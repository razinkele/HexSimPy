"""HexSim workspace file parser.

Reads/writes HexSim .hxn (hexmap), .grid (workspace geometry),
and .hbf (barrier) files. Provides hex grid geometry utilities
and export to GeoDataFrame, shapefile, GeoTIFF, and CSV.

Neighbor and coordinate methods use pointy-top odd-row offset convention,
matching HexSim 4.0.20 and verified against the original viewer: odd rows
are shifted right by half a column width.  The binary file format
historically stores data in row-major order with even rows of width W
and odd rows of width W-1 (when flag=1, narrow grid).
"""
from __future__ import annotations

import math
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── Format constants ─────────────────────────────────────────────────────────

_HXN_MAGIC = b"PATCH_HEXMAP"       # 12 bytes
_HXN_HEADER_SIZE = 37              # bytes before float32 data starts
_GRID_MAGIC = b"PATCH_GRID"        # 10 bytes
_HISTORY_MARKER = b"HISTORY"


def _build_flat_to_rowcol(height: int, width: int, flag: int):
    """Build flat-index → (row, col) lookup arrays."""
    if flag == 0:
        rows = np.repeat(np.arange(height), width)
        cols = np.tile(np.arange(width), height)
    else:
        row_list, col_list = [], []
        for r in range(height):
            rw = width if r % 2 == 0 else width - 1
            row_list.append(np.full(rw, r, dtype=np.int32))
            col_list.append(np.arange(rw, dtype=np.int32))
        rows = np.concatenate(row_list)
        cols = np.concatenate(col_list)
    return rows, cols


@dataclass
class HexMap:
    """Parsed hex-map data from a .hxn file."""

    format: str  # "patch_hexmap" or "plain"
    version: int
    height: int
    width: int
    flag: int  # 0 = wide, 1 = narrow
    max_val: float
    min_val: float
    hexzero: float
    values: np.ndarray
    cell_size: float = 0.0
    origin: tuple = (0.0, 0.0)
    nodata: int = 0
    dtype_code: int = 1
    _edge: float = field(default=0.0, repr=False)

    @property
    def n_hexagons(self) -> int:
        """Compute total hexagon count from grid dimensions and flag."""
        if self.flag == 0:
            return self.width * self.height
        # flag == 1 (narrow)
        if self.height % 2 == 0:
            n_wide = self.height // 2
            n_narrow = n_wide
        else:
            n_wide = (self.height + 1) // 2
            n_narrow = n_wide - 1
        return self.width * n_wide + (self.width - 1) * n_narrow

    def to_file(self, path: str | Path, *, format: str | None = None) -> None:
        """Write a .hxn file in the specified format.

        Parameters
        ----------
        path : str or Path
            Output file path.
        format : str, optional
            "patch_hexmap" or "plain".  Defaults to ``self.format``.
        """
        fmt = format or self.format
        if fmt == "patch_hexmap":
            self._write_patch_hexmap(path)
        elif fmt == "plain":
            self._write_plain(path)
        else:
            raise ValueError(f"Unknown format: {fmt!r}")

    def _write_patch_hexmap(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            f.write(_HXN_MAGIC)
            f.write(struct.pack("<I", self.version))
            f.write(struct.pack("<I", self.height))
            f.write(struct.pack("<I", self.width))
            f.write(struct.pack("B", self.flag))
            f.write(struct.pack("<f", self.max_val))
            f.write(struct.pack("<f", self.min_val))
            f.write(struct.pack("<f", self.hexzero))
            f.write(self.values.astype("<f4").tobytes())

    def _write_plain(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            f.write(struct.pack("<i", self.version))
            f.write(struct.pack("<i", self.width))
            f.write(struct.pack("<i", self.height))
            f.write(struct.pack("<d", self.cell_size))
            f.write(struct.pack("<d", self.origin[0]))
            f.write(struct.pack("<d", self.origin[1]))
            f.write(struct.pack("<i", self.dtype_code))
            f.write(struct.pack("<i", self.nodata))
            dtype = "<f4" if self.dtype_code == 1 else "<i4"
            f.write(self.values.astype(dtype).tobytes())

    # ------------------------------------------------------------------
    # Hex geometry methods
    # ------------------------------------------------------------------

    def _effective_edge(self) -> float:
        """_edge (Workspace) > cell_size (plain) > 1.0 fallback."""
        if self._edge > 0:
            return self._edge
        if self.cell_size > 0:
            return self.cell_size
        return 1.0

    def neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Pointy-top odd-row offset neighbors."""
        if row % 2 == 0:  # even row (not shifted)
            offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:             # odd row (shifted right)
            offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
        result = []
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if self.flag == 1 and nr % 2 == 1 and nc >= self.width - 1:
                    continue
                result.append((nr, nc))
        return result

    def hex_to_xy(self, row: int, col: int) -> tuple[float, float]:
        """Convert hex grid (row, col) to Cartesian (x, y) — pointy-top odd-row."""
        edge = self._effective_edge()
        x = math.sqrt(3.0) * edge * (col + 0.5 * (row % 2))
        y = 1.5 * edge * row
        return (x, y)

    def xy_to_hex(self, x: float, y: float) -> tuple[int, int]:
        """Convert Cartesian (x, y) to nearest hex grid (row, col)."""
        edge = self._effective_edge()
        col = round(x / (1.5 * edge))
        row = round(y / (math.sqrt(3.0) * edge) - 0.5 * (col % 2))
        return (int(row), int(col))

    @staticmethod
    def _offset_to_cube(row: int, col: int) -> tuple[int, int, int]:
        """Convert odd-q offset coords to cube coords."""
        qx = col
        qz = row - col // 2
        qy = -qx - qz
        return (qx, qy, qz)

    def hex_distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """Hex distance between two (row, col) positions."""
        ax, ay, az = self._offset_to_cube(a[0], a[1])
        bx, by, bz = self._offset_to_cube(b[0], b[1])
        return (abs(ax - bx) + abs(ay - by) + abs(az - bz)) // 2

    def hex_polygon(self, row: int, col: int) -> list[tuple[float, float]]:
        """Return 6 vertices of the flat-top hexagon at (row, col)."""
        cx, cy = self.hex_to_xy(row, col)
        edge = self._effective_edge()
        return [
            (
                cx + edge * math.cos(math.radians(60 * i)),
                cy + edge * math.sin(math.radians(60 * i)),
            )
            for i in range(6)
        ]

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def to_csv(self, path, *, skip_zeros=True):
        """Export to CSV in hxn2csv.c format: 'Hex ID,Score'."""
        path = Path(path)
        with open(path, "w") as f:
            f.write("Hex ID,Score\n")
            for i, val in enumerate(self.values):
                if skip_zeros and val == 0.0:
                    continue
                f.write(f"{i + 1},{val:f}\n")

    def to_geodataframe(self, *, edge=None, include_empty=False, crs=None):
        """Convert to GeoDataFrame with hex polygon geometry."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        if edge is not None:
            saved = self._edge
            self._edge = edge
        elif self._effective_edge() == 1.0 and self.format == "patch_hexmap":
            warnings.warn(
                "No edge length set for PATCH_HEXMAP file. "
                "Pass edge= parameter or load via Workspace for correct geometry."
            )
        try:
            all_rows, all_cols = _build_flat_to_rowcol(self.height, self.width, self.flag)
            rows_list = []
            for i, val in enumerate(self.values):
                if not include_empty and val == 0.0:
                    continue
                r = int(all_rows[i])
                c = int(all_cols[i])
                poly = Polygon(self.hex_polygon(r, c))
                rows_list.append(
                    {"hex_id": i + 1, "row": r, "col": c, "value": float(val), "geometry": poly}
                )
        finally:
            if edge is not None:
                self._edge = saved
        gdf = gpd.GeoDataFrame(rows_list, geometry="geometry")
        if crs:
            gdf = gdf.set_crs(crs)
        return gdf

    def to_shapefile(self, path, *, edge=None, include_empty=False, crs=None):
        """Export to shapefile."""
        gdf = self.to_geodataframe(edge=edge, include_empty=include_empty, crs=crs)
        gdf.to_file(str(path))

    def to_geotiff(self, path, *, crs=None):
        """Export as GeoTIFF (rectangular raster approximation)."""
        import rasterio

        if self.flag == 1:
            raise ValueError(
                "to_geotiff() is not supported for narrow grids (flag=1) "
                "— use to_geodataframe() instead"
            )

        grid = self.values.reshape((self.height, self.width))
        edge = self._effective_edge()
        col_spacing = 1.5 * edge
        row_spacing = math.sqrt(3.0) * edge
        ox, oy = self.origin
        transform = rasterio.transform.from_bounds(
            ox, oy,
            ox + self.width * col_spacing,
            oy + self.height * row_spacing,
            self.width, self.height,
        )
        with rasterio.open(
            str(path), "w", driver="GTiff",
            height=self.height, width=self.width,
            count=1, dtype="float32", crs=crs, transform=transform,
        ) as dst:
            dst.write(grid.astype(np.float32), 1)

    @classmethod
    def from_file(cls, path: str | Path) -> HexMap:
        """Read a .hxn file, auto-detecting PATCH_HEXMAP vs plain format.

        Raises ValueError if the file is a PATCH_GRID (.grid) file.
        """
        path = Path(path)
        with open(path, "rb") as f:
            peek = f.read(12)
            if peek == _HXN_MAGIC:
                return cls._read_patch_hexmap(f)
            if peek[:10] == _GRID_MAGIC:
                raise ValueError(
                    f"File is PATCH_GRID format, not a hex-map: {path}"
                )
            # Plain format – rewind and read
            f.seek(0)
            return cls._read_plain(f)

    @classmethod
    def _read_patch_hexmap(cls, f) -> HexMap:
        """Read PATCH_HEXMAP format (37-byte header)."""
        version, nrows, ncols = struct.unpack("<III", f.read(12))
        (flag,) = struct.unpack("<B", f.read(1))
        max_val, min_val, hexzero = struct.unpack("<fff", f.read(12))
        # Data runs until b"HISTORY" marker or EOF
        rest = f.read()
        hist_idx = rest.find(_HISTORY_MARKER)
        data_bytes = rest[:hist_idx] if hist_idx >= 0 else rest
        values = np.frombuffer(data_bytes, dtype=np.float32).copy()
        return cls(
            format="patch_hexmap",
            version=version,
            height=nrows,
            width=ncols,
            flag=flag,
            max_val=max_val,
            min_val=min_val,
            hexzero=hexzero,
            values=values,
        )

    @classmethod
    def _read_plain(cls, f) -> HexMap:
        """Read plain format (44-byte header)."""
        version, ncols, nrows = struct.unpack("<iii", f.read(12))
        (cell_size,) = struct.unpack("<d", f.read(8))
        origin_x, origin_y = struct.unpack("<dd", f.read(16))
        dtype_code, nodata = struct.unpack("<ii", f.read(8))
        n = ncols * nrows
        if dtype_code == 1:
            values = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
        elif dtype_code == 2:
            values = np.frombuffer(f.read(n * 4), dtype=np.int32).astype(
                np.float32
            )
        else:
            raise ValueError(f"Unknown dtype_code: {dtype_code}")
        return cls(
            format="plain",
            version=version,
            height=nrows,
            width=ncols,
            flag=0,  # plain format is always wide
            max_val=float(values.max()) if len(values) > 0 else 0.0,
            min_val=float(values.min()) if len(values) > 0 else 0.0,
            hexzero=0.0,
            values=values,
            cell_size=cell_size,
            origin=(origin_x, origin_y),
            nodata=nodata,
            dtype_code=dtype_code,
        )


@dataclass
class GridMeta:
    """Metadata from a PATCH_GRID (.grid) file."""

    ncols: int
    nrows: int
    x_extent: float
    y_extent: float
    row_spacing: float
    edge: float  # row_spacing / sqrt(3)
    version: int = 0
    n_hexes: int = 0
    flag: int = 0

    @classmethod
    def from_file(cls, path: str | Path) -> GridMeta:
        """Read a PATCH_GRID (.grid) file (67-byte header).

        Raises ValueError if the file does not start with b"PATCH_GRID".
        """
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(10)
            if magic != _GRID_MAGIC:
                raise ValueError(
                    f"Not a PATCH_GRID file (got {magic!r}): {path}"
                )
            version, n_hexes, ncols, nrows = struct.unpack("<IIII", f.read(16))
            (flag,) = struct.unpack("<B", f.read(1))
            georef = struct.unpack("<5d", f.read(40))
        return cls(
            ncols=ncols,
            nrows=nrows,
            x_extent=georef[0],
            y_extent=georef[3],
            row_spacing=georef[4],
            edge=georef[4] / math.sqrt(3),
            version=version,
            n_hexes=n_hexes,
            flag=flag,
        )


@dataclass
class Barrier:
    """A single barrier edge entry from an .hbf file."""

    hex_id: int
    edge: int
    classification: int
    class_name: str


def read_barriers(path: str | Path) -> list[Barrier]:
    """Parse an .hbf barrier file.

    The file contains C (classification) and E (edge) lines:
        C <id> <p1> <p2> "<name>"
        E <hex_id> <edge> <class_id>

    Returns a list of Barrier with class_name populated from classifications.
    Falls back to empty string if a classification is not defined.
    """
    path = Path(path)
    classifications: dict[int, str] = {}
    barriers: list[Barrier] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "C":
                class_id = int(parts[1])
                # Name is the last quoted token
                name = line.split('"')[1]
                classifications[class_id] = name
            elif parts[0] == "E":
                hex_id = int(parts[1])
                edge = int(parts[2])
                class_id = int(parts[3])
                class_name = classifications.get(class_id, "")
                barriers.append(
                    Barrier(
                        hex_id=hex_id,
                        edge=edge,
                        classification=class_id,
                        class_name=class_name,
                    )
                )
    return barriers


@dataclass
class WorldFile:
    """Affine transform from a world file (.bpw, .wld, .pgw, etc.).

    Lines: A (x-scale), D (rotation), B (rotation), E (y-scale), C (x-origin), F (y-origin).
    Maps pixel (col, row) to map (x, y) via:
        x = A*col + B*row + C
        y = D*col + E*row + F
    """

    A: float
    D: float
    B: float
    E: float
    C: float
    F: float

    @classmethod
    def from_file(cls, path: str | Path) -> WorldFile:
        """Read a 6-line world file."""
        lines = Path(path).read_text().strip().splitlines()
        if len(lines) < 6:
            raise ValueError(f"World file needs 6 lines, got {len(lines)}: {path}")
        vals = [float(line.strip()) for line in lines[:6]]
        return cls(A=vals[0], D=vals[1], B=vals[2], E=vals[3], C=vals[4], F=vals[5])

    def pixel_to_map(self, px, py):
        """Transform pixel coordinates to map coordinates (scalar or array)."""
        mx = self.A * px + self.B * py + self.C
        my = self.D * px + self.E * py + self.F
        return mx, my

    @property
    def matrix(self) -> np.ndarray:
        """3x3 affine transformation matrix."""
        return np.array([
            [self.A, self.B, self.C],
            [self.D, self.E, self.F],
            [0.0,    0.0,    1.0],
        ])


@dataclass
class Workspace:
    """A HexSim workspace directory containing grid, hexmaps, and barriers."""

    grid: GridMeta
    hexmaps: dict[str, HexMap]
    barriers: list[Barrier]
    path: Path

    @property
    def layer_names(self) -> list[str]:
        """Return sorted list of hexmap layer names."""
        return sorted(self.hexmaps.keys())

    @classmethod
    def from_dir(cls, path) -> Workspace:
        """Load a HexSim workspace from a directory.

        Expects:
        - One .grid file in the root
        - Spatial Data/Hexagons/<layer>/<layer>.*.hxn files
        - Optionally Spatial Data/barriers/<name>/<name>.*.hbf files
        """
        ws = Path(path)

        # 1. Find .grid
        grid_files = list(ws.glob("*.grid"))
        if not grid_files:
            raise FileNotFoundError(f"No .grid file found in {ws}")
        grid = GridMeta.from_file(grid_files[0])

        # 2. Find hexagon layers
        hex_dir = ws / "Spatial Data" / "Hexagons"
        if not hex_dir.exists():
            raise FileNotFoundError(f"Spatial Data/Hexagons/ not found in {ws}")
        hexmaps: dict[str, HexMap] = {}
        for hxn_path in sorted(hex_dir.glob("*/*.hxn")):
            layer_name = hxn_path.parent.name
            hm = HexMap.from_file(hxn_path)
            hm._edge = grid.edge
            hexmaps[layer_name] = hm
        if not hexmaps:
            warnings.warn(f"No .hxn files found in {hex_dir}")

        # 3. Find barriers
        barriers: list[Barrier] = []
        barrier_dir = ws / "Spatial Data" / "barriers"
        if barrier_dir.exists():
            for hbf_path in sorted(barrier_dir.glob("*/*.hbf")):
                barriers.extend(read_barriers(hbf_path))
        else:
            for hbf_path in sorted(ws.glob("*.hbf")):
                barriers.extend(read_barriers(hbf_path))

        return cls(grid=grid, hexmaps=hexmaps, barriers=barriers, path=ws)
