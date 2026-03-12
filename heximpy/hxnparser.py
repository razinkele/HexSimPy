"""HXN binary file parser for HexSim spatial data.

Reads PATCH_HEXMAP (.hxn), PATCH_GRID (.grid), and barrier (.hbf) files.
"""
from __future__ import annotations

import math
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


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
            f.write(b"PATCH_HEXMAP")
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
        """Even-q column-offset neighbors (flat-top hex)."""
        if col % 2 == 0:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1)]
        else:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (1, 1)]
        return [
            (row + dr, col + dc)
            for dr, dc in offsets
            if 0 <= row + dr < self.height and 0 <= col + dc < self.width
        ]

    def hex_to_xy(self, row: int, col: int) -> tuple[float, float]:
        """Convert hex grid (row, col) to Cartesian (x, y)."""
        edge = self._effective_edge()
        x = 1.5 * edge * col
        y = math.sqrt(3.0) * edge * (row + 0.5 * (col % 2))
        return (x, y)

    def xy_to_hex(self, x: float, y: float) -> tuple[int, int]:
        """Convert Cartesian (x, y) to nearest hex grid (row, col)."""
        edge = self._effective_edge()
        col = round(x / (1.5 * edge))
        row = round(y / (math.sqrt(3.0) * edge) - 0.5 * (col % 2))
        return (int(row), int(col))

    @staticmethod
    def _offset_to_cube(row: int, col: int) -> tuple[int, int, int]:
        """Convert even-q offset coords to cube coords."""
        qx = col
        qz = row - (col + (col & 1)) // 2
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
        rows_list = []
        for i, val in enumerate(self.values):
            if not include_empty and val == 0.0:
                continue
            r = i // self.width
            c = i % self.width
            poly = Polygon(self.hex_polygon(r, c))
            rows_list.append(
                {"hex_id": i + 1, "row": r, "col": c, "value": float(val), "geometry": poly}
            )
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

    @classmethod
    def from_file(cls, path: str | Path) -> HexMap:
        """Read a .hxn file, auto-detecting PATCH_HEXMAP vs plain format.

        Raises ValueError if the file is a PATCH_GRID (.grid) file.
        """
        path = Path(path)
        with open(path, "rb") as f:
            peek = f.read(12)
            if peek == b"PATCH_HEXMAP":
                return cls._read_patch_hexmap(f)
            if peek[:10] == b"PATCH_GRID":
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
        hist_idx = rest.find(b"HISTORY")
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
            values = np.frombuffer(f.read(n * 4), dtype=np.float32)
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
            if magic != b"PATCH_GRID":
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
