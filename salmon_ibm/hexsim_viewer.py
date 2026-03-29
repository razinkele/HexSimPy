"""HexSim grid viewer using shiny-deckgl.

This module provides functions to:
1. List available HexSim grid layers from a workspace
2. Load grid data and prepare it for deck.gl visualization
3. Create a standalone Shiny app for interactive grid viewing

Maps HexSim river corridor cells to H3 hexagonal cells along the actual
Columbia River path from Bonneville Dam to The Dalles Dam.

Usage
-----
Standalone app::

    conda run -n shiny python -m salmon_ibm.hexsim_viewer

As library::

    from salmon_ibm.hexsim_viewer import list_grids, load_grid, HexGridViewer

    # List available grids
    grids = list_grids("path/to/workspace")

    # Load a specific grid
    data = load_grid("path/to/workspace", "River [ extent ]")
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import NamedTuple

import h3
import numpy as np

# ── Hex geometry constants ────────────────────────────────────────────────────

# Flat-top hexagon vertices (unit size, angles at 0°, 60°, 120°, 180°, 240°, 300°)
# Reference: https://www.redblobgames.com/grids/hexagons/
_HEX_ANGLES_DEG = np.array([0, 60, 120, 180, 240, 300], dtype=np.float64)
_HEX_ANGLES_RAD = np.deg2rad(_HEX_ANGLES_DEG)
_HEX_COS = np.cos(_HEX_ANGLES_RAD)  # [1, 0.5, -0.5, -1, -0.5, 0.5]
_HEX_SIN = np.sin(
    _HEX_ANGLES_RAD
)  # [0, sqrt(3)/2, sqrt(3)/2, 0, -sqrt(3)/2, -sqrt(3)/2]


# ── Binary format constants (from hexsim.py) ──────────────────────────────────

_HXN_MAGIC = b"PATCH_HEXMAP"
_HXN_HEADER_SIZE = 25
_GRID_MAGIC = b"PATCH_GRID"
_HISTORY_MARKER = b"HISTORY"


# ── Data structures ──────────────────────────────────────────────────────────


class GridInfo(NamedTuple):
    """Metadata for a HexSim grid layer."""

    name: str
    path: Path
    hxn_file: Path


class HexGridData(NamedTuple):
    """Loaded HexSim grid data ready for visualization."""

    name: str
    ncols: int
    nrows: int
    values: np.ndarray  # (N,) float32 array of all cell values
    water_mask: np.ndarray  # (N,) bool array: True for non-zero (water) cells
    centroids: np.ndarray  # (N_water, 2) as [y, x] in grid units
    water_values: np.ndarray  # (N_water,) values for water cells only


# ── Binary readers ───────────────────────────────────────────────────────────


def _read_grid(path: Path) -> tuple[int, int]:
    """Parse HexSim .grid file and return (ncols, nrows)."""
    with open(path, "rb") as f:
        data = f.read()

    if not data.startswith(_GRID_MAGIC):
        raise ValueError(f"Not a HexSim .grid file: {path}")

    ncols = struct.unpack_from("<I", data, 18)[0]
    nrows = struct.unpack_from("<I", data, 22)[0]
    return ncols, nrows


def _read_hexmap(path: Path) -> np.ndarray:
    """Parse HexSim .hxn file and return flat float32 array of values."""
    with open(path, "rb") as f:
        raw = f.read()

    if not raw.startswith(_HXN_MAGIC):
        raise ValueError(f"Not a HexSim .hxn file: {path}")

    # Find footer marker to determine data extent
    hist_pos = raw.find(_HISTORY_MARKER)
    if hist_pos < 0:
        hist_pos = len(raw)

    data_bytes = raw[_HXN_HEADER_SIZE:hist_pos]
    n_floats = len(data_bytes) // 4
    arr = np.frombuffer(data_bytes[: n_floats * 4], dtype="<f4").copy()
    return arr


# ── Grid discovery ───────────────────────────────────────────────────────────


def list_grids(
    workspace_dir: str | Path, hexagons_subdir: str = "Spatial Data/Hexagons"
) -> list[GridInfo]:
    """List all available HexSim grid layers in a workspace.

    Parameters
    ----------
    workspace_dir
        Path to HexSim workspace directory (contains .grid file).
    hexagons_subdir
        Subdirectory path to Hexagons folder (default: "Spatial Data/Hexagons").

    Returns
    -------
    list[GridInfo]
        List of available grid layers with their paths.

    Example
    -------
    >>> grids = list_grids("Columbia [small]")
    >>> for g in grids:
    ...     print(g.name)
    Gradient [ distance to center ]
    Gradient [ downstream ]
    River [ depth ]
    ...
    """
    ws = Path(workspace_dir)
    hex_dir = ws / hexagons_subdir

    if not hex_dir.exists():
        raise FileNotFoundError(f"Hexagons directory not found: {hex_dir}")

    grids = []
    for subdir in sorted(hex_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Find .hxn file in the subdirectory
        hxn_files = list(subdir.glob("*.hxn"))
        if hxn_files:
            grids.append(
                GridInfo(
                    name=subdir.name,
                    path=subdir,
                    hxn_file=hxn_files[0],
                )
            )

    return grids


def find_grid_file(workspace_dir: str | Path) -> Path:
    """Find the .grid file in a HexSim workspace."""
    ws = Path(workspace_dir)
    grid_files = list(ws.glob("*.grid"))
    if not grid_files:
        raise FileNotFoundError(f"No .grid file found in {ws}")
    return grid_files[0]


# ── Grid loading ─────────────────────────────────────────────────────────────


def load_grid(
    workspace_dir: str | Path,
    grid_name: str,
    hexagons_subdir: str = "Spatial Data/Hexagons",
) -> HexGridData:
    """Load a HexSim grid layer and prepare it for visualization.

    Parameters
    ----------
    workspace_dir
        Path to HexSim workspace directory.
    grid_name
        Name of the grid layer (folder name under Hexagons/).
    hexagons_subdir
        Subdirectory path to Hexagons folder.

    Returns
    -------
    HexGridData
        Loaded grid data with centroids computed for water cells.

    Example
    -------
    >>> data = load_grid("Columbia [small]", "River [ extent ]")
    >>> print(f"Water cells: {len(data.water_values)}")
    Water cells: 88427
    """
    ws = Path(workspace_dir)

    # Get grid dimensions
    grid_file = find_grid_file(ws)
    ncols, nrows = _read_grid(grid_file)

    # Load the hexmap data
    hex_dir = ws / hexagons_subdir / grid_name
    hxn_files = list(hex_dir.glob("*.hxn"))
    if not hxn_files:
        raise FileNotFoundError(f"No .hxn file found in {hex_dir}")

    values = _read_hexmap(hxn_files[0])

    # Create water mask (non-zero values)
    water_mask = values != 0.0
    water_flat = np.where(water_mask)[0]
    water_values = values[water_flat]

    # Compute centroids for water cells — flat-top even-q convention
    # Read georeferencing from .grid for real-world meter coordinates
    from salmon_ibm.hexsim import read_grid as _read_grid_full

    grid_meta = _read_grid_full(grid_file)
    edge = grid_meta["edge"]  # hex edge length in meters

    rows = water_flat // ncols
    cols = water_flat % ncols
    col_spacing = 1.5 * edge
    row_spacing = np.sqrt(3.0) * edge
    cx = cols.astype(np.float64) * col_spacing
    cy = rows.astype(np.float64) * row_spacing + (cols % 2) * (row_spacing / 2.0)
    centroids = np.column_stack([cy, cx])  # [y, x] in meters

    return HexGridData(
        name=grid_name,
        ncols=ncols,
        nrows=nrows,
        values=values,
        water_mask=water_mask,
        centroids=centroids,
        water_values=water_values,
    )


# ── Deck.gl layer builders ───────────────────────────────────────────────────


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert hex color '#rrggbb' to (r, g, b) integers."""
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# Default colorscales
VIRIDIS = [
    [0.0, "#440154"],
    [0.25, "#3b528b"],
    [0.5, "#21918c"],
    [0.75, "#5ec962"],
    [1.0, "#fde725"],
]

BATHYMETRIC = [
    [0.0, "#0b1f2c"],
    [0.2, "#1a3d50"],
    [0.4, "#2a7a7a"],
    [0.6, "#3d9b8f"],
    [0.8, "#7ac4a5"],
    [1.0, "#c8e6c9"],
]

THERMAL = [
    [0.0, "#0b1f2c"],
    [0.15, "#1a3d50"],
    [0.3, "#2a7a7a"],
    [0.5, "#4a8fa8"],
    [0.7, "#e8d5b7"],
    [0.85, "#d4826a"],
    [1.0, "#b05a3f"],
]

# ── Columbia River corridor waypoints ─────────────────────────────────────────
# Approximate centerline from Bonneville Dam (rkm 235) to The Dalles Dam (rkm 309)

COLUMBIA_RIVER_WAYPOINTS = [
    (-121.9406, 45.6443),  # Bonneville Dam (rkm 235)
    (-121.8900, 45.6600),  # Cascade Locks area
    (-121.8200, 45.6900),  # Wind River confluence
    (-121.7500, 45.7100),
    (-121.6800, 45.7000),  # White Salmon area
    (-121.5500, 45.6900),  # Hood River area
    (-121.4000, 45.6700),  # Mosier area
    (-121.2500, 45.6400),  # Rowena area
    (-121.1800, 45.6200),  # The Dalles Dam (rkm 309)
]


def load_river_shapefile(
    gpkg_path: str | Path = "rivers/columbia.gpkg",
    bbox: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """Load Columbia River paths from GeoPackage file.

    Parameters
    ----------
    gpkg_path
        Path to the GeoPackage file containing river geometry.
    bbox
        Optional bounding box (min_lon, min_lat, max_lon, max_lat) to filter.

    Returns
    -------
    list[dict]
        List of path dicts with 'path' (list of [lon, lat]) and metadata.
    """
    try:
        import geopandas as gpd
    except ImportError:
        import logging

        logging.getLogger(__name__).warning(
            "geopandas not installed, using simple waypoints"
        )
        return []

    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        import logging

        logging.getLogger(__name__).warning("River shapefile not found: %s", gpkg_path)
        return []

    # Read and convert to WGS84
    gdf = gpd.read_file(gpkg_path, engine="fiona")
    gdf_wgs84 = gdf.to_crs("EPSG:4326")

    # Filter by bounding box if provided
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        gdf_wgs84 = gdf_wgs84.cx[min_lon:max_lon, min_lat:max_lat]

    # Extract paths
    paths = []
    for _, row in gdf_wgs84.iterrows():
        geom = row.geometry
        name = row.get("PNAME", "")
        pmile = row.get("PMILE", 0)

        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                coords = list(line.coords)
                paths.append(
                    {
                        "path": [[c[0], c[1]] for c in coords],
                        "name": name,
                        "pmile": float(pmile) if pmile else 0,
                        "color": [100, 150, 255, 200],
                    }
                )
        elif geom.geom_type == "LineString":
            coords = list(geom.coords)
            paths.append(
                {
                    "path": [[c[0], c[1]] for c in coords],
                    "name": name,
                    "pmile": float(pmile) if pmile else 0,
                    "color": [100, 150, 255, 200],
                }
            )

    return paths


class RiverCenterline:
    """River centerline for placing cells along the actual river path.

    Builds a continuous centerline from shapefile paths, computing cumulative
    distances for accurate interpolation along the river.
    """

    def __init__(self, gpkg_path: str | Path = "rivers/columbia.gpkg"):
        """Load river centerline from GeoPackage.

        Parameters
        ----------
        gpkg_path
            Path to the GeoPackage file.
        """
        self.points: list[tuple[float, float]] = []  # (lon, lat)
        self.distances: list[float] = []  # cumulative distance
        self.total_length: float = 0.0

        self._load_centerline(gpkg_path)

    def _load_centerline(self, gpkg_path: str | Path):
        """Extract and merge river segments into continuous centerline."""
        try:
            import geopandas as gpd
            from shapely.ops import linemerge
            from shapely.geometry import MultiLineString
        except ImportError:
            import logging

            logging.getLogger(__name__).warning(
                "geopandas/shapely not installed, using simple waypoints"
            )
            self._use_simple_waypoints()
            return

        gpkg_path = Path(gpkg_path)
        if not gpkg_path.exists():
            import logging

            logging.getLogger(__name__).warning(
                "River shapefile not found: %s, using simple waypoints", gpkg_path
            )
            self._use_simple_waypoints()
            return

        # Read and convert to WGS84
        gdf = gpd.read_file(gpkg_path, engine="fiona")
        gdf_wgs84 = gdf.to_crs("EPSG:4326")

        # Collect all line geometries
        lines = []
        for geom in gdf_wgs84.geometry:
            if geom.geom_type == "MultiLineString":
                lines.extend(list(geom.geoms))
            elif geom.geom_type == "LineString":
                lines.append(geom)

        if not lines:
            self._use_simple_waypoints()
            return

        # Merge lines into continuous path
        merged = linemerge(MultiLineString(lines))

        # Extract coordinates
        if merged.geom_type == "LineString":
            coords = list(merged.coords)
        elif merged.geom_type == "MultiLineString":
            # Take the longest segment if merge didn't fully connect
            longest = max(merged.geoms, key=lambda g: g.length)
            coords = list(longest.coords)
        else:
            self._use_simple_waypoints()
            return

        # Store points and compute cumulative distances
        self.points = [(c[0], c[1]) for c in coords]
        self._compute_distances()

        print(
            f"Loaded river centerline: {len(self.points)} points, "
            f"{self.total_length:.2f}° total length"
        )

    def _use_simple_waypoints(self):
        """Fall back to simple waypoints."""
        self.points = list(COLUMBIA_RIVER_WAYPOINTS)
        self._compute_distances()

    def _compute_distances(self):
        """Compute cumulative distances along centerline."""
        self.distances = [0.0]
        for i in range(1, len(self.points)):
            p1, p2 = self.points[i - 1], self.points[i]
            # Simple Euclidean distance in degrees (approximate)
            dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            self.distances.append(self.distances[-1] + dist)
        self.total_length = self.distances[-1] if self.distances else 0.0

    def get_coords(self, t: float) -> tuple[float, float]:
        """Get coordinates at position t (0-1) along river.

        Parameters
        ----------
        t
            Normalized position (0 = start, 1 = end of centerline).

        Returns
        -------
        tuple[float, float]
            (lon, lat) coordinates.
        """
        if not self.points:
            return (-121.5, 45.65)  # Default center

        if t <= 0:
            return self.points[0]
        if t >= 1:
            return self.points[-1]

        target_dist = t * self.total_length

        # Find segment containing target distance
        for i in range(1, len(self.distances)):
            if self.distances[i] >= target_dist:
                # Interpolate within this segment
                d0, d1 = self.distances[i - 1], self.distances[i]
                if d1 <= d0:
                    return self.points[i - 1]
                local_t = (target_dist - d0) / (d1 - d0)
                p0, p1 = self.points[i - 1], self.points[i]
                lon = p0[0] + local_t * (p1[0] - p0[0])
                lat = p0[1] + local_t * (p1[1] - p0[1])
                return (lon, lat)

        return self.points[-1]

    def get_perpendicular(self, t: float) -> tuple[float, float]:
        """Get perpendicular direction at position t.

        Returns unit vector perpendicular to river direction (for cross-river offset).

        Parameters
        ----------
        t
            Normalized position along river.

        Returns
        -------
        tuple[float, float]
            (dx, dy) perpendicular unit vector.
        """
        if len(self.points) < 2:
            return (0.0, 1.0)

        # Find segment at position t
        target_dist = t * self.total_length
        for i in range(1, len(self.distances)):
            if self.distances[i] >= target_dist:
                p0, p1 = self.points[i - 1], self.points[i]
                # River direction
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Perpendicular (rotate 90°)
                    return (-dy / length, dx / length)
                break

        return (0.0, 1.0)


# Global river centerline (lazy loaded)
_river_centerline: RiverCenterline | None = None


def get_river_centerline() -> RiverCenterline:
    """Get or create the global river centerline."""
    global _river_centerline
    if _river_centerline is None:
        _river_centerline = RiverCenterline()
    return _river_centerline


def _interpolate_river_position(
    downstream_val: float,
    downstream_min: float,
    downstream_max: float,
) -> float:
    """Convert downstream gradient value to position along river (0 to 1).

    Parameters
    ----------
    downstream_val
        The downstream gradient value for a cell.
    downstream_min, downstream_max
        Range of downstream values in the grid.

    Returns
    -------
    float
        Normalized position: 0 = downstream end (The Dalles), 1 = upstream end (Bonneville).
    """
    if downstream_max <= downstream_min:
        return 0.5
    t = (downstream_val - downstream_min) / (downstream_max - downstream_min)
    return float(t)


def _get_river_coords(
    t: float,
    waypoints: list[tuple[float, float]],
) -> tuple[float, float]:
    """Get lat/lon coordinates for position t (0-1) along river waypoints.

    Parameters
    ----------
    t
        Normalized position along river (0 = downstream end, 1 = upstream end).
    waypoints
        List of (lon, lat) coordinates from upstream to downstream.

    Returns
    -------
    tuple[float, float]
        (lon, lat) coordinates at position t.
    """
    if t <= 0:
        return waypoints[-1]  # End point (The Dalles)
    if t >= 1:
        return waypoints[0]  # Start point (Bonneville)

    n = len(waypoints)
    segment_t = t * (n - 1)
    idx = int(segment_t)
    if idx >= n - 1:
        return waypoints[0]

    local_t = segment_t - idx
    p1 = waypoints[n - 1 - idx]  # Going from end to start
    p2 = waypoints[n - 2 - idx]

    lon = p1[0] + local_t * (p2[0] - p1[0])
    lat = p1[1] + local_t * (p2[1] - p1[1])
    return (lon, lat)


def colorscale_rgb(
    z: np.ndarray,
    colorscale: list | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map array values to (N, 3) uint8 RGB using a colorscale.

    Parameters
    ----------
    z
        Array of values to map.
    colorscale
        List of [stop, color] pairs. Defaults to VIRIDIS.
    vmin, vmax
        Value range. Defaults to z.min(), z.max().

    Returns
    -------
    np.ndarray
        (N, 3) uint8 RGB array.
    """
    if colorscale is None:
        colorscale = VIRIDIS

    z_min = vmin if vmin is not None else float(np.nanmin(z))
    z_max = vmax if vmax is not None else float(np.nanmax(z))

    if z_max <= z_min:
        z_norm = np.zeros(len(z), dtype=np.float32)
    else:
        z_norm = ((z - z_min) / (z_max - z_min)).astype(np.float32)

    z_norm = np.clip(z_norm, 0, 1)
    stops = np.array([s[0] for s in colorscale], dtype=np.float32)
    colors = np.array([_hex_to_rgb(s[1]) for s in colorscale], dtype=np.float32)

    r = np.interp(z_norm, stops, colors[:, 0]).astype(np.uint8)
    g = np.interp(z_norm, stops, colors[:, 1]).astype(np.uint8)
    b = np.interp(z_norm, stops, colors[:, 2]).astype(np.uint8)

    return np.column_stack([r, g, b])


def compute_hex_vertices(
    centroids: np.ndarray,
    scale: float = 0.0005,
    hex_size: float = 1.0,
) -> np.ndarray:
    """Compute hexagon vertices for all centroids.

    Parameters
    ----------
    centroids
        (N, 2) array of [y, x] centroid coordinates in grid units.
    scale
        Coordinate scale factor (grid units to pseudo-lat/lon).
    hex_size
        Hex circumradius in grid units (default: 1.0).

    Returns
    -------
    np.ndarray
        (N, 6, 2) array of vertex coordinates [x, y] for each hex.
    """
    n = len(centroids)
    # Centroids are [y, x], we need [x, y] for deck.gl
    cx = centroids[:, 1] * scale  # x (longitude-like)
    cy = centroids[:, 0] * scale  # y (latitude-like)

    # Scale the hex size
    size = hex_size * scale

    # Compute all 6 vertices for each hex
    # vertices[i, j] = (cx[i] + size*cos(angle_j), cy[i] + size*sin(angle_j))
    vertices = np.zeros((n, 6, 2), dtype=np.float64)
    for j in range(6):
        vertices[:, j, 0] = cx + size * _HEX_COS[j]
        vertices[:, j, 1] = cy + size * _HEX_SIN[j]

    return vertices.astype(np.float32)


def build_h3_data(
    grid: HexGridData,
    colorscale: list | None = None,
    alpha: int = 255,
    h3_resolution: int = 9,
    hex_size_meters: float = 24.0,  # HexSim hex circumradius in meters
    lat_origin: float = 45.5,  # Columbia River southern extent
    lon_origin: float = -122.5,  # Columbia River western extent
) -> list[dict]:
    """Build deck.gl H3HexagonLayer data from HexGridData.

    Maps HexSim grid cell values to H3 hexagonal indices. When multiple
    HexSim cells map to the same H3 cell, their values are averaged.

    Parameters
    ----------
    grid
        Loaded grid data from load_grid().
    colorscale
        Color mapping. Defaults to VIRIDIS.
    alpha
        Fill opacity (0-255).
    h3_resolution
        H3 resolution (0-15). Higher = smaller hexes.
        7 ≈ 1.22km edge, 8 ≈ 461m edge, 9 ≈ 174m edge, 10 ≈ 66m edge.
    hex_size_meters
        HexSim hex circumradius in meters (from grid file, typically ~24m).
    lat_origin, lon_origin
        Geographic origin (SW corner) to place grid in real coordinates.

    Returns
    -------
    list[dict]
        List of dicts with 'hex' (H3 index), 'color', and 'value' keys.
    """
    # Convert hex size to degrees (approximate at Columbia River latitude)
    # At 46°N: 1 degree lat ≈ 111km, 1 degree lon ≈ 77km
    meters_per_deg_lat = 111_000.0
    meters_per_deg_lon = 77_000.0  # cos(46°) * 111km

    # Pointy-top hex spacing in meters
    hex_width_m = np.sqrt(3.0) * hex_size_meters  # horizontal center-to-center
    hex_height_m = 1.5 * hex_size_meters  # vertical center-to-center

    # Convert to degrees
    deg_per_col = hex_width_m / meters_per_deg_lon
    deg_per_row = hex_height_m / meters_per_deg_lat

    print(f"Building H3 data: {len(grid.water_values)} cells -> H3 res {h3_resolution}")
    print(
        f"  HexSim hex size: {hex_size_meters}m, spacing: {hex_width_m:.1f}m x {hex_height_m:.1f}m"
    )
    print(f"  Scale: {deg_per_col:.6f}°/col, {deg_per_row:.6f}°/row")

    # Map all HexSim cells to H3 indices and aggregate values
    h3_values: dict[str, list[float]] = {}

    for i in range(len(grid.water_values)):
        # Centroids are [y, x] where y=row*hex_h, x=col*hex_w (already scaled by hex spacing)
        # But we stored them as [row_scaled, col_scaled], so divide back
        row_scaled = grid.centroids[i, 0]
        col_scaled = grid.centroids[i, 1]

        # Convert to geographic coordinates
        # Note: row increases northward (lat), col increases eastward (lon)
        lat = lat_origin + (row_scaled / 1.5) * deg_per_row
        lon = lon_origin + (col_scaled / np.sqrt(3.0)) * deg_per_col

        # Get H3 index for this location
        h3_index = h3.latlng_to_cell(lat, lon, h3_resolution)

        # Aggregate values for cells that map to the same H3 hex
        if h3_index not in h3_values:
            h3_values[h3_index] = []
        h3_values[h3_index].append(float(grid.water_values[i]))

    print(f"  Mapped to {len(h3_values)} unique H3 cells")

    # Compute mean value for each H3 cell
    h3_means = {k: np.mean(v) for k, v in h3_values.items()}

    # Compute colors based on aggregated values
    hex_ids = list(h3_means.keys())
    values = np.array([h3_means[h] for h in hex_ids])
    rgb = colorscale_rgb(values, colorscale)

    # Build H3 data list
    h3_data = []
    for i, hex_id in enumerate(hex_ids):
        h3_data.append(
            {
                "hex": hex_id,
                "color": [int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]), alpha],
                "value": float(values[i]),
            }
        )

    return h3_data


def build_polygon_data(
    grid: HexGridData,
    colorscale: list | None = None,
    scale: float = 0.0005,
    max_hexes: int = 200_000,
    alpha: int = 220,
) -> list[dict]:
    """Build deck.gl PolygonLayer data from HexGridData.

    Parameters
    ----------
    grid
        Loaded grid data from load_grid().
    colorscale
        Color mapping. Defaults to VIRIDIS.
    scale
        Coordinate scale factor (grid units to pseudo-lat/lon).
    max_hexes
        Maximum hexagons to render (subsamples if exceeded).
    alpha
        Fill opacity (0-255).

    Returns
    -------
    list[dict]
        List of polygon dicts with 'polygon' and 'color' keys.
    """
    n_water = len(grid.water_values)

    # Subsample if too many hexes
    if n_water > max_hexes:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_water, max_hexes, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_water)

    # Compute colors
    rgb = colorscale_rgb(grid.water_values, colorscale)

    # Compute hex vertices
    vertices = compute_hex_vertices(grid.centroids[idx], scale=scale)

    # Build polygon list
    polygons = []
    for i, orig_idx in enumerate(idx):
        # Each polygon is a list of [x, y] vertices
        poly_coords = vertices[i].tolist()
        polygons.append(
            {
                "polygon": poly_coords,
                "color": [
                    int(rgb[orig_idx, 0]),
                    int(rgb[orig_idx, 1]),
                    int(rgb[orig_idx, 2]),
                    alpha,
                ],
                "value": float(grid.water_values[orig_idx]),
            }
        )

    return polygons


def build_layer_data(
    grid: HexGridData,
    colorscale: list | None = None,
    scale: float = 0.0005,
    max_points: int = 200_000,
    alpha: int = 220,
) -> dict:
    """Build deck.gl layer data from HexGridData.

    Parameters
    ----------
    grid
        Loaded grid data from load_grid().
    colorscale
        Color mapping. Defaults to VIRIDIS.
    scale
        Coordinate scale factor (grid units to pseudo-lat/lon).
    max_points
        Maximum hexagons to render (subsamples if exceeded).
    alpha
        Fill opacity (0-255).

    Returns
    -------
    dict
        Dict with keys: polygons, n_hexes, center, bounds, value_range.
    """
    n_water = len(grid.water_values)

    # Subsample if too many hexes
    if n_water > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_water, max_points, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_water)

    # Compute colors
    rgb = colorscale_rgb(grid.water_values, colorscale)

    # Compute hex vertices
    vertices = compute_hex_vertices(grid.centroids[idx], scale=scale)

    # Build polygon list for deck.gl
    polygons = []
    for i, orig_idx in enumerate(idx):
        poly_coords = vertices[i].tolist()
        polygons.append(
            {
                "polygon": poly_coords,
                "color": [
                    int(rgb[orig_idx, 0]),
                    int(rgb[orig_idx, 1]),
                    int(rgb[orig_idx, 2]),
                    alpha,
                ],
                "value": float(grid.water_values[orig_idx]),
            }
        )

    # Compute center for view state
    cx = float(np.mean(grid.centroids[idx, 1])) * scale
    cy = float(np.mean(grid.centroids[idx, 0])) * scale

    return {
        "polygons": polygons,
        "n_hexes": len(idx),
        "center": (cx, cy),
        "bounds": {
            "min_x": float(grid.centroids[idx, 1].min()) * scale,
            "max_x": float(grid.centroids[idx, 1].max()) * scale,
            "min_y": float(grid.centroids[idx, 0].min()) * scale,
            "max_y": float(grid.centroids[idx, 0].max()) * scale,
        },
        "value_range": {
            "min": float(np.nanmin(grid.water_values[idx])),
            "max": float(np.nanmax(grid.water_values[idx])),
        },
    }


def build_river_corridor_data(
    grid: HexGridData,
    downstream_grid: HexGridData | None = None,
    colorscale: list | None = None,
    h3_resolution: int = 9,
    max_cells: int = 20_000,
    alpha: int = 200,
    cross_river_offset: float = 0.02,
) -> dict:
    """Build deck.gl PolygonLayer data by mapping HexSim cells to river corridor.

    Maps HexSim river corridor cells to H3 hexagonal cells positioned along
    the actual river path (from shapefile) using downstream gradient values.

    Parameters
    ----------
    grid
        Grid data to visualize (values for coloring).
    downstream_grid
        Grid with downstream gradient values for positioning. If None, uses grid.
    colorscale
        Color mapping. Defaults to VIRIDIS.
    h3_resolution
        H3 resolution for output cells (9 ≈ 174m edge).
    max_cells
        Maximum cells to process (subsamples if exceeded).
    alpha
        Fill opacity (0-255).
    cross_river_offset
        Maximum cross-river offset in degrees.

    Returns
    -------
    dict
        Dict with keys: polygons (for PolygonLayer), river_paths, center, n_cells.
    """
    if downstream_grid is None:
        downstream_grid = grid

    # Get river centerline (uses shapefile if available)
    centerline = get_river_centerline()

    n_cells = len(grid.water_values)

    # Subsample if too many cells
    if n_cells > max_cells:
        sample_step = max(1, n_cells // max_cells)
        sample_idx = np.arange(0, n_cells, sample_step)
    else:
        sample_idx = np.arange(n_cells)

    # Get downstream range for normalization
    d_min = float(downstream_grid.water_values.min())
    d_max = float(downstream_grid.water_values.max())

    # Get cross-river range from centroid X values
    x_vals = downstream_grid.centroids[:, 1]
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    cross_river_range = x_max - x_min

    # Compute colors for sampled cells
    rgb = colorscale_rgb(grid.water_values[sample_idx], colorscale)

    # Build polygon data - map each cell to river position
    polygon_data = []
    for i, idx in enumerate(sample_idx):
        d_val = downstream_grid.water_values[idx]
        x_val = downstream_grid.centroids[idx, 1]

        # Get position along river (0 = start, 1 = end)
        t = _interpolate_river_position(d_val, d_min, d_max)

        # Get centerline coordinates from actual river shapefile
        center_lon, center_lat = centerline.get_coords(t)

        # Add cross-river offset perpendicular to river direction
        cross_t = (
            (x_val - x_min) / cross_river_range - 0.5 if cross_river_range > 0 else 0
        )
        perp_x, perp_y = centerline.get_perpendicular(t)
        final_lon = center_lon + cross_t * cross_river_offset * perp_x
        final_lat = center_lat + cross_t * cross_river_offset * perp_y

        # Get H3 cell and boundary
        h3_idx = h3.latlng_to_cell(final_lat, final_lon, h3_resolution)
        boundary = h3.cell_to_boundary(h3_idx)
        coords = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])  # Close polygon

        polygon_data.append(
            {
                "polygon": coords,
                "color": [int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]), alpha],
                "value": float(grid.water_values[idx]),
                "hex": h3_idx,
            }
        )

    # Deduplicate by H3 index (average colors for same cell)
    h3_to_data: dict[str, dict] = {}
    for p in polygon_data:
        h = p["hex"]
        if h not in h3_to_data:
            h3_to_data[h] = {"polygon": p["polygon"], "colors": [], "values": []}
        h3_to_data[h]["colors"].append(p["color"][:3])
        h3_to_data[h]["values"].append(p["value"])

    # Build final polygons with averaged colors
    final_polygons = []
    for h, data in h3_to_data.items():
        avg_color = np.mean(data["colors"], axis=0).astype(int).tolist() + [alpha]
        avg_value = float(np.mean(data["values"]))
        final_polygons.append(
            {
                "polygon": data["polygon"],
                "color": avg_color,
                "value": avg_value,
            }
        )

    # Simple river path for overlay (fallback using centerline points)
    simple_river_path = {"path": list(centerline.points), "color": [255, 100, 100, 255]}

    # Try to load detailed river paths from shapefile
    river_paths = load_river_shapefile()
    if not river_paths:
        river_paths = [simple_river_path]

    # Compute center from centerline
    if centerline.points:
        center_lon = np.mean([p[0] for p in centerline.points])
        center_lat = np.mean([p[1] for p in centerline.points])
    else:
        center_lon, center_lat = -121.5, 45.65

    return {
        "polygons": final_polygons,
        "river_path": simple_river_path,  # Legacy single path
        "river_paths": river_paths,  # All shapefile paths
        "center": (center_lon, center_lat),
        "n_cells": len(final_polygons),
        "n_original": n_cells,
    }


# ── Shiny-deckgl viewer class ────────────────────────────────────────────────


class HexGridViewer:
    """Interactive HexSim grid viewer using shiny-deckgl.

    This class provides a reusable viewer component that can be embedded
    in a Shiny app or run standalone.

    Parameters
    ----------
    workspace_dir
        Path to HexSim workspace directory.
    map_id
        ID for the MapWidget (default: "hexgrid_map").

    Example
    -------
    >>> viewer = HexGridViewer("Columbia [small]")
    >>> # List available grids
    >>> for name in viewer.grid_names:
    ...     print(name)
    >>> # Load and display a grid
    >>> await viewer.show_grid(session, "River [ depth ]")
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        map_id: str = "hexgrid_map",
        scale: float = 0.005,  # Larger scale for visible hexagons
    ):
        self.workspace_dir = Path(workspace_dir)
        self.map_id = map_id
        self.scale = scale
        self._grids = list_grids(workspace_dir)
        self._current_grid: HexGridData | None = None
        self._widget = None

    @property
    def grid_names(self) -> list[str]:
        """List of available grid layer names."""
        return [g.name for g in self._grids]

    def load(self, grid_name: str) -> HexGridData:
        """Load a grid layer by name."""
        self._current_grid = load_grid(self.workspace_dir, grid_name)
        return self._current_grid

    def create_widget(self):
        """Create the MapWidget. Call this before using in UI."""
        from shiny_deckgl import CARTO_POSITRON, MapWidget

        self._widget = MapWidget(
            self.map_id,
            view_state={"longitude": -121.5, "latitude": 47.0, "zoom": 7},
            style=CARTO_POSITRON,
            tooltip={
                "html": "Value: {value}",
                "style": {
                    "backgroundColor": "rgba(50, 50, 50, 0.9)",
                    "color": "#ffffff",
                    "fontSize": "12px",
                    "fontFamily": "monospace",
                    "borderRadius": "4px",
                    "padding": "6px 10px",
                },
            },
            controller=True,
        )
        return self._widget

    @property
    def widget(self):
        """Get the MapWidget (creates if needed)."""
        if self._widget is None:
            self.create_widget()
        return self._widget

    async def show_grid(
        self,
        session,
        grid_name: str,
        colorscale: list | None = None,
        h3_resolution: int = 9,
        show_river_centerline: bool = True,
    ):
        """Load and display a grid on the map along the Columbia River corridor.

        Maps HexSim cells to H3 hexagonal cells positioned along the actual
        river path using downstream gradient values. Multiple HexSim cells
        mapping to the same H3 cell have their values averaged.

        Parameters
        ----------
        session
            Shiny session object.
        grid_name
            Name of the grid layer to display.
        colorscale
            Color mapping (default: auto-select based on grid name).
        h3_resolution
            H3 resolution (7-10). Higher = smaller cells, more detail.
            9 ≈ 174m edge (default), 10 ≈ 66m edge.
        show_river_centerline
            Whether to show river centerline overlay (default: True).
        """
        from shiny_deckgl import map_view, path_layer, polygon_layer

        # Load the grid for values
        grid = self.load(grid_name)

        # Load downstream gradient for positioning (if different from display grid)
        if "downstream" not in grid_name.lower():
            try:
                downstream_grid = load_grid(
                    self.workspace_dir, "Gradient [ downstream ]"
                )
            except FileNotFoundError:
                downstream_grid = grid
        else:
            downstream_grid = grid

        # Auto-select colorscale based on grid name
        if colorscale is None:
            name_lower = grid_name.lower()
            if "depth" in name_lower:
                colorscale = BATHYMETRIC
            elif "temp" in name_lower:
                colorscale = THERMAL
            else:
                colorscale = VIRIDIS

        # Build river corridor data
        corridor_data = build_river_corridor_data(
            grid,
            downstream_grid=downstream_grid,
            colorscale=colorscale,
            h3_resolution=h3_resolution,
        )

        print(
            f"Created river corridor layer: {corridor_data['n_cells']} H3 cells "
            f"from {corridor_data['n_original']} HexSim cells"
        )

        # Build layers
        layers = []

        # Hex grid layer
        layers.append(
            polygon_layer(
                "hexgrid",
                data=corridor_data["polygons"],
                getPolygon="@@=d.polygon",
                getFillColor="@@=d.color",
                getLineColor=[50, 50, 50, 150],
                lineWidthMinPixels=1,
                filled=True,
                stroked=True,
                pickable=True,
            )
        )

        # River centerline overlay (use shapefile paths if available)
        if show_river_centerline:
            river_paths = corridor_data.get(
                "river_paths", [corridor_data["river_path"]]
            )
            layers.append(
                path_layer(
                    "river",
                    data=river_paths,
                    getPath="@@=d.path",
                    getColor="@@=d.color",
                    widthMinPixels=2,
                    pickable=True,
                )
            )

        center_lon, center_lat = corridor_data["center"]

        await self.widget.update(
            session,
            layers=layers,
            view_state={
                "longitude": center_lon,
                "latitude": center_lat,
                "zoom": 10,
                "pitch": 0,
            },
            views=[map_view(controller=True)],
        )


# ── Standalone Shiny app ─────────────────────────────────────────────────────


def create_app(workspace_dir: str | Path):
    """Create a standalone Shiny app for viewing HexSim grids.

    Parameters
    ----------
    workspace_dir
        Path to HexSim workspace directory.

    Returns
    -------
    shiny.App
        Configured Shiny application.
    """
    from shiny import App, reactive, render, ui
    from shiny_deckgl import head_includes

    viewer = HexGridViewer(workspace_dir)
    viewer.create_widget()

    app_ui = ui.page_sidebar(
        ui.sidebar(
            ui.input_select(
                "grid_select",
                "Select Grid Layer:",
                choices={name: name for name in viewer.grid_names},
                selected=viewer.grid_names[0] if viewer.grid_names else None,
            ),
            ui.input_select(
                "colorscale",
                "Colorscale:",
                choices={
                    "auto": "Auto (based on layer)",
                    "viridis": "Viridis",
                    "bathymetric": "Bathymetric",
                    "thermal": "Thermal",
                },
                selected="auto",
            ),
            ui.input_checkbox(
                "show_river",
                "Show River Centerline",
                value=True,
            ),
            ui.hr(),
            ui.output_text("grid_info"),
            width=280,
        ),
        head_includes(),
        ui.div(
            viewer.widget.ui(height="calc(100vh - 100px)"),
            style="padding: 10px;",
        ),
        title="HexSim Grid Viewer",
    )

    def server(input, output, session):
        @reactive.effect
        @reactive.event(input.grid_select, input.colorscale, input.show_river)
        async def _update_grid():
            grid_name = input.grid_select()
            if not grid_name:
                return

            # Get colorscale
            cs_choice = input.colorscale()
            if cs_choice == "viridis":
                colorscale = VIRIDIS
            elif cs_choice == "bathymetric":
                colorscale = BATHYMETRIC
            elif cs_choice == "thermal":
                colorscale = THERMAL
            else:
                colorscale = None  # auto

            await viewer.show_grid(
                session,
                grid_name,
                colorscale=colorscale,
                show_river_centerline=input.show_river(),
            )

        @render.text
        def grid_info():
            grid = viewer._current_grid
            if grid is None:
                return "No grid loaded"
            return (
                f"Grid: {grid.ncols} x {grid.nrows}\n"
                f"Water cells: {len(grid.water_values):,}\n"
                f"Value range: {grid.water_values.min():.2f} - {grid.water_values.max():.2f}"
            )

    return App(app_ui, server)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Default workspace path
    default_ws = r"Columbia [small]"

    if len(sys.argv) > 1:
        workspace = sys.argv[1]
    else:
        workspace = default_ws

    print(f"Starting HexSim Grid Viewer for: {workspace}")
    print("Open http://localhost:8000 in your browser")

    app = create_app(workspace)
    app.run(port=8000)
