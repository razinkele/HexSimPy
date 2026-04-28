"""H3 tessellation primitives + upload-flow entry points.

Extracted from scripts/build_h3_multires_landscape.py (v1.6.1). The
build script will become a thin wrapper after Task 4 of plan
2026-04-28-create-model-feature.md. Adding new entry points here
(parse_upload, preview, _fetch_emodnet_for_bbox, PreviewMesh) supports
the Create Model viewer-only feature without coupling viewer code to
script imports.
"""
from __future__ import annotations

import h3
import numpy as np
from shapely.geometry import MultiPolygon


def tessellate_reach(polygon, resolution: int) -> list[str]:
    """Return the H3 cells tessellating a buffered version of polygon.

    Buffer = half the H3 edge length at the target resolution.  This
    ensures cells whose centroid sits within half-a-cell of the
    polygon edge are included — without it, a thin meandering river
    polygon at res 10 (~75 m cells) tessellates as a sparse, often
    disconnected, cell set because polygon_to_cells uses a strict
    centroid-in-polygon test.

    Buffer is applied in degrees: convert metres → degrees at lat 55°
    (the bbox centre) using 1 deg ≈ 111 km.

    For multi-polygon inputs, each part is buffered + tessellated
    independently and the cell sets are unioned.
    """
    edge_m = h3.average_hexagon_edge_length(resolution, unit="m")
    buffer_deg = (edge_m / 2.0) / 111_000.0

    if isinstance(polygon, MultiPolygon):
        parts = list(polygon.geoms)
    else:
        parts = [polygon]

    cells: set[str] = set()
    for part in parts:
        if part.is_empty:
            continue
        buffered = part.buffer(buffer_deg)
        if hasattr(buffered, "geoms"):
            inner_parts = list(buffered.geoms)
        else:
            inner_parts = [buffered]
        for ip in inner_parts:
            if ip.is_empty:
                continue
            ext = [(y, x) for x, y in ip.exterior.coords]
            holes = [
                [(y, x) for x, y in interior.coords]
                for interior in ip.interiors
            ]
            try:
                poly = h3.LatLngPoly(ext, *holes)
            except Exception as e:
                print(f"  ! skipping polygon part: {e}")
                continue
            for c in h3.polygon_to_cells(poly, resolution):
                cells.add(c)
    return sorted(cells)


def bridge_components(
    cells: list[str], resolution: int, max_bridge_len: int = 10
) -> list[str]:
    """Merge disconnected H3 components into a single graph by adding
    shortest-path cells between adjacent components.

    See scripts/build_h3_multires_landscape.py history for the full
    rationale (v1.5.0 fix; preserved here verbatim).
    """
    if not cells or len(cells) < 2:
        return cells
    cell_set = set(cells)

    # BFS within cell_set using grid_ring(1) to find components.
    components: list[set[str]] = []
    seen: set[str] = set()
    for start in cell_set:
        if start in seen:
            continue
        comp: set[str] = set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            comp.add(c)
            for nb in h3.grid_ring(c, 1):
                if nb in cell_set and nb not in seen:
                    stack.append(nb)
        components.append(comp)

    if len(components) <= 1:
        return cells

    components.sort(key=len, reverse=True)
    anchor = components[0]
    bridges_added = 0
    for orphan in components[1:]:
        best_dist = max_bridge_len + 1
        best_pair = None
        for a in orphan:
            for b in anchor:
                d = h3.grid_distance(a, b)
                if d < best_dist:
                    best_dist = d
                    best_pair = (a, b)
                    if d == 1:
                        break
            if best_pair and best_dist == 1:
                break
        if best_pair is None or best_dist > max_bridge_len:
            print(
                f"  ! orphan component of {len(orphan)} cells too far "
                f"from anchor (>{max_bridge_len} cells) — leaving "
                f"disconnected"
            )
            continue
        path = list(h3.grid_path_cells(best_pair[0], best_pair[1]))
        for pc in path:
            if pc not in cell_set:
                cell_set.add(pc)
                bridges_added += 1

    if bridges_added:
        print(f"  bridge-cell pass: added {bridges_added} cells across "
              f"{len(components) - 1} component gaps")
    return sorted(cell_set)


def polygon_trust_water_mask(
    reach_id_arr: np.ndarray,
    water_mask: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Force water_mask=1 and depth>=1.0 where reach_id != -1.

    The v1.6.1 fix (commit d187ac9): tessellate_reach buffers each
    polygon by half-a-cell-edge to ensure narrow channels get cells.
    The buffer captures cells whose centroids sit up to ~14 m beyond
    the polygon edge — EMODnet reports many of those as dry.  Without
    this override, ~16% of tagged cells (47-87% of river cells) end
    up with reach_id != -1 but water_mask=0; the viewer drops them
    silently.

    Returns (new_water_mask, new_depth) — both arrays modified.

    See tests/test_h3_grid_quality.py::test_reach_id_implies_water_mask.
    """
    forced = (reach_id_arr != -1) & (water_mask == 0)
    new_water = np.where(forced, np.uint8(1), water_mask).astype(np.uint8)
    new_depth = np.where(forced & (depth < 1.0),
                         np.float32(1.0), depth).astype(np.float32)
    return new_water, new_depth


import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import shapely
from shapely.ops import unary_union


@dataclass
class PreviewMesh:
    """Duck-typed mesh produced by `preview()` for the Create Model
    feature. The existing rendering machinery in app.py reads these
    attributes to render hex layers; no subclass of H3MultiResMesh
    required.
    """
    h3_ids: np.ndarray
    resolutions: np.ndarray
    centroids: np.ndarray
    reach_id: np.ndarray
    reach_names: list[str]
    depth: np.ndarray
    water_mask: np.ndarray
    polygon_outlines: list

    def __post_init__(self):
        n = len(self.h3_ids)
        for name in ("resolutions", "centroids", "reach_id",
                     "depth", "water_mask"):
            arr = getattr(self, name)
            assert len(arr) == n, (
                f"PreviewMesh.{name} has length {len(arr)}, expected {n}"
            )


def parse_upload(file_bytes: bytes, suffix: str):
    """Read shp.zip / gpkg / geojson → dissolve → reproject to WGS84.

    Returns a single (Multi)Polygon geometry. Raises ValueError on
    unparseable input, missing CRS, no geometry, non-polygon features,
    or antimeridian crossing.

    suffix: one of ".geojson", ".gpkg", ".shp.zip" (case-insensitive).
    """
    suffix = suffix.lower()
    if suffix == ".geojson":
        gdf = gpd.read_file(io.BytesIO(file_bytes))
    elif suffix == ".gpkg":
        # GPKG is a SQLite container; gpd.read_file accepts BytesIO directly.
        gdf = gpd.read_file(io.BytesIO(file_bytes), layer=0)
    elif suffix == ".shp.zip":
        gdf = _read_shp_zip(file_bytes)
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    return _dissolve_and_validate(gdf)


def preview(
    polygon,
    resolution: int,
    *,
    with_bathy: bool = False,
    max_cells: int = 1_000_000,
) -> PreviewMesh:
    """Tessellate polygon at H3 resolution → PreviewMesh ready to render.

    Geometry-only when with_bathy=False (depth all zero, water_mask all 1).
    See Task 13 for the with_bathy=True branch.

    Raises ValueError if would-be cell count exceeds max_cells.
    """
    cells = tessellate_reach(polygon, resolution)
    if not cells:
        raise ValueError(
            f"Polygon is smaller than a single H3 cell at res {resolution}. "
            f"Try a finer resolution."
        )
    if len(cells) > max_cells:
        raise ValueError(
            f"Tessellation at res {resolution} would produce "
            f"{len(cells):,} cells (max {max_cells:,}). "
            f"Pick a coarser resolution or smaller polygon."
        )
    cells = bridge_components(cells, resolution)
    if len(cells) > max_cells:
        raise ValueError(
            f"Tessellation at res {resolution} produced {len(cells):,} cells "
            f"after bridging (max {max_cells:,}). "
            f"Pick a coarser resolution or smaller polygon."
        )

    n = len(cells)
    h3_ids = np.array([h3.str_to_int(c) for c in cells], dtype=np.uint64)
    centroids = np.array(
        [h3.cell_to_latlng(c) for c in cells], dtype=np.float64
    )
    resolutions = np.full(n, resolution, dtype=np.int8)
    reach_id = np.zeros(n, dtype=np.int8)
    reach_names = ["uploaded_polygon"]
    water_mask = np.ones(n, dtype=np.uint8)
    depth = np.zeros(n, dtype=np.float32)

    if isinstance(polygon, MultiPolygon):
        parts = list(polygon.geoms)
    else:
        parts = [polygon]
    polygon_outlines = []
    for part in parts:
        if part.is_empty:
            continue
        polygon_outlines.append(
            [[float(x), float(y)] for x, y in part.exterior.coords]
        )

    return PreviewMesh(
        h3_ids=h3_ids,
        resolutions=resolutions,
        centroids=centroids,
        reach_id=reach_id,
        reach_names=reach_names,
        depth=depth,
        water_mask=water_mask,
        polygon_outlines=polygon_outlines,
    )


import hashlib

_EMODNET_WCS_URL = (
    "https://ows.emodnet-bathymetry.eu/wcs"
    "?service=WCS&version=2.0.1&request=GetCoverage"
    "&CoverageId=emodnet:mean"
    "&format=image/tiff"
)
_EMODNET_CACHE_DIR = Path(".superpowers/cache")


def _fetch_emodnet_for_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Fetch EMODnet bathymetry for the bbox. Cached to .superpowers/cache/.

    Returns (depth_grid_2d, bbox_actual). Depth values are POSITIVE
    metres; cells outside coverage are NaN.

    Raises requests.RequestException on network failure (caller should
    catch and surface a UI warning).
    """
    import requests
    import rasterio

    _EMODNET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(repr(bbox).encode()).hexdigest()[:16]
    cache_path = _EMODNET_CACHE_DIR / f"emodnet_{key}.tif"
    if not cache_path.exists():
        minlon, minlat, maxlon, maxlat = bbox
        url = (
            f"{_EMODNET_WCS_URL}"
            f"&subset=Long({minlon},{maxlon})"
            f"&subset=Lat({minlat},{maxlat})"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    with rasterio.open(cache_path) as src:
        elev = src.read(1).astype(np.float32)
        depth = np.where(np.isnan(elev) | (elev > 0), np.nan, -elev)
    return depth, bbox


def _read_shp_zip(file_bytes: bytes) -> gpd.GeoDataFrame:
    """Extract a zipped shapefile bundle to a temp dir and read it.

    The zip must contain the .shp + sidecar files (.dbf, .shx, .prj).
    Raises ValueError if .dbf or .shx are missing."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            members = zf.namelist()
            exts = {Path(m).suffix.lower() for m in members}
            if ".shp" not in exts:
                raise ValueError("Zip does not contain a .shp file.")
            if ".dbf" not in exts or ".shx" not in exts:
                raise ValueError(
                    "Shapefile bundle missing .dbf or .shx — re-export "
                    "the full bundle."
                )
            zf.extractall(tmpdir)
        shp_files = list(Path(tmpdir).glob("**/*.shp"))
        if not shp_files:
            raise ValueError("No .shp file found after extraction.")
        return gpd.read_file(shp_files[0])


def _dissolve_and_validate(gdf: gpd.GeoDataFrame):
    """Common post-read processing: dissolve, reproject, validate."""
    if gdf.empty or gdf.geometry.is_empty.all():
        raise ValueError("File contains no polygon geometry.")
    if gdf.crs is None:
        raise ValueError(
            "File has no CRS — please re-export with a defined coordinate system."
        )
    if not all(g.geom_type in ("Polygon", "MultiPolygon")
               for g in gdf.geometry if not g.is_empty):
        raise ValueError("File must contain Polygon or MultiPolygon features.")
    gdf_wgs = gdf.to_crs("EPSG:4326")
    geom = unary_union([g for g in gdf_wgs.geometry if not g.is_empty])
    if geom.is_empty:
        raise ValueError("File contains no polygon geometry.")
    if not geom.is_valid:
        from shapely.validation import make_valid
        geom = make_valid(geom)
    minx, _, maxx, _ = geom.bounds
    if maxx - minx > 180:
        raise ValueError(
            "Polygons crossing the antimeridian aren't supported. "
            "Split into eastern and western parts."
        )
    return geom
