"""Build a Curonian Lagoon TriMesh-format NetCDF.

Produces the regular-lat/lon-grid NetCDF that
``salmon_ibm.mesh.TriMesh.from_netcdf`` consumes: 2-D ``lat``, ``lon``,
``mask``, and ``depth`` arrays.  Delaunay triangulation happens at
load time inside HexSimPy.

Geographic coverage matches the inSTREAM example_baltic tutorial
(``configs/example_baltic.yaml`` over there) — the Nemunas Delta +
Curonian Lagoon + Baltic-coast strip — clipped to the same bbox the
H3 backend uses so we can reuse the cached EMODnet TIF and water
polygon union.

Resolution defaults to 100 m — fine enough to actually resolve the
Nemunas Delta channel network:
  * Nemunas main channel (~600 m wide) → 5-6 cells across
  * Atmata, Minija (~150-250 m)        → 2-3 cells across
  * Šyša, Skirvytė (~100 m)            → 1-cell threads (resolved)

Approximate node count over the bbox: 1000 × 833 ≈ 833 k nodes.
Delaunay triangulation and the per-node polygon-in-water test together
take ~60-90 s on a laptop.  The resulting NetCDF is ~25 MB; ~half the
nodes are water after the mask, giving roughly 400 k water triangles
that the simulation map's ScatterplotLayer renders with binary
encoding.

Coarser fallback for headless / smoke-testing: ``--resolution 400``
gives ~58 k nodes (delta channels not resolved; Curonian Lagoon
proper still rendered cleanly).

Output: ``data/curonian_trimesh_landscape.nc``
  - lat   (y, x) float32 — node latitude (deg)
  - lon   (y, x) float32 — node longitude (deg)
  - mask  (y, x) uint8   — 1 = water node (depth > 0 ∧ inside OSM∪NE)
  - depth (y, x) float32 — EMODnet depth at node, m, positive down
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  — registers .rio accessor on xarray
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay
from shapely.geometry import Point
from shapely.strtree import STRtree

from _water_polygons import get_water_union


# Same bbox as the H3 landscape — the EMODnet TIF and CMEMS forcing
# are cached for this region.  inSTREAM example_baltic uses a slightly
# wider bbox (out to 22.08°E) but the eastward extension only adds
# upstream river segments that aren't part of HexSimPy's delta-and-sea
# domain.
BBOX = {"minlat": 54.9, "maxlat": 55.8, "minlon": 20.4, "maxlon": 21.9}
DEFAULT_RES_M = 100.0
PROJECT = Path(__file__).resolve().parent.parent


def build_grid(bbox: dict, res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) 2-D arrays at ``res_m`` metres node spacing.

    Spacing is constant in metres — converted to degrees per axis so
    the grid is square at the bbox-centre latitude.  Off-centre cells
    are slightly stretched in degrees but still ~``res_m`` wide on the
    ground.
    """
    mid_lat = 0.5 * (bbox["minlat"] + bbox["maxlat"])
    dlat_deg = res_m / 111_000.0
    dlon_deg = res_m / (111_000.0 * np.cos(np.radians(mid_lat)))

    lats_1d = np.arange(bbox["minlat"], bbox["maxlat"] + dlat_deg / 2, dlat_deg)
    lons_1d = np.arange(bbox["minlon"], bbox["maxlon"] + dlon_deg / 2, dlon_deg)
    lon2d, lat2d = np.meshgrid(lons_1d, lats_1d)

    print(f"  grid: {lat2d.shape[0]} × {lat2d.shape[1]} = "
          f"{lat2d.size:,} nodes "
          f"(dlat={dlat_deg:.5f}°, dlon={dlon_deg:.5f}°)")
    return lat2d.astype(np.float32), lon2d.astype(np.float32)


def sample_emodnet_at_nodes(
    lat2d: np.ndarray, lon2d: np.ndarray, tif_path: Path
) -> np.ndarray:
    """Sample EMODnet elevation per node; flip to depth.

    Same convention as ``scripts/build_nemunas_h3_landscape.py``: NaN
    on land becomes depth=0; positive elevations (above sea level)
    also become 0 after the ``np.maximum`` clamp.  Negative
    elevations (below sea level) become positive depth.
    """
    raw = rioxarray.open_rasterio(tif_path).squeeze()
    x, y, z = raw.x.values, raw.y.values, raw.values
    if y[0] > y[-1]:
        y, z = y[::-1], z[::-1, :]
    interp = RegularGridInterpolator(
        (y, x), z, method="linear", bounds_error=False, fill_value=np.nan,
    )
    query = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    elev = interp(query)
    depth = np.where(np.isnan(elev), 0.0, -elev)
    depth = np.maximum(depth, 0.0).astype(np.float32)
    return depth.reshape(lat2d.shape)


def polygon_water_mask(
    lat2d: np.ndarray, lon2d: np.ndarray, buffer_deg: float = 0.001
) -> np.ndarray:
    """Per-node True if inside the OSM ∪ NE water union (buffered).

    Uses shapely 2.x's vectorised ``contains_xy`` for speed — the
    Python-loop version was unacceptably slow at 948 k nodes (the
    H3 build at 106 k nodes was tolerable; that's a 9× growth that
    pushes the loop past 5 minutes).
    """
    import shapely
    union = get_water_union()
    if buffer_deg > 0:
        union = union.buffer(buffer_deg)
    flat_lon = lon2d.ravel().astype(np.float64)
    flat_lat = lat2d.ravel().astype(np.float64)
    # contains_xy operates on the prepared union; takes lon, lat (x, y).
    mask_flat = shapely.contains_xy(union, flat_lon, flat_lat).astype(np.uint8)
    return mask_flat.reshape(lat2d.shape)


def write_forcing_stub(
    out_path: Path,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    n_times: int = 30,
) -> None:
    """Write a constant-field forcing stub matching the landscape grid.

    ``salmon_ibm.environment.Environment`` indexes forcing arrays by
    ``mesh.triangles`` after a ``data.reshape(n_times, -1)``, which
    means the forcing's flattened spatial dimension must equal the
    mesh's node count.  When the user runs this scenario for real,
    they replace the stub with regridded CMEMS via
    ``scripts/fetch_cmems_forcing.py``; the stub keeps the simulation
    bootable end-to-end without that download.

    Constants chosen to match Curonian Lagoon climatology:
      tos = 12 °C, sos = 5 PSU (brackish lagoon), uo = vo = 0,
      zos (SSH) = 0.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shape2d = lat2d.shape  # (y, x)
    times = np.arange(n_times, dtype=np.float64)  # hours

    def const(value, dtype=np.float32):
        return np.full((n_times,) + shape2d, value, dtype=dtype)

    ds = xr.Dataset(
        {
            "tos": (("time", "y", "x"), const(12.0)),
            "sos": (("time", "y", "x"), const(5.0)),
            "uo":  (("time", "y", "x"), const(0.0)),
            "vo":  (("time", "y", "x"), const(0.0)),
            "zos": (("time", "y", "x"), const(0.0)),
        },
        coords={"time": times},
        attrs={
            "title": "Curonian TriMesh forcing stub — constant fields",
            "note": (
                "Replace with regridded CMEMS via "
                "scripts/fetch_cmems_forcing.py for real runs."
            ),
            "n_times": n_times,
            "grid_shape": f"{shape2d[0]} × {shape2d[1]}",
        },
    )
    ds.to_netcdf(out_path, format="NETCDF3_CLASSIC", engine="scipy")


def write_trimesh_nc(
    out_path: Path,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    mask2d: np.ndarray,
    depth2d: np.ndarray,
    res_m: float,
    bbox: dict,
) -> None:
    """Write the landscape NetCDF in NETCDF3 format (engine=scipy).

    TriMesh.from_netcdf opens with engine="scipy" so the file *must*
    be NetCDF-3.  Shape convention: dims are (y, x); triangulation
    cache uses (n_tri, 3) on a separate ``tri`` dimension.

    We embed the Delaunay triangulation result here so the runtime
    open in TriMesh.from_netcdf can skip the ~60 s Delaunay step.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  computing Delaunay triangulation (cached for fast load)…")
    nodes = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    tri = Delaunay(nodes)
    triangles = tri.simplices.astype(np.int32)
    neighbors = tri.neighbors.astype(np.int32)
    print(f"  {len(triangles):,} triangles + neighbour table cached")

    ds = xr.Dataset(
        {
            "lat":   (("y", "x"), lat2d.astype(np.float32)),
            "lon":   (("y", "x"), lon2d.astype(np.float32)),
            "mask":  (("y", "x"), mask2d.astype(np.float32)),
            "depth": (("y", "x"), depth2d.astype(np.float32)),
            "triangles": (("n_tri", "tri_corner"), triangles),
            "neighbors": (("n_tri", "tri_corner"), neighbors),
        },
        attrs={
            "title": "Curonian Lagoon TriMesh — Nemunas Delta + Baltic strip",
            "bbox": (
                f"{bbox['minlon']:.2f}-{bbox['maxlon']:.2f}E, "
                f"{bbox['minlat']:.2f}-{bbox['maxlat']:.2f}N"
            ),
            "node_spacing_m": res_m,
            "grid_shape": f"{lat2d.shape[0]} × {lat2d.shape[1]}",
            "n_water_nodes": int(mask2d.sum()),
            "source_bathymetry": "EMODnet DTM 2022 (CC-BY 4.0 EMODnet)",
            "source_coastline": "OSM Overpass + Natural Earth 1:10m ocean",
            "geographic_reference": "inSTREAM example_baltic (configs/example_baltic.yaml)",
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "license": "CC-BY 4.0 (EMODnet, Natural Earth, OpenStreetMap ODbL)",
        },
    )
    ds.to_netcdf(out_path, format="NETCDF3_CLASSIC", engine="scipy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tif", default="data/curonian_bathymetry_raw.tif",
        help="EMODnet bathymetry GeoTIFF",
    )
    parser.add_argument(
        "--out", default="data/curonian_trimesh_landscape.nc",
        help="output NetCDF path",
    )
    parser.add_argument(
        "--resolution", type=float, default=DEFAULT_RES_M,
        help=f"node spacing in metres (default {DEFAULT_RES_M:.0f})",
    )
    args = parser.parse_args()

    tif_path = PROJECT / args.tif
    out_path = PROJECT / args.out

    print(f"[1/4] Building regular grid @ {args.resolution:.0f} m…")
    lat2d, lon2d = build_grid(BBOX, args.resolution)

    print(f"[2/4] Sampling EMODnet bathymetry at each node…")
    depth2d = sample_emodnet_at_nodes(lat2d, lon2d, tif_path)
    n_pos = int((depth2d > 0).sum())
    pct = 100.0 * n_pos / depth2d.size
    print(f"  depth range: 0 – {depth2d.max():.2f} m, "
          f"{n_pos:,}/{depth2d.size:,} nodes have depth>0 ({pct:.1f}%)")

    print(f"[3/4] Computing polygon-based water mask…")
    mask2d = polygon_water_mask(lat2d, lon2d)
    # Final mask = depth>0 AND inside polygon — same convention as H3
    final_mask = ((depth2d > 0) & (mask2d > 0)).astype(np.uint8)
    n_water = int(final_mask.sum())
    print(f"  water nodes: {n_water:,}/{final_mask.size:,} "
          f"({100*n_water/final_mask.size:.1f}%)")

    print(f"[4/5] Writing {out_path}…")
    write_trimesh_nc(
        out_path, lat2d, lon2d, final_mask, depth2d, args.resolution, BBOX,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {final_mask.size:,} nodes ({size_mb:.1f} MB)")

    forcing_path = out_path.parent / "curonian_trimesh_forcing_stub.nc"
    print(f"[5/5] Writing matching forcing stub → {forcing_path.name}…")
    # n_times=2 keeps the file small (~75 MB at 100 m grid; 30 timesteps
    # would be ~1.1 GB).  Two timesteps suffice for the simulation
    # because Environment cycles via t % n_timesteps and the values
    # are constant — only SSH delta needs ≥2 frames.
    write_forcing_stub(forcing_path, lat2d, lon2d, n_times=2)
    fsize_mb = forcing_path.stat().st_size / 1e6
    print(f"  wrote {fsize_mb:.1f} MB stub (constant tos=12°C, sos=5 PSU)")
    print()
    print(f"Verify with:")
    print(f"  micromamba run -n shiny python -c \"")
    print(f"    from salmon_ibm.mesh import TriMesh; "
          f"m = TriMesh.from_netcdf('{out_path.relative_to(PROJECT)}'); "
          f"print(f'{{m.n_triangles:,}} triangles, "
          f"{{int(m.water_mask.sum()):,}} water')\"")


if __name__ == "__main__":
    main()
