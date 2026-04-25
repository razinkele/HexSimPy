"""Build a Nemunas Delta H3-native landscape from CMEMS + EMODnet.

One-shot: bbox polygon → H3 cells (default res 9) → sample EMODnet
bathymetry + CMEMS forcing at each cell centroid → write a NetCDF
that ``salmon_ibm.h3_env.H3Environment.from_netcdf`` consumes.

Output: ``data/nemunas_h3_landscape.nc`` with variables:
    - h3_id      (cell,)        uint64    — H3 cell index
    - lat, lon   (cell,)        float64   — cell centroid
    - depth      (cell,)        float32   — EMODnet, positive down (m)
    - water_mask (cell,)        uint8     — 1 = (depth > 0) ∧ inside
                                              an OSM/NE water polygon.
                                              The polygon AND-mask is
                                              required because EMODnet
                                              reports negative
                                              elevation across
                                              below-sea-level land
                                              (Nemunas Delta polders),
                                              which would otherwise
                                              leak in as 'water'.
    - tos        (time, cell)   float32   — sea-surface temperature (°C)
    - sos        (time, cell)   float32   — salinity (PSU)
    - uo, vo     (time, cell)   float32   — currents (m/s) east / north

Phase 2.1 of ``docs/superpowers/plans/2026-04-24-h3mesh-backend.md``.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import h3
import numpy as np
import rioxarray  # noqa: F401  — registers .rio accessor on xarray
import xarray as xr
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
from shapely.geometry import Point
from shapely.strtree import STRtree

from _water_polygons import get_water_union


BBOX = {"minlon": 20.4, "maxlon": 21.9, "minlat": 54.9, "maxlat": 55.8}
DEFAULT_RESOLUTION = 9


# ---------------------------------------------------------------------------
# Stage 1: build H3 cells over the bbox polygon
# ---------------------------------------------------------------------------


def build_h3_cells(bbox: dict, resolution: int) -> list[str]:
    """Tessellate the bbox polygon into H3 cells at the given resolution."""
    ring = [
        (bbox["minlat"], bbox["minlon"]),
        (bbox["minlat"], bbox["maxlon"]),
        (bbox["maxlat"], bbox["maxlon"]),
        (bbox["maxlat"], bbox["minlon"]),
        (bbox["minlat"], bbox["minlon"]),
    ]
    poly = h3.LatLngPoly(ring)
    return list(h3.polygon_to_cells(poly, resolution))


# ---------------------------------------------------------------------------
# Stage 2: sample EMODnet bathymetry at each H3 cell centroid
# ---------------------------------------------------------------------------


def sample_emodnet(cells: list[str], tif_path: Path) -> np.ndarray:
    """Sample EMODnet elevation at each H3 cell centroid; flip to depth.

    EMODnet stores elevation (positive up); we need depth (positive down).
    NaN cells (land) become depth = 0.
    """
    raw = rioxarray.open_rasterio(tif_path).squeeze()
    x, y, z = raw.x.values, raw.y.values, raw.values
    if y[0] > y[-1]:
        y, z = y[::-1], z[::-1, :]
    interp = RegularGridInterpolator(
        (y, x), z, method="linear", bounds_error=False, fill_value=np.nan,
    )
    lats, lons = zip(*(h3.cell_to_latlng(c) for c in cells))
    query = np.column_stack([lats, lons])
    elev = interp(query)
    depth = np.where(np.isnan(elev), 0.0, -elev)
    return np.maximum(depth, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 3: sample CMEMS forcing (regular lat/lon grid) at each H3 centroid
# ---------------------------------------------------------------------------


def sample_cmems(
    cells: list[str],
    cmems_path: Path,
    start: str | None = None,
    end: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sample CMEMS thetao/so/uo/vo at each H3 cell centroid.

    Returns ``(time_array, {var_name: (time, cell) float32})``.
    The optional ``start``/``end`` ISO dates subset CMEMS in time so
    output stays manageable.
    """
    raw = xr.open_dataset(cmems_path)
    if start is not None or end is not None:
        raw = raw.sel(time=slice(start, end))
        if raw.sizes["time"] == 0:
            raise ValueError(f"no CMEMS timesteps in [{start}, {end}]")

    lat_src = raw["latitude"].values
    lon_src = raw["longitude"].values
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        raw = raw.isel(latitude=slice(None, None, -1))

    lats, lons = zip(*(h3.cell_to_latlng(c) for c in cells))
    query = np.column_stack([lats, lons])
    n_time = raw.sizes["time"]
    n_cells = len(cells)

    # Pre-compute source meshgrid once for the NaN-fill nearest-neighbour
    # path; avoids rebuilding per timestep.
    src_lats_2d, src_lons_2d = np.meshgrid(lat_src, lon_src, indexing="ij")

    # CMEMS source-var → spec-name in the landscape NetCDF.
    # The loader (H3Environment) maps these spec names to the canonical
    # event keys (temperature, salinity, u_current, v_current).
    var_map = [("thetao", "tos"), ("so", "sos"), ("uo", "uo"), ("vo", "vo")]
    out: dict[str, np.ndarray] = {}
    for src, dst in var_map:
        if src not in raw:
            print(f"  ! source var {src} missing — skipping")
            continue
        arr = raw[src].squeeze().values  # (time, lat, lon) after squeeze of depth=1
        vals = np.empty((n_time, n_cells), dtype=np.float32)
        for t in range(n_time):
            src_t = arr[t]
            interp = RegularGridInterpolator(
                (lat_src, lon_src), src_t,
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            row = interp(query)
            # NaN-fill via nearest-neighbour on the source grid.  Same trick
            # the Curonian regridder uses (scripts/fetch_cmems_forcing.py)
            # to keep the thermal-kill path from killing every agent
            # because a CMEMS land cell leaks NaN into a water cell.
            nan_mask = np.isnan(row)
            if nan_mask.any():
                src_flat = src_t.ravel()
                src_valid = ~np.isnan(src_flat)
                if src_valid.any():
                    nn = NearestNDInterpolator(
                        np.column_stack([
                            src_lats_2d.ravel()[src_valid],
                            src_lons_2d.ravel()[src_valid],
                        ]),
                        src_flat[src_valid],
                    )
                    row[nan_mask] = nn(query[nan_mask])
            vals[t] = row.astype(np.float32)
        out[dst] = vals
        n_nan_day0 = int(np.isnan(out[dst][0]).sum())
        print(
            f"  ✓ {src} → {dst} (time={n_time}, cells={n_cells}, "
            f"NaN after fill day 0: {n_nan_day0})"
        )
    return raw["time"].values, out


# ---------------------------------------------------------------------------
# Stage 3.5: clip the depth>0 mask to authoritative water polygons
# ---------------------------------------------------------------------------


def polygon_water_mask(
    lats: np.ndarray, lons: np.ndarray, buffer_deg: float = 0.001
) -> np.ndarray:
    """For each (lat, lon) centroid, True if it lies inside the OSM ∪ NE
    water polygon union — buffered outward by ``buffer_deg`` first.

    A naive centroid-in-polygon test rejects cells whose centre sits a
    few tens of metres past a riverbank, even though most of the H3 hex
    is still in-water.  At res-9 cells are ~200 m across, and OSM river
    polygons hug the bank tightly, so without buffering a 200-300 m wide
    river renders as broken dots — the user-visible "scattered hexes"
    bug.

    Buffering the polygon by 0.001° (~110 m at lat 55°) — roughly half
    a cell edge — recaptures bank-adjacent cells without re-admitting
    polderland 5-10 km from any water (the original false-positive set).
    """
    union = get_water_union()
    if buffer_deg > 0:
        union = union.buffer(buffer_deg)
    # The union is typically a MultiPolygon; STRtree wants a flat list of
    # polygons.  ``.geoms`` returns the components for both Polygon and
    # MultiPolygon (after we wrap a Polygon in a 1-element list).
    parts = list(getattr(union, "geoms", [union]))
    tree = STRtree(parts)

    mask = np.zeros(len(lats), dtype=bool)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        pt = Point(lo, la)
        # `.query(pt)` returns *indices* into ``parts`` whose envelopes
        # intersect the point.  For each candidate, do the precise
        # ``.contains`` test.  This is dramatically faster than
        # ``union.contains(pt)`` for large unions.
        for idx in tree.query(pt):
            if parts[idx].contains(pt):
                mask[i] = True
                break
    return mask


# ---------------------------------------------------------------------------
# Stage 4: write the consolidated NetCDF
# ---------------------------------------------------------------------------


def write_landscape_nc(
    out_path: Path,
    cells: list[str],
    depth: np.ndarray,
    times: np.ndarray,
    forcing: dict[str, np.ndarray],
    resolution: int,
    bbox: dict,
) -> None:
    """Sort by h3_id ascending so np.searchsorted works in H3Environment."""
    h3_ids = np.array([int(h3.str_to_int(c)) for c in cells], dtype=np.uint64)
    order = np.argsort(h3_ids)
    h3_ids = h3_ids[order]
    cells_sorted = [cells[i] for i in order]
    depth_sorted = depth[order]

    lats = np.array([h3.cell_to_latlng(c)[0] for c in cells_sorted])
    lons = np.array([h3.cell_to_latlng(c)[1] for c in cells_sorted])

    # Water = bathymetry says wet AND coastline-authority says wet.
    # Either condition alone has a known failure mode:
    #   * depth>0 alone: false positives in below-sea-level polderland
    #     (Nemunas Delta — EMODnet reports negative elevation there).
    #   * polygon alone: NE ocean polygon is coarse near the spit, OSM
    #     misses small streams.  Bathymetry filters those out.
    print(f"  applying coastline polygon mask…")
    in_polygon = polygon_water_mask(lats, lons)
    water_mask = ((depth_sorted > 0.0) & in_polygon).astype(np.uint8)
    n_water = int(water_mask.sum())
    n_total = len(water_mask)
    n_dropped = int((depth_sorted > 0.0).sum() - n_water)
    print(
        f"  water_mask: {n_water:,}/{n_total:,} cells "
        f"({100*n_water/n_total:.1f}% water); "
        f"polygon-mask dropped {n_dropped:,} below-sea-level land cells"
    )

    forcing_sorted = {k: v[:, order] for k, v in forcing.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "h3_id":      (("cell",), h3_ids),
            "lat":        (("cell",), lats),
            "lon":        (("cell",), lons),
            "depth":      (("cell",), depth_sorted),
            "water_mask": (("cell",), water_mask),
            **{k: (("time", "cell"), v) for k, v in forcing_sorted.items()},
        },
        coords={"time": times},
        attrs={
            "h3_resolution": resolution,
            "source_bathymetry": "EMODnet DTM 2022 (CC-BY 4.0 EMODnet)",
            "source_forcing": (
                "CMEMS BALTICSEA_MULTIYEAR_PHY_003_011 "
                "(CC-BY 4.0 Copernicus Marine)"
            ),
            "bbox": (
                f"{bbox['minlon']:.2f}-{bbox['maxlon']:.2f}E, "
                f"{bbox['minlat']:.2f}-{bbox['maxlat']:.2f}N"
            ),
            "n_cells": len(cells),
            "n_pentagons": int(sum(h3.is_pentagon(c) for c in cells)),
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "license": "CC-BY 4.0 (EMODnet + Copernicus Marine)",
        },
    )
    # NETCDF4 (via h5netcdf) so we can store h3_id as uint64 — NetCDF3
    # variants don't have an unsigned 64-bit integer type.  H3Environment
    # loads with engine="h5netcdf" to match.
    ds.to_netcdf(out_path, format="NETCDF4", engine="h5netcdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tif", default="data/curonian_bathymetry_raw.tif",
        help="EMODnet bathymetry GeoTIFF",
    )
    parser.add_argument(
        "--cmems", default="data/curonian_forcing_cmems_raw.nc",
        help="CMEMS reanalysis NetCDF",
    )
    parser.add_argument(
        "--out", default="data/nemunas_h3_landscape.nc",
        help="output NetCDF path",
    )
    parser.add_argument(
        "--resolution", type=int, default=DEFAULT_RESOLUTION,
        help=f"H3 resolution (default {DEFAULT_RESOLUTION})",
    )
    parser.add_argument(
        "--start", default="2011-06-01",
        help="CMEMS time-window start (ISO date) — keeps output ≤10 MB",
    )
    parser.add_argument(
        "--end", default="2011-06-30",
        help="CMEMS time-window end (ISO date)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    tif_path = project_root / args.tif
    cmems_path = project_root / args.cmems
    out_path = project_root / args.out

    print(f"[1/4] H3 cells over bbox at res {args.resolution}…")
    cells = build_h3_cells(BBOX, args.resolution)
    print(f"  {len(cells):,} cells")
    n_pentas = sum(h3.is_pentagon(c) for c in cells)
    print(f"  {n_pentas} pentagons (Nemunas bbox should be 0)")

    print(f"[2/4] Sampling EMODnet bathymetry…")
    depth = sample_emodnet(cells, tif_path)
    water_pct = 100 * (depth > 0).sum() / len(depth)
    print(
        f"  depth range: {depth.min():.2f} – {depth.max():.2f} m, "
        f"{water_pct:.1f}% water"
    )

    print(f"[3/4] Sampling CMEMS forcing [{args.start} .. {args.end}]…")
    times, forcing = sample_cmems(cells, cmems_path, args.start, args.end)

    print(f"[4/4] Writing {out_path}…")
    write_landscape_nc(
        out_path, cells, depth, times, forcing, args.resolution, BBOX,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {len(cells):,} cells × {len(times)} timesteps "
          f"({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
