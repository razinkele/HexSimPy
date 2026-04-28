"""Build a variable-resolution H3 landscape (scaffold).

Phase 1 of the multi-resolution H3 backend.  See:

* ``docs/h3-multi-resolution-feasibility.md`` — design analysis.
* ``docs/h3-multires-roadmap.md``               — implementation roadmap.
* ``salmon_ibm/h3_multires.py``                  — the mesh class.

What this script does
---------------------

For each inSTREAM reach (Nemunas, Atmata, Minija, Sysa, Skirvyte,
Leite, Gilija, CuronianLagoon, BalticCoast — plus OpenBaltic from the
Natural Earth ocean polygon clip), tessellate the polygon at the
reach's *own* resolution.  Smaller delta channels get a finer
resolution (res 11, ~28 m edge) so the channel geometry is properly
resolved; the lagoon and open Baltic stay at coarse resolution
(res 9, ~200 m edge) so the cell count doesn't explode.

The result is a single H3 cell list with mixed resolutions.
``salmon_ibm.h3_multires.H3MultiResMesh.from_h3_cells`` builds the
neighbour table including cross-resolution edges (see the doc for the
algorithm).

Status
------

* Tessellate-per-reach + dedupe cell list — implemented.
* Sample bathymetry + forcing at each cell centroid — implemented.
* Write NetCDF carrying the mixed-resolution cell list + payload —
  implemented.
* Simulation integration — *not yet*.  ``Simulation.__init__`` still
  routes ``mesh_backend == "h3"`` through the uniform-res ``H3Mesh``;
  wiring the multi-res mesh through the numba movement kernels (which
  read padded-(N, 6) neighbour rows) needs the ``MAX_NBRS = 12`` bump
  to land in those kernels.  The padded compat view in
  ``H3MultiResMesh.neighbors`` is sized correctly already; the
  remaining wiring work is in ``salmon_ibm/movement.py``.

Default per-reach resolution
----------------------------

==============  ================  =====================================
Reach           H3 resolution     Rationale
==============  ================  =====================================
Nemunas         11 (~28 m)        ~600 m wide, want 20+ cells across
Atmata          11                ~250 m wide, fits 9-10 cells
Minija          11                ~150-300 m wide, fits 5-10 cells
Sysa            11                ~100 m wide, fits 3-4 cells
Skirvyte        11                ~80 m wide, fits 2-3 cells
Leite           11                Small, fits ~4 cells
Gilija          11                Kaliningrad-side delta arm
CuronianLagoon   9 (~200 m)       Large open water — coarse OK
BalticCoast      9                Narrow strip near Klaipėda
OpenBaltic       8 (~530 m)       Open sea — coarsest
==============  ================  =====================================

Override via the ``--resolutions`` arg as a comma-separated list of
``Reach=Res`` pairs, e.g. ``Nemunas=10,CuronianLagoon=9``.

Output
------

``data/curonian_h3_multires_landscape.nc`` — NETCDF4 (h5netcdf engine)
with:

* ``h3_id``      ``(cell,)`` uint64
* ``resolution`` ``(cell,)`` int8
* ``lat, lon``   ``(cell,)`` float64
* ``depth``      ``(cell,)`` float32
* ``water_mask`` ``(cell,)`` uint8
* ``reach_id``   ``(cell,)`` int8
* ``nbr_starts``  ``(cell+1,)`` int32 — CSR row pointers
* ``nbr_idx``     ``(M,)`` int32       — flat neighbour indices
* attributes: ``reach_names``, ``reach_resolutions``, etc.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import h3
import numpy as np
import xarray as xr

# Add project root to sys.path so the salmon_ibm imports below work
# whether the script is run from the project root or from scripts/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from _water_polygons import (
    BBOX as VALIDATOR_BBOX,
    fetch_instream_polygons,
    fetch_natural_earth_ocean,
)
from salmon_ibm.h3_tessellate import (
    tessellate_reach,
    bridge_components,
    polygon_trust_water_mask,
)


PROJECT = Path(__file__).resolve().parent.parent
LAND_ID = -1
DEFAULT_RES = {
    # Rivers — res 11 (~28 m edge) to resolve 50–200 m channel
    # widths as 1–7 cells across.  At this scale, polygon_to_cells
    # tessellates rivers as a connected swath naturally; bridge-cell
    # logic (Task 2) is a safety net for multi-polygon cases.
    "Nemunas":        11,
    "Atmata":         11,
    "Minija":         11,
    "Sysa":           11,
    "Skirvyte":       11,
    "Leite":          11,
    "Gilija":         11,
    # Lagoon — res 10 (~76 m edge, ~120 k cells) for spatial
    # heterogeneity in mortality, temperature, salinity gradients
    # across the lagoon's 1 600 km².  Was res 9 in v1.4.0.
    "CuronianLagoon": 10,
    # Baltic — uniform res 9 (~201 m edge) eliminates the v1.4.0
    # res-9 / res-8 boundary discontinuity at the BalticCoast↔
    # OpenBaltic interface.  Mesoscale circulation does not need
    # finer cells.  OpenBaltic gains ~7× cells (50 k vs 7 k); the
    # visible "scattered hexagons" boundary disappears.
    "BalticCoast":     9,
    "OpenBaltic":      9,
}


def sample_cmems(
    cells: list[str],
    cmems_path: Path,
    start: str | None = None,
    end: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sample CMEMS thetao/so/uo/vo at each H3 cell centroid.

    Returns ``(time_array, {var_name: (time, cell) float32})``.
    Same NaN-fill via NearestNDInterpolator as the uniform-res builder
    so the thermal-kill path doesn't kill agents on a CMEMS land cell.
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
    src_lats_2d, src_lons_2d = np.meshgrid(lat_src, lon_src, indexing="ij")
    var_map = [("thetao", "tos"), ("so", "sos"), ("uo", "uo"), ("vo", "vo")]
    out: dict[str, np.ndarray] = {}
    from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
    for src, dst in var_map:
        if src not in raw:
            print(f"  ! source var {src} missing — skipping")
            continue
        arr = raw[src].squeeze().values
        vals = np.empty((n_time, n_cells), dtype=np.float32)
        for t in range(n_time):
            src_t = arr[t]
            interp = RegularGridInterpolator(
                (lat_src, lon_src), src_t,
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            row = interp(query)
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
        print(f"  ✓ {src} → {dst} ({n_time} timesteps, {n_cells:,} cells)")
    return raw["time"].values, out


def parse_resolution_overrides(spec: str) -> dict[str, int]:
    """Parse ``Reach=N,Reach=N,...`` into a {name: res} dict."""
    if not spec:
        return {}
    out = {}
    for tok in spec.split(","):
        if "=" not in tok:
            raise ValueError(f"bad --resolutions token: {tok!r}")
        name, val = tok.split("=", 1)
        out[name.strip()] = int(val.strip())
    return out


def build_cell_list(
    reach_polygons: dict[str, "shapely.Polygon"],
    reach_res: dict[str, int],
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """For each reach, tessellate at its resolution; dedupe across reaches.

    Returns ``(cells, reach_id_arr, reach_id_map)``.  Earlier reaches in
    ``reach_polygons`` claim their cells first (so the small rivers get
    cells before the lagoon overrides them where they overlap).

    Also returns ``reach_id_map`` — the {reach_name: int_id} mapping used
    in the output NC.
    """
    seen_id_to_reach: dict[int, str] = {}  # H3 int → reach name
    for reach_name, poly in reach_polygons.items():
        res = reach_res.get(reach_name, 9)
        cells = tessellate_reach(poly, res)
        cells = bridge_components(cells, res)
        for c in cells:
            ci = int(h3.str_to_int(c))
            if ci not in seen_id_to_reach:
                seen_id_to_reach[ci] = reach_name

    # Stable order: by H3 ID for determinism + np.searchsorted on uint64.
    sorted_ids = sorted(seen_id_to_reach.keys())
    sorted_cells = [h3.int_to_str(ci) for ci in sorted_ids]

    reach_id_map = {name: i for i, name in enumerate(reach_polygons.keys())}
    reach_id_arr = np.array(
        [reach_id_map[seen_id_to_reach[ci]] for ci in sorted_ids],
        dtype=np.int8,
    )
    return sorted_cells, reach_id_arr, reach_id_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions", default="",
        help="Override per-reach resolution: 'Nemunas=10,CuronianLagoon=9'",
    )
    parser.add_argument(
        "--out", default="data/curonian_h3_multires_landscape.nc",
        help="output NetCDF path",
    )
    parser.add_argument(
        "--tif", default="data/curonian_bathymetry_raw.tif",
        help="EMODnet bathymetry GeoTIFF",
    )
    parser.add_argument(
        "--cmems", default="data/curonian_forcing_cmems_raw.nc",
        help="CMEMS reanalysis NetCDF",
    )
    parser.add_argument(
        "--start", default="2011-06-01",
        help="CMEMS time-window start (ISO date) — keeps output ≤30 MB",
    )
    parser.add_argument(
        "--end", default="2011-06-30",
        help="CMEMS time-window end (ISO date)",
    )
    args = parser.parse_args()

    overrides = parse_resolution_overrides(args.resolutions)
    reach_res = {**DEFAULT_RES, **overrides}
    print(f"Resolutions in use:")
    for name, res in reach_res.items():
        edge_m = h3.average_hexagon_edge_length(res, unit="m")
        print(f"  {name:>14s}: res {res:>2}  ({edge_m:.0f} m edge)")
    print()

    print(f"[1/4] Loading polygons…")
    instream = fetch_instream_polygons()
    by_reach = instream.dissolve(by="REACH_NAME").geometry
    ne_ocean = fetch_natural_earth_ocean()
    from shapely.ops import unary_union
    from shapely.geometry import box
    ne_clipped = ne_ocean.intersection(
        box(VALIDATOR_BBOX[1], VALIDATOR_BBOX[0],
            VALIDATOR_BBOX[3], VALIDATOR_BBOX[2])
    )
    ne_union = unary_union(list(ne_clipped[~ne_clipped.is_empty]))
    # Subtract BalticCoast AND CuronianLagoon from the NE ocean polygon.
    # Natural Earth's "ocean" polygon includes the Curonian Lagoon as a
    # contiguous water body, so without subtraction OpenBaltic's res-9
    # tessellation claims ~35% of its cells INSIDE the lagoon
    # (16k of 47k cells in v1.5.0).  Those overlapping cells then
    # generate ~6,400 spurious cross-res lagoon↔OpenBaltic links via
    # `find_cross_res_neighbours` walk-up.  We do NOT subtract the
    # rivers — they're narrow polygons whose subtraction creates
    # Swiss-cheese fragmentation (observed: 60+ components in earlier
    # experiments) without meaningfully reducing spurious links.
    open_baltic = ne_union
    if "BalticCoast" in by_reach.index:
        open_baltic = open_baltic.difference(by_reach["BalticCoast"])
    if "CuronianLagoon" in by_reach.index:
        open_baltic = open_baltic.difference(by_reach["CuronianLagoon"])
    print(f"  inSTREAM reaches: {sorted(by_reach.index)}")
    print(f"  open Baltic area (ne_ocean − BalticCoast − CuronianLagoon): {open_baltic.area:.4f} sq deg")

    reach_polygons = {name: by_reach[name] for name in DEFAULT_RES if name in by_reach.index}
    reach_polygons["OpenBaltic"] = open_baltic

    print(f"\n[2/4] Tessellating per-reach…")
    cells, reach_id_arr, reach_id_map = build_cell_list(reach_polygons, reach_res)
    print(f"  total cells: {len(cells):,}")
    for name, rid in reach_id_map.items():
        n = int((reach_id_arr == rid).sum())
        if n:
            res = reach_res.get(name, 9)
            print(f"    {name:>14s} (id={rid}, res={res}): {n:>7,}")

    print(f"\n[3/4] Sampling EMODnet bathymetry per cell…")
    import rioxarray  # noqa: F401
    from scipy.interpolate import RegularGridInterpolator
    raw = rioxarray.open_rasterio(PROJECT / args.tif).squeeze()
    x, y, z = raw.x.values, raw.y.values, raw.values
    if y[0] > y[-1]:
        y, z = y[::-1], z[::-1, :]
    interp = RegularGridInterpolator(
        (y, x), z, method="linear", bounds_error=False, fill_value=np.nan,
    )
    lats, lons = zip(*(h3.cell_to_latlng(c) for c in cells))
    elev = interp(np.column_stack([lats, lons]))
    depth = np.maximum(np.where(np.isnan(elev), 0.0, -elev), 0.0).astype(np.float32)
    water_mask = (depth > 0).astype(np.uint8)
    print(f"  depth range: {depth.min():.2f} – {depth.max():.2f} m, "
          f"water cells (EMODnet): {int(water_mask.sum()):,}")

    # Approach F (v1.6.1): trust inSTREAM polygons over EMODnet bathymetry
    # for cells inside any reach polygon. The half-cell buffer in
    # tessellate_reach (added v1.5.0 to fix narrow-channel sparseness)
    # over-captures cells up to ~14m beyond the polygon edge — EMODnet
    # often reports those cells as dry because its sampling is coarser
    # than the digitised polygon edges. Without this override, ~16% of
    # tagged cells (47-87% of river cells) have reach_id but
    # water_mask=False; the viewer (app.py:1811 filters by water_mask)
    # drops them silently — polygon outlines appear unfilled.
    #
    # Approach C (drop reach_id for dry tagged cells) was tried first but
    # catastrophically fragmented connectivity (Nemunas → 160 components,
    # CuronianLagoon → 291 components, Atmata↔Lagoon edges → 0). The
    # buffer cells were doing double duty: visual padding + connectivity
    # glue. Approach F restores the glue by treating polygon membership
    # as authoritative — same pattern the script already uses for
    # OpenBaltic = ne_ocean − BalticCoast − CuronianLagoon.
    #
    # See tests/test_h3_grid_quality.py::test_reach_id_implies_water_mask.
    n_overridden = int(((reach_id_arr != -1) & (water_mask == 0)).sum())
    water_mask, depth = polygon_trust_water_mask(reach_id_arr, water_mask, depth)
    print(f"  polygon-trust override: {n_overridden:,} buffer cells "
          f"flipped water_mask=False→True (depth set to 1m where 0)")
    print(f"  water cells (final): {int(water_mask.sum()):,}")

    print(f"\n[3.5/4] Sampling CMEMS forcing [{args.start} .. {args.end}]…")
    times, forcing = sample_cmems(
        cells, PROJECT / args.cmems, args.start, args.end,
    )

    print(f"\n[4/4] Writing {args.out}…")
    # Build the CSR neighbour table — this is the multi-res-specific
    # work that scales O(N · k · D), ~1 second for 50 k cells.
    from salmon_ibm.h3_multires import find_cross_res_neighbours
    nbr_starts, nbr_idx = find_cross_res_neighbours(cells)
    print(f"  neighbour table: {len(nbr_idx):,} edges, "
          f"avg {len(nbr_idx)/len(cells):.2f} per cell")

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Pre-existing in main() scope (v1.2.8 scaffold): cells, lats, lons,
    # depth, water_mask, reach_id_arr, nbr_starts, nbr_idx.  This task
    # adds: times, forcing.
    ds = xr.Dataset(
        {
            "h3_id":      (("cell",), np.array([int(h3.str_to_int(c)) for c in cells], dtype=np.uint64)),
            "resolution": (("cell",), np.array([h3.get_resolution(c) for c in cells], dtype=np.int8)),
            "lat":        (("cell",), np.array(lats, dtype=np.float64)),
            "lon":        (("cell",), np.array(lons, dtype=np.float64)),
            "depth":      (("cell",), depth),
            "water_mask": (("cell",), water_mask),
            "reach_id":   (("cell",), reach_id_arr),
            "nbr_starts": (("cell_p1",), nbr_starts),
            "nbr_idx":    (("edge",), nbr_idx),
            # Forcing per (time, cell) — added in this task.
            **{k: (("time", "cell"), v) for k, v in forcing.items()},
        },
        coords={"time": times},  # added in this task
        attrs={
            "title": "Curonian Lagoon multi-resolution H3 landscape",
            "reach_names": ",".join(reach_polygons.keys()),
            "reach_resolutions": ",".join(
                f"{n}={reach_res.get(n, 9)}" for n in reach_polygons.keys()
            ),
            "n_cells": len(cells),
            "n_edges": len(nbr_idx),
            # Static-viewer compat: median per-cell resolution lets the
            # legacy "h3_resolution" reader (app.py:2053 & 515) pick a
            # sensible camera zoom even though the NC is mixed-res.
            "h3_resolution": int(np.median(
                np.array([h3.get_resolution(c) for c in cells])
            )),
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )
    ds.to_netcdf(out_path, format="NETCDF4", engine="h5netcdf")
    print(f"  wrote {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
