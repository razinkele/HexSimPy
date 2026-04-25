"""Validate H3 landscape water_mask against authoritative coastline data.

For each cell that ``data/nemunas_h3_landscape.nc`` marks as water,
test whether its centroid falls inside a real water body.  Authority
is the union of OSM water polygons + Natural Earth ocean (see
``scripts/_water_polygons.py``).

Cells where landscape says water but the union says land are genuine
false positives — likely the H3 water mask spilling inland beyond the
actual coastline.

Output:
* Stats summary on stdout.
* PNG map at ``visual_check_h3_vs_osm.png`` showing the polygon
  overlay + H3 centroids colour-coded by agreement.
* CSV ``h3_outside_water_suspects.csv`` with the (lon, lat) of every
  suspect centroid for triage in QGIS.

Run:
    micromamba run -n shiny python scripts/validate_h3_landscape_vs_osm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from shapely.geometry import Point
from shapely.ops import unary_union

from _water_polygons import (
    BBOX,
    fetch_natural_earth_ocean,
    fetch_osm_water,
)


PROJECT = Path(__file__).resolve().parent.parent
LANDSCAPE_NC = PROJECT / "data" / "nemunas_h3_landscape.nc"
OUT_PNG = PROJECT / "visual_check_h3_vs_osm.png"


def main() -> int:
    if not LANDSCAPE_NC.exists():
        print(f"ERROR: {LANDSCAPE_NC} missing — build it first via "
              f"scripts/build_nemunas_h3_landscape.py", file=sys.stderr)
        return 1

    print(f"[1/4] Fetching OSM water polygons…")
    osm = fetch_osm_water()
    print(f"[1b/4] Loading Natural Earth ocean polygon (Baltic Sea coverage)…")
    ne_ocean = fetch_natural_earth_ocean()
    print(f"  NE ocean: {len(ne_ocean)} polygon(s) after bbox clip")

    print(f"[2/4] Loading landscape NetCDF…")
    ds = xr.open_dataset(LANDSCAPE_NC, engine="h5netcdf")
    lats = ds["lat"].values
    lons = ds["lon"].values
    water = ds["water_mask"].values.astype(bool)
    n_total = len(water)
    n_water = int(water.sum())
    print(f"  {n_total:,} total cells, {n_water:,} water-masked")
    ds.close()

    print(f"[3/4] Spatial join: H3 water-cell centroids ∩ {{OSM, NE}} water polygons…")
    centroids = gpd.GeoSeries(
        [Point(lon, lat) for lat, lon in zip(lats[water], lons[water])],
        crs="EPSG:4326",
    )
    osm_union = unary_union(osm.geometry.values)
    ne_union = unary_union(ne_ocean.geometry.values)

    inside_osm = centroids.within(osm_union).values
    inside_ne = centroids.within(ne_union).values
    inside_any = inside_osm | inside_ne

    n_osm = int(inside_osm.sum())
    n_ne = int(inside_ne.sum())
    n_any = int(inside_any.sum())
    n_outside = n_water - n_any

    print(f"  inside OSM (lagoon/inland):       {n_osm:,} ({100*n_osm/n_water:.1f}%)")
    print(f"  inside NE  (Baltic Sea):          {n_ne:,} ({100*n_ne/n_water:.1f}%)")
    print(f"  inside EITHER source:             {n_any:,} ({100*n_any/n_water:.1f}%)")
    print(f"  OUTSIDE both (likely inland FP):  {n_outside:,} "
          f"({100*n_outside/n_water:.1f}%)")

    print(f"[4/4] Plotting validation map → {OUT_PNG.name}")
    fig, ax = plt.subplots(figsize=(11, 12))
    ne_ocean.plot(ax=ax, color="#cfe6ff", edgecolor="#7aa9d6",
                  linewidth=0.3, alpha=0.55, label="NE ocean (Baltic)")
    osm.plot(ax=ax, color="#a8d3ff", edgecolor="#5e9ad6",
             linewidth=0.4, alpha=0.65, label="OSM water (inland/lagoon)")

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n_water, size=min(10_000, n_water), replace=False)
    cents_lat = lats[water][sample_idx]
    cents_lon = lons[water][sample_idx]
    samp_in_any = inside_any[sample_idx]

    ax.scatter(cents_lon[samp_in_any], cents_lat[samp_in_any],
               s=2, c="#1c8c4a", alpha=0.55,
               label=f"H3 water ∈ OSM∪NE (sample of {int(samp_in_any.sum())})")
    ax.scatter(cents_lon[~samp_in_any], cents_lat[~samp_in_any],
               s=6, c="#d62728", alpha=0.85,
               label=f"H3 water ∉ OSM∪NE (sample of {int((~samp_in_any).sum())})")
    ax.set_xlim(BBOX[1], BBOX[3])
    ax.set_ylim(BBOX[0], BBOX[2])
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        f"H3 res-9 water mask vs OSM ∪ Natural-Earth water\n"
        f"{n_any:,} inside / {n_outside:,} outside of {n_water:,} cells "
        f"({100*n_any/n_water:.1f}% agreement)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (BBOX[0] + BBOX[2]))))
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f"  saved {OUT_PNG}")

    out_csv = PROJECT / "h3_outside_water_suspects.csv"
    if n_outside > 0:
        sus_lat = lats[water][~inside_any]
        sus_lon = lons[water][~inside_any]
        np.savetxt(
            out_csv, np.column_stack([sus_lon, sus_lat]),
            delimiter=",", header="lon,lat", comments="", fmt="%.6f",
        )
        print(f"  wrote {n_outside:,} suspect centroids → {out_csv.name}")
    elif out_csv.exists():
        # Clean up stale CSV from previous run.
        out_csv.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
