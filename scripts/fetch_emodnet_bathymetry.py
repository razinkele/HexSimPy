"""Fetch EMODnet Bathymetry for the Curonian Lagoon + Nemunas mouth + Baltic coast.

Updates data/curonian_minimal_grid.nc in place: replaces the synthetic depth
variable with real EMODnet DTM 2022 1/16-arcmin mean depths, regridded onto
the existing Curonian mesh nodes.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 — registers .rio accessor
from owslib.wcs import WebCoverageService
from scipy.interpolate import RegularGridInterpolator


BBOX = {"minlon": 20.4, "maxlon": 21.9, "minlat": 54.9, "maxlat": 55.8}
WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"


def pick_coverage(wcs) -> str:
    candidates = list(wcs.contents.keys())
    for pattern in ["emodnet_mean_atlas_land_2022", "emodnet__mean", "mean"]:
        for c in candidates:
            if pattern.lower() in c.lower():
                return c
    if candidates:
        return candidates[0]
    raise RuntimeError("No WCS coverages listed")


def fetch_tif(bbox: dict, out_path: Path) -> str:
    wcs = WebCoverageService(WCS_URL, version="2.0.1")
    coverage = pick_coverage(wcs)
    print(f"  coverage: {coverage}")
    # owslib 2.0.1 expects CoverageID as a string, not a list — passing a
    # list causes it to URL-encode the python list repr and get HTTP 404.
    response = wcs.getCoverage(
        identifier=coverage,
        subsets=[
            ("Lat", bbox["minlat"], bbox["maxlat"]),
            ("Long", bbox["minlon"], bbox["maxlon"]),
        ],
        format="image/tiff",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.read())
    return coverage


def regrid_to_mesh(raw_tif: Path, mesh_nc: Path, coverage: str) -> None:
    raw = rioxarray.open_rasterio(raw_tif).squeeze()
    x = raw.x.values
    y = raw.y.values
    z = raw.values
    if y[0] > y[-1]:
        y = y[::-1]
        z = z[::-1, :]
    interp = RegularGridInterpolator(
        (y, x), z, method="linear", bounds_error=False, fill_value=np.nan
    )

    mesh_ds = xr.open_dataset(mesh_nc, engine="scipy").load()
    mesh_ds.close()
    ny, nx = mesh_ds.lat.shape  # 2D mesh (y, x)
    query = np.column_stack([mesh_ds.lat.values.ravel(),
                              mesh_ds.lon.values.ravel()])
    depth_at_nodes = interp(query).reshape(ny, nx)

    # EMODnet: positive up (elevation). HexSim: positive down (depth). Flip.
    depth_at_nodes = np.where(np.isnan(depth_at_nodes), 0.0, -depth_at_nodes)
    depth_at_nodes = np.maximum(depth_at_nodes, 0.0)

    mesh_ds["depth"] = (mesh_ds.depth.dims, depth_at_nodes.astype(np.float32))
    mesh_ds.attrs["depth_source"] = (
        f"EMODnet Bathymetry {coverage} via WCS {WCS_URL}, "
        f"fetched {datetime.date.today().isoformat()}, "
        f"regridded linear to mesh nodes. License: CC-BY 4.0 (EMODnet)."
    )
    mesh_ds.to_netcdf(mesh_nc, format="NETCDF3_64BIT")

    mean = float(np.nanmean(depth_at_nodes))
    maxd = float(np.nanmax(depth_at_nodes))
    print(f"  depth updated on {mesh_nc}")
    print(f"  mean: {mean:.2f} m  max: {maxd:.1f} m  (expected: mean 3-4 m, max ~14 m)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="data/curonian_minimal_grid.nc")
    parser.add_argument("--tif", default="data/curonian_bathymetry_raw.tif")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    mesh_path = project_root / args.mesh
    tif_path = project_root / args.tif

    print(f"[1/2] Fetching EMODnet DTM ...")
    coverage = fetch_tif(BBOX, tif_path)
    print(f"  TIF: {tif_path} ({tif_path.stat().st_size / 1e6:.2f} MB)")

    print(f"[2/2] Regridding onto mesh {mesh_path} ...")
    regrid_to_mesh(tif_path, mesh_path, coverage)
    print("Done.")
