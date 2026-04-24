"""Fetch CMEMS Baltic reanalysis for Curonian Lagoon forcing.

Dataset: BALTICSEA_MULTIYEAR_PHY_003_011 (daily, ~2 km native)
Variables: thetao (temp), so (salt), uo/vo (currents), zos (SSH)
Output: data/curonian_forcing_cmems.nc on the Curonian mesh nodes.

Reads credentials from .env (gitignored) — see .env.example for template.

v2 API notes (copernicusmarine >= 2.0):
  - force_download → deprecated, ignored
  - overwrite_output_data → renamed to overwrite
  - subset() returns a response object; .nc is written to output_directory/output_filename
  - username/password keywords work but also honors
    COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD env vars.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator


BBOX = {"minlon": 20.4, "maxlon": 21.9, "minlat": 54.9, "maxlat": 55.8}


def _load_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def fetch_raw(
    start: str,
    end: str,
    out_path: Path,
    bbox: dict = BBOX,
) -> None:
    """Subset BALTICSEA_MULTIYEAR_PHY_003_011 to Curonian bbox + surface layer."""
    import copernicusmarine as cm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm.subset(
        dataset_id="cmems_mod_bal_phy_my_P1D-m",
        # zos (SSH) is not in cmems_mod_bal_phy_my_P1D-m — fetch separately
        # from a wave/SSH product if needed for seiche detection.
        variables=["thetao", "so", "uo", "vo"],
        minimum_longitude=bbox["minlon"],
        maximum_longitude=bbox["maxlon"],
        minimum_latitude=bbox["minlat"],
        maximum_latitude=bbox["maxlat"],
        start_datetime=start,
        end_datetime=end,
        minimum_depth=0.0,
        maximum_depth=1.0,
        output_filename=out_path.name,
        output_directory=str(out_path.parent),
        overwrite=True,
    )


def regrid_to_mesh(raw_nc: Path, mesh_nc: Path, out_nc: Path) -> None:
    """Regrid CMEMS (regular lat/lon grid) onto the Curonian mesh (2D y×x).

    The Curonian mesh stores lat/lon/depth/mask as 2D (y, x) arrays.
    Output forcing has shape (time, y, x) matching the mesh axes.
    """
    raw = xr.open_dataset(raw_nc)
    mesh = xr.open_dataset(mesh_nc)

    rename = {}
    if "latitude" in raw.dims:
        rename["latitude"] = "lat"
    if "longitude" in raw.dims:
        rename["longitude"] = "lon"
    if rename:
        raw = raw.rename(rename)

    lat_src = raw.lat.values
    lon_src = raw.lon.values
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        raw = raw.isel(lat=slice(None, None, -1))

    n_time = raw.sizes["time"]
    ny, nx = mesh.lat.shape  # 2D mesh
    # Flatten mesh to (n_nodes, 2) for the interpolator, reshape output back to (y, x)
    query = np.column_stack([mesh.lat.values.ravel(), mesh.lon.values.ravel()])

    var_map = [("thetao", "tos"), ("so", "sos"), ("uo", "uo"), ("vo", "vo")]
    out_vars = {}
    for src, dst in var_map:
        if src not in raw:
            print(f"  ! source var {src} missing, skipping")
            continue
        arr = raw[src].squeeze().values
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        out = np.empty((n_time, ny, nx), dtype=np.float32)
        for t in range(n_time):
            src_t = arr[t]
            interp = RegularGridInterpolator(
                (lat_src, lon_src), src_t,
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            vals = interp(query).reshape(ny, nx)
            # Fill NaN (CMEMS land cells in lagoon interior) with nearest
            # neighbor from the source grid, so downstream NaN comparisons
            # in the thermal kill path don't propagate and kill every agent.
            if np.isnan(vals).any():
                src_flat = src_t.ravel()
                src_valid = ~np.isnan(src_flat)
                if src_valid.any():
                    src_lats, src_lons = np.meshgrid(lat_src, lon_src, indexing="ij")
                    src_points = np.column_stack(
                        [src_lats.ravel()[src_valid], src_lons.ravel()[src_valid]]
                    )
                    nn = NearestNDInterpolator(src_points, src_flat[src_valid])
                    nan_mask = np.isnan(vals).ravel()
                    nan_query = query[nan_mask]
                    vals.ravel()[nan_mask] = nn(nan_query)
            out[t] = vals.astype(np.float32)
        out_vars[dst] = (("time", "y", "x"), out)
        n_nan_day0 = int(np.isnan(out[0]).sum())
        print(f"  ✓ {src} → {dst} (time={n_time}, y={ny}, x={nx}, NaN after fill: {n_nan_day0})")

    # Environment loader requires ssh_var; CMEMS product doesn't provide zos.
    # Emit a zero-SSH variable so the loader succeeds — seiche detection will
    # be a no-op until a real SSH product is wired in a follow-up plan.
    out_vars["zos"] = (
        ("time", "y", "x"),
        np.zeros((n_time, ny, nx), dtype=np.float32),
    )
    print(f"  ✓ zos (zero-fill, seiche detection deferred)")

    ds = xr.Dataset(
        out_vars,
        coords={
            "time": raw.time.values,
            "lat": (("y", "x"), mesh.lat.values),
            "lon": (("y", "x"), mesh.lon.values),
        },
        attrs={
            "source": "CMEMS Baltic physics reanalysis BALTICSEA_MULTIYEAR_PHY_003_011",
            "dataset_id": "cmems_mod_bal_phy_my_P1D-m",
            "license": "CC-BY 4.0 (Copernicus Marine)",
            "regrid_method": "scipy.RegularGridInterpolator linear; NaN outside grid",
            "ssh_note": "zos is zero-filled; real SSH requires separate product",
        },
    )
    ds.to_netcdf(out_nc, format="NETCDF3_64BIT")
    print(f"Wrote {out_nc} ({out_nc.stat().st_size / 1e6:.1f} MB)")


def verify_mask_coverage(out_nc: Path, max_nan_frac: float = 0.5) -> None:
    """Sanity check: >50% NaN = CMEMS land-sea mask covers the lagoon → wrong data source."""
    ds = xr.open_dataset(out_nc, engine="scipy")
    for var in ["tos", "sos"]:
        if var not in ds:
            continue
        nan_frac = float(np.isnan(ds[var].isel(time=0)).mean())
        print(f"  {var} NaN fraction on day 0: {nan_frac:.1%}")
        if nan_frac > max_nan_frac:
            print(
                f"\n  WARNING: {nan_frac:.0%} of mesh is NaN on CMEMS day 0.\n"
                f"  The Curonian Lagoon is mostly masked as land at ~2 km CMEMS resolution.\n"
                f"  Consider falling back to SHYFEM (see scripts/fetch_shyfem_forcing.py).\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--mesh", default="data/curonian_minimal_grid.nc")
    parser.add_argument("--raw", default="data/curonian_forcing_cmems_raw.nc")
    parser.add_argument("--out", default="data/curonian_forcing_cmems.nc")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    _load_env(project_root)

    raw_path = project_root / args.raw
    out_path = project_root / args.out
    mesh_path = project_root / args.mesh

    print(f"[1/3] Fetching CMEMS {args.start}..{args.end} ...")
    fetch_raw(args.start, args.end, raw_path)
    print(f"  raw file: {raw_path} ({raw_path.stat().st_size / 1e6:.1f} MB)")

    print(f"[2/3] Regridding to mesh {mesh_path} ...")
    regrid_to_mesh(raw_path, mesh_path, out_path)

    print(f"[3/3] Land-sea mask coverage check ...")
    verify_mask_coverage(out_path)
    print("Done.")
