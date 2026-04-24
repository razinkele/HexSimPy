"""Daily Nemunas discharge for 2011-2024 at the lagoon outlet.

This script synthesises a climatological hydrograph because the authoritative
Lithuanian Environmental Protection Agency records at Smalininkai gauging
station are not currently accessible in-tree. Replace when data access is
granted.

Climatology calibrated to:
  - Mėžinė et al. 2019 (doi:10.3390/w11101970) — ~700 m³/s mean at Smalininkai
  - Valiuškevičius et al. 2019 (doi:10.5200/baltica.2018.31.09) — long-term
    mean ~530 m³/s, spring flood typically 1500-2500 m³/s (>5× mean)
  - HELCOM PLC-6 (2018) transboundary river aggregates

Hydrograph shape: winter baseline + Gaussian April-May snowmelt peak.
"""
import argparse
import datetime

import numpy as np
import pandas as pd
import xarray as xr

DATES = pd.date_range("2011-01-01", "2024-12-31", freq="D")


def synthesize_climatology() -> xr.Dataset:
    """Daily Q(t) = baseline + Gaussian peak around day 115 (Apr 25)."""
    doy = DATES.dayofyear.values
    baseline = 400.0
    amplitude = 1500.0
    peak_day = 115  # Apr 25
    sigma_days = 35
    Q = baseline + amplitude * np.exp(-((doy - peak_day) / sigma_days) ** 2)
    return xr.Dataset(
        {"Q": (("time",), Q.astype(np.float32))},
        coords={"time": DATES},
        attrs={
            "source": "synthetic climatology (awaiting Lithuanian EPA access)",
            "calibration": "Valiuskevicius 2019 + Mezine 2019 + HELCOM PLC-6 2018",
            "units": "m^3/s",
            "station_reference": "Smalininkai gauging station (Nemunas lower reach)",
            "fetched": datetime.date.today().isoformat(),
            "note": "Daily climatology — same values repeat every year.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/nemunas_discharge.nc")
    args = parser.parse_args()
    ds = synthesize_climatology()
    # HexSim's environment loader uses xarray engine="scipy" which requires
    # NetCDF3. Write with NETCDF3_64BIT_OFFSET format so the file is readable
    # without needing the netcdf4 library.
    ds.to_netcdf(args.out, format="NETCDF3_64BIT")
    Q = ds.Q.values
    print(
        f"Wrote {args.out}: {Q.shape[0]} days, "
        f"range {float(Q.min()):.0f}-{float(Q.max()):.0f} m^3/s, "
        f"mean {float(Q.mean()):.0f}, peak on DOY {int(Q.argmax() % 365) + 1}"
    )
