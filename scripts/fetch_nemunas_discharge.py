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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Add project root to sys.path so the salmon_ibm import below works whether the
# script is run from the project root or from scripts/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATES = pd.date_range("2011-01-01", "2024-12-31", freq="D")


def synthesize_climatology() -> xr.Dataset:
    """Daily Q(t) = baseline + Gaussian peak around day 115 (Apr 25)."""
    from salmon_ibm import delta_routing

    doy = DATES.dayofyear.values
    baseline = 400.0
    amplitude = 1500.0
    peak_day = 115  # Apr 25
    sigma_days = 35
    Q = (baseline + amplitude * np.exp(-((doy - peak_day) / sigma_days) ** 2)).astype(np.float32)

    fractions = list(delta_routing.BRANCH_FRACTIONS.items())
    branch_names = [br for br, _ in fractions]
    Q_per_branch = np.stack(
        [Q * f for _, f in fractions]
    ).astype(np.float32)  # (n_branches, n_time)

    # Build-time invariants (assert before writing — fail loudly if broken)
    assert Q_per_branch.shape == (len(fractions), len(Q)), (
        f"Q_per_branch shape mismatch: {Q_per_branch.shape}"
    )
    np.testing.assert_allclose(Q_per_branch.sum(axis=0), Q, rtol=1e-6)

    return xr.Dataset(
        {
            "Q": (("time",), Q),
            "Q_per_branch": (("branch", "time"), Q_per_branch),
        },
        coords={"time": DATES},
        attrs={
            "source": "synthetic climatology (awaiting Lithuanian EPA access)",
            "calibration": "Valiuskevicius 2019 + Mezine 2019 + HELCOM PLC-6 2018",
            "units": "m^3/s",
            "station_reference": "Smalininkai gauging station (Nemunas lower reach)",
            "fetched": datetime.date.today().isoformat(),
            "note": "Daily climatology — same values repeat every year.",
            "branch_names": ",".join(branch_names),
            "branch_fractions_source": (
                "Ramsar Site 629 Information Sheet (Nemunas Delta), 2010"
            ),
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
