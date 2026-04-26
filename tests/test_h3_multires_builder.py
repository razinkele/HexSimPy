"""Smoke tests for scripts/build_h3_multires_landscape.py."""
from __future__ import annotations
from pathlib import Path
import pytest

PROJECT = Path(__file__).resolve().parent.parent
LANDSCAPE_NC = PROJECT / "data" / "curonian_h3_multires_landscape.nc"


def test_multires_nc_carries_forcing_fields():
    """Builder must write tos/sos/uo/vo per (time, cell) so
    H3Environment.from_netcdf finds them at sim-init time."""
    if not LANDSCAPE_NC.exists():
        pytest.skip(
            "multi-res landscape NC not built — "
            "run scripts/build_h3_multires_landscape.py first."
        )
    import xarray as xr
    ds = xr.open_dataset(LANDSCAPE_NC, engine="h5netcdf")
    for var in ("tos", "sos", "uo", "vo"):
        assert var in ds.variables, (
            f"forcing field {var!r} missing from multi-res NC; "
            f"H3Environment will fail at simulation init."
        )
    assert "time" in ds.dims, "forcing time dimension missing"
    ds.close()
