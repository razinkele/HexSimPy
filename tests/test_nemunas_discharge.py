"""Schema test for data/nemunas_discharge.nc."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from salmon_ibm import delta_routing

PROJECT = Path(__file__).resolve().parent.parent
NC = PROJECT / "data" / "nemunas_discharge.nc"


def test_q_per_branch_present_and_consistent():
    if not NC.exists():
        pytest.skip(f"{NC.name} missing — run scripts/fetch_nemunas_discharge.py")
    ds = xr.open_dataset(NC)
    if "Q_per_branch" not in ds.variables:
        pytest.skip(
            "Q_per_branch missing — re-run scripts/fetch_nemunas_discharge.py "
            "to refresh the discharge NC with the new variable."
        )
    n_branches = len(delta_routing.BRANCH_FRACTIONS)
    n_time = ds.sizes["time"]
    assert ds["Q_per_branch"].shape == (n_branches, n_time), (
        f"Expected ({n_branches}, {n_time}), got {ds['Q_per_branch'].shape}"
    )
    branch_names_attr = ds.attrs.get("branch_names", "").split(",")
    assert branch_names_attr == list(delta_routing.BRANCH_FRACTIONS), (
        f"branch_names attr = {branch_names_attr}; "
        f"expected {list(delta_routing.BRANCH_FRACTIONS)}"
    )
    summed = ds["Q_per_branch"].values.sum(axis=0)
    np.testing.assert_allclose(summed, ds["Q"].values, rtol=1e-5)
    assert ds.attrs.get("branch_fractions_source", "").strip(), (
        "branch_fractions_source attr must be non-empty"
    )
    ds.close()
