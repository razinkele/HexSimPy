import numpy as np
import pytest
from salmon_ibm.estuary import salinity_cost, do_override, seiche_pause


def test_salinity_cost_below_tolerance():
    cost = salinity_cost(np.array([3.0]), S_opt=0.5, S_tol=6.0, k=0.6)
    np.testing.assert_allclose(cost, [1.0])


def test_salinity_cost_above_tolerance():
    cost = salinity_cost(np.array([10.0]), S_opt=0.5, S_tol=6.0, k=0.6)
    np.testing.assert_allclose(cost, [3.1])


def test_do_override_normal():
    override = do_override(np.array([8.0]), lethal=2.0, high=4.0)
    assert override[0] == 0


def test_do_override_high():
    override = do_override(np.array([3.0]), lethal=2.0, high=4.0)
    assert override[0] == 1


def test_do_override_lethal():
    override = do_override(np.array([1.5]), lethal=2.0, high=4.0)
    assert override[0] == 2


def test_seiche_pause_calm():
    paused = seiche_pause(np.array([0.005]), thresh=0.02)
    assert not paused[0]


def test_seiche_pause_active():
    paused = seiche_pause(np.array([0.05]), thresh=0.02)
    assert paused[0]


def test_salinity_cost_capped():
    """Salinity cost should not exceed a maximum cap."""
    from salmon_ibm.estuary import salinity_cost
    extreme_sal = np.array([50.0])  # extreme ocean salinity
    cost = salinity_cost(extreme_sal)
    assert cost[0] <= 5.0, f"Cost {cost[0]} should be capped at 5.0"
    assert cost[0] > 1.0, f"Cost should still be > 1.0 for high salinity"


def test_salinity_cost_nan_treated_as_zero():
    """NaN salinity should produce cost = 1.0 (no penalty)."""
    from salmon_ibm.estuary import salinity_cost
    sal = np.array([np.nan, 5.0, np.nan])
    cost = salinity_cost(sal)
    assert cost[0] == pytest.approx(1.0), "NaN salinity should give neutral cost"
    assert not np.isnan(cost).any(), "No NaN should propagate"


def test_do_override_nan_treated_as_ok():
    """NaN dissolved oxygen should be classified as DO_OK, not DO_ESCAPE."""
    from salmon_ibm.estuary import do_override, DO_OK
    do_vals = np.array([np.nan, 5.0, np.nan])
    result = do_override(do_vals)
    assert result[0] == DO_OK, "NaN DO should be classified as DO_OK"
    assert result[2] == DO_OK, "NaN DO should be classified as DO_OK"
