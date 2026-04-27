"""Unit tests for salmon_ibm.delta_routing."""
import numpy as np
import pytest

from salmon_ibm import delta_routing


def test_branch_fractions_sum_to_one():
    total = sum(delta_routing.BRANCH_FRACTIONS.values())
    assert abs(total - 1.0) < 1e-9, f"Fractions sum to {total}, not 1.0"


def test_branch_fractions_keys_are_real_reaches():
    expected = {"Skirvyte", "Atmata", "Gilija"}
    assert set(delta_routing.BRANCH_FRACTIONS) == expected


def test_delta_branch_reaches_is_frozenset():
    assert isinstance(delta_routing.DELTA_BRANCH_REACHES, frozenset)
    assert delta_routing.DELTA_BRANCH_REACHES == set(delta_routing.BRANCH_FRACTIONS)


def test_split_discharge_preserves_total_array():
    q = np.array([100.0, 200.0, 0.0, 500.0], dtype=np.float32)
    out = delta_routing.split_discharge(q)
    summed = sum(out.values())
    assert np.allclose(summed, q, rtol=1e-6)


def test_split_discharge_handles_scalar():
    out = delta_routing.split_discharge(np.float32(1000.0))
    assert pytest.approx(out["Skirvyte"], rel=1e-6) == 510.0
    assert pytest.approx(out["Atmata"],   rel=1e-6) == 270.0
    assert pytest.approx(out["Gilija"],   rel=1e-6) == 220.0


def test_split_discharge_zero_input():
    q = np.zeros(10, dtype=np.float32)
    out = delta_routing.split_discharge(q)
    for name, arr in out.items():
        assert np.all(arr == 0.0), f"{name} should be all-zero, got {arr}"


def test_split_discharge_handles_list_input():
    out = delta_routing.split_discharge([100.0, 200.0])
    import numpy as np
    np.testing.assert_allclose(out["Skirvyte"], [51.0, 102.0], rtol=1e-6)
    np.testing.assert_allclose(out["Atmata"],   [27.0, 54.0],  rtol=1e-6)
    np.testing.assert_allclose(out["Gilija"],   [22.0, 44.0],  rtol=1e-6)
