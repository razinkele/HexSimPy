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
