import numpy as np
import pytest
from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.behavior import BehaviorParams, pick_behaviors, apply_overrides


@pytest.fixture
def params():
    return BehaviorParams.defaults()


def test_pick_behaviors_returns_valid(params):
    pool = AgentPool(n=100, start_tri=5, rng_seed=42)
    t3h = pool.t3h_mean()
    behaviors = pick_behaviors(t3h, pool.target_spawn_hour, params, seed=42)
    assert behaviors.shape == (100,)
    assert np.all((behaviors >= 0) & (behaviors <= 4))


def test_override_first_move_upstream(params):
    pool = AgentPool(n=5, start_tri=5)
    pool.steps[:] = 0
    pool.behavior[:] = Behavior.RANDOM
    overridden = apply_overrides(pool, params)
    assert np.all(overridden == Behavior.UPSTREAM)


def test_override_cwr_max_residence(params):
    pool = AgentPool(n=3, start_tri=5)
    pool.steps[:] = 10
    pool.behavior[:] = Behavior.TO_CWR
    pool.cwr_hours[:] = params.max_cwr_hours + 1
    overridden = apply_overrides(pool, params)
    assert np.all(overridden == Behavior.UPSTREAM)


def test_urgent_fish_prefer_upstream(params):
    """Fish with < 15 days to spawn at cool temperature should
    strongly prefer UPSTREAM over HOLD."""
    n = 1000
    t3h = np.full(n, 14.0)  # cool water (below temp_bins[0]=16)
    hours = np.full(n, 100.0)  # < 360 hours = urgent
    behaviors = pick_behaviors(t3h, hours, params, seed=42)
    upstream_frac = (behaviors == Behavior.UPSTREAM).mean()
    hold_frac = (behaviors == Behavior.HOLD).mean()
    assert upstream_frac > 0.5, (
        f"Urgent fish in cool water: UPSTREAM fraction {upstream_frac:.2f} should exceed 0.5"
    )
    assert upstream_frac > hold_frac, (
        f"Urgent fish: UPSTREAM ({upstream_frac:.2f}) should exceed HOLD ({hold_frac:.2f})"
    )
