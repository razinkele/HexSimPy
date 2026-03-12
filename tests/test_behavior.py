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


def test_vectorized_distribution_matches_probabilities(params):
    """Verify vectorized pick_behaviors produces valid behaviors (0-4)
    and that the empirical distribution matches the probability table."""
    n = 50_000
    # Place all agents in a single bucket: urgent time (<360h), cool temp (<16)
    t3h = np.full(n, 14.0)
    hours = np.full(n, 100.0)
    behaviors = pick_behaviors(t3h, hours, params, seed=123)
    assert behaviors.shape == (n,)
    assert np.all((behaviors >= 0) & (behaviors <= 4))
    # Expected probabilities for time_idx=0, temp_idx=0
    expected = params.p_table[0, 0]
    observed = np.array([(behaviors == b).mean() for b in range(5)])
    np.testing.assert_allclose(observed, expected, atol=0.02,
                               err_msg="Empirical distribution deviates from p_table")

    # Also test a mixed-bucket scenario
    n2 = 20_000
    t3h_mixed = np.concatenate([np.full(n2, 14.0), np.full(n2, 19.0)])
    hours_mixed = np.concatenate([np.full(n2, 100.0), np.full(n2, 500.0)])
    beh_mixed = pick_behaviors(t3h_mixed, hours_mixed, params, seed=456)
    assert beh_mixed.shape == (2 * n2,)
    assert np.all((beh_mixed >= 0) & (beh_mixed <= 4))
    # Check second bucket: time_idx=1, temp_idx=2
    expected2 = params.p_table[1, 2]
    obs2 = np.array([(beh_mixed[n2:] == b).mean() for b in range(5)])
    np.testing.assert_allclose(obs2, expected2, atol=0.02,
                               err_msg="Mixed-bucket distribution deviates from p_table")


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
