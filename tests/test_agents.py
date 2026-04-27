import numpy as np
from salmon_ibm.agents import Behavior, AgentPool


def test_behavior_enum():
    assert Behavior.HOLD.value == 0
    assert Behavior.UPSTREAM.value == 3


def test_agent_pool_creation():
    pool = AgentPool(n=10, start_tri=5, rng_seed=42)
    assert pool.n == 10
    assert pool.tri_idx.shape == (10,)
    assert np.all(pool.tri_idx == 5)


def test_agent_pool_alive_mask():
    pool = AgentPool(n=10, start_tri=5)
    assert pool.alive.sum() == 10
    pool.alive[3] = False
    assert pool.alive.sum() == 9


def test_agent_view_reads_pool():
    pool = AgentPool(n=5, start_tri=7)
    agent = pool.get_agent(2)
    assert agent.tri_idx == 7
    assert agent.id == 2


def test_agent_view_writes_to_pool():
    pool = AgentPool(n=5, start_tri=7)
    agent = pool.get_agent(2)
    agent.tri_idx = 99
    assert pool.tri_idx[2] == 99


def test_pool_t3h_mean():
    pool = AgentPool(n=3, start_tri=0)
    pool.temp_history[:] = [[15, 16, 17], [10, 10, 10], [20, 22, 24]]
    means = pool.t3h_mean()
    np.testing.assert_allclose(means, [16.0, 10.0, 22.0])


def test_pool_initial_energy_density():
    pool = AgentPool(n=5, start_tri=0)
    assert np.all(pool.ed_kJ_g > 4.0)
    assert np.all(pool.ed_kJ_g < 10.0)


def test_agent_pool_zero_agents():
    """AgentPool with n=0 should not crash."""
    from salmon_ibm.agents import AgentPool

    pool = AgentPool(n=0, start_tri=np.array([], dtype=int))
    assert pool.n == 0
    assert len(pool.alive) == 0
    assert pool.t3h_mean().shape == (0,)


def test_push_temperature_and_t3h_mean():
    """push_temperature should shift history and t3h_mean should average correctly."""
    import pytest
    from salmon_ibm.agents import AgentPool

    pool = AgentPool(n=2, start_tri=np.zeros(2, dtype=int))
    # Default temp_history is all 15.0
    pool.push_temperature(np.array([10.0, 20.0]))
    pool.push_temperature(np.array([12.0, 22.0]))
    pool.push_temperature(np.array([14.0, 24.0]))
    # History should now be [10, 12, 14] and [20, 22, 24]
    np.testing.assert_array_almost_equal(pool.temp_history[0], [10.0, 12.0, 14.0])
    np.testing.assert_array_almost_equal(pool.temp_history[1], [20.0, 22.0, 24.0])
    # t3h_mean should be the mean of each row
    means = pool.t3h_mean()
    assert means[0] == pytest.approx(12.0)
    assert means[1] == pytest.approx(22.0)


def test_agent_pool_init_covers_all_array_fields():
    """Every field in ARRAY_FIELDS must be set to an ndarray by __init__.

    Guards against adding a field to ARRAY_FIELDS but forgetting to initialize it.
    """
    import numpy as np
    from salmon_ibm.agents import AgentPool

    pool = AgentPool(n=3, start_tri=0, rng_seed=0)
    missing = []
    for field in AgentPool.ARRAY_FIELDS:
        attr = getattr(pool, field, None)
        if not isinstance(attr, np.ndarray):
            missing.append(field)
    assert not missing, (
        f"AgentPool.__init__ did not initialize these ARRAY_FIELDS as ndarrays: {missing}"
    )


def test_push_temperature_preserves_dead_agents_history():
    """With alive_mask, dead agents retain their last-slot value; ring shift drops the oldest."""
    import numpy as np
    from salmon_ibm.agents import AgentPool

    pool = AgentPool(n=3, start_tri=0, rng_seed=0)
    pool.temp_history[:] = np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    pool.alive[1] = False  # agent 1 dead

    pool.push_temperature(np.array([99.0, 99.0, 99.0]), alive_mask=pool.alive)

    # Alive agents (0 and 2): last slot is 99.0 (fresh write).
    assert pool.temp_history[0, -1] == 99.0
    assert pool.temp_history[2, -1] == 99.0
    # Dead agent (1): last slot preserved at 7.0 (ring shift was NOT applied).
    assert pool.temp_history[1, -1] == 7.0, (
        f"Dead agent's temp_history must be frozen; got {pool.temp_history[1, -1]}"
    )


def test_push_temperature_backward_compat_without_mask():
    """Without alive_mask, push_temperature behaves as before (no gating)."""
    import numpy as np
    from salmon_ibm.agents import AgentPool

    pool = AgentPool(n=2, start_tri=0, rng_seed=0)
    pool.temp_history[:] = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    pool.push_temperature(np.array([99.0, 99.0]))
    # Last slot updated for all agents.
    assert pool.temp_history[0, -1] == 99.0
    assert pool.temp_history[1, -1] == 99.0


def test_array_fields_includes_natal_and_exit_ids():
    from salmon_ibm.agents import AgentPool
    assert "natal_reach_id" in AgentPool.ARRAY_FIELDS
    assert "exit_branch_id" in AgentPool.ARRAY_FIELDS


def test_pool_init_defaults_natal_and_exit_to_minus_one():
    from salmon_ibm.agents import AgentPool
    pool = AgentPool(n=5, start_tri=0)
    import numpy as np
    assert pool.natal_reach_id.dtype == np.int8
    assert pool.exit_branch_id.dtype == np.int8
    assert (pool.natal_reach_id == -1).all()
    assert (pool.exit_branch_id == -1).all()


def test_pool_compact_preserves_natal_and_exit_ids():
    """compact() must propagate the new fields like every other ARRAY_FIELD."""
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np
    pool = AgentPool(n=4, start_tri=0)
    pool.natal_reach_id[:] = np.array([1, 2, 3, 4], dtype=np.int8)
    pool.exit_branch_id[:] = np.array([5, -1, 7, -1], dtype=np.int8)
    pool.alive[:] = np.array([True, False, True, False])
    pop = Population.__new__(Population)        # bypass __init__ paths
    pop.pool = pool
    pop.group_id = np.zeros(4, dtype=np.int32)
    pop.agent_ids = np.arange(4, dtype=np.int64)
    pop.affinity_targets = np.full(4, -1, dtype=np.intp)
    pop.spatial_affinity = np.zeros(4, dtype=np.float64)
    pop.accumulator_mgr = None
    pop.trait_mgr = None
    pop.genome = None
    pop.compact()
    assert pool.n == 2
    assert (pool.natal_reach_id == np.array([1, 3], dtype=np.int8)).all()
    assert (pool.exit_branch_id == np.array([5, 7], dtype=np.int8)).all()
