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
