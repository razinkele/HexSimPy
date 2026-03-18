"""Performance regression tests for the simulation engine."""
import time

import numpy as np
import pytest

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


@pytest.mark.slow
def test_step_performance_5000_agents():
    """5000 agents, 10 steps should complete in <5s after vectorization."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=5000, data_dir="data", rng_seed=42)

    n_steps = 10
    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0

    per_step = elapsed / n_steps
    print(f"\n  Vectorized: {per_step:.4f} s/step for 5000 agents")
    assert elapsed < 5.0, f"10 steps took {elapsed:.1f}s, expected <5s"


@pytest.mark.slow
def test_step_performance_10000_agents():
    """10000 agents, 10 steps should complete in <10s after vectorization."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10000, data_dir="data", rng_seed=42)

    n_steps = 10
    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0

    per_step = elapsed / n_steps
    print(f"\n  Vectorized: {per_step:.4f} s/step for 10000 agents")
    assert elapsed < 10.0, f"10 steps took {elapsed:.1f}s, expected <10s"


def test_vectorized_movement_correctness():
    """Full simulation should still produce valid, reproducible results."""
    cfg = load_config("config_curonian_minimal.yaml")

    sim1 = Simulation(cfg, n_agents=100, data_dir="data", rng_seed=42)
    sim1.run(n_steps=20)

    cfg2 = load_config("config_curonian_minimal.yaml")
    sim2 = Simulation(cfg2, n_agents=100, data_dir="data", rng_seed=42)
    sim2.run(n_steps=20)

    np.testing.assert_array_equal(sim1.pool.tri_idx, sim2.pool.tri_idx)
    np.testing.assert_array_almost_equal(sim1.pool.ed_kJ_g, sim2.pool.ed_kJ_g)
    assert sim1.pool.alive.sum() > 0, "Some agents should survive 20 steps"
