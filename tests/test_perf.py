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


@pytest.mark.slow
def test_step_performance_100k_agents():
    """100K agents, 10 steps should complete in <50s with Numba JIT."""
    cfg = load_config("config_curonian_minimal.yaml")
    # Warmup Numba compilation with small run
    sim_warmup = Simulation(cfg, n_agents=100, data_dir="data", rng_seed=99)
    sim_warmup.run(1)

    sim = Simulation(cfg, n_agents=100_000, data_dir="data", rng_seed=42)
    n_steps = 10
    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0
    per_step = elapsed / n_steps
    print(f"\n  Numba JIT: {per_step:.4f} s/step for 100K agents")
    assert elapsed < 50.0, f"10 steps took {elapsed:.1f}s, expected <50s"
    assert sim.pool.alive.sum() > 0


@pytest.mark.slow
def test_hexsim_100k_agents_under_half_second_per_step():
    """100K agents on Columbia workspace should run < 0.5s/step.

    Measures end-to-end performance with config_columbia.yaml settings
    (estuary overrides disabled via high thresholds).
    """
    import os
    if not os.path.exists("Columbia River Migration Model/Columbia [small]"):
        pytest.skip("Columbia workspace not found")

    cfg = load_config("config_columbia.yaml")
    # Warmup Numba
    sim_warmup = Simulation(cfg, n_agents=10, rng_seed=99)
    sim_warmup.run(1)

    cfg2 = load_config("config_columbia.yaml")
    sim = Simulation(cfg2, n_agents=100_000, rng_seed=42)
    n_steps = 10
    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0
    per_step = elapsed / n_steps
    print(f"\n  HexSim 100K: {per_step:.4f} s/step")
    assert per_step < 0.5, f"{per_step:.3f}s/step exceeds 0.5s target"
    assert sim.pool.alive.sum() > 0


def test_vectorized_movement_correctness():
    """Full simulation should still produce valid, reproducible results."""
    import salmon_ibm.movement as mov
    orig = mov.FORCE_NUMPY
    try:
        mov.FORCE_NUMPY = True
        cfg = load_config("config_curonian_minimal.yaml")

        sim1 = Simulation(cfg, n_agents=100, data_dir="data", rng_seed=42)
        sim1.run(n_steps=20)

        cfg2 = load_config("config_curonian_minimal.yaml")
        sim2 = Simulation(cfg2, n_agents=100, data_dir="data", rng_seed=42)
        sim2.run(n_steps=20)

        np.testing.assert_array_equal(sim1.pool.tri_idx, sim2.pool.tri_idx)
        np.testing.assert_array_almost_equal(sim1.pool.ed_kJ_g, sim2.pool.ed_kJ_g)
        assert sim1.pool.alive.sum() > 0, "Some agents should survive 20 steps"
    finally:
        mov.FORCE_NUMPY = orig
