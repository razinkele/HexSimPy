"""Performance regression tests for the simulation engine."""
import time

import numpy as np
import pytest

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


@pytest.mark.slow
def test_step_performance_5000_agents():
    """Benchmark: 10 steps with 5000 agents should complete in <10s.

    This is a loose bound for the un-optimized baseline.
    After vectorization, tighten to <2s.
    """
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=5000, data_dir="data", rng_seed=42)

    n_steps = 10
    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0

    per_step = elapsed / n_steps
    print(f"\n  Baseline: {per_step:.4f} s/step for 5000 agents")
    assert elapsed < 10.0, f"10 steps took {elapsed:.1f}s, expected <10s"
