"""Verify NumPy fallback produces valid results when Numba is disabled."""
import numpy as np
import pytest
from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation
import salmon_ibm.movement as mov


def test_numpy_fallback_produces_valid_results():
    orig = mov.FORCE_NUMPY
    try:
        mov.FORCE_NUMPY = True
        cfg = load_config("config_curonian_minimal.yaml")
        sim = Simulation(cfg, n_agents=200, data_dir="data", rng_seed=42)
        sim.run(n_steps=10)
        assert sim.pool.alive.sum() > 0
    finally:
        mov.FORCE_NUMPY = orig


def test_numba_and_numpy_both_move_agents():
    cfg = load_config("config_curonian_minimal.yaml")
    sim_np = Simulation(cfg, n_agents=200, data_dir="data", rng_seed=42)
    start_tris = sim_np.pool.tri_idx.copy()
    orig = mov.FORCE_NUMPY
    try:
        mov.FORCE_NUMPY = True
        sim_np.run(n_steps=5)
        moved_np = (sim_np.pool.tri_idx != start_tris).sum()
    finally:
        mov.FORCE_NUMPY = orig
    assert moved_np > 0, "NumPy path should move agents"
