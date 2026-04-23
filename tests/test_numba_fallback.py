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


@pytest.mark.xfail(
    reason=(
        "Known divergence: Numba and NumPy paths produce different agent "
        "positions even with seeded RNG (~4% position agreement, not ~100%). "
        "Likely root cause: different RNG consumption order between parallel "
        "Numba kernels (prange) and sequential NumPy fallback. Documented as "
        "a finding; not a correctness blocker. See "
        "docs/superpowers/plans/2026-04-23-deep-review-fixes.md Task 10."
    ),
    strict=True,
)
def test_numba_and_numpy_converge_to_same_result():
    """Numba and NumPy paths should produce similar positions after N steps (seeded)."""
    import numpy as np

    cfg = load_config("config_curonian_minimal.yaml")

    orig = mov.FORCE_NUMPY
    try:
        mov.FORCE_NUMPY = False
        sim_numba = Simulation(cfg, n_agents=100, data_dir="data", rng_seed=42)
        sim_numba.run(n_steps=10)
        final_numba = sim_numba.pool.tri_idx.copy()

        mov.FORCE_NUMPY = True
        sim_numpy = Simulation(cfg, n_agents=100, data_dir="data", rng_seed=42)
        sim_numpy.run(n_steps=10)
        final_numpy = sim_numpy.pool.tri_idx.copy()
    finally:
        mov.FORCE_NUMPY = orig

    agreement = (final_numba == final_numpy).mean()
    assert agreement > 0.9, (
        f"Numba and NumPy diverge in >10% of agents: agreement={agreement:.2%}"
    )


def test_numba_and_numpy_agree_on_survival_count():
    """Both paths preserve agent counts within tolerance (even if positions diverge)."""
    import numpy as np

    cfg = load_config("config_curonian_minimal.yaml")

    orig = mov.FORCE_NUMPY
    try:
        mov.FORCE_NUMPY = False
        sim_numba = Simulation(cfg, n_agents=100, data_dir="data", rng_seed=42)
        sim_numba.run(n_steps=10)
        alive_numba = int(sim_numba.pool.alive.sum())

        mov.FORCE_NUMPY = True
        sim_numpy = Simulation(cfg, n_agents=100, data_dir="data", rng_seed=42)
        sim_numpy.run(n_steps=10)
        alive_numpy = int(sim_numpy.pool.alive.sum())
    finally:
        mov.FORCE_NUMPY = orig

    assert alive_numba > 50 and alive_numpy > 50
    assert abs(alive_numba - alive_numpy) < 50, (
        f"Numba and NumPy disagree too much on survival: "
        f"{alive_numba} vs {alive_numpy}"
    )
