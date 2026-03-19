"""Tests for multiprocessing ensemble runner."""
import os
import numpy as np
import pytest
from salmon_ibm.ensemble import run_ensemble

WS_PATH = "Columbia River Migration Model/Columbia [small]"
HAS_WS = os.path.exists(WS_PATH)
pytestmark = pytest.mark.skipif(not HAS_WS, reason="Columbia workspace not found")

COLUMBIA_CFG = {
    "grid": {"type": "hexsim"},
    "hexsim": {
        "workspace": WS_PATH,
        "species": "chinook",
        "temperature_csv": "River Temperature.csv",
    },
    "estuary": {
        "salinity_cost": {"S_opt": 0.5, "S_tol": 999, "k": 0.0},
        "do_avoidance": {"lethal": 0.0, "high": 0.0},
        "seiche_pause": {"dSSHdt_thresh_m_per_15min": 999},
    },
}


def test_ensemble_returns_correct_number_of_results():
    results = run_ensemble(COLUMBIA_CFG, n_replicates=4, n_agents=20, n_steps=5, n_workers=2)
    assert len(results) == 4
    for r in results:
        assert "seed" in r
        assert "history" in r
        assert len(r["history"]) == 5
        assert r["history"][-1]["n_alive"] > 0


def test_ensemble_replicates_are_different():
    results = run_ensemble(COLUMBIA_CFG, n_replicates=3, n_agents=50, n_steps=10, n_workers=1)
    mean_eds = [r["history"][-1]["mean_ed"] for r in results]
    assert len(set(round(e, 6) for e in mean_eds)) > 1, "Replicates should differ"


def test_ensemble_deterministic_with_base_seed():
    """Same base_seed should produce identical results.

    Uses n_workers=1 (sequential) because FORCE_NUMPY is a module-level
    flag that doesn't propagate to child processes spawned by Pool.
    """
    import salmon_ibm.movement as mov
    orig = mov.FORCE_NUMPY
    try:
        mov.FORCE_NUMPY = True
        r1 = run_ensemble(COLUMBIA_CFG, n_replicates=2, n_agents=20, n_steps=5,
                          n_workers=1, base_seed=42)
        r2 = run_ensemble(COLUMBIA_CFG, n_replicates=2, n_agents=20, n_steps=5,
                          n_workers=1, base_seed=42)
    finally:
        mov.FORCE_NUMPY = orig
    for a, b in zip(r1, r2):
        assert a["seed"] == b["seed"]
        for ha, hb in zip(a["history"], b["history"]):
            assert ha["n_alive"] == hb["n_alive"]
            assert abs(ha["mean_ed"] - hb["mean_ed"]) < 1e-10
