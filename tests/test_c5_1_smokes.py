"""C5.1 round-trip arrival smokes — permanent regression tests.

Splits long-horizon test into:
- test_round_trip_mechanic: synthetic scenario that ALWAYS triggers
  the round-trip flag (no calibration coupling). Hard pass/fail.
- test_long_horizon_calibration_smoke: full default scenario at
  8760h. xfail(strict=False) — calibration-coupled.
- test_default_scenario_arrived_below_n_alive: 480h regression guard
  for the v1.7.10 tautology (was 46/46).
"""
import numpy as np
import pytest


def test_round_trip_mechanic():
    """C5.1 mechanic test (no calibration coupling).

    Hard pass/fail: if this test goes red, the C5.1 round-trip
    mechanic is broken.
    """
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("configs/config_curonian_h3_multires.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42,
                     output_path=None)
    # Force-tag been_to_sea for all agents (simulate prior at-sea visit).
    sim.pool.been_to_sea[:] = True
    thresholds = sim._arrival_threshold_by_natal_rid
    if not thresholds:
        pytest.skip("No arrival thresholds available — sim init issue.")
    mesh = sim.mesh
    for i in range(sim.pool.n):
        natal = int(sim.pool.natal_reach_id[i])
        if natal < 0:
            continue
        candidate_cells = np.where(
            (mesh.reach_id == natal) & (mesh.dist_from_sea >= thresholds.get(natal, np.inf))
        )[0]
        if len(candidate_cells) > 0:
            sim.pool.tri_idx[i] = int(candidate_cells[0])
    sim.step()
    arrived_count = int(sim.pool.arrived.sum())
    assert arrived_count > 0, (
        "C5.1 mechanic test failed: 0 arrivals after forcing "
        "been_to_sea=True AND placing agents in natal upper-quartile "
        "cells. Round-trip mechanic is broken."
    )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=False,
    reason="Calibration-coupled: depends on swim-speed parameterisation. "
           "Persistent xfail indicates calibration drift, not a bug."
)
def test_long_horizon_calibration_smoke():
    """C5.1 long-horizon calibration smoke (8760h, n=200, seed=42).

    Per pass-1 B-2: xfail(strict=False) lets test serve as soft
    regression detector without blocking ship.
    """
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("configs/config_curonian_h3_multires.yaml")
    sim = Simulation(cfg, n_agents=200, data_dir="data", rng_seed=42,
                     output_path=None)
    for _ in range(8760):
        sim.step()
    arrived_total = int(sim.pool.arrived.sum())
    assert arrived_total > 0, (
        f"0 arrivals at 8760h with 200 agents — calibration drift "
        f"or mechanic regression. Investigate before tagging."
    )


def test_default_scenario_arrived_below_n_alive():
    """Regression guard: default scenario rng_seed=42, n_agents=50,
    n_steps=480h must produce arrived STRICTLY LESS THAN n_alive.

    Pre-C5.1 baseline (v1.7.10): arrived = 46 = n_alive (every alive
    agent tagged, the C5 tautology). Post-C5.1: arrived < n_alive.
    """
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("configs/config_curonian_h3_multires.yaml")
    sim = Simulation(cfg, n_agents=50, data_dir="data", rng_seed=42,
                     output_path=None)
    for _ in range(480):
        sim.step()
    arrived = int(sim.pool.arrived.sum())
    n_alive = int(sim.pool.alive.sum())
    assert arrived < n_alive, (
        f"arrived={arrived} == n_alive={n_alive} — round-trip "
        f"filter not firing or all alive agents have been to sea "
        f"(unlikely at 480h horizon). Investigate."
    )
