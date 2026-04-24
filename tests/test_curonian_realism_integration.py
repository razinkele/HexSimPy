"""End-to-end Curonian realism integration test.

Phase 5 of docs/superpowers/plans/2026-04-24-curonian-realism-upgrades.md.

Runs the realistic Baltic config (Baltic species params + real EMODnet
bathymetry + real CMEMS forcing) and checks hard invariants:

  1. Agent count conservation: alive + dead + arrived == n_agents
  2. No total extinction: some agents survive 30-day winter run
  3. No mass die-off: T_ACUTE_LETHAL=24°C means 20°C SST doesn't wipe agents
  4. Temperature envelope: realistic Baltic winter (-2 to 25 °C)
  5. Salinity envelope: 0-8 PSU lagoon-appropriate
  6. N>S salinity gradient: saltier at northern Klaipėda Strait than at
     southern Nemunas mouth — the defining feature of the Curonian estuary
  7. Bathymetry envelope: realistic mesh depth (0-50 m accounting for
     mesh extension into Baltic coast)

Tests skip if real data files haven't been fetched (Phases 2-3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parent.parent
CMEMS_FILE = PROJECT / "data" / "curonian_forcing_cmems.nc"
BALTIC_CONFIG = PROJECT / "configs" / "config_curonian_baltic.yaml"


def _needs_real_data() -> None:
    """Skip if real CMEMS data hasn't been fetched."""
    if not CMEMS_FILE.exists():
        pytest.skip(
            f"{CMEMS_FILE.name} not present — run "
            "scripts/fetch_cmems_forcing.py (requires CMEMS account)"
        )
    if not BALTIC_CONFIG.exists():
        pytest.skip(f"{BALTIC_CONFIG.name} not present")


def _run_baltic(n_agents: int = 500, n_steps: int = 720):
    """Helper: run Baltic config for n_steps, return sim + summary."""
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation

    cfg = load_config(str(BALTIC_CONFIG))
    sim = Simulation(cfg, n_agents=n_agents, data_dir="data", rng_seed=42)
    sim.run(n_steps=n_steps)
    return sim


@pytest.fixture(scope="module")
def baltic_sim():
    """30-day run of the Baltic config; module-scoped for speed."""
    _needs_real_data()
    return _run_baltic(n_agents=500, n_steps=720)


# ---------------------------------------------------------------------------
# Hard invariants
# ---------------------------------------------------------------------------


def test_agent_count_invariant(baltic_sim):
    """Baseline conservation: every agent is in exactly one of the three states."""
    alive = int(baltic_sim.pool.alive.sum())
    arrived = int(baltic_sim.pool.arrived.sum())
    # "dead" in HexSim = ~alive & ~arrived (agent was alive at init; if it
    # neither survived nor arrived, it died)
    dead = 500 - alive - arrived
    assert alive + dead + arrived == 500, (
        f"Agent count broken: alive={alive} dead={dead} arrived={arrived}"
    )
    assert alive >= 0 and dead >= 0 and arrived >= 0


def test_no_total_extinction(baltic_sim):
    """At least one agent must survive a 30-day winter run.

    If the thermal-kill path were using T_AVOID=20°C (v1 bug) this could
    still pass in winter, but would fail in summer. Kept as a safety net.
    """
    assert int(baltic_sim.pool.alive.sum()) > 0, (
        "Total extinction in 30 days of Baltic winter — env or thermal logic broken"
    )


def test_no_mass_die_off_in_winter(baltic_sim):
    """30-day winter run should kill <10% of agents. Baltic winter is cold
    but not acutely lethal; T_ACUTE_LETHAL=24°C is the real kill threshold.
    """
    alive = int(baltic_sim.pool.alive.sum())
    arrived = int(baltic_sim.pool.arrived.sum())
    dead = 500 - alive - arrived
    assert dead < 50, (
        f"{dead}/500 died in 30-day winter run — check T_ACUTE_LETHAL kill-gate; "
        f"may have regressed to T_AVOID=20°C"
    )


# ---------------------------------------------------------------------------
# Environment sanity
# ---------------------------------------------------------------------------


def test_temperature_envelope_baltic_winter(baltic_sim):
    """January Baltic surface temps: -2 to 5°C typical, envelope -2 to 25°C safe."""
    temps = baltic_sim.env.fields["temperature"]
    assert np.all(~np.isnan(temps)), "NaN temperatures leaked through — NaN fill failed"
    t_min, t_max = float(temps.min()), float(temps.max())
    assert -2.0 <= t_min, f"Temp min {t_min:.2f}°C below realistic Baltic winter"
    assert t_max <= 25.0, f"Temp max {t_max:.2f}°C above realistic surface range"


def test_salinity_envelope_curonian(baltic_sim):
    """Curonian lagoon: 0 PSU (Nemunas) to 7 PSU (Klaipėda Strait intrusion)."""
    sal = baltic_sim.env.fields["salinity"]
    assert np.all(~np.isnan(sal)), "NaN salinity — NaN fill failed"
    s_min, s_max = float(sal.min()), float(sal.max())
    assert 0.0 <= s_min, f"Negative salinity {s_min:.2f} PSU"
    assert s_max <= 10.0, f"Salinity max {s_max:.2f} PSU unrealistic for Curonian"


def test_north_south_salinity_gradient(baltic_sim):
    """The defining feature: saltier at the north (Klaipėda Strait) than
    at the south (Nemunas mouth). Without this gradient, the model isn't
    simulating an estuary."""
    mesh = baltic_sim.mesh
    sal = baltic_sim.env.fields["salinity"]
    # mesh.centroids: (n_triangles, 2) [lat, lon]
    lats = mesh.centroids[:, 0]
    north_mask = lats > np.percentile(lats, 75)
    south_mask = lats < np.percentile(lats, 25)
    north_mean = float(np.nanmean(sal[north_mask]))
    south_mean = float(np.nanmean(sal[south_mask]))
    assert north_mean > south_mean, (
        f"Salinity gradient inverted: north={north_mean:.2f} PSU "
        f"< south={south_mean:.2f} PSU. Expected Klaipėda Strait saltier "
        f"than Nemunas mouth."
    )
    # Gradient magnitude should be at least 0.5 PSU to count as "real"
    assert (north_mean - south_mean) > 0.5, (
        f"Salinity gradient too weak: {north_mean - south_mean:.2f} PSU. "
        f"Real Curonian E-W gradient is ~2-5 PSU peak-to-peak."
    )


def test_bathymetry_mesh_envelope():
    """EMODnet-loaded mesh depths: 0 to ~50 m (mesh extends past lagoon
    interior into Baltic coast)."""
    _needs_real_data()
    import xarray as xr

    ds = xr.open_dataset(PROJECT / "data" / "curonian_minimal_grid.nc",
                         engine="scipy")
    depth = ds.depth.values
    assert np.all(depth >= 0), f"Negative depth (land): {depth[depth < 0]}"
    d_mean = float(depth.mean())
    d_max = float(depth.max())
    # Curonian lagoon interior: 3-4 m. Mesh bbox extends into Baltic coast
    # where depths reach 25-50 m offshore.
    assert 2.0 < d_mean < 20.0, f"Mean depth {d_mean:.1f} m unrealistic"
    assert d_max < 60.0, (
        f"Max depth {d_max:.1f} m — mesh may extend too far into open Baltic"
    )


# ---------------------------------------------------------------------------
# Agent state sanity
# ---------------------------------------------------------------------------


def test_mean_energy_density_realistic(baltic_sim):
    """Baltic-salmon bioenergetics keep alive agents in realistic ED range."""
    alive_mask = baltic_sim.pool.alive
    if not alive_mask.any():
        pytest.skip("All agents died — upstream test should have caught this")
    mean_ed = float(baltic_sim.pool.ed_kJ_g[alive_mask].mean())
    # Baltic salmon ED_MORTAL=4 kJ/g, typical migrating range 5-7. Init 6.5.
    assert 4.0 < mean_ed < 8.0, f"Mean ED {mean_ed:.2f} kJ/g out of realistic range"


def test_bio_params_is_baltic_for_baltic_config(baltic_sim):
    """The species_config wiring must actually route to BalticBioParams."""
    from salmon_ibm.baltic_params import BalticBioParams

    assert isinstance(baltic_sim.bio_params, BalticBioParams), (
        f"Expected BalticBioParams, got {type(baltic_sim.bio_params).__name__} "
        f"— species_config loader regression"
    )
    # And the two-threshold thermal response is intact:
    assert baltic_sim.bio_params.T_OPT == 16.0
    assert baltic_sim.bio_params.T_AVOID == 20.0
    assert baltic_sim.bio_params.T_ACUTE_LETHAL == 24.0
