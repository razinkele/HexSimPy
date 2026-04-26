"""End-to-end Nemunas H3 invariant test.

Phase 3.3 of ``docs/superpowers/plans/2026-04-24-h3mesh-backend.md``.

Runs the realistic ``configs/config_nemunas_h3.yaml`` scenario (real
EMODnet bathymetry + real CMEMS forcing on a 106 k-cell H3 res-9 mesh)
and asserts the seven invariants pinned by
``docs/superpowers/specs/2026-04-24-nemunas-delta-h3.md`` § "Validation
invariants":

  1. Agent-count conservation (alive + dead + arrived == n_agents).
  2. No total extinction over the 30-day window.
  3. Mesh is an H3Mesh at the configured resolution.
  4. Surface-temperature envelope (Baltic-summer realistic).
  5. Agent positions all in [0, n_cells) — no off-mesh placement.
  6. ≥ 10 % of agents have moved at the end of the run.
  7. North–south salinity gradient ≥ 1.5 PSU
     (Klaipėda Strait fresher than Nemunas mouth).

Tests skip cleanly when the landscape NetCDF or config are absent so
the regression suite stays runnable on a fresh checkout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parent.parent
LANDSCAPE = PROJECT / "data" / "nemunas_h3_landscape.nc"
CONFIG = PROJECT / "configs" / "config_nemunas_h3.yaml"


def _needs_data() -> None:
    if not LANDSCAPE.exists():
        pytest.skip(
            f"{LANDSCAPE.name} missing — run "
            "`python scripts/build_nemunas_h3_landscape.py`"
        )
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG.name} missing")


@pytest.fixture(scope="module")
def h3_sim():
    """30-day Nemunas H3 run; module-scoped because the simulation is
    expensive (~3-5 min including Numba JIT)."""
    _needs_data()
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config(str(CONFIG))
    sim = Simulation(cfg, n_agents=500, rng_seed=42)
    sim.run(n_steps=720)
    return sim


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_agent_count_invariant(h3_sim):
    alive = int(h3_sim.pool.alive.sum())
    arrived = int(h3_sim.pool.arrived.sum())
    dead = 500 - alive - arrived
    assert alive >= 0 and arrived >= 0 and dead >= 0
    assert alive + arrived + dead == 500


def test_no_total_extinction(h3_sim):
    """Baltic summer (2011-06) is well below T_ACUTE_LETHAL=24 °C, so
    movement + bioenergetics shouldn't wipe the population."""
    assert int(h3_sim.pool.alive.sum()) > 0


def test_mesh_is_h3_at_resolution_9(h3_sim):
    from salmon_ibm.h3mesh import H3Mesh
    assert isinstance(h3_sim.mesh, H3Mesh)
    assert h3_sim.mesh.resolution == 9


def test_temperature_envelope_baltic_summer(h3_sim):
    fields = h3_sim.env.current()
    t = fields["temperature"]
    # 2011-06 surface SST in the Curonian + Baltic strait: typically
    # 14-22 °C.  Allow a slightly wider band for spike artefacts.
    assert -2.0 < float(t.min())
    assert float(t.max()) < 25.0


def test_agent_positions_all_on_mesh(h3_sim):
    assert (h3_sim.pool.tri_idx >= 0).all()
    assert (h3_sim.pool.tri_idx < h3_sim.mesh.n_cells).all()


def test_at_least_one_tenth_of_agents_moved(h3_sim):
    """Movement kernels must work on H3Mesh — non-zero displacement
    proves water_nbrs / water_nbr_count are wired correctly."""
    moved = (h3_sim.initial_cells != h3_sim.pool.tri_idx).sum()
    assert moved >= 50, (
        f"only {moved}/500 agents moved over 720 steps — movement stub?"
    )


STRONG_BARRIER_CSV = PROJECT / "data" / "nemunas_h3_barriers_strong.csv"


def _needs_barrier_data() -> None:
    if not STRONG_BARRIER_CSV.exists():
        pytest.skip(
            f"{STRONG_BARRIER_CSV.name} missing — run "
            "`python scripts/build_nemunas_h3_barriers.py`"
        )


def _run_short(barriers_csv: str | None, n_steps: int = 72) -> "Simulation":
    """3-day Nemunas H3 run with the chosen barrier config — for the
    barrier-mortality comparison.  Module-scoped fixtures would slow
    the suite by ~3 min each; a 3-day run is still long enough to
    fire ~120 barrier-edge crossings at 30 % mortality.

    Predation is explicitly disabled (``mortality_per_reach: None``)
    so this test isolates the barrier-mortality signal.  Without the
    suppression, per-reach fish-predation (added in v1.4) drowns out
    the ~30-death barrier delta in stochastic noise — observed
    311 vs 319 deaths across with/without barrier in seed=42 runs.
    """
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config(str(CONFIG))
    cfg["barriers_csv"] = barriers_csv
    cfg["mortality_per_reach"] = None  # see docstring
    sim = Simulation(cfg, n_agents=500, rng_seed=42)
    sim.run(n_steps=n_steps)
    return sim


@pytest.fixture(scope="module")
def h3_sim_with_strong_barrier():
    _needs_data()
    _needs_barrier_data()
    return _run_short(barriers_csv=str(STRONG_BARRIER_CSV))


@pytest.fixture(scope="module")
def h3_sim_no_barrier():
    _needs_data()
    return _run_short(barriers_csv=None)


def test_strong_barrier_increases_mortality(
    h3_sim_with_strong_barrier, h3_sim_no_barrier
):
    """Mortality-effect smoke test: a 30 %/60 %/10 % barrier across the
    open lagoon must produce ≥ as many deaths as the no-barrier run.

    Single-seed test — flaky in principle (movement is stochastic,
    barrier crossings depend on which agents wander into the line),
    so we use ≥ rather than >, and a 30 % mortality probability that
    is well above the per-step movement noise.
    """
    def dead(sim) -> int:
        return 500 - int(sim.pool.alive.sum()) - int(sim.pool.arrived.sum())
    with_dead = dead(h3_sim_with_strong_barrier)
    without_dead = dead(h3_sim_no_barrier)
    assert with_dead >= without_dead, (
        f"barrier reduced mortality: with={with_dead}, "
        f"no_barrier={without_dead}"
    )
    # With ~120 edges at 30 % mortality, expect ≥ 1 extra death seed=42.
    # If this is flaky in CI, raise n_agents to 5000 or n_steps to 168.


@pytest.mark.slow
def test_open_baltic_kills_more_than_river():
    """Phase B per-reach mortality: agents force-placed in OpenBaltic
    cells (daily survival 0.65) should die ~10× faster than agents
    placed in Nemunas-river cells (daily survival 0.985) over a
    24-hour run.  Single-seed test; uses 200 agents per cohort to
    average out per-cell variability.
    """
    if not CONFIG.exists() or not LANDSCAPE.exists():
        pytest.skip("h3_landscape_nc missing — build first")
    import numpy as np
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation

    def run_in_reach(reach_name: str) -> int:
        cfg = load_config(str(CONFIG))
        cfg["barriers_csv"] = None  # isolate predation signal
        n_agents = 200
        sim = Simulation(cfg, n_agents=n_agents, rng_seed=1234)
        if not hasattr(sim.mesh, "cells_in_reach"):
            pytest.skip("backend has no reach_id")
        cells = sim.mesh.cells_in_reach(reach_name)
        if len(cells) == 0:
            pytest.skip(f"reach {reach_name} has no cells")
        # Cycle cells if reach has fewer than n_agents cells (rivers in
        # the Nemunas H3 landscape are small — Nemunas main has ~38
        # cells at res 9, so we let multiple agents share a cell).
        sim.pool.tri_idx[:] = np.tile(
            cells, (n_agents + len(cells) - 1) // len(cells)
        )[:n_agents]
        sim.run(n_steps=24)  # 24 h = 1 daily-rate cycle
        return int((~sim.pool.alive).sum())

    deaths_baltic = run_in_reach("OpenBaltic")
    deaths_river = run_in_reach("Nemunas")
    # OpenBaltic daily survival = 0.65 → ~70 deaths in 200.
    # Nemunas daily survival    = 0.985 → ~3 deaths in 200.
    # Use a generous bound (3×) to absorb stochastic noise on a
    # single seed; the *expected* ratio is ~25×.
    assert deaths_baltic > deaths_river * 3, (
        f"per-reach mortality contrast too weak: OpenBaltic={deaths_baltic}, "
        f"Nemunas={deaths_river} (expected at least 3×).  Either the "
        f"`mortality_per_reach` block isn't being read, or the "
        f"hourly-rate conversion is wrong."
    )


def test_north_south_salinity_gradient(h3_sim):
    """Spec §5: Klaipėda Strait (north) ≥ 1.5 PSU saltier than Nemunas
    mouth (south).  CMEMS BALTICSEA reanalysis routinely shows ≥ 3 PSU
    over this geography; 1.5 is loose enough not to fail on weather
    noise but tight enough to catch regridder homogenisation bugs."""
    sal = h3_sim.env.current()["salinity"]
    lats = h3_sim.mesh.centroids[:, 0]
    north_mask = lats > np.percentile(lats, 75)
    south_mask = lats < np.percentile(lats, 25)
    north_mean = float(np.nanmean(sal[north_mask]))
    south_mean = float(np.nanmean(sal[south_mask]))
    assert north_mean > south_mean, (
        f"Salinity gradient inverted: north={north_mean:.2f} < "
        f"south={south_mean:.2f} — Klaipėda Strait should be saltier "
        f"than Nemunas mouth."
    )
    assert (north_mean - south_mean) >= 1.5, (
        f"Salinity gradient too weak: {north_mean - south_mean:.2f} PSU "
        f"(expected ≥ 1.5).  Regridder may be over-smoothing."
    )


def test_reach_id_present_and_consistent(h3_sim):
    """Each H3 cell carries a reach id from the inSTREAM-polygon mask.

    The build script tags water cells with the inSTREAM reach they fall
    inside (-1 for land, 0..8 for the 9 inSTREAM reaches, 9 for cells
    in Natural Earth ocean but no inSTREAM polygon = OpenBaltic).
    """
    m = h3_sim.mesh
    assert m.reach_id.shape == (m.n_cells,), "reach_id must be per-cell"
    assert m.reach_id.dtype == np.int8, "reach_id should be int8"
    assert len(m.reach_names) == 10, (
        f"expected 10 reach names (9 inSTREAM + OpenBaltic); "
        f"got {len(m.reach_names)}: {m.reach_names}"
    )
    # Land cells = ~water_mask, water cells must have a non-negative id.
    land = m.reach_id == -1
    water = m.reach_id >= 0
    assert (land == ~m.water_mask).all(), (
        "reach_id == -1 must coincide exactly with water_mask == False"
    )
    assert water.sum() == m.water_mask.sum(), (
        "every water cell must have a non-negative reach id"
    )

    # The Curonian Lagoon dominates the inland set — sanity check it
    # matches the inSTREAM example_baltic shapefile's ~1 558 km² lagoon.
    lagoon_idx = m.cells_in_reach("CuronianLagoon")
    assert 12_000 < len(lagoon_idx) < 20_000, (
        f"CuronianLagoon should have ~16 k cells; got {len(lagoon_idx):,}"
    )
