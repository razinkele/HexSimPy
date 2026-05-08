"""Tests for C5 — arrival event.

Spec: docs/superpowers/specs/2026-05-08-c5-arrival-event-design.md
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest


def test_compute_arrival_thresholds_returns_75th_percentile_per_reach():
    """C5 Test 5c (threshold-computation): build a synthetic Simulation
    fixture with a known mesh.dist_from_sea distribution; assert
    `_compute_arrival_thresholds` returns 75th-percentile per reach.

    Synthetic mesh: 12 cells.
    - cells 0..3: reach_id=0 (OpenBaltic), dist=[0, 100, 200, 300]
        → 75th percentile = 225.0
    - cells 4..7: reach_id=1 (CuronianLagoon), dist=[400, 500, 600, 700]
        → 75th percentile = 625.0
    - cells 8..11: reach_id=2 (Nemunas), dist=[800, 900, 1000, 1100]
        → 75th percentile = 1025.0
    """
    # Minimal Simulation-like object that exposes `_compute_arrival_thresholds`
    # via direct method call. Build the mesh attributes the method reads.
    class _MeshShim:
        pass
    mesh = _MeshShim()
    mesh.reach_id = np.array(
        [0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2], dtype=np.int8,
    )
    mesh.dist_from_sea = np.arange(12, dtype=np.float32) * 100.0
    mesh.water_mask = np.ones(12, dtype=bool)
    mesh.reach_names = ["OpenBaltic", "CuronianLagoon", "Nemunas"]

    # Construct a minimal Simulation-shaped class that owns the method.
    from salmon_ibm.simulation import Simulation
    sim = Simulation.__new__(Simulation)  # bypass __init__
    sim.mesh = mesh

    thresholds = sim._compute_arrival_thresholds()
    assert thresholds == pytest.approx({
        0: 225.0,
        1: 625.0,
        2: 1025.0,
    })


def test_compute_arrival_thresholds_skips_reach_with_no_water_cells(caplog):
    """C5: if a reach has zero finite-dist water cells, threshold is
    not computed; warning emitted with err-id."""
    class _MeshShim:
        pass
    mesh = _MeshShim()
    mesh.reach_id = np.array([0, 0, 1, 1], dtype=np.int8)
    mesh.dist_from_sea = np.array([0.0, 100.0, np.nan, np.nan], dtype=np.float32)
    mesh.water_mask = np.array([True, True, True, True], dtype=bool)
    mesh.reach_names = ["OpenBaltic", "Nemunas"]

    from salmon_ibm.simulation import Simulation
    sim = Simulation.__new__(Simulation)
    sim.mesh = mesh

    caplog.set_level(logging.WARNING, logger="salmon_ibm.simulation")
    thresholds = sim._compute_arrival_thresholds()
    assert 0 in thresholds  # OpenBaltic has water cells
    assert 1 not in thresholds  # Nemunas all-NaN, skipped
    matching = [
        r for r in caplog.records
        if "c5-arrival-skipped-reach" in r.getMessage()
        and "Nemunas" in r.getMessage()
    ]
    assert matching, (
        f"Expected c5-arrival-skipped-reach warning naming Nemunas; "
        f"got {[r.getMessage() for r in caplog.records]!r}"
    )


def test_compute_arrival_thresholds_returns_empty_on_legacy_mesh():
    """C5: legacy non-Baltic mesh (no dist_from_sea attribute) →
    empty thresholds dict; ArrivalEvent will no-op at execute time."""
    class _LegacyMesh:
        reach_id = np.array([0, 1], dtype=np.int8)
        # NO dist_from_sea attribute.
    from salmon_ibm.simulation import Simulation
    sim = Simulation.__new__(Simulation)
    sim.mesh = _LegacyMesh()
    assert sim._compute_arrival_thresholds() == {}


def _make_arrival_landscape(mesh, sim, fields=None, rng_seed=0):
    """Helper: minimal landscape dict for ArrivalEvent.execute tests."""
    return {
        "sim": sim,
        "mesh": mesh,
        "fields": fields or {},
        "rng": np.random.default_rng(rng_seed),
    }


def _make_arrival_pool(n, *, behavior=0, alive=True, arrived=False,
                       natal_rid=1, tri_idx=0):
    """Helper: minimal pool stand-in with the fields ArrivalEvent reads."""
    class _FakePool:
        pass
    pool = _FakePool()
    pool.tri_idx = np.full(n, tri_idx, dtype=np.intp)
    pool.alive = np.full(n, alive, dtype=bool)
    pool.arrived = np.full(n, arrived, dtype=bool)
    pool.natal_reach_id = np.full(n, natal_rid, dtype=np.int8)
    pool.behavior = np.full(n, behavior, dtype=np.int8)
    return pool


def _make_arrival_population(pool):
    """Helper: minimal Population stand-in (just .pool accessor)."""
    class _FakePop:
        pass
    pop = _FakePop()
    pop.pool = pool
    return pop


def _make_arrival_sim(thresholds: dict[int, float]):
    """Helper: minimal Simulation stand-in for landscape["sim"]."""
    class _FakeSim:
        pass
    sim = _FakeSim()
    sim._arrival_threshold_by_natal_rid = thresholds
    return sim


def _arrival_test_mesh(n=12, threshold_75th=825.0):
    """Helper: synthetic 12-cell single-reach mesh for arrival tests.

    dist_from_sea = arange * 100; reach_id = 1 for all cells.
    75th percentile = 825.0 (cells 9, 10, 11 qualify with dist 900+).
    """
    class _FakeMesh:
        pass
    mesh = _FakeMesh()
    mesh.reach_id = np.full(n, 1, dtype=np.int8)
    mesh.dist_from_sea = np.arange(n, dtype=np.float32) * 100.0
    mesh.water_mask = np.ones(n, dtype=bool)
    return mesh


def test_arrival_basic():
    """C5 Test 1: agent at upper-natal cell + above threshold → arrived."""
    from salmon_ibm.events_builtin import ArrivalEvent

    mesh = _arrival_test_mesh()
    pool = _make_arrival_pool(1, natal_rid=1, tri_idx=10)  # cell 10, dist=1000
    pop = _make_arrival_population(pool)
    sim = _make_arrival_sim({1: 825.0})
    landscape = _make_arrival_landscape(mesh, sim)

    ArrivalEvent().execute(pop, landscape, t=0, mask=pool.alive)
    assert pool.arrived[0] == True


def test_arrival_threshold_boundary():
    """C5 Test 2: at-threshold cell → arrived True (>=, not >);
    below-threshold cell → arrived stays False."""
    from salmon_ibm.events_builtin import ArrivalEvent

    mesh = _arrival_test_mesh()
    # Cell 9: dist=900, exactly above the 825 threshold (>=).
    pool_at = _make_arrival_pool(1, natal_rid=1, tri_idx=9)
    pop_at = _make_arrival_population(pool_at)
    sim_at = _make_arrival_sim({1: 825.0})
    landscape_at = _make_arrival_landscape(mesh, sim_at)
    ArrivalEvent().execute(pop_at, landscape_at, t=0, mask=pool_at.alive)
    assert pool_at.arrived[0] == True

    # Cell 8: dist=800, below 825 threshold.
    pool_below = _make_arrival_pool(1, natal_rid=1, tri_idx=8)
    pop_below = _make_arrival_population(pool_below)
    sim_below = _make_arrival_sim({1: 825.0})
    landscape_below = _make_arrival_landscape(mesh, sim_below)
    ArrivalEvent().execute(pop_below, landscape_below, t=0, mask=pool_below.alive)
    assert pool_below.arrived[0] == False

    # Exact-threshold edge: synthesise dist value matching threshold exactly.
    mesh.dist_from_sea[5] = 825.0  # cell 5 = exact threshold
    pool_exact = _make_arrival_pool(1, natal_rid=1, tri_idx=5)
    pop_exact = _make_arrival_population(pool_exact)
    sim_exact = _make_arrival_sim({1: 825.0})
    landscape_exact = _make_arrival_landscape(mesh, sim_exact)
    ArrivalEvent().execute(pop_exact, landscape_exact, t=0, mask=pool_exact.alive)
    assert pool_exact.arrived[0] == True  # >= boundary; True


def test_arrival_stray_hatchery_never_arrives():
    """C5 Test 5: agent with natal_reach_id=2 currently at upper cells
    of reach_id=1 → arrived stays False (cross-reach matching rejected)."""
    from salmon_ibm.events_builtin import ArrivalEvent

    # Two-reach mesh: cells 0-5 reach_id=1, cells 6-11 reach_id=2.
    class _FakeMesh:
        pass
    mesh = _FakeMesh()
    mesh.reach_id = np.array(
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int8,
    )
    mesh.dist_from_sea = np.arange(12, dtype=np.float32) * 100.0
    mesh.water_mask = np.ones(12, dtype=bool)

    # Agent natal=2 (Atmata) but currently at reach_id=1 (Skirvyte) cell 5.
    pool = _make_arrival_pool(1, natal_rid=2, tri_idx=5)
    pop = _make_arrival_population(pool)
    sim = _make_arrival_sim({1: 425.0, 2: 1025.0})
    landscape = _make_arrival_landscape(mesh, sim)

    ArrivalEvent().execute(pop, landscape, t=0, mask=pool.alive)
    assert pool.arrived[0] == False  # cross-reach mismatch → never arrived


def test_arrival_sentinel_natal_never_arrives():
    """C5 Test 5b: pre-tagged sentinel agent (natal_reach_id=-1) at any
    cell → arrived stays False. Guards int8→int32 + in_range clamp."""
    from salmon_ibm.events_builtin import ArrivalEvent

    mesh = _arrival_test_mesh()
    # natal_reach_id = -1 (pre-tagging sentinel); place at upper cell.
    pool = _make_arrival_pool(1, natal_rid=-1, tri_idx=11)
    pop = _make_arrival_population(pool)
    sim = _make_arrival_sim({1: 825.0})
    landscape = _make_arrival_landscape(mesh, sim)

    ArrivalEvent().execute(pop, landscape, t=0, mask=pool.alive)
    assert pool.arrived[0] == False  # sentinel → inf threshold → never arrives
