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
