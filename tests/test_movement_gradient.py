"""Tests for C4 — movement gradient (substrate fix).

Spec: docs/superpowers/specs/2026-05-07-c4-movement-gradient-design.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Import the build-script function. The script lives outside the
# salmon_ibm package, so add the scripts directory to sys.path.
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class _SyntheticMesh:
    """Minimal mesh for compute_dist_from_sea unit tests.

    Provides the attributes the function reads: nbr_starts, nbr_idx,
    centroids, water_mask, reach_id, reach_names. Adds N_cells convenience
    via len(reach_id).
    """
    def __init__(
        self,
        nbr_starts: np.ndarray,
        nbr_idx: np.ndarray,
        centroids: np.ndarray,
        water_mask: np.ndarray,
        reach_id: np.ndarray,
        reach_names: list[str],
    ):
        self.nbr_starts = nbr_starts
        self.nbr_idx = nbr_idx
        self.centroids = centroids
        self.water_mask = water_mask
        self.reach_id = reach_id
        self.reach_names = reach_names

    @property
    def N_cells(self) -> int:
        return len(self.reach_id)


def test_compute_dist_from_sea_raises_on_disconnected_component():
    """C4 Test 5: synthetic mesh with two disconnected components
    (one with sea, one without). compute_dist_from_sea must raise
    RuntimeError naming the unreachable reach."""
    from build_h3_multires_landscape import compute_dist_from_sea

    # Component A: cells 0,1 (sea + adjacent), reach=OpenBaltic.
    # Component B: cells 2,3 (river, no path to sea), reach=Nemunas.
    nbr_starts = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    nbr_idx = np.array([1, 0, 3, 2], dtype=np.int32)
    centroids = np.array([
        [55.0, 21.0],  # cell 0
        [55.0, 21.001],  # cell 1
        [55.5, 21.5],  # cell 2
        [55.5, 21.501],  # cell 3
    ], dtype=np.float64)
    water_mask = np.ones(4, dtype=bool)
    reach_id = np.array([0, 0, 1, 1], dtype=np.int8)
    reach_names = ["OpenBaltic", "Nemunas"]

    mesh = _SyntheticMesh(nbr_starts, nbr_idx, centroids, water_mask,
                          reach_id, reach_names)
    with pytest.raises(RuntimeError, match=r"Nemunas"):
        compute_dist_from_sea(mesh)


def _make_chain_mesh(n: int = 10, sea_at_zero: bool = True) -> _SyntheticMesh:
    """10-cell bidirectional chain. Cell 0 = OpenBaltic source.
    Cells 1..n-1 = Nemunas (river). Used by Tests 1, 2, 2b, 3, 5b."""
    # Bidirectional CSR: cell i has neighbors {i-1, i+1} where they exist.
    nbr_starts = np.zeros(n + 1, dtype=np.int32)
    nbrs = []
    for i in range(n):
        if i > 0:
            nbrs.append(i - 1)
        if i < n - 1:
            nbrs.append(i + 1)
        nbr_starts[i + 1] = len(nbrs)
    nbr_idx = np.array(nbrs, dtype=np.int32)
    # Centroids spaced 100m apart along a meridian (uniform haversine).
    centroids = np.array(
        [[55.0 + i * 0.0009, 21.0] for i in range(n)],
        dtype=np.float64,
    )
    water_mask = np.ones(n, dtype=bool)
    reach_id = np.zeros(n, dtype=np.int8)
    reach_id[1:] = 1  # cell 0 = OpenBaltic; cells 1..n-1 = Nemunas
    reach_names = ["OpenBaltic", "Nemunas"]
    return _SyntheticMesh(
        nbr_starts, nbr_idx, centroids, water_mask, reach_id, reach_names,
    )


def test_compute_dist_from_sea_deterministic_synthetic():
    """C4 Test 5b (synthetic part): two runs on the same mesh produce
    NaN-aware-equal output."""
    from build_h3_multires_landscape import compute_dist_from_sea

    mesh = _make_chain_mesh(n=10)
    out1 = compute_dist_from_sea(mesh)
    out2 = compute_dist_from_sea(mesh)
    assert np.array_equal(out1, out2, equal_nan=True), (
        "compute_dist_from_sea is non-deterministic on a 10-cell chain"
    )


def test_compute_dist_from_sea_y_junction_tie_break():
    """C4 Test 5c: 4-cell Y-junction with three equidistant non-source
    cells. Asserts byte-equal recompute."""
    from build_h3_multires_landscape import compute_dist_from_sea

    # Cell 0 = OpenBaltic source. Cells 1,2,3 each connected to cell 0
    # only — three equidistant arms. Bidirectional edges.
    nbr_starts = np.array([0, 3, 4, 5, 6], dtype=np.int32)
    nbr_idx = np.array([1, 2, 3,  0,  0,  0], dtype=np.int32)
    # All three arm-tips at the same lat offset but different lons
    # (cells 1, 2, 3 each differ from cell 0 by ~100m along distinct
    # bearings — haversine values are bit-identical because we use
    # identical math.cos/math.sin inputs).
    centroids = np.array([
        [55.0, 21.0],         # cell 0 (source)
        [55.0009, 21.0],      # cell 1 (north)
        [55.0, 21.00157],     # cell 2 (east; 0.00157 deg lon ~= 100m at 55N)
        [54.9991, 21.0],      # cell 3 (south)
    ], dtype=np.float64)
    water_mask = np.ones(4, dtype=bool)
    reach_id = np.array([0, 1, 1, 1], dtype=np.int8)  # 0=OpenBaltic,1=Nemunas
    reach_names = ["OpenBaltic", "Nemunas"]

    mesh = _SyntheticMesh(
        nbr_starts, nbr_idx, centroids, water_mask, reach_id, reach_names,
    )
    out1 = compute_dist_from_sea(mesh)
    out2 = compute_dist_from_sea(mesh)
    assert np.array_equal(out1, out2, equal_nan=True), (
        "Y-junction tie-break is non-deterministic"
    )
    # Cell 0 = source, distance 0.
    assert out1[0] == 0.0
    # Cells 1, 2, 3 all reachable, finite, > 0.
    assert np.all(np.isfinite(out1[1:]))
    assert np.all(out1[1:] > 0)
