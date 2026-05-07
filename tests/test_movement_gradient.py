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
