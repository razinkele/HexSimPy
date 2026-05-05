"""Tests for hatchery vs wild C3.3 — homing precision divergence.

Spec: docs/superpowers/specs/2026-05-03-hatchery-c3.3-homing-design.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from salmon_ibm.baltic_params import (
    BalticBioParams,
    BalticSpeciesConfig,
    _apply_hatchery_overrides,
    load_baltic_species_config,
)


CONFIG_PATH = Path("configs/baltic_salmon_species.yaml")


def test_homing_precision_default_loads():
    """C3.3 test 1 (partial): default BalticBioParams has wild
    homing_precision = 0.95."""
    p = BalticBioParams()
    assert p.homing_precision == 0.95


def test_homing_precision_validation_rejects_out_of_range():
    """C3.3 test 2: __post_init__ raises ValueError on -0.1 and 1.5
    with a message naming the field."""
    with pytest.raises(ValueError, match=r"homing_precision"):
        BalticBioParams(homing_precision=-0.1)
    with pytest.raises(ValueError, match=r"homing_precision"):
        BalticBioParams(homing_precision=1.5)
    # Boundaries 0.0 and 1.0 are valid.
    BalticBioParams(homing_precision=0.0)
    BalticBioParams(homing_precision=1.0)


def test_homing_precision_in_scalar_override_fields():
    """C3.3 test 3: hatchery override flows through `dataclasses.replace`
    via the existing SCALAR_OVERRIDE_FIELDS mechanism."""
    wild = BalticBioParams()
    overrides = {"homing_precision": 0.65}
    hatchery = _apply_hatchery_overrides(wild, overrides)
    assert hatchery.homing_precision == 0.65
    # Wild unchanged:
    assert wild.homing_precision == 0.95


def test_homing_precision_loads_from_yaml():
    """C3.3 test 1 (full): deployed YAML has wild=0.95 + hatchery=0.65."""
    cfg = load_baltic_species_config(CONFIG_PATH)
    assert isinstance(cfg, BalticSpeciesConfig)
    assert cfg.wild.homing_precision == 0.95
    assert cfg.hatchery is not None
    assert cfg.hatchery.homing_precision == 0.65
    assert cfg.hatchery is not cfg.wild


# --- Task 3: _BranchEntryCache + _branch_entry_cell -----------------------

from salmon_ibm.delta_routing import _branch_entry_cell


class _CacheTestMesh:
    """Minimal mesh for testing _branch_entry_cell cache invalidation.
    Only needs reach_id and reach_names attributes."""
    def __init__(self, reach_id: np.ndarray, reach_names: list[str]):
        self.reach_id = reach_id
        self.reach_names = reach_names


def test_branch_entry_cell_cache_invalidates_on_reassignment():
    """C3.3 test 15: _BranchEntryCache uses identity comparison
    (not id() int) — sound under CPython id-recycling. Reassigning
    mesh.reach_id to a new array with a DIFFERENT min-index for the
    branch must produce a fresh lookup, not return the cached old
    min-index."""
    # Original: branch rid=0 has cells at indices [3, 7, 9]; min = 3.
    original = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.int8)
    mesh = _CacheTestMesh(
        reach_id=original,
        reach_names=["Atmata", "Skirvyte"],
    )
    M_old = _branch_entry_cell(mesh, branch_rid=0)
    assert M_old == 3  # caches

    # Reassign to a new array where rid=0's cells are now at [1, 5];
    # min = 1 (DIFFERENT from cached 3 — load-bearing for the test).
    new_arr = np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int8)
    assert np.where(new_arr == 0)[0].min() != M_old  # M_new != M_old
    mesh.reach_id = new_arr

    M_new = _branch_entry_cell(mesh, branch_rid=0)
    assert M_new == 1  # NOT 3 — cache invalidated by identity check.


# --- Task 5: assert_branch_topology --------------------------------------

from salmon_ibm.delta_routing import assert_branch_topology


def test_assert_branch_topology_no_op_on_legacy_mesh():
    """assert_branch_topology no-ops when reach_names is missing
    (legacy TriMesh / HexMesh fallback paths)."""
    class _LegacyMesh:
        reach_id = np.array([0, 0, 0], dtype=np.int8)
    # No reach_names attribute → no-op, no raise.
    assert_branch_topology(_LegacyMesh())


def test_assert_branch_topology_raises_on_missing_branch_cells():
    """C3.3 test 12 (a): raises ValueError when a branch_rid has
    no cells on the mesh. Error message names at least one of
    Atmata/Skirvyte/Gilija."""
    # mesh has reach_names listing all 3 branches but reach_id only
    # has cells for indices 0 (Skirvyte) and 1 (Atmata) — Gilija
    # (index 2) has zero cells.
    class _MissingCellsMesh:
        reach_id = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        reach_names = ["Skirvyte", "Atmata", "Gilija"]
    with pytest.raises(ValueError, match=r"Atmata|Skirvyte|Gilija"):
        assert_branch_topology(_MissingCellsMesh())


def test_assert_branch_topology_raises_on_missing_fractions_entry():
    """C3.3 test 12 (b): raises ValueError when a branch_rid has
    cells but the branch name is missing from BRANCH_FRACTIONS.
    Error message references BRANCH_FRACTIONS."""
    class _MissingFractionsMesh:
        # All 3 branches have cells, but reach_names contains
        # an unknown branch name "Rusne" that's not in
        # BRANCH_FRACTIONS.
        reach_id = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
        reach_names = ["Skirvyte", "Atmata", "Rusne"]
    with pytest.raises(ValueError, match=r"BRANCH_FRACTIONS"):
        assert_branch_topology(_MissingFractionsMesh())
