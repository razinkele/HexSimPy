"""Tests for hatchery vs wild C3.3 — homing precision divergence.

Spec: docs/superpowers/specs/2026-05-03-hatchery-c3.3-homing-design.md
"""

from __future__ import annotations

import ast
import inspect
import io
import logging
import tokenize
from pathlib import Path

import numpy as np
import pytest

from salmon_ibm.baltic_params import (
    BalticBioParams,
    BalticSpeciesConfig,
    _apply_hatchery_overrides,
    load_baltic_species_config,
)
from salmon_ibm.bioenergetics import BioParams
from salmon_ibm.delta_routing import (
    BRANCH_FRACTIONS,
    _branch_entry_cell,
    _branch_reach_ids,
    assert_branch_topology,
    update_exit_branch_id,
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


# --- Task 6: update_exit_branch_id origin-aware dispatch -----------------

class _BalticTestMesh:
    """Minimal Baltic-style mesh for C3.3 dispatch tests.

    Layout: cells 0-4 = Atmata (rid=0), cells 5-9 = Skirvyte (rid=1),
    cells 10-14 = Gilija (rid=2). 15 cells total. reach_names matches
    BRANCH_FRACTIONS keys exactly so assert_branch_topology passes.
    """
    def __init__(self):
        self.reach_id = np.concatenate([
            np.full(5, 0, dtype=np.int8),   # Atmata
            np.full(5, 1, dtype=np.int8),   # Skirvyte
            np.full(5, 2, dtype=np.int8),   # Gilija
        ])
        self.reach_names = ["Atmata", "Skirvyte", "Gilija"]


def _make_baltic_landscape(seed: int = 12345, *, hatchery: bool = True):
    """Helper: minimal Baltic-configured landscape for homing tests."""
    from salmon_ibm.baltic_params import HatcheryDispatch
    cfg = load_baltic_species_config(CONFIG_PATH)
    if hatchery:
        hd = HatcheryDispatch(
            params=cfg.hatchery,
            activity_lut=np.ones(5, dtype=np.float64),
        )
    else:
        hd = None
    return {
        "rng": np.random.default_rng(seed),
        "species_config": cfg,
        "hatchery_dispatch": hd,
    }


def _make_pool_with_origin(n: int, *, origin_value: int, natal_rid: int = 0):
    """Helper: pool with N agents, all on Atmata (cell 0), tagged with
    origin and natal_reach_id."""
    class _DispatchPool:
        pass
    pool = _DispatchPool()
    pool.tri_idx = np.zeros(n, dtype=np.intp)  # all on cell 0 = Atmata
    pool.alive = np.ones(n, dtype=bool)
    pool.exit_branch_id = np.full(n, -1, dtype=np.int8)
    pool.natal_reach_id = np.full(n, natal_rid, dtype=np.int8)
    pool.origin = np.full(n, origin_value, dtype=np.int8)
    return pool


def test_wild_perfect_homing_at_p_one(monkeypatch):
    """C3.3 test 4: p=1.0, N=10000 wild agents — exact equality
    (deterministic, no tolerance band). All agents must end up
    in their natal branch (Atmata, rid=0)."""
    from salmon_ibm.origin import ORIGIN_WILD
    mesh = _BalticTestMesh()
    pool = _make_pool_with_origin(10000, origin_value=ORIGIN_WILD, natal_rid=0)
    landscape = _make_baltic_landscape(seed=12345, hatchery=False)
    monkeypatch.setattr(
        landscape["species_config"].wild, "homing_precision", 1.0,
    )
    update_exit_branch_id(pool, mesh, landscape=landscape)
    assert (pool.exit_branch_id == 0).all()


def test_hatchery_strays_at_p_zero(monkeypatch):
    """C3.3 test 5: p=0.0, N=10000 hatchery agents distribute across
    non-natal branches with BRANCH_FRACTIONS-renormalised weights.
    Natal=Atmata; non-natal stray weights:
    Skirvyte 0.51/0.73=0.6986, Gilija 0.22/0.73=0.3014.
    Expected counts at N=10000: Skirvyte ~6986, Gilija ~3014."""
    from salmon_ibm.origin import ORIGIN_HATCHERY
    mesh = _BalticTestMesh()
    pool = _make_pool_with_origin(10000, origin_value=ORIGIN_HATCHERY, natal_rid=0)
    landscape = _make_baltic_landscape(seed=12345)
    monkeypatch.setattr(
        landscape["hatchery_dispatch"].params, "homing_precision", 0.0,
    )
    update_exit_branch_id(pool, mesh, landscape=landscape)
    assert (pool.exit_branch_id != 0).all()
    counts = np.bincount(pool.exit_branch_id, minlength=3)
    assert 6800 <= counts[1] <= 7200, f"Skirvyte count {counts[1]}"
    assert 2800 <= counts[2] <= 3200, f"Gilija count {counts[2]}"


def test_homing_skipped_if_natal_not_branch():
    """C3.3 test 6: agent with natal_reach_id NOT in branch_rids_set
    falls through to passive first-touch tag."""
    from salmon_ibm.origin import ORIGIN_HATCHERY
    mesh = _BalticTestMesh()
    pool = _make_pool_with_origin(
        5, origin_value=ORIGIN_HATCHERY, natal_rid=99,
    )
    landscape = _make_baltic_landscape(seed=12345)
    update_exit_branch_id(pool, mesh, landscape=landscape)
    # Agents on Atmata (cur_reach=0) get exit_branch_id=0 via passive tag.
    assert (pool.exit_branch_id == 0).all()


def test_homing_skipped_if_non_baltic():
    """C3.3 test 7: legacy non-Baltic species_config (wild = plain
    BioParams) → falls through to passive first-touch tag."""
    from salmon_ibm.origin import ORIGIN_WILD
    mesh = _BalticTestMesh()
    pool = _make_pool_with_origin(5, origin_value=ORIGIN_WILD, natal_rid=0)
    legacy_cfg = BalticSpeciesConfig(wild=BioParams(), hatchery=None)
    landscape = {
        "rng": np.random.default_rng(0),
        "species_config": legacy_cfg,
        "hatchery_dispatch": None,
    }
    update_exit_branch_id(pool, mesh, landscape=landscape)
    assert (pool.exit_branch_id == 0).all()


def test_homing_teleports_to_entry_cell(monkeypatch):
    """C3.3 test 8: when drawn ≠ current, agent's tri_idx lands on
    the cached `_branch_entry_cell` of drawn branch; exit_branch_id
    matches drawn."""
    from salmon_ibm.origin import ORIGIN_WILD
    mesh = _BalticTestMesh()
    pool = _make_pool_with_origin(1, origin_value=ORIGIN_WILD, natal_rid=0)
    landscape = _make_baltic_landscape(seed=12345, hatchery=False)
    monkeypatch.setattr(
        landscape["species_config"].wild, "homing_precision", 0.0,
    )
    update_exit_branch_id(pool, mesh, landscape=landscape)
    drawn = pool.exit_branch_id[0]
    assert drawn in (1, 2)
    expected_cell = 5 if drawn == 1 else 10
    assert pool.tri_idx[0] == expected_cell


def test_homing_no_teleport_when_drawn_equals_current(monkeypatch):
    """C3.3 test 9: when drawn == current, tri_idx is UNCHANGED;
    only exit_branch_id is set (no spurious teleport)."""
    from salmon_ibm.origin import ORIGIN_WILD
    mesh = _BalticTestMesh()
    pool = _make_pool_with_origin(1, origin_value=ORIGIN_WILD, natal_rid=0)
    pool.tri_idx[0] = 2  # specific Atmata cell (not entry cell 0)
    landscape = _make_baltic_landscape(seed=12345, hatchery=False)
    monkeypatch.setattr(
        landscape["species_config"].wild, "homing_precision", 1.0,
    )
    update_exit_branch_id(pool, mesh, landscape=landscape)
    assert pool.exit_branch_id[0] == 0  # Atmata
    assert pool.tri_idx[0] == 2  # UNCHANGED — no teleport


def test_homing_atomic_on_branch_entry_cell_failure(monkeypatch):
    """C3.3 test 11: inject mock that raises in `_branch_entry_cell`;
    assert no pool field is mutated."""
    from salmon_ibm.origin import ORIGIN_HATCHERY
    from salmon_ibm import delta_routing
    mesh = _BalticTestMesh()
    pool = _make_pool_with_origin(3, origin_value=ORIGIN_HATCHERY, natal_rid=0)
    landscape = _make_baltic_landscape(seed=12345)
    monkeypatch.setattr(
        landscape["hatchery_dispatch"].params, "homing_precision", 0.0,
    )
    call_count = {"n": 0}
    original = delta_routing._branch_entry_cell
    def _failing(mesh_arg, branch_rid):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("injected mid-loop failure")
        return original(mesh_arg, branch_rid)
    monkeypatch.setattr(delta_routing, "_branch_entry_cell", _failing)
    pre_exit = pool.exit_branch_id.copy()
    pre_tri = pool.tri_idx.copy()
    with pytest.raises(RuntimeError, match=r"injected mid-loop"):
        update_exit_branch_id(pool, mesh, landscape=landscape)
    np.testing.assert_array_equal(pool.exit_branch_id, pre_exit)
    np.testing.assert_array_equal(pool.tri_idx, pre_tri)


# --- Task 8: integration tests 10/11b/13/14 ------------------------------

def test_homing_baltic_default_distribution():
    """C3.3 test 10: mixed wild + hatchery cohort with full Baltic
    config; exit_branch_id distribution matches expected mixture
    PER ORIGIN (not just the cohort marginal — would mask a
    precision-swap bug)."""
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY
    mesh = _BalticTestMesh()
    n_per = 10000
    pool = _make_pool_with_origin(2 * n_per, origin_value=ORIGIN_WILD, natal_rid=0)
    pool.origin[n_per:] = ORIGIN_HATCHERY
    landscape = _make_baltic_landscape(seed=12345)
    update_exit_branch_id(pool, mesh, landscape=landscape)

    wild_mask = pool.origin == ORIGIN_WILD
    hatch_mask = pool.origin == ORIGIN_HATCHERY
    wild_counts = np.bincount(pool.exit_branch_id[wild_mask], minlength=3)
    hatch_counts = np.bincount(pool.exit_branch_id[hatch_mask], minlength=3)

    # Wild (p=0.95): expected Atmata=9500, Skirvyte=350, Gilija=150.
    assert 9400 <= wild_counts[0] <= 9600, f"wild Atmata {wild_counts[0]}"
    assert 250 <= wild_counts[1] <= 450, f"wild Skirvyte {wild_counts[1]}"
    assert 90 <= wild_counts[2] <= 210, f"wild Gilija {wild_counts[2]}"

    # Hatchery (p=0.65): expected Atmata=6500, Skirvyte=2444, Gilija=1056.
    assert 6300 <= hatch_counts[0] <= 6700, f"hatch Atmata {hatch_counts[0]}"
    assert 2244 <= hatch_counts[1] <= 2644, f"hatch Skirvyte {hatch_counts[1]}"
    assert 856 <= hatch_counts[2] <= 1256, f"hatch Gilija {hatch_counts[2]}"


def test_homing_cross_pop_hatchery_warning_emitted(caplog):
    """C3.3 test 13: ORIGIN_HATCHERY agent + hatchery_dispatch=None
    + Baltic species_config + natal=Atmata (delta branch — load-
    bearing for entering homing dispatch path, not passive
    fallback). Warning emitted via stdlib logging at logger
    'salmon_ibm.delta_routing'; wild-precision fallback runs to
    completion."""
    from salmon_ibm.origin import ORIGIN_HATCHERY
    caplog.set_level(logging.WARNING, logger="salmon_ibm.delta_routing")
    mesh = _BalticTestMesh()
    pool = _make_pool_with_origin(1, origin_value=ORIGIN_HATCHERY, natal_rid=0)
    cfg = load_baltic_species_config(CONFIG_PATH)
    landscape = {
        "rng": np.random.default_rng(12345),
        "species_config": cfg,
        "hatchery_dispatch": None,  # CROSS-POP MISMATCH
    }
    update_exit_branch_id(pool, mesh, landscape=landscape)

    matching_records = [
        r for r in caplog.records
        if r.name == "salmon_ibm.delta_routing"
        and "homing-hatchery-no-dispatch" in r.getMessage()
    ]
    assert matching_records, (
        f"Expected ERR_HOMING_HATCHERY_NO_DISPATCH warning; got "
        f"records: {[(r.name, r.getMessage()) for r in caplog.records]!r}"
    )
    branch_rids_set = set(int(r) for r in _branch_reach_ids(mesh))
    assert int(pool.exit_branch_id[0]) in branch_rids_set, (
        f"exit_branch_id {pool.exit_branch_id[0]} not in branch_rids_set "
        f"{branch_rids_set} — wild-precision fallback may not have "
        f"run to completion."
    )


def test_homing_empty_target_cohort_is_noop():
    """C3.3 test 14: when no agents are on a delta branch (target
    mask is all False), update_exit_branch_id is a clean no-op:
    no mutation, no exception, even with full Baltic landscape."""
    from salmon_ibm.origin import ORIGIN_HATCHERY
    mesh = _BalticTestMesh()
    pool = _make_pool_with_origin(5, origin_value=ORIGIN_HATCHERY, natal_rid=0)
    pool.tri_idx[:] = -1  # off-mesh: target mask filters out
    landscape = _make_baltic_landscape(seed=12345)
    pre_exit = pool.exit_branch_id.copy()
    pre_tri = pool.tri_idx.copy()
    update_exit_branch_id(pool, mesh, landscape=landscape)
    np.testing.assert_array_equal(pool.exit_branch_id, pre_exit)
    np.testing.assert_array_equal(pool.tri_idx, pre_tri)


def test_homing_dispatch_commit_outside_try_block():
    """C3.3 test 11b: structural atomicity — the vectorised commit
    lines (pool.exit_branch_id[...] = ... ; pool.tri_idx[...] = ...)
    MUST be at the same lexical scope as the for-loop, NOT inside
    any try/except/finally. Tokenize + AST hybrid:
    1. Get source text via inspect.getsource.
    2. Find COMMENT token containing 'C3.3-ATOMIC-COMMIT' via tokenize.
    3. Parse with ast and build parent map.
    4. Find the two Assign nodes after the marker line.
    5. Assert no Try ancestor in their parent chain.
    """
    from salmon_ibm import delta_routing

    source = inspect.getsource(delta_routing.update_exit_branch_id)
    marker_line = None
    for tok in tokenize.tokenize(io.BytesIO(source.encode()).readline):
        if tok.type == tokenize.COMMENT and "C3.3-ATOMIC-COMMIT" in tok.string:
            marker_line = tok.start[0]
            break
    assert marker_line is not None, (
        "C3.3-ATOMIC-COMMIT marker comment missing from "
        "update_exit_branch_id — did a refactor remove it?"
    )

    tree = ast.parse(source)
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent

    post_marker_assigns = sorted(
        [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and n.lineno > marker_line],
        key=lambda n: n.lineno,
    )
    assert len(post_marker_assigns) >= 2, (
        "Expected at least 2 Assign nodes after the marker; got "
        f"{len(post_marker_assigns)}."
    )

    for assign in post_marker_assigns[:2]:
        node = assign
        while hasattr(node, "_parent"):
            node = node._parent
            if isinstance(node, ast.FunctionDef):
                break
            assert not isinstance(node, ast.Try), (
                f"Atomicity violation: Assign at line {assign.lineno} "
                f"is inside a Try block. The C3.3-ATOMIC-COMMIT "
                f"vectorised commit must NOT be wrapped in "
                f"try/except/finally."
            )
