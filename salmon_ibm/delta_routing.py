"""Nemunas delta branch identity, discharge fractions, and exit-tracking utility.

Three branches: Atmata, Skirvyte, Gilija. Pakalne is intentionally lumped
into Atmata (small distributary, absent from the inSTREAM source). Rusne
is the island between branches, not a flowing channel.

Fractions follow Ramsar Site 629 Information Sheet (Nemunas Delta), 2010
https://rsis.ramsar.org/RISapp/files/41231939/documents/LT629_lit161122.pdf
— interior-of-range midpoints summing to 1.0. They are geographic constants
of the delta, not scenario-tunable parameters.

Tagging contract: every code path that calls Population.add_agents must
follow up with population.set_natal_reach_from_cells(new_idx, mesh) — except
the inter-population transfer case in network.py, which preserves
natal_reach_id from the source agent. Population.assert_natal_tagged()
enforces this at runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from salmon_ibm.baltic_params import BalticBioParams
from salmon_ibm.origin import ORIGIN_HATCHERY

BRANCH_FRACTIONS: dict[str, float] = {
    "Skirvyte": 0.51,   # main flow to Kaliningrad / SW lagoon
    "Atmata":   0.27,   # NE distributary; Pakalne lumped in
    "Gilija":   0.22,   # easternmost; branches off upstream of Rusne island
}
_total = sum(BRANCH_FRACTIONS.values())
if abs(_total - 1.0) >= 1e-9:
    raise ValueError(
        f"BRANCH_FRACTIONS must sum to 1.0, got {_total}"
    )
del _total

DELTA_BRANCH_REACHES: frozenset[str] = frozenset(BRANCH_FRACTIONS)

ERR_HOMING_BRANCH_ENTRY_MISSING = "homing-branch-entry-missing"
ERR_HOMING_HATCHERY_NO_DISPATCH = "homing-hatchery-no-dispatch"


@dataclass
class _BranchEntryCache:
    """Cache of branch_rid → entry_cell, keyed on the identity of
    the stored mesh.reach_id array.

    Why a class (not two correlated attrs on the mesh): the cache
    invariant ("stored_reach_id is mesh.reach_id") is invisible if
    split across two raw attributes. A future caller introducing
    dynamic mesh edits MUST be able to invalidate the cache via a
    named method, not by setting two sentinel attrs in the right
    order.

    Identity comparison (NOT `id()` int): CPython reuses id() values
    after GC, so `id()`-keyed caches can silently return stale
    entries when mesh.reach_id is reassigned. `stored is not new`
    is sound under id-recycling.

    NOT detected: in-place mutation (`mesh.reach_id[k] = v`) leaves
    array identity intact. Future dynamic-mesh tiers MUST call
    `mesh._branch_entry_cache.invalidate()` after in-place mutations.
    """
    stored_reach_id: np.ndarray | None = None
    cells: dict[int, int] = field(default_factory=dict)

    def get(self, mesh, branch_rid: int) -> int:
        """Return the cached entry cell for branch_rid, rebuilding
        on identity mismatch."""
        if self.stored_reach_id is not mesh.reach_id:
            self.stored_reach_id = mesh.reach_id
            self.cells = {}
        if branch_rid not in self.cells:
            matches = np.where(mesh.reach_id == branch_rid)[0]
            self.cells[branch_rid] = (
                int(matches.min()) if len(matches) > 0 else -1
            )
        return self.cells[branch_rid]

    def invalidate(self) -> None:
        """Force full rebuild on next get() call. Required after
        in-place mutation of mesh.reach_id (no production path
        currently does this; documented for future dynamic-mesh
        tiers)."""
        self.stored_reach_id = None
        self.cells = {}


def _branch_entry_cell(mesh, branch_rid: int) -> int:
    """Return a deterministic cell of branch_rid for teleport target.

    Returns the lowest-index cell with reach_id == branch_rid. This
    is a layout artifact, NOT the seaward-most cell. Acceptable
    because UPSTREAM movement disperses the post-teleport cluster
    within ~5 steps. Hydrologically-correct entry-cell selection is
    deferred to a future tier.

    Returns -1 if no cells match. By contract, this should never be
    reachable at runtime: the init-time invariant
    `assert_branch_topology` checks all branch_rids have cells
    BEFORE any agent moves.
    """
    cache = getattr(mesh, "_branch_entry_cache", None)
    if cache is None:
        cache = _BranchEntryCache()
        mesh._branch_entry_cache = cache
    return cache.get(mesh, branch_rid)


def branch_stray_weight(branch_rid: int, mesh) -> float:
    """Look up BRANCH_FRACTIONS by branch_rid, NOT by name.

    BRANCH_FRACTIONS is name-keyed (Skirvyte/Atmata/Gilija). This
    wraps the int→str→float chain so dispatch call sites are
    int-keyed only. Future tributary-tier topology changes that
    add/remove branches localise the breakage to this helper.
    """
    return BRANCH_FRACTIONS[mesh.reach_names[branch_rid]]


def _first_touch_passive_tag(
    pool, indices: np.ndarray, cur_reach: np.ndarray
) -> None:
    """Vectorised passive first-touch tag (legacy pre-C3.3 behavior).

    Sets pool.exit_branch_id[indices] = cur_reach[indices]. Used as
    the fall-through path for non-Baltic / non-delta-natal /
    landscape-missing scenarios. Extracted so future tiers can reuse
    the "passive observer" semantics without re-deriving them.
    """
    pool.exit_branch_id[indices] = cur_reach[indices]


def assert_branch_topology(mesh) -> None:
    """Init-time invariant: every delta-branch reach has cells AND
    a BRANCH_FRACTIONS entry. Raises ValueError on violation.

    Eliminates two runtime corruption modes:
    1. _branch_entry_cell returning -1 mid-step (would desync
       exit_branch_id and tri_idx if the runtime tried to recover).
    2. branch_stray_weight raising KeyError mid-loop (would abort
       the entire batch's stage-then-commit).

    Both modes are caught at config load with actionable messages
    naming the missing branch. O(N_branches) cost, runs once.

    Early-return discipline (single check, no dead code):
    - reach_names absent (`getattr(...) is None`) → NO-OP (true
      legacy mesh; runtime path also no-ops via the same check).
    - reach_names present but `_branch_reach_ids(mesh)` empty →
      NO-OP (mesh has reach_names for non-Baltic-relevant reaches).
    """
    has_reach_names = getattr(mesh, "reach_names", None) is not None
    if not has_reach_names:
        return  # legacy mesh — no Baltic dispatch ever fires
    branch_rids = _branch_reach_ids(mesh)
    if len(branch_rids) == 0:
        return  # legacy / non-Baltic mesh with reach_names
    # If mesh has ANY BRANCH_FRACTIONS key in reach_names, ALL must be present
    # — partial delta config means some adults can't route their natal branch.
    missing_in_reach_names = [
        br for br in BRANCH_FRACTIONS if br not in mesh.reach_names
    ]
    missing_cells: list[tuple[int, str]] = []
    for rid in branch_rids:
        rid_int = int(rid)
        if _branch_entry_cell(mesh, rid_int) < 0:
            missing_cells.append((rid_int, mesh.reach_names[rid_int]))
    if missing_cells or missing_in_reach_names:
        raise ValueError(
            f"Delta branch topology invariant violated. "
            f"Branches with no cells on mesh: {missing_cells!r}. "
            f"Branches in BRANCH_FRACTIONS but missing from "
            f"mesh.reach_names: {missing_in_reach_names!r}. Both are "
            f"checked at simulation init to fail-fast rather than "
            f"corrupt state mid-step."
        )


def split_discharge(q_total: float | np.ndarray) -> dict[str, np.ndarray]:
    """Apply BRANCH_FRACTIONS to a Nemunas climatology (scalar or (T,) array).

    Returns a dict mapping branch name to the per-branch series, preserving
    insertion order of BRANCH_FRACTIONS. Lists are coerced via np.asarray
    so callers don't have to.
    """
    q = np.asarray(q_total)
    return {br: q * f for br, f in BRANCH_FRACTIONS.items()}


def _branch_reach_ids(mesh) -> np.ndarray:
    """Resolve BRANCH_FRACTIONS keys to integer reach_ids on this mesh.

    Returns an empty array if the mesh has no reach_names. Caches on the
    mesh as `_delta_branch_reach_ids` so repeated step calls don't re-scan.
    """
    # Cache assumes mesh.reach_names is immutable after the first call.
    # The Simulation.__init__ flow sets reach_names ONCE before any event
    # fires, so this is safe today. If a future caller mutates reach_names
    # post-init, invalidate by setting `mesh._delta_branch_reach_ids = None`.
    cached = getattr(mesh, "_delta_branch_reach_ids", None)
    if cached is not None:
        return cached
    if not getattr(mesh, "reach_names", None):
        return np.empty(0, dtype=np.int8)
    rids = np.array(
        [mesh.reach_names.index(br) for br in BRANCH_FRACTIONS
         if br in mesh.reach_names],
        dtype=np.int8,
    )
    try:
        mesh._delta_branch_reach_ids = rids
    except AttributeError:
        pass
    return rids


def update_exit_branch_id(pool, mesh, *, landscape=None) -> None:
    """First-touch tagging of pool.exit_branch_id by delta branch.

    Two paths:

    (A) PASSIVE (legacy, pre-C3.3 behavior). Triggered when
        landscape is None OR species_config is missing/non-Baltic
        OR agent's natal_reach_id is not a delta branch.

    (B) HOMING-BIASED (C3.3). Triggered when landscape has a
        Baltic species_config AND the agent's natal_reach_id IS a
        delta branch. Draws the chosen branch via origin-aware
        homing-precision; teleports on stray. Atomicity:
        stage-then-commit (no mid-loop pool mutation).

    Spec: docs/superpowers/specs/2026-05-03-hatchery-c3.3-homing-design.md
    """
    if not getattr(mesh, "reach_names", None):
        return  # legacy mesh fallback (TriMesh / HexMesh)
    branch_rids = _branch_reach_ids(mesh)
    if len(branch_rids) == 0:
        return

    # Prelude (mirrors pre-C3.3 logic).
    tri = pool.tri_idx
    safe_tri = np.where(tri >= 0, tri, 0)
    cur_reach = mesh.reach_id[safe_tri]
    is_branch = np.isin(cur_reach, branch_rids)
    on_mesh = tri >= 0
    untagged = pool.exit_branch_id == -1
    target = is_branch & untagged & pool.alive & on_mesh
    if not target.any():
        return
    target_indices = np.where(target)[0]

    # Path (A): no Baltic species_config OR no landscape — passive tag.
    species_cfg = landscape.get("species_config") if landscape else None
    is_baltic = (
        species_cfg is not None
        and isinstance(species_cfg.wild, BalticBioParams)
    )
    if not is_baltic:
        _first_touch_passive_tag(pool, target_indices, cur_reach)
        return

    # Path (B): Baltic — origin-aware homing draw with atomicity.
    hd = landscape.get("hatchery_dispatch")
    if "rng" not in landscape:
        raise ValueError(
            "Baltic homing dispatch requires landscape['rng']; got "
            f"keys: {sorted(landscape.keys())!r}. Calibration runs "
            "rely on a seeded RNG; a default rng would break "
            "reproducibility silently."
        )
    rng = landscape["rng"]
    branch_rids_set = set(int(r) for r in branch_rids)
    n_targets = len(target_indices)
    staged_rid = np.full(n_targets, -1, dtype=np.int8)
    staged_cell = np.full(n_targets, -1, dtype=np.intp)
    is_homing_decision = np.zeros(n_targets, dtype=bool)
    passive_indices: list[int] = []
    # Emit cross-pop warning at most once per dispatch call: 1000 hatchery
    # agents in one batch should produce 1 log line, not 1000. The first
    # offending agent's index is recorded so the operator can grep state.
    cross_pop_warned = False

    for k, i in enumerate(target_indices):
        natal_rid = int(pool.natal_reach_id[i])
        if natal_rid not in branch_rids_set:
            passive_indices.append(i)
            continue

        if pool.origin[i] == ORIGIN_HATCHERY:
            if hd is None:
                if not cross_pop_warned:
                    logging.getLogger(__name__).warning(
                        "%s: ORIGIN_HATCHERY agent %d (first of batch) at "
                        "homing time but hatchery_dispatch is None — "
                        "falling back to wild homing precision (Vasemägi "
                        "2005 wild baseline). Likely SwitchPopulationEvent "
                        "crossed populations with mismatched hatchery "
                        "configs. See spec out-of-scope: 'Cross-population "
                        "transfer alignment'.",
                        ERR_HOMING_HATCHERY_NO_DISPATCH,
                        int(i),
                    )
                    cross_pop_warned = True
                p_home = species_cfg.wild.homing_precision
            else:
                p_home = hd.params.homing_precision
        else:
            p_home = species_cfg.wild.homing_precision

        non_natal = [b for b in branch_rids_set if b != natal_rid]
        stray_w = np.array(
            [branch_stray_weight(b, mesh) for b in non_natal],
            dtype=np.float64,
        )
        stray_w /= stray_w.sum()
        all_branches = np.array([natal_rid] + non_natal, dtype=np.int8)
        all_probs = np.concatenate([[p_home], (1 - p_home) * stray_w])

        drawn_rid = int(rng.choice(all_branches, p=all_probs))
        staged_rid[k] = drawn_rid
        is_homing_decision[k] = True

        if drawn_rid != int(cur_reach[i]):
            entry_cell = _branch_entry_cell(mesh, drawn_rid)
            if entry_cell < 0:
                raise RuntimeError(
                    f"_branch_entry_cell returned -1 for branch_rid "
                    f"{drawn_rid} during homing dispatch. This should "
                    f"have been caught at simulation init by "
                    f"assert_branch_topology(). Likely cause: mesh "
                    f"mutated post-init."
                )
            staged_cell[k] = entry_cell

    # C3.3-ATOMIC-COMMIT — vectorised commit (atomic per-agent
    # across the batch). MUST stay at the same lexical scope as the
    # for-loop above; MUST NOT be inside try/except/finally. This
    # marker comment is referenced by test 11b's AST anchor — do
    # not remove or rename without updating the test.
    homing_idx = target_indices[is_homing_decision]
    pool.exit_branch_id[homing_idx] = staged_rid[is_homing_decision]
    teleport_mask = staged_cell >= 0
    pool.tri_idx[target_indices[teleport_mask]] = staged_cell[teleport_mask]
    if passive_indices:
        passive_arr = np.array(passive_indices, dtype=np.intp)
        _first_touch_passive_tag(pool, passive_arr, cur_reach)
