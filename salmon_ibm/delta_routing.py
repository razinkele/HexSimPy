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


def update_exit_branch_id(pool, mesh) -> None:
    """First-touch sticky tagging of pool.exit_branch_id by delta branch.

    Mutates pool.exit_branch_id in place. For each alive agent currently in
    a delta-branch reach (Atmata/Skirvyte/Gilija) whose exit_branch_id is
    still -1, sets it to the current reach_id. Once written, never resets
    — first-touch is the science contract.

    No-op when the mesh has no reach_names (TriMesh / HexMesh fallbacks).
    """
    if not getattr(mesh, "reach_names", None):
        return
    branch_rids = _branch_reach_ids(mesh)
    if len(branch_rids) == 0:
        return
    tri = pool.tri_idx
    safe_tri = np.where(tri >= 0, tri, 0)
    cur_reach = mesh.reach_id[safe_tri]
    is_branch = np.isin(cur_reach, branch_rids)
    on_mesh = tri >= 0
    untagged = pool.exit_branch_id == -1
    target = is_branch & untagged & pool.alive & on_mesh
    if target.any():
        pool.exit_branch_id[target] = cur_reach[target]
