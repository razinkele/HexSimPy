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

import numpy as np

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
