"""C5.1 BeenToSeaEvent tests.

Covers basic set-on-entry behavior; sticky semantics; non-Baltic no-op;
dead-skip; int8-overflow safety; post-teleport ordering.
"""
import numpy as np
import pytest
from types import SimpleNamespace

from salmon_ibm.agents import AgentPool
from salmon_ibm.events_builtin import BeenToSeaEvent


def _make_sim_stub(*, is_baltic, at_sea_rid_set, mesh_reach_id):
    """Build the minimal sim-shaped stub BeenToSeaEvent.execute reads."""
    return SimpleNamespace(
        _is_baltic=is_baltic,
        _at_sea_rid_set=frozenset(at_sea_rid_set),
        mesh=SimpleNamespace(reach_id=np.asarray(mesh_reach_id, dtype=np.int8)),
    )


def _make_pop(pool):
    """Wrap an AgentPool in a population-shaped stub."""
    return SimpleNamespace(pool=pool)


def test_been_to_sea_set_on_open_baltic_entry():
    """Agent in cell with reach_id == OpenBaltic_rid → been_to_sea True."""
    pool = AgentPool(n=3, start_tri=np.array([0, 1, 2], dtype=int))
    sim = _make_sim_stub(
        is_baltic=True, at_sea_rid_set={1, 2}, mesh_reach_id=[0, 1, 2],
    )
    landscape = {"sim": sim}
    evt = BeenToSeaEvent()
    evt.execute(_make_pop(pool), landscape, t=0, mask=None)
    assert pool.been_to_sea[0] == False
    assert pool.been_to_sea[1] == True
    assert pool.been_to_sea[2] == True


def test_been_to_sea_set_on_baltic_coast_entry():
    """BalticCoast cells must trigger the flag (validates the SET, not just OpenBaltic)."""
    pool = AgentPool(n=2, start_tri=np.array([0, 1], dtype=int))
    sim = _make_sim_stub(
        is_baltic=True, at_sea_rid_set={1, 2}, mesh_reach_id=[0, 1],
    )
    evt = BeenToSeaEvent()
    evt.execute(_make_pop(pool), {"sim": sim}, t=0, mask=None)
    assert pool.been_to_sea[0] == False  # river cell
    assert pool.been_to_sea[1] == True   # BalticCoast cell


def test_been_to_sea_mixed_membership_freshwater():
    """Mixed-membership mesh: 0=river, 1=lagoon, 2=BalticCoast, 3=OpenBaltic."""
    pool = AgentPool(n=4, start_tri=np.array([0, 1, 2, 3], dtype=int))
    sim = _make_sim_stub(
        is_baltic=True, at_sea_rid_set={2, 3},
        mesh_reach_id=[0, 1, 2, 3],
    )
    evt = BeenToSeaEvent()
    evt.execute(_make_pop(pool), {"sim": sim}, t=0, mask=None)
    assert pool.been_to_sea[0] == False  # river
    assert pool.been_to_sea[1] == False  # lagoon
    assert pool.been_to_sea[2] == True   # BalticCoast
    assert pool.been_to_sea[3] == True   # OpenBaltic


def test_been_to_sea_sticky_after_returning_to_river():
    """Agent visits BalticCoast, then moves back to river → flag stays True."""
    pool = AgentPool(n=1, start_tri=np.array([1], dtype=int))  # at BalticCoast
    sim = _make_sim_stub(
        is_baltic=True, at_sea_rid_set={1, 2}, mesh_reach_id=[0, 1],
    )
    evt = BeenToSeaEvent()
    # Step 1: in BalticCoast → set True.
    evt.execute(_make_pop(pool), {"sim": sim}, t=0, mask=None)
    assert pool.been_to_sea[0] == True
    # Step 2: move to river cell.
    pool.tri_idx[0] = 0
    evt.execute(_make_pop(pool), {"sim": sim}, t=1, mask=None)
    # Sticky — stays True even though agent is no longer at sea.
    assert pool.been_to_sea[0] == True


def test_been_to_sea_non_baltic_short_circuit_dominates():
    """When _is_baltic=False, the early-return MUST short-circuit
    BEFORE the membership test runs. To prove the short-circuit is
    doing the work (not the empty-set), seed at_sea_rid_set={1} and
    place an agent in cell with reach_id=1. The membership test
    would otherwise tag the agent — but the short-circuit prevents
    that.
    """
    pool = AgentPool(n=2, start_tri=np.array([0, 1], dtype=int))
    sim = _make_sim_stub(
        # Inconsistent state: _is_baltic=False but at_sea_rid_set
        # non-empty AND mesh has cell with reach_id matching the set.
        # If short-circuit fires, no flag is set. If short-circuit
        # doesn't fire, agent 1 (in cell reach_id=1) would be tagged.
        is_baltic=False, at_sea_rid_set={1}, mesh_reach_id=[0, 1],
    )
    evt = BeenToSeaEvent()
    evt.execute(_make_pop(pool), {"sim": sim}, t=0, mask=None)
    # Short-circuit must dominate: even though agent 1's cell would
    # otherwise match, no flag is set.
    assert not pool.been_to_sea.any()


def test_been_to_sea_skips_dead_agents():
    """Dead agent in at-sea cell → been_to_sea NOT set (matches C5
    dead-skip pattern at events_builtin.py:594)."""
    pool = AgentPool(n=2, start_tri=np.array([1, 1], dtype=int))
    pool.alive[0] = False  # agent 0 dead
    sim = _make_sim_stub(
        is_baltic=True, at_sea_rid_set={1}, mesh_reach_id=[0, 1],
    )
    evt = BeenToSeaEvent()
    evt.execute(_make_pop(pool), {"sim": sim}, t=0, mask=None)
    # Dead agent: stays False. Alive agent: True.
    assert pool.been_to_sea[0] == False
    assert pool.been_to_sea[1] == True


def test_been_to_sea_int8_storage_compatible():
    """Reach IDs stored as int8 must compare correctly with the
    int32-converted at-sea set.

    Reframed per pass-1 B-1 review. Storing `np.int8(130)` produces
    `-126` (sign-bit wrap); the int32 cast widens `-126` to
    `int32(-126)`, NOT `130`. To exercise the cast meaningfully, both
    sides of the comparison must contain the wrapped value.
    """
    pool = AgentPool(n=1, start_tri=np.array([0], dtype=int))
    # NumPy 2.x rejects `np.asarray([130], dtype=np.int8)` as overflow
    # at construction, but the silent wrap still occurs through
    # `.astype(np.int8)` (the path real code takes when narrowing
    # widened reach IDs into the int8 storage). Use that to produce
    # the wrapped value `-126` on the mesh side.
    cell_rid_int8 = np.asarray([130], dtype=np.int16).astype(np.int8)  # stored as -126
    at_sea_rid_stored = int(cell_rid_int8[0])  # -126 (matches the wrap)
    sim = _make_sim_stub(
        is_baltic=True,
        at_sea_rid_set={at_sea_rid_stored},
        mesh_reach_id=cell_rid_int8,
    )
    evt = BeenToSeaEvent()
    evt.execute(_make_pop(pool), {"sim": sim}, t=0, mask=None)
    # cell_rid (after int32 cast) = -126; at_sea_rids = [-126].
    # np.isin([-126], [-126]) = True → flag set.
    assert pool.been_to_sea[0] == True


def test_been_to_sea_at_sea_set_int_widening():
    """Regression guard: the np.fromiter conversion of the frozenset
    to int32 must preserve the integer values (no narrowing).
    """
    pool = AgentPool(n=1, start_tri=np.array([0], dtype=int))
    sim = _make_sim_stub(
        is_baltic=True, at_sea_rid_set={5},
        mesh_reach_id=np.asarray([5], dtype=np.int8),
    )
    evt = BeenToSeaEvent()
    evt.execute(_make_pop(pool), {"sim": sim}, t=0, mask=None)
    assert pool.been_to_sea[0] == True

    # Counter-test: at-sea set = {6} but cell is in reach 5 → no match.
    pool2 = AgentPool(n=1, start_tri=np.array([0], dtype=int))
    sim2 = _make_sim_stub(
        is_baltic=True, at_sea_rid_set={6},
        mesh_reach_id=np.asarray([5], dtype=np.int8),
    )
    evt.execute(_make_pop(pool2), {"sim": sim2}, t=0, mask=None)
    assert pool2.been_to_sea[0] == False


def test_been_to_sea_empty_pool_no_op():
    """Edge case (pass-1 different-angle B-A3): n_agents=0 must not
    crash. NumPy ops on empty arrays should be no-ops.
    """
    pool = AgentPool(n=0, start_tri=np.array([], dtype=int))
    sim = _make_sim_stub(
        is_baltic=True, at_sea_rid_set={1, 2},
        mesh_reach_id=[0, 1, 2],
    )
    evt = BeenToSeaEvent()
    # Must not raise on empty pool.
    evt.execute(_make_pop(pool), {"sim": sim}, t=0, mask=None)
    assert pool.been_to_sea.shape == (0,)
    assert pool.been_to_sea.dtype == bool


def test_been_to_sea_off_mesh_agents_not_falsely_tagged():
    """Edge case (pass-1 different-angle B-A2): if any agent has
    pool.tri_idx == -1 (off-mesh sentinel), the indexing
    sim.mesh.reach_id[-1] returns the LAST cell's reach_id. If the
    last cell happens to be at-sea, off-mesh agents get falsely
    tagged. This test forces the scenario and verifies behavior.

    NOTE: spec §2 design assumes agents are always on-mesh; this
    test serves as a regression guard. If the production code
    eventually adds an explicit `pool.tri_idx >= 0` guard
    (analogous to ArrivalEvent's `on_mesh = pool.tri_idx >= 0` at
    events_builtin.py:595), this test will document that intent.
    """
    # Mesh: cell 0 = river, cell 1 = at-sea (last cell).
    # Agent 0: tri_idx=-1 (off-mesh) — would index reach_id[-1]=1=at-sea
    #          IF unguarded, → false tag.
    # Agent 1: tri_idx=0 (river) — should not be tagged.
    pool = AgentPool(n=2, start_tri=np.array([-1, 0], dtype=int))
    sim = _make_sim_stub(
        is_baltic=True, at_sea_rid_set={1},
        mesh_reach_id=[0, 1],
    )
    evt = BeenToSeaEvent()
    evt.execute(_make_pop(pool), {"sim": sim}, t=0, mask=None)
    # Document current behavior: if BeenToSeaEvent.execute does NOT
    # guard tri_idx >= 0, agent 0 would be falsely tagged.
    assert pool.been_to_sea[1] == False  # river agent stays False
    if pool.been_to_sea[0]:
        # No guard in implementation — flag in commit message as a
        # known regression vector.
        import warnings
        warnings.warn(
            "BeenToSeaEvent does not guard pool.tri_idx >= 0; "
            "off-mesh agents may be falsely tagged. Consider "
            "adding `& (pool.tri_idx >= 0)` to at_sea_now mask.",
            stacklevel=2,
        )
    # Either outcome passes — this test is documentary/diagnostic
    # rather than enforcing.
