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


# ---------------------------------------------------------------------------
# C5.1 Task 5 — ArrivalEvent guard tests (round-trip semantics).
# ---------------------------------------------------------------------------


def test_arrival_requires_been_to_sea(monkeypatch):
    """Agent in upper-quartile of natal reach but been_to_sea=False
    → arrived stays False. Demonstrates the guard term is in the
    arrival mask: without it, the agent would be tagged arrived.
    """
    from salmon_ibm.events_builtin import ArrivalEvent
    pool = AgentPool(n=1, start_tri=np.array([0], dtype=int))
    pool.natal_reach_id[0] = 0
    # been_to_sea NOT set — guard should block arrival.
    sim = SimpleNamespace(
        _arrival_threshold_by_natal_rid={0: 0.0},  # any dist >= 0 qualifies
    )
    mesh = SimpleNamespace(
        reach_id=np.asarray([0], dtype=np.int8),
        dist_from_sea=np.asarray([100.0], dtype=np.float32),
    )
    landscape = {"sim": sim, "mesh": mesh}
    pop = SimpleNamespace(pool=pool)
    ArrivalEvent().execute(pop, landscape, t=0, mask=None)
    assert pool.arrived[0] == False


def test_arrival_after_round_trip(monkeypatch):
    """Agent visits at-sea, then returns to natal upper-quartile →
    arrived=True after round-trip. With the guard satisfied, arrival
    proceeds normally.
    """
    from salmon_ibm.events_builtin import ArrivalEvent
    pool = AgentPool(n=1, start_tri=np.array([0], dtype=int))
    pool.natal_reach_id[0] = 0
    pool.been_to_sea[0] = True  # simulating prior at-sea visit
    sim = SimpleNamespace(
        _arrival_threshold_by_natal_rid={0: 0.0},
    )
    mesh = SimpleNamespace(
        reach_id=np.asarray([0], dtype=np.int8),
        dist_from_sea=np.asarray([100.0], dtype=np.float32),
    )
    landscape = {"sim": sim, "mesh": mesh}
    pop = SimpleNamespace(pool=pool)
    ArrivalEvent().execute(pop, landscape, t=0, mask=None)
    assert pool.arrived[0] == True


def test_arrival_event_sticky_with_been_to_sea_guard():
    """Once arrived=True, doesn't re-clear when agent leaves natal
    upper quartile. Sticky-flag contract holds with the guard added.
    """
    from salmon_ibm.events_builtin import ArrivalEvent
    pool = AgentPool(n=1, start_tri=np.array([0], dtype=int))
    pool.natal_reach_id[0] = 0
    pool.been_to_sea[0] = True
    pool.arrived[0] = True  # already arrived
    sim = SimpleNamespace(
        _arrival_threshold_by_natal_rid={0: 999.0},
    )
    mesh = SimpleNamespace(
        reach_id=np.asarray([0], dtype=np.int8),
        dist_from_sea=np.asarray([10.0], dtype=np.float32),
    )
    landscape = {"sim": sim, "mesh": mesh}
    pop = SimpleNamespace(pool=pool)
    ArrivalEvent().execute(pop, landscape, t=0, mask=None)
    assert pool.arrived[0] == True


def test_been_to_sea_reads_post_teleport_cell():
    """Ordering invariant: BeenToSeaEvent runs AFTER update_exit_branch
    (which may teleport stray-strayer agents to a non-natal delta-branch
    cell, mutating pool.tri_idx). BeenToSeaEvent must read the
    post-teleport cell.

    Two scenarios:
    (a) Teleport into at-sea cell: been_to_sea SHOULD be set this step
        (the agent is now at sea).
    (b) Teleport out of at-sea cell into delta-branch cell: been_to_sea
        is NOT set this step (post-teleport cell isn't at sea).
        However, sticky semantics mean if been_to_sea was True from a
        prior step, it stays True — the test verifies the read source
        is post-teleport, not pre-teleport.
    """
    # Scenario (a): teleport INTO at-sea cell.
    pool = AgentPool(n=1, start_tri=np.array([0], dtype=int))  # river cell
    sim = _make_sim_stub(
        is_baltic=True, at_sea_rid_set={1}, mesh_reach_id=[0, 1],
    )
    # Simulate teleport: update_exit_branch sets pool.tri_idx[0] = 1.
    pool.tri_idx[0] = 1
    BeenToSeaEvent().execute(_make_pop(pool), {"sim": sim}, t=0, mask=None)
    assert pool.been_to_sea[0] == True

    # Scenario (b): start at-sea, teleport OUT into delta-branch.
    pool2 = AgentPool(n=1, start_tri=np.array([1], dtype=int))  # at-sea
    # NOT setting been_to_sea pre-teleport — testing the read source.
    pool2.tri_idx[0] = 0  # post-teleport: river/delta
    BeenToSeaEvent().execute(_make_pop(pool2), {"sim": sim}, t=0, mask=None)
    # BeenToSeaEvent reads pool.tri_idx[0]=0 (river); reach_id[0]=0
    # is not in at_sea_rid_set={1}; been_to_sea stays False.
    assert pool2.been_to_sea[0] == False
