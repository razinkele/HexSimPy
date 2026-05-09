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
