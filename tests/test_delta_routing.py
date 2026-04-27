"""Unit tests for salmon_ibm.delta_routing."""
import numpy as np
import pytest

from salmon_ibm import delta_routing


def test_branch_fractions_sum_to_one():
    total = sum(delta_routing.BRANCH_FRACTIONS.values())
    assert abs(total - 1.0) < 1e-9, f"Fractions sum to {total}, not 1.0"


def test_branch_fractions_keys_are_real_reaches():
    expected = {"Skirvyte", "Atmata", "Gilija"}
    assert set(delta_routing.BRANCH_FRACTIONS) == expected


def test_delta_branch_reaches_is_frozenset():
    assert isinstance(delta_routing.DELTA_BRANCH_REACHES, frozenset)
    assert delta_routing.DELTA_BRANCH_REACHES == set(delta_routing.BRANCH_FRACTIONS)


def test_split_discharge_preserves_total_array():
    q = np.array([100.0, 200.0, 0.0, 500.0], dtype=np.float32)
    out = delta_routing.split_discharge(q)
    summed = sum(out.values())
    assert np.allclose(summed, q, rtol=1e-6)


def test_split_discharge_handles_scalar():
    out = delta_routing.split_discharge(np.float32(1000.0))
    assert pytest.approx(out["Skirvyte"], rel=1e-6) == 510.0
    assert pytest.approx(out["Atmata"],   rel=1e-6) == 270.0
    assert pytest.approx(out["Gilija"],   rel=1e-6) == 220.0


def test_split_discharge_zero_input():
    q = np.zeros(10, dtype=np.float32)
    out = delta_routing.split_discharge(q)
    for name, arr in out.items():
        assert np.all(arr == 0.0), f"{name} should be all-zero, got {arr}"


def test_split_discharge_handles_list_input():
    out = delta_routing.split_discharge([100.0, 200.0])
    import numpy as np
    np.testing.assert_allclose(out["Skirvyte"], [51.0, 102.0], rtol=1e-6)
    np.testing.assert_allclose(out["Atmata"],   [27.0, 54.0],  rtol=1e-6)
    np.testing.assert_allclose(out["Gilija"],   [22.0, 44.0],  rtol=1e-6)


class _FakePool:
    """Minimal pool stand-in for update_exit_branch_id testing."""
    def __init__(self, tri_idx, alive, exit_branch_id):
        self.tri_idx = np.asarray(tri_idx, dtype=np.int64)
        self.alive = np.asarray(alive, dtype=bool)
        self.exit_branch_id = np.asarray(exit_branch_id, dtype=np.int8)


class _FakeMesh:
    def __init__(self, reach_id, reach_names):
        self.reach_id = np.asarray(reach_id, dtype=np.int8)
        self.reach_names = list(reach_names)


REACH_NAMES = ["Nemunas", "Atmata", "Skirvyte", "Gilija", "CuronianLagoon"]
RID = {name: i for i, name in enumerate(REACH_NAMES)}


def _mesh(cell_to_reach):
    """Build a mesh whose cell i sits in reach cell_to_reach[i]."""
    rid = np.array([RID[r] for r in cell_to_reach], dtype=np.int8)
    return _FakeMesh(rid, REACH_NAMES)


def test_update_exit_branch_id_first_touch():
    mesh = _mesh(["Nemunas", "Atmata", "Skirvyte"])
    pool = _FakePool(tri_idx=[1, 1, 1],
                     alive=[True, True, True],
                     exit_branch_id=[-1, -1, -1])
    delta_routing.update_exit_branch_id(pool, mesh)
    assert (pool.exit_branch_id == RID["Atmata"]).all()


def test_update_exit_branch_id_sticky():
    mesh = _mesh(["Nemunas", "Atmata", "Skirvyte"])
    pool = _FakePool(tri_idx=[2, 2],
                     alive=[True, True],
                     exit_branch_id=[RID["Atmata"], -1])
    delta_routing.update_exit_branch_id(pool, mesh)
    assert pool.exit_branch_id[0] == RID["Atmata"]
    assert pool.exit_branch_id[1] == RID["Skirvyte"]


def test_update_exit_branch_id_skips_dead_agents():
    mesh = _mesh(["Nemunas", "Atmata", "Skirvyte"])
    pool = _FakePool(tri_idx=[1, 1],
                     alive=[True, False],
                     exit_branch_id=[-1, -1])
    delta_routing.update_exit_branch_id(pool, mesh)
    assert pool.exit_branch_id[0] == RID["Atmata"]
    assert pool.exit_branch_id[1] == -1, "dead agents must not be tagged"


def test_update_exit_branch_id_skips_lagoon_only():
    mesh = _mesh(["Nemunas", "Atmata", "Skirvyte", "Gilija", "CuronianLagoon"])
    pool = _FakePool(tri_idx=[4, 0],
                     alive=[True, True],
                     exit_branch_id=[-1, -1])
    delta_routing.update_exit_branch_id(pool, mesh)
    assert (pool.exit_branch_id == -1).all()


def test_update_exit_branch_id_no_op_without_reach_meta():
    class _NoMesh:
        pass
    pool = _FakePool(tri_idx=[0], alive=[True], exit_branch_id=[-1])
    delta_routing.update_exit_branch_id(pool, _NoMesh())
    assert pool.exit_branch_id[0] == -1


def test_update_exit_branch_id_handles_negative_tri_idx():
    mesh = _mesh(["Nemunas", "Atmata", "Skirvyte"])
    pool = _FakePool(tri_idx=[-1, 1],
                     alive=[True, True],
                     exit_branch_id=[-1, -1])
    delta_routing.update_exit_branch_id(pool, mesh)
    assert pool.exit_branch_id[0] == -1
    assert pool.exit_branch_id[1] == RID["Atmata"]
