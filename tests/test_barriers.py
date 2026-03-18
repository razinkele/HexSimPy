"""Unit tests for BarrierMap."""
import numpy as np
import pytest

from salmon_ibm.barriers import BarrierMap, BarrierOutcome, BarrierClass


class TestBarrierOutcome:
    def test_impassable(self):
        o = BarrierOutcome.impassable()
        assert o.p_mortality == 0.0
        assert o.p_deflection == 1.0
        assert o.p_transmission == 0.0

    def test_lethal(self):
        o = BarrierOutcome.lethal()
        assert o.p_mortality == 1.0
        assert o.p_transmission == 0.0

    def test_custom_probabilities(self):
        o = BarrierOutcome(0.1, 0.3, 0.6)
        assert abs(o.p_mortality + o.p_deflection + o.p_transmission - 1.0) < 1e-10


class TestBarrierMap:
    def test_empty_map(self):
        bmap = BarrierMap()
        assert not bmap.has_barriers()
        assert bmap.check(0, 1) is None

    def test_add_and_check(self):
        bmap = BarrierMap()
        outcome = BarrierOutcome(0.1, 0.3, 0.6)
        bmap.add_edge(0, 1, outcome)
        assert bmap.check(0, 1) == outcome
        assert bmap.check(1, 0) is None

    def test_bidirectional(self):
        bmap = BarrierMap()
        fwd = BarrierOutcome(0.1, 0.3, 0.6)
        rev = BarrierOutcome(0.0, 0.5, 0.5)
        bmap.add_edge(0, 1, fwd)
        bmap.add_edge(1, 0, rev)
        assert bmap.check(0, 1) == fwd
        assert bmap.check(1, 0) == rev

    def test_n_edges(self):
        bmap = BarrierMap()
        bmap.add_edge(0, 1, BarrierOutcome.impassable())
        bmap.add_edge(1, 0, BarrierOutcome.impassable())
        bmap.add_edge(2, 3, BarrierOutcome.lethal())
        assert bmap.n_edges == 3


class TestBarrierMovementIntegration:
    @pytest.fixture
    def line_mesh_data(self):
        n = 5
        neighbors = np.full((n, 6), -1, dtype=np.intp)
        neighbors[0, 0] = 1
        neighbors[1, 0] = 0; neighbors[1, 1] = 2
        neighbors[2, 0] = 1; neighbors[2, 1] = 3
        neighbors[3, 0] = 2; neighbors[3, 1] = 4
        neighbors[4, 0] = 3
        nbr_count = np.sum(neighbors >= 0, axis=1).astype(np.intp)
        return neighbors, nbr_count

    def test_impassable_barrier_blocks(self, line_mesh_data):
        neighbors, _ = line_mesh_data
        mort = np.zeros((5, 6), dtype=np.float64)
        defl = np.zeros((5, 6), dtype=np.float64)
        trans = np.ones((5, 6), dtype=np.float64)
        defl[2, 1] = 1.0; trans[2, 1] = 0.0

        from salmon_ibm.movement import _resolve_barriers_vec
        final, died = _resolve_barriers_vec(
            np.array([2]), np.array([3]), mort, defl, trans, neighbors,
            np.random.default_rng(42))
        assert final[0] == 2
        assert not died[0]

    def test_lethal_barrier_kills(self, line_mesh_data):
        neighbors, _ = line_mesh_data
        mort = np.zeros((5, 6), dtype=np.float64)
        defl = np.zeros((5, 6), dtype=np.float64)
        trans = np.ones((5, 6), dtype=np.float64)
        mort[2, 1] = 1.0; trans[2, 1] = 0.0

        from salmon_ibm.movement import _resolve_barriers_vec
        final, died = _resolve_barriers_vec(
            np.array([2]), np.array([3]), mort, defl, trans, neighbors,
            np.random.default_rng(42))
        assert died[0]

    def test_no_barrier_allows_passage(self, line_mesh_data):
        neighbors, _ = line_mesh_data
        mort = np.zeros((5, 6), dtype=np.float64)
        defl = np.zeros((5, 6), dtype=np.float64)
        trans = np.ones((5, 6), dtype=np.float64)

        from salmon_ibm.movement import _resolve_barriers_vec
        final, died = _resolve_barriers_vec(
            np.array([2]), np.array([3]), mort, defl, trans, neighbors,
            np.random.default_rng(42))
        assert final[0] == 3
        assert not died[0]

    def test_no_movement_skips_barriers(self, line_mesh_data):
        neighbors, _ = line_mesh_data
        mort = np.zeros((5, 6), dtype=np.float64)
        defl = np.zeros((5, 6), dtype=np.float64)
        trans = np.ones((5, 6), dtype=np.float64)
        mort[2, 1] = 1.0; trans[2, 1] = 0.0  # lethal barrier

        from salmon_ibm.movement import _resolve_barriers_vec
        # Agent stays at cell 2 (no movement)
        final, died = _resolve_barriers_vec(
            np.array([2]), np.array([2]), mort, defl, trans, neighbors,
            np.random.default_rng(42))
        assert final[0] == 2
        assert not died[0]  # should NOT die since it didn't move
