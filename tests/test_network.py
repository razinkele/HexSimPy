"""Unit tests for stream network topology."""
import numpy as np
import pytest
from salmon_ibm.network import (
    SegmentDefinition, StreamNetwork, NetworkPosition,
    NetworkMovement, NetworkRange, NetworkRangeManager,
    SwitchPopulationEvent,
)


def make_simple_network():
    """Y-shaped network: segments 1,2 merge into 3.
       1 (100m) --\\
                    3 (200m) --> outlet
       2 (150m) --/
    """
    segs = [
        SegmentDefinition(id=1, length=100.0, upstream_ids=[], downstream_ids=[3]),
        SegmentDefinition(id=2, length=150.0, upstream_ids=[], downstream_ids=[3]),
        SegmentDefinition(id=3, length=200.0, upstream_ids=[1, 2], downstream_ids=[]),
    ]
    return StreamNetwork(segs)


class TestStreamNetwork:
    def test_segment_count(self):
        net = make_simple_network()
        assert net.n_segments == 3

    def test_upstream(self):
        net = make_simple_network()
        assert set(net.upstream(3)) == {1, 2}
        assert net.upstream(1) == []

    def test_downstream(self):
        net = make_simple_network()
        assert net.downstream(1) == [3]
        assert net.downstream(3) == []

    def test_headwater(self):
        net = make_simple_network()
        assert net.is_headwater(1)
        assert net.is_headwater(2)
        assert not net.is_headwater(3)

    def test_outlet(self):
        net = make_simple_network()
        assert net.is_outlet(3)
        assert not net.is_outlet(1)

    def test_all_upstream(self):
        net = make_simple_network()
        assert set(net.all_upstream(3)) == {1, 2}

    def test_all_downstream(self):
        net = make_simple_network()
        assert net.all_downstream(1) == [3]

    def test_segment_length(self):
        net = make_simple_network()
        assert net.segment_length(1) == 100.0
        assert net.segment_length(3) == 200.0


class TestNetworkMovement:
    def test_move_upstream_within_segment(self):
        net = make_simple_network()
        mv = NetworkMovement(net, rng_seed=42)
        pos = [NetworkPosition(3, 150.0)]
        new = mv.move_upstream(pos, np.array([50.0]))
        assert new[0].segment_id == 3
        assert new[0].offset == pytest.approx(100.0)

    def test_move_upstream_crosses_boundary(self):
        net = make_simple_network()
        mv = NetworkMovement(net, rng_seed=42)
        pos = [NetworkPosition(3, 20.0)]
        new = mv.move_upstream(pos, np.array([50.0]))
        # Should cross into segment 1 or 2
        assert new[0].segment_id in (1, 2)

    def test_move_upstream_stops_at_headwater(self):
        net = make_simple_network()
        mv = NetworkMovement(net, rng_seed=42)
        pos = [NetworkPosition(1, 10.0)]
        new = mv.move_upstream(pos, np.array([50.0]))
        assert new[0].segment_id == 1
        assert new[0].offset == 0.0

    def test_move_downstream_within_segment(self):
        net = make_simple_network()
        mv = NetworkMovement(net, rng_seed=42)
        pos = [NetworkPosition(1, 50.0)]
        new = mv.move_downstream(pos, np.array([30.0]))
        assert new[0].segment_id == 1
        assert new[0].offset == pytest.approx(80.0)

    def test_move_downstream_crosses_boundary(self):
        net = make_simple_network()
        mv = NetworkMovement(net, rng_seed=42)
        pos = [NetworkPosition(1, 80.0)]
        new = mv.move_downstream(pos, np.array([50.0]))
        assert new[0].segment_id == 3
        assert new[0].offset == pytest.approx(30.0)

    def test_move_downstream_stops_at_outlet(self):
        net = make_simple_network()
        mv = NetworkMovement(net, rng_seed=42)
        pos = [NetworkPosition(3, 180.0)]
        new = mv.move_downstream(pos, np.array([50.0]))
        assert new[0].segment_id == 3
        assert new[0].offset == pytest.approx(200.0)


class TestNetworkRange:
    def test_single_segment_length(self):
        net = make_simple_network()
        r = NetworkRange(segments=[3], start_offset=50.0, end_offset=150.0)
        assert r.total_length(net) == pytest.approx(100.0)

    def test_multi_segment_length(self):
        net = make_simple_network()
        r = NetworkRange(segments=[1, 3], start_offset=50.0, end_offset=100.0)
        # seg 1: 100 - 50 = 50, seg 3: 100
        assert r.total_length(net) == pytest.approx(150.0)


class TestNetworkRangeManager:
    def test_allocate_and_check(self):
        net = make_simple_network()
        mgr = NetworkRangeManager(net)
        assert mgr.is_available(1)
        assert mgr.allocate(0, 1)
        assert not mgr.is_available(1)
        assert mgr.owner_of(1) == 0

    def test_cannot_double_allocate(self):
        net = make_simple_network()
        mgr = NetworkRangeManager(net)
        mgr.allocate(0, 1)
        assert not mgr.allocate(1, 1)

    def test_release(self):
        net = make_simple_network()
        mgr = NetworkRangeManager(net)
        mgr.allocate(0, 1)
        mgr.release(0)
        assert mgr.is_available(1)


class TestSwitchPopulationEvent:
    def test_transfers_agents(self):
        from salmon_ibm.interactions import MultiPopulationManager

        class SimplePop:
            def __init__(self, n, positions):
                self.n = n
                self.tri_idx = np.array(positions, dtype=int)
                self.alive = np.ones(n, dtype=bool)
                self.arrived = np.zeros(n, dtype=bool)
            def add_agents(self, n, positions, **kw):
                self.n += n
                self.tri_idx = np.concatenate([self.tri_idx, positions])
                self.alive = np.concatenate([self.alive, np.ones(n, dtype=bool)])
                self.arrived = np.concatenate([self.arrived, np.zeros(n, dtype=bool)])
                return np.arange(self.n - n, self.n)

        mgr = MultiPopulationManager()
        source = SimplePop(10, [0]*10)
        target = SimplePop(5, [1]*5)
        mgr.register("source", source)
        mgr.register("target", target)

        event = SwitchPopulationEvent(
            name="switch", source_pop="source", target_pop="target",
            transfer_probability=1.0)
        landscape = {"multi_pop_mgr": mgr, "rng": np.random.default_rng(42)}
        mask = source.alive.copy()
        event.execute(source, landscape, t=0, mask=mask)

        assert source.alive.sum() == 0  # all transferred out
        assert target.n == 15  # 5 original + 10 transferred
