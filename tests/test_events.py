"""Unit tests for the event engine."""
import numpy as np
import pytest

from salmon_ibm.events import (
    EveryStep, Once, Periodic, Window, RandomTrigger,
    Event, EventSequencer, EventGroup,
)


class TestEveryStep:
    def test_always_fires(self):
        trigger = EveryStep()
        for t in range(20):
            assert trigger.should_fire(t) is True


class TestOnce:
    def test_fires_only_at_target(self):
        trigger = Once(at=5)
        assert trigger.should_fire(4) is False
        assert trigger.should_fire(5) is True
        assert trigger.should_fire(6) is False


class TestPeriodic:
    def test_fires_every_n_steps(self):
        trigger = Periodic(interval=3, offset=0)
        results = [trigger.should_fire(t) for t in range(10)]
        assert results == [True, False, False, True, False, False, True, False, False, True]

    def test_offset(self):
        trigger = Periodic(interval=4, offset=2)
        assert trigger.should_fire(0) is False
        assert trigger.should_fire(1) is False
        assert trigger.should_fire(2) is True
        assert trigger.should_fire(6) is True


class TestWindow:
    def test_fires_within_range(self):
        trigger = Window(start=3, end=6)
        results = [trigger.should_fire(t) for t in range(8)]
        assert results == [False, False, False, True, True, True, False, False]


class TestRandomTrigger:
    def test_probability_zero_never_fires(self):
        trigger = RandomTrigger(p=0.0)
        assert all(not trigger.should_fire(t) for t in range(100))

    def test_probability_one_always_fires(self):
        trigger = RandomTrigger(p=1.0)
        assert all(trigger.should_fire(t) for t in range(100))

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        t1 = RandomTrigger(p=0.5, _rng=rng1)
        t2 = RandomTrigger(p=0.5, _rng=rng2)
        results1 = [t1.should_fire(t) for t in range(50)]
        results2 = [t2.should_fire(t) for t in range(50)]
        assert results1 == results2


# ---------------------------------------------------------------------------
# Helpers for sequencer/group tests
# ---------------------------------------------------------------------------

class StubEvent(Event):
    def __init__(self, name, trigger=None):
        super().__init__(name=name, trigger=trigger or EveryStep())
        self.calls = []

    def execute(self, population, landscape, t, mask):
        self.calls.append({"t": t, "mask_sum": int(mask.sum())})


class FakePopulation:
    def __init__(self, n=10):
        self.n = n
        self.alive = np.ones(n, dtype=bool)
        self.arrived = np.zeros(n, dtype=bool)


class TestEventSequencer:
    def test_runs_events_in_order(self):
        call_order = []
        e1 = StubEvent("first")
        e2 = StubEvent("second")
        orig1, orig2 = e1.execute, e2.execute
        def track1(*a, **kw):
            call_order.append("first")
            orig1(*a, **kw)
        def track2(*a, **kw):
            call_order.append("second")
            orig2(*a, **kw)
        e1.execute = track1
        e2.execute = track2
        seq = EventSequencer([e1, e2])
        seq.step(FakePopulation(), {}, t=0)
        assert call_order == ["first", "second"]

    def test_respects_trigger(self):
        e_every = StubEvent("every", EveryStep())
        e_once = StubEvent("once", Once(at=2))
        seq = EventSequencer([e_every, e_once])
        pop = FakePopulation()
        for t in range(5):
            seq.step(pop, {}, t)
        assert len(e_every.calls) == 5
        assert len(e_once.calls) == 1
        assert e_once.calls[0]["t"] == 2

    def test_mask_excludes_dead_and_arrived(self):
        e = StubEvent("check")
        pop = FakePopulation(n=10)
        pop.alive[0:3] = False
        pop.arrived[7:10] = True
        seq = EventSequencer([e])
        seq.step(pop, {}, t=0)
        assert e.calls[0]["mask_sum"] == 4


class TestEventGroup:
    def test_runs_sub_events(self):
        sub1 = StubEvent("sub1")
        sub2 = StubEvent("sub2")
        group = EventGroup(name="group", sub_events=[sub1, sub2])
        seq = EventSequencer([group])
        seq.step(FakePopulation(), {}, t=0)
        assert len(sub1.calls) == 1
        assert len(sub2.calls) == 1

    def test_iterations(self):
        sub = StubEvent("sub")
        group = EventGroup(name="group", sub_events=[sub], iterations=3)
        seq = EventSequencer([group])
        seq.step(FakePopulation(), {}, t=0)
        assert len(sub.calls) == 3

    def test_nested_trigger_respected(self):
        sub_every = StubEvent("every", EveryStep())
        sub_once = StubEvent("once", Once(at=5))
        group = EventGroup(name="group", sub_events=[sub_every, sub_once])
        seq = EventSequencer([group])
        seq.step(FakePopulation(), {}, t=0)
        assert len(sub_every.calls) == 1
        assert len(sub_once.calls) == 0

    def test_group_trigger_gates_sub_events(self):
        sub = StubEvent("sub")
        group = EventGroup(name="group", trigger=Once(at=3), sub_events=[sub])
        seq = EventSequencer([group])
        for t in range(5):
            seq.step(FakePopulation(), {}, t)
        assert len(sub.calls) == 1
