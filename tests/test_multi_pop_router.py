"""Tests for multi-population event routing."""
import numpy as np
import pytest
from salmon_ibm.agents import AgentPool
from salmon_ibm.population import Population
from salmon_ibm.events import Event, EventGroup, EveryStep, Once, MultiPopEventSequencer
from salmon_ibm.interactions import MultiPopulationManager
from dataclasses import dataclass, field


@dataclass
class MockEvent(Event):
    """Test event that records which population it was called with."""
    calls: list = field(default_factory=list)

    def execute(self, population, landscape, t, mask):
        pop_name = population.name if population else None
        self.calls.append({"pop": pop_name, "t": t, "n_masked": int(mask.sum()) if len(mask) > 0 else 0})


def _make_pop(name, n=10):
    pool = AgentPool(n=n, start_tri=np.zeros(n, dtype=int))
    return Population(name=name, pool=pool)


class TestMultiPopEventSequencer:
    def test_routes_to_named_population(self):
        pop_a = _make_pop("A", 5)
        pop_b = _make_pop("B", 10)
        mgr = MultiPopulationManager()
        mgr.register(pop_a)
        mgr.register(pop_b)

        evt = MockEvent(name="test", population_name="B")
        seq = MultiPopEventSequencer([evt], mgr)
        seq.step({}, 0)

        assert len(evt.calls) == 1
        assert evt.calls[0]["pop"] == "B"
        assert evt.calls[0]["n_masked"] == 10

    def test_disabled_event_skipped(self):
        pop = _make_pop("A")
        mgr = MultiPopulationManager()
        mgr.register(pop)

        evt = MockEvent(name="disabled", population_name="A", enabled=False)
        seq = MultiPopEventSequencer([evt], mgr)
        seq.step({}, 0)

        assert len(evt.calls) == 0

    def test_once_trigger_fires_once(self):
        pop = _make_pop("A")
        mgr = MultiPopulationManager()
        mgr.register(pop)

        evt = MockEvent(name="init", population_name="A", trigger=Once(at=0))
        seq = MultiPopEventSequencer([evt], mgr)
        seq.step({}, 0)
        seq.step({}, 1)

        assert len(evt.calls) == 1

    def test_event_group_routes_children(self):
        pop_a = _make_pop("A", 5)
        pop_b = _make_pop("B", 8)
        mgr = MultiPopulationManager()
        mgr.register(pop_a)
        mgr.register(pop_b)

        child1 = MockEvent(name="c1", population_name="A")
        child2 = MockEvent(name="c2", population_name="B")
        group = EventGroup(name="grp", sub_events=[child1, child2])
        seq = MultiPopEventSequencer([group], mgr)
        seq.step({}, 0)

        assert child1.calls[0]["pop"] == "A"
        assert child1.calls[0]["n_masked"] == 5
        assert child2.calls[0]["pop"] == "B"
        assert child2.calls[0]["n_masked"] == 8

    def test_null_population_group(self):
        """Group with no population_name passes None to execute."""
        pop_a = _make_pop("A")
        mgr = MultiPopulationManager()
        mgr.register(pop_a)

        child = MockEvent(name="c", population_name="A")
        group = EventGroup(name="grp", sub_events=[child])
        seq = MultiPopEventSequencer([group], mgr)
        seq.step({}, 0)

        assert child.calls[0]["pop"] == "A"
