"""Unit tests for Phase 3 events."""
import numpy as np
import pytest


class MockPopulation:
    def __init__(self, n):
        self.n = n
        self.alive = np.ones(n, dtype=bool)
        self.arrived = np.zeros(n, dtype=bool)
        self.tri_idx = np.zeros(n, dtype=int)


class TestGeneratedHexmapEvent:
    def test_simple_expression(self):
        from salmon_ibm.events_phase3 import GeneratedHexmapEvent
        pop = MockPopulation(5)
        landscape = {"temperature": np.array([10.0, 15.0, 20.0, 25.0]), "n_cells": 4}
        event = GeneratedHexmapEvent(name="stress", expression="maximum(temperature - 18.0, 0.0)", output_name="stress_map")
        mask = np.ones(5, dtype=bool)
        event.execute(pop, landscape, t=0, mask=mask)
        np.testing.assert_array_almost_equal(landscape["stress_map"], [0.0, 0.0, 2.0, 7.0])

    def test_time_varying_expression(self):
        from salmon_ibm.events_phase3 import GeneratedHexmapEvent
        pop = MockPopulation(5)
        base = np.array([10.0, 12.0, 14.0])
        landscape = {"base_temp": base, "n_cells": 3}
        event = GeneratedHexmapEvent(name="seasonal", expression="base_temp + 5.0 * sin(t * pi / 180)", output_name="current_temp")
        mask = np.ones(5, dtype=bool)
        event.execute(pop, landscape, t=90, mask=mask)
        expected = base + 5.0
        np.testing.assert_array_almost_equal(landscape["current_temp"], expected)


class TestRangeDynamicsEvent:
    def test_no_crash_without_ranges(self):
        from salmon_ibm.events_phase3 import RangeDynamicsEvent
        pop = MockPopulation(5)
        landscape = {"resources": np.array([1.0, 2.0, 3.0])}
        event = RangeDynamicsEvent(name="range_expand", mode="expand")
        mask = np.ones(5, dtype=bool)
        event.execute(pop, landscape, t=0, mask=mask)


class TestSetAffinityEvent:
    def test_spatial_affinity_sets_strength(self):
        from salmon_ibm.events_phase3 import SetAffinityEvent
        pop = MockPopulation(5)
        pop.spatial_affinity = np.zeros(5, dtype=np.float64)
        pop.spatial_affinity_map_name = None
        landscape = {"habitat_quality": np.array([1.0, 2.0, 3.0])}
        event = SetAffinityEvent(name="attract", affinity_type="spatial", affinity_map_name="habitat_quality", strength=0.8)
        mask = np.array([True, True, False, False, True])
        event.execute(pop, landscape, t=0, mask=mask)
        assert pop.spatial_affinity[0] == 0.8
        assert pop.spatial_affinity[2] == 0.0
        assert pop.spatial_affinity[4] == 0.8

    def test_group_affinity_sets_strength(self):
        from salmon_ibm.events_phase3 import SetAffinityEvent
        pop = MockPopulation(3)
        pop.group_affinity = np.zeros(3, dtype=np.float64)
        event = SetAffinityEvent(name="cohesion", affinity_type="group", strength=1.5)
        mask = np.ones(3, dtype=bool)
        event.execute(pop, {}, t=0, mask=mask)
        assert np.all(pop.group_affinity == 1.5)

    def test_no_crash_without_attributes(self):
        from salmon_ibm.events_phase3 import SetAffinityEvent
        pop = MockPopulation(3)
        event = SetAffinityEvent(name="test", affinity_type="spatial", strength=1.0)
        mask = np.ones(3, dtype=bool)
        event.execute(pop, {}, t=0, mask=mask)
