"""Unit tests for Phase 3 events."""

import numpy as np

from tests.helpers import MockPopulation


class TestGeneratedHexmapEvent:
    def test_simple_expression(self):
        from salmon_ibm.events_phase3 import GeneratedHexmapEvent

        pop = MockPopulation(5)
        landscape = {"temperature": np.array([10.0, 15.0, 20.0, 25.0]), "n_cells": 4}
        event = GeneratedHexmapEvent(
            name="stress",
            expression="maximum(temperature - 18.0, 0.0)",
            output_name="stress_map",
            allowed_landscape_keys=("temperature",),
        )
        mask = np.ones(5, dtype=bool)
        event.execute(pop, landscape, t=0, mask=mask)
        np.testing.assert_array_almost_equal(
            landscape["stress_map"], [0.0, 0.0, 2.0, 7.0]
        )

    def test_time_varying_expression(self):
        from salmon_ibm.events_phase3 import GeneratedHexmapEvent

        pop = MockPopulation(5)
        base = np.array([10.0, 12.0, 14.0])
        landscape = {"base_temp": base, "n_cells": 3}
        event = GeneratedHexmapEvent(
            name="seasonal",
            expression="base_temp + 5.0 * sin(t * pi / 180)",
            output_name="current_temp",
            allowed_landscape_keys=("base_temp",),
        )
        mask = np.ones(5, dtype=bool)
        event.execute(pop, landscape, t=90, mask=mask)
        expected = base + 5.0
        np.testing.assert_array_almost_equal(landscape["current_temp"], expected)

    def test_landscape_key_not_injected_without_allowlist(self):
        """Top-level landscape ndarrays must NOT be auto-injected by substring match."""
        import pytest
        from salmon_ibm.events_phase3 import GeneratedHexmapEvent

        pop = MockPopulation(5)
        landscape = {
            "temperature": np.array([10.0, 15.0, 20.0, 25.0]),
            "n_cells": 4,
        }
        # Expression references 'temperature' but event does NOT list it in allowlist.
        event = GeneratedHexmapEvent(
            name="stress",
            expression="maximum(temperature - 18.0, 0.0)",
            output_name="stress_map",
            # allowed_landscape_keys defaults to ()
        )
        mask = np.ones(5, dtype=bool)
        # eval will fail because 'temperature' is not in the namespace.
        with pytest.raises(NameError):
            event.execute(pop, landscape, t=0, mask=mask)

    def test_spatial_data_key_cannot_shadow_safe_math(self):
        """A malicious spatial layer named 'sqrt' must NOT overwrite np.sqrt."""
        from salmon_ibm.events_phase3 import GeneratedHexmapEvent

        pop = MockPopulation(5)
        landscape = {
            "n_cells": 4,
            "spatial_data": {
                "sqrt": np.array([999.0, 999.0, 999.0, 999.0]),  # attempt to shadow
                "depth": np.array([1.0, 4.0, 9.0, 16.0]),
            },
        }
        event = GeneratedHexmapEvent(
            name="rooted",
            expression="sqrt(depth)",
            output_name="rooted_depth",
        )
        mask = np.ones(5, dtype=bool)
        event.execute(pop, landscape, t=0, mask=mask)
        # np.sqrt should have been used, not the spatial_data "sqrt" array.
        np.testing.assert_array_almost_equal(
            landscape["rooted_depth"], [1.0, 2.0, 3.0, 4.0]
        )

    def test_output_name_rejected_for_protected_landscape_key(self):
        """output_name='fields' or 'mesh' must raise ValueError."""
        import pytest
        from salmon_ibm.events_phase3 import GeneratedHexmapEvent

        pop = MockPopulation(5)
        landscape = {"n_cells": 4}
        event = GeneratedHexmapEvent(
            name="oops",
            expression="t + 1",
            output_name="fields",  # protected
        )
        mask = np.ones(5, dtype=bool)
        with pytest.raises(ValueError, match="protected"):
            event.execute(pop, landscape, t=0, mask=mask)


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
        event = SetAffinityEvent(
            name="attract",
            affinity_type="spatial",
            affinity_map_name="habitat_quality",
            strength=0.8,
        )
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


class TestPlantDynamicsEvent:
    def test_produces_seedlings(self):
        from salmon_ibm.events_phase3 import PlantDynamicsEvent
        from salmon_ibm.population import Population
        from salmon_ibm.agents import AgentPool

        pool = AgentPool(n=5, start_tri=np.array([0, 1, 2, 3, 4]), rng_seed=42)
        pop = Population("plants", pool)

        # Simple mesh mock
        class SimpleMesh:
            n_cells = 10
            n_triangles = 10
            _water_nbrs = np.full((10, 6), -1, dtype=np.intp)
            _water_nbr_count = np.zeros(10, dtype=np.intp)

        for i in range(10):
            k = 0
            if i > 0:
                SimpleMesh._water_nbrs[i, k] = i - 1
                k += 1
            if i < 9:
                SimpleMesh._water_nbrs[i, k] = i + 1
                k += 1
            SimpleMesh._water_nbr_count[i] = k

        landscape = {
            "rng": np.random.default_rng(42),
            "mesh": SimpleMesh(),
            "resources": np.ones(10) * 2.0,
        }
        event = PlantDynamicsEvent(name="plant", seed_production_rate=3.0)
        mask = pop.alive.copy()
        n_before = pop.n
        event.execute(pop, landscape, t=0, mask=mask)
        assert pop.n > n_before, "Seedlings should have been created"

    def test_no_crash_without_mesh(self):
        from salmon_ibm.events_phase3 import PlantDynamicsEvent

        pop = MockPopulation(3)
        landscape = {"rng": np.random.default_rng(42)}
        event = PlantDynamicsEvent(name="plant")
        event.execute(pop, landscape, t=0, mask=np.ones(3, dtype=bool))
