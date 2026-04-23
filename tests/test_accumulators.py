import numpy as np
import pytest
from salmon_ibm.accumulators import AccumulatorDef, AccumulatorManager


class TestAccumulatorManager:
    def test_create_manager_with_definitions(self):
        defs = [
            AccumulatorDef(name="energy", min_val=0.0, max_val=100.0),
            AccumulatorDef(name="age"),
        ]
        mgr = AccumulatorManager(n_agents=10, definitions=defs)
        assert mgr.data.shape == (10, 2)
        assert mgr.data.dtype == np.float64
        assert np.all(mgr.data == 0.0)

    def test_get_set_by_name(self):
        defs = [AccumulatorDef(name="energy"), AccumulatorDef(name="age")]
        mgr = AccumulatorManager(n_agents=5, definitions=defs)
        mgr.set("energy", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = mgr.get("energy")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_get_set_by_index(self):
        defs = [AccumulatorDef(name="energy")]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        mgr.set(0, np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(mgr.get(0), [10.0, 20.0, 30.0])

    def test_bounds_clamping(self):
        defs = [AccumulatorDef(name="energy", min_val=0.0, max_val=100.0)]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        mgr.set("energy", np.array([-5.0, 50.0, 150.0]))
        np.testing.assert_array_equal(mgr.get("energy"), [0.0, 50.0, 100.0])

    def test_unknown_name_raises(self):
        defs = [AccumulatorDef(name="energy")]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        with pytest.raises(KeyError):
            mgr.get("nonexistent")

    def test_index_lookup(self):
        defs = [
            AccumulatorDef(name="a"),
            AccumulatorDef(name="b"),
            AccumulatorDef(name="c"),
        ]
        mgr = AccumulatorManager(n_agents=2, definitions=defs)
        assert mgr.index_of("a") == 0
        assert mgr.index_of("b") == 1
        assert mgr.index_of("c") == 2

    def test_masked_set(self):
        defs = [AccumulatorDef(name="energy", min_val=0.0)]
        mgr = AccumulatorManager(n_agents=4, definitions=defs)
        mask = np.array([True, False, True, False])
        mgr.set("energy", np.array([99.0, 99.0]), mask=mask)
        np.testing.assert_array_equal(mgr.get("energy"), [99.0, 0.0, 99.0, 0.0])


from salmon_ibm.accumulators import (
    updater_clear,
    updater_increment,
    updater_stochastic_increment,
    updater_expression,
    updater_time_step,
    updater_individual_id,
    updater_stochastic_trigger,
    updater_quantify_location,
)


class TestSimpleUpdaters:
    def _make_manager(self):
        defs = [AccumulatorDef(name="energy", min_val=0.0, max_val=100.0)]
        mgr = AccumulatorManager(n_agents=5, definitions=defs)
        mgr.set("energy", np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        return mgr

    def test_clear_resets_to_zero(self):
        mgr = self._make_manager()
        mask = np.ones(5, dtype=bool)
        updater_clear(mgr, "energy", mask)
        np.testing.assert_array_equal(mgr.get("energy"), [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_clear_respects_mask(self):
        mgr = self._make_manager()
        mask = np.array([True, False, True, False, True])
        updater_clear(mgr, "energy", mask)
        np.testing.assert_array_equal(mgr.get("energy"), [0.0, 20.0, 0.0, 40.0, 0.0])

    def test_increment_adds_fixed_quantity(self):
        mgr = self._make_manager()
        mask = np.ones(5, dtype=bool)
        updater_increment(mgr, "energy", mask, amount=5.0)
        np.testing.assert_array_equal(mgr.get("energy"), [15.0, 25.0, 35.0, 45.0, 55.0])

    def test_increment_clamps_to_bounds(self):
        mgr = self._make_manager()
        mask = np.ones(5, dtype=bool)
        updater_increment(mgr, "energy", mask, amount=70.0)
        np.testing.assert_array_equal(
            mgr.get("energy"), [80.0, 90.0, 100.0, 100.0, 100.0]
        )

    def test_stochastic_increment_adds_uniform_random(self):
        mgr = self._make_manager()
        mask = np.ones(5, dtype=bool)
        updater_stochastic_increment(
            mgr, "energy", mask, low=1.0, high=2.0, rng=np.random.default_rng(42)
        )
        result = mgr.get("energy")
        original = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        increments = result - original
        assert np.all(increments >= 1.0)
        assert np.all(increments < 2.0)

    def test_stochastic_increment_respects_mask(self):
        mgr = self._make_manager()
        mask = np.array([True, False, False, False, True])
        updater_stochastic_increment(
            mgr, "energy", mask, low=5.0, high=6.0, rng=np.random.default_rng(0)
        )
        result = mgr.get("energy")
        assert result[1] == 20.0
        assert result[2] == 30.0
        assert result[3] == 40.0
        assert result[0] > 10.0
        assert result[4] > 50.0


class TestExpressionUpdater:
    def _make_manager(self):
        defs = [
            AccumulatorDef(name="energy"),
            AccumulatorDef(name="age"),
            AccumulatorDef(name="result"),
        ]
        mgr = AccumulatorManager(n_agents=4, definitions=defs)
        mgr.set("energy", np.array([10.0, 20.0, 30.0, 40.0]))
        mgr.set("age", np.array([1.0, 2.0, 3.0, 4.0]))
        return mgr

    def test_simple_addition(self):
        mgr = self._make_manager()
        mask = np.ones(4, dtype=bool)
        updater_expression(mgr, "result", mask, expression="energy + age")
        np.testing.assert_array_equal(mgr.get("result"), [11.0, 22.0, 33.0, 44.0])

    def test_multiplication_and_constants(self):
        mgr = self._make_manager()
        mask = np.ones(4, dtype=bool)
        updater_expression(mgr, "result", mask, expression="energy * 0.5 + 1.0")
        np.testing.assert_array_equal(mgr.get("result"), [6.0, 11.0, 16.0, 21.0])

    def test_respects_mask(self):
        mgr = self._make_manager()
        mask = np.array([True, False, True, False])
        updater_expression(mgr, "result", mask, expression="energy * 2")
        np.testing.assert_array_equal(mgr.get("result"), [20.0, 0.0, 60.0, 0.0])

    def test_math_functions(self):
        mgr = self._make_manager()
        mask = np.ones(4, dtype=bool)
        updater_expression(mgr, "result", mask, expression="sqrt(energy)")
        expected = np.sqrt(np.array([10.0, 20.0, 30.0, 40.0]))
        np.testing.assert_allclose(mgr.get("result"), expected)

    def test_rejects_dangerous_expressions(self):
        mgr = self._make_manager()
        mask = np.ones(4, dtype=bool)
        with pytest.raises(ValueError):
            updater_expression(
                mgr, "result", mask, expression="__import__('os').system('rm -rf /')"
            )

    def test_rejects_unknown_variable(self):
        mgr = self._make_manager()
        mask = np.ones(4, dtype=bool)
        with pytest.raises((KeyError, NameError)):
            updater_expression(mgr, "result", mask, expression="nonexistent + 1")


class TestTimestepUpdater:
    def test_writes_current_timestep(self):
        defs = [AccumulatorDef(name="step")]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        mask = np.ones(3, dtype=bool)
        updater_time_step(mgr, "step", mask, timestep=42)
        np.testing.assert_array_equal(mgr.get("step"), [42.0, 42.0, 42.0])

    def test_modulus(self):
        defs = [AccumulatorDef(name="day_of_year")]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        mask = np.ones(3, dtype=bool)
        updater_time_step(mgr, "day_of_year", mask, timestep=370, modulus=365)
        np.testing.assert_array_equal(mgr.get("day_of_year"), [5.0, 5.0, 5.0])

    def test_respects_mask(self):
        defs = [AccumulatorDef(name="step")]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        mask = np.array([True, False, True])
        updater_time_step(mgr, "step", mask, timestep=10)
        np.testing.assert_array_equal(mgr.get("step"), [10.0, 0.0, 10.0])


class TestIndividualIDUpdater:
    def test_writes_agent_ids(self):
        defs = [AccumulatorDef(name="id")]
        mgr = AccumulatorManager(n_agents=5, definitions=defs)
        mask = np.ones(5, dtype=bool)
        agent_ids = np.array([100, 101, 102, 103, 104])
        updater_individual_id(mgr, "id", mask, agent_ids=agent_ids)
        np.testing.assert_array_equal(
            mgr.get("id"), [100.0, 101.0, 102.0, 103.0, 104.0]
        )

    def test_respects_mask(self):
        defs = [AccumulatorDef(name="id")]
        mgr = AccumulatorManager(n_agents=4, definitions=defs)
        mask = np.array([False, True, False, True])
        agent_ids = np.array([0, 1, 2, 3])
        updater_individual_id(mgr, "id", mask, agent_ids=agent_ids)
        np.testing.assert_array_equal(mgr.get("id"), [0.0, 1.0, 0.0, 3.0])


class TestStochasticTriggerUpdater:
    def test_returns_zero_or_one(self):
        defs = [AccumulatorDef(name="trigger")]
        mgr = AccumulatorManager(n_agents=1000, definitions=defs)
        mask = np.ones(1000, dtype=bool)
        updater_stochastic_trigger(
            mgr, "trigger", mask, probability=0.5, rng=np.random.default_rng(42)
        )
        vals = mgr.get("trigger")
        assert set(np.unique(vals)).issubset({0.0, 1.0})

    def test_probability_respected_statistically(self):
        defs = [AccumulatorDef(name="trigger")]
        mgr = AccumulatorManager(n_agents=10000, definitions=defs)
        mask = np.ones(10000, dtype=bool)
        updater_stochastic_trigger(
            mgr, "trigger", mask, probability=0.3, rng=np.random.default_rng(123)
        )
        frac = mgr.get("trigger").mean()
        assert 0.27 < frac < 0.33, f"Expected ~0.3, got {frac}"

    def test_probability_zero_all_zeros(self):
        defs = [AccumulatorDef(name="trigger")]
        mgr = AccumulatorManager(n_agents=100, definitions=defs)
        mask = np.ones(100, dtype=bool)
        updater_stochastic_trigger(
            mgr, "trigger", mask, probability=0.0, rng=np.random.default_rng(0)
        )
        assert np.all(mgr.get("trigger") == 0.0)

    def test_probability_one_all_ones(self):
        defs = [AccumulatorDef(name="trigger")]
        mgr = AccumulatorManager(n_agents=100, definitions=defs)
        mask = np.ones(100, dtype=bool)
        updater_stochastic_trigger(
            mgr, "trigger", mask, probability=1.0, rng=np.random.default_rng(0)
        )
        assert np.all(mgr.get("trigger") == 1.0)


class TestQuantifyLocationUpdater:
    def test_samples_hexmap_at_agent_positions(self):
        defs = [AccumulatorDef(name="temperature")]
        mgr = AccumulatorManager(n_agents=4, definitions=defs)
        mask = np.ones(4, dtype=bool)
        hex_map = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
        agent_cells = np.array([0, 3, 7, 9])
        updater_quantify_location(
            mgr, "temperature", mask, hex_map=hex_map, cell_indices=agent_cells
        )
        np.testing.assert_array_equal(mgr.get("temperature"), [15.0, 18.0, 22.0, 24.0])

    def test_respects_mask(self):
        defs = [AccumulatorDef(name="depth")]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        mask = np.array([True, False, True])
        hex_map = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        agent_cells = np.array([0, 2, 4])
        updater_quantify_location(
            mgr, "depth", mask, hex_map=hex_map, cell_indices=agent_cells
        )
        np.testing.assert_array_equal(mgr.get("depth"), [5.0, 0.0, 25.0])


from salmon_ibm.agents import AgentPool
from salmon_ibm.traits import TraitType, TraitDefinition, TraitManager


class TestAgentPoolIntegration:
    def test_pool_has_no_accumulators_by_default(self):
        pool = AgentPool(n=5, start_tri=0)
        assert pool.accumulators is None
        assert pool.traits is None

    def test_pool_with_accumulators(self):
        pool = AgentPool(n=5, start_tri=0)
        defs = [AccumulatorDef(name="energy", min_val=0.0)]
        pool.accumulators = AccumulatorManager(n_agents=5, definitions=defs)
        pool.accumulators.set("energy", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        np.testing.assert_array_equal(
            pool.accumulators.get("energy"), [1.0, 2.0, 3.0, 4.0, 5.0]
        )

    def test_pool_with_traits(self):
        pool = AgentPool(n=4, start_tri=0)
        td = TraitDefinition(
            name="stage",
            trait_type=TraitType.PROBABILISTIC,
            categories=["juv", "adult"],
        )
        pool.traits = TraitManager(n_agents=4, definitions=[td])
        pool.traits.set("stage", np.array([0, 1, 1, 0], dtype=np.int32))
        np.testing.assert_array_equal(pool.traits.get("stage"), [0, 1, 1, 0])

    def test_updater_with_pool_cell_indices(self):
        from salmon_ibm.accumulators import updater_quantify_location

        pool = AgentPool(n=3, start_tri=np.array([0, 5, 9]))
        defs = [AccumulatorDef(name="depth")]
        pool.accumulators = AccumulatorManager(n_agents=3, definitions=defs)
        hex_map = np.arange(10, dtype=np.float64) * 10.0
        mask = np.ones(3, dtype=bool)
        updater_quantify_location(
            pool.accumulators, "depth", mask, hex_map=hex_map, cell_indices=pool.tri_idx
        )
        np.testing.assert_array_equal(pool.accumulators.get("depth"), [0.0, 50.0, 90.0])

    def test_accumulated_trait_with_pool(self):
        pool = AgentPool(n=4, start_tri=0)
        acc_defs = [AccumulatorDef(name="energy", linked_trait="condition")]
        pool.accumulators = AccumulatorManager(n_agents=4, definitions=acc_defs)
        trait_def = TraitDefinition(
            name="condition",
            trait_type=TraitType.ACCUMULATED,
            categories=["low", "medium", "high"],
            accumulator_name="energy",
            thresholds=np.array([30.0, 70.0]),
        )
        pool.traits = TraitManager(n_agents=4, definitions=[trait_def])
        pool.accumulators.set("energy", np.array([10.0, 40.0, 80.0, 70.0]))
        pool.traits.evaluate_accumulated("condition", pool.accumulators)
        np.testing.assert_array_equal(pool.traits.get("condition"), [0, 1, 2, 2])


from salmon_ibm.accumulators import (
    updater_accumulator_transfer,
    updater_group_size,
    updater_group_sum,
    updater_hexagon_presence,
    updater_uptake,
    updater_individual_locations,
    updater_subpopulation_assign,
    updater_trait_value_index,
    updater_data_lookup,
)


class TestRemainingUpdaters:
    def test_accumulator_transfer(self):
        defs = [AccumulatorDef("src"), AccumulatorDef("tgt")]
        mgr = AccumulatorManager(3, defs)
        mgr.set("src", np.array([10.0, 20.0, 30.0]))
        mask = np.ones(3, dtype=bool)
        updater_accumulator_transfer(mgr, "src", "tgt", mask, fraction=0.5)
        np.testing.assert_array_almost_equal(mgr.get("src"), [5.0, 10.0, 15.0])
        np.testing.assert_array_almost_equal(mgr.get("tgt"), [5.0, 10.0, 15.0])

    def test_group_size(self):
        defs = [AccumulatorDef("gsize")]
        mgr = AccumulatorManager(6, defs)
        mask = np.ones(6, dtype=bool)
        group_ids = np.array([0, 0, 0, 1, 1, -1], dtype=np.int32)
        updater_group_size(mgr, "gsize", mask, group_ids=group_ids)
        np.testing.assert_array_equal(mgr.get("gsize"), [3.0, 3.0, 3.0, 2.0, 2.0, 0.0])

    def test_group_sum(self):
        defs = [AccumulatorDef("energy"), AccumulatorDef("group_energy")]
        mgr = AccumulatorManager(4, defs)
        mgr.set("energy", np.array([10.0, 20.0, 30.0, 40.0]))
        mask = np.ones(4, dtype=bool)
        group_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        updater_group_sum(mgr, "group_energy", "energy", mask, group_ids=group_ids)
        np.testing.assert_array_equal(mgr.get("group_energy"), [30.0, 30.0, 70.0, 70.0])

    def test_hexagon_presence(self):
        defs = [AccumulatorDef("present")]
        mgr = AccumulatorManager(3, defs)
        mask = np.ones(3, dtype=bool)
        hex_map = np.array([0.5, 1.5, 0.0, 2.0])
        cells = np.array([0, 1, 2])
        updater_hexagon_presence(
            mgr, "present", mask, hex_map=hex_map, cell_indices=cells, threshold=1.0
        )
        np.testing.assert_array_equal(mgr.get("present"), [0.0, 1.0, 0.0])

    def test_uptake_depletes_map(self):
        defs = [AccumulatorDef("food")]
        mgr = AccumulatorManager(2, defs)
        mask = np.ones(2, dtype=bool)
        hex_map = np.array([10.0, 20.0, 30.0])
        cells = np.array([0, 2])
        updater_uptake(mgr, "food", mask, hex_map=hex_map, cell_indices=cells, rate=0.5)
        np.testing.assert_array_almost_equal(mgr.get("food"), [5.0, 15.0])
        np.testing.assert_array_almost_equal(hex_map, [5.0, 20.0, 15.0])

    def test_individual_locations(self):
        defs = [AccumulatorDef("loc")]
        mgr = AccumulatorManager(3, defs)
        mask = np.ones(3, dtype=bool)
        cells = np.array([7, 12, 3])
        updater_individual_locations(mgr, "loc", mask, cell_indices=cells)
        np.testing.assert_array_equal(mgr.get("loc"), [7.0, 12.0, 3.0])

    def test_subpopulation_assign(self):
        defs = [AccumulatorDef("selected")]
        mgr = AccumulatorManager(10, defs)
        mask = np.ones(10, dtype=bool)
        updater_subpopulation_assign(
            mgr, "selected", mask, n_select=3, value=1.0, rng=np.random.default_rng(42)
        )
        assert (mgr.get("selected") == 1.0).sum() == 3

    def test_trait_value_index(self):
        from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType

        defs = [AccumulatorDef("stage_val")]
        mgr = AccumulatorManager(4, defs)
        trait_defs = [
            TraitDefinition("stage", TraitType.PROBABILISTIC, ["juv", "adult"])
        ]
        tmgr = TraitManager(4, trait_defs)
        tmgr._data["stage"] = np.array([0, 1, 1, 0], dtype=np.int32)
        mask = np.ones(4, dtype=bool)
        updater_trait_value_index(
            mgr, "stage_val", mask, trait_mgr=tmgr, trait_name="stage"
        )
        np.testing.assert_array_equal(mgr.get("stage_val"), [0.0, 1.0, 1.0, 0.0])

    def test_data_lookup(self):
        defs = [AccumulatorDef("key"), AccumulatorDef("result")]
        mgr = AccumulatorManager(3, defs)
        mgr.set("key", np.array([0.0, 2.0, 1.0]))
        lookup = np.array([100.0, 200.0, 300.0])
        mask = np.ones(3, dtype=bool)
        updater_data_lookup(
            mgr, "result", mask, lookup_table=lookup, key_acc_name="key"
        )
        np.testing.assert_array_equal(mgr.get("result"), [100.0, 300.0, 200.0])


def test_updater_uptake_can_overdeplete():
    """Multiple agents extracting from a low-resource cell can drive hex_map negative."""
    from salmon_ibm.accumulators import (
        AccumulatorManager,
        AccumulatorDef,
        updater_uptake,
    )

    mgr = AccumulatorManager(3, [AccumulatorDef("food")])
    hex_map = np.array([5.0, 100.0])
    cell_indices = np.array([0, 0, 0])  # all 3 on cell 0
    mask = np.array([True, True, True])
    updater_uptake(
        mgr, "food", mask, hex_map=hex_map, cell_indices=cell_indices, rate=1.0
    )
    # Each agent extracts 5.0. Total depletion = 15. hex_map goes to -10.
    assert hex_map[0] == pytest.approx(-10.0)
    # Each agent got 5.0
    assert np.all(mgr.data[:, 0] == pytest.approx(5.0))


def test_updater_uptake_multi_agent_same_cell():
    """Two agents on the same cell should each deplete the resource."""
    from salmon_ibm.accumulators import (
        AccumulatorManager,
        AccumulatorDef,
        updater_uptake,
    )

    mgr = AccumulatorManager(2, [AccumulatorDef("food")])
    hex_map = np.array([100.0, 50.0, 50.0])
    cell_indices = np.array([0, 0])  # both agents on cell 0
    mask = np.array([True, True])
    updater_uptake(
        mgr, "food", mask, hex_map=hex_map, cell_indices=cell_indices, rate=0.1
    )
    # Each agent extracts 100 * 0.1 = 10. Total depletion should be 20.
    assert hex_map[0] == pytest.approx(80.0), f"Expected 80.0, got {hex_map[0]}"
    # Each agent should have received 10.0
    assert mgr.data[0, 0] == pytest.approx(10.0)
    assert mgr.data[1, 0] == pytest.approx(10.0)


def test_individual_locations_writes_cell_indices():
    """IndividualLocations must write each agent's tri_idx to its accumulator."""
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_individual_locations

    defs = [AccumulatorDef(name="pos", min_val=0.0, max_val=None)]
    mgr = AccumulatorManager(n_agents=4, definitions=defs)
    cell_indices = np.array([10, 20, 30, 40], dtype=np.int64)
    mask = np.ones(4, dtype=bool)

    updater_individual_locations(mgr, "pos", mask, cell_indices=cell_indices)

    np.testing.assert_array_equal(mgr.data[:, 0], [10.0, 20.0, 30.0, 40.0])


def test_accumulator_transfer_clamps_source_to_min_val():
    """Source accumulator must not fall below its own min_val after transfer."""
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_accumulator_transfer

    defs = [
        AccumulatorDef(name="src", min_val=1.0, max_val=100.0),
        AccumulatorDef(name="tgt", min_val=0.0, max_val=100.0),
    ]
    mgr = AccumulatorManager(n_agents=2, definitions=defs)
    mgr.data[:, 0] = 5.0  # src = 5, min_val = 1.0
    mask = np.ones(2, dtype=bool)

    # Fraction=1.0 would drive src to 0, below min_val=1.0
    updater_accumulator_transfer(mgr, "src", "tgt", mask, fraction=1.0)

    assert np.all(mgr.data[:, 0] >= 1.0), f"src must clamp to min_val=1.0, got {mgr.data[:, 0]}"


def test_accumulator_transfer_conserves_mass_under_clamp():
    """When source clamp reduces the actual subtracted amount, target must
    receive the actual amount — not the pre-clamp nominal amount.
    Conservation: delta_src + delta_tgt == 0 (modulo target clamping)."""
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_accumulator_transfer

    defs = [
        AccumulatorDef(name="src", min_val=1.0, max_val=100.0),
        AccumulatorDef(name="tgt", min_val=0.0, max_val=100.0),
    ]
    mgr = AccumulatorManager(n_agents=2, definitions=defs)
    mgr.data[:, 0] = 5.0  # src = 5, clamp floor 1.0
    mgr.data[:, 1] = 0.0
    mask = np.ones(2, dtype=bool)

    updater_accumulator_transfer(mgr, "src", "tgt", mask, fraction=1.0)

    # src is clamped from 0 back up to 1.0 → actual amount moved was 4.0, not 5.0.
    # Target must receive exactly 4.0 (no phantom mass).
    np.testing.assert_array_almost_equal(mgr.data[:, 0], [1.0, 1.0])
    np.testing.assert_array_almost_equal(mgr.data[:, 1], [4.0, 4.0])


def test_ast_sandbox_allows_rng_whitelisted_methods():
    """Legitimate _rng methods in the allowlist must pass validation."""
    from salmon_ibm.accumulators import _validate_expression, _HEXSIM_FUNCTIONS

    # Should not raise:
    _validate_expression("_rng.random(1)", extra_names=_HEXSIM_FUNCTIONS)
    _validate_expression("_rng.uniform(0, 1)", extra_names=_HEXSIM_FUNCTIONS)
    _validate_expression("_rng.normal(0, 1)", extra_names=_HEXSIM_FUNCTIONS)
    _validate_expression("_rng.integers(0, 10)", extra_names=_HEXSIM_FUNCTIONS)


def test_ast_sandbox_rejects_non_allowlisted_rng_methods():
    """_rng.seed, _rng.bytes, _rng.choice, etc. must be rejected (DoS/attack surface)."""
    import pytest
    from salmon_ibm.accumulators import _validate_expression, _HEXSIM_FUNCTIONS

    for malicious in [
        "_rng.seed(42)",
        "_rng.bytes(1000)",
        "_rng.permutation(10)",
        "_rng.choice([1, 2, 3])",
    ]:
        with pytest.raises(ValueError, match=r"method|Disallowed"):
            _validate_expression(malicious, extra_names=_HEXSIM_FUNCTIONS)


def test_ast_sandbox_rejects_huge_rng_arg_dos():
    """_rng.random(10**9) must fail — prevents GB-scale allocation DoS from scenario XML."""
    import pytest
    from salmon_ibm.accumulators import _validate_expression, _HEXSIM_FUNCTIONS, _RNG_ARG_MAX

    with pytest.raises(ValueError, match="too large"):
        _validate_expression("_rng.random(10000000000)", extra_names=_HEXSIM_FUNCTIONS)

    # Normal-sized arg still passes.
    _validate_expression(f"_rng.random({_RNG_ARG_MAX - 1})", extra_names=_HEXSIM_FUNCTIONS)


def test_ast_sandbox_rejects_method_call_on_non_rng_attribute():
    """Method calls via attribute access on anything other than _rng must fail."""
    import pytest
    from salmon_ibm.accumulators import _validate_expression, _HEXSIM_FUNCTIONS

    with pytest.raises(ValueError):
        _validate_expression("foo.bar(1)", extra_names=_HEXSIM_FUNCTIONS)


def test_compiled_expr_cache_bounded_to_lru_size():
    """Cache must evict oldest entries when over capacity; should not grow unbounded."""
    from salmon_ibm.accumulators import _compiled_expr_cache, _EXPR_CACHE_MAX
    from collections import OrderedDict

    _compiled_expr_cache.clear()
    # Use the cache's own insertion API by calling updater_expression, or simulate:
    for i in range(_EXPR_CACHE_MAX + 10):
        expr = f"expr_{i}"
        if expr not in _compiled_expr_cache:
            if len(_compiled_expr_cache) >= _EXPR_CACHE_MAX:
                _compiled_expr_cache.popitem(last=False)
            _compiled_expr_cache[expr] = compile(str(i), "<test>", "eval")
    assert len(_compiled_expr_cache) <= _EXPR_CACHE_MAX, (
        f"Cache grew to {len(_compiled_expr_cache)}, expected <= {_EXPR_CACHE_MAX}"
    )
    # Oldest entries should have been evicted (first 10).
    assert "expr_0" not in _compiled_expr_cache
    assert f"expr_{_EXPR_CACHE_MAX + 9}" in _compiled_expr_cache


def test_translate_cache_bounded_to_lru_size():
    """hexsim_expr._translate_cache must also be LRU-bounded."""
    from salmon_ibm.hexsim_expr import translate_hexsim_expr, _translate_cache, _TRANSLATE_CACHE_MAX

    _translate_cache.clear()
    for i in range(_TRANSLATE_CACHE_MAX + 10):
        translate_hexsim_expr(f"'global_{i}' + 1")
    assert len(_translate_cache) <= _TRANSLATE_CACHE_MAX
