"""Tests for HexSim-specific event types."""
import numpy as np
import pytest
from salmon_ibm.agents import AgentPool
from salmon_ibm.population import Population
from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
from salmon_ibm.events_hexsim import (
    HexSimSurvivalEvent,
    PatchIntroductionEvent,
    DataLookupEvent,
    SetSpatialAffinityEvent,
    HexSimMoveEvent,
    DataProbeEvent,
)


def _make_pop(name, n=10, n_cells=100):
    pool = AgentPool(n=n, start_tri=np.zeros(n, dtype=int))
    return Population(name=name, pool=pool)


class TestHexSimSurvival:
    def test_kills_low_accumulator_agents(self):
        pop = _make_pop("fish", 5)
        defs = [AccumulatorDef("Survival [ switch ]")]
        pop.accumulator_mgr = AccumulatorManager(5, defs)
        pop.accumulator_mgr.data[:, 0] = [0.0, 1.0, 0.0, 1.0, 1.0]
        mask = pop.alive.copy()
        evt = HexSimSurvivalEvent(name="surv", survival_accumulator="Survival [ switch ]")
        evt.execute(pop, {}, 0, mask)
        assert pop.alive.tolist() == [False, True, False, True, True]

    def test_no_acc_mgr_is_noop(self):
        pop = _make_pop("fish", 3)
        pop.accumulator_mgr = None
        mask = pop.alive.copy()
        evt = HexSimSurvivalEvent(name="surv", survival_accumulator="x")
        evt.execute(pop, {}, 0, mask)
        assert pop.alive.all()

    def test_empty_accumulator_name_is_noop(self):
        pop = _make_pop("fish", 3)
        mask = pop.alive.copy()
        evt = HexSimSurvivalEvent(name="surv", survival_accumulator="")
        evt.execute(pop, {}, 0, mask)
        assert pop.alive.all()

    def test_all_survive_when_all_above_threshold(self):
        pop = _make_pop("fish", 4)
        defs = [AccumulatorDef("surv_acc")]
        pop.accumulator_mgr = AccumulatorManager(4, defs)
        pop.accumulator_mgr.data[:, 0] = [1.0, 2.0, 5.0, 1.0]
        mask = pop.alive.copy()
        evt = HexSimSurvivalEvent(name="surv", survival_accumulator="surv_acc")
        evt.execute(pop, {}, 0, mask)
        assert pop.alive.all()

    def test_all_die_when_all_below_threshold(self):
        pop = _make_pop("fish", 4)
        defs = [AccumulatorDef("surv_acc")]
        pop.accumulator_mgr = AccumulatorManager(4, defs)
        pop.accumulator_mgr.data[:, 0] = [0.0, 0.5, 0.0, 0.99]
        mask = pop.alive.copy()
        evt = HexSimSurvivalEvent(name="surv", survival_accumulator="surv_acc")
        evt.execute(pop, {}, 0, mask)
        assert not pop.alive.any()


class TestPatchIntroduction:
    def test_adds_agents_at_nonzero_cells(self):
        pop = _make_pop("refuges", 0)
        pop.pool.n = 0
        pop.pool.alive = np.array([], dtype=bool)
        layer = np.array([0, 5, 0, 3, 0, 7])
        landscape = {"spatial_data": {"refuges_layer": layer}}
        evt = PatchIntroductionEvent(name="add", patch_spatial_data="refuges_layer")
        evt.execute(pop, landscape, 0, np.array([], dtype=bool))
        assert pop.pool.n == 3  # cells 1, 3, 5 are non-zero

    def test_missing_layer_is_noop(self):
        pop = _make_pop("refuges", 0)
        pop.pool.n = 0
        pop.pool.alive = np.array([], dtype=bool)
        landscape = {"spatial_data": {}}
        evt = PatchIntroductionEvent(name="add", patch_spatial_data="missing_layer")
        evt.execute(pop, landscape, 0, np.array([], dtype=bool))
        assert pop.pool.n == 0

    def test_all_zero_layer_is_noop(self):
        pop = _make_pop("refuges", 0)
        pop.pool.n = 0
        pop.pool.alive = np.array([], dtype=bool)
        layer = np.zeros(10, dtype=float)
        landscape = {"spatial_data": {"empty_layer": layer}}
        evt = PatchIntroductionEvent(name="add", patch_spatial_data="empty_layer")
        evt.execute(pop, landscape, 0, np.array([], dtype=bool))
        assert pop.pool.n == 0

    def test_agents_placed_at_correct_cells(self):
        pop = _make_pop("refuges", 0)
        pop.pool.n = 0
        pop.pool.alive = np.array([], dtype=bool)
        layer = np.array([0, 0, 9, 0, 4])
        landscape = {"spatial_data": {"layer": layer}}
        evt = PatchIntroductionEvent(name="add", patch_spatial_data="layer")
        evt.execute(pop, landscape, 0, np.array([], dtype=bool))
        assert pop.pool.n == 2
        # Agents should be at cells 2 and 4
        placed_cells = set(pop.pool.tri_idx.tolist())
        assert placed_cells == {2, 4}


class TestDataLookup:
    def test_2d_lookup(self):
        pop = _make_pop("fish", 3)
        defs = [AccumulatorDef("row"), AccumulatorDef("col"), AccumulatorDef("result")]
        pop.accumulator_mgr = AccumulatorManager(3, defs)
        pop.accumulator_mgr.data[:, 0] = [0, 1, 2]  # row keys
        pop.accumulator_mgr.data[:, 1] = [0, 1, 0]  # col keys
        table = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
        mask = pop.alive.copy()
        evt = DataLookupEvent(
            name="lookup",
            row_accumulator="row",
            column_accumulator="col",
            target_accumulator="result",
            lookup_table=table,
        )
        evt.execute(pop, {}, 0, mask)
        expected = [10.0, 40.0, 50.0]
        np.testing.assert_array_equal(pop.accumulator_mgr.get("result"), expected)

    def test_1d_lookup(self):
        pop = _make_pop("fish", 3)
        defs = [AccumulatorDef("row"), AccumulatorDef("result")]
        pop.accumulator_mgr = AccumulatorManager(3, defs)
        pop.accumulator_mgr.data[:, 0] = [0, 2, 1]
        table = np.array([100.0, 200.0, 300.0])
        mask = pop.alive.copy()
        evt = DataLookupEvent(
            name="lookup",
            row_accumulator="row",
            target_accumulator="result",
            lookup_table=table,
        )
        evt.execute(pop, {}, 0, mask)
        np.testing.assert_array_equal(pop.accumulator_mgr.get("result"), [100.0, 300.0, 200.0])

    def test_out_of_bounds_rows_give_zero(self):
        pop = _make_pop("fish", 3)
        defs = [AccumulatorDef("row"), AccumulatorDef("result")]
        pop.accumulator_mgr = AccumulatorManager(3, defs)
        pop.accumulator_mgr.data[:, 0] = [-1, 0, 99]  # -1 and 99 are OOB
        table = np.array([10.0, 20.0, 30.0])
        mask = pop.alive.copy()
        evt = DataLookupEvent(
            name="lookup",
            row_accumulator="row",
            target_accumulator="result",
            lookup_table=table,
        )
        evt.execute(pop, {}, 0, mask)
        result = pop.accumulator_mgr.get("result")
        assert result[0] == 0.0   # OOB -> 0
        assert result[1] == 10.0  # valid
        assert result[2] == 0.0   # OOB -> 0

    def test_no_acc_mgr_is_noop(self):
        pop = _make_pop("fish", 3)
        pop.accumulator_mgr = None
        mask = pop.alive.copy()
        table = np.array([1.0, 2.0])
        evt = DataLookupEvent(
            name="lookup",
            row_accumulator="row",
            target_accumulator="result",
            lookup_table=table,
        )
        evt.execute(pop, {}, 0, mask)  # should not raise

    def test_no_lookup_table_is_noop(self):
        pop = _make_pop("fish", 3)
        defs = [AccumulatorDef("row"), AccumulatorDef("result")]
        pop.accumulator_mgr = AccumulatorManager(3, defs)
        mask = pop.alive.copy()
        evt = DataLookupEvent(
            name="lookup",
            row_accumulator="row",
            target_accumulator="result",
            lookup_table=None,
        )
        evt.execute(pop, {}, 0, mask)  # should not raise


class TestDataProbe:
    def test_noop(self):
        pop = _make_pop("fish")
        evt = DataProbeEvent(name="probe")
        evt.execute(pop, {}, 0, pop.alive.copy())  # should not crash

    def test_noop_with_none_population(self):
        evt = DataProbeEvent(name="probe")
        evt.execute(None, {}, 0, np.array([], dtype=bool))  # should not crash


class TestPopulationAffinityArrays:
    def test_affinity_arrays_exist(self):
        pop = _make_pop("fish", 5)
        assert hasattr(pop, 'affinity_targets')
        assert hasattr(pop, 'spatial_affinity')
        assert len(pop.affinity_targets) == 5
        assert len(pop.spatial_affinity) == 5

    def test_affinity_targets_initialized_to_minus_one(self):
        pop = _make_pop("fish", 4)
        assert (pop.affinity_targets == -1).all()

    def test_spatial_affinity_initialized_to_zero(self):
        pop = _make_pop("fish", 4)
        assert (pop.spatial_affinity == 0.0).all()

    def test_affinity_arrays_grow_with_add_agents(self):
        pop = _make_pop("fish", 3)
        pop.add_agents(2, np.array([5, 6]))
        assert len(pop.affinity_targets) == 5
        assert len(pop.spatial_affinity) == 5
        # New entries default to -1 and 0
        assert (pop.affinity_targets[3:] == -1).all()
        assert (pop.spatial_affinity[3:] == 0.0).all()

    def test_affinity_arrays_compact_with_dead_agents(self):
        pop = _make_pop("fish", 5)
        pop.affinity_targets[:] = [10, 20, 30, 40, 50]
        pop.spatial_affinity[:] = [1.0, 2.0, 3.0, 4.0, 5.0]
        pop.alive[1] = False
        pop.alive[3] = False
        pop.compact()
        assert len(pop.affinity_targets) == 3
        assert pop.affinity_targets.tolist() == [10, 30, 50]
        assert pop.spatial_affinity.tolist() == [1.0, 3.0, 5.0]


class TestEventRegistration:
    def test_all_events_registered(self):
        from salmon_ibm.events import EVENT_REGISTRY
        # Force registration by importing the module
        import salmon_ibm.events_hexsim  # noqa: F401
        assert "hexsim_survival" in EVENT_REGISTRY
        assert "patch_introduction" in EVENT_REGISTRY
        assert "data_lookup" in EVENT_REGISTRY
        assert "set_spatial_affinity" in EVENT_REGISTRY
        assert "move" in EVENT_REGISTRY
        assert "data_probe" in EVENT_REGISTRY
