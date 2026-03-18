import numpy as np
import pytest
from salmon_ibm.traits import TraitType, TraitDefinition, TraitManager


class TestTraitManager:
    def test_create_manager(self):
        td = TraitDefinition(name="life_stage", trait_type=TraitType.PROBABILISTIC, categories=["juvenile", "subadult", "adult"])
        mgr = TraitManager(n_agents=10, definitions=[td])
        assert mgr.get("life_stage").shape == (10,)
        assert mgr.get("life_stage").dtype == np.int32
        assert np.all(mgr.get("life_stage") == 0)

    def test_set_trait_values(self):
        td = TraitDefinition(name="sex", trait_type=TraitType.PROBABILISTIC, categories=["female", "male"])
        mgr = TraitManager(n_agents=4, definitions=[td])
        mgr.set("sex", np.array([0, 1, 1, 0], dtype=np.int32))
        np.testing.assert_array_equal(mgr.get("sex"), [0, 1, 1, 0])

    def test_set_with_mask(self):
        td = TraitDefinition(name="status", trait_type=TraitType.PROBABILISTIC, categories=["healthy", "sick"])
        mgr = TraitManager(n_agents=4, definitions=[td])
        mask = np.array([False, True, True, False])
        mgr.set("status", np.array([1, 1], dtype=np.int32), mask=mask)
        np.testing.assert_array_equal(mgr.get("status"), [0, 1, 1, 0])

    def test_category_name_lookup(self):
        td = TraitDefinition(name="life_stage", trait_type=TraitType.PROBABILISTIC, categories=["egg", "fry", "smolt", "adult"])
        mgr = TraitManager(n_agents=3, definitions=[td])
        mgr.set("life_stage", np.array([0, 2, 3], dtype=np.int32))
        names = mgr.category_names("life_stage")
        assert names == ["egg", "smolt", "adult"]

    def test_unknown_trait_raises(self):
        mgr = TraitManager(n_agents=3, definitions=[])
        with pytest.raises(KeyError):
            mgr.get("nonexistent")

    def test_multiple_traits(self):
        defs = [
            TraitDefinition(name="sex", trait_type=TraitType.PROBABILISTIC, categories=["F", "M"]),
            TraitDefinition(name="stage", trait_type=TraitType.PROBABILISTIC, categories=["juv", "adult"]),
        ]
        mgr = TraitManager(n_agents=3, definitions=defs)
        mgr.set("sex", np.array([1, 0, 1], dtype=np.int32))
        mgr.set("stage", np.array([0, 1, 1], dtype=np.int32))
        np.testing.assert_array_equal(mgr.get("sex"), [1, 0, 1])
        np.testing.assert_array_equal(mgr.get("stage"), [0, 1, 1])


class TestAccumulatedTrait:
    def _make_system(self):
        from salmon_ibm.accumulators import AccumulatorDef, AccumulatorManager
        acc_defs = [AccumulatorDef(name="energy", linked_trait="condition")]
        acc_mgr = AccumulatorManager(n_agents=6, definitions=acc_defs)
        trait_def = TraitDefinition(
            name="condition", trait_type=TraitType.ACCUMULATED,
            categories=["critical", "poor", "fair", "good"],
            accumulator_name="energy", thresholds=np.array([20.0, 50.0, 80.0]),
        )
        trait_mgr = TraitManager(n_agents=6, definitions=[trait_def])
        return acc_mgr, trait_mgr

    def test_evaluate_accumulated_basic(self):
        acc_mgr, trait_mgr = self._make_system()
        acc_mgr.set("energy", np.array([5.0, 20.0, 35.0, 50.0, 80.0, 95.0]))
        trait_mgr.evaluate_accumulated("condition", acc_mgr)
        np.testing.assert_array_equal(trait_mgr.get("condition"), [0, 1, 1, 2, 3, 3])

    def test_evaluate_accumulated_all_below(self):
        acc_mgr, trait_mgr = self._make_system()
        acc_mgr.set("energy", np.array([0.0, 1.0, 5.0, 10.0, 15.0, 19.9]))
        trait_mgr.evaluate_accumulated("condition", acc_mgr)
        assert np.all(trait_mgr.get("condition") == 0)

    def test_evaluate_accumulated_all_above(self):
        acc_mgr, trait_mgr = self._make_system()
        acc_mgr.set("energy", np.array([80.0, 90.0, 100.0, 200.0, 80.1, 999.0]))
        trait_mgr.evaluate_accumulated("condition", acc_mgr)
        assert np.all(trait_mgr.get("condition") == 3)

    def test_evaluate_accumulated_with_mask(self):
        acc_mgr, trait_mgr = self._make_system()
        acc_mgr.set("energy", np.array([5.0, 95.0, 5.0, 95.0, 5.0, 95.0]))
        mask = np.array([True, True, True, False, False, False])
        trait_mgr.evaluate_accumulated("condition", acc_mgr, mask=mask)
        np.testing.assert_array_equal(trait_mgr.get("condition"), [0, 3, 0, 0, 0, 0])

    def test_raises_for_probabilistic_trait(self):
        trait_def = TraitDefinition(name="sex", trait_type=TraitType.PROBABILISTIC, categories=["F", "M"])
        trait_mgr = TraitManager(n_agents=3, definitions=[trait_def])
        from salmon_ibm.accumulators import AccumulatorDef, AccumulatorManager
        acc_mgr = AccumulatorManager(n_agents=3, definitions=[AccumulatorDef(name="dummy")])
        with pytest.raises(ValueError):
            trait_mgr.evaluate_accumulated("sex", acc_mgr)


class TestTraitFiltering:
    def _make_manager(self):
        defs = [
            TraitDefinition(name="sex", trait_type=TraitType.PROBABILISTIC, categories=["F", "M"]),
            TraitDefinition(name="stage", trait_type=TraitType.PROBABILISTIC, categories=["juv", "sub", "adult"]),
        ]
        mgr = TraitManager(n_agents=6, definitions=defs)
        mgr.set("sex",   np.array([0, 1, 0, 1, 0, 1], dtype=np.int32))
        mgr.set("stage", np.array([0, 0, 1, 1, 2, 2], dtype=np.int32))
        return mgr

    def test_filter_single_trait_single_value(self):
        mgr = self._make_manager()
        mask = mgr.filter_by_traits(sex=0)
        np.testing.assert_array_equal(mask, [True, False, True, False, True, False])

    def test_filter_single_trait_multiple_values(self):
        mgr = self._make_manager()
        mask = mgr.filter_by_traits(stage=[0, 2])
        np.testing.assert_array_equal(mask, [True, True, False, False, True, True])

    def test_filter_multiple_traits_and_logic(self):
        mgr = self._make_manager()
        mask = mgr.filter_by_traits(sex=1, stage=2)
        np.testing.assert_array_equal(mask, [False, False, False, False, False, True])

    def test_filter_no_criteria_returns_all_true(self):
        mgr = self._make_manager()
        mask = mgr.filter_by_traits()
        assert np.all(mask)
        assert mask.shape == (6,)

    def test_filter_specific_combo(self):
        mgr = self._make_manager()
        mask = mgr.filter_by_traits(sex=1, stage=0)
        np.testing.assert_array_equal(mask, [False, True, False, False, False, False])

    def test_filter_by_category_name(self):
        mgr = self._make_manager()
        mask = mgr.filter_by_traits(sex="M")
        np.testing.assert_array_equal(mask, [False, True, False, True, False, True])

    def test_filter_by_mixed_name_and_index(self):
        mgr = self._make_manager()
        mask = mgr.filter_by_traits(sex="F", stage=2)
        np.testing.assert_array_equal(mask, [False, False, False, False, True, False])
