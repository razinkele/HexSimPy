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
        defs = [AccumulatorDef(name="a"), AccumulatorDef(name="b"), AccumulatorDef(name="c")]
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
