"""Tests for HexSim expression DSL translator."""
import numpy as np
import pytest
from salmon_ibm.hexsim_expr import translate_hexsim_expr, build_hexsim_namespace


class TestTranslate:
    def test_single_quoted_global(self):
        assert '_g["Hexagon Area"]' in translate_hexsim_expr("'Hexagon Area'")

    def test_double_quoted_accumulator(self):
        assert '_a["Fitness [ weight ]"]' in translate_hexsim_expr('"Fitness [ weight ]"')

    def test_cond_renamed(self):
        result = translate_hexsim_expr("Cond(x, 1, 0)")
        assert "_cond" in result
        assert "Cond" not in result

    def test_mixed_quotes(self):
        expr = "'Hexagon Area' * \"Refuge Depth [ chinook ]\""
        result = translate_hexsim_expr(expr)
        assert '_g["Hexagon Area"]' in result
        assert '_a["Refuge Depth [ chinook ]"]' in result

    def test_nested_cond(self):
        expr = 'Cond("a", Cond("b" - 5, 1, 0), 1)'
        result = translate_hexsim_expr(expr)
        assert result.count("_cond") == 2

    def test_functions_not_translated(self):
        """Floor, Pow, Exp stay as-is — they're injected into namespace."""
        expr = "Floor(Pow(\"x\", 2))"
        result = translate_hexsim_expr(expr)
        assert "Floor" in result
        assert "Pow" in result


class TestBuildNamespace:
    def test_contains_functions(self):
        ns = build_hexsim_namespace({}, {}, np.random.default_rng(), 10)
        assert callable(ns["_cond"])
        assert callable(ns["Floor"])
        assert callable(ns["Rand"])
        assert callable(ns["GasDev"])

    def test_cond_semantics(self):
        ns = build_hexsim_namespace({}, {}, np.random.default_rng(), 3)
        result = ns["_cond"](np.array([1.0, 0.0, -1.0]), 10, 20)
        np.testing.assert_array_equal(result, [10, 20, 20])

    def test_rand_returns_correct_size(self):
        ns = build_hexsim_namespace({}, {}, np.random.default_rng(42), 5)
        result = ns["Rand"]()
        assert result.shape == (5,)

    def test_gasdev_returns_correct_size(self):
        ns = build_hexsim_namespace({}, {}, np.random.default_rng(42), 5)
        result = ns["GasDev"]()
        assert result.shape == (5,)


class TestEndToEnd:
    def test_evaluate_simple_expression(self):
        from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_expression
        defs = [AccumulatorDef("x"), AccumulatorDef("y"), AccumulatorDef("result")]
        mgr = AccumulatorManager(3, defs)
        mgr.data[:, 0] = [1.0, 2.0, 3.0]  # x
        mgr.data[:, 1] = [10.0, 20.0, 30.0]  # y
        mask = np.ones(3, dtype=bool)
        updater_expression(mgr, "result", mask,
                          expression='"x" + "y"',
                          globals_dict={"unused": 0})
        np.testing.assert_array_equal(mgr.data[:, 2], [11.0, 22.0, 33.0])

    def test_evaluate_with_global(self):
        from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_expression
        defs = [AccumulatorDef("weight"), AccumulatorDef("result")]
        mgr = AccumulatorManager(2, defs)
        mgr.data[:, 0] = [100.0, 200.0]
        mask = np.ones(2, dtype=bool)
        updater_expression(mgr, "result", mask,
                          expression="'scale' * \"weight\"",
                          globals_dict={"scale": 2.0})
        np.testing.assert_array_equal(mgr.data[:, 1], [200.0, 400.0])

    def test_evaluate_cond(self):
        from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_expression
        defs = [AccumulatorDef("val"), AccumulatorDef("result")]
        mgr = AccumulatorManager(3, defs)
        mgr.data[:, 0] = [5.0, -1.0, 0.0]  # positive, negative, zero
        mask = np.ones(3, dtype=bool)
        updater_expression(mgr, "result", mask,
                          expression='Cond("val", 1, 0)',
                          globals_dict={})
        np.testing.assert_array_equal(mgr.data[:, 1], [1.0, 0.0, 0.0])

    def test_evaluate_nested_cond(self):
        from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_expression
        defs = [AccumulatorDef("a"), AccumulatorDef("result")]
        mgr = AccumulatorManager(2, defs)
        mgr.data[:, 0] = [10.0, 3.0]
        mask = np.ones(2, dtype=bool)
        # Cond(a - 5, Cond(a - 8, 100, 50), 0)
        # a=10: 10-5>0 → Cond(10-8>0 → 100)
        # a=3: 3-5<0 → 0
        updater_expression(mgr, "result", mask,
                          expression='Cond("a" - 5, Cond("a" - 8, 100, 50), 0)',
                          globals_dict={})
        np.testing.assert_array_equal(mgr.data[:, 1], [100.0, 0.0])

    def test_legacy_mode_still_works(self):
        """Without globals_dict, existing bare-name expressions still work."""
        from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_expression
        defs = [AccumulatorDef("x"), AccumulatorDef("result")]
        mgr = AccumulatorManager(2, defs)
        mgr.data[:, 0] = [3.0, 4.0]
        mask = np.ones(2, dtype=bool)
        updater_expression(mgr, "result", mask, expression="x + 1")
        np.testing.assert_array_equal(mgr.data[:, 1], [4.0, 5.0])
