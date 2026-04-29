# Phase 1a: Accumulator and Trait System — Implementation Plan

> **STATUS: ✅ EXECUTED** — `salmon_ibm/accumulators.py` (AccumulatorManager + 25 updaters) and `salmon_ibm/traits.py` (TraitDefinition/TraitManager) shipped. Tests in `tests/test_accumulators.py`, `tests/test_traits.py`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a general-purpose accumulator system (8 updater functions) and trait system (probabilistic + accumulated types) to the agent pool, enabling configurable agent state beyond the current hardcoded salmon fields.

**Architecture:** The accumulator system stores per-agent floating-point state in a 2D NumPy array `[n_agents, n_accumulators]`, modified by pluggable updater functions. The trait system stores per-agent categorical indices as `int[n_agents]` per trait, with accumulated traits auto-reevaluating when their linked accumulator changes. Both systems integrate into `AgentPool` as optional managers, preserving backward compatibility with the existing salmon model.

**Tech Stack:** NumPy, dataclasses

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `salmon_ibm/accumulators.py` | **Create** | AccumulatorManager, AccumulatorDef, 8 updater functions |
| `salmon_ibm/traits.py` | **Create** | TraitDefinition, TraitManager, trait evaluation and filtering |
| `salmon_ibm/agents.py` | **Modify** | Add optional `accumulators` and `traits` attributes to AgentPool |
| `tests/test_accumulators.py` | **Create** | Accumulator unit tests (storage, bounds, all 8 updaters) |
| `tests/test_traits.py` | **Create** | Trait unit tests (probabilistic, accumulated, filtering) |

---

## Test Command

All tasks use this command to run tests:

```bash
conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

To run only the new tests:

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py tests/test_traits.py -v
```

---

## Chunk 1: AccumulatorManager — Storage, Get/Set, Bounds

### Task 1: AccumulatorDef dataclass and AccumulatorManager core

Create the accumulator storage layer. Each accumulator has a name, optional min/max bounds, and an optional linked trait name. The manager holds a 2D NumPy array and provides get/set with automatic bounds clamping.

**Files:**
- Create: `salmon_ibm/accumulators.py`
- Create: `tests/test_accumulators.py`

- [ ] **Step 1: Write failing tests for AccumulatorManager basics**

Add to `tests/test_accumulators.py`:

```python
import numpy as np
import pytest
from salmon_ibm.accumulators import AccumulatorDef, AccumulatorManager


class TestAccumulatorManager:
    def test_create_manager_with_definitions(self):
        """Manager initializes storage from a list of AccumulatorDefs."""
        defs = [
            AccumulatorDef(name="energy", min_val=0.0, max_val=100.0),
            AccumulatorDef(name="age"),
        ]
        mgr = AccumulatorManager(n_agents=10, definitions=defs)
        assert mgr.data.shape == (10, 2)
        assert mgr.data.dtype == np.float64
        assert np.all(mgr.data == 0.0)

    def test_get_set_by_name(self):
        """Can get and set accumulator columns by name."""
        defs = [AccumulatorDef(name="energy"), AccumulatorDef(name="age")]
        mgr = AccumulatorManager(n_agents=5, definitions=defs)
        mgr.set("energy", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = mgr.get("energy")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_get_set_by_index(self):
        """Can get and set accumulator columns by integer index."""
        defs = [AccumulatorDef(name="energy")]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        mgr.set(0, np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(mgr.get(0), [10.0, 20.0, 30.0])

    def test_bounds_clamping(self):
        """Values are clamped to [min_val, max_val] on set."""
        defs = [AccumulatorDef(name="energy", min_val=0.0, max_val=100.0)]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        mgr.set("energy", np.array([-5.0, 50.0, 150.0]))
        np.testing.assert_array_equal(mgr.get("energy"), [0.0, 50.0, 100.0])

    def test_unknown_name_raises(self):
        """Accessing a non-existent accumulator name raises KeyError."""
        defs = [AccumulatorDef(name="energy")]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        with pytest.raises(KeyError):
            mgr.get("nonexistent")

    def test_index_lookup(self):
        """index_of returns the column index for a named accumulator."""
        defs = [AccumulatorDef(name="a"), AccumulatorDef(name="b"), AccumulatorDef(name="c")]
        mgr = AccumulatorManager(n_agents=2, definitions=defs)
        assert mgr.index_of("a") == 0
        assert mgr.index_of("b") == 1
        assert mgr.index_of("c") == 2

    def test_masked_set(self):
        """Can set values for a subset of agents using a boolean mask."""
        defs = [AccumulatorDef(name="energy", min_val=0.0)]
        mgr = AccumulatorManager(n_agents=4, definitions=defs)
        mask = np.array([True, False, True, False])
        mgr.set("energy", np.array([99.0, 99.0]), mask=mask)
        np.testing.assert_array_equal(mgr.get("energy"), [99.0, 0.0, 99.0, 0.0])
```

- [ ] **Step 2: Run tests — verify they fail (ImportError)**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py -v
```

Expected: `ImportError` — module does not exist yet.

- [ ] **Step 3: Implement AccumulatorDef and AccumulatorManager**

Create `salmon_ibm/accumulators.py`:

```python
"""Accumulator system: general-purpose per-agent floating-point state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union
import numpy as np


@dataclass
class AccumulatorDef:
    """Definition for a single accumulator column."""
    name: str
    min_val: float | None = None
    max_val: float | None = None
    linked_trait: str | None = None  # trait name to reevaluate on change


class AccumulatorManager:
    """Vectorized storage and manipulation of per-agent accumulators.

    Storage: 2D NumPy array of shape (n_agents, n_accumulators).
    Each column corresponds to one AccumulatorDef.
    """

    def __init__(self, n_agents: int, definitions: list[AccumulatorDef]):
        self.n_agents = n_agents
        self.definitions = list(definitions)
        self._name_to_idx: dict[str, int] = {
            d.name: i for i, d in enumerate(definitions)
        }
        n_acc = len(definitions)
        self.data = np.zeros((n_agents, n_acc), dtype=np.float64)

    def index_of(self, name: str) -> int:
        return self._name_to_idx[name]

    def _resolve_idx(self, key: Union[str, int]) -> int:
        if isinstance(key, str):
            return self._name_to_idx[key]  # raises KeyError if missing
        return key

    def get(self, key: Union[str, int]) -> np.ndarray:
        idx = self._resolve_idx(key)
        return self.data[:, idx]

    def set(
        self,
        key: Union[str, int],
        values: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        idx = self._resolve_idx(key)
        defn = self.definitions[idx]
        # Clamp to bounds
        clamped = values
        if defn.min_val is not None:
            clamped = np.maximum(clamped, defn.min_val)
        if defn.max_val is not None:
            clamped = np.minimum(clamped, defn.max_val)
        if mask is not None:
            self.data[mask, idx] = clamped
        else:
            self.data[:, idx] = clamped
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py::TestAccumulatorManager -v
```

Expected: all 7 tests pass.

**Commit message:** `feat(accumulators): add AccumulatorDef and AccumulatorManager with bounds clamping`

---

## Chunk 2: Updaters — Clear, Increment, Stochastic Increment

### Task 2: Implement the 3 simplest updater functions

Updaters are standalone functions that modify one accumulator column for a subset of agents. They follow a common signature: `(manager, acc_name, mask, **params) -> None`. They mutate the manager in place.

**Files:**
- Modify: `salmon_ibm/accumulators.py`
- Modify: `tests/test_accumulators.py`

- [ ] **Step 1: Write failing tests for Clear, Increment, Stochastic Increment**

Add to `tests/test_accumulators.py`:

```python
from salmon_ibm.accumulators import updater_clear, updater_increment, updater_stochastic_increment


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
        np.testing.assert_array_equal(mgr.get("energy"), [80.0, 90.0, 100.0, 100.0, 100.0])

    def test_stochastic_increment_adds_uniform_random(self):
        mgr = self._make_manager()
        mask = np.ones(5, dtype=bool)
        updater_stochastic_increment(mgr, "energy", mask, low=1.0, high=2.0, rng=np.random.default_rng(42))
        result = mgr.get("energy")
        # All values should have increased by [1.0, 2.0)
        original = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        increments = result - original
        assert np.all(increments >= 1.0)
        assert np.all(increments < 2.0)

    def test_stochastic_increment_respects_mask(self):
        mgr = self._make_manager()
        mask = np.array([True, False, False, False, True])
        updater_stochastic_increment(mgr, "energy", mask, low=5.0, high=6.0, rng=np.random.default_rng(0))
        result = mgr.get("energy")
        # Unmasked agents unchanged
        assert result[1] == 20.0
        assert result[2] == 30.0
        assert result[3] == 40.0
        # Masked agents increased
        assert result[0] > 10.0
        assert result[4] > 50.0
```

- [ ] **Step 2: Run tests — verify they fail (ImportError)**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py::TestSimpleUpdaters -v
```

- [ ] **Step 3: Implement updater_clear, updater_increment, updater_stochastic_increment**

Add to `salmon_ibm/accumulators.py`:

```python
def updater_clear(
    manager: AccumulatorManager, acc_name: str, mask: np.ndarray
) -> None:
    """Reset accumulator to zero (or min_val if > 0) for masked agents."""
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]
    val = 0.0
    if defn.min_val is not None:
        val = max(val, defn.min_val)
    manager.data[mask, idx] = val


def updater_increment(
    manager: AccumulatorManager, acc_name: str, mask: np.ndarray,
    *, amount: float,
) -> None:
    """Add a fixed quantity to accumulator for masked agents."""
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]
    new_vals = manager.data[mask, idx] + amount
    if defn.min_val is not None:
        new_vals = np.maximum(new_vals, defn.min_val)
    if defn.max_val is not None:
        new_vals = np.minimum(new_vals, defn.max_val)
    manager.data[mask, idx] = new_vals


def updater_stochastic_increment(
    manager: AccumulatorManager, acc_name: str, mask: np.ndarray,
    *, low: float, high: float, rng: np.random.Generator,
) -> None:
    """Add uniform random quantity in [low, high) to accumulator for masked agents."""
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]
    n_masked = mask.sum()
    increments = rng.uniform(low, high, size=n_masked)
    new_vals = manager.data[mask, idx] + increments
    if defn.min_val is not None:
        new_vals = np.maximum(new_vals, defn.min_val)
    if defn.max_val is not None:
        new_vals = np.minimum(new_vals, defn.max_val)
    manager.data[mask, idx] = new_vals
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py::TestSimpleUpdaters -v
```

Expected: all 6 tests pass.

**Commit message:** `feat(accumulators): add Clear, Increment, and Stochastic Increment updaters`

---

## Chunk 3: Updater — Accumulator Expression

### Task 3: Safe algebraic evaluation over accumulators

The Accumulator Expression updater evaluates a mathematical expression where variables are accumulator names. For example, `"energy * 0.5 + age"` reads accumulators `energy` and `age`, computes the expression element-wise, and writes the result to a target accumulator. Must be safe (no arbitrary code execution).

**Files:**
- Modify: `salmon_ibm/accumulators.py`
- Modify: `tests/test_accumulators.py`

- [ ] **Step 1: Write failing tests for Accumulator Expression**

Add to `tests/test_accumulators.py`:

```python
from salmon_ibm.accumulators import updater_expression


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
        """Expressions can use numpy math functions like sqrt, abs, exp, log."""
        mgr = self._make_manager()
        mask = np.ones(4, dtype=bool)
        updater_expression(mgr, "result", mask, expression="sqrt(energy)")
        expected = np.sqrt(np.array([10.0, 20.0, 30.0, 40.0]))
        np.testing.assert_allclose(mgr.get("result"), expected)

    def test_rejects_dangerous_expressions(self):
        """Expressions containing dangerous constructs are rejected."""
        mgr = self._make_manager()
        mask = np.ones(4, dtype=bool)
        with pytest.raises(ValueError):
            updater_expression(mgr, "result", mask, expression="__import__('os').system('rm -rf /')")

    def test_rejects_unknown_variable(self):
        """Expressions referencing non-existent accumulators raise an error."""
        mgr = self._make_manager()
        mask = np.ones(4, dtype=bool)
        with pytest.raises((KeyError, NameError)):
            updater_expression(mgr, "result", mask, expression="nonexistent + 1")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py::TestExpressionUpdater -v
```

- [ ] **Step 3: Implement updater_expression with safe evaluation**

Add to `salmon_ibm/accumulators.py`. Use a whitelist approach: build a namespace dict containing only accumulator arrays and safe NumPy math functions, then evaluate with Python's `eval()` against that restricted namespace.

```python
import re

# Safe math functions available in expressions
_SAFE_MATH = {
    "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp, "log": np.log,
    "log10": np.log10, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "minimum": np.minimum, "maximum": np.maximum,
    "clip": np.clip, "where": np.where,
    "pi": np.pi, "e": np.e,
}

import ast

# AST whitelist for safe expression evaluation (defense in depth)
_ALLOWED_NODE_TYPES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant,
    ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.Mod, ast.USub, ast.UAdd,
)


def _validate_expression(expr: str) -> None:
    """Validate expression AST contains only safe constructs."""
    tree = ast.parse(expr, mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(f"Disallowed construct in expression: {type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_MATH:
                raise ValueError(f"Unknown function in expression: {node.func.id}")


def updater_expression(
    manager: AccumulatorManager, acc_name: str, mask: np.ndarray,
    *, expression: str,
) -> None:
    """Evaluate algebraic expression over accumulators, write result to acc_name.

    Variables in the expression are accumulator names. Safe NumPy math
    functions (sqrt, abs, exp, log, etc.) are available.
    Uses AST whitelist validation for safety — only arithmetic, constants,
    accumulator names, and whitelisted math functions are permitted.
    """
    _validate_expression(expression)

    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]

    # Build namespace: accumulator arrays (masked subset) + safe math
    namespace = dict(_SAFE_MATH)
    for name, col_idx in manager._name_to_idx.items():
        namespace[name] = manager.data[mask, col_idx]

    result = eval(expression, {"__builtins__": {}}, namespace)  # noqa: S307
    result = np.asarray(result, dtype=np.float64)

    # Handle NaN/inf from division by zero etc.
    result = np.nan_to_num(result, nan=0.0,
                           posinf=defn.max_val if defn.max_val is not None else 0.0,
                           neginf=defn.min_val if defn.min_val is not None else 0.0)

    if defn.min_val is not None:
        result = np.maximum(result, defn.min_val)
    if defn.max_val is not None:
        result = np.minimum(result, defn.max_val)
    manager.data[mask, idx] = result
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py::TestExpressionUpdater -v
```

Expected: all 6 tests pass.

**Commit message:** `feat(accumulators): add Accumulator Expression updater with safe evaluation`

---

## Chunk 4: Updaters — Time Step, Individual ID, Stochastic Trigger, Quantify Location

### Task 4: Implement the remaining 4 priority updaters

These updaters read external state (timestep, agent IDs, spatial maps) and write values into accumulators.

**Files:**
- Modify: `salmon_ibm/accumulators.py`
- Modify: `tests/test_accumulators.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_accumulators.py`:

```python
from salmon_ibm.accumulators import (
    updater_time_step, updater_individual_id,
    updater_stochastic_trigger, updater_quantify_location,
)


class TestTimestepUpdater:
    def test_writes_current_timestep(self):
        defs = [AccumulatorDef(name="step")]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        mask = np.ones(3, dtype=bool)
        updater_time_step(mgr, "step", mask, timestep=42)
        np.testing.assert_array_equal(mgr.get("step"), [42.0, 42.0, 42.0])

    def test_modulus(self):
        """With modulus, writes timestep % modulus."""
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
        np.testing.assert_array_equal(mgr.get("id"), [100.0, 101.0, 102.0, 103.0, 104.0])

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
        updater_stochastic_trigger(mgr, "trigger", mask, probability=0.5, rng=np.random.default_rng(42))
        vals = mgr.get("trigger")
        assert set(np.unique(vals)).issubset({0.0, 1.0})

    def test_probability_respected_statistically(self):
        """With p=0.3, roughly 30% of agents should get 1."""
        defs = [AccumulatorDef(name="trigger")]
        mgr = AccumulatorManager(n_agents=10000, definitions=defs)
        mask = np.ones(10000, dtype=bool)
        updater_stochastic_trigger(mgr, "trigger", mask, probability=0.3, rng=np.random.default_rng(123))
        frac = mgr.get("trigger").mean()
        assert 0.27 < frac < 0.33, f"Expected ~0.3, got {frac}"

    def test_probability_zero_all_zeros(self):
        defs = [AccumulatorDef(name="trigger")]
        mgr = AccumulatorManager(n_agents=100, definitions=defs)
        mask = np.ones(100, dtype=bool)
        updater_stochastic_trigger(mgr, "trigger", mask, probability=0.0, rng=np.random.default_rng(0))
        assert np.all(mgr.get("trigger") == 0.0)

    def test_probability_one_all_ones(self):
        defs = [AccumulatorDef(name="trigger")]
        mgr = AccumulatorManager(n_agents=100, definitions=defs)
        mask = np.ones(100, dtype=bool)
        updater_stochastic_trigger(mgr, "trigger", mask, probability=1.0, rng=np.random.default_rng(0))
        assert np.all(mgr.get("trigger") == 1.0)


class TestQuantifyLocationUpdater:
    def test_samples_hexmap_at_agent_positions(self):
        defs = [AccumulatorDef(name="temperature")]
        mgr = AccumulatorManager(n_agents=4, definitions=defs)
        mask = np.ones(4, dtype=bool)
        # Simulated hex map: 10 cells with temperature values
        hex_map = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
        agent_cells = np.array([0, 3, 7, 9])  # agent positions (cell indices)
        updater_quantify_location(mgr, "temperature", mask, hex_map=hex_map, cell_indices=agent_cells)
        np.testing.assert_array_equal(mgr.get("temperature"), [15.0, 18.0, 22.0, 24.0])

    def test_respects_mask(self):
        defs = [AccumulatorDef(name="depth")]
        mgr = AccumulatorManager(n_agents=3, definitions=defs)
        mask = np.array([True, False, True])
        hex_map = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        agent_cells = np.array([0, 2, 4])
        updater_quantify_location(mgr, "depth", mask, hex_map=hex_map, cell_indices=agent_cells)
        np.testing.assert_array_equal(mgr.get("depth"), [5.0, 0.0, 25.0])
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py::TestTimestepUpdater tests/test_accumulators.py::TestIndividualIDUpdater tests/test_accumulators.py::TestStochasticTriggerUpdater tests/test_accumulators.py::TestQuantifyLocationUpdater -v
```

- [ ] **Step 3: Implement all 4 updaters**

Add to `salmon_ibm/accumulators.py`:

```python
def updater_time_step(
    manager: AccumulatorManager, acc_name: str, mask: np.ndarray,
    *, timestep: int, modulus: int | None = None,
) -> None:
    """Write current timestep (optionally with modulus) to accumulator."""
    idx = manager._resolve_idx(acc_name)
    value = float(timestep % modulus) if modulus is not None and modulus > 0 else float(timestep)
    manager.data[mask, idx] = value


def updater_individual_id(
    manager: AccumulatorManager, acc_name: str, mask: np.ndarray,
    *, agent_ids: np.ndarray,
) -> None:
    """Write each agent's unique ID to accumulator."""
    idx = manager._resolve_idx(acc_name)
    manager.data[mask, idx] = agent_ids[mask].astype(np.float64)


def updater_stochastic_trigger(
    manager: AccumulatorManager, acc_name: str, mask: np.ndarray,
    *, probability: float, rng: np.random.Generator,
) -> None:
    """Write 1.0 with probability p, else 0.0, for each masked agent."""
    idx = manager._resolve_idx(acc_name)
    n_masked = mask.sum()
    triggers = (rng.random(n_masked) < probability).astype(np.float64)
    manager.data[mask, idx] = triggers


def updater_quantify_location(
    manager: AccumulatorManager, acc_name: str, mask: np.ndarray,
    *, hex_map: np.ndarray, cell_indices: np.ndarray,
) -> None:
    """Sample hex-map values at each agent's current cell position."""
    idx = manager._resolve_idx(acc_name)
    manager.data[mask, idx] = hex_map[cell_indices[mask]]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py -v
```

Expected: all accumulator tests pass (7 + 6 + 6 + 12 = 31 tests).

**Commit message:** `feat(accumulators): add TimeStep, IndividualID, StochasticTrigger, QuantifyLocation updaters`

---

## Chunk 5: TraitDefinition and TraitManager

### Task 5: Trait storage, evaluation, and category assignment

A trait assigns each agent a categorical index (integer). `TraitDefinition` describes the trait (name, type, categories). `TraitManager` stores all traits and provides evaluation methods.

**Files:**
- Create: `salmon_ibm/traits.py`
- Create: `tests/test_traits.py`

- [ ] **Step 1: Write failing tests for TraitManager basics**

Create `tests/test_traits.py`:

```python
import numpy as np
import pytest
from salmon_ibm.traits import TraitType, TraitDefinition, TraitManager


class TestTraitManager:
    def test_create_manager(self):
        td = TraitDefinition(
            name="life_stage",
            trait_type=TraitType.PROBABILISTIC,
            categories=["juvenile", "subadult", "adult"],
        )
        mgr = TraitManager(n_agents=10, definitions=[td])
        assert mgr.get("life_stage").shape == (10,)
        assert mgr.get("life_stage").dtype == np.int32
        # All agents start at category 0 by default
        assert np.all(mgr.get("life_stage") == 0)

    def test_set_trait_values(self):
        td = TraitDefinition(
            name="sex", trait_type=TraitType.PROBABILISTIC,
            categories=["female", "male"],
        )
        mgr = TraitManager(n_agents=4, definitions=[td])
        mgr.set("sex", np.array([0, 1, 1, 0], dtype=np.int32))
        np.testing.assert_array_equal(mgr.get("sex"), [0, 1, 1, 0])

    def test_set_with_mask(self):
        td = TraitDefinition(
            name="status", trait_type=TraitType.PROBABILISTIC,
            categories=["healthy", "sick"],
        )
        mgr = TraitManager(n_agents=4, definitions=[td])
        mask = np.array([False, True, True, False])
        mgr.set("status", np.array([1, 1], dtype=np.int32), mask=mask)
        np.testing.assert_array_equal(mgr.get("status"), [0, 1, 1, 0])

    def test_category_name_lookup(self):
        td = TraitDefinition(
            name="life_stage", trait_type=TraitType.PROBABILISTIC,
            categories=["egg", "fry", "smolt", "adult"],
        )
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
```

- [ ] **Step 2: Run tests — verify they fail (ImportError)**

```bash
conda run -n shiny python -m pytest tests/test_traits.py::TestTraitManager -v
```

- [ ] **Step 3: Implement TraitType, TraitDefinition, TraitManager**

Create `salmon_ibm/traits.py`:

```python
"""Trait system: categorical per-agent state with auto-evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union
import numpy as np


class TraitType(Enum):
    PROBABILISTIC = "probabilistic"
    ACCUMULATED = "accumulated"


@dataclass
class TraitDefinition:
    """Definition for a single trait."""
    name: str
    trait_type: TraitType
    categories: list[str]
    # For ACCUMULATED type: linked accumulator name and thresholds
    accumulator_name: str | None = None
    thresholds: np.ndarray | None = None  # ascending; len = len(categories) - 1


class TraitManager:
    """Vectorized storage and evaluation of per-agent categorical traits."""

    def __init__(self, n_agents: int, definitions: list[TraitDefinition]):
        self.n_agents = n_agents
        self.definitions: dict[str, TraitDefinition] = {
            d.name: d for d in definitions
        }
        self._data: dict[str, np.ndarray] = {
            d.name: np.zeros(n_agents, dtype=np.int32) for d in definitions
        }

    def get(self, name: str) -> np.ndarray:
        if name not in self._data:
            raise KeyError(f"Unknown trait: {name!r}")
        return self._data[name]

    def set(
        self, name: str, values: np.ndarray, mask: np.ndarray | None = None,
    ) -> None:
        if name not in self._data:
            raise KeyError(f"Unknown trait: {name!r}")
        if mask is not None:
            self._data[name][mask] = values
        else:
            self._data[name][:] = values

    def category_names(self, name: str) -> list[str]:
        """Return the category name for each agent's current trait value."""
        defn = self.definitions[name]
        indices = self._data[name]
        return [defn.categories[i] for i in indices]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_traits.py::TestTraitManager -v
```

Expected: all 6 tests pass.

**Commit message:** `feat(traits): add TraitDefinition and TraitManager with categorical storage`

---

## Chunk 6: Accumulated Trait Auto-Evaluation

### Task 6: Reevaluate accumulated traits when linked accumulator changes

An accumulated trait maps a continuous accumulator value to a discrete category using threshold bins. For example, if accumulator `"energy"` has thresholds `[20.0, 50.0, 80.0]`, then categories are assigned as:
- energy < 20 -> category 0
- 20 <= energy < 50 -> category 1
- 50 <= energy < 80 -> category 2
- energy >= 80 -> category 3

The `TraitManager.evaluate_accumulated()` method performs this mapping. It should be called after any updater modifies a linked accumulator.

**Files:**
- Modify: `salmon_ibm/traits.py`
- Modify: `tests/test_traits.py`

- [ ] **Step 1: Write failing tests for accumulated trait evaluation**

Add to `tests/test_traits.py`:

```python
from salmon_ibm.accumulators import AccumulatorDef, AccumulatorManager


class TestAccumulatedTrait:
    def _make_system(self):
        """Create linked accumulator + accumulated trait."""
        acc_defs = [AccumulatorDef(name="energy", linked_trait="condition")]
        acc_mgr = AccumulatorManager(n_agents=6, definitions=acc_defs)

        trait_def = TraitDefinition(
            name="condition",
            trait_type=TraitType.ACCUMULATED,
            categories=["critical", "poor", "fair", "good"],
            accumulator_name="energy",
            thresholds=np.array([20.0, 50.0, 80.0]),
        )
        trait_mgr = TraitManager(n_agents=6, definitions=[trait_def])
        return acc_mgr, trait_mgr

    def test_evaluate_accumulated_basic(self):
        acc_mgr, trait_mgr = self._make_system()
        acc_mgr.set("energy", np.array([5.0, 20.0, 35.0, 50.0, 80.0, 95.0]))
        trait_mgr.evaluate_accumulated("condition", acc_mgr)
        expected = np.array([0, 1, 1, 2, 3, 3], dtype=np.int32)
        np.testing.assert_array_equal(trait_mgr.get("condition"), expected)

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
        # Masked agents updated, unmasked agents stay at default (0)
        expected = np.array([0, 3, 0, 0, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(trait_mgr.get("condition"), expected)

    def test_raises_for_probabilistic_trait(self):
        """evaluate_accumulated should raise if trait is not ACCUMULATED type."""
        trait_def = TraitDefinition(
            name="sex", trait_type=TraitType.PROBABILISTIC,
            categories=["F", "M"],
        )
        trait_mgr = TraitManager(n_agents=3, definitions=[trait_def])
        acc_defs = [AccumulatorDef(name="dummy")]
        acc_mgr = AccumulatorManager(n_agents=3, definitions=acc_defs)
        with pytest.raises(ValueError):
            trait_mgr.evaluate_accumulated("sex", acc_mgr)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
conda run -n shiny python -m pytest tests/test_traits.py::TestAccumulatedTrait -v
```

- [ ] **Step 3: Implement evaluate_accumulated on TraitManager**

Add to `TraitManager` in `salmon_ibm/traits.py`:

```python
def evaluate_accumulated(
    self, name: str, acc_manager, mask: np.ndarray | None = None,
) -> None:
    """Re-evaluate an accumulated trait by binning its linked accumulator.

    Uses np.digitize with the trait's thresholds to assign category indices.
    """
    defn = self.definitions[name]
    if defn.trait_type != TraitType.ACCUMULATED:
        raise ValueError(
            f"Trait {name!r} is {defn.trait_type.value}, not accumulated"
        )
    acc_values = acc_manager.get(defn.accumulator_name)
    categories = np.digitize(acc_values, defn.thresholds).astype(np.int32)
    if mask is not None:
        self._data[name][mask] = categories[mask]
    else:
        self._data[name][:] = categories
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_traits.py::TestAccumulatedTrait -v
```

Expected: all 5 tests pass.

**Commit message:** `feat(traits): add accumulated trait auto-evaluation via threshold binning`

---

## Chunk 7: Trait Filtering (Stratification)

### Task 7: Boolean mask generation from trait criteria

HexSim events apply to subsets of agents filtered by trait combinations. The `filter_by_traits` method takes a dict of `{trait_name: category_index_or_list}` and returns a boolean mask of agents matching ALL criteria (AND logic).

**Files:**
- Modify: `salmon_ibm/traits.py`
- Modify: `tests/test_traits.py`

- [ ] **Step 1: Write failing tests for trait filtering**

Add to `tests/test_traits.py`:

```python
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
        """Filter by multiple categories of the same trait (OR within trait)."""
        mgr = self._make_manager()
        mask = mgr.filter_by_traits(stage=[0, 2])
        np.testing.assert_array_equal(mask, [True, True, False, False, True, True])

    def test_filter_multiple_traits_and_logic(self):
        """Multiple traits use AND logic."""
        mgr = self._make_manager()
        mask = mgr.filter_by_traits(sex=1, stage=2)
        np.testing.assert_array_equal(mask, [False, False, False, False, False, True])

    def test_filter_no_criteria_returns_all_true(self):
        mgr = self._make_manager()
        mask = mgr.filter_by_traits()
        assert np.all(mask)
        assert mask.shape == (6,)

    def test_filter_no_match_returns_all_false(self):
        mgr = self._make_manager()
        # No male juveniles exist? Actually agent 1 is male juvenile.
        # Use a combo that doesn't exist: female adult = agent 4
        mask = mgr.filter_by_traits(sex=1, stage=0)
        # Agent 1 is male (1) and juvenile (0) -> True
        np.testing.assert_array_equal(mask, [False, True, False, False, False, False])

    def test_filter_by_category_name(self):
        """Can filter using category name strings instead of indices."""
        mgr = self._make_manager()
        mask = mgr.filter_by_traits(sex="M")
        np.testing.assert_array_equal(mask, [False, True, False, True, False, True])

    def test_filter_by_mixed_name_and_index(self):
        mgr = self._make_manager()
        mask = mgr.filter_by_traits(sex="F", stage=2)
        np.testing.assert_array_equal(mask, [False, False, False, False, True, False])
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
conda run -n shiny python -m pytest tests/test_traits.py::TestTraitFiltering -v
```

- [ ] **Step 3: Implement filter_by_traits on TraitManager**

Add to `TraitManager` in `salmon_ibm/traits.py`:

```python
def _resolve_category(self, trait_name: str, value) -> list[int]:
    """Convert a category value (int, str, or list) to list of int indices."""
    defn = self.definitions[trait_name]
    if isinstance(value, (list, tuple)):
        return [self._resolve_single_category(trait_name, v, defn) for v in value]
    return [self._resolve_single_category(trait_name, value, defn)]

def _resolve_single_category(self, trait_name: str, value, defn: TraitDefinition) -> int:
    if isinstance(value, str):
        return defn.categories.index(value)
    return int(value)

def filter_by_traits(self, **criteria) -> np.ndarray:
    """Return boolean mask for agents matching all trait criteria (AND logic).

    Each kwarg is trait_name=value where value is:
    - int: category index
    - str: category name
    - list[int|str]: any of these categories (OR within trait)

    Multiple traits use AND logic across traits.
    """
    mask = np.ones(self.n_agents, dtype=bool)
    for trait_name, value in criteria.items():
        indices = self._resolve_category(trait_name, value)
        trait_vals = self._data[trait_name]
        trait_mask = np.zeros(self.n_agents, dtype=bool)
        for idx in indices:
            trait_mask |= (trait_vals == idx)
        mask &= trait_mask
    return mask
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_traits.py::TestTraitFiltering -v
```

Expected: all 7 tests pass.

**Commit message:** `feat(traits): add filter_by_traits for stratified agent selection`

---

## Chunk 8: Integration with AgentPool

### Task 8: Add optional accumulator and trait managers to AgentPool

Add optional `accumulators` and `traits` attributes to `AgentPool`. These are `None` by default (backward compatible). When present, the pool provides convenience methods for common operations.

**Files:**
- Modify: `salmon_ibm/agents.py`
- Modify: `tests/test_accumulators.py` (integration tests)

- [ ] **Step 1: Write failing tests for AgentPool integration**

Add to `tests/test_accumulators.py`:

```python
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
        np.testing.assert_array_equal(pool.accumulators.get("energy"), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_pool_with_traits(self):
        pool = AgentPool(n=4, start_tri=0)
        td = TraitDefinition(name="stage", trait_type=TraitType.PROBABILISTIC, categories=["juv", "adult"])
        pool.traits = TraitManager(n_agents=4, definitions=[td])
        pool.traits.set("stage", np.array([0, 1, 1, 0], dtype=np.int32))
        np.testing.assert_array_equal(pool.traits.get("stage"), [0, 1, 1, 0])

    def test_updater_with_pool_cell_indices(self):
        """Quantify Location can read cell indices from pool.tri_idx."""
        pool = AgentPool(n=3, start_tri=np.array([0, 5, 9]))
        defs = [AccumulatorDef(name="depth")]
        pool.accumulators = AccumulatorManager(n_agents=3, definitions=defs)
        hex_map = np.arange(10, dtype=np.float64) * 10.0  # 0, 10, 20, ..., 90
        mask = np.ones(3, dtype=bool)
        updater_quantify_location(pool.accumulators, "depth", mask, hex_map=hex_map, cell_indices=pool.tri_idx)
        np.testing.assert_array_equal(pool.accumulators.get("depth"), [0.0, 50.0, 90.0])

    def test_accumulated_trait_with_pool(self):
        """End-to-end: update accumulator -> reevaluate accumulated trait."""
        pool = AgentPool(n=4, start_tri=0)
        # Set up accumulators
        acc_defs = [AccumulatorDef(name="energy", linked_trait="condition")]
        pool.accumulators = AccumulatorManager(n_agents=4, definitions=acc_defs)
        # Set up traits
        trait_def = TraitDefinition(
            name="condition", trait_type=TraitType.ACCUMULATED,
            categories=["low", "medium", "high"],
            accumulator_name="energy",
            thresholds=np.array([30.0, 70.0]),
        )
        pool.traits = TraitManager(n_agents=4, definitions=[trait_def])
        # Set accumulator values and evaluate trait
        pool.accumulators.set("energy", np.array([10.0, 40.0, 80.0, 70.0]))
        pool.traits.evaluate_accumulated("condition", pool.accumulators)
        np.testing.assert_array_equal(pool.traits.get("condition"), [0, 1, 2, 2])
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py::TestAgentPoolIntegration -v
```

Expected: `AttributeError: 'AgentPool' object has no attribute 'accumulators'`

- [ ] **Step 3: Add accumulators and traits attributes to AgentPool**

In `salmon_ibm/agents.py`, add to `AgentPool.__init__()` after the existing attribute assignments:

```python
# Optional general-purpose state (Phase 1a)
self.accumulators = None  # AccumulatorManager | None
self.traits = None        # TraitManager | None
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py::TestAgentPoolIntegration -v
```

- [ ] **Step 5: Run ALL tests to verify no regressions**

```bash
conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

Expected: all tests pass, including existing salmon model tests.

**Commit message:** `feat(agents): integrate optional AccumulatorManager and TraitManager into AgentPool`

---

## Summary of Deliverables

| Chunk | Tests | Files Modified/Created |
|-------|-------|----------------------|
| 1. AccumulatorManager core | 7 | `accumulators.py` (create), `test_accumulators.py` (create) |
| 2. Clear, Increment, Stochastic Increment | 6 | `accumulators.py`, `test_accumulators.py` |
| 3. Accumulator Expression | 6 | `accumulators.py`, `test_accumulators.py` |
| 4. TimeStep, IndividualID, StochasticTrigger, QuantifyLocation | 12 | `accumulators.py`, `test_accumulators.py` |
| 5. TraitDefinition and TraitManager | 6 | `traits.py` (create), `test_traits.py` (create) |
| 6. Accumulated trait auto-evaluation | 5 | `traits.py`, `test_traits.py` |
| 7. Trait filtering (stratification) | 7 | `traits.py`, `test_traits.py` |
| 8. AgentPool integration | 5 | `agents.py`, `test_accumulators.py` |
| **Total** | **54** | **5 files** |

## What This Does NOT Include (Deferred)

- **17 remaining updaters** (Resources, Group operations, etc.) — depend on Phase 2 population/group management
- **Genetic traits** (2 types) — deferred to Phase 3
- **Transition events** (probability matrix state changes for probabilistic traits) — deferred to Phase 1b event engine
- **Event engine integration** — accumulators/traits are standalone data structures; wiring them into an event sequencer is Phase 1b
- **Performance optimization** (Numba, CuPy) — Phase 0 concern; the pure NumPy implementation is the correct first step
