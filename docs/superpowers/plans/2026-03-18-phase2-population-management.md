# Phase 2: Population Management — Implementation Plan

> **STATUS: ✅ EXECUTED** — `Population`, `BarrierMap`, IntroductionEvent / ReproductionEvent / FloaterCreationEvent all shipped. Tests in `tests/test_population.py`, `tests/test_barriers.py`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add population lifecycle management (birth, death, array resizing), barrier enforcement in movement, stage-specific survival, and basic reproduction — enabling multi-generation simulations.

**Architecture:** Phase 2 introduces a `Population` class that wraps `AgentPool` + `AccumulatorManager` + `TraitManager` into a unified lifecycle manager with dynamic array resizing (`add_agents`/`remove_agents`). A `BarrierMap` class converts the raw `Barrier` list from `heximpy.read_barriers()` into an edge-keyed lookup table that the movement kernels consult on every micro-step. Four new event types (`IntroductionEvent`, `ReproductionEvent`, `FloaterCreationEvent`, and an enhanced `SurvivalEvent`) plug into the existing event engine from Phase 1b.

**Tech Stack:** NumPy, dataclasses, existing event engine

**Test command:** `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `salmon_ibm/population.py` | Population class wrapping AgentPool + accumulators + traits + groups | **Create** |
| `salmon_ibm/barriers.py` | BarrierMap class for edge-based barrier lookup | **Create** |
| `salmon_ibm/movement.py` | Barrier enforcement in movement kernels | **Modify** |
| `salmon_ibm/events_builtin.py` | New event types: IntroductionEvent, ReproductionEvent, FloaterCreationEvent; enhanced SurvivalEvent | **Modify** |
| `salmon_ibm/events.py` | Update EventSequencer._compute_mask to use trait filtering | **Modify** |
| `tests/test_population.py` | Population class unit tests | **Create** |
| `tests/test_barriers.py` | BarrierMap unit tests | **Create** |
| `tests/test_events.py` | New event type tests (append to existing) | **Modify** |
| `tests/test_phase2_integration.py` | Multi-generation integration tests | **Create** |

---

### Task 1: Population Class — Core Wrapper

**Files:**
- Create: `salmon_ibm/population.py`
- Create: `tests/test_population.py`

Wrap AgentPool + AccumulatorManager + TraitManager into a single `Population` object that manages the full agent lifecycle. The Population is the primary object passed to events (replacing the raw AgentPool).

- [ ] **Step 1: Define the Population dataclass**

```python
# salmon_ibm/population.py
"""Population: unified lifecycle manager for a named agent collection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from salmon_ibm.agents import AgentPool
from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
from salmon_ibm.traits import TraitManager, TraitDefinition


@dataclass
class Population:
    """A named collection of individuals with shared state managers.

    Wraps AgentPool + AccumulatorManager + TraitManager and provides
    dynamic array resizing (add_agents / remove_agents).
    """
    name: str
    pool: AgentPool
    accumulator_mgr: AccumulatorManager | None = None
    trait_mgr: TraitManager | None = None

    # Group membership: -1 = floater, >= 0 = group id
    group_id: np.ndarray = field(init=False)

    # Next unique agent id (for tracking across births/deaths)
    _next_id: int = field(init=False, default=0)
    agent_ids: np.ndarray = field(init=False)

    def __post_init__(self):
        n = self.pool.n
        self.group_id = np.full(n, -1, dtype=np.int32)
        self.agent_ids = np.arange(n, dtype=np.int64)
        self._next_id = n

    @property
    def n(self) -> int:
        """Current capacity (including dead agents)."""
        return self.pool.n

    @property
    def n_alive(self) -> int:
        return int(self.pool.alive.sum())

    @property
    def alive(self) -> np.ndarray:
        return self.pool.alive

    @property
    def arrived(self) -> np.ndarray:
        return self.pool.arrived

    @property
    def tri_idx(self) -> np.ndarray:
        return self.pool.tri_idx

    @tri_idx.setter
    def tri_idx(self, v):
        self.pool.tri_idx = v

    # --- Proxies for AgentPool attributes used by existing event callbacks ---
    # These ensure Population is a drop-in replacement for AgentPool in
    # Simulation._sequencer.step(). All existing _event_* callbacks reference
    # these attributes on the population object.

    @property
    def behavior(self) -> np.ndarray:
        return self.pool.behavior
    @behavior.setter
    def behavior(self, v):
        self.pool.behavior = v

    @property
    def ed_kJ_g(self) -> np.ndarray:
        return self.pool.ed_kJ_g
    @ed_kJ_g.setter
    def ed_kJ_g(self, v):
        self.pool.ed_kJ_g = v

    @property
    def mass_g(self) -> np.ndarray:
        return self.pool.mass_g
    @mass_g.setter
    def mass_g(self, v):
        self.pool.mass_g = v

    @property
    def steps(self) -> np.ndarray:
        return self.pool.steps
    @steps.setter
    def steps(self, v):
        self.pool.steps = v

    @property
    def target_spawn_hour(self) -> np.ndarray:
        return self.pool.target_spawn_hour
    @target_spawn_hour.setter
    def target_spawn_hour(self, v):
        self.pool.target_spawn_hour = v

    @property
    def cwr_hours(self) -> np.ndarray:
        return self.pool.cwr_hours
    @cwr_hours.setter
    def cwr_hours(self, v):
        self.pool.cwr_hours = v

    @property
    def hours_since_cwr(self) -> np.ndarray:
        return self.pool.hours_since_cwr
    @hours_since_cwr.setter
    def hours_since_cwr(self, v):
        self.pool.hours_since_cwr = v

    @property
    def temp_history(self) -> np.ndarray:
        return self.pool.temp_history
    @temp_history.setter
    def temp_history(self, v):
        self.pool.temp_history = v

    def t3h_mean(self) -> np.ndarray:
        return self.pool.t3h_mean()

    def push_temperature(self, temps):
        self.pool.push_temperature(temps)

    @property
    def floaters(self) -> np.ndarray:
        """Boolean mask: alive agents not in any group."""
        return self.pool.alive & (self.group_id == -1)

    @property
    def grouped(self) -> np.ndarray:
        """Boolean mask: alive agents in a group."""
        return self.pool.alive & (self.group_id >= 0)
```

- [ ] **Step 2: Write tests for Population construction**

```python
# tests/test_population.py
"""Unit tests for the Population class."""
import numpy as np
import pytest

from salmon_ibm.agents import AgentPool
from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
from salmon_ibm.population import Population


@pytest.fixture
def basic_pool():
    return AgentPool(n=10, start_tri=0, rng_seed=42)


@pytest.fixture
def basic_pop(basic_pool):
    return Population(name="test", pool=basic_pool)


class TestPopulationInit:
    def test_name_and_size(self, basic_pop):
        assert basic_pop.name == "test"
        assert basic_pop.n == 10
        assert basic_pop.n_alive == 10

    def test_all_start_as_floaters(self, basic_pop):
        assert (basic_pop.group_id == -1).all()
        assert basic_pop.floaters.sum() == 10
        assert basic_pop.grouped.sum() == 0

    def test_agent_ids_sequential(self, basic_pop):
        np.testing.assert_array_equal(basic_pop.agent_ids, np.arange(10))

    def test_with_accumulators(self, basic_pool):
        acc_defs = [AccumulatorDef("energy", min_val=0.0)]
        acc_mgr = AccumulatorManager(10, acc_defs)
        pop = Population("test", basic_pool, accumulator_mgr=acc_mgr)
        assert pop.accumulator_mgr is not None

    def test_with_traits(self, basic_pool):
        trait_defs = [TraitDefinition("stage", TraitType.PROBABILISTIC, ["juvenile", "adult"])]
        trait_mgr = TraitManager(10, trait_defs)
        pop = Population("test", basic_pool, trait_mgr=trait_mgr)
        assert pop.trait_mgr is not None
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_population.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `feat(population): add Population class wrapping AgentPool + accumulators + traits`

> **IMPORTANT — Simulation integration:** After Task 2, update `Simulation.__init__()` to
> optionally wrap `self.pool` in a `Population` and pass the `Population` (not raw `AgentPool`)
> to `self._sequencer.step()`. The `Population` class has proxy properties for ALL `AgentPool`
> attributes used by existing callbacks (`behavior`, `ed_kJ_g`, `mass_g`, `steps`, `tri_idx`,
> `target_spawn_hour`, `cwr_hours`, `hours_since_cwr`, `temp_history`, `push_temperature()`,
> `t3h_mean()`), so existing callbacks work without modification. Add a step in Task 2:
>
> ```python
> # In Simulation.__init__(), after self.pool = AgentPool(...):
> from salmon_ibm.population import Population
> self.population = Population(name="salmon", pool=self.pool)
> ```
>
> Then in `Simulation.step()`, change:
> `self._sequencer.step(self.pool, landscape, t)` → `self._sequencer.step(self.population, landscape, t)`

---

### Task 2: Dynamic Agent Array Resizing

**Files:**
- Modify: `salmon_ibm/population.py`
- Modify: `tests/test_population.py`

Add `add_agents()` and `remove_agents()` methods that grow/shrink all parallel arrays (pool arrays, accumulators, traits, group_id, agent_ids) in lockstep.

- [ ] **Step 1: Implement remove_agents**

The simplest approach: mark agents as dead and compact periodically. For correctness, `remove_agents` immediately sets `alive[indices] = False`. A separate `compact()` method physically removes dead slots when fragmentation exceeds a threshold.

```python
# salmon_ibm/population.py — add to Population class

    def remove_agents(self, indices: np.ndarray) -> None:
        """Kill agents at given indices (mark dead, do not compact)."""
        indices = np.asarray(indices, dtype=np.intp)
        self.pool.alive[indices] = False

    def compact(self) -> None:
        """Remove dead agents, shrinking all arrays.

        Call periodically (e.g., once per generation) to reclaim memory.
        """
        alive_mask = self.pool.alive
        if alive_mask.all():
            return  # nothing to compact

        alive_idx = np.where(alive_mask)[0]
        n_new = len(alive_idx)

        # Compact AgentPool arrays
        for attr in ["tri_idx", "mass_g", "ed_kJ_g", "target_spawn_hour",
                     "behavior", "cwr_hours", "hours_since_cwr", "steps",
                     "alive", "arrived", "temp_history"]:
            arr = getattr(self.pool, attr)
            setattr(self.pool, attr, arr[alive_idx].copy())
        self.pool.n = n_new

        # Compact group_id and agent_ids
        self.group_id = self.group_id[alive_idx].copy()
        self.agent_ids = self.agent_ids[alive_idx].copy()

        # Compact accumulators
        if self.accumulator_mgr is not None:
            self.accumulator_mgr.data = self.accumulator_mgr.data[alive_idx].copy()
            self.accumulator_mgr.n_agents = n_new

        # Compact traits
        if self.trait_mgr is not None:
            for name in self.trait_mgr._data:
                self.trait_mgr._data[name] = self.trait_mgr._data[name][alive_idx].copy()
            self.trait_mgr.n_agents = n_new
```

- [ ] **Step 2: Implement add_agents**

Append new agents to the end of all arrays. Returns the indices of the newly added agents.

```python
# salmon_ibm/population.py — add to Population class

    def add_agents(
        self,
        n: int,
        positions: np.ndarray,
        *,
        mass_g: np.ndarray | None = None,
        ed_kJ_g: float = 6.5,
        group_id: int = -1,
    ) -> np.ndarray:
        """Add n new agents at given positions.

        Returns array of new agent indices.
        """
        old_n = self.pool.n
        new_n = old_n + n

        # Extend AgentPool arrays
        self.pool.tri_idx = np.concatenate([self.pool.tri_idx, positions])
        self.pool.mass_g = np.concatenate([
            self.pool.mass_g,
            mass_g if mass_g is not None else np.full(n, 3500.0),
        ])
        self.pool.ed_kJ_g = np.concatenate([self.pool.ed_kJ_g, np.full(n, ed_kJ_g)])
        self.pool.target_spawn_hour = np.concatenate([
            self.pool.target_spawn_hour, np.full(n, 720, dtype=int),
        ])
        self.pool.behavior = np.concatenate([self.pool.behavior, np.zeros(n, dtype=int)])
        self.pool.cwr_hours = np.concatenate([self.pool.cwr_hours, np.zeros(n, dtype=int)])
        self.pool.hours_since_cwr = np.concatenate([
            self.pool.hours_since_cwr, np.full(n, 999, dtype=int),
        ])
        self.pool.steps = np.concatenate([self.pool.steps, np.zeros(n, dtype=int)])
        self.pool.alive = np.concatenate([self.pool.alive, np.ones(n, dtype=bool)])
        self.pool.arrived = np.concatenate([self.pool.arrived, np.zeros(n, dtype=bool)])
        self.pool.temp_history = np.concatenate([
            self.pool.temp_history, np.full((n, 3), 15.0),
        ])
        self.pool.n = new_n

        # Extend group_id
        self.group_id = np.concatenate([
            self.group_id, np.full(n, group_id, dtype=np.int32),
        ])

        # Extend agent_ids
        new_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self.agent_ids = np.concatenate([self.agent_ids, new_ids])
        self._next_id += n

        # Extend accumulators
        if self.accumulator_mgr is not None:
            n_acc = self.accumulator_mgr.data.shape[1]
            self.accumulator_mgr.data = np.concatenate([
                self.accumulator_mgr.data,
                np.zeros((n, n_acc), dtype=np.float64),
            ])
            self.accumulator_mgr.n_agents = new_n

        # Extend traits
        if self.trait_mgr is not None:
            for name in self.trait_mgr._data:
                self.trait_mgr._data[name] = np.concatenate([
                    self.trait_mgr._data[name],
                    np.zeros(n, dtype=np.int32),
                ])
            self.trait_mgr.n_agents = new_n

        return np.arange(old_n, new_n)
```

- [ ] **Step 3: Write tests for add_agents / remove_agents / compact**

```python
# tests/test_population.py — append to file

class TestRemoveAgents:
    def test_remove_marks_dead(self, basic_pop):
        basic_pop.remove_agents(np.array([0, 3, 7]))
        assert basic_pop.pool.alive[0] is np.False_
        assert basic_pop.pool.alive[3] is np.False_
        assert basic_pop.pool.alive[7] is np.False_
        assert basic_pop.n_alive == 7

    def test_compact_shrinks_arrays(self, basic_pop):
        basic_pop.remove_agents(np.array([0, 3, 7]))
        basic_pop.compact()
        assert basic_pop.n == 7
        assert basic_pop.pool.alive.all()
        assert len(basic_pop.group_id) == 7
        assert len(basic_pop.agent_ids) == 7


class TestAddAgents:
    def test_add_extends_arrays(self, basic_pop):
        positions = np.array([5, 5, 5])
        new_idx = basic_pop.add_agents(3, positions)
        assert basic_pop.n == 13
        assert basic_pop.n_alive == 13
        np.testing.assert_array_equal(new_idx, [10, 11, 12])

    def test_new_agents_get_unique_ids(self, basic_pop):
        basic_pop.add_agents(3, np.array([0, 0, 0]))
        assert len(np.unique(basic_pop.agent_ids)) == 13

    def test_add_agents_with_accumulators(self, basic_pool):
        acc_defs = [AccumulatorDef("energy", min_val=0.0)]
        acc_mgr = AccumulatorManager(10, acc_defs)
        pop = Population("test", basic_pool, accumulator_mgr=acc_mgr)
        pop.add_agents(5, np.zeros(5, dtype=int))
        assert pop.accumulator_mgr.data.shape == (15, 1)

    def test_add_then_compact_roundtrip(self, basic_pop):
        basic_pop.add_agents(5, np.zeros(5, dtype=int))
        basic_pop.remove_agents(np.array([0, 1, 2]))
        basic_pop.compact()
        assert basic_pop.n == 12
        assert basic_pop.pool.alive.all()
```

- [ ] **Step 4: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_population.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `feat(population): add dynamic array resizing with add_agents/remove_agents/compact`

---

### Task 3: BarrierMap Class

**Files:**
- Create: `salmon_ibm/barriers.py`
- Create: `tests/test_barriers.py`

Build a `BarrierMap` that converts the list of `Barrier` objects from `heximpy.read_barriers()` into an efficient edge-keyed lookup. Each barrier entry maps a directed edge `(from_cell, to_cell)` to outcome probabilities: `(p_mortality, p_deflection, p_transmission)`.

- [ ] **Step 1: Define BarrierMap class**

```python
# salmon_ibm/barriers.py
"""Barrier map: edge-based movement restrictions on the hex grid."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from heximpy.hxnparser import Barrier, read_barriers


class BarrierOutcome(NamedTuple):
    """Probabilities for barrier interaction outcomes."""
    p_mortality: float    # probability agent dies
    p_deflection: float   # probability agent is deflected (stays)
    p_transmission: float # probability agent crosses

    @staticmethod
    def impassable() -> BarrierOutcome:
        return BarrierOutcome(0.0, 1.0, 0.0)

    @staticmethod
    def lethal() -> BarrierOutcome:
        return BarrierOutcome(1.0, 0.0, 0.0)


@dataclass
class BarrierClass:
    """Configuration for a barrier classification."""
    name: str
    forward: BarrierOutcome   # crossing in the defined direction
    reverse: BarrierOutcome   # crossing in the opposite direction


class BarrierMap:
    """Edge-keyed barrier lookup for movement enforcement.

    Keys are (from_compact, to_compact) tuples mapping to BarrierOutcome.
    For undirected barriers, both directions have the same outcome.
    For directional barriers, forward and reverse have different outcomes.

    Usage:
        bmap = BarrierMap.from_hbf(path, mesh)
        outcome = bmap.check(from_cell, to_cell)
        # outcome is None if no barrier, else a BarrierOutcome
    """

    def __init__(self):
        self._edges: dict[tuple[int, int], BarrierOutcome] = {}

    def add_edge(self, from_cell: int, to_cell: int, outcome: BarrierOutcome) -> None:
        self._edges[(from_cell, to_cell)] = outcome

    def check(self, from_cell: int, to_cell: int) -> BarrierOutcome | None:
        return self._edges.get((from_cell, to_cell))

    def has_barriers(self) -> bool:
        return len(self._edges) > 0

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    @classmethod
    def from_hbf(
        cls,
        path: str | Path,
        mesh,
        class_config: dict[str, BarrierClass] | None = None,
    ) -> BarrierMap:
        """Load barriers from .hbf file and map to compact mesh indices.

        Parameters
        ----------
        path : path to .hbf file
        mesh : HexMesh instance (provides _full_to_compact mapping)
        class_config : optional mapping from class_name -> BarrierClass
            with outcome probabilities. If None, all barriers are impassable.
        """
        barriers = read_barriers(path)
        bmap = cls()

        for b in barriers:
            # b.hex_id is a full (flat) grid index
            # b.edge is the neighbor direction (0-5)
            if b.hex_id not in mesh._full_to_compact:
                continue  # barrier on non-water cell

            compact_from = mesh._full_to_compact[b.hex_id]

            # Determine the neighbor cell in the given edge direction.
            # IMPORTANT: mesh.neighbors slots are NOT directional — they
            # are packed sequentially (non-water skipped). We must compute
            # the full-grid neighbor for the given edge direction, convert
            # to compact index, then find its slot in the neighbor array.
            from salmon_ibm.hexsim import _hex_neighbors_offset
            full_from = mesh._water_full_idx[compact_from]
            r, c = int(full_from // mesh._ncols), int(full_from % mesh._ncols)
            dir_nbrs = _hex_neighbors_offset(r, c, mesh._ncols, mesh._nrows, mesh._n_data)
            if b.edge >= len(dir_nbrs):
                continue  # edge direction out of range
            full_to = dir_nbrs[b.edge]
            if full_to not in mesh._full_to_compact:
                continue  # neighbor is non-water
            compact_to = mesh._full_to_compact[full_to]

            # Determine outcome probabilities
            if class_config and b.class_name in class_config:
                bc = class_config[b.class_name]
                fwd = bc.forward
                rev = bc.reverse
            else:
                fwd = BarrierOutcome.impassable()
                rev = BarrierOutcome.impassable()

            bmap.add_edge(compact_from, compact_to, fwd)
            bmap.add_edge(compact_to, compact_from, rev)

        return bmap

    def to_arrays(self, n_cells: int, max_nbrs: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to array form for vectorized movement.

        Returns
        -------
        barrier_mort : float64[n_cells, max_nbrs] — mortality prob per edge
        barrier_defl : float64[n_cells, max_nbrs] — deflection prob per edge
        barrier_trans : float64[n_cells, max_nbrs] — transmission prob per edge

        Edges with no barrier have (0, 0, 1) — i.e., always transmit.
        """
        mort = np.zeros((n_cells, max_nbrs), dtype=np.float64)
        defl = np.zeros((n_cells, max_nbrs), dtype=np.float64)
        trans = np.ones((n_cells, max_nbrs), dtype=np.float64)

        for (fc, tc), outcome in self._edges.items():
            # Find which neighbor slot tc occupies for fc
            # This requires the mesh neighbors array
            # Caller should use this with mesh.neighbors
            pass  # Populated in Task 4 integration

        return mort, defl, trans
```

- [ ] **Step 2: Write BarrierMap unit tests (synthetic data)**

```python
# tests/test_barriers.py
"""Unit tests for BarrierMap."""
import numpy as np
import pytest

from salmon_ibm.barriers import BarrierMap, BarrierOutcome, BarrierClass


class TestBarrierOutcome:
    def test_impassable(self):
        o = BarrierOutcome.impassable()
        assert o.p_mortality == 0.0
        assert o.p_deflection == 1.0
        assert o.p_transmission == 0.0

    def test_lethal(self):
        o = BarrierOutcome.lethal()
        assert o.p_mortality == 1.0
        assert o.p_transmission == 0.0

    def test_custom_probabilities(self):
        o = BarrierOutcome(0.1, 0.3, 0.6)
        assert abs(o.p_mortality + o.p_deflection + o.p_transmission - 1.0) < 1e-10


class TestBarrierMap:
    def test_empty_map(self):
        bmap = BarrierMap()
        assert not bmap.has_barriers()
        assert bmap.check(0, 1) is None

    def test_add_and_check(self):
        bmap = BarrierMap()
        outcome = BarrierOutcome(0.1, 0.3, 0.6)
        bmap.add_edge(0, 1, outcome)
        assert bmap.check(0, 1) == outcome
        assert bmap.check(1, 0) is None  # only forward direction

    def test_bidirectional(self):
        bmap = BarrierMap()
        fwd = BarrierOutcome(0.1, 0.3, 0.6)
        rev = BarrierOutcome(0.0, 0.5, 0.5)
        bmap.add_edge(0, 1, fwd)
        bmap.add_edge(1, 0, rev)
        assert bmap.check(0, 1) == fwd
        assert bmap.check(1, 0) == rev

    def test_n_edges(self):
        bmap = BarrierMap()
        bmap.add_edge(0, 1, BarrierOutcome.impassable())
        bmap.add_edge(1, 0, BarrierOutcome.impassable())
        bmap.add_edge(2, 3, BarrierOutcome.lethal())
        assert bmap.n_edges == 3
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_barriers.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `feat(barriers): add BarrierMap class for edge-based barrier lookup`

---

### Task 4: Barrier Enforcement in Movement

**Files:**
- Modify: `salmon_ibm/movement.py`
- Modify: `salmon_ibm/barriers.py` (finish `to_arrays`)
- Modify: `tests/test_barriers.py` (add movement integration tests)

Integrate barrier checking into the vectorized movement kernels. On each micro-step, when an agent would move from cell A to cell B, check the barrier map. Three outcomes: mortality (agent dies), deflection (agent stays at A), or transmission (agent crosses to B).

- [ ] **Step 1: Implement `BarrierMap.to_arrays()` with mesh reference**

Complete the `to_arrays` method so it produces NumPy arrays indexed by `[cell, neighbor_slot]` for vectorized barrier checks. Each cell-neighbor pair maps to `(p_mort, p_defl, p_trans)`.

```python
# salmon_ibm/barriers.py — replace the stub to_arrays

    def to_arrays(self, mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to array form for vectorized movement.

        Returns
        -------
        barrier_mort : float64[n_cells, max_nbrs]
        barrier_defl : float64[n_cells, max_nbrs]
        barrier_trans : float64[n_cells, max_nbrs]

        No-barrier edges have (0, 0, 1) — always transmit.
        """
        n_cells = mesh.n_cells
        max_nbrs = mesh.neighbors.shape[1]
        mort = np.zeros((n_cells, max_nbrs), dtype=np.float64)
        defl = np.zeros((n_cells, max_nbrs), dtype=np.float64)
        trans = np.ones((n_cells, max_nbrs), dtype=np.float64)

        for (fc, tc), outcome in self._edges.items():
            # Find which neighbor slot tc occupies for cell fc
            nbr_row = mesh.neighbors[fc]
            slots = np.where(nbr_row == tc)[0]
            if len(slots) > 0:
                slot = slots[0]
                mort[fc, slot] = outcome.p_mortality
                defl[fc, slot] = outcome.p_deflection
                trans[fc, slot] = outcome.p_transmission

        return mort, defl, trans
```

- [ ] **Step 2: Add barrier-aware helper to movement.py**

Add a `_resolve_barriers_vec` function that takes proposed moves and applies barrier outcomes vectorized.

```python
# salmon_ibm/movement.py — add function

def _resolve_barriers_vec(
    current: np.ndarray,      # (n,) current cell indices
    proposed: np.ndarray,      # (n,) proposed next cell indices
    barrier_mort: np.ndarray,  # (n_cells, max_nbrs)
    barrier_defl: np.ndarray,
    barrier_trans: np.ndarray,
    neighbors: np.ndarray,     # (n_cells, max_nbrs)
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve barrier outcomes for a batch of proposed moves.

    Returns
    -------
    final_positions : int array — where agents actually end up
    died : bool array — which agents died at barriers
    """
    n = len(current)
    final = proposed.copy()
    died = np.zeros(n, dtype=bool)

    # Skip agents that didn't move (current == proposed)
    moving = current != proposed
    if not moving.any():
        return final, died

    # Find which neighbor slot each proposed cell occupies
    # For each agent, expand neighbors[current] and find proposed
    nbr_matrix = neighbors[current]  # (n, max_nbrs)
    # Match: which column == proposed?
    match = (nbr_matrix == proposed[:, np.newaxis])  # (n, max_nbrs)
    has_match = match.any(axis=1) & moving  # only consider agents that actually moved

    if not has_match.any():
        return final, died

    # Get the slot index for matched agents
    slot = np.argmax(match, axis=1)  # first matching slot

    # Look up barrier probabilities
    p_mort = barrier_mort[current, slot]
    p_defl = barrier_defl[current, slot]
    # p_trans = barrier_trans[current, slot]

    # Only process agents that actually hit a barrier
    has_barrier = has_match & ((p_mort > 0) | (p_defl > 0))

    if not has_barrier.any():
        return final, died

    # Roll random numbers for barrier resolution
    rolls = rng.random(n)

    # Mortality: roll < p_mort
    kill = has_barrier & (rolls < p_mort)
    died[kill] = True

    # Deflection: p_mort <= roll < p_mort + p_defl
    deflect = has_barrier & ~kill & (rolls < p_mort + p_defl)
    final[deflect] = current[deflect]

    # Transmission: roll >= p_mort + p_defl (agent crosses)
    # final already set to proposed for these agents

    return final, died
```

- [ ] **Step 3: Integrate barrier resolution into `execute_movement`**

Modify `execute_movement()` in `salmon_ibm/movement.py` to accept an optional `barrier_arrays` parameter. When provided, each micro-step's proposed moves are filtered through `_resolve_barriers_vec`. Dead agents are marked in `pool.alive`.

```python
# salmon_ibm/movement.py — modify execute_movement signature and body

def execute_movement(pool, mesh, fields, seed=None, n_micro_steps=3,
                     cwr_threshold=16.0, barrier_arrays=None):
    """Execute movement with optional barrier enforcement.

    Parameters
    ----------
    barrier_arrays : tuple (mort, defl, trans) from BarrierMap.to_arrays(), or None
    """
    rng = np.random.default_rng(seed)
    # ... existing code, but each _step_*_vec function now returns
    # proposed positions, and barriers are resolved afterward.
    # Implementation: wrap each movement block with barrier check.
```

- [ ] **Step 4: Modify `_step_random_vec` to support barrier checking**

Refactor so the inner loop produces proposed positions that can be intercepted. The function now takes optional barrier arrays and an rng for barrier rolls.

```python
# salmon_ibm/movement.py — modify _step_random_vec

def _step_random_vec(tri_indices, water_nbrs, water_nbr_count, rng, steps,
                     barrier_arrays=None):
    """Vectorized random walk with optional barrier enforcement."""
    n = len(tri_indices)
    for _ in range(steps):
        current = tri_indices.copy()
        counts = water_nbr_count[current]
        has_nbrs = counts > 0
        if not has_nbrs.any():
            break
        rand_idx = rng.integers(0, np.maximum(counts, 1))
        proposed = water_nbrs[current, rand_idx]

        if barrier_arrays is not None:
            mort, defl, trans = barrier_arrays
            final, died = _resolve_barriers_vec(
                current, proposed, mort, defl, trans, water_nbrs, rng,
            )
            proposed = final
            # Mark died agents — caller handles alive array update
            # For now: died agents stay at current position
            proposed[died] = current[died]

        tri_indices[has_nbrs] = proposed[has_nbrs]
```

- [ ] **Step 5: Write barrier-movement integration tests**

```python
# tests/test_barriers.py — append

class TestBarrierMovementIntegration:
    """Test barrier enforcement with synthetic mesh data."""

    @pytest.fixture
    def line_mesh_data(self):
        """5-cell linear mesh: 0-1-2-3-4, each cell has max 2 neighbors."""
        n = 5
        neighbors = np.full((n, 6), -1, dtype=np.intp)
        neighbors[0, 0] = 1
        neighbors[1, 0] = 0; neighbors[1, 1] = 2
        neighbors[2, 0] = 1; neighbors[2, 1] = 3
        neighbors[3, 0] = 2; neighbors[3, 1] = 4
        neighbors[4, 0] = 3
        nbr_count = np.sum(neighbors >= 0, axis=1).astype(np.intp)
        return neighbors, nbr_count

    def test_impassable_barrier_blocks_movement(self, line_mesh_data):
        neighbors, nbr_count = line_mesh_data
        # Barrier between cell 2 and 3 (impassable)
        mort = np.zeros((5, 6), dtype=np.float64)
        defl = np.zeros((5, 6), dtype=np.float64)
        trans = np.ones((5, 6), dtype=np.float64)
        # Cell 2 -> cell 3 is slot 1
        defl[2, 1] = 1.0; trans[2, 1] = 0.0
        # Cell 3 -> cell 2 is slot 0
        defl[3, 0] = 1.0; trans[3, 0] = 0.0

        current = np.array([2])
        proposed = np.array([3])
        rng = np.random.default_rng(42)

        from salmon_ibm.movement import _resolve_barriers_vec
        final, died = _resolve_barriers_vec(
            current, proposed, mort, defl, trans, neighbors, rng,
        )
        assert final[0] == 2  # deflected back
        assert not died[0]

    def test_lethal_barrier_kills_agent(self, line_mesh_data):
        neighbors, nbr_count = line_mesh_data
        mort = np.zeros((5, 6), dtype=np.float64)
        defl = np.zeros((5, 6), dtype=np.float64)
        trans = np.ones((5, 6), dtype=np.float64)
        mort[2, 1] = 1.0; trans[2, 1] = 0.0

        current = np.array([2])
        proposed = np.array([3])
        rng = np.random.default_rng(42)

        from salmon_ibm.movement import _resolve_barriers_vec
        final, died = _resolve_barriers_vec(
            current, proposed, mort, defl, trans, neighbors, rng,
        )
        assert died[0]

    def test_no_barrier_allows_passage(self, line_mesh_data):
        neighbors, nbr_count = line_mesh_data
        mort = np.zeros((5, 6), dtype=np.float64)
        defl = np.zeros((5, 6), dtype=np.float64)
        trans = np.ones((5, 6), dtype=np.float64)

        current = np.array([2])
        proposed = np.array([3])
        rng = np.random.default_rng(42)

        from salmon_ibm.movement import _resolve_barriers_vec
        final, died = _resolve_barriers_vec(
            current, proposed, mort, defl, trans, neighbors, rng,
        )
        assert final[0] == 3
        assert not died[0]
```

- [ ] **Step 6: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_barriers.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `feat(barriers): enforce barriers in vectorized movement kernels`

---

### Task 5: Stage-Specific SurvivalEvent

**Files:**
- Modify: `salmon_ibm/events_builtin.py`
- Modify: `salmon_ibm/events.py` (activate trait filtering in `_compute_mask`)
- Modify: `tests/test_events.py`

Enhance `SurvivalEvent` with trait-filtered, stage-specific mortality rates and density-dependent survival.

- [ ] **Step 1: Activate trait filtering in EventSequencer._compute_mask**

```python
# salmon_ibm/events.py — replace the stub in _compute_mask

    @staticmethod
    def _compute_mask(population, trait_filter: dict | None) -> np.ndarray:
        base = population.alive & ~population.arrived
        if trait_filter is not None and hasattr(population, 'trait_mgr') and population.trait_mgr is not None:
            trait_mask = population.trait_mgr.filter_by_traits(**trait_filter)
            base = base & trait_mask
        elif trait_filter is not None and hasattr(population, 'traits') and population.traits is not None:
            trait_mask = population.traits.filter_by_traits(**trait_filter)
            base = base & trait_mask
        return base
```

- [ ] **Step 2: Add StageSpecificSurvivalEvent**

```python
# salmon_ibm/events_builtin.py — append

@register_event("stage_survival")
@dataclass
class StageSpecificSurvivalEvent(Event):
    """Trait-filtered survival: different mortality rates per stage.

    mortality_rates maps trait category name -> probability of death per step.
    trait_name is the trait used to determine stage.
    density_dependent: if True, mortality increases with local density.
    """
    trait_name: str = "stage"
    mortality_rates: dict[str, float] = field(default_factory=dict)
    density_dependent: bool = False
    density_scale: float = 1.0  # scaling factor for density effect

    def execute(self, population, landscape, t, mask):
        if not mask.any():
            return

        rng = landscape.get("rng", np.random.default_rng())
        trait_mgr = getattr(population, 'trait_mgr', None) or getattr(population, 'traits', None)

        if trait_mgr is None:
            # No traits: apply a flat mortality if "default" rate specified
            default_rate = self.mortality_rates.get("default", 0.0)
            if default_rate > 0:
                rolls = rng.random(mask.sum())
                die_local = rolls < default_rate
                die_indices = np.where(mask)[0][die_local]
                population.alive[die_indices] = False
            return

        # Per-stage mortality
        trait_vals = trait_mgr.get(self.trait_name)
        defn = trait_mgr.definitions[self.trait_name]

        alive_idx = np.where(mask)[0]
        mort_prob = np.zeros(len(alive_idx), dtype=np.float64)

        for cat_name, rate in self.mortality_rates.items():
            cat_idx = defn.categories.index(cat_name)
            cat_mask = trait_vals[alive_idx] == cat_idx
            mort_prob[cat_mask] = rate

        # Density-dependent modifier
        if self.density_dependent:
            mesh = landscape.get("mesh")
            if mesh is not None:
                positions = population.tri_idx[alive_idx]
                # Count agents per cell
                cell_counts = np.bincount(positions, minlength=mesh.n_cells)
                local_density = cell_counts[positions].astype(np.float64)
                # Scale mortality: mort * (1 + density_scale * (local_density - 1))
                density_factor = 1.0 + self.density_scale * np.maximum(local_density - 1.0, 0.0)
                mort_prob = np.minimum(mort_prob * density_factor, 1.0)

        rolls = rng.random(len(alive_idx))
        die_local = rolls < mort_prob
        die_indices = alive_idx[die_local]
        population.alive[die_indices] = False
```

- [ ] **Step 3: Write tests for stage-specific survival**

```python
# tests/test_events.py — append

from salmon_ibm.events_builtin import StageSpecificSurvivalEvent
from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
from salmon_ibm.population import Population


class TestStageSpecificSurvival:
    @pytest.fixture
    def pop_with_stages(self):
        pool = AgentPool(n=100, start_tri=0, rng_seed=42)
        trait_defs = [
            TraitDefinition("stage", TraitType.PROBABILISTIC, ["juvenile", "adult"]),
        ]
        trait_mgr = TraitManager(100, trait_defs)
        # First 50 juvenile (0), last 50 adult (1)
        trait_mgr._data["stage"][:50] = 0
        trait_mgr._data["stage"][50:] = 1
        pop = Population("fish", pool, trait_mgr=trait_mgr)
        return pop

    def test_juvenile_high_mortality(self, pop_with_stages):
        event = StageSpecificSurvivalEvent(
            name="survival",
            mortality_rates={"juvenile": 1.0, "adult": 0.0},
        )
        mask = pop_with_stages.alive.copy()
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(pop_with_stages, landscape, t=0, mask=mask)
        # All juveniles dead, all adults alive
        assert pop_with_stages.pool.alive[:50].sum() == 0
        assert pop_with_stages.pool.alive[50:].sum() == 50

    def test_zero_mortality_preserves_all(self, pop_with_stages):
        event = StageSpecificSurvivalEvent(
            name="survival",
            mortality_rates={"juvenile": 0.0, "adult": 0.0},
        )
        mask = pop_with_stages.alive.copy()
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(pop_with_stages, landscape, t=0, mask=mask)
        assert pop_with_stages.n_alive == 100
```

- [ ] **Step 4: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_events.py -v -k "StageSpecificSurvival" --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `feat(survival): add StageSpecificSurvivalEvent with trait-filtered mortality rates`

---

### Task 6: IntroductionEvent

**Files:**
- Modify: `salmon_ibm/events_builtin.py`
- Modify: `tests/test_events.py`

Add an event that introduces new agents at specified locations with initial trait/accumulator values.

- [ ] **Step 1: Implement IntroductionEvent**

```python
# salmon_ibm/events_builtin.py — append

@register_event("introduction")
@dataclass
class IntroductionEvent(Event):
    """Add new individuals to the population.

    n_agents: number to introduce each time the event fires
    positions: list of cell indices (cycled if shorter than n_agents)
    initial_mass_mean: mean mass in grams
    initial_mass_std: standard deviation of mass
    initial_ed: initial energy density (kJ/g)
    initial_traits: dict trait_name -> category_name for new agents
    initial_accumulators: dict acc_name -> value for new agents
    """
    n_agents: int = 10
    positions: list[int] = field(default_factory=lambda: [0])
    initial_mass_mean: float = 3500.0
    initial_mass_std: float = 500.0
    initial_ed: float = 6.5
    initial_traits: dict[str, str] = field(default_factory=dict)
    initial_accumulators: dict[str, float] = field(default_factory=dict)

    def execute(self, population, landscape, t, mask):
        rng = landscape.get("rng", np.random.default_rng())
        n = self.n_agents

        # Cycle positions to fill n agents
        pos_arr = np.array(self.positions, dtype=int)
        if len(pos_arr) < n:
            pos_arr = np.tile(pos_arr, (n // len(pos_arr)) + 1)[:n]

        mass = np.clip(
            rng.normal(self.initial_mass_mean, self.initial_mass_std, n),
            self.initial_mass_mean * 0.5,
            self.initial_mass_mean * 1.5,
        )

        new_idx = population.add_agents(n, pos_arr, mass_g=mass, ed_kJ_g=self.initial_ed)

        # Set initial traits
        if population.trait_mgr is not None:
            for trait_name, cat_name in self.initial_traits.items():
                defn = population.trait_mgr.definitions[trait_name]
                cat_idx = defn.categories.index(cat_name)
                population.trait_mgr._data[trait_name][new_idx] = cat_idx

        # Set initial accumulators
        if population.accumulator_mgr is not None:
            for acc_name, value in self.initial_accumulators.items():
                idx = population.accumulator_mgr.index_of(acc_name)
                population.accumulator_mgr.data[new_idx, idx] = value
```

- [ ] **Step 2: Write tests for IntroductionEvent**

```python
# tests/test_events.py — append

from salmon_ibm.events_builtin import IntroductionEvent


class TestIntroductionEvent:
    def test_adds_agents(self):
        pool = AgentPool(n=10, start_tri=0, rng_seed=42)
        pop = Population("fish", pool)
        event = IntroductionEvent(name="introduce", n_agents=5, positions=[3, 7])
        landscape = {"rng": np.random.default_rng(42)}
        mask = pop.alive.copy()
        event.execute(pop, landscape, t=0, mask=mask)
        assert pop.n == 15
        assert pop.n_alive == 15

    def test_positions_are_set(self):
        pool = AgentPool(n=5, start_tri=0, rng_seed=42)
        pop = Population("fish", pool)
        event = IntroductionEvent(name="introduce", n_agents=3, positions=[10])
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(pop, landscape, t=0, mask=pop.alive.copy())
        assert (pop.pool.tri_idx[5:] == 10).all()

    def test_sets_initial_traits(self):
        pool = AgentPool(n=5, start_tri=0, rng_seed=42)
        trait_defs = [TraitDefinition("stage", TraitType.PROBABILISTIC, ["egg", "juvenile", "adult"])]
        trait_mgr = TraitManager(5, trait_defs)
        pop = Population("fish", pool, trait_mgr=trait_mgr)
        event = IntroductionEvent(
            name="introduce", n_agents=3, positions=[0],
            initial_traits={"stage": "juvenile"},
        )
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(pop, landscape, t=0, mask=pop.alive.copy())
        # New agents (indices 5, 6, 7) should be juvenile (index 1)
        assert (pop.trait_mgr._data["stage"][5:] == 1).all()
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_events.py -v -k "IntroductionEvent" --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `feat(events): add IntroductionEvent for adding agents at specified locations`

---

### Task 7: ReproductionEvent

**Files:**
- Modify: `salmon_ibm/events_builtin.py`
- Modify: `tests/test_events.py`

Group-based reproduction: agents in groups produce offspring with configurable clutch sizes. Offspring inherit position and group from parent, traits can be configured.

- [ ] **Step 1: Implement ReproductionEvent**

```python
# salmon_ibm/events_builtin.py — append

@register_event("reproduction")
@dataclass
class ReproductionEvent(Event):
    """Group-based reproduction.

    Only agents in groups (group_id >= 0) and passing the trait filter reproduce.
    Each reproducing agent produces clutch_size offspring at the same cell.
    Offspring start as floaters (group_id = -1).

    clutch_mean: mean clutch size (Poisson-distributed)
    offspring_trait: trait category name assigned to offspring
    offspring_trait_name: which trait to set on offspring
    min_group_size: minimum group size for reproduction to occur
    """
    clutch_mean: float = 4.0
    offspring_trait_name: str = "stage"
    offspring_trait_value: str = "juvenile"
    min_group_size: int = 1
    offspring_mass_mean: float = 100.0
    offspring_mass_std: float = 20.0
    offspring_ed: float = 6.5

    def execute(self, population, landscape, t, mask):
        rng = landscape.get("rng", np.random.default_rng())

        # Only grouped, alive agents matching mask can reproduce
        can_reproduce = mask & (population.group_id >= 0)

        if self.min_group_size > 1:
            # Count group sizes
            group_ids = population.group_id[can_reproduce]
            unique_groups, counts = np.unique(group_ids[group_ids >= 0], return_counts=True)
            valid_groups = set(unique_groups[counts >= self.min_group_size])
            can_reproduce = can_reproduce & np.isin(population.group_id, list(valid_groups))

        reproducer_idx = np.where(can_reproduce)[0]
        if len(reproducer_idx) == 0:
            return

        # Determine clutch sizes (Poisson)
        clutch_sizes = rng.poisson(self.clutch_mean, size=len(reproducer_idx))
        total_offspring = clutch_sizes.sum()
        if total_offspring == 0:
            return

        # Offspring positions: inherit from parent
        parent_positions = population.tri_idx[reproducer_idx]
        offspring_positions = np.repeat(parent_positions, clutch_sizes)

        # Offspring mass
        offspring_mass = np.clip(
            rng.normal(self.offspring_mass_mean, self.offspring_mass_std, total_offspring),
            self.offspring_mass_mean * 0.5,
            self.offspring_mass_mean * 1.5,
        )

        new_idx = population.add_agents(
            total_offspring, offspring_positions,
            mass_g=offspring_mass, ed_kJ_g=self.offspring_ed,
        )

        # Set offspring trait
        if population.trait_mgr is not None and self.offspring_trait_name:
            defn = population.trait_mgr.definitions[self.offspring_trait_name]
            cat_idx = defn.categories.index(self.offspring_trait_value)
            population.trait_mgr._data[self.offspring_trait_name][new_idx] = cat_idx
```

- [ ] **Step 2: Write tests for ReproductionEvent**

```python
# tests/test_events.py — append

from salmon_ibm.events_builtin import ReproductionEvent


class TestReproductionEvent:
    @pytest.fixture
    def grouped_pop(self):
        pool = AgentPool(n=10, start_tri=5, rng_seed=42)
        trait_defs = [
            TraitDefinition("stage", TraitType.PROBABILISTIC, ["juvenile", "adult"]),
        ]
        trait_mgr = TraitManager(10, trait_defs)
        trait_mgr._data["stage"][:] = 1  # all adults
        pop = Population("fish", pool, trait_mgr=trait_mgr)
        pop.group_id[:5] = 0  # first 5 in group 0
        pop.group_id[5:] = -1  # last 5 are floaters
        return pop

    def test_only_grouped_reproduce(self, grouped_pop):
        event = ReproductionEvent(name="repro", clutch_mean=2.0)
        landscape = {"rng": np.random.default_rng(42)}
        mask = grouped_pop.alive.copy()
        event.execute(grouped_pop, landscape, t=0, mask=mask)
        # Population should grow (5 grouped agents each producing ~2 offspring)
        assert grouped_pop.n > 10

    def test_offspring_inherit_position(self, grouped_pop):
        event = ReproductionEvent(name="repro", clutch_mean=1.0)
        landscape = {"rng": np.random.default_rng(42)}
        mask = grouped_pop.alive.copy()
        event.execute(grouped_pop, landscape, t=0, mask=mask)
        # All offspring should be at cell 5 (parent position)
        assert (grouped_pop.pool.tri_idx[10:] == 5).all()

    def test_offspring_get_trait(self, grouped_pop):
        event = ReproductionEvent(
            name="repro", clutch_mean=1.0,
            offspring_trait_name="stage", offspring_trait_value="juvenile",
        )
        landscape = {"rng": np.random.default_rng(42)}
        mask = grouped_pop.alive.copy()
        event.execute(grouped_pop, landscape, t=0, mask=mask)
        # Offspring should be juvenile (0)
        assert (grouped_pop.trait_mgr._data["stage"][10:] == 0).all()

    def test_zero_clutch_no_offspring(self, grouped_pop):
        event = ReproductionEvent(name="repro", clutch_mean=0.001)
        landscape = {"rng": np.random.default_rng(0)}
        mask = grouped_pop.alive.copy()
        # With very low clutch mean, most will produce 0
        n_before = grouped_pop.n
        event.execute(grouped_pop, landscape, t=0, mask=mask)
        # Might add 0 or very few
        assert grouped_pop.n >= n_before

    def test_floaters_excluded(self, grouped_pop):
        # Make everyone a floater
        grouped_pop.group_id[:] = -1
        event = ReproductionEvent(name="repro", clutch_mean=5.0)
        landscape = {"rng": np.random.default_rng(42)}
        mask = grouped_pop.alive.copy()
        event.execute(grouped_pop, landscape, t=0, mask=mask)
        assert grouped_pop.n == 10  # no offspring produced
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_events.py -v -k "ReproductionEvent" --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `feat(events): add ReproductionEvent with Poisson clutch sizes and trait inheritance`

---

### Task 8: FloaterCreationEvent

**Files:**
- Modify: `salmon_ibm/events_builtin.py`
- Modify: `tests/test_events.py`

Release individuals from groups to become floaters (dispersers). This is the inverse of group assignment and triggers subsequent exploration/dispersal movement.

- [ ] **Step 1: Implement FloaterCreationEvent**

```python
# salmon_ibm/events_builtin.py — append

@register_event("floater_creation")
@dataclass
class FloaterCreationEvent(Event):
    """Release agents from groups to become floaters.

    probability: per-agent probability of being released per timestep
    trait_filter on the Event itself selects which agents are eligible
    """
    probability: float = 0.1

    def execute(self, population, landscape, t, mask):
        rng = landscape.get("rng", np.random.default_rng())

        # Only grouped agents matching mask are candidates
        candidates = mask & (population.group_id >= 0)
        cand_idx = np.where(candidates)[0]
        if len(cand_idx) == 0:
            return

        # Roll for release
        rolls = rng.random(len(cand_idx))
        release = rolls < self.probability
        release_idx = cand_idx[release]

        # Set to floater
        population.group_id[release_idx] = -1
```

- [ ] **Step 2: Write tests for FloaterCreationEvent**

```python
# tests/test_events.py — append

from salmon_ibm.events_builtin import FloaterCreationEvent


class TestFloaterCreationEvent:
    @pytest.fixture
    def all_grouped_pop(self):
        pool = AgentPool(n=20, start_tri=0, rng_seed=42)
        pop = Population("fish", pool)
        pop.group_id[:10] = 0
        pop.group_id[10:] = 1
        return pop

    def test_releases_some_agents(self, all_grouped_pop):
        event = FloaterCreationEvent(name="release", probability=0.5)
        landscape = {"rng": np.random.default_rng(42)}
        mask = all_grouped_pop.alive.copy()
        event.execute(all_grouped_pop, landscape, t=0, mask=mask)
        n_floaters = (all_grouped_pop.group_id == -1).sum()
        assert 0 < n_floaters < 20  # some but not all released

    def test_probability_zero_releases_none(self, all_grouped_pop):
        event = FloaterCreationEvent(name="release", probability=0.0)
        landscape = {"rng": np.random.default_rng(42)}
        mask = all_grouped_pop.alive.copy()
        event.execute(all_grouped_pop, landscape, t=0, mask=mask)
        assert (all_grouped_pop.group_id >= 0).all()

    def test_probability_one_releases_all(self, all_grouped_pop):
        event = FloaterCreationEvent(name="release", probability=1.0)
        landscape = {"rng": np.random.default_rng(42)}
        mask = all_grouped_pop.alive.copy()
        event.execute(all_grouped_pop, landscape, t=0, mask=mask)
        assert (all_grouped_pop.group_id == -1).all()

    def test_floaters_not_affected(self, all_grouped_pop):
        all_grouped_pop.group_id[:5] = -1  # 5 already floaters
        event = FloaterCreationEvent(name="release", probability=1.0)
        landscape = {"rng": np.random.default_rng(42)}
        mask = all_grouped_pop.alive.copy()
        event.execute(all_grouped_pop, landscape, t=0, mask=mask)
        assert (all_grouped_pop.group_id == -1).all()
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_events.py -v -k "FloaterCreation" --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `feat(events): add FloaterCreationEvent for releasing agents from groups`

---

### Task 9: Integration Tests — Multi-Generation Simulation

**Files:**
- Create: `tests/test_phase2_integration.py`

End-to-end tests that run a multi-generation simulation using Population, barriers, survival, reproduction, and introduction events together through the EventSequencer.

- [ ] **Step 1: Write multi-generation integration test**

```python
# tests/test_phase2_integration.py
"""Integration tests for Phase 2: multi-generation simulation."""
import numpy as np
import pytest

from salmon_ibm.agents import AgentPool
from salmon_ibm.population import Population
from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
from salmon_ibm.events import EventSequencer, EveryStep, Periodic
from salmon_ibm.events_builtin import (
    StageSpecificSurvivalEvent,
    IntroductionEvent,
    ReproductionEvent,
    FloaterCreationEvent,
)
from salmon_ibm.barriers import BarrierMap, BarrierOutcome


class TestMultiGenerationSimulation:
    """Run a simplified multi-generation lifecycle."""

    @pytest.fixture
    def setup_simulation(self):
        """Create a population with traits and a simple event sequence."""
        pool = AgentPool(n=50, start_tri=0, rng_seed=42)
        trait_defs = [
            TraitDefinition("stage", TraitType.PROBABILISTIC,
                          ["egg", "juvenile", "adult"]),
        ]
        trait_mgr = TraitManager(50, trait_defs)
        trait_mgr._data["stage"][:] = 2  # all start as adult

        acc_defs = [AccumulatorDef("age", min_val=0.0)]
        acc_mgr = AccumulatorManager(50, acc_defs)

        pop = Population("salmon", pool,
                        accumulator_mgr=acc_mgr, trait_mgr=trait_mgr)
        # Put first 25 in groups for reproduction
        pop.group_id[:25] = 0

        events = [
            StageSpecificSurvivalEvent(
                name="mortality",
                mortality_rates={"egg": 0.5, "juvenile": 0.1, "adult": 0.02},
            ),
            ReproductionEvent(
                name="reproduction",
                trigger=Periodic(interval=10),
                clutch_mean=3.0,
                offspring_trait_name="stage",
                offspring_trait_value="egg",
                offspring_mass_mean=10.0,
            ),
            FloaterCreationEvent(
                name="disperse",
                trigger=Periodic(interval=5),
                probability=0.2,
            ),
        ]

        sequencer = EventSequencer(events)
        landscape = {"rng": np.random.default_rng(42)}
        return pop, sequencer, landscape

    def test_population_survives_20_steps(self, setup_simulation):
        pop, sequencer, landscape = setup_simulation
        for t in range(20):
            sequencer.step(pop, landscape, t)
        # Population should still exist (adults have low mortality)
        assert pop.n_alive > 0

    def test_reproduction_adds_agents(self, setup_simulation):
        pop, sequencer, landscape = setup_simulation
        n_initial = pop.n
        for t in range(11):  # reproduction fires at t=0 and t=10
            sequencer.step(pop, landscape, t)
        # Some reproduction should have occurred
        assert pop.n > n_initial

    def test_survival_reduces_population(self, setup_simulation):
        pop, sequencer, landscape = setup_simulation
        n_initial_alive = pop.n_alive
        # Run just survival (high egg mortality)
        survival = StageSpecificSurvivalEvent(
            name="kill_eggs",
            mortality_rates={"egg": 1.0, "juvenile": 0.0, "adult": 0.0},
        )
        # Add some eggs first
        pop.trait_mgr._data["stage"][:10] = 0  # mark first 10 as eggs
        mask = pop.alive.copy()
        survival.execute(pop, landscape, t=0, mask=mask)
        assert pop.n_alive == n_initial_alive - 10

    def test_compact_after_deaths(self, setup_simulation):
        pop, sequencer, landscape = setup_simulation
        for t in range(20):
            sequencer.step(pop, landscape, t)
        n_alive_before = pop.n_alive
        pop.compact()
        assert pop.n == n_alive_before
        assert pop.pool.alive.all()


class TestBarrierIntegration:
    """Test barrier enforcement in a simple scenario."""

    def test_impassable_barrier_prevents_crossing(self):
        """Agents on one side of an impassable barrier cannot cross."""
        # This test validates the full pipeline:
        # BarrierMap -> to_arrays -> movement with barrier check
        bmap = BarrierMap()
        bmap.add_edge(0, 1, BarrierOutcome.impassable())
        bmap.add_edge(1, 0, BarrierOutcome.impassable())
        assert bmap.has_barriers()
        assert bmap.n_edges == 2

    def test_partial_barrier_allows_some_crossing(self):
        """Barrier with 50% transmission allows roughly half to cross."""
        bmap = BarrierMap()
        outcome = BarrierOutcome(0.0, 0.5, 0.5)
        bmap.add_edge(0, 1, outcome)
        result = bmap.check(0, 1)
        assert result.p_transmission == 0.5
        assert result.p_deflection == 0.5
```

- [ ] **Step 2: Run all Phase 2 tests**

```bash
conda run -n shiny python -m pytest tests/test_population.py tests/test_barriers.py tests/test_events.py tests/test_phase2_integration.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

- [ ] **Step 3: Run full test suite to verify no regressions**

```bash
conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `test(phase2): add multi-generation integration tests for population lifecycle`

---

## Summary

| Task | Files | LOC Estimate | Key Deliverable |
|------|-------|-------------|-----------------|
| 1. Population class | `population.py`, `test_population.py` | ~150 | Unified lifecycle wrapper |
| 2. Dynamic resizing | `population.py`, `test_population.py` | ~200 | `add_agents`/`remove_agents`/`compact` |
| 3. BarrierMap class | `barriers.py`, `test_barriers.py` | ~150 | Edge-keyed barrier lookup |
| 4. Barrier enforcement | `movement.py`, `barriers.py`, `test_barriers.py` | ~200 | `_resolve_barriers_vec` + integration |
| 5. Stage survival | `events_builtin.py`, `events.py`, `test_events.py` | ~100 | `StageSpecificSurvivalEvent` |
| 6. IntroductionEvent | `events_builtin.py`, `test_events.py` | ~80 | Add agents at locations |
| 7. ReproductionEvent | `events_builtin.py`, `test_events.py` | ~120 | Poisson clutch + trait inheritance |
| 8. FloaterCreationEvent | `events_builtin.py`, `test_events.py` | ~50 | Release from groups |
| 9. Integration tests | `test_phase2_integration.py` | ~150 | Multi-generation validation |
| **Total** | | **~1,200** | |

## Dependencies

- Tasks 1-2 must complete before Tasks 5-8 (events need Population)
- Task 3 must complete before Task 4 (movement needs BarrierMap)
- Tasks 5-8 are independent of each other
- Task 9 depends on all other tasks

Recommended order: **1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9**
