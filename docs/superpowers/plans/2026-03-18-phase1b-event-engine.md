# Phase 1b: Event Engine and Salmon Model Refactor — Implementation Plan

> **STATUS: ✅ EXECUTED** — `salmon_ibm/events.py` (Event ABC + EventTrigger + EventSequencer) and `salmon_ibm/events_builtin.py` shipped. `EVENT_REGISTRY` and `@register_event` are live conventions (see CLAUDE.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded Simulation.step() with a configurable event sequencer, then refactor the salmon migration model as a sequence of events — preserving all existing behavior while enabling arbitrary event composition.

**Architecture:** The event engine introduces three layers: (1) an `Event` ABC with pluggable `EventTrigger` objects that control when events fire, (2) an `EventSequencer` that iterates over a user-defined list of events each timestep, and (3) concrete event wrappers (`MovementEvent`, `SurvivalEvent`, `AccumulateEvent`, `CustomEvent`) that delegate to existing salmon functions. `Simulation.step()` is refactored to build its current hardcoded sequence as a list of `Event` objects and delegate to `EventSequencer.step()`, preserving the public API and all existing test behavior.

**Tech Stack:** NumPy, ABC, dataclasses, PyYAML

**Test command:** `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `salmon_ibm/events.py` | Event ABC, EventTrigger, EventSequencer, EventGroup | **Create** |
| `salmon_ibm/events_builtin.py` | MovementEvent, SurvivalEvent, AccumulateEvent, CustomEvent | **Create** |
| `salmon_ibm/simulation.py` | Refactor to use EventSequencer internally | **Modify** |
| `tests/test_events.py` | Event engine unit tests | **Create** |
| `tests/test_simulation.py` | Existing tests must still pass (backward compat) | **Verify** |

---

### Task 1: Event ABC and EventTrigger

**Files:**
- Create: `salmon_ibm/events.py`

Define the base abstractions for the event system. All trigger types are implemented here. The `Event` base class uses `@dataclass` with an abstract `execute()` method.

- [ ] **Step 1: Write EventTrigger classes**

```python
# salmon_ibm/events.py
"""Event engine: base classes, triggers, and sequencer."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------

class EventTrigger(ABC):
    """Determines whether an event should fire at timestep t."""

    @abstractmethod
    def should_fire(self, t: int) -> bool: ...


class EveryStep(EventTrigger):
    """Fire every timestep."""

    def should_fire(self, t: int) -> bool:
        return True


@dataclass
class Once(EventTrigger):
    """Fire only at a specific timestep."""
    at: int

    def should_fire(self, t: int) -> bool:
        return t == self.at


@dataclass
class Periodic(EventTrigger):
    """Fire every *interval* steps, with optional offset."""
    interval: int
    offset: int = 0

    def should_fire(self, t: int) -> bool:
        return (t - self.offset) % self.interval == 0 and t >= self.offset


@dataclass
class Window(EventTrigger):
    """Fire only within [start, end) timestep range."""
    start: int
    end: int

    def should_fire(self, t: int) -> bool:
        return self.start <= t < self.end


@dataclass
class RandomTrigger(EventTrigger):
    """Fire with probability *p* each timestep."""
    p: float
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(), repr=False
    )

    def should_fire(self, t: int) -> bool:
        return self._rng.random() < self.p
```

- [ ] **Step 2: Write Event ABC**

Append to `salmon_ibm/events.py`:

```python
# ---------------------------------------------------------------------------
# Event base class
# ---------------------------------------------------------------------------

@dataclass
class Event(ABC):
    """Base class for all simulation events."""
    name: str
    trigger: EventTrigger = field(default_factory=EveryStep)
    trait_filter: dict | None = None

    @abstractmethod
    def execute(
        self,
        population,       # AgentPool
        landscape,        # mesh + env fields
        t: int,
        mask: np.ndarray, # bool[n_agents] — which agents this applies to
    ) -> None: ...
```

- [ ] **Step 3: Write unit tests for triggers**

```python
# tests/test_events.py
"""Unit tests for the event engine."""
import numpy as np
import pytest

from salmon_ibm.events import (
    EveryStep, Once, Periodic, Window, RandomTrigger,
)


class TestEveryStep:
    def test_always_fires(self):
        trigger = EveryStep()
        for t in range(20):
            assert trigger.should_fire(t) is True


class TestOnce:
    def test_fires_only_at_target(self):
        trigger = Once(at=5)
        assert trigger.should_fire(4) is False
        assert trigger.should_fire(5) is True
        assert trigger.should_fire(6) is False


class TestPeriodic:
    def test_fires_every_n_steps(self):
        trigger = Periodic(interval=3, offset=0)
        results = [trigger.should_fire(t) for t in range(10)]
        assert results == [True, False, False, True, False, False, True, False, False, True]

    def test_offset(self):
        trigger = Periodic(interval=4, offset=2)
        assert trigger.should_fire(0) is False
        assert trigger.should_fire(1) is False
        assert trigger.should_fire(2) is True
        assert trigger.should_fire(6) is True


class TestWindow:
    def test_fires_within_range(self):
        trigger = Window(start=3, end=6)
        results = [trigger.should_fire(t) for t in range(8)]
        assert results == [False, False, False, True, True, True, False, False]


class TestRandomTrigger:
    def test_probability_zero_never_fires(self):
        trigger = RandomTrigger(p=0.0)
        assert all(not trigger.should_fire(t) for t in range(100))

    def test_probability_one_always_fires(self):
        trigger = RandomTrigger(p=1.0)
        assert all(trigger.should_fire(t) for t in range(100))

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        t1 = RandomTrigger(p=0.5, _rng=rng1)
        t2 = RandomTrigger(p=0.5, _rng=rng2)
        results1 = [t1.should_fire(t) for t in range(50)]
        results2 = [t2.should_fire(t) for t in range(50)]
        assert results1 == results2
```

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_events.py -v`
Expected: All trigger tests PASS.

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/events.py tests/test_events.py
git commit -m "feat(events): add Event ABC and EventTrigger types"
```

---

### Task 2: EventSequencer

**Files:**
- Modify: `salmon_ibm/events.py`

The sequencer iterates over events, checks triggers, computes agent masks (combining `alive & ~arrived` with the event's `trait_filter`), and calls `execute()`.

- [ ] **Step 1: Implement EventSequencer**

Append to `salmon_ibm/events.py`:

```python
class EventSequencer:
    """Executes a list of events in order each timestep."""

    def __init__(self, events: list[Event]):
        self.events = events

    def step(self, population, landscape, t: int) -> None:
        """Run all events whose triggers fire at timestep *t*.

        Computes a 'step mask' (alive & ~arrived at step start) and stores
        it in ``landscape["step_alive_mask"]`` so events that need the
        beginning-of-step mask (e.g., behavior selection, timer updates)
        can use it instead of a freshly recomputed mask.
        """
        landscape["step_alive_mask"] = population.alive & ~population.arrived
        for event in self.events:
            if event.trigger.should_fire(t):
                mask = self._compute_mask(population, event.trait_filter)
                event.execute(population, landscape, t, mask)

    @staticmethod
    def _compute_mask(population, trait_filter: dict | None) -> np.ndarray:
        """Compute boolean mask: alive & ~arrived, optionally filtered by traits.

        Parameters
        ----------
        population : AgentPool
            The agent pool (must have .alive and .arrived bool arrays).
        trait_filter : dict | None
            Currently unused; reserved for future trait-based filtering.
            When None, returns alive & ~arrived.
        """
        base = population.alive & ~population.arrived
        if trait_filter is not None:
            # Future: apply trait-based filtering here
            # For now, trait_filter is a no-op placeholder
            pass
        return base
```

- [ ] **Step 2: Write tests for EventSequencer**

Append to `tests/test_events.py`:

```python
from salmon_ibm.events import Event, EventSequencer, EveryStep, Once


class StubEvent(Event):
    """Test double that records calls."""

    def __init__(self, name, trigger=None):
        super().__init__(name=name, trigger=trigger or EveryStep())
        self.calls = []

    def execute(self, population, landscape, t, mask):
        self.calls.append({"t": t, "mask_sum": int(mask.sum())})


class FakePopulation:
    """Minimal stand-in for AgentPool with alive/arrived arrays."""

    def __init__(self, n=10):
        self.n = n
        self.alive = np.ones(n, dtype=bool)
        self.arrived = np.zeros(n, dtype=bool)


class TestEventSequencer:
    def test_runs_events_in_order(self):
        call_order = []
        e1 = StubEvent("first")
        e2 = StubEvent("second")
        # Monkey-patch to track ordering
        orig_exec_1 = e1.execute
        orig_exec_2 = e2.execute
        def track1(*a, **kw):
            call_order.append("first")
            orig_exec_1(*a, **kw)
        def track2(*a, **kw):
            call_order.append("second")
            orig_exec_2(*a, **kw)
        e1.execute = track1
        e2.execute = track2

        seq = EventSequencer([e1, e2])
        seq.step(FakePopulation(), {}, t=0)
        assert call_order == ["first", "second"]

    def test_respects_trigger(self):
        e_every = StubEvent("every", EveryStep())
        e_once = StubEvent("once", Once(at=2))

        seq = EventSequencer([e_every, e_once])
        pop = FakePopulation()
        for t in range(5):
            seq.step(pop, {}, t)

        assert len(e_every.calls) == 5
        assert len(e_once.calls) == 1
        assert e_once.calls[0]["t"] == 2

    def test_mask_excludes_dead_and_arrived(self):
        e = StubEvent("check")
        pop = FakePopulation(n=10)
        pop.alive[0:3] = False
        pop.arrived[7:10] = True
        # alive & ~arrived = indices 3..6 → 4 agents

        seq = EventSequencer([e])
        seq.step(pop, {}, t=0)
        assert e.calls[0]["mask_sum"] == 4
```

- [ ] **Step 3: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_events.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/events.py tests/test_events.py
git commit -m "feat(events): add EventSequencer with trigger and mask logic"
```

---

### Task 3: EventGroup

**Files:**
- Modify: `salmon_ibm/events.py`

An `EventGroup` is itself an `Event` that contains sub-events. When executed, it runs its sub-events in order for `iterations` loops. This enables nested sequences (e.g., movement micro-steps, multi-phase reproduction).

- [ ] **Step 1: Implement EventGroup**

Append to `salmon_ibm/events.py`:

```python
@dataclass
class EventGroup(Event):
    """Container for nested event sequences with iteration count.

    When executed, runs sub_events in order for *iterations* loops.
    The group's own trigger controls whether it fires at all;
    sub-events' triggers are checked within each iteration.
    """
    sub_events: list[Event] = field(default_factory=list)
    iterations: int = 1

    def execute(self, population, landscape, t, mask):
        for _ in range(self.iterations):
            for event in self.sub_events:
                if event.trigger.should_fire(t):
                    sub_mask = self._compute_sub_mask(population, event.trait_filter, mask)
                    event.execute(population, landscape, t, sub_mask)

    @staticmethod
    def _compute_sub_mask(population, trait_filter, parent_mask):
        """Intersect parent mask with sub-event's trait filter."""
        child_mask = population.alive & ~population.arrived
        if trait_filter is not None:
            pass  # Future: apply trait-based filtering
        return parent_mask & child_mask
```

- [ ] **Step 2: Write tests for EventGroup**

Append to `tests/test_events.py`:

```python
from salmon_ibm.events import EventGroup


class TestEventGroup:
    def test_runs_sub_events(self):
        sub1 = StubEvent("sub1")
        sub2 = StubEvent("sub2")
        group = EventGroup(name="group", sub_events=[sub1, sub2])

        seq = EventSequencer([group])
        seq.step(FakePopulation(), {}, t=0)

        assert len(sub1.calls) == 1
        assert len(sub2.calls) == 1

    def test_iterations(self):
        sub = StubEvent("sub")
        group = EventGroup(name="group", sub_events=[sub], iterations=3)

        seq = EventSequencer([group])
        seq.step(FakePopulation(), {}, t=0)

        assert len(sub.calls) == 3

    def test_nested_trigger_respected(self):
        sub_every = StubEvent("every", EveryStep())
        sub_once = StubEvent("once", Once(at=5))
        group = EventGroup(name="group", sub_events=[sub_every, sub_once])

        seq = EventSequencer([group])
        seq.step(FakePopulation(), {}, t=0)

        assert len(sub_every.calls) == 1
        assert len(sub_once.calls) == 0  # t=0 != 5

    def test_group_trigger_gates_sub_events(self):
        sub = StubEvent("sub")
        group = EventGroup(name="group", trigger=Once(at=3), sub_events=[sub])

        seq = EventSequencer([group])
        for t in range(5):
            seq.step(FakePopulation(), {}, t)

        assert len(sub.calls) == 1  # only fired at t=3
```

- [ ] **Step 3: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_events.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/events.py tests/test_events.py
git commit -m "feat(events): add EventGroup for nested event sequences"
```

---

### Task 4: MovementEvent

**Files:**
- Create: `salmon_ibm/events_builtin.py`

Wrap the existing `execute_movement()` function as an event. The event receives its parameters (e.g., `n_micro_steps`, `cwr_threshold`) at construction time.

- [ ] **Step 1: Implement MovementEvent**

```python
# salmon_ibm/events_builtin.py
"""Built-in event types that wrap existing salmon IBM logic."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from salmon_ibm.events import Event, EveryStep, EventTrigger
from salmon_ibm.movement import execute_movement


@dataclass
class MovementEvent(Event):
    """Wraps execute_movement() as an event.

    Parameters
    ----------
    n_micro_steps : int
        Number of micro-steps per movement call.
    cwr_threshold : float
        Temperature threshold for cold-water refuge seeking.
    """
    n_micro_steps: int = 3
    cwr_threshold: float = 16.0

    def execute(self, population, landscape, t, mask):
        """Delegate to execute_movement().

        Note: execute_movement() computes its own alive mask internally,
        so we pass the full population and let it handle filtering.
        The *mask* parameter is available for future trait-based filtering.
        """
        mesh = landscape["mesh"]
        fields = landscape["fields"]
        rng = landscape["rng"]
        execute_movement(
            population, mesh, fields,
            seed=int(rng.integers(2**31)),
            n_micro_steps=self.n_micro_steps,
            cwr_threshold=self.cwr_threshold,
        )
```

- [ ] **Step 2: Write tests for MovementEvent**

Append to `tests/test_events.py`:

```python
from salmon_ibm.events_builtin import MovementEvent


class TestMovementEvent:
    def test_calls_execute_movement(self, mocker):
        """MovementEvent should delegate to execute_movement()."""
        mock_move = mocker.patch("salmon_ibm.events_builtin.execute_movement")
        rng = np.random.default_rng(42)
        pop = FakePopulation(n=5)
        landscape = {"mesh": object(), "fields": {}, "rng": rng}
        mask = np.ones(5, dtype=bool)

        event = MovementEvent(name="movement", n_micro_steps=3, cwr_threshold=16.0)
        event.execute(pop, landscape, t=0, mask=mask)

        mock_move.assert_called_once()
        call_args = mock_move.call_args
        assert call_args.kwargs.get("n_micro_steps", call_args[1].get("n_micro_steps")) == 3
```

- [ ] **Step 3: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_events.py::TestMovementEvent -v`
Expected: PASS. (Requires `pytest-mock` — verify it is installed or add to test deps.)

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/events_builtin.py tests/test_events.py
git commit -m "feat(events): add MovementEvent wrapping execute_movement()"
```

---

### Task 5: SurvivalEvent

**Files:**
- Modify: `salmon_ibm/events_builtin.py`

Wrap the existing thermal mortality and starvation check (lines 111-126 of `simulation.py`) as a single event.

- [ ] **Step 1: Implement SurvivalEvent**

Append to `salmon_ibm/events_builtin.py`:

```python
from salmon_ibm.bioenergetics import BioParams, update_energy
from salmon_ibm.estuary import salinity_cost


@dataclass
class SurvivalEvent(Event):
    """Bioenergetics energy update + thermal/starvation mortality.

    Combines:
    1. Wisconsin bioenergetics respiration (energy drain)
    2. Starvation mortality (ED < ED_MORTAL)
    3. Thermal mortality (temperature >= T_MAX)

    Parameters
    ----------
    bio_params : BioParams
        Bioenergetics parameters.
    thermal : bool
        Whether to apply thermal mortality.
    starvation : bool
        Whether to apply starvation mortality via bioenergetics.
    """
    bio_params: BioParams = field(default_factory=BioParams)
    thermal: bool = True
    starvation: bool = True

    def execute(self, population, landscape, t, mask):
        fields = landscape["fields"]
        activity_lut = landscape["activity_lut"]
        est_cfg = landscape.get("est_cfg", {})

        temps = fields["temperature"][population.tri_idx]
        alive = mask  # alive & ~arrived, computed by sequencer

        # Bioenergetics (starvation)
        if self.starvation and alive.any():
            activity = activity_lut[population.behavior]
            sal = fields.get("salinity", np.zeros(len(fields["temperature"])))
            sal_at_agents = sal[population.tri_idx]
            s_cfg = est_cfg.get("salinity_cost", {})
            sal_cost_arr = salinity_cost(
                sal_at_agents,
                S_opt=s_cfg.get("S_opt", 0.5),
                S_tol=s_cfg.get("S_tol", 6.0),
                k=s_cfg.get("k", 0.6),
            )

            new_ed, dead, new_mass = update_energy(
                population.ed_kJ_g[alive], population.mass_g[alive],
                temps[alive], activity[alive], sal_cost_arr[alive],
                self.bio_params,
            )
            population.ed_kJ_g[alive] = new_ed
            population.mass_g[alive] = new_mass
            dead_indices = np.where(alive)[0][dead]
            population.alive[dead_indices] = False

        # Thermal mortality
        if self.thermal:
            thermal_kill = alive & (temps >= self.bio_params.T_MAX)
            population.alive[thermal_kill] = False
```

- [ ] **Step 2: Write tests for SurvivalEvent**

Append to `tests/test_events.py`:

```python
from salmon_ibm.events_builtin import SurvivalEvent
from salmon_ibm.bioenergetics import BioParams


class TestSurvivalEvent:
    def _make_landscape(self, n_cells=20, temperature=15.0):
        """Minimal landscape dict for SurvivalEvent."""
        fields = {
            "temperature": np.full(n_cells, temperature),
        }
        # Activity LUT: all 1.0
        activity_lut = np.ones(5)
        return {"fields": fields, "activity_lut": activity_lut, "est_cfg": {}}

    def test_thermal_mortality(self):
        """Agents should die when temperature >= T_MAX."""
        pop = FakePopulation(n=5)
        pop.tri_idx = np.zeros(5, dtype=int)
        pop.ed_kJ_g = np.full(5, 6.5)
        pop.mass_g = np.full(5, 3500.0)
        pop.behavior = np.zeros(5, dtype=int)

        bio = BioParams(T_MAX=20.0)
        landscape = self._make_landscape(temperature=25.0)
        mask = pop.alive & ~pop.arrived

        event = SurvivalEvent(name="survival", bio_params=bio)
        event.execute(pop, landscape, t=0, mask=mask)
        assert not pop.alive.any(), "All agents should die at temp > T_MAX"

    def test_no_mortality_at_safe_temp(self):
        """Agents should survive at normal temperature with high energy."""
        pop = FakePopulation(n=5)
        pop.tri_idx = np.zeros(5, dtype=int)
        pop.ed_kJ_g = np.full(5, 6.5)
        pop.mass_g = np.full(5, 3500.0)
        pop.behavior = np.zeros(5, dtype=int)

        bio = BioParams(T_MAX=26.0)
        landscape = self._make_landscape(temperature=10.0)
        mask = pop.alive & ~pop.arrived

        event = SurvivalEvent(name="survival", bio_params=bio)
        event.execute(pop, landscape, t=0, mask=mask)
        assert pop.alive.all(), "All agents should survive at safe temperature"
```

- [ ] **Step 3: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_events.py::TestSurvivalEvent -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/events_builtin.py tests/test_events.py
git commit -m "feat(events): add SurvivalEvent with thermal and starvation mortality"
```

---

### Task 6: AccumulateEvent and CustomEvent

**Files:**
- Modify: `salmon_ibm/events_builtin.py`

`AccumulateEvent` runs a list of updater callables on population accumulators. `CustomEvent` is a generic callback wrapper for domain-specific logic (behavior selection, estuarine overrides, CWR counter updates, etc.).

- [ ] **Step 1: Implement AccumulateEvent**

Append to `salmon_ibm/events_builtin.py`:

```python
from typing import Callable


@dataclass
class AccumulateEvent(Event):
    """Runs updater functions that modify agent accumulators/state.

    Each updater is a callable with signature:
        updater(population, landscape, t, mask) -> None

    Updaters modify population state in-place (e.g., ed_kJ_g, mass_g).
    """
    updaters: list[Callable] = field(default_factory=list)

    def execute(self, population, landscape, t, mask):
        for updater in self.updaters:
            updater(population, landscape, t, mask)
```

- [ ] **Step 2: Implement CustomEvent**

Append to `salmon_ibm/events_builtin.py`:

```python
@dataclass
class CustomEvent(Event):
    """Generic event that delegates to a Python callback.

    The callback receives (population, landscape, t, mask) and performs
    arbitrary domain-specific logic. Used to wrap existing simulation
    methods (behavior selection, estuarine overrides, timer updates, etc.)
    without creating a separate Event subclass for each.
    """
    callback: Callable = field(default=lambda pop, land, t, mask: None)

    def execute(self, population, landscape, t, mask):
        self.callback(population, landscape, t, mask)
```

- [ ] **Step 3: Write tests**

Append to `tests/test_events.py`:

```python
from salmon_ibm.events_builtin import AccumulateEvent, CustomEvent


class TestAccumulateEvent:
    def test_runs_updaters_in_order(self):
        call_log = []

        def updater_a(pop, land, t, mask):
            call_log.append("a")

        def updater_b(pop, land, t, mask):
            call_log.append("b")

        event = AccumulateEvent(name="acc", updaters=[updater_a, updater_b])
        pop = FakePopulation()
        mask = pop.alive & ~pop.arrived
        event.execute(pop, {}, t=0, mask=mask)
        assert call_log == ["a", "b"]

    def test_updater_modifies_population(self):
        def increment_steps(pop, land, t, mask):
            pop.steps = getattr(pop, "steps", 0) + 1

        event = AccumulateEvent(name="acc", updaters=[increment_steps])
        pop = FakePopulation()
        pop.steps = 0
        mask = pop.alive & ~pop.arrived
        event.execute(pop, {}, t=0, mask=mask)
        assert pop.steps == 1


class TestCustomEvent:
    def test_calls_callback(self):
        calls = []

        def my_callback(pop, land, t, mask):
            calls.append(t)

        event = CustomEvent(name="custom", callback=my_callback)
        pop = FakePopulation()
        mask = pop.alive & ~pop.arrived
        event.execute(pop, {}, t=7, mask=mask)
        assert calls == [7]

    def test_callback_receives_correct_mask(self):
        received_masks = []

        def capture_mask(pop, land, t, mask):
            received_masks.append(mask.copy())

        pop = FakePopulation(n=10)
        pop.alive[0:3] = False  # 7 alive
        mask = pop.alive & ~pop.arrived

        event = CustomEvent(name="custom", callback=capture_mask)
        event.execute(pop, {}, t=0, mask=mask)
        assert received_masks[0].sum() == 7
```

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_events.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/events_builtin.py tests/test_events.py
git commit -m "feat(events): add AccumulateEvent and CustomEvent"
```

---

### Task 7: Refactor Simulation.step() to Use EventSequencer

**Files:**
- Modify: `salmon_ibm/simulation.py`

This is the critical refactoring task. The existing `Simulation.step()` method (lines 62-143) is replaced by an `EventSequencer` that runs the same operations in the same order. The `Simulation.__init__()` builds the event list. The public API (`step()`, `run()`, `close()`, `.history`, `.current_t`) remains identical.

**Strategy:** Each block in the current `step()` becomes either a `CustomEvent` wrapping a private method, or a built-in event type. The `landscape` dict bundles `mesh`, `env.fields`, `rng`, and other context needed by events.

- [ ] **Step 1: Add event imports and build_events() method**

Add to `simulation.py` imports:

```python
from salmon_ibm.events import EventSequencer, EveryStep
from salmon_ibm.events_builtin import (
    MovementEvent, SurvivalEvent, CustomEvent,
)
```

Add a `_build_events()` method to `Simulation`:

```python
def _build_events(self):
    """Build the default salmon migration event sequence.

    This replicates the original hardcoded step() logic as a
    configurable sequence of events.
    """
    return [
        CustomEvent(
            name="push_temperature",
            callback=self._event_push_temperature,
        ),
        CustomEvent(
            name="behavior_selection",
            callback=self._event_behavior_selection,
        ),
        CustomEvent(
            name="estuarine_overrides",
            callback=self._event_estuarine_overrides,
        ),
        CustomEvent(
            name="update_cwr_counters",
            callback=self._event_update_cwr_counters,
        ),
        MovementEvent(
            name="movement",
            n_micro_steps=3,
            cwr_threshold=self.beh_params.temp_bins[0],
        ),
        CustomEvent(
            name="update_timers",
            callback=self._event_update_timers,
        ),
        CustomEvent(
            name="bioenergetics",
            callback=self._event_bioenergetics,
        ),
        CustomEvent(
            name="logging",
            callback=self._event_logging,
        ),
    ]
```

- [ ] **Step 2: Extract each step() block into a private _event_* method**

Each method has the signature `(self, population, landscape, t, mask)` and performs exactly one block from the original `step()`. This is a mechanical extraction — no logic changes.

```python
def _event_push_temperature(self, population, landscape, t, mask):
    """Push current temperature into agent history."""
    temps_at_agents = landscape["fields"]["temperature"][population.tri_idx]
    population.push_temperature(temps_at_agents)

def _event_behavior_selection(self, population, landscape, t, mask):
    """Pick behaviors based on temperature and spawn timing.

    Uses ``landscape["step_alive_mask"]`` (computed at step start) to match
    the original step() behavior — the alive mask is NOT recomputed after
    earlier events may have killed agents.
    """
    step_mask = landscape["step_alive_mask"]
    t3h = population.t3h_mean()
    population.behavior[step_mask] = pick_behaviors(
        t3h[step_mask], population.target_spawn_hour[step_mask],
        self.beh_params, seed=int(self._rng.integers(2**31)),
    )
    population.behavior = apply_overrides(population, self.beh_params)

def _event_estuarine_overrides(self, population, landscape, t, mask):
    """Apply seiche pause and DO overrides."""
    self._apply_estuarine_overrides()

def _event_update_cwr_counters(self, population, landscape, t, mask):
    """Update cold-water refuge tracking counters."""
    self._update_cwr_counters()

def _event_update_timers(self, population, landscape, t, mask):
    """Increment step counters and decrement spawn timers.

    Uses ``landscape["step_alive_mask"]`` to match the original step()
    behavior where timers use the beginning-of-step alive mask.
    """
    step_mask = landscape["step_alive_mask"]
    population.steps[step_mask] += 1
    population.target_spawn_hour[step_mask] = np.maximum(
        population.target_spawn_hour[step_mask] - 1, 0
    )

def _event_bioenergetics(self, population, landscape, t, mask):
    """Run bioenergetics and thermal/starvation mortality."""
    fields = landscape["fields"]
    temps_at_agents = fields["temperature"][population.tri_idx]

    activity = self._activity_lut[population.behavior]
    sal = fields.get("salinity", np.zeros(self.mesh.n_triangles))
    sal_at_agents = sal[population.tri_idx]
    s_cfg = self.est_cfg.get("salinity_cost", {})
    sal_cost = salinity_cost(
        sal_at_agents,
        S_opt=s_cfg.get("S_opt", 0.5),
        S_tol=s_cfg.get("S_tol", 6.0),
        k=s_cfg.get("k", 0.6),
    )

    alive = population.alive & ~population.arrived
    if alive.any():
        new_ed, dead, new_mass = update_energy(
            population.ed_kJ_g[alive], population.mass_g[alive],
            temps_at_agents[alive], activity[alive], sal_cost[alive],
            self.bio_params,
        )
        population.ed_kJ_g[alive] = new_ed
        population.mass_g[alive] = new_mass
        dead_indices = np.where(alive)[0][dead]
        population.alive[dead_indices] = False

    thermal_kill = alive & (temps_at_agents >= self.bio_params.T_MAX)
    population.alive[thermal_kill] = False

def _event_logging(self, population, landscape, t, mask):
    """Log step data and append to history."""
    if self.logger:
        self.logger.log_step(t, population)

    summary = {
        "time": t,
        "n_alive": int(population.alive.sum()),
        "n_arrived": int(population.arrived.sum()),
        "mean_ed": float(population.ed_kJ_g[population.alive].mean())
            if population.alive.any() else 0.0,
        "mean_mass": float(population.mass_g[population.alive].mean())
            if population.alive.any() else 0.0,
        "behavior_counts": {
            int(b): int((population.behavior[population.alive] == b).sum())
            for b in range(5)
        },
    }
    self.history.append(summary)
```

- [ ] **Step 3: Replace step() with sequencer delegation**

Replace the entire `step()` method body:

```python
def step(self):
    t = self.current_t
    self.env.advance(t)

    landscape = {
        "mesh": self.mesh,
        "fields": self.env.fields,
        "rng": self._rng,
        "activity_lut": self._activity_lut,
        "est_cfg": self.est_cfg,
    }

    self._sequencer.step(self.pool, landscape, t)
    self.current_t += 1
```

- [ ] **Step 4: Initialize sequencer in __init__()**

Add at the end of `__init__()`, after `self.history = []`:

```python
self._sequencer = EventSequencer(self._build_events())
```

- [ ] **Step 5: Run ALL existing tests to verify backward compatibility**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

Expected: **ALL tests pass.** Specifically:
- `tests/test_simulation.py` — all 13 tests PASS unchanged
- `tests/test_movement.py` — all tests PASS
- `tests/test_bioenergetics.py` — all tests PASS
- `tests/test_behavior.py` — all tests PASS
- `tests/test_estuary.py` — all tests PASS
- `tests/test_events.py` — all tests PASS

**This is the most critical step.** If any test fails, the refactoring has changed behavior. Debug by comparing the event method output against the original `step()` logic line-by-line.

- [ ] **Step 6: Run reproducibility test specifically**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py::test_simulation_reproducibility -v`

This test verifies that `rng_seed=42` produces identical agent positions and energy values across two runs. If this fails, the RNG call order has changed during refactoring (likely the `_rng.integers()` calls are happening in a different sequence).

**Debugging tip:** If reproducibility breaks, ensure that each `_event_*` method calls `self._rng.integers()` exactly the same number of times and in the same order as the original `step()`.

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/simulation.py
git commit -m "refactor(simulation): replace hardcoded step() with EventSequencer

Simulation.step() now delegates to an EventSequencer that runs the
same operations in the same order. All existing tests pass unchanged.
The public API (step, run, close, history) is preserved."
```

---

### Task 8: Backward Compatibility Verification

**Files:**
- No changes — verification only

This task is a dedicated verification step to confirm the refactoring is behavior-preserving.

- [ ] **Step 1: Run the full test suite**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

Every test listed here must PASS:

| Test file | Expected |
|-----------|----------|
| `tests/test_simulation.py` | 13 tests PASS |
| `tests/test_movement.py` | All PASS |
| `tests/test_bioenergetics.py` | All PASS |
| `tests/test_behavior.py` | All PASS |
| `tests/test_estuary.py` | All PASS |
| `tests/test_events.py` | All PASS (new) |
| `tests/test_agents.py` | All PASS |
| `tests/test_config.py` | All PASS |
| `tests/test_environment.py` | All PASS |
| `tests/test_mesh.py` | All PASS |
| `tests/test_integration.py` | All PASS |

- [ ] **Step 2: Run performance benchmark**

Run: `conda run -n shiny python -m pytest tests/test_perf.py -v -s`

The refactoring should not degrade performance. EventSequencer overhead is one Python `for` loop over 8 events — negligible compared to the NumPy operations inside each event.

- [ ] **Step 3: Verify the Simulation public API is unchanged**

Manually verify these attributes and methods still exist and work:
- `Simulation(config, n_agents, data_dir, rng_seed, output_path)`
- `sim.step()`
- `sim.run(n_steps)`
- `sim.close()`
- `sim.current_t`
- `sim.pool`
- `sim.history`
- `sim.mesh`
- `sim.env`
- `sim.beh_params`
- `sim.bio_params`
- `sim.est_cfg`
- `sim._apply_estuarine_overrides()` — still accessible for `test_simulation.py` tests
- `sim._update_cwr_counters()` — still accessible for `test_simulation.py` tests

---

### Task 9: YAML Event Sequence Loading

**Files:**
- Modify: `salmon_ibm/events.py` (add `load_events_from_config`)
- Modify: `salmon_ibm/simulation.py` (optional: load from config if `events:` key present)
- Modify: `tests/test_events.py` (add YAML loading tests)

This enables users to configure event order from YAML config files. The default salmon sequence is used when no `events:` key is present (backward compatible).

- [ ] **Step 1: Define the YAML schema**

The `events:` key in a config file maps to a list of event descriptors:

```yaml
events:
  - type: custom
    name: push_temperature
  - type: custom
    name: behavior_selection
  - type: custom
    name: estuarine_overrides
  - type: custom
    name: update_cwr_counters
  - type: movement
    name: movement
    params:
      n_micro_steps: 3
  - type: custom
    name: update_timers
  - type: custom
    name: bioenergetics
  - type: custom
    name: logging
```

- [ ] **Step 2: Implement EventFactory**

Add to `salmon_ibm/events.py`:

```python
# At module level
EVENT_REGISTRY: dict[str, type[Event]] = {}


def register_event(type_name: str):
    """Decorator to register an Event subclass under a type name."""
    def decorator(cls):
        EVENT_REGISTRY[type_name] = cls
        return cls
    return decorator


def load_events_from_config(
    event_defs: list[dict],
    callback_registry: dict[str, Callable] | None = None,
) -> list[Event]:
    """Build an event list from YAML event definitions.

    Parameters
    ----------
    event_defs : list[dict]
        Each dict has keys: type, name, and optionally params, trigger.
    callback_registry : dict[str, Callable] | None
        For 'custom' events, maps name -> callback function.

    Returns
    -------
    list[Event]
        Instantiated events in config order.
    """
    events = []
    callback_registry = callback_registry or {}

    for defn in event_defs:
        event_type = defn["type"]
        name = defn.get("name", event_type)
        params = defn.get("params", {})
        trigger = _parse_trigger(defn.get("trigger"))

        if event_type == "custom":
            cb = callback_registry.get(name)
            if cb is None:
                raise ValueError(
                    f"No callback registered for custom event '{name}'. "
                    f"Available: {list(callback_registry.keys())}"
                )
            from salmon_ibm.events_builtin import CustomEvent
            events.append(CustomEvent(name=name, trigger=trigger, callback=cb))

        elif event_type in EVENT_REGISTRY:
            cls = EVENT_REGISTRY[event_type]
            events.append(cls(name=name, trigger=trigger, **params))

        else:
            raise ValueError(
                f"Unknown event type '{event_type}'. "
                f"Registered types: {list(EVENT_REGISTRY.keys())}"
            )

    return events


def _parse_trigger(trigger_def: dict | None) -> EventTrigger:
    """Parse a trigger definition from YAML."""
    if trigger_def is None:
        return EveryStep()

    kind = trigger_def.get("type", "every_step")
    if kind == "every_step":
        return EveryStep()
    elif kind == "once":
        return Once(at=trigger_def["at"])
    elif kind == "periodic":
        return Periodic(
            interval=trigger_def["interval"],
            offset=trigger_def.get("offset", 0),
        )
    elif kind == "window":
        return Window(start=trigger_def["start"], end=trigger_def["end"])
    elif kind == "random":
        return RandomTrigger(p=trigger_def["p"])
    else:
        raise ValueError(f"Unknown trigger type: {kind}")
```

- [ ] **Step 3: Register built-in events**

In `salmon_ibm/events_builtin.py`, add the `@register_event` decorator:

```python
from salmon_ibm.events import register_event

@register_event("movement")
@dataclass
class MovementEvent(Event):
    ...

@register_event("survival")
@dataclass
class SurvivalEvent(Event):
    ...

@register_event("accumulate")
@dataclass
class AccumulateEvent(Event):
    ...
```

- [ ] **Step 4: Update Simulation to optionally load events from config**

In `Simulation.__init__()`, after building the default events:

```python
def _build_events(self):
    """Build the event sequence from config or use the default salmon sequence."""
    # If config has an 'events' key, load from YAML
    event_defs = self.config.get("events")
    if event_defs is not None:
        callback_registry = self._build_callback_registry()
        return load_events_from_config(event_defs, callback_registry)

    # Default: hardcoded salmon migration sequence
    return [
        # ... same as Task 7
    ]

def _build_callback_registry(self):
    """Map custom event names to their callback methods."""
    return {
        "push_temperature": self._event_push_temperature,
        "behavior_selection": self._event_behavior_selection,
        "estuarine_overrides": self._event_estuarine_overrides,
        "update_cwr_counters": self._event_update_cwr_counters,
        "update_timers": self._event_update_timers,
        "bioenergetics": self._event_bioenergetics,
        "logging": self._event_logging,
    }
```

- [ ] **Step 5: Write tests for YAML loading**

Append to `tests/test_events.py`:

```python
from salmon_ibm.events import load_events_from_config, EVENT_REGISTRY


class TestLoadEventsFromConfig:
    def test_loads_custom_event(self):
        calls = []
        def my_cb(pop, land, t, mask):
            calls.append(t)

        defs = [{"type": "custom", "name": "my_cb"}]
        events = load_events_from_config(defs, {"my_cb": my_cb})
        assert len(events) == 1
        assert events[0].name == "my_cb"

    def test_loads_movement_event(self):
        defs = [{"type": "movement", "name": "move", "params": {"n_micro_steps": 5}}]
        events = load_events_from_config(defs)
        assert len(events) == 1
        assert events[0].n_micro_steps == 5

    def test_unknown_type_raises(self):
        defs = [{"type": "nonexistent", "name": "bad"}]
        with pytest.raises(ValueError, match="Unknown event type"):
            load_events_from_config(defs)

    def test_missing_custom_callback_raises(self):
        defs = [{"type": "custom", "name": "missing"}]
        with pytest.raises(ValueError, match="No callback registered"):
            load_events_from_config(defs, {})

    def test_trigger_parsing(self):
        defs = [{
            "type": "custom", "name": "x",
            "trigger": {"type": "periodic", "interval": 5, "offset": 2},
        }]
        events = load_events_from_config(defs, {"x": lambda *a: None})
        assert events[0].trigger.should_fire(2) is True
        assert events[0].trigger.should_fire(3) is False
        assert events[0].trigger.should_fire(7) is True
```

- [ ] **Step 6: Run ALL tests**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

Expected: ALL PASS. Existing configs without `events:` key use the default sequence (backward compatible).

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/events.py salmon_ibm/events_builtin.py salmon_ibm/simulation.py tests/test_events.py
git commit -m "feat(events): add YAML event sequence loading with EventFactory

Users can now configure event order via the 'events:' key in YAML config.
Custom events reference callbacks by name. Built-in events (movement,
survival, accumulate) are loaded from the registry. Existing configs
without 'events:' key continue to use the default salmon sequence."
```

---

## Summary of Deliverables

| Deliverable | Files | Tests |
|-------------|-------|-------|
| Event ABC + 5 trigger types | `salmon_ibm/events.py` | `tests/test_events.py::TestEveryStep` through `TestRandomTrigger` |
| EventSequencer | `salmon_ibm/events.py` | `tests/test_events.py::TestEventSequencer` |
| EventGroup | `salmon_ibm/events.py` | `tests/test_events.py::TestEventGroup` |
| MovementEvent | `salmon_ibm/events_builtin.py` | `tests/test_events.py::TestMovementEvent` |
| SurvivalEvent | `salmon_ibm/events_builtin.py` | `tests/test_events.py::TestSurvivalEvent` |
| AccumulateEvent | `salmon_ibm/events_builtin.py` | `tests/test_events.py::TestAccumulateEvent` |
| CustomEvent | `salmon_ibm/events_builtin.py` | `tests/test_events.py::TestCustomEvent` |
| Refactored Simulation.step() | `salmon_ibm/simulation.py` | All existing `tests/test_simulation.py` tests |
| YAML event loading | `salmon_ibm/events.py` | `tests/test_events.py::TestLoadEventsFromConfig` |

## Critical Constraints

1. **ALL existing tests must pass unchanged** after the refactoring. The test files `test_simulation.py`, `test_movement.py`, `test_bioenergetics.py`, `test_behavior.py`, and `test_estuary.py` are not modified.

2. **RNG call order must be preserved** in the refactored `step()`. The `test_simulation_reproducibility` test will catch any deviation.

3. **The Simulation public API does not change.** External callers use `sim.step()`, `sim.run(n)`, `sim.close()` exactly as before.

4. **Performance must not degrade.** The event engine adds one Python `for` loop over ~8 events per timestep — negligible overhead. Verify with `test_perf.py`.
