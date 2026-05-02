# Hatchery vs Wild Distinction — Tier C1 Implementation Plan

> **STATUS: ✅ EXECUTED 2026-04-30** — All 7 tasks complete. salmon_ibm/origin.py module created; AgentPool.ARRAY_FIELDS extended; Population.add_agents and both introduction events propagate origin; network.py preserves on transfer; OutputLogger 7-touch wire-through complete; YAML loader handles string→int conversion. Full pytest suite green at 829 passing. Spec at docs/superpowers/specs/2026-04-30-hatchery-origin-c1-design.md. Closes the first of three planned tiers (C1 → C2 → C3); next: C2 BalticHatcheryBioParams parameter divergence.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track origin (wild vs hatchery) on every agent as int8 metadata; thread it through introduction events, inter-population transfer, OutputLogger CSV export, and YAML scenario loading. No physics or behaviour change — C1 is the scaffold for C2 (parameter divergence) and C3 (behaviour divergence) to build on.

**Architecture:** New `salmon_ibm/origin.py` module with `Origin` IntEnum + module constants + `ORIGIN_NAMES` tuple (mirrors `DOState` precedent in `salmon_ibm/estuary.py:61-69`). New `origin: int8` field on `AgentPool.ARRAY_FIELDS` propagated through `Population.add_agents()`, both introduction events, inter-population transfer, and OutputLogger. YAML `origin: hatchery` strings parsed via `ORIGIN_NAMES.index()` in `scenario_loader._build_single_event` before the existing setattr loop.

**Tech Stack:** Python 3.10+, NumPy, dataclasses, IntEnum from stdlib enum; pytest for testing; conda env `shiny`.

**Spec:** [`docs/superpowers/specs/2026-04-30-hatchery-origin-c1-design.md`](../specs/2026-04-30-hatchery-origin-c1-design.md) (commits `967ac74` + `718f3c6`).

---

## File structure

**Modified files (9 total: 1 new + 8 modified, plus 1 new test file):**

Production code (8):
- `salmon_ibm/origin.py` — **new file** (~15 lines)
- `salmon_ibm/agents.py` — `ARRAY_FIELDS` tuple + `__init__` (3-touch field addition)
- `salmon_ibm/population.py` — `add_agents()` signature + bulk-prealloc block
- `salmon_ibm/events_builtin.py` — `IntroductionEvent` gains `origin` field
- `salmon_ibm/events_hexsim.py` — `PatchIntroductionEvent` gains `origin` field
- `salmon_ibm/network.py` — `MultiPopulationManager` transfer preservation (next to existing `natal_reach_id` line)
- `salmon_ibm/output.py` — `OutputLogger` 7 touch points
- `salmon_ibm/scenario_loader.py` — `_build_single_event` adds string→int conversion before setattr loop

Tests (1 new file):
- `tests/test_origin.py` — 8 new tests

**Test runner:**
```bash
micromamba run -n shiny python -m pytest tests/path/file.py::test_name -v
# whole suite
micromamba run -n shiny python -m pytest tests/ -v
```

Suite is ~14 minutes. Baseline before this plan: 821 passing on `main` (post-v1.7.3). Expected after: **829 passing** (821 + 8 new tests, 0 regressions).

**Commit cadence (5 commits):**
- Tasks 1-2 → commit 1 (origin module + agent storage)
- Task 3 → commit 2 (event propagation)
- Task 4 → commit 3 (network preservation)
- Task 5 → commit 4 (OutputLogger wire-through)
- Tasks 6-7 → commit 5 (YAML loader + final stamp)

Branch: `hatchery-origin-c1` (created from `main` at the start of Task 1).

---

## Tasks

### Task 1: Create `salmon_ibm/origin.py` module (TDD)

**Files:**
- Create: `salmon_ibm/origin.py`
- Create: `tests/test_origin.py`

This task ships the new module with three module-level tests. Mirrors the `DOState` precedent in `salmon_ibm/estuary.py:61-69` — IntEnum class plus module-constant aliases plus a names tuple for serialization.

- [ ] **Step 1.1: Create the branch**

```bash
git switch main
git pull origin main
git checkout -b hatchery-origin-c1
```

- [ ] **Step 1.2: Create failing tests**

Create `tests/test_origin.py`:

```python
"""Tests for the Origin enum and module constants (Tier C1)."""
import pytest


def test_origin_enum_values():
    """Origin enum values match the int8 column convention (WILD=0, HATCHERY=1)."""
    from salmon_ibm.origin import Origin
    assert Origin.WILD == 0
    assert Origin.HATCHERY == 1


def test_origin_names_roundtrip():
    """ORIGIN_NAMES index aligns with Origin enum values."""
    from salmon_ibm.origin import Origin, ORIGIN_NAMES
    assert ORIGIN_NAMES.index("wild") == 0
    assert ORIGIN_NAMES.index("hatchery") == 1
    assert ORIGIN_NAMES[Origin.WILD] == "wild"
    assert ORIGIN_NAMES[Origin.HATCHERY] == "hatchery"


def test_origin_names_invalid_raises():
    """Unknown origin string raises ValueError via list.index."""
    from salmon_ibm.origin import ORIGIN_NAMES
    with pytest.raises(ValueError):
        ORIGIN_NAMES.index("salmon")
```

- [ ] **Step 1.3: Run tests; expect 3 failures**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py -v
```

Expected: 3 failed with `ModuleNotFoundError: No module named 'salmon_ibm.origin'`.

- [ ] **Step 1.4: Create the origin module**

Create `salmon_ibm/origin.py`:

```python
"""Origin enum for tracking wild vs hatchery agents.

Tier C1 of the hatchery-vs-wild plan (see
docs/superpowers/specs/2026-04-30-hatchery-origin-c1-design.md).
Used as int8 metadata on AgentPool agents; no behaviour change in
C1 — physiology divergence ships in C2.

Mirrors the DOState precedent in salmon_ibm/estuary.py:61-69 — IntEnum
class plus module-constant aliases plus a names tuple for YAML / CSV
serialization.
"""
from enum import IntEnum


class Origin(IntEnum):
    WILD = 0
    HATCHERY = 1


ORIGIN_WILD = Origin.WILD
ORIGIN_HATCHERY = Origin.HATCHERY
ORIGIN_NAMES = ("wild", "hatchery")  # index aligns with enum value
```

- [ ] **Step 1.5: Run tests; expect 3 passes**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py -v
```

Expected: `3 passed`.

---

### Task 2: Add `origin` field to `AgentPool` and `Population.add_agents` (TDD)

**Files:**
- Modify: `salmon_ibm/agents.py` (top imports + `ARRAY_FIELDS` tuple + `__init__`)
- Modify: `salmon_ibm/population.py` (`add_agents` signature + bulk-prealloc block)
- Modify: `tests/test_origin.py` (append 2 new tests)

Three-touch field addition: declare in `ARRAY_FIELDS`, initialize in `__init__`, extend in `add_agents`. The defensive assertions at `agents.py:80-86` and `population.py:240-245` will catch a forgotten step. Then `Population.add_agents` accepts an `origin` kwarg.

- [ ] **Step 2.1: Append two new tests**

Append to `tests/test_origin.py`:

```python
def test_agent_pool_origin_default_wild():
    """A freshly-allocated AgentPool has all agents tagged as WILD (0)."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.origin import ORIGIN_WILD
    pool = AgentPool(n=10, start_tri=0, rng_seed=42)
    assert pool.origin.shape == (10,)
    assert pool.origin.dtype == np.int8
    assert (pool.origin == ORIGIN_WILD).all()


def test_population_add_agents_with_origin():
    """add_agents(origin=ORIGIN_HATCHERY) writes 1 to new agents only."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY
    pool = AgentPool(n=3, start_tri=0, rng_seed=42)
    # Population is a @dataclass with `name` as first required field.
    pop = Population(name="test", pool=pool)
    new_idx = pop.add_agents(
        n=2,
        positions=np.array([0, 0]),
        origin=ORIGIN_HATCHERY,
    )
    # Existing 3 agents stay WILD
    assert (pool.origin[:3] == ORIGIN_WILD).all()
    # New 2 agents are HATCHERY
    assert (pool.origin[new_idx] == ORIGIN_HATCHERY).all()
```

- [ ] **Step 2.2: Run new tests; expect 2 failures**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py::test_agent_pool_origin_default_wild tests/test_origin.py::test_population_add_agents_with_origin -v
```

Expected: 2 failed with `AttributeError: 'AgentPool' object has no attribute 'origin'` and similar.

- [ ] **Step 2.3: Add `origin` to `AgentPool.ARRAY_FIELDS`**

In `salmon_ibm/agents.py`, locate the `ARRAY_FIELDS` tuple (around line 21-35) and append `"origin"` after `"exit_branch_id"`:

```python
    ARRAY_FIELDS = (
        "tri_idx",
        "mass_g",
        "ed_kJ_g",
        "target_spawn_hour",
        "behavior",
        "cwr_hours",
        "hours_since_cwr",
        "steps",
        "alive",
        "arrived",
        "temp_history",
        "natal_reach_id",   # int8: cell's reach_id at introduction; -1 if pre-tagging
        "exit_branch_id",   # int8: first delta-branch reach_id touched; sticky; -1 if never
        "origin",           # int8: 0=wild, 1=hatchery; permanent at introduction
    )
```

- [ ] **Step 2.4: Add origin import + initialization to `AgentPool.__init__`**

In `salmon_ibm/agents.py`, add to the top imports (around line 1-15):

```python
from salmon_ibm.origin import ORIGIN_WILD
```

In `AgentPool.__init__`, immediately after the existing `self.exit_branch_id` initialization (around line 71), add:

```python
        self.origin = np.full(n, ORIGIN_WILD, dtype=np.int8)
```

- [ ] **Step 2.5: Update `Population.add_agents` signature and bulk-prealloc block**

In `salmon_ibm/population.py`, locate the `add_agents` method (line 192-200). The current signature has keyword-only args after the `*` separator:

```python
    def add_agents(
        self,
        n: int,
        positions: np.ndarray,
        *,
        mass_g=None,
        ed_kJ_g: float = 6.5,
        group_id: int = -1,
    ) -> np.ndarray:
```

Add `origin: int = ORIGIN_WILD,` immediately after `group_id: int = -1,` (i.e., as another keyword-only arg, matching the existing pattern):

```python
    def add_agents(
        self,
        n: int,
        positions: np.ndarray,
        *,
        mass_g=None,
        ed_kJ_g: float = 6.5,
        group_id: int = -1,
        origin: int = ORIGIN_WILD,
    ) -> np.ndarray:
```

Then in the bulk-prealloc block (around line 219-231), add the origin extension line immediately after `new_arrays["exit_branch_id"][old_n:] = -1`:

```python
        new_arrays["origin"][old_n:] = origin
```

Add to the top imports of `population.py`:

```python
from salmon_ibm.origin import ORIGIN_WILD
```

- [ ] **Step 2.6: Run the 5 origin tests; expect 5 passes**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py -v
```

Expected: `5 passed`.

- [ ] **Step 2.7: Run smoke checks against AgentPool + Population**

Quick check that the defensive assertions still pass:

```bash
micromamba run -n shiny python -m pytest tests/test_population.py -v
```

Expected: all population tests pass (the assertions at `agents.py:80-86` and `population.py:240-245` would raise `AssertionError` if the new field isn't initialized or extended properly).

- [ ] **Step 2.8: Commit Tasks 1-2**

```bash
git add salmon_ibm/origin.py salmon_ibm/agents.py salmon_ibm/population.py tests/test_origin.py
git commit -m "feat(origin): track wild vs hatchery agents (Tier C1, Tasks 1-2)

New module salmon_ibm/origin.py with Origin IntEnum + module constants
(ORIGIN_WILD, ORIGIN_HATCHERY) + ORIGIN_NAMES tuple for serialization.
Mirrors the DOState precedent in salmon_ibm/estuary.py.

AgentPool gains origin int8 column (default 0 = WILD); 3-touch
addition: ARRAY_FIELDS tuple, __init__, Population.add_agents extension
block. The defensive assertions in agents.py and population.py would
catch a forgotten step.

Population.add_agents accepts origin: int = ORIGIN_WILD keyword;
existing callers (with no kwarg) silently default to WILD, matching
the scope-OUT decisions for ReproductionEvent / Phase-3 seedling.

Tests: 5 new in tests/test_origin.py covering enum values, names
roundtrip, invalid-string raises, default-WILD on pool init, and
add_agents kwarg propagation.

Spec: docs/superpowers/specs/2026-04-30-hatchery-origin-c1-design.md
This is the first of three planned tiers (C1 → C2 → C3)."
```

---

### Task 3: Propagate `origin` through both introduction events (TDD)

**Files:**
- Modify: `salmon_ibm/events_builtin.py` (`IntroductionEvent` dataclass + execute call site)
- Modify: `salmon_ibm/events_hexsim.py` (`PatchIntroductionEvent` dataclass + execute call site)
- Modify: `tests/test_origin.py` (append 1 new test)

`IntroductionEvent` and `PatchIntroductionEvent` both call `population.add_agents(...)`. Each gains an `origin: int = ORIGIN_WILD` dataclass field that is passed through. ReproductionEvent (`events_builtin.py:306`) and Phase-3 seedling (`events_phase3.py:295`) intentionally don't propagate origin per the spec's scope-OUT — they default to WILD.

- [ ] **Step 3.1: Append the propagation test**

Append to `tests/test_origin.py`:

```python
def test_introduction_event_propagates_origin():
    """IntroductionEvent(origin=ORIGIN_HATCHERY) tags the new agents
    as hatchery in the population's origin column."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.events_builtin import IntroductionEvent
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY

    pool = AgentPool(n=2, start_tri=0, rng_seed=42)
    # Population is a @dataclass with `name` as first required field.
    pop = Population(name="test", pool=pool)
    landscape = {"rng": np.random.default_rng(0)}

    # Event base class requires `name`; trigger defaults to EveryStep.
    evt = IntroductionEvent(
        name="intro_test",
        n_agents=3,
        positions=[0],
        origin=ORIGIN_HATCHERY,
    )
    evt.execute(pop, landscape, t=0, mask=None)

    # Pre-existing 2 agents stay WILD
    assert (pool.origin[:2] == ORIGIN_WILD).all()
    # New 3 agents are HATCHERY
    assert (pool.origin[2:5] == ORIGIN_HATCHERY).all()
```

- [ ] **Step 3.2: Run the new test; expect failure**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py::test_introduction_event_propagates_origin -v
```

Expected: FAILED with `TypeError: ...IntroductionEvent.__init__() got an unexpected keyword argument 'origin'`.

- [ ] **Step 3.3: Add `origin` field to `IntroductionEvent`**

In `salmon_ibm/events_builtin.py`, add the import to the top of the file (find the existing `from salmon_ibm.estuary import salinity_cost, EstuaryParams` line and add a sibling line):

```python
from salmon_ibm.origin import ORIGIN_WILD
```

Modify the `IntroductionEvent` dataclass (around line 198-263) to add the `origin` field. Place it after `initialization_spatial_data`:

```python
@register_event("introduction")
@dataclass
class IntroductionEvent(Event):
    """Add new individuals to the population."""

    n_agents: int = 10
    positions: list[int] = field(default_factory=lambda: [0])
    initial_mass_mean: float = 3500.0
    initial_mass_std: float = 500.0
    initial_ed: float = 6.5
    initial_traits: dict[str, str] = field(default_factory=dict)
    initial_accumulators: dict[str, float] = field(default_factory=dict)
    initialization_spatial_data: str = ""
    origin: int = ORIGIN_WILD
```

In the `execute` method, find the `population.add_agents(...)` call (around line 247) and add `origin=self.origin`:

```python
        new_idx = population.add_agents(
            n, pos_arr, mass_g=mass, ed_kJ_g=self.initial_ed,
            origin=self.origin,
        )
```

- [ ] **Step 3.4: Add `origin` field to `PatchIntroductionEvent`**

In `salmon_ibm/events_hexsim.py`, add the import to the top of the file:

```python
from salmon_ibm.origin import ORIGIN_WILD
```

Modify the `PatchIntroductionEvent` dataclass (around line 395-421) to add the `origin` field:

```python
@register_event("patch_introduction")
@dataclass
class PatchIntroductionEvent(Event):
    """Place one agent on every non-zero cell of a named spatial data layer."""

    patch_spatial_data: str = ""
    origin: int = ORIGIN_WILD
```

In the `execute` method (around line 418), update the `add_agents` call:

```python
        new_idx = population.add_agents(
            len(nonzero_cells),
            nonzero_cells,
            origin=self.origin,
        )
```

- [ ] **Step 3.5: Run the propagation test; expect pass**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py::test_introduction_event_propagates_origin -v
```

Expected: `1 passed`.

- [ ] **Step 3.6: Run wider regression check**

```bash
micromamba run -n shiny python -m pytest tests/test_events.py tests/test_origin.py -v
```

Expected: all event-related tests still pass; the 6 origin tests pass.

- [ ] **Step 3.7: Commit Task 3**

```bash
git add salmon_ibm/events_builtin.py salmon_ibm/events_hexsim.py tests/test_origin.py
git commit -m "feat(origin): propagate origin through introduction events (Task 3)

IntroductionEvent and PatchIntroductionEvent each gain an origin: int
= ORIGIN_WILD dataclass field that is passed to population.add_agents.

ReproductionEvent (events_builtin.py:306) and the Phase-3 seedling
event (events_phase3.py:295) intentionally don't propagate origin —
they call add_agents without the kwarg, so offspring/seedlings default
to WILD per the spec's scope-OUT decisions.

Test: test_introduction_event_propagates_origin verifies that
IntroductionEvent(origin=ORIGIN_HATCHERY).execute() tags the new
agents."
```

---

### Task 4: Preserve origin on inter-population transfer (TDD)

**Files:**
- Modify: `salmon_ibm/network.py` (after the existing `natal_reach_id` preservation block, around line 199-201)
- Modify: `tests/test_origin.py` (append 1 new test)

`MultiPopulationManager.execute` already preserves `natal_reach_id` and `exit_branch_id` from source to target on transfer. Origin must follow the same pattern — without this, a hatchery agent transferring populations would silently revert to WILD, breaking ensemble post-processing.

- [ ] **Step 4.1: Append the preservation test**

Append to `tests/test_origin.py`:

```python
def test_origin_preserved_on_population_transfer():
    """Origin must persist when an agent transfers between populations
    via MultiPopulationManager. Without this, a hatchery agent moving
    populations would silently reset to WILD."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.interactions import MultiPopulationManager
    from salmon_ibm.network import SwitchPopulationEvent
    from salmon_ibm.origin import ORIGIN_HATCHERY

    # Source: 3 hatchery agents
    src_pool = AgentPool(n=3, start_tri=0, rng_seed=42)
    src_pool.origin[:] = ORIGIN_HATCHERY
    src = Population(name="src", pool=src_pool)

    # Target: empty
    dst_pool = AgentPool(n=0, start_tri=0, rng_seed=43)
    dst = Population(name="dst", pool=dst_pool)

    mpm = MultiPopulationManager()
    mpm.register(src)
    mpm.register(dst)

    # SwitchPopulationEvent looks up source/target by name from
    # landscape["multi_pop_mgr"] (verified at network.py:177).
    landscape = {"rng": np.random.default_rng(0), "multi_pop_mgr": mpm}
    # Event base class requires `name`; trigger defaults to EveryStep.
    evt = SwitchPopulationEvent(
        name="transfer_test",
        source_pop="src",
        target_pop="dst",
        transfer_probability=1.0,
    )
    mask = np.ones(3, dtype=bool)
    evt.execute(src, landscape, t=0, mask=mask)

    # All source agents transferred; target should now have 3 agents,
    # all tagged HATCHERY (origin must have been carried over).
    assert dst.pool.n == 3
    assert (dst.pool.origin[:3] == ORIGIN_HATCHERY).all()
```

- [ ] **Step 4.2: Run the test; expect failure**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py::test_origin_preserved_on_population_transfer -v
```

Expected: FAILED — the dst pool's origin column will be all 0 (WILD) because the existing transfer code doesn't preserve origin.

(If the test fails for a different reason — e.g., `ImportError`, `TypeError` — the engineer should adjust the test setup to match the actual `network.py` API rather than the implementation. Report BLOCKED if the API can't be matched.)

- [ ] **Step 4.3: Add origin preservation in `network.py`**

In `salmon_ibm/network.py`, find the existing preservation block (around line 199-201):

```python
        if hasattr(target, "natal_reach_id") and hasattr(source, "natal_reach_id"):
            target.natal_reach_id[new_idx] = source.natal_reach_id[transfer]
            target.exit_branch_id[new_idx] = source.exit_branch_id[transfer]
```

Add origin preservation immediately after, mirroring the same defensive `hasattr` guard:

```python
        if hasattr(target, "origin") and hasattr(source, "origin"):
            target.origin[new_idx] = source.origin[transfer]
```

(The hasattr guards mirror the existing block — both are defensive against alternative Population implementations that might not have these fields, e.g. test stand-ins.)

- [ ] **Step 4.4: Run the test; expect pass**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py::test_origin_preserved_on_population_transfer -v
```

Expected: `1 passed`.

- [ ] **Step 4.5: Run network/transfer regression check**

```bash
micromamba run -n shiny python -m pytest tests/test_network.py tests/test_interactions.py tests/test_origin.py -v
```

Expected: all pass. Existing network tests didn't create non-WILD agents, so they couldn't have caught a regression — the new test 8 is the only one that verifies origin preservation.

- [ ] **Step 4.6: Commit Task 4**

```bash
git add salmon_ibm/network.py tests/test_origin.py
git commit -m "feat(origin): preserve origin on inter-population transfer (Task 4)

MultiPopulationManager.execute already preserves natal_reach_id and
exit_branch_id from source to target on transfer; origin now follows
the same pattern. Without this, a hatchery agent transferring
populations would silently reset to WILD.

The hasattr guard mirrors the existing pattern — defensive against
alternative Population implementations (test stand-ins, etc.).

Test: test_origin_preserved_on_population_transfer locks in the
invariant. Existing network tests don't create non-WILD agents, so
this test is the only safeguard against regressing this code path."
```

---

### Task 5: Wire `origin` through `OutputLogger` (7 touch points)

**Files:**
- Modify: `salmon_ibm/output.py` (7 touch points)

OutputLogger has parallel preallocated and list-based code paths (per the v1.7.1 deep-review fix). Each tracked agent column needs 7 edits: 2 in `__init__` (alloc per branch), 2 in `log_step` (assign per branch), 1 in `to_dataframe` empty_cols, 2 in `to_dataframe` filled-DataFrame dicts (preallocated + list). No new test needed for this — existing OutputLogger tests round-trip through every tracked column; if `origin` is in the schema, those tests verify it.

- [ ] **Step 5.1: Add `__init__` allocations (preallocated + list branches)**

In `salmon_ibm/output.py`, find the preallocated branch in `__init__` (around line 41-42, next to `_natal_reach_id_arr` and `_exit_branch_id_arr`). Add:

```python
            self._origin_arr = np.empty((max_steps, max_agents), dtype=np.int8)
```

In the same `__init__`, find the list branch (around line 53-54). Add:

```python
            self._origin: list[np.ndarray] = []
```

- [ ] **Step 5.2: Add `log_step` assignments (preallocated + list branches)**

In `log_step` preallocated branch (around line 80, next to `self._natal_reach_id_arr[r, :n] = ...`). Add:

```python
            self._origin_arr[r, :n] = pool.origin[:n]
```

In `log_step` list branch (around line 92, next to `self._natal_reach_id.append(...)`). Add:

```python
            self._origin.append(pool.origin.copy())
```

- [ ] **Step 5.3: Update `to_dataframe` empty_cols list**

Find `to_dataframe` (around line 95). The `empty_cols` list (around line 96-100) currently ends with `"natal_reach_id", "exit_branch_id",`. Append `"origin"` after `"exit_branch_id"`:

```python
        empty_cols = [
            "time", "agent_id", "tri_idx", "lat", "lon",
            "ed_kJ_g", "behavior", "alive", "arrived",
            "natal_reach_id", "exit_branch_id", "origin",
        ]
```

- [ ] **Step 5.4: Update `to_dataframe` preallocated DataFrame dict**

In the preallocated branch of `to_dataframe` (around line 107-119), add the `origin` key/value after the `exit_branch_id` line:

```python
                parts.append(pd.DataFrame({
                    "time": self._times_arr[r, :n],
                    "agent_id": self._agent_ids_arr[r, :n],
                    "tri_idx": self._tri_idxs_arr[r, :n],
                    "lat": self._lats_arr[r, :n],
                    "lon": self._lons_arr[r, :n],
                    "ed_kJ_g": self._eds_arr[r, :n],
                    "behavior": self._behaviors_arr[r, :n],
                    "alive": self._alive_arr[r, :n],
                    "arrived": self._arrived_arr[r, :n],
                    "natal_reach_id": self._natal_reach_id_arr[r, :n],
                    "exit_branch_id": self._exit_branch_id_arr[r, :n],
                    "origin": self._origin_arr[r, :n],
                }))
```

- [ ] **Step 5.5: Update `to_dataframe` list DataFrame dict**

In the list branch of `to_dataframe` (around line 123-137), add the `origin` key/value after the `exit_branch_id` line:

```python
        return pd.DataFrame(
            {
                "time": np.concatenate(self._times),
                "agent_id": np.concatenate(self._agent_ids),
                "tri_idx": np.concatenate(self._tri_idxs),
                "lat": np.concatenate(self._lats),
                "lon": np.concatenate(self._lons),
                "ed_kJ_g": np.concatenate(self._eds),
                "behavior": np.concatenate(self._behaviors),
                "alive": np.concatenate(self._alive),
                "arrived": np.concatenate(self._arrived),
                "natal_reach_id": np.concatenate(self._natal_reach_id),
                "exit_branch_id": np.concatenate(self._exit_branch_id),
                "origin": np.concatenate(self._origin),
            }
        )
```

- [ ] **Step 5.6: Run OutputLogger regression check**

```bash
micromamba run -n shiny python -m pytest tests/test_output.py tests/test_origin.py -v
```

(If `tests/test_output.py` does not exist in the project, run a broader check that exercises OutputLogger via integration tests:)

```bash
micromamba run -n shiny python -m pytest tests/ -v -k "output or logger or census"
```

Expected: all OutputLogger-touching tests pass; the 7 origin tests pass.

- [ ] **Step 5.7: Commit Task 5**

```bash
git add salmon_ibm/output.py
git commit -m "feat(origin): export origin column in OutputLogger (Task 5)

Seven touch points wire origin through OutputLogger's parallel
preallocated and list-based code paths (per the v1.7.1 deep-review
preallocation work):

  1. __init__ preallocated: allocate self._origin_arr int8 buffer
  2. __init__ list: declare self._origin: list[np.ndarray]
  3. log_step preallocated: copy pool.origin[:n] into row r
  4. log_step list: append pool.origin.copy()
  5. to_dataframe empty_cols: append 'origin' string
  6. to_dataframe preallocated dict: add 'origin' column
  7. to_dataframe list dict: add 'origin' column

CSV format is int8 values (0 / 1) for performance and Pandas
convention; downstream consumers can map via ORIGIN_NAMES if they want
human-readable strings. Decision recorded in spec.

No new test — existing OutputLogger round-trip tests would surface a
regression if the schema diverges between empty/filled paths."
```

---

### Task 6: YAML scenario loader string→int conversion (TDD)

**Files:**
- Modify: `salmon_ibm/scenario_loader.py` (`_build_single_event`, just before the existing `for key, val in params.items()` loop)
- Modify: `tests/test_origin.py` (append 1 new test)

YAML scenarios may carry `origin: hatchery` (string) under the event's `params` block. The existing `_build_single_event` loop applies params via `setattr(evt, key, val)` — without conversion, `setattr` would store the string in the int-typed dataclass field, breaking downstream NumPy ops. Convert before the loop runs.

- [ ] **Step 6.1: Append the YAML test**

Append to `tests/test_origin.py` (single test covering both happy path and error path, matching the spec's wording for test 7):

```python
def test_yaml_origin_string_parses():
    """YAML 'origin: hatchery' string is converted to int8 by the
    scenario loader; invalid strings raise ValueError at load time."""
    import pytest
    from salmon_ibm.scenario_loader import ScenarioLoader
    from salmon_ibm.origin import ORIGIN_HATCHERY

    loader = ScenarioLoader()
    # Happy path: 'hatchery' converts to int8
    edef = {
        "type": "introduction",
        "name": "intro_hatchery",
        "params": {"n_agents": 5, "origin": "hatchery"},
    }
    evt = loader._build_single_event(edef)
    assert evt is not None
    assert evt.origin == ORIGIN_HATCHERY

    # Error path: invalid string raises ValueError at load time
    bad_edef = {
        "type": "introduction",
        "name": "intro_bad",
        "params": {"n_agents": 5, "origin": "salmon"},
    }
    with pytest.raises(ValueError, match="origin"):
        loader._build_single_event(bad_edef)
```

(If `ScenarioLoader` requires constructor arguments to instantiate, the engineer should mirror an existing test pattern from `tests/test_scenario_loader.py` for setup. The test's purpose is to exercise the conversion path inside `_build_single_event`.)

- [ ] **Step 6.2: Run the new test; expect failure**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py::test_yaml_origin_string_parses -v
```

Expected: FAILED — `evt.origin` is the string `"hatchery"`, not the int `1`. (The `pytest.raises` block won't run because the assert before it fails first.)

- [ ] **Step 6.3: Add the conversion block to `_build_single_event`**

In `salmon_ibm/scenario_loader.py`, locate `_build_single_event` (around line 223). Find the line `params = edef.get("params", {})` (around line 317).

Immediately before the `for key, val in params.items()` loop (around line 318), insert:

```python
        # Convert origin string ("wild"/"hatchery") to int8. Surfaces
        # invalid values at scenario-load time, not first simulation
        # step. See spec
        # docs/superpowers/specs/2026-04-30-hatchery-origin-c1-design.md.
        if "origin" in params and isinstance(params["origin"], str):
            from salmon_ibm.origin import ORIGIN_NAMES
            s = params["origin"]
            try:
                params["origin"] = ORIGIN_NAMES.index(s)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid origin '{s}'; expected one of {ORIGIN_NAMES}"
                ) from exc
```

- [ ] **Step 6.4: Run the YAML test; expect pass**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py::test_yaml_origin_string_parses -v
```

Expected: 1 passed.

- [ ] **Step 6.5: Run the full origin test file**

```bash
micromamba run -n shiny python -m pytest tests/test_origin.py -v
```

Expected: 8 passed (spec-faithful — both happy path and error path are inside the single `test_yaml_origin_string_parses` test).

- [ ] **Step 6.6: Run scenario-loader regression**

```bash
micromamba run -n shiny python -m pytest tests/test_scenario_loader.py tests/test_config.py -v
```

Expected: all pass — old-schema YAML configs without `origin` parse normally (the new conversion block is gated on `"origin" in params`).

---

### Task 7: Full pytest suite + final commit + push

**Files:**
- Modify: `docs/superpowers/plans/2026-04-30-hatchery-origin-c1.md` (add ✅ EXECUTED stamp)

Run the whole suite to surface regressions; if green, stamp the plan and push.

- [ ] **Step 7.1: Run full pytest suite**

```bash
micromamba run -n shiny python -m pytest tests/ -q --no-header
```

Expected (~14 minutes): `829 passed, 34 skipped, 7 deselected, 1 xfailed`. Zero failures.

- [ ] **Step 7.2: If failures, triage**

If failures appear, classify and fix:
- **(A) AssertionError from agents.py:80-86**: a step skipped `__init__` initialization. Fix in Task 2's diff.
- **(B) AssertionError from population.py:240-245**: a step skipped `add_agents` extension. Fix in Task 2's diff.
- **(C) TypeError on event construction**: a step missed propagating `origin` to `add_agents`. Fix in Task 3's diff.
- **(D) DataFrame schema mismatch in OutputLogger tests**: empty_cols out of sync with filled paths. Fix in Task 5's diff.
- **(E) Unexpected regression in unrelated test**: should not happen since C1 is metadata-only. If it does, escalate.

Re-run the suite after each fix; commit fixes as separate small commits with `fix(origin): ...` messages.

- [ ] **Step 7.3: Stamp the plan as ✅ EXECUTED**

In `docs/superpowers/plans/2026-04-30-hatchery-origin-c1.md`, replace the line:

```markdown
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
```

With:

```markdown
> **STATUS: ✅ EXECUTED 2026-MM-DD** — All 7 tasks complete. salmon_ibm/origin.py module created; AgentPool.ARRAY_FIELDS extended; Population.add_agents and both introduction events propagate origin; network.py preserves on transfer; OutputLogger 7-touch wire-through complete; YAML loader handles string→int conversion. Full pytest suite green at NNN passing. Spec at docs/superpowers/specs/2026-04-30-hatchery-origin-c1-design.md. Closes the first of three planned tiers (C1 → C2 → C3); next: C2 BalticHatcheryBioParams parameter divergence.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
```

Replace `2026-MM-DD` with the actual completion date and `NNN` with the verified passing count from Step 7.1.

- [ ] **Step 7.4: Final commit**

```bash
git add docs/superpowers/plans/2026-04-30-hatchery-origin-c1.md tests/test_origin.py salmon_ibm/scenario_loader.py
git commit -m "feat(origin): YAML loader string conversion + plan EXECUTED (Tasks 6-7)

scenario_loader._build_single_event now converts string origin values
('wild'/'hatchery') to int8 before applying params via setattr. Invalid
strings raise ValueError at scenario-load time, not first simulation
step.

Tier C1 of the hatchery-vs-wild plan complete. Suite at NNN passing.
Closes the first of three planned tiers (C1 -> C2 -> C3); next plan
will be C2 BalticHatcheryBioParams parameter divergence."
```

(Update `NNN` to match the actual count.)

- [ ] **Step 7.5: Push the branch and open PR**

```bash
git push -u origin hatchery-origin-c1
```

Then open a PR via `gh pr create` (matching the v1.7.3 osmoregulation PR pattern). Title: `Hatchery vs wild C1: tag-only origin tracking`. Body should reference the spec, list the 9 modified files, summarize the test count delta, and note that C2/C3 are deferred.

- [ ] **Step 7.6: Update memory after merge**

Once the PR is merged + tagged + deployed, update `~/.claude/projects/.../memory/curonian_h3_grid_state.md` to reflect the new deployed version, and update `~/.claude/projects/.../memory/curonian_deferred.md` to mark hatchery-vs-wild as "C1 RESOLVED, C2/C3 still queued."

---

## Plan summary

- **7 tasks**, **5 commits** (Tasks 1-2 → commit 1, Task 3 → commit 2, Task 4 → commit 3, Task 5 → commit 4, Tasks 6-7 → commit 5).
- **9 files modified** (1 new + 8 modified) plus 1 new test file.
- **+8 net tests** (matches spec exactly).
- **Estimated time:** 1-2 days. All work is mechanical (3-touch agent-field, 7-touch OutputLogger, 2 event class updates, 1 network preservation line, 1 YAML conversion block).
- **Risk profile:** very low — C1 is metadata-only; no agent takes a different action because of origin.
- **No backward-compat shim** needed — origin defaults to WILD everywhere, so old YAML configs without origin keys continue to work.

## Spec coverage check

| Spec section | Implementing task |
|---|---|
| `salmon_ibm/origin.py` (Origin IntEnum + constants + ORIGIN_NAMES) | Task 1 |
| `AgentPool.ARRAY_FIELDS` extension + `__init__` initialization | Task 2 |
| `Population.add_agents()` signature + extension block | Task 2 |
| `IntroductionEvent` origin field | Task 3 |
| `PatchIntroductionEvent` origin field | Task 3 |
| `network.py` MultiPopulationManager origin preservation | Task 4 |
| `OutputLogger` 7-touch wire-through | Task 5 |
| `scenario_loader._build_single_event` string→int conversion | Task 6 |
| 8 new tests (test_origin.py) | Tasks 1, 2, 3, 4, 6 |
| Plan stamp on completion | Task 7.3 |
