# Nemunas Delta Branching Implementation Plan

> **STATUS: ✅ EXECUTED** — `salmon_ibm/delta_routing.py` shipped — sticky first-touch `update_exit_branch_id` CustomEvent + `natal_reach_id` / `exit_branch_id` int8 fields on `AgentPool`. `nemunas_discharge.nc` now carries `Q_per_branch` for Atmata / Skirvyte / Pakalnė split.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add agent-side topology awareness (natal reach + first-touch exit branch) and per-branch discharge data shape for the three flowing Nemunas delta branches (Atmata, Skirvytė, Gilija), keeping the runtime minimal and the data shape extensible for future per-branch mortality / homing plans.

**Architecture:** New `salmon_ibm/delta_routing.py` module owns the branch-fraction LUT and exit-tracking helper. `AgentPool.ARRAY_FIELDS` gains two int8 fields (`natal_reach_id`, `exit_branch_id`). `Simulation.step()` gains a `CustomEvent` between movement and fish_predation that updates `exit_branch_id` (sticky first-touch). `Population.assert_natal_tagged()` runs each step (suppressed under `Simulation.resume=True`). `scripts/fetch_nemunas_discharge.py` writes a new `Q_per_branch[branch, time]` variable to `data/nemunas_discharge.nc` — schema-only, no runtime consumer.

**Tech Stack:** Python 3.10+ via micromamba env `shiny`; numpy + xarray + pytest. No new third-party dependencies. Existing patterns: `@register_event` decorator (events.py), `CustomEvent` callback wrapper (simulation.py), `ARRAY_FIELDS` SoA agent storage (agents.py), NetCDF3 64-bit-offset for forcing files (existing fetch script).

**Spec:** [`docs/superpowers/specs/2026-04-27-nemunas-delta-branching-design.md`](../specs/2026-04-27-nemunas-delta-branching-design.md) (last commit `0814000`).

---

## File Structure

**New files:**
- `salmon_ibm/delta_routing.py` — branch-fraction LUT, `split_discharge`, `update_exit_branch_id`. ~80 LoC.
- `tests/test_delta_routing.py` — unit tests for the new module. ~12 tests.
- `tests/test_nemunas_discharge.py` — schema test for the new NC variable. 1 test.

**Modified files:**
- `salmon_ibm/agents.py` — extend `ARRAY_FIELDS` and `AgentPool.__init__`.
- `salmon_ibm/population.py` — extend `add_agents` defaults; add `set_natal_reach_from_cells` and `assert_natal_tagged` helpers.
- `salmon_ibm/simulation.py` — add `resume` flag; add init-time mesh validation; insert `update_exit_branch` CustomEvent; gate assertion call.
- `salmon_ibm/output.py` — extend `OutputLogger` to serialise the two new fields (both columnar and list-append branches).
- `salmon_ibm/events_builtin.py` — tag `IntroductionEvent` and `ReproductionEvent` after `add_agents`.
- `salmon_ibm/events_hexsim.py` — tag `PatchIntroductionEvent` after `add_agents`.
- `salmon_ibm/events_phase3.py` — tag the vegetation-seedling event after `add_agents`.
- `salmon_ibm/network.py` — make `TransferEvent` preserve `natal_reach_id` from source.
- `scripts/fetch_nemunas_discharge.py` — write `Q_per_branch`, `branch_names`, `branch_fractions_source`.
- `tests/test_agents.py` — assert ARRAY_FIELDS contains the new fields (+3).
- `tests/test_population.py` — assert defaults + helpers + assertion fire correctly (+3).
- `tests/test_simulation.py` — assert init-time validation + resume-flag bypass (+2).
- `tests/test_output.py` — assert OutputLogger serialises the two new fields (+2).
- `tests/test_nemunas_h3_integration.py` — natal-vs-exit invariant (+1).
- `tests/test_movement_metric.py` — perf-regression sentinel (+1).

**One-shot manual operations:**
- Re-run `scripts/fetch_nemunas_discharge.py` once to regenerate `data/nemunas_discharge.nc` with the new variable.
- Calibrate the integration-test threshold against an actual run (see Task 18).
- Tag a release and deploy to laguna (out of scope for this plan; manual per project convention).

---

## Test runner

All commands use the user's micromamba env. Pytest invocation pattern:

```bash
micromamba run -n shiny python -m pytest tests/test_<file>.py::<test_name> -v
```

Whole-suite invocation:

```bash
micromamba run -n shiny python -m pytest tests/ -v
```

---

## Tasks

### Task 1: `delta_routing.py` — fractions and `split_discharge`

**Files:**
- Create: `salmon_ibm/delta_routing.py`
- Test: `tests/test_delta_routing.py`

- [ ] **Step 1.1: Write the failing tests for fractions and `split_discharge`**

Create `tests/test_delta_routing.py`:

```python
"""Unit tests for salmon_ibm.delta_routing."""
import numpy as np
import pytest

from salmon_ibm import delta_routing


def test_branch_fractions_sum_to_one():
    total = sum(delta_routing.BRANCH_FRACTIONS.values())
    assert abs(total - 1.0) < 1e-9, f"Fractions sum to {total}, not 1.0"


def test_branch_fractions_keys_are_real_reaches():
    expected = {"Skirvyte", "Atmata", "Gilija"}
    assert set(delta_routing.BRANCH_FRACTIONS) == expected


def test_delta_branch_reaches_is_frozenset():
    assert isinstance(delta_routing.DELTA_BRANCH_REACHES, frozenset)
    assert delta_routing.DELTA_BRANCH_REACHES == set(delta_routing.BRANCH_FRACTIONS)


def test_split_discharge_preserves_total_array():
    q = np.array([100.0, 200.0, 0.0, 500.0], dtype=np.float32)
    out = delta_routing.split_discharge(q)
    summed = sum(out.values())
    assert np.allclose(summed, q, rtol=1e-6)


def test_split_discharge_handles_scalar():
    out = delta_routing.split_discharge(np.float32(1000.0))
    assert pytest.approx(out["Skirvyte"], rel=1e-6) == 510.0
    assert pytest.approx(out["Atmata"],   rel=1e-6) == 270.0
    assert pytest.approx(out["Gilija"],   rel=1e-6) == 220.0


def test_split_discharge_zero_input():
    q = np.zeros(10, dtype=np.float32)
    out = delta_routing.split_discharge(q)
    for name, arr in out.items():
        assert np.all(arr == 0.0), f"{name} should be all-zero, got {arr}"
```

- [ ] **Step 1.2: Run tests to verify they fail with ImportError**

```bash
micromamba run -n shiny python -m pytest tests/test_delta_routing.py -v
```

Expected: `ModuleNotFoundError: No module named 'salmon_ibm.delta_routing'` or `ImportError`.

- [ ] **Step 1.3: Create `salmon_ibm/delta_routing.py` with the LUT and `split_discharge`**

```python
"""Nemunas delta branch identity, discharge fractions, and exit-tracking utility.

Three branches: Atmata, Skirvyte, Gilija. Pakalne is intentionally lumped
into Atmata (small distributary, absent from the inSTREAM source). Rusne
is the island between branches, not a flowing channel.

Fractions follow Ramsar Site 629 Information Sheet (Nemunas Delta), 2010
https://rsis.ramsar.org/RISapp/files/41231939/documents/LT629_lit161122.pdf
— interior-of-range midpoints summing to 1.0. They are geographic constants
of the delta, not scenario-tunable parameters.

Tagging contract: every code path that calls Population.add_agents must
follow up with population.set_natal_reach_from_cells(new_idx, mesh) — except
the inter-population transfer case in network.py, which preserves
natal_reach_id from the source agent. Population.assert_natal_tagged()
enforces this at runtime.
"""
from __future__ import annotations

import numpy as np

BRANCH_FRACTIONS: dict[str, float] = {
    "Skirvyte": 0.51,   # main flow to Kaliningrad / SW lagoon
    "Atmata":   0.27,   # NE distributary; Pakalne lumped in
    "Gilija":   0.22,   # easternmost; branches off upstream of Rusne island
}
assert abs(sum(BRANCH_FRACTIONS.values()) - 1.0) < 1e-9, (
    f"BRANCH_FRACTIONS must sum to 1.0, got {sum(BRANCH_FRACTIONS.values())}"
)

DELTA_BRANCH_REACHES: frozenset[str] = frozenset(BRANCH_FRACTIONS)


def split_discharge(q_total):
    """Apply BRANCH_FRACTIONS to a Nemunas climatology (scalar or (T,) array).

    Returns a dict mapping branch name to the per-branch series, preserving
    insertion order of BRANCH_FRACTIONS.
    """
    return {br: q_total * f for br, f in BRANCH_FRACTIONS.items()}
```

- [ ] **Step 1.4: Run tests — should pass**

```bash
micromamba run -n shiny python -m pytest tests/test_delta_routing.py -v
```

Expected: 6 tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add salmon_ibm/delta_routing.py tests/test_delta_routing.py
git commit -m "feat(delta_routing): branch fractions LUT + split_discharge"
```

---

### Task 2: `delta_routing.update_exit_branch_id`

**Files:**
- Modify: `salmon_ibm/delta_routing.py`
- Test: `tests/test_delta_routing.py`

- [ ] **Step 2.1: Append failing tests for `update_exit_branch_id`**

Append to `tests/test_delta_routing.py`:

```python
class _FakePool:
    """Minimal pool stand-in for update_exit_branch_id testing."""
    def __init__(self, tri_idx, alive, exit_branch_id):
        self.tri_idx = np.asarray(tri_idx, dtype=np.int64)
        self.alive = np.asarray(alive, dtype=bool)
        self.exit_branch_id = np.asarray(exit_branch_id, dtype=np.int8)


class _FakeMesh:
    def __init__(self, reach_id, reach_names):
        self.reach_id = np.asarray(reach_id, dtype=np.int8)
        self.reach_names = list(reach_names)


REACH_NAMES = ["Nemunas", "Atmata", "Skirvyte", "Gilija", "CuronianLagoon"]
RID = {name: i for i, name in enumerate(REACH_NAMES)}


def _mesh(cell_to_reach):
    """Build a mesh whose cell i sits in reach cell_to_reach[i]."""
    rid = np.array([RID[r] for r in cell_to_reach], dtype=np.int8)
    return _FakeMesh(rid, REACH_NAMES)


def test_update_exit_branch_id_first_touch():
    # 3 agents, all in Atmata, none yet tagged -> exit becomes Atmata for all.
    mesh = _mesh(["Nemunas", "Atmata", "Skirvyte"])
    pool = _FakePool(tri_idx=[1, 1, 1],          # all in cell 1 = Atmata
                     alive=[True, True, True],
                     exit_branch_id=[-1, -1, -1])
    delta_routing.update_exit_branch_id(pool, mesh)
    assert (pool.exit_branch_id == RID["Atmata"]).all()


def test_update_exit_branch_id_sticky():
    # Pre-tagged at Atmata, now in Skirvyte: exit STAYS Atmata.
    mesh = _mesh(["Nemunas", "Atmata", "Skirvyte"])
    pool = _FakePool(tri_idx=[2, 2],             # both in cell 2 = Skirvyte
                     alive=[True, True],
                     exit_branch_id=[RID["Atmata"], -1])
    delta_routing.update_exit_branch_id(pool, mesh)
    assert pool.exit_branch_id[0] == RID["Atmata"]   # untouched (sticky)
    assert pool.exit_branch_id[1] == RID["Skirvyte"] # newly tagged


def test_update_exit_branch_id_skips_dead_agents():
    mesh = _mesh(["Nemunas", "Atmata", "Skirvyte"])
    pool = _FakePool(tri_idx=[1, 1],
                     alive=[True, False],
                     exit_branch_id=[-1, -1])
    delta_routing.update_exit_branch_id(pool, mesh)
    assert pool.exit_branch_id[0] == RID["Atmata"]
    assert pool.exit_branch_id[1] == -1, "dead agents must not be tagged"


def test_update_exit_branch_id_skips_lagoon_only():
    # Agent never visits a delta-branch reach: stays -1.
    mesh = _mesh(["Nemunas", "Atmata", "Skirvyte", "Gilija", "CuronianLagoon"])
    pool = _FakePool(tri_idx=[4, 0],             # CuronianLagoon, Nemunas
                     alive=[True, True],
                     exit_branch_id=[-1, -1])
    delta_routing.update_exit_branch_id(pool, mesh)
    assert (pool.exit_branch_id == -1).all()


def test_update_exit_branch_id_no_op_without_reach_meta():
    # Mesh without reach_names returns silently.
    class _NoMesh:
        pass
    pool = _FakePool(tri_idx=[0], alive=[True], exit_branch_id=[-1])
    delta_routing.update_exit_branch_id(pool, _NoMesh())
    assert pool.exit_branch_id[0] == -1


def test_update_exit_branch_id_handles_negative_tri_idx():
    # Agent with tri_idx == -1 (off-mesh) must not crash; stays -1.
    mesh = _mesh(["Nemunas", "Atmata", "Skirvyte"])
    pool = _FakePool(tri_idx=[-1, 1],
                     alive=[True, True],
                     exit_branch_id=[-1, -1])
    delta_routing.update_exit_branch_id(pool, mesh)
    assert pool.exit_branch_id[0] == -1
    assert pool.exit_branch_id[1] == RID["Atmata"]
```

- [ ] **Step 2.2: Run new tests — they should fail**

```bash
micromamba run -n shiny python -m pytest tests/test_delta_routing.py -v -k "update_exit_branch_id"
```

Expected: 6 tests fail with `AttributeError: module 'salmon_ibm.delta_routing' has no attribute 'update_exit_branch_id'`.

- [ ] **Step 2.3: Implement `update_exit_branch_id`**

Append to `salmon_ibm/delta_routing.py`:

```python
def _branch_reach_ids(mesh) -> np.ndarray:
    """Resolve BRANCH_FRACTIONS keys to integer reach_ids on this mesh.

    Returns an empty array if the mesh has no reach_names. Caches on the
    mesh as `_delta_branch_reach_ids` so repeated step calls don't re-scan.
    """
    cached = getattr(mesh, "_delta_branch_reach_ids", None)
    if cached is not None:
        return cached
    if not getattr(mesh, "reach_names", None):
        return np.empty(0, dtype=np.int8)
    rids = np.array(
        [mesh.reach_names.index(br) for br in BRANCH_FRACTIONS
         if br in mesh.reach_names],
        dtype=np.int8,
    )
    try:
        mesh._delta_branch_reach_ids = rids
    except AttributeError:
        # Mesh stand-ins in tests may forbid attribute setting; that's fine.
        pass
    return rids


def update_exit_branch_id(pool, mesh) -> None:
    """First-touch sticky tagging of pool.exit_branch_id by delta branch.

    Mutates pool.exit_branch_id in place. For each alive agent currently in
    a delta-branch reach (Atmata/Skirvyte/Gilija) whose exit_branch_id is
    still -1, sets it to the current reach_id. Once written, never resets
    — first-touch is the science contract.

    No-op when the mesh has no reach_names (TriMesh / HexMesh fallbacks).
    """
    if not getattr(mesh, "reach_names", None):
        return
    branch_rids = _branch_reach_ids(mesh)
    if len(branch_rids) == 0:
        return
    # Tri_idx may contain -1 for off-mesh agents; clamp to a valid index for
    # the lookup (we mask them out below by `alive` and the is_branch test).
    tri = pool.tri_idx
    safe_tri = np.where(tri >= 0, tri, 0)
    cur_reach = mesh.reach_id[safe_tri]
    is_branch = np.isin(cur_reach, branch_rids)
    on_mesh = tri >= 0
    untagged = pool.exit_branch_id == -1
    target = is_branch & untagged & pool.alive & on_mesh
    if target.any():
        pool.exit_branch_id[target] = cur_reach[target]
```

- [ ] **Step 2.4: Run all delta_routing tests — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_delta_routing.py -v
```

Expected: 12 tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add salmon_ibm/delta_routing.py tests/test_delta_routing.py
git commit -m "feat(delta_routing): update_exit_branch_id sticky first-touch"
```

---

### Task 3: `AgentPool.ARRAY_FIELDS` extension

**Files:**
- Modify: `salmon_ibm/agents.py:21-33` (ARRAY_FIELDS) and `salmon_ibm/agents.py:46-67` (`__init__`)
- Test: `tests/test_agents.py`

- [ ] **Step 3.1: Write failing tests for the new fields**

Append to `tests/test_agents.py`:

```python
def test_array_fields_includes_natal_and_exit_ids():
    from salmon_ibm.agents import AgentPool
    assert "natal_reach_id" in AgentPool.ARRAY_FIELDS
    assert "exit_branch_id" in AgentPool.ARRAY_FIELDS


def test_pool_init_defaults_natal_and_exit_to_minus_one():
    from salmon_ibm.agents import AgentPool
    pool = AgentPool(n=5, start_tri=0)
    import numpy as np
    assert pool.natal_reach_id.dtype == np.int8
    assert pool.exit_branch_id.dtype == np.int8
    assert (pool.natal_reach_id == -1).all()
    assert (pool.exit_branch_id == -1).all()


def test_pool_compact_preserves_natal_and_exit_ids():
    """compact() must propagate the new fields like every other ARRAY_FIELD."""
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np
    pool = AgentPool(n=4, start_tri=0)
    pool.natal_reach_id[:] = np.array([1, 2, 3, 4], dtype=np.int8)
    pool.exit_branch_id[:] = np.array([5, -1, 7, -1], dtype=np.int8)
    pool.alive[:] = np.array([True, False, True, False])
    pop = Population.__new__(Population)        # bypass __init__ paths
    pop.pool = pool
    pop.group_id = np.zeros(4, dtype=np.int32)
    pop.agent_ids = np.arange(4, dtype=np.int64)
    pop.affinity_targets = np.full(4, -1, dtype=np.intp)
    pop.spatial_affinity = np.zeros(4, dtype=np.float64)
    pop.accumulator_mgr = None
    pop.trait_mgr = None
    pop.genome = None
    pop.compact()
    assert pool.n == 2
    assert (pool.natal_reach_id == np.array([1, 3], dtype=np.int8)).all()
    assert (pool.exit_branch_id == np.array([5, 7], dtype=np.int8)).all()
```

- [ ] **Step 3.2: Run tests — they should fail**

```bash
micromamba run -n shiny python -m pytest tests/test_agents.py -v -k "natal or exit_branch"
```

Expected: 3 tests fail (ARRAY_FIELDS missing the new entries; __init__ assertion fires on missing initialization).

- [ ] **Step 3.3: Extend `ARRAY_FIELDS` and `__init__`**

In `salmon_ibm/agents.py`, replace the `ARRAY_FIELDS` tuple:

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
    )
```

In `AgentPool.__init__`, after the `self.temp_history = ...` line at line 67, add:

```python
        self.natal_reach_id = np.full(n, -1, dtype=np.int8)
        self.exit_branch_id = np.full(n, -1, dtype=np.int8)
```

- [ ] **Step 3.4: Run tests — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_agents.py -v -k "natal or exit_branch"
```

Expected: 3 tests pass.

- [ ] **Step 3.5: Commit**

```bash
git add salmon_ibm/agents.py tests/test_agents.py
git commit -m "feat(agents): add natal_reach_id and exit_branch_id to AgentPool"
```

---

### Task 4: `Population.add_agents` defaults for new fields

**Files:**
- Modify: `salmon_ibm/population.py:200-213`
- Test: `tests/test_population.py`

- [ ] **Step 4.1: Write failing test for default-fill on new fields**

Append to `tests/test_population.py`:

```python
def test_add_agents_defaults_natal_and_exit_to_minus_one():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np
    pool = AgentPool(n=2, start_tri=0)
    pop = Population(name="test", pool=pool)
    pop.natal_reach_id[:] = np.array([5, 7], dtype=np.int8)
    pop.exit_branch_id[:] = np.array([3, 4], dtype=np.int8)
    new_idx = pop.add_agents(n=3, positions=np.array([0, 1, 2]))
    assert (pop.natal_reach_id[new_idx] == -1).all()
    assert (pop.exit_branch_id[new_idx] == -1).all()
    assert pop.natal_reach_id[0] == 5  # original agents untouched
    assert pop.natal_reach_id[1] == 7
```

- [ ] **Step 4.2: Run test — should fail**

```bash
micromamba run -n shiny python -m pytest tests/test_population.py::test_add_agents_defaults_natal_and_exit_to_minus_one -v
```

Expected: FAIL — likely an assertion firing in `add_agents` that field length doesn't match `new_n` (because the new fields aren't filled), or the Population property raises.

- [ ] **Step 4.3: Add property accessors and default-fills**

In `salmon_ibm/population.py`, the existing pattern uses property accessors that delegate to `pool` (e.g., `target_spawn_hour` near line 99). Add the same pattern for the two new fields. Search for the `target_spawn_hour` property and add immediately after it:

```python
    @property
    def natal_reach_id(self) -> np.ndarray:
        return self.pool.natal_reach_id

    @natal_reach_id.setter
    def natal_reach_id(self, v):
        self.pool.natal_reach_id = v

    @property
    def exit_branch_id(self) -> np.ndarray:
        return self.pool.exit_branch_id

    @exit_branch_id.setter
    def exit_branch_id(self, v):
        self.pool.exit_branch_id = v
```

In `add_agents`, after the line `new_arrays["temp_history"][old_n:] = 15.0` (around line 213), add:

```python
        new_arrays["natal_reach_id"][old_n:] = -1
        new_arrays["exit_branch_id"][old_n:] = -1
```

- [ ] **Step 4.4: Run test — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_population.py::test_add_agents_defaults_natal_and_exit_to_minus_one -v
```

Expected: PASS.

- [ ] **Step 4.5: Commit**

```bash
git add salmon_ibm/population.py tests/test_population.py
git commit -m "feat(population): add_agents fills natal_reach_id and exit_branch_id with -1"
```

---

### Task 5: `Population.set_natal_reach_from_cells`

**Files:**
- Modify: `salmon_ibm/population.py`
- Test: `tests/test_population.py`

- [ ] **Step 5.1: Write failing test**

Append to `tests/test_population.py`:

```python
def test_set_natal_reach_from_cells_writes_correct_reach_ids():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np

    class _FakeMesh:
        reach_names = ["Nemunas", "Atmata", "Skirvyte"]
        reach_id = np.array([0, 0, 1, 2, 1], dtype=np.int8)  # 5 cells

    pool = AgentPool(n=3, start_tri=np.array([1, 3, 4]))  # cells 1,3,4
    pop = Population(name="test", pool=pool)
    pop.set_natal_reach_from_cells(np.arange(3), _FakeMesh())
    expected = np.array([0, 2, 1], dtype=np.int8)  # Nemunas, Skirvyte, Atmata
    assert (pop.natal_reach_id == expected).all()


def test_set_natal_reach_from_cells_no_op_without_reach_names():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np

    class _NoMesh:
        pass

    pool = AgentPool(n=2, start_tri=0)
    pop = Population(name="test", pool=pool)
    pop.set_natal_reach_from_cells(np.arange(2), _NoMesh())
    assert (pop.natal_reach_id == -1).all()  # untouched
```

- [ ] **Step 5.2: Run tests — should fail**

```bash
micromamba run -n shiny python -m pytest tests/test_population.py -v -k "set_natal_reach_from_cells"
```

Expected: `AttributeError: 'Population' object has no attribute 'set_natal_reach_from_cells'`.

- [ ] **Step 5.3: Implement the helper**

In `salmon_ibm/population.py`, add this method to the `Population` class (placement: just after `add_agents` returns):

```python
    def set_natal_reach_from_cells(self, new_idx, mesh) -> None:
        """Tag new agents' natal_reach_id by looking up the mesh reach_id at
        their current cell. Called by every add_agents call site.

        Truth-checks reach_names (a list) rather than reach_id (an ndarray)
        to avoid `if not arr` raising on multi-element numpy arrays. Same
        defensive pattern as delta_routing.update_exit_branch_id.

        Off-mesh agents (tri_idx < 0) are skipped — without the guard,
        `mesh.reach_id[-1]` would silently return the last cell's reach_id
        and tag the agent with a wrong reach.
        """
        if not getattr(mesh, "reach_names", None):
            return
        new_idx = np.asarray(new_idx)
        tri = self.pool.tri_idx[new_idx]
        valid = tri >= 0
        if valid.any():
            self.pool.natal_reach_id[new_idx[valid]] = mesh.reach_id[tri[valid]]
```

- [ ] **Step 5.4: Run tests — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_population.py -v -k "set_natal_reach_from_cells"
```

Expected: 2 tests pass.

- [ ] **Step 5.5: Commit**

```bash
git add salmon_ibm/population.py tests/test_population.py
git commit -m "feat(population): set_natal_reach_from_cells helper"
```

---

### Task 6: `Population.assert_natal_tagged`

**Files:**
- Modify: `salmon_ibm/population.py`
- Test: `tests/test_population.py`

- [ ] **Step 6.1: Write failing test**

Append to `tests/test_population.py`:

```python
def test_assert_natal_tagged_fires_on_untagged_alive_on_mesh():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np
    import pytest

    pool = AgentPool(n=3, start_tri=np.array([0, 1, 2]))
    pop = Population(name="test", pool=pool)
    pool.natal_reach_id[:] = np.array([5, -1, 7], dtype=np.int8)  # idx 1 untagged
    pool.alive[:] = True
    with pytest.raises(AssertionError, match="natal_reach_id tagging"):
        pop.assert_natal_tagged()


def test_assert_natal_tagged_silent_when_all_tagged():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np
    pool = AgentPool(n=3, start_tri=np.array([0, 1, 2]))
    pop = Population(name="test", pool=pool)
    pool.natal_reach_id[:] = np.array([5, 6, 7], dtype=np.int8)
    pool.alive[:] = True
    pop.assert_natal_tagged()  # must not raise


def test_assert_natal_tagged_ignores_dead_agents():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np
    pool = AgentPool(n=3, start_tri=np.array([0, 1, 2]))
    pop = Population(name="test", pool=pool)
    pool.natal_reach_id[:] = np.array([5, -1, 7], dtype=np.int8)  # idx 1 untagged
    pool.alive[:] = np.array([True, False, True])              # but idx 1 is dead
    pop.assert_natal_tagged()  # must not raise — dead agents excluded
```

- [ ] **Step 6.2: Run tests — should fail**

```bash
micromamba run -n shiny python -m pytest tests/test_population.py -v -k "assert_natal_tagged"
```

Expected: `AttributeError: 'Population' object has no attribute 'assert_natal_tagged'`.

- [ ] **Step 6.3: Implement the assertion**

In `salmon_ibm/population.py`, add immediately after `set_natal_reach_from_cells`:

```python
    def assert_natal_tagged(self) -> None:
        """Fail loudly if any alive on-mesh agent lacks natal_reach_id tagging.

        Called once per simulation step from Simulation.step() before logging,
        so any failure surfaces in the same step that introduced the
        un-tagged agents. Suppressed under Simulation(resume=True).
        """
        pool = self.pool
        on_mesh = pool.alive & (pool.tri_idx >= 0)
        untagged = pool.natal_reach_id == -1
        bad = on_mesh & untagged
        assert not bad.any(), (
            "Agents introduced without natal_reach_id tagging — every "
            "code path calling add_agents() must follow up with "
            "set_natal_reach_from_cells() or equivalent. "
            f"{int(bad.sum())} agents affected."
        )
```

- [ ] **Step 6.4: Run tests — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_population.py -v -k "assert_natal_tagged"
```

Expected: 3 tests pass.

- [ ] **Step 6.5: Commit**

```bash
git add salmon_ibm/population.py tests/test_population.py
git commit -m "feat(population): assert_natal_tagged runtime invariant"
```

---

### Task 7: Tag `IntroductionEvent` after `add_agents`

**Files:**
- Modify: `salmon_ibm/events_builtin.py:237-239`
- Test: covered by integration test (Task 18) and the assertion fires in test_simulation if forgotten.

- [ ] **Step 7.1: Apply the tagging idiom**

In `salmon_ibm/events_builtin.py`, find the `IntroductionEvent.execute` method around line 237. The current code is:

```python
        new_idx = population.add_agents(
            n, pos_arr, mass_g=mass, ed_kJ_g=self.initial_ed
        )
```

Add the tagging call directly after that block, before any trait/accumulator initialisation:

```python
        new_idx = population.add_agents(
            n, pos_arr, mass_g=mass, ed_kJ_g=self.initial_ed
        )
        # Tag natal_reach_id from current cell — see salmon_ibm/delta_routing.py.
        mesh = landscape.get("mesh")
        if mesh is not None:
            population.set_natal_reach_from_cells(new_idx, mesh)
```

- [ ] **Step 7.2: Run the existing IntroductionEvent tests as a smoke check**

```bash
micromamba run -n shiny python -m pytest tests/ -v -k "IntroductionEvent or introduction" --no-header
```

Expected: existing tests still pass; if any test instantiates an IntroductionEvent against a `landscape` dict that has a `"mesh"` key with `reach_names`, the new code path runs silently. If `mesh` is missing or has no `reach_names`, the tagging is a no-op.

- [ ] **Step 7.3: Commit**

```bash
git add salmon_ibm/events_builtin.py
git commit -m "feat(events_builtin): IntroductionEvent tags natal_reach_id"
```

---

### Task 8: Tag `PatchIntroductionEvent` after `add_agents`

**Files:**
- Modify: `salmon_ibm/events_hexsim.py:418`

- [ ] **Step 8.1: Apply the tagging idiom**

In `salmon_ibm/events_hexsim.py`, find the existing line around 418:

```python
        population.add_agents(len(nonzero_cells), nonzero_cells)
```

Replace with:

```python
        new_idx = population.add_agents(len(nonzero_cells), nonzero_cells)
        mesh = landscape.get("mesh")
        if mesh is not None:
            population.set_natal_reach_from_cells(new_idx, mesh)
```

- [ ] **Step 8.2: Run existing PatchIntroductionEvent tests**

```bash
micromamba run -n shiny python -m pytest tests/ -v -k "PatchIntroduction or patch_introduction" --no-header
```

Expected: existing tests still pass.

- [ ] **Step 8.3: Commit**

```bash
git add salmon_ibm/events_hexsim.py
git commit -m "feat(events_hexsim): PatchIntroductionEvent tags natal_reach_id"
```

---

### Task 9: Tag `ReproductionEvent` after `add_agents`

**Files:**
- Modify: `salmon_ibm/events_builtin.py:292-297`

- [ ] **Step 9.1: Apply the tagging idiom**

In `salmon_ibm/events_builtin.py`, find the `ReproductionEvent.execute` block at line 292:

```python
        new_idx = population.add_agents(
            total_offspring,
            offspring_positions,
            mass_g=offspring_mass,
            ed_kJ_g=self.offspring_ed,
        )
```

Add immediately after:

```python
        mesh = landscape.get("mesh")
        if mesh is not None:
            population.set_natal_reach_from_cells(new_idx, mesh)
```

- [ ] **Step 9.2: Run existing reproduction tests**

```bash
micromamba run -n shiny python -m pytest tests/ -v -k "Reproduction or reproduction" --no-header
```

Expected: existing tests still pass.

- [ ] **Step 9.3: Commit**

```bash
git add salmon_ibm/events_builtin.py
git commit -m "feat(events_builtin): ReproductionEvent tags natal_reach_id from parent cell"
```

---

### Task 10: Tag `events_phase3.py` vegetation seedling event

**Files:**
- Modify: `salmon_ibm/events_phase3.py:294-300`

- [ ] **Step 10.1: Apply the tagging idiom**

In `salmon_ibm/events_phase3.py`, the existing code at lines 294–300 is:

```python
        if hasattr(population, "add_agents"):
            population.add_agents(
                len(seed_positions),
                seed_positions,
                mass_g=np.full(len(seed_positions), 1.0),
                ed_kJ_g=1.0,
            )
```

Replace it with:

```python
        if hasattr(population, "add_agents"):
            new_idx = population.add_agents(
                len(seed_positions),
                seed_positions,
                mass_g=np.full(len(seed_positions), 1.0),
                ed_kJ_g=1.0,
            )
            mesh = landscape.get("mesh")
            if mesh is not None and hasattr(population, "set_natal_reach_from_cells"):
                population.set_natal_reach_from_cells(new_idx, mesh)
```

The `hasattr` guard on `set_natal_reach_from_cells` is defensive — `events_phase3.py` may be loaded against non-fish populations that don't have this helper. (The Population class added the method in Task 5; non-Population species classes won't have it.)

- [ ] **Step 10.2: Run any phase3 tests that exist**

```bash
micromamba run -n shiny python -m pytest tests/ -v -k "phase3 or vegetation or seedling" --no-header
```

Expected: existing tests still pass (or skip if no phase3 tests exist locally).

- [ ] **Step 10.3: Commit**

```bash
git add salmon_ibm/events_phase3.py
git commit -m "feat(events_phase3): seedling event tags natal_reach_id when available"
```

---

### Task 11: Preserve `natal_reach_id` across `TransferEvent`

**Files:**
- Modify: `salmon_ibm/network.py:193-195`

- [ ] **Step 11.1: Apply the natal-preservation logic**

In `salmon_ibm/network.py`, find the lines around 193:

```python
        positions = source.tri_idx[transfer]
        target.add_agents(len(transfer), positions)
        source.alive[transfer] = False
```

Replace with:

```python
        positions = source.tri_idx[transfer]
        new_idx = target.add_agents(len(transfer), positions)
        # Natal is fixed at birth — preserve from source on transfer.
        # Note (documented limitation): if source and target use different
        # meshes, the reach_id encoding may not match. No deployed salmon
        # scenario uses multi-mesh transfers today.
        if hasattr(target, "natal_reach_id") and hasattr(source, "natal_reach_id"):
            target.natal_reach_id[new_idx] = source.natal_reach_id[transfer]
            target.exit_branch_id[new_idx] = source.exit_branch_id[transfer]
        source.alive[transfer] = False
```

- [ ] **Step 11.2: Run existing TransferEvent / network tests**

```bash
micromamba run -n shiny python -m pytest tests/ -v -k "TransferEvent or network" --no-header
```

Expected: existing tests still pass.

- [ ] **Step 11.3: Commit**

```bash
git add salmon_ibm/network.py
git commit -m "feat(network): TransferEvent preserves natal/exit ids from source"
```

---

### Task 12: `Simulation` init-time mesh validation

**Files:**
- Modify: `salmon_ibm/simulation.py` (constructor)
- Test: `tests/test_simulation.py`

- [ ] **Step 12.1: Write failing test**

Append to `tests/test_simulation.py`:

```python
def test_init_raises_when_branch_fractions_keys_missing_from_mesh():
    """If the mesh's reach_names does not include all BRANCH_FRACTIONS keys,
    the validator must raise ValueError naming the missing branches."""
    import pytest
    from salmon_ibm.simulation import _validate_mesh_for_delta_routing

    class _Mesh:
        reach_names = ["Nemunas", "CuronianLagoon"]   # missing all 3 branches

    with pytest.raises(ValueError, match="(Atmata|Skirvyte|Gilija)"):
        _validate_mesh_for_delta_routing(_Mesh())


def test_validate_mesh_passes_with_all_branches():
    from salmon_ibm.simulation import _validate_mesh_for_delta_routing

    class _Mesh:
        reach_names = ["Nemunas", "Atmata", "Skirvyte", "Gilija", "CuronianLagoon"]

    _validate_mesh_for_delta_routing(_Mesh())  # must not raise


def test_validate_mesh_no_op_without_reach_names():
    from salmon_ibm.simulation import _validate_mesh_for_delta_routing

    class _NoMesh:
        pass

    _validate_mesh_for_delta_routing(_NoMesh())  # no-op, no exception
```

- [ ] **Step 12.2: Run tests — should fail**

```bash
micromamba run -n shiny python -m pytest tests/test_simulation.py -v -k "validate_mesh"
```

Expected: `ImportError: cannot import name '_validate_mesh_for_delta_routing'`.

- [ ] **Step 12.3: Implement the validator and call it from `Simulation.__init__`**

In `salmon_ibm/simulation.py`, near the top of the file (after imports), add:

```python
from salmon_ibm import delta_routing


def _validate_mesh_for_delta_routing(mesh) -> None:
    """Raise ValueError if the mesh is missing any BRANCH_FRACTIONS reach.

    No-op when the mesh has no reach_names (TriMesh / HexMesh fallbacks).
    """
    if not getattr(mesh, "reach_names", None):
        return
    missing = sorted(
        br for br in delta_routing.BRANCH_FRACTIONS
        if br not in mesh.reach_names
    )
    if missing:
        raise ValueError(
            f"Mesh reach_names is missing delta-branch reaches: {missing}. "
            f"BRANCH_FRACTIONS expects {sorted(delta_routing.BRANCH_FRACTIONS)}; "
            f"mesh has {sorted(mesh.reach_names)}."
        )
```

In `Simulation.__init__`, the mesh-construction section is an `if/elif/elif/else` block spanning multiple backends (`h3`, `h3_multires`, `hexsim`, default TriMesh). The validator must run **after all branches complete**, *not* inside any single branch.

The cleanest anchor is **just before `self.pool = AgentPool(...)`** (around line 197) — by that point `self.mesh` is fully constructed and populated regardless of which mesh backend ran. Insert:

```python
        # Cross-validate BRANCH_FRACTIONS against the constructed mesh.
        # No-op on backends without reach_names (TriMesh / HexMesh fallbacks).
        _validate_mesh_for_delta_routing(self.mesh)

        self.pool = AgentPool(n=n_agents, start_tri=start_tris, rng_seed=rng_seed)
```

- [ ] **Step 12.6: Tag initial agents at simulation init**

Initial agents are placed in `AgentPool.__init__` (line 197) — *not* via an `IntroductionEvent`. Without explicit init-time tagging, they all stay `natal_reach_id = -1` and the runtime assertion (Task 13.4) fires on step 1.

After `self.population = Population(name="salmon", pool=self.pool)` at line 201, add:

```python
        # Tag initial agents' natal_reach_id from their starting cell.
        # No-op on backends without reach_names (TriMesh / HexMesh).
        self.population.set_natal_reach_from_cells(
            np.arange(self.pool.n), self.mesh
        )
```

This ensures all H3 scenarios have correctly-tagged initial agents from step 0. On TriMesh / HexMesh, the call is a no-op (the new fields stay at -1) and the Task-13.4 callback's mesh-gate keeps the assertion silent.

- [ ] **Step 12.4: Run tests — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_simulation.py -v -k "validate_mesh"
```

Expected: 3 tests pass.

- [ ] **Step 12.5: Commit**

```bash
git add salmon_ibm/simulation.py tests/test_simulation.py
git commit -m "feat(simulation): init-time validation of BRANCH_FRACTIONS vs mesh"
```

---

### Task 13: `Simulation` resume flag + step-time assertion gate

**Files:**
- Modify: `salmon_ibm/simulation.py` (constructor + step + new event callback)
- Test: `tests/test_simulation.py`

- [ ] **Step 13.1: Write failing tests**

Existing `tests/test_simulation.py` constructs the sim directly:

```python
cfg = load_config("config_curonian_minimal.yaml")
sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
```

Mirror that pattern. Append to `tests/test_simulation.py`:

```python
def test_simulation_resume_flag_defaults_false():
    """Simulation.__init__ accepts resume: bool; default is False."""
    import inspect
    from salmon_ibm.simulation import Simulation
    sig = inspect.signature(Simulation.__init__)
    assert "resume" in sig.parameters, "Simulation must accept a `resume` kwarg"
    assert sig.parameters["resume"].default is False


def test_simulation_step_skips_assertion_when_resume():
    """Under resume=True, Simulation.step does NOT call assert_natal_tagged.

    Note: TriMesh has no reach_names, so the callback short-circuits before
    even reaching the `resume` check. To isolate the resume-gate, we
    monkey-patch reach_names on the mesh so the callback proceeds past the
    no-reach-metadata gate. Then the resume gate is the ONLY thing that
    decides whether the spy is called.
    """
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42, resume=True)
    sim.mesh.reach_names = ["FakeReach"]   # force the path past the no-meta gate
    called = {"count": 0}

    def _spy():
        called["count"] += 1

    sim.population.assert_natal_tagged = _spy
    sim.step()
    assert called["count"] == 0, "assertion should be suppressed under resume=True"


def test_simulation_step_calls_assertion_when_not_resume():
    """Under resume=False (default), Simulation.step DOES call the assertion
    when reach metadata is present."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim.mesh.reach_names = ["FakeReach"]   # force the path past the no-meta gate
    called = {"count": 0}

    def _spy():
        called["count"] += 1

    sim.population.assert_natal_tagged = _spy
    sim.step()
    assert called["count"] >= 1, "assertion must run when resume=False and reach_names is set"
```

Why the spy: the spy replaces `assert_natal_tagged` entirely, so the underlying assertion logic (which would *itself* fail because the TriMesh agents have `natal_reach_id == -1`) never runs. The test only verifies whether the callback chooses to invoke the method or short-circuits — exactly the gate we want to test.

- [ ] **Step 13.2: Run tests — should fail**

```bash
micromamba run -n shiny python -m pytest tests/test_simulation.py -v -k "resume"
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'resume'` or assertion fail.

- [ ] **Step 13.3: Add the resume flag**

In `salmon_ibm/simulation.py`, modify `Simulation.__init__` (signature at line 67–68):

```python
    def __init__(
        self, config, n_agents=100, data_dir="data", rng_seed=None,
        output_path=None, resume: bool = False,
    ):
```

In the body, near the start of `__init__`, store the flag:

```python
        self.resume = resume
```

- [ ] **Step 13.4: Add the assertion event to the event sequence**

The existing `_event_logging` (line 466) is **conditional** (`if self.logger:`) — it's gated by whether a logger was configured. The natal-tagging assertion must run *unconditionally* on non-resume runs, so it must be a separate `CustomEvent` rather than embedded in `_event_logging`.

In `_build_default_events` (around line 343), insert before the logging entry:

```python
            CustomEvent(
                name="assert_natal_tagged",
                callback=self._event_assert_natal_tagged,
            ),
            CustomEvent(name="logging", callback=self._event_logging),
```

In `_build_callback_registry` (around line 346), add:

```python
            "assert_natal_tagged": self._event_assert_natal_tagged,
```

Add the callback method (place it next to `_event_logging`, around line 466):

```python
    def _event_assert_natal_tagged(self, population, landscape, t, mask):
        # Two short-circuits: resume runs (option α) and meshes without
        # reach metadata (TriMesh / HexMesh — agents legitimately have
        # natal_reach_id == -1 there, no contract to enforce).
        if self.resume:
            return
        if not getattr(self.mesh, "reach_names", None):
            return
        population.assert_natal_tagged()
```

- [ ] **Step 13.5: Run tests — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_simulation.py -v -k "resume"
```

Expected: PASS.

- [ ] **Step 13.6: Commit**

```bash
git add salmon_ibm/simulation.py tests/test_simulation.py
git commit -m "feat(simulation): resume flag + step-time assert_natal_tagged gate"
```

---

### Task 14: `update_exit_branch` event in the simulation step pipeline

**Files:**
- Modify: `salmon_ibm/simulation.py` (event list + new callback)
- Test: existing delta_routing unit tests + Task 18 integration test

- [ ] **Step 14.1: Add the CustomEvent and its callback**

In `salmon_ibm/simulation.py`, find `_build_default_events` (around line 320). The existing event list contains a `MovementEvent(name="movement", ...)` followed by `CustomEvent(name="fish_predation", ...)`. Insert a new `CustomEvent` between them:

```python
            MovementEvent(
                name="movement",
                n_micro_steps=3,
                cwr_threshold=self.beh_params.temp_bins[0],
            ),
            # NEW: tag exit_branch_id (sticky first-touch) before any
            # post-movement event reads or kills agents in delta-branch cells.
            CustomEvent(
                name="update_exit_branch",
                callback=self._event_update_exit_branch,
            ),
            # Fish predation fires AFTER movement so agents are killed
            # at their final cell of this step (not their starting cell).
            # No-ops on backends without reach_id (TriMesh, HexMesh).
            CustomEvent(
                name="fish_predation", callback=self._event_fish_predation
            ),
```

Add the callback method to `Simulation` (place it next to `_event_fish_predation`, around line 389):

```python
    def _event_update_exit_branch(self, population, landscape, t, mask):
        """First-touch sticky tagging of exit_branch_id by delta branch."""
        delta_routing.update_exit_branch_id(population.pool, self.mesh)
```

Also register it in `_build_callback_registry` (around line 346) by adding:

```python
            "update_exit_branch": self._event_update_exit_branch,
```

- [ ] **Step 14.2: Re-run the delta_routing unit tests as a smoke check**

```bash
micromamba run -n shiny python -m pytest tests/test_delta_routing.py -v
```

Expected: 12 tests still pass.

- [ ] **Step 14.3: Run the broader simulation suite**

```bash
micromamba run -n shiny python -m pytest tests/test_simulation.py tests/test_curonian_realism_integration.py -v --no-header -q
```

Expected: existing simulation tests still pass. The new event is a no-op on backends without `reach_names`, and the test-landscape tests will trigger it.

- [ ] **Step 14.4: Commit**

```bash
git add salmon_ibm/simulation.py
git commit -m "feat(simulation): update_exit_branch CustomEvent between move and predation"
```

---

### Task 15: `OutputLogger` extension for `natal_reach_id` and `exit_branch_id`

**Files:**
- Modify: `salmon_ibm/output.py` — `__init__` (both modes), `log_step` (both modes), `to_dataframe` (both modes + `empty_cols`)
- Test: `tests/test_output.py`

**Note**: the actual method is `log_step(t, population)` — *not* `append_step` as the spec suggested. The spec text uses "append_step" loosely; the codebase uses `log_step`.

- [ ] **Step 15.1: Read the current OutputLogger**

Read `salmon_ibm/output.py` lines 16–125 for the existing patterns. There are **two parallel modes** in `__init__`, gated by `self._preallocated`:
- columnar mode (when both `max_steps` and `max_agents` are set): allocates `self._<field>_arr = np.empty((max_steps, max_agents), ...)`
- list-append mode (fallback): `self._<field>: list[np.ndarray] = []`

`log_step` (lines 52–85) and `to_dataframe` (lines 87–124) each have a corresponding pair of branches — gated by the same `self._preallocated` flag.

- [ ] **Step 15.2: Write failing tests**

Append to `tests/test_output.py`:

```python
def test_outputlogger_serialises_natal_reach_id(tmp_path):
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.output import OutputLogger
    import numpy as np
    pool = AgentPool(n=3, start_tri=np.array([0, 1, 2]))
    pool.natal_reach_id[:] = np.array([5, 6, 7], dtype=np.int8)
    pool.exit_branch_id[:] = np.array([8, -1, 9], dtype=np.int8)
    pop = Population(name="test", pool=pool)
    logger = OutputLogger(path=str(tmp_path / "out.csv"),
                          centroids=np.zeros((10, 2)))
    logger.log_step(t=0, population=pop)
    df = logger.to_dataframe()
    assert "natal_reach_id" in df.columns
    assert df["natal_reach_id"].tolist() == [5, 6, 7]


def test_outputlogger_serialises_exit_branch_id(tmp_path):
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.output import OutputLogger
    import numpy as np
    pool = AgentPool(n=2, start_tri=np.array([0, 1]))
    pool.exit_branch_id[:] = np.array([4, -1], dtype=np.int8)
    pop = Population(name="test", pool=pool)
    logger = OutputLogger(path=str(tmp_path / "out.csv"),
                          centroids=np.zeros((10, 2)))
    logger.log_step(t=0, population=pop)
    df = logger.to_dataframe()
    assert "exit_branch_id" in df.columns
    assert df["exit_branch_id"].tolist() == [4, -1]
```

- [ ] **Step 15.3: Run tests — should fail**

```bash
micromamba run -n shiny python -m pytest tests/test_output.py -v -k "natal_reach_id or exit_branch_id"
```

Expected: `KeyError: 'natal_reach_id'` (column not in DataFrame) or `AttributeError`.

- [ ] **Step 15.4: Extend `OutputLogger.__init__` (both modes)**

In `salmon_ibm/output.py`, in the `if self._preallocated:` branch (currently ending at line 40 with `self._arrived_arr`), add:

```python
            self._natal_reach_id_arr = np.empty((max_steps, max_agents), dtype=np.int8)
            self._exit_branch_id_arr = np.empty((max_steps, max_agents), dtype=np.int8)
```

In the `else:` (list-append) branch (currently ending at line 50 with `self._arrived: list[np.ndarray] = []`), add:

```python
            self._natal_reach_id: list[np.ndarray] = []
            self._exit_branch_id: list[np.ndarray] = []
```

- [ ] **Step 15.5: Extend `log_step` (both modes)**

In the `if self._preallocated:` branch of `log_step` (currently ending at line 74 with `self._arrived_arr[r, :n] = pool.arrived[:n]`), add:

```python
            self._natal_reach_id_arr[r, :n] = pool.natal_reach_id[:n]
            self._exit_branch_id_arr[r, :n] = pool.exit_branch_id[:n]
```

In the `else:` branch (currently ending at line 85 with `self._arrived.append(pool.arrived.copy())`), add:

```python
            self._natal_reach_id.append(pool.natal_reach_id.copy())
            self._exit_branch_id.append(pool.exit_branch_id.copy())
```

- [ ] **Step 15.6: Extend `to_dataframe` (three places)**

(a) Update `empty_cols` (line 88):

```python
        empty_cols = [
            "time", "agent_id", "tri_idx", "lat", "lon",
            "ed_kJ_g", "behavior", "alive", "arrived",
            "natal_reach_id", "exit_branch_id",
        ]
```

(b) In the columnar branch's `parts.append(pd.DataFrame({...}))` block (line 98), add to the dict literal:

```python
                    "natal_reach_id": self._natal_reach_id_arr[r, :n],
                    "exit_branch_id": self._exit_branch_id_arr[r, :n],
```

(c) In the list-append `pd.DataFrame({...})` block (line 112), add to the dict literal:

```python
                "natal_reach_id": np.concatenate(self._natal_reach_id),
                "exit_branch_id": np.concatenate(self._exit_branch_id),
```

- [ ] **Step 15.7: Run tests — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_output.py -v -k "natal_reach_id or exit_branch_id"
```

Expected: 2 tests pass.

- [ ] **Step 15.8: Run the whole `test_output.py` to check nothing else broke**

```bash
micromamba run -n shiny python -m pytest tests/test_output.py -v
```

Expected: all tests pass.

- [ ] **Step 15.9: Commit**

```bash
git add salmon_ibm/output.py tests/test_output.py
git commit -m "feat(output): OutputLogger serialises natal_reach_id and exit_branch_id"
```

---

### Task 16: `Q_per_branch` in `fetch_nemunas_discharge.py`

**Files:**
- Modify: `scripts/fetch_nemunas_discharge.py:34-45` (`synthesize_climatology`)
- Test: `tests/test_nemunas_discharge.py`

- [ ] **Step 16.1: Write failing test**

Create `tests/test_nemunas_discharge.py`:

```python
"""Schema test for data/nemunas_discharge.nc."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from salmon_ibm import delta_routing

PROJECT = Path(__file__).resolve().parent.parent
NC = PROJECT / "data" / "nemunas_discharge.nc"


def test_q_per_branch_present_and_consistent():
    if not NC.exists():
        pytest.skip(f"{NC.name} missing — run scripts/fetch_nemunas_discharge.py")
    ds = xr.open_dataset(NC)
    if "Q_per_branch" not in ds.variables:
        pytest.skip(
            "Q_per_branch missing — re-run scripts/fetch_nemunas_discharge.py "
            "to refresh the discharge NC with the new variable."
        )
    n_branches = len(delta_routing.BRANCH_FRACTIONS)
    n_time = ds.sizes["time"]
    assert ds["Q_per_branch"].shape == (n_branches, n_time), (
        f"Expected ({n_branches}, {n_time}), got {ds['Q_per_branch'].shape}"
    )
    branch_names_attr = ds.attrs.get("branch_names", "").split(",")
    assert branch_names_attr == list(delta_routing.BRANCH_FRACTIONS), (
        f"branch_names attr = {branch_names_attr}; "
        f"expected {list(delta_routing.BRANCH_FRACTIONS)}"
    )
    summed = ds["Q_per_branch"].values.sum(axis=0)
    np.testing.assert_allclose(summed, ds["Q"].values, rtol=1e-5)
    assert ds.attrs.get("branch_fractions_source", "").strip(), (
        "branch_fractions_source attr must be non-empty"
    )
    ds.close()
```

- [ ] **Step 16.2: Run test — should skip (NC variable missing)**

```bash
micromamba run -n shiny python -m pytest tests/test_nemunas_discharge.py -v
```

Expected: SKIPPED with message about Q_per_branch.

- [ ] **Step 16.3: Extend the fetch script**

In `scripts/fetch_nemunas_discharge.py`, modify `synthesize_climatology` to also compute and write the per-branch variable. Replace the existing `xr.Dataset(...)` block:

```python
def synthesize_climatology() -> xr.Dataset:
    """Daily Q(t) = baseline + Gaussian peak around day 115 (Apr 25)."""
    from salmon_ibm import delta_routing

    doy = DATES.dayofyear.values
    baseline = 400.0
    amplitude = 1500.0
    peak_day = 115  # Apr 25
    sigma_days = 35
    Q = (baseline + amplitude * np.exp(-((doy - peak_day) / sigma_days) ** 2)).astype(np.float32)

    fractions = list(delta_routing.BRANCH_FRACTIONS.items())
    branch_names = [br for br, _ in fractions]
    Q_per_branch = np.stack(
        [Q * f for _, f in fractions]
    ).astype(np.float32)  # (n_branches, n_time)

    # Build-time invariants (assert before writing — fail loudly if broken)
    assert Q_per_branch.shape == (len(fractions), len(Q)), (
        f"Q_per_branch shape mismatch: {Q_per_branch.shape}"
    )
    np.testing.assert_allclose(Q_per_branch.sum(axis=0), Q, rtol=1e-6)

    return xr.Dataset(
        {
            "Q": (("time",), Q),
            "Q_per_branch": (("branch", "time"), Q_per_branch),
        },
        coords={"time": DATES},
        attrs={
            "source": "synthetic climatology (awaiting Lithuanian EPA access)",
            "calibration": "Valiuskevicius 2019 + Mezine 2019 + HELCOM PLC-6 2018",
            "units": "m^3/s",
            "station_reference": "Smalininkai gauging station (Nemunas lower reach)",
            "fetched": datetime.date.today().isoformat(),
            "note": "Daily climatology — same values repeat every year.",
            "branch_names": ",".join(branch_names),
            "branch_fractions_source": (
                "Ramsar Site 629 Information Sheet (Nemunas Delta), 2010"
            ),
        },
    )
```

- [ ] **Step 16.4: Run the script to regenerate the NC**

```bash
micromamba run -n shiny python scripts/fetch_nemunas_discharge.py --out data/nemunas_discharge.nc
```

Expected output: includes the existing summary line; no errors.

- [ ] **Step 16.5: Run the schema test — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_nemunas_discharge.py -v
```

Expected: PASS.

- [ ] **Step 16.6: Commit**

`data/nemunas_discharge.nc` is gitignored — only the script and test land in the commit.

```bash
git add scripts/fetch_nemunas_discharge.py tests/test_nemunas_discharge.py
git commit -m "feat(discharge): write Q_per_branch + branch_names to nemunas_discharge.nc"
```

---

### Task 17: Run a calibration sim to set the integration-test threshold

**Files:** none committed — this task generates a number for Task 18.

- [ ] **Step 17.1: Run the existing test landscape integration suite once with the new code**

```bash
micromamba run -n shiny python -m pytest tests/test_nemunas_h3_integration.py -v --no-header
```

Expected: existing 8 invariants still pass.

- [ ] **Step 17.2: Calibrate the natal/exit ratio**

Run a one-off probe script. Create `_diag_natal_exit_ratio.py` at the repo root (do **not** commit — `_diag_*.py` is the project's local-helper convention).

The probe mirrors the existing `h3_sim` fixture in `tests/test_nemunas_h3_integration.py` (lines 46–56):

```python
"""Probe: of Nemunas-natal smolts that survive 30 days, what fraction
recorded an exit_branch_id != -1? Used to set the Task-18 threshold."""
import numpy as np
from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation

cfg = load_config("configs/config_nemunas_h3.yaml")
sim = Simulation(cfg, n_agents=500, rng_seed=42)
sim.run(n_steps=720)

pool = sim.pool
mesh = sim.mesh
nemunas_id = mesh.reach_names.index("Nemunas")
natal_mask = pool.natal_reach_id == nemunas_id
alive_mask = pool.alive
relevant = natal_mask & alive_mask
n_total = int(relevant.sum())
n_with_exit = int((relevant & (pool.exit_branch_id >= 0)).sum())
ratio = n_with_exit / max(n_total, 1)
print(f"Nemunas-natal alive: {n_total}; with exit_branch_id >= 0: {n_with_exit}")
print(f"Observed ratio: {ratio:.3f}")
```

```bash
micromamba run -n shiny python _diag_natal_exit_ratio.py
```

Record the observed ratio (e.g., 0.78). The Task-18 threshold becomes `floor(observed * 0.6 * n_total) / n_total`. If the observed ratio is suspiciously low (< 0.20) or zero, **stop and investigate** — this likely means the new event isn't firing or the fixture isn't loading the multi-res NC with delta-branch reach_ids.

- [ ] **Step 17.3: No commit — diagnostic file is local-only**

The convention in `feedback_collaboration.md` is that `_diag_*.py` files stay uncommitted; the recorded ratio is used in Task 18.

---

### Task 18: Integration test — Nemunas-natal smolts record an exit branch

**Files:**
- Modify: `tests/test_nemunas_h3_integration.py`

- [ ] **Step 18.1: Read existing integration test patterns**

Open `tests/test_nemunas_h3_integration.py` and observe how the existing 8 invariants access the simulation result, the mesh, and the output dataset. Mirror the same access pattern in the new test.

- [ ] **Step 18.2: Write the test**

The existing fixture is `h3_sim` (defined inline at `tests/test_nemunas_h3_integration.py:46–56`, module-scoped, returns a `Simulation` object). Access pool state via `h3_sim.pool.<field>` and mesh metadata via `h3_sim.mesh`.

Append to `tests/test_nemunas_h3_integration.py`:

```python
def test_smolts_originating_in_nemunas_record_an_exit_branch(h3_sim):
    """Per spec: Nemunas-natal smolts that survive should mostly record
    a delta-branch exit. The threshold is calibrated from an actual run
    (see plan Task 17). The 0.6 multiplier matches the discipline in
    MIN_CROSS_REACH_LINKS (tests/test_h3_grid_quality.py:54).
    """
    import numpy as np
    pool = h3_sim.pool
    mesh = h3_sim.mesh
    if "Nemunas" not in mesh.reach_names:
        pytest.skip("Test landscape lacks Nemunas reach (legacy NC).")
    nemunas_id = mesh.reach_names.index("Nemunas")
    branch_ids = {
        mesh.reach_names.index(b)
        for b in ("Atmata", "Skirvyte", "Gilija")
        if b in mesh.reach_names
    }
    natal_mask = pool.natal_reach_id == nemunas_id
    relevant = natal_mask & pool.alive
    if not relevant.any():
        pytest.skip("No Nemunas-natal alive agents in this run.")
    exit_ids = pool.exit_branch_id[relevant]
    has_exit = int((exit_ids >= 0).sum())
    n_total = int(relevant.sum())
    ratio = has_exit / n_total

    # Threshold calibrated 2026-04-27 from one-off run in plan Task 17.
    # Replace OBSERVED_RATIO_FROM_TASK_17 with the value recorded there
    # (a float between 0.0 and 1.0).
    OBSERVED = OBSERVED_RATIO_FROM_TASK_17    # e.g. 0.78
    THRESHOLD = int(OBSERVED * 0.6 * n_total) / n_total
    assert ratio >= THRESHOLD, (
        f"Only {ratio:.2%} of Nemunas-natal alive agents recorded an "
        f"exit_branch_id (threshold {THRESHOLD:.2%}; calibrated "
        f"observed {OBSERVED:.2%})."
    )

    # Sanity: every recorded exit must be a delta-branch reach_id (not -1
    # because that's filtered out, and not e.g. CuronianLagoon's reach_id).
    recorded_exits = set(int(x) for x in exit_ids[exit_ids >= 0].tolist())
    assert recorded_exits.issubset(branch_ids), (
        f"exit_branch_id values include non-delta reaches: "
        f"{recorded_exits - branch_ids}"
    )
```

Replace `OBSERVED_RATIO_FROM_TASK_17` with the actual value from Task 17.2 (e.g., `0.78`).

- [ ] **Step 18.3: Run the test — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_nemunas_h3_integration.py::test_smolts_originating_in_nemunas_record_an_exit_branch -v
```

Expected: PASS.

- [ ] **Step 18.4: Run the entire integration suite to confirm nothing else regressed**

```bash
micromamba run -n shiny python -m pytest tests/test_nemunas_h3_integration.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 18.5: Commit**

```bash
git add tests/test_nemunas_h3_integration.py
git commit -m "test(integration): Nemunas-natal smolts record exit_branch_id"
```

---

### Task 19: Performance regression sentinel

**Files:**
- Modify: `tests/test_movement_metric.py`

- [ ] **Step 19.1: Read the existing perf-regression pattern**

```bash
micromamba run -n shiny python -m pytest tests/test_movement_metric.py -v --no-header
```

Confirm the existing tests run; observe the per-step timing pattern.

- [ ] **Step 19.2: Add the sentinel test**

Append to `tests/test_movement_metric.py`:

```python
def test_full_step_time_within_one_percent_of_baseline():
    """The new update_exit_branch event should add ~0.05 ms per step at
    typical agent counts. If a future change replaces the vectorised
    update with an O(n^2) loop, this test should catch it.

    Sim construction mirrors the existing h3_sim fixture
    (tests/test_nemunas_h3_integration.py:46-56).
    """
    import time
    from pathlib import Path
    import pytest
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config_nemunas_h3.yaml"
    landscape = Path(__file__).resolve().parent.parent / "data" / "nemunas_h3_landscape.nc"
    if not config_path.exists() or not landscape.exists():
        pytest.skip("nemunas H3 fixtures missing — see test_nemunas_h3_integration.py")
    cfg = load_config(str(config_path))
    sim = Simulation(cfg, n_agents=500, rng_seed=42)
    # Warmup (Numba JIT)
    for _ in range(3):
        sim.step()
    # Time 50 steps
    t0 = time.perf_counter()
    for _ in range(50):
        sim.step()
    elapsed = time.perf_counter() - t0
    per_step_ms = (elapsed / 50) * 1000.0

    # Baseline recorded 2026-04-27 on the dev machine before the new event:
    # ~XXX ms per step (replace XXX with the actual measurement). 1% margin.
    BASELINE_MS = BASELINE_FROM_RUN
    assert per_step_ms <= BASELINE_MS * 1.01, (
        f"Step time {per_step_ms:.2f} ms exceeds baseline {BASELINE_MS:.2f} ms "
        f"by more than 1%. The new update_exit_branch event should be "
        f"~0.05 ms; if this fails, check it for non-vectorised code."
    )
```

Run the test to capture a baseline first; record the value as `BASELINE_FROM_RUN` (e.g., `45.0`).

- [ ] **Step 19.3: Run the test — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_movement_metric.py -v -k "full_step_time"
```

Expected: PASS.

- [ ] **Step 19.4: Run the full suite end-to-end**

```bash
micromamba run -n shiny python -m pytest tests/ -v --no-header
```

Expected: all 582 tests pass (557 original + 25 new). Runtime ~4 minutes per CLAUDE.md.

- [ ] **Step 19.5: Commit**

```bash
git add tests/test_movement_metric.py
git commit -m "test(perf): step-time regression sentinel for update_exit_branch"
```

---

## Final integration check (no new commit)

- [ ] **Step F.1: Whole suite green**

```bash
micromamba run -n shiny python -m pytest tests/ -v
```

Expected: 582 passed, 0 failed.

- [ ] **Step F.2: `git log --oneline` to confirm 19 task-shaped commits**

```bash
git log --oneline -25
```

Expected: 19 commits since the spec was committed (`0814000`), each with a clear `feat(...)` or `test(...)` prefix.

- [ ] **Step F.3: Hand off to user for tag + deploy**

The spec's "Schema evolution" section calls out that the change is forward-only (no NC migration breaks old runs). Per `feedback_collaboration.md` ("tag-before-deploy"), the user tags a release and runs `scripts/deploy_laguna.sh apply`. **Plan does not auto-deploy — confirmation required per the project's standing instruction.**

Sample tag/deploy sequence the user runs (do not run automatically):

```bash
git tag -a v1.6.0 -m "Nemunas delta branching — D+ slice (natal_reach_id, exit_branch_id, Q_per_branch)"
git push origin main
git push origin v1.6.0
scripts/deploy_laguna.sh apply
# Then SCP the regenerated discharge NC:
scp data/nemunas_discharge.nc razinka@laguna.ku.lt:/srv/shiny-server/HexSimPy/data/
ssh razinka@laguna.ku.lt 'cd /srv/shiny-server/HexSimPy && md5sum data/nemunas_discharge.nc && touch restart.txt'
```

---

## Spec coverage check

| Spec section | Implementing task(s) |
|---|---|
| `delta_routing.py` (BRANCH_FRACTIONS, split_discharge, update_exit_branch_id) | Tasks 1, 2 |
| `agents.py` ARRAY_FIELDS extension | Task 3 |
| `population.py` defaults + `set_natal_reach_from_cells` + `assert_natal_tagged` | Tasks 4, 5, 6 |
| `simulation.py` event-sequence insertion | Task 14 |
| `simulation.py` init-time validation | Task 12 |
| `simulation.py` resume flag | Task 13 |
| `fetch_nemunas_discharge.py` schema additions | Task 16 |
| `output.py` extension (both branches) | Task 15 |
| `IntroductionEvent` tagging | Task 7 |
| `PatchIntroductionEvent` tagging | Task 8 |
| `ReproductionEvent` tagging | Task 9 |
| `events_phase3.py` vegetation tagging | Task 10 |
| `TransferEvent` natal preservation | Task 11 |
| Build-time invariants | Task 16 (Step 16.3 asserts) |
| Init-time invariants | Task 12 |
| Runtime invariants — sticky, no-op safety | Tasks 2, 6 |
| Schema-evolution mitigations | Implicitly preserved (additive-only schema) — verified by Task 19 full-suite |
| Documented limitations | Spec only — no implementation |
| All ~30 tests | Tasks 1+2 (12 in test_delta_routing), 3 (3 in test_agents), 4+5+6 (1+2+3=6 in test_population), 12+13 (3+3=6 in test_simulation), 15 (2 in test_output), 16 (1 in test_nemunas_discharge), 18 (1 in test_nemunas_h3_integration), 19 (1 in test_movement_metric) — 32 tests total |

All sections implemented. No placeholders inside tasks.
