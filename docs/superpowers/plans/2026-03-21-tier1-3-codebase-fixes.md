# HexSimPy Tier 1–3 Implementation Plan

> **STATUS: ✅ EXECUTED** — All 15 tasks shipped via comprehensive 'phase 4 completion' commit and follow-ups: bioenergetics double-mass-deduction, accumulator uptake race, simulation/estuary/behavior/events_hexsim/hexsim_env/environment/population fixes. `bioenergetics.py` `MASS_FLOOR_FRACTION` is now a documented project convention (CLAUDE.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all critical bugs, silent failures, and performance bottlenecks identified in the deep codebase analysis (Tiers 1–3).

**Architecture:** Each task is a self-contained fix with a test-first approach. Tasks are ordered by risk: scientific correctness bugs first, then silent failures, then performance, then robustness. Each fix touches 1–2 files and has no cross-dependencies, so tasks can be parallelized.

**Tech Stack:** Python 3.10+, NumPy, Numba (optional), pytest, conda env `shiny`

**Test command:** `conda run -n shiny python -m pytest tests/ -v`

---

## File Map

| Task | Files Modified | Files Created/Modified (Tests) |
|------|---------------|-------------------------------|
| 1 | `salmon_ibm/bioenergetics.py` | `tests/test_bioenergetics.py` |
| 2 | `salmon_ibm/accumulators.py` | `tests/test_accumulators.py` |
| 3 | `salmon_ibm/simulation.py` | `tests/test_simulation.py` |
| 4 | `salmon_ibm/estuary.py` | `tests/test_estuary.py` |
| 5 | `salmon_ibm/behavior.py` | `tests/test_behavior.py` |
| 6 | `salmon_ibm/events_hexsim.py` | `tests/test_events_hexsim.py` |
| 7 | `salmon_ibm/hexsim_env.py` | `tests/test_hexsim.py` |
| 8 | `salmon_ibm/events_hexsim.py` | `tests/test_events_hexsim.py` |
| 9 | `salmon_ibm/events_hexsim.py` | `tests/test_events_hexsim.py` |
| 10 | `salmon_ibm/events_hexsim.py` | `tests/test_events_hexsim.py` |
| 11 | `salmon_ibm/environment.py` | `tests/test_environment.py` |
| 12 | `salmon_ibm/population.py` | `tests/test_population.py` |
| 13 | `tests/test_agents.py`, `tests/test_events.py` | (test-only) |
| 14 | `salmon_ibm/estuary.py` | `tests/test_estuary.py` |
| 15 | `tests/conftest.py`, various test files | (test-only) |

---

## Tier 1 — Scientific Correctness (Fix Now)

### Task 1: Fix bioenergetics double mass deduction

The `update_energy` function deducts respiration energy from `e_total_j`, then independently computes `mass_loss_g` from the same `r_hourly`, then recomputes `new_ed = e_total_j / (new_mass * 1000)`. Since both energy and mass shrink from the same term, energy density paradoxically rises as fish waste away.

**Fix:** Use proportional mass loss so that energy density stays constant (or decreases) under pure respiration. This matches standard Wisconsin bioenergetics model behavior for non-feeding migrants: `new_mass = mass_g * (e_total_j / original_e_total_j)`.

**Files:**
- Modify: `salmon_ibm/bioenergetics.py:35-52`
- Test: `tests/test_bioenergetics.py`

- [ ] **Step 1: Write failing test — energy density must not increase under pure respiration**

Add to `tests/test_bioenergetics.py`:

```python
def test_energy_density_decreases_monotonically():
    """ED must decrease (or stay flat) every hour for non-feeding migrants."""
    from salmon_ibm.bioenergetics import update_energy, BioParams
    params = BioParams()
    n = 100
    ed = np.full(n, 6.0)
    mass = np.full(n, 3500.0)
    temp = np.full(n, 15.0)
    activity = np.ones(n)
    sal_cost = np.ones(n)

    for _ in range(48):
        new_ed, dead, new_mass = update_energy(ed, mass, temp, activity, sal_cost, params)
        # Energy density must never increase for starving fish
        assert np.all(new_ed[~dead] <= ed[~dead] + 1e-12), \
            f"ED increased: max delta = {(new_ed[~dead] - ed[~dead]).max()}"
        ed = new_ed
        mass = new_mass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_bioenergetics.py::test_energy_density_decreases_monotonically -v`
Expected: FAIL — ED increases because mass shrinks faster than energy

- [ ] **Step 3: Fix `update_energy` — proportional mass loss**

Replace `salmon_ibm/bioenergetics.py` lines 43–52 with:

```python
    r_hourly = hourly_respiration(mass_g, temperature_c, activity_mult, params) * salinity_cost
    e_total_j = ed_kJ_g * 1000.0 * mass_g
    new_e_total_j = np.maximum(e_total_j - r_hourly, 0.0)
    # Proportional mass loss: mass shrinks in proportion to energy loss.
    # This keeps ED constant under pure respiration (standard Wisconsin model).
    energy_fraction = np.where(e_total_j > 0, new_e_total_j / e_total_j, 0.0)
    new_mass = mass_g * energy_fraction
    # Floor at 50% original mass to prevent numerical collapse
    new_mass = np.maximum(new_mass, mass_g * 0.5)
    new_ed = np.where(new_mass > 0, new_e_total_j / (new_mass * 1000.0), 0.0)
    dead = new_ed < params.ED_MORTAL
    return new_ed, dead, new_mass
```

**Note:** With proportional mass loss, ED should stay constant (not increase) because both numerator and denominator shrink by the same fraction. The test verifies `new_ed <= ed + 1e-12` to account for floating-point rounding.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n shiny python -m pytest tests/test_bioenergetics.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `conda run -n shiny python -m pytest tests/ -v`

- [ ] **Step 6: Commit**

```
git add salmon_ibm/bioenergetics.py tests/test_bioenergetics.py
git commit -m "fix: derive mass from energy in update_energy — prevent ED inflation"
```

---

### Task 2: Fix `updater_uptake` race condition with repeated cell indices

When multiple agents occupy the same cell, `hex_map[cell_indices[mask]] -= extracted` only applies the last agent's subtraction due to NumPy fancy-indexing semantics. Resources are underconsumed.

**Files:**
- Modify: `salmon_ibm/accumulators.py:355-369`
- Test: `tests/test_accumulators.py`

- [ ] **Step 1: Write failing test — multiple agents on same cell**

Add to `tests/test_accumulators.py`:

```python
def test_updater_uptake_multi_agent_same_cell():
    """Two agents on the same cell should each deplete the resource."""
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_uptake
    mgr = AccumulatorManager(2, [AccumulatorDef("food")])
    hex_map = np.array([100.0, 50.0, 50.0])
    cell_indices = np.array([0, 0])  # both agents on cell 0
    mask = np.array([True, True])
    updater_uptake(mgr, "food", mask, hex_map=hex_map, cell_indices=cell_indices, rate=0.1)
    # Each agent extracts 100 * 0.1 = 10. Total depletion should be 20.
    assert hex_map[0] == pytest.approx(80.0), f"Expected 80.0, got {hex_map[0]}"
    # Each agent should have received 10.0
    assert mgr.data[0, 0] == pytest.approx(10.0)
    assert mgr.data[1, 0] == pytest.approx(10.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_accumulators.py::test_updater_uptake_multi_agent_same_cell -v`
Expected: FAIL — hex_map[0] is 90.0 instead of 80.0

- [ ] **Step 3: Fix `updater_uptake` with `np.subtract.at`**

Replace `salmon_ibm/accumulators.py` lines 355–369 with:

```python
def updater_uptake(
    manager, acc_name: str, mask, *, hex_map, cell_indices, rate=1.0,
):
    """Transfer value from hex-map into accumulator (resource extraction)."""
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]
    cells = cell_indices[mask]
    extracted = hex_map[cells] * rate
    new_vals = manager.data[mask, idx] + extracted
    if defn.min_val is not None:
        new_vals = np.maximum(new_vals, defn.min_val)
    if defn.max_val is not None:
        new_vals = np.minimum(new_vals, defn.max_val)
    manager.data[mask, idx] = new_vals
    # Use unbuffered subtract to handle repeated cell indices correctly
    np.subtract.at(hex_map, cells, extracted)
```

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_accumulators.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```
git add salmon_ibm/accumulators.py tests/test_accumulators.py
git commit -m "fix: use np.subtract.at in updater_uptake for correct multi-agent depletion"
```

---

### Task 3: Harden `step_alive_mask` usage — recompute in `_event_update_timers`

`landscape["step_alive_mask"]` is computed once at step start. In the default event order, `update_timers` (position 6) runs BEFORE `bioenergetics` (position 7), so stale masks don't manifest. However, if events are reordered via YAML config, dead agents could have timers incremented. This is a defensive fix to prevent bugs in custom event orderings.

**Files:**
- Modify: `salmon_ibm/simulation.py:156-161`

- [ ] **Step 1: Fix `_event_update_timers` to recompute mask**

In `salmon_ibm/simulation.py`, change `_event_update_timers` (lines 156–161):

```python
    def _event_update_timers(self, population, landscape, t, mask):
        # Recompute alive mask — defensive against custom event orderings
        # where mortality events may run before this callback.
        step_mask = population.alive & ~population.arrived
        population.steps[step_mask] += 1
        population.target_spawn_hour[step_mask] = np.maximum(
            population.target_spawn_hour[step_mask] - 1, 0
        )
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py -v`

- [ ] **Step 3: Commit**

```
git add salmon_ibm/simulation.py
git commit -m "fix: recompute alive mask in _event_update_timers — defensive against event reordering"
```

---

### Task 4: Add DO threshold validation

`do_override` silently produces wrong results if `lethal > high` (misconfigured).

**Files:**
- Modify: `salmon_ibm/estuary.py:24-35`
- Test: `tests/test_estuary.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_estuary.py`:

```python
def test_do_override_rejects_inverted_thresholds():
    """lethal > high should raise ValueError."""
    from salmon_ibm.estuary import do_override
    with pytest.raises(ValueError, match="lethal.*must be.*<=.*high"):
        do_override(np.array([3.0]), lethal=5.0, high=2.0)
```

- [ ] **Step 2: Add validation to `do_override`**

In `salmon_ibm/estuary.py`, add after line 28 (inside the function, before `result = ...`):

```python
    if lethal > high:
        raise ValueError(
            f"lethal threshold ({lethal}) must be <= high threshold ({high})"
        )
```

- [ ] **Step 3: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_estuary.py -v`

- [ ] **Step 4: Commit**

```
git add salmon_ibm/estuary.py tests/test_estuary.py
git commit -m "fix: validate DO thresholds — reject lethal > high"
```

---

### Task 5: Fix `apply_overrides` writing to dead agents

`apply_overrides` operates on the full pool (including dead agents), then the result replaces the entire behavior array.

**Files:**
- Modify: `salmon_ibm/simulation.py:146`
- Test: `tests/test_behavior.py`

- [ ] **Step 1: Write test documenting correct behavior after fix**

Add to `tests/test_behavior.py`:

```python
def test_apply_overrides_returns_upstream_for_step_zero():
    """Verify apply_overrides sets UPSTREAM for step==0 agents (baseline behavior)."""
    from salmon_ibm.agents import AgentPool, Behavior
    from salmon_ibm.behavior import apply_overrides, BehaviorParams
    pool = AgentPool(n=5, start_tri=np.zeros(5, dtype=int), rng_seed=42)
    pool.steps[:] = 0  # triggers first_move override
    pool.behavior[:] = Behavior.HOLD
    params = BehaviorParams.defaults()
    result = apply_overrides(pool, params)
    # All agents at step 0 should be UPSTREAM
    assert np.all(result == Behavior.UPSTREAM)
```

Note: The actual fix is in `simulation.py` where the result is applied. `apply_overrides` itself operates on the full pool (correct), but `simulation.py` should only write the result to alive agents. The test above documents baseline behavior; the integration-level constraint (dead agents unaffected) is verified by the simulation tests.

- [ ] **Step 2: Fix in `simulation.py` — apply overrides only to alive agents**

In `salmon_ibm/simulation.py`, change line 146:

```python
        overridden = apply_overrides(population, self.beh_params)
        alive_mask = population.alive & ~population.arrived
        population.behavior[alive_mask] = overridden[alive_mask]
```

- [ ] **Step 3: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_behavior.py tests/test_simulation.py -v`

- [ ] **Step 4: Commit**

```
git add salmon_ibm/simulation.py tests/test_behavior.py
git commit -m "fix: apply behavior overrides only to alive agents"
```

---

### Task 6: Narrow `except Exception` in HexSimAccumulateEvent

The broad catch swallows all errors silently. Narrow to expected types and re-raise unexpected ones.

**Files:**
- Modify: `salmon_ibm/events_hexsim.py:259-305`
- Test: `tests/test_events_hexsim.py`

- [ ] **Step 1: Write test — expected errors warn, unexpected errors propagate**

Add to `tests/test_events_hexsim.py`:

```python
def test_accumulate_event_warns_on_known_errors():
    """KeyError/ValueError in accumulator dispatch should warn, not crash."""
    from salmon_ibm.events_hexsim import HexSimAccumulateEvent
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
    from salmon_ibm.population import Population
    from salmon_ibm.agents import AgentPool
    import warnings

    pool = AgentPool(n=3, start_tri=np.zeros(3, dtype=int))
    pop = Population(name="test", pool=pool)
    pop.accumulator_mgr = AccumulatorManager(3, [AccumulatorDef("test_acc")])

    evt = HexSimAccumulateEvent(
        name="bad_acc",
        updater_functions=[{
            "function": "QuantifyLocation",
            "accumulator": "test_acc",
            "spatial_data": "nonexistent_layer",
        }],
    )
    mask = np.ones(3, dtype=bool)
    landscape = {"rng": np.random.default_rng(42), "spatial_data": {}, "global_variables": {}}
    # Should not raise — missing spatial data is a known error case
    evt.execute(pop, landscape, 0, mask)


def test_accumulate_event_propagates_type_error():
    """TypeError in accumulator dispatch should NOT be swallowed after fix."""
    from salmon_ibm.events_hexsim import HexSimAccumulateEvent
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
    from salmon_ibm.population import Population
    from salmon_ibm.agents import AgentPool

    pool = AgentPool(n=3, start_tri=np.zeros(3, dtype=int))
    pop = Population(name="test", pool=pool)
    pop.accumulator_mgr = AccumulatorManager(3, [AccumulatorDef("test_acc")])

    # Patch dispatch to inject a TypeError
    evt = HexSimAccumulateEvent(
        name="bad_acc",
        updater_functions=[{
            "function": "Clear",
            "accumulator": "test_acc",
        }],
    )
    evt._init_dispatch()
    evt._dispatch["Clear"] = lambda *a, **kw: (_ for _ in ()).throw(TypeError("injected"))

    mask = np.ones(3, dtype=bool)
    landscape = {"rng": np.random.default_rng(42), "spatial_data": {}, "global_variables": {}}
    with pytest.raises(TypeError, match="injected"):
        evt.execute(pop, landscape, 0, mask)
```

- [ ] **Step 2: Narrow the except clause**

In `salmon_ibm/events_hexsim.py`, replace lines 300–305:

```python
            except (KeyError, ValueError, IndexError) as e:
                import warnings
                warnings.warn(
                    f"Updater {func_name} for '{acc_name}' failed: {e}",
                    RuntimeWarning, stacklevel=2,
                )
```

- [ ] **Step 3: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_events_hexsim.py -v`

- [ ] **Step 4: Commit**

```
git add salmon_ibm/events_hexsim.py tests/test_events_hexsim.py
git commit -m "fix: narrow except clause in HexSimAccumulateEvent — propagate unexpected errors"
```

---

### Task 7: Warn loudly on missing temperature data

Silent fallback to constant 15C disables the entire thermal ecology model.

**Files:**
- Modify: `salmon_ibm/hexsim_env.py:67-70`
- Test: `tests/test_hexsim.py`

- [ ] **Step 1: Add warning to temperature fallback**

In `salmon_ibm/hexsim_env.py`, replace lines 67–70:

```python
        if not self._has_temperature:
            import warnings
            warnings.warn(
                "No temperature zone data found in workspace. "
                "Using constant 15°C for all cells. "
                "Temperature-dependent processes (respiration, behavior, thermal mortality) "
                "will not function correctly.",
                UserWarning, stacklevel=2,
            )
            self._zone_ids = np.zeros(mesh.n_cells, dtype=int)
            self._temp_table = np.full((1, 1), 15.0, dtype=np.float32)
            self.n_timesteps = 1
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest tests/ -v`

- [ ] **Step 3: Commit**

```
git add salmon_ibm/hexsim_env.py
git commit -m "fix: emit loud warning when temperature data is missing from workspace"
```

---

### Task 8: Warn on missing gradient data (movement silently disabled)

Missing gradient silently replaced with `np.ones`, making agents immobile.

**Files:**
- Modify: `salmon_ibm/events_hexsim.py:491-495`

- [ ] **Step 1: Add warning when gradient is missing**

In `salmon_ibm/events_hexsim.py`, replace lines 492–494:

```python
            gradient = spatial_data.get(self.dispersal_spatial_data)
            if gradient is None:
                import warnings
                available = list(spatial_data.keys()) if spatial_data else []
                warnings.warn(
                    f"HexSimMoveEvent '{self.name}': spatial data layer "
                    f"'{self.dispersal_spatial_data}' not found. "
                    f"Available layers: {available}. "
                    f"Falling back to uniform gradient (no directed movement).",
                    UserWarning, stacklevel=2,
                )
                gradient = np.ones(mesh.n_cells)
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest tests/ -v`

- [ ] **Step 3: Commit**

```
git add salmon_ibm/events_hexsim.py
git commit -m "fix: warn when gradient layer missing — movement falls back to uniform"
```

---

### Task 9: Log warning on NaN/Inf in expression evaluator

`nan_to_num` silently converts computation errors to zeros.

**Files:**
- Modify: `salmon_ibm/accumulators.py:200-205`

- [ ] **Step 1: Add NaN/Inf detection before `nan_to_num`**

In `salmon_ibm/accumulators.py`, insert before line 203 (before the `nan_to_num` call):

```python
    n_bad = np.count_nonzero(~np.isfinite(result))
    if n_bad > 0:
        import warnings
        warnings.warn(
            f"Expression '{expression}' produced {n_bad} NaN/Inf values "
            f"out of {len(result)} agents. Replacing with bounds/zero.",
            RuntimeWarning, stacklevel=2,
        )
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_accumulators.py -v`

- [ ] **Step 3: Commit**

```
git add salmon_ibm/accumulators.py
git commit -m "fix: warn when expression evaluator produces NaN/Inf values"
```

---

## Tier 2 — Edge Cases & Type Safety

### Task 10: Warn on silent trait-combo mask fallback

`_apply_trait_combo_mask` silently returns full mask on data inconsistency, applying events to wrong agents.

**Files:**
- Modify: `salmon_ibm/events_hexsim.py:147-193`

- [ ] **Step 1: Add warnings at each fallback point**

In `salmon_ibm/events_hexsim.py`, at each `return base_mask` in `_apply_trait_combo_mask`:

Line 159 (missing stratified/combos):
```python
    if not stratified or combos_str is None:
        return base_mask  # intentional: no filtering requested
```

Line 162 (missing trait_mgr) — add warning:
```python
    if trait_mgr is None:
        import warnings
        warnings.warn(
            f"Trait-combo filter requested but population has no trait_mgr. "
            f"Event will fire for ALL agents.",
            RuntimeWarning, stacklevel=2,
        )
        return base_mask
```

Line 182 (unknown trait name) — add warning:
```python
        if tname not in trait_mgr.definitions:
            import warnings
            warnings.warn(
                f"Trait '{tname}' not found in trait_mgr. "
                f"Available: {list(trait_mgr.definitions.keys())}. "
                f"Event will fire for ALL agents.",
                RuntimeWarning, stacklevel=2,
            )
            return base_mask
```

Line 189 (combo size mismatch) — add warning:
```python
    if len(combo_flags) != stride:
        import warnings
        warnings.warn(
            f"Trait combo flags length ({len(combo_flags)}) doesn't match "
            f"expected stride ({stride}). Event will fire for ALL agents.",
            RuntimeWarning, stacklevel=2,
        )
        return base_mask
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest tests/ -v`

- [ ] **Step 3: Commit**

```
git add salmon_ibm/events_hexsim.py
git commit -m "fix: warn when trait-combo mask falls back to unfiltered"
```

---

### Task 11: Pre-load xarray data in Environment (performance)

`Environment.advance()` does 5 xarray reads per step. Pre-load into NumPy arrays at init.

**Files:**
- Modify: `salmon_ibm/environment.py:10-54`
- Test: `tests/test_environment.py`

- [ ] **Step 1: Pre-load data in `__init__`**

In `salmon_ibm/environment.py`, add after `self._var` dict (line 43), pre-load all variables:

```python
        # Pre-load all time-varying data into memory for fast advance()
        self._preloaded: dict[str, np.ndarray] = {}
        for field_name, var_name in self._var.items():
            raw = self._phy[var_name].values  # shape: (n_times, ...)
            self._preloaded[field_name] = raw
```

- [ ] **Step 2: Simplify `advance()` to use pre-loaded data**

Replace `advance()` body:

```python
    def advance(self, t: int):
        self._prev_ssh = self.fields.get("ssh")
        self.current_t = t
        t_idx = t % self.n_timesteps

        for field_name in self._var:
            raw = self._preloaded[field_name][t_idx]
            flat = raw.ravel()
            tri_vals = flat[self.mesh.triangles].mean(axis=1)
            self.fields[field_name] = tri_vals
```

- [ ] **Step 3: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_environment.py tests/test_simulation.py -v`

- [ ] **Step 4: Commit**

```
git add salmon_ibm/environment.py
git commit -m "perf: pre-load xarray data at init — avoid 5 dataset reads per step"
```

---

### Task 12: Add `parallel=True` to Numba movement kernels (performance)

The three Numba kernels in `events_hexsim.py` use sequential `for i in range(n)` instead of `prange`.

**Files:**
- Modify: `salmon_ibm/events_hexsim.py:24-137`

- [ ] **Step 1: Add `prange` import**

At top of file, update the numba import block:

```python
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range
```

- [ ] **Step 2: Add `parallel=True` to `_move_gradient_numba` and change inner loop**

Change line 24: `@njit(cache=True)` → `@njit(cache=True, parallel=True)`
Change line 37: `for i in range(n):` → `for i in prange(n):`

**Important:** The outer `for step in range(n_steps)` must remain sequential (agents read positions written by prior steps). Only the inner agent loop can be parallel.

NOTE: Each agent reads only its own `positions[i]` and the read-only gradient, never another agent's position, so `positions` is race-free. However, `any_moved` is a shared boolean written by multiple threads. Under `prange`, this is a benign race (worst case: an extra iteration). To be safe, **remove the `any_moved` early-exit** when using `parallel=True`. Delete the `any_moved` variable, the `any_moved = True` assignment, and the `if not any_moved: break` check from both `_move_gradient_numba` and `_move_affinity_numba`. The early-exit is a micro-optimization that rarely triggers (most steps have at least one moving agent). Simply iterate for all `n_steps`.

- [ ] **Step 3: Add `parallel=True` to `_set_affinity_numba`**

Change line 111: `@njit(cache=True)` → `@njit(cache=True, parallel=True)`
Change line 117: `for i in range(n):` → `for i in prange(n):`

This kernel is fully independent per agent (reads gradient, writes only to `targets[i]`).

- [ ] **Step 4: Parallelize `_move_affinity_numba` (same treatment)**

`_move_affinity_numba` has the same structure as `_move_gradient_numba` — each agent reads its own `positions[i]` and `targets[i]`, so it is safe to parallelize the inner loop. Remove the `any_moved` early-exit (same as above). Apply the same treatment:

Change line 68: `@njit(cache=True)` → `@njit(cache=True, parallel=True)`
Change line 80: `for i in range(n):` → `for i in prange(n):`

- [ ] **Step 5: Delete Numba cache files to force recompilation**

```bash
find . -name "*.nbi" -o -name "*.nbc" | head -20
# Delete all Numba caches in salmon_ibm
find ./salmon_ibm -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

- [ ] **Step 6: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_events_hexsim.py tests/test_simulation.py -v`

- [ ] **Step 7: Commit**

```
git add salmon_ibm/events_hexsim.py
git commit -m "perf: add parallel=True + prange to Numba movement kernels"
```

---

## Tier 3 — Robustness & Test Coverage

### Task 13: Add zero-agent edge case tests

No tests cover empty pools or all-dead populations.

**Files:**
- Modify: `tests/test_agents.py`, `tests/test_events.py`

- [ ] **Step 1: Add zero-agent AgentPool test**

Add to `tests/test_agents.py`:

```python
def test_agent_pool_zero_agents():
    """AgentPool with n=0 should not crash."""
    from salmon_ibm.agents import AgentPool
    pool = AgentPool(n=0, start_tri=np.array([], dtype=int))
    assert pool.n == 0
    assert len(pool.alive) == 0
    assert pool.t3h_mean().shape == (0,)
```

- [ ] **Step 2: Add all-dead event execution test**

Add to `tests/test_events.py`:

```python
def test_event_execution_with_all_dead_mask():
    """Events should handle all-False mask without error."""
    from salmon_ibm.events import EventSequencer, EveryStep
    from salmon_ibm.events_builtin import CustomEvent

    called = []
    def cb(pop, landscape, t, mask):
        called.append(mask.sum())

    seq = EventSequencer([CustomEvent(name="test", callback=cb)])
    # Create a population where all agents are dead
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    pool = AgentPool(n=5, start_tri=np.zeros(5, dtype=int))
    pool.alive[:] = False
    pop = Population(name="test", pool=pool)
    seq.step(pop, {}, 0)
    assert called == [0]
```

- [ ] **Step 3: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_agents.py::test_agent_pool_zero_agents tests/test_events.py::test_event_execution_with_all_dead_mask -v`

- [ ] **Step 4: Commit**

```
git add tests/test_agents.py tests/test_events.py
git commit -m "test: add zero-agent and all-dead edge case tests"
```

---

### Task 14: Convert DO states to IntEnum

`DO_OK`, `DO_ESCAPE`, `DO_LETHAL` are bare integer constants that could cause comparison bugs.

**Files:**
- Modify: `salmon_ibm/estuary.py:19-21`
- Test: `tests/test_estuary.py`

- [ ] **Step 1: Convert to IntEnum**

In `salmon_ibm/estuary.py`, replace lines 19–21:

```python
from enum import IntEnum

class DOState(IntEnum):
    OK = 0
    ESCAPE = 1
    LETHAL = 2

DO_OK = DOState.OK
DO_ESCAPE = DOState.ESCAPE
DO_LETHAL = DOState.LETHAL
```

- [ ] **Step 2: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_estuary.py tests/test_simulation.py -v`
Expected: ALL PASS (IntEnum is backward-compatible with int comparisons)

- [ ] **Step 3: Commit**

```
git add salmon_ibm/estuary.py
git commit -m "refactor: convert DO states to IntEnum for type safety"
```

---

### Task 15: Consolidate duplicate test fixtures into conftest

`FakePopulation`/`MockPopulation` is defined in 3+ test files.

**Files:**
- Modify: `tests/conftest.py`, `tests/test_events.py`, `tests/test_events_phase3.py`, `tests/test_interactions.py`

- [ ] **Step 1: Identify all duplicate fixtures**

```bash
grep -rn "class FakePopulation\|class MockPopulation\|class FakeMesh" tests/
```

- [ ] **Step 2: Move common fixtures to `tests/conftest.py`**

Add to `tests/conftest.py` (after existing fixtures) the consolidated `FakePopulation` class and a `@pytest.fixture` that creates it.

- [ ] **Step 3: Remove duplicates from individual test files**

Update each test file to import from conftest instead of defining its own.

- [ ] **Step 4: Run full test suite**

Run: `conda run -n shiny python -m pytest tests/ -v`

- [ ] **Step 5: Commit**

```
git add tests/conftest.py tests/test_events.py tests/test_events_phase3.py tests/test_interactions.py
git commit -m "refactor: consolidate duplicate test fixtures into conftest.py"
```

---

## Verification Checklist

After all tasks are complete:

- [ ] Run full test suite: `conda run -n shiny python -m pytest tests/ -v`
- [ ] Verify no new warnings in clean run
- [ ] Check that bioenergetics ED is monotonically decreasing (Task 1)
- [ ] Check that DO threshold validation works (Task 4)
- [ ] Check that Numba kernels recompile successfully (Task 12)
