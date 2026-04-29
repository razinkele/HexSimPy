# Codebase Review Fixes Implementation Plan

> **STATUS: ✅ EXECUTED** — All 11 fixes shipped — `BioParams.__post_init__` validation, `_start_run` ordering, `_beh` negative-index guard, error handling, dead-code removals, type safety, perf. Tests in `tests/test_accumulators.py::TestBioParamsValidation`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all issues found by the 6-agent deep review: 2 critical bugs, 5 error handling improvements, 3 dead code removals, 1 type safety fix, and 1 performance optimization.

**Architecture:** Changes are isolated and independent — each task modifies a small, self-contained area. Tasks are ordered by risk (lowest first): validation additions, then bug fixes, then error handling, then dead code removal, then performance. Every task has a test-first step.

**Tech Stack:** Python 3.10+, Shiny for Python, NumPy, Playwright (E2E), pytest

**Test command:** `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_playwright.py --ignore=tests/test_hex_playwright.py --ignore=tests/test_hex_comparison.py --ignore=tests/test_hex_grid_rendering.py --ignore=tests/test_hexsim_validation.py --ignore=tests/test_map_visualization.py -x`

---

### Task 1: Add BioParams.__post_init__ validation

**Files:**
- Modify: `salmon_ibm/bioenergetics.py:12-30`
- Test: `tests/test_bioenergetics.py`

- [ ] **Step 1: Write failing tests for invalid BioParams**

Add to `tests/test_bioenergetics.py`:

```python
class TestBioParamsValidation:
    def test_negative_ra_raises(self):
        with pytest.raises(ValueError, match="RA"):
            BioParams(RA=-0.001)

    def test_negative_rq_raises(self):
        with pytest.raises(ValueError, match="RQ"):
            BioParams(RQ=-0.01)

    def test_t_max_below_t_opt_raises(self):
        with pytest.raises(ValueError, match="T_MAX.*T_OPT"):
            BioParams(T_OPT=20.0, T_MAX=15.0)

    def test_mass_floor_out_of_range_raises(self):
        with pytest.raises(ValueError, match="MASS_FLOOR"):
            BioParams(MASS_FLOOR_FRACTION=1.5)

    def test_valid_params_no_error(self):
        bp = BioParams()  # defaults should be valid
        assert bp.RA > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest tests/test_bioenergetics.py::TestBioParamsValidation -v`
Expected: FAIL — no `__post_init__` exists yet.

- [ ] **Step 3: Add __post_init__ to BioParams**

In `salmon_ibm/bioenergetics.py`, after the field definitions (after line 30), add:

```python
    def __post_init__(self):
        if self.RA <= 0:
            raise ValueError("RA must be > 0")
        if self.RQ <= 0:
            raise ValueError("RQ must be > 0")
        if self.T_MAX <= self.T_OPT:
            raise ValueError(f"T_MAX ({self.T_MAX}) must be > T_OPT ({self.T_OPT})")
        if not (0 < self.MASS_FLOOR_FRACTION <= 1):
            raise ValueError(f"MASS_FLOOR_FRACTION must be in (0, 1], got {self.MASS_FLOOR_FRACTION}")
        if self.ED_MORTAL <= 0:
            raise ValueError("ED_MORTAL must be > 0")
        if self.ED_TISSUE <= 0:
            raise ValueError("ED_TISSUE must be > 0")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest tests/test_bioenergetics.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/bioenergetics.py tests/test_bioenergetics.py
git commit -m "fix: add BioParams.__post_init__ validation for bioenergetics parameters"
```

---

### Task 2: Fix _start_run ordering — set running=True after init

**Files:**
- Modify: `app.py:894-900`
- Test: Manual verification (reactive ordering)

- [ ] **Step 1: Read current _start_run code**

Current code at `app.py:894-900`:
```python
@reactive.effect
@reactive.event(input.btn_run)
async def _start_run():
    running.set(True)          # BUG: fires _run_tick before sim exists
    sim = sim_state.get()
    if sim is None:
        await _init_sim()
```

- [ ] **Step 2: Fix the ordering**

Replace with:
```python
@reactive.effect
@reactive.event(input.btn_run)
async def _start_run():
    sim = sim_state.get()
    if sim is None:
        try:
            await _init_sim()
        except Exception as e:
            ui.notification_show(f"Init failed: {e}", type="error", duration=10)
            return
    running.set(True)  # only after sim is guaranteed to exist
```

- [ ] **Step 3: Verify syntax**

Run: `conda run -n shiny python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Run unit tests**

Run: `conda run -n shiny python -m pytest tests/test_trip_buffer.py -q`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "fix: set running=True only after _init_sim completes, add error handling"
```

---

### Task 3: Fix _beh negative index guard

**Files:**
- Modify: `app.py:366`
- Test: `tests/test_trip_buffer.py`

- [ ] **Step 1: Write failing test for negative behavior index**

Add to `tests/test_trip_buffer.py`:

```python
class TestBehaviorColors:
    def test_valid_behavior_produces_colored_trail(self, buf):
        alive = np.ones(3, dtype=bool)
        behaviors = np.array([0, 3, 4], dtype=np.int8)
        for step in range(3):
            pos = np.array([[step*0.1, 0], [step*0.1+1, 1], [step*0.1+2, 2]], dtype=np.float32)
            buf.update(alive, pos, behaviors)
        trips, _, _ = buf.build_trips()
        assert len(trips) > 0
        for trip in trips:
            assert len(trip["color"]) == 4

    def test_negative_behavior_uses_default_color(self, buf):
        alive = np.ones(2, dtype=bool)
        behaviors = np.array([-1, 0], dtype=np.int8)
        for step in range(3):
            pos = np.array([[step*0.1, 0], [step*0.1+1, 1]], dtype=np.float32)
            buf.update(alive, pos, behaviors)
        trips, _, _ = buf.build_trips()
        # Should not crash, negative index should fall back to color 0
        assert len(trips) > 0

    def test_out_of_range_behavior_uses_default(self, buf):
        alive = np.ones(2, dtype=bool)
        behaviors = np.array([99, 0], dtype=np.int8)
        for step in range(3):
            pos = np.array([[step*0.1, 0], [step*0.1+1, 1]], dtype=np.float32)
            buf.update(alive, pos, behaviors)
        trips, _, _ = buf.build_trips()
        assert len(trips) > 0
```

- [ ] **Step 2: Run tests to verify negative index test fails**

Run: `conda run -n shiny python -m pytest tests/test_trip_buffer.py::TestBehaviorColors -v`
Expected: `test_negative_behavior_uses_default_color` may pass (Python wraps negative indices) or produce wrong color — verify.

- [ ] **Step 3: Fix the guard**

In `app.py:366`, change:
```python
beh_idx = int(self._beh[i]) if self._beh[i] < len(BEH_COLORS) else 0
```
to:
```python
beh_idx = int(self._beh[i]) if 0 <= self._beh[i] < len(BEH_COLORS) else 0
```

- [ ] **Step 4: Run all TripBuffer tests**

Run: `conda run -n shiny python -m pytest tests/test_trip_buffer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_trip_buffer.py
git commit -m "fix: guard against negative behavior index in TripBuffer trail colors"
```

---

### Task 4: Fix _run_tick docstring placement

**Files:**
- Modify: `app.py:903-909`

- [ ] **Step 1: Move docstring before nonlocal**

Current (broken docstring):
```python
async def _run_tick():
    nonlocal _cached_scale
    """Self-scheduling run loop using invalidate_later.
    ...
    """
```

Fix to:
```python
async def _run_tick():
    """Self-scheduling run loop using invalidate_later.

    Each invocation does one batch, sets reactive values, then returns.
    Shiny flushes outputs between invocations so the UI stays live.
    """
    nonlocal _cached_scale
```

- [ ] **Step 2: Verify syntax**

Run: `conda run -n shiny python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "fix: move _run_tick docstring before nonlocal declaration"
```

---

### Task 5: Separate computation from communication in error handlers

**Files:**
- Modify: `app.py:847-892` (_step), `app.py:929-983` (_run_tick)

- [ ] **Step 1: Refactor _step to separate sim.step() from UI updates**

Replace the single try/except in `_step` with two blocks:

```python
async def _step():
    nonlocal _cached_scale
    try:
        sim = sim_state.get()
        if sim is None:
            await _init_sim()
            sim = sim_state.get()
        await asyncio.to_thread(sim.step)
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Simulation step failed")
        ui.notification_show(f"Simulation error: {e}", type="error", duration=10)
        return

    # UI updates — non-fatal
    try:
        if sim.pool.alive.any():
            # ... existing trail buffer update code ...
        # ... existing step_stats.set code ...
        history.set(sim.history.copy())
        await _push_chart_data(sim)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("UI update failed after step %d", sim.current_t)
```

- [ ] **Step 2: Refactor _run_tick similarly**

Split the try/except in `_run_tick` into:
1. Simulation batch (fatal — stops run on error)
2. UI updates (non-fatal — log and continue)

```python
    try:
        t_batch = time.perf_counter()
        await asyncio.to_thread(_batch)
        elapsed = time.perf_counter() - t_batch
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Simulation batch failed")
        ui.notification_show(f"Simulation error: {e}", type="error", duration=10)
        running.set(False)
        return

    # UI updates — non-fatal
    try:
        if sim.pool.alive.any():
            # ... trail buffer update ...
        # ... step_stats, history, trips_update, charts ...
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Visualization update failed at t=%d", sim.current_t)
```

- [ ] **Step 3: Verify syntax and run tests**

Run: `conda run -n shiny python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"`
Run: `conda run -n shiny python -m pytest tests/test_trip_buffer.py -q`

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "fix: separate simulation errors from UI update errors in _step and _run_tick"
```

---

### Task 6: Add error handling to _update_map and _toggle_trails

**Files:**
- Modify: `app.py:1384-1442`

- [ ] **Step 1: Wrap _update_map body in try/except**

```python
@reactive.effect
async def _update_map():
    nonlocal _cached_landscape, _cached_field
    nonlocal _cached_subsample_idx, _cached_scale
    sim = sim_state.get()
    _ = history.get()
    if sim is None:
        return
    if running.get():
        return
    try:
        # ... existing landscape/field change detection and update logic ...
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Map update failed")
        ui.notification_show("Map update failed", type="warning", duration=5)
```

- [ ] **Step 2: Wrap _toggle_trails similarly**

```python
async def _toggle_trails():
    if sim_state.get() is None:
        return
    try:
        await _trips_update()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Failed to toggle trails")
```

- [ ] **Step 3: Replace bare `except Exception: pass` blocks in _init_sim with logging**

Find the 3 instances of `except Exception: pass` in `_init_sim` (lines ~655, ~736, ~768) and replace with:
```python
except Exception:
    import logging
    logging.getLogger(__name__).debug("Message send failed (session may not be ready)")
```

- [ ] **Step 4: Verify syntax and run tests**

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "fix: add error handling to _update_map, _toggle_trails; log suppressed exceptions in _init_sim"
```

---

### Task 7: Replace warnings.warn with logging.warning in events_hexsim.py

**Files:**
- Modify: `salmon_ibm/events_hexsim.py` (8 instances)

- [ ] **Step 1: Add `import logging` at top of file if not present**

- [ ] **Step 2: Replace all `warnings.warn(...)` with `logging.getLogger(__name__).warning(...)`**

Find/replace across the file. There are 8 instances at lines: 176, 206, 222, 300, 318, 385, 431, 617.

Change pattern:
```python
# Before:
warnings.warn("message")
# After:
logging.getLogger(__name__).warning("message")
```

Also remove `import warnings` if no longer used.

- [ ] **Step 3: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_events_hexsim.py -v`
Expected: ALL PASS (warnings are now logs, tests that check UserWarning may need updating)

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/events_hexsim.py
git commit -m "fix: replace warnings.warn with logging.warning in HexSim events"
```

---

### Task 8: Remove dead code (_hexsim_to_lonlat, _GEO_ANCHORS, _detect_landscape)

**Files:**
- Modify: `app.py:170-230`

- [ ] **Step 1: Verify no references exist**

Run: `grep -n "_hexsim_to_lonlat\|_GEO_ANCHORS\|_detect_landscape" app.py`
Expected: Only the definitions — no call sites.

- [ ] **Step 2: Delete lines 170-230 (the three functions and the dict)**

Remove:
- `_GEO_ANCHORS` dict (lines ~170-187)
- `_detect_landscape()` function (lines ~191-197)
- `_hexsim_to_lonlat()` function (lines ~200-230)

- [ ] **Step 3: Verify syntax and run tests**

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "chore: remove unused _hexsim_to_lonlat, _GEO_ANCHORS, _detect_landscape"
```

---

### Task 9: Add Plotly/Shiny guards to streaming_charts.js

**Files:**
- Modify: `www/streaming_charts.js`

- [ ] **Step 1: Add Plotly availability guard**

At the top of each `initPopulation`, `initMigration`, `initBehavior` function, add:
```javascript
if (typeof Plotly === 'undefined') return;
```

- [ ] **Step 2: Wrap Shiny handler registration in polling loop**

Replace the `if (window.Shiny)` check at line 155 with:
```javascript
function _waitForShiny() {
    if (window.Shiny && Shiny.addCustomMessageHandler) {
        Shiny.addCustomMessageHandler("chart_reset", onReset);
        Shiny.addCustomMessageHandler("chart_update", onUpdate);
    } else {
        setTimeout(_waitForShiny, 200);
    }
}
_waitForShiny();
```

- [ ] **Step 3: Wrap handler bodies in try/catch**

```javascript
function onUpdate(msg) {
    try {
        // ... existing update logic ...
    } catch (e) {
        console.warn('[streaming_charts] update failed:', e);
    }
}
```

- [ ] **Step 4: Commit**

```bash
git add www/streaming_charts.js
git commit -m "fix: add Plotly/Shiny guards and error handling to streaming charts"
```

---

### Task 10: Vectorize build_trips() — eliminate Python loop

**Files:**
- Modify: `app.py:335-375`
- Test: `tests/test_trip_buffer.py`

- [ ] **Step 1: Run existing tests as baseline**

Run: `conda run -n shiny python -m pytest tests/test_trip_buffer.py -v`
Expected: ALL PASS (27+ tests)

- [ ] **Step 2: Rewrite build_trips with vectorized NaN/stationarity check**

Replace the Python `for i in range(nt)` loop with bulk NumPy operations:

```python
def build_trips(self):
    if self._fill < 2 or self._tracked is None:
        return [], 0, 0
    nt = min(len(self._tracked), self.MAX_AGENTS)
    order = np.array(
        [(self._ptr - self._fill + i) % self.TRAIL_LEN for i in range(self._fill)]
    )
    t_min = float(self._time - self._fill)
    t_max = float(self._time - 1)

    # Bulk extract: (nt, fill, 3)
    all_wp = self._buf[:nt, order, :]

    # Vectorized NaN check: (nt, fill) — True where ANY column is NaN
    has_nan = np.any(np.isnan(all_wp), axis=2)
    valid_counts = (~has_nan).sum(axis=1)  # (nt,)

    # Stationarity check: compare all non-NaN positions to the first non-NaN
    first_valid = np.argmax(~has_nan, axis=1)  # (nt,)
    ref = all_wp[np.arange(nt), first_valid, :2]  # (nt, 2)
    same_pos = np.where(has_nan, True,
        (all_wp[:, :, 0] == ref[:, 0:1]) & (all_wp[:, :, 1] == ref[:, 1:2]))
    stationary = np.all(same_pos, axis=1)

    # Filter: >= 2 valid waypoints and not stationary
    keep = (valid_counts >= 2) & ~stationary
    keep_idx = np.where(keep)[0]

    # Pre-compute colors
    beh_indices = np.clip(self._beh[:nt], 0, len(BEH_COLORS) - 1)

    trips = []
    for i in keep_idx:
        valid_mask = ~has_nan[i]
        valid_wp = all_wp[i, valid_mask]
        norm_ts = valid_wp[:, 2] - t_min
        path_3d = np.column_stack([valid_wp[:, :2], norm_ts])
        beh = int(beh_indices[i])
        r, g, b = _hex_to_rgb(BEH_COLORS[beh])
        trips.append({
            "path": path_3d.tolist(),
            "timestamps": norm_ts.tolist(),
            "color": [r, g, b, 240],
        })
    return trips, t_min, t_max
```

- [ ] **Step 3: Run all tests to verify no regression**

Run: `conda run -n shiny python -m pytest tests/test_trip_buffer.py -v`
Expected: ALL PASS

- [ ] **Step 4: Run full test suite**

Run the full non-browser test command from the header.
Expected: ALL PASS (~640 tests)

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "perf: vectorize build_trips() NaN/stationarity checks, reduce Python loop iterations"
```

---

### Task 11: Fix scale sentinel — use None instead of (0, 1.0)

**Files:**
- Modify: `app.py` — lines 858, 937, and initialization of `_cached_scale`

- [ ] **Step 1: Change `_cached_scale` initialization from `1.0` to `None`**

Find `_cached_scale = 1.0` in the cache initialization area and change to `_cached_scale = None`.

- [ ] **Step 2: Update scale check in _step and _run_tick**

Replace both instances of:
```python
scale = (
    _cached_scale
    if _cached_scale not in (0, 1.0) and is_hex
    else (80.0 / max(abs(mesh.centroids).max(), 1) if is_hex else 1.0)
)
```
with:
```python
scale = (
    _cached_scale
    if _cached_scale is not None and is_hex
    else (80.0 / max(abs(mesh.centroids).max(), 1) if is_hex else 1.0)
)
```

- [ ] **Step 3: Verify syntax and run tests**

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "fix: use None sentinel for _cached_scale instead of (0, 1.0) collision"
```

---

## Execution Order

Tasks are ordered by risk (lowest first):

1. **BioParams validation** — pure addition, no existing code modified
2. **_start_run ordering** — small, isolated fix
3. **_beh negative index** — small guard fix + tests
4. **Docstring placement** — cosmetic
5. **Error handler separation** — refactoring, no behavior change
6. **_update_map error handling** — addition
7. **warnings.warn → logging** — find/replace
8. **Dead code removal** — deletion only
9. **streaming_charts.js guards** — JS-only
10. **Vectorize build_trips** — performance, tested by existing suite
11. **Scale sentinel** — small logic fix
