# Movement Animation & Live Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add smooth agent movement animation (deck.gl transitions), trail visualization (PathLayer), and a real-time stats bar to the HexSim Shiny app.

**Architecture:** Three features built incrementally on the existing three-tier map update system. Each task produces a working, visually testable change. Binary-encode agent data (replacing Python for-loop), add deck.gl transitions for smooth interpolation, add a NumPy-backed trail buffer rendered as PathLayer, and add a reactive HTML stats strip.

**Tech Stack:** Shiny for Python, shiny-deckgl (deck.gl), NumPy, Plotly

**Spec reference:** `docs/superpowers/specs/2026-03-19-movement-animation-design.md`

**Test command:** `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py --ignore=tests/test_hexsim_validation.py --ignore=tests/test_hexsim_compat.py --ignore=tests/test_columbia_validation.py --tb=short`

**Visual test:** `conda run -n shiny shiny run app.py --port 8765` then open http://localhost:8765

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `app.py` | **Modify** | Binary agent encoding, transition config, trail buffer wiring, live stats output, adaptive sleep, chart throttling |
| `ui/run_controls.py` | **Modify** | Add trail toggle switch + live stats output placeholder |
| `www/style.css` | **Modify** | Live stats bar styling |

---

## Task 1: Binary-Encode Agent Data (replace Python for-loop)

**Priority:** Highest — prerequisite for transitions and performance
**Files:**
- Modify: `app.py:51-54,131,252-280,730-745`

The current `_build_agent_data()` uses a Python for-loop over every alive agent to build JSON dicts. Replace with vectorized NumPy + binary encoding (same pattern as the water layer).

- [ ] **Step 1: Add BEH_COLORS_ARRAY constant**

At `app.py:131`, after `BEH_COLORS_RGB`, add:

```python
BEH_COLORS_ARRAY = np.array(
    [list(_hex_to_rgb(c)) + [240] for c in BEH_COLORS], dtype=np.uint8
)  # (5, 4) — RGBA per behavior
```

- [ ] **Step 2: Create `_build_agent_binary()` function**

Add after `_build_agent_data()` (around line 281):

```python
def _build_agent_binary(sim, mesh, scale=1.0):
    """Build binary-encoded agent position + color arrays for deck.gl."""
    alive = sim.pool.alive
    if not alive.any():
        return None, None, 0
    tris = sim.pool.tri_idx[alive]
    behaviors = sim.pool.behavior[alive]
    is_hexsim = hasattr(mesh, '_edge')

    if is_hexsim:
        positions = np.column_stack([
            mesh.centroids[tris, 1] * scale,
            -mesh.centroids[tris, 0] * scale,
        ]).astype(np.float32)
    else:
        positions = np.column_stack([
            mesh.centroids[tris, 1] * scale,
            mesh.centroids[tris, 0] * scale,
        ]).astype(np.float32)

    colors = BEH_COLORS_ARRAY[behaviors]
    n = int(alive.sum())
    return encode_binary_attribute(positions), encode_binary_attribute(colors), n
```

- [ ] **Step 3: Create `_agent_layer_binary()` that uses binary encoding**

Replace `_agent_layer()` (app.py:730-745) with:

```python
def _agent_layer_binary(sim, scale=1.0, use_transitions=False):
    """Build binary-encoded agent ScatterplotLayer."""
    pos_bin, col_bin, n = _build_agent_binary(sim, sim.mesh, scale=scale)
    if n == 0:
        return scatterplot_layer("agents", {"length": 0},
            getPosition=encode_binary_attribute(np.zeros((0, 2), dtype=np.float32)),
            getFillColor=encode_binary_attribute(np.zeros((0, 4), dtype=np.uint8)),
        )
    props = dict(
        getPosition=pos_bin,
        getFillColor=col_bin,
        getRadius=150,
        radiusMinPixels=5,
        radiusMaxPixels=12,
        stroked=True,
        getLineColor=[0, 0, 0, 140],
        lineWidthMinPixels=1,
        pickable=True,
    )
    if use_transitions:
        props["transitions"] = {"getPosition": {"duration": 200, "type": "spring"}}
    return scatterplot_layer("agents", {"length": n}, **props)
```

- [ ] **Step 4: Add agent-count tracking for transition safety**

In the server function, add a variable to track previous agent count:

```python
    _prev_agent_count = 0
```

Update `_agent_layer_binary` calls to conditionally enable transitions:

```python
    def _should_transition(sim):
        nonlocal _prev_agent_count
        n = int(sim.pool.alive.sum())
        changed = n != _prev_agent_count
        _prev_agent_count = n
        return not changed  # skip transitions when count changes
```

- [ ] **Step 5: Wire `_agent_layer_binary` into `_full_update`, `_color_and_agent_update`, `_agent_only_update`**

In `_full_update()` (~line 792), replace:
```python
layers=[water, _agent_layer(sim, landscape=landscape)],
```
with:
```python
layers=[water, _agent_layer_binary(sim, scale=_cached_scale)],
```

In `_color_and_agent_update()` (~line 810-811), replace `_agent_layer(sim, ...)` with `_agent_layer_binary(sim, scale=_cached_scale, use_transitions=_should_transition(sim))`.

In `_agent_only_update()` (~line 815), replace with:
```python
async def _agent_only_update(sim):
    await map_widget.partial_update(session, [
        _agent_layer_binary(sim, scale=_cached_scale, use_transitions=_should_transition(sim))
    ])
```

- [ ] **Step 5: Visual test**

Run: `conda run -n shiny shiny run app.py --port 8765`
- Load Columbia, click Step — agents should appear on map
- Click Run — agents should glide smoothly between positions (200ms spring transition)

- [ ] **Step 6: Commit**

**Commit message:** `feat(ui): binary-encode agents + deck.gl transitions for smooth movement`

---

## Task 2: Adaptive Sleep in Run Loop

**Priority:** High — prevents sluggish animation
**Files:**
- Modify: `app.py:549-572`

- [ ] **Step 1: Add `import time` at top of server function if not present**

- [ ] **Step 2: Replace the run loop sleep**

In `_run()` (app.py:549-572), replace:

```python
            await asyncio.to_thread(_batch)
            history.set(sim.history.copy())
            await asyncio.sleep(0.05)
```

with:

```python
            t_batch = time.perf_counter()
            await asyncio.to_thread(_batch)
            elapsed = time.perf_counter() - t_batch
            history.set(sim.history.copy())
            # Adaptive: ensure 200ms transition completes, but don't add
            # delay when simulation step already took longer
            await asyncio.sleep(max(0.05, 0.25 - elapsed))
```

- [ ] **Step 3: Visual test**

Run app, click Run with speed=1 on Columbia — should see smooth ~4 FPS animation. With speed=10, should still animate smoothly (no extra delay since batch takes >250ms).

- [ ] **Step 4: Commit**

**Commit message:** `feat(ui): adaptive sleep for smooth animation timing`

---

## Task 3: Trail Visualization

**Priority:** Medium — visual enhancement
**Files:**
- Modify: `app.py` (add TrailBuffer class + wiring)
- Modify: `ui/run_controls.py` (add toggle)

- [ ] **Step 0: Add `path_layer` import to app.py**

At `app.py:13-22` (the shiny_deckgl imports), add `path_layer` to the import list:

```python
from shiny_deckgl import ..., path_layer
```

- [ ] **Step 1: Add trail toggle to run_controls.py**

In `ui/run_controls.py`, add after the speed control div (line 19):

```python
        ui.div(
            ui.input_switch("show_trails", "Trails", value=False),
            class_="trail-toggle",
        ),
```

- [ ] **Step 2: Add TrailBuffer class to app.py**

Add after `_build_agent_binary()`:

```python
class TrailBuffer:
    """NumPy rolling buffer for agent movement trails."""
    MAX_AGENTS = 2000
    TRAIL_LEN = 10

    def __init__(self):
        self._buf = np.zeros((self.MAX_AGENTS, self.TRAIL_LEN, 2), dtype=np.float32)
        self._ptr = 0
        self._fill = 0
        self._tracked = None

    def update(self, alive_mask, positions_xy):
        """Push current positions. Subsamples to MAX_AGENTS."""
        alive_idx = np.where(alive_mask)[0]
        n = len(alive_idx)
        if n == 0:
            return
        if self._tracked is None:
            if n <= self.MAX_AGENTS:
                self._tracked = alive_idx.copy()
            else:
                step = n // self.MAX_AGENTS
                self._tracked = alive_idx[::step][:self.MAX_AGENTS]
            self._fill = 0
            self._ptr = 0

        nt = min(len(self._tracked), self.MAX_AGENTS)
        # Get positions for tracked agents still alive
        valid = alive_mask[self._tracked[:nt]]
        if valid.any() and len(positions_xy) > 0:
            xy = positions_xy[valid[:min(len(valid), len(positions_xy))]]
            m = min(len(xy), nt)
            self._buf[:m, self._ptr, :] = xy[:m]
        self._ptr = (self._ptr + 1) % self.TRAIL_LEN
        self._fill = min(self._fill + 1, self.TRAIL_LEN)

    def clear(self):
        self._buf[:] = 0
        self._fill = 0
        self._ptr = 0
        self._tracked = None

    def build_paths(self):
        """Build list of path dicts for PathLayer."""
        if self._fill < 2 or self._tracked is None:
            return []
        nt = min(len(self._tracked), self.MAX_AGENTS)
        order = [(self._ptr - self._fill + i) % self.TRAIL_LEN
                 for i in range(self._fill)]
        paths = []
        for i in range(nt):
            trail = self._buf[i, order, :].tolist()
            if trail[0] == trail[-1]:
                continue
            paths.append({"path": trail, "color": [100, 180, 160, 100]})
        return paths
```

- [ ] **Step 3: Initialize TrailBuffer in server function**

In the server function, after `sim_state = reactive.Value(None)`:

```python
    trail_buffer = TrailBuffer()
```

In `_init_sim()`, add: `trail_buffer.clear()`

- [ ] **Step 4: Update trail buffer after each step**

In `_step()`, after `await asyncio.to_thread(sim.step)`:

```python
            # Update trail buffer
            if sim.pool.alive.any():
                mesh = sim.mesh
                is_hex = hasattr(mesh, '_edge')
                tris = sim.pool.tri_idx[sim.pool.alive]
                if is_hex:
                    xy = np.column_stack([mesh.centroids[tris, 1] * _cached_scale,
                                          -mesh.centroids[tris, 0] * _cached_scale])
                else:
                    xy = np.column_stack([mesh.centroids[tris, 1], mesh.centroids[tris, 0]])
                trail_buffer.update(sim.pool.alive, xy.astype(np.float32))
```

Add the same block in `_run()` after `await asyncio.to_thread(_batch)`.

- [ ] **Step 5: Add trail layer to map updates**

In `_agent_only_update()` and `_color_and_agent_update()`, conditionally add the trail layer:

```python
    layers = [_agent_layer_binary(sim, scale=_cached_scale, use_transitions=True)]
    if input.show_trails():
        trail_data = trail_buffer.build_paths()
        if trail_data:
            layers.insert(0, path_layer("trails", trail_data,
                getPath="@@d.path", getColor="@@d.color",
                widthMinPixels=1, widthMaxPixels=3,
                jointRounded=True, capRounded=True,
            ))
    else:
        # Hide trail layer explicitly (shallow merge keeps stale layers)
        layers.insert(0, layer("PathLayer", "trails",
            data={"length": 0}, getPath=[], getColor=[0,0,0,0],
            visible=False,
        ))
    await map_widget.partial_update(session, layers)
```

- [ ] **Step 6: Visual test**

Run app, enable Trails toggle, click Run on Columbia — should see fading teal lines behind moving agents.

- [ ] **Step 7: Commit**

**Commit message:** `feat(ui): add trail visualization with PathLayer + toggle`

---

## Task 4: Live Stats Bar

**Priority:** Medium — real-time feedback
**Files:**
- Modify: `app.py` (add live_stats output)
- Modify: `ui/run_controls.py` (add output placeholder)
- Modify: `www/style.css` (add styling)

- [ ] **Step 1: Add live_stats output placeholder in run_controls.py**

After the status_text div (line 28), add:

```python
        ui.output_ui("live_stats"),
```

- [ ] **Step 2: Add live_stats render function in app.py server**

In the server function, after the `status_text` output:

```python
    @output
    @render.ui
    def live_stats():
        h = history.get()
        if not h:
            return ui.div(class_="live-stats-bar")
        last = h[-1]
        t = last.get("time", 0)
        n_alive = last.get("n_alive", 0)
        n_arrived = last.get("n_arrived", 0)
        beh = last.get("behavior_counts", {})
        return ui.div(
            ui.span(f"t = {t} h", class_="stat-val"),
            ui.span("|", class_="stat-sep"),
            ui.span(f"{n_alive:,} alive", class_="stat-val stat-alive"),
            ui.span("|", class_="stat-sep"),
            ui.span(f"{n_arrived:,} arrived", class_="stat-val"),
            ui.span("|", class_="stat-sep"),
            ui.span(
                f"\u2191{beh.get(3,0)} \u2193{beh.get(1,0)} \u2192{beh.get(0,0)} \u25cb{beh.get(4,0)}",
                class_="stat-val stat-behaviors",
            ),
            class_="live-stats-bar",
        )
```

- [ ] **Step 3: Add CSS for live-stats-bar in www/style.css**

Append:

```css
/* Live stats bar */
.live-stats-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    background: var(--card-bg);
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-primary);
    margin-top: 8px;
    min-height: 28px;
}
.stat-sep { opacity: 0.3; }
.stat-alive { color: var(--teal); font-weight: 600; }
.stat-behaviors { letter-spacing: 0.5px; }
.trail-toggle {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.78rem;
}
```

- [ ] **Step 4: Visual test**

Run app, click Step — live stats bar should show "t = 0 h | 50 alive | 0 arrived | ..."

- [ ] **Step 5: Commit**

**Commit message:** `feat(ui): add live stats bar with real-time population counts`

---

## Task 5: Chart Throttling

**Priority:** Low — reduces unnecessary chart regeneration
**Files:**
- Modify: `app.py` (chart render functions)

- [ ] **Step 1: Add throttle check to chart render functions**

The three chart functions are `survival_plot()`, `energy_plot()`, `behavior_plot()` in app.py's server function. Each calls `_plotly_iframe(fig, name)` which writes HTML + returns iframe. Add early return for throttling.

In each function, after `h = history.get()` and `if not h: return`, add:

```python
        # Throttle during Run: only regenerate every 5th step
        if running.get() and len(h) % 5 != 0:
            ts = int(time.time() * 1000)
            return ui.tags.iframe(
                src=f"{name}.html?t={ts}",
                width="100%", height="280px",
                style="border: none; border-radius: 8px; background: transparent;",
            )
```

Where `name` is `"survival"`, `"energy"`, or `"behavior"` respectively (matching the filenames already used in each function's `_plotly_iframe()` call).

- [ ] **Step 2: Visual test**

Run app with speed=5 — charts should update every ~5 steps instead of every batch. Live stats bar still updates every batch.

- [ ] **Step 3: Commit**

**Commit message:** `perf(ui): throttle Plotly chart regeneration to every 5th step`

---

## Dependency Graph

```
Task 1 (Binary encoding + transitions) ─┐
                                          ├──> Task 3 (Trails — uses binary positions)
Task 2 (Adaptive sleep)                 ─┘
Task 4 (Live stats bar)                   [independent]
Task 5 (Chart throttling)                 [independent]
```

**Recommended order:** Task 1 → Task 2 → Task 4 → Task 3 → Task 5

---

## Estimated Scope

| Task | Lines Changed | Priority |
|------|-------------|----------|
| 1. Binary encoding + transitions | ~80 | HIGH |
| 2. Adaptive sleep | ~5 | HIGH |
| 3. Trail visualization | ~100 | MEDIUM |
| 4. Live stats bar | ~40 | MEDIUM |
| 5. Chart throttling | ~15 | LOW |
| **Total** | **~240** | |
