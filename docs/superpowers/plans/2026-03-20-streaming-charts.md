# Streaming Interactive Charts Implementation Plan

> **STATUS: ✅ EXECUTED** — Streaming `Plotly.extendTraces` shipped — `ui/charts_panel.py`, `www/streaming_charts.js`, `_push_chart_data`, `_push_chart_reset` all in place; iframe Plotly rebuilds eliminated.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace iframe-based Plotly chart rebuilds with streaming Plotly.js charts alongside the map, eliminating spinners during simulation.

**Architecture:** Python pushes lightweight JSON per step via `session.send_custom_message()`. JS receives via `Shiny.addCustomMessageHandler()` and calls `Plotly.extendTraces()` for O(1) append. Charts live in a collapsible panel below the deck.gl map.

**Tech Stack:** Shiny for Python, Plotly.js 2.27+ (CDN), custom JS message handlers

**Spec:** `docs/superpowers/specs/2026-03-20-streaming-charts-design.md`

**Test command:** `conda run -n shiny python -m pytest tests/ -v`

---

## File Structure

### Files to Create
| File | Responsibility |
|------|---------------|
| `www/streaming_charts.js` | Plotly chart init, Shiny message handlers, extendTraces/react, panel collapse/resize |
| `ui/charts_panel.py` | Shiny UI function: HTML container + Plotly CDN + script tag |

### Files to Modify
| File | Changes |
|------|---------|
| `app.py` | Add `_push_chart_data()` + `_push_chart_reset()`, wire into step/run/init. Add `charts_panel()` to Map tab. Fix `_should_transition()`. Decouple status text from history. |
| `www/style.css` | Add charts panel styles (collapsible container, drag handle, chart cells) |

---

## Task 1: Create the charts panel UI component

**Files:**
- Create: `ui/charts_panel.py`

- [ ] **Step 1: Create `ui/charts_panel.py`**

```python
"""Streaming charts panel — collapsible panel below the map with live Plotly.js charts."""
from shiny import ui


def charts_panel() -> ui.Tag:
    """Return the HTML container for three streaming charts + Plotly.js scripts."""
    return ui.div(
        # Drag handle / toggle
        ui.div(
            ui.span("\u25b2 LIVE CHARTS \u25b2"),
            class_="charts-panel-handle",
            id="charts-panel-handle",
        ),
        # Three chart containers
        ui.div(
            ui.div(
                ui.div("POPULATION", class_="chart-cell-title"),
                ui.div(id="chart-population", class_="chart-cell-plot"),
                class_="chart-cell",
            ),
            ui.div(
                ui.div("MIGRATION", class_="chart-cell-title"),
                ui.div(id="chart-migration", class_="chart-cell-plot"),
                class_="chart-cell",
            ),
            ui.div(
                ui.div("BEHAVIOR", class_="chart-cell-title"),
                ui.div(id="chart-behavior", class_="chart-cell-plot"),
                class_="chart-cell",
            ),
            class_="charts-panel-body",
            id="charts-panel-body",
        ),
        # Load Plotly.js from CDN + our streaming handler
        ui.tags.script(src="https://cdn.plot.ly/plotly-2.27.0.min.js"),
        ui.tags.script(src="streaming_charts.js"),
        class_="charts-panel",
        id="charts-panel",
    )
```

- [ ] **Step 2: Verify import works**

```bash
conda run -n shiny python -c "from ui.charts_panel import charts_panel; print(charts_panel())"
```

Expected: HTML output with chart divs and script tags.

- [ ] **Step 3: Commit**

```bash
git add ui/charts_panel.py
git commit -m "feat: add charts_panel UI component for streaming charts"
```

---

## Task 2: Create the streaming charts JavaScript

**Files:**
- Create: `www/streaming_charts.js`

- [ ] **Step 1: Create `www/streaming_charts.js`**

```javascript
/* Streaming charts — receives JSON pushes from Python, updates Plotly traces.
 * No re-renders, no iframes, no disk I/O. O(1) append per step.
 */
(function() {
  "use strict";

  var MAX_POINTS = 200;
  var initialized = false;

  // ── Chart colors ──────────────────────────────────────────────────────
  var COLORS = {
    alive:      "#6bcb77",
    dead:       "#ff6b6b",
    arrived:    "#ffd93d",
    upstream:   "#2d8cf0",
    hold:       "#6bcb77",
    random:     "#ffd93d",
    cwr:        "#ff6b6b",
    downstream: "#a855f7",
    migration:  "#ffd93d",
  };

  var CHART_LAYOUT = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(0,0,0,0)",
    font: { color: "#aaa", size: 10 },
    margin: { l: 35, r: 10, t: 5, b: 25 },
    showlegend: false,
    xaxis: { gridcolor: "rgba(255,255,255,0.05)", zerolinecolor: "rgba(255,255,255,0.1)" },
    yaxis: { gridcolor: "rgba(255,255,255,0.05)", zerolinecolor: "rgba(255,255,255,0.1)" },
  };

  // ── Initialization ────────────────────────────────────────────────────

  function initPopulation(msg) {
    var el = document.getElementById("chart-population");
    if (!el) return;
    var traces = [
      { x: [], y: [], mode: "lines", name: "Alive",
        line: { color: COLORS.alive, width: 2 }, fill: "tozeroy",
        fillcolor: "rgba(107,203,119,0.15)" },
      { x: [], y: [], mode: "lines", name: "Dead",
        line: { color: COLORS.dead, width: 1.5, dash: "dash" } },
      { x: [], y: [], mode: "lines", name: "Arrived",
        line: { color: COLORS.arrived, width: 1.5, dash: "dot" } },
    ];
    var layout = Object.assign({}, CHART_LAYOUT, {
      yaxis: Object.assign({}, CHART_LAYOUT.yaxis, {
        title: { text: "Count", font: { size: 9 } },
        range: [0, msg.n_agents * 1.05],
      }),
      xaxis: Object.assign({}, CHART_LAYOUT.xaxis, {
        title: { text: "Hour", font: { size: 9 } },
      }),
    });
    Plotly.newPlot(el, traces, layout, { displayModeBar: false, responsive: true });
  }

  function initMigration(msg) {
    var el = document.getElementById("chart-migration");
    if (!el) return;
    var nBins = msg.n_bins || 50;
    var edges = msg.bin_edges || [];
    var centers = [];
    for (var i = 0; i < edges.length - 1; i++) {
      centers.push((edges[i] + edges[i + 1]) / 2);
    }
    el._binCenters = centers;
    var traces = [{
      x: centers, y: new Array(centers.length).fill(0),
      type: "bar", marker: { color: COLORS.migration, opacity: 0.8 },
    }];
    var layout = Object.assign({}, CHART_LAYOUT, {
      xaxis: Object.assign({}, CHART_LAYOUT.xaxis, {
        title: { text: "Upstream (km)", font: { size: 9 } },
      }),
      yaxis: Object.assign({}, CHART_LAYOUT.yaxis, {
        title: { text: "Count", font: { size: 9 } },
      }),
      bargap: 0.05,
    });
    Plotly.newPlot(el, traces, layout, { displayModeBar: false, responsive: true });
  }

  function initBehavior(msg) {
    var el = document.getElementById("chart-behavior");
    if (!el) return;
    var names  = ["Upstream", "Hold", "Random", "CWR", "Downstream"];
    var colors = [COLORS.upstream, COLORS.hold, COLORS.random, COLORS.cwr, COLORS.downstream];
    var traces = names.map(function(name, i) {
      return {
        x: [], y: [], mode: "lines", name: name,
        stackgroup: "one", line: { color: colors[i], width: 0.5 },
        fillcolor: colors[i],
      };
    });
    var layout = Object.assign({}, CHART_LAYOUT, {
      yaxis: Object.assign({}, CHART_LAYOUT.yaxis, {
        title: { text: "Count", font: { size: 9 } },
      }),
      xaxis: Object.assign({}, CHART_LAYOUT.xaxis, {
        title: { text: "Hour", font: { size: 9 } },
      }),
    });
    Plotly.newPlot(el, traces, layout, { displayModeBar: false, responsive: true });
  }

  // ── Updates ───────────────────────────────────────────────────────────

  function updatePopulation(msg) {
    var el = document.getElementById("chart-population");
    if (!el || !el.data) return;
    Plotly.extendTraces(el,
      { x: [[msg.t], [msg.t], [msg.t]], y: [[msg.alive], [msg.dead], [msg.arrived]] },
      [0, 1, 2], MAX_POINTS);
  }

  function updateMigration(msg) {
    var el = document.getElementById("chart-migration");
    if (!el || !el.data || !el._binCenters) return;
    var bins = msg.migration_bins || [];
    Plotly.react(el, [{
      x: el._binCenters, y: bins,
      type: "bar", marker: { color: COLORS.migration, opacity: 0.8 },
    }], el.layout, { displayModeBar: false, responsive: true });
  }

  function updateBehavior(msg) {
    var el = document.getElementById("chart-behavior");
    if (!el || !el.data) return;
    var b = msg.behaviors;
    Plotly.extendTraces(el,
      { x: [[msg.t], [msg.t], [msg.t], [msg.t], [msg.t]],
        y: [[b.upstream], [b.hold], [b.random], [b.cwr], [b.downstream]] },
      [0, 1, 2, 3, 4], MAX_POINTS);
  }

  // ── Panel collapse/resize ─────────────────────────────────────────────

  function setupPanel() {
    var handle = document.getElementById("charts-panel-handle");
    var body = document.getElementById("charts-panel-body");
    if (!handle || !body) return;

    handle.addEventListener("click", function() {
      var collapsed = body.style.display === "none";
      body.style.display = collapsed ? "flex" : "none";
      handle.querySelector("span").textContent = collapsed
        ? "\u25b2 LIVE CHARTS \u25b2"
        : "\u25bc LIVE CHARTS \u25bc";
      // Resize Plotly charts after visibility change
      if (collapsed) {
        setTimeout(function() {
          ["chart-population", "chart-migration", "chart-behavior"].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) Plotly.Plots.resize(el);
          });
        }, 50);
      }
    });
  }

  // ── Shiny message handlers ────────────────────────────────────────────

  if (window.Shiny) {
    Shiny.addCustomMessageHandler("chart_reset", function(msg) {
      initPopulation(msg);
      initMigration(msg);
      initBehavior(msg);
      initialized = true;
    });

    Shiny.addCustomMessageHandler("chart_update", function(msg) {
      if (!initialized) return;
      updatePopulation(msg);
      updateMigration(msg);
      updateBehavior(msg);
    });
  }

  // Setup panel toggle when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupPanel);
  } else {
    setupPanel();
  }

})();
```

- [ ] **Step 2: Verify file is served by Shiny**

Start the app and check the JS loads:
```bash
curl -s http://localhost:8001/streaming_charts.js | head -5
```

Expected: First 5 lines of the JS file.

- [ ] **Step 3: Commit**

```bash
git add www/streaming_charts.js
git commit -m "feat: add streaming charts JS — Plotly extendTraces message handlers"
```

---

## Task 3: Add CSS styles for the charts panel

**Files:**
- Modify: `www/style.css`

- [ ] **Step 1: Append chart panel styles to `www/style.css`**

```css
/* ── Streaming Charts Panel ──────────────────────────────────────────── */
.charts-panel {
  width: 100%;
  background: var(--card-bg, #141e2b);
  border-top: 1px solid var(--border-color, #2a3a4a);
}

.charts-panel-handle {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 24px;
  background: var(--card-bg, #1a2633);
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--border-color, #2a3a4a);
}
.charts-panel-handle span {
  color: var(--accent, #4ecdc4);
  font-size: 11px;
  letter-spacing: 1px;
  font-weight: 600;
}
.charts-panel-handle:hover {
  background: var(--hover-bg, #1f2f3f);
}

.charts-panel-body {
  display: flex;
  gap: 2px;
  padding: 2px;
  height: 150px;
}

.chart-cell {
  flex: 1;
  background: var(--card-bg, #141e2b);
  border-radius: 4px;
  padding: 4px 6px;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.chart-cell-title {
  font-size: 10px;
  color: var(--accent, #4ecdc4);
  font-weight: 700;
  letter-spacing: 0.5px;
  margin-bottom: 2px;
  flex-shrink: 0;
}
.chart-cell-plot {
  flex: 1;
  min-height: 0;
}
```

- [ ] **Step 2: Commit**

```bash
git add www/style.css
git commit -m "feat: add CSS styles for streaming charts panel"
```

---

## Task 4: Wire charts panel into the Map tab

**Files:**
- Modify: `app.py:487-498` (Map tab nav_panel)

- [ ] **Step 1: Add import for charts_panel at the top of app.py**

After the existing UI imports (around line 40), add:
```python
from ui.charts_panel import charts_panel
```

- [ ] **Step 2: Add charts_panel() inside the Map tab**

Replace `app.py` lines 488-498 (the Map nav_panel):

```python
        ui.nav_panel(
            "Map",
            ui.div(
                ui.div(
                    map_widget.ui(height="520px"),
                    ui.output_ui("map_legend"),
                    style="position: relative;",
                ),
                charts_panel(),
                class_="chart-card",
            ),
            value="map",
        ),
```

The only change is adding `charts_panel()` after the map widget div.

- [ ] **Step 3: Verify app starts without errors**

```bash
conda run -n shiny python -m shiny run app.py --port 8002 &
sleep 5 && curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/
```

Expected: `200`

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: wire streaming charts panel into Map tab"
```

---

## Task 5: Add `_push_chart_reset()` to simulation init

**Files:**
- Modify: `app.py` (inside `_init_sim()` effect, around line 620-628)

- [ ] **Step 1: Add migration bins initialization after sim is created**

In `_init_sim()`, after `sim_state.set(sim)` and before `history.set([])`, add:

```python
        # ── Initialize streaming chart bins ──
        ws = sim.mesh._workspace
        if ws and 'Gradient [ upstream ]' in ws.hexmaps:
            up_hm = ws.hexmaps['Gradient [ upstream ]']
            up_vals = up_hm.values[up_hm.values > 0]
            max_dist = float(up_vals.max()) if len(up_vals) > 0 else 1.0
            sim._upstream_distances = up_hm.values[sim.mesh._water_full_idx].astype(np.float64)
        else:
            max_dist = float(abs(sim.mesh.centroids[:, 0]).max())
            sim._upstream_distances = sim.mesh.centroids[:, 0].copy()
        sim._migration_bins = np.linspace(0, max_dist, 51)
```

- [ ] **Step 2: Add `_push_chart_reset()` call at the end of `_init_sim()`**

After `history.set([])`, add:

```python
        # Push chart reset to JS
        try:
            await session.send_custom_message("chart_reset", {
                "max_time": int(input.n_steps()),
                "n_agents": int(cfg["grid"]["n_agents"]),
                "river_length_km": float(max_dist),
                "n_bins": 50,
                "bin_edges": sim._migration_bins.tolist(),
            })
        except Exception:
            pass  # Session not ready yet on first load
```

- [ ] **Step 3: Run app and verify chart_reset message is sent on init**

Start app, open browser, check browser console for Plotly chart initialization (empty traces).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add _push_chart_reset to init — migration bins + JS init"
```

---

## Task 6: Add `_push_chart_data()` and wire into step/run

**Files:**
- Modify: `app.py` (add function + calls in `_step()` and `_run()`)

- [ ] **Step 1: Add `_push_chart_data()` function**

Add after `_init_sim()` (around line 630):

```python
    async def _push_chart_data(sim):
        """Push chart update to JS — lightweight JSON, no reactive cascade."""
        try:
            pool = sim.pool
            alive_mask = pool.alive.astype(bool)
            n_alive = int(alive_mask.sum())
            n_total = len(pool.alive)

            # Behavior counts: 0=Hold, 1=Random, 2=CWR, 3=Upstream, 4=Downstream
            beh = pool.behavior[alive_mask] if n_alive > 0 else np.array([], dtype=int)
            beh_counts = {
                "upstream": int((beh == 3).sum()),
                "downstream": int((beh == 4).sum()),
                "hold": int((beh == 0).sum()),
                "random": int((beh == 1).sum()),
                "cwr": int((beh == 2).sum()),
            }

            # Migration histogram
            if n_alive > 0 and hasattr(sim, '_upstream_distances'):
                compact_idx = np.where(alive_mask)[0]
                dists = sim._upstream_distances[pool.tri_idx[alive_mask]]
                bin_counts, _ = np.histogram(dists, bins=sim._migration_bins)
            else:
                bin_counts = np.zeros(50, dtype=int)

            await session.send_custom_message("chart_update", {
                "t": sim.current_t,
                "alive": n_alive,
                "dead": n_total - n_alive,
                "arrived": int(getattr(pool, 'n_arrived', 0)),
                "behaviors": beh_counts,
                "migration_bins": bin_counts.tolist(),
            })
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("chart push failed: %s", exc)
```

- [ ] **Step 2: Wire into `_step()` effect**

In `_step()` (around line 677, after `history.set(sim.history.copy())`), add:

```python
            await _push_chart_data(sim)
```

- [ ] **Step 3: Wire into `_run()` loop**

In `_run()` (around line 714, after `history.set(sim.history.copy())`), add with backpressure:

```python
                speed = input.speed()
                is_final = sim.current_t >= steps or not sim.pool.alive.any()
                if is_final or speed <= 5 or sim.current_t % max(1, speed // 2) == 0:
                    await _push_chart_data(sim)
```

- [ ] **Step 4: Test manually — run sim, verify charts stream**

Start app, click Run, watch browser — charts should update smoothly without spinners.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add _push_chart_data with backpressure — streaming chart updates"
```

---

## Task 7: Fix `_should_transition()` inverted logic

**Files:**
- Modify: `app.py:963-968`

- [ ] **Step 1: Fix the inverted return**

Replace lines 963-968:

```python
    def _should_transition(sim):
        nonlocal _prev_agent_count
        n = int(sim.pool.alive.sum())
        changed = n != _prev_agent_count
        _prev_agent_count = n
        return changed  # Was: return not changed
```

- [ ] **Step 2: Run full test suite**

```bash
conda run -n shiny python -m pytest tests/ -v
```

Expected: No regressions.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "fix: correct _should_transition inverted logic — enable spring animations"
```

---

## Task 8: Decouple status text from history dependency

**Files:**
- Modify: `app.py` (status_text, progress_text, live_stats renders)

- [ ] **Step 1: Add a lightweight reactive value for step stats**

After the existing reactive values (line 569), add:

```python
    step_stats = reactive.Value({"t": 0, "alive": 0, "dead": 0, "arrived": 0, "behaviors": {}})
```

- [ ] **Step 2: Update step_stats in `_step()` and `_run()` instead of relying on history**

After `sim.step()` completes (in both `_step()` and `_run()`), before `history.set(...)`, add:

```python
                pool = sim.pool
                alive_mask = pool.alive.astype(bool)
                n_alive = int(alive_mask.sum())
                beh = pool.behavior[alive_mask] if n_alive > 0 else np.array([], dtype=int)
                step_stats.set({
                    "t": sim.current_t,
                    "alive": n_alive,
                    "dead": len(pool.alive) - n_alive,
                    "arrived": int(getattr(pool, 'n_arrived', 0)),
                    "behaviors": {i: int((beh == i).sum()) for i in range(5)},
                })
```

- [ ] **Step 3: Update `status_text` to use step_stats instead of history**

Replace the current `status_text` render to depend on `step_stats` instead of `history`:

```python
    @render.text
    def status_text():
        s = step_stats.get()
        sim = sim_state.get()
        if sim is None:
            return ""
        total = len(sim.pool.alive)
        return f"{s['alive']}/{total} alive \u00b7 {s['arrived']} arrived"
```

- [ ] **Step 4: Update `progress_text` similarly**

```python
    @render.text
    def progress_text():
        s = step_stats.get()
        return f"t = {s['t']} h"
```

- [ ] **Step 5: Update `live_stats` similarly**

Replace the `history.get()` dependency with `step_stats.get()` and read behavior counts from it.

- [ ] **Step 6: Run full test suite + manual verification**

```bash
conda run -n shiny python -m pytest tests/ -v
```

Then start the app and verify status text updates without chart re-renders.

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "perf: decouple status text from history — use lightweight step_stats"
```

---

## Post-Implementation Checklist

- [ ] Run full test suite: `conda run -n shiny python -m pytest tests/ -v`
- [ ] Manual test: Start app, click Run at speed=5, verify:
  - Charts stream smoothly (no spinners, no flicker)
  - Population line extends incrementally
  - Migration histogram bars update
  - Behavior stacked area grows
  - Map agents still move correctly
  - Panel collapses/expands without breaking charts
- [ ] Test Reset: click Reset, verify charts clear and reinitialize
- [ ] Test theme toggle: switch dark/light, verify charts remain readable
- [ ] Test long run (500+ steps): verify no memory growth, auto-scroll works
- [ ] Existing Charts tab still works (migration strategy phase 1-2)

## Future Work (Not in This Plan)

- Remove old iframe Charts tab (migration strategy phase 3)
- Add energy chart as optional 4th column
- Viewport-aware hex subsampling for map performance
- Switch SolidPolygonLayer to a type supporting partial color updates
