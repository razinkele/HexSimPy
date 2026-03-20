# Streaming Interactive Charts — Design Spec

**Date:** 2026-03-20
**Approach:** Replace iframe-based Plotly chart rebuilds with streaming Plotly.js via `Plotly.extendTraces()` push from Python
**Priority:** Eliminate spinners during simulation run; show charts alongside map

## Problem

The current UI rebuilds 3 Plotly charts from scratch on each simulation step:
1. `go.Figure()` → `pio.to_html()` → write to disk → serve in iframe
2. **400-700ms UI latency per cycle** — user sees spinners instead of smooth updates
3. Charts live in a separate "Charts" tab — must switch away from the map to see them
4. `history.copy()` on every step triggers 8 downstream `@render` functions unnecessarily

## Solution

Push-based streaming charts embedded directly below the map. Python sends a lightweight JSON message per step (~1KB), JS appends to existing Plotly traces. No re-render, no iframe, no disk I/O.

**Estimated latency:** ~7ms per step (down from 400-700ms).

---

## Layout

Charts live in a collapsible panel below the map in the Map tab:

```
┌─────────────────────────────────────────────┐
│  [Run] [Step] [Pause] [Reset]   Speed ●──── │
│  50/50 alive · 0 arrived        t = 24 h    │
├─────────────────────────────────────────────┤
│                                             │
│              deck.gl Map                    │
│                                             │
├──────── ▲ LIVE CHARTS ▲ ────────────────────┤  ← drag handle
│ POPULATION    │ MIGRATION     │ BEHAVIOR    │
│ ╱╲            │ ▐█▌           │ ████████    │
│╱  ╲___        │ ▐██▌          │ ▓▓▓▓▓▓▓▓   │
│               │ ▐███▌         │ ░░░░░░░░   │
└─────────────────────────────────────────────┘
```

- **Default height:** ~150px, collapsible to 0 (map takes full height)
- **Three equal columns:** Population (line), Migration (histogram), Behavior (stacked area)
- **Dark theme:** transparent Plotly backgrounds, matching app palette
- **Drag handle:** "LIVE CHARTS" bar — click to toggle, drag to resize

---

## Data Flow

### Current (broken)
```
sim.step()
  → history.set(sim.history.copy())     # 5-50ms, FULL COPY
    → @render.ui survival_plot()         # 150-300ms: go.Figure → to_html → disk → iframe
    → @render.ui energy_plot()           # 150-300ms
    → @render.ui behavior_plot()         # 150-300ms
    → @render.text status_text()         # 5ms
    → @render.text progress_text()       # 5ms
    → @render.ui live_stats()            # 5ms
    → @render.ui map_legend()            # 50ms
    → @reactive.effect _update_map()     # 50-150ms
  Total: 400-700ms, spinners on chart panels
```

### Proposed (streaming)
```
sim.step()
  → _push_chart_data(session, sim)       # 1-3ms: extract stats, JSON push
    → JS: Plotly.extendTraces()          # <2ms: append point
    → JS: Plotly.react() (histogram)     # <3ms: diff update
  → _update_map() (agent-only)           # 50-150ms: unchanged
  → status_text (from sim_state direct)  # 1ms: no history dependency
  Total: ~55-160ms, zero spinners on charts
```

### Push Message Format

Python sends per step:
```python
await session.send_custom_message("chart_update", {
    "t": sim.current_t,
    "alive": int(alive_count),
    "dead": int(dead_count),
    "arrived": int(arrived_count),
    "behaviors": {
        "upstream": int(n_upstream),
        "downstream": int(n_downstream),
        "hold": int(n_hold),
        "random": int(n_random),
        "cwr": int(n_cwr),
    },
    "migration_bins": list(bin_counts),  # ~50 ints, see extraction logic below
})
```

**Extraction logic for `_push_chart_data()`:**
```python
async def _push_chart_data(session, sim):
    """Push chart data to JS — called after each sim.step()."""
    try:
        pool = sim.pool
        alive_mask = pool.alive.astype(bool)
        n_alive = int(alive_mask.sum())
        n_total = len(pool.alive)

        # Population counts
        n_dead = n_total - n_alive
        n_arrived = int(getattr(pool, 'arrived', np.zeros(1)).sum())

        # Behavior counts — pool.behavior uses int indices:
        #   0=Hold, 1=Random/Downstream, 2=CWR, 3=Upstream, 4=Downstream
        # Map to named keys for JS
        beh = pool.behavior[alive_mask] if n_alive > 0 else np.array([])
        BEH_MAP = {3: "upstream", 4: "downstream", 0: "hold", 1: "random", 2: "cwr"}
        beh_counts = {name: 0 for name in BEH_MAP.values()}
        for idx, name in BEH_MAP.items():
            beh_counts[name] = int((beh == idx).sum())

        # Migration histogram — upstream distance for alive agents
        # Uses the "Gradient [ upstream ]" layer if available, else mesh centroid Y
        if hasattr(sim, '_upstream_distances'):
            dists = sim._upstream_distances[alive_mask]
        else:
            # Fallback: use centroid y-coordinate as proxy for upstream distance
            water_idx = sim.mesh._water_full_idx
            compact_idx = np.where(alive_mask)[0]
            dists = sim.mesh.centroids[compact_idx, 0]  # y = upstream proxy
        bin_counts, _ = np.histogram(dists, bins=sim._migration_bins)

        await session.send_custom_message("chart_update", {
            "t": sim.current_t,
            "alive": n_alive,
            "dead": n_dead,
            "arrived": n_arrived,
            "behaviors": beh_counts,
            "migration_bins": bin_counts.tolist(),
        })
    except Exception:
        pass  # Don't crash the simulation loop on chart push failure
```

**Note on energy chart:** The current app has a 3rd chart showing mean energy density (kJ/g). This spec intentionally replaces it with Migration Progress (upstream distance histogram), which better demonstrates the spatial model behavior. The energy chart remains available in the existing Charts tab during the migration period. If energy data is needed in the streaming panel later, add `"mean_ed": float(pool.energy[alive_mask].mean())` to the push message and a 4th chart column.

On reset:
```python
await session.send_custom_message("chart_reset", {
    "max_time": total_steps,
    "n_agents": initial_population,
    "river_length_km": max_upstream_distance,
    "n_bins": 50,
    "bin_edges": list(bin_edges),  # ~51 floats, computed once from upstream gradient range
})
```

**Migration bins initialization** (in `_push_chart_reset()` or `_init_sim()`):
```python
# Compute fixed bin edges from the upstream gradient layer range
if 'Gradient [ upstream ]' in sim.mesh._workspace.hexmaps:
    up_hm = sim.mesh._workspace.hexmaps['Gradient [ upstream ]']
    up_vals = up_hm.values[up_hm.values > 0]
    max_dist = float(up_vals.max()) if len(up_vals) > 0 else 1.0
else:
    max_dist = float(sim.mesh.centroids[:, 0].max())
sim._migration_bins = np.linspace(0, max_dist, 51)  # 50 bins
sim._upstream_distances = up_hm.values[sim.mesh._water_full_idx] if up_hm else sim.mesh.centroids[:, 0]
```

**Backpressure:** During fast Run loops (speed > 5), throttle chart pushes to every Nth step (e.g., `if sim.current_t % max(1, speed // 2) == 0`). This prevents WebSocket message queue buildup while still providing smooth visual updates.
```

---

## Chart Specifications

### Population Dynamics (line chart)
- **Traces:** green solid = alive, red dashed = cumulative dead, yellow dotted = arrived
- **X axis:** time (hours), auto-scrolling window (last 200 steps)
- **Y axis:** agent count (0 to initial population)
- **Update:** `Plotly.extendTraces(el, {x:[[t],[t],[t]], y:[[alive],[dead],[arrived]]}, [0,1,2])`
- **Reset:** `Plotly.purge(el)` then reinitialize with empty traces

### Migration Progress (histogram)
- **X axis:** upstream distance in km (0 = mouth, max = headwaters), ~50 fixed bins
- **Y axis:** agent count per bin
- **Color:** yellow/gold bars (`#ffd93d`)
- **Update:** `Plotly.react(el, [{x: bin_centers, y: bin_counts, type:'bar'}], layout)`
- **Reset:** empty bars, same bin structure

### Behavior State (stacked area)
- **Traces:** upstream (blue `#2d8cf0`), hold (green `#6bcb77`), random (yellow `#ffd93d`), CWR (red `#ff6b6b`), downstream (purple `#a855f7`)
- **X axis:** time (hours), same auto-scrolling window as Population
- **Y axis:** agent count (absolute, stacked)
- **Stacking:** `stackgroup: 'one'` with `fill: 'tonexty'`
- **Update:** `Plotly.extendTraces(el, {x:[[t],...], y:[[up],[hold],[rand],[cwr],[down]]}, [0,1,2,3,4])`
- **Reset:** purge and reinitialize

### Shared Properties
- `Plotly.newPlot(el, traces, layout, {displayModeBar: false, responsive: true})`
- Layout: `{paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:'#aaa', size:10}, margin:{l:35,r:10,t:5,b:25}}`
- Auto-scroll: `Plotly.extendTraces(el, update, traceIndices, maxPoints)` — the 4th argument `maxPoints` (integer) caps each trace to N points, removing oldest automatically. Example: `Plotly.extendTraces(el, {x:[[t]], y:[[v]]}, [0], 200)` keeps at most 200 points per trace.
- On panel resize/collapse: call `Plotly.Plots.resize(el)` for each chart div to refit the layout

---

## Implementation Components

### New Files

| File | Lines | Responsibility |
|------|-------|---------------|
| `www/streaming_charts.js` | ~150 | Chart init, Shiny message handlers, extendTraces/react, panel collapse/resize |
| `ui/charts_panel.py` | ~40 | Shiny UI function returning HTML container + Plotly CDN script + streaming_charts.js script tag |

### Plotly.js Loading

The main page must load Plotly.js (currently only loaded inside iframes). Add to the `charts_panel()` UI output:
```python
ui.tags.script(src="https://cdn.plot.ly/plotly-2.27.0.min.js"),
ui.tags.script(src="streaming_charts.js"),
```

### JS Message Handler Registration

In `www/streaming_charts.js`, register handlers on document ready:
```javascript
// Register Shiny custom message handlers
Shiny.addCustomMessageHandler("chart_update", function(msg) {
    updatePopulationChart(msg);
    updateMigrationChart(msg);
    updateBehaviorChart(msg);
});

Shiny.addCustomMessageHandler("chart_reset", function(msg) {
    initPopulationChart(msg);
    initMigrationChart(msg);
    initBehaviorChart(msg);
});
```

### Modified Files

| File | Changes |
|------|---------|
| `app.py` | Add `_push_chart_data()` called after each `sim.step()`. Add `_push_chart_reset()` called on init. Decouple `status_text`/`progress_text` from `history`. Fix `_should_transition()` inverted logic. Remove 3 iframe chart `@render.ui` functions. Add `charts_panel()` to Map tab layout. |
| `www/style.css` | Add `.charts-panel`, `.charts-panel-handle`, `.chart-cell` styles. Collapsible panel CSS. |

### Files Removed from Render Pipeline

| File | Status |
|------|--------|
| `www/survival.html` | No longer generated (keep for backward compat if Charts tab retained) |
| `www/energy.html` | Same |
| `www/behavior.html` | Same |

### Unchanged
- Map update pipeline (agent-only, color+agent, full)
- HexSim Viewer tab
- Science tab
- Sidebar controls
- Trail buffer

---

## Migration Strategy

1. Add streaming charts panel to Map tab (new code, no changes to existing)
2. Verify streaming works alongside existing Charts tab
3. Once validated, remove the 3 iframe `@render.ui` functions from `app.py`
4. Optionally repurpose Charts tab for detailed/exportable static views

---

## Bug Fixes Included

### `_should_transition()` logic inversion (app.py:963-968)
Current code returns `True` when agent count **hasn't** changed (disabling transitions when agents die). Fix: return `True` when count **has** changed, enabling smooth spring animation on deaths/arrivals.

### `status_text` / `progress_text` history dependency
Current code depends on `history.get()` just to trigger re-render, then reads from `sim_state`. Fix: depend on `sim_state` directly via a lightweight `reactive.Value` that updates with just `{t, alive, arrived}` — no full history copy needed.

---

## Testing Strategy

- **Manual:** Run simulation, verify charts stream smoothly without spinners
- **Performance:** Measure `_push_chart_data()` latency (target: <5ms)
- **Reset:** Verify charts clear and restart on Reset button
- **Theme:** Verify charts adapt to dark/light toggle
- **Collapse:** Verify panel collapse/expand doesn't break chart state
- **Long runs:** Verify auto-scrolling window works for 1000+ steps without memory growth
