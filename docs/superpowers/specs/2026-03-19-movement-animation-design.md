# Movement Animation & Live Dashboard — Design Spec

## Goal

Add smooth agent movement animation, trail visualization, and a real-time stats dashboard to the HexSim Shiny app.

## Architecture

Three independent features sharing the same update loop. The simulation step runs in a thread; on completion, binary-encoded agent positions (with deck.gl transitions), trail paths, and live stats are pushed to the browser. Plotly charts are throttled to every 5th step to reduce overhead.

## Feature 1: Smooth Agent Movement

### Problem

Agents teleport between cells every ~100ms. No interpolation between positions.

### Solution

Add deck.gl `transitions` to the agent ScatterplotLayer and switch agent data from per-agent Python dicts to binary-encoded NumPy arrays.

### Agent Data — Binary Encoding

Replace the Python for-loop in `_build_agent_data()` (app.py:252-280) with vectorized NumPy:

```python
def _build_agent_binary(sim, mesh, scale=1.0):
    """Build binary-encoded agent position + color arrays."""
    alive = sim.pool.alive
    if not alive.any():
        return None, None, 0
    tris = sim.pool.tri_idx[alive]
    behaviors = sim.pool.behavior[alive]

    # Vectorized position computation
    if hasattr(mesh, '_edge'):  # HexSim
        positions = np.column_stack([
            mesh.centroids[tris, 1] * scale,
            -mesh.centroids[tris, 0] * scale,
        ]).astype(np.float32)
    else:  # TriMesh
        positions = np.column_stack([
            mesh.centroids[tris, 1] * scale,
            mesh.centroids[tris, 0] * scale,
        ]).astype(np.float32)

    # Vectorized color lookup
    colors = BEH_COLORS_ARRAY[behaviors].astype(np.uint8)  # (n, 4)

    pos_bin = encode_binary_attribute(positions)
    col_bin = encode_binary_attribute(colors)
    return pos_bin, col_bin, int(alive.sum())
```

Pre-compute `BEH_COLORS_ARRAY` as a (5, 4) uint8 array at module level.

### Transition Configuration

```python
scatterplot_layer(
    "agents", {"length": n_agents},
    getPosition=pos_bin,
    getFillColor=col_bin,
    getRadius=150,
    radiusMinPixels=5,
    radiusMaxPixels=12,
    stroked=True,
    getLineColor=[0, 0, 0, 140],
    lineWidthMinPixels=1,
    pickable=True,
    transitions={"getPosition": {"duration": 200, "type": "spring"}},
)
```

### Update Interval

Change `await asyncio.sleep(0.05)` to `await asyncio.sleep(0.25)` during Run, so the 200ms transition completes before the next position update. Single Step keeps no sleep (immediate update).

### Agent Count Change Handling

When agent count changes between frames (births/deaths/introductions), deck.gl transitions break. Detect this and skip transitions on that frame:

```python
if n_agents != _prev_agent_count:
    # Full agent layer rebuild (no transition)
    _prev_agent_count = n_agents
else:
    # Partial update with transitions
```

## Feature 2: Trail Visualization

### Data Structure

Server-side rolling buffer per agent:

```python
from collections import deque

TRAIL_LENGTH = 10  # positions per agent trail

class TrailBuffer:
    """Per-agent position history for trail rendering."""
    def __init__(self):
        self._trails = {}  # agent_index → deque of (x, y)

    def update(self, alive_mask, positions_xy):
        """Push current positions for all alive agents."""
        alive_idx = np.where(alive_mask)[0]
        for i, (ax, ay) in zip(alive_idx, positions_xy):
            if i not in self._trails:
                self._trails[i] = deque(maxlen=TRAIL_LENGTH)
            self._trails[i].append((float(ax), float(ay)))

    def clear(self):
        self._trails.clear()

    def build_path_data(self):
        """Build PathLayer data: list of paths with fading colors."""
        paths = []
        for agent_id, trail in self._trails.items():
            if len(trail) < 2:
                continue
            paths.append({
                "path": list(trail),
                "color": [100, 180, 160, 120],  # teal, semi-transparent
            })
        return paths
```

### PathLayer Configuration

```python
path_layer("trails", trail_data,
    getPath="@@d.path",
    getColor="@@d.color",
    widthMinPixels=1,
    widthMaxPixels=3,
    jointRounded=True,
    capRounded=True,
    billboard=False,
)
```

### Toggle Control

Add a toggle button in `ui/run_controls.py`:

```python
ui.input_switch("show_trails", "Trails", value=False)
```

Only build and send trail data when `input.show_trails()` is True.

### Clear on Reset

`TrailBuffer.clear()` called in `_init_sim()`.

### Payload Estimate

- 2000 agents × 10 trail points × 8 bytes/point = 160 KB
- With JSON overhead: ~200 KB per update (acceptable alongside 6 KB agent data)
- For very large populations (>10K agents), subsample trails to MAX_TRAIL_AGENTS=2000.

## Feature 3: Live Stats Bar

### Layout

A horizontal stats strip rendered as reactive HTML below the run controls:

```
 t=156 h  │  4,892 alive  │  327 dead  │  0.18s/step  │  ↑2031  ↓1488  →894  ⊙479
```

### Implementation

```python
@output
@render.ui
def live_stats():
    sim = sim_state.get()
    h = history.get()
    if not sim or not h:
        return ui.div()
    last = h[-1]
    t = last["time"]
    n_alive = last["n_alive"]
    n_arrived = last["n_arrived"]
    beh = last.get("behavior_counts", {})
    # Format as styled HTML
    return ui.div(
        ui.span(f"t = {t} h", class_="stat-val"),
        ui.span("│", class_="stat-sep"),
        ui.span(f"{n_alive} alive", class_="stat-val stat-alive"),
        ui.span("│", class_="stat-sep"),
        ui.span(f"{n_arrived} arrived", class_="stat-val"),
        ui.span("│", class_="stat-sep"),
        ui.span(f"↑{beh.get(3,0)} ↓{beh.get(1,0)} →{beh.get(0,0)} ⊙{beh.get(4,0)}",
                class_="stat-val stat-behaviors"),
        class_="live-stats-bar",
    )
```

### CSS (www/style.css)

```css
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
}
.stat-sep { opacity: 0.3; }
.stat-alive { color: var(--teal); font-weight: 600; }
.stat-behaviors { letter-spacing: 0.5px; }
```

### Chart Throttling

In the `_run()` loop, only update Plotly charts every 5th step:

```python
while running.get() and sim.current_t < steps:
    ...
    await asyncio.to_thread(_batch)
    history.set(sim.history.copy())  # always update (triggers live stats + map)
    # Plotly charts throttled separately
    await asyncio.sleep(0.25)
```

The Plotly chart render functions check `len(history) % 5 == 0` before regenerating HTML files.

## Update Loop — Revised Flow

```
Run pressed:
  while running:
    1. Execute speed×steps in thread (0.2-2s depending on model)
    2. history.set() → triggers:
       a. live_stats bar (instant HTML update, <1ms)
       b. _update_map() → builds:
          - Binary agent positions + colors (vectorized, <1ms)
          - Trail paths if enabled (<5ms for 2K agents)
          - Sends partial_update with transitions
    3. Plotly charts regenerated every 5th step only
    4. await asyncio.sleep(0.25) — allows 200ms transition to complete
```

## Files to Modify

| File | Changes |
|------|---------|
| `app.py` | Binary agent encoding, trail buffer, transition config, chart throttling, live stats output |
| `ui/run_controls.py` | Add trail toggle switch |
| `www/style.css` | Live stats bar styling |

## Testing

- Visual: run Columbia scenario, verify smooth agent gliding
- Visual: enable trails, verify fading path lines behind agents
- Visual: verify live stats update every frame
- Performance: verify step time not degraded by trail buffer
- Regression: all 446 existing tests still pass
