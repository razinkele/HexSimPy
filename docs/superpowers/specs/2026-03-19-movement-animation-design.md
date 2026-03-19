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

### Update Interval — Adaptive Sleep

Use adaptive sleep so transitions complete without artificially slowing fast models:

```python
t_batch_start = time.perf_counter()
await asyncio.to_thread(_batch)
elapsed = time.perf_counter() - t_batch_start
# Ensure at least 200ms total cycle for transition, but don't add delay
# when the batch itself took longer than that
await asyncio.sleep(max(0.05, 0.25 - elapsed))
```

This gives ~4 FPS when steps are fast (<50ms) and doesn't slow down when steps take >200ms. Single Step keeps no extra sleep (immediate update).

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

### Data Structure — NumPy Rolling Buffer

Use a fixed-size NumPy array (not per-agent dicts) with subsampling as
the default path. This handles both 2K-agent Columbia and 4M-agent
Gnatcatcher scenarios.

```python
TRAIL_LENGTH = 10       # positions per trail
MAX_TRAIL_AGENTS = 2000 # subsample limit

class TrailBuffer:
    """Fixed-size rolling position buffer for trail rendering."""

    def __init__(self, max_agents=MAX_TRAIL_AGENTS, trail_len=TRAIL_LENGTH):
        self.max_agents = max_agents
        self.trail_len = trail_len
        # (max_agents, trail_len, 2) float32 rolling buffer
        self._buf = np.zeros((max_agents, trail_len, 2), dtype=np.float32)
        self._write_ptr = 0          # circular write index
        self._fill = 0               # how many slots are filled (up to trail_len)
        self._tracked_idx = None     # which agent indices we're tracking
        self._initialized = False

    def update(self, alive_mask, positions_xy):
        """Push current positions. Subsamples if needed."""
        alive_idx = np.where(alive_mask)[0]
        n_alive = len(alive_idx)

        if n_alive == 0:
            return

        # Select which agents to track (subsample if too many)
        if not self._initialized or self._tracked_idx is None:
            if n_alive <= self.max_agents:
                self._tracked_idx = alive_idx
            else:
                step = n_alive // self.max_agents
                self._tracked_idx = alive_idx[::step][:self.max_agents]
            self._initialized = True
            self._fill = 0
            self._write_ptr = 0

        # Get positions for tracked agents (vectorized)
        n_tracked = min(len(self._tracked_idx), self.max_agents)
        valid = self._tracked_idx[:n_tracked]
        # Filter to only agents still alive
        still_alive = alive_mask[valid]
        xy = positions_xy[still_alive[:len(positions_xy)]] if len(positions_xy) > 0 else np.empty((0, 2))

        # Handle size mismatch gracefully
        n = min(len(xy), n_tracked)
        if n > 0:
            self._buf[:n, self._write_ptr, :] = xy[:n]
            self._write_ptr = (self._write_ptr + 1) % self.trail_len
            self._fill = min(self._fill + 1, self.trail_len)

    def clear(self):
        self._buf[:] = 0
        self._fill = 0
        self._write_ptr = 0
        self._tracked_idx = None
        self._initialized = False

    def build_path_data(self):
        """Build PathLayer data as list of path dicts."""
        if self._fill < 2:
            return []
        # Reorder buffer into chronological order
        n = min(len(self._tracked_idx) if self._tracked_idx is not None else 0,
                self.max_agents)
        if n == 0:
            return []
        filled = self._fill
        indices = [(self._write_ptr - filled + i) % self.trail_len
                   for i in range(filled)]
        paths = []
        for i in range(n):
            trail = self._buf[i, indices, :].tolist()
            # Skip if all positions are the same (agent didn't move)
            if trail[0] == trail[-1]:
                continue
            paths.append({
                "path": trail,
                "color": [100, 180, 160, 120],
            })
        return paths
```

**Scalability:** For 4M agents (Gnatcatcher), only 2000 are tracked. The
`update()` method is fully vectorized — no Python per-agent loop. Memory:
`2000 × 10 × 2 × 4 = 160 KB` regardless of population size.

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

### Trail Toggle Behavior

- **TrailBuffer.update() always runs** for populations ≤10K agents. This
  ensures trails are immediately visible when toggled ON mid-run.
- **For populations >10K agents**, `update()` is gated on `show_trails` to
  avoid the subsampling overhead when trails are hidden.
- **When toggled OFF:** send the trail layer with `visible=False` via
  `partial_update` to hide it (don't just omit it — shallow merge keeps
  stale layers visible).
- **When toggled ON:** send trail layer with `visible=True` + current path data.

### Clear on Reset

`TrailBuffer.clear()` called in `_init_sim()`.

### Payload Estimate

- 2000 agents × 10 trail points × 8 bytes/point = 160 KB (fixed, regardless of population size)
- With JSON overhead: ~200 KB per update (acceptable alongside 6 KB agent data)
- MAX_TRAIL_AGENTS = 2000 is the default subsample limit for all scenarios.

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
    h = history.get()  # sole reactive dependency — triggers on every step
    if not h:
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
    await asyncio.sleep(max(0.05, 0.25 - elapsed))  # adaptive
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
    4. await asyncio.sleep(max(0.05, 0.25 - elapsed)) — adaptive sleep
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
