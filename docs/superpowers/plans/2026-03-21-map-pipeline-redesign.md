# Map Pipeline Redesign Implementation Plan

> **STATUS: ✅ EXECUTED** — `BLANK_STYLE` data URL, `_should_transition` deletion, `_color_and_agent_update` → `_hex_color_update` replacement, and `_agent_only_update` → `_agent_trail_update` rename all shipped.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix hex grid disappearing during run, add smooth agent animation with trails, fix BLANK_STYLE URL bug.

**Architecture:** Split map into 3 independent layer groups (hex grid, agents, trails) with different update frequencies. Hex grid static during run, agents animate with spring transitions every step, trails via PathLayer.

**Tech Stack:** Shiny for Python, deck.gl (SolidPolygonLayer, ScatterplotLayer, PathLayer), MapLibre

**Spec:** `docs/superpowers/specs/2026-03-21-map-pipeline-redesign.md`

**Test command:** `conda run -n shiny python -m pytest tests/ -v`

---

## File Structure

### Files to Modify
| File | Changes |
|------|---------|
| `app.py` | Fix BLANK_STYLE, delete `_should_transition()`, simplify `_agent_layer_binary()`, replace `_color_and_agent_update()` with `_hex_color_update()`, rename `_agent_only_update()` → `_agent_trail_update()`, rewrite `_update_map()` branching |

No new files needed — this is a restructure of existing functions.

---

## Task 1: Fix BLANK_STYLE URL bug

**Find:** `BLANK_STYLE = _json.dumps(` in `app.py`

- [ ] **Step 1: Replace BLANK_STYLE with data URL**

Find these lines:
```python
BLANK_STYLE = _json.dumps({"version": 8, "sources": {}, "layers": [
    {"id": "background", "type": "background",
     "paint": {"background-color": "#0b1f2c"}}
]})
```

Replace with:
```python
import base64 as _b64
_blank_json = '{"version":8,"sources":{},"layers":[{"id":"background","type":"background","paint":{"background-color":"#0b1f2c"}}]}'
BLANK_STYLE = "data:application/json;base64," + _b64.b64encode(_blank_json.encode()).decode()
```

- [ ] **Step 2: Run test suite**

```bash
conda run -n shiny python -m pytest tests/ -v
```

Expected: No regressions.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "fix: convert BLANK_STYLE to data URL — MapLibre was treating JSON string as URL"
```

---

## Task 2: Delete `_should_transition()` and always enable agent transitions

**Find:** `def _should_transition(sim):` and `def _agent_layer_binary(sim,` in `app.py`

- [ ] **Step 1: Delete `_should_transition()`**

Find and delete these lines (including the `_prev_agent_count` variable above it):
```python
    _prev_agent_count = 0

    def _should_transition(sim):
        nonlocal _prev_agent_count
        n = int(sim.pool.alive.sum())
        changed = n != _prev_agent_count
        _prev_agent_count = n
        return not changed
```

- [ ] **Step 2: Simplify `_agent_layer_binary()` — always enable transitions**

Find `def _agent_layer_binary(sim, scale=1.0, use_transitions=False):`

Change signature to remove `use_transitions` parameter:
```python
def _agent_layer_binary(sim, scale=1.0):
```

Inside the function, find the transitions conditional block:
```python
        if use_transitions:
            props["transitions"] = {"getPosition": {"duration": 200, "type": "spring"}}
```

Replace with (always enabled, slightly longer duration for smoother motion):
```python
        props["transitions"] = {"getPosition": {"duration": 250, "type": "spring"}}
```

- [ ] **Step 3: Update all callers of `_agent_layer_binary`**

Search for all calls to `_agent_layer_binary` in app.py. Remove any `use_transitions=...` arguments:

Before:
```python
_agent_layer_binary(sim, scale=_cached_scale, use_transitions=_should_transition(sim))
```

After:
```python
_agent_layer_binary(sim, scale=_cached_scale)
```

Do this for ALL call sites (there should be 3-4 in `_full_update`, `_color_and_agent_update`, `_agent_only_update`).

- [ ] **Step 4: Run test suite**

```bash
conda run -n shiny python -m pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: always enable spring transitions on agent layer — smooth animation every step"
```

---

## Task 3: Replace `_color_and_agent_update()` with `_hex_color_update()`

**Find:** `async def _color_and_agent_update(sim, landscape=None):` in `app.py`

- [ ] **Step 1: Replace the function**

Delete the entire `_color_and_agent_update` function and replace with:

```python
    async def _hex_color_update(sim, landscape=None):
        """Rebuild hex grid with new field colors — no set_style, no view state change.

        For HexSim: SolidPolygonLayer requires full vertex+color data,
        so this resends the complete hex polygon layer. The difference from
        _full_update(): no set_style() call, no view state change. Also sends
        current agents + trails to avoid stale state after field switch.
        """
        is_hexsim = hasattr(sim.mesh, '_edge')
        z, cscale = _resolve_field(sim)
        idx = _water_idx(sim)
        n = len(idx)

        if is_hexsim:
            # Same pattern as _full_update() — must use layer() + encode_binary_attribute()
            mesh = sim.mesh
            scale = _cached_scale
            cx = mesh.centroids[idx, 1] * scale
            cy = -mesh.centroids[idx, 0] * scale
            edge_s = mesh._edge * scale
            verts, start_idx = _build_hex_polygons(cx, cy, edge_s)
            rgb = _colorscale_rgb(z, cscale)
            colors = np.column_stack([
                rgb[idx, 0], rgb[idx, 1], rgb[idx, 2],
                np.full(n, 220, dtype=np.uint8),
            ]).astype(np.uint8)
            water = layer(
                "SolidPolygonLayer", "water",
                data={"length": n, "startIndices": start_idx},
                getPolygon=encode_binary_attribute(verts),
                getFillColor=encode_binary_attribute(colors),
                filled=True, extruded=False, pickable=False,
            )
            layers = [water]
        else:
            col_bin = _build_color_binary(sim.mesh, z, cscale, idx)
            layers = [{"id": "water", "getFillColor": col_bin,
                       "data": {"length": _cached_water_n}}]

        # Also send current agents + trails to avoid stale state
        layers.append(_agent_layer_binary(sim, scale=_cached_scale))
        if input.show_trails():
            trail_data = trail_buffer.build_paths()
            if trail_data:
                layers.insert(0, layer("PathLayer", "trails",
                    data=trail_data,
                    getPath="@@d.path", getColor="@@d.color",
                    widthMinPixels=1, widthMaxPixels=3,
                    jointRounded=True, capRounded=True,
                ))
        else:
            layers.insert(0, {"id": "trails", "visible": False})
        await map_widget.partial_update(session, layers)
```

**Important:** You need to check what helper functions exist: `_build_hex_polygons`, `_colorscale_rgb`, `solid_polygon_layer`, `_build_color_binary`. These are already used in `_full_update()` — read that function to see the exact calling convention and replicate it.

- [ ] **Step 2: Run test suite**

```bash
conda run -n shiny python -m pytest tests/ -v
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: replace _color_and_agent_update with _hex_color_update — no set_style call"
```

---

## Task 4: Rename `_agent_only_update()` → `_agent_trail_update()`

**Find:** `async def _agent_only_update(sim, landscape=None):` in `app.py`

- [ ] **Step 1: Rename the function**

Change the function name and docstring:
```python
    async def _agent_trail_update(sim, landscape=None):
        """Send only agents + trails. Hex grid untouched in JS cache."""
```

The function body stays the same — it already sends agents and trails via `partial_update()`.

- [ ] **Step 2: Update the caller reference**

Find where `_agent_only_update` is called (inside `_update_map()`) and rename to `_agent_trail_update`.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "refactor: rename _agent_only_update → _agent_trail_update"
```

---

## Task 5: Rewrite `_update_map()` with 3-branch logic

**Find:** `async def _update_map():` in `app.py`

- [ ] **Step 1: Rewrite the branching logic**

The current `_update_map()` has this structure:
```python
if landscape_changed:
    # set_style + _full_update
elif field_changed:
    _color_and_agent_update()      # <-- was falling back to _full_update for HexSim
else:
    if field_name in DYNAMIC_FIELDS:
        _color_and_agent_update()  # <-- was falling back to _full_update for HexSim
    else:
        _agent_only_update()
```

Replace the `elif field_changed` and `else` branches with:
```python
        elif field_changed:
            _cached_field = field_name
            _cached_scale = scale
            await _hex_color_update(sim, landscape=landscape)
        else:
            # Step only — agents + trails, hex grid untouched
            _cached_scale = scale
            await _agent_trail_update(sim, landscape=landscape)
```

The key change: the `else` branch NO LONGER checks `DYNAMIC_FIELDS`. It always calls `_agent_trail_update()` — hex colors only update on field dropdown change, never per-step.

- [ ] **Step 2: Verify the full `_update_map()` function reads correctly**

The complete function should now be:
```python
    @reactive.effect
    async def _update_map():
        nonlocal _cached_landscape, _cached_field
        nonlocal _cached_subsample_idx, _cached_scale

        sim = sim_state.get()
        _ = history.get()
        if sim is None:
            return

        landscape = input.landscape()
        field_name = input.map_field()
        is_hexsim = hasattr(sim.mesh, "n_cells")
        if is_hexsim:
            max_coord = max(abs(sim.mesh.centroids[:, 0]).max(),
                            abs(sim.mesh.centroids[:, 1]).max())
            scale = 80.0 / max(max_coord, 1)
        else:
            scale = 1.0

        landscape_changed = landscape != _cached_landscape
        field_changed = field_name != _cached_field

        if landscape_changed:
            _cached_landscape = landscape
            _cached_field = field_name
            _cached_subsample_idx = None
            _cached_scale = scale
            if is_hexsim:
                await map_widget.set_style(session, BLANK_STYLE)
            else:
                style = CARTO_POSITRON if _cached_theme == "light" else CARTO_DARK
                await map_widget.set_style(session, style)
            await _full_update(sim, landscape=landscape)
        elif field_changed:
            _cached_field = field_name
            _cached_scale = scale
            await _hex_color_update(sim, landscape=landscape)
        else:
            _cached_scale = scale
            await _agent_trail_update(sim, landscape=landscape)
```

- [ ] **Step 3: Run test suite**

```bash
conda run -n shiny python -m pytest tests/ -v
```

- [ ] **Step 4: Start app and manually test**

```bash
PYTHONPATH=. conda run -n shiny python -m shiny run app.py --port 8010
```

Test in browser:
1. Click Step — hex grid should stay visible, agents should move smoothly
2. Click Run — agents animate continuously, hex grid stable, no spinners
3. Switch Temperature → Depth — hex colors change, agents remain
4. Reset — full rebuild, grid reappears

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: rewrite _update_map with 3-branch logic — hex grid stable during run"
```

---

## Post-Implementation Checklist

- [ ] Run full test suite: `conda run -n shiny python -m pytest tests/ -v`
- [ ] Manual test with Columbia River workspace:
  - Hex grid stays visible throughout entire simulation run
  - Agents animate smoothly with spring transitions (no jumping)
  - Trails appear when Trails toggle is enabled
  - Field switch (Temperature → Depth) updates hex colors without flicker
  - Landscape switch (Columbia → Curonian) does full rebuild
  - Streaming charts still update simultaneously
  - No 500 errors in browser console for BLANK_STYLE
- [ ] Check browser console for MapLibre/deck.gl errors
