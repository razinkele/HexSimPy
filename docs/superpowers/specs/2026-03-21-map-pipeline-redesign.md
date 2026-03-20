# Map Pipeline Redesign — Layered Update Architecture

**Date:** 2026-03-21
**Approach:** Split map into 3 independent layer groups with different update frequencies
**Priority:** Fix hex grid disappearing during run; smooth agent animation with trails

## Problems

1. **Hex grid disappears during run:** `_color_and_agent_update()` falls back to `_full_update()` for HexSim grids because SolidPolygonLayer can't do partial color updates. Every step with a dynamic field triggers a full 880K-hex rebuild (2-5s), causing flicker/disappear via async race conditions.

2. **BLANK_STYLE URL bug:** `BLANK_STYLE = json.dumps({...})` is a JSON string. MapLibre's `setStyle()` treats strings as URLs and tries to fetch them, causing 500 errors. The initial widget creation works but runtime `set_style()` calls fail.

3. **No agent animation:** `_should_transition()` has inverted logic AND is only conditionally applied. Agents jump between positions instead of smoothly interpolating.

## Solution

### Layered Update Architecture

Three deck.gl layer groups with independent update cycles:

| Layer | ID | Type | Update Frequency | Cost |
|-------|-----|------|-------------------|------|
| Hex grid | `"water"` | SolidPolygonLayer | Init + field change only | 2-5s (once) |
| Agents | `"agents"` | ScatterplotLayer | Every step | ~50ms |
| Trails | `"trails"` | PathLayer | Every step | ~20ms |

### New `_update_map()` Logic

```
if landscape_changed:
    _full_update()              # Rebuild everything (rare)
elif field_changed:
    _hex_color_update()         # Rebuild hex grid colors only
else:
    _agent_trail_update()       # Agents + trails only (every step)
```

**Key invariant:** `_agent_trail_update()` NEVER touches the hex grid layer. It sends only `agents` and `trails` layers via `partial_update()`. The hex grid persists in deck.gl's JS-side layer cache.

---

## BLANK_STYLE Fix

**Workaround (this plan):** Convert to a data URL that MapLibre can fetch:

```python
import base64 as _b64
_blank_json = '{"version":8,"sources":{},"layers":[{"id":"background","type":"background","paint":{"background-color":"#0b1f2c"}}]}'
BLANK_STYLE = "data:application/json;base64," + _b64.b64encode(_blank_json.encode()).decode()
```

**Proper fix (later, in shiny-deckgl):** Modify `deckgl-init.js` to detect JSON strings and `JSON.parse()` before passing to `instance.map.setStyle()`.

---

## Agent Animation + Trails

### ScatterplotLayer (agents)
- Always enable spring transitions: `transitions: {getPosition: {duration: 250, type: 'spring'}}`
- Remove `_should_transition()` function entirely
- Agent dots smoothly interpolate between positions every step
- Colors based on behavior state (BEH_COLORS)

### PathLayer (trails)
- Uses existing `TrailBuffer` (last 10 positions per agent, up to 2000 agents)
- Trail color matches agent behavior color with fading opacity
- `widthMinPixels: 1`, `widthMaxPixels: 2` — thin, subtle
- `jointRounded: true`, `capRounded: true` — smooth curves

### Per-Step Update Flow
```python
async def _agent_trail_update(sim):
    """Agents + trails only. Hex grid untouched. ~70ms total."""
    layers = [
        _agent_layer_binary(sim, scale=_cached_scale, use_transitions=True),
    ]
    if input.show_trails():
        trail_data = trail_buffer.build_paths()
        if trail_data:
            layers.insert(0, path_layer("trails", trail_data))
    else:
        layers.insert(0, {"id": "trails", "visible": False})
    await map_widget.partial_update(session, layers)
```

Estimated: ~70ms per step (50ms agents + 20ms trails) vs current 2-5s full rebuild.

---

## New `_hex_color_update()` Function

Triggered only on field dropdown change. Rebuilds hex polygon colors without calling `set_style()`:

```python
async def _hex_color_update(sim, landscape=None):
    """Rebuild hex grid colors only — no set_style, no agent layer."""
    is_hexsim = hasattr(sim.mesh, '_edge')
    if is_hexsim:
        # Rebuild hex polygon layer with new colors
        z, cscale = _resolve_field(sim)
        idx = _water_idx(sim)
        # Build full hex layer (same as _full_update but skip set_style)
        # ... vertex + color construction ...
        layers = [hex_layer]
        await map_widget.partial_update(session, layers)
    else:
        # TriMesh: can do incremental color update
        z, cscale = _resolve_field(sim)
        idx = _water_idx(sim)
        col_bin = _build_color_binary(sim.mesh, z, cscale, idx)
        layers = [{"id": "water", "getFillColor": col_bin, "data": {"length": _cached_water_n}}]
        await map_widget.partial_update(session, layers)
```

The critical difference from `_full_update()`: no `set_style()` call, no view state change, no agent layer rebuild. Just hex colors via `partial_update()`.

---

## Files Modified

| File | Lines | Changes |
|------|-------|---------|
| `app.py:377-380` | BLANK_STYLE definition | Convert to data URL |
| `app.py:961-968` | `_should_transition()` | Delete entirely |
| `app.py:970-991` | `_agent_layer_binary()` | Remove `use_transitions` param, always enable transitions |
| `app.py:1132-1138` | `_color_and_agent_update()` | Replace with `_hex_color_update()` — no `_full_update()` fallback |
| `app.py:1164-1181` | `_agent_only_update()` | Rename to `_agent_trail_update()` |
| `app.py:1183-1230` | `_update_map()` | Rewrite with 3-branch logic: landscape → field → agent-only |

## Unchanged
- `_full_update()` — still used on landscape change
- `TrailBuffer` — already works
- Streaming charts — independent pipeline
- `map_widget.partial_update()` API
- HexSim Viewer tab

## Testing
- Manual: Run simulation — hex grid stays visible, agents animate smoothly with trails
- Field switch: Temperature → Salinity rebuilds hex colors without flicker
- Landscape switch: Columbia → Curonian does full rebuild correctly
- Run full test suite: `conda run -n shiny python -m pytest tests/ -v`
