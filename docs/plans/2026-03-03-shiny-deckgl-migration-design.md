# Replace iframe deck.gl & matplotlib map with shiny-deckgl MapWidget

**Date:** 2026-03-03
**Status:** Approved

## Problem

The current map visualization uses two separate rendering pipelines:

- **Curonian Lagoon (TriMesh):** matplotlib scatter → PNG → base64 `<img>` tag
- **Columbia River (HexMesh):** standalone HTML with raw deck.gl JS → file write → iframe with cache-busting

Both approaches have drawbacks: matplotlib produces static raster images with no interactivity, and the iframe pipeline requires file I/O on every simulation step, full HTML regeneration (~5.5 MB), and cannot communicate events back to the Shiny server.

## Solution

Replace both rendering paths with a single `shiny-deckgl` `MapWidget` that uses native Shiny messaging. The widget updates in-place via `widget.update(session, layers=[...])` — no iframes, no file I/O, no cache-busting.

**shiny-deckgl v1.3.0** is already installed in the `shiny` micromamba environment.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Both renderers | Unified pipeline, removes matplotlib dependency for maps |
| Approach | Single shared MapWidget | One widget, two update paths. Cleanest architecture |
| Curonian basemap | CARTO Dark Matter tiles | Real lat/lon coords benefit from geographic context |
| Columbia basemap | None (OrthographicView) | Grid-unit coords, no geographic projection |
| Interactivity | Hover/click tooltips | Moderate value, low complexity |
| Legend | Shiny UI overlay | Easier to update reactively, matches app theme |

## Architecture

### Single MapWidget

```python
from shiny_deckgl import MapWidget, head_includes, scatterplot_layer
from shiny_deckgl import map_view, orthographic_view, COORDINATE_SYSTEM

widget = MapWidget("map",
    view_state={"longitude": 21.07, "latitude": 55.31, "zoom": 10},
    style=CARTO_DARK_MATTER,
    tooltip={...},
    controller=True,
)
```

The widget is created once in module scope and placed in the UI. On each simulation step or field change, `widget.update()` pushes new layer data.

### Dual Update Paths

**Curonian Lagoon** (`_update_curonian`):
- `map_view(controller=True)` with CARTO Dark Matter basemap
- `LNGLAT` coordinate system (default)
- `ScatterplotLayer("water")` for triangle centroids (lat/lon)
- `ScatterplotLayer("agents")` for salmon agents
- Radius in metres (~30m per point)

**Columbia River** (`_update_columbia`):
- `orthographic_view(controller=True)` with no basemap
- `COORDINATE_SYSTEM.CARTESIAN` for grid-unit coordinates
- `ScatterplotLayer("water")` for hex centroids (subsampled 880K → 200K)
- `ScatterplotLayer("agents")` for salmon agents
- Radius in grid units (~1.0 for water, ~10.0 for agents)

### Routing Logic

```python
@reactive.effect
async def _update_map():
    sim = sim_state.get()
    if sim is None:
        return
    mesh = sim.mesh
    is_hexsim = hasattr(mesh, "n_cells")

    if is_hexsim:
        await _update_columbia(session, sim, mesh, z, cscale, cbar_title)
    else:
        await _update_curonian(session, sim, mesh, z, cscale, cbar_title)
```

### Data Format

Both paths build data as numpy arrays → `.tolist()`:
```
water_data: [[x, y, r, g, b, info_str], ...]
agent_data: [[x, y, r, g, b, info_str], ...]
```

The `info` field is used for tooltips:
- Water cells: field value (e.g., "Temp: 14.2°C", "Depth: 3.1m")
- Agents: behavior + energy (e.g., "Agent #42 | Upstream | 5.2 kJ/g")

### Tooltip

```python
tooltip = {
    "html": "<b>{5}</b>",  # index 5 = info field
    "style": {
        "backgroundColor": "rgba(19, 47, 62, 0.9)",
        "color": "#e4e8e6",
        "fontSize": "11px",
        "fontFamily": "Work Sans, sans-serif",
    }
}
```

### Legend Overlay

A Shiny `@render.ui` producing an absolutely-positioned div with:
- Title (field name)
- CSS gradient bar matching the active colorscale
- Min/max value labels
- Behavior color chips when agents are visible

### UI Layout

```python
ui.div(
    widget.ui(height="520px"),
    ui.output_ui("map_legend"),
    style="position: relative;",
)
```

`head_includes()` added to `ui.head_content()` for CDN script injection.

## What Gets Removed

- `DECK_TEMPLATE` (60-line HTML/JS string)
- `_render_deckgl()` function
- `_render_mpl()` function
- `www/deck_map.html` file generation
- `import matplotlib` and related imports (for map rendering only)
- `import base64`, `import io` (for map PNG encoding)
- iframe cache-busting pattern for the map

**Kept unchanged:**
- `_plotly_iframe()` for chart rendering (survival, energy, behavior)
- `_colorscale_rgb()` helper (reused for layer coloring)
- `_hex_to_rgb()` helpers
- All simulation, mesh, environment, and agent code

## Error Handling

1. **Widget not ready:** Guard `widget.update()` behind `sim_state.get() is not None`
2. **Landscape switch:** `_init_sim` creates new Simulation, map reactive effect calls appropriate path
3. **Large data:** Start with JSON `.tolist()` (~6-8 MB for Columbia). Fall back to `encode_binary_attribute()` if WebSocket becomes bottleneck
4. **Session reconnection:** shiny-deckgl handles this natively

## Testing

1. All 95 existing tests pass unchanged (they test simulation logic, not rendering)
2. Manual smoke test: both landscapes, field switching, simulation stepping
3. New integration test: verify layer data structure from mock simulation
