# shiny-deckgl Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace both map renderers (matplotlib PNG + iframe deck.gl) with a single `shiny-deckgl` `MapWidget`, adding hover tooltips.

**Architecture:** One `MapWidget("map")` in the UI. Landscape switch triggers `widget.set_style()` (basemap toggle) + `widget.update()` (layers + views). Curonian uses `map_view()` with CARTO_DARK tiles and LNGLAT coords. Columbia uses `orthographic_view()` with CARTESIAN coords and no basemap. A Shiny UI overlay provides the colorbar legend.

**Tech Stack:** shiny-deckgl 1.3.0, Shiny for Python, numpy

**Design doc:** `docs/plans/2026-03-03-shiny-deckgl-migration-design.md`

---

### Task 1: Add shiny-deckgl imports and MapWidget to UI

**Files:**
- Modify: `app.py:1-24` (imports)
- Modify: `app.py:185-247` (app_ui)

**Step 1: Update imports**

Replace the matplotlib/iframe-related imports and add shiny-deckgl:

```python
"""Baltic Salmon IBM — Shiny for Python Application (Lagoon Field Station theme)."""
import asyncio
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from shiny import App, reactive, render, ui

from shiny_deckgl import (
    CARTO_DARK,
    COORDINATE_SYSTEM,
    MapWidget,
    head_includes,
    map_view,
    orthographic_view,
    scatterplot_layer,
)

from salmon_ibm.config import load_config
from salmon_ibm.bioenergetics import BioParams
from salmon_ibm.simulation import Simulation
from ui.sidebar import sidebar_panel
from ui.run_controls import run_controls_panel
from ui.science_tab import science_panel
```

Removed: `base64`, `io`, `time`, `matplotlib`, `matplotlib.pyplot`, `LinearSegmentedColormap`.

Note: `import json` stays — still needed for `_plotly_iframe` fallback and potential data debugging. `time` is removed because the iframe cache-busting `int(time.time() * 1000)` is no longer needed for the map. However, `_plotly_iframe` also uses `time` — check if charts still need it. If yes, keep `import time`.

**Actually — keep `import time`** because `_plotly_iframe` (lines 536-555) still uses `int(time.time() * 1000)` for chart iframes.

**Step 2: Create MapWidget instance and update app_ui**

Add after the constants section (after line 122):

```python
# --- shiny-deckgl map widget (shared by both landscapes) ---
TOOLTIP_STYLE = {
    "backgroundColor": "rgba(19, 47, 62, 0.92)",
    "color": "#e4e8e6",
    "fontSize": "11px",
    "fontFamily": "Work Sans, sans-serif",
    "borderRadius": "4px",
    "padding": "6px 10px",
    "border": "1px solid rgba(42, 122, 122, 0.3)",
}

map_widget = MapWidget(
    "map",
    view_state={"longitude": 21.07, "latitude": 55.31, "zoom": 10},
    style=CARTO_DARK,
    tooltip={"html": "{info}", "style": TOOLTIP_STYLE},
    controller=True,
    parameters={"clearColor": [11/255, 31/255, 44/255, 1]},
)
```

Replace the Map nav_panel in `app_ui` (lines 193-200):

```python
ui.nav_panel(
    "Map",
    ui.div(
        ui.div(
            map_widget.ui(height="520px"),
            ui.output_ui("map_legend"),
            style="position: relative;",
        ),
        class_="chart-card",
    ),
),
```

Add `head_includes()` to the `ui.head_content(...)` block (line 188-191):

```python
ui.head_content(
    ui.tags.link(rel="stylesheet", href="style.css"),
    ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
    head_includes(),
),
```

**Step 3: Run tests to verify nothing breaks at import time**

Run: `micromamba run -n shiny python -c "from app import app_ui; print('UI OK')"`
Expected: `UI OK` (app_ui builds without error)

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add shiny-deckgl MapWidget to UI layout"
```

---

### Task 2: Implement the legend overlay

**Files:**
- Modify: `app.py` (add `map_legend` render function inside `server()`)
- Modify: `www/style.css` (add legend overlay CSS)

**Step 1: Add legend CSS to style.css**

Append to `www/style.css`:

```css
/* --- Map legend overlay --- */
.map-legend {
    position: absolute;
    bottom: 12px;
    right: 12px;
    background: rgba(11, 31, 44, 0.88);
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'Work Sans', sans-serif;
    color: #e4e8e6;
    z-index: 10;
    pointer-events: none;
    border: 1px solid rgba(42, 122, 122, 0.15);
}
.map-legend-title {
    font-size: 10px;
    color: #6a8a8a;
    margin-bottom: 4px;
}
.map-legend-bar {
    width: 120px;
    height: 10px;
    border-radius: 3px;
}
.map-legend-range {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: #6a8a8a;
    margin-top: 2px;
}
.map-legend-behaviors {
    margin-top: 6px;
    display: flex;
    flex-wrap: wrap;
    gap: 4px 8px;
}
.map-legend-beh-item {
    display: flex;
    align-items: center;
    gap: 3px;
    font-size: 9px;
    color: #6a8a8a;
}
.map-legend-beh-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
```

**Step 2: Add the `map_legend` render function inside `server()`**

Add inside `server()`, after the existing `map_display` block will be replaced (Task 3). For now, add it as a new function:

```python
@render.ui
def map_legend():
    sim = sim_state.get()
    _ = history.get()
    if sim is None:
        return ui.HTML("")

    field_name = input.map_field()
    if field_name == "depth":
        cscale = BATHY_COLORSCALE
        cbar_title = "Depth (m)"
        z = sim.mesh.depth
    elif field_name in sim.env.fields:
        z = sim.env.fields[field_name]
        cscale = TEMP_COLORSCALE
        field_labels = {
            "temperature": "Temp (\u00b0C)", "salinity": "Sal (PSU)",
            "ssh": "SSH (m)",
        }
        cbar_title = field_labels.get(field_name, field_name)
    else:
        cscale = BATHY_COLORSCALE
        cbar_title = "Depth (m)"
        z = sim.mesh.depth

    z_min = f"{float(np.nanmin(z)):.1f}"
    z_max = f"{float(np.nanmax(z)):.1f}"
    gradient = ", ".join(f"{s[1]} {int(s[0]*100)}%" for s in cscale)

    # Behavior chips (only if agents alive)
    beh_html = ""
    if sim.pool.alive.any():
        chips = []
        for i, (name, color) in enumerate(zip(BEH_NAMES, BEH_COLORS)):
            chips.append(
                f'<span class="map-legend-beh-item">'
                f'<span class="map-legend-beh-dot" style="background:{color}"></span>'
                f'{name}</span>'
            )
        beh_html = f'<div class="map-legend-behaviors">{"".join(chips)}</div>'

    return ui.HTML(
        f'<div class="map-legend">'
        f'<div class="map-legend-title">{cbar_title}</div>'
        f'<div class="map-legend-bar" style="background:linear-gradient(to right,{gradient})"></div>'
        f'<div class="map-legend-range"><span>{z_min}</span><span>{z_max}</span></div>'
        f'{beh_html}'
        f'</div>'
    )
```

**Step 3: Verify CSS loads**

Run: `micromamba run -n shiny python -c "from app import app; print('App OK')"`
Expected: `App OK`

**Step 4: Commit**

```bash
git add app.py www/style.css
git commit -m "feat: add map legend as Shiny UI overlay"
```

---

### Task 3: Replace map_display with widget.update() reactive effect

This is the core task — replace `@render.ui def map_display()` with an async reactive effect that calls `map_widget.update()`.

**Files:**
- Modify: `app.py:350-533` (replace `map_display`, `_render_mpl`, `_render_deckgl`)

**Step 1: Add shared helper functions**

Add these helper functions in the module scope (after `BEH_COLORS_RGB`, around line 116):

```python
def _subsample_indices(n, max_pts):
    """Return sorted random indices for subsampling large meshes."""
    if n > max_pts:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_pts, replace=False)
        idx.sort()
        return idx
    return np.arange(n)


def _build_water_data(mesh, z, cscale, field_label, idx=None):
    """Build water layer data: list of dicts with position, color, info."""
    rgb = _colorscale_rgb(z, cscale)
    if idx is None:
        idx = np.arange(len(z))

    xs = mesh.centroids[idx, 1]
    ys = mesh.centroids[idx, 0]
    rs, gs, bs = rgb[idx, 0], rgb[idx, 1], rgb[idx, 2]
    vals = z[idx]

    return [
        {
            "position": [round(float(xs[i]), 2), round(float(ys[i]), 2)],
            "color": [int(rs[i]), int(gs[i]), int(bs[i]), 220],
            "info": f"{field_label}: {vals[i]:.1f}",
        }
        for i in range(len(idx))
    ]


def _build_agent_data(sim, mesh):
    """Build agent layer data: list of dicts with position, color, info."""
    alive = sim.pool.alive
    if not alive.any():
        return []

    tris = sim.pool.tri_idx[alive]
    behaviors = sim.pool.behavior[alive]
    energies = sim.pool.ed_kJ_g[alive]
    agent_ids = np.where(alive)[0]

    pts = []
    for i, (tri, beh, ed, aid) in enumerate(
        zip(tris, behaviors, energies, agent_ids)
    ):
        beh_int = int(beh)
        rc, gc, bc = BEH_COLORS_RGB[beh_int]
        pts.append({
            "position": [
                round(float(mesh.centroids[tri, 1]), 2),
                round(float(mesh.centroids[tri, 0]), 2),
            ],
            "color": [rc, gc, bc, 240],
            "info": f"#{aid} {BEH_NAMES[beh_int]} | {ed:.1f} kJ/g",
        })
    return pts
```

**Step 2: Replace map_display and renderers inside server()**

Remove the old `map_display`, `_render_mpl`, and `_render_deckgl` functions (lines 350-533).

Replace with:

```python
# Track current landscape to detect switches
_current_landscape = reactive.Value("")

@reactive.effect
async def _update_map():
    sim = sim_state.get()
    _ = history.get()
    if sim is None:
        return

    mesh = sim.mesh
    field_name = input.map_field()
    is_hexsim = hasattr(mesh, "n_cells")
    landscape = input.landscape()

    # Resolve field + colorscale + label
    if field_name == "depth":
        z = mesh.depth
        cscale = BATHY_COLORSCALE
        field_label = "Depth (m)"
    elif field_name in sim.env.fields:
        z = sim.env.fields[field_name]
        cscale = TEMP_COLORSCALE
        field_labels = {
            "temperature": "Temp (\u00b0C)",
            "salinity": "Sal (PSU)",
            "ssh": "SSH (m)",
        }
        field_label = field_labels.get(field_name, field_name)
    else:
        z = mesh.depth
        cscale = BATHY_COLORSCALE
        field_label = "Depth (m)"

    # Switch basemap when landscape changes
    if landscape != _current_landscape.get():
        _current_landscape.set(landscape)
        if is_hexsim:
            # Columbia: no basemap (orthographic view, dark bg via CSS)
            await map_widget.set_style(session, CARTO_DARK)
            # Note: OrthographicView won't render the basemap tiles,
            # but having a dark style set prevents a white flash.
        else:
            await map_widget.set_style(session, CARTO_DARK)

    if is_hexsim:
        await _update_columbia(session, sim, mesh, z, cscale, field_label)
    else:
        await _update_curonian(session, sim, mesh, z, cscale, field_label)


async def _update_curonian(session, sim, mesh, z, cscale, field_label):
    """Push Curonian Lagoon layers (geographic, LNGLAT)."""
    water_idx = np.where(mesh.water_mask)[0]
    water_data = _build_water_data(mesh, z, cscale, field_label, idx=water_idx)
    agent_data = _build_agent_data(sim, mesh)

    await map_widget.update(
        session,
        layers=[
            scatterplot_layer("water", water_data,
                getPosition="@@d.position",
                getFillColor="@@d.color",
                getRadius=30,
                radiusMinPixels=2,
                radiusMaxPixels=6,
                pickable=True,
            ),
            scatterplot_layer("agents", agent_data,
                getPosition="@@d.position",
                getFillColor="@@d.color",
                getRadius=150,
                radiusMinPixels=5,
                radiusMaxPixels=12,
                stroked=True,
                getLineColor=[0, 0, 0, 140],
                lineWidthMinPixels=1,
                pickable=True,
            ),
        ],
        view_state={
            "longitude": 21.07,
            "latitude": 55.31,
            "zoom": 10,
        },
        views=[map_view(controller=True)],
    )


async def _update_columbia(session, sim, mesh, z, cscale, field_label):
    """Push Columbia River layers (orthographic, CARTESIAN)."""
    idx = _subsample_indices(mesh.n_cells, MAX_DECK_POINTS)
    water_data = _build_water_data(mesh, z, cscale, field_label, idx=idx)
    agent_data = _build_agent_data(sim, mesh)

    cx = float(np.mean(mesh.centroids[:, 1]))
    cy = float(np.mean(mesh.centroids[:, 0]))
    x_range = float(np.ptp(mesh.centroids[:, 1]))
    y_range = float(np.ptp(mesh.centroids[:, 0]))
    extent = max(x_range, y_range, 1.0)
    zoom = -np.log2(extent / 512)

    await map_widget.update(
        session,
        layers=[
            scatterplot_layer("water", water_data,
                getPosition="@@d.position",
                getFillColor="@@d.color",
                getRadius=1.0,
                radiusMinPixels=3,
                radiusMaxPixels=8,
                coordinateSystem=COORDINATE_SYSTEM.CARTESIAN,
                pickable=False,
            ),
            scatterplot_layer("agents", agent_data,
                getPosition="@@d.position",
                getFillColor="@@d.color",
                getRadius=10.0,
                radiusMinPixels=5,
                radiusMaxPixels=12,
                stroked=True,
                getLineColor=[0, 0, 0, 140],
                lineWidthMinPixels=1,
                coordinateSystem=COORDINATE_SYSTEM.CARTESIAN,
                pickable=True,
            ),
        ],
        view_state={"target": [cx, cy, 0], "zoom": round(zoom, 2)},
        views=[orthographic_view(controller=True)],
    )
```

**Step 3: Run existing tests to confirm no regressions**

Run: `micromamba run -n shiny python -m pytest tests/ -v --tb=short`
Expected: All 95 tests pass (tests don't import rendering functions)

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: replace matplotlib/iframe renderers with shiny-deckgl widget.update()"
```

---

### Task 4: Remove dead code (DECK_TEMPLATE, matplotlib map imports)

**Files:**
- Modify: `app.py` (remove DECK_TEMPLATE, _hex_to_rgb_f, dead imports)

**Step 1: Remove DECK_TEMPLATE**

Delete lines 124-182 (the entire `DECK_TEMPLATE = """..."""` string).

**Step 2: Remove `_hex_to_rgb_f` helper**

Delete lines 94-97. This function converted hex→RGB floats for matplotlib colormaps. No longer needed since we removed the matplotlib renderer.

**Step 3: Clean up imports**

Remove these imports that are no longer used:
- `import base64`
- `import io`
- `import matplotlib` / `matplotlib.use("Agg")`
- `import matplotlib.pyplot as plt`
- `from matplotlib.colors import LinearSegmentedColormap`

Keep:
- `import json` (used by `_plotly_iframe` if needed for debugging)
- `import time` (used by `_plotly_iframe` for chart cache-busting)

**Step 4: Verify app still loads**

Run: `micromamba run -n shiny python -c "from app import app; print('App OK')"`
Expected: `App OK`

**Step 5: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 6: Commit**

```bash
git add app.py
git commit -m "refactor: remove DECK_TEMPLATE, matplotlib map code, and dead imports"
```

---

### Task 5: Manual smoke test and edge cases

**Files:**
- No code changes — manual verification

**Step 1: Launch the app**

Run: `micromamba run -n shiny shiny run app.py --port 8123`

**Step 2: Test Curonian Lagoon**

1. Select "Curonian Lagoon" in landscape dropdown
2. Click "Step" — verify map shows with CARTO dark basemap, scatter points colored by temperature
3. Hover over a water cell — verify tooltip shows "Temp: XX.X°C"
4. Change "Color mesh by" to "Bathymetry" — verify colors change, legend updates
5. Click "Run" for a few steps — verify agents appear with behavior colors
6. Hover over an agent — verify tooltip shows "#42 Upstream | 5.2 kJ/g"

**Step 3: Test Columbia River**

1. Switch to "Columbia River"
2. Click "Step" — verify map shows orthographic view, hex cells visible
3. Verify legend shows correct field/range
4. Step a few times — verify agents appear

**Step 4: Test landscape switching**

1. Switch from Columbia back to Curonian — verify basemap reappears
2. Switch back to Columbia — verify orthographic view restores

**Step 5: Test charts still work**

1. Click "Charts" tab — verify survival, energy, behavior plots render in iframes
2. These should be completely unaffected by the map changes

**Step 6: Document any issues found**

If issues are found, create follow-up tasks to fix them before the final commit.

---

### Task 6: Performance check for Columbia (200K points)

**Files:**
- Possibly modify: `app.py` (if binary transport needed)

**Step 1: Time the Columbia update**

Add temporary timing in `_update_columbia`:

```python
import time as _time
t0 = _time.perf_counter()
# ... existing water_data build ...
print(f"Data build: {_time.perf_counter() - t0:.2f}s")
t1 = _time.perf_counter()
await map_widget.update(...)
print(f"Widget update: {_time.perf_counter() - t1:.2f}s")
```

**Step 2: Evaluate**

- If data build < 1s and update < 2s: acceptable, remove timing code
- If data build > 2s: switch to numpy vectorized dict building or `encode_binary_attribute()`
- If WebSocket transfer > 3s: reduce `MAX_DECK_POINTS` to 150K or use binary transport

**Step 3: If binary transport needed (optional)**

Replace list-of-dicts with binary attribute encoding:

```python
from shiny_deckgl import encode_binary_attribute

positions = np.column_stack([xs, ys]).astype(np.float32)
colors = np.column_stack([rs, gs, bs, np.full(len(rs), 220, dtype=np.uint8)])

scatterplot_layer("water",
    data={"length": len(positions)},
    getPosition=encode_binary_attribute(positions),
    getFillColor=encode_binary_attribute(colors),
    ...
)
```

This would reduce ~6-8 MB JSON to ~2 MB binary.

**Step 4: Remove timing code and commit if changes were made**

```bash
git add app.py
git commit -m "perf: optimize Columbia data transfer (if applicable)"
```

---

### Task 7: Final cleanup and commit

**Files:**
- Modify: `app.py` (final review)
- Possibly delete: `www/deck_map.html` (if it still exists as a leftover)

**Step 1: Remove www/deck_map.html if present**

```bash
rm -f www/deck_map.html
```

This file was generated dynamically by the old renderer. It's no longer needed.

**Step 2: Run full test suite**

Run: `micromamba run -n shiny python -m pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: remove leftover deck_map.html, finalize shiny-deckgl migration"
```

---

## Summary of Changes

| File | Action | Description |
|---|---|---|
| `app.py` | Major modify | Replace imports, add MapWidget, replace renderers, add legend |
| `www/style.css` | Append | Add `.map-legend*` CSS classes |
| `www/deck_map.html` | Delete | No longer generated |
| `docs/plans/` | Create | Design doc + this plan |

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| OrthographicView + CARTESIAN doesn't render | Fall back to `map_view` with fake geographic coords near 0,0 |
| 200K points too slow over WebSocket | Reduce to 150K or use `encode_binary_attribute()` |
| Tooltip accessor `@@d.info` doesn't work | Switch to index-based `{5}` or embed info in position array |
| Landscape switch causes white flash | Keep `CARTO_DARK` set even for Columbia (ortho view ignores basemap tiles) |
