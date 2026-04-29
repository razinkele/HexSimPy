# Light Theme ("Cool Aquatic") Implementation Plan

> **STATUS: ✅ EXECUTED** — Light theme + navbar toggle + theme-reactive map basemaps + Plotly colors all shipped. See `www/style.css` `[data-theme="light"]` selectors and `app.py` `theme_mode` reactive.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a toggleable light theme to the Salmon IBM dashboard with navbar toggle button, CSS variable overrides, and theme-aware map/chart rendering.

**Architecture:** CSS custom property overrides via `[data-theme="light"]` selector on `<html>`. A JS snippet handles toggle + localStorage persistence + Shiny input notification. Server reacts to `input.theme_mode()` to swap map basemaps and Plotly chart colors.

**Tech Stack:** CSS custom properties, vanilla JS, Shiny for Python reactive system, shiny_deckgl (`CARTO_DARK`/`CARTO_POSITRON`), Plotly

**Spec:** `docs/superpowers/specs/2026-03-14-light-theme-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `www/style.css` | Modify | Add `[data-theme="light"]` variable overrides, convert hardcoded legend/badge colors to variables |
| `app.py` | Modify | Add toggle button + JS, import `CARTO_POSITRON`, theme-aware `_base_layout()`, theme-reactive map updates, convert inline colors |

No new files needed. All changes are modifications to existing files.

---

## Chunk 1: CSS Light Theme Variables and Hardcoded Color Fixes

### Task 1: Add light theme CSS variable overrides

**Files:**
- Modify: `www/style.css:9-46` (after `:root` block)

- [ ] **Step 1: Add `[data-theme="light"]` variable override block**

Add this block immediately after the `:root { ... }` block (after line 46):

```css
/* --- Light Theme: Cool Aquatic --- */
[data-theme="light"] {
    --lagoon-deep:      #f8faf9;
    --lagoon-dark:      #f0f6f4;
    --lagoon-mid:       #e8f0ee;
    --lagoon-surface:   #dce8e4;
    --lagoon-teal:      #2a7a7a;
    --lagoon-shallow:   #2d8a7f;
    --sediment:         #8a7040;
    --sediment-light:   #5a4a32;
    --sediment-pale:    #f5ede0;
    --reed-gold:        #8a7020;
    --reed-dark:        #6a5520;
    --riparian:         #2d5a3f;
    --riparian-light:   #3d7a55;
    --salmon:           #c06a50;
    --salmon-bright:    #d4826a;
    --salmon-deep:      #b05a3f;
    --mist:             #1a3d50;
    --mist-warm:        #e8f0ee;
    --water-glass:      rgba(240, 246, 244, 0.9);
    --card-bg:          rgba(255, 255, 255, 0.95);
    --card-border:      rgba(42, 122, 122, 0.2);
    --text-primary:     #1a3d50;
    --text-secondary:   #4a7a7a;
    --text-accent:      #1a3d50;
    --text-muted:       #507070;

    --shadow-card:      0 1px 4px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04);
    --shadow-elevated:  0 4px 16px rgba(0, 0, 0, 0.1), 0 1px 4px rgba(0, 0, 0, 0.06);
}
```

- [ ] **Step 2: Add light theme overrides for body background and decorative elements**

Add after the `[data-theme="light"]` variables block:

```css
/* Light theme: background, scrollbar, selection */
[data-theme="light"] body,
[data-theme="light"] html {
    background: var(--lagoon-deep);
}

[data-theme="light"] body::before {
    background:
        radial-gradient(ellipse 120% 80% at 20% 90%, rgba(42, 122, 122, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 100% 60% at 80% 20%, rgba(61, 107, 79, 0.04) 0%, transparent 50%),
        linear-gradient(175deg, #f8faf9 0%, #f0f6f4 40%, #e8f0ee 70%, #f8faf9 100%);
}

[data-theme="light"] body::after {
    background:
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 60px,
            rgba(42, 122, 122, 0.02) 60px,
            rgba(42, 122, 122, 0.02) 61px
        );
}

[data-theme="light"] ::-webkit-scrollbar-track { background: #f0f6f4; }
[data-theme="light"] ::-webkit-scrollbar-thumb { background: #2a7a7a; }
[data-theme="light"] ::-webkit-scrollbar-thumb:hover { background: #2d8a7f; }

[data-theme="light"] ::selection {
    background: rgba(42, 122, 122, 0.2);
    color: #1a3d50;
}
```

- [ ] **Step 3: Add light navbar box-shadow override**

```css
[data-theme="light"] .navbar,
[data-theme="light"] .navbar-default,
[data-theme="light"] .navbar-static-top,
[data-theme="light"] nav.navbar,
[data-theme="light"] header.navbar {
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08) !important;
}
```

- [ ] **Step 4: Verify file saves without syntax errors**

Run: Open the app (or validate CSS manually). No test command needed for CSS — visual verification in Task 5.

- [ ] **Step 5: Commit**

```bash
git add www/style.css
git commit -m "feat: add light theme CSS variable overrides (Cool Aquatic palette)"
```

---

### Task 2: Convert hardcoded colors in CSS to variables

**Files:**
- Modify: `www/style.css:666-730` (map legend and model badge)

- [ ] **Step 1: Convert `.model-badge` background to variable**

Change line 676 from:
```css
    background: rgba(19, 47, 62, 0.6);
```
to:
```css
    background: var(--water-glass);
```

- [ ] **Step 2: Convert `.map-legend` hardcoded colors to variables**

Replace the `.map-legend` block (lines 682-694) with:
```css
.map-legend {
    position: absolute;
    bottom: 12px;
    right: 12px;
    background: var(--water-glass);
    border-radius: 6px;
    padding: 8px 12px;
    font-family: var(--font-body);
    color: var(--text-primary);
    z-index: 10;
    pointer-events: none;
    border: 1px solid var(--card-border);
}
```

- [ ] **Step 3: Convert `.map-legend-title`, `.map-legend-range`, `.map-legend-beh-item` colors**

Replace:
```css
.map-legend-title {
    font-size: 10px;
    color: #6a8a8a;
    margin-bottom: 4px;
}
```
with:
```css
.map-legend-title {
    font-size: 10px;
    color: var(--text-muted);
    margin-bottom: 4px;
}
```

Replace `color: #6a8a8a;` in `.map-legend-range` with `color: var(--text-muted);`

Replace `color: #6a8a8a;` in `.map-legend-beh-item` with `color: var(--text-muted);`

- [ ] **Step 4: Confirm viewer_legend and map_legend are covered**

Both `map_legend` and `viewer_legend` render functions in `app.py` emit HTML using `.map-legend` CSS classes. The CSS variable conversions in Steps 2-3 above cover both automatically. No Python code changes needed for these renderers.

- [ ] **Step 5: Note on `.param-hint` background**

The `.param-hint` background `rgba(42, 122, 122, 0.06)` is fine on both themes (very subtle). No change needed — the `border-left` already uses `var(--lagoon-teal)`.

- [ ] **Step 6: Commit**

```bash
git add www/style.css
git commit -m "refactor: convert hardcoded colors in map legend and badges to CSS variables"
```

---

## Chunk 2: Theme Toggle Button and JS

### Task 3: Add theme toggle button and JS to app.py

**Files:**
- Modify: `app.py:11-21` (imports)
- Modify: `app.py:314-390` (app_ui definition)

- [ ] **Step 1: Add `CARTO_POSITRON` import**

Change line 13-21 from:
```python
from shiny_deckgl import (
    CARTO_DARK,
    MapWidget,
    encode_binary_attribute,
    head_includes,
    layer,
    map_view,
    scatterplot_layer,
)
```
to:
```python
from shiny_deckgl import (
    CARTO_DARK,
    CARTO_POSITRON,
    MapWidget,
    encode_binary_attribute,
    head_includes,
    layer,
    map_view,
    scatterplot_layer,
)
```

- [ ] **Step 2: Add theme JS script and toggle button to app_ui**

Add a `THEME_JS` constant before `app_ui` (after `_list_hxn_layers` function, around line 313):

```python
# --- Theme toggle JS (flash prevention + toggle logic) ---
THEME_JS = """
(function() {
    var stored = localStorage.getItem('salmon-ibm-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', stored);
})();

function toggleTheme() {
    var html = document.documentElement;
    var current = html.getAttribute('data-theme') || 'dark';
    var next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('salmon-ibm-theme', next);
    var btn = document.getElementById('theme-toggle-btn');
    if (btn) btn.textContent = next === 'dark' ? '\\u263E' : '\\u2600';
    if (window.Shiny) Shiny.setInputValue('theme_mode', next);
}

document.addEventListener('DOMContentLoaded', function() {
    var btn = document.getElementById('theme-toggle-btn');
    var theme = document.documentElement.getAttribute('data-theme') || 'dark';
    if (btn) btn.textContent = theme === 'dark' ? '\\u263E' : '\\u2600';
});

document.addEventListener('shiny:connected', function() {
    var theme = document.documentElement.getAttribute('data-theme') || 'dark';
    Shiny.setInputValue('theme_mode', theme);
});
"""
```

- [ ] **Step 3: Add theme JS to head_content and toggle button to title**

Change `ui.head_content(...)` (lines 317-320) to:
```python
    ui.head_content(
        ui.tags.script(THEME_JS),
        ui.tags.link(rel="stylesheet", href="style.css"),
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
    ),
```

Change `title=ui.div(...)` (lines 380-389) to:
```python
    title=ui.div(
        ui.tags.span(
            "Salmon IBM",
            class_="navbar-title",
        ),
        ui.tags.span(
            " \u2014 Individual-Based Migration Model",
            class_="navbar-subtitle",
        ),
        ui.tags.button(
            "\u263E",
            id="theme-toggle-btn",
            onclick="toggleTheme()",
            class_="theme-toggle",
        ),
        style="display: flex; align-items: center; width: 100%;",
    ),
```

- [ ] **Step 4: Add CSS for navbar title classes and theme toggle button**

Add to `www/style.css` (after the navbar section, around line 122):

```css
/* --- Navbar title text --- */
.navbar-title {
    font-family: var(--font-display);
    font-weight: 700;
    color: var(--sediment-light);
    font-size: 1.35rem;
}

.navbar-subtitle {
    font-family: var(--font-body);
    font-weight: 300;
    color: var(--text-muted);
    font-size: 0.9rem;
}

/* --- Theme toggle button --- */
.theme-toggle {
    margin-left: auto;
    background: var(--lagoon-mid);
    border: 1px solid var(--card-border);
    color: var(--text-secondary);
    border-radius: 20px;
    width: 36px;
    height: 36px;
    font-size: 1.1rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all var(--transition);
    padding: 0;
    line-height: 1;
}

.theme-toggle:hover {
    background: var(--lagoon-teal);
    color: var(--text-primary);
}
```

- [ ] **Step 5: Commit**

```bash
git add app.py www/style.css
git commit -m "feat: add theme toggle button with JS localStorage persistence"
```

---

## Chunk 3: Server-Side Theme Reactions

### Task 4: Make map widgets react to theme changes

**Files:**
- Modify: `app.py:256-282` (TOOLTIP_STYLE and map widget definitions)
- Modify: `app.py:678-715` (_update_map function)
- Modify: `app.py:697-703` (landscape_changed branch)

- [ ] **Step 1: Add theme-reactive map style update**

Note: `MapWidget.update()` does not accept `parameters` or `tooltip` kwargs at runtime — only the constructor does. So we can only swap the basemap style via `set_style()`. The `clearColor` and tooltip styling cannot be changed after construction. This is acceptable: the basemap swap is the most visible change, and tooltip colors are subtle. To handle clearColor, we use a CSS background on the map container as a fallback.

Add a new reactive effect in the server function (after `_pause`, around line 498):

```python
    _cached_theme = ""

    @reactive.effect
    @reactive.event(input.theme_mode)
    async def _update_theme():
        nonlocal _cached_theme
        theme = input.theme_mode()
        if not theme or theme == _cached_theme:
            return
        _cached_theme = theme

        is_light = theme == "light"
        style = CARTO_POSITRON if is_light else CARTO_DARK

        await map_widget.set_style(session, style)
        await viewer_map_widget.set_style(session, style)

        # Force full map rebuild if simulation is loaded
        sim = sim_state.get()
        if sim is not None:
            nonlocal _cached_subsample_idx, _cached_landscape
            _cached_subsample_idx = None
            _cached_landscape = ""  # force full update on next render
```

- [ ] **Step 2: Make landscape-change branch theme-aware**

In `_update_map` (line 703), change:
```python
            await map_widget.set_style(session, CARTO_DARK)
```
to:
```python
            style = CARTO_POSITRON if _cached_theme == "light" else CARTO_DARK
            await map_widget.set_style(session, style)
```

- [ ] **Step 3: Verify the app starts without errors**

Run: `conda run -n shiny python -c "from app import app; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: react to theme toggle — swap map basemap on both widgets"
```

---

### Task 5: Make Plotly charts theme-aware

**Files:**
- Modify: `app.py:32-91` (Plotly constants and `_base_layout`)
- Modify: `app.py:740-821` (chart render functions)

- [ ] **Step 1: Add light Plotly constants and modify `_base_layout`**

Add light constants after the dark ones (after line 40):

```python
# --- Plotly light theme constants ---
PLOT_BG_LIGHT = "rgba(248, 250, 249, 0.0)"
PAPER_BG_LIGHT = "rgba(255, 255, 255, 0.0)"
GRID_COLOR_LIGHT = "rgba(42, 122, 122, 0.1)"
AXIS_COLOR_LIGHT = "#4a7a7a"
TEXT_COLOR_LIGHT = "#1a3d50"
ACCENT_COLOR_LIGHT = "#8a7040"
```

Change `_base_layout` signature and body (lines 67-91):
```python
def _base_layout(theme="dark", **overrides):
    """Shared Plotly layout for all charts."""
    is_light = theme == "light"
    _plot_bg = PLOT_BG_LIGHT if is_light else PLOT_BG
    _paper_bg = PAPER_BG_LIGHT if is_light else PAPER_BG
    _grid = GRID_COLOR_LIGHT if is_light else GRID_COLOR
    _axis = AXIS_COLOR_LIGHT if is_light else AXIS_COLOR
    _text = TEXT_COLOR_LIGHT if is_light else TEXT_COLOR
    layout = dict(
        plot_bgcolor=_plot_bg,
        paper_bgcolor=_paper_bg,
        font=dict(family="Work Sans, sans-serif", color=_text, size=12),
        margin=dict(l=48, r=16, t=36, b=44),
        xaxis=dict(
            gridcolor=_grid, zerolinecolor=_grid,
            tickfont=dict(size=10, color=_axis),
            title_font=dict(size=11, color=_axis),
        ),
        yaxis=dict(
            gridcolor=_grid, zerolinecolor=_grid,
            tickfont=dict(size=10, color=_axis),
            title_font=dict(size=11, color=_axis),
        ),
        legend=dict(
            font=dict(size=11, color=_text),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    layout.update(overrides)
    return layout
```

- [ ] **Step 2: Update chart render functions to pass theme**

Each chart render function needs to read `input.theme_mode()` and pass it. Update `survival_plot` (lines 740-762):

```python
    @render.ui
    def survival_plot():
        h = history.get()
        theme = input.theme_mode() or "dark"
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(theme=theme, height=280))
            return _plotly_iframe(fig, "survival")
        times = [r["time"] for r in h]
        alive = [r["n_alive"] for r in h]
        fig.add_trace(go.Scatter(
            x=times, y=alive, mode="lines",
            line=dict(color=TEAL, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(61, 155, 143, 0.15)",
            name="Alive",
        ))
        fig.update_layout(**_base_layout(
            theme=theme,
            height=280,
            xaxis_title="Hour",
            yaxis_title="N alive",
            showlegend=False,
        ))
        return _plotly_iframe(fig, "survival")
```

Apply same pattern to `energy_plot` and `behavior_plot`: add `theme = input.theme_mode() or "dark"` as first line, pass `theme=theme` to all `_base_layout()` calls.

For `energy_plot`, the `ACCENT_COLOR` in the line trace should also be theme-aware:
```python
        accent = ACCENT_COLOR_LIGHT if theme == "light" else ACCENT_COLOR
        fig.add_trace(go.Scatter(
            x=times, y=ed, mode="lines",
            line=dict(color=accent, width=2.5),
            ...
```

- [ ] **Step 3: Verify the app starts without errors**

Run: `conda run -n shiny python -c "from app import app; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: make Plotly charts theme-aware (dark/light color constants)"
```

---

## Chunk 4: Convert Remaining Inline Colors

### Task 6: Convert viewer_metadata and navbar inline colors

**Files:**
- Modify: `app.py:989-1007` (viewer_metadata render)

- [ ] **Step 1: Convert viewer_metadata hardcoded colors to CSS classes**

Replace the `viewer_metadata` HTML template (lines 989-1007) with:
```python
        return ui.HTML(f'''
            <div class="viewer-meta">
                <strong class="viewer-meta-title">{meta["layer_name"]}</strong><br>
                <span class="viewer-meta-label">Format:</span> {meta["format"]}<br>
                <span class="viewer-meta-label">Version:</span> {meta["version"]}<br>
                <span class="viewer-meta-label">Grid:</span> {meta["ncols"]} &times; {meta["nrows"]}<br>
                <span class="viewer-meta-label">Total hexagons:</span> {meta["total_cells"]:,}<br>
                <span class="viewer-meta-label">Water cells:</span> {meta["water_cells"]:,}
                    ({100*meta["water_cells"]/meta["total_cells"]:.1f}%)<br>
                <span class="viewer-meta-label">Hex edge:</span> {meta["edge"]:.3f} m<br>
                <span class="viewer-meta-label">Hex area:</span> {hex_area:.1f} m&sup2;<br>
                <span class="viewer-meta-label">Cell size:</span> {cell_size_str}<br>
                <span class="viewer-meta-label">Origin:</span> ({origin[0]:.1f}, {origin[1]:.1f})<br>
                <hr class="viewer-meta-hr">
                <span class="viewer-meta-label">Values:</span><br>
                &nbsp; min={meta["vmin"]:.2f} &nbsp; max={meta["vmax"]:.2f}<br>
                &nbsp; mean={meta["vmean"]:.2f} &nbsp; unique={meta["n_unique"]}<br>
            </div>
        ''')
```

- [ ] **Step 2: Add viewer-meta CSS classes to style.css**

Add to `www/style.css` (after the map legend section):

```css
/* --- Viewer metadata panel --- */
.viewer-meta {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.7;
}

.viewer-meta-title {
    color: var(--sediment-light);
}

.viewer-meta-label {
    color: var(--text-muted);
}

.viewer-meta-hr {
    border-color: var(--card-border);
}
```

- [ ] **Step 3: Commit**

```bash
git add app.py www/style.css
git commit -m "refactor: convert viewer metadata inline colors to CSS classes"
```

---

### Task 7: Manual visual verification

- [ ] **Step 1: Start the app**

Run: `conda run -n shiny python -m shiny run app.py --port 8001`

- [ ] **Step 2: Verify dark theme (default)**

Open `http://localhost:8001`. Confirm:
- Dark background, teal/sediment accents
- Moon icon visible in navbar top-right
- Map uses dark basemap
- All text readable

- [ ] **Step 3: Click theme toggle — verify light theme**

Click the moon icon. Confirm:
- Background switches to light `#f8faf9`
- Sidebar switches to `#f0f6f4`
- Cards become white
- Text becomes dark `#1a3d50`
- Map basemap switches to CARTO Positron (light)
- Toggle icon changes to sun
- Charts re-render with light colors on next step/update
- Map legend text is readable
- Viewer metadata (if loaded) uses correct colors

- [ ] **Step 4: Refresh page — verify persistence**

Refresh the browser. Confirm light theme persists (no dark flash).

- [ ] **Step 5: Toggle back to dark — verify round-trip**

Click sun icon. Confirm everything returns to the original dark theme.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete light theme (Cool Aquatic) with toggle, map, and chart support"
```
