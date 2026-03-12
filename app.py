"""Baltic Salmon IBM — Shiny for Python Application (Lagoon Field Station theme)."""
import asyncio
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from shiny import App, reactive, render, ui

from shiny_deckgl import (
    CARTO_DARK,
    MapWidget,
    encode_binary_attribute,
    head_includes,
    layer,
    map_view,
    scatterplot_layer,
)

from salmon_ibm.config import load_config
from salmon_ibm.bioenergetics import BioParams
from salmon_ibm.simulation import Simulation
from ui.sidebar import sidebar_panel
from ui.run_controls import run_controls_panel
from ui.science_tab import science_panel

# --- Plotly theme constants (Lagoon palette) ---
PLOT_BG = "rgba(11, 31, 44, 0.0)"
PAPER_BG = "rgba(19, 47, 62, 0.0)"
GRID_COLOR = "rgba(42, 122, 122, 0.15)"
AXIS_COLOR = "#6a8a8a"
TEXT_COLOR = "#e4e8e6"
ACCENT_COLOR = "#e8d5b7"
TEAL = "#3d9b8f"
SALMON_COLOR = "#d4826a"

# Behavior colors — nature-inspired
BEH_NAMES = ["Hold", "Random", "CWR", "Upstream", "Downstream"]
BEH_COLORS = ["#7a8b7a", "#4a8fa8", "#3d9b8f", "#d4826a", "#b8963e"]

# Bathymetric colorscale
BATHY_COLORSCALE = [
    [0.0, "#0b1f2c"],
    [0.2, "#1a3d50"],
    [0.4, "#2a7a7a"],
    [0.6, "#3d9b8f"],
    [0.8, "#7ac4a5"],
    [1.0, "#c8e6c9"],
]

TEMP_COLORSCALE = [
    [0.0, "#0b1f2c"],
    [0.15, "#1a3d50"],
    [0.3, "#2a7a7a"],
    [0.5, "#4a8fa8"],
    [0.7, "#e8d5b7"],
    [0.85, "#d4826a"],
    [1.0, "#b05a3f"],
]


def _base_layout(**overrides):
    """Shared Plotly layout for all charts."""
    layout = dict(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Work Sans, sans-serif", color=TEXT_COLOR, size=12),
        margin=dict(l=48, r=16, t=36, b=44),
        xaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            tickfont=dict(size=10, color=AXIS_COLOR),
            title_font=dict(size=11, color=AXIS_COLOR),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            tickfont=dict(size=10, color=AXIS_COLOR),
            title_font=dict(size=11, color=AXIS_COLOR),
        ),
        legend=dict(
            font=dict(size=11, color=TEXT_COLOR),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    layout.update(overrides)
    return layout


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert hex color '#rrggbb' to (r, g, b) integers."""
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _colorscale_rgb(z: np.ndarray, colorscale: list) -> np.ndarray:
    """Map array values → (N, 3) uint8 RGB using a Plotly-style colorscale."""
    z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    if z_max <= z_min:
        z_norm = np.zeros(len(z), dtype=np.float32)
    else:
        z_norm = ((z - z_min) / (z_max - z_min)).astype(np.float32)
    z_norm = np.clip(z_norm, 0, 1)
    stops = np.array([s[0] for s in colorscale], dtype=np.float32)
    colors = np.array([_hex_to_rgb(s[1]) for s in colorscale], dtype=np.float32)
    r = np.interp(z_norm, stops, colors[:, 0]).astype(np.uint8)
    g = np.interp(z_norm, stops, colors[:, 1]).astype(np.uint8)
    b = np.interp(z_norm, stops, colors[:, 2]).astype(np.uint8)
    return np.column_stack([r, g, b])


BEH_COLORS_RGB = [_hex_to_rgb(c) for c in BEH_COLORS]


def _subsample_indices(n, max_pts):
    """Return sorted random indices for subsampling large meshes."""
    if n > max_pts:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_pts, replace=False)
        idx.sort()
        return idx
    return np.arange(n)


def _build_water_binary(mesh, z, cscale, idx, scale=1.0):
    """Build binary-encoded position and color arrays for the water layer.

    Returns (positions_binary, colors_binary, n_points).
    """
    rgb = _colorscale_rgb(z, cscale)
    positions = np.column_stack([
        mesh.centroids[idx, 1] * scale,
        mesh.centroids[idx, 0] * scale,
    ]).astype(np.float32)
    colors = np.column_stack([
        rgb[idx, 0], rgb[idx, 1], rgb[idx, 2],
        np.full(len(idx), 220, dtype=np.uint8),
    ]).astype(np.uint8)
    return (
        encode_binary_attribute(positions),
        encode_binary_attribute(colors),
        len(idx),
    )


def _build_color_binary(mesh, z, cscale, idx):
    """Build only the color binary for a partial update."""
    rgb = _colorscale_rgb(z, cscale)
    colors = np.column_stack([
        rgb[idx, 0], rgb[idx, 1], rgb[idx, 2],
        np.full(len(idx), 220, dtype=np.uint8),
    ]).astype(np.uint8)
    return encode_binary_attribute(colors)


def _build_agent_data(sim, mesh, scale=1.0):
    """Build agent layer data: list of dicts with position, color, info."""
    alive = sim.pool.alive
    if not alive.any():
        return []

    tris = sim.pool.tri_idx[alive]
    behaviors = sim.pool.behavior[alive]
    energies = sim.pool.ed_kJ_g[alive]
    agent_ids = np.where(alive)[0]

    pts = []
    for tri, beh, ed, aid in zip(tris, behaviors, energies, agent_ids):
        beh_int = int(beh)
        rc, gc, bc = BEH_COLORS_RGB[beh_int]
        pts.append({
            "position": [
                round(float(mesh.centroids[tri, 1] * scale), 5),
                round(float(mesh.centroids[tri, 0] * scale), 5),
            ],
            "color": [rc, gc, bc, 240],
            "info": f"#{aid} {BEH_NAMES[beh_int]} | {ed:.1f} kJ/g",
        })
    return pts


# Static assets directory
WWW_DIR = Path(__file__).parent / "www"

# Max hex background points for deck.gl (200K → ~5.5 MB HTML, less banding)
MAX_DECK_POINTS = 200_000

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

# --- App UI ---
app_ui = ui.page_sidebar(
    sidebar_panel(),
    ui.head_content(
        ui.tags.link(rel="stylesheet", href="style.css"),
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
    ),
    head_includes(),
    run_controls_panel(),
    ui.navset_tab(
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
        ui.nav_panel(
            "Charts",
            ui.row(
                ui.column(
                    6,
                    ui.div(
                        ui.div("Population Survival", class_="chart-card-title"),
                        ui.output_ui("survival_plot"),
                        class_="chart-card",
                    ),
                ),
                ui.column(
                    6,
                    ui.div(
                        ui.div("Energy Reserve", class_="chart-card-title"),
                        ui.output_ui("energy_plot"),
                        class_="chart-card",
                    ),
                ),
            ),
            ui.row(
                ui.column(
                    12,
                    ui.div(
                        ui.div("Behavioral State Distribution", class_="chart-card-title"),
                        ui.output_ui("behavior_plot"),
                        class_="chart-card",
                    ),
                ),
            ),
        ),
        ui.nav_panel(
            "Science",
            science_panel(),
        ),
    ),
    title=ui.div(
        ui.tags.span(
            "Salmon IBM",
            style="font-family: 'Cormorant Garamond', serif; font-weight: 700; color: #e8d5b7; font-size: 1.35rem;",
        ),
        ui.tags.span(
            " \u2014 Individual-Based Migration Model",
            style="font-family: 'Work Sans', sans-serif; font-weight: 300; color: #6a8a8a; font-size: 0.9rem;",
        ),
    ),
)


# --- Server ---
def server(input, output, session):
    sim_state = reactive.Value(None)
    running = reactive.Value(False)
    history = reactive.Value([])

    @reactive.effect
    @reactive.event(input.btn_reset, input.landscape, ignore_none=False)
    async def _init_sim():
        landscape = input.landscape()

        if landscape == "columbia":
            cfg = load_config("config_columbia.yaml")
        else:
            cfg = load_config("config_curonian_minimal.yaml")

        # Apply estuary overrides from UI controls
        est = cfg.setdefault("estuary", {})
        est.setdefault("salinity_cost", {}).update({
            "S_opt": input.s_opt(), "S_tol": input.s_tol(), "k": input.sal_k(),
        })
        est.setdefault("do_avoidance", {}).update({
            "lethal": input.do_lethal(), "high": input.do_high(),
        })
        est.setdefault("seiche_pause", {}).update({
            "dSSHdt_thresh_m_per_15min": input.seiche_thresh(),
        })

        # Capture reactive values before entering thread
        n_agents = input.n_agents()
        rng_seed = input.rng_seed()
        ra, rb, rq = input.ra(), input.rb(), input.rq()
        ed_mortal = input.ed_mortal()
        t_opt, t_max = input.t_opt(), input.t_max()
        ed_init = input.ed_init()

        # Run blocking init in a thread to avoid freezing the event loop
        sim = await asyncio.to_thread(
            Simulation, cfg, n_agents=n_agents, data_dir="data", rng_seed=rng_seed,
        )
        sim.bio_params = BioParams(
            RA=ra, RB=rb, RQ=rq, ED_MORTAL=ed_mortal,
            T_OPT=t_opt, T_MAX=t_max,
        )
        sim._activity_lut = sim._build_activity_lut()
        sim.pool.ed_kJ_g[:] = ed_init
        sim_state.set(sim)
        history.set([])
        running.set(False)

    @reactive.effect
    @reactive.event(input.btn_step)
    async def _step():
        sim = sim_state.get()
        if sim is None:
            await _init_sim()
            sim = sim_state.get()
        await asyncio.to_thread(sim.step)
        history.set(sim.history.copy())

    @reactive.effect
    @reactive.event(input.btn_run)
    async def _run():
        running.set(True)
        sim = sim_state.get()
        if sim is None:
            await _init_sim()
            sim = sim_state.get()
        steps = input.n_steps()
        while running.get() and sim.current_t < steps:
            speed = input.speed()
            def _batch():
                for _ in range(speed):
                    if sim.current_t >= steps:
                        break
                    sim.step()
            await asyncio.to_thread(_batch)
            history.set(sim.history.copy())
            await asyncio.sleep(0.05)
        running.set(False)

    @reactive.effect
    @reactive.event(input.btn_pause)
    def _pause():
        running.set(False)

    @render.text
    def status_text():
        _ = history.get()  # re-render on each step
        sim = sim_state.get()
        if sim is None:
            return "Awaiting initialization"
        alive = int(sim.pool.alive.sum())
        total = sim.pool.n
        arrived = int(sim.pool.arrived.sum())
        return f"{alive}/{total} alive \u00b7 {arrived} arrived"

    @render.text
    def progress_text():
        _ = history.get()  # re-render on each step
        sim = sim_state.get()
        if sim is None:
            return "t = 0 h"
        return f"t = {sim.current_t} h"

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
                "temperature": "Temp (\u00b0C)",
                "salinity": "Sal (PSU)",
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

    # --- Map update: three-tier strategy ---
    # Cache state for change detection
    _cached_landscape = ""
    _cached_field = ""
    _cached_subsample_idx = None  # Columbia subsample / Curonian water indices (computed once)
    _cached_scale = 1.0
    _cached_water_n = 0

    # Dynamic fields whose colors change on env.advance()
    DYNAMIC_FIELDS = {"temperature", "salinity", "ssh"}

    def _resolve_field(sim):
        """Resolve current field → (z, colorscale)."""
        field_name = input.map_field()
        if field_name == "depth":
            return sim.mesh.depth, BATHY_COLORSCALE
        elif field_name in sim.env.fields:
            return sim.env.fields[field_name], TEMP_COLORSCALE
        return sim.mesh.depth, BATHY_COLORSCALE

    def _water_idx(sim):
        """Return subsample indices for the current mesh."""
        nonlocal _cached_subsample_idx
        mesh = sim.mesh
        is_hexsim = hasattr(mesh, "n_cells")
        if is_hexsim:
            if _cached_subsample_idx is None:
                _cached_subsample_idx = _subsample_indices(
                    mesh.n_cells, MAX_DECK_POINTS
                )
            return _cached_subsample_idx
        if _cached_subsample_idx is None:
            _cached_subsample_idx = np.where(mesh.water_mask)[0]
        return _cached_subsample_idx

    def _view_state(sim):
        """Return the appropriate view_state for the current landscape."""
        mesh = sim.mesh
        is_hexsim = hasattr(mesh, "n_cells")
        if is_hexsim:
            cx = float(np.mean(mesh.centroids[:, 1])) * _cached_scale
            cy = float(np.mean(mesh.centroids[:, 0])) * _cached_scale
            return {"longitude": cx, "latitude": cy, "zoom": 7}
        return {"longitude": 21.07, "latitude": 55.31, "zoom": 10}

    def _agent_layer(sim):
        """Build the complete agent ScatterplotLayer dict."""
        agent_data = _build_agent_data(sim, sim.mesh, scale=_cached_scale)
        return scatterplot_layer(
            "agents", agent_data,
            getPosition="@@d.position",
            getFillColor="@@d.color",
            getRadius=150,
            radiusMinPixels=5,
            radiusMaxPixels=12,
            stroked=True,
            getLineColor=[0, 0, 0, 140],
            lineWidthMinPixels=1,
            pickable=True,
        )

    async def _full_update(sim):
        """Send everything: positions + colors + agents (~3 MB)."""
        nonlocal _cached_water_n
        z, cscale = _resolve_field(sim)
        idx = _water_idx(sim)
        pos_bin, col_bin, n = _build_water_binary(
            sim.mesh, z, cscale, idx, scale=_cached_scale
        )
        _cached_water_n = n

        is_hexsim = hasattr(sim.mesh, "n_cells")
        water = layer(
            "ScatterplotLayer", "water",
            data={"length": n},
            getPosition=pos_bin,
            getFillColor=col_bin,
            getRadius=30,
            radiusMinPixels=3 if is_hexsim else 2,
            radiusMaxPixels=8 if is_hexsim else 6,
            pickable=False,
        )
        await map_widget.update(
            session,
            layers=[water, _agent_layer(sim)],
            view_state=_view_state(sim),
            views=[map_view(controller=True)],
        )

    async def _color_and_agent_update(sim):
        """Send new colors + agents, positions stay in JS cache (~805 KB)."""
        z, cscale = _resolve_field(sim)
        idx = _water_idx(sim)
        col_bin = _build_color_binary(sim.mesh, z, cscale, idx)
        await map_widget.partial_update(session, [
            {"id": "water", "getFillColor": col_bin,
             "data": {"length": _cached_water_n}},
            _agent_layer(sim),
        ])

    async def _agent_only_update(sim):
        """Send only agents, water layer untouched in JS cache (~5 KB)."""
        await map_widget.partial_update(session, [_agent_layer(sim)])

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
        scale = 0.0005 if is_hexsim else 1.0

        # Detect what changed
        landscape_changed = landscape != _cached_landscape
        field_changed = field_name != _cached_field

        if landscape_changed:
            # Reset cache for new landscape
            _cached_landscape = landscape
            _cached_field = field_name
            _cached_subsample_idx = None
            _cached_scale = scale
            await map_widget.set_style(session, CARTO_DARK)
            await _full_update(sim)
        elif field_changed:
            _cached_field = field_name
            _cached_scale = scale
            await _color_and_agent_update(sim)
        else:
            # Step only — choose path based on field dynamism
            _cached_scale = scale
            if field_name in DYNAMIC_FIELDS:
                await _color_and_agent_update(sim)
            else:
                await _agent_only_update(sim)

    # --- Chart helper: Plotly → HTML file + iframe (no widget/comm layer) ---
    def _plotly_iframe(fig, name, height="280px"):
        """Write a Plotly Figure to www/<name>.html and return an iframe.

        Script tags in innerHTML don't execute (browser security), so we
        can't use to_html() inline. Instead, write a standalone HTML file
        and serve it via iframe — same pattern as the deck.gl map.
        """
        html_str = pio.to_html(
            fig, include_plotlyjs="cdn", full_html=True,
            config={"displayModeBar": False, "responsive": True},
        )
        chart_path = WWW_DIR / f"{name}.html"
        chart_path.write_text(html_str, encoding="utf-8")
        ts = int(time.time() * 1000)
        return ui.tags.iframe(
            src=f"{name}.html?t={ts}",
            width="100%",
            height=height,
            style="border: none; border-radius: 8px; background: transparent;",
        )

    # --- Survival Plot ---
    @render.ui
    def survival_plot():
        h = history.get()
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(height=280))
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
            height=280,
            xaxis_title="Hour",
            yaxis_title="N alive",
            showlegend=False,
        ))
        return _plotly_iframe(fig, "survival")

    # --- Energy Plot ---
    @render.ui
    def energy_plot():
        h = history.get()
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(height=280))
            return _plotly_iframe(fig, "energy")
        times = [r["time"] for r in h]
        ed = [r.get("mean_ed", 0) for r in h]
        fig.add_trace(go.Scatter(
            x=times, y=ed, mode="lines",
            line=dict(color=ACCENT_COLOR, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(232, 213, 183, 0.1)",
            name="Mean ED",
        ))
        fig.add_hline(
            y=4.0, line_dash="dot", line_color=SALMON_COLOR, line_width=1.5,
            annotation_text="Mortality",
            annotation_font=dict(size=10, color=SALMON_COLOR),
            annotation_position="top left",
        )
        fig.update_layout(**_base_layout(
            height=280,
            xaxis_title="Hour",
            yaxis_title="kJ/g",
            showlegend=False,
        ))
        return _plotly_iframe(fig, "energy")

    # --- Behavior Plot ---
    @render.ui
    def behavior_plot():
        h = history.get()
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(height=280))
            return _plotly_iframe(fig, "behavior")
        times = [r["time"] for r in h]
        for b in range(5):
            counts = [r.get("behavior_counts", {}).get(b, 0) for r in h]
            fig.add_trace(go.Scatter(
                x=times, y=counts, mode="lines", name=BEH_NAMES[b],
                stackgroup="one",
                line=dict(color=BEH_COLORS[b], width=0.5),
                fillcolor=BEH_COLORS[b],
            ))
        fig.update_layout(**_base_layout(
            height=280,
            xaxis_title="Hour",
            yaxis_title="Count",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=10),
            ),
        ))
        return _plotly_iframe(fig, "behavior")


app = App(app_ui, server, static_assets=str(Path(__file__).parent / "www"))
