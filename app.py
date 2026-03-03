"""Baltic Salmon IBM — Shiny for Python Application (Lagoon Field Station theme)."""
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
BEH_SYMBOLS = ["circle", "diamond", "star", "triangle-up", "triangle-down"]

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


def _hex_to_rgb_f(h: str) -> tuple[float, float, float]:
    """Convert hex color '#rrggbb' to (r, g, b) floats 0-1 for matplotlib."""
    r, g, b = _hex_to_rgb(h)
    return r / 255.0, g / 255.0, b / 255.0


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

# deck.gl standalone HTML template (bypasses pydeck widget system)
DECK_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<script src="https://unpkg.com/deck.gl@9.1.4/dist.min.js"></script>
<style>
  body {{ margin:0; overflow:hidden; background:{bg_color}; }}
  #deck-container {{ width:100vw; height:100vh; background:{bg_color}; }}
  #deck-container canvas {{ background:transparent !important; }}
  .legend {{
    position:absolute; bottom:12px; right:12px;
    background:rgba(11,31,44,0.85); border-radius:6px;
    padding:8px 12px; font:11px 'Work Sans',sans-serif; color:#e4e8e6;
  }}
  .legend-title {{ font-size:10px; color:#6a8a8a; margin-bottom:4px; }}
  .legend-bar {{
    width:120px; height:10px; border-radius:3px;
    background:linear-gradient(to right, {gradient_css});
  }}
  .legend-range {{ display:flex; justify-content:space-between; font-size:9px; color:#6a8a8a; margin-top:2px; }}
</style>
</head><body>
<div id="deck-container"></div>
<div class="legend">
  <div class="legend-title">{cbar_title}</div>
  <div class="legend-bar"></div>
  <div class="legend-range"><span>{z_min}</span><span>{z_max}</span></div>
</div>
<script>
const W={water_data};
const A={agent_data};
new deck.DeckGL({{
  container:'deck-container',
  style:{{backgroundColor:'{bg_color}'}},
  glOptions:{{alpha:true}},
  parameters:{{clearColor:[{clear_r},{clear_g},{clear_b},1]}},
  views:[new deck.OrthographicView()],
  initialViewState:{{target:[{cx},{cy},0],zoom:{zoom}}},
  controller:true,
  layers:[
    new deck.ScatterplotLayer({{
      id:'water',data:W,
      getPosition:d=>[d[0],d[1]],
      getFillColor:d=>[d[2],d[3],d[4],220],
      getRadius:{hex_radius},
      radiusMinPixels:3,radiusMaxPixels:8,
      pickable:false,updateTriggers:{{getFillColor:[Date.now()]}}
    }}),
    new deck.ScatterplotLayer({{
      id:'agents',data:A,
      getPosition:d=>[d[0],d[1]],
      getFillColor:d=>[d[2],d[3],d[4],240],
      getRadius:{hex_radius}*10,
      radiusMinPixels:5,radiusMaxPixels:12,
      getLineColor:[0,0,0,140],lineWidthMinPixels:1,stroked:true
    }})
  ]
}});
</script>
</body></html>"""


# --- App UI ---
app_ui = ui.page_sidebar(
    sidebar_panel(),
    ui.head_content(
        ui.tags.link(rel="stylesheet", href="style.css"),
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        head_includes(),
    ),
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
    @reactive.event(input.btn_reset, ignore_none=False)
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

        # Run blocking init in a thread to avoid freezing the event loop
        sim = await asyncio.to_thread(
            Simulation, cfg, n_agents=n_agents, data_dir="data", rng_seed=rng_seed,
        )
        sim.bio_params = BioParams(
            RA=ra, RB=rb, RQ=rq, ED_MORTAL=ed_mortal,
            T_OPT=t_opt, T_MAX=t_max,
        )
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
        sim.step()
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
        speed = input.speed()
        while running.get() and sim.current_t < steps:
            for _ in range(speed):
                if sim.current_t >= steps:
                    break
                sim.step()
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

    # --- Map update (push to shiny-deckgl widget) ---
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
