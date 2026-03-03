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

    # --- Map Display (unified — @render.ui) ---
    @render.ui
    def map_display():
        sim = sim_state.get()
        _ = history.get()

        if sim is None:
            return ui.HTML(
                '<div style="height:520px;display:flex;align-items:center;'
                'justify-content:center;color:#6a8a8a;'
                "font-family:'Cormorant Garamond',serif;font-size:1.1rem\">"
                "Click Step or Run to initialize</div>"
            )

        mesh = sim.mesh
        field_name = input.map_field()
        is_hexsim = hasattr(mesh, "n_cells")

        # Resolve field + colorscale
        if field_name == "depth":
            z = mesh.depth
            cscale = BATHY_COLORSCALE
            cbar_title = "Depth (m)"
        elif field_name in sim.env.fields:
            z = sim.env.fields[field_name]
            cscale = TEMP_COLORSCALE
            field_labels = {
                "temperature": "Temp (\u00b0C)", "salinity": "Sal (PSU)",
                "ssh": "SSH (m)",
            }
            cbar_title = field_labels.get(field_name, field_name)
        else:
            z = mesh.depth
            cscale = BATHY_COLORSCALE
            cbar_title = "Depth (m)"

        if is_hexsim:
            return _render_deckgl(sim, mesh, z, cscale, cbar_title)
        return _render_mpl(sim, mesh, z, cscale, cbar_title)

    # ── Matplotlib renderer (Curonian / TriMesh) ──
    def _render_mpl(sim, mesh, z, cscale, cbar_title):
        # Build matplotlib colormap from Plotly-style stops
        cmap = LinearSegmentedColormap.from_list(
            "lagoon",
            [(s[0], _hex_to_rgb_f(s[1])) for s in cscale],
        )

        bg = "#0d2233"
        fig_m, ax = plt.subplots(figsize=(7.5, 5.2), dpi=120, facecolor=bg)
        ax.set_facecolor(bg)

        water_idx = np.where(mesh.water_mask)[0]
        sc = ax.scatter(
            mesh.centroids[water_idx, 1], mesh.centroids[water_idx, 0],
            c=z[water_idx], cmap=cmap, s=8, edgecolors="none",
            rasterized=True,
        )
        cb = fig_m.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(cbar_title, color=ACCENT_COLOR, fontsize=9)
        cb.ax.tick_params(colors=AXIS_COLOR, labelsize=8)

        # Agent overlay
        alive = sim.pool.alive
        if alive.any():
            agent_tris = sim.pool.tri_idx[alive]
            behaviors = sim.pool.behavior[alive]
            mpl_markers = ["o", "D", "*", "^", "v"]
            for b in range(5):
                b_mask = behaviors == b
                if b_mask.any():
                    tris_b = agent_tris[b_mask]
                    ax.scatter(
                        mesh.centroids[tris_b, 1], mesh.centroids[tris_b, 0],
                        c=BEH_COLORS[b], s=50, marker=mpl_markers[b],
                        edgecolors="black", linewidths=0.8, zorder=5,
                        label=BEH_NAMES[b],
                    )
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=5,
                fontsize=8, frameon=False, labelcolor=TEXT_COLOR,
            )

        ax.set_xlabel("Longitude", color=AXIS_COLOR, fontsize=9)
        ax.set_ylabel("Latitude", color=AXIS_COLOR, fontsize=9)
        ax.tick_params(colors=AXIS_COLOR, labelsize=8)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_color((42/255, 122/255, 122/255, 0.15))
        fig_m.tight_layout()

        buf = io.BytesIO()
        fig_m.savefig(buf, format="png", facecolor=bg, bbox_inches="tight")
        plt.close(fig_m)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return ui.HTML(
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;height:520px;object-fit:contain;border-radius:8px;" />'
        )

    # ── deck.gl renderer (Columbia / HexMesh) ──
    def _render_deckgl(sim, mesh, z, cscale, cbar_title):
        rgb = _colorscale_rgb(z, cscale)

        # Random subsample for reasonable HTML size (~3.5 MB)
        # Random avoids banding artifacts from regular stepping on row-major data
        n = mesh.n_cells
        if n > MAX_DECK_POINTS:
            rng = np.random.default_rng(0)  # fixed seed for stable rendering
            idx = rng.choice(n, MAX_DECK_POINTS, replace=False)
            idx.sort()  # sort for cache-friendly access
        else:
            idx = np.arange(n)

        # Compact: each point is [x, y, r, g, b]
        xs = mesh.centroids[idx, 1]
        ys = mesh.centroids[idx, 0]
        water_pts = [
            [round(float(xs[i]), 1), round(float(ys[i]), 1),
             int(rgb[idx[i], 0]), int(rgb[idx[i], 1]), int(rgb[idx[i], 2])]
            for i in range(len(idx))
        ]

        # Agent overlay
        agent_pts = []
        alive = sim.pool.alive
        if alive.any():
            agent_tris = sim.pool.tri_idx[alive]
            behaviors = sim.pool.behavior[alive]
            for t, b in zip(agent_tris, behaviors):
                rc, gc, bc = BEH_COLORS_RGB[int(b)]
                agent_pts.append([
                    round(float(mesh.centroids[t, 1]), 1),
                    round(float(mesh.centroids[t, 0]), 1),
                    rc, gc, bc,
                ])

        cx = float(np.mean(mesh.centroids[:, 1]))
        cy = float(np.mean(mesh.centroids[:, 0]))

        # Hex radius: spacing between adjacent hex centers ≈ sqrt(3) grid units;
        # circle radius to visually fill hex ≈ 1.0 grid unit.
        # At typical zoom, radiusMinPixels dominates anyway.
        hex_radius = 1.0

        # Auto-zoom: fit full extent into ~520px viewport
        x_range = float(np.ptp(mesh.centroids[:, 1]))
        y_range = float(np.ptp(mesh.centroids[:, 0]))
        extent = max(x_range, y_range, 1.0)
        zoom = -np.log2(extent / 512)

        # Colorbar legend
        z_min_val = f"{float(np.nanmin(z)):.1f}"
        z_max_val = f"{float(np.nanmax(z)):.1f}"
        gradient_stops = ",".join(
            f"{s[1]} {int(s[0]*100)}%" for s in cscale
        )

        # WebGL clear color (background): #0b1f2c → normalized floats
        html = DECK_TEMPLATE.format(
            bg_color="#0b1f2c",
            clear_r=round(11/255, 4), clear_g=round(31/255, 4),
            clear_b=round(44/255, 4),
            water_data=json.dumps(water_pts, separators=(",", ":")),
            agent_data=json.dumps(agent_pts, separators=(",", ":")),
            cx=round(cx, 1), cy=round(cy, 1),
            zoom=round(zoom, 2),
            hex_radius=round(hex_radius, 1),
            cbar_title=cbar_title,
            z_min=z_min_val, z_max=z_max_val,
            gradient_css=gradient_stops,
        )

        deck_path = WWW_DIR / "deck_map.html"
        deck_path.write_text(html, encoding="utf-8")

        ts = int(time.time() * 1000)
        return ui.tags.iframe(
            src=f"deck_map.html?t={ts}",
            width="100%",
            height="520px",
            style="border: none; border-radius: 8px; background: #0b1f2c;",
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
