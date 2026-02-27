"""Baltic Salmon IBM — Shiny for Python Application (Lagoon Field Station theme)."""
import asyncio
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

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


# --- App UI ---
app_ui = ui.page_sidebar(
    sidebar_panel(),
    ui.head_content(
        ui.tags.link(rel="stylesheet", href="style.css"),
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
    ),
    run_controls_panel(),
    ui.navset_tab(
        ui.nav_panel(
            "Map",
            ui.div(
                output_widget("map_plot", height="520px"),
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
                        output_widget("survival_plot", height="280px"),
                        class_="chart-card",
                    ),
                ),
                ui.column(
                    6,
                    ui.div(
                        ui.div("Energy Reserve", class_="chart-card-title"),
                        output_widget("energy_plot", height="280px"),
                        class_="chart-card",
                    ),
                ),
            ),
            ui.row(
                ui.column(
                    12,
                    ui.div(
                        ui.div("Behavioral State Distribution", class_="chart-card-title"),
                        output_widget("behavior_plot", height="280px"),
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
            "Baltic Salmon IBM",
            style="font-family: 'Cormorant Garamond', serif; font-weight: 700; color: #e8d5b7; font-size: 1.35rem;",
        ),
        ui.tags.span(
            " \u2014 Curonian Lagoon",
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
    def _init_sim():
        cfg = load_config("config_curonian_minimal.yaml")
        cfg["estuary"]["salinity_cost"]["S_opt"] = input.s_opt()
        cfg["estuary"]["salinity_cost"]["S_tol"] = input.s_tol()
        cfg["estuary"]["salinity_cost"]["k"] = input.sal_k()
        cfg["estuary"]["do_avoidance"]["lethal"] = input.do_lethal()
        cfg["estuary"]["do_avoidance"]["high"] = input.do_high()
        cfg["estuary"]["seiche_pause"]["dSSHdt_thresh_m_per_15min"] = input.seiche_thresh()

        sim = Simulation(
            cfg, n_agents=input.n_agents(), data_dir="data", rng_seed=input.rng_seed(),
        )
        sim.bio_params = BioParams(
            RA=input.ra(), RB=input.rb(), RQ=input.rq(), ED_MORTAL=input.ed_mortal(),
        )
        sim_state.set(sim)
        history.set([])
        running.set(False)

    @reactive.effect
    @reactive.event(input.btn_step)
    def _step():
        sim = sim_state.get()
        if sim is None:
            _init_sim()
            sim = sim_state.get()
        sim.step()
        history.set(sim.history.copy())

    @reactive.effect
    @reactive.event(input.btn_run)
    async def _run():
        running.set(True)
        sim = sim_state.get()
        if sim is None:
            _init_sim()
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
        sim = sim_state.get()
        if sim is None:
            return "Awaiting initialization"
        alive = int(sim.pool.alive.sum())
        total = sim.pool.n
        arrived = int(sim.pool.arrived.sum())
        return f"{alive}/{total} alive \u00b7 {arrived} arrived"

    @render.text
    def progress_text():
        sim = sim_state.get()
        if sim is None:
            return "t = 0 h"
        return f"t = {sim.current_t} h"

    # --- Map Plot ---
    @render_widget
    def map_plot():
        sim = sim_state.get()
        _ = history.get()
        if sim is None:
            fig = go.Figure()
            fig.update_layout(**_base_layout(height=520))
            fig.add_annotation(
                text="Click Step or Run to initialize",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color=AXIS_COLOR, family="Cormorant Garamond, serif"),
            )
            return fig

        mesh = sim.mesh
        field_name = input.map_field()

        if field_name == "depth":
            z = mesh.depth
            cscale = BATHY_COLORSCALE
            cbar_title = "Depth (m)"
        elif field_name in sim.env.fields:
            z = sim.env.fields[field_name]
            cscale = TEMP_COLORSCALE
            field_labels = {"temperature": "Temp (\u00b0C)", "salinity": "Sal (PSU)", "ssh": "SSH (m)"}
            cbar_title = field_labels.get(field_name, field_name)
        else:
            z = mesh.depth
            cscale = BATHY_COLORSCALE
            cbar_title = "Depth (m)"

        fig = go.Figure()
        water = mesh.water_mask

        fig.add_trace(go.Scatter(
            x=mesh.centroids[water, 1], y=mesh.centroids[water, 0],
            mode="markers",
            marker=dict(
                size=9, color=z[water], colorscale=cscale,
                colorbar=dict(
                    title=dict(text=cbar_title, font=dict(size=11, color=ACCENT_COLOR)),
                    tickfont=dict(size=10, color=AXIS_COLOR),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    len=0.6,
                ),
                line=dict(width=0.5, color="rgba(42, 122, 122, 0.3)"),
            ),
            name="Lagoon",
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Lon: %{x:.3f}\u00b0E<br>"
                "Lat: %{y:.3f}\u00b0N<br>"
                "Value: %{marker.color:.2f}"
                "<extra></extra>"
            ),
            customdata=[cbar_title] * int(water.sum()),
        ))

        # Agent overlay
        alive = sim.pool.alive
        if alive.any():
            agent_tris = sim.pool.tri_idx[alive]
            for b in range(5):
                b_mask = sim.pool.behavior[alive] == b
                if b_mask.any():
                    tris_b = agent_tris[b_mask]
                    fig.add_trace(go.Scatter(
                        x=mesh.centroids[tris_b, 1], y=mesh.centroids[tris_b, 0],
                        mode="markers",
                        marker=dict(
                            size=11, color=BEH_COLORS[b],
                            symbol=BEH_SYMBOLS[b],
                            line=dict(width=1.5, color="rgba(0,0,0,0.5)"),
                        ),
                        name=BEH_NAMES[b],
                    ))

        fig.update_layout(**_base_layout(
            height=520,
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=11, color=TEXT_COLOR),
                bgcolor="rgba(0,0,0,0)",
            ),
        ))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    # --- Survival Plot ---
    @render_widget
    def survival_plot():
        h = history.get()
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(height=280))
            return fig
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
        return fig

    # --- Energy Plot ---
    @render_widget
    def energy_plot():
        h = history.get()
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(height=280))
            return fig
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
        return fig

    # --- Behavior Plot ---
    @render_widget
    def behavior_plot():
        h = history.get()
        fig = go.Figure()
        if not h:
            fig.update_layout(**_base_layout(height=280))
            return fig
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
        return fig


app = App(app_ui, server, static_assets=str(Path(__file__).parent / "www"))
