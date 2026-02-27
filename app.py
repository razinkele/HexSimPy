"""Baltic Salmon IBM — Shiny for Python Application."""
import asyncio

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from salmon_ibm.config import load_config
from salmon_ibm.bioenergetics import BioParams
from salmon_ibm.simulation import Simulation
from ui.sidebar import sidebar_panel
from ui.run_controls import run_controls_panel


app_ui = ui.page_sidebar(
    sidebar_panel(),
    ui.navset_tab(
        ui.nav_panel(
            "Map",
            run_controls_panel(),
            output_widget("map_plot", height="500px"),
        ),
        ui.nav_panel(
            "Charts",
            ui.row(
                ui.column(6, output_widget("survival_plot", height="300px")),
                ui.column(6, output_widget("energy_plot", height="300px")),
            ),
            ui.row(
                ui.column(12, output_widget("behavior_plot", height="300px")),
            ),
        ),
    ),
    title="Baltic Salmon IBM — Curonian Lagoon",
)


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
            return "Not initialized"
        return f"Alive: {sim.pool.alive.sum()}/{sim.pool.n} | Arrived: {sim.pool.arrived.sum()}"

    @render.text
    def progress_text():
        sim = sim_state.get()
        if sim is None:
            return "t = 0"
        return f"t = {sim.current_t} h"

    @render_widget
    def map_plot():
        sim = sim_state.get()
        _ = history.get()
        if sim is None:
            return go.Figure()

        mesh = sim.mesh
        field_name = input.map_field()

        fig = go.Figure()

        if field_name == "depth":
            z = mesh.depth
        elif field_name in sim.env.fields:
            z = sim.env.fields[field_name]
        else:
            z = mesh.depth

        water = mesh.water_mask
        fig.add_trace(go.Scatter(
            x=mesh.centroids[water, 1], y=mesh.centroids[water, 0],
            mode="markers",
            marker=dict(size=8, color=z[water], colorscale="Viridis",
                        colorbar=dict(title=field_name)),
            name="Mesh",
            hovertemplate="lon: %{x:.3f}<br>lat: %{y:.3f}<br>value: %{marker.color:.2f}",
        ))

        alive = sim.pool.alive
        if alive.any():
            agent_tris = sim.pool.tri_idx[alive]
            behavior_names = ["Hold", "Random", "CWR", "Upstream", "Downstream"]
            colors = ["gray", "blue", "cyan", "red", "orange"]
            for b in range(5):
                b_mask = sim.pool.behavior[alive] == b
                if b_mask.any():
                    tris_b = agent_tris[b_mask]
                    fig.add_trace(go.Scatter(
                        x=mesh.centroids[tris_b, 1], y=mesh.centroids[tris_b, 0],
                        mode="markers",
                        marker=dict(size=10, color=colors[b], symbol="diamond",
                                    line=dict(width=1, color="black")),
                        name=behavior_names[b],
                    ))

        fig.update_layout(
            xaxis_title="Longitude", yaxis_title="Latitude",
            height=500, margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    @render_widget
    def survival_plot():
        h = history.get()
        if not h:
            return go.Figure().update_layout(title="Survival")
        times = [r["time"] for r in h]
        alive = [r["n_alive"] for r in h]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=alive, mode="lines", name="Alive"))
        fig.update_layout(title="Survival", xaxis_title="Hour", yaxis_title="N alive",
                          height=300, margin=dict(l=40, r=20, t=40, b=40))
        return fig

    @render_widget
    def energy_plot():
        h = history.get()
        if not h:
            return go.Figure().update_layout(title="Energy Density")
        times = [r["time"] for r in h]
        ed = [r.get("mean_ed", 0) for r in h]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=ed, mode="lines", name="Mean ED"))
        fig.add_hline(y=4.0, line_dash="dash", line_color="red",
                      annotation_text="Mortality threshold")
        fig.update_layout(title="Mean Energy Density", xaxis_title="Hour",
                          yaxis_title="kJ/g", height=300, margin=dict(l=40, r=20, t=40, b=40))
        return fig

    @render_widget
    def behavior_plot():
        h = history.get()
        if not h:
            return go.Figure().update_layout(title="Behavior Distribution")
        times = [r["time"] for r in h]
        names = ["Hold", "Random", "CWR", "Upstream", "Downstream"]
        colors = ["gray", "blue", "cyan", "red", "orange"]
        fig = go.Figure()
        for b in range(5):
            counts = [r.get("behavior_counts", {}).get(b, 0) for r in h]
            fig.add_trace(go.Scatter(
                x=times, y=counts, mode="lines", name=names[b],
                stackgroup="one", line=dict(color=colors[b]),
            ))
        fig.update_layout(title="Behavior Distribution", xaxis_title="Hour",
                          yaxis_title="Count", height=300, margin=dict(l=40, r=20, t=40, b=40))
        return fig


app = App(app_ui, server)
