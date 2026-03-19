"""Run control bar — Lagoon Field Station theme."""
from shiny import ui


def run_controls_panel():
    return ui.div(
        ui.div(
            ui.input_action_button("btn_run", "Run", class_="btn-run"),
            ui.input_action_button("btn_step", "Step", class_="btn-step"),
            ui.input_action_button("btn_pause", "Pause", class_="btn-pause"),
            ui.input_action_button("btn_reset", "Reset", class_="btn-reset"),
            class_="btn-group",
        ),
        ui.div(class_="spacer"),
        ui.div(
            ui.tags.label("Speed", **{"for": "speed"}),
            ui.input_slider("speed", None, min=1, max=10, value=1, width="120px"),
            class_="speed-control",
        ),
        ui.div(
            ui.input_switch("show_trails", "Trails", value=False),
            class_="trail-toggle",
        ),
        ui.div(class_="spacer"),
        ui.div(
            ui.output_text("progress_text"),
            style="font-family: var(--font-mono); font-size: 0.82rem; color: var(--lagoon-shallow);",
        ),
        ui.div(
            ui.output_text("status_text"),
            class_="status-badge",
        ),
        ui.output_ui("live_stats"),
        class_="run-controls-bar",
    )
