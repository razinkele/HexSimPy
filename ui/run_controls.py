"""Run control buttons and progress display."""
from shiny import ui


def run_controls_panel():
    return ui.div(
        ui.row(
            ui.column(3, ui.input_action_button("btn_run", "Run", class_="btn-primary")),
            ui.column(3, ui.input_action_button("btn_step", "Step")),
            ui.column(3, ui.input_action_button("btn_pause", "Pause")),
            ui.column(3, ui.input_action_button("btn_reset", "Reset", class_="btn-warning")),
        ),
        ui.row(
            ui.column(6, ui.input_slider("speed", "Steps/update", min=1, max=10, value=1)),
            ui.column(6, ui.output_text("status_text")),
        ),
        ui.output_text("progress_text"),
    )
