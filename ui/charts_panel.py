"""Streaming charts panel — collapsible panel below the map with live Plotly.js charts."""
from shiny import ui


def charts_panel() -> ui.Tag:
    """Return the HTML container for three streaming charts + Plotly.js scripts."""
    return ui.div(
        # Drag handle / toggle
        ui.div(
            ui.span("\u25b2 LIVE CHARTS \u25b2"),
            class_="charts-panel-handle",
            id="charts-panel-handle",
        ),
        # Three chart containers
        ui.div(
            ui.div(
                ui.div("POPULATION", class_="chart-cell-title"),
                ui.div(id="chart-population", class_="chart-cell-plot"),
                class_="chart-cell",
            ),
            ui.div(
                ui.div("MIGRATION", class_="chart-cell-title"),
                ui.div(id="chart-migration", class_="chart-cell-plot"),
                class_="chart-cell",
            ),
            ui.div(
                ui.div("BEHAVIOR", class_="chart-cell-title"),
                ui.div(id="chart-behavior", class_="chart-cell-plot"),
                class_="chart-cell",
            ),
            class_="charts-panel-body",
            id="charts-panel-body",
        ),
        # Load Plotly.js from CDN + our streaming handler
        ui.tags.script(src="https://cdn.plot.ly/plotly-2.27.0.min.js"),
        ui.tags.script(src="streaming_charts.js"),
        class_="charts-panel",
        id="charts-panel",
    )
