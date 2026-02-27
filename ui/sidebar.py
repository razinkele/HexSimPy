"""Sidebar parameter controls for the Shiny app."""
from shiny import ui


def sidebar_panel():
    return ui.sidebar(
        ui.h4("Simulation Parameters"),
        ui.input_numeric("n_agents", "Number of agents", value=50, min=1, max=1000),
        ui.input_numeric("n_steps", "Simulation hours", value=24, min=1, max=8760),
        ui.input_numeric("rng_seed", "Random seed", value=42),
        ui.hr(),
        ui.h5("Bioenergetics (Salmo salar)"),
        ui.input_numeric("ra", "RA (resp. intercept)", value=0.00264, step=0.0001),
        ui.input_numeric("rb", "RB (allometric exp.)", value=-0.217, step=0.01),
        ui.input_numeric("rq", "RQ (temp coeff.)", value=0.06818, step=0.001),
        ui.input_numeric("ed_init", "Initial energy density (kJ/g)", value=6.5, step=0.1),
        ui.input_numeric("ed_mortal", "Mortality threshold (kJ/g)", value=4.0, step=0.1),
        ui.hr(),
        ui.h5("Estuary"),
        ui.input_numeric("s_opt", "Salinity optimum (PSU)", value=0.5, step=0.1),
        ui.input_numeric("s_tol", "Salinity tolerance (PSU)", value=6.0, step=0.5),
        ui.input_numeric("sal_k", "Salinity cost coefficient", value=0.6, step=0.1),
        ui.input_numeric("do_lethal", "DO lethal (mg/L)", value=2.0, step=0.5),
        ui.input_numeric("do_high", "DO avoidance (mg/L)", value=4.0, step=0.5),
        ui.input_numeric("seiche_thresh", "Seiche dSSH/dt threshold", value=0.02, step=0.005),
        ui.hr(),
        ui.h5("Map Display"),
        ui.input_select(
            "map_field", "Color mesh by",
            choices={"temperature": "Temperature", "salinity": "Salinity",
                     "ssh": "SSH", "depth": "Depth"},
        ),
        width=320,
    )
