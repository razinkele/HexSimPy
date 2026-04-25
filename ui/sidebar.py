"""Sidebar — run controls + parameter accordions."""

from shiny import ui


def _hint(text):
    """Subtle scientific annotation below an input group."""
    return ui.tags.div(text, class_="param-hint")


def sidebar_panel():
    return ui.sidebar(
        # ── Status strip — single tight row, no duplicate app title (the
        # navbar already shows it).  progress_text shows simulation hour;
        # status_text shows alive/arrived counts.  live_stats fills in
        # behaviour breakdown once the sim is stepping. ──
        ui.div(
            ui.output_text("progress_text"),
            ui.output_text("status_text"),
            class_="sidebar-status-strip",
        ),
        ui.output_ui("live_stats"),
        # ── Accordion ── Landscape opens first so the user picks a study
        # area before anything else; Run + parameter groups stay collapsed.
        ui.accordion(
            ui.accordion_panel(
                "Landscape",
                ui.input_select(
                    "landscape",
                    "Study area",
                    choices={
                        # Curonian Lagoon H3 is now the default
                        # (v1.2.6).  Internal key stays "nemunas"
                        # (matches configs/config_nemunas_h3.yaml +
                        # data/nemunas_h3_*.nc filenames).  Water
                        # mask now uses inSTREAM example_baltic
                        # polygons + NE ocean — tighter inland than
                        # the prior OSM source, eliminates the
                        # ~1 000-cell Nemunas-Delta-polderland leak.
                        "nemunas": "Curonian Lagoon H3",
                        "curonian_trimesh": "Curonian Lagoon TriMesh",
                        "columbia": "Columbia River",
                    },
                    selected="nemunas",
                ),
                ui.input_numeric("n_agents", "Agents", value=50, min=1, max=1000),
                ui.input_numeric("n_steps", "Hours", value=480, min=1, max=8760),
                ui.input_numeric("rng_seed", "Seed", value=42),
                _hint("Parameter changes take effect on Reset."),
            ),
            ui.accordion_panel(
                "Run",
                ui.div(
                    ui.input_action_button("btn_run", "Run", class_="btn-run"),
                    ui.input_action_button("btn_step", "Step", class_="btn-step"),
                    ui.input_action_button("btn_pause", "Pause", class_="btn-pause"),
                    ui.input_action_button("btn_reset", "Reset", class_="btn-reset"),
                    class_="btn-group sidebar-btn-group",
                ),
                ui.div(
                    ui.tags.label("Speed", **{"for": "speed"}),
                    ui.input_slider(
                        "speed", None, min=1, max=10, value=1, width="100%"
                    ),
                    class_="speed-control sidebar-speed",
                ),
                ui.div(
                    ui.input_switch("show_trails", "Trails", value=True),
                    class_="trail-toggle",
                ),
                ui.div(
                    ui.input_select(
                        "map_field",
                        "Map field",
                        choices={
                            "depth": "Bathymetry",
                            "temperature": "Temperature",
                            "salinity": "Salinity",
                            "ssh": "SSH",
                        },
                        selected="depth",
                        width="100%",
                    ),
                    class_="field-selector",
                ),
            ),
            ui.accordion_panel(
                "Bioenergetics",
                _hint(
                    "Wisconsin model (Forseth et al. 2001, Salmo salar). "
                    "Non-feeding migrants: C=0, hourly energy loss from respiration."
                ),
                ui.input_numeric("ra", "RA intercept", value=0.00264, step=0.0001),
                ui.input_numeric("rb", "RB allometric", value=-0.217, step=0.01),
                ui.input_numeric("rq", "RQ thermal", value=0.06818, step=0.001),
                _hint(
                    "RA × mass^RB × exp(RQ × T) × OXY_CAL × activity"
                ),
                ui.input_numeric("ed_init", "Init. ED (kJ/g)", value=6.5, step=0.1),
                ui.input_numeric("ed_mortal", "Lethal ED (kJ/g)", value=4.0, step=0.1),
                _hint("Death at ED < 4.0 kJ/g (Snyder et al. 2019)."),
                ui.input_numeric("t_opt", "T optimal (°C)", value=16.0, step=0.5),
                ui.input_numeric("t_max", "T lethal (°C)", value=26.0, step=0.5),
                _hint(
                    "Monotonic R(T): exponential increase with temperature (Wisconsin model)."
                ),
            ),
            ui.accordion_panel(
                "Osmoregulation",
                _hint(
                    "Brackish inflows in Klaipėda Strait impose osmoregulatory costs. "
                    "m_osmo(S) = 1 + k × max(0, S − (S_opt + S_tol)). "
                    "Salinity intrusions decay ~20 km from the sea."
                ),
                ui.input_numeric("s_opt", "S optimum (PSU)", value=0.5, step=0.1),
                ui.input_numeric("s_tol", "S tolerance (PSU)", value=6.0, step=0.5),
                ui.input_numeric("sal_k", "Cost coefficient k", value=0.6, step=0.1),
            ),
            ui.accordion_panel(
                "Dissolved Oxygen",
                _hint(
                    "Hypertrophic system with cyanobacterial hyperblooms "
                    "causing transient hypoxia. Agents avoid low-DO zones; "
                    "lethal exposure kills."
                ),
                ui.input_numeric("do_lethal", "DO lethal (mg/L)", value=2.0, step=0.5),
                ui.input_numeric("do_high", "DO avoidance (mg/L)", value=4.0, step=0.5),
            ),
            ui.accordion_panel(
                "Seiches & Long Waves",
                _hint(
                    "Klaipėda port experiences harbor seiches and meteotsunami-like "
                    "events with rapid current reversals. Movement pauses when "
                    "|dSSH/dt| exceeds threshold."
                ),
                ui.input_numeric(
                    "seiche_thresh", "dSSH/dt thresh. (m/15min)", value=0.02, step=0.005
                ),
            ),
            # Landscape opens by default — picking the study area is the
            # first decision; everything else has sensible defaults.
            open=["Landscape"],
        ),
        width=290,
    )
