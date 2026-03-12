"""Sidebar parameter controls — Lagoon Field Station theme with scientific context."""
from shiny import ui


def _section(label):
    """Section header with decorative divider."""
    return ui.div(label, class_="section-divider")


def _hint(text):
    """Subtle scientific annotation below an input group."""
    return ui.tags.div(
        text,
        class_="param-hint",
    )


def sidebar_panel():
    return ui.sidebar(
        ui.div(
            ui.tags.span(
                ui.HTML("&#127843;"), " ",
                ui.tags.strong("Salmon IBM"),
                style="font-family: var(--font-display); font-size: 1.1rem; color: var(--sediment-light);",
            ),
            style="margin-bottom: 12px;",
        ),
        ui.hr(),

        # --- Landscape ---
        _section("Landscape"),
        ui.input_select(
            "landscape", "Study area",
            choices={
                "curonian": "Curonian Lagoon",
                "columbia": "Columbia River",
            },
        ),

        # --- Population ---
        _section("Population"),
        ui.input_numeric("n_agents", "Agents", value=50, min=1, max=1000),
        ui.input_numeric("n_steps", "Hours", value=24, min=1, max=8760),
        ui.input_numeric("rng_seed", "Seed", value=42),
        _hint("Parameter changes take effect on Reset."),

        # --- Bioenergetics ---
        _section("Bioenergetics"),
        _hint("Wisconsin model (Forseth et al. 2001, Salmo salar). "
              "Non-feeding migrants: C=0, hourly energy loss from respiration."),
        ui.input_numeric("ra", "RA intercept", value=0.00264, step=0.0001),
        ui.input_numeric("rb", "RB allometric", value=-0.217, step=0.01),
        ui.input_numeric("rq", "RQ thermal", value=0.06818, step=0.001),
        _hint("RA \u00d7 mass^RB \u00d7 exp(RQ \u00d7 T) \u00d7 OXY_CAL \u00d7 activity"),
        ui.input_numeric("ed_init", "Init. ED (kJ/g)", value=6.5, step=0.1),
        ui.input_numeric("ed_mortal", "Lethal ED (kJ/g)", value=4.0, step=0.1),
        _hint("Death at ED < 4.0 kJ/g (Snyder et al. 2019)."),
        ui.input_numeric("t_opt", "T optimal (\u00b0C)", value=16.0, step=0.5),
        ui.input_numeric("t_max", "T lethal (\u00b0C)", value=26.0, step=0.5),
        _hint("Monotonic R(T): exponential increase with temperature (Wisconsin model)."),

        # --- Osmoregulation & Salinity ---
        _section("Osmoregulation"),
        _hint("Brackish inflows in Klaip\u0117da Strait impose osmoregulatory costs. "
              "m_osmo(S) = 1 + k \u00d7 max(0, S \u2212 (S_opt + S_tol)). "
              "Salinity intrusions decay ~20 km from the sea."),
        ui.input_numeric("s_opt", "S optimum (PSU)", value=0.5, step=0.1),
        ui.input_numeric("s_tol", "S tolerance (PSU)", value=6.0, step=0.5),
        ui.input_numeric("sal_k", "Cost coefficient k", value=0.6, step=0.1),

        # --- Dissolved Oxygen ---
        _section("Dissolved Oxygen"),
        _hint("Hypertrophic system with cyanobacterial hyperblooms "
              "causing transient hypoxia. Agents avoid low-DO zones; "
              "lethal exposure kills."),
        ui.input_numeric("do_lethal", "DO lethal (mg/L)", value=2.0, step=0.5),
        ui.input_numeric("do_high", "DO avoidance (mg/L)", value=4.0, step=0.5),

        # --- Seiches ---
        _section("Seiches & Long Waves"),
        _hint("Klaip\u0117da port experiences harbor seiches and meteotsunami-like "
              "events with rapid current reversals. Movement pauses when "
              "|dSSH/dt| exceeds threshold."),
        ui.input_numeric("seiche_thresh", "dSSH/dt thresh. (m/15min)", value=0.02, step=0.005),

        # --- Display ---
        _section("Map Display"),
        ui.input_select(
            "map_field", "Color mesh by",
            choices={
                "temperature": "Temperature",
                "salinity": "Salinity",
                "ssh": "Sea Surface Height",
                "depth": "Bathymetry",
            },
        ),
        width=290,
    )
