"""Science & References tab content for the IBM dashboard."""
from shiny import ui


def _ref_block(title, body, sources):
    """A styled reference card."""
    source_items = [ui.tags.li(ui.tags.a(s[0], href=s[1], target="_blank")) for s in sources]
    return ui.div(
        ui.tags.h5(title, class_="ref-title"),
        ui.tags.p(body, class_="ref-body"),
        ui.tags.ul(*source_items, class_="ref-sources"),
        class_="ref-card",
    )


def science_panel():
    """Full scientific background panel."""
    return ui.div(
        ui.div(
            ui.tags.h3(
                "Scientific Basis",
                style="font-family: var(--font-display); color: var(--sediment-light);"
                " margin-bottom: 4px;",
            ),
            ui.tags.p(
                "This IBM models adult Atlantic salmon (Salmo salar) migration "
                "through the Curonian Lagoon, the largest European coastal lagoon, "
                "from Klaip\u0117da Strait to spawning grounds in the Nemunas River basin. "
                "The model integrates hydrodynamic forcing, bioenergetics, and "
                "estuary-specific stressors based on the following scientific foundations.",
                class_="ref-intro",
            ),
            class_="ref-header",
        ),
        ui.row(
            ui.column(6, _ref_block(
                "Hydrodynamics & Salinity",
                "The Curonian Lagoon is microtidal; exchange is dominated by wind- "
                "and pressure-driven flows through Klaip\u0117da Strait with frequent "
                "two-layer structure (surface outflow, bottom inflow). Salinity "
                "intrusions decay within ~20 km from the sea, with strong vertical "
                "gradients near the strait. Movement rules and calibration targets "
                "explicitly reflect two-directional flow that toggles with wind "
                "and sea-level set-up.",
                [
                    ("Zemlys et al. 2013 \u2014 Curonian Lagoon circulation",
                     "https://os.copernicus.org"),
                    ("Umgiesser et al. \u2014 Lagoon hydrodynamic modelling",
                     "https://iris.cnr.it"),
                ],
            )),
            ui.column(6, _ref_block(
                "Bioenergetics Model",
                "Wisconsin bioenergetics model for non-feeding migrants "
                "(consumption C=0). Respiration parameterized per Forseth et al. "
                "(2001) for Salmo salar: RA=0.00264, RB=\u22120.217, RQ=0.06818. "
                "Energy density below 4.0 kJ/g triggers mortality (Snyder et al. 2019). "
                "Osmoregulation cost multiplier m_osmo(S) adds energy penalties "
                "in brackish waters above tolerance.",
                [
                    ("Forseth et al. 2001 \u2014 S. salar bioenergetics",
                     "https://doi.org/10.1139/f01-022"),
                    ("Snyder et al. 2019 \u2014 HexSim migration IBM",
                     "https://doi.org/10.1016/j.ecolmodel.2019.108776"),
                ],
            )),
        ),
        ui.row(
            ui.column(6, _ref_block(
                "Eutrophication & Hypoxia",
                "The system is hypertrophic with recurring cyanobacterial "
                "\"hyperblooms\" that promote transient hypoxia and internal P "
                "regeneration. The IBM includes hypoxia-avoidance rules with "
                "lethal/high DO thresholds reflecting field-observed feedbacks "
                "during bloom-heat-calm episodes.",
                [
                    ("Bartosevi\u010dien\u0117 et al. \u2014 Lagoon eutrophication",
                     "https://link.springer.com"),
                    ("Zilius et al. \u2014 Hypoxia & nutrient cycling",
                     "https://frontiersin.org"),
                ],
            )),
            ui.column(6, _ref_block(
                "Seiches & Meteotsunamis",
                "Klaip\u0117da port experiences hazardous long-wave oscillations "
                "(harbor seiches, meteotsunami-like events) causing rapid current "
                "reversals. The \"seiche-pause\" trigger delays fish movement when "
                "|dSSH/dt| exceeds a threshold, preventing unrealistic transport "
                "during extreme oscillation events.",
                [
                    ("Papreckien\u0117 et al. \u2014 Harbor seiches in Klaip\u0117da",
                     "https://link.springer.com"),
                    ("Galli & Soomere \u2014 Long waves in the Baltic",
                     "https://jstor.org"),
                ],
            )),
        ),
        ui.row(
            ui.column(6, _ref_block(
                "Water Renewal & Connectivity",
                "Lagoon-wide water renewal varies with Nemunas discharge and "
                "prevailing wind regimes. Discharge largely sets residence times, "
                "while wind controls north-south exchange and internal connectivity. "
                "Boundary forcing hooks (Nemunas Q, wind) and validation metrics "
                "for seasonal renewal reflect these dynamics.",
                [
                    ("Idu\u017eyt\u0117 et al. \u2014 Renewal time modelling",
                     "https://os.copernicus.org"),
                    ("Ferrarin et al. \u2014 Lagoon water residence times",
                     "https://docslib.org"),
                ],
            )),
            ui.column(6, _ref_block(
                "Forcing Data & Operational Products",
                "The model uses CMEMS Baltic Physics (NEMO v4.2.1, ~2 km, 56 z-levels) "
                "for currents, salinity, SSH, and temperature. Wave data is available from "
                "CMEMS Baltic Waves. Cross-checking against BOOS multi-model ensembles "
                "is supported. Nemunas discharge provides the primary freshwater forcing.",
                [
                    ("CMEMS BALTICSEA_ANALYSISFORECAST_PHY_003_006",
                     "https://data.marine.copernicus.eu"),
                    ("BOOS Baltic Operational Oceanographic System",
                     "https://boos.org"),
                ],
            )),
        ),
        ui.div(
            ui.row(
                ui.column(12, _ref_block(
                    "Calibration Targets",
                    "Salinity penetration length (~20 km south of Kiaul\u0117s Nugara); "
                    "two-directional flow statistics near the strait; seasonal renewal "
                    "time sensitivity to discharge and wind; hypoxia exposure timing "
                    "during bloom/heat/calm spells; port long-wave event frequency "
                    "(movement pausing under strong dSSH/dt).",
                    [
                        ("Dredging & salinity sensitivity analyses",
                         "https://academia.edu"),
                        ("Long-term salinity trends",
                         "https://researchgate.net"),
                    ],
                )),
            ),
        ),
        class_="science-content",
    )
