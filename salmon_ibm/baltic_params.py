"""Baltic Atlantic salmon (Salmo salar) bioenergetics parameters.

All scientifically-sensitive values are sourced from peer-reviewed
literature; citations inline below. Parameter provenance summary:

  cmax_A, cmax_B: Smith, Booker & Wells 2009 (doi:10.1016/j.marenvres.2008.12.010)
                  — verify exact values against Table 2 before production use.
  T_OPT:          Jensen, Jonsson & Forseth 2001 (doi:10.1046/j.0269-8463.2001.00572.x)
                  — primary source for 16-17°C Atlantic salmon thermal peak;
                  Koskela et al. 1997 (doi:10.1111/j.1095-8649.1997.tb01976.x)
                  is Baltic low-temperature supporting evidence only.
  T_AVOID:        20.0°C — behavioral avoidance / reduced growth (Handeland
                  et al. 2008 doi:10.1016/j.aquaculture.2008.06.042). NOT a
                  hard mortality cap.
  T_ACUTE_LETHAL: 24.0°C — acute thermal mortality (Elliott & Elliott 2010
                  review; Smialek, Pander & Geist 2021 doi:10.1111/fme.12507).
  LW_a, LW_b:     Provenance needs verification. Baltic length-weight typically
                  from ICES WGBAST reports or Kallio-Nyberg & Ikonen 1992.
  fecundity:      Linear 2.0 eggs/g defensible for mid-sized (Heinimaa &
                  Heinimaa 2003: 1845 eggs/kg = 1.85 eggs/g for 9 kg females).
                  Real form declines with size — consider allometric for
                  production runs.
  spawn window:   Late Oct – Nov 30 for Lithuanian Nemunas tributaries
                  (Žeimena, Merkys, Dubysa). Use Heinimaa & Heinimaa 2003
                  or local Lithuanian sources for Nemunas basin.

These values supersede the generic Snyder-Chinook BioParams for Baltic
salmon simulations. See docs/superpowers/plans/2026-04-24-curonian-realism-upgrades.md
for full context and calibration notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BalticBioParams:
    """Baltic Atlantic salmon bioenergetics + life-history parameters.

    NOT inheriting from BioParams — Baltic ED_TISSUE uses lipid-first catabolism
    (36 kJ/g) vs Chinook whole-muscle (5 kJ/g), so defaults must diverge.
    Duplicated field names are deliberate.
    """

    # Wisconsin bioenergetics (compatible with existing update_energy signature)
    RA: float = 0.00264
    RB: float = -0.217
    RQ: float = 0.06818
    ED_MORTAL: float = 4.0
    ED_TISSUE: float = 36.0
    MASS_FLOOR_FRACTION: float = 0.5

    # Species-specific Baltic values. Two-threshold thermal response:
    #   T_OPT → peak growth
    #   T_AVOID → behavioral avoidance (NOT a mortality gate)
    #   T_ACUTE_LETHAL → hard mortality threshold
    cmax_A: float = 0.303
    cmax_B: float = -0.275
    T_OPT: float = 16.0
    T_AVOID: float = 20.0
    T_ACUTE_LETHAL: float = 24.0
    LW_a: float = 0.0077
    LW_b: float = 3.05
    fecundity_per_g: float = 2.0

    # Spawning phenology (day of year; Lithuanian Nemunas basin populations)
    spawn_window_start_day: int = 288  # Oct 15
    spawn_window_end_day: int = 334    # Nov 30
    spawn_temp_min_c: float = 5.0
    spawn_temp_max_c: float = 14.0

    # Activity by behavior (keep Snyder structure, Baltic-tunable)
    activity_by_behavior: dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    )

    def __post_init__(self):
        if self.T_AVOID <= self.T_OPT:
            raise ValueError(
                f"T_AVOID ({self.T_AVOID}) must be > T_OPT ({self.T_OPT})"
            )
        if self.T_ACUTE_LETHAL <= self.T_AVOID:
            raise ValueError(
                f"T_ACUTE_LETHAL ({self.T_ACUTE_LETHAL}) must be > T_AVOID ({self.T_AVOID})"
            )
        if self.cmax_A <= 0:
            raise ValueError(f"cmax_A must be > 0, got {self.cmax_A}")
        if self.LW_a <= 0 or self.LW_b <= 0:
            raise ValueError("Length-weight coefficients must be positive")
        if not (0 < self.MASS_FLOOR_FRACTION <= 1):
            raise ValueError(
                f"MASS_FLOOR_FRACTION must be in (0, 1], got {self.MASS_FLOOR_FRACTION}"
            )
        if self.ED_TISSUE <= 0 or self.ED_MORTAL <= 0:
            raise ValueError("ED_TISSUE and ED_MORTAL must be > 0")
        if not (0 <= self.spawn_window_start_day <= 365):
            raise ValueError("spawn_window_start_day must be 0-365")

    @property
    def T_MAX(self) -> float:
        """Back-compat for legacy callers expecting a single thermal threshold.

        Returns T_AVOID (the softer, behavioral threshold). Callers needing
        the hard acute-mortality threshold must read T_ACUTE_LETHAL explicitly.
        """
        return self.T_AVOID


def load_baltic_species_config(path: str | Path) -> BalticBioParams:
    """Load the canonical baltic_salmon_species.yaml into BalticBioParams.

    YAML schema (minimum):

        species:
          BalticAtlanticSalmon:
            cmax_A: 0.303
            T_OPT: 16.0
            # ... other fields from BalticBioParams

    Unknown keys are silently filtered (the inSTREAM source YAML has
    additional fields HexSim doesn't consume).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("species", {}).get("BalticAtlanticSalmon")
    if block is None:
        raise ValueError(
            f"{path}: missing 'species.BalticAtlanticSalmon' block"
        )
    # Filter to fields BalticBioParams knows about; extra keys tolerated.
    known = {f.name for f in BalticBioParams.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in block.items() if k in known}
    return BalticBioParams(**kwargs)
