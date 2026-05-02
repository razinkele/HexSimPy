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

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

from salmon_ibm.bioenergetics import BioParams  # for BalticSpeciesConfig.wild type widening (used in Task 3)


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
        if not self.activity_by_behavior:
            raise ValueError("activity_by_behavior must be non-empty")
        for k, v in self.activity_by_behavior.items():
            if not isinstance(k, int) or k < 0:
                raise ValueError(
                    f"activity_by_behavior keys must be non-negative ints, got {k!r}"
                )
            if not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(
                    f"activity_by_behavior values must be positive floats, "
                    f"got {k}: {v!r}"
                )

    @property
    def T_MAX(self) -> float:
        """Back-compat for legacy callers expecting a single thermal threshold.

        Returns T_AVOID (the softer, behavioral threshold). Callers needing
        the hard acute-mortality threshold must read T_ACUTE_LETHAL explicitly.
        """
        return self.T_AVOID


@dataclass(frozen=True)
class HatcheryDispatch:
    """Bundle hatchery params + their derived activity LUT atomically.

    Holding params and LUT separately as paired nullables on Simulation
    invites desync (one rebuilt, one stale). This bundle makes the
    invariant `params is None ↔ lut is None` structurally impossible
    to violate, and gives callers a single nullable to guard on
    (`if landscape.get('hatchery_dispatch') is None: ...`).
    """
    params: BalticBioParams
    activity_lut: np.ndarray


class BalticSpeciesConfig(NamedTuple):
    """Loaded species config — wild + optional hatchery override.

    Always returned by load_baltic_species_config(); legacy non-Baltic
    path returns BalticSpeciesConfig(wild=plain_BioParams, hatchery=None)
    so callers don't need isinstance branching. The `wild` field is
    typed as `BioParams | BalticBioParams` because the legacy path
    wraps a plain `BioParams`.
    """
    wild: BioParams | BalticBioParams
    hatchery: BalticBioParams | None


def _apply_hatchery_overrides(
    wild_params: BalticBioParams,
    overrides: dict,
) -> BalticBioParams:
    """Build a hatchery BalticBioParams by overlaying overrides on wild.

    For C2, the only allowed override key is `activity_by_behavior`.
    Sub-keys must be valid Behavior enum values (0-4). Non-numeric or
    out-of-range sub-keys raise ValueError at load time.

    See docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md.
    """
    from salmon_ibm.agents import Behavior  # local import to avoid cycle

    ALLOWED_OVERRIDE_KEYS = {"activity_by_behavior"}
    VALID_BEHAVIORS = {int(b) for b in Behavior}  # {0, 1, 2, 3, 4}

    unknown = set(overrides) - ALLOWED_OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            f"hatchery_overrides supports only "
            f"{sorted(ALLOWED_OVERRIDE_KEYS)} in C2; unsupported keys: "
            f"{sorted(unknown)}"
        )

    activity_overrides_raw = overrides.get("activity_by_behavior", {})
    # Coerce YAML string keys to int (PyYAML may emit '1' rather than 1).
    # Wrap in try/except so non-numeric keys produce an actionable message
    # rather than a bare 'invalid literal for int()' traceback.
    try:
        activity_overrides = {
            int(k): float(v) for k, v in activity_overrides_raw.items()
        }
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"hatchery_overrides.activity_by_behavior keys must be integers "
            f"(Behavior enum values 0-4); got non-integer key in "
            f"{activity_overrides_raw!r}"
        ) from exc

    invalid_keys = set(activity_overrides) - VALID_BEHAVIORS
    if invalid_keys:
        raise ValueError(
            f"hatchery_overrides.activity_by_behavior keys must be valid "
            f"Behavior enum values {sorted(VALID_BEHAVIORS)}; got invalid: "
            f"{sorted(invalid_keys)}"
        )

    # Shallow-merge over wild base: missing keys keep wild values.
    merged_dict = {**wild_params.activity_by_behavior, **activity_overrides}
    # dataclasses.replace re-runs __post_init__ validation on the merged
    # instance, catching e.g. negative override values.
    return dataclasses.replace(wild_params, activity_by_behavior=merged_dict)


def load_baltic_species_config(path: str | Path) -> BalticSpeciesConfig:
    """Load the canonical baltic_salmon_species.yaml into BalticSpeciesConfig.

    Always returns a BalticSpeciesConfig NamedTuple. If the YAML's
    species.BalticAtlanticSalmon block contains a hatchery_overrides:
    sub-block, .hatchery is populated; otherwise .hatchery is None.

    YAML schema (minimum):

        species:
          BalticAtlanticSalmon:
            cmax_A: 0.303
            activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
            # optional:
            hatchery_overrides:
              activity_by_behavior:
                1: 1.5
                3: 1.875

    Unknown top-level keys in BalticAtlanticSalmon are silently filtered
    (legacy tolerance). Unknown keys in hatchery_overrides RAISE
    ValueError (strict — typos there are scientifically dangerous).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("species", {}).get("BalticAtlanticSalmon")
    if block is None:
        raise ValueError(
            f"{path}: missing 'species.BalticAtlanticSalmon' block"
        )
    # Pop hatchery_overrides BEFORE the known-field filter so it doesn't
    # get silently dropped.
    hatchery_overrides = block.pop("hatchery_overrides", None)
    # Filter to fields BalticBioParams knows about; extra keys tolerated.
    known = {f.name for f in BalticBioParams.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in block.items() if k in known}
    wild = BalticBioParams(**kwargs)

    if hatchery_overrides is None:
        return BalticSpeciesConfig(wild=wild, hatchery=None)
    hatchery = _apply_hatchery_overrides(wild, hatchery_overrides)
    return BalticSpeciesConfig(wild=wild, hatchery=hatchery)
