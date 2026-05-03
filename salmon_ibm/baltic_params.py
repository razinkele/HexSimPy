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
from salmon_ibm.sea_age import VALID_SEA_AGES


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

    # Pre-spawn skip probability (C3.1). Bernoulli gate on reproducers
    # before Poisson clutch sampling; wild=0.0 (always spawns), hatchery
    # may divert via hatchery_overrides.pre_spawn_skip_prob.
    # Empirical anchor: Bouchard et al. 2022 (doi:10.1111/eva.13374)
    # — captive-bred Atlantic salmon RRS 0.65-0.80, "fewer mating events,
    # not smaller clutches" → skip-rate model intervention is the
    # right shape (matches Bouchard's mechanistic finding).
    pre_spawn_skip_prob: float = 0.0

    # Sea-age distribution (C3.2). Trinomial over VALID_SEA_AGES.
    # Wild baseline anchored to WGBAST annual-report ranges.
    # Validated in __post_init__: keys ⊂ {1,2,3} (rejects bool AND
    # numpy ints via `type(k) is int`), positive floats, sum to 1.0
    # within 1e-6.
    sea_age_distribution: dict[int, float] = field(
        default_factory=lambda: {1: 0.35, 2: 0.55, 3: 0.10}
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
        if not (0.0 <= self.pre_spawn_skip_prob <= 1.0):
            raise ValueError(
                f"pre_spawn_skip_prob must be in [0, 1], got "
                f"{self.pre_spawn_skip_prob!r}"
            )
        # C3.2: sea_age_distribution validation.
        if not self.sea_age_distribution:
            raise ValueError("sea_age_distribution must be non-empty")
        for k, v in self.sea_age_distribution.items():
            # type(k) is int — rejects bool (type(True) is bool) and
            # numpy integer types (type(np.int64(1)) is np.int64).
            # The error message includes type(k).__name__ so users see
            # "got numpy.int64; pass int(k)" rather than a confusing
            # "must be in {1,2,3}" when they passed numerically valid keys.
            if type(k) is not int:
                raise ValueError(
                    f"sea_age_distribution keys must be Python int "
                    f"(got {type(k).__module__}.{type(k).__name__} "
                    f"for key {k!r}); cast via int(k) at the call site"
                )
            if k not in VALID_SEA_AGES:
                raise ValueError(
                    f"sea_age_distribution keys must be in {{1, 2, 3}}, "
                    f"got {k!r}"
                )
            if not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(
                    f"sea_age_distribution values must be positive floats, "
                    f"got {k}: {v!r}"
                )
        total = sum(self.sea_age_distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"sea_age_distribution must sum to 1.0, got {total!r}"
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

    Override semantics by field:
    - activity_by_behavior (C2): shallow-merge over wild base — empirical
      reports per-behavior delta tables.
    - pre_spawn_skip_prob (C3.1): scalar replacement.
    - sea_age_distribution (C3.2): FULL replacement — empirical reports
      whole trinomials (Jokikokko 2004); a partial merge that happens to
      produce identity-with-wild would silent-no-op, hiding the
      intervention.

    Sub-keys of activity_by_behavior must be valid Behavior enum values
    (0-4). Sub-keys of sea_age_distribution must be exactly {1, 2, 3}
    (the override path applies the same `type(k) is int` rejection as
    BalticBioParams.__post_init__ to prevent int(True) == 1 from
    silently passing).

    Post-replace `__post_init__` re-validation is the merge-output safety
    net — never bypass `dataclasses.replace`.

    See docs/superpowers/specs/2026-05-03-hatchery-c3.2-seaage-design.md.
    """
    from salmon_ibm.agents import Behavior  # local import to avoid cycle

    ALLOWED_OVERRIDE_KEYS = {
        "activity_by_behavior",
        "pre_spawn_skip_prob",
        "sea_age_distribution",
    }
    SCALAR_OVERRIDE_FIELDS = {"pre_spawn_skip_prob"}
    VALID_BEHAVIORS = {int(b) for b in Behavior}

    unknown = set(overrides) - ALLOWED_OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            f"hatchery_overrides supports only "
            f"{sorted(ALLOWED_OVERRIDE_KEYS)} in C3.2; unsupported keys: "
            f"{sorted(unknown)}"
        )

    # --- activity_by_behavior (C2: shallow-merge) -------------------------
    activity_overrides_raw = overrides.get("activity_by_behavior", {})
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
    merged_activity = {**wild_params.activity_by_behavior, **activity_overrides}

    # --- sea_age_distribution (C3.2: FULL replacement) --------------------
    sea_age_kwargs: dict = {}
    if "sea_age_distribution" in overrides:
        sea_age_raw = overrides["sea_age_distribution"]
        # Reject bool / numpy-int keys BEFORE int() coercion. Mirrors
        # BalticBioParams.__post_init__'s `type(k) is int` discipline.
        for k in sea_age_raw:
            if type(k) is not int:
                raise ValueError(
                    f"hatchery_overrides.sea_age_distribution keys must be "
                    f"Python int (got "
                    f"{type(k).__module__}.{type(k).__name__} for key "
                    f"{k!r}); cast via int(k) at the call site"
                )
        # Coerce values to float (PyYAML may emit int).
        try:
            sea_age_coerced = {int(k): float(v) for k, v in sea_age_raw.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"hatchery_overrides.sea_age_distribution values must be "
                f"numeric; got {sea_age_raw!r}"
            ) from exc
        # Full replacement — keys must be exactly {1, 2, 3}.
        if set(sea_age_coerced.keys()) != {1, 2, 3}:
            raise ValueError(
                f"hatchery_overrides.sea_age_distribution keys must be "
                f"exactly {{1, 2, 3}} (full replacement; partial merge "
                f"would silent-no-op against the wild baseline); got "
                f"keys {sorted(sea_age_coerced.keys())!r}"
            )
        sea_age_kwargs["sea_age_distribution"] = sea_age_coerced

    # --- scalar fields (C3.1+) --------------------------------------------
    scalar_kwargs = {
        k: v for k, v in overrides.items() if k in SCALAR_OVERRIDE_FIELDS
    }

    # dataclasses.replace re-runs __post_init__ on the new instance,
    # catching any merge-output that violates the field invariants.
    return dataclasses.replace(
        wild_params,
        activity_by_behavior=merged_activity,
        **sea_age_kwargs,
        **scalar_kwargs,
    )


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
