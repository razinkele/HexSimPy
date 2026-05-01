"""Wisconsin Bioenergetics Model — hourly budget for non-feeding migrants."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from salmon_ibm.origin import ORIGIN_HATCHERY


OXY_CAL_J_PER_GO2 = 13_560.0


@dataclass
class BioParams:
    RA: float = 0.00264
    RB: float = -0.217
    RQ: float = 0.06818
    ED_MORTAL: float = 4.0
    T_OPT: float = 16.0  # optimal temperature for respiration (Macnaughton 2019)
    T_MAX: float = 26.0  # upper lethal temperature
    ED_TISSUE: float = 36.0  # lipid catabolism (Brett 1995 / Breck 2008)
    MASS_FLOOR_FRACTION: float = 0.5  # minimum mass as fraction of current mass
    activity_by_behavior: dict[int, float] = field(
        default_factory=lambda: {
            0: 1.0,
            1: 1.2,
            2: 0.8,
            3: 1.5,
            4: 1.0,
        }
    )

    def __post_init__(self):
        if self.RA <= 0:
            raise ValueError("RA must be > 0")
        if self.RQ <= 0:
            raise ValueError("RQ must be > 0")
        if self.T_MAX <= self.T_OPT:
            raise ValueError(f"T_MAX ({self.T_MAX}) must be > T_OPT ({self.T_OPT})")
        if not (0 < self.MASS_FLOOR_FRACTION <= 1):
            raise ValueError(
                f"MASS_FLOOR_FRACTION must be in (0, 1], got {self.MASS_FLOOR_FRACTION}"
            )
        if self.ED_MORTAL <= 0:
            raise ValueError("ED_MORTAL must be > 0")
        if self.ED_TISSUE <= 0:
            raise ValueError("ED_TISSUE must be > 0")


def hourly_respiration(
    mass_g: np.ndarray,
    temperature_c: np.ndarray,
    activity_mult: np.ndarray,
    params: BioParams,
) -> np.ndarray:
    mass_g = np.maximum(mass_g, 1e-6)
    r_daily = (
        params.RA
        * np.power(mass_g, params.RB)
        * np.exp(params.RQ * temperature_c)
        * activity_mult
    )
    return r_daily * OXY_CAL_J_PER_GO2 * mass_g / 24.0


def update_energy(
    ed_kJ_g: np.ndarray,
    mass_g: np.ndarray,
    temperature_c: np.ndarray,
    activity_mult: np.ndarray,
    salinity_cost: np.ndarray,
    params: BioParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Lipid-first catabolism (Option A; Brett 1995 / Breck 2008). Mass lost
    # per joule respired is set by ED_TISSUE (energy density of catabolized
    # tissue, ~36 kJ/g for pure lipid). Because ED_TISSUE > whole-body ED,
    # mass declines slower than total energy → remaining body's mean ED
    # smoothly drops toward ED_MORTAL. The proportional-mass formula
    # previously here kept ED flat by construction (only the mass-floor
    # engaged it, abruptly), which masked starvation mortality.
    r_hourly = (
        hourly_respiration(mass_g, temperature_c, activity_mult, params) * salinity_cost
    )
    e_total_j = ed_kJ_g * 1000.0 * mass_g
    new_e_total_j = np.maximum(e_total_j - r_hourly, 0.0)
    actual_loss_j = e_total_j - new_e_total_j
    mass_lost_g = actual_loss_j / (params.ED_TISSUE * 1000.0)
    new_mass = np.maximum(mass_g - mass_lost_g, mass_g * params.MASS_FLOOR_FRACTION)
    new_ed = np.where(new_mass > 0, new_e_total_j / (new_mass * 1000.0), 0.0)
    dead = new_ed < params.ED_MORTAL
    return new_ed, dead, new_mass


def origin_aware_activity_mult(
    behavior: np.ndarray,
    origin: np.ndarray,
    lut_wild: np.ndarray,
    lut_hatch: np.ndarray | None,
) -> np.ndarray:
    """Per-agent activity multiplier with origin-aware dispatch.

    Returns lut_wild[behavior] when lut_hatch is None — graceful for
    pre-C2 paths, test fixtures, and scenarios without hatchery
    overrides. Otherwise dispatches per-agent via origin column:
    HATCHERY agents read from lut_hatch, WILD from lut_wild.

    See docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md.
    """
    if lut_hatch is None:
        return lut_wild[behavior]
    return np.where(
        origin == ORIGIN_HATCHERY,
        lut_hatch[behavior],
        lut_wild[behavior],
    )
