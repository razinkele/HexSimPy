"""Wisconsin Bioenergetics Model — hourly budget for non-feeding migrants."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


OXY_CAL_J_PER_GO2 = 13_560.0


@dataclass
class BioParams:
    RA: float = 0.00264
    RB: float = -0.217
    RQ: float = 0.06818
    ED_MORTAL: float = 4.0
    T_OPT: float = 16.0  # optimal temperature for respiration (Macnaughton 2019)
    T_MAX: float = 26.0  # upper lethal temperature
    ED_TISSUE: float = 5.0  # energy density of catabolized tissue (kJ/g)
    activity_by_behavior: dict[int, float] = field(
        default_factory=lambda: {
            0: 1.0,
            1: 1.2,
            2: 0.8,
            3: 1.5,
            4: 1.0,
        }
    )


def hourly_respiration(
    mass_g: np.ndarray,
    temperature_c: np.ndarray,
    activity_mult: np.ndarray,
    params: BioParams,
) -> np.ndarray:
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
    r_hourly = (
        hourly_respiration(mass_g, temperature_c, activity_mult, params) * salinity_cost
    )
    e_total_j = ed_kJ_g * 1000.0 * mass_g
    new_e_total_j = np.maximum(e_total_j - r_hourly, 0.0)
    # Proportional mass loss: mass shrinks in proportion to energy loss.
    # This keeps ED constant under pure respiration (standard Wisconsin model).
    energy_fraction = np.where(e_total_j > 0, new_e_total_j / e_total_j, 0.0)
    new_mass = mass_g * energy_fraction
    # Floor at 50% original mass to prevent numerical collapse
    new_mass = np.maximum(new_mass, mass_g * 0.5)
    new_ed = np.where(new_mass > 0, new_e_total_j / (new_mass * 1000.0), 0.0)
    dead = new_ed < params.ED_MORTAL
    return new_ed, dead, new_mass
