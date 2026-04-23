"""Estuarine extensions: salinity cost, DO avoidance, seiche pause."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


@dataclass
class EstuaryParams:
    """Parameters for estuarine stressors.

    DO defaults cite Liland et al. (2024) for *Salmo salar* performance:
    reduced growth below ~60% O2 saturation (~5.5 mg/L at 15°C). Acute
    mortality approaches ~3 mg/L (Davis 1975 criteria).

    Units:
        do_lethal, do_high: mg/L
        s_opt, s_tol: PSU
        seiche_threshold: m/s (|dSSH/dt|)
    """
    do_lethal: float = 3.0
    do_high: float = 5.5
    s_opt: float = 0.5
    s_tol: float = 6.0
    seiche_threshold_m_per_s: float = 0.02


def validate_do_field_units(do_field: np.ndarray) -> None:
    """Sanity-check that a DO field is in mg/L, not mmol/m^3.

    CMEMS Baltic products often ship DO in mmol/m^3 with typical values 150-400.
    mg/L values should be in [0, 20]. A field with max > 30 almost certainly
    needs unit conversion: mg/L = mmol/m^3 * 32 / 1000.

    Raises ValueError on suspected unit mismatch.
    """
    finite = do_field[np.isfinite(do_field)]
    if finite.size and finite.max() > 30.0:
        raise ValueError(
            f"DO field max={finite.max():.1f} suggests mmol/m^3 input. "
            f"Expected mg/L (typical range 0-20). "
            f"Convert via: mg/L = mmol/m^3 * 32 / 1000."
        )


def salinity_cost(
    salinity: np.ndarray,
    S_opt: float = 0.5,
    S_tol: float = 6.0,
    k: float = 0.6,
    max_cost: float = 5.0,
) -> np.ndarray:
    safe_sal = np.where(np.isnan(salinity), 0.0, salinity)
    excess = np.maximum(safe_sal - (S_opt + S_tol), 0.0)
    return np.minimum(1.0 + k * excess, max_cost)


class DOState(IntEnum):
    OK = 0
    ESCAPE = 1
    LETHAL = 2


DO_OK = DOState.OK
DO_ESCAPE = DOState.ESCAPE
DO_LETHAL = DOState.LETHAL


def do_override(
    do_mg_l: np.ndarray,
    lethal: float = 3.0,
    high: float = 5.5,
) -> np.ndarray:
    """Map per-cell DO in mg/L to DOState.

    Default thresholds from Liland et al. 2024 (*Salmo salar*):
      high=5.5 mg/L (sub-optimal / avoidance)
      lethal=3.0 mg/L (acute mortality, per Davis 1975).
    """
    if lethal > high:
        raise ValueError(
            f"lethal threshold ({lethal}) must be <= high threshold ({high})"
        )
    result = np.full(len(do_mg_l), DO_OK, dtype=int)
    # NaN-safe comparisons: NaN < X is False in numpy, so NaN gets DO_OK
    valid = ~np.isnan(do_mg_l)
    result[valid & (do_mg_l < high)] = DO_ESCAPE
    result[valid & (do_mg_l < lethal)] = DO_LETHAL
    return result


def seiche_pause(
    dSSH_dt: np.ndarray,
    thresh: float = 0.02,
) -> np.ndarray:
    return np.abs(dSSH_dt) > thresh
