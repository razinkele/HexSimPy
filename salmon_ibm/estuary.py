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

    Salinity-cost parameters cite Wilson 2002 (blood iso-osmotic point
    for S. salar) and Brett & Groves 1979 (hyper/hypo cost slopes for
    euryhaline salmonids).

    Units:
        do_lethal, do_high: mg/L
        salinity_iso_osmotic: PSU (blood iso-osmotic point)
        salinity_hyper_cost: dimensionless multiplier slope (above iso)
        salinity_hypo_cost: dimensionless multiplier slope (below iso)
        seiche_threshold_m_per_s: m/s (|dSSH/dt|)
    """
    do_lethal: float = 3.0
    do_high: float = 5.5
    salinity_iso_osmotic: float = 10.0   # Wilson 2002 — S. salar blood iso-osmotic ~9-12 PSU
    salinity_hyper_cost: float = 0.30    # Brett & Groves 1979 — verified value from Task 1
    salinity_hypo_cost: float = 0.05     # Brett & Groves 1979 — verified value from Task 1
    seiche_threshold_m_per_s: float = 0.02

    def __post_init__(self):
        if not (0 < self.salinity_iso_osmotic < 35):
            raise ValueError(
                f"salinity_iso_osmotic must be in (0, 35) PSU, "
                f"got {self.salinity_iso_osmotic}"
            )
        if not (0 <= self.salinity_hyper_cost <= 1):
            raise ValueError(
                f"salinity_hyper_cost must be in [0, 1], "
                f"got {self.salinity_hyper_cost}"
            )
        if not (0 <= self.salinity_hypo_cost <= 1):
            raise ValueError(
                f"salinity_hypo_cost must be in [0, 1], "
                f"got {self.salinity_hypo_cost}"
            )


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
    params: EstuaryParams,
) -> np.ndarray:
    """Osmoregulation cost multiplier on respiration for *Salmo salar*.

    Linear-with-anchors function with separate slopes for hyper-osmotic
    (above the blood iso-osmotic point) and hypo-osmotic (below iso)
    stress.

    Returns: multiplier ≥ 1.0 array with same shape as `salinity`;
    equals 1.0 exactly at salinity == params.salinity_iso_osmotic.

    NaN inputs are treated as iso (cost 1.0).

    Citations: Wilson 2002 for iso-osmotic point; Brett & Groves 1979
    for hyper/hypo cost magnitudes.
    """
    iso = params.salinity_iso_osmotic
    hyper = params.salinity_hyper_cost
    hypo = params.salinity_hypo_cost
    safe = np.where(np.isnan(salinity), iso, salinity)  # NaN → iso (cost 1.0)
    s = np.clip(safe, 0.0, 35.0)
    above = np.maximum(s - iso, 0.0) / max(35.0 - iso, 1.0)
    below = np.maximum(iso - s, 0.0) / max(iso, 1.0)
    return 1.0 + hyper * above + hypo * below


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
