"""Estuarine extensions: salinity cost, DO avoidance, seiche pause."""

from __future__ import annotations

import numpy as np


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


from enum import IntEnum


class DOState(IntEnum):
    OK = 0
    ESCAPE = 1
    LETHAL = 2


DO_OK = DOState.OK
DO_ESCAPE = DOState.ESCAPE
DO_LETHAL = DOState.LETHAL


def do_override(
    do_mg_l: np.ndarray,
    lethal: float = 2.0,
    high: float = 4.0,
) -> np.ndarray:
    if lethal > high:
        raise ValueError(
            f"lethal threshold ({lethal}) must be <= high threshold ({high})"
        )
    result = np.full(len(do_mg_l), DO_OK, dtype=int)
    # NaN-safe comparisons: NaN < X is False in numpy, so NaN gets DO_OK
    # This is already the correct behavior, but make it explicit
    valid = ~np.isnan(do_mg_l)
    result[valid & (do_mg_l < high)] = DO_ESCAPE
    result[valid & (do_mg_l < lethal)] = DO_LETHAL
    return result


def seiche_pause(
    dSSH_dt: np.ndarray,
    thresh: float = 0.02,
) -> np.ndarray:
    return np.abs(dSSH_dt) > thresh
