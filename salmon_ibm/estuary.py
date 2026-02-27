"""Estuarine extensions: salinity cost, DO avoidance, seiche pause."""
from __future__ import annotations

import numpy as np


def salinity_cost(
    salinity: np.ndarray,
    S_opt: float = 0.5,
    S_tol: float = 6.0,
    k: float = 0.6,
) -> np.ndarray:
    excess = np.maximum(salinity - (S_opt + S_tol), 0.0)
    return 1.0 + k * excess


DO_OK = 0
DO_ESCAPE = 1
DO_LETHAL = 2


def do_override(
    do_mg_l: np.ndarray,
    lethal: float = 2.0,
    high: float = 4.0,
) -> np.ndarray:
    result = np.full(len(do_mg_l), DO_OK, dtype=int)
    result[do_mg_l < high] = DO_ESCAPE
    result[do_mg_l < lethal] = DO_LETHAL
    return result


def seiche_pause(
    dSSH_dt: np.ndarray,
    thresh: float = 0.02,
) -> np.ndarray:
    return np.abs(dSSH_dt) > thresh
