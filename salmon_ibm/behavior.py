"""Behavioral decision table and overrides (Snyder et al. 2019)."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

from salmon_ibm.agents import Behavior


@dataclass
class BehaviorParams:
    temp_bins: list[float] = field(default_factory=lambda: [16.0, 18.0, 20.0])
    time_bins: list[float] = field(default_factory=lambda: [360, 720])
    p_table: np.ndarray | None = None
    max_cwr_hours: int = 48
    avoid_cwr_cooldown_h: int = 12
    max_dist_to_cwr: float = 5000.0

    def __post_init__(self):
        # Cache a contiguous copy of p_table so pick_behaviors() doesn't call
        # np.ascontiguousarray(...) every step (the table never changes).
        if self.p_table is not None:
            self._p_table_c = np.ascontiguousarray(self.p_table)
        else:
            self._p_table_c = None

    @classmethod
    def defaults(cls):
        p = np.array([
            # <15 days to spawn — urgent: strongly UPSTREAM
            [[0.00, 0.20, 0.00, 0.80, 0.00],
             [0.00, 0.10, 0.20, 0.70, 0.00],
             [0.00, 0.00, 0.50, 0.50, 0.00],
             [0.00, 0.00, 0.60, 0.40, 0.00]],
            # 15-30 days — moderate urgency
            [[0.20, 0.20, 0.00, 0.60, 0.00],
             [0.00, 0.20, 0.30, 0.50, 0.00],
             [0.00, 0.00, 0.40, 0.40, 0.20],
             [0.00, 0.00, 0.70, 0.00, 0.30]],
            # >30 days — relaxed: more HOLD and RANDOM
            [[0.60, 0.10, 0.00, 0.30, 0.00],
             [0.40, 0.00, 0.20, 0.40, 0.00],
             [0.30, 0.00, 0.50, 0.20, 0.00],
             [0.20, 0.00, 0.80, 0.00, 0.00]],
        ])
        return cls(p_table=p)


@njit(cache=True, parallel=True)
def _pick_behaviors_numba(temp_idx, time_idx, p_table, rand_vals):
    n = len(temp_idx)
    behaviors = np.empty(n, dtype=np.int32)
    for i in prange(n):
        ti = time_idx[i]
        te = temp_idx[i]
        r = rand_vals[i]
        cumsum = 0.0
        chosen = 4
        for b in range(5):
            cumsum += p_table[ti, te, b]
            if r < cumsum:
                chosen = b
                break
        behaviors[i] = chosen
    return behaviors


def pick_behaviors(t3h_mean, hours_to_spawn, params, seed=None):
    rng = np.random.default_rng(seed)
    n = len(t3h_mean)
    temp_idx = np.clip(np.digitize(t3h_mean, params.temp_bins),
                       0, params.p_table.shape[1] - 1)
    time_idx = np.clip(np.digitize(hours_to_spawn, params.time_bins),
                       0, params.p_table.shape[0] - 1)

    if HAS_NUMBA:
        rand_vals = rng.random(n)
        # Use cached contiguous view; falls back to explicit conversion if params
        # was mutated after construction (defensive, not expected).
        p_table_c = getattr(params, "_p_table_c", None)
        if p_table_c is None:
            p_table_c = np.ascontiguousarray(params.p_table)
        return _pick_behaviors_numba(
            temp_idx.astype(np.int32), time_idx.astype(np.int32),
            p_table_c, rand_vals,
        )
    else:
        behaviors = np.empty(n, dtype=int)
        for ti in range(params.p_table.shape[0]):
            for te in range(params.p_table.shape[1]):
                mask = (time_idx == ti) & (temp_idx == te)
                count = mask.sum()
                if count > 0:
                    probs = params.p_table[ti, te]
                    behaviors[mask] = rng.choice(5, size=count, p=probs)
        return behaviors


def apply_overrides(pool, params):
    beh = pool.behavior.copy()
    first_move = pool.steps == 0
    beh[first_move] = Behavior.UPSTREAM
    cwr_exceeded = pool.cwr_hours > params.max_cwr_hours
    beh[cwr_exceeded] = Behavior.UPSTREAM
    cooldown_active = pool.hours_since_cwr < params.avoid_cwr_cooldown_h
    to_cwr = beh == Behavior.TO_CWR
    beh[to_cwr & cooldown_active] = Behavior.UPSTREAM
    return beh
