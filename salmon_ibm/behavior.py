"""Behavioral decision table and overrides (Snyder et al. 2019)."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from salmon_ibm.agents import Behavior


@dataclass
class BehaviorParams:
    temp_bins: list[float] = field(default_factory=lambda: [16.0, 18.0, 20.0])
    time_bins: list[float] = field(default_factory=lambda: [360, 720])
    p_table: np.ndarray | None = None
    max_cwr_hours: int = 48
    avoid_cwr_cooldown_h: int = 12
    max_dist_to_cwr: float = 5000.0

    @classmethod
    def defaults(cls):
        p = np.array([
            # >30 days to spawn
            [[0.60, 0.10, 0.00, 0.30, 0.00],
             [0.40, 0.00, 0.20, 0.40, 0.00],
             [0.30, 0.00, 0.50, 0.20, 0.00],
             [0.20, 0.00, 0.80, 0.00, 0.00]],
            # 15-30 days
            [[0.20, 0.20, 0.00, 0.60, 0.00],
             [0.00, 0.20, 0.30, 0.50, 0.00],
             [0.00, 0.00, 0.40, 0.40, 0.20],
             [0.00, 0.00, 0.70, 0.00, 0.30]],
            # <15 days
            [[0.00, 0.20, 0.00, 0.80, 0.00],
             [0.00, 0.10, 0.20, 0.70, 0.00],
             [0.00, 0.00, 0.50, 0.50, 0.00],
             [0.00, 0.00, 0.60, 0.40, 0.00]],
        ])
        return cls(p_table=p)


def pick_behaviors(t3h_mean, hours_to_spawn, params, seed=None):
    rng = np.random.default_rng(seed)
    n = len(t3h_mean)
    temp_idx = np.digitize(t3h_mean, params.temp_bins)
    time_idx = np.digitize(hours_to_spawn, params.time_bins)
    behaviors = np.empty(n, dtype=int)
    for i in range(n):
        ti = int(np.clip(time_idx[i], 0, params.p_table.shape[0] - 1))
        te = int(np.clip(temp_idx[i], 0, params.p_table.shape[1] - 1))
        probs = params.p_table[ti, te]
        behaviors[i] = rng.choice(5, p=probs)
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
