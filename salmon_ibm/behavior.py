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


def pick_behaviors(t3h_mean, hours_to_spawn, params, seed=None):
    rng = np.random.default_rng(seed)
    n = len(t3h_mean)
    temp_idx = np.clip(np.digitize(t3h_mean, params.temp_bins),
                       0, params.p_table.shape[1] - 1)
    time_idx = np.clip(np.digitize(hours_to_spawn, params.time_bins),
                       0, params.p_table.shape[0] - 1)
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
