"""Fish agent state: FishAgent (OOP view) + AgentPool (vectorized arrays)."""

from __future__ import annotations

from enum import IntEnum
import numpy as np


class Behavior(IntEnum):
    HOLD = 0
    RANDOM = 1
    TO_CWR = 2
    UPSTREAM = 3
    DOWNSTREAM = 4


class AgentPool:
    """Vectorized storage for all fish agents (structure-of-arrays)."""

    # Array fields that must be compacted/extended together
    ARRAY_FIELDS = (
        "tri_idx",
        "mass_g",
        "ed_kJ_g",
        "target_spawn_hour",
        "behavior",
        "cwr_hours",
        "hours_since_cwr",
        "steps",
        "alive",
        "arrived",
        "temp_history",
    )

    def __init__(
        self,
        n: int,
        start_tri: int | np.ndarray,
        rng_seed: int | None = None,
        mass_mean: float = 3500.0,
        mass_std: float = 500.0,
        ed_init: float = 6.5,
        spawn_hours_mean: float = 720.0,
        spawn_hours_std: float = 168.0,
    ):
        self.n = n
        rng = np.random.default_rng(rng_seed)

        if isinstance(start_tri, (int, np.integer)):
            self.tri_idx = np.full(n, start_tri, dtype=int)
        else:
            self.tri_idx = np.asarray(start_tri, dtype=int)

        self.mass_g = np.clip(
            rng.normal(mass_mean, mass_std, n), mass_mean * 0.5, mass_mean * 1.5
        )
        self.ed_kJ_g = np.full(n, ed_init)
        self.target_spawn_hour = np.clip(
            rng.normal(spawn_hours_mean, spawn_hours_std, n).astype(int), 1, None
        )
        self.behavior = np.full(n, Behavior.HOLD, dtype=int)
        self.cwr_hours = np.zeros(n, dtype=int)
        self.hours_since_cwr = np.full(n, 999, dtype=int)
        self.steps = np.zeros(n, dtype=int)
        self.alive = np.ones(n, dtype=bool)
        self.arrived = np.zeros(n, dtype=bool)
        self.temp_history = np.full((n, 3), 15.0)

        # Optional general-purpose state (Phase 1a)
        self.accumulators = None  # AccumulatorManager | None
        self.traits = None  # TraitManager | None

        # Invariant: every declared array field must be initialized as an ndarray.
        # Guards against adding a field to ARRAY_FIELDS but forgetting to init it.
        missing = [
            f for f in self.ARRAY_FIELDS
            if not isinstance(getattr(self, f, None), np.ndarray)
        ]
        assert not missing, (
            f"AgentPool.__init__ missed ARRAY_FIELDS: {missing}. "
            f"Add initialization or update ARRAY_FIELDS."
        )

    def get_agent(self, idx: int) -> "FishAgent":
        return FishAgent(self, idx)

    def t3h_mean(self) -> np.ndarray:
        # Inline column sum avoids numpy.ufunc.reduce dispatch overhead
        # on the fixed (N, 3) shape — ~3x faster than .mean(axis=1).
        h = self.temp_history
        return (h[:, 0] + h[:, 1] + h[:, 2]) * (1.0 / 3.0)

    def push_temperature(self, temps: np.ndarray):
        self.temp_history[:, :-1] = self.temp_history[:, 1:]
        self.temp_history[:, -1] = temps


class FishAgent:
    """OOP view into a single agent within an AgentPool. Zero-copy."""

    def __init__(self, pool: AgentPool, idx: int):
        self._pool = pool
        self._idx = idx

    @property
    def id(self) -> int:
        return self._idx

    @property
    def tri_idx(self) -> int:
        return int(self._pool.tri_idx[self._idx])

    @tri_idx.setter
    def tri_idx(self, v: int):
        self._pool.tri_idx[self._idx] = v

    @property
    def mass_g(self) -> float:
        return float(self._pool.mass_g[self._idx])

    @property
    def ed_kJ_g(self) -> float:
        return float(self._pool.ed_kJ_g[self._idx])

    @ed_kJ_g.setter
    def ed_kJ_g(self, v: float):
        self._pool.ed_kJ_g[self._idx] = v

    @property
    def behavior(self) -> int:
        return int(self._pool.behavior[self._idx])

    @behavior.setter
    def behavior(self, v: int):
        self._pool.behavior[self._idx] = v

    @property
    def alive(self) -> bool:
        return bool(self._pool.alive[self._idx])

    @property
    def arrived(self) -> bool:
        return bool(self._pool.arrived[self._idx])

    @property
    def steps(self) -> int:
        return int(self._pool.steps[self._idx])

    @property
    def cwr_hours(self) -> int:
        return int(self._pool.cwr_hours[self._idx])

    @property
    def hours_since_cwr(self) -> int:
        return int(self._pool.hours_since_cwr[self._idx])
