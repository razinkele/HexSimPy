"""Track logging and diagnostics output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from salmon_ibm.agents import AgentPool

if TYPE_CHECKING:
    from salmon_ibm.population import Population


class OutputLogger:
    def __init__(self, path: str, centroids: np.ndarray):
        self.path = path
        self.centroids = centroids
        self._times: list[np.ndarray] = []
        self._agent_ids: list[np.ndarray] = []
        self._tri_idxs: list[np.ndarray] = []
        self._lats: list[np.ndarray] = []
        self._lons: list[np.ndarray] = []
        self._eds: list[np.ndarray] = []
        self._behaviors: list[np.ndarray] = []
        self._alive: list[np.ndarray] = []
        self._arrived: list[np.ndarray] = []

    def log_step(self, t: int, population: "Population"):
        pool = population.pool
        n = pool.n
        self._times.append(np.full(n, t, dtype=np.int32))
        # Use Population.agent_ids (stable across compact()) not np.arange(n).
        self._agent_ids.append(population.agent_ids.astype(np.int32).copy())
        self._tri_idxs.append(pool.tri_idx.copy())
        self._lats.append(self.centroids[pool.tri_idx, 0].copy())
        self._lons.append(self.centroids[pool.tri_idx, 1].copy())
        self._eds.append(pool.ed_kJ_g.copy())
        self._behaviors.append(pool.behavior.copy())
        self._alive.append(pool.alive.copy())
        self._arrived.append(pool.arrived.copy())

    def to_dataframe(self) -> pd.DataFrame:
        if not self._times:
            return pd.DataFrame(
                columns=[
                    "time",
                    "agent_id",
                    "tri_idx",
                    "lat",
                    "lon",
                    "ed_kJ_g",
                    "behavior",
                    "alive",
                    "arrived",
                ]
            )
        return pd.DataFrame(
            {
                "time": np.concatenate(self._times),
                "agent_id": np.concatenate(self._agent_ids),
                "tri_idx": np.concatenate(self._tri_idxs),
                "lat": np.concatenate(self._lats),
                "lon": np.concatenate(self._lons),
                "ed_kJ_g": np.concatenate(self._eds),
                "behavior": np.concatenate(self._behaviors),
                "alive": np.concatenate(self._alive),
                "arrived": np.concatenate(self._arrived),
            }
        )

    def close(self):
        df = self.to_dataframe()
        df.to_csv(self.path, index=False)

    def summary(self, t: int, pool: AgentPool) -> dict:
        alive = pool.alive
        return {
            "time": t,
            "n_alive": int(alive.sum()),
            "n_arrived": int(pool.arrived.sum()),
            "mean_ed": float(pool.ed_kJ_g[alive].mean()) if alive.any() else 0.0,
            "behavior_counts": {
                int(b): int((pool.behavior[alive] == b).sum()) for b in range(5)
            },
        }
