"""Track logging and diagnostics output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from salmon_ibm.agents import AgentPool

if TYPE_CHECKING:
    from salmon_ibm.population import Population


class OutputLogger:
    def __init__(
        self,
        path: str,
        centroids: np.ndarray,
        max_steps: int | None = None,
        max_agents: int | None = None,
    ):
        self.path = path
        self.centroids = centroids
        self._preallocated = max_steps is not None and max_agents is not None
        if self._preallocated:
            self._max_steps = max_steps
            self._max_agents = max_agents
            self._n_rows = 0
            self._step_counts = np.zeros(max_steps, dtype=np.int32)
            self._times_arr = np.empty((max_steps, max_agents), dtype=np.int32)
            self._agent_ids_arr = np.empty((max_steps, max_agents), dtype=np.int32)
            self._tri_idxs_arr = np.empty((max_steps, max_agents), dtype=np.int64)
            self._lats_arr = np.empty((max_steps, max_agents), dtype=np.float64)
            self._lons_arr = np.empty((max_steps, max_agents), dtype=np.float64)
            self._eds_arr = np.empty((max_steps, max_agents), dtype=np.float64)
            self._behaviors_arr = np.empty((max_steps, max_agents), dtype=np.int32)
            self._alive_arr = np.empty((max_steps, max_agents), dtype=bool)
            self._arrived_arr = np.empty((max_steps, max_agents), dtype=bool)
        else:
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
        if self._preallocated:
            if self._n_rows >= self._max_steps:
                raise ValueError(
                    f"OutputLogger: exceeded max_steps={self._max_steps}"
                )
            if n > self._max_agents:
                raise ValueError(
                    f"OutputLogger: agent count {n} > max_agents={self._max_agents}"
                )
            r = self._n_rows
            self._step_counts[r] = n
            self._times_arr[r, :n] = t
            self._agent_ids_arr[r, :n] = population.agent_ids[:n].astype(np.int32)
            self._tri_idxs_arr[r, :n] = pool.tri_idx[:n]
            self._lats_arr[r, :n] = self.centroids[pool.tri_idx[:n], 0]
            self._lons_arr[r, :n] = self.centroids[pool.tri_idx[:n], 1]
            self._eds_arr[r, :n] = pool.ed_kJ_g[:n]
            self._behaviors_arr[r, :n] = pool.behavior[:n]
            self._alive_arr[r, :n] = pool.alive[:n]
            self._arrived_arr[r, :n] = pool.arrived[:n]
            self._n_rows += 1
        else:
            self._times.append(np.full(n, t, dtype=np.int32))
            self._agent_ids.append(population.agent_ids.astype(np.int32).copy())
            self._tri_idxs.append(pool.tri_idx.copy())
            self._lats.append(self.centroids[pool.tri_idx, 0].copy())
            self._lons.append(self.centroids[pool.tri_idx, 1].copy())
            self._eds.append(pool.ed_kJ_g.copy())
            self._behaviors.append(pool.behavior.copy())
            self._alive.append(pool.alive.copy())
            self._arrived.append(pool.arrived.copy())

    def to_dataframe(self) -> pd.DataFrame:
        empty_cols = [
            "time", "agent_id", "tri_idx", "lat", "lon",
            "ed_kJ_g", "behavior", "alive", "arrived",
        ]
        if self._preallocated:
            if self._n_rows == 0:
                return pd.DataFrame(columns=empty_cols)
            parts = []
            for r in range(self._n_rows):
                n = int(self._step_counts[r])
                parts.append(pd.DataFrame({
                    "time": self._times_arr[r, :n],
                    "agent_id": self._agent_ids_arr[r, :n],
                    "tri_idx": self._tri_idxs_arr[r, :n],
                    "lat": self._lats_arr[r, :n],
                    "lon": self._lons_arr[r, :n],
                    "ed_kJ_g": self._eds_arr[r, :n],
                    "behavior": self._behaviors_arr[r, :n],
                    "alive": self._alive_arr[r, :n],
                    "arrived": self._arrived_arr[r, :n],
                }))
            return pd.concat(parts, ignore_index=True)
        if not self._times:
            return pd.DataFrame(columns=empty_cols)
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
