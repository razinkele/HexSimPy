"""Track logging and diagnostics output."""
from __future__ import annotations

import numpy as np
import pandas as pd

from salmon_ibm.agents import AgentPool


class OutputLogger:
    def __init__(self, path: str, centroids: np.ndarray):
        self.path = path
        self.centroids = centroids
        self._records: list[dict] = []

    def log_step(self, t: int, pool: AgentPool):
        lats = self.centroids[pool.tri_idx, 0]
        lons = self.centroids[pool.tri_idx, 1]
        for i in range(pool.n):
            self._records.append({
                "time": t,
                "agent_id": i,
                "tri_idx": int(pool.tri_idx[i]),
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "ed_kJ_g": float(pool.ed_kJ_g[i]),
                "behavior": int(pool.behavior[i]),
                "alive": bool(pool.alive[i]),
                "arrived": bool(pool.arrived[i]),
            })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

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
