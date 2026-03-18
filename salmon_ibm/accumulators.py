"""Accumulator system: general-purpose per-agent floating-point state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class AccumulatorDef:
    """Definition for a single accumulator column."""
    name: str
    min_val: float | None = None
    max_val: float | None = None
    linked_trait: str | None = None


class AccumulatorManager:
    """Vectorized storage and manipulation of per-agent accumulators.

    Storage: 2D NumPy array of shape (n_agents, n_accumulators).
    """

    def __init__(self, n_agents: int, definitions: list[AccumulatorDef]):
        self.n_agents = n_agents
        self.definitions = list(definitions)
        self._name_to_idx: dict[str, int] = {
            d.name: i for i, d in enumerate(definitions)
        }
        n_acc = len(definitions)
        self.data = np.zeros((n_agents, n_acc), dtype=np.float64)

    def index_of(self, name: str) -> int:
        return self._name_to_idx[name]

    def _resolve_idx(self, key: Union[str, int]) -> int:
        if isinstance(key, str):
            return self._name_to_idx[key]
        return key

    def get(self, key: Union[str, int]) -> np.ndarray:
        idx = self._resolve_idx(key)
        return self.data[:, idx]

    def set(
        self,
        key: Union[str, int],
        values: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        idx = self._resolve_idx(key)
        defn = self.definitions[idx]
        clamped = values
        if defn.min_val is not None:
            clamped = np.maximum(clamped, defn.min_val)
        if defn.max_val is not None:
            clamped = np.minimum(clamped, defn.max_val)
        if mask is not None:
            self.data[mask, idx] = clamped
        else:
            self.data[:, idx] = clamped
