"""Accumulator system: general-purpose per-agent floating-point state."""
from __future__ import annotations

import ast
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


# ---------------------------------------------------------------------------
# Updater functions (Tasks 2-4)
# ---------------------------------------------------------------------------

def updater_clear(manager, acc_name, mask):
    """Reset accumulator to zero (or min_val if > 0) for masked agents."""
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]
    val = 0.0
    if defn.min_val is not None:
        val = max(val, defn.min_val)
    manager.data[mask, idx] = val


def updater_increment(manager, acc_name, mask, *, amount):
    """Add a fixed quantity to accumulator for masked agents."""
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]
    new_vals = manager.data[mask, idx] + amount
    if defn.min_val is not None:
        new_vals = np.maximum(new_vals, defn.min_val)
    if defn.max_val is not None:
        new_vals = np.minimum(new_vals, defn.max_val)
    manager.data[mask, idx] = new_vals


def updater_stochastic_increment(manager, acc_name, mask, *, low, high, rng):
    """Add uniform random quantity in [low, high) to accumulator for masked agents."""
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]
    n_masked = mask.sum()
    increments = rng.uniform(low, high, size=n_masked)
    new_vals = manager.data[mask, idx] + increments
    if defn.min_val is not None:
        new_vals = np.maximum(new_vals, defn.min_val)
    if defn.max_val is not None:
        new_vals = np.minimum(new_vals, defn.max_val)
    manager.data[mask, idx] = new_vals


# --- Expression updater helpers ---

_SAFE_MATH = {
    "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp, "log": np.log,
    "log10": np.log10, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "minimum": np.minimum, "maximum": np.maximum,
    "clip": np.clip, "where": np.where,
    "pi": np.pi, "e": np.e,
}

_ALLOWED_NODE_TYPES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant,
    ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.Mod, ast.USub, ast.UAdd,
)


def _validate_expression(expr):
    tree = ast.parse(expr, mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(f"Disallowed construct in expression: {type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_MATH:
                raise ValueError(f"Unknown function in expression: {node.func.id}")


def updater_expression(manager, acc_name, mask, *, expression):
    """Evaluate algebraic expression over accumulators with AST safety validation."""
    _validate_expression(expression)
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]
    namespace = dict(_SAFE_MATH)
    for name, col_idx in manager._name_to_idx.items():
        namespace[name] = manager.data[mask, col_idx]
    result = eval(expression, {"__builtins__": {}}, namespace)
    result = np.asarray(result, dtype=np.float64)
    result = np.nan_to_num(result, nan=0.0,
                           posinf=defn.max_val if defn.max_val is not None else 0.0,
                           neginf=defn.min_val if defn.min_val is not None else 0.0)
    if defn.min_val is not None:
        result = np.maximum(result, defn.min_val)
    if defn.max_val is not None:
        result = np.minimum(result, defn.max_val)
    manager.data[mask, idx] = result


def updater_time_step(manager, acc_name, mask, *, timestep, modulus=None):
    """Write current timestep (optionally with modulus) to accumulator."""
    idx = manager._resolve_idx(acc_name)
    value = float(timestep % modulus) if modulus is not None and modulus > 0 else float(timestep)
    manager.data[mask, idx] = value


def updater_individual_id(manager, acc_name, mask, *, agent_ids):
    """Write each agent's unique ID to accumulator."""
    idx = manager._resolve_idx(acc_name)
    manager.data[mask, idx] = agent_ids[mask].astype(np.float64)


def updater_stochastic_trigger(manager, acc_name, mask, *, probability, rng):
    """Write 1.0 with probability p, else 0.0, for each masked agent."""
    idx = manager._resolve_idx(acc_name)
    n_masked = mask.sum()
    triggers = (rng.random(n_masked) < probability).astype(np.float64)
    manager.data[mask, idx] = triggers


def updater_quantify_location(manager, acc_name, mask, *, hex_map, cell_indices):
    """Sample hex-map values at each agent's current cell position."""
    idx = manager._resolve_idx(acc_name)
    manager.data[mask, idx] = hex_map[cell_indices[mask]]
