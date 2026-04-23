"""Accumulator system: general-purpose per-agent floating-point state."""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass
from typing import Union
import numpy as np

# LRU-bounded cache: prevents scenario-authored expression flooding
# (attacker could submit 9999 distinct expressions to keep cache pegged
# at near-capacity forever under the old "bulk clear at 10000" approach).
_EXPR_CACHE_MAX = 256
_compiled_expr_cache: "OrderedDict[str, object]" = OrderedDict()


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
    "sqrt": np.sqrt,
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    "where": np.where,
    "pi": np.pi,
    "e": np.e,
}

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Subscript,
    ast.Attribute,  # validated: only _rng.<allowlisted method> (see _validate_expression)
)

# Methods on _rng that scenario expressions may call (numpy.random.Generator).
# Extend only if a real HexSim scenario uses it AND the method is non-mutating
# (i.e., does NOT include seed, bytes, or any method with side effects).
_ALLOWED_RNG_METHODS = frozenset({"random", "uniform", "normal", "integers"})

# DoS guard: reject _rng.<method>(N) when N exceeds this bound.
# Prevents scenario XML from allocating GB-scale arrays via e.g. _rng.random(10**9).
# 1M floats = 8 MB; typical legitimate usage is <= n_agents (<100k).
_RNG_ARG_MAX = 1_000_000

# Functions available in the HexSim DSL namespace (produced by translate_hexsim_expr)
_HEXSIM_FUNCTIONS = frozenset(
    {
        "_cond",
        "_g",
        "_a",
        "Floor",
        "Pow",
        "Exp",
        "Max",
        "Min",
        "GasDev",
        "Rand",
    }
)


def _validate_expression(expr, extra_names=None):
    """Validate an expression AST for disallowed constructs.

    Parameters
    ----------
    expr : str
        The expression string to validate.
    extra_names : frozenset or None
        Additional function/name identifiers to allow beyond _SAFE_MATH.
        Pass _HEXSIM_FUNCTIONS when validating translated HexSim expressions.
    """
    allowed_names = set(_SAFE_MATH)
    if extra_names:
        allowed_names |= extra_names
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(
                f"Disallowed construct in expression: {type(node).__name__}"
            )
        if isinstance(node, ast.Attribute):
            # Only allow attribute access on _rng, and only for allowlisted methods.
            if not (isinstance(node.value, ast.Name) and node.value.id == "_rng"):
                raise ValueError(
                    f"Disallowed attribute access in expression: "
                    f"only '_rng.<method>' is permitted, got "
                    f"'{ast.dump(node.value)}.{node.attr}'"
                )
            if node.attr not in _ALLOWED_RNG_METHODS:
                raise ValueError(
                    f"Disallowed _rng method '{node.attr}'. "
                    f"Allowed: {sorted(_ALLOWED_RNG_METHODS)}"
                )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_names:
                    raise ValueError(f"Unknown function in expression: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                # Attribute-form calls were already validated by the ast.Attribute
                # branch above. Add a DoS guard on literal numeric args to prevent
                # e.g. _rng.random(10**9).
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(
                        arg.value, (int, float)
                    ):
                        if arg.value > _RNG_ARG_MAX:
                            raise ValueError(
                                f"Argument {arg.value} too large in _rng call "
                                f"(exceeds _RNG_ARG_MAX={_RNG_ARG_MAX})"
                            )
            else:
                raise ValueError(
                    f"Disallowed call target: {type(node.func).__name__}"
                )


class _LazyAccDict:
    """Dict-like that lazily copies accumulator columns on first access.

    Avoids building a full {name: masked_column} dict for all 74 accumulators
    when the expression only references 1-2.
    """

    __slots__ = ("_data", "_mask", "_idx", "_cache")

    def __init__(self, data, mask, name_to_idx):
        self._data = data
        self._mask = mask
        self._idx = name_to_idx
        self._cache = {}

    def __getitem__(self, name):
        if name not in self._cache:
            col = self._idx.get(name)
            if col is None:
                raise KeyError(name)
            self._cache[name] = self._data[self._mask, col]
        return self._cache[name]

    def __contains__(self, name):
        return name in self._idx

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default


def updater_expression(
    manager, acc_name, mask, *, expression, globals_dict=None, rng=None
):
    """Evaluate algebraic expression over accumulators with AST safety validation.

    If globals_dict is provided, use HexSim DSL translation mode:
    - Single-quoted names → global variable lookup in globals_dict
    - Double-quoted names → accumulator lookup
    - Cond() → vectorised sign-of-difference conditional

    If globals_dict is None, use legacy bare-name mode (backward compatible).
    """
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]

    if globals_dict is not None:
        # HexSim mode: translate and evaluate with full namespace
        from salmon_ibm.hexsim_expr import translate_hexsim_expr, build_hexsim_namespace

        translated = translate_hexsim_expr(expression)
        _validate_expression(translated, extra_names=_HEXSIM_FUNCTIONS)
        n_masked = int(mask.sum())
        # Lazy dict: only copies columns that the expression actually reads
        # (avoids copying all 74 accumulators when expression uses 1-2)
        acc_dict = _LazyAccDict(manager.data, mask, manager._name_to_idx)
        if rng is None:
            rng = np.random.default_rng()
        namespace = build_hexsim_namespace(globals_dict, acc_dict, rng, n_masked)
        if translated not in _compiled_expr_cache:
            if len(_compiled_expr_cache) >= _EXPR_CACHE_MAX:
                _compiled_expr_cache.popitem(last=False)  # LRU eviction (oldest)
            _compiled_expr_cache[translated] = compile(
                translated, "<hexsim-expr>", "eval"
            )
        else:
            _compiled_expr_cache.move_to_end(translated)  # mark as recently used
        result = eval(_compiled_expr_cache[translated], {"__builtins__": {}}, namespace)
    else:
        # Legacy mode: bare accumulator names in namespace
        _validate_expression(expression)
        namespace = dict(_SAFE_MATH)
        for name, col_idx in manager._name_to_idx.items():
            namespace[name] = manager.data[mask, col_idx]
        result = eval(expression, {"__builtins__": {}}, namespace)

    result = np.asarray(result, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(mask.sum(), float(result))
    n_bad = np.count_nonzero(~np.isfinite(result))
    if n_bad > 0:
        import warnings

        warnings.warn(
            f"Expression '{expression}' produced {n_bad} NaN/Inf values "
            f"out of {len(result)} agents. Replacing with bounds/zero.",
            RuntimeWarning,
            stacklevel=2,
        )
    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=defn.max_val if defn.max_val is not None else 0.0,
        neginf=defn.min_val if defn.min_val is not None else 0.0,
    )
    if defn.min_val is not None:
        result = np.maximum(result, defn.min_val)
    if defn.max_val is not None:
        result = np.minimum(result, defn.max_val)
    manager.data[mask, idx] = result


def updater_time_step(manager, acc_name, mask, *, timestep, modulus=None):
    """Write current timestep (optionally with modulus) to accumulator."""
    idx = manager._resolve_idx(acc_name)
    value = (
        float(timestep % modulus)
        if modulus is not None and modulus > 0
        else float(timestep)
    )
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
    cells = cell_indices[mask]
    # Bounds-check: clip cell indices to valid range
    valid = (cells >= 0) & (cells < len(hex_map))
    result = np.zeros(int(mask.sum()), dtype=np.float64)
    result[valid] = hex_map[cells[valid]]
    manager.data[mask, idx] = result


# --- Remaining 17 updater functions (complete HexSim set) ---


def updater_accumulator_transfer(
    manager,
    source_name: str,
    target_name: str,
    mask,
    *,
    fraction: float = 1.0,
):
    """Transfer a fraction of one accumulator's value to another.

    Conservation: the actual amount moved from source (which may differ from
    `fraction * src_value` if the source clamps at min_val) is what the target
    receives. The pre-clamp nominal amount is never added to the target.
    """
    src_idx = manager._resolve_idx(source_name)
    tgt_idx = manager._resolve_idx(target_name)

    src_before = manager.data[mask, src_idx].copy()
    nominal_amount = src_before * fraction
    src_defn = manager.definitions[src_idx]
    new_src = src_before - nominal_amount
    if src_defn.min_val is not None:
        new_src = np.maximum(new_src, src_defn.min_val)
    if src_defn.max_val is not None:
        new_src = np.minimum(new_src, src_defn.max_val)
    manager.data[mask, src_idx] = new_src

    # Actual mass moved = how much source actually changed. Preserves conservation.
    actual_amount = src_before - new_src

    tgt_defn = manager.definitions[tgt_idx]
    new_tgt = manager.data[mask, tgt_idx] + actual_amount
    if tgt_defn.min_val is not None:
        new_tgt = np.maximum(new_tgt, tgt_defn.min_val)
    if tgt_defn.max_val is not None:
        new_tgt = np.minimum(new_tgt, tgt_defn.max_val)
    manager.data[mask, tgt_idx] = new_tgt


def updater_allocated_hexagons(
    manager,
    acc_name: str,
    mask,
    *,
    range_allocator,
    agent_indices,
):
    """Count hexagons in each agent's allocated territory."""
    idx = manager._resolve_idx(acc_name)
    for i in np.where(mask)[0]:
        agent_range = (
            range_allocator.get_range(i)
            if hasattr(range_allocator, "get_range")
            else None
        )
        count = (
            len(agent_range.cells)
            if agent_range is not None and hasattr(agent_range, "cells")
            else 0
        )
        manager.data[i, idx] = float(count)


def updater_explored_hexagons(
    manager,
    acc_name: str,
    mask,
    *,
    explored_sets,
    agent_indices,
):
    """Count hexagons in each agent's explored area."""
    idx = manager._resolve_idx(acc_name)
    for i in np.where(mask)[0]:
        explored = (
            explored_sets.get(i, set()) if isinstance(explored_sets, dict) else set()
        )
        manager.data[i, idx] = float(len(explored))


def updater_group_size(
    manager,
    acc_name: str,
    mask,
    *,
    group_ids,
):
    """Write the size of each agent's group to accumulator."""
    idx = manager._resolve_idx(acc_name)
    masked_idx = np.where(mask)[0]
    groups = group_ids[masked_idx]
    unique, counts = np.unique(groups[groups >= 0], return_counts=True)
    size_map = dict(zip(unique, counts))
    for i in masked_idx:
        gid = group_ids[i]
        manager.data[i, idx] = float(size_map.get(gid, 0)) if gid >= 0 else 0.0


def updater_group_sum(
    manager,
    acc_name: str,
    source_name: str,
    mask,
    *,
    group_ids,
):
    """Sum a source accumulator across all group members, write to target."""
    src_idx = manager._resolve_idx(source_name)
    tgt_idx = manager._resolve_idx(acc_name)
    masked_idx = np.where(mask)[0]
    groups = group_ids[masked_idx]
    unique_groups = np.unique(groups[groups >= 0])
    for gid in unique_groups:
        members = masked_idx[groups == gid]
        total = manager.data[members, src_idx].sum()
        manager.data[members, tgt_idx] = total


def updater_births(
    manager,
    acc_name: str,
    mask,
    *,
    birth_counts,
):
    """Write the number of offspring produced by each agent."""
    idx = manager._resolve_idx(acc_name)
    manager.data[mask, idx] = birth_counts[mask].astype(np.float64)


def updater_mate_verification(
    manager,
    acc_name: str,
    mask,
    *,
    mate_ids,
    alive,
):
    """Clear mate accumulator if the mate has died."""
    idx = manager._resolve_idx(acc_name)
    masked_idx = np.where(mask)[0]
    for i in masked_idx:
        mate_id = int(manager.data[i, idx])
        if mate_id >= 0 and mate_id < len(alive) and not alive[mate_id]:
            manager.data[i, idx] = -1.0


def updater_quantify_extremes(
    manager,
    acc_name: str,
    mask,
    *,
    hex_map,
    cell_indices,
    mode="max",
):
    """Write min or max hex-map value in each agent's explored/allocated area.

    Simplified: uses the single cell at agent's position.
    """
    idx = manager._resolve_idx(acc_name)
    cells = cell_indices[mask]
    valid = (cells >= 0) & (cells < len(hex_map))
    result = np.zeros(int(mask.sum()), dtype=np.float64)
    result[valid] = hex_map[cells[valid]]
    manager.data[mask, idx] = result


def updater_hexagon_presence(
    manager,
    acc_name: str,
    mask,
    *,
    hex_map,
    cell_indices,
    threshold=0.0,
):
    """Write 1.0 if hex-map value at agent's cell exceeds threshold, else 0.0."""
    idx = manager._resolve_idx(acc_name)
    cells = cell_indices[mask]
    valid = (cells >= 0) & (cells < len(hex_map))
    result = np.zeros(int(mask.sum()), dtype=np.float64)
    result[valid] = (hex_map[cells[valid]] > threshold).astype(np.float64)
    manager.data[mask, idx] = result


def updater_uptake(
    manager,
    acc_name: str,
    mask,
    *,
    hex_map,
    cell_indices,
    rate=1.0,
):
    """Transfer value from hex-map into accumulator (resource extraction)."""
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]
    cells = cell_indices[mask]
    extracted = hex_map[cells] * rate
    new_vals = manager.data[mask, idx] + extracted
    if defn.min_val is not None:
        new_vals = np.maximum(new_vals, defn.min_val)
    if defn.max_val is not None:
        new_vals = np.minimum(new_vals, defn.max_val)
    manager.data[mask, idx] = new_vals
    # Use unbuffered subtract to handle repeated cell indices correctly
    np.subtract.at(hex_map, cells, extracted)


def updater_individual_locations(
    manager,
    acc_name: str,
    mask,
    *,
    cell_indices,
):
    """Write each agent's current cell index (patch ID) to accumulator."""
    idx = manager._resolve_idx(acc_name)
    manager.data[mask, idx] = cell_indices[mask].astype(np.float64)


def updater_resources_allocated(
    manager,
    acc_name: str,
    mask,
    *,
    resource_map,
    range_allocator,
):
    """Percentage of resource target met by allocated territory (simplified)."""
    idx = manager._resolve_idx(acc_name)
    for i in np.where(mask)[0]:
        agent_range = (
            range_allocator.get_range(i)
            if hasattr(range_allocator, "get_range")
            else None
        )
        if agent_range is not None and hasattr(agent_range, "cells"):
            total_resource = sum(resource_map[c] for c in agent_range.cells)
            manager.data[i, idx] = total_resource
        else:
            manager.data[i, idx] = 0.0


def updater_resources_explored(
    manager,
    acc_name: str,
    mask,
    *,
    resource_map,
    explored_sets,
):
    """Resource total in explored area (simplified)."""
    idx = manager._resolve_idx(acc_name)
    for i in np.where(mask)[0]:
        explored = (
            explored_sets.get(i, set()) if isinstance(explored_sets, dict) else set()
        )
        total = sum(resource_map[c] for c in explored)
        manager.data[i, idx] = float(total)


def updater_subpopulation_assign(
    manager,
    acc_name: str,
    mask,
    *,
    n_select,
    value,
    rng,
):
    """Randomly select N agents from masked set and assign a value."""
    idx = manager._resolve_idx(acc_name)
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return
    n = min(n_select, len(candidates))
    selected = rng.choice(candidates, size=n, replace=False)
    manager.data[selected, idx] = value


def updater_subpopulation_selector(
    manager,
    acc_name: str,
    mask,
    *,
    group_ids,
    n_per_group,
    value,
):
    """Select N agents per group (non-randomly, first N) and set value."""
    idx = manager._resolve_idx(acc_name)
    masked_idx = np.where(mask)[0]
    groups = group_ids[masked_idx]
    for gid in np.unique(groups[groups >= 0]):
        members = masked_idx[groups == gid]
        selected = members[:n_per_group]
        manager.data[selected, idx] = value


def updater_trait_value_index(
    manager,
    acc_name: str,
    mask,
    *,
    trait_mgr,
    trait_name,
):
    """Write each agent's trait category index to accumulator."""
    idx = manager._resolve_idx(acc_name)
    trait_vals = trait_mgr.get(trait_name)
    manager.data[mask, idx] = trait_vals[mask].astype(np.float64)


def updater_data_lookup(
    manager,
    acc_name: str,
    mask,
    *,
    lookup_table,
    key_acc_name,
):
    """Look up values in a table keyed by another accumulator's integer value."""
    idx = manager._resolve_idx(acc_name)
    key_idx = manager._resolve_idx(key_acc_name)
    keys = manager.data[mask, key_idx].astype(int)
    valid = (keys >= 0) & (keys < len(lookup_table))
    result = np.zeros(mask.sum(), dtype=np.float64)
    result[valid] = lookup_table[keys[valid]]
    manager.data[mask, idx] = result
