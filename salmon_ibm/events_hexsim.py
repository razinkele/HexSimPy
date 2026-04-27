"""HexSim-specific event types for real scenario XML compatibility."""

from __future__ import annotations
from dataclasses import dataclass, field
import logging
import numpy as np
from salmon_ibm.events import Event, register_event

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


# ---------------------------------------------------------------------------
# Numba JIT kernels for HexSim movement (hot path)
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _move_gradient_numba(
    positions, water_nbrs, water_nbr_count, gradient, n_steps, walk_up, halt_dist
):
    """JIT kernel: move agents along gradient for n_steps.

    Returns (final_positions, distances_moved).
    """
    n = len(positions)
    distances = np.zeros(n, dtype=np.float64)

    for step in range(n_steps):
        for i in prange(n):
            if halt_dist > 0 and distances[i] >= halt_dist:
                continue
            c = positions[i]
            count = water_nbr_count[c]
            if count <= 0:
                continue
            best_nb = c
            best_val = gradient[c]
            for k in range(count):
                nb = water_nbrs[c, k]
                if nb < 0:
                    continue
                val = gradient[nb]
                if walk_up:
                    if val > best_val:
                        best_val = val
                        best_nb = nb
                else:
                    if val < best_val:
                        best_val = val
                        best_nb = nb
            if best_nb != c:
                positions[i] = best_nb
                distances[i] += 1.0
    return positions, distances


@njit(cache=True, parallel=True)
def _move_affinity_numba(
    positions,
    water_nbrs,
    water_nbr_count,
    centroids_x,
    centroids_y,
    targets,
    n_steps,
    halt_dist,
):
    """JIT kernel: move agents toward affinity targets for n_steps.

    Returns (final_positions, distances_moved).
    """
    n = len(positions)
    distances = np.zeros(n, dtype=np.float64)

    for step in range(n_steps):
        for i in prange(n):
            if halt_dist > 0 and distances[i] >= halt_dist:
                continue
            if targets[i] < 0:
                continue
            c = positions[i]
            count = water_nbr_count[c]
            if count <= 0:
                continue
            target = targets[i]
            tx = centroids_x[target]
            ty = centroids_y[target]
            best_nb = c
            best_dist_sq = (centroids_x[c] - tx) ** 2 + (centroids_y[c] - ty) ** 2
            for k in range(count):
                nb = water_nbrs[c, k]
                if nb < 0:
                    continue
                d_sq = (centroids_x[nb] - tx) ** 2 + (centroids_y[nb] - ty) ** 2
                if d_sq < best_dist_sq:
                    best_dist_sq = d_sq
                    best_nb = nb
            if best_nb != c:
                positions[i] = best_nb
                distances[i] += 1.0
    return positions, distances


@njit(cache=True, parallel=True)
def _set_affinity_numba(
    current_cells,
    water_nbrs,
    water_nbr_count,
    gradient,
    min_bounds,
    max_bounds,
    better_strategy,
):
    """JIT kernel: find best neighbor within bounds for each agent."""
    n = len(current_cells)
    targets = np.full(n, -1, dtype=np.intp)
    for i in prange(n):
        cell = current_cells[i]
        count = water_nbr_count[cell]
        if count <= 0:
            continue
        cur_val = gradient[cell]
        best_nb = -1
        best_val = cur_val if better_strategy else -1e30
        for k in range(count):
            nb = water_nbrs[cell, k]
            if nb < 0:
                continue
            nb_val = gradient[nb]
            diff = abs(nb_val - cur_val)
            if diff < min_bounds[i] or diff > max_bounds[i]:
                continue
            if nb_val > best_val:
                best_val = nb_val
                best_nb = nb
        targets[i] = best_nb
    return targets


_combo_flags_cache: dict[str, np.ndarray] = {}
# Per-step cache: (population_id, combos_str) → enabled_mask
# Cleared at start of each step by clear_combo_mask_cache().
# Thread safety: assumes single-threaded event dispatch (no concurrent runs).
_combo_mask_cache: dict[tuple, np.ndarray] = {}


def _apply_trait_combo_mask(base_mask, uf, population):
    """Narrow base_mask to agents matching the trait combination in a updater dict.

    Caches the enabled-combo mask per (population, combo_string, step) to avoid
    recomputing the same mixed-radix encoding thousands of times per step.

    Returns the (possibly unchanged) mask.
    """

    stratified = uf.get("stratified_traits")
    combos_str = uf.get("trait_combinations")
    if not stratified or combos_str is None:
        return base_mask
    trait_mgr = getattr(population, "trait_mgr", None)
    if trait_mgr is None:
        logging.getLogger(__name__).warning(
            "Trait-combo filter requested but population has no trait_mgr. "
            "Event will fire for ALL agents."
        )
        return base_mask

    # Check per-step cache
    pop_id = id(population)
    cache_key = (pop_id, combos_str)
    cached = _combo_mask_cache.get(cache_key)
    if cached is not None and len(cached) == len(base_mask):
        return base_mask & cached

    # Cache parsed combo_flags (same string → same array, permanent)
    combo_flags = _combo_flags_cache.get(combos_str)
    if combo_flags is None:
        combo_flags = np.array([int(x) for x in str(combos_str).split()], dtype=np.int8)
        _combo_flags_cache[combos_str] = combo_flags
        if len(_combo_flags_cache) > 10000:
            _combo_flags_cache.clear()  # prevent unbounded growth

    # Build flat combo index using mixed-radix encoding
    flat_idx = np.zeros(len(base_mask), dtype=np.int32)
    stride = 1
    for tname in reversed(stratified):
        if tname not in trait_mgr.definitions:
            logging.getLogger(__name__).warning(
                f"Trait '{tname}' not found in trait_mgr. "
                f"Available: {list(trait_mgr.definitions.keys())}. "
                f"Event will fire for ALL agents."
            )
            return base_mask
        defn = trait_mgr.definitions[tname]
        n_cat = len(defn.categories)
        flat_idx += trait_mgr.get(tname).astype(np.int32) * stride
        stride *= n_cat

    if len(combo_flags) != stride:
        logging.getLogger(__name__).warning(
            f"Trait combo flags length ({len(combo_flags)}) doesn't match "
            f"expected stride ({stride}). Event will fire for ALL agents."
        )
        return base_mask

    enabled = combo_flags[flat_idx] == 1
    _combo_mask_cache[cache_key] = enabled
    return base_mask & enabled


def clear_combo_mask_cache():
    """Call at the start of each step to invalidate trait-combo caches."""
    _combo_mask_cache.clear()


@register_event("accumulate")  # overrides AccumulateEvent from events_builtin
@dataclass
class HexSimAccumulateEvent(Event):
    """HexSim-style accumulate event that dispatches updater_functions dicts at runtime."""

    updater_functions: list = field(default_factory=list)

    # Lazy-initialized dispatch table (avoids repeated imports + string comparisons)
    _dispatch: dict = field(init=False, default=None, repr=False)

    def _init_dispatch(self):
        from salmon_ibm.accumulators import (
            updater_expression,
            updater_clear,
            updater_increment,
            updater_time_step,
            updater_quantify_location,
            updater_trait_value_index,
        )

        try:
            from salmon_ibm.accumulators import updater_individual_locations
        except ImportError:
            updater_individual_locations = updater_quantify_location
        self._dispatch = {
            "Expression": updater_expression,
            "Clear": updater_clear,
            "Increment": updater_increment,
            "TimeStep": updater_time_step,
            "IndividualLocations": updater_individual_locations,
            "QuantifyLocation": updater_quantify_location,
            "TraitId": updater_trait_value_index,
            "ExploredRunningSum": updater_quantify_location,
        }

    def execute(self, population, landscape, t, mask):
        if not mask.any() or not population:
            return
        acc_mgr = population.accumulator_mgr
        if acc_mgr is None:
            return
        if self._dispatch is None:
            self._init_dispatch()

        rng = landscape.get("rng", np.random.default_rng())
        spatial_data = landscape.get("spatial_data", {})
        global_vars = landscape.get("global_variables", {})

        for uf in self.updater_functions:
            func_name = uf.get("function", "")
            acc_name = uf.get("accumulator", "")
            params = uf.get("parameters", [])
            spatial = uf.get("spatial_data")
            source_trait = uf.get("source_trait")

            if not acc_name:
                continue
            if acc_name not in acc_mgr._name_to_idx:
                logging.getLogger(__name__).warning(
                    f"Updater '{func_name}' references unknown accumulator '{acc_name}'. "
                    f"Available: {list(acc_mgr._name_to_idx.keys())}"
                )
                continue

            # Apply trait-combination sub-filtering if specified
            uf_mask = _apply_trait_combo_mask(mask, uf, population)
            if not uf_mask.any():
                continue

            try:
                handler = self._dispatch.get(func_name)
                if handler is None:
                    logging.getLogger(__name__).warning(
                        f"Unknown updater function '{func_name}' for accumulator "
                        f"'{acc_name}' — skipped. Known: {list(self._dispatch.keys())}"
                    )
                    continue

                if func_name == "Expression":
                    expr_str = params[0] if params else ""
                    if expr_str:
                        handler(
                            acc_mgr,
                            acc_name,
                            uf_mask,
                            expression=expr_str,
                            globals_dict=global_vars,
                            rng=rng,
                        )
                elif func_name == "Clear":
                    handler(acc_mgr, acc_name, uf_mask)
                elif func_name == "Increment":
                    amount = float(params[0]) if params else 1.0
                    handler(acc_mgr, acc_name, uf_mask, amount=amount)
                elif func_name == "TimeStep":
                    modulus = int(float(params[0])) if params else 0
                    handler(
                        acc_mgr,
                        acc_name,
                        uf_mask,
                        timestep=t,
                        modulus=modulus if modulus > 0 else None,
                    )
                elif func_name == "TraitId":
                    if source_trait and population.trait_mgr:
                        self._dispatch["TraitId"](
                            acc_mgr,
                            acc_name,
                            uf_mask,
                            trait_mgr=population.trait_mgr,
                            trait_name=source_trait,
                        )
                else:
                    # Spatial data updaters (IndividualLocations, QuantifyLocation, ExploredRunningSum)
                    if spatial and spatial in spatial_data:
                        layer = spatial_data[spatial]
                        self._dispatch.get("QuantifyLocation", handler)(
                            acc_mgr,
                            acc_name,
                            uf_mask,
                            hex_map=layer,
                            cell_indices=population.tri_idx,
                        )
                    elif func_name == "IndividualLocations":
                        # Write each agent's cell index to the accumulator.
                        updater_individual_locations(
                            acc_mgr,
                            acc_name,
                            uf_mask,
                            cell_indices=population.tri_idx,
                        )

            except (KeyError, ValueError, IndexError) as e:
                logging.getLogger(__name__).warning(
                    f"Updater {func_name} for '{acc_name}' failed: {e}"
                )


@register_event("hexsim_survival")
@dataclass
class HexSimSurvivalEvent(Event):
    """Accumulator-driven survival: agents in trait category 0 die.

    In HexSim, survivalEvent checks the trait linked to the named accumulator.
    Category 0 = "Will Die", any other category = survive.
    """

    survival_accumulator: str = ""

    def execute(self, population, landscape, t, mask):
        if not mask.any() or not self.survival_accumulator:
            return
        acc_mgr = population.accumulator_mgr
        if acc_mgr is None:
            return
        # Get accumulator values for all agents
        acc_vals = acc_mgr.get(self.survival_accumulator)
        # Agents where accumulator value < 1.0 are in category 0 (die)
        alive_idx = np.where(mask)[0]
        die_mask = acc_vals[alive_idx] < 1.0
        die_indices = alive_idx[die_mask]
        population.alive[die_indices] = False


@register_event("patch_introduction")
@dataclass
class PatchIntroductionEvent(Event):
    """Place one agent on every non-zero cell of a named spatial data layer."""

    patch_spatial_data: str = ""

    def execute(self, population, landscape, t, mask):
        spatial_registry = landscape.get("spatial_data", {})
        layer = spatial_registry.get(self.patch_spatial_data)
        if layer is None:
            available = list(spatial_registry.keys()) if spatial_registry else []
            logging.getLogger(__name__).warning(
                f"PatchIntroductionEvent '{self.name}': spatial data layer "
                f"'{self.patch_spatial_data}' not found. Available: {available}. "
                f"No agents will be introduced."
            )
            return
        # HexSim hex-maps store integer-valued floats (0.0 = no-data, nonzero = water).
        # Exact float comparison with 0 is safe for this data format.
        nonzero_cells = np.where(layer != 0)[0]
        if len(nonzero_cells) == 0:
            return
        new_idx = population.add_agents(len(nonzero_cells), nonzero_cells)
        mesh = landscape.get("mesh")
        if mesh is not None:
            population.set_natal_reach_from_cells(new_idx, mesh)


@register_event("data_lookup")
@dataclass
class DataLookupEvent(Event):
    """Look up values from a table using accumulator values as row/column keys."""

    file_name: str = ""
    row_accumulator: str = ""
    column_accumulator: str = ""
    target_accumulator: str = ""
    lookup_table: np.ndarray | None = field(default=None, repr=False)

    def execute(self, population, landscape, t, mask):
        if not mask.any():
            return
        acc_mgr = population.accumulator_mgr
        if acc_mgr is None or self.lookup_table is None:
            return
        alive_idx = np.where(mask)[0]
        row_vals = acc_mgr.get(self.row_accumulator)[alive_idx].astype(int)

        if self.column_accumulator:
            col_vals = acc_mgr.get(self.column_accumulator)[alive_idx].astype(int)
            # 2D lookup
            valid = (
                (row_vals >= 0)
                & (row_vals < self.lookup_table.shape[0])
                & (col_vals >= 0)
                & (col_vals < self.lookup_table.shape[1])
            )
            result = np.zeros(len(alive_idx), dtype=np.float64)
            result[valid] = self.lookup_table[row_vals[valid], col_vals[valid]]
        else:
            # 1D lookup
            valid = (row_vals >= 0) & (row_vals < len(self.lookup_table))
            result = np.zeros(len(alive_idx), dtype=np.float64)
            result[valid] = self.lookup_table[row_vals[valid]]

        tgt_idx = acc_mgr._resolve_idx(self.target_accumulator)
        acc_mgr.data[tgt_idx, alive_idx] = result


@register_event("set_spatial_affinity")
@dataclass
class SetSpatialAffinityEvent(Event):
    """Set a spatial affinity goal — pick best cell within bounds on a gradient."""

    affinity_name: str = ""
    strategy: str = "better"
    spatial_series: str = ""
    error_accumulator: str = ""
    min_accumulator: str = ""
    max_accumulator: str = ""

    def execute(self, population, landscape, t, mask):
        if not mask.any():
            return
        spatial_data = landscape.get("spatial_data", {})
        gradient = spatial_data.get(self.spatial_series)
        mesh = landscape.get("mesh")
        if gradient is None or mesh is None:
            return
        acc_mgr = population.accumulator_mgr
        alive_idx = np.where(mask)[0]

        # Get search bounds from accumulators
        if acc_mgr and self.min_accumulator:
            min_bounds = acc_mgr.get(self.min_accumulator)[alive_idx]
        else:
            min_bounds = np.zeros(len(alive_idx))
        if acc_mgr and self.max_accumulator:
            max_bounds = acc_mgr.get(self.max_accumulator)[alive_idx]
        else:
            max_bounds = np.full(len(alive_idx), np.inf)

        # Find best neighbor within bounds for each agent
        current_cells = population.tri_idx[alive_idx]

        if HAS_NUMBA:
            targets = _set_affinity_numba(
                current_cells,
                mesh._water_nbrs,
                mesh._water_nbr_count,
                gradient,
                min_bounds,
                max_bounds,
                self.strategy == "better",
            )
        else:
            # NumPy fallback
            n = len(alive_idx)
            nbr_matrix = mesh._water_nbrs[current_cells]
            valid = nbr_matrix >= 0
            safe_nbrs = np.where(nbr_matrix >= 0, nbr_matrix, 0)
            nbr_grad = gradient[safe_nbrs]
            cur_grad = gradient[current_cells]
            dist = np.abs(nbr_grad - cur_grad[:, np.newaxis])
            in_bounds = (
                valid
                & (dist >= min_bounds[:, np.newaxis])
                & (dist <= max_bounds[:, np.newaxis])
            )
            candidate_vals = np.where(in_bounds, nbr_grad, -np.inf)
            if self.strategy == "better":
                candidate_vals = np.where(
                    candidate_vals > cur_grad[:, np.newaxis], candidate_vals, -np.inf
                )
            best_local = np.argmax(candidate_vals, axis=1)
            best_vals = candidate_vals[np.arange(n), best_local]
            has_valid = best_vals > -np.inf
            targets = np.full(n, -1, dtype=np.intp)
            targets[has_valid] = nbr_matrix[
                np.where(has_valid)[0], best_local[has_valid]
            ]

        # Write targets to population affinity
        population.affinity_targets[alive_idx] = targets

        # Set error accumulator for agents where no valid cell found
        if acc_mgr and self.error_accumulator:
            err_idx = acc_mgr._resolve_idx(self.error_accumulator)
            failed = targets < 0
            acc_mgr.data[err_idx, alive_idx[failed]] = 1.0
            acc_mgr.data[err_idx, alive_idx[~failed]] = 0.0


@register_event("move")
@dataclass
class HexSimMoveEvent(Event):
    """HexSim movement event — gradient-following with optional affinity targeting."""

    move_strategy: str = "onlyDisperse"
    dispersal_spatial_data: str = ""
    walk_up_gradient: bool = False
    dispersal_accumulator: str = ""
    distance_accumulator: str = ""
    dispersal_use_affinity: object = None  # True (empty elem) or str (named affinity)
    dispersal_halt_target: float = 0.0
    resource_threshold: float = 0.0

    # Cached references (resolved on first execute, reused on subsequent calls).
    # NOTE: assumes gradient is static (loaded once from workspace hex-maps).
    # If time-varying spatial data is added, invalidate _cached_gradient per step.
    _cached_mesh: object = field(init=False, default=None, repr=False)
    _cached_gradient: np.ndarray = field(init=False, default=None, repr=False)
    _cached_nbrs: np.ndarray = field(init=False, default=None, repr=False)
    _cached_nbr_count: np.ndarray = field(init=False, default=None, repr=False)
    _cached_cx: np.ndarray = field(init=False, default=None, repr=False)
    _cached_cy: np.ndarray = field(init=False, default=None, repr=False)

    def execute(self, population, landscape, t, mask):
        if not mask.any():
            return

        # Cache mesh and gradient references (same across all calls within a step).
        # Invalidate cache if mesh changed (e.g., across ensemble runs).
        mesh = landscape.get("mesh")
        if mesh is None:
            return
        if self._cached_mesh is not mesh:
            self._cached_mesh = mesh
            self._cached_nbrs = mesh._water_nbrs
            self._cached_nbr_count = mesh._water_nbr_count
            self._cached_cx = np.ascontiguousarray(mesh.centroids[:, 0])
            self._cached_cy = np.ascontiguousarray(mesh.centroids[:, 1])
            spatial_data = landscape.get("spatial_data", {})
            gradient = spatial_data.get(self.dispersal_spatial_data)
            if gradient is None:
                available = list(spatial_data.keys()) if spatial_data else []
                logging.getLogger(__name__).warning(
                    f"HexSimMoveEvent '{self.name}': spatial data layer "
                    f"'{self.dispersal_spatial_data}' not found. "
                    f"Available layers: {available}. "
                    f"Falling back to uniform gradient (no directed movement)."
                )
                gradient = np.ones(mesh.n_cells)
            self._cached_gradient = gradient

        alive_idx = np.where(mask)[0]
        positions = population.tri_idx[alive_idx].copy()

        acc_mgr = population.accumulator_mgr
        halt_dist = self.dispersal_halt_target

        # Use affinity targets if dispersalUseAffinity is set and targets exist
        use_affinity = self.dispersal_use_affinity is not None
        affinity_targets = None
        if use_affinity and hasattr(population, "affinity_targets"):
            affinity_targets = population.affinity_targets[alive_idx]

        # Movement kernel dispatch
        n_agents = len(alive_idx)
        n_steps = max(1, int(halt_dist)) if halt_dist > 0 else 1
        water_nbrs = self._cached_nbrs
        water_nbr_count = self._cached_nbr_count
        gradient = self._cached_gradient
        mesh = self._cached_mesh

        if affinity_targets is not None and (affinity_targets >= 0).any():
            # Split: affinity-targeted agents vs gradient-following
            has_target = affinity_targets >= 0
            aff_mask = np.where(has_target)[0]
            grad_mask = np.where(~has_target)[0]

            distances = np.zeros(n_agents)

            if len(aff_mask) > 0 and HAS_NUMBA:
                aff_pos = positions[aff_mask].copy()
                aff_pos, aff_dist = _move_affinity_numba(
                    aff_pos,
                    water_nbrs,
                    water_nbr_count,
                    self._cached_cx,
                    self._cached_cy,
                    affinity_targets[aff_mask],
                    n_steps,
                    halt_dist,
                )
                positions[aff_mask] = aff_pos
                distances[aff_mask] = aff_dist
            elif len(aff_mask) > 0:
                # NumPy fallback for affinity
                for i in aff_mask:
                    for _ in range(n_steps):
                        if halt_dist > 0 and distances[i] >= halt_dist:
                            break
                        c = positions[i]
                        tgt = affinity_targets[i]
                        if tgt < 0 or water_nbr_count[c] <= 0:
                            break
                        best = c
                        best_d = (
                            mesh.centroids[c, 0] - mesh.centroids[tgt, 0]
                        ) ** 2 + (mesh.centroids[c, 1] - mesh.centroids[tgt, 1]) ** 2
                        for k in range(water_nbr_count[c]):
                            nb = water_nbrs[c, k]
                            if nb < 0:
                                continue
                            d = (
                                mesh.centroids[nb, 0] - mesh.centroids[tgt, 0]
                            ) ** 2 + (
                                mesh.centroids[nb, 1] - mesh.centroids[tgt, 1]
                            ) ** 2
                            if d < best_d:
                                best_d = d
                                best = nb
                        if best != c:
                            positions[i] = best
                            distances[i] += 1.0

            if len(grad_mask) > 0:
                grad_pos = positions[grad_mask].copy()
                if HAS_NUMBA:
                    grad_pos, grad_dist = _move_gradient_numba(
                        grad_pos,
                        water_nbrs,
                        water_nbr_count,
                        gradient,
                        n_steps,
                        self.walk_up_gradient,
                        halt_dist,
                    )
                else:
                    grad_dist = np.zeros(len(grad_mask))
                    for i in range(len(grad_mask)):
                        for _ in range(n_steps):
                            if halt_dist > 0 and grad_dist[i] >= halt_dist:
                                break
                            c = grad_pos[i]
                            if water_nbr_count[c] <= 0:
                                break
                            best = c
                            best_val = gradient[c]
                            for k in range(water_nbr_count[c]):
                                nb = water_nbrs[c, k]
                                if nb < 0:
                                    continue
                                val = gradient[nb]
                                if self.walk_up_gradient and val > best_val:
                                    best_val = val
                                    best = nb
                                elif not self.walk_up_gradient and val < best_val:
                                    best_val = val
                                    best = nb
                            if best != c:
                                grad_pos[i] = best
                                grad_dist[i] += 1.0
                positions[grad_mask] = grad_pos
                distances[grad_mask] = grad_dist
        else:
            # All agents follow gradient — single Numba call
            if HAS_NUMBA:
                positions, distances = _move_gradient_numba(
                    positions,
                    water_nbrs,
                    water_nbr_count,
                    gradient,
                    n_steps,
                    self.walk_up_gradient,
                    halt_dist,
                )
            else:
                distances = np.zeros(n_agents)
                for i in range(n_agents):
                    for _ in range(n_steps):
                        if halt_dist > 0 and distances[i] >= halt_dist:
                            break
                        c = positions[i]
                        if water_nbr_count[c] <= 0:
                            break
                        best = c
                        best_val = gradient[c]
                        for k in range(water_nbr_count[c]):
                            nb = water_nbrs[c, k]
                            if nb < 0:
                                continue
                            val = gradient[nb]
                            if self.walk_up_gradient and val > best_val:
                                best_val = val
                                best = nb
                            elif not self.walk_up_gradient and val < best_val:
                                best_val = val
                                best = nb
                        if best != c:
                            positions[i] = best
                            distances[i] += 1.0

        population.tri_idx[alive_idx] = positions

        # Write distance to accumulator if specified
        if acc_mgr and self.dispersal_accumulator:
            d_idx = acc_mgr._resolve_idx(self.dispersal_accumulator)
            acc_mgr.data[d_idx, alive_idx] = distances


@register_event("data_probe")
@dataclass
class DataProbeEvent(Event):
    """No-op data probe event — output logging only, no behavioral effect."""

    @classmethod
    def from_descriptor(cls, descriptor):
        """Construct from a typed EventDescriptor."""
        return cls(
            name=descriptor.name,
            trigger=None,
            population_name=descriptor.population_name,
            enabled=descriptor.enabled,
        )

    def execute(self, population, landscape, t, mask):
        pass  # Output logging not implemented; prevents parse failure
