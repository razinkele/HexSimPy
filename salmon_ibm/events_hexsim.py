"""HexSim-specific event types for real scenario XML compatibility."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from salmon_ibm.events import Event, register_event


def _apply_trait_combo_mask(base_mask, uf, population):
    """Narrow base_mask to agents matching the trait combination in a updater dict.

    If the updater has 'stratified_traits' and 'trait_combinations', the
    space-separated '1'/'0' string encodes which Cartesian-product combinations
    of trait categories are targeted.  We build the intersection of base_mask
    with all agents that fall in any enabled combination.

    Returns the (possibly unchanged) mask.
    """
    stratified = uf.get("stratified_traits")
    combos_str = uf.get("trait_combinations")
    if not stratified or combos_str is None:
        return base_mask
    trait_mgr = getattr(population, "trait_mgr", None)
    if trait_mgr is None:
        return base_mask

    combo_flags = np.array([int(x) for x in str(combos_str).split()], dtype=np.int8)

    # Build list of category arrays and category counts for each trait
    cat_arrays = []
    n_cats = []
    for tname in stratified:
        if tname not in trait_mgr.definitions:
            return base_mask  # trait unknown, can't filter
        defn = trait_mgr.definitions[tname]
        cat_arrays.append(trait_mgr.get(tname))
        n_cats.append(len(defn.categories))

    total_combos = 1
    for n in n_cats:
        total_combos *= n
    if len(combo_flags) != total_combos:
        return base_mask  # mismatch, skip filtering

    # Compute a flat combo index for each agent using mixed-radix encoding
    # flat_idx = cat0 * (n1 * n2 * ...) + cat1 * (n2 * ...) + ...
    n_agents = len(base_mask)
    flat_idx = np.zeros(n_agents, dtype=np.int32)
    stride = 1
    for arr, n in zip(reversed(cat_arrays), reversed(n_cats)):
        flat_idx += arr.astype(np.int32) * stride
        stride *= n

    # Agents matching any enabled combination
    enabled_combo = combo_flags[flat_idx] == 1  # vectorized lookup
    return base_mask & enabled_combo


@register_event("accumulate")  # overrides AccumulateEvent from events_builtin
@dataclass
class HexSimAccumulateEvent(Event):
    """HexSim-style accumulate event that dispatches updater_functions dicts at runtime."""
    updater_functions: list = field(default_factory=list)

    def execute(self, population, landscape, t, mask):
        if not mask.any() or not population:
            return
        acc_mgr = population.accumulator_mgr
        if acc_mgr is None:
            return

        rng = landscape.get("rng", np.random.default_rng())
        spatial_data = landscape.get("spatial_data", {})
        global_vars = landscape.get("global_variables", {})

        for uf in self.updater_functions:
            func_name = uf.get("function", "")
            acc_name = uf.get("accumulator", "")
            params = uf.get("parameters", [])
            spatial = uf.get("spatial_data")
            source_trait = uf.get("source_trait")

            if not acc_name or acc_name not in acc_mgr._name_to_idx:
                continue

            # Apply trait-combination sub-filtering if specified
            uf_mask = _apply_trait_combo_mask(mask, uf, population)
            if not uf_mask.any():
                continue

            try:
                if func_name == "Expression":
                    expr_str = params[0] if params else ""
                    if expr_str:
                        from salmon_ibm.accumulators import updater_expression
                        updater_expression(acc_mgr, acc_name, uf_mask,
                                           expression=expr_str,
                                           globals_dict=global_vars,
                                           rng=rng)

                elif func_name == "Clear":
                    from salmon_ibm.accumulators import updater_clear
                    updater_clear(acc_mgr, acc_name, uf_mask)

                elif func_name == "Increment":
                    from salmon_ibm.accumulators import updater_increment
                    amount = float(params[0]) if params else 1.0
                    updater_increment(acc_mgr, acc_name, uf_mask, amount=amount)

                elif func_name == "TimeStep":
                    from salmon_ibm.accumulators import updater_time_step
                    modulus = int(float(params[0])) if params else 0
                    updater_time_step(acc_mgr, acc_name, uf_mask,
                                     timestep=t,
                                     modulus=modulus if modulus > 0 else None)

                elif func_name == "IndividualLocations":
                    if spatial and spatial in spatial_data:
                        layer = spatial_data[spatial]
                        from salmon_ibm.accumulators import updater_quantify_location
                        updater_quantify_location(acc_mgr, acc_name, uf_mask,
                                                  hex_map=layer,
                                                  cell_indices=population.tri_idx)
                    else:
                        from salmon_ibm.accumulators import updater_individual_locations
                        updater_individual_locations(acc_mgr, acc_name, uf_mask,
                                                    cell_indices=population.tri_idx)

                elif func_name == "QuantifyLocation":
                    if spatial and spatial in spatial_data:
                        layer = spatial_data[spatial]
                        from salmon_ibm.accumulators import updater_quantify_location
                        updater_quantify_location(acc_mgr, acc_name, uf_mask,
                                                  hex_map=layer,
                                                  cell_indices=population.tri_idx)

                elif func_name == "TraitId":
                    if source_trait and population.trait_mgr:
                        from salmon_ibm.accumulators import updater_trait_value_index
                        updater_trait_value_index(acc_mgr, acc_name, uf_mask,
                                                  trait_mgr=population.trait_mgr,
                                                  trait_name=source_trait)

                elif func_name == "ExploredRunningSum":
                    if spatial and spatial in spatial_data:
                        layer = spatial_data[spatial]
                        from salmon_ibm.accumulators import updater_quantify_location
                        updater_quantify_location(acc_mgr, acc_name, uf_mask,
                                                  hex_map=layer,
                                                  cell_indices=population.tri_idx)

            except Exception as e:
                import warnings
                warnings.warn(
                    f"Updater {func_name} for '{acc_name}' failed: {e}",
                    RuntimeWarning, stacklevel=2,
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
            return
        nonzero_cells = np.where(layer != 0)[0]
        if len(nonzero_cells) == 0:
            return
        population.add_agents(len(nonzero_cells), nonzero_cells)


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
                (row_vals >= 0) & (row_vals < self.lookup_table.shape[0]) &
                (col_vals >= 0) & (col_vals < self.lookup_table.shape[1])
            )
            result = np.zeros(len(alive_idx), dtype=np.float64)
            result[valid] = self.lookup_table[row_vals[valid], col_vals[valid]]
        else:
            # 1D lookup
            valid = (row_vals >= 0) & (row_vals < len(self.lookup_table))
            result = np.zeros(len(alive_idx), dtype=np.float64)
            result[valid] = self.lookup_table[row_vals[valid]]

        tgt_idx = acc_mgr._resolve_idx(self.target_accumulator)
        acc_mgr.data[alive_idx, tgt_idx] = result


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

        # Vectorized: find best neighbor within bounds for each agent
        current_cells = population.tri_idx[alive_idx]
        n = len(alive_idx)

        # Gather neighbor matrix
        nbr_matrix = mesh._water_nbrs[current_cells]  # (n, max_nbrs)
        valid = nbr_matrix >= 0
        safe_nbrs = np.where(nbr_matrix >= 0, nbr_matrix, 0)

        # Gradient values at neighbors and current cells
        nbr_grad = gradient[safe_nbrs]  # (n, max_nbrs)
        cur_grad = gradient[current_cells]  # (n,)

        # Distance = |gradient[nb] - gradient[cell]|
        dist = np.abs(nbr_grad - cur_grad[:, np.newaxis])

        # Bounds filtering: valid & (dist >= min_bounds) & (dist <= max_bounds)
        in_bounds = valid & (dist >= min_bounds[:, np.newaxis]) & (dist <= max_bounds[:, np.newaxis])

        # Among in-bounds neighbors, find the one with highest gradient value
        candidate_vals = np.where(in_bounds, nbr_grad, -np.inf)

        # For "better" strategy: only accept if better than current
        if self.strategy == "better":
            candidate_vals = np.where(candidate_vals > cur_grad[:, np.newaxis], candidate_vals, -np.inf)

        best_local = np.argmax(candidate_vals, axis=1)
        best_vals = candidate_vals[np.arange(n), best_local]
        has_valid = best_vals > -np.inf

        targets = np.full(n, -1, dtype=np.intp)
        targets[has_valid] = nbr_matrix[np.where(has_valid)[0], best_local[has_valid]]

        # Write targets to population affinity
        population.affinity_targets[alive_idx] = targets

        # Set error accumulator for agents where no valid cell found
        if acc_mgr and self.error_accumulator:
            err_idx = acc_mgr._resolve_idx(self.error_accumulator)
            failed = targets < 0
            acc_mgr.data[alive_idx[failed], err_idx] = 1.0
            acc_mgr.data[alive_idx[~failed], err_idx] = 0.0


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

    def execute(self, population, landscape, t, mask):
        if not mask.any():
            return
        mesh = landscape.get("mesh")
        spatial_data = landscape.get("spatial_data", {})
        rng = landscape.get("rng", np.random.default_rng())

        if mesh is None:
            return

        gradient = spatial_data.get(self.dispersal_spatial_data)
        if gradient is None:
            gradient = np.ones(mesh.n_cells)

        alive_idx = np.where(mask)[0]
        positions = population.tri_idx[alive_idx].copy()

        acc_mgr = population.accumulator_mgr
        halt_dist = self.dispersal_halt_target

        # Use affinity targets if dispersalUseAffinity is set and targets exist
        use_affinity = self.dispersal_use_affinity is not None
        affinity_targets = None
        if use_affinity and hasattr(population, 'affinity_targets'):
            affinity_targets = population.affinity_targets[alive_idx]

        # Vectorized movement loop
        n_agents = len(alive_idx)
        distances = np.zeros(n_agents)
        n_steps = max(1, int(halt_dist)) if halt_dist > 0 else 1
        water_nbrs = mesh._water_nbrs
        water_nbr_count = mesh._water_nbr_count

        for step in range(n_steps):
            counts = water_nbr_count[positions]
            has_nbrs = counts > 0
            if not has_nbrs.any():
                break

            active = has_nbrs
            if halt_dist > 0:
                active = active & (distances < halt_dist)
                if not active.any():
                    break

            act_idx = np.where(active)[0]
            act_pos = positions[act_idx]

            # Gather neighbor matrix for active agents
            nbr_matrix = water_nbrs[act_pos]  # (n_active, max_nbrs)
            valid = nbr_matrix >= 0

            if affinity_targets is not None:
                # Split into affinity-targeted and gradient-following
                has_target = affinity_targets[act_idx] >= 0
                aff_idx = np.where(has_target)[0]
                grad_idx = np.where(~has_target)[0]

                # --- Affinity targeting (move toward target centroid) ---
                if len(aff_idx) > 0:
                    aff_pos = act_pos[aff_idx]
                    aff_targets = affinity_targets[act_idx[aff_idx]]
                    aff_nbrs = nbr_matrix[aff_idx]
                    aff_valid = valid[aff_idx]
                    # Target centroids
                    tc = mesh.centroids[aff_targets]  # (n_aff, 2)
                    # Neighbor centroids
                    safe_nbrs = np.where(aff_nbrs >= 0, aff_nbrs, 0)
                    nb_cx = mesh.centroids[safe_nbrs, 0]  # (n_aff, max_nbrs)
                    nb_cy = mesh.centroids[safe_nbrs, 1]
                    # Distance from each neighbor to target
                    dx = nb_cx - tc[:, 0:1]
                    dy = nb_cy - tc[:, 1:2]
                    dist_to_target = np.sqrt(dx * dx + dy * dy)
                    dist_to_target[~aff_valid] = np.inf
                    best_local = np.argmin(dist_to_target, axis=1)
                    chosen = aff_nbrs[np.arange(len(aff_idx)), best_local]
                    # Only move if the chosen neighbor is valid
                    moved = aff_valid[np.arange(len(aff_idx)), best_local]
                    global_aff = act_idx[aff_idx]
                    positions[global_aff[moved]] = chosen[moved]
                    distances[global_aff[moved]] += 1.0

                # --- Gradient following for non-affinity agents ---
                if len(grad_idx) > 0:
                    g_pos = act_pos[grad_idx]
                    g_nbrs = nbr_matrix[grad_idx]
                    g_valid = valid[grad_idx]
                    safe_g = np.where(g_nbrs >= 0, g_nbrs, 0)
                    nbr_vals = gradient[safe_g]
                    if self.walk_up_gradient:
                        nbr_vals[~g_valid] = -np.inf
                        best_local = np.argmax(nbr_vals, axis=1)
                    else:
                        nbr_vals[~g_valid] = np.inf
                        best_local = np.argmin(nbr_vals, axis=1)
                    chosen = g_nbrs[np.arange(len(grad_idx)), best_local]
                    moved = g_valid[np.arange(len(grad_idx)), best_local]
                    global_grad = act_idx[grad_idx]
                    positions[global_grad[moved]] = chosen[moved]
                    distances[global_grad[moved]] += 1.0
            else:
                # All agents follow gradient (no affinity targets)
                safe_nbrs = np.where(nbr_matrix >= 0, nbr_matrix, 0)
                nbr_vals = gradient[safe_nbrs]
                if self.walk_up_gradient:
                    nbr_vals[~valid] = -np.inf
                    best_local = np.argmax(nbr_vals, axis=1)
                else:
                    nbr_vals[~valid] = np.inf
                    best_local = np.argmin(nbr_vals, axis=1)
                chosen = nbr_matrix[np.arange(len(act_idx)), best_local]
                moved = valid[np.arange(len(act_idx)), best_local]
                positions[act_idx[moved]] = chosen[moved]
                distances[act_idx[moved]] += 1.0

        population.tri_idx[alive_idx] = positions

        # Write distance to accumulator if specified
        if acc_mgr and self.dispersal_accumulator:
            d_idx = acc_mgr._resolve_idx(self.dispersal_accumulator)
            acc_mgr.data[alive_idx, d_idx] = distances


@register_event("data_probe")
@dataclass
class DataProbeEvent(Event):
    """No-op data probe event — output logging only, no behavioral effect."""

    def execute(self, population, landscape, t, mask):
        pass  # Output logging not implemented; prevents parse failure
