"""HexSim-specific event types for real scenario XML compatibility."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from salmon_ibm.events import Event, register_event


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

        # For each agent, find best cell within bounds
        current_cells = population.tri_idx[alive_idx]
        targets = np.full(len(alive_idx), -1, dtype=np.intp)

        for i in range(len(alive_idx)):
            cell = current_cells[i]
            best_cell = -1
            best_val = gradient[cell] if self.strategy == "better" else -np.inf

            # Search neighbors within bounds
            count = mesh._water_nbr_count[cell]
            for k in range(count):
                nb = mesh._water_nbrs[cell, k]
                if nb < 0:
                    continue
                dist = abs(gradient[nb] - gradient[cell])
                if dist < min_bounds[i] or dist > max_bounds[i]:
                    continue
                if gradient[nb] > best_val:
                    best_val = gradient[nb]
                    best_cell = nb

            targets[i] = best_cell

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

        # Movement loop: step toward gradient or affinity target
        distances = np.zeros(len(alive_idx))
        n_steps = max(1, int(halt_dist)) if halt_dist > 0 else 1
        for step in range(n_steps):
            for i in range(len(alive_idx)):
                c = positions[i]
                count = mesh._water_nbr_count[c]
                if count <= 0:
                    continue

                if affinity_targets is not None and affinity_targets[i] >= 0:
                    # Move toward affinity target
                    target = affinity_targets[i]
                    best_nb = c
                    best_dist = np.inf
                    tc = mesh.centroids[target]
                    for k in range(count):
                        nb = mesh._water_nbrs[c, k]
                        if nb < 0:
                            continue
                        d = np.sqrt(
                            (mesh.centroids[nb, 0] - tc[0]) ** 2 +
                            (mesh.centroids[nb, 1] - tc[1]) ** 2
                        )
                        if d < best_dist:
                            best_dist = d
                            best_nb = nb
                    positions[i] = best_nb
                else:
                    # Gradient following
                    best_nb = c
                    best_val = gradient[c]
                    for k in range(count):
                        nb = mesh._water_nbrs[c, k]
                        if nb < 0:
                            continue
                        val = gradient[nb]
                        if self.walk_up_gradient and val > best_val:
                            best_val = val
                            best_nb = nb
                        elif not self.walk_up_gradient and val < best_val:
                            best_val = val
                            best_nb = nb
                    positions[i] = best_nb

                distances[i] += 1.0

            if halt_dist > 0 and (distances >= halt_dist).all():
                break

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
