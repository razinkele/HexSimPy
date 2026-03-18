"""Phase 3 event types: genetics, interactions, network, and remaining events."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from salmon_ibm.events import Event, register_event


@register_event("mutation")
@dataclass
class MutationEvent(Event):
    """Apply allele mutations at specified loci each timestep."""
    locus_name: str = ""
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
        genome = getattr(population, 'genome', None)
        if genome is None:
            return
        genome.mutate(
            self.locus_name, self.transition_matrix, mask=mask
        )


@register_event("transition")
@dataclass
class TransitionEvent(Event):
    """Modify a probabilistic trait using a transition matrix.

    transition_matrix[i, j] = P(category i -> category j) per timestep.
    Rows must sum to 1.0. Used for SEIR disease states, life stages, etc.
    """
    trait_name: str = ""
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
        traits = getattr(population, 'trait_mgr', None) or getattr(population, 'traits', None)
        if traits is None:
            return
        defn = traits.definitions[self.trait_name]
        n_categories = len(defn.categories)
        current = traits.get(self.trait_name)

        masked_indices = np.where(mask)[0]
        if len(masked_indices) == 0:
            return

        rng = landscape.get("rng", np.random.default_rng())
        current_cats = current[masked_indices]
        new_cats = np.empty_like(current_cats)

        for cat_val in range(n_categories):
            cat_mask = current_cats == cat_val
            if not cat_mask.any():
                continue
            n = cat_mask.sum()
            probs = self.transition_matrix[cat_val]
            drawn = rng.choice(n_categories, size=n, p=probs)
            new_cats[cat_mask] = drawn

        traits.set(self.trait_name, new_cats, mask=mask)


@register_event("generated_hexmap")
@dataclass
class GeneratedHexmapEvent(Event):
    """Create or update a hex map using an algebraic expression."""
    expression: str = ""
    output_name: str = ""

    def execute(self, population, landscape, t, mask):
        from salmon_ibm.accumulators import _validate_expression, _SAFE_MATH
        _validate_expression(self.expression)
        namespace = dict(_SAFE_MATH)
        namespace["t"] = t
        if "density" in self.expression:
            n_cells = landscape.get("n_cells", 0)
            if n_cells > 0:
                alive = population.alive if hasattr(population, 'alive') else mask
                positions = population.tri_idx[alive & mask]
                density = np.bincount(positions, minlength=n_cells).astype(np.float64)
                namespace["density"] = density
        for key, value in landscape.items():
            if isinstance(value, np.ndarray) and key not in namespace:
                namespace[key] = value
        result = eval(self.expression, {"__builtins__": {}}, namespace)
        landscape[self.output_name] = np.asarray(result, dtype=np.float64)


@register_event("range_dynamics")
@dataclass
class RangeDynamicsEvent(Event):
    """Adjust agent territories based on resource availability."""
    mode: str = "expand"
    resource_map_name: str = "resources"
    resource_threshold: float = 0.0
    max_range_size: int = 50

    def execute(self, population, landscape, t, mask):
        range_alloc = getattr(population, 'ranges', None)
        if range_alloc is None:
            return
        resource_map = landscape.get(self.resource_map_name)
        if resource_map is None:
            return
        indices = np.where(mask)[0]
        for i in indices:
            agent_range = range_alloc.get_range(i)
            if agent_range is None:
                continue
            if self.mode == "expand":
                self._expand(i, agent_range, resource_map, range_alloc, landscape)
            elif self.mode == "contract":
                self._contract(i, agent_range, resource_map, range_alloc)
            elif self.mode == "shift":
                self._contract(i, agent_range, resource_map, range_alloc)
                self._expand(i, agent_range, resource_map, range_alloc, landscape)

    def _expand(self, agent_idx, agent_range, resource_map, range_alloc, landscape):
        mesh = landscape.get("mesh")
        if mesh is None:
            return
        current_cells = set(agent_range.cells)
        if len(current_cells) >= self.max_range_size:
            return
        for cell in list(current_cells):
            count = mesh._water_nbr_count[cell]
            neighbors = mesh._water_nbrs[cell, :count]
            for nb in neighbors:
                if nb not in current_cells and resource_map[nb] >= self.resource_threshold:
                    if range_alloc.try_add_cell(agent_idx, nb):
                        current_cells.add(nb)
                        if len(current_cells) >= self.max_range_size:
                            return

    def _contract(self, agent_idx, agent_range, resource_map, range_alloc):
        for cell in list(agent_range.cells):
            if resource_map[cell] < self.resource_threshold:
                range_alloc.release_cell(agent_idx, cell)


@register_event("set_affinity")
@dataclass
class SetAffinityEvent(Event):
    """Set movement affinity for agents."""
    affinity_type: str = "spatial"
    affinity_map_name: str | None = None
    strength: float = 1.0

    def execute(self, population, landscape, t, mask):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return
        if self.affinity_type == "spatial" and self.affinity_map_name:
            if not hasattr(population, 'spatial_affinity'):
                return
            affinity_map = landscape.get(self.affinity_map_name)
            if affinity_map is None:
                return
            population.spatial_affinity[indices] = self.strength
            population.spatial_affinity_map_name = self.affinity_map_name
        elif self.affinity_type == "group":
            if not hasattr(population, 'group_affinity'):
                return
            population.group_affinity[indices] = self.strength
