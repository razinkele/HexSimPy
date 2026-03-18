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
