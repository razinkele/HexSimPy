"""Multi-species interaction system: population management and encounter events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from salmon_ibm.events import Event, register_event


class MultiPopulationManager:
    """Manages multiple named populations on a shared landscape.

    Provides cross-population spatial queries using cell-based hashing.
    """

    def __init__(self):
        self.populations: dict[str, Any] = {}  # name -> Population
        self._cell_index: dict[
            str, dict[int, np.ndarray]
        ] = {}  # pop_name -> {cell_id -> agent_indices}

    def register(self, name_or_population, population=None) -> None:
        """Register a population.

        Supports two calling conventions:
          - register("name", population)  — explicit name
          - register(population)           — uses population.name attribute
        """
        if population is None:
            # Single-argument form: register(population)
            population = name_or_population
            name = population.name
        else:
            # Two-argument form: register("name", population)
            name = name_or_population
        self.populations[name] = population

    def get(self, name: str):
        return self.populations[name]

    def build_cell_index(self, name: str) -> None:
        """Build cell-based spatial hash for a population.

        Maps cell_id -> array of alive agent indices at that cell.
        """
        pop = self.populations[name]
        alive = pop.alive
        positions = pop.tri_idx  # cell indices
        index: dict[int, list[int]] = {}
        alive_indices = np.where(alive)[0]
        for i in alive_indices:
            cell = int(positions[i])
            if cell not in index:
                index[cell] = []
            index[cell].append(i)
        self._cell_index[name] = {
            k: np.array(v, dtype=np.int64) for k, v in index.items()
        }

    def agents_at_cell(self, name: str, cell_id: int) -> np.ndarray:
        """Return indices of alive agents from population `name` at `cell_id`."""
        idx = self._cell_index.get(name, {})
        return idx.get(cell_id, np.array([], dtype=np.int64))

    def co_located_pairs(
        self, pop_a: str, pop_b: str
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Find cells where both populations have agents.

        Returns list of (agents_a, agents_b) for each shared cell.
        """
        self.build_cell_index(pop_a)
        self.build_cell_index(pop_b)
        idx_a = self._cell_index[pop_a]
        idx_b = self._cell_index[pop_b]
        shared_cells = set(idx_a.keys()) & set(idx_b.keys())
        pairs = []
        for cell in shared_cells:
            pairs.append((idx_a[cell], idx_b[cell]))
        return pairs


class InteractionOutcome(Enum):
    PREDATION = "predation"  # prey dies, predator gains resource
    COMPETITION = "competition"  # loser incurs penalty
    DISEASE = "disease"  # pathogen transmission (handled by TransitionEvent)


@register_event("interaction")
@dataclass
class InteractionEvent(Event):
    """Probabilistic encounter between two co-located populations.

    Follows the Event ABC. Receives the MultiPopulationManager via
    ``landscape["multi_pop_mgr"]`` and the RNG via ``landscape["rng"]``.
    The ``population`` parameter in execute() is the primary population;
    the partner population is retrieved from the multi-pop manager.
    """

    pop_a_name: str = ""  # e.g., "predator"
    pop_b_name: str = ""  # e.g., "prey"
    encounter_probability: float = 0.1  # P(encounter) per co-located pair
    outcome: InteractionOutcome = InteractionOutcome.PREDATION
    resource_gain_acc: str | None = None
    resource_gain_amount: float = 0.0
    penalty_acc: str | None = None  # accumulator to decrement on loser
    penalty_amount: float = 0.0

    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
        """Run encounters. Retrieves MultiPopulationManager from landscape."""
        if not self.pop_a_name or not self.pop_b_name:
            return
        multi_pop_mgr = landscape["multi_pop_mgr"]
        rng = landscape.get("rng", np.random.default_rng())
        pairs = multi_pop_mgr.co_located_pairs(self.pop_a_name, self.pop_b_name)
        pop_a = multi_pop_mgr.get(self.pop_a_name)
        pop_b = multi_pop_mgr.get(self.pop_b_name)

        stats = {"encounters": 0, "kills": 0}

        for agents_a, agents_b in pairs:
            # Each agent in A encounters each agent in B with probability p
            for a_idx in agents_a:
                for b_idx in agents_b:
                    if not pop_b.alive[b_idx]:
                        continue  # Skip already-dead prey
                    if rng.random() < self.encounter_probability:
                        stats["encounters"] += 1
                        if self.outcome == InteractionOutcome.PREDATION:
                            pop_b.alive[b_idx] = False
                            stats["kills"] += 1
                            if (
                                self.resource_gain_acc
                                and pop_a.accumulator_mgr is not None
                            ):
                                acc_idx = pop_a.accumulator_mgr._resolve_idx(
                                    self.resource_gain_acc
                                )
                                pop_a.accumulator_mgr.data[a_idx, acc_idx] += (
                                    self.resource_gain_amount
                                )

        # Store stats in landscape for observability (previously returned but discarded)
        interaction_stats = landscape.setdefault("interaction_stats", [])
        interaction_stats.append({"event": self.name, "t": t, **stats})
