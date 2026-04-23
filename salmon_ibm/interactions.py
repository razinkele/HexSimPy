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
        """Run encounters. Retrieves MultiPopulationManager from landscape.

        Vectorized per-cell using "Option 2" semantics: for each shared cell,
        a dice-roll matrix rng.random((|A|, |B|)) determines candidate
        encounters. For each B column, only the FIRST row (smallest a_idx)
        with a successful roll claims the kill — matching the natural
        iteration order of the prior nested Python loop.

        RNG consumption order differs from the prior scalar version: the
        scalar path drew one random per (a, b) iteration; this path draws
        |A|*|B| randoms in one matrix per shared cell. Kill counts under a
        given seed will differ numerically, but distributions are
        statistically equivalent.
        """
        if not self.pop_a_name or not self.pop_b_name:
            return
        multi_pop_mgr = landscape["multi_pop_mgr"]
        rng = landscape.get("rng", np.random.default_rng())
        pairs = multi_pop_mgr.co_located_pairs(self.pop_a_name, self.pop_b_name)
        pop_a = multi_pop_mgr.get(self.pop_a_name)
        pop_b = multi_pop_mgr.get(self.pop_b_name)

        total_encounters = 0
        total_kills = 0

        # Buffer resource gains per A agent; commit in one scatter at end.
        resource_gains = None
        acc_idx = None
        if (
            self.resource_gain_acc
            and pop_a.accumulator_mgr is not None
            and self.outcome == InteractionOutcome.PREDATION
        ):
            acc_idx = pop_a.accumulator_mgr._resolve_idx(self.resource_gain_acc)
            resource_gains = np.zeros(pop_a.pool.n, dtype=np.float64)

        for agents_a, agents_b in pairs:
            if agents_a.size == 0 or agents_b.size == 0:
                continue
            # Filter already-dead B agents (possible if same B index appeared
            # in an earlier cell's pairs — defensive).
            alive_b_mask = pop_b.alive[agents_b]
            if not alive_b_mask.any():
                continue
            live_b = agents_b[alive_b_mask]

            # Dice-roll matrix: (|A|, |B|) of uniform[0, 1).
            rolls = rng.random((agents_a.size, live_b.size))
            hits = rolls < self.encounter_probability
            total_encounters += int(hits.sum())

            if self.outcome != InteractionOutcome.PREDATION:
                # Non-predation outcomes (competition, disease) were non-
                # functional in the scalar path too — skip rather than
                # silently double-count.
                continue

            # Option 2 dedup: first A row with a hit wins each B column.
            col_has_hit = hits.any(axis=0)
            if not col_has_hit.any():
                continue
            winning_a_rows = np.argmax(hits, axis=0)

            b_cols = np.where(col_has_hit)[0]
            a_rows = winning_a_rows[b_cols]

            # Global agent indices.
            a_globals = agents_a[a_rows]
            b_globals = live_b[b_cols]

            pop_b.alive[b_globals] = False
            total_kills += int(b_globals.size)

            if resource_gains is not None:
                np.add.at(resource_gains, a_globals, self.resource_gain_amount)

        # Commit resource gains in one pass (contiguous write).
        if resource_gains is not None and acc_idx is not None:
            pop_a.accumulator_mgr.data[acc_idx, :] += resource_gains

        interaction_stats = landscape.setdefault("interaction_stats", [])
        interaction_stats.append({
            "event": self.name, "t": t,
            "encounters": total_encounters, "kills": total_kills,
        })
