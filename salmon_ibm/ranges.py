"""Range allocation: non-overlapping territory management on hex grids."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class AgentRange:
    """A contiguous set of hex cells owned by one agent."""
    owner: int  # agent index
    cells: set[int] = field(default_factory=set)
    resource_total: float = 0.0


class RangeAllocator:
    """Manages non-overlapping territory allocation on a hex mesh.

    Each cell can be owned by at most one agent. Agents expand their
    range by claiming adjacent unoccupied cells that meet resource thresholds.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        n_cells = mesh.n_cells if hasattr(mesh, 'n_cells') else mesh.n_triangles
        self._cell_owner = np.full(n_cells, -1, dtype=np.int32)  # -1 = unoccupied
        self._ranges: dict[int, AgentRange] = {}

    @property
    def n_occupied(self) -> int:
        return int((self._cell_owner >= 0).sum())

    def get_range(self, agent_idx: int) -> AgentRange | None:
        return self._ranges.get(agent_idx)

    def owner_of(self, cell_id: int) -> int:
        return int(self._cell_owner[cell_id])

    def is_available(self, cell_id: int) -> bool:
        return self._cell_owner[cell_id] == -1

    def allocate_cell(self, agent_idx: int, cell_id: int) -> bool:
        """Try to claim a single cell for an agent. Returns True if successful."""
        if not self.is_available(cell_id):
            return False
        self._cell_owner[cell_id] = agent_idx
        if agent_idx not in self._ranges:
            self._ranges[agent_idx] = AgentRange(owner=agent_idx)
        self._ranges[agent_idx].cells.add(cell_id)
        return True

    def release_cell(self, agent_idx: int, cell_id: int) -> None:
        """Release a single cell from an agent's range."""
        if self._cell_owner[cell_id] == agent_idx:
            self._cell_owner[cell_id] = -1
            if agent_idx in self._ranges:
                self._ranges[agent_idx].cells.discard(cell_id)
                if not self._ranges[agent_idx].cells:
                    del self._ranges[agent_idx]

    def release_all(self, agent_idx: int) -> None:
        """Release all cells owned by an agent."""
        rng = self._ranges.pop(agent_idx, None)
        if rng:
            for cell in rng.cells:
                self._cell_owner[cell] = -1

    def try_add_cell(self, agent_idx: int, cell_id: int) -> bool:
        """Alias for allocate_cell (used by RangeDynamicsEvent)."""
        return self.allocate_cell(agent_idx, cell_id)

    def expand_range(self, agent_idx: int, resource_map: np.ndarray,
                     resource_threshold: float = 0.0,
                     max_cells: int = 50) -> int:
        """Expand an agent's territory by claiming adjacent viable cells.

        Uses BFS from current range cells. Stops when max_cells reached
        or no more viable neighbors available.

        Returns number of cells added.
        """
        if agent_idx not in self._ranges:
            return 0

        current = self._ranges[agent_idx].cells
        if len(current) >= max_cells:
            return 0

        added = 0
        frontier = set()
        for cell in current:
            count = self.mesh._water_nbr_count[cell]
            for j in range(count):
                nb = int(self.mesh._water_nbrs[cell, j])
                if nb >= 0 and self.is_available(nb) and nb not in current:
                    frontier.add(nb)

        # Sort frontier by resource value (highest first)
        if not frontier:
            return 0
        frontier_list = sorted(frontier, key=lambda c: resource_map[c], reverse=True)

        for cell in frontier_list:
            if resource_map[cell] < resource_threshold:
                continue
            if not self.is_available(cell):
                continue
            if self.allocate_cell(agent_idx, cell):
                added += 1
                if len(self._ranges[agent_idx].cells) >= max_cells:
                    break

        return added

    def contract_range(self, agent_idx: int, resource_map: np.ndarray,
                       resource_threshold: float = 0.0) -> int:
        """Release cells below resource threshold from an agent's range.

        Returns number of cells released.
        """
        if agent_idx not in self._ranges:
            return 0

        released = 0
        cells_to_release = []
        for cell in self._ranges[agent_idx].cells:
            if resource_map[cell] < resource_threshold:
                cells_to_release.append(cell)

        for cell in cells_to_release:
            self.release_cell(agent_idx, cell)
            released += 1

        return released

    def compute_resources(self, agent_idx: int, resource_map: np.ndarray) -> float:
        """Total resource value in an agent's range."""
        rng = self._ranges.get(agent_idx)
        if rng is None:
            return 0.0
        cells = list(rng.cells)
        if not cells:
            return 0.0
        return float(resource_map[cells].sum())

    def summary(self) -> dict:
        """Summary statistics of range allocation."""
        sizes = [len(r.cells) for r in self._ranges.values()]
        return {
            "n_agents_with_ranges": len(self._ranges),
            "n_occupied_cells": self.n_occupied,
            "mean_range_size": float(np.mean(sizes)) if sizes else 0.0,
            "max_range_size": max(sizes) if sizes else 0,
        }
