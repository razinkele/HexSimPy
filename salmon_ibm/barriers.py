"""Barrier map: edge-based movement restrictions on the hex grid."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from heximpy.hxnparser import Barrier, read_barriers


class BarrierOutcome(NamedTuple):
    p_mortality: float
    p_deflection: float
    p_transmission: float

    @staticmethod
    def impassable() -> BarrierOutcome:
        return BarrierOutcome(0.0, 1.0, 0.0)

    @staticmethod
    def lethal() -> BarrierOutcome:
        return BarrierOutcome(1.0, 0.0, 0.0)


@dataclass
class BarrierClass:
    name: str
    forward: BarrierOutcome
    reverse: BarrierOutcome


class BarrierMap:
    """Edge-keyed barrier lookup for movement enforcement."""

    def __init__(self):
        self._edges: dict[tuple[int, int], BarrierOutcome] = {}

    def add_edge(self, from_cell: int, to_cell: int, outcome: BarrierOutcome) -> None:
        self._edges[(from_cell, to_cell)] = outcome

    def check(self, from_cell: int, to_cell: int) -> BarrierOutcome | None:
        return self._edges.get((from_cell, to_cell))

    def has_barriers(self) -> bool:
        return len(self._edges) > 0

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    @classmethod
    def empty(cls) -> "BarrierMap":
        """No-barrier map — for landscapes without barriers (e.g. Nemunas H3)."""
        return cls()

    @classmethod
    def from_hbf_hexsim(cls, path, mesh, class_config=None):
        """Load barriers from HexSim ``.hbf`` file into compact mesh indices.

        Requires a :class:`HexMesh` (reads ``_ncols``, ``_nrows``,
        ``_full_to_compact``).  Raises ``TypeError`` on any other mesh
        backend — H3 / Tri landscapes should use their own sibling loader
        (``BarrierMap.from_csv_h3`` once Phase 4 lands, or ``empty()``
        for barrier-free scenarios).
        """
        if not hasattr(mesh, "_ncols"):
            raise TypeError(
                f"BarrierMap.from_hbf_hexsim requires a HexMesh; "
                f"got {type(mesh).__name__}"
            )
        barriers = read_barriers(str(path))
        bmap = cls()
        from salmon_ibm.hexsim import _hex_neighbors_offset
        for b in barriers:
            if b.hex_id not in mesh._full_to_compact:
                continue
            compact_from = mesh._full_to_compact[b.hex_id]
            full_from = mesh._water_full_idx[compact_from]
            r, c = int(full_from // mesh._ncols), int(full_from % mesh._ncols)
            dir_nbrs = _hex_neighbors_offset(r, c, mesh._ncols, mesh._nrows, mesh._n_data)
            if b.edge >= len(dir_nbrs):
                continue
            full_to = dir_nbrs[b.edge]
            if full_to not in mesh._full_to_compact:
                continue
            compact_to = mesh._full_to_compact[full_to]
            if class_config and b.class_name in class_config:
                bc = class_config[b.class_name]
                fwd, rev = bc.forward, bc.reverse
            else:
                fwd = BarrierOutcome.impassable()
                rev = BarrierOutcome.impassable()
            bmap.add_edge(compact_from, compact_to, fwd)
            bmap.add_edge(compact_to, compact_from, rev)
        return bmap

    def to_arrays(self, mesh):
        """Convert to array form for vectorized movement."""
        n_cells = mesh.n_cells if hasattr(mesh, 'n_cells') else mesh.n_triangles
        max_nbrs = mesh.neighbors.shape[1] if hasattr(mesh, 'neighbors') else mesh._water_nbrs.shape[1]
        mort = np.zeros((n_cells, max_nbrs), dtype=np.float64)
        defl = np.zeros((n_cells, max_nbrs), dtype=np.float64)
        trans = np.ones((n_cells, max_nbrs), dtype=np.float64)
        nbrs = mesh.neighbors if hasattr(mesh, 'neighbors') else mesh._water_nbrs
        for (fc, tc), outcome in self._edges.items():
            nbr_row = nbrs[fc]
            slots = np.where(nbr_row == tc)[0]
            if len(slots) > 0:
                slot = slots[0]
                mort[fc, slot] = outcome.p_mortality
                defl[fc, slot] = outcome.p_deflection
                trans[fc, slot] = outcome.p_transmission
        return mort, defl, trans
