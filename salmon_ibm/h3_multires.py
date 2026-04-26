"""Variable-resolution H3 mesh — scaffold (v1.2.8).

Phase 1 scaffolding for the multi-resolution H3 backend.  See
``docs/h3-multi-resolution-feasibility.md`` for the design and
``docs/h3-multires-roadmap.md`` for the implementation tracking.

Key differences from ``H3Mesh``:

* **Per-cell resolution.**  Each cell carries its own H3 resolution
  (``resolutions: (N,) int8``), supporting e.g. rivers at res 11
  (~28 m edge) inside the same mesh as the lagoon at res 9 (~200 m
  edge).
* **Ragged neighbour table.**  ``H3Mesh`` uses ``(N, 6)`` because every
  cell has the same neighbour count.  Cross-resolution boundaries
  produce up to ~12 neighbours for the *coarse* side (it can border
  several finer children of the same nominal ring neighbour), so we
  store neighbours in CSR form: ``nbr_starts: (N+1,) int32`` indexes
  into ``nbr_idx: (M,) int32``.
* **Compatibility surface.**  A padded ``(N, MAX_NBRS) int32``
  ``neighbors`` view is exposed for legacy numba kernels — same
  ``-1``-sentinel convention as ``H3Mesh.neighbors``, but
  ``MAX_NBRS`` is bumped to 64 (was 12 in the v1.2.8 scaffold;
  v1.2.10 raised it after the v1.2.9 overflow guard caught real
  rows of 31+ entries at res-10/res-8 boundaries — see comment on
  ``MAX_NBRS`` for the math).

Status:

* Cross-resolution neighbour finder — **implemented** with tests.
* Mesh-from-reach-polygons builder — **implemented** for the simplest
  2-reach case; multi-source build (inSTREAM 9-reach + Baltic) is a
  follow-up.
* Simulation integration — **not yet**.  The ``Simulation.__init__``
  H3 branch still uses ``H3Mesh``.  Wiring this class through the
  numba movement kernels needs the kernels to read CSR (or read the
  padded compat view).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Iterable

import h3
import numpy as np


# Cap for the padded compat view.  At a coarse-fine boundary, a
# coarse cell of resolution r has up to 6 same-res ring-1 neighbours
# PLUS up to ~7^Δr fine children at each missing ring slot, where
# Δr is the resolution drop.  In practice (res-10 rivers ↔ res-8
# OpenBaltic, drop=2) we see ~30 neighbours; the production config
# (res-11 rivers ↔ res-8 OpenBaltic, drop=3) could see ~60.  Set 64
# as a generous cap that covers the production case with margin.
# Per-mesh memory cost: 36 k cells × 64 cols × 4 bytes ≈ 9 MB,
# fits in L3 trivially.  v1.2.8 had MAX_NBRS=12, v1.2.10 raised
# this after the v1.2.9 overflow guard caught real overflows.
MAX_NBRS = 64


@dataclass
class _CellSpec:
    """Internal: a (cell, resolution) pair carrying optional payload."""
    h3_str: str
    resolution: int
    depth: float = 0.0
    water: bool = True
    reach_id: int = -1


def _h3_int(cell: str) -> int:
    """Convenience: H3 string → Python int."""
    return int(h3.str_to_int(cell))


# ---------------------------------------------------------------------------
# Cross-resolution neighbour finder — the core algorithmic piece
# ---------------------------------------------------------------------------


def find_cross_res_neighbours(
    cells: Sequence[str],
    *,
    max_resolution_drop: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CSR neighbour table for a mixed-resolution H3 cell list.

    For each input cell at its own resolution, returns the list of
    *compact mesh indices* of cells that share a real-world edge with
    it — possibly at a different resolution.

    Parameters
    ----------
    cells
        Iterable of H3 cell strings (e.g. ``"891f7396c8fffff"``).
        Cells may be at different resolutions.  The list defines the
        compact-index ordering: ``cells[i]`` ⇔ index ``i``.
    max_resolution_drop
        Search depth for coarse-side parent lookups.  When a fine
        cell's same-resolution ring neighbour is missing from the mesh,
        we walk *up* (lower resolution) by at most this many steps to
        find a coarser cell that contains the missing neighbour's
        centroid.  Default 3 — enough for a res-11→res-9 transition
        in the inSTREAM-river-vs-lagoon use case.  Capped to avoid
        false matches in unrelated coarse zones.

    Returns
    -------
    nbr_starts : ``(N+1,) int32``
        CSR row pointers.  Cell ``i``'s neighbour indices are
        ``nbr_idx[nbr_starts[i]:nbr_starts[i+1]]``.
    nbr_idx : ``(M,) int32``
        Flat list of neighbour indices.  All entries are valid (no
        ``-1`` sentinels in this representation).

    Algorithm
    ---------
    A cell ``c`` at resolution ``r_c`` may have neighbours at:

    1. *Same resolution* — `h3.grid_ring(c, 1)` ∩ the mesh.
       Standard case, captures all-same-res zones.
    2. *Coarser resolution* (``r_n < r_c``).  When a ring neighbour
       at res ``r_c`` doesn't exist in the mesh, its centroid still
       has a position in the world; the cell *containing* that
       position at a lower resolution may exist.
       :func:`h3.latlng_to_cell(lat, lon, r_lower)` resolves the
       coarse cell at any target resolution; we try `r_c-1, r_c-2, …`
       until either a match is found or ``max_resolution_drop`` is
       exceeded.
    3. *Finer resolution* (``r_n > r_c``).  If ``c`` is on the
       boundary of a coarse zone, the ring of res-``r_c`` cells around
       it may contain a fine-resolution zone where the actual
       neighbours are children of the missing ring members.  We
       enumerate `h3.cell_to_children(missing_ring_member, r_finer)`
       up to ``r_c + max_resolution_drop`` and intersect with the
       mesh.

    Behaviour at resolution boundaries is symmetric: the fine cell
    sees its coarse parent neighbour, and the coarse cell sees the
    fine children of its same-res ring neighbours that fall inside
    the fine zone.

    Notes
    -----
    * The function is O(N · k · D) where k=6 (ring-1 size) and
      D=`max_resolution_drop`.  At ~50 k cells with D=3 this is
      ~1 second.  Pre-built id-to-index dict makes the hot loop
      fast.
    * Pentagons are handled: their `grid_ring` returns 5 entries
      not 6, the rest of the algorithm doesn't care.
    """
    cells = list(cells)
    n = len(cells)
    if n == 0:
        return np.array([0], dtype=np.int32), np.array([], dtype=np.int32)

    resolutions = np.array(
        [h3.get_resolution(c) for c in cells], dtype=np.int8
    )
    id_to_idx: dict[int, int] = {_h3_int(c): i for i, c in enumerate(cells)}

    rows: list[list[int]] = [[] for _ in range(n)]
    for i, c in enumerate(cells):
        r = int(resolutions[i])
        seen: set[int] = set()
        # ── Step 1: same-resolution ring-1 ──────────────────────
        try:
            ring = h3.grid_ring(c, 1)
        except Exception:
            ring = []
        for nb in ring:
            nb_int = _h3_int(nb)
            j = id_to_idx.get(nb_int)
            if j is not None and j != i and nb_int not in seen:
                rows[i].append(j)
                seen.add(nb_int)
                continue
            # Step 2: walk UP the resolution hierarchy.  The missing
            # ring member at res r might be enclosed by a coarser cell
            # in the mesh.
            try:
                lat, lon = h3.cell_to_latlng(nb)
            except Exception:
                continue
            for r_lower in range(r - 1, max(0, r - 1 - max_resolution_drop), -1):
                parent = h3.latlng_to_cell(lat, lon, r_lower)
                p_int = _h3_int(parent)
                if p_int == _h3_int(c):
                    # Self — happens when the coarse cell IS this cell.
                    break
                k = id_to_idx.get(p_int)
                if k is not None and k != i and p_int not in seen:
                    rows[i].append(k)
                    seen.add(p_int)
                    break
            # Step 3: enumerate children of the missing ring member
            # at finer resolutions, intersect with the mesh.  Captures
            # the coarse-cell-bordering-fine-zone case.
            for r_finer in range(r + 1, min(15, r + 1 + max_resolution_drop) + 1):
                try:
                    children = h3.cell_to_children(nb, r_finer)
                except Exception:
                    children = []
                for ch in children:
                    ch_int = _h3_int(ch)
                    k = id_to_idx.get(ch_int)
                    if k is not None and k != i and ch_int not in seen:
                        rows[i].append(k)
                        seen.add(ch_int)

    max_row_len = max((len(r) for r in rows), default=0)
    if max_row_len > MAX_NBRS:
        raise RuntimeError(
            f"Cross-resolution neighbour finder produced a row with "
            f"{max_row_len} entries; MAX_NBRS is {MAX_NBRS}.  Bump "
            f"MAX_NBRS in h3_multires.py and rebuild the landscape NC, "
            f"or reduce ``max_resolution_drop`` to limit cross-res "
            f"reach."
        )

    # Convert lists to CSR
    nbr_starts = np.empty(n + 1, dtype=np.int32)
    nbr_starts[0] = 0
    for i in range(n):
        nbr_starts[i + 1] = nbr_starts[i] + len(rows[i])
    nbr_idx = np.empty(int(nbr_starts[-1]), dtype=np.int32)
    for i, row in enumerate(rows):
        if row:
            nbr_idx[nbr_starts[i]:nbr_starts[i + 1]] = row
    return nbr_starts, nbr_idx


# ---------------------------------------------------------------------------
# H3MultiResMesh — variable-resolution mesh class
# ---------------------------------------------------------------------------


class H3MultiResMesh:
    """Hexagonal mesh whose cells may have different H3 resolutions.

    Carries the same downstream contract as :class:`H3Mesh`
    (``centroids``, ``water_mask``, ``depth``, ``areas``, ``neighbors``,
    ``reach_id``, ``reach_names``) plus:

    * ``resolutions: (N,) int8`` — per-cell H3 resolution.
    * ``nbr_starts``, ``nbr_idx`` — CSR neighbour table (the canonical
      representation).  ``neighbors`` is a derived ``(N, MAX_NBRS)``
      padded view for legacy consumers.

    Build via :meth:`from_h3_cells` (explicit cell list, mixed
    resolutions allowed).  Higher-level builders that tessellate
    per-reach polygons at per-reach resolutions are in
    ``scripts/build_h3_multires_landscape.py`` (also Phase-1 scaffold).

    The numba movement kernels in ``salmon_ibm/movement.py`` currently
    read ``mesh.neighbors[i]`` as a ``(MAX_NBRS,) int32`` row with
    ``-1`` sentinels.  Setting ``MAX_NBRS = 12`` and emitting a
    consistent padded view keeps them working unchanged for the
    multi-res case (with at most 12 neighbours per cell).
    """

    MAX_NBRS = MAX_NBRS

    def __init__(
        self,
        h3_ids: np.ndarray,
        resolutions: np.ndarray,
        centroids: np.ndarray,
        nbr_starts: np.ndarray,
        nbr_idx: np.ndarray,
        water_mask: np.ndarray,
        depth: np.ndarray,
        areas: np.ndarray,
        reach_id: np.ndarray | None = None,
        reach_names: list[str] | None = None,
    ) -> None:
        self.h3_ids = h3_ids
        self.resolutions = np.asarray(resolutions, dtype=np.int8)
        self.centroids = centroids
        self.nbr_starts = np.asarray(nbr_starts, dtype=np.int32)
        self.nbr_idx = np.asarray(nbr_idx, dtype=np.int32)
        self.water_mask = water_mask.astype(bool)
        self.depth = depth.astype(np.float32)
        self.areas = areas.astype(np.float32)

        if reach_id is None:
            self.reach_id = np.full(len(h3_ids), -1, dtype=np.int8)
        else:
            self.reach_id = np.asarray(reach_id, dtype=np.int8)
        self.reach_names = list(reach_names) if reach_names else []

        # Derived padded neighbour view for legacy numba kernels that
        # expect (N, MAX_NBRS) int32 with -1 sentinels.  Worst case
        # 12 neighbours; rows shorter than that get -1 padding.
        self.neighbors = np.full((len(h3_ids), MAX_NBRS), -1, dtype=np.int32)
        for i in range(len(h3_ids)):
            row = nbr_idx[nbr_starts[i]:nbr_starts[i + 1]]
            n_row = min(len(row), MAX_NBRS)
            self.neighbors[i, :n_row] = row[:n_row]

        # Numba caches — same naming as H3Mesh so duck-typing works.
        self.centroids_c = np.ascontiguousarray(centroids)
        self._water_nbrs = self.neighbors
        self._water_nbr_count = (self.neighbors >= 0).sum(axis=1).astype(np.int32)

    @property
    def n_cells(self) -> int:
        return len(self.h3_ids)

    @property
    def n_triangles(self) -> int:
        """Alias for ``n_cells`` — duck-types ``TriMesh`` / ``H3Mesh``."""
        return self.n_cells

    def neighbours_of(self, idx: int) -> np.ndarray:
        """Return the *exact* neighbour list (no padding) for cell ``idx``."""
        return self.nbr_idx[self.nbr_starts[idx]:self.nbr_starts[idx + 1]]

    def water_neighbors(self, idx: int) -> list[int]:
        return [int(n) for n in self.neighbours_of(idx)]

    def reach_name_of(self, idx: int) -> str:
        rid = int(self.reach_id[idx])
        if rid < 0:
            return "Land"
        if 0 <= rid < len(self.reach_names):
            return self.reach_names[rid]
        return "Unknown"

    def cells_in_reach(self, reach_name: str) -> np.ndarray:
        if reach_name not in self.reach_names:
            return np.array([], dtype=np.int64)
        rid = self.reach_names.index(reach_name)
        return np.where(self.reach_id == rid)[0]

    def metric_scale(self, lat: float) -> tuple[float, float]:
        """``(metres / deg lon at lat, metres / deg lat)`` — duck-types H3Mesh."""
        from .geomconst import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR
        return (
            M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat)),
            M_PER_DEG_LAT,
        )

    def gradient(self, field: np.ndarray, idx: int) -> tuple[float, float]:
        """Approximate normalised ``(dlat, dlon)`` gradient of ``field`` at ``idx``.

        Same algorithm as :meth:`H3Mesh.gradient` — centroid diffs scaled
        by :meth:`metric_scale` so a degree of longitude doesn't outweigh
        a degree of latitude at mid-latitude.  Returns ``(0.0, 0.0)`` when
        the cell has no valid neighbours.
        """
        row = self.neighbors[idx]
        valid = row[row >= 0]
        if len(valid) == 0:
            return (0.0, 0.0)
        here = self.centroids[idx]
        scale_x, scale_y = self.metric_scale(float(here[0]))
        dlat = dlon = 0.0
        for n in valid:
            there = self.centroids[n]
            df = field[n] - field[idx]
            dlat += df * (there[0] - here[0]) * scale_y
            dlon += df * (there[1] - here[1]) * scale_x
        norm = (dlat * dlat + dlon * dlon) ** 0.5
        if norm < 1e-12:
            return (0.0, 0.0)
        return (dlat / norm, dlon / norm)

    @classmethod
    def from_h3_cells(
        cls,
        h3_cells: Sequence[str],
        *,
        depth: np.ndarray | None = None,
        water_mask: np.ndarray | None = None,
        reach_id: np.ndarray | None = None,
        reach_names: Sequence[str] | None = None,
        max_resolution_drop: int = 3,
    ) -> "H3MultiResMesh":
        """Build a mesh from an explicit list of H3 cell strings.

        Parameters
        ----------
        h3_cells
            Cell strings.  Resolutions may differ between cells.
        depth, water_mask, reach_id
            Optional per-cell payload arrays of length ``len(h3_cells)``.
        reach_names
            Optional reach-name decode list.  ``reach_id`` indexes into
            this list; ``-1`` is reserved for land.
        max_resolution_drop
            Passed through to :func:`find_cross_res_neighbours`.

        Notes
        -----
        Pentagons are accepted unconditionally (no ``pentagon_policy``
        argument like :class:`H3Mesh`) because the CSR neighbour
        representation handles 5-vs-6 ring sizes naturally.
        """
        h3_cells = list(h3_cells)
        n = len(h3_cells)
        if n == 0:
            raise ValueError("H3MultiResMesh.from_h3_cells: empty cell list")

        h3_ids = np.array(
            [_h3_int(c) for c in h3_cells], dtype=np.uint64,
        )
        resolutions = np.array(
            [h3.get_resolution(c) for c in h3_cells], dtype=np.int8
        )
        centroids = np.array(
            [h3.cell_to_latlng(c) for c in h3_cells], dtype=np.float64,
        )
        areas = np.array(
            [h3.cell_area(c, unit="m^2") for c in h3_cells], dtype=np.float32,
        )

        nbr_starts, nbr_idx = find_cross_res_neighbours(
            h3_cells, max_resolution_drop=max_resolution_drop,
        )

        if water_mask is None:
            water_mask = np.ones(n, dtype=bool)
        if depth is None:
            depth = np.zeros(n, dtype=np.float32)

        return cls(
            h3_ids=h3_ids,
            resolutions=resolutions,
            centroids=centroids,
            nbr_starts=nbr_starts,
            nbr_idx=nbr_idx,
            water_mask=np.asarray(water_mask, dtype=bool),
            depth=np.asarray(depth, dtype=np.float32),
            areas=areas,
            reach_id=reach_id,
            reach_names=list(reach_names) if reach_names else None,
        )
