"""H3-native mesh backend for the Salmon IBM.

``H3Mesh`` duck-types :class:`salmon_ibm.mesh.TriMesh` and
:class:`salmon_ibm.hexsim.HexMesh` — same attribute names and shapes —
so the Numba movement / accumulator kernels work unchanged.  Cells are
addressed by compact integer indices ``0..N-1``; their globally-unique
H3 IDs are carried in :attr:`h3_ids` as ``uint64`` for environment
binding and viewer rendering.

Phase 1 of ``docs/superpowers/plans/2026-04-24-h3mesh-backend.md``.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import h3


class H3Mesh:
    """Hexagonal mesh built from explicit H3 cell IDs.

    Parameters
    ----------
    h3_ids
        ``(N,)`` ``uint64`` — H3 cell IDs as integers.
    centroids
        ``(N, 2)`` ``float64`` — ``[lat, lon]`` per cell.
    neighbors
        ``(N, 6)`` ``int32`` — compact mesh-index of each ring-1
        neighbour, with ``-1`` sentinel for missing slots (pentagons
        have 5; cells at the mesh boundary may have fewer).
    water_mask
        ``(N,)`` ``bool`` — True where the cell is water.
    depth
        ``(N,)`` ``float32`` — per-cell depth in metres.
    areas
        ``(N,)`` ``float32`` — per-cell area in m² (from
        :func:`h3.cell_area`).
    resolution
        H3 resolution (0–15) — same for every cell.
    """

    MAX_NBRS = 6  # pentagons fit with a -1 sentinel in the 6th slot

    def __init__(
        self,
        h3_ids: np.ndarray,
        centroids: np.ndarray,
        neighbors: np.ndarray,
        water_mask: np.ndarray,
        depth: np.ndarray,
        areas: np.ndarray,
        resolution: int,
        reach_id: np.ndarray | None = None,
        reach_names: list[str] | None = None,
    ) -> None:
        self.h3_ids = h3_ids
        self.centroids = centroids
        self.neighbors = neighbors
        self.water_mask = water_mask
        self.depth = depth
        self.areas = areas
        self.resolution = resolution

        # Per-cell reach IDs (optional — only populated for landscapes
        # built with the inSTREAM-polygon water mask; older NCs have
        # this missing).  -1 = land, 0..8 = inSTREAM reaches in the
        # order recorded in ``reach_names``, 9 = OpenBaltic.
        if reach_id is None:
            self.reach_id = np.full(len(h3_ids), -1, dtype=np.int8)
        else:
            self.reach_id = np.asarray(reach_id, dtype=np.int8)
        self.reach_names: list[str] = list(reach_names) if reach_names else []

        # Numba caches — same layout as TriMesh / HexMesh.
        self.centroids_c = np.ascontiguousarray(centroids)
        self._water_nbrs = neighbors
        self._water_nbr_count = (neighbors >= 0).sum(axis=1).astype(np.int32)

    def reach_name_of(self, idx: int) -> str:
        """Return the reach name for cell ``idx``, or ``"Land"`` / ``"Unknown"``."""
        rid = int(self.reach_id[idx])
        if rid < 0:
            return "Land"
        if 0 <= rid < len(self.reach_names):
            return self.reach_names[rid]
        return "Unknown"

    def cells_in_reach(self, reach_name: str) -> np.ndarray:
        """Return ``(N,)`` int array of compact indices for cells in ``reach_name``."""
        if reach_name not in self.reach_names:
            return np.array([], dtype=np.int64)
        rid = self.reach_names.index(reach_name)
        return np.where(self.reach_id == rid)[0]

    # --- duck-typed properties ---------------------------------------

    @property
    def n_cells(self) -> int:
        return len(self.h3_ids)

    @property
    def n_triangles(self) -> int:
        """Alias for ``n_cells`` — duck-types :class:`TriMesh`."""
        return self.n_cells

    # --- duck-typed methods ------------------------------------------

    def water_neighbors(self, idx: int) -> list[int]:
        row = self.neighbors[idx]
        return [int(n) for n in row if n >= 0]

    def find_triangle(self, lat: float, lon: float) -> int:
        """Return the compact mesh index containing ``(lat, lon)``, or ``-1``."""
        hid = h3.latlng_to_cell(float(lat), float(lon), self.resolution)
        hid_int = np.uint64(int(h3.str_to_int(hid)))
        hits = np.where(self.h3_ids == hid_int)[0]
        return int(hits[0]) if len(hits) else -1

    def metric_scale(self, lat: float) -> tuple[float, float]:
        """Return ``(metres / deg lon at lat, metres / deg lat)``.

        Mirrors :meth:`TriMesh.metric_scale` so the Phase-0 advection
        kernel works identically on H3 landscapes.
        """
        from .geomconst import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR
        return (M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat)),
                M_PER_DEG_LAT)

    def gradient(self, field: np.ndarray, idx: int) -> tuple[float, float]:
        """Approximate normalised ``(dlat, dlon)`` gradient of ``field`` at ``idx``.

        Centroid diffs are scaled by :meth:`metric_scale` so a degree of
        longitude doesn't outweigh a degree of latitude at mid-latitude
        — same correction Task 0.1 applies to ``_advection_numba``.
        Returns the zero vector when the cell has no valid neighbours
        or when the gradient magnitude is below ``1e-12``.
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

    # --- factories ---------------------------------------------------

    @classmethod
    def from_h3_cells(
        cls,
        h3_cells: Sequence[str],
        *,
        depth: np.ndarray | None = None,
        water_mask: np.ndarray | None = None,
        pentagon_policy: str = "raise",
    ) -> "H3Mesh":
        """Build a mesh from an explicit list of H3 cell IDs.

        Parameters
        ----------
        h3_cells
            Iterable of H3 cell strings (e.g. ``"891f7396c8fffff"``).
            Must be non-empty, must all share the same resolution.
        depth
            Optional ``(N,)`` per-cell depth in metres.  Defaults to zeros.
        water_mask
            Optional ``(N,)`` bool mask.  Defaults to all True.
        pentagon_policy
            * ``"raise"`` (default): refuse if any cell is an H3
              pentagon — pentagons have 5 neighbours and break some
              downstream invariants that assume exactly 6.
            * ``"skip"``: drop pentagon cells from the mesh; ``depth``
              and ``water_mask`` are sliced consistently.
            * ``"allow"``: include pentagons; their row in
              :attr:`neighbors` has 5 valid entries and one ``-1``
              sentinel.
        """
        if pentagon_policy not in ("raise", "skip", "allow"):
            raise ValueError(
                f"pentagon_policy must be 'raise', 'skip', or 'allow'; "
                f"got {pentagon_policy!r}"
            )
        if len(h3_cells) == 0:
            raise ValueError("H3Mesh.from_h3_cells: empty cell list")

        # Pentagon guard. Compute the keep-mask BEFORE mutating h3_cells
        # so depth/water_mask get filtered against the original ordering.
        h3_cells = list(h3_cells)
        is_penta = [h3.is_pentagon(c) for c in h3_cells]
        if any(is_penta):
            if pentagon_policy == "raise":
                n_pentas = sum(is_penta)
                raise ValueError(
                    f"{n_pentas} pentagon cell(s) in input; pass "
                    f"pentagon_policy='skip' or 'allow' to proceed."
                )
            elif pentagon_policy == "skip":
                keep = np.array([not p for p in is_penta], dtype=bool)
                h3_cells = [c for c, k in zip(h3_cells, keep) if k]
                if depth is not None:
                    depth = np.asarray(depth)[keep]
                if water_mask is not None:
                    water_mask = np.asarray(water_mask)[keep]
                if not h3_cells:
                    raise ValueError(
                        "all input cells were pentagons — nothing left "
                        "after pentagon_policy='skip'"
                    )
            # "allow": fall through deliberately — pentagons stay in the
            # cell list and get 5 valid neighbours + a -1 sentinel in
            # the 6th slot.

        n = len(h3_cells)
        # Use Python int consistently at the dict boundary — numpy uint64
        # hash-equals Python int but mixing them in dict keys has bitten
        # people before, so be explicit.
        h3_ids_pyint = [int(h3.str_to_int(c)) for c in h3_cells]
        h3_ids = np.array(h3_ids_pyint, dtype=np.uint64)

        resolution = h3.get_resolution(h3_cells[0])
        centroids = np.array(
            [h3.cell_to_latlng(c) for c in h3_cells],
            dtype=np.float64,
        )

        # Reverse lookup: H3 ID (Python int) → compact index.
        id_to_idx = {cid: i for i, cid in enumerate(h3_ids_pyint)}

        # Neighbours from H3's ring-1 query.  Rows for cells at the
        # mesh boundary or for pentagons may have fewer than 6 entries
        # — fill the unused slots with -1, but **compact valid entries
        # into the first slots** so movement kernels can read
        # ``water_nbrs[c, :count]`` without skipping holes.  Without
        # this compaction, an interior cell whose 0th h3.grid_ring
        # neighbour happens to fall outside the bbox would have
        # ``water_nbrs[c, 0] == -1`` even though count > 0, and
        # ``_step_directed_numba`` would write ``best_nbr = -1`` into
        # the agent's tri_idx.
        neighbours = np.full((n, cls.MAX_NBRS), -1, dtype=np.int32)
        for i, cell in enumerate(h3_cells):
            slot = 0
            for nb in h3.grid_ring(cell, 1):
                nb_int = int(h3.str_to_int(nb))
                idx = id_to_idx.get(nb_int)
                if idx is not None:
                    neighbours[i, slot] = idx
                    slot += 1

        if water_mask is None:
            water_mask = np.ones(n, dtype=bool)
        else:
            water_mask = np.asarray(water_mask, dtype=bool)

        if depth is None:
            depth = np.zeros(n, dtype=np.float32)
        else:
            depth = np.asarray(depth, dtype=np.float32)

        areas = np.array(
            [h3.cell_area(c, unit="m^2") for c in h3_cells],
            dtype=np.float32,
        )

        return cls(
            h3_ids=h3_ids,
            centroids=centroids,
            neighbors=neighbours,
            water_mask=water_mask,
            depth=depth,
            areas=areas,
            resolution=resolution,
        )

    @classmethod
    def from_polygon(
        cls,
        polygon: "h3.LatLngPoly",
        resolution: int,
        *,
        depth: dict[int, float] | None = None,
        water_mask: dict[int, bool] | None = None,
        pentagon_policy: str = "raise",
    ) -> "H3Mesh":
        """Build a mesh by H3-tessellating a lat/lon polygon.

        Parameters
        ----------
        polygon
            ``h3.LatLngPoly`` whose interior is filled with H3 cells at
            ``resolution``.  Use :func:`h3.LatLngPoly` with a list of
            ``(lat, lon)`` corner tuples (closed ring — first point
            repeated at the end).
        resolution
            H3 resolution (0–15).  See
            :func:`h3.average_hexagon_edge_length` for cell size at
            each resolution.
        depth, water_mask
            Optional ``{h3_int_id: value}`` lookups.  Cells not in the
            dict get zero / True defaults.
        pentagon_policy
            Forwarded to :meth:`from_h3_cells` — see that docstring for
            the three options (``raise`` / ``skip`` / ``allow``).
        """
        cells = list(h3.polygon_to_cells(polygon, resolution))
        if not cells:
            raise ValueError(
                "polygon produced zero H3 cells — degenerate polygon "
                "or resolution too coarse for the given extent"
            )

        # Resolve the optional dict-keyed inputs into per-cell arrays
        # before delegating to from_h3_cells.
        n = len(cells)
        depth_arr = np.zeros(n, dtype=np.float32)
        mask_arr = np.ones(n, dtype=bool)
        ids_int = [int(h3.str_to_int(c)) for c in cells]
        if depth is not None:
            for i, cid in enumerate(ids_int):
                if cid in depth:
                    depth_arr[i] = float(depth[cid])
        if water_mask is not None:
            for i, cid in enumerate(ids_int):
                if cid in water_mask:
                    mask_arr[i] = bool(water_mask[cid])

        return cls.from_h3_cells(
            cells,
            depth=depth_arr,
            water_mask=mask_arr,
            pentagon_policy=pentagon_policy,
        )
