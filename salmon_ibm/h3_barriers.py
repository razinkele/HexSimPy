"""H3-native barrier loading & geometric helpers.

A barrier is an **edge** between two adjacent H3 cells: when an agent
tries to step from cell A to cell B (or vice versa) and an edge
``(A, B)`` is in the :class:`BarrierMap`, a stochastic outcome
(mortality / deflection / transmission) fires.

This module provides:

* :func:`line_barrier_to_h3_edges` — convert a lat/lon line segment
  into the set of H3-cell-pair edges it crosses, so callers can spec
  a weir as two endpoints rather than enumerating every edge.
* :func:`load_h3_barrier_csv` — parse a barrier CSV and return the
  ``{(from_idx, to_idx): outcome}`` dict that
  :class:`salmon_ibm.barriers.BarrierMap` consumes.

Phase 4 of ``docs/superpowers/plans/2026-04-24-h3mesh-backend.md``.
"""
from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import h3
import numpy as np

from .geomconst import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometric helper: lat/lon line → H3 edge set
# ---------------------------------------------------------------------------


def line_barrier_to_h3_edges(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    resolution: int,
    *,
    bidirectional: bool = True,
    n_samples: int | None = None,
) -> list[tuple[str, str]]:
    """Return all H3 cell-pair edges crossed by the line from
    ``(lat1, lon1)`` to ``(lat2, lon2)`` at the given H3 resolution.

    The line is sampled densely in lat/lon space; each transition
    between adjacent H3 cells emits one ``(prev, curr)`` edge.

    Parameters
    ----------
    lat1, lon1, lat2, lon2
        Segment endpoints in WGS84 degrees.
    resolution
        H3 resolution (0–15) at which to find crossings.
    bidirectional
        When True (default), emit both ``(a, b)`` and ``(b, a)`` so a
        barrier blocks travel in either direction.  When False, only
        the traversal direction is emitted.
    n_samples
        Number of points along the segment to test.  Defaults to
        ``max(16, 4 · seg_length / h3_edge)`` so a long segment isn't
        under-sampled.

    Returns
    -------
    list[tuple[str, str]]
        H3 cell-string edge pairs.  Empty for zero-length segments.
        Order is insertion-into-a-set, so callers needing reproducibility
        should sort.

    Notes
    -----
    Sampling is linear in lat/lon space — accurate at scales where the
    H3 cell edge ≫ Earth-curvature-induced deviation over the segment
    length, i.e. sub-continental.  For ≳ 100 km segments at high
    latitude, prefer splitting the line or using a geodesic library.
    """
    edge_m = h3.average_hexagon_edge_length(resolution, unit="m")
    lat_mid = 0.5 * (lat1 + lat2)
    cos_lat_mid = math.cos(math.radians(lat_mid))
    dlat_m = (lat2 - lat1) * M_PER_DEG_LAT
    dlon_m = (lon2 - lon1) * M_PER_DEG_LON_EQUATOR * cos_lat_mid
    seg_m = math.hypot(dlat_m, dlon_m)
    if n_samples is None:
        n_samples = max(16, int(4 * seg_m / max(edge_m, 1.0)))

    edges: set[tuple[str, str]] = set()
    prev_cell: str | None = None
    for t in np.linspace(0.0, 1.0, n_samples):
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)
        cell = h3.latlng_to_cell(lat, lon, resolution)
        if prev_cell is not None and cell != prev_cell:
            # Only emit the edge if the two cells are actual neighbours
            # — for a nearly-straight short line they always are, but
            # rounding near a 3-cell-vertex can occasionally produce
            # a "diagonal" jump of distance ≥ 2.
            if cell in h3.grid_ring(prev_cell, 1):
                edges.add((prev_cell, cell))
                if bidirectional:
                    edges.add((cell, prev_cell))
        prev_cell = cell
    return list(edges)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


_REQUIRED_COLUMNS = {
    "from_h3", "to_h3", "mortality", "deflection", "transmission",
}


def load_h3_barrier_csv(
    path: Path,
    mesh,
) -> dict[tuple[int, int], "BarrierOutcome"]:
    """Parse a barrier CSV and return the ``{(from_idx, to_idx): outcome}``
    dict that :class:`salmon_ibm.barriers.BarrierMap` stores internally.

    CSV schema
    ----------
    Required columns: ``from_h3``, ``to_h3``, ``mortality``,
    ``deflection``, ``transmission``.  Optional: ``note``.

    Validation
    ----------
    * Each row's mortality + deflection + transmission must equal
      1.0 ± 1 e-6.
    * ``from_h3`` and ``to_h3`` must be H3 ring-1 neighbours.  Rows
      whose cells aren't in ``mesh`` are skipped with a warning so a
      single CSV can target multiple landscapes.

    Parameters
    ----------
    path
        Path to the CSV.
    mesh
        :class:`H3Mesh` whose ``h3_ids`` and ``neighbors`` provide the
        compact-index mapping.
    """
    from .barriers import BarrierOutcome

    # H3-int → mesh-compact-index reverse lookup, built once.
    id_to_idx = {int(mid): i for i, mid in enumerate(mesh.h3_ids)}

    edges: dict[tuple[int, int], BarrierOutcome] = {}
    n_skipped = 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"barrier CSV is empty: {path}")
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"barrier CSV missing columns: {sorted(missing)} "
                f"(have: {sorted(reader.fieldnames)})"
            )
        for row_num, row in enumerate(reader, start=2):
            from_str = row["from_h3"].strip()
            to_str = row["to_h3"].strip()
            from_id = int(h3.str_to_int(from_str))
            to_id = int(h3.str_to_int(to_str))
            if from_id not in id_to_idx or to_id not in id_to_idx:
                n_skipped += 1
                continue
            from_idx = id_to_idx[from_id]
            to_idx = id_to_idx[to_id]
            # Validate H3 adjacency at the *cell* level — relies on H3
            # itself, not on the mesh's compacted neighbour table (which
            # may have dropped some neighbours that are outside the mesh).
            if to_str not in h3.grid_ring(from_str, 1):
                raise ValueError(
                    f"row {row_num}: {from_str} → {to_str} are not H3 "
                    f"neighbours"
                )
            mort = float(row["mortality"])
            deflect = float(row["deflection"])
            trans = float(row["transmission"])
            total = mort + deflect + trans
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"row {row_num}: mortality+deflection+transmission "
                    f"= {total:.6f} (expected 1.0 ± 1e-6)"
                )
            edges[(from_idx, to_idx)] = BarrierOutcome(
                p_mortality=mort,
                p_deflection=deflect,
                p_transmission=trans,
            )

    if n_skipped:
        log.warning(
            "H3 barrier CSV %s: skipped %d row(s) with off-mesh cells",
            path, n_skipped,
        )
    return edges
