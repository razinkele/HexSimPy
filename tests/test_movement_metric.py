"""Phase-0 regression: ``_advection_numba`` must be metric-correct under
both lat/lon centroids (TriMesh, H3Mesh) and meter centroids (HexMesh).

Guards against the latent Euclidean-in-degrees bias that would bite any
mesh backend placing cells far from the equator.
"""
from __future__ import annotations

import numpy as np

from salmon_ibm.movement import _advection_numba


def test_advection_east_flow_picks_east_neighbor_at_55N():
    """At 55°N, cos(lat) ≈ 0.57 — without metric scaling, a 100-m easterly
    offset in degrees looks smaller than a 100-m northerly offset, so a
    pure east flow would incorrectly pick the north neighbour.

    With ``scale_x = 111320·cos(55°) ≈ 63860`` and ``scale_y = 110540``,
    both axes are in meters and the agent correctly picks east.
    """
    # col-0 = lat (y), col-1 = lon (x), per movement.py convention
    centroids = np.array([
        [55.00000, 21.00000],
        [55.00090, 21.00000],   # ~100 m north of cell 0
        [55.00000, 21.00156],   # ~100 m east of cell 0
    ], dtype=np.float64)
    water_nbrs = np.array([[1, 2, -1, -1, -1, -1],
                           [0, -1, -1, -1, -1, -1],
                           [0, -1, -1, -1, -1, -1]], dtype=np.int32)
    water_nbr_count = np.array([2, 1, 1], dtype=np.int32)
    tri_indices = np.array([0], dtype=np.int32)
    u = np.array([1.0, 1.0, 1.0])            # pure east flow at every cell
    v = np.array([0.0, 0.0, 0.0])
    speeds = np.array([0.5])                  # above speed_threshold
    rand_drift = np.array([0.0])              # always drift
    scale_x = 111320.0 * np.cos(np.deg2rad(55.0))  # ≈ 63860
    scale_y = 110540.0

    # Kernel mutates tri_indices in place and returns None.
    _advection_numba(
        tri_indices, water_nbrs, water_nbr_count, centroids,
        u, v, speeds, rand_drift,
        scale_x, scale_y,
    )
    assert tri_indices[0] == 2, (
        f"expected east neighbour (2), got {tri_indices[0]}"
    )


def test_advection_back_compat_on_hexmesh_meters():
    """HexMesh centroids are already in meters; ``metric_scale`` returns
    (1.0, 1.0) for a no-op correction.  Pre-refactor behaviour must be
    bit-for-bit preserved for the HexMesh Columbia path.
    """
    # col-0 = y (north), col-1 = x (east)
    centroids = np.array([
        [0.0, 0.0],
        [100.0, 0.0],   # 100 m north
        [0.0, 100.0],   # 100 m east
    ], dtype=np.float64)
    water_nbrs = np.array([[1, 2, -1, -1, -1, -1],
                           [0, -1, -1, -1, -1, -1],
                           [0, -1, -1, -1, -1, -1]], dtype=np.int32)
    water_nbr_count = np.array([2, 1, 1], dtype=np.int32)
    tri_indices = np.array([0], dtype=np.int32)
    u = np.array([0.0, 0.0, 0.0])            # pure north flow (v only)
    v = np.array([1.0, 1.0, 1.0])
    speeds = np.array([0.5])
    rand_drift = np.array([0.0])
    _advection_numba(
        tri_indices, water_nbrs, water_nbr_count, centroids,
        u, v, speeds, rand_drift,
        1.0, 1.0,
    )
    assert tri_indices[0] == 1, (
        f"expected north neighbour (1), got {tri_indices[0]}"
    )
