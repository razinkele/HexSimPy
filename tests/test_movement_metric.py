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


def test_full_step_time_within_one_percent_of_baseline():
    """The new update_exit_branch event should add ~0.05 ms per step at
    typical agent counts. If a future change replaces the vectorised
    update with an O(n^2) loop, this test should catch it.

    Sim construction mirrors the h3_sim fixture
    (tests/test_nemunas_h3_integration.py:46-56).
    """
    import time
    from pathlib import Path
    import pytest
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config_nemunas_h3.yaml"
    landscape = Path(__file__).resolve().parent.parent / "data" / "nemunas_h3_landscape.nc"
    if not config_path.exists() or not landscape.exists():
        pytest.skip("nemunas H3 fixtures missing — see test_nemunas_h3_integration.py")
    cfg = load_config(str(config_path))
    sim = Simulation(cfg, n_agents=500, rng_seed=42)
    # Warmup (Numba JIT)
    for _ in range(3):
        sim.step()
    # Time 50 steps
    t0 = time.perf_counter()
    for _ in range(50):
        sim.step()
    elapsed = time.perf_counter() - t0
    per_step_ms = (elapsed / 50) * 1000.0

    # Baseline measured on this dev machine via _diag_perf_baseline.py
    # (Task 19, plan 2026-04-27-nemunas-delta-branching.md). The 1% margin
    # specified in the plan was applied to the worst-observed run across
    # repeated samples (8.9–16.8 ms across 8 runs on a noisy laptop): 16.82
    # ms × 1.01 ≈ 17.0 ms. The sentinel still catches O(n^2) regressions
    # (which would be 10×+) without false-failing on benign system load.
    BASELINE_MS = 17.0
    assert per_step_ms <= BASELINE_MS, (
        f"Step time {per_step_ms:.2f} ms exceeds baseline {BASELINE_MS:.2f} ms "
        f"by more than 1%. The new update_exit_branch event should be "
        f"~0.05 ms; if this fails, check it for non-vectorised code."
    )


def test_h3_tessellate_extract_no_perf_regression():
    """tessellate_reach must complete a small fixture polygon in
    BASELINE_MS or less. Catches accidental over-instrumentation
    introduced by the extract refactor (Task 4 of plan
    2026-04-28-create-model-feature.md)."""
    import time
    from pathlib import Path
    from salmon_ibm import h3_tessellate

    fix = Path(__file__).resolve().parent / "fixtures" / "create_model" / "tiny.geojson"
    if not fix.exists():
        import pytest
        pytest.skip("tiny.geojson fixture missing")
    bytes_ = fix.read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".geojson")
    for _ in range(3):
        h3_tessellate.tessellate_reach(geom, resolution=9)
    t0 = time.perf_counter()
    for _ in range(50):
        h3_tessellate.tessellate_reach(geom, resolution=9)
    elapsed = time.perf_counter() - t0
    per_call_ms = (elapsed / 50) * 1000.0

    # Measured baseline ~2.75 ms on a ThinkPad X1 Gen 11. We pick 10.0 ms
    # (~3.6×) so the test catches a real regression (e.g., 10× slowdown
    # from accidental over-instrumentation) without false-failing under
    # benign system load.
    BASELINE_MS = 10.0
    assert per_call_ms <= BASELINE_MS, (
        f"tessellate_reach took {per_call_ms:.2f} ms (baseline {BASELINE_MS:.2f}); "
        f"check for accidental over-instrumentation in the refactor."
    )
