"""Tests for C4 — movement gradient (substrate fix).

Spec: docs/superpowers/specs/2026-05-07-c4-movement-gradient-design.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Import the build-script function. The script lives outside the
# salmon_ibm package, so add the scripts directory to sys.path.
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class _SyntheticMesh:
    """Minimal mesh for compute_dist_from_sea unit tests.

    Provides the attributes the function reads: nbr_starts, nbr_idx,
    centroids, water_mask, reach_id, reach_names. Adds N_cells convenience
    via len(reach_id).
    """
    def __init__(
        self,
        nbr_starts: np.ndarray,
        nbr_idx: np.ndarray,
        centroids: np.ndarray,
        water_mask: np.ndarray,
        reach_id: np.ndarray,
        reach_names: list[str],
    ):
        self.nbr_starts = nbr_starts
        self.nbr_idx = nbr_idx
        self.centroids = centroids
        self.water_mask = water_mask
        self.reach_id = reach_id
        self.reach_names = reach_names

    @property
    def N_cells(self) -> int:
        return len(self.reach_id)


def test_compute_dist_from_sea_raises_on_disconnected_component():
    """C4 Test 5: synthetic mesh with two disconnected components
    (one with sea, one without). compute_dist_from_sea must raise
    RuntimeError naming the unreachable reach."""
    from build_h3_multires_landscape import compute_dist_from_sea

    # Component A: cells 0,1 (sea + adjacent), reach=OpenBaltic.
    # Component B: cells 2,3 (river, no path to sea), reach=Nemunas.
    nbr_starts = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    nbr_idx = np.array([1, 0, 3, 2], dtype=np.int32)
    centroids = np.array([
        [55.0, 21.0],  # cell 0
        [55.0, 21.001],  # cell 1
        [55.5, 21.5],  # cell 2
        [55.5, 21.501],  # cell 3
    ], dtype=np.float64)
    water_mask = np.ones(4, dtype=bool)
    reach_id = np.array([0, 0, 1, 1], dtype=np.int8)
    reach_names = ["OpenBaltic", "Nemunas"]

    mesh = _SyntheticMesh(nbr_starts, nbr_idx, centroids, water_mask,
                          reach_id, reach_names)
    with pytest.raises(RuntimeError, match=r"Nemunas"):
        compute_dist_from_sea(mesh)


def _make_chain_mesh(n: int = 10) -> _SyntheticMesh:
    """10-cell bidirectional chain. Cell 0 = OpenBaltic source.
    Cells 1..n-1 = Nemunas (river). Used by Tests 1, 2, 2b, 3, 5b."""
    # Bidirectional CSR: cell i has neighbors {i-1, i+1} where they exist.
    nbr_starts = np.zeros(n + 1, dtype=np.int32)
    nbrs = []
    for i in range(n):
        if i > 0:
            nbrs.append(i - 1)
        if i < n - 1:
            nbrs.append(i + 1)
        nbr_starts[i + 1] = len(nbrs)
    nbr_idx = np.array(nbrs, dtype=np.int32)
    # Centroids spaced 100m apart along a meridian (uniform haversine).
    centroids = np.array(
        [[55.0 + i * 0.0009, 21.0] for i in range(n)],
        dtype=np.float64,
    )
    water_mask = np.ones(n, dtype=bool)
    reach_id = np.zeros(n, dtype=np.int8)
    reach_id[1:] = 1  # cell 0 = OpenBaltic; cells 1..n-1 = Nemunas
    reach_names = ["OpenBaltic", "Nemunas"]
    return _SyntheticMesh(
        nbr_starts, nbr_idx, centroids, water_mask, reach_id, reach_names,
    )


def test_compute_dist_from_sea_deterministic_synthetic():
    """C4 Test 5b (synthetic part): two runs on the same mesh produce
    NaN-aware-equal output."""
    from build_h3_multires_landscape import compute_dist_from_sea

    mesh = _make_chain_mesh(n=10)
    out1 = compute_dist_from_sea(mesh)
    out2 = compute_dist_from_sea(mesh)
    assert np.array_equal(out1, out2, equal_nan=True), (
        "compute_dist_from_sea is non-deterministic on a 10-cell chain"
    )


def test_compute_dist_from_sea_y_junction_tie_break():
    """C4 Test 5c: 4-cell Y-junction with three equidistant non-source
    cells. Asserts byte-equal recompute."""
    from build_h3_multires_landscape import compute_dist_from_sea

    # Cell 0 = OpenBaltic source. Cells 1,2,3 each connected to cell 0
    # only — three equidistant arms. Bidirectional edges.
    nbr_starts = np.array([0, 3, 4, 5, 6], dtype=np.int32)
    nbr_idx = np.array([1, 2, 3,  0,  0,  0], dtype=np.int32)
    # All three arm-tips at the same lat offset but different lons
    # (cells 1, 2, 3 each differ from cell 0 by ~100m along distinct
    # bearings — haversine values are bit-identical because we use
    # identical math.cos/math.sin inputs).
    centroids = np.array([
        [55.0, 21.0],         # cell 0 (source)
        [55.0009, 21.0],      # cell 1 (north)
        [55.0, 21.00157],     # cell 2 (east; 0.00157 deg lon ~= 100m at 55N)
        [54.9991, 21.0],      # cell 3 (south)
    ], dtype=np.float64)
    water_mask = np.ones(4, dtype=bool)
    reach_id = np.array([0, 1, 1, 1], dtype=np.int8)  # 0=OpenBaltic,1=Nemunas
    reach_names = ["OpenBaltic", "Nemunas"]

    mesh = _SyntheticMesh(
        nbr_starts, nbr_idx, centroids, water_mask, reach_id, reach_names,
    )
    out1 = compute_dist_from_sea(mesh)
    out2 = compute_dist_from_sea(mesh)
    assert np.array_equal(out1, out2, equal_nan=True), (
        "Y-junction tie-break is non-deterministic"
    )
    # Cell 0 = source, distance 0.
    assert out1[0] == 0.0
    # Cells 1, 2, 3 all reachable, finite, > 0.
    assert np.all(np.isfinite(out1[1:]))
    assert np.all(out1[1:] > 0)
    # Cells 1 and 3 are placed at lat offsets +/- 0.0009 from cell 0
    # with identical lon — haversine produces bit-identical distances
    # for them. Cell 2 (lon-offset arm) is only approximately
    # equidistant; the determinism check above already covers it.
    assert out1[1] == out1[3]


# -----------------------------------------------------------------------------
# Task 6 — _check_dormant_gradient helper tests (4e, 4f + sanity)
# -----------------------------------------------------------------------------

# Reuse synthetic-NC builders from tests/test_h3_env.py for the Case-B
# happy-path test (Test 4f).
from tests.test_h3_env import _build_minimal_h3_nc, _build_mesh_from_nc


def _make_landscape(env, dist_from_sea_arr=None):
    """Helper: minimal landscape dict for dormancy-check tests.

    `dist_from_sea_arr` overrides what the helper inspects; if None,
    uses env.fields["dist_from_sea"] as-is.
    """
    fields = dict(env.fields) if hasattr(env, "fields") else {}
    if dist_from_sea_arr is not None:
        fields["dist_from_sea"] = dist_from_sea_arr
    return {
        "env": env,
        "fields": fields,
        "rng": np.random.default_rng(0),
    }


def _make_pool(behaviors):
    """Helper: minimal pool stand-in with .behavior array."""
    class _FakePool:
        pass
    pool = _FakePool()
    pool.behavior = np.asarray(behaviors, dtype=np.int8)
    return pool


def test_check_dormant_gradient_raises_on_flat_zero_with_directed_agents():
    """C4 Test 4e (part 1): env-A loaded via Case A path -> flat-zero ->
    directed agent -> raise containing the err-id."""
    from salmon_ibm.movement import _check_dormant_gradient
    from salmon_ibm.agents import Behavior
    from salmon_ibm.h3_env import ERR_DIST_FROM_SEA_MISSING

    class _FakeEnv:
        pass
    env_a = _FakeEnv()
    env_a.fields = {"dist_from_sea": np.zeros(10, dtype=np.float32)}
    env_a._dormant_gradient_check_done = False

    pool = _make_pool([int(Behavior.UPSTREAM), int(Behavior.HOLD)])
    landscape = _make_landscape(env_a)

    with pytest.raises(RuntimeError, match=ERR_DIST_FROM_SEA_MISSING):
        _check_dormant_gradient(landscape, pool)


def test_check_dormant_gradient_per_env_isolation():
    """C4 Test 4e (part 2): env-A's latch does NOT affect env-B's
    independent check. Regression test for the pass-7 module-global
    -> per-env-instance refactor."""
    from salmon_ibm.movement import _check_dormant_gradient
    from salmon_ibm.agents import Behavior

    class _FakeEnv:
        pass

    env_a = _FakeEnv()
    env_a.fields = {"dist_from_sea": np.zeros(10, dtype=np.float32)}
    env_a._dormant_gradient_check_done = False

    pool = _make_pool([int(Behavior.UPSTREAM)])

    with pytest.raises(RuntimeError):
        _check_dormant_gradient(_make_landscape(env_a), pool)
    assert env_a._dormant_gradient_check_done is True

    env_b = _FakeEnv()
    env_b.fields = {"dist_from_sea": np.zeros(10, dtype=np.float32)}
    env_b._dormant_gradient_check_done = False
    with pytest.raises(RuntimeError):
        _check_dormant_gradient(_make_landscape(env_b), pool)
    assert env_b._dormant_gradient_check_done is True


def test_check_dormant_gradient_happy_path_latches(tmp_path, caplog):
    """C4 Test 4f: load env via H3Environment.from_netcdf with a
    valid Case-B NC; check fires no raise and latches True. Tests the
    end-to-end Case-B init -> helper-call sequence."""
    from salmon_ibm.movement import _check_dormant_gradient
    from salmon_ibm.h3_env import H3Environment
    from salmon_ibm.agents import Behavior

    nc_path = tmp_path / "happy_path.nc"
    valid = np.array([0.0, 100.0, 200.0, np.nan], dtype=np.float32)
    _build_minimal_h3_nc(nc_path, dist_from_sea_arr=valid)
    mesh = _build_mesh_from_nc(nc_path)
    env = H3Environment.from_netcdf(str(nc_path), mesh)
    assert env._dormant_gradient_check_done is False  # Case B init OK

    pool = _make_pool([int(Behavior.UPSTREAM)])
    # No raise — gradient has positive values.
    _check_dormant_gradient(_make_landscape(env), pool)
    assert env._dormant_gradient_check_done is True

    # Second call: latched. Even with the gradient now-zeroed, no raise.
    env.fields["dist_from_sea"][:] = 0.0
    _check_dormant_gradient(_make_landscape(env), pool)


def test_check_dormant_gradient_no_directed_agents_no_raise():
    """C4 sanity: flat-zero gradient + only HOLD/RANDOM/TO_CWR agents
    -> no raise (the check is gated on UPSTREAM/DOWNSTREAM presence)."""
    from salmon_ibm.movement import _check_dormant_gradient
    from salmon_ibm.agents import Behavior

    class _FakeEnv:
        pass
    env = _FakeEnv()
    env.fields = {"dist_from_sea": np.zeros(10, dtype=np.float32)}
    env._dormant_gradient_check_done = False

    pool = _make_pool([
        int(Behavior.HOLD),
        int(Behavior.RANDOM),
        int(Behavior.TO_CWR),
    ])
    _check_dormant_gradient(_make_landscape(env), pool)
    assert env._dormant_gradient_check_done is True


def test_check_dormant_gradient_no_env_in_landscape_no_raise():
    """C4 sanity: legacy non-Baltic landscape (no env key) -> no-op."""
    from salmon_ibm.movement import _check_dormant_gradient
    from salmon_ibm.agents import Behavior

    landscape = {
        "fields": {"dist_from_sea": np.zeros(10, dtype=np.float32)},
        "rng": np.random.default_rng(0),
        # NO "env" key.
    }
    pool = _make_pool([int(Behavior.UPSTREAM)])
    _check_dormant_gradient(landscape, pool)  # must not raise
