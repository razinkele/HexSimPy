"""Tests for HexSim workspace loading, HexMesh, and HexSimEnvironment."""
import os

import numpy as np
import pytest

from heximpy.hxnparser import GridMeta, HexMap, Workspace, read_barriers
from salmon_ibm.config import load_config
from salmon_ibm.hexsim import HexMesh, _hex_neighbors_offset


# ── Workspace path ───────────────────────────────────────────────────────────

WS = "Columbia River Migration Model/Columbia [small]"
HEX_DIR = f"{WS}/Spatial Data/Hexagons"
GRID_FILE = f"{WS}/Columbia Fish Model [small].grid"


# ── hxnparser reader tests ──────────────────────────────────────────────────

def test_read_grid_returns_dimensions():
    meta = GridMeta.from_file(GRID_FILE)
    assert meta.ncols == 1574
    assert meta.nrows == 10195


def test_read_grid_returns_georef():
    """Grid file should contain georeferencing with valid hex geometry."""
    meta = GridMeta.from_file(GRID_FILE)
    # row_spacing ≈ 24.028 m, edge ≈ 13.876 m
    assert 23.0 < meta.row_spacing < 25.0
    assert 13.0 < meta.edge < 15.0
    # Verify flat-top relationship: row_spacing = √3 × edge
    assert abs(meta.row_spacing - np.sqrt(3.0) * meta.edge) < 0.01


def test_read_hexmap_river_extent():
    """River extent values should be 0 (land) or small integers (water types)."""
    hm = HexMap.from_file(f"{HEX_DIR}/River [ extent ]/River [ extent ].1.hxn")
    arr = hm.values
    assert arr.dtype == np.float32
    assert len(arr) > 1_000_000  # should be ~16M values

    # Water cells have nonzero values, land cells are 0
    water = arr[arr != 0]
    assert len(water) > 50_000   # tens to hundreds of thousands of water cells
    assert len(water) < 1_000_000
    # All water values should be small positive integers (1, 2, or 3)
    unique_water = np.unique(water)
    assert all(v > 0 for v in unique_water)
    assert unique_water.max() <= 3.0


def test_read_hexmap_temperature_zones():
    """Temperature zone values should be 0 (land) or integers 1–45."""
    hm = HexMap.from_file(f"{HEX_DIR}/Temperature Zones/Temperature Zones.1.hxn")
    arr = hm.values
    # Filter to non-zero (water cells only)
    water = arr[arr != 0]
    unique_water = np.unique(water)
    assert unique_water.min() >= 1.0
    assert unique_water.max() <= 45.0
    # Should be integer-like values
    for v in unique_water:
        assert v == int(v), f"Non-integer zone value: {v}"


def test_read_hxn_patch_hexmap_format():
    """HexMap.from_file should handle PATCH_HEXMAP files."""
    hm = HexMap.from_file(f"{HEX_DIR}/River [ extent ]/River [ extent ].1.hxn")
    assert hm.format == "patch_hexmap"
    assert len(hm.values) > 1_000_000
    # Water cells have nonzero values
    water = hm.values[hm.values != 0]
    assert len(water) > 50_000


def test_read_barriers_parses_text():
    hbf = f"{WS}/Spatial Data/barriers/Fish Ladder Available/Fish Ladder Available.1.hbf"
    barriers = read_barriers(hbf)
    assert len(barriers) > 0
    first = barriers[0]
    assert isinstance(first.hex_id, int)
    assert isinstance(first.edge, int)
    assert isinstance(first.classification, int)


def test_workspace_from_dir():
    """Workspace.from_dir should load grid, hexmaps, and barriers."""
    ws = Workspace.from_dir(WS)
    assert ws.grid.ncols == 1574
    assert ws.grid.nrows == 10195
    assert "River [ extent ]" in ws.hexmaps
    assert len(ws.barriers) > 0


# ── Flat-top neighbor convention tests ────────────────────────────────────────

def test_flat_top_odd_q_neighbors():
    """Odd-q flat-top: even column neighbors differ from odd column."""
    ncols, nrows, n_data = 10, 10, 100
    # Even column (col=4): neighbors should shift up for diagonal
    nbrs_even = _hex_neighbors_offset(5, 4, ncols, nrows, n_data)
    # Odd column (col=5): neighbors should shift down for diagonal
    nbrs_odd = _hex_neighbors_offset(5, 5, ncols, nrows, n_data)
    # Both should have 6 neighbors (interior cell)
    assert len(nbrs_even) == 6
    assert len(nbrs_odd) == 6
    # Even col: diagonals go to row-1 (up)
    # Expected: (4,4),(6,4),(5,3),(5,5),(4,3),(4,5) → flat: 44,64,53,55,43,45
    assert 44 in nbrs_even  # (row-1, same col)
    assert 64 in nbrs_even  # (row+1, same col)
    assert 43 in nbrs_even  # (row-1, col-1) — even col shifts up
    # Odd col: diagonals go to row+1 (down)
    # Expected: (4,5),(6,5),(5,4),(5,6),(6,4),(6,6) → flat: 45,65,54,56,64,66
    assert 45 in nbrs_odd   # (row-1, same col)
    assert 65 in nbrs_odd   # (row+1, same col)
    assert 64 in nbrs_odd   # (row+1, col-1) — odd col shifts down


# ── HexMesh tests ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mesh():
    """Load HexMesh once for all mesh tests (expensive I/O)."""
    return HexMesh.from_hexsim(WS, species="chinook")


def test_hexmesh_water_only_compaction(mesh):
    """Compacted mesh should have far fewer cells than full grid."""
    assert mesh.n_cells < 1_000_000  # much less than 16M
    assert mesh.n_cells > 50_000     # but at least tens of thousands
    assert mesh.water_mask.all()     # all stored cells are water


def test_hexmesh_n_triangles_alias(mesh):
    assert mesh.n_triangles == mesh.n_cells


def test_hexmesh_centroids_shape(mesh):
    assert mesh.centroids.shape == (mesh.n_cells, 2)


def test_hexmesh_depth_shape(mesh):
    assert mesh.depth.shape == (mesh.n_cells,)


def test_hexmesh_neighbors_shape(mesh):
    assert mesh.neighbors.shape == (mesh.n_cells, 6)


def test_hexmesh_neighbors_are_symmetric(mesh):
    """If A neighbors B, then B should neighbor A."""
    # Check a random sample of cells (checking all 880K is slow)
    rng = np.random.default_rng(42)
    sample = rng.choice(mesh.n_cells, size=min(1000, mesh.n_cells), replace=False)

    for a in sample:
        for b in mesh.water_neighbors(a):
            b_nbrs = mesh.water_neighbors(b)
            assert a in b_nbrs, f"Cell {a} neighbors {b}, but {b} does not neighbor {a}"


def test_hexmesh_find_triangle(mesh):
    """find_triangle should return a valid cell index."""
    # Pick a known water cell's centroid
    idx = mesh.n_cells // 2
    y, x = mesh.centroids[idx]
    found = mesh.find_triangle(y, x)
    assert found == idx  # should find itself


def test_hexmesh_gradient_returns_tuple(mesh):
    """gradient() should return (dy, dx) tuple."""
    field = mesh.depth.copy()
    idx = mesh.n_cells // 2
    result = mesh.gradient(field, idx)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_hexmesh_areas_positive(mesh):
    assert (mesh.areas > 0).all()


def test_hexmesh_areas_approx_500m2(mesh):
    """HexSim hex area should be ≈500 m² (edge ≈ 13.876 m)."""
    assert abs(mesh.areas[0] - 500.0) < 5.0  # within 5 m²


def test_hexmesh_centroids_in_meters(mesh):
    """Centroids should be in meters, not arbitrary grid units."""
    # Columbia River extent is ~32 km wide and ~245 km tall
    y_range = mesh.centroids[:, 0].max() - mesh.centroids[:, 0].min()
    x_range = mesh.centroids[:, 1].max() - mesh.centroids[:, 1].min()
    # Should be in the thousands-of-meters range, not single digits
    assert y_range > 10_000   # at least 10 km
    assert x_range > 1_000    # at least 1 km


def test_hexmesh_stores_workspace(mesh):
    """HexMesh should store the parsed Workspace for reuse."""
    assert mesh._workspace is not None
    assert isinstance(mesh._workspace, Workspace)
    assert mesh._workspace.grid.ncols == 1574


# ── HexSimEnvironment tests ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env(mesh):
    """Create HexSimEnvironment once."""
    from salmon_ibm.hexsim_env import HexSimEnvironment
    return HexSimEnvironment(WS, mesh)


def test_hexsim_env_has_fields(env):
    env.advance(0)
    assert "temperature" in env.fields
    assert "salinity" in env.fields
    assert "ssh" in env.fields
    assert "u_current" in env.fields
    assert "v_current" in env.fields


def test_hexsim_env_temperature_from_zones(env, mesh):
    """Temperature should come from zone lookup, not be all zeros."""
    env.advance(0)
    temp = env.fields["temperature"]
    assert temp.shape == (mesh.n_cells,)
    # Temperature should be in a reasonable range for Columbia River
    assert temp.min() > 5.0   # not freezing
    assert temp.max() < 35.0  # not boiling


def test_hexsim_env_advance_changes_temp(env):
    """Different timesteps should give different temperature distributions."""
    env.advance(0)
    temp0 = env.fields["temperature"].copy()
    env.advance(100)
    temp100 = env.fields["temperature"].copy()
    # At least some cells should have different temps at different hours
    assert not np.array_equal(temp0, temp100)


def test_hexsim_env_sample_returns_dict(env, mesh):
    env.advance(0)
    s = env.sample(0)
    assert isinstance(s, dict)
    assert isinstance(s["temperature"], float)


def test_hexsim_env_gradient(env, mesh):
    env.advance(0)
    result = env.gradient("temperature", mesh.n_cells // 2)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_hexsim_env_dSSH_dt(env, mesh):
    env.advance(0)
    env.advance(1)
    val = env.dSSH_dt(0)
    assert isinstance(val, float)


def test_hexsim_env_advance_reuses_temp_array_dtype():
    """Temperature field should be float64 without per-step conversion."""
    from salmon_ibm.hexsim import HexMesh
    from salmon_ibm.hexsim_env import HexSimEnvironment
    WS = "Columbia River Migration Model/Columbia [small]"
    if not os.path.exists(WS):
        pytest.skip("Columbia workspace not found")
    mesh = HexMesh.from_hexsim(WS)
    env = HexSimEnvironment(WS, mesh)
    env.advance(0)
    assert env.fields["temperature"].dtype == np.float64
    # Advance again — should reuse the same buffer (identity check)
    env.advance(1)
    assert env.fields["temperature"] is env._temp_buf


def test_hexsim_env_ssh_is_static_no_copy():
    """SSH field should not be re-copied each step for static gradient."""
    from salmon_ibm.hexsim import HexMesh
    from salmon_ibm.hexsim_env import HexSimEnvironment
    WS = "Columbia River Migration Model/Columbia [small]"
    if not os.path.exists(WS):
        pytest.skip("Columbia workspace not found")
    mesh = HexMesh.from_hexsim(WS)
    env = HexSimEnvironment(WS, mesh)
    env.advance(0)
    ssh0 = env.fields["ssh"]
    env.advance(1)
    ssh1 = env.fields["ssh"]
    assert ssh0 is ssh1


def test_hexsim_env_dssh_dt_always_zero():
    """dSSH_dt_array should return zeros for static SSH (no computation)."""
    from salmon_ibm.hexsim import HexMesh
    from salmon_ibm.hexsim_env import HexSimEnvironment
    WS = "Columbia River Migration Model/Columbia [small]"
    if not os.path.exists(WS):
        pytest.skip("Columbia workspace not found")
    mesh = HexMesh.from_hexsim(WS)
    env = HexSimEnvironment(WS, mesh)
    env.advance(0)
    env.advance(1)
    dssh = env.dSSH_dt_array()
    assert (dssh == 0.0).all()
    dssh2 = env.dSSH_dt_array()
    assert dssh is dssh2


# ── Integration test ─────────────────────────────────────────────────────────

def test_simulation_runs_on_hexsim():
    """Full 24h simulation run on Columbia River landscape."""
    from salmon_ibm.simulation import Simulation
    cfg = load_config("config_columbia.yaml")
    sim = Simulation(cfg, n_agents=20, rng_seed=42)
    sim.run(n_steps=24)

    assert sim.pool.alive.sum() > 0
    assert len(sim.history) == 24

    sim.close()
