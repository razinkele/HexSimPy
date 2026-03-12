"""Tests for HexSim binary readers, HexMesh, and HexSimEnvironment."""
import numpy as np
import pytest

from salmon_ibm.config import load_config
from salmon_ibm.hexsim import read_grid, read_hexmap, read_barriers, HexMesh


# ── Workspace path ───────────────────────────────────────────────────────────

WS = "Columbia River Migration Model/Columbia [small]"
HEX_DIR = f"{WS}/Spatial Data/Hexagons"
GRID_FILE = f"{WS}/Columbia Fish Model [small].grid"


# ── Binary reader tests ─────────────────────────────────────────────────────

def test_read_grid_returns_dimensions():
    meta = read_grid(GRID_FILE)
    assert meta["ncols"] == 1574
    assert meta["nrows"] == 10195


def test_read_hexmap_river_extent():
    """River extent values should be 0 (land) or small integers (water types)."""
    arr = read_hexmap(f"{HEX_DIR}/River [ extent ]/River [ extent ].1.hxn")
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
    arr = read_hexmap(f"{HEX_DIR}/Temperature Zones/Temperature Zones.1.hxn")
    # Filter to non-zero (water cells only)
    water = arr[arr != 0]
    unique_water = np.unique(water)
    assert unique_water.min() >= 1.0
    assert unique_water.max() <= 45.0
    # Should be integer-like values
    for v in unique_water:
        assert v == int(v), f"Non-integer zone value: {v}"


def test_read_barriers_parses_text():
    hbf = f"{WS}/Spatial Data/barriers/Fish Ladder Available/Fish Ladder Available.1.hbf"
    edges = read_barriers(hbf)
    assert len(edges) > 0
    first = edges[0]
    assert "hex_id" in first
    assert "edge" in first
    assert "classification" in first
    assert isinstance(first["hex_id"], int)


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
