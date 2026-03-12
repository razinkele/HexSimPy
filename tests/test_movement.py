import numpy as np
import pytest
from salmon_ibm.mesh import TriMesh
from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.movement import execute_movement


@pytest.fixture
def mesh():
    return TriMesh.from_netcdf("data/curonian_minimal_grid.nc")


def test_hold_does_not_move(mesh):
    water_ids = np.where(mesh.water_mask)[0]
    start = water_ids[0]
    pool = AgentPool(n=5, start_tri=start)
    pool.behavior[:] = Behavior.HOLD
    fields = {"ssh": np.zeros(mesh.n_triangles), "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=42)
    assert np.all(pool.tri_idx == start)


def test_random_moves_to_neighbor(mesh):
    water_ids = np.where(mesh.water_mask)[0]
    for start in water_ids:
        if len(mesh.water_neighbors(start)) > 0:
            break
    pool = AgentPool(n=20, start_tri=start)
    pool.behavior[:] = Behavior.RANDOM
    fields = {"ssh": np.zeros(mesh.n_triangles), "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=42)
    moved = pool.tri_idx != start
    assert moved.sum() > 0
    assert np.all(mesh.water_mask[pool.tri_idx])


def test_upstream_follows_ssh_gradient(mesh):
    water_ids = np.where(mesh.water_mask)[0]
    ssh = mesh.centroids[:, 0].copy()
    for start in water_ids:
        if len(mesh.water_neighbors(start)) > 0:
            break
    pool = AgentPool(n=10, start_tri=start)
    pool.behavior[:] = Behavior.UPSTREAM
    fields = {"ssh": ssh, "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=42)
    assert np.all(mesh.water_mask[pool.tri_idx])


def test_cwr_stops_at_threshold(mesh):
    """TO_CWR movement should stop when temperature drops below the threshold."""
    water_ids = np.where(mesh.water_mask)[0]
    # Find a cell with water neighbors
    for start in water_ids:
        if len(mesh.water_neighbors(start)) > 0:
            break
    pool = AgentPool(n=5, start_tri=start)
    pool.behavior[:] = Behavior.TO_CWR
    # Set all temperatures below CWR threshold — fish should not move
    temperature = np.full(mesh.n_triangles, 10.0)  # well below 16.0
    fields = {"ssh": np.zeros(mesh.n_triangles), "temperature": temperature,
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=42, cwr_threshold=16.0)
    assert np.all(pool.tri_idx == start), "Fish should stay put when already below CWR threshold"


def test_upstream_net_movement_follows_gradient(mesh):
    """UPSTREAM fish should make net progress toward lower SSH."""
    water_ids = np.where(mesh.water_mask)[0]
    # Create an SSH field that increases with latitude
    ssh = mesh.centroids[:, 0].copy()
    for start in water_ids:
        if len(mesh.water_neighbors(start)) > 0:
            break
    pool = AgentPool(n=20, start_tri=start)
    pool.behavior[:] = Behavior.UPSTREAM
    pool.steps[:] = 10  # not first move
    initial_ssh = ssh[start]
    fields = {"ssh": ssh, "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=42)
    # Most agents should have moved to lower SSH
    final_ssh = np.array([ssh[t] for t in pool.tri_idx])
    moved_lower = (final_ssh < initial_ssh).mean()
    assert moved_lower > 0.5, (
        f"Most UPSTREAM fish should move to lower SSH, but only {moved_lower:.0%} did"
    )
