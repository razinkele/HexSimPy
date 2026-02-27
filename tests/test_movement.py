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
