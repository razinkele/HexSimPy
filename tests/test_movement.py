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


def test_downstream_follows_ascending_ssh(mesh):
    """DOWNSTREAM fish should move toward higher SSH values (ascending=True)."""
    water_ids = np.where(mesh.water_mask)[0]
    # Use SSH that increases with the x-coordinate
    ssh = mesh.centroids[:, 0].copy()
    # Pick a start cell near the median SSH so there is room to move higher
    water_ssh = ssh[water_ids]
    median_ssh = np.median(water_ssh)
    # Find the water cell closest to median that has neighbors
    order = np.argsort(np.abs(water_ssh - median_ssh))
    for idx in order:
        start = water_ids[idx]
        if len(mesh.water_neighbors(start)) > 0:
            break
    pool = AgentPool(n=20, start_tri=start)
    pool.behavior[:] = Behavior.DOWNSTREAM
    pool.steps[:] = 10  # not first move
    initial_ssh = ssh[start]
    fields = {"ssh": ssh, "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=42)
    # Most agents should have moved to higher SSH (ascending)
    final_ssh = np.array([ssh[t] for t in pool.tri_idx])
    moved_higher = (final_ssh > initial_ssh).mean()
    assert moved_higher > 0.5, (
        f"Most DOWNSTREAM fish should move to higher SSH, but only {moved_higher:.0%} did"
    )


def test_random_movement_vectorized_lands_on_water(mesh):
    """Vectorized random movement should land on valid water cells."""
    water_ids = np.where(mesh.water_mask)[0]
    for start in water_ids:
        if len(mesh.water_neighbors(start)) > 0:
            break
    pool = AgentPool(n=50, start_tri=start)
    pool.behavior[:] = Behavior.RANDOM
    fields = {"ssh": np.zeros(mesh.n_triangles), "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=99)
    assert np.all(mesh.water_mask[pool.tri_idx])
    assert (pool.tri_idx != start).sum() > 0


def test_to_cwr_seeks_cooler_water(mesh):
    """TO_CWR fish above threshold should move toward cooler neighbors."""
    water_ids = np.where(mesh.water_mask)[0]
    # Find a start cell with water neighbors
    for start in water_ids:
        nbrs = mesh.water_neighbors(start)
        if len(nbrs) > 0:
            break
    pool = AgentPool(n=20, start_tri=start)
    pool.behavior[:] = Behavior.TO_CWR
    # Create a temperature field above CWR threshold with a gradient:
    # start cell is hot, and temperature decreases with distance from start
    # Use negative of distance-from-start so neighbors further away are cooler
    temperature = np.full(mesh.n_triangles, 22.0)
    # Make neighbors cooler based on their centroid x-coordinate
    # Use a gradient so that fish have a clear cooler direction
    temperature = 18.0 + 4.0 * (mesh.centroids[:, 0] - mesh.centroids[:, 0].min()) / (
        mesh.centroids[:, 0].max() - mesh.centroids[:, 0].min() + 1e-12
    )
    initial_temp = temperature[start]
    fields = {"ssh": np.zeros(mesh.n_triangles), "temperature": temperature,
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=42, cwr_threshold=16.0)
    # Fish should have moved to cooler cells (lower temperature)
    final_temps = np.array([temperature[t] for t in pool.tri_idx])
    moved_cooler = (final_temps < initial_temp).mean()
    assert moved_cooler > 0.5, (
        f"Most TO_CWR fish should move to cooler water, but only {moved_cooler:.0%} did"
    )
