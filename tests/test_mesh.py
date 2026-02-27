import numpy as np
import pytest
from salmon_ibm.mesh import TriMesh


@pytest.fixture
def mesh():
    return TriMesh.from_netcdf("data/curonian_minimal_grid.nc")


def test_mesh_loads(mesh):
    assert mesh.nodes.shape[1] == 2
    assert mesh.triangles.shape[1] == 3


def test_mesh_has_centroids(mesh):
    assert mesh.centroids.shape == (mesh.n_triangles, 2)


def test_mesh_has_neighbors(mesh):
    assert mesh.neighbors.shape == (mesh.n_triangles, 3)


def test_mesh_mask_filters_land(mesh):
    assert mesh.water_mask.dtype == bool
    n_water = mesh.water_mask.sum()
    assert 0 < n_water < mesh.n_triangles


def test_mesh_depth_at_water_cells(mesh):
    water_depths = mesh.depth[mesh.water_mask]
    assert np.all(water_depths > 0)


def test_water_neighbors_returns_valid_indices(mesh):
    water_ids = np.where(mesh.water_mask)[0]
    tri = water_ids[0]
    nbrs = mesh.water_neighbors(tri)
    assert all(n >= 0 for n in nbrs)
    assert all(mesh.water_mask[n] for n in nbrs)


def test_find_triangle(mesh):
    water_ids = np.where(mesh.water_mask)[0]
    tri = water_ids[0]
    lat, lon = mesh.centroids[tri]
    found = mesh.find_triangle(lat, lon)
    assert found == tri


def test_gradient_returns_vector(mesh):
    field = mesh.depth.copy()
    water_ids = np.where(mesh.water_mask)[0]
    tri = water_ids[0]
    grad = mesh.gradient(field, tri)
    assert len(grad) == 2
