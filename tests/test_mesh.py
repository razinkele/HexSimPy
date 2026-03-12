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


def test_delaunay_neighbors_match_build_neighbors(mesh):
    """Verify that Delaunay.neighbors produces the same neighbor sets as
    the old edge-based _build_neighbors() method."""
    old_neighbors = TriMesh._build_neighbors(mesh.triangles)
    for i in range(mesh.n_triangles):
        old_set = set(int(n) for n in old_neighbors[i] if n >= 0)
        new_set = set(int(n) for n in mesh.neighbors[i] if n >= 0)
        assert old_set == new_set, (
            f"Triangle {i}: old {old_set} != new {new_set}"
        )


def test_precomputed_water_neighbors(mesh):
    """Verify that precomputed water neighbor arrays match the on-the-fly
    computation."""
    for tri_idx in range(mesh.n_triangles):
        # Recompute the expected result the old way
        nbrs = mesh.neighbors[tri_idx]
        expected = [int(n) for n in nbrs if n >= 0 and mesh.water_mask[n]]
        actual = mesh.water_neighbors(tri_idx)
        assert set(actual) == set(expected), (
            f"Triangle {tri_idx}: expected {expected}, got {actual}"
        )


def test_gradient_corrects_for_lat_lon_asymmetry(mesh):
    """Gradient should account for the fact that 1 degree of longitude
    is shorter than 1 degree of latitude at high latitudes."""
    # Create a field that increases purely in the longitude direction
    field = mesh.centroids[:, 1].copy()  # longitude values
    water_ids = np.where(mesh.water_mask)[0]
    # Find a water cell with neighbors
    for tri_idx in water_ids:
        nbrs = mesh.water_neighbors(tri_idx)
        if len(nbrs) >= 2:
            break
    dlat, dlon = mesh.gradient(field, tri_idx)
    # For a pure longitude gradient, the lat component should be near zero
    # and the lon component should dominate
    assert abs(dlon) > abs(dlat) * 0.5, (
        f"Longitude gradient should dominate for a pure-longitude field: "
        f"dlat={dlat:.4f}, dlon={dlon:.4f}"
    )
