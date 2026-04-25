"""Unit tests for H3Mesh — Phase 1 of the H3 backend plan."""
from __future__ import annotations

import math

import h3
import numpy as np
import pytest

from salmon_ibm.h3mesh import H3Mesh


# ---------------------------------------------------------------------------
# Task 1.1 — from_h3_cells + duck-typed Mesh contract
# ---------------------------------------------------------------------------


def test_h3mesh_from_cells_builds_neighbors():
    """Centre cell + 6 ring-1 neighbours: every neighbour must point back."""
    center = h3.latlng_to_cell(55.3, 21.1, 9)
    ring = list(h3.grid_ring(center, 1))
    cells = [center] + ring
    depth = np.full(7, 5.0, dtype=np.float32)
    mesh = H3Mesh.from_h3_cells(cells, depth=depth)

    assert mesh.n_cells == 7
    assert mesh.n_triangles == 7  # duck-typed alias
    assert mesh.neighbors.shape == (7, 6)
    assert mesh.centroids.shape == (7, 2)
    assert mesh.centroids.dtype == np.float64

    # Centre cell has all 6 neighbours present (no -1).
    assert (mesh.neighbors[0] >= 0).sum() == 6
    # Every neighbour reaches back to the centre.
    for i in range(1, 7):
        assert 0 in mesh.neighbors[i].tolist(), (
            f"cell {i} doesn't reach back to centre"
        )


def test_h3mesh_resolution_inferred():
    cells = [h3.latlng_to_cell(55.3, 21.1, 9)]
    cells += list(h3.grid_ring(cells[0], 1))
    mesh = H3Mesh.from_h3_cells(cells)
    assert mesh.resolution == 9


def test_h3mesh_water_neighbors_returns_compact_ints():
    center = h3.latlng_to_cell(55.3, 21.1, 9)
    cells = [center] + list(h3.grid_ring(center, 1))
    mesh = H3Mesh.from_h3_cells(cells)
    nbrs = mesh.water_neighbors(0)
    assert isinstance(nbrs, list)
    assert all(isinstance(n, int) for n in nbrs)
    assert sorted(nbrs) == [1, 2, 3, 4, 5, 6]  # 6 valid neighbours, no -1


def test_h3mesh_find_triangle_round_trips():
    """find_triangle(lat, lon) returns the cell whose centroid covers (lat, lon)."""
    center_id = h3.latlng_to_cell(55.3, 21.1, 9)
    cells = [center_id] + list(h3.grid_ring(center_id, 1))
    mesh = H3Mesh.from_h3_cells(cells)
    lat, lon = h3.cell_to_latlng(center_id)
    idx = mesh.find_triangle(lat, lon)
    # 0 is the centre cell in our construction order.
    assert idx == 0


def test_h3mesh_find_triangle_off_mesh_returns_negative_one():
    center_id = h3.latlng_to_cell(55.3, 21.1, 9)
    cells = [center_id] + list(h3.grid_ring(center_id, 1))
    mesh = H3Mesh.from_h3_cells(cells)
    # Far away — shouldn't be in the mesh.
    idx = mesh.find_triangle(0.0, 0.0)
    assert idx == -1


def test_h3mesh_metric_scale_at_lat55():
    """metric_scale matches the Mesh contract used by Task 0.1."""
    cells = [h3.latlng_to_cell(55.3, 21.1, 9)]
    cells += list(h3.grid_ring(cells[0], 1))
    mesh = H3Mesh.from_h3_cells(cells)
    sx, sy = mesh.metric_scale(55.0)
    assert sy == 110540.0
    assert sx == pytest.approx(111320.0 * math.cos(math.radians(55.0)), rel=1e-6)


def test_h3mesh_areas_match_h3_per_cell():
    cells = [h3.latlng_to_cell(55.3, 21.1, 9)]
    cells += list(h3.grid_ring(cells[0], 1))
    mesh = H3Mesh.from_h3_cells(cells)
    expected = np.array([h3.cell_area(c, unit="m^2") for c in cells], dtype=np.float32)
    np.testing.assert_allclose(mesh.areas, expected, rtol=1e-5)


def test_h3mesh_neighbors_are_compacted_to_low_slots():
    """Valid neighbours must occupy slots [0, count); -1 only at the tail.

    Regression for a movement-kernel hazard: ``_step_directed_numba``
    initialises ``best_nbr = water_nbrs[c, 0]`` and only updates if a
    later neighbour has a better field value.  If slot 0 is -1 (because
    the first h3.grid_ring neighbour falls outside the mesh), the
    agent's tri_idx becomes -1 and downstream array indexing breaks.
    """
    # Build a mesh that excludes one ring-1 neighbour of the centre.
    center = h3.latlng_to_cell(55.3, 21.1, 9)
    ring = list(h3.grid_ring(center, 1))
    cells = [center, ring[1], ring[2], ring[3], ring[4], ring[5]]  # ring[0] dropped
    mesh = H3Mesh.from_h3_cells(cells)
    centre_row = mesh.neighbors[0]
    cnt = int(mesh._water_nbr_count[0])
    # First `cnt` slots all valid (no -1 holes), tail all -1.
    assert (centre_row[:cnt] >= 0).all(), (
        f"non-compact neighbours: row[:cnt]={centre_row[:cnt]}"
    )
    assert (centre_row[cnt:] == -1).all()


def test_h3mesh_water_nbrs_count_matches_neighbors():
    """_water_nbr_count equals the count of non-(-1) entries in each row."""
    cells = [h3.latlng_to_cell(55.3, 21.1, 9)]
    cells += list(h3.grid_ring(cells[0], 1))
    mesh = H3Mesh.from_h3_cells(cells)
    assert mesh._water_nbr_count.dtype == np.int32
    expected_counts = (mesh.neighbors >= 0).sum(axis=1).astype(np.int32)
    np.testing.assert_array_equal(mesh._water_nbr_count, expected_counts)


def test_h3mesh_default_water_mask_is_all_true():
    cells = [h3.latlng_to_cell(55.3, 21.1, 9)]
    cells += list(h3.grid_ring(cells[0], 1))
    mesh = H3Mesh.from_h3_cells(cells)
    assert mesh.water_mask.all()
    assert mesh.water_mask.dtype == bool


def test_h3mesh_explicit_water_mask_round_trips():
    cells = [h3.latlng_to_cell(55.3, 21.1, 9)]
    cells += list(h3.grid_ring(cells[0], 1))
    mask = np.array([True, False, True, False, True, False, True])
    mesh = H3Mesh.from_h3_cells(cells, water_mask=mask)
    np.testing.assert_array_equal(mesh.water_mask, mask)


def test_h3mesh_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        H3Mesh.from_h3_cells([])


def test_h3mesh_centroids_c_is_contiguous():
    """Numba kernels need contiguous arrays — centroids_c must satisfy that."""
    cells = [h3.latlng_to_cell(55.3, 21.1, 9)]
    cells += list(h3.grid_ring(cells[0], 1))
    mesh = H3Mesh.from_h3_cells(cells)
    assert mesh.centroids_c.flags["C_CONTIGUOUS"]


def test_h3mesh_h3_ids_are_uint64():
    cells = [h3.latlng_to_cell(55.3, 21.1, 9)]
    cells += list(h3.grid_ring(cells[0], 1))
    mesh = H3Mesh.from_h3_cells(cells)
    assert mesh.h3_ids.dtype == np.uint64
    # Round-trip via int_to_str / str_to_int
    for i, cell_str in enumerate(cells):
        assert h3.int_to_str(int(mesh.h3_ids[i])) == cell_str


def test_h3mesh_gradient_metric_scaled_matches_uniform_field():
    """A flat field gives zero gradient — sanity check the metric_scale path."""
    cells = [h3.latlng_to_cell(55.3, 21.1, 9)]
    cells += list(h3.grid_ring(cells[0], 1))
    mesh = H3Mesh.from_h3_cells(cells)
    field = np.ones(mesh.n_cells, dtype=np.float32)
    grad = mesh.gradient(field, 0)
    assert grad == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Task 1.3 — pentagon guard (also belongs to from_h3_cells's contract)
# ---------------------------------------------------------------------------


def test_h3mesh_from_h3_cells_raises_on_pentagon_by_default():
    """Pentagon guard fires inside from_h3_cells, not just from_polygon."""
    penta = list(h3.get_pentagons(5))[0]
    nbrs = list(h3.grid_ring(penta, 1))
    assert h3.is_pentagon(penta) and len(nbrs) == 5
    with pytest.raises(ValueError, match="pentagon"):
        H3Mesh.from_h3_cells([penta] + nbrs)


def test_h3mesh_pentagon_skip_filters_pentagons():
    penta = list(h3.get_pentagons(5))[0]
    nbrs = list(h3.grid_ring(penta, 1))
    cells = [penta] + nbrs
    depth = np.arange(len(cells), dtype=np.float32)
    mesh = H3Mesh.from_h3_cells(cells, depth=depth, pentagon_policy="skip")
    assert mesh.n_cells == 5  # 5 ring neighbours, pentagon dropped
    # Depth array got sliced consistently — pentagon's row (index 0) is gone.
    np.testing.assert_array_equal(mesh.depth, depth[1:])


def test_h3mesh_pentagon_allow_keeps_pentagons_with_5_neighbours():
    """allow lets pentagons in; their row has 5 neighbours + 1 sentinel slot."""
    penta = list(h3.get_pentagons(5))[0]
    nbrs = list(h3.grid_ring(penta, 1))
    cells = [penta] + nbrs
    mesh = H3Mesh.from_h3_cells(cells, pentagon_policy="allow")
    assert mesh.n_cells == 6
    row = mesh.neighbors[0]  # the pentagon's row
    assert (row >= 0).sum() == 5
    assert (row == -1).sum() == 1


# ---------------------------------------------------------------------------
# Task 1.2 — from_polygon factory (bbox tessellation)
# ---------------------------------------------------------------------------


def test_h3mesh_from_polygon_builds_nemunas_subbox_at_res9():
    """Small bbox inside the Nemunas delta → reasonable cell count, every
    interior cell has all 6 neighbours present."""
    ring = [
        (55.30, 21.10), (55.30, 21.20),
        (55.35, 21.20), (55.35, 21.10),
        (55.30, 21.10),
    ]
    poly = h3.LatLngPoly(ring)
    mesh = H3Mesh.from_polygon(poly, resolution=9)
    # ~5.5 km × 5 km box at res 9 (200 m cells) → expect 100-500 cells.
    assert 50 < mesh.n_cells < 800
    assert mesh.neighbors.shape == (mesh.n_cells, 6)
    # At least some cells are interior (all 6 neighbours present).
    full_rows = (mesh.neighbors >= 0).sum(axis=1) == 6
    assert full_rows.sum() > 0
    # Resolution is read off the produced cells.
    assert mesh.resolution == 9


def test_h3mesh_from_polygon_forwards_pentagon_policy():
    """from_polygon must hand pentagon_policy through to from_h3_cells.

    A small box around an icosahedral vertex covers a pentagon; default
    'raise' should refuse, 'skip' should succeed with no pentagons.
    Resolution 5 + 0.05° half-box is a known-working combo for h3-py 4.4.2
    (lower resolutions / larger boxes can hit H3FailedError around pentagons).
    """
    penta = list(h3.get_pentagons(5))[0]
    lat0, lon0 = h3.cell_to_latlng(penta)
    half = 0.05
    ring = [
        (lat0 - half, lon0 - half), (lat0 - half, lon0 + half),
        (lat0 + half, lon0 + half), (lat0 + half, lon0 - half),
        (lat0 - half, lon0 - half),
    ]
    poly = h3.LatLngPoly(ring)
    with pytest.raises(ValueError, match="pentagon"):
        H3Mesh.from_polygon(poly, resolution=5)
    # At res 5 with this 0.05° half-box around the pentagon, h3 returns
    # only the pentagon itself — skip filters it out, leaving zero cells.
    # That triggers the "all input cells were pentagons" guard.
    with pytest.raises(ValueError, match="pentagons"):
        H3Mesh.from_polygon(poly, resolution=5, pentagon_policy="skip")
    # At res 7 (smaller cells) the box covers the pentagon plus 16 hexes,
    # so skip leaves 16 valid cells — none of which are pentagons.
    mesh = H3Mesh.from_polygon(poly, resolution=7, pentagon_policy="skip")
    assert mesh.n_cells > 0
    for hid in mesh.h3_ids:
        assert not h3.is_pentagon(h3.int_to_str(int(hid)))


def test_h3mesh_from_polygon_zero_cells_raises():
    """A polygon smaller than one H3 cell at the chosen resolution
    produces zero cells — must raise ValueError, not silently build an
    empty mesh."""
    # 0.000001° ≈ 11 cm box, well below any H3 cell.
    ring = [
        (55.300000, 21.100000), (55.300000, 21.100001),
        (55.300001, 21.100001), (55.300001, 21.100000),
        (55.300000, 21.100000),
    ]
    poly = h3.LatLngPoly(ring)
    with pytest.raises(ValueError, match="zero H3 cells"):
        H3Mesh.from_polygon(poly, resolution=2)
