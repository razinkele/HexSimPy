"""Tests for the variable-resolution H3 mesh scaffold (v1.2.8).

Focus is the cross-resolution neighbour finder — the algorithmic core
of the multi-res backend.  The simulation-integration tests come in a
later phase; for now we just make sure the mesh structure is correct
on toy 2-resolution inputs.
"""
from __future__ import annotations

import h3
import numpy as np
import pytest

from salmon_ibm.h3_multires import (
    H3MultiResMesh,
    MAX_NBRS,
    find_cross_res_neighbours,
)


# A reference centre cell at res 9 over the Curonian Lagoon.
LAGOON_CENTRE = h3.latlng_to_cell(55.35, 21.10, 9)


# ---------------------------------------------------------------------------
# Same-resolution neighbour finding (regression: must match H3Mesh)
# ---------------------------------------------------------------------------


def test_uniform_resolution_matches_h3_grid_ring():
    """When all cells are the same resolution, the CSR table reduces to
    `h3.grid_ring`. This is the regression case — multi-res must not
    break the uniform-res behaviour."""
    cells = [LAGOON_CENTRE] + list(h3.grid_ring(LAGOON_CENTRE, 1))
    nbr_starts, nbr_idx = find_cross_res_neighbours(cells)

    # Cell 0 is the centre — should have all 6 ring neighbours.
    centre_nbrs = set(nbr_idx[nbr_starts[0]:nbr_starts[1]])
    assert centre_nbrs == {1, 2, 3, 4, 5, 6}, (
        f"centre cell should see all 6 ring neighbours; got {centre_nbrs}"
    )

    # Each ring cell should at least see the centre back.
    for i in range(1, 7):
        ring_nbrs = nbr_idx[nbr_starts[i]:nbr_starts[i + 1]]
        assert 0 in ring_nbrs, f"ring cell {i} doesn't see the centre cell"


# ---------------------------------------------------------------------------
# Cross-resolution: a coarse cell flanked by fine children
# ---------------------------------------------------------------------------


def test_coarse_cell_sees_fine_children_of_neighbours():
    """A coarse cell on the boundary of a fine zone should list the
    fine children of its (missing) coarse-resolution ring neighbours.

    Setup: 1 coarse centre (res 9) + the 7 fine children (res 10) of
    one of its ring neighbours.  The coarse centre's ring at res 9
    has 6 entries; one of them is *missing* from the mesh (replaced
    by 7 res-10 children of that cell).  The neighbour finder should
    include all 7 fine children for the coarse centre."""
    coarse_centre = LAGOON_CENTRE  # res 9
    ring = list(h3.grid_ring(coarse_centre, 1))
    # Pick one ring neighbour and replace it with its res-10 children.
    fine_zone_anchor = ring[0]
    fine_children = list(h3.cell_to_children(fine_zone_anchor, 10))
    other_ring = ring[1:]

    cells = [coarse_centre, *other_ring, *fine_children]
    nbr_starts, nbr_idx = find_cross_res_neighbours(cells)

    centre_nbrs = set(nbr_idx[nbr_starts[0]:nbr_starts[1]])
    other_ring_idx = set(range(1, 1 + len(other_ring)))
    fine_children_idx = set(
        range(1 + len(other_ring), 1 + len(other_ring) + len(fine_children))
    )

    # Coarse centre must see the 5 same-res ring neighbours that exist.
    assert other_ring_idx <= centre_nbrs, (
        f"coarse centre missing same-res neighbours; "
        f"expected {other_ring_idx} ⊂ {centre_nbrs}"
    )
    # Coarse centre must see at least *some* of the fine children of
    # the missing ring entry (those bordering the centre).
    seen_fine = centre_nbrs & fine_children_idx
    assert len(seen_fine) >= 1, (
        f"coarse centre should see fine children of missing ring entry; "
        f"got 0 of {len(fine_children)}"
    )


# ---------------------------------------------------------------------------
# Cross-resolution: a fine cell on the boundary of a coarse zone
# ---------------------------------------------------------------------------


def test_fine_cell_sees_coarse_parent_neighbour():
    """A fine cell on the boundary of a coarse zone should see the
    coarse cell as a neighbour, even though that coarse cell isn't a
    same-resolution ring neighbour.

    Setup: a single coarse cell (res 9) + the 7 fine children (res
    10) of an adjacent res-9 cell.  Each fine child whose centroid is
    spatially adjacent to the coarse cell should list the coarse cell
    as a neighbour."""
    coarse_centre = LAGOON_CENTRE
    ring = list(h3.grid_ring(coarse_centre, 1))
    fine_zone_anchor = ring[0]
    fine_children = list(h3.cell_to_children(fine_zone_anchor, 10))

    cells = [coarse_centre, *fine_children]  # 1 + 7 cells
    nbr_starts, nbr_idx = find_cross_res_neighbours(cells)

    # At least one fine child must list the coarse centre (idx 0)
    # as a neighbour — the children physically touching the boundary.
    saw_coarse = 0
    for fc_i in range(1, len(cells)):
        nbrs = set(nbr_idx[nbr_starts[fc_i]:nbr_starts[fc_i + 1]])
        if 0 in nbrs:
            saw_coarse += 1
    assert saw_coarse >= 1, (
        "no fine child saw the coarse parent neighbour — cross-res "
        "lookup is broken"
    )


# ---------------------------------------------------------------------------
# Sanity: pentagons stay handled
# ---------------------------------------------------------------------------


def test_pentagon_doesnt_crash():
    """Cells near an H3 pentagon have only 5 ring-1 neighbours, not 6.
    The neighbour finder must not crash."""
    # H3 pentagons live at fixed positions; one of them is in the North
    # Atlantic at res 0.  Get a pentagon at res 9 by descending into
    # one of its children.
    res0_cells = list(h3.get_res0_cells())
    pentagon_res0 = next(c for c in res0_cells if h3.is_pentagon(c))
    pentagon_res9 = list(h3.cell_to_children(pentagon_res0, 9))[0]

    cells = [pentagon_res9, *list(h3.grid_ring(pentagon_res9, 1))]
    # Should not raise.
    nbr_starts, nbr_idx = find_cross_res_neighbours(cells)
    pentagon_nbrs = nbr_idx[nbr_starts[0]:nbr_starts[1]]
    # H3 pentagons have 5 ring-1 neighbours — pentagon cell IS in
    # the input, so it should see at least 4-5 of them.
    assert len(pentagon_nbrs) >= 4


# ---------------------------------------------------------------------------
# H3MultiResMesh — basic construction
# ---------------------------------------------------------------------------


def test_mesh_construction_attributes():
    """Mesh exposes all the expected attributes for downstream consumers."""
    cells = [LAGOON_CENTRE, *list(h3.grid_ring(LAGOON_CENTRE, 1))]
    mesh = H3MultiResMesh.from_h3_cells(cells)

    assert mesh.n_cells == 7
    assert mesh.h3_ids.shape == (7,) and mesh.h3_ids.dtype == np.uint64
    assert mesh.resolutions.shape == (7,) and (mesh.resolutions == 9).all()
    assert mesh.centroids.shape == (7, 2)
    assert mesh.water_mask.dtype == bool
    assert mesh.depth.shape == (7,)
    assert mesh.areas.shape == (7,)

    # CSR neighbour table integrity
    assert mesh.nbr_starts.shape == (8,) and mesh.nbr_starts[0] == 0
    assert mesh.nbr_idx.shape[0] == int(mesh.nbr_starts[-1])
    # Padded compat view
    assert mesh.neighbors.shape == (7, MAX_NBRS)
    assert (mesh.neighbors[mesh.neighbors != -1] >= 0).all()


def test_mesh_uniform_res_neighbours_match_h3mesh():
    """For uniform-res input, H3MultiResMesh's neighbour table must
    match the same set H3Mesh produces — the multi-res backend must
    not be a regression on the uniform case."""
    from salmon_ibm.h3mesh import H3Mesh

    cells = [LAGOON_CENTRE, *list(h3.grid_ring(LAGOON_CENTRE, 1))]
    multi = H3MultiResMesh.from_h3_cells(cells)
    single = H3Mesh.from_h3_cells(cells)

    # Compare neighbour SETS for each cell — order may differ.
    for i in range(multi.n_cells):
        m_nbrs = set(multi.neighbours_of(i).tolist())
        s_nbrs = set(int(n) for n in single.neighbors[i] if n >= 0)
        assert m_nbrs == s_nbrs, (
            f"cell {i}: multi-res {m_nbrs} != H3Mesh {s_nbrs}"
        )


def test_mesh_mixed_resolution():
    """Build a 2-resolution mesh and confirm cross-res neighbours land
    in the padded view."""
    coarse = LAGOON_CENTRE  # res 9
    fine = list(h3.cell_to_children(list(h3.grid_ring(coarse, 1))[0], 10))

    cells = [coarse] + fine
    mesh = H3MultiResMesh.from_h3_cells(cells)
    assert mesh.resolutions[0] == 9
    assert (mesh.resolutions[1:] == 10).all()

    # The coarse centre must see at least 1 of the fine children.
    coarse_nbrs = set(mesh.neighbours_of(0).tolist())
    fine_indices = set(range(1, len(cells)))
    assert (coarse_nbrs & fine_indices), (
        "coarse cell should see at least one fine-zone neighbour"
    )


def test_reach_helpers():
    """`cells_in_reach` and `reach_name_of` work with multi-res meshes."""
    cells = [LAGOON_CENTRE, *list(h3.grid_ring(LAGOON_CENTRE, 1))]
    reach_id = np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    reach_names = ["Centre", "Ring"]
    mesh = H3MultiResMesh.from_h3_cells(
        cells, reach_id=reach_id, reach_names=reach_names,
    )
    assert mesh.reach_name_of(0) == "Centre"
    assert mesh.reach_name_of(3) == "Ring"
    assert list(mesh.cells_in_reach("Ring")) == [1, 2, 3, 4, 5, 6]
    assert list(mesh.cells_in_reach("Centre")) == [0]
    assert list(mesh.cells_in_reach("Nope")) == []
