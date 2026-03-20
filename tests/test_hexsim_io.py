"""Ground-truth HexSim I/O tests using HexSim 4.0.20 workspace fixtures."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from heximpy.hxnparser import GridMeta, HexMap, Workspace, read_barriers

# ── Fixture paths ────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent

WIDE_HXN = BASE / "HexSim Examples" / "Spatial Data" / "Hexagons" / "Habitat Map" / "Habitat Map.1.hxn"
WIDE_GRID = BASE / "HexSim Examples" / "HexSim Examples.grid"
NARROW_HXN = BASE / "Columbia [small]" / "Spatial Data" / "Hexagons" / "River [ extent ]" / "River [ extent ].1.hxn"
NARROW_GRID = BASE / "Columbia [small]" / "Columbia Fish Model [small].grid"
BARRIER_FILE = BASE / "Columbia [small]" / "Spatial Data" / "barriers" / "Fish Ladder Available" / "Fish Ladder Available.1.hbf"

NARROW_WORKSPACE = BASE / "Columbia [small]"

# Skip all tests if fixture files are not present
pytestmark = pytest.mark.skipif(
    not WIDE_HXN.exists() or not NARROW_HXN.exists(),
    reason="HexSim workspace fixtures not available",
)


def _pointy_top_neighbors(row, col, height, width, flag):
    """Compute neighbors using pointy-top odd-row convention."""
    if row % 2 == 0:
        offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    else:
        offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

    result = []
    for dr, dc in offsets:
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width:
            row_width = width if flag == 0 or nr % 2 == 0 else width - 1
            if nc < row_width:
                result.append((nr, nc))
    return result


def _pointy_top_center(row, col, edge):
    """Compute hex center using pointy-top odd-row convention."""
    cx = np.sqrt(3.0) * edge * (col + 0.5 * (row % 2))
    cy = 1.5 * edge * row
    return cx, cy


class TestDataLayoutVerification:
    """Verify that pointy-top odd-row offsets produce geometrically adjacent centroids."""

    def test_neighbor_centroids_are_adjacent(self):
        """Load narrow workspace, verify neighbor distance ≈ √3 × edge."""
        grid = GridMeta.from_file(NARROW_GRID)
        edge = grid.edge
        expected_dist = np.sqrt(3.0) * edge

        hm = HexMap.from_file(NARROW_HXN)
        h, w = hm.height, hm.width

        row_list, col_list = [], []
        for r in range(h):
            rw = w if hm.flag == 0 or r % 2 == 0 else w - 1
            row_list.append(np.full(rw, r, dtype=np.int32))
            col_list.append(np.arange(rw, dtype=np.int32))
        all_rows = np.concatenate(row_list)
        all_cols = np.concatenate(col_list)

        interior = [
            i for i in range(len(all_rows))
            if 2 <= all_rows[i] < h - 2 and 2 <= all_cols[i] < w - 2
            and hm.values[i] != 0.0
        ]
        rng = np.random.default_rng(42)
        sample = rng.choice(interior, size=min(200, len(interior)), replace=False)

        bad = []
        for idx in sample:
            r, c = int(all_rows[idx]), int(all_cols[idx])
            cx0, cy0 = _pointy_top_center(r, c, edge)
            nbrs = _pointy_top_neighbors(r, c, h, w, hm.flag)
            for nr, nc in nbrs:
                cx1, cy1 = _pointy_top_center(nr, nc, edge)
                dist = np.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)
                rel_err = abs(dist - expected_dist) / expected_dist
                if rel_err > 0.01:
                    bad.append((r, c, nr, nc, dist, expected_dist))

        assert not bad, f"Non-adjacent neighbors found: {bad[:5]}"


from salmon_ibm.hexsim import _hex_neighbors_offset


class TestNarrowGridNeighbors:
    """Test neighbor computation for narrow grids (flag=1)."""

    def test_rowcol_to_flat_narrow_basic(self):
        """Verify flat index computation for narrow grid, width=5."""
        from salmon_ibm.hexsim import _rowcol_to_flat_narrow
        assert _rowcol_to_flat_narrow(0, 0, 5) == 0
        assert _rowcol_to_flat_narrow(0, 4, 5) == 4
        assert _rowcol_to_flat_narrow(1, 0, 5) == 5
        assert _rowcol_to_flat_narrow(1, 3, 5) == 8
        assert _rowcol_to_flat_narrow(2, 0, 5) == 9
        assert _rowcol_to_flat_narrow(2, 4, 5) == 13

    def test_neighbor_symmetry_narrow_grid(self):
        """If A is neighbor of B, then B must be neighbor of A."""
        hm = HexMap.from_file(NARROW_HXN)
        h, w, flag = hm.height, hm.width, hm.flag
        n_data = len(hm.values)

        row_list, col_list = [], []
        for r in range(h):
            rw = w if r % 2 == 0 else w - 1
            row_list.append(np.full(rw, r, dtype=np.int32))
            col_list.append(np.arange(rw, dtype=np.int32))
        all_rows = np.concatenate(row_list)
        all_cols = np.concatenate(col_list)

        rng = np.random.default_rng(42)
        sample = rng.choice(len(all_rows), size=min(500, len(all_rows)), replace=False)

        asymmetric = []
        for idx in sample:
            r, c = int(all_rows[idx]), int(all_cols[idx])
            nbrs = _hex_neighbors_offset(r, c, w, h, n_data, flag)
            for nbr_flat in nbrs:
                nr, nc = int(all_rows[nbr_flat]), int(all_cols[nbr_flat])
                reverse_nbrs = _hex_neighbors_offset(nr, nc, w, h, n_data, flag)
                if idx not in reverse_nbrs:
                    asymmetric.append((idx, r, c, nbr_flat, nr, nc))

        assert not asymmetric, f"Asymmetric neighbors: {asymmetric[:5]}"

    def test_interior_cells_have_six_neighbors(self):
        """Interior cells in narrow grid should have exactly 6 neighbors."""
        hm = HexMap.from_file(NARROW_HXN)
        h, w, flag = hm.height, hm.width, hm.flag
        n_data = len(hm.values)

        nbrs = _hex_neighbors_offset(4, 2, w, h, n_data, flag)
        assert len(nbrs) == 6, f"Expected 6 neighbors, got {len(nbrs)}"


class TestHxnparserNarrowGrid:
    """Test hxnparser methods handle narrow grids correctly."""

    def test_neighbors_pointy_top(self):
        """HexMap.neighbors() uses pointy-top odd-row convention."""
        hm = HexMap.from_file(NARROW_HXN)
        # Even row: diagonal neighbors at lower col indices
        nbrs_even = hm.neighbors(2, 2)
        assert (1, 1) in nbrs_even, "Even row upper-left should be (-1,-1)"
        # Odd row: diagonal neighbors at higher col indices
        nbrs_odd = hm.neighbors(3, 2)
        assert (2, 3) in nbrs_odd, "Odd row upper-right should be (-1,+1)"

    def test_to_geodataframe_narrow_grid(self):
        """to_geodataframe should not crash on narrow grids."""
        hm = HexMap.from_file(NARROW_HXN)
        if hm.flag != 1:
            pytest.skip("Not a narrow grid")
        gdf = hm.to_geodataframe(edge=1.0, include_empty=False)
        assert len(gdf) > 0

    def test_to_geotiff_narrow_grid_raises(self):
        """to_geotiff should raise ValueError for narrow grids."""
        hm = HexMap.from_file(NARROW_HXN)
        if hm.flag != 1:
            pytest.skip("Not a narrow grid")
        with pytest.raises(ValueError, match="narrow"):
            hm.to_geotiff("/tmp/test.tif")


class TestDimensionSwapSafety:
    """Test GridMeta ↔ HexMap dimension mapping."""

    def test_data_height_width_properties(self):
        """GridMeta.data_height/data_width match HexMap.height/width."""
        grid = GridMeta.from_file(NARROW_GRID)
        hm = HexMap.from_file(NARROW_HXN)
        assert grid.data_height == hm.height
        assert grid.data_width == hm.width

    def test_wide_grid_dimensions(self):
        """Same check for wide grid."""
        grid = GridMeta.from_file(WIDE_GRID)
        hm = HexMap.from_file(WIDE_HXN)
        assert grid.data_height == hm.height
        assert grid.data_width == hm.width


class TestRoundTripFidelity:
    """Write → read round-trip must preserve values exactly."""

    def test_wide_grid_roundtrip(self):
        hm = HexMap.from_file(WIDE_HXN)
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            hm.to_file(tmp)
            hm2 = HexMap.from_file(tmp)
            np.testing.assert_array_equal(hm.values, hm2.values)
            assert hm.width == hm2.width
            assert hm.height == hm2.height
            assert hm.flag == hm2.flag
        finally:
            tmp.unlink(missing_ok=True)

    def test_narrow_grid_roundtrip(self):
        hm = HexMap.from_file(NARROW_HXN)
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            hm.to_file(tmp)
            hm2 = HexMap.from_file(tmp)
            np.testing.assert_array_equal(hm.values, hm2.values)
            assert hm.width == hm2.width
            assert hm.height == hm2.height
            assert hm.flag == hm2.flag
        finally:
            tmp.unlink(missing_ok=True)


class TestNarrowGridCellCount:
    """Verify n_hexagons matches actual data array length."""

    def test_narrow_grid_cell_count(self):
        hm = HexMap.from_file(NARROW_HXN)
        assert hm.n_hexagons == len(hm.values), (
            f"n_hexagons={hm.n_hexagons} != len(values)={len(hm.values)}"
        )

    def test_wide_grid_cell_count(self):
        hm = HexMap.from_file(WIDE_HXN)
        assert hm.n_hexagons == len(hm.values)
        assert hm.n_hexagons == hm.width * hm.height


class TestBarrierFileParsing:
    """Verify barrier file entries are within grid bounds."""

    @pytest.mark.skipif(not BARRIER_FILE.exists(), reason="Barrier file not available")
    def test_barrier_hex_ids_in_bounds(self):
        barriers = read_barriers(BARRIER_FILE)
        grid = GridMeta.from_file(NARROW_GRID)
        hm = HexMap.from_file(NARROW_HXN)
        n_hex = hm.n_hexagons
        for b in barriers:
            assert 0 <= b.hex_id < n_hex, f"Barrier hex_id {b.hex_id} out of bounds"
            assert 0 <= b.edge <= 5, f"Barrier edge {b.edge} out of range"
