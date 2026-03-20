"""Ground-truth HexSim I/O tests using HexSim 4.0.20 workspace fixtures."""
from pathlib import Path

import numpy as np
import pytest

from heximpy.hxnparser import GridMeta, HexMap, Workspace

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
