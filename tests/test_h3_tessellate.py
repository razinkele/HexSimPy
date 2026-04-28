"""Unit tests for salmon_ibm.h3_tessellate."""
import h3
import numpy as np
from shapely.geometry import Polygon

from salmon_ibm import h3_tessellate


def test_tessellate_reach_simple_polygon():
    # 0.02° × 0.02° square at lat ~55° (~ 2.2 km × 1.3 km).
    poly = Polygon([(21.20, 55.30), (21.22, 55.30),
                    (21.22, 55.32), (21.20, 55.32)])
    cells = h3_tessellate.tessellate_reach(poly, resolution=9)
    assert len(cells) > 0, "should tessellate to at least one cell"
    # All returned cells must be valid H3 indices at the given resolution.
    for c in cells:
        assert h3.is_valid_cell(c)
        assert h3.get_resolution(c) == 9


def test_bridge_components_connects_two_pieces():
    # Two disjoint single-cell components exactly 3 cells apart.
    # h3.grid_ring(a, 3) returns cells at distance EXACTLY 3 from a.
    # (h3.grid_disk(a, k) returns cells at distance <= k including a.)
    a = h3.latlng_to_cell(55.30, 21.20, 11)
    b = list(h3.grid_ring(a, 3))[0]
    assert h3.grid_distance(a, b) == 3, "fixture: b should be 3 cells from a"
    out = h3_tessellate.bridge_components([a, b], resolution=11, max_bridge_len=10)
    assert a in out and b in out
    assert len(out) > 2, "bridge should add intermediate cells between two distance-3 components"
