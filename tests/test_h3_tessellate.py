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


def test_polygon_trust_water_mask_flips_dry_to_wet():
    # 5 cells: cells 0-3 tagged with reach_id=0; cell 4 untagged.
    # cells 1 and 4 are "dry" per EMODnet (depth=0, water_mask=0).
    # Trust override should flip cell 1 (tagged AND dry) but leave cell 4.
    reach_id = np.array([0, 0, 0, 0, -1], dtype=np.int8)
    water_mask = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    depth = np.array([5.0, 0.0, 3.0, 2.0, 0.0], dtype=np.float32)

    new_water, new_depth = h3_tessellate.polygon_trust_water_mask(
        reach_id, water_mask, depth)

    expected_water = np.array([1, 1, 1, 1, 0], dtype=np.uint8)  # cell 1 flipped, cell 4 left
    expected_depth = np.array([5.0, 1.0, 3.0, 2.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(new_water, expected_water)
    np.testing.assert_array_equal(new_depth, expected_depth)


import io
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "create_model"


def test_parse_upload_geojson_round_trips():
    bytes_ = (FIXTURES / "tiny.geojson").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".geojson")
    assert geom is not None
    assert geom.geom_type in ("Polygon", "MultiPolygon")
    # Round-trip: bbox should contain the centroid (21.21, 55.31).
    minx, miny, maxx, maxy = geom.bounds
    assert minx <= 21.21 <= maxx
    assert miny <= 55.31 <= maxy
