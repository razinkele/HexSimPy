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


def test_parse_upload_gpkg_reads_first_layer():
    bytes_ = (FIXTURES / "tiny_wgs84.gpkg").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".gpkg")
    assert geom is not None
    assert geom.geom_type in ("Polygon", "MultiPolygon")


def test_parse_upload_shp_zip_extracts_correctly():
    bytes_ = (FIXTURES / "tiny_3035.shp.zip").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".shp.zip")
    assert geom is not None
    assert geom.geom_type in ("Polygon", "MultiPolygon")


def test_parse_upload_dissolves_multi_feature_to_single():
    """tiny_3035.shp.zip has 3 features (a + b adjacent, c disjoint).
    After dissolve we expect a MultiPolygon with 2 parts (a+b merged, c separate)."""
    bytes_ = (FIXTURES / "tiny_3035.shp.zip").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".shp.zip")
    if geom.geom_type == "MultiPolygon":
        assert len(geom.geoms) == 2


def test_parse_upload_reprojects_from_3035_to_4326():
    """tiny_3035.shp.zip is in EPSG:3035 (LAEA). After parse, geometry must be in WGS84."""
    bytes_ = (FIXTURES / "tiny_3035.shp.zip").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".shp.zip")
    minx, miny, maxx, maxy = geom.bounds
    assert -180 <= minx <= maxx <= 180
    assert -90 <= miny <= maxy <= 90
    assert 10 < minx < 30, f"Expected European longitude, got {minx}"
    assert 50 < miny < 60, f"Expected northern-Europe latitude, got {miny}"


import pytest


def test_parse_upload_raises_on_missing_crs():
    import geopandas as gpd
    from shapely.geometry import Polygon
    gdf = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        h3_tessellate._dissolve_and_validate(gdf)


def test_parse_upload_raises_on_no_geometry():
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    with pytest.raises(ValueError, match="no polygon geometry"):
        h3_tessellate._dissolve_and_validate(gdf)


def test_parse_upload_raises_on_non_polygon_features():
    import geopandas as gpd
    from shapely.geometry import LineString
    gdf = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
    with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
        h3_tessellate._dissolve_and_validate(gdf)


def test_parse_upload_raises_on_antimeridian_crossing():
    import geopandas as gpd
    from shapely.geometry import Polygon
    poly = Polygon([(170, 0), (170, 10), (-170, 10), (-170, 0)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    with pytest.raises(ValueError, match="antimeridian"):
        h3_tessellate._dissolve_and_validate(gdf)


def test_preview_returns_consistent_dataclass():
    bytes_ = (FIXTURES / "tiny.geojson").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".geojson")
    mesh = h3_tessellate.preview(geom, resolution=9)
    assert isinstance(mesh, h3_tessellate.PreviewMesh)
    assert mesh.h3_ids.dtype == np.uint64
    assert mesh.water_mask.dtype == np.uint8
    assert mesh.depth.dtype == np.float32


def test_preview_water_mask_all_ones():
    bytes_ = (FIXTURES / "tiny.geojson").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".geojson")
    mesh = h3_tessellate.preview(geom, resolution=9)
    assert (mesh.water_mask == 1).all(), (
        "By construction, all cells in the upload preview are water"
    )


def test_preview_with_bathy_off_returns_zero_depth():
    bytes_ = (FIXTURES / "tiny.geojson").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".geojson")
    mesh = h3_tessellate.preview(geom, resolution=9, with_bathy=False)
    assert (mesh.depth == 0.0).all(), "with_bathy=False → depth all zero"


def test_preview_caps_cell_count_at_max_cells():
    bytes_ = (FIXTURES / "tiny.geojson").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".geojson")
    with pytest.raises(ValueError, match=r"would produce.*cells.*max"):
        h3_tessellate.preview(geom, resolution=9, max_cells=1)


def test_preview_mesh_dataclass_post_init_assertion():
    """All array fields must have matching length."""
    n = 5
    pm = h3_tessellate.PreviewMesh(
        h3_ids=np.zeros(n, dtype=np.uint64),
        resolutions=np.full(n, 9, dtype=np.int8),
        centroids=np.zeros((n, 2)),
        reach_id=np.zeros(n, dtype=np.int8),
        reach_names=["uploaded_polygon"],
        depth=np.zeros(n, dtype=np.float32),
        water_mask=np.ones(n, dtype=np.uint8),
        polygon_outlines=[],
    )
    assert pm.h3_ids.shape == (n,)
    with pytest.raises(AssertionError):
        h3_tessellate.PreviewMesh(
            h3_ids=np.zeros(n, dtype=np.uint64),
            resolutions=np.full(n - 1, 9, dtype=np.int8),
            centroids=np.zeros((n, 2)),
            reach_id=np.zeros(n, dtype=np.int8),
            reach_names=["uploaded_polygon"],
            depth=np.zeros(n, dtype=np.float32),
            water_mask=np.ones(n, dtype=np.uint8),
            polygon_outlines=[],
        )
