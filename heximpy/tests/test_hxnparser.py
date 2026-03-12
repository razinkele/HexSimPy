"""Tests for heximpy.hxnparser module."""
import math
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from heximpy.hxnparser import Barrier, GridMeta, HexMap, read_barriers

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
HEXIMPY_DIR = Path(__file__).resolve().parent.parent
SALMON_DIR = HEXIMPY_DIR.parent

HABITAT_HXN = HEXIMPY_DIR / "HabitatMap.hxn"
RIVER_HXN = HEXIMPY_DIR / "River.hxn"
STUDY_HXN = HEXIMPY_DIR / "StudyArea.hxn"
GRID_FILE = SALMON_DIR / "Columbia [small]" / "Columbia Fish Model [small].grid"
HBF_FILE = (
    SALMON_DIR
    / "Columbia [small]"
    / "Spatial Data"
    / "barriers"
    / "Fish Ladder Available"
    / "Fish Ladder Available.1.hbf"
)

has_habitat = pytest.mark.skipif(
    not HABITAT_HXN.exists(), reason="HabitatMap.hxn not available"
)
has_river = pytest.mark.skipif(
    not RIVER_HXN.exists(), reason="River.hxn not available"
)
has_study = pytest.mark.skipif(
    not STUDY_HXN.exists(), reason="StudyArea.hxn not available"
)
has_grid = pytest.mark.skipif(
    not GRID_FILE.exists(), reason=".grid file not available"
)
has_hbf = pytest.mark.skipif(
    not HBF_FILE.exists(), reason=".hbf file not available"
)


# ===================================================================
# Task 1 – Dataclasses and n_hexagons property
# ===================================================================
class TestHexMapDataclass:
    """Test HexMap creation and n_hexagons computation."""

    def test_create_hexmap(self):
        hm = HexMap(
            format="patch_hexmap",
            version=8,
            height=100,
            width=50,
            flag=0,
            max_val=1.0,
            min_val=0.0,
            hexzero=0.0,
            values=np.zeros(5000, dtype=np.float32),
        )
        assert hm.version == 8
        assert hm.height == 100
        assert hm.width == 50

    def test_n_hexagons_wide(self):
        """flag=0 (wide): width * height."""
        hm = HexMap(
            format="patch_hexmap", version=8, height=10, width=5, flag=0,
            max_val=1.0, min_val=0.0, hexzero=0.0,
            values=np.zeros(50, dtype=np.float32),
        )
        assert hm.n_hexagons == 50

    def test_n_hexagons_narrow_even(self):
        """flag=1, even height: n_wide=height/2, n_narrow=n_wide."""
        # height=10, width=5 -> n_wide=5, n_narrow=5
        # total = 5*5 + 4*5 = 25+20 = 45
        hm = HexMap(
            format="patch_hexmap", version=8, height=10, width=5, flag=1,
            max_val=1.0, min_val=0.0, hexzero=0.0,
            values=np.zeros(45, dtype=np.float32),
        )
        assert hm.n_hexagons == 45

    def test_n_hexagons_narrow_odd(self):
        """flag=1, odd height: n_wide=(height+1)/2, n_narrow=n_wide-1."""
        # height=11, width=5 -> n_wide=6, n_narrow=5
        # total = 5*6 + 4*5 = 30+20 = 50
        hm = HexMap(
            format="patch_hexmap", version=8, height=11, width=5, flag=1,
            max_val=1.0, min_val=0.0, hexzero=0.0,
            values=np.zeros(50, dtype=np.float32),
        )
        assert hm.n_hexagons == 50


class TestGridMetaDataclass:
    """Test GridMeta creation."""

    def test_create_gridmeta(self):
        gm = GridMeta(
            ncols=10,
            nrows=20,
            x_extent=100.0,
            y_extent=200.0,
            row_spacing=24.0,
            edge=24.0 / math.sqrt(3),
            version=6,
            n_hexes=100,
            flag=1,
        )
        assert gm.n_hexes == 100
        assert gm.edge == pytest.approx(24.0 / math.sqrt(3))


class TestBarrierDataclass:
    """Test Barrier creation."""

    def test_create_barrier(self):
        b = Barrier(hex_id=123, edge=2, classification=1, class_name="Dam")
        assert b.hex_id == 123
        assert b.edge == 2
        assert b.classification == 1
        assert b.class_name == "Dam"


# ===================================================================
# Task 2 – HexMap.from_file (PATCH_HEXMAP and plain)
# ===================================================================
class TestHexMapFromFile:
    """Test reading .hxn files."""

    @has_habitat
    def test_read_habitat_hxn(self):
        hm = HexMap.from_file(HABITAT_HXN)
        assert hm.version == 8
        assert hm.height == 1033
        assert hm.width == 757
        assert hm.flag == 1
        assert hm.max_val == pytest.approx(1.0)
        assert hm.min_val == pytest.approx(0.0)
        assert len(hm.values) == hm.n_hexagons

    @has_habitat
    def test_header_size_is_37(self):
        """The PATCH_HEXMAP header is 37 bytes, not 25."""
        with open(HABITAT_HXN, "rb") as f:
            f.read(12)  # magic
            f.read(12)  # ver, height, width
            f.read(1)   # flag
            f.read(12)  # max, min, hexzero
            assert f.tell() == 37

    @has_river
    def test_read_river_hxn(self):
        hm = HexMap.from_file(RIVER_HXN)
        assert hm.height == 1574
        assert hm.width == 10195
        assert hm.flag == 1
        assert len(hm.values) == hm.n_hexagons

    @has_study
    def test_read_study_hxn(self):
        hm = HexMap.from_file(STUDY_HXN)
        assert hm.height == 1574
        assert hm.width == 10195
        assert len(hm.values) == hm.n_hexagons

    @has_grid
    def test_grid_file_rejected(self):
        """A .grid file should raise ValueError, not be parsed as HexMap."""
        with pytest.raises(ValueError, match="PATCH_GRID"):
            HexMap.from_file(GRID_FILE)

    @has_habitat
    @has_grid
    def test_habitat_dims_vs_grid(self):
        """HabitatMap uses a different (smaller) grid; just verify it loads."""
        hm = HexMap.from_file(HABITAT_HXN)
        gm = GridMeta.from_file(GRID_FILE)
        # Different workspaces so dims won't match, but both should parse.
        assert hm.n_hexagons > 0
        assert gm.n_hexes > 0

    def test_read_plain_format(self):
        """Synthesize a plain-format .hxn and read it back."""
        width, height = 4, 3
        n = width * height
        vals = np.arange(n, dtype=np.float32)
        header = struct.pack(
            "<iiidddi i",
            1,           # version
            width,       # width
            height,      # height
            1.0,         # cell_size
            0.0,         # origin_x
            0.0,         # origin_y
            1,           # dtype_code (float32)
            -9999,       # nodata
        )
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            f.write(header)
            f.write(vals.tobytes())
            tmp = Path(f.name)
        try:
            hm = HexMap.from_file(tmp)
            assert hm.height == height
            assert hm.width == width
            assert hm.flag == 0
            assert len(hm.values) == n
            np.testing.assert_array_equal(hm.values, vals)
        finally:
            tmp.unlink()

    def test_read_plain_int32(self):
        """Plain format with dtype_code=2 (int32 values stored, returned as float32)."""
        width, height = 2, 2
        n = width * height
        int_vals = np.array([10, 20, 30, 40], dtype=np.int32)
        header = struct.pack(
            "<iiidddi i",
            1, width, height,
            1.0, 0.0, 0.0,
            2,      # int32
            -9999,
        )
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            f.write(header)
            f.write(int_vals.tobytes())
            tmp = Path(f.name)
        try:
            hm = HexMap.from_file(tmp)
            assert hm.values.dtype == np.float32
            np.testing.assert_array_equal(hm.values, int_vals.astype(np.float32))
        finally:
            tmp.unlink()


# ===================================================================
# Task 3 – GridMeta.from_file and read_barriers
# ===================================================================
class TestGridMetaFromFile:
    """Test reading .grid files."""

    @has_grid
    def test_read_grid(self):
        gm = GridMeta.from_file(GRID_FILE)
        assert gm.version == 6
        assert gm.n_hexes == 16_046_143
        assert gm.ncols == 1574
        assert gm.nrows == 10195
        assert gm.flag == 1
        assert gm.row_spacing == pytest.approx(24.028114141347544, rel=1e-6)
        assert gm.edge == pytest.approx(24.028114141347544 / math.sqrt(3), rel=1e-6)

    @has_grid
    @has_river
    def test_grid_dims_match_river_hxn(self):
        """Grid width/height should correspond to River.hxn height/width."""
        gm = GridMeta.from_file(GRID_FILE)
        hm = HexMap.from_file(RIVER_HXN)
        # The .hxn stores (height, width) and .grid stores (width, height)
        # but n_hexes should match
        assert gm.n_hexes == hm.n_hexagons

    @has_habitat
    def test_hxn_file_rejected(self):
        """An .hxn file should raise ValueError when read as GridMeta."""
        with pytest.raises(ValueError):
            GridMeta.from_file(HABITAT_HXN)


class TestReadBarriers:
    """Test barrier file parsing."""

    @has_hbf
    def test_read_real_hbf(self):
        barriers = read_barriers(HBF_FILE)
        assert len(barriers) > 0
        # All barriers should have class_name populated
        assert all(b.class_name for b in barriers)
        # Check first barrier properties
        b = barriers[0]
        assert isinstance(b.hex_id, int)
        assert isinstance(b.edge, int)
        assert isinstance(b.classification, int)
        assert isinstance(b.class_name, str)

    @has_hbf
    def test_barrier_class_names_from_real(self):
        barriers = read_barriers(HBF_FILE)
        class_names = {b.class_name for b in barriers}
        assert "Fish Ladder Entry" in class_names

    def test_synthetic_hbf(self):
        """Parse a synthetic .hbf file."""
        content = (
            'C 1 0 0 "Dam"\n'
            'C 2 0 1 "Fish Ladder"\n'
            "E 100 2 1\n"
            "E 200 3 2\n"
            "E 300 0 1\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".hbf", delete=False
        ) as f:
            f.write(content)
            tmp = Path(f.name)
        try:
            barriers = read_barriers(tmp)
            assert len(barriers) == 3
            assert barriers[0] == Barrier(
                hex_id=100, edge=2, classification=1, class_name="Dam"
            )
            assert barriers[1] == Barrier(
                hex_id=200, edge=3, classification=2, class_name="Fish Ladder"
            )
            assert barriers[2] == Barrier(
                hex_id=300, edge=0, classification=1, class_name="Dam"
            )
        finally:
            tmp.unlink()

    def test_hbf_missing_classification_fallback(self):
        """Edge referencing unknown classification should fall back to empty string."""
        content = "E 100 2 1\n"  # no C line for classification=1
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".hbf", delete=False
        ) as f:
            f.write(content)
            tmp = Path(f.name)
        try:
            barriers = read_barriers(tmp)
            assert len(barriers) == 1
            assert barriers[0].classification == 1
            assert barriers[0].class_name == ""
        finally:
            tmp.unlink()


# ===================================================================
# Task 4 – HexMap.to_file (write support)
# ===================================================================
class TestHexMapToFile:
    """Test writing .hxn files and round-tripping."""

    def test_roundtrip_patch_hexmap(self):
        """Create a PATCH_HEXMAP, write, read back, compare all fields."""
        vals = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
        hm = HexMap(
            format="patch_hexmap", version=8, height=2, width=2, flag=0,
            max_val=3.0, min_val=0.5, hexzero=-1.0, values=vals,
        )
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            hm.to_file(tmp)
            hm2 = HexMap.from_file(tmp)
            assert hm2.format == "patch_hexmap"
            assert hm2.version == 8
            assert hm2.height == 2
            assert hm2.width == 2
            assert hm2.flag == 0
            assert hm2.max_val == pytest.approx(3.0)
            assert hm2.min_val == pytest.approx(0.5)
            assert hm2.hexzero == pytest.approx(-1.0)
            np.testing.assert_array_equal(hm2.values, vals)
        finally:
            tmp.unlink()

    def test_roundtrip_plain(self):
        """Create a plain-format HexMap, write, read back, compare all fields."""
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)
        hm = HexMap(
            format="plain", version=1, height=2, width=3, flag=0,
            max_val=60.0, min_val=10.0, hexzero=0.0, values=vals,
            cell_size=5.0, origin=(100.0, 200.0), nodata=-9999, dtype_code=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            hm.to_file(tmp)
            hm2 = HexMap.from_file(tmp)
            assert hm2.format == "plain"
            assert hm2.version == 1
            assert hm2.height == 2
            assert hm2.width == 3
            assert hm2.cell_size == pytest.approx(5.0)
            assert hm2.origin == pytest.approx((100.0, 200.0))
            assert hm2.nodata == -9999
            assert hm2.dtype_code == 1
            np.testing.assert_array_equal(hm2.values, vals)
        finally:
            tmp.unlink()

    @has_habitat
    def test_roundtrip_real_file(self):
        """Read HabitatMap.hxn, write, read back, compare values."""
        hm = HexMap.from_file(HABITAT_HXN)
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            hm.to_file(tmp)
            hm2 = HexMap.from_file(tmp)
            assert hm2.format == hm.format
            assert hm2.version == hm.version
            assert hm2.height == hm.height
            assert hm2.width == hm.width
            assert hm2.flag == hm.flag
            assert hm2.max_val == pytest.approx(hm.max_val)
            assert hm2.min_val == pytest.approx(hm.min_val)
            np.testing.assert_array_equal(hm2.values, hm.values)
        finally:
            tmp.unlink()

    def test_format_override(self):
        """Write a patch_hexmap as plain, read back, verify format."""
        vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        hm = HexMap(
            format="patch_hexmap", version=1, height=2, width=2, flag=0,
            max_val=4.0, min_val=1.0, hexzero=0.0, values=vals,
            cell_size=1.0, origin=(0.0, 0.0), nodata=0, dtype_code=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            hm.to_file(tmp, format="plain")
            hm2 = HexMap.from_file(tmp)
            assert hm2.format == "plain"
            np.testing.assert_array_equal(hm2.values, vals)
        finally:
            tmp.unlink()


# ===================================================================
# Task 5 – Hex geometry methods
# ===================================================================
def _make_hex(height=10, width=8, edge=10.0):
    """Helper to create a HexMap with a known _edge for geometry tests."""
    n = height * width
    return HexMap(
        format="patch_hexmap", version=8, height=height, width=width, flag=0,
        max_val=1.0, min_val=0.0, hexzero=0.0,
        values=np.zeros(n, dtype=np.float32),
        _edge=edge,
    )


class TestNeighbors:
    """Test HexMap.neighbors (even-q flat-top)."""

    def test_neighbors_even_col(self):
        hm = _make_hex()
        result = sorted(hm.neighbors(3, 4))
        expected = sorted([(2, 3), (2, 4), (2, 5), (3, 3), (3, 5), (4, 4)])
        assert result == expected

    def test_neighbors_odd_col(self):
        hm = _make_hex()
        result = sorted(hm.neighbors(3, 5))
        expected = sorted([(2, 5), (3, 4), (3, 6), (4, 4), (4, 5), (4, 6)])
        assert result == expected

    def test_neighbors_corner(self):
        hm = _make_hex()
        result = hm.neighbors(0, 0)
        # (0,0) is even col: offsets (-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1)
        # valid: (1,0), (0,1) only — (-1,*) and (*,-1) are out of bounds
        assert len(result) <= 3
        assert (1, 0) in result
        assert (0, 1) in result


class TestHexToXy:
    """Test hex_to_xy coordinate conversion."""

    def test_origin(self):
        hm = _make_hex(edge=10.0)
        x, y = hm.hex_to_xy(0, 0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_col1(self):
        hm = _make_hex(edge=10.0)
        x, y = hm.hex_to_xy(0, 1)
        assert x == pytest.approx(15.0)
        assert y == pytest.approx(math.sqrt(3) * 5)


class TestXyToHexRoundtrip:
    """Test xy_to_hex round-trips with hex_to_xy for a 3x3 grid."""

    def test_roundtrip_3x3(self):
        hm = _make_hex(height=3, width=3, edge=10.0)
        for r in range(3):
            for c in range(3):
                x, y = hm.hex_to_xy(r, c)
                r2, c2 = hm.xy_to_hex(x, y)
                assert (r2, c2) == (r, c), f"Failed for ({r},{c})"


class TestHexDistance:
    """Test hex_distance computations."""

    def test_same_cell(self):
        hm = _make_hex()
        assert hm.hex_distance((3, 4), (3, 4)) == 0

    def test_adjacent(self):
        hm = _make_hex()
        # (3,4) and (2,4) are neighbors
        assert hm.hex_distance((3, 4), (2, 4)) == 1

    def test_two_steps(self):
        hm = _make_hex()
        # (3,4) to (1,4): two rows straight up
        assert hm.hex_distance((3, 4), (1, 4)) == 2


class TestHexPolygon:
    """Test hex_polygon returns correct vertices."""

    def test_six_vertices(self):
        hm = _make_hex(edge=10.0)
        poly = hm.hex_polygon(0, 0)
        assert len(poly) == 6

    def test_vertices_at_edge_distance(self):
        hm = _make_hex(edge=10.0)
        cx, cy = hm.hex_to_xy(2, 3)
        poly = hm.hex_polygon(2, 3)
        for vx, vy in poly:
            dist = math.sqrt((vx - cx) ** 2 + (vy - cy) ** 2)
            assert dist == pytest.approx(10.0)


class TestEffectiveEdge:
    """Test _effective_edge priority: _edge > cell_size > 1.0."""

    def test_edge_from_workspace(self):
        hm = _make_hex(edge=10.0)
        assert hm._effective_edge() == 10.0

    def test_edge_from_cell_size(self):
        hm = HexMap(
            format="plain", version=1, height=2, width=2, flag=0,
            max_val=1.0, min_val=0.0, hexzero=0.0,
            values=np.zeros(4, dtype=np.float32),
            cell_size=5.0,
        )
        assert hm._effective_edge() == 5.0

    def test_edge_fallback(self):
        hm = HexMap(
            format="patch_hexmap", version=1, height=2, width=2, flag=0,
            max_val=1.0, min_val=0.0, hexzero=0.0,
            values=np.zeros(4, dtype=np.float32),
        )
        assert hm._effective_edge() == 1.0
