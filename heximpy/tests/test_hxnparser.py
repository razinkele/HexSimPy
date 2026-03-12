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
