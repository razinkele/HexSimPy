# HXN Parser Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `hxnparser.py` module that reads/writes HexSim `.hxn`, `.grid`, `.hbf` files with hex geometry utilities and GeoDataFrame/shapefile/GeoTIFF/CSV export.

**Architecture:** Single file `heximpy/hxnparser.py` (~700 lines) with three dataclasses (`HexMap`, `GridMeta`, `Barrier`) plus a `Workspace` loader. TDD with `tests/test_hxnparser.py`. Consolidates scattered implementations from `io.py`, `grid.py`, `geometry.py`, `export.py`, `salmon_ibm/hexsim.py`, and `HexSim.py.txt`.

**Tech Stack:** Python 3.10+, numpy, struct, geopandas, shapely, rasterio, pytest.

**Spec:** `docs/superpowers/specs/2026-03-12-hxn-parser-design.md`

**Test data paths:**
- PATCH_HEXMAP examples: `heximpy/HabitatMap.hxn`, `heximpy/River.hxn`, `heximpy/StudyArea.hxn`
- Full workspace: `Columbia [small]/` (has `.grid`, `.hbf`, `.hxn` files)
- HexSimPLE workspace: `HexSimPLE/` (has `.grid`, `.hxn` files)

**Run tests:** `cd heximpy && python -m pytest tests/ -v`

---

## Chunk 1: Data Model + PATCH_HEXMAP Reader

### Task 1: Scaffold test file and dataclasses

**Files:**
- Create: `heximpy/tests/__init__.py`
- Create: `heximpy/tests/test_hxnparser.py`
- Create: `heximpy/hxnparser.py`

- [ ] **Step 1: Create empty test package**

```python
# heximpy/tests/__init__.py
# (empty)
```

- [ ] **Step 2: Write test for HexMap dataclass**

```python
# heximpy/tests/test_hxnparser.py
import numpy as np
import pytest
from heximpy.hxnparser import HexMap, GridMeta, Barrier


class TestHexMapDataclass:
    def test_create_patch_hexmap(self):
        hm = HexMap(
            format="patch_hexmap",
            version=8,
            width=10,
            height=5,
            values=np.zeros(50, dtype=np.float32),
            flag=0,
            max_val=1.0,
            min_val=0.0,
            hexzero=0.0,
        )
        assert hm.format == "patch_hexmap"
        assert hm.width == 10
        assert hm.height == 5
        assert len(hm.values) == 50

    def test_create_plain_hexmap(self):
        hm = HexMap(
            format="plain",
            version=3,
            width=20,
            height=15,
            values=np.ones(300, dtype=np.float32),
            cell_size=100.0,
            origin=(500000.0, 4100000.0),
            nodata=-9999,
            dtype_code=1,
        )
        assert hm.format == "plain"
        assert hm.cell_size == 100.0
        assert hm.origin == (500000.0, 4100000.0)

    def test_n_hexagons_wide(self):
        hm = HexMap(
            format="patch_hexmap", version=8, width=10, height=5,
            values=np.zeros(50, dtype=np.float32), flag=0,
        )
        assert hm.n_hexagons == 50  # wide: ncols * nrows

    def test_n_hexagons_narrow_even_rows(self):
        # nrows=4 (even): n_wide=2, n_narrow=2
        # hexagons = 10*2 + 9*2 = 38
        hm = HexMap(
            format="patch_hexmap", version=8, width=10, height=4,
            values=np.zeros(38, dtype=np.float32), flag=1,
        )
        assert hm.n_hexagons == 38

    def test_n_hexagons_narrow_odd_rows(self):
        # nrows=5 (odd): n_wide=3, n_narrow=2
        # hexagons = 10*3 + 9*2 = 48
        hm = HexMap(
            format="patch_hexmap", version=8, width=10, height=5,
            values=np.zeros(48, dtype=np.float32), flag=1,
        )
        assert hm.n_hexagons == 48
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestHexMapDataclass -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'heximpy.hxnparser'`

- [ ] **Step 4: Implement dataclasses in hxnparser.py**

```python
# heximpy/hxnparser.py
"""HexSim workspace file parser.

Reads/writes HexSim .hxn (hexmap), .grid (workspace geometry),
and .hbf (barrier) files. Provides hex grid geometry utilities
and export to GeoDataFrame, shapefile, GeoTIFF, and CSV.

Binary format specs derived from C source code (hxn2csv.c,
get_hexmap_means.c) and reverse-engineering of HexSim workspaces.
"""
from __future__ import annotations

import math
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── Format constants ─────────────────────────────────────────────────────────

_HXN_MAGIC = b"PATCH_HEXMAP"       # 12 bytes
_HXN_HEADER_SIZE = 37              # bytes before float32 data starts
_GRID_MAGIC = b"PATCH_GRID"        # 10 bytes
_HISTORY_MARKER = b"HISTORY"


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class HexMap:
    """Parsed .hxn file — supports both PATCH_HEXMAP and plain formats."""

    format: str                          # "patch_hexmap" or "plain"
    version: int
    width: int                           # ncols
    height: int                          # nrows
    values: np.ndarray                   # flat float32/int32 array
    flag: int = 0                        # 0=wide, 1=narrow (PATCH_HEXMAP)
    max_val: float = 0.0                 # PATCH_HEXMAP only
    min_val: float = 0.0                 # PATCH_HEXMAP only
    hexzero: float = 0.0                 # PATCH_HEXMAP only
    cell_size: float = 0.0              # plain format only
    origin: tuple = (0.0, 0.0)          # plain format only
    nodata: int = 0                      # plain format only
    dtype_code: int = 1                  # 1=float32, 2=int32 (plain only)
    _edge: float = field(default=0.0, repr=False)  # set by Workspace

    @property
    def n_hexagons(self) -> int:
        """Total hexagon count respecting wide/narrow flag.

        Wide  (flag=0): ncols * nrows
        Narrow (flag=1): ncols * n_wide + (ncols - 1) * n_narrow
        """
        if self.flag == 0:
            return self.width * self.height
        # Narrow grid
        if self.height % 2 == 0:
            n_wide = self.height // 2
            n_narrow = n_wide
        else:
            n_wide = (self.height + 1) // 2
            n_narrow = n_wide - 1
        return self.width * n_wide + (self.width - 1) * n_narrow


@dataclass
class GridMeta:
    """Parsed .grid file — workspace georeferencing."""

    ncols: int
    nrows: int
    x_extent: float
    y_extent: float
    row_spacing: float
    edge: float  # hex edge length in meters


@dataclass
class Barrier:
    """Single barrier edge from .hbf file."""

    hex_id: int
    edge: int
    classification: int
    class_name: str = ""
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestHexMapDataclass -v`
Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add heximpy/hxnparser.py heximpy/tests/__init__.py heximpy/tests/test_hxnparser.py
git commit -m "feat(hxnparser): add HexMap, GridMeta, Barrier dataclasses with hex count formula"
```

---

### Task 2: PATCH_HEXMAP reader

**Files:**
- Modify: `heximpy/hxnparser.py`
- Modify: `heximpy/tests/test_hxnparser.py`

**Reference:** `hxn2csv.c` lines 43-50 define the binary layout. Header is 37 bytes: 12B magic + 4B version + 4B nrows + 4B ncols + 1B flag + 4B max_val + 4B min_val + 4B hexzero.

- [ ] **Step 1: Write test for PATCH_HEXMAP reading with real file**

Append to `heximpy/tests/test_hxnparser.py`:

```python
import os

# Path to test data (relative to repo root)
_HEXIMPY_DIR = Path(__file__).resolve().parent.parent
_SALMON_DIR = _HEXIMPY_DIR.parent

# Real .hxn files in heximpy/
_HABITAT_HXN = _HEXIMPY_DIR / "HabitatMap.hxn"
_RIVER_HXN = _HEXIMPY_DIR / "River.hxn"

# Columbia [small] workspace
_COLUMBIA_DIR = _SALMON_DIR / "Columbia [small]"
_COLUMBIA_GRID = _COLUMBIA_DIR / "Columbia Fish Model [small].grid"


class TestPatchHexmapReader:
    @pytest.mark.skipif(not _HABITAT_HXN.exists(), reason="test data not available")
    def test_read_habitat_hxn(self):
        hm = HexMap.from_file(_HABITAT_HXN)
        assert hm.format == "patch_hexmap"
        assert hm.version == 8
        assert hm.width > 0
        assert hm.height > 0
        assert len(hm.values) > 0
        # max_val, min_val should be parsed (not be part of values)
        assert hm.max_val != 0.0 or hm.min_val != 0.0 or hm.hexzero != 0.0

    @pytest.mark.skipif(not _HABITAT_HXN.exists(), reason="test data not available")
    def test_header_size_37_not_25(self):
        """Verify data starts at byte 37, not byte 25 (existing bug)."""
        hm = HexMap.from_file(_HABITAT_HXN)
        # First data value should be a reasonable hex score (0.0 or small float),
        # not max_val (typically 1.0 for habitat maps)
        # If header was only 25 bytes, values[0] would be max_val
        with open(_HABITAT_HXN, "rb") as f:
            f.seek(25)
            max_val_bytes = struct.unpack("<f", f.read(4))[0]
        # values[0] should NOT equal the max_val that sits at offset 25
        # (unless by coincidence, but this is a sanity check)
        assert hm.max_val == max_val_bytes  # we correctly parsed max_val
        assert hm.values[0] != max_val_bytes or hm.values[0] == 0.0

    @pytest.mark.skipif(
        not (_COLUMBIA_GRID.exists() and _HABITAT_HXN.exists()),
        reason="test data not available",
    )
    def test_rows_cols_order_matches_grid(self):
        """Verify rows at offset 16, cols at offset 20 (not swapped).

        Cross-checks .hxn dimensions against .grid file dimensions.
        This catches the rows/cols swap bug in the old parser.
        """
        hm = HexMap.from_file(_HABITAT_HXN)
        gm = GridMeta.from_file(_COLUMBIA_GRID)
        assert hm.width == gm.ncols, (
            f"width/ncols mismatch: .hxn={hm.width}, .grid={gm.ncols}"
        )
        assert hm.height == gm.nrows, (
            f"height/nrows mismatch: .hxn={hm.height}, .grid={gm.nrows}"
        )

    @pytest.mark.skipif(not _COLUMBIA_GRID.exists(), reason="test data not available")
    def test_rejects_grid_file(self):
        """Passing a .grid file to HexMap.from_file should raise ValueError."""
        with pytest.raises(ValueError, match="grid file"):
            HexMap.from_file(_COLUMBIA_GRID)

    @pytest.mark.skipif(not _HABITAT_HXN.exists(), reason="test data not available")
    def test_hex_count_matches_values(self):
        """Verify n_hexagons matches actual data length."""
        hm = HexMap.from_file(_HABITAT_HXN)
        assert hm.n_hexagons == len(hm.values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestPatchHexmapReader -v`
Expected: FAIL — `AttributeError: type object 'HexMap' has no attribute 'from_file'`

- [ ] **Step 3: Implement HexMap.from_file for PATCH_HEXMAP**

Add to `HexMap` class in `heximpy/hxnparser.py`:

```python
    @classmethod
    def from_file(cls, path: str | Path) -> HexMap:
        """Read a .hxn file, auto-detecting format.

        PATCH_HEXMAP: first 12 bytes == b"PATCH_HEXMAP"
        Plain: 44-byte header starting with int32 version.
        """
        path = Path(path)
        with open(path, "rb") as f:
            peek = f.read(12)
            f.seek(0)
            raw = f.read()

        if peek == _HXN_MAGIC:
            return cls._read_patch_hexmap(raw)
        elif peek[:10] == _GRID_MAGIC:
            raise ValueError(
                f"This is a .grid file, not .hxn: {path}. "
                "Use GridMeta.from_file() instead."
            )
        else:
            return cls._read_plain(raw)

    @classmethod
    def _read_patch_hexmap(cls, raw: bytes) -> HexMap:
        """Parse PATCH_HEXMAP binary format (37-byte header).

        Layout (from hxn2csv.c):
            0-11:  char[12]  "PATCH_HEXMAP"
            12-15: uint32    version
            16-19: uint32    nrows (C variable 'rows')
            20-23: uint32    ncols (C variable 'cols')
            24:    uint8     flag (0=wide, 1=narrow)
            25-28: float32   max_val
            29-32: float32   min_val
            33-36: float32   hexzero
            37+:   float32[] data until HISTORY marker or EOF
        """
        version = struct.unpack_from("<I", raw, 12)[0]
        nrows = struct.unpack_from("<I", raw, 16)[0]
        ncols = struct.unpack_from("<I", raw, 20)[0]
        flag = raw[24]
        max_val = struct.unpack_from("<f", raw, 25)[0]
        min_val = struct.unpack_from("<f", raw, 29)[0]
        hexzero = struct.unpack_from("<f", raw, 33)[0]

        # Data runs from byte 37 to HISTORY marker (or EOF)
        hist_pos = raw.find(_HISTORY_MARKER, _HXN_HEADER_SIZE)
        if hist_pos < 0:
            hist_pos = len(raw)
        data_bytes = raw[_HXN_HEADER_SIZE:hist_pos]
        n_floats = len(data_bytes) // 4
        values = np.frombuffer(
            data_bytes[:n_floats * 4], dtype="<f4"
        ).copy()

        return cls(
            format="patch_hexmap",
            version=version,
            width=ncols,
            height=nrows,
            values=values,
            flag=flag,
            max_val=max_val,
            min_val=min_val,
            hexzero=hexzero,
        )

    @classmethod
    def _read_plain(cls, raw: bytes) -> HexMap:
        """Parse plain .hxn format (44-byte header).

        Layout:
            0-3:   int32    version
            4-7:   int32    width (ncols)
            8-11:  int32    height (nrows)
            12-19: float64  cell_size
            20-27: float64  origin_x
            28-35: float64  origin_y
            36-39: int32    dtype_code (1=float32, 2=int32)
            40-43: int32    nodata
            44+:   values   width*height elements
        """
        version = struct.unpack_from("<i", raw, 0)[0]
        width = struct.unpack_from("<i", raw, 4)[0]
        height = struct.unpack_from("<i", raw, 8)[0]
        cell_size = struct.unpack_from("<d", raw, 12)[0]
        origin_x = struct.unpack_from("<d", raw, 20)[0]
        origin_y = struct.unpack_from("<d", raw, 28)[0]
        dtype_code = struct.unpack_from("<i", raw, 36)[0]
        nodata = struct.unpack_from("<i", raw, 40)[0]

        dtype = np.dtype("<f4") if dtype_code == 1 else np.dtype("<i4")
        count = width * height
        values = np.frombuffer(raw, dtype=dtype, offset=44, count=count).copy()

        return cls(
            format="plain",
            version=version,
            width=width,
            height=height,
            values=values,
            cell_size=cell_size,
            origin=(origin_x, origin_y),
            nodata=nodata,
            dtype_code=dtype_code,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add heximpy/hxnparser.py heximpy/tests/test_hxnparser.py
git commit -m "feat(hxnparser): implement HexMap.from_file for PATCH_HEXMAP and plain formats"
```

---

### Task 3: GridMeta.from_file and read_barriers

**Files:**
- Modify: `heximpy/hxnparser.py`
- Modify: `heximpy/tests/test_hxnparser.py`

- [ ] **Step 1: Write tests for GridMeta and barriers**

Append to `heximpy/tests/test_hxnparser.py`:

```python
from heximpy.hxnparser import read_barriers

# Barrier test data
_BARRIER_HBF = (
    _COLUMBIA_DIR / "Spatial Data" / "barriers"
    / "Fish Ladder Available" / "Fish Ladder Available.1.hbf"
)


class TestGridMetaReader:
    @pytest.mark.skipif(not _COLUMBIA_GRID.exists(), reason="test data not available")
    def test_read_grid(self):
        gm = GridMeta.from_file(_COLUMBIA_GRID)
        assert gm.ncols > 0
        assert gm.nrows > 0
        assert gm.edge > 0
        assert gm.row_spacing > 0
        # edge = row_spacing / sqrt(3)
        assert abs(gm.edge - gm.row_spacing / np.sqrt(3.0)) < 0.001

    @pytest.mark.skipif(not _COLUMBIA_GRID.exists(), reason="test data not available")
    def test_grid_rejects_hxn(self):
        """Passing a .hxn file to GridMeta.from_file should raise."""
        with pytest.raises(ValueError, match="not a .grid file"):
            GridMeta.from_file(_HABITAT_HXN)


class TestBarrierReader:
    @pytest.mark.skipif(not _BARRIER_HBF.exists(), reason="test data not available")
    def test_read_barriers(self):
        barriers = read_barriers(_BARRIER_HBF)
        assert len(barriers) > 0
        assert all(isinstance(b, Barrier) for b in barriers)
        # First barrier should have valid fields
        b = barriers[0]
        assert b.hex_id > 0
        assert 0 <= b.edge <= 5
        assert b.classification >= 1

    @pytest.mark.skipif(not _BARRIER_HBF.exists(), reason="test data not available")
    def test_barrier_class_names(self):
        barriers = read_barriers(_BARRIER_HBF)
        # At least some barriers should have class names
        names = {b.class_name for b in barriers if b.class_name}
        assert len(names) > 0

    def test_read_barriers_from_string(self, tmp_path):
        """Test with synthetic .hbf content."""
        hbf = tmp_path / "test.hbf"
        hbf.write_text(
            'C 1 0.5 0.0 "Dam"\n'
            'C 2 0.3 0.0 "Waterfall"\n'
            "E 100 3 1\n"
            "E 200 0 2\n"
        )
        barriers = read_barriers(hbf)
        assert len(barriers) == 2
        assert barriers[0].hex_id == 100
        assert barriers[0].edge == 3
        assert barriers[0].classification == 1
        assert barriers[0].class_name == "Dam"
        assert barriers[1].class_name == "Waterfall"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestGridMetaReader tests/test_hxnparser.py::TestBarrierReader -v`
Expected: FAIL — `AttributeError` / `ImportError`

- [ ] **Step 3: Implement GridMeta.from_file**

Add to `GridMeta` class in `heximpy/hxnparser.py`:

```python
    @classmethod
    def from_file(cls, path: str | Path) -> GridMeta:
        """Parse a HexSim .grid file.

        Layout (PATCH_GRID):
            0-9:   char[10]   "PATCH_GRID"
            10-13: uint32     version
            14-17: uint32     n_hexes
            18-21: uint32     ncols
            22-25: uint32     nrows
            26:    uint8      flag
            27-66: 5×float64  georef [x_extent, 0.0, 0.0, y_extent, row_spacing]
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = f.read()

        if not data.startswith(_GRID_MAGIC):
            raise ValueError(f"Not a .grid file (missing PATCH_GRID magic): {path}")

        ncols = struct.unpack_from("<I", data, 18)[0]
        nrows = struct.unpack_from("<I", data, 22)[0]
        georef = struct.unpack_from("<5d", data, 27)
        x_extent = georef[0]
        y_extent = georef[3]
        row_spacing = georef[4]
        edge = row_spacing / np.sqrt(3.0)

        return cls(
            ncols=ncols,
            nrows=nrows,
            x_extent=x_extent,
            y_extent=y_extent,
            row_spacing=row_spacing,
            edge=edge,
        )
```

- [ ] **Step 4: Implement read_barriers**

Add as a module-level function in `heximpy/hxnparser.py`:

```python
def read_barriers(path: str | Path) -> list[Barrier]:
    """Parse a HexSim .hbf text file → list of Barrier objects.

    Lines starting with C define classifications:
        C <id> <p1> <p2> "<name>"
    Lines starting with E define barrier edges:
        E <hex_id> <edge> <class_id>
    """
    classifications: dict[int, str] = {}
    barriers: list[Barrier] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == "C" and len(parts) >= 5:
                cid = int(parts[1])
                name = line.split('"')[1] if '"' in line else f"class_{cid}"
                classifications[cid] = name
            elif parts[0] == "E" and len(parts) >= 4:
                class_id = int(parts[3])
                barriers.append(Barrier(
                    hex_id=int(parts[1]),
                    edge=int(parts[2]),
                    classification=class_id,
                    class_name=classifications.get(class_id, ""),
                ))

    return barriers
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py -v`
Expected: ALL PASSED

- [ ] **Step 6: Commit**

```bash
git add heximpy/hxnparser.py heximpy/tests/test_hxnparser.py
git commit -m "feat(hxnparser): add GridMeta.from_file and read_barriers"
```

---

## Chunk 2: Writer + Hex Geometry

### Task 4: HexMap.to_file (write support)

**Files:**
- Modify: `heximpy/hxnparser.py`
- Modify: `heximpy/tests/test_hxnparser.py`

- [ ] **Step 1: Write round-trip tests**

Append to `heximpy/tests/test_hxnparser.py`:

```python
class TestHexMapWriter:
    def test_roundtrip_patch_hexmap(self, tmp_path):
        """Write PATCH_HEXMAP, read back, compare."""
        original = HexMap(
            format="patch_hexmap",
            version=8,
            width=10,
            height=5,
            values=np.arange(50, dtype=np.float32),
            flag=0,
            max_val=49.0,
            min_val=0.0,
            hexzero=0.0,
        )
        out = tmp_path / "test.hxn"
        original.to_file(out)
        loaded = HexMap.from_file(out)

        assert loaded.format == "patch_hexmap"
        assert loaded.version == 8
        assert loaded.width == 10
        assert loaded.height == 5
        assert loaded.flag == 0
        assert loaded.max_val == pytest.approx(49.0)
        assert loaded.min_val == pytest.approx(0.0)
        np.testing.assert_array_almost_equal(loaded.values, original.values)

    def test_roundtrip_plain(self, tmp_path):
        """Write plain format, read back, compare."""
        original = HexMap(
            format="plain",
            version=3,
            width=5,
            height=4,
            values=np.arange(20, dtype=np.float32),
            cell_size=100.0,
            origin=(500000.0, 4100000.0),
            nodata=-9999,
            dtype_code=1,
        )
        out = tmp_path / "test.hxn"
        original.to_file(out)
        loaded = HexMap.from_file(out)

        assert loaded.format == "plain"
        assert loaded.width == 5
        assert loaded.height == 4
        assert loaded.cell_size == pytest.approx(100.0)
        assert loaded.origin == pytest.approx((500000.0, 4100000.0))
        assert loaded.nodata == -9999
        np.testing.assert_array_almost_equal(loaded.values, original.values)

    @pytest.mark.skipif(not _HABITAT_HXN.exists(), reason="test data not available")
    def test_roundtrip_real_file(self, tmp_path):
        """Read real file, write, read back, compare values."""
        original = HexMap.from_file(_HABITAT_HXN)
        out = tmp_path / "roundtrip.hxn"
        original.to_file(out)
        loaded = HexMap.from_file(out)
        np.testing.assert_array_almost_equal(loaded.values, original.values)
        assert loaded.width == original.width
        assert loaded.height == original.height

    def test_force_format(self, tmp_path):
        """Write with forced format override."""
        hm = HexMap(
            format="patch_hexmap",
            version=8,
            width=3,
            height=2,
            values=np.ones(6, dtype=np.float32),
            flag=0,
        )
        out = tmp_path / "forced.hxn"
        hm.to_file(out, format="plain")
        loaded = HexMap.from_file(out)
        assert loaded.format == "plain"
        np.testing.assert_array_almost_equal(loaded.values, hm.values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestHexMapWriter -v`
Expected: FAIL — `AttributeError: 'HexMap' object has no attribute 'to_file'`

- [ ] **Step 3: Implement to_file**

Add to `HexMap` class in `heximpy/hxnparser.py`:

```python
    def to_file(self, path: str | Path, *, format: str | None = None) -> None:
        """Write this HexMap to a .hxn file.

        Parameters
        ----------
        path : output file path
        format : force "patch_hexmap" or "plain" (default: use self.format)
        """
        fmt = format or self.format
        path = Path(path)

        if fmt == "patch_hexmap":
            self._write_patch_hexmap(path)
        else:
            self._write_plain(path)

    def _write_patch_hexmap(self, path: Path) -> None:
        with open(path, "wb") as f:
            f.write(_HXN_MAGIC)
            f.write(struct.pack("<I", self.version))
            f.write(struct.pack("<I", self.height))   # nrows at offset 16
            f.write(struct.pack("<I", self.width))    # ncols at offset 20
            f.write(struct.pack("B", self.flag))
            f.write(struct.pack("<f", self.max_val))
            f.write(struct.pack("<f", self.min_val))
            f.write(struct.pack("<f", self.hexzero))
            f.write(self.values.astype("<f4").tobytes())

    def _write_plain(self, path: Path) -> None:
        with open(path, "wb") as f:
            f.write(struct.pack("<i", self.version))
            f.write(struct.pack("<i", self.width))
            f.write(struct.pack("<i", self.height))
            f.write(struct.pack("<d", self.cell_size))
            f.write(struct.pack("<d", self.origin[0]))
            f.write(struct.pack("<d", self.origin[1]))
            f.write(struct.pack("<i", self.dtype_code))
            f.write(struct.pack("<i", self.nodata))
            dtype = "<f4" if self.dtype_code == 1 else "<i4"
            f.write(self.values.astype(dtype).tobytes())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestHexMapWriter -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add heximpy/hxnparser.py heximpy/tests/test_hxnparser.py
git commit -m "feat(hxnparser): add HexMap.to_file with round-trip support"
```

---

### Task 5: Hex geometry methods

**Files:**
- Modify: `heximpy/hxnparser.py`
- Modify: `heximpy/tests/test_hxnparser.py`

**Reference:** Even-q column-offset, flat-top hexagons. Neighbor offsets from `salmon_ibm/hexsim.py` lines 220-222. Coordinate math from `geometry.py` and `HexSim.py.txt`.

- [ ] **Step 1: Write geometry tests**

Append to `heximpy/tests/test_hxnparser.py`:

```python
class TestHexGeometry:
    def _make_hexmap(self, width=10, height=8):
        return HexMap(
            format="patch_hexmap", version=8,
            width=width, height=height,
            values=np.zeros(width * height, dtype=np.float32),
            flag=0, _edge=10.0,
        )

    def test_neighbors_even_col(self):
        hm = self._make_hexmap()
        nbrs = hm.neighbors(3, 4)  # col 4 = even
        expected = {(2, 4), (4, 4), (3, 3), (3, 5), (2, 3), (2, 5)}
        assert set(nbrs) == expected

    def test_neighbors_odd_col(self):
        hm = self._make_hexmap()
        nbrs = hm.neighbors(3, 5)  # col 5 = odd
        expected = {(2, 5), (4, 5), (3, 4), (3, 6), (4, 4), (4, 6)}
        assert set(nbrs) == expected

    def test_neighbors_corner(self):
        hm = self._make_hexmap()
        nbrs = hm.neighbors(0, 0)
        # Top-left corner, even col: only (1,0) and (0,1) are in bounds
        assert len(nbrs) <= 3

    def test_hex_to_xy_origin(self):
        hm = self._make_hexmap()
        x, y = hm.hex_to_xy(0, 0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_hex_to_xy_col1(self):
        hm = self._make_hexmap()
        x, y = hm.hex_to_xy(0, 1)
        # col_spacing = 1.5 * edge = 15.0
        assert x == pytest.approx(15.0)
        # odd col shifted down by row_spacing/2 = sqrt(3)*10/2
        assert y == pytest.approx(10.0 * math.sqrt(3) / 2.0)

    def test_xy_to_hex_roundtrip(self):
        hm = self._make_hexmap()
        for r in range(3):
            for c in range(3):
                x, y = hm.hex_to_xy(r, c)
                rr, cc = hm.xy_to_hex(x, y)
                assert (rr, cc) == (r, c), f"Failed roundtrip for ({r},{c})"

    def test_hex_distance_same_cell(self):
        hm = self._make_hexmap()
        assert hm.hex_distance((3, 3), (3, 3)) == 0

    def test_hex_distance_adjacent(self):
        hm = self._make_hexmap()
        assert hm.hex_distance((3, 4), (2, 4)) == 1  # direct neighbor

    def test_hex_distance_two_steps(self):
        hm = self._make_hexmap()
        # (0,0) to (0,2): two column steps
        assert hm.hex_distance((0, 0), (0, 2)) == 2

    def test_hex_polygon_six_vertices(self):
        hm = self._make_hexmap()
        poly = hm.hex_polygon(0, 0)
        assert len(poly) == 6
        # All vertices should be at distance ~edge from center
        cx, cy = 0.0, 0.0
        for px, py in poly:
            dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            assert dist == pytest.approx(10.0, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestHexGeometry -v`
Expected: FAIL — `AttributeError: 'HexMap' object has no attribute 'neighbors'`

- [ ] **Step 3: Implement geometry methods**

Add to `HexMap` class in `heximpy/hxnparser.py`:

```python
    def _effective_edge(self) -> float:
        """Resolve hex edge length: _edge (from Workspace) > cell_size > 1.0."""
        if self._edge > 0:
            return self._edge
        if self.cell_size > 0:
            return self.cell_size
        return 1.0

    def neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Neighbors of (row, col) in even-q column-offset grid.

        Even columns: [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1)]
        Odd columns:  [(-1,0),(1,0),(0,-1),(0,1),(1,-1),(1,1)]
        """
        if col % 2 == 0:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1)]
        else:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (1, 1)]

        result = []
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                result.append((nr, nc))
        return result

    def hex_to_xy(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid (row, col) → (x, y) in spatial units.

        Flat-top hex: x = 1.5 * edge * col
                      y = √3 * edge * (row + 0.5 * (col % 2))
        """
        edge = self._effective_edge()
        x = 1.5 * edge * col
        y = math.sqrt(3.0) * edge * (row + 0.5 * (col % 2))
        return (x, y)

    def xy_to_hex(self, x: float, y: float) -> tuple[int, int]:
        """Convert (x, y) → nearest grid (row, col)."""
        edge = self._effective_edge()
        col = round(x / (1.5 * edge))
        row = round(y / (math.sqrt(3.0) * edge) - 0.5 * (col % 2))
        return (int(row), int(col))

    @staticmethod
    def _offset_to_cube(row: int, col: int) -> tuple[int, int, int]:
        """Convert even-q offset → cube coordinates."""
        qx = col
        qz = row - (col + (col & 1)) // 2
        qy = -qx - qz
        return (qx, qy, qz)

    def hex_distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        """Minimum hex steps between two cells (cube coordinate distance)."""
        ax, ay, az = self._offset_to_cube(a[0], a[1])
        bx, by, bz = self._offset_to_cube(b[0], b[1])
        return (abs(ax - bx) + abs(ay - by) + abs(az - bz)) // 2

    def hex_polygon(self, row: int, col: int) -> list[tuple[float, float]]:
        """Six vertices of the hex at (row, col)."""
        cx, cy = self.hex_to_xy(row, col)
        edge = self._effective_edge()
        pts = []
        for i in range(6):
            angle = math.radians(60 * i)
            pts.append((cx + edge * math.cos(angle),
                         cy + edge * math.sin(angle)))
        return pts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestHexGeometry -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add heximpy/hxnparser.py heximpy/tests/test_hxnparser.py
git commit -m "feat(hxnparser): add hex geometry — neighbors, coordinates, distance, polygon"
```

---

## Chunk 3: Export + Workspace

### Task 6: CSV and GeoDataFrame export

**Files:**
- Modify: `heximpy/hxnparser.py`
- Modify: `heximpy/tests/test_hxnparser.py`

- [ ] **Step 1: Write export tests**

Append to `heximpy/tests/test_hxnparser.py`:

```python
import geopandas as gpd


class TestExport:
    def _make_hexmap(self):
        vals = np.array([0.0, 1.0, 0.0, 2.5, 3.0, 0.0], dtype=np.float32)
        return HexMap(
            format="patch_hexmap", version=8,
            width=3, height=2,
            values=vals, flag=0, _edge=10.0,
        )

    def test_to_csv_skip_zeros(self, tmp_path):
        hm = self._make_hexmap()
        out = tmp_path / "out.csv"
        hm.to_csv(out, skip_zeros=True)
        lines = out.read_text().strip().split("\n")
        assert lines[0] == "Hex ID,Score"
        # 3 nonzero values: indices 1, 3, 4 → hex IDs 2, 4, 5
        assert len(lines) == 4  # header + 3 data

    def test_to_csv_include_zeros(self, tmp_path):
        hm = self._make_hexmap()
        out = tmp_path / "out.csv"
        hm.to_csv(out, skip_zeros=False)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 7  # header + 6 data

    def test_to_geodataframe(self):
        hm = self._make_hexmap()
        gdf = hm.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        # Default skips zeros: 3 nonzero cells
        assert len(gdf) == 3
        assert "hex_id" in gdf.columns
        assert "row" in gdf.columns
        assert "col" in gdf.columns
        assert "value" in gdf.columns
        assert gdf.geometry.name == "geometry"
        # Each geometry should be a polygon with 6 exterior coords + closing
        for geom in gdf.geometry:
            assert len(geom.exterior.coords) == 7  # 6 vertices + close

    def test_to_geodataframe_include_empty(self):
        hm = self._make_hexmap()
        gdf = hm.to_geodataframe(include_empty=True)
        assert len(gdf) == 6  # all cells

    def test_to_shapefile(self, tmp_path):
        hm = self._make_hexmap()
        out = tmp_path / "out.shp"
        hm.to_shapefile(out)
        gdf = gpd.read_file(out)
        assert len(gdf) == 3

    def test_to_geodataframe_with_crs(self):
        hm = self._make_hexmap()
        gdf = hm.to_geodataframe(crs="EPSG:32610")
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 32610
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestExport -v`
Expected: FAIL — `AttributeError: 'HexMap' object has no attribute 'to_csv'`

- [ ] **Step 3: Implement export methods**

Add imports at top of `heximpy/hxnparser.py`:

```python
import geopandas as gpd
from shapely.geometry import Polygon
```

Add to `HexMap` class:

```python
    def to_csv(self, path: str | Path, *, skip_zeros: bool = True) -> None:
        """Export to CSV in hxn2csv.c format: 'Hex ID,Score'.

        Parameters
        ----------
        skip_zeros : if True, omit cells with value == 0.0 (default).
        """
        path = Path(path)
        with open(path, "w") as f:
            f.write("Hex ID,Score\n")
            for i, val in enumerate(self.values):
                if skip_zeros and val == 0.0:
                    continue
                f.write(f"{i + 1},{val:f}\n")

    def to_geodataframe(
        self,
        *,
        edge: float | None = None,
        include_empty: bool = False,
        crs: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Convert to GeoDataFrame with hex polygon geometry.

        Parameters
        ----------
        edge : hex edge length (uses _edge or cell_size if not given)
        include_empty : if True, include zero/nodata cells
        crs : coordinate reference system string (e.g. "EPSG:32610")
        """
        if edge is not None:
            saved = self._edge
            self._edge = edge
        elif self._effective_edge() == 1.0 and self.format == "patch_hexmap":
            warnings.warn(
                "No edge length set for PATCH_HEXMAP file. "
                "Pass edge= parameter or load via Workspace for correct geometry."
            )

        rows_list = []
        for i, val in enumerate(self.values):
            if not include_empty and val == 0.0:
                continue
            r = i // self.width
            c = i % self.width
            poly = Polygon(self.hex_polygon(r, c))
            rows_list.append({
                "hex_id": i + 1,
                "row": r,
                "col": c,
                "value": float(val),
                "geometry": poly,
            })

        if edge is not None:
            self._edge = saved

        gdf = gpd.GeoDataFrame(rows_list, geometry="geometry")
        if crs:
            gdf = gdf.set_crs(crs)
        return gdf

    def to_shapefile(
        self,
        path: str | Path,
        *,
        edge: float | None = None,
        include_empty: bool = False,
        crs: str | None = None,
    ) -> None:
        """Export to shapefile (delegates to to_geodataframe)."""
        gdf = self.to_geodataframe(
            edge=edge, include_empty=include_empty, crs=crs,
        )
        gdf.to_file(str(path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestExport -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add heximpy/hxnparser.py heximpy/tests/test_hxnparser.py
git commit -m "feat(hxnparser): add CSV, GeoDataFrame, and shapefile export"
```

---

### Task 7: GeoTIFF export

**Files:**
- Modify: `heximpy/hxnparser.py`
- Modify: `heximpy/tests/test_hxnparser.py`

- [ ] **Step 1: Write GeoTIFF test**

Append to `heximpy/tests/test_hxnparser.py`:

```python
import rasterio


class TestGeoTIFFExport:
    def test_to_geotiff(self, tmp_path):
        hm = HexMap(
            format="plain", version=3,
            width=5, height=4,
            values=np.arange(20, dtype=np.float32),
            cell_size=100.0,
            origin=(500000.0, 4100000.0),
            nodata=-9999,
            dtype_code=1,
        )
        out = tmp_path / "out.tif"
        hm.to_geotiff(out)
        with rasterio.open(out) as src:
            assert src.width == 5
            assert src.height == 4
            data = src.read(1)
            assert data.shape == (4, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestGeoTIFFExport -v`
Expected: FAIL — `AttributeError: 'HexMap' object has no attribute 'to_geotiff'`

- [ ] **Step 3: Implement to_geotiff**

Add import at top of `heximpy/hxnparser.py`:

```python
import rasterio
from rasterio.transform import from_bounds
```

Add to `HexMap` class:

```python
    def to_geotiff(self, path: str | Path, *, crs: str | None = None) -> None:
        """Export as GeoTIFF (rectangular raster approximation).

        Uses origin and cell_size for georeferencing when available.
        """
        grid = self.values.reshape((self.height, self.width))
        edge = self._effective_edge()
        col_spacing = 1.5 * edge
        row_spacing = math.sqrt(3.0) * edge

        ox, oy = self.origin
        transform = rasterio.transform.from_bounds(
            ox, oy,
            ox + self.width * col_spacing,
            oy + self.height * row_spacing,
            self.width, self.height,
        )

        with rasterio.open(
            str(path), "w",
            driver="GTiff",
            height=self.height,
            width=self.width,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(grid.astype(np.float32), 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestGeoTIFFExport -v`
Expected: ALL PASSED

- [ ] **Step 5: Commit**

```bash
git add heximpy/hxnparser.py heximpy/tests/test_hxnparser.py
git commit -m "feat(hxnparser): add GeoTIFF export via rasterio"
```

---

### Task 8: Workspace loader

**Files:**
- Modify: `heximpy/hxnparser.py`
- Modify: `heximpy/tests/test_hxnparser.py`

- [ ] **Step 1: Write Workspace tests**

Append to `heximpy/tests/test_hxnparser.py`:

```python
from heximpy.hxnparser import Workspace


class TestWorkspace:
    @pytest.mark.skipif(not _COLUMBIA_DIR.exists(), reason="test data not available")
    def test_load_columbia_workspace(self):
        ws = Workspace.from_dir(_COLUMBIA_DIR)
        assert ws.grid.ncols > 0
        assert ws.grid.nrows > 0
        assert ws.grid.edge > 0
        assert len(ws.hexmaps) > 0
        assert len(ws.barriers) > 0

    @pytest.mark.skipif(not _COLUMBIA_DIR.exists(), reason="test data not available")
    def test_layer_names(self):
        ws = Workspace.from_dir(_COLUMBIA_DIR)
        names = ws.layer_names
        assert isinstance(names, list)
        assert len(names) > 0

    @pytest.mark.skipif(not _COLUMBIA_DIR.exists(), reason="test data not available")
    def test_edge_auto_populated(self):
        ws = Workspace.from_dir(_COLUMBIA_DIR)
        for name, hm in ws.hexmaps.items():
            assert hm._edge == pytest.approx(ws.grid.edge), (
                f"Layer {name} edge not set from grid"
            )

    @pytest.mark.skipif(not _COLUMBIA_DIR.exists(), reason="test data not available")
    def test_hexmap_dimensions_match_grid(self):
        """Verify .hxn nrows/ncols are consistent with .grid."""
        ws = Workspace.from_dir(_COLUMBIA_DIR)
        # At least one layer should have matching dimensions
        for name, hm in ws.hexmaps.items():
            if hm.format == "patch_hexmap":
                # Width and height should be reasonable (not swapped)
                assert hm.width > 0
                assert hm.height > 0
                break

    @pytest.mark.skipif(not _COLUMBIA_DIR.exists(), reason="test data not available")
    def test_workspace_export_layer(self):
        """Load workspace and export a layer to GeoDataFrame."""
        ws = Workspace.from_dir(_COLUMBIA_DIR)
        name = ws.layer_names[0]
        gdf = ws.hexmaps[name].to_geodataframe()
        assert len(gdf) > 0

    def test_workspace_missing_grid(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .grid file"):
            Workspace.from_dir(tmp_path)

    def test_workspace_missing_hexagons_dir(self, tmp_path):
        # Create a fake .grid file
        grid_path = tmp_path / "test.grid"
        grid_path.write_bytes(
            _GRID_MAGIC + b"\x00" * 57  # minimal fake grid
        )
        with pytest.raises(FileNotFoundError, match="Spatial Data"):
            Workspace.from_dir(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestWorkspace -v`
Expected: FAIL — `ImportError: cannot import name 'Workspace'`

- [ ] **Step 3: Implement Workspace**

Add to `heximpy/hxnparser.py`:

```python
@dataclass
class Workspace:
    """A loaded HexSim workspace directory."""

    grid: GridMeta
    hexmaps: dict[str, HexMap]
    barriers: list[Barrier]
    path: Path

    @property
    def layer_names(self) -> list[str]:
        return sorted(self.hexmaps.keys())

    @classmethod
    def from_dir(cls, path: str | Path) -> Workspace:
        """Load a HexSim workspace directory.

        Expected structure:
            workspace/
                *.grid
                *.hbf  (optional)
                Spatial Data/
                    Hexagons/
                        Layer Name/
                            Layer Name.1.hxn
        """
        ws = Path(path)

        # 1. Find and parse .grid
        grid_files = list(ws.glob("*.grid"))
        if not grid_files:
            raise FileNotFoundError(f"No .grid file found in {ws}")
        grid = GridMeta.from_file(grid_files[0])

        # 2. Find hexagon layers
        hex_dir = ws / "Spatial Data" / "Hexagons"
        if not hex_dir.exists():
            raise FileNotFoundError(
                f"Spatial Data/Hexagons/ not found in {ws}"
            )

        hexmaps: dict[str, HexMap] = {}
        for hxn_path in sorted(hex_dir.glob("*/*.hxn")):
            layer_name = hxn_path.parent.name
            hm = HexMap.from_file(hxn_path)
            hm._edge = grid.edge
            hexmaps[layer_name] = hm

        if not hexmaps:
            warnings.warn(f"No .hxn files found in {hex_dir}")

        # 3. Find and parse barriers
        barriers: list[Barrier] = []
        barrier_dir = ws / "Spatial Data" / "barriers"
        if barrier_dir.exists():
            for hbf_path in sorted(barrier_dir.glob("*/*.hbf")):
                barriers.extend(read_barriers(hbf_path))
        else:
            for hbf_path in sorted(ws.glob("*.hbf")):
                barriers.extend(read_barriers(hbf_path))

        return cls(
            grid=grid,
            hexmaps=hexmaps,
            barriers=barriers,
            path=ws,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py::TestWorkspace -v`
Expected: ALL PASSED

- [ ] **Step 5: Run full test suite**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py -v`
Expected: ALL PASSED

- [ ] **Step 6: Commit**

```bash
git add heximpy/hxnparser.py heximpy/tests/test_hxnparser.py
git commit -m "feat(hxnparser): add Workspace.from_dir with layer discovery and barrier loading"
```

---

## Chunk 4: Integration Verification

### Task 9: Cross-validation with C code and final cleanup

**Files:**
- Modify: `heximpy/tests/test_hxnparser.py`

- [ ] **Step 1: Write cross-validation integration test**

Append to `heximpy/tests/test_hxnparser.py`:

```python
class TestCrossValidation:
    """Cross-validate Python parser against C code behavior."""

    @pytest.mark.skipif(not _HABITAT_HXN.exists(), reason="test data not available")
    def test_patch_hexmap_header_byte_37(self):
        """Verify we read data starting at byte 37, not 25."""
        with open(_HABITAT_HXN, "rb") as f:
            raw = f.read()

        # Bytes 25-36 are max_val, min_val, hexzero (header, not data)
        max_val = struct.unpack_from("<f", raw, 25)[0]
        min_val = struct.unpack_from("<f", raw, 29)[0]
        hexzero = struct.unpack_from("<f", raw, 33)[0]

        # First actual data value at byte 37
        first_data = struct.unpack_from("<f", raw, 37)[0]

        hm = HexMap.from_file(_HABITAT_HXN)
        assert hm.max_val == pytest.approx(max_val)
        assert hm.min_val == pytest.approx(min_val)
        assert hm.hexzero == pytest.approx(hexzero)
        assert hm.values[0] == pytest.approx(first_data)

    @pytest.mark.skipif(not _HABITAT_HXN.exists(), reason="test data not available")
    def test_csv_matches_hxn2csv_format(self, tmp_path):
        """Verify CSV output matches hxn2csv.c format."""
        hm = HexMap.from_file(_HABITAT_HXN)
        out = tmp_path / "out.csv"
        hm.to_csv(out, skip_zeros=True)

        lines = out.read_text().strip().split("\n")
        assert lines[0] == "Hex ID,Score"
        # Each data line: "int,float"
        for line in lines[1:5]:  # check first few
            parts = line.split(",")
            assert len(parts) == 2
            int(parts[0])         # hex ID is integer
            float(parts[1])       # score is float

    @pytest.mark.skipif(not _RIVER_HXN.exists(), reason="test data not available")
    def test_multiple_files_consistent(self):
        """Read multiple .hxn files and verify consistent dimensions."""
        hm1 = HexMap.from_file(_HABITAT_HXN)
        hm2 = HexMap.from_file(_RIVER_HXN)
        # Same workspace should have same grid dimensions
        assert hm1.width == hm2.width
        assert hm1.height == hm2.height
```

- [ ] **Step 2: Run full test suite**

Run: `cd heximpy && python -m pytest tests/test_hxnparser.py -v`
Expected: ALL PASSED

- [ ] **Step 3: Commit**

```bash
git add heximpy/tests/test_hxnparser.py
git commit -m "test(hxnparser): add cross-validation and integration tests"
```

- [ ] **Step 4: Add `__init__.py` to ensure heximpy is importable as package**

Check if `heximpy/__init__.py` exists. If not, create it:

```python
# heximpy/__init__.py
# (empty — makes heximpy importable as a package)
```

- [ ] **Step 5: Final commit**

```bash
git add heximpy/__init__.py
git commit -m "chore: add heximpy __init__.py for package imports"
```
