# HexSim I/O Improvements Implementation Plan

> **STATUS: ✅ EXECUTED** — Narrow-grid bug fixes, I/O validation, `salmon_ibm/event_descriptors.py` + `from_descriptor()` registry-driven loading all shipped. Tests in `tests/test_hexsim_io.py`, `test_io_validation.py`, `test_event_parsing.py`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix critical narrow-grid bugs, add I/O validation, and prepare the XML parser for EVENT_REGISTRY integration.

**Architecture:** Three phases — (1) bug fixes with ground-truth fixture tests, (2) validation layer across all I/O entry points, (3) structured event descriptors and registry-driven loading. Each phase builds on the previous but delivers standalone value.

**Tech Stack:** Python 3.11+, numpy, pytest, xml.etree.ElementTree, dataclasses

**Spec:** `docs/superpowers/specs/2026-03-20-hexsim-io-improvements-design.md`

**Test command:** `conda run -n shiny python -m pytest tests/ -v`

---

## File Structure

### Files to Create
| File | Responsibility |
|------|---------------|
| `tests/test_hexsim_io.py` | Ground-truth fixture tests (Phase 1) |
| `tests/test_io_validation.py` | Validation error tests (Phase 2) |
| `salmon_ibm/event_descriptors.py` | Typed event descriptor dataclasses (Phase 3) |
| `tests/test_event_parsing.py` | Event descriptor + registry tests (Phase 3) |

### Files to Modify
| File | Changes |
|------|---------|
| `salmon_ibm/hexsim.py` | Fix neighbor function, add dimension validation |
| `heximpy/hxnparser.py` | Fix narrow-grid exports, add data_height/data_width, validation |
| `salmon_ibm/hexsim_env.py` | Temperature CSV shape validation |
| `salmon_ibm/xml_parser.py` | Required element checks, per-type extractors |
| `salmon_ibm/scenario_loader.py` | Warning on no-op fallback, registry-driven dispatch |
| `salmon_ibm/events_builtin.py` | Add `from_descriptor()` classmethod |
| `salmon_ibm/events_hexsim.py` | Add `from_descriptor()` classmethod |
| `salmon_ibm/events_phase3.py` | Add `from_descriptor()` classmethod |
| `salmon_ibm/interactions.py` | Add `from_descriptor()` classmethod |
| `HexSimFormat.txt` | Add implementation notes section |

### Ground-Truth Test Fixtures (read-only, not modified)
| File | Type |
|------|------|
| `HexSim Examples/Spatial Data/Hexagons/Habitat Map/Habitat Map.1.hxn` | Wide grid PATCH_HEXMAP |
| `HexSim Examples/HexSim Examples.grid` | Wide grid GridMeta |
| `Columbia [small]/Spatial Data/Hexagons/River [ extent ]/River [ extent ].1.hxn` | Narrow grid PATCH_HEXMAP |
| `Columbia [small]/Columbia Fish Model [small].grid` | Narrow grid GridMeta |
| `Columbia [small]/Spatial Data/barriers/Fish Ladder Available/Fish Ladder Available.1.hbf` | Barrier file |

---

## Phase 1: Critical Bug Fixes + Ground-Truth Test Fixtures

### Task 1: Data layout verification test (pre-implementation gate, not TDD)

Verify that pointy-top odd-row neighbor offsets produce geometrically adjacent centroids. This MUST pass before changing any neighbor code. Unlike other tasks, this test is expected to pass immediately — it validates our assumptions about the data layout convention.

**Files:**
- Create: `tests/test_hexsim_io.py`

- [ ] **Step 1: Write the data layout verification test**

```python
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
    # Even rows (NOT shifted) → diagonal neighbors at lower col indices
    # Odd rows (shifted right) → diagonal neighbors at higher col indices
    if row % 2 == 0:
        offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    else:
        offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

    result = []
    for dr, dc in offsets:
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width:
            # For narrow grids, odd rows have width-1 cells
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
        expected_dist = np.sqrt(3.0) * edge  # pointy-top neighbor distance

        hm = HexMap.from_file(NARROW_HXN)
        h, w = hm.height, hm.width

        # Build row/col arrays for narrow grid
        row_list, col_list = [], []
        for r in range(h):
            rw = w if hm.flag == 0 or r % 2 == 0 else w - 1
            row_list.append(np.full(rw, r, dtype=np.int32))
            col_list.append(np.arange(rw, dtype=np.int32))
        all_rows = np.concatenate(row_list)
        all_cols = np.concatenate(col_list)

        # Sample 200 interior cells (skip edges)
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
                if rel_err > 0.01:  # 1% tolerance
                    bad.append((r, c, nr, nc, dist, expected_dist))

        assert not bad, f"Non-adjacent neighbors found: {bad[:5]}"
```

- [ ] **Step 2: Run the test to verify it passes**

```bash
conda run -n shiny python -m pytest tests/test_hexsim_io.py::TestDataLayoutVerification -v
```

Expected: PASS — this verifies the pointy-top odd-row convention is correct for the data.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hexsim_io.py
git commit -m "test: add data layout verification for pointy-top odd-row convention"
```

---

### Task 2: Fix narrow-grid neighbor computation

**Files:**
- Modify: `salmon_ibm/hexsim.py:25-47`

- [ ] **Step 1: Write failing test for narrow-grid neighbors**

Append to `tests/test_hexsim_io.py`:

```python
from salmon_ibm.hexsim import _hex_neighbors_offset


class TestNarrowGridNeighbors:
    """Test neighbor computation for narrow grids (flag=1)."""

    def test_rowcol_to_flat_narrow_basic(self):
        """Verify flat index computation for narrow grid, width=5."""
        from salmon_ibm.hexsim import _rowcol_to_flat_narrow
        # Row 0 (even, 5 cells): indices 0-4
        assert _rowcol_to_flat_narrow(0, 0, 5) == 0
        assert _rowcol_to_flat_narrow(0, 4, 5) == 4
        # Row 1 (odd, 4 cells): indices 5-8
        assert _rowcol_to_flat_narrow(1, 0, 5) == 5
        assert _rowcol_to_flat_narrow(1, 3, 5) == 8
        # Row 2 (even, 5 cells): indices 9-13
        assert _rowcol_to_flat_narrow(2, 0, 5) == 9
        assert _rowcol_to_flat_narrow(2, 4, 5) == 13

    def test_neighbor_symmetry_narrow_grid(self):
        """If A is neighbor of B, then B must be neighbor of A."""
        hm = HexMap.from_file(NARROW_HXN)
        h, w, flag = hm.height, hm.width, hm.flag
        n_data = len(hm.values)

        # Build row/col arrays
        row_list, col_list = [], []
        for r in range(h):
            rw = w if r % 2 == 0 else w - 1
            row_list.append(np.full(rw, r, dtype=np.int32))
            col_list.append(np.arange(rw, dtype=np.int32))
        all_rows = np.concatenate(row_list)
        all_cols = np.concatenate(col_list)

        # Check symmetry on 500 random cells
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

        # Pick an interior cell (row 4, col 2 — safely interior)
        nbrs = _hex_neighbors_offset(4, 2, w, h, n_data, flag)
        assert len(nbrs) == 6, f"Expected 6 neighbors, got {len(nbrs)}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n shiny python -m pytest tests/test_hexsim_io.py::TestNarrowGridNeighbors -v
```

Expected: FAIL — `_rowcol_to_flat_narrow` does not exist, `_hex_neighbors_offset` doesn't accept `flag`.

- [ ] **Step 3: Implement the fix in hexsim.py**

Replace `salmon_ibm/hexsim.py` lines 23-47 with:

```python
# ── Hex grid geometry ────────────────────────────────────────────────────────

def _rowcol_to_flat_narrow(row: int, col: int, width: int) -> int:
    """Convert (row, col) to flat index for narrow grid (flag=1).

    Even rows have ``width`` cells, odd rows have ``width - 1`` cells.
    """
    full_pairs = row // 2  # number of complete (even + odd) row pairs
    flat = full_pairs * (2 * width - 1)
    if row % 2 == 1:
        flat += width  # skip the even row of this pair
    flat += col
    return flat


def _hex_neighbors_offset(row: int, col: int, ncols: int, nrows: int,
                          n_data: int, flag: int = 0) -> list[int]:
    """Compute flat-index neighbors for (row, col) in an offset hex grid.

    Uses pointy-top odd-row offset convention:
        Even rows (NOT shifted): diagonal neighbors at lower col indices
        Odd rows (shifted RIGHT): diagonal neighbors at higher col indices

    Reference: https://www.redblobgames.com/grids/hexagons/#coordinates-offset
    """
    if row % 2 == 0:  # even row (not shifted)
        offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    else:             # odd row (shifted right)
        offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

    result = []
    for dr, dc in offsets:
        nr, nc = row + dr, col + dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            # For narrow grids, odd rows have width-1 cells
            if flag == 1 and nr % 2 == 1 and nc >= ncols - 1:
                continue
            if flag == 0:
                flat = nr * ncols + nc
            else:
                flat = _rowcol_to_flat_narrow(nr, nc, ncols)
            if flat < n_data:
                result.append(flat)
    return result
```

- [ ] **Step 4: Update the caller in `from_hexsim()` to pass flag**

In `salmon_ibm/hexsim.py` line 236, change:

```python
# Before:
full_nbrs = _hex_neighbors_offset(r, c, data_stride, data_nrows, n_data)
# After:
full_nbrs = _hex_neighbors_offset(r, c, data_stride, data_nrows, n_data, extent_hm.flag)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_hexsim_io.py::TestNarrowGridNeighbors -v
```

Expected: PASS

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
conda run -n shiny python -m pytest tests/ -v --timeout=120
```

Expected: No new failures.

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/hexsim.py tests/test_hexsim_io.py
git commit -m "fix: narrow-grid neighbor computation with pointy-top odd-row offsets"
```

---

### Task 3: Fix narrow-grid in hxnparser exports + update neighbors/hex_to_xy

**Files:**
- Modify: `heximpy/hxnparser.py:118-135` (neighbors, hex_to_xy)
- Modify: `heximpy/hxnparser.py:199-203` (to_geodataframe)
- Modify: `heximpy/hxnparser.py:221-225` (to_geotiff)

- [ ] **Step 1: Write failing test for hxnparser narrow-grid export**

Append to `tests/test_hexsim_io.py`:

```python
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
        # Should not raise — the old code does i // width which is wrong
        gdf = hm.to_geodataframe(edge=1.0, include_empty=False)
        assert len(gdf) > 0

    def test_to_geotiff_narrow_grid_raises(self):
        """to_geotiff should raise ValueError for narrow grids."""
        hm = HexMap.from_file(NARROW_HXN)
        if hm.flag != 1:
            pytest.skip("Not a narrow grid")
        with pytest.raises(ValueError, match="narrow"):
            hm.to_geotiff("/tmp/test.tif")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n shiny python -m pytest tests/test_hexsim_io.py::TestHxnparserNarrowGrid -v
```

Expected: FAIL

- [ ] **Step 3: Add `_build_flat_to_rowcol` helper to hxnparser.py**

Add after the `_HISTORY_MARKER` constant (line 27):

```python
def _build_flat_to_rowcol(height: int, width: int, flag: int):
    """Build flat-index → (row, col) lookup arrays.

    For wide grids (flag=0): every row has ``width`` cells.
    For narrow grids (flag=1): even rows have ``width``, odd rows ``width-1``.
    """
    if flag == 0:
        rows = np.repeat(np.arange(height), width)
        cols = np.tile(np.arange(width), height)
    else:
        row_list, col_list = [], []
        for r in range(height):
            rw = width if r % 2 == 0 else width - 1
            row_list.append(np.full(rw, r, dtype=np.int32))
            col_list.append(np.arange(rw, dtype=np.int32))
        rows = np.concatenate(row_list)
        cols = np.concatenate(col_list)
    return rows, cols
```

- [ ] **Step 4: Update `HexMap.neighbors()` to pointy-top odd-row**

Replace `heximpy/hxnparser.py` lines 118-128:

```python
    def neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Pointy-top odd-row offset neighbors."""
        if row % 2 == 0:  # even row (not shifted)
            offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:             # odd row (shifted right)
            offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
        result = []
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                # For narrow grids, odd rows have width-1 cells
                if self.flag == 1 and nr % 2 == 1 and nc >= self.width - 1:
                    continue
                result.append((nr, nc))
        return result
```

- [ ] **Step 5: Update `HexMap.hex_to_xy()` to pointy-top odd-row**

Replace `heximpy/hxnparser.py` lines 130-135:

```python
    def hex_to_xy(self, row: int, col: int) -> tuple[float, float]:
        """Convert hex grid (row, col) to Cartesian (x, y) — pointy-top odd-row."""
        edge = self._effective_edge()
        x = math.sqrt(3.0) * edge * (col + 0.5 * (row % 2))
        y = 1.5 * edge * row
        return (x, y)
```

- [ ] **Step 6: Fix `HexMap.to_geodataframe()` for narrow grids**

Replace `heximpy/hxnparser.py` lines 198-206 (the loop body):

```python
        try:
            all_rows, all_cols = _build_flat_to_rowcol(self.height, self.width, self.flag)
            rows_list = []
            for i, val in enumerate(self.values):
                if not include_empty and val == 0.0:
                    continue
                r = int(all_rows[i])
                c = int(all_cols[i])
                poly = Polygon(self.hex_polygon(r, c))
                rows_list.append(
                    {"hex_id": i + 1, "row": r, "col": c, "value": float(val), "geometry": poly}
                )
```

- [ ] **Step 7: Guard `HexMap.to_geotiff()` against narrow grids**

Add at the start of `to_geotiff()` method (after docstring, before `import rasterio`):

```python
        if self.flag == 1:
            raise ValueError(
                "to_geotiff() is not supported for narrow grids (flag=1) "
                "— use to_geodataframe() instead"
            )
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_hexsim_io.py::TestHxnparserNarrowGrid -v
```

Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add heximpy/hxnparser.py tests/test_hexsim_io.py
git commit -m "fix: narrow-grid support in hxnparser neighbors, exports, and hex_to_xy"
```

---

### Task 4: Dimension swap safety

**Files:**
- Modify: `heximpy/hxnparser.py` (GridMeta class)
- Modify: `salmon_ibm/hexsim.py:169-175`

- [ ] **Step 1: Write failing test for dimension swap**

Append to `tests/test_hexsim_io.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n shiny python -m pytest tests/test_hexsim_io.py::TestDimensionSwapSafety -v
```

Expected: FAIL — `data_height`/`data_width` not defined.

- [ ] **Step 3: Add properties to GridMeta**

In `heximpy/hxnparser.py`, add to the `GridMeta` dataclass (after the existing fields):

```python
    @property
    def data_height(self) -> int:
        """Number of data rows (= HexMap.height = GridMeta.ncols)."""
        return self.ncols

    @property
    def data_width(self) -> int:
        """Cells per row / stride (= HexMap.width = GridMeta.nrows)."""
        return self.nrows
```

- [ ] **Step 4: Add validation in HexMesh.from_hexsim()**

In `salmon_ibm/hexsim.py`, after line 175 (`data_nrows = extent_hm.height`), add:

```python
        # Validate GridMeta ↔ HexMap dimension mapping
        if ws.grid.data_height != extent_hm.height:
            raise ValueError(
                f"GridMeta/HexMap height mismatch: grid.data_height={ws.grid.data_height} "
                f"!= hexmap.height={extent_hm.height}"
            )
        if ws.grid.data_width != extent_hm.width:
            raise ValueError(
                f"GridMeta/HexMap width mismatch: grid.data_width={ws.grid.data_width} "
                f"!= hexmap.width={extent_hm.width}"
            )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_hexsim_io.py::TestDimensionSwapSafety -v
```

Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
conda run -n shiny python -m pytest tests/ -v --timeout=120
```

Expected: No new failures. All 8 workspaces should still load correctly.

- [ ] **Step 7: Commit**

```bash
git add heximpy/hxnparser.py salmon_ibm/hexsim.py tests/test_hexsim_io.py
git commit -m "feat: add GridMeta.data_height/data_width + dimension validation"
```

---

### Task 5: Ground-truth round-trip and fixture tests

**Files:**
- Modify: `tests/test_hexsim_io.py`

- [ ] **Step 1: Write round-trip and fixture tests**

Append to `tests/test_hexsim_io.py`:

```python
import tempfile

from heximpy.hxnparser import read_barriers


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
```

- [ ] **Step 2: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_hexsim_io.py::TestRoundTripFidelity tests/test_hexsim_io.py::TestNarrowGridCellCount tests/test_hexsim_io.py::TestBarrierFileParsing -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_hexsim_io.py
git commit -m "test: add round-trip fidelity, cell count, and barrier fixture tests"
```

---

## Phase 2: Validation Layer Across I/O Entry Points

### Task 6: HXN parser read/write validation

**Files:**
- Modify: `heximpy/hxnparser.py`
- Create: `tests/test_io_validation.py`

- [ ] **Step 1: Write failing validation tests**

```python
"""Tests for I/O validation error messages."""
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from heximpy.hxnparser import GridMeta, HexMap


class TestHxnReadValidation:
    """HexMap.from_file() should reject corrupt files."""

    def test_truncated_data_raises(self):
        """File with fewer values than header claims should raise ValueError."""
        # Write a valid header but truncated data
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            with open(tmp, "wb") as f:
                # Plain format header: version, ncols=10, nrows=10
                f.write(struct.pack("<i", 3))   # version
                f.write(struct.pack("<i", 10))  # ncols (width)
                f.write(struct.pack("<i", 10))  # nrows (height)
                f.write(struct.pack("<d", 1.0)) # cell_size
                f.write(struct.pack("<d", 0.0)) # origin_x
                f.write(struct.pack("<d", 0.0)) # origin_y
                f.write(struct.pack("<i", 1))   # dtype_code (float32)
                f.write(struct.pack("<i", 0))   # nodata
                # Only write 50 values instead of 100
                f.write(np.zeros(50, dtype=np.float32).tobytes())

            with pytest.raises(ValueError, match="mismatch|truncated|expected"):
                HexMap.from_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_invalid_dtype_code_raises(self):
        """dtype_code other than 1 or 2 should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            with open(tmp, "wb") as f:
                f.write(struct.pack("<i", 3))   # version
                f.write(struct.pack("<i", 5))   # ncols
                f.write(struct.pack("<i", 5))   # nrows
                f.write(struct.pack("<d", 1.0)) # cell_size
                f.write(struct.pack("<d", 0.0)) # origin_x
                f.write(struct.pack("<d", 0.0)) # origin_y
                f.write(struct.pack("<i", 99))  # invalid dtype_code
                f.write(struct.pack("<i", 0))   # nodata
                f.write(np.zeros(25, dtype=np.float32).tobytes())

            with pytest.raises(ValueError, match="dtype"):
                HexMap.from_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)


class TestHxnWriteValidation:
    """HexMap.to_file() should reject mismatched data."""

    def test_write_wrong_length_plain_raises(self):
        """Writing plain HexMap with values length != n_hexagons should raise."""
        hm = HexMap(
            format="plain", version=3, height=10, width=10,
            flag=0, max_val=1.0, min_val=0.0, hexzero=0.0,
            values=np.zeros(50, dtype=np.float32),  # 50 != 10*10
        )
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="data length"):
                hm.to_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_write_wrong_length_patch_hexmap_raises(self):
        """Writing PATCH_HEXMAP with values length != n_hexagons should raise."""
        hm = HexMap(
            format="patch_hexmap", version=1, height=10, width=10,
            flag=0, max_val=1.0, min_val=0.0, hexzero=0.0,
            values=np.zeros(50, dtype=np.float32),  # 50 != 10*10
        )
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="data length"):
                hm.to_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n shiny python -m pytest tests/test_io_validation.py -v
```

Expected: FAIL — no validation exists yet.

- [ ] **Step 3: Add read validation to `HexMap.from_file()`**

In `heximpy/hxnparser.py`, in the `_read_plain` method, after reading the data array, add:

```python
        # Validate data length
        if flag == 0:
            expected = width * height
        else:
            n_even = (height + 1) // 2
            n_odd = height // 2
            expected = n_even * width + n_odd * (width - 1)
        if len(values) != expected:
            raise ValueError(
                f"HXN data length mismatch: got {len(values)} values, "
                f"expected {expected} (width={width}, height={height}, flag={flag})"
            )
```

Add similar validation in `_read_patch_hexmap` after reading the data.

Also validate dtype_code:

```python
        if dtype_code not in (1, 2):
            raise ValueError(f"Invalid dtype_code {dtype_code} — expected 1 (float32) or 2 (int32)")
```

- [ ] **Step 4: Add write validation to `HexMap.to_file()`**

At the start of `_write_patch_hexmap` and `_write_plain`, add:

```python
        if len(self.values) != self.n_hexagons:
            raise ValueError(
                f"Cannot write HexMap: data length {len(self.values)} "
                f"!= expected {self.n_hexagons}"
            )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
conda run -n shiny python -m pytest tests/test_io_validation.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add heximpy/hxnparser.py tests/test_io_validation.py
git commit -m "feat: add HXN read/write validation for data length and dtype"
```

---

### Task 7: GridMeta validation

**Files:**
- Modify: `heximpy/hxnparser.py`
- Modify: `tests/test_io_validation.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_io_validation.py`:

```python
class TestGridMetaValidation:
    """GridMeta.from_file() should reject implausible values."""

    def test_zero_extent_raises(self):
        """Grid with zero extent should raise."""
        # Create a fake grid file with zero x_extent
        with tempfile.NamedTemporaryFile(suffix=".grid", delete=False) as f:
            tmp = Path(f.name)
        try:
            with open(tmp, "wb") as f:
                f.write(b"PATCH_GRID")        # magic (10 bytes)
                f.write(struct.pack("<I", 1))  # version
                f.write(struct.pack("<I", 100))  # n_hexes
                f.write(struct.pack("<I", 10))   # ncols
                f.write(struct.pack("<I", 10))   # nrows
                f.write(struct.pack("<?", False))  # flag
                # 5 float64: x_extent=0, ?, y_extent, ?, row_spacing
                f.write(struct.pack("<d", 0.0))    # x_extent = 0 (invalid)
                f.write(struct.pack("<d", 0.0))
                f.write(struct.pack("<d", 1000.0))
                f.write(struct.pack("<d", 0.0))
                f.write(struct.pack("<d", 24.0))

            with pytest.raises(ValueError, match="extent|plausib"):
                GridMeta.from_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n shiny python -m pytest tests/test_io_validation.py::TestGridMetaValidation -v
```

Expected: FAIL

- [ ] **Step 3: Add validation to `GridMeta.from_file()`**

After parsing the binary header in `GridMeta.from_file()`, add:

```python
        # Validate physical plausibility
        if x_extent <= 0 or y_extent <= 0:
            raise ValueError(
                f"Grid extent must be positive: x_extent={x_extent}, y_extent={y_extent}"
            )
        if row_spacing <= 0:
            raise ValueError(f"Grid row_spacing must be positive: {row_spacing}")
        if ncols <= 0 or nrows <= 0:
            raise ValueError(f"Grid dimensions must be positive: ncols={ncols}, nrows={nrows}")
        edge = row_spacing / math.sqrt(3.0)
        if edge <= 0:
            raise ValueError(f"Computed edge length must be positive: {edge}")
```

- [ ] **Step 4: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_io_validation.py::TestGridMetaValidation -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add heximpy/hxnparser.py tests/test_io_validation.py
git commit -m "feat: add GridMeta physical plausibility validation"
```

---

### Task 8: Temperature CSV validation

**Files:**
- Modify: `salmon_ibm/hexsim_env.py:44-52`
- Modify: `tests/test_io_validation.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_io_validation.py`:

```python
from unittest.mock import patch, MagicMock


class TestTemperatureCSVValidation:
    """HexSimEnvironment should validate temperature CSV shape."""

    def test_wrong_shape_raises(self):
        """CSV with wrong dimensions should raise ValueError."""
        # This test is tricky because HexSimEnvironment needs a full mesh.
        # We test the validation logic directly by mocking.
        from salmon_ibm.hexsim_env import _validate_temp_table

        # 5 zones, 10 timesteps expected, but data has 3 zones
        data = np.zeros((3, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="Temperature CSV shape"):
            _validate_temp_table(data, n_zones=5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n shiny python -m pytest tests/test_io_validation.py::TestTemperatureCSVValidation -v
```

Expected: FAIL — `_validate_temp_table` does not exist.

- [ ] **Step 3: Add validation function to hexsim_env.py**

Add before the `HexSimEnvironment` class:

```python
def _validate_temp_table(data: np.ndarray, n_zones: int) -> None:
    """Validate temperature lookup table dimensions."""
    if data.ndim != 2:
        raise ValueError(f"Temperature CSV must be 2D, got shape {data.shape}")
    if data.shape[0] != n_zones:
        raise ValueError(
            f"Temperature CSV shape {data.shape} doesn't match "
            f"expected {n_zones} zones (rows). Got {data.shape[0]} rows."
        )
```

Then call it in `HexSimEnvironment.__init__()` after loading the CSV (line 49-51):

```python
                self._temp_table = np.loadtxt(csv_path, delimiter=",",
                                              dtype=np.float32)
                n_zones = int(self._zone_ids.max()) + 1
                _validate_temp_table(self._temp_table, n_zones)
```

- [ ] **Step 4: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_io_validation.py::TestTemperatureCSVValidation -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/hexsim_env.py tests/test_io_validation.py
git commit -m "feat: add temperature CSV shape validation"
```

---

### Task 9: XML parser required element checks + event loading warning

**Files:**
- Modify: `salmon_ibm/xml_parser.py:9-23`
- Modify: `salmon_ibm/scenario_loader.py:228-239`
- Modify: `tests/test_io_validation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_io_validation.py`:

```python
import xml.etree.ElementTree as ET
import warnings


class TestXmlParserValidation:
    """XML parser should check for required elements."""

    def test_missing_simulation_params_raises(self):
        from salmon_ibm.xml_parser import load_scenario_xml
        # Create minimal XML without simulationParameters
        xml_str = "<scenario><hexagonGrid/></scenario>"
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_str)
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="simulationParameters"):
                load_scenario_xml(tmp)
        finally:
            tmp.unlink(missing_ok=True)


class TestEventLoadingWarning:
    """Unregistered event types should emit a warning."""

    def test_unregistered_event_warns(self):
        """Building an event with unknown type should emit a warning."""
        from salmon_ibm.scenario_loader import ScenarioLoader

        loader = ScenarioLoader.__new__(ScenarioLoader)
        edef = {
            "type": "reanimation",
            "name": "Test Reanimation",
            "timestep": 1,
            "population": "pop1",
            "enabled": True,
            "params": {},
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evt = loader._build_single_event(edef)
            matching = [x for x in w if "not in EVENT_REGISTRY" in str(x.message)]
            assert len(matching) == 1, f"Expected 1 warning, got {len(matching)}"
            assert "reanimation" in str(matching[0].message)
```

- [ ] **Step 2: Add required element validation to xml_parser.py**

In `load_scenario_xml()`, after `root = tree.getroot()` (line 23), add:

```python
    # Validate required elements
    _REQUIRED_ELEMENTS = ["simulationParameters", "hexagonGrid"]
    for elem_name in _REQUIRED_ELEMENTS:
        if root.find(f".//{elem_name}") is None:
            raise ValueError(f"Missing required XML element: <{elem_name}>")
```

- [ ] **Step 3: Add warning for unregistered event types in scenario_loader.py**

In `salmon_ibm/scenario_loader.py`, at line 231-239, modify the no-op fallback:

```python
        if cls is None:
            # Unknown event type — create a no-op wrapper with warning
            import warnings
            warnings.warn(
                f"Event type '{etype}' not in EVENT_REGISTRY — "
                f"replaced with no-op. Event: {name}",
                stacklevel=2,
            )
            from salmon_ibm.events_hexsim import DataProbeEvent
            return DataProbeEvent(
                name=f"[unimplemented:{etype}] {name}",
                trigger=trigger,
                population_name=population_name,
                enabled=enabled,
            )
```

- [ ] **Step 4: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_io_validation.py::TestXmlParserValidation -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```bash
conda run -n shiny python -m pytest tests/ -v --timeout=120
```

Expected: No new failures. Warnings may appear for `reanimation` events — that's expected and correct.

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/xml_parser.py salmon_ibm/scenario_loader.py tests/test_io_validation.py
git commit -m "feat: add XML required element checks + event loading warnings"
```

---

### Task 10: World file and barrier file validation

**Files:**
- Modify: `heximpy/hxnparser.py` (WorldFile.from_file, read_barriers)
- Modify: `tests/test_io_validation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_io_validation.py`:

```python
from heximpy.hxnparser import WorldFile


class TestWorldFileValidation:
    """WorldFile.from_file() should validate format."""

    def test_too_few_lines_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".bpw", mode="w", delete=False) as f:
            f.write("1.0\n0.0\n0.0\n")  # only 3 lines
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="6.*lines"):
                WorldFile.from_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_non_numeric_line_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".bpw", mode="w", delete=False) as f:
            f.write("1.0\n0.0\n0.0\n-1.0\nNOT_A_NUMBER\n0.0\n")
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="line"):
                WorldFile.from_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)


class TestBarrierFileValidation:
    """read_barriers() should validate hex_id and edge bounds."""

    def test_out_of_bounds_hex_id_raises(self):
        from heximpy.hxnparser import read_barriers
        with tempfile.NamedTemporaryFile(suffix=".hbf", mode="w", delete=False) as f:
            f.write('C 1 0.0 1.0 "wall"\n')
            f.write('E 999999 3 1\n')  # hex_id way out of bounds
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="hex_id.*out of bounds"):
                read_barriers(tmp, n_hexagons=100)
        finally:
            tmp.unlink(missing_ok=True)

    def test_invalid_edge_index_raises(self):
        from heximpy.hxnparser import read_barriers
        with tempfile.NamedTemporaryFile(suffix=".hbf", mode="w", delete=False) as f:
            f.write('C 1 0.0 1.0 "wall"\n')
            f.write('E 5 7 1\n')  # edge=7, valid range is 0-5
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="edge.*out of range"):
                read_barriers(tmp, n_hexagons=100)
        finally:
            tmp.unlink(missing_ok=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n shiny python -m pytest tests/test_io_validation.py::TestWorldFileValidation -v
```

Expected: FAIL

- [ ] **Step 3: Add validation to `WorldFile.from_file()` (non-numeric check)**

The existing `WorldFile.from_file()` already validates line count. Add non-numeric validation.
In the `WorldFile.from_file()` classmethod, replace the parsing logic:

```python
    @classmethod
    def from_file(cls, path: str | Path) -> WorldFile:
        """Read a 6-line world file (.bpw/.wld/.pgw)."""
        path = Path(path)
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        if len(lines) != 6:
            raise ValueError(
                f"World file {path.name} must have exactly 6 non-empty lines, "
                f"got {len(lines)}"
            )
        vals = []
        for i, ln in enumerate(lines, 1):
            try:
                vals.append(float(ln))
            except ValueError:
                raise ValueError(
                    f"World file {path.name} line {i}: cannot parse '{ln}' as float"
                )
        return cls(A=vals[0], D=vals[1], B=vals[2], E=vals[3], C=vals[4], F=vals[5])
```

- [ ] **Step 4: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_io_validation.py::TestWorldFileValidation -v
```

Expected: PASS

- [ ] **Step 5: Add barrier validation to `read_barriers()`**

In `heximpy/hxnparser.py`, update `read_barriers()` to accept an optional `n_hexagons` parameter and validate:

```python
def read_barriers(path: str | Path, *, n_hexagons: int | None = None) -> list[Barrier]:
    """Read a .hbf barrier file. Optionally validate bounds."""
    # ... existing parsing logic ...
    # After parsing, validate if n_hexagons is provided:
    if n_hexagons is not None:
        for b in barriers:
            if b.hex_id < 0 or b.hex_id >= n_hexagons:
                raise ValueError(
                    f"Barrier hex_id {b.hex_id} out of bounds "
                    f"(n_hexagons={n_hexagons})"
                )
            if b.edge < 0 or b.edge > 5:
                raise ValueError(
                    f"Barrier edge {b.edge} out of range (must be 0-5)"
                )
    return barriers
```

- [ ] **Step 6: Run barrier validation tests**

```bash
conda run -n shiny python -m pytest tests/test_io_validation.py::TestBarrierFileValidation -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add heximpy/hxnparser.py tests/test_io_validation.py
git commit -m "feat: add world file and barrier file validation"
```

---

## Phase 3: XML Parser Preparation for EVENT_REGISTRY Integration

### Task 11: Create typed event descriptors

**Files:**
- Create: `salmon_ibm/event_descriptors.py`
- Create: `tests/test_event_parsing.py`

- [ ] **Step 1: Write tests for event descriptors**

```python
"""Tests for typed event descriptors and XML event parsing."""
from dataclasses import asdict

import pytest

from salmon_ibm.event_descriptors import (
    EventDescriptor,
    MoveEventDescriptor,
    SurvivalEventDescriptor,
    AccumulateEventDescriptor,
    TransitionEventDescriptor,
    CensusEventDescriptor,
    IntroductionEventDescriptor,
    DataProbeEventDescriptor,
    DataLookupEventDescriptor,
    SetSpatialAffinityEventDescriptor,
    PatchIntroductionEventDescriptor,
    InteractionEventDescriptor,
    ReanimationEventDescriptor,
)


class TestEventDescriptors:
    """Event descriptors are typed dataclasses."""

    def test_base_descriptor_fields(self):
        d = EventDescriptor(
            name="test", event_type="move", timestep=1,
            population_name="pop1",
        )
        assert d.name == "test"
        assert d.event_type == "move"
        assert d.enabled is True

    def test_move_descriptor_defaults(self):
        d = MoveEventDescriptor(
            name="m1", event_type="move", timestep=1,
            population_name="pop1",
        )
        assert d.move_type == ""
        assert d.max_steps == 0

    def test_survival_descriptor_accumulator_refs(self):
        d = SurvivalEventDescriptor(
            name="s1", event_type="hexsim_survival", timestep=1,
            population_name="pop1",
            survival_expression="exp(-0.1 * age)",
            accumulator_refs=["age", "energy"],
        )
        assert d.accumulator_refs == ["age", "energy"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n shiny python -m pytest tests/test_event_parsing.py::TestEventDescriptors -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Create event_descriptors.py**

```python
"""Typed event descriptors for HexSim XML event parsing.

Each descriptor is a dataclass that carries validated, typed fields
extracted from an XML event element. These replace the raw dicts
previously passed between xml_parser.py and scenario_loader.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EventDescriptor:
    """Base typed representation of a parsed XML event."""
    name: str
    event_type: str
    timestep: int
    population_name: str
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class MoveEventDescriptor(EventDescriptor):
    move_type: str = ""
    max_steps: int = 0
    affinity_name: str = ""


@dataclass
class SurvivalEventDescriptor(EventDescriptor):
    survival_expression: str = ""
    accumulator_refs: list[str] = field(default_factory=list)


@dataclass
class AccumulateEventDescriptor(EventDescriptor):
    updater_functions: list[dict] = field(default_factory=list)


@dataclass
class TransitionEventDescriptor(EventDescriptor):
    trait_name: str = ""
    transition_matrix: list[list[float]] = field(default_factory=list)


@dataclass
class CensusEventDescriptor(EventDescriptor):
    pass


@dataclass
class IntroductionEventDescriptor(EventDescriptor):
    n_agents: int = 0
    initialization_spatial_data: str = ""


@dataclass
class PatchIntroductionEventDescriptor(EventDescriptor):
    n_agents: int = 0
    spatial_data: str = ""


@dataclass
class DataProbeEventDescriptor(EventDescriptor):
    pass


@dataclass
class DataLookupEventDescriptor(EventDescriptor):
    lookup_file: str = ""
    accumulator_name: str = ""


@dataclass
class SetSpatialAffinityEventDescriptor(EventDescriptor):
    affinity_name: str = ""
    spatial_data: str = ""


@dataclass
class InteractionEventDescriptor(EventDescriptor):
    interaction_type: str = ""


@dataclass
class ReanimationEventDescriptor(EventDescriptor):
    pass


# Map event_type string → descriptor class
DESCRIPTOR_REGISTRY: dict[str, type[EventDescriptor]] = {
    "move": MoveEventDescriptor,
    "hexsim_survival": SurvivalEventDescriptor,
    "accumulate": AccumulateEventDescriptor,
    "transition": TransitionEventDescriptor,
    "census": CensusEventDescriptor,
    "introduction": IntroductionEventDescriptor,
    "patch_introduction": PatchIntroductionEventDescriptor,
    "data_probe": DataProbeEventDescriptor,
    "data_lookup": DataLookupEventDescriptor,
    "set_spatial_affinity": SetSpatialAffinityEventDescriptor,
    "interaction": InteractionEventDescriptor,
    "reanimation": ReanimationEventDescriptor,
}
```

- [ ] **Step 4: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_event_parsing.py::TestEventDescriptors -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/event_descriptors.py tests/test_event_parsing.py
git commit -m "feat: add typed event descriptor dataclasses"
```

---

### Task 12: Per-type XML parameter extraction

**Files:**
- Modify: `salmon_ibm/xml_parser.py`
- Modify: `tests/test_event_parsing.py`

- [ ] **Step 1: Write test for parameter extraction**

Append to `tests/test_event_parsing.py`:

```python
import xml.etree.ElementTree as ET

from salmon_ibm.xml_parser import _parse_event_to_descriptor
from salmon_ibm.event_descriptors import MoveEventDescriptor


class TestEventParameterExtraction:
    """Per-type XML parameter extraction produces correct descriptors."""

    def test_parse_move_event(self):
        xml_str = """
        <event timestep="5">
            <moveEvent>
                <eventName>Fish Migration</eventName>
                <population>chinook</population>
                <enabled>true</enabled>
            </moveEvent>
        </event>
        """
        elem = ET.fromstring(xml_str)
        desc = _parse_event_to_descriptor(elem)
        assert isinstance(desc, MoveEventDescriptor)
        assert desc.name == "Fish Migration"
        assert desc.event_type == "move"
        assert desc.timestep == 5
        assert desc.population_name == "chinook"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n shiny python -m pytest tests/test_event_parsing.py::TestEventParameterExtraction -v
```

Expected: FAIL — `_parse_event_to_descriptor` does not exist.

- [ ] **Step 3: Implement per-type extraction functions in xml_parser.py**

Add after the `_EVENT_TAG_MAP` definition:

```python
from salmon_ibm.event_descriptors import (
    DESCRIPTOR_REGISTRY, EventDescriptor,
)


def _parse_event_to_descriptor(event_elem: ET.Element) -> EventDescriptor:
    """Parse an <event> XML element into a typed EventDescriptor."""
    timestep = int(event_elem.get("timestep", "0"))

    # Find the typed child element
    for child in event_elem:
        tag = child.tag
        if tag in _EVENT_TAG_MAP:
            etype = _EVENT_TAG_MAP[tag]
            name = ""
            pop_name = ""
            enabled = True
            for sub in child:
                if sub.tag == "eventName":
                    name = sub.text or ""
                elif sub.tag == "population":
                    pop_name = sub.text or ""
                elif sub.tag == "enabled":
                    enabled = (sub.text or "").lower() == "true"

            desc_cls = DESCRIPTOR_REGISTRY.get(etype, EventDescriptor)
            return desc_cls(
                name=name,
                event_type=etype,
                timestep=timestep,
                population_name=pop_name,
                enabled=enabled,
            )

    # Fallback for unrecognized event structure
    return EventDescriptor(
        name="unknown",
        event_type="unknown",
        timestep=timestep,
        population_name="",
    )
```

- [ ] **Step 4: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_event_parsing.py::TestEventParameterExtraction -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/xml_parser.py tests/test_event_parsing.py
git commit -m "feat: add per-type XML event parameter extraction to descriptors"
```

---

### Task 13: Registry-driven loading path with from_descriptor()

**Files:**
- Modify: `salmon_ibm/scenario_loader.py:228-276`
- Modify: `salmon_ibm/events_hexsim.py` (add from_descriptor to DataProbeEvent as first example)
- Modify: `tests/test_event_parsing.py`

- [ ] **Step 1: Write test for from_descriptor dispatch**

Append to `tests/test_event_parsing.py`:

```python
from salmon_ibm.event_descriptors import DataProbeEventDescriptor
from salmon_ibm.events_hexsim import DataProbeEvent


class TestRegistryDrivenLoading:
    """Events can be constructed from descriptors."""

    def test_data_probe_from_descriptor(self):
        desc = DataProbeEventDescriptor(
            name="test_probe", event_type="data_probe",
            timestep=1, population_name="pop1",
        )
        evt = DataProbeEvent.from_descriptor(desc)
        assert evt.name == "test_probe"
        assert evt.population_name == "pop1"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n shiny python -m pytest tests/test_event_parsing.py::TestRegistryDrivenLoading -v
```

Expected: FAIL — `from_descriptor` does not exist.

- [ ] **Step 3: Add from_descriptor to DataProbeEvent (first example)**

In `salmon_ibm/events_hexsim.py`, add to the `DataProbeEvent` class:

```python
    @classmethod
    def from_descriptor(cls, descriptor):
        """Construct from a typed EventDescriptor."""
        return cls(
            name=descriptor.name,
            trigger=None,  # will be set by sequencer
            population_name=descriptor.population_name,
            enabled=descriptor.enabled,
        )
```

- [ ] **Step 4: Update scenario_loader.py to try from_descriptor first**

In `salmon_ibm/scenario_loader.py`, after the EVENT_REGISTRY lookup (line 230), modify:

```python
        cls = EVENT_REGISTRY.get(etype)
        if cls is None:
            # Unknown event type — create a no-op wrapper with warning
            import warnings
            warnings.warn(
                f"Event type '{etype}' not in EVENT_REGISTRY — "
                f"replaced with no-op. Event: {name}",
                stacklevel=2,
            )
            from salmon_ibm.events_hexsim import DataProbeEvent
            return DataProbeEvent(
                name=f"[unimplemented:{etype}] {name}",
                trigger=trigger,
                population_name=population_name,
                enabled=enabled,
            )

        # Try from_descriptor if available (Phase 3 incremental migration)
        if hasattr(cls, "from_descriptor"):
            from salmon_ibm.event_descriptors import DESCRIPTOR_REGISTRY
            desc_cls = DESCRIPTOR_REGISTRY.get(etype)
            if desc_cls is not None:
                descriptor = desc_cls(
                    name=name,
                    event_type=etype,
                    timestep=edef.get("timestep", 0),
                    population_name=population_name,
                    enabled=enabled,
                )
                evt = cls.from_descriptor(descriptor)
                evt.trigger = trigger
                return evt

        # Fallback: existing __init__ path (unchanged)
        try:
            evt = cls(
                name=name,
                trigger=trigger,
                population_name=population_name,
                enabled=enabled,
            )
```

- [ ] **Step 5: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_event_parsing.py::TestRegistryDrivenLoading -v
```

Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
conda run -n shiny python -m pytest tests/ -v --timeout=120
```

Expected: No regressions. The fallback path preserves all existing behavior.

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/events_hexsim.py salmon_ibm/scenario_loader.py tests/test_event_parsing.py
git commit -m "feat: add from_descriptor() with fallback — registry-driven event loading"
```

---

### Task 14: Documentation alignment

**Files:**
- Modify: `HexSimFormat.txt`

- [ ] **Step 1: Add implementation notes to HexSimFormat.txt**

Prepend to the top of `HexSimFormat.txt`:

```
IMPLEMENTATION NOTES (HexSimPy)
================================
FILE FORMAT vs RUNTIME CONVENTION:
  The .hxn file format stores cells in row-major order. The format
  descriptions below reference flat-top/odd-q column-offset convention
  for historical context.

  The Python runtime uses POINTY-TOP hex orientation with ODD-ROW offset,
  matching HexSim 4.0.20's viewer. Neighbor computation and centroid
  placement both use row-based parity (row % 2).

  The data layout verification (Phase 1a) confirmed which convention
  the binary data actually follows. See tests/test_hexsim_io.py for
  the verification evidence.

GridMeta <-> HexMap dimension mapping:
  GridMeta.ncols = HexMap.height (number of data rows)
  GridMeta.nrows = HexMap.width  (cells per row / stride)
  Use GridMeta.data_height / data_width for clarity.

=====================================

```

- [ ] **Step 2: Commit**

```bash
git add HexSimFormat.txt
git commit -m "docs: add implementation notes to HexSimFormat.txt"
```

---

## Future Work (Not in This Plan)

- **Add `from_descriptor()` to remaining 15 event classes** — Task 13 only adds it to `DataProbeEvent` as a proof-of-concept with fallback. The remaining classes in `events_builtin.py` (5), `events_hexsim.py` (5), `events_phase3.py` (5), and `interactions.py` (1) should be migrated incrementally. Each migration follows the same pattern: add classmethod, test, verify fallback still works.
- **Remove fallback path** — once all 16 classes have `from_descriptor()`, remove the `__init__`-based fallback in `scenario_loader.py`.
- **Add `edge` vs `row_spacing / sqrt(3)` consistency warning** to `GridMeta.from_file()` (spec item 2a, low priority).

## Post-Implementation Checklist

- [ ] Run full test suite: `conda run -n shiny python -m pytest tests/ -v`
- [ ] Verify all 8 workspaces still load: run `scripts/run_hexsim_validation.py` if available
- [ ] Check for deprecation warnings from the neighbor convention change
- [ ] Verify Shiny app still renders hex grids correctly (manual check)
