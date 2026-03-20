# HexSim Format I/O Improvements — Design Spec

**Date:** 2026-03-20
**Approach:** Correctness Sprint + Structural Foundation (3 phases)
**Priority:** Correctness/reliability first, extensibility second

## Context

Review of the HexSim I/O codebase identified six improvement areas:

1. **Narrow-grid neighbor bug** — `_hex_neighbors_offset()` uses `flat = nr * ncols + nc` which is wrong for flag=1 grids (majority of workspaces). Same bug exists in `HexMap.to_geodataframe()` and `to_geotiff()` in `hxnparser.py`.
2. **GridMeta/HexMap dimension swap** — `GridMeta.ncols = HexMap.height` is only documented in comments, not enforced
3. **No I/O validation** — corrupt files, dimension mismatches, and missing elements produce silent failures
4. **XML event loading gaps** — Two separate issues: (a) `build_events_from_xml()` in `xml_parser.py` is a dead-code stub not yet wired into the runtime path; (b) `scenario_loader.py:_build_single_event()` silently replaces unregistered event types (e.g., `reanimation`) with no-op DataProbeEvent wrappers (the name prefix `[unimplemented:{etype}]` is set but no warning is emitted)
5. **Temperature CSV fragility** — no header, no dimension validation
6. **Documentation drift** — `HexSimFormat.txt` describes flat-top/odd-q but code uses pointy-top/odd-row

## Ground Truth

Existing .hxn files produced by HexSim 4.0.20 Java application are treated as ground-truth test fixtures. Round-trip tests must preserve byte-level fidelity against these files.

---

## Phase 1: Critical Bug Fixes + Ground-Truth Test Fixtures

### 1a. Fix narrow-grid neighbor computation

**File:** `salmon_ibm/hexsim.py` — `_hex_neighbors_offset()` (lines 25-47)

**Problem:** The function computes `flat = nr * ncols + nc` (line 44). For narrow grids (flag=1), odd rows have `width-1` cells, so this formula produces wrong flat indices — it over-counts by 1 for every odd row above the target.

**Fix:**
- Add `flag` parameter to `_hex_neighbors_offset()`
- For flag=0: existing formula is correct
- For flag=1: compute flat index using cumulative row widths:
  ```python
  def _rowcol_to_flat_narrow(row, col, width):
      """Convert (row, col) to flat index for narrow grid."""
      # Even rows: width cells, odd rows: width-1 cells
      full_pairs = row // 2  # pairs of (even + odd) rows
      flat = full_pairs * (2 * width - 1)
      if row % 2 == 1:
          flat += width  # skip the even row
      flat += col
      return flat
  ```
- Replace the neighbor offset tables with pointy-top odd-row convention (matching the display and centroid code). The current function uses flat-top odd-q column-offset (`col % 2`) but must use odd-row (`row % 2`):
  ```python
  # Pointy-top, odd-row offset (ref: redblobgames.com/grids/hexagons/#coordinates-offset)
  # Odd rows are shifted RIGHT: cx = sqrt(3) * edge * (col + 0.5 * (row % 2))
  # Even rows (NOT shifted) → diagonal neighbors at lower col indices
  # Odd rows (shifted right) → diagonal neighbors at higher col indices
  if row % 2 == 0:  # even row (not shifted)
      offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
  else:             # odd row (shifted right)
      offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
  ```
  **Verification:** for cell (2,1) on even row at x=√3·edge, upper-left neighbor is (1,0)
  at x=0.5·√3·edge → offset (-1,-1). For cell (1,1) on odd row at x=1.5·√3·edge,
  upper-left neighbor is (0,1) at x=√3·edge → offset (-1,0). Both match the table above.
- Update all callers in `HexMesh.from_hexsim()` to pass the flag
- Also fix `HexMap.neighbors()` and `HexMap.hex_to_xy()` in `heximpy/hxnparser.py` which currently use flat-top odd-q convention (`col % 2`) — must be updated to pointy-top odd-row (`row % 2`) for consistency
- Fix `HexMap.to_geodataframe()` in `heximpy/hxnparser.py` which uses incorrect `flat = row * width + col` for narrow grids. Add a `_flat_to_rowcol()` helper (inverse of `_rowcol_to_flat_narrow`):
  ```python
  def _build_flat_to_rowcol(height, width, flag):
      """Build flat-index → (row, col) lookup arrays."""
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
- Fix `HexMap.to_geotiff()`: for narrow grids, `values.reshape((height, width))` will crash because `len(values) != height * width`. Add a guard that raises `ValueError("to_geotiff() is not supported for narrow grids (flag=1) — use to_geodataframe() instead")` or pad odd rows with nodata to fill the rectangle

**Data layout verification (must be done first):**
The .hxn binary data is row-major. The centroid code (verified against HexSim 4.0.20) uses pointy-top odd-row. Before changing the neighbor offsets, verify by loading a known workspace, computing neighbors with the proposed offsets, and confirming that neighbor centroids are geometrically adjacent (distance ≈ √3 × edge). This confirms the data layout matches the pointy-top convention. If it doesn't, the neighbor function must stay flat-top for data indexing and only the display uses pointy-top.

**Note:** `hxnparser.py` has its own `neighbors()` method (line 118) and `hex_to_xy()` (line 130) using flat-top odd-q convention. These must also be updated if the verification confirms pointy-top data layout, or documented as using a different convention if not.

**Acceptance criteria:**
- Data layout verification passes (neighbor centroids are geometrically adjacent)
- Neighbor symmetry holds for all cells in narrow-grid workspaces
- Neighbor count is 6 for interior cells, <6 for edge cells
- No cell references an out-of-bounds flat index

### 1b. Dimension swap safety

**Files:** `heximpy/hxnparser.py`, `salmon_ibm/hexsim.py`

**Problem:** `GridMeta.ncols` maps to `HexMap.height` and `GridMeta.nrows` maps to `HexMap.width`. This is only documented in comments and is a recurring source of bugs.

**Fix:**
- Add a `GridMeta.data_height` and `GridMeta.data_width` property pair that returns the values in HexMap's convention:
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
- Add a validation check in `HexMesh.from_hexsim()` where GridMeta and HexMap are joined (use `raise ValueError`, not `assert`, since assertions can be disabled with `python -O`):
  ```python
  if grid.data_height != extent_hm.height:
      raise ValueError(
          f"GridMeta/HexMap height mismatch: grid.data_height={grid.data_height} "
          f"!= hexmap.height={extent_hm.height}"
      )
  if grid.data_width != extent_hm.width:
      raise ValueError(
          f"GridMeta/HexMap width mismatch: grid.data_width={grid.data_width} "
          f"!= hexmap.width={extent_hm.width}"
      )
  ```

**Acceptance criteria:**
- All 8 workspaces load without assertion failures
- Any future dimension mismatch raises immediately with a clear message

### 1c. Ground-truth fixture tests

**File:** New test file `tests/test_hexsim_io.py`

**Test fixtures** (selected from existing workspaces):
- Wide grid PATCH_HEXMAP: `HexSim Examples/Spatial Data/Hexagons/Habitat/Habitat.1.hxn`
- Narrow grid PATCH_HEXMAP: `Columbia River Migration Model/Spatial Data/Hexagons/Extent/Extent.1.hxn` (or first available layer)
- Plain HXN: from TestWorkspace if available, otherwise create one via `HexMap.to_file()`
- GridMeta: `.grid` files from both wide and narrow workspaces

**Tests:**
1. **Read ground-truth:** Load each fixture, verify header fields match known values
2. **Round-trip fidelity:** Load → write to temp → read back → assert `values` arrays are identical
3. **Narrow-grid cell count:** Verify `n_hexagons` property matches actual data array length
4. **Neighbor symmetry (narrow):** Load narrow-grid workspace, test symmetry on all water cells
5. **Dimension consistency:** Load .grid + .hxn from same workspace, verify `grid.data_height == hm.height`
6. **Barrier file parsing:** Load .hbf, verify hex_id and edge values are within grid bounds

---

## Phase 2: Validation Layer Across I/O Entry Points

### 2a. HXN parser validation

**File:** `heximpy/hxnparser.py`

**`HexMap.from_file()` — read validation:**
- After reading header, compute expected cell count:
  ```python
  if flag == 0:
      expected = width * height
  else:
      # Narrow grid: even rows (0-indexed) have `width` cells, odd rows have `width-1`
      n_even = (height + 1) // 2   # ceil(height / 2)
      n_odd = height // 2          # floor(height / 2)
      expected = n_even * width + n_odd * (width - 1)
  ```
- Verify `len(values) == expected`, raise `ValueError` with details on mismatch
- Verify dtype_code is 1 or 2 (already done, confirm)

**`HexMap.to_file()` — write validation:**
- Verify `len(self.values) == self.n_hexagons` before writing
- Raise `ValueError(f"Cannot write HexMap: data length {len(self.values)} != expected {self.n_hexagons}")` on mismatch

**`GridMeta.from_file()` — physical plausibility:**
- `x_extent > 0`, `y_extent > 0`, `row_spacing > 0`, `edge > 0`
- `n_hexes > 0`, `ncols > 0`, `nrows > 0`
- Warn (don't error) if `edge` doesn't match `row_spacing / sqrt(3)` within tolerance

### 2b. Temperature CSV validation

**File:** `salmon_ibm/hexsim_env.py`

- After loading CSV with `np.loadtxt()`, verify shape:
  ```python
  if data.shape != (n_zones, n_timesteps):
      raise ValueError(
          f"Temperature CSV shape {data.shape} doesn't match "
          f"expected ({n_zones}, {n_timesteps})"
      )
  ```
- `n_zones` comes from the temperature zones HXN (number of unique non-zero values)
- `n_timesteps` comes from scenario XML `simulationParameters`

### 2c. XML parser validation

**File:** `salmon_ibm/xml_parser.py`, `salmon_ibm/scenario_loader.py`

**Required element validation** (xml_parser.py):
- `load_scenario_xml()` checks for required top-level elements: `simulationParameters`, `hexagonGrid`, at least one `population`
- Raise `ValueError("Missing required XML element: <{name}>")` if absent

**Event loading transparency** (scenario_loader.py):
- The existing code already prefixes unregistered events with `[unimplemented:{etype}]` in the event name. Augment this with `warnings.warn()` so the message is visible outside of debug inspection:
  ```
  "Event type '{etype}' not in EVENT_REGISTRY — replaced with no-op. Event: {name}"
  ```
- After all events are built, log summary: `"Loaded {n} events: {registered} registered, {skipped} unimplemented (no-op)"`

### 2d. World file / barrier file validation

**File:** `heximpy/hxnparser.py`

**World file:**
- Verify exactly 6 non-empty lines exist
- Verify all 6 parse as float; raise `ValueError` with line number on failure

**Barrier file:**
- After parsing, validate `hex_id < n_hexagons` and `0 <= edge <= 5`
- Raise `ValueError` identifying the offending line

---

## Phase 3: XML Parser Preparation for EVENT_REGISTRY Integration

### 3a. Structured event descriptors

**File:** New file `salmon_ibm/event_descriptors.py`

Define a base dataclass and typed subclasses:

```python
@dataclass
class EventDescriptor:
    """Typed representation of a parsed XML event."""
    name: str
    event_type: str           # matches EVENT_REGISTRY key
    timestep: int
    population_name: str
    enabled: bool = True

@dataclass
class MoveEventDescriptor(EventDescriptor):
    move_type: str = ""       # "random_walk", "target", etc.
    max_steps: int = 0
    affinity_name: str = ""

@dataclass
class SurvivalEventDescriptor(EventDescriptor):
    survival_expression: str = ""
    accumulator_refs: list[str] = field(default_factory=list)

# ... one per event type
```

This replaces the current raw dicts with validated, typed structures.

### 3b. Per-type parameter extraction

**File:** `salmon_ibm/xml_parser.py`

Factor current inline XML element reading into focused functions:

```python
def _parse_move_params(elem: ET.Element) -> dict:
    """Extract move event parameters from XML element."""
    ...

def _parse_survival_params(elem: ET.Element) -> dict:
    """Extract survival event parameters from XML element."""
    ...
```

Each returns a typed dict matching the corresponding `EventDescriptor` fields. The main `_parse_event()` function dispatches to these based on `_EVENT_TAG_MAP`.

Add `_parse_reanimation_params()` for the currently missing event type.

### 3c. Registry-driven loading path

**File:** `salmon_ibm/scenario_loader.py`

Refactor `_build_single_event()` to:

1. Receive an `EventDescriptor` (from Phase 3a) instead of a raw dict
2. Look up the `EVENT_REGISTRY` entry by `descriptor.event_type`
3. Call `cls.from_descriptor(descriptor)` — each event class implements this classmethod
4. The loader has no event-type-specific parameter mapping; it's just a dispatcher

**Breaking change:** Existing event classes currently use `__init__` with `**params` keyword unpacking (called from `scenario_loader.py`). Adding `from_descriptor()` is a new classmethod on each event class. The migration strategy:
1. Add `from_descriptor()` to each registered event class while keeping `__init__` unchanged
2. Update `_build_single_event()` to try `cls.from_descriptor(descriptor)` first, falling back to the old `cls(**params)` path if `from_descriptor` is not implemented — this allows incremental migration and rollback
3. Once all 16 event classes have `from_descriptor()` and tests pass, remove the fallback

Event classes that need modification: all 16 registered types in `events_builtin.py`, `events_hexsim.py`, `events_phase3.py`, and `interactions.py`.

This means adding a new event type requires:
1. Add `_parse_<type>_params()` in xml_parser.py
2. Add `<Type>EventDescriptor` in event_descriptors.py
3. Add `from_descriptor()` classmethod on the event class
4. Register with `@register_event()`

### 3d. Documentation alignment

**File:** `HexSimFormat.txt`

Add a section at the top:

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

GridMeta ↔ HexMap dimension mapping:
  GridMeta.ncols = HexMap.height (number of data rows)
  GridMeta.nrows = HexMap.width  (cells per row / stride)
  Use GridMeta.data_height / data_width for clarity.
```

---

## Testing Strategy

All tests use existing HexSim 4.0.20 workspace files as ground-truth fixtures.

**Test command:** `conda run -n shiny python -m pytest tests/ -v`

| Phase | Test Type | Files |
|-------|-----------|-------|
| 1 | Round-trip fidelity, neighbor symmetry, dimension consistency | `tests/test_hexsim_io.py` |
| 2 | Validation error messages (corrupt file fixtures, truncated data) | `tests/test_io_validation.py` |
| 3 | EventDescriptor parsing, registry dispatch, round-trip XML→Event | `tests/test_event_parsing.py` |

## Dependency Order

- Phase 1 has no dependencies — can start immediately
- Phase 2 depends on Phase 1c (fixture infrastructure)
- Phase 3a-3b can start in parallel with Phase 2
- Phase 3c depends on 3a and 3b
- Phase 3d can happen at any time

## Files Modified

| File | Phase | Changes |
|------|-------|---------|
| `salmon_ibm/hexsim.py` | 1a, 1b | Fix neighbor function, add dimension validation |
| `heximpy/hxnparser.py` | 1a, 1b, 2a, 2d | Fix narrow-grid in to_geodataframe/to_geotiff, add data_height/data_width, read/write validation |
| `tests/test_hexsim_io.py` | 1c | New: ground-truth fixture tests |
| `salmon_ibm/hexsim_env.py` | 2b | Temperature CSV validation |
| `salmon_ibm/xml_parser.py` | 2c, 3b | Required element checks, per-type extractors |
| `salmon_ibm/scenario_loader.py` | 2c, 3c | Warning on no-op fallback, registry-driven dispatch |
| `tests/test_io_validation.py` | 2 | New: validation error tests |
| `salmon_ibm/event_descriptors.py` | 3a | New: typed event descriptor dataclasses |
| `salmon_ibm/events_builtin.py` | 3c | Add `from_descriptor()` classmethod |
| `salmon_ibm/events_hexsim.py` | 3c | Add `from_descriptor()` classmethod |
| `salmon_ibm/events_phase3.py` | 3c | Add `from_descriptor()` classmethod |
| `salmon_ibm/interactions.py` | 3c | Add `from_descriptor()` classmethod |
| `tests/test_event_parsing.py` | 3 | New: event parsing tests |
| `HexSimFormat.txt` | 3d | Add implementation notes (clarify that file format is flat-top data layout, runtime converts to pointy-top display) |
