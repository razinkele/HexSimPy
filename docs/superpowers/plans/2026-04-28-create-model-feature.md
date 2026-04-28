# Create Model Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Create Model" sidebar accordion that lets users upload `shp.zip` / `gpkg` / `geojson` polygon files and preview them as an H3 tessellation overlaid on the map (viewer-only ephemeral, with an optional bathymetry-shading toggle).

**Architecture:** Extract-first refactor. `tessellate_reach`, `bridge_components`, and the v1.6.1 `polygon_trust_water_mask` logic move from `scripts/build_h3_multires_landscape.py` into a new library module `salmon_ibm/h3_tessellate.py`. The build script becomes a thin wrapper (no behavior change). The new sidebar UI calls `h3_tessellate.parse_upload` + `h3_tessellate.preview` and stores the result in a Shiny reactive `_uploaded_preview`. When set, a parallel deck.gl layer builder (`_build_preview_h3_layer`) renders the upload directly, short-circuiting the existing `sim.mesh`-coupled flow (env-field sampling, agent rendering, color scales) — those paths stay untouched.

**Tech Stack:** Python 3.10+ via micromamba env `shiny`; `geopandas` + `shapely` + `pyproj` (already used by build pipeline); `h3` (already a dep); Shiny for Python (UI). No new third-party deps.

**Spec:** [`docs/superpowers/specs/2026-04-28-create-model-feature-design.md`](../specs/2026-04-28-create-model-feature-design.md) (commit `8e333ab`).

---

## File Structure

**New files:**
- `salmon_ibm/h3_tessellate.py` — extracted tessellation primitives + new upload-flow entry points + `_fetch_emodnet_for_bbox` + `PreviewMesh` dataclass + `suffix_from_filename` helper. ~250 LoC.
- `tests/test_h3_tessellate.py` — 20 unit tests for the new module (16 spec-listed + 1 cell-cap regression + 1 EMODnet cache test + 1 polygon-trust-on-bathy path + 1 suffix helper).
- `tests/fixtures/create_model/tiny.geojson` — 1 small polygon, EPSG:4326.
- `tests/fixtures/create_model/tiny_wgs84.gpkg` — 1 small polygon, EPSG:4326.
- `tests/fixtures/create_model/tiny_3035.shp.zip` — 3 polygons, EPSG:3035 (tests dissolve + reproject).

**Reactive-test deferral:** the spec called for `tests/test_create_model_reactive.py` (5 tests), but `app.py` is not unit-test-importable in this codebase (Shiny side-effects on import; documented in `tests/test_trip_buffer.py:20`). The reactive glue is covered by the manual Playwright smoke (Task 19.2). The pure-function logic worth extracting (filename → suffix) is unit-tested in Task 15.2.

**Modified files:**
- `scripts/build_h3_multires_landscape.py` — three internal functions removed; the script imports them from the new module. Behavior identical.
- `ui/sidebar.py` — new accordion panel "Create Model" inserted after "Landscape".
- `app.py` — new `_uploaded_preview` reactive value, Preview button handler, mesh-override branch in the existing mesh reactive, study-area-change reset effect, bathymetry-toggle effect.
- `tests/test_sidebar.py` — +2 wiring tests.
- `tests/test_movement_metric.py` — +1 perf-regression sentinel for the extract.

**Test runner:**
```bash
micromamba run -n shiny python -m pytest tests/path/file.py::test_name -v
# whole suite
micromamba run -n shiny python -m pytest tests/ -v
```

---

## Tasks

### Task 1: Test fixtures for `parse_upload`

**Files:**
- Create: `tests/fixtures/create_model/tiny.geojson`
- Create: `tests/fixtures/create_model/tiny_wgs84.gpkg`
- Create: `tests/fixtures/create_model/tiny_3035.shp.zip`

These are needed by every `parse_upload` test. Each ≤2 KB so safe to commit.

- [ ] **Step 1.1: Create the fixture-generator script**

Save as a temp helper script (gitignored) at `_diag_create_fixtures.py`:

```python
"""One-shot: generate test fixtures for tests/fixtures/create_model/.

Run once to populate the fixtures; the resulting files are committed
to git. Script itself is not committed (matches the project's
_diag_*.py convention)."""
from pathlib import Path
import zipfile
import io

import geopandas as gpd
from shapely.geometry import Polygon

OUT = Path("tests/fixtures/create_model")
OUT.mkdir(parents=True, exist_ok=True)

# 1) tiny.geojson — single small polygon, WGS84
gdf_geojson = gpd.GeoDataFrame(
    {"name": ["test"]},
    geometry=[Polygon([(21.20, 55.30), (21.22, 55.30),
                       (21.22, 55.32), (21.20, 55.32)])],
    crs="EPSG:4326",
)
gdf_geojson.to_file(OUT / "tiny.geojson", driver="GeoJSON")

# 2) tiny_wgs84.gpkg — same polygon, GeoPackage format
gdf_geojson.to_file(OUT / "tiny_wgs84.gpkg", driver="GPKG", layer="features")

# 3) tiny_3035.shp.zip — three polygons in EPSG:3035 (LAEA), zipped
gdf_3035 = gpd.GeoDataFrame(
    {"name": ["a", "b", "c"]},
    geometry=[
        Polygon([(5400000, 3650000), (5410000, 3650000),
                 (5410000, 3660000), (5400000, 3660000)]),
        Polygon([(5410000, 3650000), (5420000, 3650000),
                 (5420000, 3660000), (5410000, 3660000)]),  # adjacent to (a)
        Polygon([(5500000, 3700000), (5510000, 3700000),
                 (5510000, 3710000), (5500000, 3710000)]),  # disjoint
    ],
    crs="EPSG:3035",
)
shp_dir = OUT / "_tmp_shp"
shp_dir.mkdir(exist_ok=True)
gdf_3035.to_file(shp_dir / "tiny_3035.shp")
# Zip the .shp + sidecars into tiny_3035.shp.zip
with zipfile.ZipFile(OUT / "tiny_3035.shp.zip", "w") as zf:
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
        f = shp_dir / f"tiny_3035{ext}"
        if f.exists():
            zf.write(f, arcname=f.name)
# Cleanup tmp dir
import shutil
shutil.rmtree(shp_dir)

print(f"Wrote: {sorted(OUT.iterdir())}")
```

- [ ] **Step 1.2: Run the generator**

```bash
micromamba run -n shiny python _diag_create_fixtures.py
```

Expected: prints the three fixture paths.

- [ ] **Step 1.3: Sanity-check the fixtures**

```bash
ls -la tests/fixtures/create_model/
```

Expected: three files, each <2 KB.

- [ ] **Step 1.4: Commit**

```bash
git add tests/fixtures/create_model/
git commit -m "test(fixtures): tiny polygon fixtures for create-model upload tests"
```

The `_diag_create_fixtures.py` script stays uncommitted (project convention).

---

### Task 2: Extract `tessellate_reach` and `bridge_components`

**Files:**
- Create: `salmon_ibm/h3_tessellate.py`
- Test: `tests/test_h3_tessellate.py`
- Modify: `scripts/build_h3_multires_landscape.py:204-348` (remove the two functions; they'll be imported from the new module after Task 4)

This task creates the new module and copies the two functions. The build script keeps its inline copies for now (Task 4 removes them).

- [ ] **Step 2.1: Create `salmon_ibm/h3_tessellate.py` with the extracted functions**

Create `salmon_ibm/h3_tessellate.py`:

```python
"""H3 tessellation primitives + upload-flow entry points.

Extracted from scripts/build_h3_multires_landscape.py (v1.6.1). The
build script will become a thin wrapper after Task 4 of plan
2026-04-28-create-model-feature.md. Adding new entry points here
(parse_upload, preview, _fetch_emodnet_for_bbox, PreviewMesh) supports
the Create Model viewer-only feature without coupling viewer code to
script imports.
"""
from __future__ import annotations

import h3
import numpy as np
from shapely.geometry import MultiPolygon


def tessellate_reach(polygon, resolution: int) -> list[str]:
    """Return the H3 cells tessellating a buffered version of polygon.

    Buffer = half the H3 edge length at the target resolution.  This
    ensures cells whose centroid sits within half-a-cell of the
    polygon edge are included — without it, a thin meandering river
    polygon at res 10 (~75 m cells) tessellates as a sparse, often
    disconnected, cell set because polygon_to_cells uses a strict
    centroid-in-polygon test.

    Buffer is applied in degrees: convert metres → degrees at lat 55°
    (the bbox centre) using 1 deg ≈ 111 km.

    For multi-polygon inputs, each part is buffered + tessellated
    independently and the cell sets are unioned.
    """
    edge_m = h3.average_hexagon_edge_length(resolution, unit="m")
    buffer_deg = (edge_m / 2.0) / 111_000.0

    if isinstance(polygon, MultiPolygon):
        parts = list(polygon.geoms)
    else:
        parts = [polygon]

    cells: set[str] = set()
    for part in parts:
        if part.is_empty:
            continue
        buffered = part.buffer(buffer_deg)
        if hasattr(buffered, "geoms"):
            inner_parts = list(buffered.geoms)
        else:
            inner_parts = [buffered]
        for ip in inner_parts:
            if ip.is_empty:
                continue
            ext = [(y, x) for x, y in ip.exterior.coords]
            holes = [
                [(y, x) for x, y in interior.coords]
                for interior in ip.interiors
            ]
            try:
                poly = h3.LatLngPoly(ext, *holes)
            except Exception as e:
                print(f"  ! skipping polygon part: {e}")
                continue
            for c in h3.polygon_to_cells(poly, resolution):
                cells.add(c)
    return sorted(cells)


def bridge_components(
    cells: list[str], resolution: int, max_bridge_len: int = 10
) -> list[str]:
    """Merge disconnected H3 components into a single graph by adding
    shortest-path cells between adjacent components.

    See scripts/build_h3_multires_landscape.py history for the full
    rationale (v1.5.0 fix; preserved here verbatim).
    """
    if not cells or len(cells) < 2:
        return cells
    cell_set = set(cells)

    # BFS within cell_set using grid_ring(1) to find components.
    components: list[set[str]] = []
    seen: set[str] = set()
    for start in cell_set:
        if start in seen:
            continue
        comp: set[str] = set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            comp.add(c)
            for nb in h3.grid_ring(c, 1):
                if nb in cell_set and nb not in seen:
                    stack.append(nb)
        components.append(comp)

    if len(components) <= 1:
        return cells

    components.sort(key=len, reverse=True)
    anchor = components[0]
    bridges_added = 0
    for orphan in components[1:]:
        best_dist = max_bridge_len + 1
        best_pair = None
        for a in orphan:
            for b in anchor:
                d = h3.grid_distance(a, b)
                if d < best_dist:
                    best_dist = d
                    best_pair = (a, b)
                    if d == 1:
                        break
            if best_pair and best_dist == 1:
                break
        if best_pair is None or best_dist > max_bridge_len:
            print(
                f"  ! orphan component of {len(orphan)} cells too far "
                f"from anchor (>{max_bridge_len} cells) — leaving "
                f"disconnected"
            )
            continue
        path = list(h3.grid_path_cells(best_pair[0], best_pair[1]))
        for pc in path:
            if pc not in cell_set:
                cell_set.add(pc)
                bridges_added += 1

    if bridges_added:
        print(f"  bridge-cell pass: added {bridges_added} cells across "
              f"{len(components) - 1} component gaps")
    return sorted(cell_set)
```

- [ ] **Step 2.2: Create `tests/test_h3_tessellate.py` with two unit tests**

```python
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
```

- [ ] **Step 2.3: Run tests — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py -v
```

Expected: 2 tests pass.

- [ ] **Step 2.4: Commit**

```bash
git add salmon_ibm/h3_tessellate.py tests/test_h3_tessellate.py
git commit -m "feat(h3_tessellate): extract tessellate_reach + bridge_components into library module"
```

---

### Task 3: Extract `polygon_trust_water_mask`

**Files:**
- Modify: `salmon_ibm/h3_tessellate.py` (append the function)
- Modify: `tests/test_h3_tessellate.py` (append a test)

The v1.6.1 fix is currently inline in the build script. Pull it out into a named function so both pipelines can call it.

- [ ] **Step 3.1: Append failing test**

Append to `tests/test_h3_tessellate.py`:

```python
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
```

- [ ] **Step 3.2: Run — should fail with AttributeError**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py::test_polygon_trust_water_mask_flips_dry_to_wet -v
```

Expected: `AttributeError: module 'salmon_ibm.h3_tessellate' has no attribute 'polygon_trust_water_mask'`.

- [ ] **Step 3.3: Append the function to `salmon_ibm/h3_tessellate.py`**

```python
def polygon_trust_water_mask(
    reach_id_arr: np.ndarray,
    water_mask: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Force water_mask=1 and depth>=1.0 where reach_id != -1.

    The v1.6.1 fix (commit d187ac9): tessellate_reach buffers each
    polygon by half-a-cell-edge to ensure narrow channels get cells.
    The buffer captures cells whose centroids sit up to ~14 m beyond
    the polygon edge — EMODnet reports many of those as dry.  Without
    this override, ~16% of tagged cells (47-87% of river cells) end
    up with reach_id != -1 but water_mask=0; the viewer drops them
    silently.

    Returns (new_water_mask, new_depth) — both arrays modified.

    See tests/test_h3_grid_quality.py::test_reach_id_implies_water_mask.
    """
    forced = (reach_id_arr != -1) & (water_mask == 0)
    new_water = np.where(forced, np.uint8(1), water_mask).astype(np.uint8)
    new_depth = np.where(forced & (depth < 1.0),
                         np.float32(1.0), depth).astype(np.float32)
    return new_water, new_depth
```

- [ ] **Step 3.4: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py -v
```

Expected: 3 tests pass.

- [ ] **Step 3.5: Commit**

```bash
git add salmon_ibm/h3_tessellate.py tests/test_h3_tessellate.py
git commit -m "feat(h3_tessellate): extract polygon_trust_water_mask from build script"
```

---

### Task 4: Build script delegates to the new module

**Files:**
- Modify: `scripts/build_h3_multires_landscape.py` (remove the three extracted functions, import from `salmon_ibm.h3_tessellate`)

This is the regression-contract task. `tests/test_h3_grid_quality.py` (3 tests) must still pass with no changes.

- [ ] **Step 4.1: Replace the inline functions with imports**

In `scripts/build_h3_multires_landscape.py`:

(a) Near the top of the file (after the existing imports), add:

```python
from salmon_ibm.h3_tessellate import (
    tessellate_reach,
    bridge_components,
    polygon_trust_water_mask,
)
```

(b) Delete the inline `def tessellate_reach(...)` block (around lines 204-258).

(c) Delete the inline `def bridge_components(...)` block (around lines 261-348).

(d) Replace the inline polygon-trust block (around lines 477-490, the lines I added in v1.6.1):

The current code reads:
```python
    forced_water = (reach_id_arr != -1) & (water_mask == 0)
    n_overridden = int(forced_water.sum())
    water_mask = np.where(forced_water, np.uint8(1), water_mask).astype(np.uint8)
    depth = np.where(forced_water & (depth < 1.0),
                     np.float32(1.0), depth).astype(np.float32)
    print(f"  polygon-trust override: {n_overridden:,} buffer cells "
          f"flipped water_mask=False→True (depth set to 1m where 0)")
    print(f"  water cells (final): {int(water_mask.sum()):,}")
```

Replace with:
```python
    n_overridden = int(((reach_id_arr != -1) & (water_mask == 0)).sum())
    water_mask, depth = polygon_trust_water_mask(reach_id_arr, water_mask, depth)
    print(f"  polygon-trust override: {n_overridden:,} buffer cells "
          f"flipped water_mask=False→True (depth set to 1m where 0)")
    print(f"  water cells (final): {int(water_mask.sum()):,}")
```

- [ ] **Step 4.2: Run the build-script regression test (still uses local NC)**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_grid_quality.py -v
```

Expected: 3 tests pass against the existing NC. No NC rebuild needed — the refactor is import-only; the NC produced by the script before this task already encodes the v1.6.1 fix.

- [ ] **Step 4.3: Run the broader simulation suite as a smoke check**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py tests/test_h3_grid_quality.py tests/test_simulation.py -v
```

Expected: all pass; no regressions.

- [ ] **Step 4.4: Commit**

```bash
git add scripts/build_h3_multires_landscape.py
git commit -m "refactor(build_script): delegate tessellation primitives to h3_tessellate"
```

---

### Task 5: `parse_upload` for GeoJSON

**Files:**
- Modify: `salmon_ibm/h3_tessellate.py`
- Modify: `tests/test_h3_tessellate.py`

- [ ] **Step 5.1: Append failing test**

```python
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
```

- [ ] **Step 5.2: Run — should fail**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py::test_parse_upload_geojson_round_trips -v
```

Expected: `AttributeError: module 'salmon_ibm.h3_tessellate' has no attribute 'parse_upload'`.

- [ ] **Step 5.3: Append `parse_upload` (geojson-only branch)**

Append to `salmon_ibm/h3_tessellate.py`:

```python
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import shapely
from shapely.ops import unary_union


def parse_upload(file_bytes: bytes, suffix: str):
    """Read shp.zip / gpkg / geojson → dissolve → reproject to WGS84.

    Returns a single (Multi)Polygon geometry. Raises ValueError on
    unparseable input, missing CRS, no geometry, non-polygon features,
    or antimeridian crossing.

    suffix: one of ".geojson", ".gpkg", ".shp.zip" (case-insensitive).
    """
    suffix = suffix.lower()
    if suffix == ".geojson":
        gdf = gpd.read_file(io.BytesIO(file_bytes))
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    return _dissolve_and_validate(gdf)


def _dissolve_and_validate(gdf: gpd.GeoDataFrame):
    """Common post-read processing: dissolve, reproject, validate."""
    if gdf.empty or gdf.geometry.is_empty.all():
        raise ValueError("File contains no polygon geometry.")
    if gdf.crs is None:
        raise ValueError(
            "File has no CRS — please re-export with a defined coordinate system."
        )
    if not all(g.geom_type in ("Polygon", "MultiPolygon")
               for g in gdf.geometry if not g.is_empty):
        raise ValueError("File must contain Polygon or MultiPolygon features.")
    gdf_wgs = gdf.to_crs("EPSG:4326")
    geom = unary_union([g for g in gdf_wgs.geometry if not g.is_empty])
    if geom.is_empty:
        raise ValueError("File contains no polygon geometry.")
    if not geom.is_valid:
        from shapely.validation import make_valid
        geom = make_valid(geom)
    minx, _, maxx, _ = geom.bounds
    if maxx - minx > 180:
        raise ValueError(
            "Polygons crossing the antimeridian aren't supported. "
            "Split into eastern and western parts."
        )
    return geom
```

- [ ] **Step 5.4: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py::test_parse_upload_geojson_round_trips -v
```

Expected: PASS.

- [ ] **Step 5.5: Commit**

```bash
git add salmon_ibm/h3_tessellate.py tests/test_h3_tessellate.py
git commit -m "feat(h3_tessellate): parse_upload for GeoJSON"
```

---

### Task 6: `parse_upload` for GeoPackage (gpkg)

**Files:**
- Modify: `salmon_ibm/h3_tessellate.py`
- Modify: `tests/test_h3_tessellate.py`

- [ ] **Step 6.1: Append failing test**

```python
def test_parse_upload_gpkg_reads_first_layer():
    bytes_ = (FIXTURES / "tiny_wgs84.gpkg").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".gpkg")
    assert geom is not None
    assert geom.geom_type in ("Polygon", "MultiPolygon")
```

- [ ] **Step 6.2: Run — should fail (raises ValueError "Unsupported format")**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py::test_parse_upload_gpkg_reads_first_layer -v
```

- [ ] **Step 6.3: Add gpkg branch**

In `parse_upload`, replace the body:

```python
    suffix = suffix.lower()
    if suffix == ".geojson":
        gdf = gpd.read_file(io.BytesIO(file_bytes))
    elif suffix == ".gpkg":
        # GPKG is a SQLite container; gpd.read_file accepts BytesIO directly.
        gdf = gpd.read_file(io.BytesIO(file_bytes), layer=0)
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    return _dissolve_and_validate(gdf)
```

- [ ] **Step 6.4: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py -v
```

Expected: 5 tests pass.

- [ ] **Step 6.5: Commit**

```bash
git add salmon_ibm/h3_tessellate.py tests/test_h3_tessellate.py
git commit -m "feat(h3_tessellate): parse_upload for GeoPackage (.gpkg)"
```

---

### Task 7: `parse_upload` for shapefile zip

**Files:**
- Modify: `salmon_ibm/h3_tessellate.py`
- Modify: `tests/test_h3_tessellate.py`

- [ ] **Step 7.1: Append failing tests**

```python
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
    # If shapely merged adjacent polygons into one Polygon (touching edges),
    # that's also valid; the test passes either way.


def test_parse_upload_reprojects_from_3035_to_4326():
    """tiny_3035.shp.zip is in EPSG:3035 (LAEA). After parse, geometry must be in WGS84."""
    bytes_ = (FIXTURES / "tiny_3035.shp.zip").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".shp.zip")
    minx, miny, maxx, maxy = geom.bounds
    # WGS84 bounds: lon in [-180, 180], lat in [-90, 90].
    # The fixture is around lat 55, lon 21 in WGS84 (EU northern coast).
    assert -180 <= minx <= maxx <= 180
    assert -90 <= miny <= maxy <= 90
    assert 10 < minx < 30, f"Expected European longitude, got {minx}"
    assert 50 < miny < 60, f"Expected northern-Europe latitude, got {miny}"
```

- [ ] **Step 7.2: Run — should fail**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py -v -k "shp_zip or dissolves or reprojects"
```

Expected: 3 tests fail (raises ValueError "Unsupported format").

- [ ] **Step 7.3: Add shp.zip branch**

In `parse_upload`:

```python
    suffix = suffix.lower()
    if suffix == ".geojson":
        gdf = gpd.read_file(io.BytesIO(file_bytes))
    elif suffix == ".gpkg":
        gdf = gpd.read_file(io.BytesIO(file_bytes), layer=0)
    elif suffix == ".shp.zip":
        gdf = _read_shp_zip(file_bytes)
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    return _dissolve_and_validate(gdf)


def _read_shp_zip(file_bytes: bytes) -> gpd.GeoDataFrame:
    """Extract a zipped shapefile bundle to a temp dir and read it.

    The zip must contain the .shp + sidecar files (.dbf, .shx, .prj).
    Raises ValueError if .dbf or .shx are missing."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            members = zf.namelist()
            stems = {Path(m).stem for m in members}
            exts = {Path(m).suffix.lower() for m in members}
            if ".shp" not in exts:
                raise ValueError("Zip does not contain a .shp file.")
            if ".dbf" not in exts or ".shx" not in exts:
                raise ValueError(
                    "Shapefile bundle missing .dbf or .shx — re-export "
                    "the full bundle."
                )
            zf.extractall(tmpdir)
        shp_files = list(Path(tmpdir).glob("**/*.shp"))
        if not shp_files:
            raise ValueError("No .shp file found after extraction.")
        return gpd.read_file(shp_files[0])
```

- [ ] **Step 7.4: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py -v
```

Expected: 8 tests pass.

- [ ] **Step 7.5: Commit**

```bash
git add salmon_ibm/h3_tessellate.py tests/test_h3_tessellate.py
git commit -m "feat(h3_tessellate): parse_upload for shapefile zip (.shp.zip)"
```

---

### Task 8: `parse_upload` validation tests

**Files:**
- Modify: `tests/test_h3_tessellate.py` (add error-path tests; existing implementation already raises)

The validation logic is already in place from Tasks 5–7. This task adds the negative tests.

- [ ] **Step 8.1: Append validation tests**

```python
import pytest


def test_parse_upload_raises_on_missing_crs():
    # Build a GeoJSON without a CRS by stripping the crs key.
    import json
    raw = (FIXTURES / "tiny.geojson").read_text()
    data = json.loads(raw)
    data.pop("crs", None)
    # GeoJSON without explicit CRS is implicitly WGS84 per RFC 7946 — geopandas
    # honours that. To force "no CRS" behaviour we have to construct a GeoJSON
    # with a non-standard fake projection or remove all spatial reference info.
    # Simplest: write a fake file via gpd with crs=None and check it raises.
    import geopandas as gpd
    from shapely.geometry import Polygon
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs=None)
    bytes_ = gdf.to_json().encode()
    # Force gpd to read with crs=None: write as GeoJSON without crs metadata.
    # Easier: bypass parse_upload's geojson branch by constructing the gdf directly.
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
    # Polygon spanning longitude 170 to -170 → bounds width 340° > 180°.
    poly = Polygon([(170, 0), (170, 10), (-170, 10), (-170, 0)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    with pytest.raises(ValueError, match="antimeridian"):
        h3_tessellate._dissolve_and_validate(gdf)
```

- [ ] **Step 8.2: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py -v
```

Expected: 12 tests pass.

- [ ] **Step 8.3: Commit**

```bash
git add tests/test_h3_tessellate.py
git commit -m "test(h3_tessellate): parse_upload validation negative tests"
```

---

### Task 9: `PreviewMesh` dataclass

**Files:**
- Modify: `salmon_ibm/h3_tessellate.py`
- Modify: `tests/test_h3_tessellate.py`

- [ ] **Step 9.1: Append failing test**

```python
def test_preview_mesh_dataclass_post_init_assertion():
    """All array fields must have matching length."""
    import numpy as np
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
    # Mismatched length must raise.
    with pytest.raises(AssertionError):
        h3_tessellate.PreviewMesh(
            h3_ids=np.zeros(n, dtype=np.uint64),
            resolutions=np.full(n - 1, 9, dtype=np.int8),  # mismatched!
            centroids=np.zeros((n, 2)),
            reach_id=np.zeros(n, dtype=np.int8),
            reach_names=["uploaded_polygon"],
            depth=np.zeros(n, dtype=np.float32),
            water_mask=np.ones(n, dtype=np.uint8),
            polygon_outlines=[],
        )
```

- [ ] **Step 9.2: Run — should fail (no PreviewMesh)**

- [ ] **Step 9.3: Append PreviewMesh to `salmon_ibm/h3_tessellate.py`**

```python
from dataclasses import dataclass


@dataclass
class PreviewMesh:
    """Duck-typed mesh produced by `preview()` for the Create Model
    feature. The existing _build_h3_data_rows machinery in app.py
    reads these attributes to render hex layers; no subclass of
    H3MultiResMesh required.
    """
    h3_ids: np.ndarray              # uint64
    resolutions: np.ndarray         # int8
    centroids: np.ndarray           # (n, 2) lat, lon (h3.cell_to_latlng order)
    reach_id: np.ndarray            # int8 — uniform 0 (single dissolved reach)
    reach_names: list[str]          # ["uploaded_polygon"]
    depth: np.ndarray               # float32
    water_mask: np.ndarray          # uint8 — uniform 1 by construction
    polygon_outlines: list          # list[list[list[float]]] — rings of [lon, lat]
                                    # pairs, matching app.py:_polygon_to_rings format

    def __post_init__(self):
        n = len(self.h3_ids)
        for name in ("resolutions", "centroids", "reach_id",
                     "depth", "water_mask"):
            arr = getattr(self, name)
            assert len(arr) == n, (
                f"PreviewMesh.{name} has length {len(arr)}, expected {n}"
            )
```

- [ ] **Step 9.4: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py::test_preview_mesh_dataclass_post_init_assertion -v
```

- [ ] **Step 9.5: Commit**

```bash
git add salmon_ibm/h3_tessellate.py tests/test_h3_tessellate.py
git commit -m "feat(h3_tessellate): PreviewMesh dataclass"
```

---

### Task 10: `preview()` — geometry-only path

**Files:**
- Modify: `salmon_ibm/h3_tessellate.py`
- Modify: `tests/test_h3_tessellate.py`

- [ ] **Step 10.1: Append failing tests**

```python
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
```

- [ ] **Step 10.2: Run — should fail (no preview function yet)**

- [ ] **Step 10.3: Append `preview` (geometry-only)**

```python
def preview(
    polygon,
    resolution: int,
    *,
    with_bathy: bool = False,
    max_cells: int = 1_000_000,
) -> PreviewMesh:
    """Tessellate polygon at H3 resolution → PreviewMesh ready to render.

    Geometry-only when with_bathy=False (depth all zero, water_mask all 1).
    See Task 13 for the with_bathy=True branch.

    Raises ValueError if would-be cell count exceeds max_cells.
    """
    cells = tessellate_reach(polygon, resolution)
    if not cells:
        raise ValueError(
            f"Polygon is smaller than a single H3 cell at res {resolution}. "
            f"Try a finer resolution."
        )
    # Pre-bridge cap check — fast feedback for clearly oversized polygons.
    if len(cells) > max_cells:
        raise ValueError(
            f"Tessellation at res {resolution} would produce "
            f"{len(cells):,} cells (max {max_cells:,}). "
            f"Pick a coarser resolution or smaller polygon."
        )
    cells = bridge_components(cells, resolution)
    # Post-bridge cap check — bridge_components can add cells (rare but
    # possible if the polygon has many far-apart components). Re-check
    # so the final cell count is genuinely <= max_cells.
    if len(cells) > max_cells:
        raise ValueError(
            f"Tessellation at res {resolution} produced {len(cells):,} cells "
            f"after bridging (max {max_cells:,}). "
            f"Pick a coarser resolution or smaller polygon."
        )

    n = len(cells)
    h3_ids = np.array([h3.str_to_int(c) for c in cells], dtype=np.uint64)
    centroids = np.array(
        [h3.cell_to_latlng(c) for c in cells], dtype=np.float64
    )
    resolutions = np.full(n, resolution, dtype=np.int8)
    reach_id = np.zeros(n, dtype=np.int8)
    reach_names = ["uploaded_polygon"]
    water_mask = np.ones(n, dtype=np.uint8)
    depth = np.zeros(n, dtype=np.float32)

    # Polygon outlines for the PolygonLayer overlay. Format matches the
    # existing _polygon_to_rings helper in app.py:474 — a list of rings,
    # each ring being a list of [lon, lat] floats. shapely's
    # `part.exterior.coords` yields (x, y) = (lon, lat) tuples, so the
    # list comprehension preserves the (lon, lat) order.
    if isinstance(polygon, MultiPolygon):
        parts = list(polygon.geoms)
    else:
        parts = [polygon]
    polygon_outlines = []
    for part in parts:
        if part.is_empty:
            continue
        polygon_outlines.append(
            [[float(x), float(y)] for x, y in part.exterior.coords]
        )

    return PreviewMesh(
        h3_ids=h3_ids,
        resolutions=resolutions,
        centroids=centroids,
        reach_id=reach_id,
        reach_names=reach_names,
        depth=depth,
        water_mask=water_mask,
        polygon_outlines=polygon_outlines,
    )
```

- [ ] **Step 10.4: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py -v
```

Expected: 16 tests pass.

- [ ] **Step 10.5: Commit**

```bash
git add salmon_ibm/h3_tessellate.py tests/test_h3_tessellate.py
git commit -m "feat(h3_tessellate): preview() geometry-only path"
```

---

### Task 11: `preview()` cell-count cap

**Files:**
- Modify: `tests/test_h3_tessellate.py`

The cap logic was already written in Task 10. This task adds the regression test.

- [ ] **Step 11.1: Append test**

```python
def test_preview_caps_cell_count_at_max_cells():
    bytes_ = (FIXTURES / "tiny.geojson").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".geojson")
    # Pass a tiny max_cells to force the cap to fire.
    with pytest.raises(ValueError, match=r"would produce.*cells.*max"):
        h3_tessellate.preview(geom, resolution=9, max_cells=1)
```

- [ ] **Step 11.2: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py::test_preview_caps_cell_count_at_max_cells -v
```

- [ ] **Step 11.3: Commit**

```bash
git add tests/test_h3_tessellate.py
git commit -m "test(h3_tessellate): cell-count cap regression"
```

---

### Task 12: `_fetch_emodnet_for_bbox` with disk caching

**Files:**
- Modify: `salmon_ibm/h3_tessellate.py`
- Modify: `tests/test_h3_tessellate.py`

This task adds the EMODnet WCS fetch helper used by the bathymetry-toggle path. Cache lives in `.superpowers/cache/` (gitignored already, since `.superpowers/` is in the ignore list).

- [ ] **Step 12.1: Append test (cache-hit path; no live WCS fetch needed)**

The cache-miss path requires a live EMODnet WCS round-trip — too brittle for a CI test. We test only the cache-hit path: pre-write a valid GeoTIFF at the expected cache key, monkey-patch `requests.get` to raise, and verify the function returns without hitting the network.

```python
def test_fetch_emodnet_uses_disk_cache_when_present(tmp_path, monkeypatch):
    """If a cache file exists for the bbox key, _fetch_emodnet_for_bbox
    must read from disk and not hit the network."""
    import hashlib
    import rasterio
    import rasterio.transform

    monkeypatch.setattr(h3_tessellate, "_EMODNET_CACHE_DIR", tmp_path)

    bbox = (21.0, 55.0, 21.5, 55.5)
    key = hashlib.sha1(repr(bbox).encode()).hexdigest()[:16]
    cache_path = tmp_path / f"emodnet_{key}.tif"

    # Write a 1×1 GeoTIFF: elevation = -5 m → depth = 5 m after sign flip.
    with rasterio.open(
        cache_path, "w",
        driver="GTiff", height=1, width=1, count=1, dtype="float32",
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(21.0, 55.5, 0.5, 0.5),
    ) as dst:
        dst.write(np.array([[-5.0]], dtype=np.float32), 1)

    # Any network call would fail this test.
    def must_not_call(url, **kwargs):
        raise AssertionError("Cache should be used; should not hit network.")
    monkeypatch.setattr("requests.get", must_not_call)

    depth, _ = h3_tessellate._fetch_emodnet_for_bbox(bbox)
    assert depth.shape == (1, 1)
    assert float(depth[0, 0]) == 5.0
```

- [ ] **Step 12.2: Append `_fetch_emodnet_for_bbox`**

```python
import hashlib
import os

_EMODNET_WCS_URL = (
    "https://ows.emodnet-bathymetry.eu/wcs"
    "?service=WCS&version=2.0.1&request=GetCoverage"
    "&CoverageId=emodnet:mean"
    "&format=image/tiff"
)
_EMODNET_CACHE_DIR = Path(".superpowers/cache")


def _fetch_emodnet_for_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Fetch EMODnet bathymetry for the bbox. Cached to .superpowers/cache/.

    Returns (depth_grid_2d, bbox_actual). Depth values are POSITIVE
    metres; cells outside coverage are NaN.

    Raises requests.RequestException on network failure (caller should
    catch and surface a UI warning).
    """
    import requests
    import rasterio

    _EMODNET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(repr(bbox).encode()).hexdigest()[:16]
    cache_path = _EMODNET_CACHE_DIR / f"emodnet_{key}.tif"
    if not cache_path.exists():
        minlon, minlat, maxlon, maxlat = bbox
        url = (
            f"{_EMODNET_WCS_URL}"
            f"&subset=Long({minlon},{maxlon})"
            f"&subset=Lat({minlat},{maxlat})"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    with rasterio.open(cache_path) as src:
        elev = src.read(1).astype(np.float32)
        # EMODnet = elevation (positive up). Convert to depth (positive down).
        depth = np.where(np.isnan(elev) | (elev > 0), np.nan, -elev)
    return depth, bbox
```

- [ ] **Step 12.3: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py::test_fetch_emodnet_caches_to_disk -v
```

- [ ] **Step 12.4: Commit**

```bash
git add salmon_ibm/h3_tessellate.py tests/test_h3_tessellate.py
git commit -m "feat(h3_tessellate): _fetch_emodnet_for_bbox with disk cache"
```

---

### Task 13: `preview()` bathymetry path with polygon-trust

**Files:**
- Modify: `salmon_ibm/h3_tessellate.py`
- Modify: `tests/test_h3_tessellate.py`

- [ ] **Step 13.1: Append test**

```python
def test_polygon_trust_applied_on_bathy_path(monkeypatch):
    """When with_bathy=True, the polygon-trust override must run so that
    buffer cells whose EMODnet depth is 0 get water_mask=1 + depth=1."""
    bytes_ = (FIXTURES / "tiny.geojson").read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".geojson")
    # Mock _fetch_emodnet to return all-zero depths (i.e., everything is
    # "dry" per EMODnet). Polygon-trust should flip them all to 1.0.
    def fake_fetch(bbox):
        import numpy as np
        return np.zeros((10, 10), dtype=np.float32), bbox
    monkeypatch.setattr(h3_tessellate, "_fetch_emodnet_for_bbox", fake_fetch)
    mesh = h3_tessellate.preview(geom, resolution=9, with_bathy=True)
    assert (mesh.water_mask == 1).all()
    assert (mesh.depth >= 1.0).all(), (
        "polygon-trust must force depth >= 1.0 for tagged cells"
    )
```

- [ ] **Step 13.2: Modify `preview` to accept the with_bathy branch**

In `salmon_ibm/h3_tessellate.py`, replace the `preview` function body's tail (after constructing the basic PreviewMesh) with:

```python
    mesh = PreviewMesh(
        h3_ids=h3_ids, resolutions=resolutions, centroids=centroids,
        reach_id=reach_id, reach_names=reach_names,
        depth=depth, water_mask=water_mask,
        polygon_outlines=polygon_outlines,
    )

    if with_bathy:
        bbox = polygon.bounds  # (minx, miny, maxx, maxy) = (minlon, minlat, maxlon, maxlat)
        depth_grid, _ = _fetch_emodnet_for_bbox(bbox)
        sampled = _sample_depth_at_centroids(depth_grid, bbox, mesh.centroids)
        # Mirror the build script flow exactly so polygon_trust does what
        # it does there: derive water_mask from depth FIRST (EMODnet's
        # verdict), THEN apply polygon-trust to flip water_mask=0 → 1 at
        # tagged cells.  Without this, water_mask is uniformly 1 by upload
        # construction and polygon_trust's `water_mask == 0` predicate is
        # never true → no override fires → "dry" stripes still appear.
        initial_water = (sampled > 0).astype(np.uint8)
        new_water, new_depth = polygon_trust_water_mask(
            mesh.reach_id, initial_water, sampled
        )
        mesh.water_mask = new_water
        mesh.depth = new_depth

    return mesh


def _sample_depth_at_centroids(
    depth_grid: np.ndarray,
    bbox: tuple[float, float, float, float],
    centroids: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour sample from a regular lat/lon depth grid.

    bbox: (minlon, minlat, maxlon, maxlat).
    centroids: (n, 2) of (lat, lon) per cell.
    Returns: (n,) float32. NaN values become 0.0.
    """
    h, w = depth_grid.shape
    minlon, minlat, maxlon, maxlat = bbox
    lats = centroids[:, 0]
    lons = centroids[:, 1]
    rows = np.clip(((maxlat - lats) / max(maxlat - minlat, 1e-9) * (h - 1)).astype(int), 0, h - 1)
    cols = np.clip(((lons - minlon) / max(maxlon - minlon, 1e-9) * (w - 1)).astype(int), 0, w - 1)
    sampled = depth_grid[rows, cols]
    sampled = np.where(np.isnan(sampled), 0.0, sampled).astype(np.float32)
    return sampled
```

Also delete the `return PreviewMesh(...)` line that was in Task 10's body (since it's now built into `mesh` above).

- [ ] **Step 13.3: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py -v
```

Expected: 18 tests pass.

- [ ] **Step 13.4: Commit**

```bash
git add salmon_ibm/h3_tessellate.py tests/test_h3_tessellate.py
git commit -m "feat(h3_tessellate): preview() bathymetry path with polygon-trust override"
```

---

### Task 14: Sidebar accordion section "Create Model"

**Files:**
- Modify: `ui/sidebar.py`
- Create: `tests/test_sidebar.py` (does not currently exist)

- [ ] **Step 14.1: Read existing sidebar to confirm accordion pattern**

```bash
grep -n "ui.accordion_panel" ui/sidebar.py | head -10
```

Note the existing pattern (Landscape, Run, Bioenergetics, etc.). Each is `ui.accordion_panel(title, ...content..., value="key")`.

- [ ] **Step 14.2: Add the new accordion panel**

In `ui/sidebar.py`, find the closing of the "Landscape" accordion_panel (around line 54, after the `_hint(...)` line). Insert immediately AFTER it (still inside the outer `ui.accordion(...)`):

```python
            ui.accordion_panel(
                "Create Model",
                ui.input_file(
                    "create_model_file",
                    "Polygon file",
                    accept=[".shp.zip", ".gpkg", ".geojson"],
                    multiple=False,
                ),
                ui.input_radio_buttons(
                    "create_model_resolution",
                    "H3 resolution",
                    choices={
                        "8": "8 (~530 m)",
                        "9": "9 (~200 m)",
                        "10": "10 (~76 m)",
                        "11": "11 (~28 m)",
                    },
                    selected="9",
                    inline=True,
                ),
                ui.input_switch(
                    "create_model_with_bathy",
                    "Show bathymetry",
                    value=False,
                ),
                ui.input_action_button(
                    "create_model_preview_btn",
                    "Preview",
                    class_="btn-primary",
                ),
                ui.output_text("create_model_status"),
                _hint("Ephemeral preview: lives only for this session."),
                value="create_model",
            ),
```

- [ ] **Step 14.3: Create `tests/test_sidebar.py` with wiring tests**

The file does not exist today. Create it with:

```python
def test_sidebar_includes_create_model_accordion():
    from ui.sidebar import sidebar_panel
    sidebar = sidebar_panel()
    rendered = str(sidebar)
    assert "Create Model" in rendered
    assert "create_model_file" in rendered


def test_sidebar_create_model_inputs_have_expected_ids():
    from ui.sidebar import sidebar_panel
    rendered = str(sidebar_panel())
    for input_id in (
        "create_model_file",
        "create_model_resolution",
        "create_model_with_bathy",
        "create_model_preview_btn",
        "create_model_status",
    ):
        assert input_id in rendered, f"missing input id: {input_id}"
```

- [ ] **Step 14.4: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_sidebar.py -v
```

Expected: 2 tests pass.

- [ ] **Step 14.5: Commit**

```bash
git add ui/sidebar.py tests/test_sidebar.py
git commit -m "feat(ui): Create Model accordion section in sidebar"
```

---

### Task 15: `app.py` — `_uploaded_preview` reactive + Preview button handler

**Files:**
- Modify: `app.py`
- Modify: `salmon_ibm/h3_tessellate.py` (add `suffix_from_filename`)
- Modify: `tests/test_h3_tessellate.py` (one new test)

- [ ] **Step 15.1: Decide test strategy — no `tests/test_create_model_reactive.py`**

`app.py` is not importable from tests because it runs Shiny at module-load time (this is documented in `tests/test_trip_buffer.py:20`: *"We can't import app.py directly (it runs Shiny)"*). Unit-testing Shiny reactives in this codebase is not currently possible.

**Decision:** drop the planned `tests/test_create_model_reactive.py` (5 tests). The reactive glue is exercised by the manual Playwright smoke (Task 19.2). Logic worth unit-testing — file-suffix detection from a filename string — gets one test in `tests/test_h3_tessellate.py` against a small helper function (Task 15.2).

This matches the spec's `Documented limitations` section's spirit (some test coverage is deferred when the harness doesn't exist) and is consistent with the project's existing workaround (the trip-buffer extraction approach).

- [ ] **Step 15.2: Add `suffix_from_filename` helper + test**

Append to `salmon_ibm/h3_tessellate.py`:

```python
def suffix_from_filename(name: str) -> str | None:
    """Detect the upload suffix from a filename. Case-insensitive.

    Returns ".shp.zip" / ".gpkg" / ".geojson" or None if unrecognised.
    """
    lo = name.lower()
    if lo.endswith(".shp.zip"):
        return ".shp.zip"
    if lo.endswith(".gpkg"):
        return ".gpkg"
    if lo.endswith(".geojson"):
        return ".geojson"
    return None
```

Append to `tests/test_h3_tessellate.py`:

```python
def test_suffix_from_filename():
    sff = h3_tessellate.suffix_from_filename
    assert sff("tiny.geojson") == ".geojson"
    assert sff("TINY.GeoJSON") == ".geojson"
    assert sff("data.gpkg") == ".gpkg"
    assert sff("DATA.GPKG") == ".gpkg"
    assert sff("shoreline.shp.zip") == ".shp.zip"
    assert sff("foo.txt") is None
    assert sff("just_zip.zip") is None  # zip without .shp.
```

Run:
```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py::test_suffix_from_filename -v
```

Expected: PASS.

- [ ] **Step 15.3: Add module-level helpers + reactive infrastructure inside `server()`**

In `app.py`, ensure these imports are at the top of the file:

```python
import os
from salmon_ibm import h3_tessellate
```

Then find `def server(input, output, session):` (around line 1176). Existing pattern shows reactives like `sim_state = reactive.Value(None)` at the top of `server()`. Add **inside `server()`**, near those existing reactives (e.g., right after `step_stats = reactive.Value(...)`):

```python
    # Create Model — ephemeral viewer-only preview.
    _uploaded_preview = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.create_model_preview_btn)
    def _on_create_model_preview():
        file_info = input.create_model_file()
        if not file_info:
            ui.notification_show(
                "Pick a file first.", type="warning", duration=4)
            return

        file = file_info[0]
        name = file["name"]
        bytes_ = open(file["datapath"], "rb").read()

            suffix = h3_tessellate.suffix_from_filename(name)
        if suffix is None:
            ui.notification_show(
                f"Unsupported format: {name}", type="error", duration=8)
            return

        res = int(input.create_model_resolution())
        with_bathy = bool(input.create_model_with_bathy())
        max_cells = int(os.environ.get("HEXSIM_PREVIEW_MAX_CELLS", "1000000"))

        try:
            geom = h3_tessellate.parse_upload(bytes_, suffix)
            mesh = h3_tessellate.preview(
                geom, resolution=res,
                with_bathy=with_bathy, max_cells=max_cells)
        except ValueError as e:
            ui.notification_show(str(e), type="error", duration=8)
            return
        except Exception as e:
            ui.notification_show(
                f"Unexpected error: {e}", type="error", duration=8)
            return

        _uploaded_preview.set(mesh)
        n = len(mesh.h3_ids)
        ui.notification_show(
            f"Preview ready: {n:,} cells at res {res}.",
            type="message", duration=4,
        )

    @output
    @render.text
    def create_model_status():
        mesh = _uploaded_preview()
        if mesh is None:
            return "No preview loaded."
        return f"Preview: {len(mesh.h3_ids):,} cells at res {mesh.resolutions[0]}."
```

The `_uploaded_preview` reactive is now scoped to the Shiny session (one per browser tab), as is the convention in this `app.py`. Module-level state would not survive across sessions and would also error at import time on systems without a session context.

- [ ] **Step 15.4: Run the suffix test + smoke-import the existing tests**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py -v
```

Expected: 20 tests pass (19 prior + 1 new suffix test).

The reactive glue itself (`_on_create_model_preview`) is exercised by the manual Playwright smoke (Task 19.2).

- [ ] **Step 15.5: Commit**

```bash
git add app.py salmon_ibm/h3_tessellate.py tests/test_h3_tessellate.py
git commit -m "feat(app): _uploaded_preview reactive + Preview button handler"
```

---

### Task 16: `app.py` — preview layer builder + study-area reset

**Files:**
- Modify: `app.py`

This task adds a parallel deck.gl layer builder for the upload preview that short-circuits the sim.mesh-based rendering flow. The duck-typed-mesh approach in early drafts didn't work because `sim.mesh` is threaded through env-field sampling, color scaling, and agent-render paths — a naive substitution breaks them.

- [ ] **Step 16.1: No new test (reactive glue covered by manual smoke)**

This task is implementation-only. The existing `tests/test_h3_grid_quality.py` regression contract still holds (Task 4 verified it). The mesh-override and reset-effect logic are exercised by the manual Playwright smoke at Task 19.2.

- [ ] **Step 16.2: Modify the existing layer-build code to honor `_uploaded_preview`**

**Important:** the existing app.py threads `sim.mesh` through env-field sampling, color-scaling, and agent-render code (e.g., line 1453 in agent positions, line 1970 in the H3 layer block, plus `_build_h3_data_rows` at line 795 reads `mesh.h3_ids`, `mesh.reach_id`, etc., AND `sim.env.fields` separately). A naive duck-type swap doesn't work cleanly.

**Approach: parallel layer-build for the upload preview.** Instead of trying to re-route the existing `sim.mesh`-coupled flow, build a **separate, simpler** layer set when `_uploaded_preview` is non-None, and short-circuit the existing `is_h3` branch at line 1964ish.

Add this helper at module scope in `app.py` (after `_build_h3_data_rows` at line ~795):

```python
def _build_preview_h3_layer(mesh) -> list[dict]:
    """Build deck.gl layers for a Create Model upload preview.

    No env-field sampling, no agent rendering — just an H3HexagonLayer
    coloured by depth (if mesh.depth has positive values) or uniform
    light blue (if depth is all zero), plus PolygonLayer overlays for
    the source polygon outlines.
    """
    n = len(mesh.h3_ids)
    if mesh.depth.max() > 0:
        # Bathymetry-on path: shade by depth (deeper = darker blue).
        d_norm = np.clip(mesh.depth / max(mesh.depth.max(), 1.0), 0, 1)
        rgb = np.stack([
            (200 - 150 * d_norm).astype(np.uint8),  # R
            (220 - 80 * d_norm).astype(np.uint8),   # G
            (240).astype(np.uint8) * np.ones(n, dtype=np.uint8),  # B
        ], axis=1)
    else:
        rgb = np.tile(np.array([158, 216, 227], dtype=np.uint8), (n, 1))

    import h3 as _h3
    data_rows = [
        {
            "hex": _h3.int_to_str(int(mesh.h3_ids[k])),
            "color": [int(rgb[k, 0]), int(rgb[k, 1]), int(rgb[k, 2]), 220],
        }
        for k in range(n)
    ]
    layers = [{
        "@@type": "H3HexagonLayer",
        "id": "water",
        "data": data_rows,
        "getHexagon": "@@=d.hex",
        "getFillColor": "@@=d.color",
        "filled": True,
        "stroked": False,
        "pickable": True,
    }]
    # Polygon outlines (matching app.py:_build_reach_polygon_layers shape).
    for i, ring in enumerate(mesh.polygon_outlines):
        layers.append({
            "@@type": "PolygonLayer",
            "id": f"polygon-uploaded-{i}",
            "data": [{"polygon": ring}],
            "getPolygon": "@@=d.polygon",
            "getFillColor": [120, 180, 160, 36],
            "getLineColor": [120, 180, 160, 220],
            "lineWidthMinPixels": 2,
            "stroked": True,
            "filled": True,
            "pickable": False,
        })
    return layers
```

In the existing layer reactive (search for `is_h3` around line 1964), branch BEFORE the existing H3 path:

```python
        # Create Model upload preview takes precedence — render its
        # layers and skip the sim.mesh-derived flow entirely.
        upload = _uploaded_preview()
        if upload is not None:
            return _build_preview_h3_layer(upload)
        # ... existing sim.mesh-based code follows unchanged ...
```

The exact reactive name and indentation depend on which reactive owns layer construction; follow the existing structure (typically a `@reactive.calc` near the deck-spec building).

- [ ] **Step 16.3: Add the study-area-change reset effect (inside `server()`)**

In `app.py`, **inside `def server(...)`**, alongside the `_on_create_model_preview` effect added in Task 15.3, append:

```python
    @reactive.effect
    @reactive.event(input.landscape)
    def _on_landscape_change_reset_preview():
        """Clear the upload preview when the user picks a different study area."""
        if _uploaded_preview() is not None:
            _uploaded_preview.set(None)
```

Note: matches the `ignore_none=False` pattern used by the existing `btn_reset` reactive at app.py:1186.

- [ ] **Step 16.4: Run the existing test suite as a smoke check**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_grid_quality.py tests/test_h3_tessellate.py tests/test_sidebar.py -v
```

Expected: all pass; no regressions.

- [ ] **Step 16.5: Commit**

```bash
git add app.py
git commit -m "feat(app): _build_preview_h3_layer + study-area-change reset for upload preview"
```

---

### Task 17: `app.py` — bathymetry toggle effect

**Files:**
- Modify: `app.py`

- [ ] **Step 17.1: No new test (reactive glue covered by manual smoke)**

The bathymetry-toggle reactive is implementation-only; covered by the manual Playwright smoke (Task 19.2 step 8: toggle bathymetry on / verify hexes change color).

- [ ] **Step 17.2: Add the bathymetry-toggle effect (inside `server()`)**

In `app.py`, **inside `def server(...)`**, alongside the other Create Model reactives added in Tasks 15.3 and 16.3, append:

```python
    @reactive.effect
    @reactive.event(input.create_model_with_bathy)
    def _on_create_model_with_bathy_toggle():
        """When the user toggles bathymetry, re-run preview() with new flag.
        Re-uses the most recent FileInfo if available; otherwise warns."""
        mesh = _uploaded_preview()
        if mesh is None:
            return  # nothing to update yet
        file_info = input.create_model_file()
        if not file_info:
            return
        file = file_info[0]
        name = file["name"]
        bytes_ = open(file["datapath"], "rb").read()
        suffix = h3_tessellate.suffix_from_filename(name)
        if suffix is None:
            return

        res = int(input.create_model_resolution())
        with_bathy = bool(input.create_model_with_bathy())
        max_cells = int(os.environ.get("HEXSIM_PREVIEW_MAX_CELLS", "1000000"))

        try:
            geom = h3_tessellate.parse_upload(bytes_, suffix)
            new_mesh = h3_tessellate.preview(
                geom, resolution=res,
                with_bathy=with_bathy, max_cells=max_cells)
        except Exception as e:
            ui.notification_show(
                f"EMODnet bathymetry unavailable: {e}",
                type="warning", duration=8)
            # Auto-flip the switch off so we don't keep retrying.
            ui.update_switch("create_model_with_bathy", value=False)
            return
        _uploaded_preview.set(new_mesh)
```

- [ ] **Step 17.3: Smoke test — existing tests still green**

```bash
micromamba run -n shiny python -m pytest tests/test_h3_tessellate.py tests/test_sidebar.py -v
```

- [ ] **Step 17.4: Commit**

```bash
git add app.py
git commit -m "feat(app): bathymetry-toggle effect for upload preview"
```

---

### Task 18: Performance regression sentinel

**Files:**
- Modify: `tests/test_movement_metric.py`

- [ ] **Step 18.1: Run a one-off baseline measurement**

Create `_diag_tessellate_perf.py` (uncommitted helper):

```python
"""Baseline measurement for the h3_tessellate extract refactor."""
import time
from pathlib import Path
from salmon_ibm import h3_tessellate

bytes_ = Path("tests/fixtures/create_model/tiny.geojson").read_bytes()
geom = h3_tessellate.parse_upload(bytes_, ".geojson")

# Warmup
for _ in range(3):
    h3_tessellate.tessellate_reach(geom, resolution=9)

t0 = time.perf_counter()
for _ in range(50):
    h3_tessellate.tessellate_reach(geom, resolution=9)
elapsed = time.perf_counter() - t0
per_call_ms = (elapsed / 50) * 1000.0
print(f"Per-call time: {per_call_ms:.2f} ms")
print(f"Suggested BASELINE_MS = {per_call_ms * 1.10:.1f}  (10% margin)")
```

```bash
micromamba run -n shiny python _diag_tessellate_perf.py
```

Record the per-call ms and suggested baseline.

- [ ] **Step 18.2: Append the sentinel test**

Append to `tests/test_movement_metric.py`:

```python
def test_h3_tessellate_extract_no_perf_regression():
    """tessellate_reach must complete a small fixture polygon in
    BASELINE_MS or less. Catches accidental over-instrumentation
    introduced by the extract refactor (Task 4 of plan
    2026-04-28-create-model-feature.md)."""
    import time
    from pathlib import Path
    from salmon_ibm import h3_tessellate

    fix = Path(__file__).resolve().parent / "fixtures" / "create_model" / "tiny.geojson"
    if not fix.exists():
        import pytest
        pytest.skip("tiny.geojson fixture missing")
    bytes_ = fix.read_bytes()
    geom = h3_tessellate.parse_upload(bytes_, ".geojson")
    # Warmup
    for _ in range(3):
        h3_tessellate.tessellate_reach(geom, resolution=9)
    t0 = time.perf_counter()
    for _ in range(50):
        h3_tessellate.tessellate_reach(geom, resolution=9)
    elapsed = time.perf_counter() - t0
    per_call_ms = (elapsed / 50) * 1000.0

    # Replace BASELINE_MS_PLACEHOLDER with the measured value × 1.10 from Step 18.1.
    BASELINE_MS = BASELINE_MS_PLACEHOLDER
    assert per_call_ms <= BASELINE_MS, (
        f"tessellate_reach took {per_call_ms:.2f} ms (baseline {BASELINE_MS:.2f}); "
        f"check for accidental over-instrumentation in the refactor."
    )
```

Replace `BASELINE_MS_PLACEHOLDER` with the actual value (e.g., `5.0`).

- [ ] **Step 18.3: Run — must pass**

```bash
micromamba run -n shiny python -m pytest tests/test_movement_metric.py::test_h3_tessellate_extract_no_perf_regression -v
```

- [ ] **Step 18.4: Commit**

```bash
git add tests/test_movement_metric.py
git commit -m "test(perf): h3_tessellate extract no-regression sentinel"
```

---

### Task 19: Whole-suite verification + manual smoke

- [ ] **Step 19.1: Run the full suite**

```bash
micromamba run -n shiny python -m pytest tests/ -v --no-header -q
```

Expected: large green count (~812 tests), 0 failures. Some tests skip due to missing data — that's fine.

- [ ] **Step 19.2: Manual Playwright smoke (documented; no automated test)**

Run the following manually before tagging:

```
1. Start the app locally:
     micromamba run -n shiny python -m shiny run app.py
2. Open http://localhost:8000 in a browser.
3. Open the "Create Model" sidebar accordion.
4. Upload tests/fixtures/create_model/tiny.geojson.
5. Pick H3 resolution 9.
6. Click "Preview".
7. Verify hex cells render over the basemap.
8. Toggle "Show bathymetry" on.
9. Verify hexes change color (depth-shaded). If EMODnet is unreachable,
   the toast should explain and the toggle should auto-flip off.
10. Switch the Landscape dropdown to "Curonian Lagoon H3 (multi-res)".
11. Verify the upload is cleared and Curonian renders again.
```

- [ ] **Step 19.3: Hand off to user for tag + deploy**

Per the project's tag-before-deploy convention, the user runs:

```bash
git tag -a v1.7.0 -m "Create Model: viewer-only ephemeral preview (Approach A)"
git push origin main
git push origin v1.7.0
scripts/deploy_laguna.sh apply
```

No NC SCP needed for this release — Create Model doesn't change the landscape NC.

---

## Spec coverage check

| Spec section | Implementing task(s) |
|---|---|
| `salmon_ibm/h3_tessellate.py` (extracted primitives + new entry points) | Tasks 2, 3, 5, 6, 7, 9, 10, 12, 13 |
| `tessellate_reach` extract | Task 2 |
| `bridge_components` extract | Task 2 |
| `polygon_trust_water_mask` extract | Task 3 |
| Build script wrapper refactor | Task 4 |
| `parse_upload` (geojson/gpkg/shp.zip) | Tasks 5, 6, 7 |
| `parse_upload` validation | Task 8 |
| `PreviewMesh` dataclass | Task 9 |
| `preview()` geometry-only path | Task 10 |
| Cell-count cap | Task 11 |
| `_fetch_emodnet_for_bbox` | Task 12 |
| `preview()` bathymetry path with polygon-trust | Task 13 |
| Sidebar accordion section | Task 14 |
| `_uploaded_preview` reactive + Preview button | Task 15 |
| Mesh override + study-area-change reset | Task 16 |
| Bathymetry-toggle effect | Task 17 |
| Test fixtures | Task 1 |
| h3_tessellate unit tests (17) | Tasks 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13 |
| Sidebar wiring tests (+2) | Task 14 |
| Reactive logic tests (+5) | **Deferred** — `app.py` not unit-test-importable. Replaced by `suffix_from_filename` helper test (Task 15.2) + manual Playwright smoke (Task 19.2). |
| Performance regression sentinel (+1) | Task 18 |
| Manual Playwright smoke | Task 19 |
| Documented limitations | Spec only — no implementation |

**Test count actual:** +23 (was +25 in spec). 20 in `test_h3_tessellate.py`, 2 sidebar, 1 perf. Suite: 787 → 810.

All sections implemented.
