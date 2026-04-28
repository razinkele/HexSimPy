# Create Model Feature — Design Spec

> **STATUS: DRAFT 2026-04-28.** Awaiting user review before handoff to
> `superpowers:writing-plans`. No code changes yet.

A new sidebar accordion section ("Create Model") that lets users upload
`shp.zip` / `gpkg` / `geojson` polygon files and preview them as an H3
hexagonal tessellation overlaid on the map. **Viewer-only, ephemeral** —
no persistence, no simulation, no NetCDF generation. The smallest unit
that delivers "see what my polygons would look like at H3 resolution N"
without committing to a full landscape build.

## Purpose

The deployed Curonian salmon IBM (v1.6.1 at <http://laguna.ku.lt/HexSimPy/>)
ships four hardcoded study areas (Curonian Lagoon H3 multi-res, Curonian
Lagoon H3 single-res, Curonian TriMesh, Columbia River). Researchers
working on different geographies (e.g., a Norwegian fjord, a Spanish
estuary, a custom Curonian sub-region) currently have to:

1. Hand-edit polygon shapefiles
2. Run `scripts/build_h3_multires_landscape.py` from the CLI
3. Edit `configs/*.yaml` to register the new landscape
4. Restart the app to see results

That's a 30-minute round trip just to answer "would my polygons
tessellate sensibly at H3 res 10?". The Create Model feature collapses
that to a 5-second upload-and-preview cycle.

This spec is the **viewer-only first slice (Approach A)** of a planned
graduated rollout (decided 2026-04-28):

* **Approach A (this plan):** ephemeral preview, no persistence, no
  simulation. ~1 week.
* **Approach D (future plan):** auto-fetch EMODnet bathymetry +
  Curonian-default forcing, build a runnable NC, register as a study
  area. ~2 weeks.
* **Approach B (future plan):** full multi-source data ingestion with
  user-supplied bathymetry/forcing, multi-feature reach support. ~3-4
  weeks.

## Scope decisions reached during brainstorming (2026-04-28)

| # | Decision | Rationale |
|---|---|---|
| 1 | Approach A: viewer-only, ephemeral. Graduate to D, then B as separate plans. | Smallest unit with real value. Each follow-up has its own spec. |
| 2 | No persistence — preview lives only for the current Shiny session. | Avoids storage / multi-user / cleanup concerns until they're worth it. |
| 3 | New **sidebar accordion section** "Create Model", below "Landscape". | Matches existing pattern (Landscape / Run / Bioenergetics / Osmoregulation / Dissolved Oxygen / Seiches accordions). Smallest UI surface. |
| 4 | **Single dissolved area** — multi-feature files merged via `unary_union()` before tessellation. | Avoids attribute-column picker UI. Multi-color per-feature treatment is a follow-up plan. |
| 5 | **Geometry + polygon outline only**, with a "show bathymetry" toggle (default off). | Pure-geometry preview answers the user's stated need. Bathymetry adds an EMODnet WCS round-trip (~3-10s) on demand. |
| 6 | **User-pickable H3 resolution {8/9/10/11} + cell-count cap (default 1M)**. | Different polygons want different resolutions. Cap prevents browser-killing previews; configurable via `HEXSIM_PREVIEW_MAX_CELLS`. |
| 7 | **Implementation 1: extract + reuse.** `tessellate_reach`, `bridge_components`, `polygon_trust_water_mask` move from `scripts/build_h3_multires_landscape.py` into a new library module. | The v1.6.1 polygon-trust fix automatically applies to both pipelines. Build script becomes a thin wrapper; no behavior change. |

## Architecture

Three layers, one new library module, one new sidebar section.

```
┌────────────────────────────────────────────────────────────┐
│  ui/sidebar.py                                              │
│  NEW accordion section "Create Model":                      │
│   ├── ui.input_file (.shp.zip / .gpkg / .geojson)           │
│   ├── ui.input_radio_buttons (H3 res: 8 / 9 / 10 / 11)      │
│   ├── ui.input_switch (show bathymetry, default off)        │
│   ├── ui.input_action_button ("Preview")                    │
│   └── ui.output_text("create_model_status")                 │
└─────────────────────────┬───────────────────────────────────┘
                          │ FileInfo + resolution + bathy flag
                          ▼
┌────────────────────────────────────────────────────────────┐
│  app.py                                                     │
│  NEW reactive value `_uploaded_preview` (None | PreviewMesh)│
│  When "Preview" clicked: parse_upload → preview → set       │
│  Cell-cap exceeded → ui.notification_show toast             │
│  When _uploaded_preview is set: mesh reactive returns it    │
│  When user changes study area dropdown: _uploaded_preview=None│
└─────────────────────────┬───────────────────────────────────┘
                          │ delegates to
                          ▼
┌────────────────────────────────────────────────────────────┐
│  salmon_ibm/h3_tessellate.py  (NEW — extracted library)     │
│  Pure functions, no Shiny / I/O coupling:                   │
│   • tessellate_reach(polygon, res)        ← extracted from  │
│   • bridge_components(cells, res)         ← build script    │
│   • polygon_trust_water_mask(...)         ← v1.6.1 logic    │
│  PLUS new entry-points:                                     │
│   • parse_upload(bytes, suffix) → GeoSeries (WGS84)         │
│   • preview(polygon, res, with_bathy, max_cells) → PreviewMesh│
│   • _fetch_emodnet_for_bbox(bbox)  (cached on disk)         │
│  scripts/build_h3_multires_landscape.py becomes a thin      │
│   wrapper over this module (no behavior change).            │
└────────────────────────────────────────────────────────────┘
```

**Three architectural calls:**

1. **Extract-first refactor.** `tessellate_reach`, `bridge_components`,
   and the v1.6.1 `polygon_trust_water_mask` logic move from
   `scripts/build_h3_multires_landscape.py` into
   `salmon_ibm/h3_tessellate.py`. The build script imports from there;
   the new upload feature imports from there. Existing
   `tests/test_h3_grid_quality.py` (3 tests) is the regression contract:
   no behavior change for landscape builds.

2. **PreviewMesh = duck-typed mesh.** A `@dataclass` with the four
   arrays the existing `_build_h3_data_rows` machinery needs, plus a
   polygon-outline list. No subclass of `H3MultiResMesh` — overkill for
   the dataclass-of-arrays this needs to be.

3. **No persistence layer.** The `_uploaded_preview` reactive lives only
   in the Shiny session. Future Approach B (persistent study areas)
   adds storage / CRUD; not in scope here.

## Components

### 1. `salmon_ibm/h3_tessellate.py` (NEW, ~250 LoC)

Three categories of functions.

**Tessellation primitives** (moved verbatim from
`scripts/build_h3_multires_landscape.py`, no behavior change):

```python
def tessellate_reach(polygon, resolution: int) -> list[str]: ...
def bridge_components(cells, resolution, max_bridge_len: int = 10) -> list[str]: ...
def polygon_trust_water_mask(reach_id_arr, water_mask, depth) -> tuple[ndarray, ndarray]:
    """Force water_mask=True and depth=max(depth, 1.0) for cells with reach_id != -1.
    The v1.6.1 fix; extracted here so both the build pipeline and the
    bathymetry-toggle preview path use the same logic.
    """
```

**Upload-flow entry points** (new):

```python
def parse_upload(file_bytes: bytes, suffix: str) -> shapely.Geometry:
    """Read shp.zip / gpkg / geojson → GeoDataFrame → dissolve →
    reproject to EPSG:4326. Raises ValueError on unparseable input,
    missing CRS, no geometry, non-polygon features, or antimeridian
    crossing. Returns a single (Multi)Polygon geometry.
    """

@dataclass
class PreviewMesh:
    h3_ids: np.ndarray              # uint64
    resolutions: np.ndarray         # int8 (uniform for now; future-proof)
    centroids: np.ndarray           # (n, 2) lat/lon
    reach_id: np.ndarray            # int8 — uniform 0 (single dissolved reach)
    reach_names: list[str]          # ["uploaded_polygon"]
    depth: np.ndarray               # float32 (zeros if with_bathy=False)
    water_mask: np.ndarray          # uint8 (uniform 1 — see runtime invariants)
    polygon_outlines: list[list[tuple[float, float]]]  # for the PolygonLayer

    def __post_init__(self):
        n = len(self.h3_ids)
        assert all(len(arr) == n for arr in [self.resolutions, self.centroids,
                   self.reach_id, self.depth, self.water_mask]), (
            "PreviewMesh array lengths must match")

def preview(
    polygon, resolution: int, *,
    with_bathy: bool = False,
    max_cells: int = 1_000_000,
) -> PreviewMesh:
    """Tessellate + bridge + (optionally) sample bathymetry.

    Raises ValueError if the would-be cell count exceeds max_cells.
    Caller catches and translates to a user-facing toast.
    """
```

**Bathymetry sub-helper** (new, only used when toggle is on):

```python
def _fetch_emodnet_for_bbox(bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Fetch EMODnet bathymetry GeoTIFF for the bbox via WCS; return
    depth grid. Cached to .superpowers/cache/emodnet_<hash>.tif so
    re-toggling doesn't re-fetch. Reuses the WCS endpoint pattern from
    data/_water_polygons.py.
    """
```

### 2. `ui/sidebar.py` — NEW accordion section

Inserted immediately after the existing `"Landscape"` accordion. Pattern
matches the other accordion sections:

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
        choices={"8": "8 (~530 m)", "9": "9 (~200 m)",
                 "10": "10 (~76 m)", "11": "11 (~28 m)"},
        selected="9",
        inline=True,
    ),
    ui.input_switch("create_model_with_bathy", "Show bathymetry", value=False),
    ui.input_action_button(
        "create_model_preview_btn", "Preview", class_="btn-primary"),
    ui.output_text("create_model_status"),
    value="create_model",
),
```

The `_status` text shows progress messages ("Reading file…", "Tessellating
at res 9…", "Preview ready: 12,341 cells") and error messages ("Cap
exceeded: 4.2M cells > 1M; pick a coarser resolution.").

### 3. `app.py` — reactives + integration (~+50 LoC)

Three new pieces near the existing study-area reactive (~line 1330):

(a) `_uploaded_preview = reactive.Value(None)` — holds the current
`PreviewMesh` or None.

(b) `@reactive.effect @reactive.event(input.create_model_preview_btn)`
— when the user clicks Preview:
  * read the FileInfo (path, name, size, type)
  * call `h3_tessellate.parse_upload(bytes, suffix)`
  * call `h3_tessellate.preview(geom, res, with_bathy, max_cells=cap)`
  * catch `ValueError` (cap exceeded, parse error) → show toast via
    `ui.notification_show(message, type="error", duration=8)`
  * on success: `_uploaded_preview.set(mesh)` and update the status text

(c) Modify the existing `mesh` reactive: when `_uploaded_preview()` is
non-None, return it; otherwise fall through to the existing
study-area-mesh path. The H3HexagonLayer + PolygonLayer rebuild
reactives already depend on `mesh` and pick up the change automatically.

A separate `@reactive.effect` on the study-area dropdown clears
`_uploaded_preview` when the user picks a different study area.

### 4. Bathymetry toggle path (new reactive in `app.py`)

```python
@reactive.effect
@reactive.event(input.create_model_with_bathy)
def _on_bathy_toggle():
    mesh = _uploaded_preview.get()
    if mesh is None:
        return  # no preview to update
    if input.create_model_with_bathy():
        bbox = compute_bbox(mesh.polygon_outlines)
        try:
            depth_grid = h3_tessellate._fetch_emodnet_for_bbox(bbox)
        except (requests.RequestException, OSError):
            ui.notification_show(
                "EMODnet bathymetry unavailable. Toggle disabled.",
                type="warning",
            )
            ui.update_switch("create_model_with_bathy", value=False)
            return
        mesh.depth = sample_depth(depth_grid, mesh.centroids)
    else:
        mesh.depth = np.zeros_like(mesh.depth)
    _uploaded_preview.set(mesh)  # triggers re-render
```

### 5. `scripts/build_h3_multires_landscape.py` — refactor only

Three internal functions move to `salmon_ibm/h3_tessellate.py`. The
build script imports them and otherwise keeps its top-to-bottom flow
unchanged. Existing `tests/test_h3_grid_quality.py` continues to pass
without modification — that's the regression contract.

## Data flow

### Upload → Preview (happy path)

```
User clicks Preview button
        │
        ▼
@reactive.event reads FileInfo + resolution + with_bathy
        │
        ▼
status_text = "Reading file…"
        │
        ▼
geom = h3_tessellate.parse_upload(file_bytes, suffix)
        │   • dispatch on suffix (.shp.zip / .gpkg / .geojson)
        │   • dissolve all features via unary_union
        │   • reproject to EPSG:4326
        │   • make_valid + antimeridian-cross check
        ▼
status_text = "Tessellating at res N…"
        │
        ▼
mesh = h3_tessellate.preview(geom, resolution, with_bathy, max_cells)
        │   • tessellate_reach (buffered)
        │   • cell-count cap check → ValueError if exceeded
        │   • bridge_components
        │   • build h3_ids / centroids / uniform reach_id / water_mask=1 / depth
        │   • optional: _fetch_emodnet_for_bbox + interpolate
        │   • polygon_outlines from geom (exterior + interiors)
        │   • return PreviewMesh
        ▼
status_text = "Preview ready: 12,341 cells from 3 features"
        │
        ▼
_uploaded_preview.set(mesh)
        │
        ▼
mesh reactive returns PreviewMesh; H3HexagonLayer + PolygonLayer rebuild;
map renders the new tessellation. Existing study-area dropdown stays
where it was; the displayed mesh is the upload.
```

### Bathymetry-toggle ON path

```
With _uploaded_preview already set, user toggles input.create_model_with_bathy = True.
        │
        ▼
@reactive.effect on the toggle:
  • bbox = compute_bbox(mesh.polygon_outlines)
  • depth_grid = h3_tessellate._fetch_emodnet_for_bbox(bbox)
  •   (cached at .superpowers/cache/emodnet_<hash>.tif)
  • sampled_depth = sample_depth(depth_grid, mesh.centroids)
  • mesh.water_mask, mesh.depth = h3_tessellate.polygon_trust_water_mask(
        mesh.reach_id, mesh.water_mask, sampled_depth)
  •   (Applies the v1.6.1 fix to the upload preview path: buffer cells
  •    beyond the polygon edge that EMODnet reports as dry get
  •    water_mask=1 + depth=1.0 — so the depth-color scale doesn't
  •    show "dry" stripes inside the user's polygon.)
  • _uploaded_preview.set(mesh)  # triggers re-render with depth-shaded hexes
        │
        ▼
On unreachable EMODnet: warning toast + auto-flip toggle off.
On bbox outside coverage: depth_grid all-NaN → mesh.depth zeros + warning.
```

### Reset path

```
User picks a different study area in the existing dropdown.
        │
        ▼
@reactive.effect on input.study_area:
  _uploaded_preview.set(None)
        │
        ▼
mesh reactive returns the existing study-area mesh.
Map re-renders the chosen study area.
The Create Model accordion's file-picker stays populated (Shiny UX);
clicking Preview again would re-tessellate.
```

## Error handling and invariants

### Upload-time errors (toast + status text)

| Trigger | User-facing message |
|---|---|
| File suffix not in `{.shp.zip, .gpkg, .geojson}` | `"Unsupported format: .X. Use .shp.zip / .gpkg / .geojson."` |
| `.shp.zip` missing required sidecar (.dbf/.shx) | `"Shapefile bundle missing .dbf or .shx — re-export the full bundle."` |
| Corrupt / truncated file | `"Couldn't read the file as a vector dataset."` |
| No geometry column | `"File contains no polygon geometry."` |
| Non-Polygon types only (e.g., LineString) | `"File must contain Polygon or MultiPolygon features."` |
| Missing CRS | `"File has no CRS — please re-export with a defined coordinate system."` |
| Invalid EPSG / `to_crs` raises | `"Invalid CRS in file: {detail}."` |
| Polygon crosses antimeridian | `"Polygons crossing the antimeridian aren't supported. Split into eastern and western parts."` |

All shown via `ui.notification_show(..., type="error", duration=8)` plus
inline `output_text("create_model_status")`. `parse_upload` raises typed
exceptions; the calling reactive translates to user messages.

### Tessellation-time errors

| Trigger | User-facing message |
|---|---|
| Cell count > `max_cells` | `"Tessellation at res N would produce {n:,} cells (max {max:,}). Pick a coarser resolution or smaller polygon."` |
| 0 cells (polygon smaller than one cell at res N) | `"Polygon is smaller than a single H3 cell at res N. Try a finer resolution."` |
| Polygon validity error | `"Polygon geometry is invalid and couldn't be repaired."` |

The cap message must show the would-be cell count and the cap so users
can see the magnitude of their overage.

### Bathymetry-toggle errors

| Trigger | Behaviour |
|---|---|
| EMODnet WCS unreachable (network) | Warning toast; auto-flip toggle off. |
| Bbox outside EMODnet coverage | depth_grid all-NaN → mesh.depth zeros; info message. |
| Cache directory unwritable (laguna's `data/naturalearth_ocean` permission issue) | Silent fallback: re-fetch every time (warn once in logs). |

### Runtime invariants

| Invariant | Where |
|---|---|
| `PreviewMesh.water_mask` is uniformly 1 | constructed in `preview()`; tested |
| Array length consistency across all PreviewMesh fields | dataclass `__post_init__` assertion |
| `PreviewMesh.reach_names` is single-element list | constructed in `preview()`; tested |
| `mesh` reactive returns study-area mesh when `_uploaded_preview` is None | reactive unit test |
| Setting `_uploaded_preview` does not affect simulation state | enforced by construction — sim reactive ignores it |

### Documented limitations (intentionally not addressed)

1. **Antimeridian-crossing polygons** are rejected with an error message
   rather than handled. Splitting them is a polygon-fixup feature out of
   scope for ephemeral preview.
2. **Multi-CRS files** (rare in practice — gpkg can have per-layer CRS)
   reproject the first layer's CRS. Other layers ignored.
3. **Z-coordinates in features** (3D polygons) are flattened to 2D via
   `wkb.loads(wkb.dumps(geom, output_dimension=2))`.
4. **The cell-count cap is a UI safety net, not a hard limit on
   `preview()`'s capability.** A direct call to `h3_tessellate.preview(geom, res, max_cells=10_000_000)`
   would work; only the Shiny entry-point enforces the default cap.
5. **Bathymetry depth interpolation** uses nearest-neighbor at cell
   centroid (existing pattern in the build script).
6. **No multi-feature reach model.** All features in the uploaded file
   are dissolved into a single area. Per-feature reach support is the
   first job of the future Approach B plan.
7. **No simulation against the upload.** The simulation reactive ignores
   `_uploaded_preview`. Running a simulation requires a registered study
   area with bathymetry + forcing — Approach D / B territory.

## Testing

### Unit tests — `tests/test_h3_tessellate.py` (NEW, ~16 tests)

```
# Tessellation primitives (extracted, behavior-preserving)
test_tessellate_reach_simple_polygon
test_tessellate_reach_buffered_for_narrow_channel
test_bridge_components_connects_two_pieces
test_polygon_trust_water_mask_flips_dry_to_wet

# parse_upload — file format coverage
test_parse_upload_geojson_round_trips
test_parse_upload_shp_zip_extracts_correctly
test_parse_upload_gpkg_reads_first_layer
test_parse_upload_dissolves_multi_feature_to_single
test_parse_upload_reprojects_from_3035_to_4326
test_parse_upload_raises_on_missing_crs
test_parse_upload_raises_on_no_geometry
test_parse_upload_raises_on_non_polygon_features
test_parse_upload_raises_on_antimeridian_crossing

# preview — happy path + caps
test_preview_returns_consistent_dataclass
test_preview_water_mask_all_ones
test_preview_caps_cell_count_at_max_cells
test_preview_with_bathy_off_returns_zero_depth
test_polygon_trust_applied_on_bathy_path        # v1.6.1 logic carried forward
```

Each parse_upload format test uses a tiny fixture under
`tests/fixtures/create_model/{tiny.geojson, tiny_wgs84.gpkg, tiny_3035.shp.zip}`
— small enough to commit (each <2 KB).

### Refactor regression — `tests/test_h3_grid_quality.py` unchanged

The extract-first refactor moves three functions from
`scripts/build_h3_multires_landscape.py` into
`salmon_ibm/h3_tessellate.py`. The build script imports them. Existing
`tests/test_h3_grid_quality.py` (3 tests) must continue to pass without
modification — that's the regression contract.

### Sidebar wiring — `tests/test_sidebar.py` extension (+2)

```
test_sidebar_includes_create_model_accordion
test_sidebar_create_model_inputs_have_expected_ids
```

Both shallow — render the sidebar, search for input IDs in the
resulting HTML. No file upload, no Shiny session.

### Reactive logic — `tests/test_create_model_reactive.py` (NEW, +5)

Tests the `_uploaded_preview` reactive flow without a real browser:

```
test_uploaded_preview_default_none
test_preview_button_sets_uploaded_preview_on_success
test_preview_button_shows_toast_on_cap_exceeded
test_preview_button_shows_toast_on_parse_error
test_study_area_dropdown_change_resets_uploaded_preview
```

Each test mocks `h3_tessellate.preview` to control its return value /
exceptions, then asserts the reactive value transitions correctly.

### Performance regression — `tests/test_movement_metric.py` extension (+1)

```
test_h3_tessellate_extract_no_perf_regression
```

Calls `h3_tessellate.tessellate_reach` on a fixture polygon and asserts
wall-time within 10% of a recorded baseline. Catches accidental
over-instrumentation in the extract refactor.

### Integration — manual Playwright smoke (no automated test in MVP)

The full upload→render flow involves Shiny WebSocket frames + deck.gl
rendering. Documented as a manual step:

```
1. Start the app locally:
     micromamba run -n shiny python -m shiny run app.py
2. Open http://localhost:8000
3. Open the "Create Model" accordion
4. Upload tests/fixtures/create_model/tiny.geojson
5. Pick res 9, click Preview
6. Verify hex cells render over the basemap
7. Toggle "show bathymetry" on
8. Verify hexes change color (depth-shaded)
9. Pick a different study area in the Landscape dropdown
10. Verify the upload is cleared and Curonian renders again
```

A future plan can add Playwright automation; out of scope here.

### Test count delta

| Bucket | Count |
|---|---|
| `test_h3_tessellate.py` (new) | +17 |
| `test_sidebar.py` extension | +2 |
| `test_create_model_reactive.py` (new) | +5 |
| `test_movement_metric.py` extension | +1 |
| **Total** | **+25** |

Suite: 787 → 812 tests (787 = 786 from v1.6.0 + 1 from v1.6.1's `test_reach_id_implies_water_mask`). Runtime impact: ~+8 s.

## Deferred work (carry-forward, future plans)

This spec deliberately does **not** include the following — each
becomes its own future plan:

1. **Approach D — auto-fetch landscape build.** User uploads polygon,
   app auto-fetches EMODnet bathymetry + uses Curonian-default forcing,
   builds NC, registers as a study area. ~2 weeks.
2. **Approach B — full multi-source data ingestion.** User-supplied
   bathymetry, user-supplied forcing, multi-feature reach model with
   attribute-based reach names, persistent study-area library. ~3-4
   weeks.
3. **Per-feature reach model.** Multi-color hex preview where each
   feature in the upload is treated as a distinct reach (like the
   existing inSTREAM-derived 9-reach Curonian landscape). Adds an
   attribute-column picker to the upload UI.
4. **Persistent uploaded study areas.** Save / name / delete uploaded
   models from a library. Survives page refresh.
5. **Playwright integration test** — automate the manual smoke as a
   CI-runnable Playwright session. Requires an app harness.
6. **Visual diff test** — pixel-snapshot the rendered hex preview vs
   a stored golden image.
7. **Antimeridian-crossing polygon support** — currently rejected;
   future polygon-fixup feature could split + re-tessellate.
8. **Bathymetry-fetch retry semantics** — exponential backoff on
   transient EMODnet failures.

## References

- Project tagging history: `v1.5.x` series (visual fix sprint),
  `v1.6.0` (Nemunas delta branching D+ slice, 2026-04-27),
  `v1.6.1` (polygon-trust water_mask override, 2026-04-28 — the bug
  this spec's polygon-trust extraction depends on).
- Build script polygon-trust override:
  `scripts/build_h3_multires_landscape.py:475` (will be extracted to
  `salmon_ibm/h3_tessellate.py:polygon_trust_water_mask`).
- Existing accordion-section pattern: `ui/sidebar.py` (Landscape / Run
  / Bioenergetics / Osmoregulation / Dissolved Oxygen / Seiches).
- Existing study-area dropdown registration: `ui/sidebar.py:33-47`
  (4 hardcoded entries: Curonian H3 multi-res / Curonian H3 / Curonian
  TriMesh / Columbia River).
- Existing H3HexagonLayer + PolygonLayer machinery: `app.py:_build_h3_data_rows`
  (~line 800) and the polygon-overlay reactives (~line 1500).
- Bathymetry fetching reference: `data/_water_polygons.py`
  (`fetch_natural_earth_ocean`, `fetch_instream_polygons` use the same
  WCS/HTTP patterns).
