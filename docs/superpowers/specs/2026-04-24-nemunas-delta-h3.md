# Nemunas Delta H3 Test Landscape — Spec

> **STATUS: ✅ IMPLEMENTED 2026-04-25.**  All eleven invariants in
> § "Validation invariants" are exercised by
> `tests/test_nemunas_h3_integration.py` and pass against the real
> EMODnet + CMEMS-bound `data/nemunas_h3_landscape.nc`.  See the
> sibling plan's "Execution status" section for commit details and
> deviations.

Sibling to `docs/superpowers/plans/2026-04-24-h3mesh-backend.md`. This document pins down what the test landscape *is* — its spatial extent, resolution, data sources, validation invariants — so Phase 2.1 of the plan (`scripts/build_nemunas_h3_landscape.py`) has unambiguous targets.

## Purpose

First end-to-end proof that the H3Mesh backend can host a real salmon migration simulation. Chosen over Curonian Lagoon proper because the delta (south end of the lagoon, where the Nemunas river meets the brackish water) is narrow enough to be computationally cheap, includes the critical estuary gradient, and has real CMEMS/EMODnet data already fetched and cached.

## Spatial extent

**Bounding box** (same as the CMEMS Curonian fetch — reuses existing raw data):
- Longitude: 20.4° – 21.9° E
- Latitude: 54.9° – 55.8° N
- Physical size: ~96 km (W–E) × ~100 km (S–N)
- Covers: the entire Curonian Lagoon, Nemunas Delta, Klaipėda Strait, and a strip of Baltic coast

The test run places agents anywhere in the water-masked region; for this smoke test that's fine. A narrower follow-up spec can tighten to the delta proper (21.0–21.6°E, 55.15–55.50°N) if needed.

## H3 resolution

**Resolution: 9** (avg edge ≈ 200.79 m, avg area ≈ 105,332 m² / 10.5 ha).

Justification:
- Existing Curonian TriMesh is 30 × 30 = 900 nodes with ~750 m × 1840 m cells — coarser than the Nemunas delta geometry needs.
- CMEMS native grid for BALTICSEA is ~2 km; at H3 res 9 (200 m), one CMEMS cell covers ~100 H3 cells, so linear interpolation of temperature/salinity produces smooth fields with no aliasing.
- EMODnet DTM 2022 at 1/16 arc-min ≈ 116 m native resolution: H3 res 9 (200 m) slightly undersamples, giving average depth per cell — acceptable for movement, preserves Klaipėda strait bathymetric gradient.
- Payload: ~5.3 MB per forcing snapshot, 37 MB per 30-day run with 1 h CMEMS resolution → comfortable for Shiny/WebSocket and viewer H3HexagonLayer.

Alternative resolutions (kept documented for future scenarios):

| Res | Cells | Edge | Purpose | Payload |
|-----|-------|------|---------|---------|
| 8 | 15,167 | 531 m | Regional overview / low-res fast runs | 0.8 MB |
| **9** | **106,188** | **201 m** | **Default** — adequate for estuary movement | **5.3 MB** |
| 10 | 743,198 | 76 m | High-res spawning-site studies | 37 MB |
| 11 | 5.2 M | 29 m | Not recommended — exceeds typical WebSocket frame |

Pentagon status: **0 pentagons at every resolution in this bbox** (verified via `h3.is_pentagon()` sweep). No pentagon-handling code needed; `H3Mesh.from_polygon(..., pentagon_policy="raise")` is the default because any future bbox that crosses a pentagon will fail loudly.

## Data sources

### Bathymetry
- **EMODnet Bathymetry DTM 2022** via WCS (already fetched): `data/curonian_bathymetry_raw.tif`.
- License: CC-BY 4.0 (EMODnet).
- Mapping: for each H3 cell, sample at the cell's centroid (`h3.cell_to_latlng`) via `scipy.interpolate.RegularGridInterpolator` with linear interpolation.
- Sign convention: EMODnet = positive-up elevation. HexSim convention = positive-down depth. **Flip sign and clamp at zero** (NaN → 0) to produce `depth (m)` where >0 = water.

### Temperature, salinity, currents
- **CMEMS BALTICSEA_MULTIYEAR_PHY_003_011** via `copernicusmarine` (already fetched): `data/curonian_forcing_cmems_raw.nc`.
- License: CC-BY 4.0 (Copernicus Marine).
- Variables: `thetao` → `tos` (sea-surface temperature °C), `so` → `sos` (salinity PSU), `uo`/`vo` (surface currents m/s east/north).
- Temporal: daily 2011–2024; downstream scenarios pick their window.
- Spatial mapping: same RegularGridInterpolator + NearestNDInterpolator NaN-fill as the existing Curonian regridder (`scripts/fetch_cmems_forcing.py`).

### SSH (sea-surface height for seiche detection)
- **Not in this dataset.** `zos` is zero-filled. Real SSH requires a separate CMEMS product (deferred in the parent Curonian plan; same deferral applies here).

## Output file: `data/nemunas_h3_landscape.nc`

NetCDF3 64-bit offset (so scipy engine can load it on Windows), single file, ~6–8 MB.

Dimensions:
- `cell`: 106,188 (res 9 over bbox, minus any pentagons — 0 here)
- `time`: depends on the CMEMS window loaded (default 2011-01-01 to 2024-12-31 daily ≈ 5113 days → if we load it all, 1.1 GB. For the test landscape, **load 30 days (2011-06-01 to 2011-06-30)** to keep file size ≤10 MB.)

Variables:

| Name | Dims | Dtype | Units | Meaning |
|------|------|-------|-------|---------|
| `h3_id` | (cell,) | uint64 | — | H3 cell index (integer form of `8a...` hex string) |
| `lat` | (cell,) | float64 | deg N | Cell centroid latitude |
| `lon` | (cell,) | float64 | deg E | Cell centroid longitude |
| `depth` | (cell,) | float32 | m | EMODnet bathymetry, positive down |
| `water_mask` | (cell,) | uint8 | — | 1 = water (depth > 0), 0 = land |
| `tos` | (time, cell) | float32 | °C | Sea-surface temperature |
| `sos` | (time, cell) | float32 | PSU | Salinity |
| `uo` | (time, cell) | float32 | m/s | Eastward current |
| `vo` | (time, cell) | float32 | m/s | Northward current |

Global attrs:
- `h3_resolution = 9`
- `source_bathymetry = "EMODnet DTM 2022"`
- `source_forcing = "CMEMS BALTICSEA_MULTIYEAR_PHY_003_011"`
- `bbox = "{'minlon': 20.4, 'maxlon': 21.9, 'minlat': 54.9, 'maxlat': 55.8}"`
- `created_utc = "YYYY-MM-DDTHH:MM:SSZ"`
- `pentagon_count = 0`

Cell ordering: stable, sorted by `h3_id` ascending. The loader (`H3Environment.from_netcdf`) uses `np.searchsorted` on this sorted order to bind mesh-cell → dataset-cell, so ordering must be consistent.

## Validation invariants (hard, enforced in `tests/test_nemunas_h3_integration.py`)

1. **Agent conservation.** alive + dead + arrived == n_agents (same as Curonian test).
2. **No total extinction.** alive > 0 after 30-day run. (A 2011-06 window, not winter — temperatures 14–20 °C, well below T_ACUTE_LETHAL=24 °C.)
3. **Temperature envelope.** -2 < t.min() and t.max() < 25 — Baltic surface realism.
4. **Salinity envelope.** 0 ≤ s ≤ 10 PSU — lagoon-appropriate.
5. **North-south salinity gradient.** Cells with lat > 75th-percentile (Klaipėda Strait area) have mean salinity **≥ 1.5 PSU** higher than cells with lat < 25th-percentile (Nemunas mouth). CMEMS BALTICSEA reanalysis routinely shows ≥ 3 PSU difference across this geography; 1.5 PSU is loose enough not to fail on weather noise but tight enough to catch regridder bugs that silently homogenise the lagoon from the strait end.
6. **Bathymetry envelope.** 2 < d_mean < 20 m, d_max < 60 m. Lagoon interior 3–4 m, strait bathymetry up to 50 m, nothing in the open Baltic abyssal plain (would signal a bbox extension bug).
7. **Mesh type assertion.** `isinstance(sim.mesh, H3Mesh)` and `sim.mesh.resolution == 9`.
8. **Agent placement sanity.** All `pool.tri_idx` values are ≥ 0 and < `mesh.n_cells` — no off-mesh placement.
9. **Movement is happening.** ≥ 10 % of agents have moved after 30 days. (Sanity guard against the movement kernel silently no-oping on the new mesh.)

## Known limitations

- **Methodological test, not an ecological claim.** Every invariant below is a **software-mechanics** guarantee (mesh loads, movement kernel runs, thermal/salinity fields bind correctly). None of them validate salmon migration timing, natal-river fidelity, density-dependent dynamics, or any ecological quantity. A published scenario should not cite this test landscape as evidence of realism.
- **Movement speed ceiling at res-9 / 1 h.** `movement_advection` moves one cell per tick; at 200 m cells and 1 h steps, the realised swimming speed is ≤ 5.6 cm/s — roughly 50–100× slower than a cruising *S. salar* (0.5–1 body length/s → 50–100 cm/s). Agents cannot actually *migrate* on this mesh; they drift locally. For migration-timing studies, either drop timestep to ~minutes at res 9 or coarsen to res 7–8 (530 m – 1.2 km) at 1 h.
- **Agent density is sparse.** 500 agents over 106 k water cells ≈ 0.3 agents/km², 2–3 orders of magnitude below realistic post-smolt coastal densities (Baltic rivers release 10⁵–10⁶ smolts per year). The current IBM has no density-dependent term, so this does not bias invariants *today* — but any future density-dependent mortality, disease transmission, or predator-functional-response must re-tune `n_agents` before inference. A realistic Nemunas stock scale is ~50 k – 500 k agents.
- **`uniform_random_water` placement is biologically false.** Agents are scattered over any water cell regardless of natal-river fidelity or homing behaviour. Use `spawn_site` (existing TriMesh path) for ecological scenarios.
- **Synthetic mid-lagoon barrier is illustrative only.** The 5/90/5 probability split is not calibrated to any real structure. The line is a 25 km E–W traverse at lat 55.30° N — chosen during execution because the original "Klaipėda Strait" placement was only 2–4 cells wide at H3 res 9 (1–4 water-water edges after the EMODnet land-mask filter, not enough signal for a 3-day mortality-effect test). The lagoon traverse yields 116 in-mesh edges. *Not a fisheries-management claim* — `scripts/build_nemunas_h3_barriers.py` is reproducible from any landscape NetCDF.
- **Two barrier variants ship.** `data/nemunas_h3_barriers.csv` (gentle 5/90/5) is the default the YAML config points at. `data/nemunas_h3_barriers_strong.csv` (aggressive 30/60/10) is used only by the integration mortality-effect test fixtures, which override `barriers_csv` at simulation construction.
- **No delta-branching model.** The real Nemunas delta splits into the Atmata, Skirvytė, and Minija branches. Branching is implicitly present via the CMEMS velocity field (`uo`/`vo`) but not as discrete paths with assigned flow fractions.
- **Zero-fill SSH.** Seiche detection is disabled — same deferral as the Curonian plan. Not an issue for this test.
- **Surface-only forcing (no DO, no vertical structure).** `temperature` and `salinity` are sea-surface only. The lagoon is 3–4 m deep so depth stratification is negligible, but any thermal-refuge-seeking behaviour at the 50 m Klaipėda-strait bathymetric deep is modelled at surface temperature.
- **Per-cell area varies ~1.4× across H3 res 9.** The plan uses per-cell `h3.cell_area()` for the `areas` array (correct), but movement kernels rely only on centroid differences — latent concern for any future flux-across-edge term (Lagrangian tracer, bioenergetic integration). Fix via `cell_to_boundary` when needed.
- **No basemap context in viewer.** H3 cells are in real lat/lon but the viewer's `BLANK_STYLE` hides coastline tiles. Swap to `CARTO_POSITRON` if geographic context is wanted — no other changes needed.

## Regeneration

```bash
# Assumes data/curonian_bathymetry_raw.tif and data/curonian_forcing_cmems_raw.nc
# already exist (from the Curonian realism upgrade work).

micromamba run -n shiny python scripts/build_nemunas_h3_landscape.py \
    --out data/nemunas_h3_landscape.nc
```

Runtime: ~30 s for 106k cells at res 9 with a 30-day CMEMS window.

## Relation to existing Curonian TriMesh

This landscape is **not** a replacement for `data/curonian_minimal_grid.nc` + `data/curonian_forcing_cmems.nc` — those stay canonical for the TriMesh backend and are validated by `tests/test_curonian_realism_integration.py`. The H3 landscape is a *parallel* representation of the same geography at ~100× finer resolution, built to exercise and validate the H3Mesh backend. If model results from the two mesh backends ever diverge on the same geography, that divergence is itself data: either the H3 conversion lost something meaningful or the TriMesh is under-resolving the estuary.

## As-built artifacts

Files committed to git as part of the H3Mesh implementation
(plan-sibling commits `6360f7d` … `e0fa529`):

| Path | Kind | Purpose |
|---|---|---|
| `salmon_ibm/h3mesh.py` | code | `H3Mesh` class, factories, pentagon guard |
| `salmon_ibm/h3_env.py` | code | `H3Environment.from_netcdf` (CMEMS/EMODnet → mesh) |
| `salmon_ibm/h3_barriers.py` | code | `line_barrier_to_h3_edges` + CSV loader |
| `salmon_ibm/geomconst.py` | code | `M_PER_DEG_LAT`, `M_PER_DEG_LON_EQUATOR` |
| `scripts/build_nemunas_h3_landscape.py` | tool | bbox polygon → 106 k cells + CMEMS forcing |
| `scripts/build_nemunas_h3_barriers.py` | tool | mid-lagoon line → barrier CSV pair |
| `configs/config_nemunas_h3.yaml` | config | scenario file the integration test loads |
| `data/nemunas_h3_barriers.csv` | data | gentle 5/90/5, 116 edges (committed, ~12 KB) |
| `data/nemunas_h3_barriers_strong.csv` | data | strong 30/60/10, same 116 edges (committed) |
| `tests/test_h3mesh.py` | tests | 21 unit tests |
| `tests/test_h3_env.py` | tests | 7 unit tests |
| `tests/test_h3_barriers.py` | tests | 12 unit tests |
| `tests/test_nemunas_h3_integration.py` | tests | 8 end-to-end invariants |
| `tests/test_movement_metric.py` | tests | 2 metric_scale regressions |

Files **not** committed (gitignored, regeneratable):

| Path | Size | Regenerated by |
|---|---|---|
| `data/nemunas_h3_landscape.nc` | 54 MB | `scripts/build_nemunas_h3_landscape.py` (~30 s) |
