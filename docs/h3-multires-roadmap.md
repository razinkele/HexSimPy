# Multi-Resolution H3 — Implementation Roadmap

**Companion to** `docs/h3-multi-resolution-feasibility.md` (the design
analysis) and `docs/per-reach-ecology-plan.md` (the alternative
lower-effort path).

The feasibility doc estimated **1–2 weeks of focused work** for a true
variable-resolution H3 backend. This document tracks what's shipped
versus what remains.

## Phase 1 — Mesh class + builder + cross-res neighbour algorithm

**Status: shipped in v1.2.8** (this commit).

* `salmon_ibm/h3_multires.py` — `H3MultiResMesh` class.
  * Per-cell `resolutions` array; CSR neighbour table (`nbr_starts`,
    `nbr_idx`); padded `(N, 12)` compat view as `neighbors` for legacy
    numba kernels.
  * `find_cross_res_neighbours()` — the algorithmic core. Walks the
    H3 resolution hierarchy up *and* down to find neighbours that
    don't sit at the cell's own resolution.
  * Reach helpers (`reach_name_of`, `cells_in_reach`) carried over
    from `H3Mesh`.
* `scripts/build_h3_multires_landscape.py` — multi-res builder.
  * Per-reach resolution config (default: rivers res 11, lagoon res 9,
    OpenBaltic res 8). Override via `--resolutions Nemunas=10,…`.
  * Tessellates each inSTREAM polygon at its own resolution; dedupes
    across reach overlaps; samples EMODnet bathymetry per cell.
  * Writes a NetCDF carrying the CSR neighbour table inline so the
    runtime reader doesn't need to recompute neighbours.
* `tests/test_h3_multires.py` — 8 tests covering:
  * Uniform-res regression (must match `H3Mesh.neighbors` exactly).
  * Coarse-cell-sees-fine-children edge case.
  * Fine-cell-sees-coarse-parent edge case.
  * Pentagon handling.
  * Mesh attributes / construction.
  * Reach helpers.

Test smoke run with rivers at res 10 (one step coarser than the
documented default of res 11 for build-time speed):

```
total cells: 36,418
  Nemunas         (res 10):  1,319    CuronianLagoon (res  9): 17,264
  Atmata          (res 10):    152    BalticCoast    (res  9):  9,663
  Minija          (res 10):    219    OpenBaltic     (res  8):  6,788
  Sysa            (res 10):    153
  Skirvyte        (res 10):    201
  Leite           (res 10):    102
  Gilija          (res 10):    557
neighbour table: 209,900 edges, avg 5.76 per cell
```

For the production-target res 11 rivers: expect ~7× more river cells
(rough estimate ~17 k river cells); total mesh ~50 k cells; build runs
in 5–10 s.

## Phase 2 — Simulation integration (**shipped in v1.3.0**)

`Simulation(config_with_h3_multires)` now works end-to-end with
movement, bioenergetics, and the existing event sequencer.  Plan
executed against `docs/superpowers/plans/2026-04-26-h3-multires-phase2.md`
(post multi-agent review) over 5 tasks:

* **Prereq (v1.2.9 + v1.2.10):** scaffold corrections — `MAX_NBRS`
  overflow guard, `gradient()` port, then `MAX_NBRS` 12 → 64 after
  the guard caught real overflows at res-10 / res-8 boundaries.
* **Task 1 (`a65ee8e`):** builder samples CMEMS forcing per cell.
* **Task 2 (`2a84a87`):** `mesh_backend: h3_multires` Simulation
  init branch + `configs/config_curonian_h3_multires.yaml`.
* **Task 3 (`ee14384`):** `resolution` compat property + sidebar
  dropdown (now default) + viewer routing.
* **Task 4 (`c98d393`):** five new ecological invariants + pytest.ini
  with `slow` marker.  Two plan-spec corrections applied during
  execution: per-step displacement test rewritten as a static
  neighbour-table check (substep-independent); no-one-way-trap test
  filtered to OpenBaltic boundary cells (random walk in 72 h
  doesn't reach the boundary from deep Baltic).
* **Task 5 (v1.3.0):** tag + deploy + browser verify.  Multi-res
  mesh renders correctly with mixed cell sizes visible in the deck.gl
  H3HexagonLayer.

Test counts at completion:

* Default (`-m "not slow"`): 17 passed, 1 skipped, 2 deselected.
* Slow (`-m slow`):           2 passed, 9 deselected.
* H3 regression suite:        49/49 passed (no regression).

### 2a. Config schema

`configs/config_curonian_h3_multires.yaml`:

```yaml
mesh_backend: h3_multires             # new value, sibling of "h3"
h3_multires_landscape_nc: data/curonian_h3_multires_landscape.nc
species_config: configs/baltic_salmon_species.yaml
initial_state:
  initial_cell_strategy: uniform_random_water
# barriers_csv supported but optional
```

### 2b. `Simulation.__init__` h3_multires branch

`salmon_ibm/simulation.py`:

```python
if mesh_backend == "h3_multires":
    from salmon_ibm.h3_multires import H3MultiResMesh
    nc = config["h3_multires_landscape_nc"]
    ds = xr.open_dataset(nc, engine="h5netcdf")
    cells = [_h3.int_to_str(int(i)) for i in ds["h3_id"].values]
    self.mesh = H3MultiResMesh(
        h3_ids=ds["h3_id"].values,
        resolutions=ds["resolution"].values,
        centroids=np.column_stack([ds["lat"].values, ds["lon"].values]),
        nbr_starts=ds["nbr_starts"].values,
        nbr_idx=ds["nbr_idx"].values,
        water_mask=ds["water_mask"].values.astype(bool),
        depth=ds["depth"].values,
        areas=...,        # compute from h3.cell_area(c)
        reach_id=ds["reach_id"].values,
        reach_names=ds.attrs["reach_names"].split(","),
    )
```

`H3Environment.from_netcdf` works unchanged because forcing arrays
are already keyed by compact cell index.

### 2c. Numba movement kernels — bump `MAX_NBRS`

`salmon_ibm/movement.py` calls into `_step_directed_numba` and
`_advection_numba`, both of which read `mesh._water_nbrs` as a fixed-
shape `(N, 6)` array. The padded-`(N, 12)` view in `H3MultiResMesh`
is shape-compatible *if* the kernels read `_water_nbrs[i, :count]`
where `count = _water_nbr_count[i]`. Verify by inspection — if any
kernel hardcodes `range(6)`, change to `range(MAX_NBRS)` and ensure
the `-1`-sentinel guard is in place.

Estimated effort: **half a day**, mostly mechanical. The `H3MultiResMesh`
class already exposes the right view; the kernels just need to use the
count-aware iteration pattern.

### 2d. Movement disaggregation across resolutions

When an agent at a coarse cell (`res 9`) chooses a neighbour that's a
fine cell (`res 11`), it lands in *one* fine cell. Fine. But when an
agent at a fine cell crosses *into* a coarse zone, the coarse cell is
~50× larger. The agent should land somewhere "inside" that coarse cell
— which is fine for the IBM (the cell IS the spatial unit), but the
agent's *future* movement is now coarser-grained.

No code change needed for this case — the existing movement loop
treats each cell as a homogeneous spatial unit. Agents naturally
"jump" into coarser cells when they cross resolution boundaries; the
movement physics is correct.

The *only* subtlety: when computing things like "expected movement
distance per step" or visualising agent tracks, displaying centroids
of mixed-resolution cells produces visibly uneven step sizes. The
viewer needs to handle this gracefully — at high zoom, fine cells
look right; at low zoom, coarse cells look right; in mixed zooms,
either looks fine but adjacent steps may have very different lengths.
Document this in the viewer rather than try to "fix" it.

## Phase 3 — Viewer

`H3HexagonLayer` already supports mixed resolutions natively — each
hex string carries its own resolution and deck.gl tessellates
accordingly. The existing `_load_viewer_h3_landscape` path needs:

* Read `resolution` per-cell instead of from the global `h3_resolution`
  attribute.
* When picking the camera zoom, use the *median* resolution rather than
  the file-level one. Fine zones get cropped at low zoom; coarse zones
  stay visible.

Estimated effort: **2–3 hours**.

## Phase 4 — Tests

* End-to-end `test_h3_multires_integration.py` mirroring
  `test_nemunas_h3_integration.py`: place 50 agents, run 30 days, check
  invariants. Uses the smaller `--resolutions` override so the test
  finishes in <60 s.
* Cross-resolution movement test: place an agent in a fine river cell,
  run until it transitions into the coarser lagoon, verify the cell
  index switches resolution.

Estimated effort: **half a day**.

## Phase 5 — Validation against published Nemunas Delta hydraulics

The whole point of variable resolution is that the rivers should
represent the actual channel hydraulics. Phase 5 cross-checks the
simulation against:

* **Channel velocity**: at res 11 (~28 m), Nemunas main channel
  cells should show ~0.4–0.6 m/s mean current per CMEMS (or the
  reach-specific time-series in the inSTREAM example_baltic config).
* **Smolt outmigration timing**: junior agents released in the
  Nemunas headwaters should reach Klaipėda Strait in 7–14 days at
  the current res-11 grid resolution, matching telemetry studies.

This is real ecological calibration work — beyond the *scaffolding*
scope of v1.2.8 but the natural place this leads.

## Total remaining effort to "production-ready multi-res"

Conservative estimate from the roadmap above:

| Phase | Effort |
|---|---|
| 2a. Config schema | 1 hour |
| 2b. Simulation init branch | 2–3 hours |
| 2c. Numba MAX_NBRS bump | 4 hours |
| 2d. Movement disaggregation (mostly docs) | 1 hour |
| 3. Viewer | 2–3 hours |
| 4. Integration test + cross-res move test | 4 hours |
| 5. Calibration | open-ended |

**Total Phase 2-4: 2–3 days** beyond the v1.2.8 scaffold.

The feasibility doc's original "1–2 weeks" estimate is conservative —
because the scaffold separates the *novel* work (cross-res adjacency)
from the *integration* work (sim, viewer), and the novel work is now
done and tested. Phase 2-4 is mostly plumbing.

## Why phase the scaffold first

* The hardest part — cross-res neighbour finding — is an algorithm
  that's easier to reason about and test in isolation than as part of
  a fully wired-up simulation. Shipping it on its own with 8 unit
  tests gives a stable foundation.
* The mesh class is a clean drop-in replacement for `H3Mesh` once the
  kernels are bumped to `MAX_NBRS=12` — no other code changes
  cascade.
* Phase 2-4 can land in any order; they don't depend on each other.
  Convenient for iterative shipping.
