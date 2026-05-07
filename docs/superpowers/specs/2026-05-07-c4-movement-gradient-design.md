# C4 — Movement Gradient (substrate fix)

**Date:** 2026-05-07
**Owner:** @razinkele
**Status:** ✅ CONVERGED v11 — **12-pass review-loop complete; deeper than the C3.3 8-pass cycle.** v10 added 9 new tests addressing pass-11's test-coverage gaps; pass-12 verified closure + flagged 3 LOW polish items (Test 5c centroid construction detail, Test 7b RNG seeding mechanism, header test-count off-by-one) — fixed inline as v11 final. Implementation-ready. Awaiting writing-plans.

C4 fixes a **substrate-level correctness defect** that has been latent
since the H3 multi-resolution mesh shipped (v1.5.0, 2026-03 cohort).
Returning adults' UPSTREAM movement degenerates to an
approximately-zero-net-displacement oscillation because the directed-
movement kernel reads a flat (all-zero) `ssh` field. The four-tier
hatchery-vs-wild architecture (C1+C2+C3.1+C3.2+C3.3, all shipped
through v1.7.7) is structurally correct but **biologically dormant in
production** — agents do not reach delta branches, so C3.3's homing-
precision dispatch never fires for any agent in a real run.

C4 sits *underneath* the hatchery tiers as a substrate concern. The
numbering "C4" preserves the C-tier naming convention from the
hatchery-vs-wild work, but architecturally C4 is closer to a movement-
layer prerequisite than a behavioral-divergence tier.

## Why now

A code review of the post-C3.3 movement layer + a Playwright live-test
on the deployed app (https://laguna.ku.lt/HexSimPy/, v1.7.7) on
2026-05-06 surfaced that:

- Default 50-agent / 480-hour run produced **0/50 arrivals**.
- The 3 surviving agents wandered in lon×lat boxes of only ~80m × 70m
  over 480 hours — observed displacement ~2 m/hour.
- Atlantic salmon adults swim 0.5-1 body-lengths/sec ≈ 1.5-3 m/s.
  Observed speed is **~1000× too slow**.
- The BEHAVIOR chart showed agents in active UPSTREAM mode — they
  were trying to move, not in HOLD or barrier-blocked states.

A subsequent code-explorer agent traced the rate-limiting factor to
`salmon_ibm/h3_env.py:113-125`, where `H3Environment.from_netcdf()`
explicitly zero-fills the `ssh` field with the comment *"gradient-
following degenerates to first-neighbour selection."* CMEMS Baltic
reanalysis does not include sea-surface height; the build pipeline
ships a flat-zero array. With all field values = 0.0, the directed-
movement kernel `_step_directed_numba` (`movement.py:208-228`) falls
back to `water_nbrs[c, 0]` (slot-0 neighbor, deterministic) on even
micro-steps and a random neighbor on odd. Slot-0 directions vary
arbitrarily across cells, so net displacement of "directed" hops
approximately cancels. The agent oscillates in a small neighborhood
instead of progressing upstream.

## Why bathymetry-elev was rejected

A natural first instinct is to use bathymetric elevation
(`elev = -depth`) as an upstream-distance proxy: deeper sea cells
should be "downstream", shallower delta cells "upstream". An empirical
diagnostic on the production Curonian H3 mesh (commit `b18ecaf`,
2026-05-06) rejected this:

| Reach            | Cells   | Mean depth | Expected real-world |
|------------------|---------|------------|---------------------|
| Nemunas (river)  | 10,061  | 4.77 m     | 1-3 m               |
| CuronianLagoon   | 121,666 | **46.63 m** | **3.8 m mean**     |
| OpenBaltic       | 31,493  | **12.81 m** | **50-200 m**       |
| Atmata           | 1,252   | 1.47 m     | ~1 m                |
| Skirvyte         | 1,580   | 1.59 m     | ~1 m                |
| Gilija           | 4,546   | 2.70 m     | ~2 m                |

The lagoon reports ~12× deeper than reality, and OpenBaltic — which
should be ~10× deeper than the lagoon — reports *shallower*. Greedy
ascent from an OpenBaltic cell on `elev = -depth` hit a local maximum
at step 1 (a sandbar artifact at depth 0.43 m, surrounded by
descending neighbors). The bathymetry data on the deployed mesh is
not a usable upstream gradient.

The data is degenerate either because EMODnet sampling failed and
filled with defaults, or because the build script's polygon-overlay
logic mis-attributes depth. Investigating the bathymetry build is out
of scope for C4 — even with corrected EMODnet sampling, river depth
isn't EMODnet's domain (rivers aren't coastal); the river-side of the
gradient would still need a separate signal.

## Scope

### In

- New scalar field `dist_from_sea`: per-cell distance (in meters)
  from the nearest open-Baltic boundary cell, computed by
  multi-source edge-length-weighted Dijkstra on the mesh's
  neighbor graph.
- `dist_from_sea` saved as a `float32` variable in the landscape NC
  by the build script; loaded by `H3Environment.from_netcdf` if
  present.
- `_step_directed_numba` and `_step_directed_vec` UPSTREAM/DOWNSTREAM
  paths read `fields["dist_from_sea"]` instead of `fields["ssh"]`.
  **The `ascending` flag must flip from current values.** Current
  code (`movement.py:107`, `:125`): UPSTREAM uses `ascending=False`,
  DOWNSTREAM uses `ascending=True`. This was correct under the
  legacy semantic where SSH was assumed to be lower upstream.
  `dist_from_sea` is higher upstream by construction, so the
  polarity reverses: **UPSTREAM must use `ascending=True`**
  (climb toward higher `dist_from_sea`); **DOWNSTREAM must use
  `ascending=False`** (descend toward lower). The kernel logic
  itself is unchanged; the field swap forces the polarity flip.
  An implementer who copies the existing `ascending=False` for
  UPSTREAM unmodified will produce a silent regression where
  upstream-behaving fish chase the open sea.
- Backward-compat: NCs without `dist_from_sea` produce a clear
  `RuntimeWarning` at sim init; the field defaults to all-zeros so
  legacy behavior is preserved (broken, but unchanged).
- Build-time validation: every water cell must have a finite
  `dist_from_sea`. Disconnected sub-graphs cause Dijkstra to leave
  `inf`; the build script asserts on this and reports the
  unreachable cells with their reach names.
- Build-time sanity output: per-reach `dist_from_sea` distribution
  printed (min / max / mean) so anomalies surface during the build.
- Unit test in `tests/test_movement_gradient.py` (NEW): synthetic
  10-cell linear chain mesh with a known gradient; assert a
  directed-UPSTREAM agent climbs 5 cells in ≤ 5 timesteps.
- Integration test asserting `_step_directed_*` reads the new field
  name (mock landscape with `dist_from_sea` populated, agent moves
  toward higher values).
- Existing test suite (893 passing cases as of the 2026-05-06
  full sweep on commit `b18ecaf`) continues to pass with no
  regressions. Plain `def test_` count alone is lower (~420)
  because pytest fixtures, classes, and parametrize multiply
  case counts — use the sweep count as the canonical baseline.
- Live-test contract: post-deploy Playwright run on laguna confirms
  ≥ 1 agent reaches a delta branch within 480 hours at default
  settings (50 agents, seed=42). This is the test that should have
  existed all along; C4 lands it.

### Out (deferred)

- **Per-tributary granularity** (Žeimena/Merkys/Dubysa). The
  deployed mesh's reach taxonomy ends at delta-branch level
  (Atmata/Skirvyte/Gilija/Nemunas/etc.); per-tributary homing
  requires a mesh upgrade with finer reach polygons. Future tier.
- **Per-natal-reach gradient.** C3.3's branch-level dispatch
  handles natal-vs-stray decisions at delta entry. A single global
  gradient is sufficient given the current data model. Per-natal-
  reach fields would be O(N_branches × N_cells) memory for marginal
  biological gain.
- **Per-agent goal-cell pathfinding.** Most flexible (each agent
  has its own gradient toward its specific natal cell), but
  requires a kernel-signature change and per-agent state. Out of
  scope for the substrate fix.
- **Real hydrodynamic SSH** from a Baltic Sea model (e.g., NEMO
  output). Replaces the synthetic gradient with measured sea-surface
  height; requires data-pipeline changes. If/when it lands later,
  it can drop in via the same field name without movement-layer
  changes.
- **Bathymetry data correction.** The diagnostic showed broken
  depth values on the production mesh; fixing the EMODnet build
  pipeline is its own concern, not bundled with C4. C4 sidesteps
  the broken bathymetry by computing `dist_from_sea` purely from
  graph topology, not from depth.
- **TO_CWR behavior.** Already uses `temperature` field, not
  affected by the SSH=0 issue. Unchanged.
- **Movement-kernel refactor** (e.g., A*-style pathfinding,
  multi-objective behavior dispatch). The existing
  `_step_directed_numba` / `_vec` architecture is sound; C4 is a
  data-source change, not an algorithmic one.
- **Nemunas NC support** (`data/nemunas_h3_landscape.nc`).
  Initially in scope but rejected at v3 review. The Nemunas
  build script (`scripts/build_nemunas_h3_landscape.py`) is a
  separate uniform-resolution pipeline that does NOT build a
  CSR neighbour table — there is no `nbr_starts`/`nbr_idx` on
  the Nemunas NC, so multi-source Dijkstra on the mesh's
  neighbour graph cannot run. Adding `dist_from_sea` to the
  Nemunas pipeline requires either (a) building an inline
  H3 ring-1 neighbour table in that script, or (b) refactoring
  to share `H3MultiResMesh` machinery. Neither is part of C4.
  The Nemunas NC is a development/research asset, not the
  production deploy (laguna.ku.lt loads the Curonian multi-res
  NC); deferring this is low-impact for production. Future tier.

## Architecture

C4 is a **topology-only movement abstraction**. It does not encode
olfaction, magnetic compass cues, hydroacoustic recognition, or any
of the multi-modal cues real Atlantic salmon use during the homing
phase of migration. The gradient is sufficient to deliver agents
from open Baltic into the lagoon and through the delta-mouth funnel;
once at the delta, **C3.3's branch-level dispatch handles the
natal-vs-stray decision** (which is the cue C4 cannot encode at
single-field granularity). A future per-natal-reach gradient tier
could replace this single field with an N-field array if individual-
fidelity homing becomes a research requirement; that's deferred.

### Data structure

A single new variable in the H3-multires landscape NC:

```
dist_from_sea: float32, shape=(N_cells,), units=meters
```

Semantics: distance (in meters) from each cell's centroid to the
nearest cell where `reach_id == OpenBaltic AND water_mask == True`,
along the mesh's neighbor graph, using great-circle edge weights.
Cells that are themselves OpenBaltic water cells get distance 0.
Land cells (water_mask=False) get distance NaN; they are not
reachable in the graph and never used by movement.

### Computation algorithm

Multi-source Dijkstra on the mesh's neighbor graph:

1. **Source set:** `S = {c : reach_id[c] == OpenBaltic AND water_mask[c]}`.
2. **Edge weights:** for each ordered neighbor pair `(c, n)`,
   `w(c, n) = haversine(centroid[c], centroid[n])` in meters.
   Distance is symmetric on the H3 graph, so we compute one
   direction and reuse.
3. **Dijkstra:** initialize `dist[S] = 0`, push all sources to a
   min-heap, expand greedy. Only traverse edges where both
   endpoints have `water_mask=True`.
4. **Output:** `dist_from_sea = float32(dist)`. Land cells remain
   NaN.

Cost: O(E log V) ≈ a few seconds for the production mesh
(N_cells ≈ 185k, average degree ≈ 6, so E ≈ 1.1M edges per direction,
matching the NC's 1.09M `edge` dimension). Runs once at landscape
build time.

### Determinism

`compute_dist_from_sea(mesh)` MUST produce bit-exact identical
output across runs given identical input. This is required so a
committed NC and an in-process recomputation agree exactly,
preventing silent drift in seeded reproducibility tests. Two
sources of non-determinism to neutralise:

1. **Source set iteration order.** The source set is built from
   `np.where((mesh.reach_id == OpenBaltic_id) & mesh.water_mask)[0]`.
   Always `np.sort(...)` the resulting array before pushing to
   the heap — do NOT rely on any implicit ordering from `np.where`.
   NumPy's API does not guarantee a specific iteration order
   across versions or array shapes; explicit sort by cell index
   pins it deterministically.
2. **Heap tie-breaking.** Python's `heapq` orders tuples
   lexicographically; ties in distance are broken by the next
   tuple element. Push entries as `(distance, cell_index)` so
   ties break by cell index (deterministic) rather than by
   insertion order or pointer-comparison fallback. Never push
   an entry containing a non-comparable object (e.g., a dict)
   that would force fallback to id()-based comparison.

The `_step_directed_*` kernel itself does not use `dist_from_sea`
in any reduction-order-sensitive way (it reads scalar values per
neighbor independently), so the kernel side is already
deterministic. Determinism discipline is fully contained in
`compute_dist_from_sea`.

### Where it lives

- `scripts/build_h3_multires_landscape.py` — add a
  `compute_dist_from_sea(mesh) -> np.ndarray` function that
  returns a `float32[N_cells]` distance array (in meters from the
  nearest OpenBaltic water cell). The function is **pure**: it
  does NOT mutate `mesh`. The build script calls it after the
  neighbor table is built, writes the returned array to NC as a
  variable named `dist_from_sea`, and prints per-reach
  distribution to stdout for sanity-check. The `mesh.dist_from_sea`
  attribute is set later by `H3Environment.from_netcdf`
  post-construction (see h3_env.py load sequence below), NOT by
  the build script — keeping the build path side-effect-free.
- **Mesh attribute (post-hoc):** `mesh.dist_from_sea` is set as
  a post-construction attribute, NOT via `H3MultiResMesh.__init__`.
  The class has no `__slots__`, so dynamic attribute assignment
  works at runtime. Intentional tradeoff for C4: avoiding a
  constructor signature change keeps the mesh-build / mesh-load
  paths unchanged for non-Baltic / legacy callers. A future tier
  can promote `dist_from_sea` to a constructor parameter if the
  mesh ever gains static-typing enforcement. The analogy with
  `reach_id`/`water_mask` applies to the *runtime data shape*
  (per-cell static array on the mesh), not to the constructor
  contract (those fields ARE in `__init__`; `dist_from_sea` is
  not).
- `salmon_ibm/h3_env.py` — `H3Environment.from_netcdf` loads
  `dist_from_sea` (1D, `float32[N_cells]`, time-independent) if
  present.

  **Storage path — load order matters.** `dist_from_sea` MUST
  NOT enter `full_fields` / `self._full_fields`. The existing
  `advance()` (`h3_env.py:153`) iterates `self._full_fields.items()`
  and does `np.copyto(self.fields[name], arr[self._time_idx])`.
  If `dist_from_sea` (shape `(N,)`) entered `_full_fields`, then
  `arr[self._time_idx]` would be a *scalar* (single float) and
  `np.copyto` would broadcast-fill the whole `(N,)` field every
  step — silently corrupting the gradient into a uniform value.
  Additionally, the `n_time = full_fields[next(iter(...))].shape[0]`
  inference at `h3_env.py:121-122` would mis-fire on a 1D entry.

  **Correct load sequence in `from_netcdf`:**
  1. Build `full_fields` from `_FIELD_RENAME` time-indexed vars
     (existing logic — unchanged).
  2. Run the existing `n_time` inference + ssh zero-fill
     (existing logic — unchanged).
  3. Construct: `env = cls(mesh=mesh, full_fields=full_fields,
     time=ds["time"].values)`.
  4. Load `dist_from_sea` if present in `ds.variables`. Inject
     directly: `env.fields["dist_from_sea"] = arr` and
     `mesh.dist_from_sea = arr`. (See Architecture §"Where it
     lives" for the post-hoc-attribute rationale.)
  5. If absent: emit `logging.getLogger("salmon_ibm.h3_env").warning(
     "%s: dist_from_sea missing from NC ...", ERR_DIST_FROM_SEA_MISSING)`
     and zero-fill: `env.fields["dist_from_sea"] = np.zeros(N,
     dtype=np.float32)`; `mesh.dist_from_sea = ...` same.
  6. Return env.

  Because step 4/5 happens AFTER `cls(...)` and writes directly
  to `env.fields`, `advance()` never touches the entry. The
  `(N,)` shape stays intact across all timesteps.
- `salmon_ibm/movement.py` — `_step_directed_numba` and
  `_step_directed_vec` UPSTREAM/DOWNSTREAM paths read
  `fields["dist_from_sea"]` instead of `fields["ssh"]`. The
  dispatch block spans **lines 94-127** (UPSTREAM at 94-109,
  DOWNSTREAM at 111-127); both call sites must be updated. Also
  flip the `ascending` flag at lines 107 and 125 per the
  Architecture-section note. The `ssh` field stays in the env
  (other code may still reference it); not worth the cleanup
  churn in C4's scope.

### Validation discipline

**Build-time** (in `build_h3_multires_landscape.py`):

1. Assert `len(S) > 0` (mesh has at least one OpenBaltic water cell).
2. Run Dijkstra.
3. Assert every water cell has a finite `dist_from_sea`. If not:
   collect unreachable cells, group by reach, raise a
   `RuntimeError` listing reach name + cell count. This catches
   disconnected sub-graphs (e.g., a river cluster with no neighbor
   path to the sea) at build time rather than at sim time.
4. Print per-reach distribution: `f"{reach}: min={...:.0f}m
   max={...:.0f}m mean={...:.0f}m"`. The expected pattern is
   `OpenBaltic ≪ BalticCoast ≪ CuronianLagoon ≪ delta branches ≪
   Nemunas`. A reach that doesn't fit the order signals a
   topology problem.

**Sim-init** (in `H3Environment.from_netcdf`):

The build-time validations only run when the build script is
re-executed; the production NC is committed pre-built and CI
does not rebuild it. Sim-init MUST therefore mirror the
correctness checks against the *loaded* `dist_from_sea` array,
or a stale/corrupt NC ships silently to production.

**Two distinct cases, deliberately handled differently:**

**Case A — `dist_from_sea` variable absent from NC** (backward-
compat with pre-C4 NCs):

The 4-step Case A sequence (executed in `from_netcdf` AFTER the
existing `cls(...)` construction; never as part of `full_fields`):

1. Emit `logging.getLogger("salmon_ibm.h3_env").warning(
   "%s: dist_from_sea missing from NC; movement gradient will be
   flat — agents will not migrate. Rebuild landscape with
   build_h3_multires_landscape.py.", ERR_DIST_FROM_SEA_MISSING)`.
2. `env.fields["dist_from_sea"] = np.zeros(N_cells, dtype=np.float32)`.
3. `mesh.dist_from_sea = env.fields["dist_from_sea"]` (same array,
   not a copy — both references must stay in sync if the array is
   later replaced by a hot-reload mechanism).
4. **Initialize the per-env latch flag:**
   `env._dormant_gradient_check_done = False`. Without this, the
   sim-time `_check_dormant_gradient` helper (which reads
   `getattr(env, "_dormant_gradient_check_done", True)`) will see
   the missing attribute, default to `True`, and silently skip
   the dormancy raise — defeating the entire backward-compat
   safety net for Case A.

This case is graceful by design: an old NC predating C4 should
load and run — just with dormant movement, identical to its
pre-C4 behavior. The sim-time raise (gated by step 4's flag)
will catch any dormant-and-directed-agents combination explicitly.

**Case B — `dist_from_sea` variable present but structurally
invalid** (corrupt build, schema regression, post-C4 NC defect):

Each failed check **RAISES** `RuntimeError(<err-id>: <details>)`
at sim init. No zero-fill, no graceful continuation — a
post-C4 NC that fails any structural check indicates a build
defect that won't self-heal during the sim. Fail fast at init:

a. `arr.shape == (mesh.N_cells,)` else raise
   `ERR_DIST_FROM_SEA_SHAPE_MISMATCH` — stale NC built against
   different mesh.
b. `np.all(np.isfinite(arr[mesh.water_mask]))` else raise
   `ERR_DIST_FROM_SEA_NAN_ON_WATER` — NaN/Inf on water cells.
   (NaN land cells are expected per the spec data structure
   and do NOT trigger this.)
c. `arr.max() > 0` else raise `ERR_DIST_FROM_SEA_ALL_ZERO` —
   build ran but failed mid-Dijkstra.
d. `np.any(arr[mesh.reach_id == OpenBaltic_id] == 0)` else raise
   `ERR_DIST_FROM_SEA_NO_SOURCES` — no source cells.

Where `OpenBaltic_id = mesh.reach_names.index("OpenBaltic")`
(direct list lookup; raises `ValueError` if "OpenBaltic" isn't
in `reach_names`, which itself indicates a non-Baltic mesh and
should skip the entire `dist_from_sea` validation block — a
non-Baltic mesh has nothing to validate against). Pin this
lookup explicitly in the implementation; do NOT use a stale
constant or a separate id-mapping dict.

On all checks passing, the Case B 3-step injection (mirroring
Case A steps 2-4):

1. `arr32 = arr.astype(np.float32)` (single cast; reused).
2. `env.fields["dist_from_sea"] = arr32`;
   `mesh.dist_from_sea = arr32` (same array reference, not
   independent copies — see Case A rationale).
3. `env._dormant_gradient_check_done = False` — per-instance
   latch flag for the sim-time check. Same purpose and same
   default as Case A step 4. Both paths MUST initialize this
   attribute so the sim-time helper's `getattr(env,
   "_dormant_gradient_check_done", True)` default-True branch is
   never reachable on a properly-initialized env.

All err-id constants live in `salmon_ibm/h3_env.py` as module-
level strings, mirroring `ERR_HOMING_HATCHERY_NO_DISPATCH` in
`delta_routing.py:43`:

```python
ERR_DIST_FROM_SEA_MISSING = "dist-from-sea-missing"
ERR_DIST_FROM_SEA_SHAPE_MISMATCH = "dist-from-sea-shape-mismatch"
ERR_DIST_FROM_SEA_NAN_ON_WATER = "dist-from-sea-nan-on-water"
ERR_DIST_FROM_SEA_ALL_ZERO = "dist-from-sea-all-zero"
ERR_DIST_FROM_SEA_NO_SOURCES = "dist-from-sea-no-sources"
```

**Reconciliation rationale:** Case A (missing variable) is a
*backward-compat* path — an old NC that predates C4. Sim should
run with degraded movement so legacy scenarios still execute.
Case B (present-but-invalid) is a *correctness defect* — a
post-C4 build that produced bad data. No legacy reason to
tolerate it; raise at init and force a rebuild. The two cases
look superficially similar but have opposite remediation paths.

**Sim-time:** the sim-init Case A path leaves `dist_from_sea`
as flat-zeros. To avoid a deployed app oscillating silently
when sim-init logging is suppressed, add a per-env latched check
in `salmon_ibm/movement.py`:

```python
def _check_dormant_gradient(landscape, buckets):
    """Raise once per env-instance if dist_from_sea is flat-zero
    AND any agent is in directed (UPSTREAM/DOWNSTREAM) behavior."""
    env = landscape.get("env")  # H3Environment instance
    if env is None or getattr(env, "_dormant_gradient_check_done", True):
        return  # legacy non-Baltic env, or check already run
    has_directed = (
        buckets.get(int(Behavior.UPSTREAM)) is not None
        or buckets.get(int(Behavior.DOWNSTREAM)) is not None
    )
    if has_directed and not np.any(landscape["fields"]["dist_from_sea"]):
        env._dormant_gradient_check_done = True  # latch BEFORE raise to avoid loop
        raise RuntimeError(
            f"{ERR_DIST_FROM_SEA_MISSING}: dist_from_sea is "
            "flat-zero AND agents are in UPSTREAM/DOWNSTREAM "
            "behavior. Movement will not progress (legacy SSH=0 "
            "dormant state). Rebuild the landscape NC with "
            "build_h3_multires_landscape.py to populate dist_from_sea."
        )
    env._dormant_gradient_check_done = True  # latch on happy path too
```

Called once per `execute_movement` invocation, before the
behavior dispatch. The latch is **per-env-instance**
(`env._dormant_gradient_check_done`), NOT module-global. This
prevents cross-test leakage: a test that loads a degraded env,
then a healthy env, will check each independently. The
`_dormant_gradient_check_done` attribute is initialized to
`False` in `from_netcdf` after a successful load (Case B all-
checks-pass) AND in the Case A fallback path (after the warn +
zero-fill). It is never set elsewhere; future code that injects
a new `dist_from_sea` array onto an existing env should reset
this flag.

The check is **landscape-aware** via `landscape.get("env")` —
the env reference must therefore be added to the landscape
dict at sim-step time. Two updates required in `simulation.py`,
both per the CLAUDE.md convention "`Landscape` TypedDict in
`simulation.py` defines the event context schema. Add new keys
there when extending the landscape dict":

1. **TypedDict definition** (current location around
   `simulation.py:12-33`): add an `env: H3Environment | None`
   field. Use `| None` because TriMesh / HexMesh fallback paths
   construct the landscape without an `H3Environment`.
2. **Landscape construction site** (current `step()` method,
   around `simulation.py:602`): include
   `"env": getattr(self, "env", None)` in the dict. Use
   `getattr` defensively in case any sim path constructs the
   landscape before `self.env` is bound.

If `env` is absent (e.g., a TriMesh / HexMesh fallback with no
`H3Environment`), the check no-ops via the `env is None` guard
— backward-compat with non-Baltic configs.

Why landscape-not-module: the C3.3 spec memory entry explicitly
notes module-level state can leak across in-process landscape
swaps; per-env state is the safer default for any latch-style
check.

## Implementation files

| File                                               | Change type | Notes                                                        |
|----------------------------------------------------|-------------|--------------------------------------------------------------|
| `scripts/build_h3_multires_landscape.py`           | Modify      | Add `compute_dist_from_sea` step + sanity-output             |
| `salmon_ibm/h3_env.py`                             | Modify      | Load `dist_from_sea` (Case A absent / Case B structural validation); err-id constants; per-env latch init |
| `salmon_ibm/movement.py`                           | Modify      | UPSTREAM/DOWNSTREAM read `dist_from_sea` (2 identifier sites at lines 102, 120; ascending flag flips at 107, 125); add `_check_dormant_gradient(landscape, buckets)` helper called once before the dispatch block at line 94 |
| `salmon_ibm/simulation.py`                         | Modify      | Add `"env": self.env` key to the landscape dict at the existing `step()` landscape construction site; add `env: H3Environment \| None` (or equivalent) to the `Landscape` TypedDict per the project convention "`Landscape` TypedDict in `simulation.py` defines the event context schema" (CLAUDE.md). |
| `tests/test_movement_gradient.py` (NEW)            | Create      | Synthetic 10-cell chain + gradient-following assertion       |
| `tests/test_h3_env.py`                             | Modify      | Add load + missing-field warning test                        |
| `tests/test_movement.py` or equivalent             | Modify      | Replace `ssh` references in test fixtures with `dist_from_sea`|
| `data/curonian_h3_multires_landscape.nc`           | Rebuild     | Run the build script with the new step (separate from PR). This is the NC the deployed app uses. |

## Tests

### Unit (new)

**Test 1: linear-chain gradient.** Construct a synthetic 10-cell
**bidirectional** chain mesh with:
- `dist_from_sea = np.arange(10, dtype=np.float32) * 100.0`
  (i.e., `[0, 100, 200, ..., 900]`)
- `water_nbrs = np.full((10, 2), -1, dtype=np.int32)`;
  `water_nbrs[i, 0] = i+1` for `i in 0..8`;
  `water_nbrs[i, 1] = i-1` for `i in 1..9`. Cell 0 has only
  one forward neighbor (slot 0); cell 9 has only one backward
  neighbor (slot 1); interior cells have both.
- `water_nbr_count = np.array([1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
  dtype=np.int32)`.

The chain is bidirectional so Test 2's DOWNSTREAM agent at cell
9 has a valid backward neighbor; Test 1's UPSTREAM agent at cell
0 still picks the higher-gradient (forward) neighbor by gradient
comparison.

Place a single agent at cell 0 in UPSTREAM behavior. Pin
`n_micro_steps_per_cell = np.ones(10, dtype=np.int32)` (one hop
per cell — the directed kernel's even-step gradient + odd-step
random pattern means with `n_micro=1` the only hop is even-indexed
and deterministic). Run 1 timestep. Assert the agent ends at cell
1 (climbed one step). Run 5 timesteps. Assert the agent reaches
cell ≥ 5 (climbed at least half the chain).

**Test 2: gradient symmetry for DOWNSTREAM.** Same bidirectional
chain (same fixture as Test 1), same
`n_micro_steps_per_cell = np.ones(10, dtype=np.int32)` pin, agent
at cell 9, DOWNSTREAM behavior. The DOWNSTREAM dispatch picks the
LOWER-gradient neighbor (`ascending=False`); cell 9's only
neighbor is cell 8 (backward), which has a lower `dist_from_sea`
value, so the agent steps to 8. Assert the agent ends at cell
≤ 4 after 5 timesteps.

**Test 2b: mesh-edge fallback.** Same bidirectional chain. Place
an UPSTREAM agent at cell 9 (the maximum-gradient cell — the
mesh edge). Cell 9 has only one neighbor (cell 8) which has a
LOWER `dist_from_sea`. The directed kernel's gradient comparison
finds no strictly-higher neighbor, so it should fall back to the
slot-0 / random selection per the Open-Question 2 RESOLVED
behavior. Run 5 timesteps. Assert: (a) no exception raised;
(b) the agent stayed within `{8, 9}` — confirming the
documented oscillation-near-mesh-edge behavior. This pins the
fallback semantics that the spec relies on at Open Question 2
and at the Interactions section's "post-teleport" failure mode.

**Test 3: zero-gradient fallback (behavioral).** Use the
bidirectional 10-cell chain fixture from Test 1, but override
`dist_from_sea = np.zeros(10, dtype=np.float32)`. Place an UPSTREAM
agent at cell 5. Run 10 timesteps. Assert: (a) no exception is
raised (the kernel handles zero gradient gracefully); (b) the
agent's final cell is in `{4, 5, 6}` — i.e., it visited at most
its immediate neighbors, confirming the dormant-state oscillation
the spec documents. A weaker "does not raise" assertion would
silently pass even if the kernel decided to teleport agents to
arbitrary cells. Note: this test runs WITHOUT a populated `env`
in the landscape (so `_check_dormant_gradient` no-ops via its
`env is None` guard); a separate test (Test 4f below) covers the
"env present + zero gradient + directed agent → raise" path.

**Test 4: missing field warning (Case A).**
`H3Environment.from_netcdf` on an NC without `dist_from_sea`. Use
`caplog.set_level(logging.WARNING, logger="salmon_ibm.h3_env")`.
Assert at least one record contains `dist-from-sea-missing`.
Assert `fields["dist_from_sea"]` exists and is all-zeros. Assert
`env._dormant_gradient_check_done is False` (Case A step 4 latch
init fired).

**Test 4a: Case B shape-mismatch raise.** Construct a synthetic NC
with `dist_from_sea` shaped `(N+5,)` instead of `(N,)`. Assert
`H3Environment.from_netcdf` raises `RuntimeError` whose message
contains `dist-from-sea-shape-mismatch`.

**Test 4b: Case B NaN-on-water raise.** Construct an NC with a
correctly-shaped `dist_from_sea` array, but inject a `NaN` at a
cell where `water_mask=True`. Assert `H3Environment.from_netcdf`
raises `RuntimeError` whose message contains
`dist-from-sea-nan-on-water`. Sanity-check: a NaN at a land cell
(`water_mask=False`) does NOT trigger the raise.

**Test 4c: Case B all-zero raise.** Construct an NC with a
correctly-shaped `dist_from_sea` array but all values = 0.0
(simulating a build that wrote the file but failed mid-Dijkstra).
Assert `RuntimeError` containing `dist-from-sea-all-zero`.

**Test 4d: Case B no-sources raise.** Construct an NC where no
OpenBaltic water cell has `dist_from_sea == 0` (e.g., all
distances ≥ 1). Assert `RuntimeError` containing
`dist-from-sea-no-sources`.

**Test 4e: per-env latch isolation.** Load env-A via the Case A
path (NC missing `dist_from_sea`). Construct a `landscape` dict
with `env=env_A`, an UPSTREAM agent's bucket, and zero-filled
fields. Call `_check_dormant_gradient(landscape, buckets)`;
assert `RuntimeError` with `dist-from-sea-missing` err-id. THEN
load env-B independently (also Case A). Construct a fresh
landscape with `env=env_B`. Assert env-B's
`_dormant_gradient_check_done` is `False` (no leakage from env-A)
AND that calling `_check_dormant_gradient` on env-B ALSO raises
(env-B's check fires independently, not silenced by env-A's
latch). This is the regression test for the pass-7 module-global
→ per-env-instance refactor.

**Test 4f: sim-time happy-path latch.** Load env-C via the Case B
all-checks-pass path (synthetic NC with valid `dist_from_sea`).
Construct a landscape with an UPSTREAM agent. Call
`_check_dormant_gradient`; assert NO raise. Assert
`env_C._dormant_gradient_check_done is True` (latched on the
happy path so subsequent calls are no-ops). Call again; assert
no raise and no work (the early-return-on-latch path).

**Test 5: build-time disconnected-graph check.** Synthetic mesh
with two disconnected components (one with sea, one without).
`compute_dist_from_sea` raises `RuntimeError` naming the
unreachable reach.

**Test 5b: determinism.** Same synthetic mesh. Run
`compute_dist_from_sea` twice and assert
`np.array_equal(out1, out2, equal_nan=True)` — NaN-aware (land
cells carry NaN by spec; `tobytes()` would require byte-identical
NaN payloads which NumPy does not guarantee across runs even when
the *value* is consistent). With at least one tied-distance pair
(two cells equidistant from a single source), this catches
non-deterministic heap-ordering or source-set iteration
regressions. Also run on the production NC's mesh (via the
H3MultiResMesh constructor) and assert
`np.array_equal(saved_nc, recomputed, equal_nan=True)` — pinning
down "committed NC matches what the build script would produce
now". If a future bathymetry / mesh edit changes any cell, this
test fails and forces an explicit NC rebuild commit. **CI wiring:
this test runs UNCONDITIONALLY in the default `pytest tests/`
collection** — not gated behind `@pytest.mark.slow` or
`@pytest.mark.production_data`. If the production NC is missing
locally, the test SKIPS with a clear `pytest.skip("production NC
not available; run `python scripts/build_h3_multires_landscape.py`
to generate it")`, NOT silently passes. Without unconditional CI
wiring, drift between `compute_dist_from_sea` and the committed
NC ships silently — defeating the purpose of the determinism
contract.

**Test 5c: Y-junction tie-break determinism.** Construct a 4-cell
mesh: cell 0 (source, OpenBaltic) connected to cells 1, 2, 3 at
identical haversine distance. **Centroid construction:** place
cell 0 at `(lat=55.0, lon=21.0)`; place cells 1, 2, 3 at
identical latitude offset and longitude offsets that give
bit-identical haversine output — easiest construction is to set
all three at the same lat (`55.0 + δ` for small δ) and use
longitudes computed as `21.0 + n*Δ` for `n in {-1, 0, 1}` and
`Δ` chosen so adjusted lon falls on a great-circle equidistant
from cell 0. A simpler fixture-friendly alternative: monkeypatch
the haversine function in the test to return a fixed scalar `d`
for any centroid pair — sidesteps floating-point geometry while
still exercising the heap tie-break. Use whichever is simpler in
implementation; the test's purpose is determinism, not geometry.
All four cells should compute
`dist_from_sea = [0, d, d, d]` with `d` exact. The interesting
case: cells 1, 2, 3 each have ONE neighbor at the same gradient
(each other, via a back-edge through cell 0). Run
`compute_dist_from_sea` twice; assert byte-equal output. Then run
the directed kernel: place an UPSTREAM agent at cell 0; assert
the agent moves to a DETERMINISTIC neighbor (e.g., cell 1, the
lowest-index — the slot-0 fallback when all gradients tie). This
exercises the heap-tie-break determinism contract from the
Architecture §Determinism subsection AND the kernel's neighbor-
selection determinism, neither of which Test 1 (linear chain)
exercises.

### Integration (new)

**Test 6: end-to-end production-mesh gradient sanity.** Load
`data/curonian_h3_multires_landscape.nc` (rebuilt with the new
step). **First**, assert two NC-rebuilt preconditions:
(i) `dist_from_sea` variable exists in the NC (`"dist_from_sea"
in ds.variables`); without this, the test was run before the
rebuild and should fail with a clear "rebuild the NC first"
error rather than a misleading downstream gradient assertion.
(ii) `dist_from_sea.max() > 0` — distinguishes "rebuilt with
working compute" from "rebuilt with broken compute that wrote
all-zeros". Then assert
`np.all(np.isfinite(dist_from_sea[mesh.water_mask]))` — no
NaN/Inf on water cells. A single NaN poisons `mean()` and
produces confusing "nan < nan = False" failures downstream that
look like topology bugs but are really NC-corruption bugs; the
explicit isfinite check gives a distinct error message. Then assert: (a) `mean(OpenBaltic) <
mean(Nemunas)` — the sanity floor that the gradient points the
right way overall; (b) `min(Nemunas) > mean(OpenBaltic)` — no
Nemunas (river) cell is closer to sea than the typical OpenBaltic
cell, catching gross inversions; (c) every delta-branch cell has
`dist_from_sea > mean(BalticCoast)` — delta cells are inland of
the coastal strip; (d) **per-delta-branch inversion check:** for
each branch in {Atmata, Skirvyte, Gilija}, `min(dist_from_sea[
reach == branch]) > mean(dist_from_sea[reach == OpenBaltic])`.
Catches single-branch inversion (e.g., a polygon-overlay bug
that gives Skirvyte cells 5m while Atmata cells get 5km) which
the reach-aggregate assertions (a)-(c) would miss. The full
chain order (OpenBaltic < BalticCoast < CuronianLagoon < delta <
Nemunas) is NOT asserted because CuronianLagoon spans both the
strait (close to sea) and the eastern shore (far from sea), so
its per-reach mean is bimodal and doesn't fit a strict total
order. Pure data assertion, runs fast.

**Test 7: post-C3.3-teleport invariant.** Lives in
`tests/test_movement_gradient.py` alongside the other unit tests.
Load `data/curonian_h3_multires_landscape.nc` (production mesh,
rebuilt with `dist_from_sea`). For each delta-branch reach
(Atmata, Skirvyte, Gilija): compute the entry cell via
`_branch_entry_cell(mesh, branch_rid)`. **First** assert
`np.isfinite(dist_from_sea[entry])` with a distinct error message
("entry cell N for branch X has non-finite dist_from_sea — NC is
corrupt") — without this guard, a NaN entry cell would produce
"no neighbor has higher dist_from_sea" (since `nbr > nan` is
False) and the failure would read as a topology-degeneracy bug
when really it's an NC-build bug. Then assert at least one of
the entry cell's water neighbors has strictly higher
`dist_from_sea` than the entry cell itself. This guarantees that
a returning adult teleported by C3.3's stray dispatch can
progress inland on the next UPSTREAM step rather than oscillate
at the branch mouth. If the topology assertion fails on the
production mesh, the test surfaces a topology-config defect
(e.g., an entry cell positioned at a confluence where lagoon-side
and inland-side neighbors have similar gradient values). Pure
data assertion; runs in milliseconds.

**Test 7b: teleport-then-step end-to-end behavioral.** Test 7
checks topology; Test 7b checks the actual multi-event sequence.
Set up a Baltic-configured simulation with the production NC. For
each delta branch (Atmata, Skirvyte, Gilija):
1. Construct a hatchery agent with `natal_reach_id = <branch_rid>`
   and place them on the lagoon side of a different branch's
   entry cell. (E.g., natal=Atmata, agent at Skirvyte mouth — so
   they're poised to "stray" upon delta entry per C3.3.)
2. Run `_event_update_exit_branch` with a seeded RNG configured
   to force the stray dispatch to choose the natal branch.
   **Seeding mechanism:** override `landscape["rng"] =
   np.random.default_rng(<seed>)` with a seed that produces the
   intended branch choice for the agent's natal_reach_id under
   the C3.3 dispatch's `rng.choice(...)` call. Find a working
   seed via the existing C3.3 test harness pattern in
   `tests/test_hatchery_c3_3_homing.py::test_hatchery_strays_at_p_zero`
   (which uses `np.random.default_rng(12345)` and forces
   `homing_precision = 0.0` via `monkeypatch`); the same idiom
   transfers here. The teleport fires; agent's `tri_idx` jumps
   to natal-branch's `_branch_entry_cell`.
3. Record `dist_from_sea[pre_teleport_cell]` and
   `dist_from_sea[post_teleport_cell]`.
4. Run one `MovementEvent` step with the agent in UPSTREAM
   behavior.
5. Assert
   `dist_from_sea[final_cell] > dist_from_sea[post_teleport_cell]`
   — the agent advanced inland after teleport, did NOT regress
   to the lagoon. This is the behavioral check that Test 7's
   topology assertion is the prerequisite for; if Test 7 passes
   but Test 7b fails, the kernel's slot-0 fallback won the
   tie-break in a way the topology check didn't catch.

Test 7b is `@pytest.mark.integration` because it composes
`_event_update_exit_branch` + `MovementEvent` against the
production NC. Allow it to be slow (1-2 seconds) — runs in the
default suite.

### Regression (existing)

The full test suite (893 cases at the 2026-05-06 baseline) must
continue to pass. The two tests most likely to drift:

- `tests/test_movement.py::*` — any test fixture mocking the
  landscape `fields` dict needs `dist_from_sea` added (or to
  remain on the SSH-based legacy path explicitly).
- `tests/test_h3_env.py::*` — load tests need to either skip the
  new variable or include it in the test NC.

### Live test (post-deploy)

After deploy, a Playwright run on https://laguna.ku.lt/HexSimPy/
asserts ≥ 1 agent reaches a delta branch (Atmata/Skirvyte/Gilija)
within 480 hours at default settings (50 agents, seed=42). Captured
via `__deckgl_instances.map.lastLayers` inspection of the trips
layer or by polling the agent state. This is the C4 acceptance
test and the ongoing smoke-test for the migration pipeline.

**Calibration sanity assertion** (in addition to ≥ 1 arrival):
median fraction of alive agents reaching at least 50% of the mesh's
`max(dist_from_sea)` by hour 240 should exceed 30%. Free-swim
calculation: ~30 km Baltic-to-delta path at ~1 m/s gives ~8 hours
straight-line; real arrivals with diffusive overhead and odd-step
random hops are expected at 100-300 hours. Catches the
"barely-1-arrival" pathology where the gradient is technically
non-zero but too weak to drive realistic timescales — a passing
acceptance test under that pathology would still leave C3.3
effectively dormant.

**Known limitation: mesh-edge oscillation indistinguishable from
arrival-and-hold.** An agent stuck oscillating in a high-
`dist_from_sea` mesh-edge cell registers as "reached 50% of
max(dist_from_sea)" identically to an agent that successfully
migrated and is holding at its natal reach. The calibration
assertion above does NOT distinguish these two states. Detecting
the oscillation requires per-agent telemetry like
`unique_cells_visited / total_steps` (oscillation: ratio ≪ 1;
real migration: ratio ≈ 1). This telemetry is **deferred** to a
future hardening tier: it requires adding a per-agent counter
field to `AgentPool.ARRAY_FIELDS`, a per-step `unique_cells_seen`
update, and a sim-end summary log. C4 ships without it; the live-
test acceptance criterion is "at least one arrival" plus the
calibration sanity, both of which a well-functioning gradient
satisfies. Oscillation-detection telemetry is tracked in the
follow-up notes (see "Related deferred items").

## Performance

- **Build-time:** Dijkstra on 185k cells × ~6 average degree =
  ~1.1M edges. With a binary heap, expected ~3-5 seconds. Runs
  once per landscape rebuild; landscapes are rebuilt rarely
  (per-NC build is already minutes-scale). Negligible.
- **Sim-init:** Loading one float32[N] variable from NetCDF.
  ~750 KB for the production mesh. Negligible.
- **Sim-step:** identical to the current SSH read path. No new
  per-step cost. The kernel reads a different field; the field
  has the same shape and dtype.
- **Memory:** float32[185k] = 750 KB additional in `fields` dict.
  Negligible.

## Backward compatibility

- **NCs without `dist_from_sea`:** loaded with a
  `logging.warning` at the `salmon_ibm.h3_env` logger (err-id
  `dist-from-sea-missing`), zero-filled. Movement produces a flat
  gradient that the directed kernel handles identically to the
  pre-C4 SSH=0 dormant state. **Note:** post-C4 the legacy `ssh`
  field is no longer read by `_step_directed_*` (movement reads
  `dist_from_sea` instead); the `ssh` field remains in the env
  for non-movement consumers but is dead code from movement's
  perspective. Any future contributor who re-introduces an `ssh`
  reader will silently get a zero-filled array with no test
  coverage — flag this in code review. The sim-init warning makes
  the dormancy visible; the sim-time latched raise (see
  Validation discipline → Sim-time) makes a deployed-with-
  suppressed-logging dormancy unmissable on the first directed
  movement call.
- **Code paths that reference `ssh`:** unchanged. The `ssh` field
  remains in the env, zero-filled by `H3Environment.from_netcdf`
  as before. C4 only changes which field UPSTREAM/DOWNSTREAM
  *reads*; it does not remove `ssh` from the env.
- **Test fixtures that build a synthetic landscape:** if they
  populate `fields["ssh"]` and then run UPSTREAM/DOWNSTREAM
  movement, they need to populate `fields["dist_from_sea"]`
  instead (or in addition). Single search-and-replace pass during
  implementation.
- **External reproducibility:** seeded runs on the same NC produce
  the same trajectories before and after C4 *only if the NC has
  `dist_from_sea` already*. Replicating a pre-C4 published run
  requires either (a) using the legacy NC without
  `dist_from_sea` (and accepting the dormant movement), or
  (b) re-running the analysis with the C4-rebuilt NC.

## Interactions with C1–C3.3

C4 sits underneath the four-tier hatchery-vs-wild architecture. Each
prior tier has at least one interaction worth documenting; some are
*reactivations* of behavior that was dormant pre-C4, not new effects.

### C3.3 (homing precision at delta entry) — primary interaction

C3.3's stray dispatch teleports an agent to `_branch_entry_cell` of
the chosen branch on first delta-branch entry. The teleport target
is the lowest-index swimmable cell of the chosen branch (per the M1
hotfix on branch `c3.3-teleport-water-check`). Under C4 this cell
sits at the *low* end of the chosen branch's `dist_from_sea`
gradient: the branch mouth, just inland of the lagoon. The agent's
next UPSTREAM step climbs the gradient inland; biologically
defensible.

**Failure mode worth a test:** at a confluence/multi-branch node,
the lagoon-side neighbor's `dist_from_sea` could be comparable to
the inland-side neighbor's. The directed kernel's tie-breaking
(slot-0 fallback when no strict-higher neighbor exists) could push
a teleported strayer back toward the lagoon. Their `exit_branch_id`
is now sticky-set, so they won't re-trigger C3.3 — they would
oscillate near the branch mouth instead of progressing inland. C4's
test suite should include a post-teleport invariant: "after a C3.3
stray teleport, the agent has at least one strictly-higher
`dist_from_sea` neighbor available." If this fails for any branch
on the production mesh, the build script should warn (not error —
the agent could still arrive eventually via odd-step random walks).

### C3.3 (homing precision) — natal-homing degeneracy at iso-distance contours

C3.3 only fires *at* delta-branch entry. Before that, an agent's
choice of *which* branch to approach is governed entirely by C4's
gradient. **The gradient does NOT encode branch identity.** Two
delta-branch mouths at similar lagoon-shore positions have similar
`dist_from_sea`; an agent in the lagoon already heading inland sees
zero lateral signal distinguishing Atmata from Skirvyte from Gilija.
This means **a wild fish whose natal branch is Gilija but who
happens to drift toward the Skirvyte mouth will preferentially enter
Skirvyte first** — at which point C3.3's homing dispatch fires and
either (a) homes them to Skirvyte (treating them as a Skirvyte fish
even though they're natal-Gilija), or (b) draws Gilija via the
homing-precision probability and teleports them. The second path
recovers the natal signal. The first path is a known limitation: a
strict reading of C3.3's "first delta-branch entry" plus C4's
identity-blind gradient produces an over-estimate of the
"non-natal-branch first-touch" rate, even for wild fish.

**This is documented as a known limitation, not a bug to fix in
C4.** The proper fix is per-natal-reach gradient (one field per
delta branch, agents look up `gradient_for_reach[natal_reach_id]`)
— deferred to a future tier per the Out-of-scope section. C4's
single global gradient is sufficient for the IBM's current
research goal (population-level wild-vs-hatchery contrast) but not
for individual-fidelity homing.

### C2 (activity multiplier) — semantic reactivation

C2 gives hatchery agents +25% activity multiplier on RANDOM and
UPSTREAM behaviors. **Pre-C4, this multiplier translated to +25%
diffusion rate** because UPSTREAM degenerated to an oscillating
random walk under flat-zero SSH. **Post-C4, the same multiplier
becomes +25% directed swim** — hatchery agents arrive at delta
branches *earlier* than wild agents, by ~25% in expectation.

This is a *reactivation* of existing C2 behavior, not a new effect.
But the operational consequence shifts from "metabolic-cost-only"
to "metabolic-cost + earlier-arrival timing". Downstream analyses
that compare arrival-time distributions should expect a
hatchery-vs-wild shift on the C4-deployed mesh that was absent on
the pre-C4 mesh. Cite this in any results that compare the two
deploys.

### C3.1 (pre-spawn skip) — no direct C4 interaction

C3.1 fires at the reproduction event (post-arrival). C4 affects
*whether and when* an agent reaches the spawning ground; once
there, C3.1's hatchery-skip-probability is unchanged. No spec
update needed for C3.1.

### C3.2 (sea-age) — coupling intentionally absent

C3.2 sets `sea_age` at agent introduction. Real biology: 3SW > 2SW
> 1SW body length → swim speed scales (weakly). The IBM uses
`n_micro_steps_per_cell` from `simulation.py:251`, which depends on
cell edge length, NOT on agent state. **C4 does not couple sea_age
to swim speed** — all agents move identically regardless of
sea_age. This matches the pre-C4 movement model (no behavior change
under C4); coupling sea_age → swim speed is a future tier
("body-size-dependent movement", not in the deferred Curonian-
realism list as of 2026-05-07).

### C1 (origin tag) — required infrastructure

C4's correctness depends on C1's `origin` tag being set at
introduction (so C2/C3.1/C3.3 can dispatch by origin). C4 doesn't
read `origin` directly — the gradient is origin-blind by design.
But C4 unblocks the *observable* expression of the origin-aware
behaviors layered on top.

### Compound (3+ tier) interactions

The Interactions enumeration above covers *pairwise* C4↔Cn
interactions. Compound interactions involving three or more tiers
are NOT enumerated — the combinatorial space is too large for
exhaustive analysis (e.g., C2 +25% activity × C3.2 3SW body size
× C4 directed gradient × thermal-stress timing windows × seasonal
discharge → arrival distribution shift). The integrative check is
the post-deploy live-test arrival distribution. **If hatchery-
vs-wild arrival KDEs diverge unexpectedly post-deploy** (e.g.,
hatchery 2σ earlier than the pairwise C2-only model would
predict), audit the compound stack — typically by toggling
individual tiers off in scenario configs and re-running. C4
itself does not introduce compound-interaction safeguards beyond
the integrative check.

## Open questions

All open questions from earlier drafts are RESOLVED in v3.

1. **Source-set definition: RESOLVED.** Source set = all cells
   where `reach_id == OpenBaltic AND water_mask == True`. Multi-
   source Dijkstra computes "distance to the nearest OpenBaltic
   water cell". An "outer-boundary-only" alternative was
   considered (would give "distance to the open Atlantic"
   semantically) but rejected because (a) the OpenBaltic polygon
   on the deployed mesh already represents the seaward extent
   (everything beyond is land or off-mesh), so the inner cells
   already give a usable nearest-sea reference, and (b) outer-
   boundary detection adds a polygon-geometry step that's not
   warranted given the simpler choice works.

2. **Mesh-edge fallback: RESOLVED.** When an upstream-swimming
   agent reaches the bbox's eastern edge (where the mesh ends)
   and has no neighbor with higher `dist_from_sea`, the directed
   kernel's existing `cnt > 0` + best-not-found path applies:
   the gradient comparison fails (no neighbor strictly higher),
   so `best_nbr` defaults to `water_nbrs[c, 0]` (slot-0 neighbor)
   on even micro-steps and a random neighbor on odd. The agent
   degrades to a random walk in the mesh-edge region —
   biologically defensible (real salmon reach their natal
   tributary somewhere along the river and stop) and matches
   the existing kernel semantics without a code change.

3. **Backward-compat warning channel: RESOLVED in v2.** Use
   `logging.getLogger("salmon_ibm.h3_env").warning(...)` to match
   C3.3's pattern. ERR_ID constant
   `ERR_DIST_FROM_SEA_MISSING = "dist-from-sea-missing"` defined
   in `h3_env.py` alongside the load step.

4. **Test 6's monotonicity ordering: RESOLVED in v2.** Assert
   only `mean(OpenBaltic) < mean(Nemunas)`,
   `min(Nemunas) > mean(OpenBaltic)`, and `delta-cells >
   mean(BalticCoast)`. Strict full-chain ordering dropped
   because CuronianLagoon's per-reach mean is bimodal.

## References

- C1+C2+C3.1+C3.2+C3.3 specs and plans in
  `docs/superpowers/specs/` and `docs/superpowers/plans/`,
  particularly `2026-05-03-hatchery-c3.3-homing-design.md` for
  the layer C4 unblocks.
- `docs/scientific-foundations-hatchery-wild.md` §7 — describes
  C3.3's expected wild-vs-hatchery branch distribution divergence
  (Vasemägi 2005, doi:10.1038/sj.hdy.6800693). C4 is the
  prerequisite for this divergence to be observable in production.
- Movement plan from the v1.5 cohort:
  `docs/superpowers/plans/2026-03-19-movement-animation.md`.
- Diagnostic outputs from 2026-05-06: bathymetry-elev rejection
  (this spec, "Why bathymetry-elev was rejected" section);
  WebSocket payload sizes from Playwright (informational, not
  load-bearing for C4).

## Related deferred items

C4 is one of the 9 Curonian-realism items deferred in
`docs/superpowers/plans/2026-04-24-curonian-realism-upgrades.md`,
though it surfaced not from that plan but from post-C3.3
Playwright observation. Adjacent items that may compose with C4:

- **Real Nemunas EPA discharge** (deferred item #5) — would feed
  current advection magnitudes; combines with C4's directed
  gradient to give the full force balance on a swimming agent.
- **Reach-level habitat attributes** (deferred item #1) — per-cell
  shelter / drift_conc; orthogonal to C4 but part of the same
  movement-realism cluster.
- **Bathymetry data correction** (NOT in the deferred list,
  surfaced by C4's diagnostic) — the EMODnet sampling on the
  production mesh produces non-physical values. C4 sidesteps this
  by computing `dist_from_sea` from topology, but the underlying
  bathymetry issue should be flagged as its own follow-up.
- **Movement oscillation telemetry** (NOT in the deferred list,
  surfaced by C4's pass-6 silent-failure-hunter review) — adds a
  per-agent counter (`unique_cells_visited`, `total_steps`) so the
  ratio distinguishes "successfully migrated and holding" from
  "stuck oscillating at mesh edge or barrier-blocked cell". C4
  ships without it; the calibration sanity assertion in the live
  test cannot tell the two states apart. Adding the telemetry
  requires `AgentPool.ARRAY_FIELDS` extensions and a sim-end log
  step. Future hardening tier; low scope but touches the SoA
  agent state.
