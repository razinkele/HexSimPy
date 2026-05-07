# C4 — Movement Gradient (substrate fix)

**Date:** 2026-05-07
**Owner:** @razinkele
**Status:** ✅ CONVERGED v5 — 5-pass review-loop complete (pass-1 implementation details + ascending inversion; pass-2 advance() trap + Nemunas pipeline; pass-3 cross-tier interactions; pass-4 implementation-readiness; pass-5 verification — no new issues, all findings closed). Implementation-ready. Awaiting writing-plans.

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

1. If `dist_from_sea` variable is present, load it.
2. If absent: emit `logging.getLogger("salmon_ibm.h3_env").warning(
   "%s: dist_from_sea missing from NC; movement gradient will be
   flat — agents will not migrate. Rebuild landscape with
   build_h3_multires_landscape.py to populate it.",
   ERR_DIST_FROM_SEA_MISSING)` (using the err-id constant defined
   in `h3_env.py` alongside the load step).
3. Zero-fill as fallback (preserves the legacy SSH=0 behavior; no
   crash, just visible at sim init).

**Sim-time:** no new validation. The directed kernel already handles
zero-gradient gracefully (degenerates to slot-0 neighbor selection,
which is the legacy behavior; documented as the broken state if
`dist_from_sea` is missing).

## Implementation files

| File                                               | Change type | Notes                                                        |
|----------------------------------------------------|-------------|--------------------------------------------------------------|
| `scripts/build_h3_multires_landscape.py`           | Modify      | Add `compute_dist_from_sea` step + sanity-output             |
| `salmon_ibm/h3_env.py`                             | Modify      | Load `dist_from_sea`; warn + zero-fill if missing            |
| `salmon_ibm/movement.py`                           | Modify      | UPSTREAM/DOWNSTREAM read `dist_from_sea` (2 identifier sites)|
| `tests/test_movement_gradient.py` (NEW)            | Create      | Synthetic 10-cell chain + gradient-following assertion       |
| `tests/test_h3_env.py`                             | Modify      | Add load + missing-field warning test                        |
| `tests/test_movement.py` or equivalent             | Modify      | Replace `ssh` references in test fixtures with `dist_from_sea`|
| `data/curonian_h3_multires_landscape.nc`           | Rebuild     | Run the build script with the new step (separate from PR). This is the NC the deployed app uses. |

## Tests

### Unit (new)

**Test 1: linear-chain gradient.** Construct a synthetic 10-cell
chain mesh with:
- `dist_from_sea = np.arange(10, dtype=np.float32) * 100.0`
  (i.e., `[0, 100, 200, ..., 900]`)
- `water_nbr_count = np.ones(10, dtype=np.int32)`
  (one neighbor per cell — unidirectional chain)
- `water_nbrs = np.full((10, 1), -1, dtype=np.int32);
  water_nbrs[:9, 0] = np.arange(1, 10)`
  (cell `i` has exactly one neighbor: cell `i+1`; cell 9 is a
  dead-end with no neighbor)

Place a single agent at cell 0 in UPSTREAM behavior. Pin
`n_micro_steps_per_cell = np.ones(10, dtype=np.int32)` (one hop
per cell — the directed kernel's even-step gradient + odd-step
random pattern means with `n_micro=1` the only hop is even-indexed
and deterministic). Run 1 timestep. Assert the agent ends at cell
1 (climbed one step). Run 5 timesteps. Assert the agent reaches
cell ≥ 5 (climbed at least half the chain).

**Test 2: gradient symmetry for DOWNSTREAM.** Same chain, same
`n_micro_steps_per_cell = np.ones(10, dtype=np.int32)` pin, agent
at cell 9, DOWNSTREAM behavior. Assert the agent ends at cell
≤ 4 after 5 timesteps.

**Test 3: zero-gradient fallback.** Mesh with `dist_from_sea =
zeros(N)`. UPSTREAM agent. Assert it does NOT raise; movement
degenerates to slot-0 selection (legacy broken-but-not-crashing
behavior). Document the expected dormancy.

**Test 4: missing field warning.** `H3Environment.from_netcdf` on
an NC without `dist_from_sea`. Use `caplog.set_level(
logging.WARNING, logger="salmon_ibm.h3_env")`. Assert at least
one record contains `dist-from-sea-missing`. Assert
`fields["dist_from_sea"]` exists and is all-zeros.

**Test 5: build-time disconnected-graph check.** Synthetic mesh
with two disconnected components (one with sea, one without).
`compute_dist_from_sea` raises `RuntimeError` naming the
unreachable reach.

### Integration (new)

**Test 6: end-to-end production-mesh gradient sanity.** Load
`data/curonian_h3_multires_landscape.nc` (rebuilt with the new
step). Assert: (a) `mean(OpenBaltic) < mean(Nemunas)` — the
sanity floor that the gradient points the right way overall;
(b) `min(Nemunas) > mean(OpenBaltic)` — no Nemunas (river) cell
is closer to sea than the typical OpenBaltic cell, catching gross
inversions; (c) every delta-branch cell has `dist_from_sea >
mean(BalticCoast)` — delta cells are inland of the coastal strip.
The full chain order (OpenBaltic < BalticCoast < CuronianLagoon
< delta < Nemunas) is NOT asserted because CuronianLagoon spans
both the strait (close to sea) and the eastern shore (far from
sea), so its per-reach mean is bimodal and doesn't fit a strict
total order. Pure data assertion, runs fast.

**Test 7: post-C3.3-teleport invariant.** Lives in
`tests/test_movement_gradient.py` alongside the other unit tests.
Load `data/curonian_h3_multires_landscape.nc` (production mesh,
rebuilt with `dist_from_sea`). For each delta-branch reach
(Atmata, Skirvyte, Gilija): compute the entry cell via
`_branch_entry_cell(mesh, branch_rid)`; assert at least one of
the entry cell's water neighbors has strictly higher
`dist_from_sea` than the entry cell itself. This guarantees that
a returning adult teleported by C3.3's stray dispatch can
progress inland on the next UPSTREAM step rather than oscillate
at the branch mouth. If the assertion fails on the production
mesh, the test surfaces a topology-config defect (e.g., an entry
cell positioned at a confluence where lagoon-side and inland-side
neighbors have similar gradient values). Pure data assertion;
runs in milliseconds.

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
  `dist-from-sea-missing`), zero-filled. Movement degenerates to
  legacy SSH=0 behavior (oscillation in place). The warning makes
  the dormancy visible at sim init rather than silent.
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
