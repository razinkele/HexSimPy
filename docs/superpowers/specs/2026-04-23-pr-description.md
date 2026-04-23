# PR Description Draft — Deep Review Fixes + Perf Wins

Use when you're ready to push main to origin and open a PR.

**Suggested title:** `Deep codebase review: correctness + security + perf (2.77× + 5.7×)`

**Branch:** `main` (37 commits ahead of origin/main at the time of this draft)

---

## Body (copy below the `---` into the PR)

---

## Summary

Five-pass deep review of the Baltic Salmon IBM followed by 37 commits that land 14 correctness/security/science fixes, two major performance wins, one latent bug caught by end-to-end bench, and complete decision documents for the two remaining modeler-blocked items. Zero test regressions.

### Performance (measured)

| Hot path | Before | After | Speedup |
|---|---|---|---|
| Accumulator expression (50k agents × 74 accs × 200 iters) | 49.5s | 17.8s | **2.78×** |
| Interaction event (500 pred × 5000 prey × 100 cells) | 59.2 ms | 10.4 ms | **5.7×** |
| Full accumulator hot loop (total) | 53.0s | 19.2s | **2.77×** |

### Correctness

- **`bioenergetics.py`** — `MASS_FLOOR_FRACTION` interaction with ED was hiding starvation mortality; surfaced + documented. Implementation blocked on modeler decision (`docs/superpowers/specs/2026-04-23-bioenergetics-starvation-decision.md`).
- **`accumulators.py`** — `IndividualLocations` fallback was writing zeros instead of cell indices; `Transfer` was creating phantom mass when source clamped at `min_val`.
- **`events.py`** — single-pop `EventSequencer.step()` wasn't clearing `_combo_mask_cache` between steps (fixed, parity with multi-pop).
- **`agents.py`** — `push_temperature` was polluting the t3h history for dead agents; now gated via optional `alive_mask`.
- **`agents.py`** — `AgentPool.__init__` now asserts every `ARRAY_FIELDS` entry is initialized, preventing silent field drift.
- **`output.py`** — `log_step` now uses `Population.agent_ids` so cross-step agent tracking survives `compact()`.
- **`hexsim.py`** — `HexMesh` was missing the `centroids_c` property added to `TriMesh` in the centroids-cache perf commit, causing `AttributeError` on Columbia runs. Caught only by `scripts/bench_e2e.py` — unit tests didn't exercise the path. Bug was latent through ~11 commits before the bench surfaced it.

### Security

- **`accumulators.py`** — AST sandbox for scenario-expression `eval()` tightened: `_rng` method calls restricted to `{random, uniform, normal, integers}` allowlist (was any method); literal numeric args to `_rng.*` capped at `_RNG_ARG_MAX = 1_000_000` (prevents `_rng.random(10**9)` memory DoS).
- **`accumulators.py` + `hexsim_expr.py`** — expression caches converted from unbounded `dict` (bulk clear at 10k entries) to `OrderedDict` LRU capped at 256; prevents cache-flood DoS from attacker-authored scenario XML.
- **`events_phase3.py`** — `GeneratedHexmapEvent` substring-match landscape injection replaced with explicit `allowed_landscape_keys` allowlist. Spatial-data keys that would shadow `_SAFE_MATH` (`sqrt`, `clip`, etc.) are now skipped. `output_name` validated against `_PROTECTED_LANDSCAPE_KEYS` (can't overwrite `fields`, `mesh`, `spatial_data`, etc.).

### Science

- **`estuary.py`** — DO thresholds updated from Roegner 2011 (Pacific salmon) to Liland et al. 2024 (Atlantic salmon, *S. salar*-specific): `do_lethal = 3.0`, `do_high = 5.5` mg/L. New `EstuaryParams` dataclass + `validate_do_field_units()` that rejects DO fields in mmol/m³ (CMEMS Baltic format) before they silently pass mg/L-calibrated thresholds as always-OK.

### Performance

- **`accumulators.py`** — `AccumulatorManager.data` transposed from `(n_agents, n_acc)` row-major to `(n_acc, n_agents)` column-major so single-column updater accesses are contiguous. Micro-benchmark: **2.77× total speedup** on the three hot updater patterns (increment, expression, transfer). Decision gate was ≥ 1.3×; target blown past. See `scripts/bench_accumulators.py` + `scripts/bench_probe.py` for baseline capture + post-refactor measurement.
- **`interactions.py`** — `InteractionEvent.execute` replaced its nested Python loop with a vectorized per-cell dice-roll matrix (`rng.random((|A|, |B|)) < p`). "Option 2" (first-A-wins) dedup preserves scalar iteration semantics. **5.7× speedup** on the 500-predator × 5000-prey × 100-cell benchmark. Five new characterization tests lock in Option 2's invariants (p=1 kills all, p=0 kills none, resource conservation, first-A-wins tie-breaking).
- **`mesh.py` + `hexsim.py`** — `centroids_c` `@cached_property` caches the contiguous-array view instead of calling `np.ascontiguousarray(mesh.centroids)` every step.
- **`output.py`** — optional preallocated buffer mode (`max_steps`, `max_agents`) removes ~1-8 GB of list-append + `copy()` per run at 50k-agent scale; list-append mode still the default.

### Tests

- **+50 new tests** covering the fixes and both perf wins (characterization tests for the vectorized interactions, AST-sandbox attack-surface tests, accumulator-layout assertions, transfer-conservation invariants, etc.).
- **`test_numba_fallback.py`** — added parity test that revealed Numba↔NumPy paths diverge ~4% in agent positions due to different RNG consumption order; documented as `xfail` with full explanation rather than silently passing or silently failing.
- **`test_snyder_reference.py`** — replaced two tautology assertions (`0.5 == approx(0.5)`) with formula-evaluating versions that actually test the logistic curve at LC50.
- **`test_playwright.py`** — replaced two consecutive `wait_for_timeout` calls with `expect().to_have_text(timeout=...)` — proves the pause is stable for 2s rather than just sleeping.

### Documentation

- **`docs/superpowers/plans/2026-04-23-deep-review-fixes.md`** — 21-task plan with five revisions (v1 draft → v2 post-review → v3 post-verification → v4 post-security-depth → v5 post-execution-dry-run), full revision history preserved.
- **`docs/superpowers/specs/2026-04-23-bioenergetics-starvation-decision.md`** — modeler-facing decision doc for `ED_TISSUE` (options A: lipid-first 36 kJ/g, B: mixed 25-30, C: retain proportional).
- **`docs/superpowers/specs/2026-04-23-activity-multiplier-decision.md`** — modeler-facing decision doc for activity scaling (options A: Snyder temp-dependent, B: recalibrate RA, C: document-only). Both specs flag the interaction: picking A+A yields the most defensible combined physiology.
- **`docs/superpowers/plans/2026-04-24-accumulator-layout-transpose.md`** + **`2026-04-24-vectorize-interactions.md`** — dedicated sub-plans for the two deferred perf items; the transpose plan was executed and landed here, the vectorize plan was the execution guide for commit `7c3f3ad`.
- **Four benchmark harnesses** — `scripts/bench_accumulators.py`, `bench_probe.py`, `bench_interactions.py`, `bench_e2e.py`, `profile_step.py` — all now in-tree so future perf work has its own baseline-capture infrastructure.

## Still blocked on external decisions

These items are ready to implement but need the modeler's call:

1. **Task 1 — starvation catabolism** (`bioenergetics.py`): pick A/B/C from `2026-04-23-bioenergetics-starvation-decision.md`
2. **Task 17 — activity-by-temperature** (`bioenergetics.py`): pick A/B/C from `2026-04-23-activity-multiplier-decision.md`

## Follow-up observed but not actioned

- **`hexsim_env.advance`** is now 60% of Columbia step time (per `scripts/profile_step.py`). The naive `float32 → float64` dtype change regressed performance (11 → 17 ms/call) because the larger temp-table hurt cache behavior more than the conversion cost; reverted. Real optimization needs deeper design — documented in commit message `6587583`.

## Test plan

- [ ] Full test suite passes (should be ~495 passing, 73 skipped, 1 xfail, 1 pre-existing `pytest-mock` error)
- [ ] Parity tests pass if Columbia `[small]` workspace is present (39/39)
- [ ] Benchmarks reproducible: `micromamba run -n shiny python scripts/bench_accumulators.py` and `bench_interactions.py` show the reported speedups
- [ ] End-to-end bench runs cleanly on both Curonian and Columbia configs: `micromamba run -n shiny python scripts/bench_e2e.py`
- [ ] Review decision docs and either sign off on Task 1/17 options or leave TODOs
