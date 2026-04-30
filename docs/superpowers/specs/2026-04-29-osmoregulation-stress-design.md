# Salinity-driven osmoregulation stress for *Salmo salar*

**Date:** 2026-04-29
**Owner:** @razinkele
**Status:** 📋 DRAFT — awaiting writing-plans

This is the first of four queued Curonian-realism deferred items
(ordered: A osmoregulation → C hatchery vs wild → B predation → E habitat
realism). Each gets its own spec → plan → ship cycle.

## Why now

The v1.7.1 memory note flagged this exact gap: *"plan wires salinity
into `salinity_cost()` only; ion-balance turnover / osmoregulation is
still Chinook defaults, not defensible for S. salar."* Today's
lipid-first catabolism fix (commit `4247a11`) makes physiology the most
recently-touched code surface.

What the memory got slightly wrong: the existing parameter values are
already Baltic-flavored (`S_opt=0.5, S_tol=6.0`), but the function
**shape** is Pacific-salmon-style. It returns cost = 1.0 for all
salinities below `S_opt + S_tol = 6.5` PSU, including freshwater (0
PSU). For *Salmo salar*, blood is iso-osmotic at ~10 PSU, so freshwater
*does* impose an osmoregulation cost (hypo-osmotic stress: excrete
water, retain ions). The threshold-linear shape models marine stress
correctly and freshwater stress as zero, which is wrong.

This plan replaces the *shape*, not just the numbers.

## Current state (verified 2026-04-29)

- Function: `salmon_ibm/estuary.py:49-58`
- Signature: `salinity_cost(salinity, S_opt=0.5, S_tol=6.0, k=0.6, max_cost=5.0)`
- Logic: `excess = max(salinity - (S_opt + S_tol), 0); return min(1.0 + k*excess, max_cost)` — threshold-linear, NaN-safe via `np.where(np.isnan, 0)`.
- Parameters supplied via YAML config (`est.salinity_cost.{S_opt, S_tol, k}`) at two call sites:
  - `salmon_ibm/events_builtin.py:83` (the `SurvivalEvent`).
  - `salmon_ibm/simulation.py:481` (the `_event_bioenergetics` handler — an active event in the simulation event loop, NOT a legacy/parallel path).
- Output is the `salinity_cost` ndarray fed into `update_energy()` in `bioenergetics.py:65`.
- `EstuaryParams` dataclass (`estuary.py:11-28`) declares `s_opt: 0.5` and `s_tol: 6.0` but the function takes its own defaults — `EstuaryParams` is currently **not** wired through to `salinity_cost()`.

## Scope

**In:**
- Replace the body of `salmon_ibm/estuary.py::salinity_cost` with a
  linear-with-anchors function modeling iso-osmotic physiology.
- Replace the function's parameter set: drop `S_opt, S_tol, k, max_cost`
  (Pacific-style threshold-linear), introduce
  `iso, hyper_cost, hypo_cost` (S. salar iso-osmotic with asymmetric
  costs).
- Move parameters onto `EstuaryParams` (which currently has unused
  `s_opt, s_tol`); add validation in `__post_init__`.
- Update the 2 call sites to pass the new parameters from
  `EstuaryParams`.
- Migrate 5 YAML configs to the new schema.
- Update `tests/test_estuary.py` to test the new shape.
- Add 8 new tests (5 functional + 3 validation).

**Out:**
- Calibration to field data (V1: literature-only validation).
- Ion-balance state on agents (gill ATPase activity tracking, acute
  transition spikes, recovery dynamics) — A3 ambition tier; deferred
  indefinitely unless a use case surfaces.
- Salinity data plumbing — already in place via CMEMS forcing →
  `H3Environment` → `landscape["salinity"]`.
- Comparison to other *S. salar* IBMs (inSTREAM-Baltic, Säterberg 2023)
  — possible future plan.
- `BioParams` / `BalticBioParams` changes — these dataclasses are about
  Wisconsin bioenergetics (RA, RB, RQ, ED_TISSUE, etc.); osmoregulation
  is an estuarine stressor, so it lives on `EstuaryParams`.

## Choice of approach

Three functional-form options were considered:

1. **Linear-with-anchors** — two slope parameters anchored to specific
   published values (selected).
2. **Asymmetric tanh-saturating** — four parameters; smoother edges but
   two of the four have no clear literature anchor.
3. **Lookup-table** — same shape as (1) precomputed at fixed bins;
   adds YAML maintenance overhead with no expressiveness gain.

**(1) selected** because every parameter has a single, named citation;
salinity is naturally bounded so saturation isn't required; smallest
defensible change.

## Functional form

The function takes the salinity array plus a single `params:
EstuaryParams` dataclass (matches the `update_energy(..., params:
BioParams)` precedent) instead of multiple scalar keyword args:

```python
def salinity_cost(
    salinity: np.ndarray,
    params: EstuaryParams,
) -> np.ndarray:
    """Osmoregulation cost multiplier on respiration for S. salar.

    Linear with separate slopes for hyper-osmotic (above iso) and
    hypo-osmotic (below iso) stress, anchored to literature values
    for blood iso-osmotic point and marine/freshwater respiration
    increments.

    Returns: multiplier ≥ 1.0; equals 1.0 at salinity == iso.
    """
    iso = params.salinity_iso_osmotic
    hyper = params.salinity_hyper_cost
    hypo = params.salinity_hypo_cost
    safe = np.where(np.isnan(salinity), iso, salinity)  # NaN → iso (cost 1.0)
    s = np.clip(safe, 0.0, 35.0)
    above = np.maximum(s - iso, 0.0) / max(35.0 - iso, 1.0)
    below = np.maximum(iso - s, 0.0) / max(iso, 1.0)
    return 1.0 + hyper * above + hypo * below
```

The function is pure (no I/O), vectorised, and stable for the input
range CMEMS produces. NaN handling preserves the existing behavior of
"NaN → no cost" but routes it through iso (semantically clearer than
defaulting to 0).

The output is no longer capped (no `max_cost`) because the linear-from-iso
form has a natural maximum of `1.0 + hyper_cost ≈ 1.30` at full marine
salinity. The previous `max_cost=5.0` cap existed because the
threshold-linear form could grow unbounded for arbitrary salinity input.

**Why the signature change:** the existing function takes 4 scalar
keyword args (`S_opt, S_tol, k, max_cost`); the call sites build them
from a YAML dict. This means `EstuaryParams` (which already exists with
`s_opt, s_tol`) is **not wired through** to the function — it's
decorative and tested but its values never reach production code.
Switching to `(salinity, params: EstuaryParams)` wires it through
properly and makes `__post_init__` validation real (vs theatrical), at
the cost of a small refactor at the two call sites and a YAML schema
migration.

## Parameters

Three new fields on `EstuaryParams`. Two old fields (`s_opt, s_tol`)
are removed — they referred to a Pacific-salmon physiology that doesn't
apply to *S. salar*.

Note: `EstuaryParams.s_opt` and `EstuaryParams.s_tol` are **currently
dead code** — they're declared on the dataclass but `salinity_cost()`
takes its own function defaults (also `S_opt=0.5, S_tol=6.0`) without
reading from `EstuaryParams`. So removing them is safer than it sounds:
no caller depends on them. The YAML config layer (`est.salinity_cost.{S_opt, S_tol, k}`)
*is* live and feeds the function; that's what the migration handles.

| Field | Default | Citation | Validation |
|---|---|---|---|
| `salinity_iso_osmotic` | `10.0` ppt | Wilson 2002 — *S. salar* blood plasma iso-osmolality is in the 9-12 ppt range | `0 < iso < 35` |
| `salinity_hyper_cost` | `0.30` (verify) | Brett & Groves 1979 — chapter reports ~25-35% above iso-osmotic respiration at full marine salinity for euryhaline salmonids; **plan task: verify exact number from Table 8.x of the chapter and update default if needed** | `0 ≤ x ≤ 1` |
| `salinity_hypo_cost` | `0.05` (verify) | Brett & Groves 1979 — chapter reports ~5-10% above iso-osmotic respiration at freshwater for euryhaline salmonids; the asymmetry reflects hyper-osmotic stress being more energetically expensive than hypo. **Plan task: verify exact number** | `0 ≤ x ≤ 1` |

Validation lives in `EstuaryParams.__post_init__`, raising `ValueError`
on out-of-range values. Currently `EstuaryParams` has no `__post_init__`
— this plan adds one. Validation covers the three new salinity fields
**only**; validation for the existing `do_lethal`, `do_high`,
`seiche_threshold_m_per_s` fields is deferred (see Out-of-scope).

### Citations

- **Wilson, J. M., & Laurent, P. (2002).** Fish gill morphology: inside
  out. *Journal of Experimental Zoology*, 293(3), 192–213.
  https://doi.org/10.1002/jez.10124 — reviews iso-osmolality; *S. salar*
  blood plasma is ~340 mOsm ≈ 10-12 ppt.
- **Brett, J. R., & Groves, T. D. D. (1979).** Physiological energetics.
  In *Fish Physiology* (Vol. 8). Academic Press, pp. 279–352. — chapter
  reports salinity effects on respiration for migratory salmonids;
  ~25-35% increase at marine vs iso, ~5-10% at freshwater. Implementation
  plan must verify exact values from the chapter and adjust defaults
  accordingly.

## Architecture

The change is wider than originally scoped — touches one function body,
one dataclass, two call sites, five YAML configs, two documentation
files. No new modules, no new files.

**Files modified:**

1. `salmon_ibm/estuary.py`
   - `salinity_cost()` body + signature replaced.
   - `EstuaryParams` extended with 3 new fields, `s_opt` and `s_tol`
     fields removed, `__post_init__` added.
2. `salmon_ibm/events_builtin.py:83` — call site builds an
   `EstuaryParams` instance from the YAML `estuary.salinity_cost`
   subsection and passes it to `salinity_cost(sal, est_params)`.
   Likely best to construct `EstuaryParams` once at `SurvivalEvent`
   init rather than per-step (perf-friendly, also surfaces validation
   errors at scenario-load time, not first step).

   **Partial population:** the salinity_cost YAML key only carries the
   3 salinity fields. The other `EstuaryParams` fields (`do_lethal,
   do_high, seiche_threshold_m_per_s`) will take their dataclass
   defaults — which preserves current behaviour exactly, since the
   existing code also doesn't flow DO/seiche YAML values through to
   `EstuaryParams` (it reads `do_avoidance.{lethal,high}` and
   `seiche_pause.dSSHdt_thresh_m_per_15min` directly into the relevant
   call sites, bypassing the dataclass). Wiring those YAML keys through
   `EstuaryParams` is a separate concern and is in this plan's
   Out-of-scope list.
3. `salmon_ibm/simulation.py:481` — same pattern; this is the
   `_event_bioenergetics` event handler (active, not legacy).
4. `tests/test_estuary.py` — 4 existing salinity tests rewritten or
   deleted for the new shape; 8 new tests added.
5. `tests/test_config.py:26`, `tests/test_ensemble.py:19`,
   `tests/test_simulation.py:235` — three test files with hardcoded
   old YAML schema; migrate fixtures to new field names.
6. YAML configs (5 files): `config_columbia.yaml`,
   `config_curonian_minimal.yaml`,
   `configs/config_curonian_trimesh.yaml`,
   `configs/config_curonian_baltic.yaml`,
   `config_curonian_hexsim.yaml` — replace
   `salinity_cost: {S_opt, S_tol, k}` block with
   `salinity_cost: {salinity_iso_osmotic, salinity_hyper_cost, salinity_hypo_cost}`.
   Columbia uses the "disable" pattern (currently `S_tol: 999`); migrate
   to `salinity_hyper_cost: 0.0, salinity_hypo_cost: 0.0` for the same
   effect.
7. Documentation: `docs/api-reference.md` and `docs/model-manual.md` —
   update the function signature reference and YAML schema examples.

Total: 1 module (estuary.py), 2 production call sites
(events_builtin.py, simulation.py), 4 test files (test_estuary.py +
3 fixture-only updates), 5 YAML configs, 2 doc files. **14 files.**

The salinity field per agent already flows from CMEMS forcing through
the env model. No data wiring changes.

## Migration

The old YAML schema (`S_opt`, `S_tol`, `k`) is removed without
backward-compat — the parameter semantics are scientifically wrong for
*S. salar* (zero freshwater cost), so silently mapping old values to
new ones would preserve a bug. Following the lipid-first precedent
(commit `4247a11`, which deleted the proportional-mass formula
without a compat shim).

For any user with custom configs not in this repo:
- `S_opt` and `S_tol` no longer have meaning. Closest equivalent is
  `iso` (the salinity at which cost is minimum).
- `k` (slope above threshold) maps roughly to `hyper_cost` after
  appropriate rescaling, but the math is not 1:1.
- `max_cost` no longer exists — the new function's output is naturally
  bounded by `1 + hyper_cost` since salinity is clipped to [0, 35].
- Migration documented in the implementation plan's commit message;
  default values produce the published *S. salar* curve, so most users
  can simply delete their `salinity_cost` block from YAML.

**Special case — disabling salinity cost entirely.** `config_columbia.yaml`
currently uses `S_tol: 999` to effectively disable salinity cost (the
threshold becomes unreachable, so cost stays at 1.0). The new
equivalent: set both `salinity_hyper_cost: 0.0` and `salinity_hypo_cost:
0.0`, which makes cost = 1.0 for all salinities. The Columbia config
should be migrated to that pattern (Columbia is freshwater throughout,
no osmoregulation cost makes sense for that scenario).

**Note on the executed Curonian-realism plan.**
`docs/superpowers/plans/2026-04-24-curonian-realism-upgrades.md:1183`
references the old `salinity_cost(salinity, S_opt, S_tol, k, max_cost)`
signature. **That plan is stamped ✅ EXECUTED and should not be
amended** — its reference was correct at execution time. The reference
will become stale post-migration; future readers wanting current
signatures should consult `salmon_ibm/estuary.py` directly. This is
the standard convention: executed plans are historical records.

## Tests

### Existing tests — review and update

`tests/test_estuary.py` currently tests the threshold-linear shape and
includes `test_estuary_params_defaults_match_liland_2024` (line ~79)
which asserts `EstuaryParams()` defaults. The implementation plan will:

- Identify which tests assert the `cost = 1.0 below threshold` behavior
  and either rewrite them (now `cost > 1.0 in freshwater`) or delete
  them.
- Preserve any tests that aren't shape-dependent (NaN-handling,
  vectorisation, monotonicity outside iso).
- Update `test_estuary_params_defaults_match_liland_2024` to drop
  assertions about `s_opt, s_tol` (removed) and add assertions about
  the new `salinity_iso_osmotic, salinity_hyper_cost, salinity_hypo_cost`
  defaults.

The other 5+ test files referencing `salinity_cost` need plan-time
updates. Verified by direct grep against the codebase 2026-04-29:

**Files with hardcoded old YAML schema or kwargs (need updating):**

- `tests/test_config.py:26` — asserts
  `cfg["estuary"]["salinity_cost"]["S_opt"] == 0.5`. **Break** —
  S_opt no longer in schema. Rewrite to assert `salinity_iso_osmotic`.
- `tests/test_ensemble.py:19` — fixture dict
  `{"S_opt": 0.5, "S_tol": 999, "k": 0.0}` (the "disable salinity
  cost" pattern). **Break** — old keys ignored. Rewrite as
  `{"salinity_hyper_cost": 0.0, "salinity_hypo_cost": 0.0}`.
- `tests/test_simulation.py:235` — same fixture pattern as ensemble.
  **Break** — same fix.

**Files that may need indirect updates:**

- `tests/test_bioenergetics.py` — uses `salinity_cost` indirectly via
  `update_energy()`. May or may not break depending on what it asserts.
- `tests/test_h3_multires_integration.py`,
  `tests/test_curonian_realism_integration.py` — integration tests;
  may shift on Baltic-cost change documented above. Verify during
  implementation.

Full suite run will surface any remaining breaks; the implementation
plan must budget for fixing all 3 hardcoded-schema sites + triaging
the integration-assertion breaks.

### New tests in `tests/test_estuary.py`

All function-level tests construct an `EstuaryParams()` (defaults) and
pass it as the second argument; the new signature requires it.

Five functional:

1. `test_salinity_cost_at_iso_returns_unity` — `salinity_cost(np.array([10.0]), EstuaryParams())[0] == pytest.approx(1.0)`. Lock the iso anchor.
2. `test_salinity_cost_marine_matches_brett_groves` — at salinity=35 with default params, cost ≈ 1.30 (or whatever the verified Brett & Groves number is).
3. `test_salinity_cost_freshwater_above_one` — at salinity=0 with default params, cost > 1.0 and strictly less than the marine cost. Locks the asymmetry.
4. `test_salinity_cost_smooth_monotonic_outside_iso` — sweep 0..35; cost is monotonic non-decreasing as |salinity − iso| grows.
5. `test_salinity_cost_handles_nan` — input `np.array([np.nan, 10.0, 35.0])` returns `[1.0, 1.0, 1+hyper]` (NaN treated as iso → cost 1.0).

Three validation in a new `TestEstuaryParamsValidation` class:

6. `test_negative_iso_raises` — `EstuaryParams(salinity_iso_osmotic=-1.0)` raises.
7. `test_iso_above_35_raises` — `EstuaryParams(salinity_iso_osmotic=40.0)` raises.
8. `test_negative_hyper_cost_raises` — `EstuaryParams(salinity_hyper_cost=-0.1)` raises.

**Total:** 8 new tests + ~3 existing tests rewritten.

## Risk + regression surface

**Behavior change surface:** any test or run that exercises agents at
salinities in the Curonian Lagoon (~5 PSU) or Baltic Sea (~7 PSU)
ranges. The two functions diverge most sharply at the boundary
between the old threshold (`S_opt + S_tol = 6.5 PSU`) and the new
iso-osmotic point (10 PSU):

| Salinity | OLD cost | NEW cost | Direction |
|---|---|---|---|
| 0 PSU (freshwater) | 1.00 | ~1.05 | small increase |
| 5 PSU (lagoon) | 1.00 | ~1.025 | tiny increase |
| 7 PSU (Baltic) | **1.30** | ~1.015 | **large decrease** |
| 10 PSU (iso) | 3.10 | 1.00 | very large decrease |
| 35 PSU (full marine) | 5.00 (capped) | ~1.30 | large decrease |

The dominant biological effect is **decreased respiration cost for
Baltic migrants** — the old function was over-penalising Baltic
salinity (which is actually close to *S. salar*'s iso-osmotic point).
Net result: under new physics, Baltic migrants conserve more energy,
likely survive longer, return to spawn at higher condition — the
correct direction relative to *S. salar* physiology.

The lagoon (~5 PSU) sees a slight cost increase (was zero, now ~2.5%);
this is biologically correct (hypo-osmotic stress is real, just small).

**Tests that need updating** (verified by reading `tests/test_estuary.py`):
- `test_salinity_cost_below_tolerance` (line 6) — asserts `cost = 1.0`
  at salinity=3 with old kwargs. **Break** (new cost ≈ 1.035 at 3 PSU).
  Rewrite as a new-physics equivalent.
- `test_salinity_cost_above_tolerance` (line 11) — asserts `cost = 3.1`
  at salinity=10 with old kwargs. **Break** (new cost = 1.0 — that's
  the iso-osmotic minimum). Rewrite to test `cost ≈ 1 + hyper_cost` at
  salinity=35 instead.
- `test_salinity_cost_capped` (line 41) — asserts `cost ≤ 5.0` at
  extreme salinity. **Delete** — no cap in new function (input is
  clipped to [0, 35] so output is bounded by `1 + hyper_cost ≈ 1.30`).
- `test_estuary_params_defaults_match_liland_2024` (line 79) — asserts
  `p.s_opt == 0.5, p.s_tol == 6.0`. **Break** (those fields are
  removed). Rewrite to assert the three new fields' defaults.
- `test_salinity_cost_nan_treated_as_zero` (line 51) — asserts NaN
  → cost=1.0. **Survives** under new physics (NaN → iso → cost=1.0).
  Could be merged with the new `test_salinity_cost_handles_nan`; or
  kept as-is.

Plus possibly `test_curonian_realism_integration.py` if it asserts
specific energy/mortality values post-Baltic-transit (verify during
implementation; large drop in Baltic cost may shift migration outcomes
non-trivially).

**Mitigation:** run full pytest suite before pushing, surface
regressions, fix or update them. Same playbook as v1.7.1 lipid-first.

## Success criteria

- [ ] `salinity_cost()` body + signature replaced with linear-with-anchors form.
- [ ] `EstuaryParams` extended with 3 new fields + `__post_init__`
      validation.
- [ ] Both call sites (`events_builtin.py`, `simulation.py`) updated.
- [ ] All 5 YAML configs migrated.
- [ ] Documentation (`api-reference.md`, `model-manual.md`) updated.
- [ ] All 8 new tests pass; existing salinity tests in `test_estuary.py`
      rewritten or deleted as appropriate.
- [ ] Full pytest suite stays green (current: 815 passing on main;
      expected post-change: ~821-823 passing depending on how many
      existing tests are absorbed/replaced vs added).
- [ ] Manual sanity check: cost curve plotted from 0 to 35 ppt looks
      smooth, has minimum at iso (10 ppt), and matches the published
      *S. salar* shape.
- [ ] Plan stamp on the implementation plan: ✅ EXECUTED.

## Estimated implementation time

**1-2 days** including TDD cycle, YAML migration, doc updates, and
regression sweep. Most of the work is mechanical (YAML/doc sync); the
function rewrite itself is ~10 lines.

## Out-of-scope (deferred)

- **Ion-balance state on agents** (the A3 ambition tier from
  brainstorming). Deferred indefinitely. If a future use case requires
  per-agent acclimation history, that's a separate plan.
- **Field-data calibration** (V2/V3 from brainstorming). Deferred. The
  V1 literature-only path closes the headline critique without a
  calibration phase. If the model later needs to match observed
  Curonian/Baltic migrant energy trajectories, a follow-on plan can
  add calibration.
- **Acute transition spikes.** Modeled via Brett & Groves' chronic-cost
  framing only. Not modeled because IBM-level signals are dominated by
  movement and encounter, not individual transition events.
- **`EstuaryParams` validation for non-salinity fields** (`do_lethal`,
  `do_high`, `seiche_threshold_m_per_s`). Adding these to the new
  `__post_init__` is scope creep; the plan can add them opportunistically
  if it's free, but they're not blocking and not a requirement.
