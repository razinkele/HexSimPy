# Hatchery vs Wild C3.1 — Pre-Spawn Skip Probability

**Date:** 2026-05-02
**Owner:** @razinkele
**Status:** 📋 DRAFT — awaiting writing-plans

This is the **third tier, first sub-task** of hatchery-vs-wild origin support.
Tier sequence:

- C1 (shipped, PR #3 awaiting merge) — tag-only origin int8 column.
- C2 (shipped locally, awaits PR #3 merge) — `activity_by_behavior`
  parameter divergence (+25% on RANDOM/UPSTREAM for hatchery).
- **C3.1 (this spec)** — pre-spawn skip probability divergence (hatchery
  reproducers skip pre-spawn with `p_skip = 0.3`; wild always proceed).
  ~2 days.
- C3.2 (future) — sea-age sampling divergence (new `sea_age` field on
  AgentPool + sampling event).
- C3.3 (future) — homing precision divergence (modifies migration /
  delta-routing logic).

C3 was originally projected as a single ~1-2 week tier per the C1 spec.
Splitting into C3.1/C3.2/C3.3 mirrors the C1→C2 cadence (each tier
ships independently in ~2-4 days) and keeps each sub-tier's literature
review focused.

## Why now

C2 made the origin column physically meaningful for swimming
metabolism — hatchery agents pay +25% activity cost during active
swimming. C3.1 makes it physically meaningful for **reproduction** —
hatchery reproducers have lower per-attempt success rate (RRS ≈ 0.7),
modeled as a Bernoulli skip gate before clutch sampling.

Together with C2, this captures two distinct empirical signals: (a)
hatchery fish are mechanically less efficient swimmers (Enders 2004,
Pedersen 2008), and (b) hatchery fish have fewer mating events per
spawning attempt in nature (Bouchard 2022, Christie 2014).

## Scientific basis

**Primary anchor:** Bouchard, Wellband, Lecomte et al. (2022),
*Evolutionary Applications* 15(5):838-852, doi:10.1111/eva.13374 —
*"Effects of stocking at the parr stage on the reproductive fitness
and genetic diversity of a wild population of Atlantic salmon (Salmo
salar L.)"*. Key findings for *S. salar* specifically:

- Captive-bred MSW females and males averaged **~80% RRS**
- Captive-bred 1SW males averaged **~65% RRS**
- **Mechanism: "captive-breeding did not directly affect the number of
  offspring per mating event but instead the number of mating events"**

This last finding is decisive for C3.1's model intervention shape:
the empirically-documented contrast is in mating frequency, not
clutch size. The pre-spawn skip-probability dispatch (Bernoulli gate
before Poisson clutch sampling) maps cleanly to this mechanism.

**Cross-species meta-analytic baseline:** Christie, Ford & Blouin
(2014), *Evolutionary Applications* 7(1):37-46, doi:10.1111/eva.12183
— *"On the reproductive success of early-generation hatchery fish in
the wild"*. Meta-analysis of 51 estimates from 6 studies / 4 salmon
species: **early-generation hatchery fish averaged HALF the reproductive
success of wild-origin counterparts**. Reduction more severe for males.
Mechanism (genetic vs. environmental) discussed but unresolved.

**Population-scale anchor:** Jonsson, Jönsson & Jonsson (2019),
*Conservation Science and Practice* 1:e85, doi:10.1111/csp2.85 —
*"Supportive breeders of Atlantic salmon Salmo salar have reduced
fitness in nature"*. River Imsa long-term monitoring (1976-2013):
mean smolts per 100m² river area / female breeder dropped from 0.47
(wild only) to 0.088 (5% wild females). The metric MIXES spawning
behaviour and offspring survival — uses Christie 2014 / Bouchard 2022
for the spawning-only signal that C3.1 models.

**Generation-level applicability to the Lithuanian programme:** The
Žeimena/Simnas programme has been running since 1997 (~29 years; ~4-7
salmon generations). This places Lithuanian hatchery fish closer to
Bouchard 2022's intermediate-generation Atlantic salmon stocks
(RRS 65-80%) than to Christie 2014's cross-species 50% meta-analytic
baseline. The +0.3 skip probability default sits in the upper-middle
of Bouchard's Atlantic salmon range and is calibration-grade pending
direct Lithuanian RRS measurement.

## Scope

**In:**

- New `pre_spawn_skip_prob: float = 0.0` field on `BalticBioParams` in
  `salmon_ibm/baltic_params.py`. Wild default 0.0 (always spawn).
- `BalticBioParams.__post_init__` validates `0.0 <= pre_spawn_skip_prob
  <= 1.0`; raises `ValueError` on out-of-range.
- `_apply_hatchery_overrides` extended: `ALLOWED_OVERRIDE_KEYS` now
  includes `"pre_spawn_skip_prob"`. Scalar overrides applied via
  `dataclasses.replace(..., **scalar_kwargs)` after the existing
  `activity_by_behavior` shallow-merge.
- New `SCALAR_OVERRIDE_FIELDS = {"pre_spawn_skip_prob"}` set in the
  loader, separating scalar-replacement semantics from the existing
  `activity_by_behavior` shallow-merge semantics. Forward-compatible:
  future C3.x scalar fields just add to this set.
- `ReproductionEvent.execute` (in `salmon_ibm/events_builtin.py`) gains
  a Bernoulli skip filter inserted between `reproducer_idx` computation
  and the empty-check / Poisson clutch sampling. Filter is gated by
  `landscape.get("hatchery_dispatch") is not None and
  hd.params.pre_spawn_skip_prob > 0`.
- Top-of-file import updated: `from salmon_ibm.origin import ORIGIN_WILD,
  ORIGIN_HATCHERY` (currently only `ORIGIN_WILD`).
- `configs/baltic_salmon_species.yaml` gains `pre_spawn_skip_prob: 0.0`
  at the wild level (explicit baseline) and `pre_spawn_skip_prob: 0.3`
  inside `hatchery_overrides`. Provenance comment block documents
  Bouchard 2022 + Christie 2014 + Jonsson 2019 + calibration status.
- 6 new tests in `tests/test_hatchery_c3_spawn.py` covering: loader
  happy path with identity check, `__post_init__` validation, dispatch
  at p=1.0 (deterministic skip-all), dispatch at p=0.0 (no-skip
  regression check), graceful fallback when `hatchery_dispatch` absent,
  strict-loader rejection of unknown keys (extended set still rejects).

**Out:**

- **Origin inheritance on reproduction.** Per C1 scope-OUT decision,
  offspring of hatchery parents continue to default `ORIGIN_WILD`.
  Bouchard 2022 / Christie 2014 attribute the RRS deficit to
  environmental + epigenetic effects on parents, not on offspring per
  se. Inheritance becomes its own future tier (could be C3.1.5 or
  fold into a later C3.x).
- **Sex-specific skip rates.** Bouchard 2022 found males more affected
  than females (1SW males ~65% RRS vs MSW females/males ~80%). The
  current model doesn't track sex; adding it would substantially
  expand C3.1 scope. Defer to future tier.
- **Stochastic skip-rate variation by year/cohort.** A single
  Bernoulli with constant `p_skip` per scenario. Hatchery RRS may
  vary year-to-year due to environmental conditions; not modeled.
- **C3.2 sea-age sampling.** Separate spec.
- **C3.3 homing precision.** Separate spec.
- **Replacement of clutch_mean reduction (Option β).** Bouchard 2022's
  "fewer mating events, not smaller clutches" finding directly excludes
  this approach.
- **Per-offspring viability gate (Option γ).** Mixes spawning with
  offspring survival; out of C3.1's narrow "spawning behaviour" scope.

## Divergent values

```python
# Wild baseline (existing field with default 0.0):
pre_spawn_skip_prob = 0.0   # always spawn

# Hatchery (NEW C3.1):
pre_spawn_skip_prob = 0.3   # corresponds to RRS ≈ 0.7
```

**Wild baseline provenance:** 0.0 reflects the implicit assumption
that wild reproducers always proceed once they're in `reproducer_idx`
(group-membership + min_group_size + alive + mask gates). This was
the pre-C3.1 ReproductionEvent behaviour; C3.1 makes the assumption
explicit as a configurable field.

**Effect size at default 0.3:** for a population of N hatchery
reproducers with clutch_mean=4.0, expected offspring drop is ~30%
(`N × 4.0 × 0.7` vs `N × 4.0`). For a mixed scenario with k% hatchery,
the population-level offspring count drops by ~`0.3 × k%`. Combined
with C2's activity divergence (~0.21pp post-migration mass loss
differential), C3.1 produces meaningful population-level contrasts in
multi-year cohort tracking.

**Calibration sensitivity:** the 0.3 default is calibration-grade,
**bracketed** by Bouchard 2022's Atlantic salmon range (1 - 0.65 =
0.35 upper; 1 - 0.80 = 0.20 lower). Mandatory sensitivity sweep
before publication: **{0.0, 0.15, 0.30, 0.50}**, anchored to
Bouchard 2022 (Atlantic salmon-specific) on the lower end and
Christie 2014 (cross-species meta-analytic) on the upper end.
**Cap at 0.6** — beyond exceeds Christie 2014's lower bound for
*S. salar*-specific estimates and would require separate citation
support.

## Choice of approach

Three options were considered for the model intervention shape
(brainstormed 2026-05-02):

1. **Pre-spawn skip probability via Bernoulli filter** (selected) —
   Bernoulli gate after `reproducer_idx` filtering, before Poisson
   clutch sampling. Single new dataclass field; single new dispatch
   site in `ReproductionEvent.execute`; reuses C2's HatcheryDispatch
   bundle pattern.
2. **Reduced clutch_mean for hatchery** — multiply existing
   `clutch_mean` by a factor (e.g., 0.5) for hatchery reproducers.
   Reuses existing field. **REJECTED:** Bouchard 2022 explicitly
   found that captive-breeding affects mating frequency, NOT offspring
   per mating event. Choosing this option would conflate two distinct
   biological signals (fecundity vs. spawning success) and contradict
   the cited mechanistic finding.
3. **Per-offspring viability gate** — Bernoulli per-offspring after
   Poisson sampling. Most flexible (could model partial-clutch
   survival). **REJECTED:** mixes spawning behaviour with offspring
   survival; out of C3.1's narrow "spawning behaviour" scope.
   Population-scale Jonsson 2019 metric mixes these too, which is
   why C3.1 uses Bouchard 2022 (mechanistically separable) instead.

**Approach 1 selected** for: (a) cleanest mapping to Bouchard 2022's
mechanistic finding, (b) symmetry with C2's "one helper per divergent
field" pattern, (c) zero touch on the Poisson clutch sampling /
offspring placement code (reduces regression surface), (d) the
`pre_spawn_skip_prob > 0` guard means existing pre-C3.1 reproductive
tests are unaffected.

## Architecture

The change touches **4 files** (1 new test file + 3 modified). All
modifications are additive; no existing field/method is removed or
renamed.

**Files modified:**

1. `salmon_ibm/baltic_params.py`:

   - Append to `BalticBioParams`:
     ```python
     pre_spawn_skip_prob: float = 0.0
     ```
   - Append to `BalticBioParams.__post_init__`:
     ```python
     if not (0.0 <= self.pre_spawn_skip_prob <= 1.0):
         raise ValueError(
             f"pre_spawn_skip_prob must be in [0, 1], got "
             f"{self.pre_spawn_skip_prob!r}"
         )
     ```
   - Extend `_apply_hatchery_overrides`:
     ```python
     ALLOWED_OVERRIDE_KEYS = {"activity_by_behavior", "pre_spawn_skip_prob"}
     SCALAR_OVERRIDE_FIELDS = {"pre_spawn_skip_prob"}
     # ... existing unknown-key + activity_by_behavior coercion blocks ...

     # Existing activity_by_behavior shallow-merge:
     merged_dict = {**wild_params.activity_by_behavior, **activity_overrides}

     # NEW: scalar field overrides
     scalar_kwargs = {
         k: v for k, v in overrides.items() if k in SCALAR_OVERRIDE_FIELDS
     }
     return dataclasses.replace(
         wild_params,
         activity_by_behavior=merged_dict,
         **scalar_kwargs,
     )
     ```

2. `salmon_ibm/events_builtin.py`:

   - Update top import (currently only ORIGIN_WILD from C1):
     ```python
     from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY
     ```
   - In `ReproductionEvent.execute` (around line 314), insert the skip
     filter immediately after `reproducer_idx = np.where(can_reproduce)[0]`
     and before the existing `if len(reproducer_idx) == 0: return`:
     ```python
     # C3.1: pre-spawn skip filter for hatchery reproducers.
     # Bernoulli gate before Poisson clutch sampling; matches Bouchard
     # et al. 2022 (doi:10.1111/eva.13374) finding that captive-breeding
     # reduces "number of mating events, not offspring per mating".
     # Wild reproducers (origin=ORIGIN_WILD) always pass through.
     # Spec: docs/superpowers/specs/2026-05-02-hatchery-c3-spawn-design.md
     hd = landscape.get("hatchery_dispatch")
     if hd is not None and hd.params.pre_spawn_skip_prob > 0 and len(reproducer_idx) > 0:
         is_hatchery = population.pool.origin[reproducer_idx] == ORIGIN_HATCHERY
         skip_rolls = rng.random(len(reproducer_idx)) < hd.params.pre_spawn_skip_prob
         keep = ~(is_hatchery & skip_rolls)
         reproducer_idx = reproducer_idx[keep]
     ```

3. `configs/baltic_salmon_species.yaml`:

   - Add `pre_spawn_skip_prob: 0.0` at the wild level (alongside other
     scalar fields like `cmax_A`, `T_OPT`).
   - Inside `hatchery_overrides:`, add `pre_spawn_skip_prob: 0.3` after
     the existing `activity_by_behavior:` block.
   - Provenance comment block above the new `pre_spawn_skip_prob`
     entries citing Bouchard 2022 (primary), Christie 2014 (meta-
     analytic baseline), Jonsson 2019 (population-scale), with
     calibration status, sensitivity sweep, scope-OUT items.

**New test file:**

`tests/test_hatchery_c3_spawn.py` — 6 tests:

1. `test_pre_spawn_skip_prob_loads_from_yaml` — happy path with
   identity check (`cfg.hatchery is not cfg.wild`).
2. `test_pre_spawn_skip_prob_rejects_out_of_range` — `__post_init__`
   raises on `-0.1` and `1.5`.
3. `test_reproduction_skips_hatchery_at_p_skip_one` — deterministic
   test at `p_skip=1.0`; asserts offspring count consistent with
   wild-only Poisson.
4. `test_reproduction_no_skip_at_p_zero` — explicit `p_skip=0.0` →
   skip path bypassed; offspring count consistent with all-reproducers
   Poisson.
5. `test_reproduction_graceful_without_hatchery_dispatch` — landscape
   without `"hatchery_dispatch"` key → no skip code executes; pre-C3.1
   scenarios unaffected.
6. `test_extended_overrides_still_reject_unknown_keys` — strict
   loader still raises on unknown keys after `ALLOWED_OVERRIDE_KEYS`
   expansion.

## Risk + regression surface

**Behaviour change for pre-C3.1 scenarios: zero.**

- YAMLs without `pre_spawn_skip_prob` → dataclass default 0.0 → guard
  short-circuits before any new code path executes
- Existing reproduction tests (test_events.py, integration tests)
  construct WILD-only populations → `is_hatchery` mask all-False →
  filter is no-op
- `landscape.get("hatchery_dispatch")` returns None for tests without
  C2 setup → guard short-circuits before pool.origin access

**Behaviour change for C3.1-aware scenarios: by design.**

- Hatchery reproducers skip with the configured probability
- Default 0.3 corresponds to RRS ≈ 0.7, mid-range of Bouchard 2022
- Effect compounds with C2 activity divergence

**Silent-failure modes (closed by design):**

- ❌ Hatchery agents present but skip path silently disabled →
  caught by C2's `IntroductionEvent`/`PatchIntroductionEvent` runtime
  guards (raise if HATCHERY agents but no `hatchery_dispatch`); C3.1
  inherits this protection.
- ❌ Out-of-range skip probability silently accepted → caught by
  `__post_init__` validation at load time. Locked by test 2.
- ❌ YAML typo in `pre_spawn_skip_prob` → caught by strict
  `ALLOWED_OVERRIDE_KEYS` check. Locked by test 6.
- ❌ Skip applied to wild reproducers → mask requires both `is_hatchery`
  AND `skip_rolls`; `population.pool.origin == ORIGIN_HATCHERY` is
  unambiguous. Locked by test 3.
- ❌ Origin inheritance accidentally enabled → C3.1 doesn't touch
  `add_agents` calls; offspring continue to default `ORIGIN_WILD` per
  C1 scope-OUT.
- ❌ Numerical instability from skip > 1.0 → `__post_init__` rejects.
  Locked by test 2.

**Tests that may need updating:** none expected. The
`pre_spawn_skip_prob > 0` guard means existing reproduction tests
(which never construct C3.1 scenarios) bypass the new code path
entirely.

## Success criteria

- [ ] `BalticBioParams.pre_spawn_skip_prob: float = 0.0` field added.
- [ ] `BalticBioParams.__post_init__` validates `0 <= p <= 1`.
- [ ] `_apply_hatchery_overrides` extended: `ALLOWED_OVERRIDE_KEYS`
      includes `pre_spawn_skip_prob`; `SCALAR_OVERRIDE_FIELDS` set
      established; `dataclasses.replace(..., **scalar_kwargs)` applied.
- [ ] `ReproductionEvent.execute` reads `landscape["hatchery_dispatch"]`
      and applies Bernoulli skip on hatchery reproducers when
      `pre_spawn_skip_prob > 0`.
- [ ] `events_builtin.py` top import includes `ORIGIN_HATCHERY`.
- [ ] `configs/baltic_salmon_species.yaml` gains `pre_spawn_skip_prob:
      0.0` at wild level + `pre_spawn_skip_prob: 0.3` in
      hatchery_overrides + full provenance comment block.
- [ ] All 6 new tests pass.
- [ ] Full pytest suite stays green (current C2 baseline 842 passing;
      expected post-C3.1: 848 = baseline + 6).
- [ ] Plan stamp on the eventual implementation plan: ✅ EXECUTED.

## Estimated implementation time

**~2 days** including TDD cycle, validation logic, dispatch wiring,
YAML provenance, regression sweep. Smaller than C2 because:

- Single new dataclass field (no new types)
- Single new dispatch site (no factor-out / refactor)
- Single new test file (no caller-fix updates expected)
- No Numba kernel touch
- No `app.py` changes
- No `simulation.py` changes
- Reuses C2's HatcheryDispatch + landscape injection without
  modification

## Out-of-scope (deferred)

- **Origin inheritance on reproduction** — offspring of hatchery
  parents stay tagged `ORIGIN_WILD` per C1 scope-OUT. Could be its
  own future plan once Bouchard 2022's parent-environmental-effect
  finding has stronger Atlantic salmon support. Some literature
  suggests epigenetic carryover (Rodriguez-Barreto 2019); not
  scoped here.
- **Sex-specific skip rates** — Bouchard 2022 found 1SW males ~65%
  RRS vs MSW ~80%. Modeling this requires sex tracking (new field +
  sampling event) — substantially larger than C3.1's scope. Defer.
- **C3.2 sea-age sampling** — separate spec.
- **C3.3 homing precision** — separate spec.
- **Year-to-year RRS stochasticity** — single Bernoulli with constant
  `p_skip`. Bouchard 2022 reported per-year variation; not modeled.
- **Mate-finding submodel** — Bouchard 2022 mechanism is "fewer
  mating events". C3.1 abstracts this as a per-reproducer Bernoulli
  rather than modeling mate-search behaviour explicitly. The
  abstraction is honest and matches the Wisconsin/IBM tradition of
  population-level rates rather than individual mate-choice mechanics.
- **Hatchery-only group filter** — C3.1 leaves group-based
  reproduction logic untouched. A hatchery agent in an all-hatchery
  group still qualifies as a reproducer (subject to skip).

## YAML schema documentation

The deployed `configs/baltic_salmon_species.yaml` block (current C2
state has `hatchery_overrides:` with `activity_by_behavior:`) gets
the C3.1 additions:

```yaml
species:
  BalticAtlanticSalmon:
    # ... ALL existing fields stay (cmax_A, T_OPT, LW_a, LW_b, etc.) ...
    activity_by_behavior:
      0: 1.0   # HOLD
      1: 1.2   # RANDOM
      2: 0.8   # TO_CWR
      3: 1.5   # UPSTREAM
      4: 1.0   # DOWNSTREAM
    # C3.1: wild reproducers always proceed (skip probability 0.0)
    pre_spawn_skip_prob: 0.0
    # ... existing C2 provenance comment block stays here ...
    hatchery_overrides:
      activity_by_behavior:                    # ← C2
        1: 1.5     # RANDOM:  +25% (1.2 → 1.5)
        3: 1.875   # UPSTREAM: +25% (1.5 → 1.875)
      # C3.1: pre-spawn skip probability for hatchery reproducers.
      # Empirical anchor: Bouchard, Wellband, Lecomte et al. 2022
      # (doi:10.1111/eva.13374) — captive-bred Atlantic salmon RRS
      # 0.65-0.80 (MSW females + males 80%; 1SW males 65%).
      # Cross-species meta-analytic baseline: Christie, Ford & Blouin
      # 2014 (doi:10.1111/eva.12183) — early-generation hatchery fish
      # average HALF the reproductive success of wild counterparts
      # across 4 salmon species.
      # Population-scale supportive-breeders study: Jonsson et al.
      # 2019 (doi:10.1111/csp2.85) — 81% reduction in smolts/breeder
      # at 95% hatchery, but mixes spawning + offspring survival.
      # Lithuanian Žeimena/Simnas programme (~4-7 generations since
      # 1997) is intermediate between Bouchard 2022 and Christie 2014.
      #
      # MECHANISTIC SHAPE: Bouchard 2022 found "captive-breeding did
      # not directly affect the number of offspring per mating event
      # but instead the number of mating events" — directly validates
      # the pre-spawn skip-rate model intervention (rather than clutch-
      # size reduction). This is the strongest single empirical
      # rationale for the C3.1 model intervention shape.
      #
      # CALIBRATION STATUS: 0.3 corresponds to RRS ≈ 0.7, mid-range
      # of Bouchard 2022 Atlantic salmon estimates. Treat as
      # calibration-grade. Mandatory sensitivity sweep before
      # publication: {0.0, 0.15, 0.30, 0.50}. Cap at 0.6 — beyond
      # exceeds Christie 2014 lower bound for Salmo salar.
      #
      # SCOPE-OUT (deferred): origin inheritance (offspring stay
      # tagged ORIGIN_WILD per C1 scope-OUT); sex-specific skip
      # (Bouchard 2022 found males more affected; current model
      # doesn't track sex); year-to-year RRS stochasticity (single
      # constant skip probability per scenario).
      pre_spawn_skip_prob: 0.3
```
