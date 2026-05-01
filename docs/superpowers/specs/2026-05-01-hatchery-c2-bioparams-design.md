# Hatchery vs Wild Distinction — C2 activity_by_behavior Divergence

**Date:** 2026-05-01
**Owner:** @razinkele
**Status:** 📋 DRAFT — awaiting writing-plans

This is the **second of three planned tiers** for hatchery-vs-wild origin
support, following C1 (`docs/superpowers/specs/2026-04-30-hatchery-origin-c1-design.md`,
shipped as PR #3 on 2026-04-30):

- C1 (shipped) — tag-only origin int8 column on every agent, no behaviour
  difference. ~1 day.
- **C2 (this spec)** — parameter-level: hatchery agents get a different
  `activity_by_behavior` Wisconsin-bioenergetics multiplier than wild.
  ~2 days.
- C3 (future plan) — full origin-aware behaviour: spawn-success, homing
  precision, sea-age sampling differ by origin. Not addressed here.

C2 ships *one* literature-anchored divergent field plus the dispatch
plumbing. Future tiers can layer additional divergent fields with the
same pattern.

## Why now

C1 added the `origin` column on every agent but no agent takes a
different action because of it. C2 makes the column physically meaningful:
hatchery-tagged agents pay a higher metabolic cost during active
swimming, capturing the empirical observation that domesticated *Salmo
salar* are mechanically less efficient swimmers than wild conspecifics.

The single divergent field is `activity_by_behavior` — the per-behavior
multiplier on Wisconsin respiration in `bioenergetics.py:hourly_respiration`.
Hatchery values are raised on swim-active behaviors only.

## Scientific basis

**Primary anchor:** Enders, Boisclair & Roy (2004),
*Canadian Journal of Fisheries and Aquatic Sciences* 61(12):2302-2313,
doi:10.1139/f04-211 — direct measurements of total swimming costs across
wild, farmed (F1 from wild broodstock), and domesticated (seventh-generation
Norwegian aquaculture strain) *Salmo salar*. **Wild and F1 fish showed
no statistically significant difference (6.7% average). Seventh-generation
domesticated fish paid 12–29% higher swimming costs**, attributable to
deeper body morphology and reduced fin area. The +25% increment used in
C2 sits inside the seventh-generation bracket; it is not directly
supported by the F1 result.

**Generation-level applicability to the Lithuanian programme:** The
Žeimena/Simnas programme has been running since 1997 (~29 years; ~4–7
salmon generations). This places Lithuanian hatchery fish *between* the
F1 arm of Enders and the seventh-generation arm. The +25% increment is
applied as a plausible extrapolation for that generation level; it is
not directly measured for Baltic stocks. An implementation using
strictly F1 broodstock would over-state the contrast at +25%. A
seventh-generation strain would likely under-state it.

**Supporting:** Zhang et al. (2016), *Aquaculture* 463:79-88,
doi:10.1016/j.aquaculture.2016.05.015 — domesticated *Salmo salar* show
significantly reduced aerobic scope (MO2max, AAS, FAS) under aerobic
exercise training. Pedersen, Koed & Malte (2008), *Ecology of Freshwater
Fish* 17(3):425-431, doi:10.1111/j.1600-0633.2008.00293.x — F1 hatchery
Atlantic salmon smolts show ~30% slower **burst swimming speed** (a
kinematic capacity metric, not aerobic cost) and faster fatigue.
Hammenstig et al. (2014), *Journal of Fish Biology* 85(4):1177-1191,
doi:10.1111/jfb.12511 — wild smolts metabolically more efficient than
hatchery in burst-swim recovery.

**Population-specific null result:** Hatanpää, Huuskonen & Kekäläinen
(2020), doi:10.1139/cjfas-2019-0079 — landlocked *Salmo salar m.
sebago* in Lake Saimaa: hatchery vs semi-wild swimming endurance was
not significantly different, attributed to extreme genetic bottleneck
in that population. The null result does not generalize: anadromous
Lithuanian Nemunas/Žeimena stocks operate without a comparable
genetic constraint.

**Framing:** the `activity_by_behavior` divergence is an
empirically-bounded activity scaling factor representing the net effect
of (a) reduced aerobic scope (Zhang 2016) and (b) higher swimming cost
per unit displacement (Enders 2004) observed in domesticated *S. salar*.
This is not a mechanistic respiration estimate — Wisconsin's
activity_mult scales steady-state aerobic respiration, while the cited
papers measured swimming costs across mixed metabolic regimes. The
phenomenological framing is standard in IBM literature (e.g. Lennox
et al. 2018 use the same Wisconsin pattern for migrating *S. salar*).

## Scope

**In:**

- New `HatcheryDispatch` frozen dataclass in `baltic_params.py` bundling
  the paired nullables (params + LUT) so the invariant `params is None
  ↔ lut is None` becomes structurally impossible to violate:
  ```python
  @dataclass(frozen=True)
  class HatcheryDispatch:
      params: BalticBioParams
      activity_lut: np.ndarray
  ```
- New optional attribute `Simulation.hatchery_dispatch: HatcheryDispatch | None`,
  built at simulation init by overlaying YAML `hatchery_overrides:` onto
  the existing wild `BalticBioParams` instance via `dataclasses.replace`,
  then bundled with its derived activity LUT.
- New vectorized helper `origin_aware_activity_mult(behavior, origin,
  lut_wild, lut_hatch) -> ndarray` in `bioenergetics.py`. Returns
  `lut_wild[behavior]` when `lut_hatch is None` (graceful for legacy
  / pre-C2 callers).
- New simulation method `rebuild_luts()` that rebuilds wild
  `_activity_lut` AND the `hatchery_dispatch` (params + LUT bundled
  together) from a cached `BalticSpeciesConfig` (the config is loaded
  once at `__init__` and stored on `self._species_config`, so
  `rebuild_luts()` does not re-read disk on every slider event).
- `Landscape` TypedDict gains `hatchery_dispatch: HatcheryDispatch | None`
  (with `total=False` so existing consumers are unaffected). One key,
  not two — the bundle is atomic.
- `Simulation.step()` injects `hatchery_dispatch` into the per-step
  landscape dict.
- `_event_bioenergetics` and `events_builtin.py:SurvivalEvent.execute`
  both call the dispatch helper.
- YAML schema gains a `hatchery_overrides:` sub-block under
  `species.BalticAtlanticSalmon`. **For C2 the only allowed key is
  `activity_by_behavior`** — other dataclass-field keys are rejected at
  load time (would be biologically inert under the C2 dispatch).
- Strict loader: typos and unsupported keys raise `ValueError` at
  scenario-load time, not at first simulation step.
- `BalticBioParams.__post_init__` gains validation that
  `activity_by_behavior` is non-empty and contains only positive floats
  (already implicitly required by Wisconsin respiration).
- Runtime guard inside `IntroductionEvent.execute()`: if `self.origin
  == ORIGIN_HATCHERY` and `landscape.get("hatchery_dispatch") is None`,
  raise. Catches the case where a runtime introduction tries to add
  hatchery agents but no overrides are configured.
- Startup log message at `Simulation.__init__` when
  `hatchery_dispatch is not None`: `"C2 hatchery dispatch active:
  activity_by_behavior keys {1, 3} overridden"`. Uses
  `{k for k in hatchery_dict if wild_dict.get(k) != hatchery_dict[k]}`
  (robust to extra hatchery keys, although the loader prevents that case).
- `load_baltic_species_config()` returns a `NamedTuple
  BalticSpeciesConfig(wild: BioParams | BalticBioParams, hatchery:
  BalticBioParams | None)`. **Always** returns this NamedTuple — even
  the legacy non-Baltic path returns `BalticSpeciesConfig(wild=plain_BioParams,
  hatchery=None)`. This eliminates the heterogeneous-return isinstance
  branch at the caller and makes `AttributeError` failure modes
  impossible.
- `load_bio_params_from_config()` follows the same pattern: always
  returns `BalticSpeciesConfig`.
- 13 new tests in `tests/test_hatchery_params.py` covering all happy
  paths, all rejection paths, and the integration invariants.

**Out:**

- **Other BioParams field divergences** — RA, RB, RQ, T_OPT, T_AVOID,
  T_ACUTE_LETHAL, ED_TISSUE, MASS_FLOOR_FRACTION, fecundity_per_g, etc.
  are NOT origin-aware in C2. The dispatch helper is field-specific
  (one helper per divergent field). This is deliberate — C2 ships ONE
  literature-supported divergence; future tiers add more by adding
  helpers, not by generalizing.
- **Reproduction / Phase-3 seedling origin inheritance** — same
  scope-OUT as C1. Offspring default to WILD.
- **Time-decaying multiplier** — hatchery-vs-wild physiological gap may
  narrow over time post-release (developmental plasticity). No
  published time-decay curve exists for *S. salar*; static
  approximation retained pending Curonian Lagoon tagging data.
- **Hatchery activity multiplier sliders in the Shiny UI** — sidebar
  exposes wild RA/RB/RQ/ED_MORTAL/T_OPT/T_MAX only. Hatchery overrides
  are research parameters fixed in YAML for C2. Flagged as C3 backlog.
- **Generic deep-merge for nested dict overrides** — only
  `activity_by_behavior` is dict-typed in BalticBioParams today.
  Shallow-merge is hand-rolled for that single field. C3 may factor a
  helper if more dict fields are added.
- **Pre-existing app.py:1493 bug** — sidebar replaces `bio_params` with
  plain `BioParams` (drops Baltic-specific fields like
  `T_ACUTE_LETHAL`). C2 does NOT fix this; it is a separate cleanup.
  C2's `rebuild_luts()` re-derives the wild base from
  `load_bio_params_from_config(self.config)` to avoid compounding the
  bug.
- **`SwitchPopulationEvent` hatchery-into-no-hatchery-target warning**
  — when an agent transfers populations and the target population has
  no `bio_params_hatchery`, the agent silently uses the target's wild
  values. No deployed multi-population hatchery scenario exists today;
  the runtime guard inside `IntroductionEvent` covers the introduction
  path. Defer to follow-up.
- **C3 behaviour divergence** — separate spec.

## Divergent values

```python
# Wild baseline (existing in BalticBioParams, line 76-79):
activity_by_behavior = {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
#   HOLD  (0): 1.0   stationary; no contrast
#   RANDOM (1): 1.2  active wandering swim
#   TO_CWR (2): 0.8  efficient directed refuge-seeking
#   UPSTREAM (3): 1.5  high-cost active swim against current
#   DOWNSTREAM (4): 1.0  drift-assisted, mostly passive

# Hatchery (new, +25% on swim-active behaviors only):
activity_by_behavior = {0: 1.0, 1: 1.5, 2: 0.8, 3: 1.875, 4: 1.0}
#   RANDOM:  1.2 → 1.5    (+25%)
#   UPSTREAM: 1.5 → 1.875  (+25%)
#   DOWNSTREAM unchanged: passive drift; morphological inefficiency
#     suppressed when current provides thrust (hydrodynamic rationale,
#     no direct citation). Including it would over-extend Pedersen 2008
#     beyond what was tested.
#   HOLD, TO_CWR unchanged: stationary or efficient directed.
```

**Wild baseline provenance:** `{0:1.0, 1:1.2, 2:0.8, 3:1.5, 4:1.0}` is
inherited verbatim from the inSTREAM Snyder Chinook reference
(`baltic_params.py:76-79` comment "keep Snyder structure,
Baltic-tunable"). Baltic-specific verification has not been done; the
hatchery contrast is layered atop a known-uncertain baseline. The YAML
must carry a provenance note acknowledging this.

**Effect size:** under realistic mixed-behavior migration
(p_HOLD=0.05, p_RANDOM=0.20, p_TO_CWR=0.10, p_UPSTREAM=0.55,
p_DOWNSTREAM=0.10) the hatchery cohort accumulates **~0.21 percentage
points more 21-day mass loss** than wild (1.94% vs 1.73% at 10°C, 2 kg
smolts, BalticBioParams defaults). C2 alone does NOT produce a large
standalone hatchery-vs-wild survival contrast — ED stays well above
ED_MORTAL=4.0 for both cohorts over typical Curonian-corridor durations.
The contrast emerges in compound with osmoregulation stress
(v1.7.3) or longer corridors, or with future C3 behaviour
divergences. C2 is the architectural enabler for compounding, not a
standalone mortality driver.

**Calibration sensitivity:** the +25% magnitude is calibration-grade.
Mandatory sensitivity sweep before publication: **{0%, +12.5%, +25%,
+37.5%}**, anchored to the 12-29% empirical bracket from Enders 2004.
**Cap at +40%** — beyond that, no published support exists. Document the
sweep results in any paper using C2 outputs.

## Choice of approach

Three architectural patterns were considered (brainstorming 2026-05-01):

1. **Single dispatch point per divergent field** (selected) — one
   helper function per divergent field, called at the point where
   per-agent multiplier arrays are computed. Wisconsin kernel signature
   unchanged. Other BioParams fields read from the wild instance only.
2. **Per-agent param-vector struct** — at sim init, build NumPy arrays
   length-N for each scalar field by indexing param_table[origin]. Hot
   loops read `params_arr.RA[mask]` instead of `params.RA`. Rejected:
   forces Wisconsin kernel signature to change (12 ndarrays instead of
   one BioParams object); ramps C2 to ~5 days; premature abstraction
   for one divergent field.
3. **Resolver class** — `BioParamsResolver(wild, hatchery)` with
   `.resolve(field_name, agents)`. Rejected: reflection / string-keyed
   field lookup is slow and incompatible with Numba JIT; no consumer
   in C2 benefits from generic dispatch.

**Approach 1 selected** for: (a) zero impact on Wisconsin kernel
signature, (b) honest scope (one divergence in C2), (c) symmetric with
C1's "one helper per concern" pattern, (d) future C3 divergences add
their own helpers without retrofitting the existing kernel.

## Architecture

The change touches **8 files** (1 new test file + 7 modified). All
modifications are additive; no existing field/method is removed or
renamed.

**Files modified:**

1. `salmon_ibm/bioenergetics.py` — add the dispatch helper:

   ```python
   from salmon_ibm.origin import ORIGIN_HATCHERY

   def origin_aware_activity_mult(
       behavior: np.ndarray,
       origin: np.ndarray,
       lut_wild: np.ndarray,
       lut_hatch: np.ndarray | None,
   ) -> np.ndarray:
       """Per-agent activity multiplier with origin-aware dispatch.

       Returns lut_wild[behavior] when lut_hatch is None — graceful for
       pre-C2 paths, test fixtures, and scenarios without hatchery
       overrides. Otherwise dispatches per-agent via origin column.

       See docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md.
       """
       if lut_hatch is None:
           return lut_wild[behavior]
       return np.where(
           origin == ORIGIN_HATCHERY,
           lut_hatch[behavior],
           lut_wild[behavior],
       )
   ```

   No changes to `hourly_respiration` or `update_energy` signatures.

2. `salmon_ibm/baltic_params.py`:

   - New frozen dataclass at module top-level:
     ```python
     @dataclass(frozen=True)
     class HatcheryDispatch:
         """Bundle hatchery params + their derived activity LUT atomically.

         Holding params and LUT separately as paired nullables on
         Simulation invites desync (one rebuilt, one stale). This bundle
         makes the invariant `params is None ↔ lut is None` structurally
         impossible to violate, and gives callers a single nullable to
         guard on (`if landscape.get('hatchery_dispatch') is None: ...`).
         """
         params: BalticBioParams
         activity_lut: np.ndarray
     ```

   - `BalticBioParams.__post_init__` adds:
     ```python
     if not self.activity_by_behavior:
         raise ValueError("activity_by_behavior must be non-empty")
     for k, v in self.activity_by_behavior.items():
         if not isinstance(k, int) or k < 0:
             raise ValueError(f"activity_by_behavior keys must be non-negative ints, got {k!r}")
         if not isinstance(v, (int, float)) or v <= 0:
             raise ValueError(f"activity_by_behavior values must be positive floats, got {k}: {v!r}")
     ```
   - New `NamedTuple BalticSpeciesConfig(wild: BalticBioParams, hatchery: BalticBioParams | None)`.
   - `load_baltic_species_config(path) -> BalticSpeciesConfig` (replaces
     bare-`BalticBioParams` return type). Builds wild as before. If
     `species.BalticAtlanticSalmon.hatchery_overrides` block exists,
     parses it via `_apply_hatchery_overrides(wild, overrides_block)`.
     Returns `(wild, hatchery_or_None)`.
   - New `_apply_hatchery_overrides(wild_params: BalticBioParams,
     overrides: dict) -> BalticBioParams`:
     ```python
     from salmon_ibm.agents import Behavior  # for sub-key validation
     ALLOWED_OVERRIDE_KEYS = {"activity_by_behavior"}  # C2 scope
     VALID_BEHAVIORS = {int(b) for b in Behavior}  # {0, 1, 2, 3, 4}

     unknown = set(overrides) - ALLOWED_OVERRIDE_KEYS
     if unknown:
         raise ValueError(
             f"hatchery_overrides supports only {sorted(ALLOWED_OVERRIDE_KEYS)} in C2; "
             f"unsupported keys: {sorted(unknown)}"
         )
     activity_overrides = overrides.get("activity_by_behavior", {})
     # Coerce YAML string keys to int (PyYAML may emit e.g. "1" rather than 1).
     # Raise an actionable ValueError if a sub-key isn't numeric (e.g. user
     # accidentally typed a behavior name like "hold" instead of 0).
     try:
         activity_overrides = {int(k): float(v) for k, v in activity_overrides.items()}
     except (ValueError, TypeError) as exc:
         raise ValueError(
             f"hatchery_overrides.activity_by_behavior keys must be integers "
             f"(Behavior enum values 0-4); got non-integer key in {activity_overrides!r}"
         ) from exc
     # Validate sub-keys are valid Behavior enum values:
     invalid_keys = set(activity_overrides) - VALID_BEHAVIORS
     if invalid_keys:
         raise ValueError(
             f"hatchery_overrides.activity_by_behavior keys must be valid "
             f"Behavior enum values {sorted(VALID_BEHAVIORS)}; got invalid: "
             f"{sorted(invalid_keys)}"
         )
     merged_dict = {**wild_params.activity_by_behavior, **activity_overrides}
     return dataclasses.replace(wild_params, activity_by_behavior=merged_dict)
     ```
     `dataclasses.replace` re-runs `__post_init__` validation on the
     merged instance. The Behavior-enum check at load time means an
     `{999: 5.0}` typo raises `ValueError` immediately rather than
     silently producing an oversized LUT.

3. `salmon_ibm/config.py` — `load_bio_params_from_config()` returns
   `BalticSpeciesConfig` for **all** paths. The legacy non-Baltic path
   wraps a plain `BioParams` into `BalticSpeciesConfig(wild=BioParams(...),
   hatchery=None)`. The Baltic-with-overrides path returns
   `BalticSpeciesConfig(wild=baltic_params, hatchery=baltic_params_with_overrides)`.

   The caller at `simulation.py:294` becomes a single attribute access
   without isinstance branching:
   ```python
   loaded = load_bio_params_from_config(config)
   self.bio_params = loaded.wild
   self._species_config = loaded  # cached for rebuild_luts()
   if loaded.hatchery is not None:
       hatch_lut = build_activity_lut(loaded.hatchery.activity_by_behavior)
       self.hatchery_dispatch = HatcheryDispatch(
           params=loaded.hatchery,
           activity_lut=hatch_lut,
       )
   else:
       self.hatchery_dispatch = None
   ```
   No `AttributeError` failure mode possible — every path produces a
   well-typed `BalticSpeciesConfig`.

4. `salmon_ibm/simulation.py`:

   - **Imports** (top of file): add
     `from salmon_ibm.baltic_params import BalticSpeciesConfig, HatcheryDispatch`.

   - `Simulation.__init__` (around line 292-295):
     - Receive `BalticSpeciesConfig` from loader; assign as shown in
       item 3 above. Cache `self._species_config = loaded` for
       `rebuild_luts()` to use without re-reading disk on every
       sidebar event.
     - Call `self.rebuild_luts()` instead of `_build_activity_lut()`.
     - If `self.hatchery_dispatch is not None`, log:
       ```python
       wild_dict = self.bio_params.activity_by_behavior
       hatch_dict = self.hatchery_dispatch.params.activity_by_behavior
       overridden = sorted({k for k in hatch_dict if wild_dict.get(k) != hatch_dict[k]})
       logger.info("C2 hatchery dispatch active: activity_by_behavior keys overridden: %s", overridden)
       ```

   - New method `Simulation.rebuild_luts()`:
     ```python
     def rebuild_luts(self):
         """Rebuild wild and hatchery activity LUTs from cached species-config.

         Reads from self._species_config (cached at __init__) — does NOT
         re-read disk on every sidebar event. Slider adjustments that
         affect RA/RB/etc on self.bio_params do NOT affect the activity
         LUTs, which stay locked to the YAML-loaded baseline. This is
         intentional: sliders are for exploring metabolic-scale params,
         not for replacing the species-config activity profile.

         No-ops on non-Baltic configs (no _species_config cached).
         """
         if not hasattr(self, "_species_config") or self._species_config is None:
             return
         species_cfg = self._species_config  # cached BalticSpeciesConfig
         self._activity_lut = self._build_lut(species_cfg.wild.activity_by_behavior)
         if species_cfg.hatchery is not None:
             hatch_lut = self._build_lut(species_cfg.hatchery.activity_by_behavior)
             self.hatchery_dispatch = HatcheryDispatch(
                 params=species_cfg.hatchery,
                 activity_lut=hatch_lut,
             )
         else:
             self.hatchery_dispatch = None
     ```

   - `Simulation.step()` (around line 560-571) — landscape dict gains:
     ```python
     "hatchery_dispatch": self.hatchery_dispatch,
     ```
     (Single key, single nullable. The `HatcheryDispatch` bundle is
     read by both consumers below.)

   - `_event_bioenergetics` (around line 488-510) — reads from
     `self.*` directly (NOT from landscape, since this method is on
     Simulation):
     ```python
     hatch_lut = (self.hatchery_dispatch.activity_lut
                  if self.hatchery_dispatch is not None else None)
     activity = origin_aware_activity_mult(
         pool.behavior[mask], pool.origin[mask],
         self._activity_lut, hatch_lut,
     )
     ```

   - `Landscape` TypedDict (existing definition at `simulation.py:9`,
     `class Landscape(TypedDict, total=False)`) gains a single new key:
     `hatchery_dispatch: HatcheryDispatch | None`. The `total=False`
     attribute means existing consumers that don't reference this key
     are unaffected. **One key**, not two — the bundle keeps the
     params + LUT atomic.

5. `salmon_ibm/events_builtin.py` — three edits in this single file:

   **5a. `SurvivalEvent.execute` (lines 56-124):**

   - Add comment at line 56 above `bio_params: BioParams = field(...)`:
     ```python
     # NOTE: bio_params.activity_by_behavior on this event is dead
     # weight — activity dispatch goes through landscape["activity_lut"]
     # and landscape["activity_lut_hatchery"]. Local activity values
     # here are silently ignored. See C2 spec.
     ```

   - Replace the line 69 dispatch. **Note dispatch sources differ
     between the two consumers:** `_event_bioenergetics` (item 4 above)
     reads `self.hatchery_dispatch` directly because it's on Simulation;
     `SurvivalEvent.execute` reads `landscape["hatchery_dispatch"]`
     because it's an event without direct Simulation access. Both call
     the same helper. The `from salmon_ibm.bioenergetics import
     origin_aware_activity_mult` import goes at the **top of the file**,
     not inline inside the method:
     ```python
     # Was: activity = activity_lut[population.behavior]
     hd = landscape.get("hatchery_dispatch")
     hatch_lut = hd.activity_lut if hd is not None else None
     activity = origin_aware_activity_mult(
         population.behavior,
         population.pool.origin,
         landscape["activity_lut"],
         hatch_lut,
     )
     ```

   **5b. `IntroductionEvent.execute()` (around line 249)** — runtime
   guard before `add_agents`. Note: the guard checks `landscape.get(
   "hatchery_dispatch") is None` (single key) instead of two
   separate keys:

   ```python
   if self.origin == ORIGIN_HATCHERY and landscape.get("hatchery_dispatch") is None:
       raise ValueError(
           f"IntroductionEvent '{self.name}' tags new agents as HATCHERY, "
           f"but the simulation has no hatchery_dispatch configured. "
           f"Add a 'hatchery_overrides:' block under "
           f"species.BalticAtlanticSalmon in the species YAML."
       )
   ```

   `landscape["hatchery_dispatch"]` is set in `Simulation.step()` (item 4
   above) — a single key carrying both params and LUT bundled in the
   `HatcheryDispatch` dataclass.

6. `salmon_ibm/events_hexsim.py:PatchIntroductionEvent.execute()` —
   same runtime guard mirror as IntroductionEvent (5b above).

7. `app.py` (around line 1493-1501) — replace
   `sim._build_activity_lut()` with `sim.rebuild_luts()`. (One-line
   change; the slider-driven `BioParams(...)` replacement is left
   alone — the `rebuild_luts()` method re-derives the species-config
   base, so the hatchery LUT is anchored to Baltic defaults.)

**New test file:**

`tests/test_hatchery_params.py` — 13 tests covering all happy paths,
all rejection paths, and integration invariants. All tests use
`tmp_path` YAML fixtures unless they require a full `Simulation`
(tests 11, 12, 13).

**Loader / merge tests (1-3 + new 8-10):**

1. `test_hatchery_overrides_activity_by_behavior_loads` — YAML with
   `hatchery_overrides.activity_by_behavior: {1: 1.5, 3: 1.875}` loads
   into `BalticSpeciesConfig(wild=..., hatchery=...)` with merged dict
   `{0: 1.0, 1: 1.5, 2: 0.8, 3: 1.875, 4: 1.0}`. **Plus identity
   assertion: `assert result.hatchery is not result.wild`** (catches a
   mutation-in-place implementation that contaminates wild).
2. `test_hatchery_overrides_unsupported_key_raises` — `hatchery_overrides:
   {T_OPT: 14.0}` raises `ValueError` matching `r"hatchery_overrides supports only"`.
3. `test_hatchery_overrides_typo_raises` — `hatchery_overrides:
   {activity_for_behavior: ...}` (typo: "for" vs "by") raises
   `ValueError`.
8. `test_hatchery_overrides_invalid_behavior_key_raises` — sub-key
   `999` (not a valid Behavior enum value): `hatchery_overrides:
   {activity_by_behavior: {999: 5.0}}` raises `ValueError` matching
   `r"valid Behavior enum values"`. Without this test, an implementer
   who omits `VALID_BEHAVIORS` validation would silently produce an
   oversized 1000-element LUT.
9. `test_hatchery_overrides_nonnumeric_behavior_key_raises` —
   non-numeric sub-key like `{"hold": 1.5}` (after YAML parse, key is
   the string `"hold"`) raises `ValueError` matching
   `r"keys must be integers"`. Without this test, an implementer who
   omits the `try/except` around `int(k)` would emit a bare
   `ValueError: invalid literal for int()` traceback.
10. `test_activity_by_behavior_rejects_nonpositive_value` — direct
    construction `BalticBioParams(activity_by_behavior={0: 1.0, 1:
    -0.5, ...})` raises `ValueError`. Locks in the new
    `__post_init__` validation that the spec mandates.

**Helper test (4):**

4. `test_origin_aware_activity_mult_dispatch` — wild origin gets
   `lut_wild[behavior]`, HATCHERY origin gets `lut_hatch[behavior]`,
   mixed origin produces correct per-agent dispatch. Graceful path:
   when `lut_hatch is None`, `np.array_equal(result, lut_wild[behavior])`
   element-by-element (NOT just shape — a stub that returns wrong
   values would pass a shape-only check).

**Runtime guard tests (5 + new 11):**

5. `test_introduction_event_runtime_guard_no_hatchery_params` — call
   `IntroductionEvent(origin=ORIGIN_HATCHERY, ...).execute(pop, {}, 0,
   mask)` with empty landscape (so `landscape.get("hatchery_dispatch")
   is None`). Asserts `ValueError` matching `r"HATCHERY.*no
   hatchery_dispatch"`.
11. `test_patch_introduction_event_runtime_guard_no_hatchery_params`
    — mirror of test 5 for `PatchIntroductionEvent` in
    `events_hexsim.py`. Without this, the hexsim-mode introduction
    guard could be silently omitted; existing
    `test_events_hexsim.py` doesn't construct hatchery scenarios.

**Init / rebuild / step integration tests (6, 7 + new 12, 13):**

6. `test_rebuild_luts_resilient_to_slider_replacement` — load a
   Baltic YAML `tmp_path` fixture with hatchery overrides into a full
   `Simulation`. Replace `sim.bio_params` with a mimic of `app.py:1493`'s
   slider construction: `BioParams(RA=0.005, RB=-0.2, RQ=0.07,
   ED_MORTAL=4.0, T_OPT=15.0, T_MAX=24.0)`. Call `sim.rebuild_luts()`.
   Numeric assertions: `sim._activity_lut[1] == pytest.approx(1.2)`
   (Baltic wild RANDOM, NOT Chinook plain-BioParams default of 1.2 —
   coincidentally same; assert UPSTREAM `sim._activity_lut[3] ==
   pytest.approx(1.5)` — this is Baltic-specific). Plus
   `sim.hatchery_dispatch.activity_lut[1] == pytest.approx(1.5)`
   (merged override) AND `sim.hatchery_dispatch.activity_lut[3] ==
   pytest.approx(1.875)`.
7. `test_step_injects_hatchery_dispatch_landscape_key` — load a
   Baltic YAML `tmp_path` fixture with `hatchery_overrides:` into a
   full `Simulation`. Use `monkeypatch.setattr(sim._sequencer, "step",
   spy_fn)` where `spy_fn` captures `args[1]` (the landscape dict).
   After `sim.step()` runs once, assert `"hatchery_dispatch" in
   landscape AND landscape["hatchery_dispatch"] is not None AND
   isinstance(landscape["hatchery_dispatch"], HatcheryDispatch)`.
   Without this strict-typed assertion, an implementer could ship a
   `step()` that injects `None` for the key and the test would still
   pass with a membership-only check.
12. `test_simulation_init_non_baltic_has_no_hatchery_dispatch` —
    construct a Simulation from a non-Baltic config (no
    `species_config:` key). Assert `sim.hatchery_dispatch is None`
    AND `isinstance(sim.bio_params, BioParams)` (NOT
    `BalticBioParams`). Locks in the legacy-path return-type
    semantics that fall out of the unified-return refactor.
13. `test_rebuild_luts_noop_on_non_baltic_sim` — call
    `sim.rebuild_luts()` on the non-Baltic simulation from test 12.
    Asserts no exception raised, and `sim.hatchery_dispatch` remains
    `None`. Without this test, a future implementation could throw
    `KeyError` on the missing `_species_config` cache.

## Tests

Existing tests that may need updating:

- **None expected** — the dispatch helper is graceful when
  `lut_hatch is None`, so all pre-C2 tests (which never construct a
  scenario with hatchery overrides) continue to use the wild path
  unchanged.
- The 13 new tests above lock in the C2-specific paths.

Existing tests that exercise the new init path (`Simulation`
end-to-end, no modifications expected because they don't supply
hatchery overrides):

- `tests/test_simulation.py:8,15,23,30` (function `def` lines)
- `tests/test_integration.py:9,30`
- `tests/test_curonian_realism_integration.py:50`
- `tests/test_curonian_trimesh_integration.py:30`
- `tests/test_hexsim_compat.py:46,61,73`
- `tests/test_nemunas_h3_integration.py:54,135`

## Risk + regression surface

**Behaviour change surface for pre-C2 scenarios: zero.** When
`hatchery_overrides:` is absent from the YAML, `hatchery_dispatch`
remains None, the dispatch helper falls back to wild-only via
`lut_hatch is None`, and every agent — wild by default — receives
exactly the same activity multipliers as before C2.

**Behaviour change surface for C2-aware scenarios:** non-zero by
design. Hatchery-tagged agents accumulate ~0.21 pp more mass loss over
21 days under realistic mixed behavior (calibration-grade; sensitivity
sweep mandatory for publications using these outputs).

**Silent-failure modes (closed by design):**

- ❌ Hatchery agent at runtime, no overrides → caught by runtime
  guard inside `IntroductionEvent.execute()` and
  `PatchIntroductionEvent.execute()`. Locked by tests 5 and 11.
- ❌ YAML typo in top-level override key (e.g., `activity_for_behavior`)
  → caught by strict loader (`ALLOWED_OVERRIDE_KEYS` check). Locked
  by test 3.
- ❌ Invalid Behavior enum sub-key (e.g. `999`) → caught by
  `VALID_BEHAVIORS` check after coercion. Locked by test 8.
- ❌ Non-numeric sub-key (e.g., `"hold"`) → caught by `try/except`
  around `int(k)` coercion with actionable message. Locked by test 9.
- ❌ Non-positive activity value → caught by new `__post_init__`
  validation. Locked by test 10.
- ❌ Sparse override dict truncating LUT → caught by shallow-merge
  into wild base.
- ❌ Sidebar slider stale-LUT bug → caught by `rebuild_luts()`
  reading from cached `BalticSpeciesConfig`.
- ❌ Slider replacement anchoring hatchery LUT to Chinook defaults →
  caught by the species-config re-derivation in `rebuild_luts()`.
  Locked by test 6.
- ❌ Per-slider disk I/O regression → caught by caching
  `BalticSpeciesConfig` on `self._species_config` at init.
- ❌ `step()` not injecting `hatchery_dispatch` into landscape →
  caught by spy-fn test 7 with strict `isinstance(HatcheryDispatch)`
  assertion.
- ❌ Heterogeneous return type from loader producing `AttributeError`
  on non-Baltic configs → caught by unified-return architecture (loader
  ALWAYS returns `BalticSpeciesConfig`). Locked by test 12.
- ❌ Pre-pass-7 paired-nullable design (`bio_params_hatchery` and
  `_activity_lut_hatchery` separately nullable) was prone to silent
  desync (one rebuilt, one stale) → eliminated by `HatcheryDispatch`
  dataclass bundling: the params and LUT are constructed together and
  carried as one nullable.
- ❌ `rebuild_luts()` raising on non-Baltic sim → caught by early
  return when `_species_config` not cached. Locked by test 13.
- ❌ Mutating wild dict via override path → caught by
  `dataclasses.replace` returning new instance. Locked by identity
  assertion in test 1.

## Success criteria

- [ ] `BalticSpeciesConfig` NamedTuple AND `HatcheryDispatch` frozen
      dataclass defined in `baltic_params.py`.
- [ ] `BalticBioParams.__post_init__` validates `activity_by_behavior`
      (non-empty, non-negative int keys, positive float values).
- [ ] `_apply_hatchery_overrides` raises on unsupported keys, invalid
      Behavior sub-keys, and non-numeric sub-key types.
- [ ] `load_bio_params_from_config` ALWAYS returns `BalticSpeciesConfig`
      (legacy path returns `BalticSpeciesConfig(wild=BioParams, hatchery=None)`).
- [ ] `Simulation._species_config` cached at init; `rebuild_luts()`
      reads from cache (no per-slider disk I/O).
- [ ] `Simulation.hatchery_dispatch: HatcheryDispatch | None` attribute
      bundles params + LUT atomically.
- [ ] `Simulation.step()` injects `hatchery_dispatch` into landscape
      dict (single key).
- [ ] `_event_bioenergetics` reads `self.hatchery_dispatch` directly;
      `SurvivalEvent.execute` reads `landscape["hatchery_dispatch"]`.
      Both call `origin_aware_activity_mult` helper.
- [ ] `IntroductionEvent.execute()` and `PatchIntroductionEvent.execute()`
      raise when origin=HATCHERY but `landscape.get("hatchery_dispatch")
      is None`.
- [ ] Startup log message fires when `hatchery_dispatch is not None`.
- [ ] `app.py:1493-1501` Shiny sidebar calls `rebuild_luts()` instead
      of `_build_activity_lut()`.
- [ ] All 13 new tests pass.
- [ ] Full pytest suite stays green (current baseline 829 passing on
      `hatchery-origin-c1` branch; expected post-C2: 842 = baseline + 13).
- [ ] Plan stamp on the eventual implementation plan: ✅ EXECUTED.

## Estimated implementation time

**~2 days** including TDD cycle, the dispatch helper + 7 file edits,
new test file, sensitivity-sweep documentation in YAML, regression
sweep. Reduced from the original 3-5 day estimate because:

- Wisconsin kernel signature is preserved (no Numba recompile / parity
  test changes).
- Only one dataclass field diverges (no per-field dispatch buildout).
- The dispatch hides behind the existing per-agent activity_mult array.
- Override merge uses `dataclasses.replace` + `__post_init__` for
  validation (no custom validation framework needed).

## Out-of-scope (deferred)

- **C3 behaviour divergence** — origin-specific spawn-success, homing
  precision, sea-age sampling, predator-naïveté. Separate spec.
- **Time-decaying multiplier** — `activity_mult(t) = wild + delta *
  exp(-t / tau)` where delta narrows over migration days. Needs
  Curonian Lagoon tagging data not currently available.
- **Other dataclass-field hatchery overrides** (`T_OPT`,
  `T_ACUTE_LETHAL`, `RA`, `MASS_FLOOR_FRACTION`, etc.) — each new
  divergent field gets its own helper following the C2 pattern.
- **Hatchery activity multiplier sliders in Shiny UI** — sidebar
  sliders affect wild only in C2. UX gap acknowledged; defer to C3.
- **`SwitchPopulationEvent` cross-population hatchery target warning**
  — multi-population hatchery scenarios are not deployed today.
- **Generic recursive deep-merge for nested dict fields** — only
  `activity_by_behavior` is dict-typed in BalticBioParams; the C2
  shallow-merge is field-specific. Generalize if/when C3 adds another
  dict field.
- **Pre-existing `app.py:1493` Baltic→plain-BioParams replacement bug**
  — sidebar reset drops Baltic-specific fields like `T_ACUTE_LETHAL`.
  C2 routes around the bug via `rebuild_luts()` re-deriving from
  config, but does not fix the underlying replacement. Separate
  cleanup PR.

## YAML schema documentation

The hatchery overrides block in
`configs/baltic_salmon_species.yaml` (or whichever the deployed config
is) MUST carry the following provenance comment block:

```yaml
species:
  BalticAtlanticSalmon:
    # ... existing wild fields ...
    activity_by_behavior:
      0: 1.0   # HOLD
      1: 1.2   # RANDOM
      2: 0.8   # TO_CWR
      3: 1.5   # UPSTREAM
      4: 1.0   # DOWNSTREAM
    # NOTE: wild baseline inherited from inSTREAM Snyder Chinook;
    # Baltic-specific verification has not been done. The hatchery
    # contrast below is layered on a known-uncertain baseline.
    #
    # C2 hatchery vs wild parameter divergence (see
    # docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md).
    # Empirically-bounded activity scaling factor representing reduced
    # aerobic scope (Zhang et al. 2016, doi:10.1016/j.aquaculture.2016.05.015)
    # and higher swimming cost per unit displacement (Enders, Boisclair
    # & Roy 2004, doi:10.1139/f04-211: 12-29% range in seventh-generation
    # domesticated Salmo salar; F1 costs not statistically different
    # from wild in same study). Lithuanian programme (~4-7 generations
    # since 1997) falls in intermediate range.
    #
    # CALIBRATION STATUS: +25% increment is empirically bracketed but
    # not directly measured for Lithuanian Žeimena/Simnas hatchery
    # stocks. Treat as calibration-grade. Mandatory sensitivity sweep
    # before publication: {0%, +12.5%, +25%, +37.5%}. Cap at +40%
    # without additional citation support.
    #
    # EFFECT SIZE: ~0.21 pp mass-loss differential over 21d realistic
    # mixed-behavior migration. C2 alone is the architectural enabler
    # for compound contrast (with osmoregulation stress, longer
    # corridors, future C3 behaviour divergences); not standalone
    # survival driver.
    #
    # DOWNSTREAM (4) UNCHANGED — passive drift; morphological
    # inefficiency suppressed when current provides thrust. No direct
    # citation; rationale is hydrodynamic (drag vs thrust allocation).
    hatchery_overrides:
      activity_by_behavior:
        1: 1.5     # RANDOM:  +25% (1.2 → 1.5)
        3: 1.875   # UPSTREAM: +25% (1.5 → 1.875)
        # HOLD (0), TO_CWR (2), DOWNSTREAM (4) inherit wild values
        # via shallow-merge.
```
