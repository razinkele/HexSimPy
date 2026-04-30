# Hatchery vs Wild Distinction — C1 Tag-Only

**Date:** 2026-04-30
**Owner:** @razinkele
**Status:** 📋 DRAFT — awaiting writing-plans

This is the **first of three planned tiers** for hatchery-vs-wild origin
support. Sequencing was decided during 2026-04-30 brainstorming:

- **C1 (this spec)** — tag-only: track origin on agents, no behaviour
  difference yet. ~1-2 days.
- **C2 (future plan)** — parameter-level: origin-specific
  `BalticHatcheryBioParams` sister class with hatchery-specific
  physiology constants. ~3-5 days.
- **C3 (future plan)** — full origin-aware behaviour: spawn-success,
  homing precision, sea-age sampling differ by origin. ~1-2 weeks.

After C1 ships, C2 and C3 become entries in `curonian_deferred.md`.

This spec covers C1 *only*.

## Why now

Per memory `curonian_deferred.md`: "most Nemunas salmon are
hatchery-origin from Žeimena and Simnas (Lithuanian programme since
1997). May need separate `BalticBioParams` subconfig (e.g.,
`origin_type: hatchery | wild`). Current plan assumes 'wild' — retrofit
needed when hatchery data arrive."

C1 lays the *scaffold* — origin tracked on agents and exported in
output — without changing simulation physics. This unblocks ensemble
post-processing (partition by origin) and lets C2/C3 add physiology /
behaviour later without re-architecting.

## Scope

**In:**
- New module `salmon_ibm/origin.py` (~10 lines): `Origin` IntEnum +
  module constants `ORIGIN_WILD` / `ORIGIN_HATCHERY` + `ORIGIN_NAMES`
  tuple.
- New `origin: int8` field on `AgentPool.ARRAY_FIELDS`. Default
  `ORIGIN_WILD = 0`. Three-touch change per the existing pattern
  (ARRAY_FIELDS tuple, `__init__` ndarray init, `add_agents` extension
  block).
- New `origin: int = ORIGIN_WILD` field on `IntroductionEvent`
  (`events_builtin.py:200`) and `PatchIntroductionEvent`
  (`events_hexsim.py:397`). Propagated to `Population.add_agents()`.
- `Population.add_agents()` accepts `origin: int = ORIGIN_WILD` keyword
  and writes it into the new column.
- `OutputLogger` tracks `origin` per agent per step (four-touch change:
  `__init__` allocation, `log_step` preallocated assignment, `log_step`
  list append, `to_dataframe()` column) — same pattern as the v1.7.0
  `natal_reach_id` precedent.
- YAML scenario loader: events with `origin: wild` or `origin: hatchery`
  string parse correctly via `ORIGIN_NAMES.index()`. Invalid values
  raise `ValueError` at scenario-load time.
- 7 new tests in `tests/test_origin.py`.

**Out:**
- **No behaviour difference** between origins — physics is identical.
  C2 ships physiology divergence; C3 ships behaviour divergence.
- **No origin inheritance on reproduction** — `ReproductionEvent` (at
  `events_builtin.py:306`) calls `add_agents(...)` without `origin`,
  so offspring default to `ORIGIN_WILD`. The question "should
  hatchery-origin parents produce hatchery-origin offspring?" is
  meaningful but only matters once C2/C3 give origin behavioural
  weight. Defer.
- **No origin inheritance on Phase 3 seedling creation** (at
  `events_phase3.py:295`) — same reasoning as ReproductionEvent.
  Defaults to WILD.
- **Origin IS preserved on inter-population transfer** (at
  `network.py:194`) — see Architecture point 8. Origin is permanent
  per-agent metadata; an agent moving between populations doesn't
  change its origin. This is in scope (it's a correctness bug
  otherwise).
- **No spatial assignment** (the C1.b option from brainstorming —
  origin determined by spawn cell). YAML author tags introductions
  explicitly.
- **No fraction-based assignment** (C1.c) — same reasoning.
- **No `BalticBioParams` changes**. C2 introduces
  `BalticHatcheryBioParams` and the parameter-divergence work; this
  spec just creates the field for C2 to read from.

## Choice of approach

Three options were considered for the origin field representation
(brainstormed 2026-04-30):

1. **IntEnum + module constants** (selected) — mirrors `DOState`
   precedent in `salmon_ibm/estuary.py:61-69`.
2. **Module constants only** — simpler but no type safety; relies on
   ad-hoc validation.
3. **String tag on agents** — rejected; doesn't fit `AgentPool`
   int8-array convention.

**(1) selected** because it matches an established project precedent
(`DOState`), gives ergonomic constants for callers
(`Origin.WILD == ORIGIN_WILD`), supports clean YAML round-trip via
`ORIGIN_NAMES.index()`, and integrates with `OutputLogger`'s CSV export
(`ORIGIN_NAMES[agent.origin]` for human-readable column).

## Architecture

The change touches **9 files** (1 new + 8 modified). All modifications
are additive — no existing field/method is removed or renamed.

**Files modified:**

1. `salmon_ibm/origin.py` — **new file** (~15 lines):

   ```python
   """Origin enum for tracking wild vs hatchery agents.

   See docs/superpowers/specs/2026-04-30-hatchery-origin-c1-design.md.
   Defaults to WILD. Used as int8 metadata on AgentPool agents; no
   behaviour change in C1 — physiology divergence ships in C2.
   """
   from enum import IntEnum

   class Origin(IntEnum):
       WILD = 0
       HATCHERY = 1

   ORIGIN_WILD = Origin.WILD
   ORIGIN_HATCHERY = Origin.HATCHERY
   ORIGIN_NAMES = ("wild", "hatchery")  # index = enum value
   ```

2. `salmon_ibm/agents.py:21-35` (ARRAY_FIELDS tuple) — append `"origin"`.

3. `salmon_ibm/agents.py` (`AgentPool.__init__`, around line 71-72) —
   add `self.origin = np.full(n, ORIGIN_WILD, dtype=np.int8)` immediately
   after the existing `self.exit_branch_id = ...` line. The defensive
   assertion at lines 80-86 will catch this if forgotten.

4. `salmon_ibm/agents.py:1-15` (top imports) — add
   `from salmon_ibm.origin import Origin, ORIGIN_WILD`.

5. `salmon_ibm/population.py:200-245` (`Population.add_agents()`):
   - Accept `origin: int = ORIGIN_WILD` as keyword arg.
   - Add `new_arrays["origin"][old_n:] = origin` line in the
     bulk-prealloc block (next to the `natal_reach_id` line).
   - The existing length-assertion (lines 240-245) automatically
     verifies the new field is extended.

6. `salmon_ibm/events_builtin.py:198-263` (`IntroductionEvent`):
   - Add `origin: int = ORIGIN_WILD` to the dataclass.
   - In `execute()`, pass `origin=self.origin` to
     `population.add_agents(...)`.
   - Top-of-file import: add `from salmon_ibm.origin import ORIGIN_WILD`.

7. `salmon_ibm/events_hexsim.py:395-421` (`PatchIntroductionEvent`):
   - Currently the dataclass has only `patch_spatial_data` field;
     `execute()` calls `population.add_agents(len(nonzero_cells), nonzero_cells)`
     at line 418.
   - Add `origin: int = ORIGIN_WILD` to the dataclass.
   - Update the `add_agents()` call to pass `origin=self.origin`.
   - Top-of-file import: add `from salmon_ibm.origin import ORIGIN_WILD`.

8. `salmon_ibm/network.py:194-201` (`MultiPopulationManager` transfer
   between populations):
   - The transfer path already preserves `natal_reach_id` and
     `exit_branch_id` from source to target on agent migration (lines
     199-201). **Origin must be preserved the same way** — otherwise a
     hatchery agent transferring to another population would silently
     reset to WILD, breaking origin-aware ensemble post-processing.

   After line 201, add:

   ```python
   if hasattr(target, "origin") and hasattr(source, "origin"):
       target.origin[new_idx] = source.origin[transfer]
   ```

   This sits next to the existing `natal_reach_id`/`exit_branch_id`
   preservation block. The `hasattr` guards mirror the existing
   defensive pattern.

   No new tests needed — `network.py` likely already has tests for
   transfer; an existing test that transfers an agent with non-WILD
   origin would catch any regression.

9. `salmon_ibm/output.py` (`OutputLogger`) — five touch points (verified
   2026-04-30 against the existing `natal_reach_id` precedent at lines
   41, 53, 79, 99, 117, 134):

   - `__init__` (preallocated branch, line 42): add
     `self._origin_arr = np.empty((max_steps, max_agents), dtype=np.int8)`.
   - `__init__` (list branch, line 54): add
     `self._origin: list[np.ndarray] = []`.
   - `log_step` (preallocated branch, line 80): add
     `self._origin_arr[r, :n] = pool.origin[:n]`.
   - `log_step` (list branch, line ~91): add
     `self._origin.append(pool.origin.copy())`.
   - `to_dataframe()` empty-cols list (line 99): add `"origin"` to the
     `empty_cols` list.
   - `to_dataframe()` preallocated dict (line 118): add
     `"origin": self._origin_arr[r, :n]`.
   - `to_dataframe()` list dict (line 135): add
     `"origin": np.concatenate(self._origin)`.

   **CSV format decision: int8 values** (0 / 1) — for performance and
   Pandas convention. Downstream consumers map via
   `ORIGIN_NAMES[origin_col]` if they want human-readable strings.

10. `salmon_ibm/scenario_loader.py` (`_build_single_event`, around line 309-323):
   The loader applies event params via `setattr(evt, key, val)` after
   constructing the event with defaults. YAML/XML carries
   `params: {origin: "hatchery"}` as a string; the dataclass field
   expects an int. So we must convert before the setattr loop.

   Insert before the `for key, val in params.items()` loop:

   ```python
   # Convert origin string ("wild"/"hatchery") to int8. Surfaces
   # invalid values at scenario-load time, not first simulation step.
   if "origin" in params and isinstance(params["origin"], str):
       from salmon_ibm.origin import ORIGIN_NAMES
       s = params["origin"]
       try:
           params["origin"] = ORIGIN_NAMES.index(s)
       except ValueError as exc:
           raise ValueError(
               f"Invalid origin '{s}'; expected one of {ORIGIN_NAMES}"
           ) from exc
   ```

   Integer values (e.g. `origin: 0`) pass through unchanged. Already-int
   values from earlier code paths also pass through unchanged.

## Tests

**New file: `tests/test_origin.py`** — 7 tests:

1. `test_origin_enum_values` — `Origin.WILD == 0`, `Origin.HATCHERY == 1`.
2. `test_origin_names_roundtrip` — `ORIGIN_NAMES.index("wild") == 0`;
   `ORIGIN_NAMES[Origin.HATCHERY] == "hatchery"`.
3. `test_origin_names_invalid_raises` — `ORIGIN_NAMES.index("salmon")`
   raises `ValueError`.
4. `test_agent_pool_origin_default_wild` — `AgentPool(n=10, ...)` has
   `pool.origin == np.zeros(10, dtype=np.int8)`.
5. `test_population_add_agents_with_origin` — `pop.add_agents(5, ...,
   origin=ORIGIN_HATCHERY)` writes `1` to the new agents' origin slots
   while leaving existing agents at `0`.
6. `test_introduction_event_propagates_origin` — running an
   `IntroductionEvent(origin=ORIGIN_HATCHERY).execute(pop, landscape, t,
   mask)` results in the new agents having `pop.pool.origin[new_idx]
   == 1`.
7. `test_yaml_origin_string_parses` — scenario YAML with
   `origin: hatchery` loads the event with the int field set; with
   `origin: salmon` raises `ValueError` at load time.

## Risk + regression surface

**Behaviour change surface: zero.** C1 is metadata-only. No agent
takes a different action because of origin.

**Tests that may need updating:** none expected. The `AgentPool`
defensive assertion at line 80-86 will fire if `__init__` fails to
initialize the new field. The `add_agents()` defensive assertion at
line 240-245 will fire if the field isn't extended. Both surface
issues at first instantiation, not in production. Existing tests that
construct `AgentPool` or call `add_agents` will continue to work
because `origin` defaults to `ORIGIN_WILD` and existing tests don't
care about origin.

**Mitigation:** run full pytest suite, surface regressions, fix or
update them. Same playbook as v1.7.3 osmoregulation.

## Success criteria

- [ ] `salmon_ibm/origin.py` exists with `Origin` IntEnum +
      `ORIGIN_WILD`/`ORIGIN_HATCHERY` + `ORIGIN_NAMES`.
- [ ] `AgentPool.ARRAY_FIELDS` includes `origin`; defensive assertions
      pass.
- [ ] `Population.add_agents(origin=...)` accepts the keyword and
      writes to the new column.
- [ ] `IntroductionEvent` and `PatchIntroductionEvent` have an
      `origin` dataclass field; both propagate to `add_agents`.
- [ ] `OutputLogger` writes an `origin` column in `to_dataframe()`.
- [ ] YAML `origin: hatchery` string round-trips through
      `scenario_loader`.
- [ ] All 7 new tests pass.
- [ ] Full pytest suite stays green (current baseline 821 passing on
      main; expected post-C1: 828).
- [ ] Plan stamp on the eventual implementation plan: ✅ EXECUTED.

## Estimated implementation time

**1-2 days** including TDD cycle, OutputLogger four-touch wire-through,
scenario-loader update, regression sweep. Most tasks are mechanical.

## Out-of-scope (deferred)

- **Behaviour divergence (C2, C3)** — physiology constants per origin,
  origin-specific decisions. Each gets its own future plan once C1
  ships.
- **Origin inheritance on reproduction** — meaningful only when origin
  drives behaviour; defer to C2.
- **Spatial origin assignment** (C1.b) — agents at Žeimena/Simnas cells
  auto-tagged hatchery. Possible follow-on; not blocking.
- **Fraction-based origin assignment** (C1.c) — `hatchery_fraction`
  YAML key sampled at population init. Possible follow-on.
- **`BalticBioParams` changes** — that's C2.
