# Deep Review Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all verified correctness, security, robustness, and performance issues identified in the 2026-04-23 five-pass codebase review of the Baltic Salmon IBM.

**Architecture:** Small, focused, TDD-driven commits grouped by subsystem. Each task is independent and lands its own commit. Critical science correctness first; then security hardening; then robustness; then performance; finally test polish. Two large architectural refactors (viewer split, interactions vectorization) are deferred to separate plans.

**Tech Stack:** Python 3.10+, NumPy, Numba `@njit`, pytest, conda env `shiny`.

**Test command:** `conda run -n shiny python -m pytest tests/ -v` (full suite, ~4 min) or targeted: `conda run -n shiny python -m pytest tests/test_FILE.py::TEST_NAME -v`.

> **Env note:** On a machine where only `micromamba` is installed (no `conda` on PATH), substitute `micromamba run -n shiny` for every `conda run -n shiny` in this plan. They are interchangeable for pytest invocation. Verify with `which conda` / `which micromamba` at the start of execution.
>
> **Pre-existing env gap (not plan-caused):** the `shiny` env does not have `pytest-mock` installed, so `tests/test_events.py::TestMovementEvent::test_calls_execute_movement` errors at setup with "fixture 'mocker' not found". This is a pre-existing issue — not your concern when implementing plan tasks. Expect 1 error in test_events.py baseline.

**Scope note — what is NOT in this plan:**
- The reviewer's `events_hexsim.py:43-68` "`prange` data race on `distances[i]`" — **false positive**. Each `prange` iteration owns a unique `i`; reads/writes are on disjoint memory. The CLAUDE.md `any_moved` rule is about *shared scalars*, not per-agent arrays. No fix needed.
- Big refactors (`hexsim_viewer.py` split, `interactions.py` full vectorization rewrite) — create separate plans.

**Plan revision history:**
- **v1 (initial)** — drafted from 5-pass review synthesis.
- **v2 (2026-04-23, post-review)** — three review agents audited v1 and caught:
  - Task 1's `ED_TISSUE=5 kJ/g` was algebraically inverted (ED would *rise*, not fall). Corrected to `ED_TISSUE=36 kJ/g` (lipid catabolism, Brett 1995 / Breck 2008), added blocking note requiring modeler sign-off, added guidance for updating pre-existing conservation/monotonicity tests.
  - Task 2's "add `updater_individual_locations`" was wrong — the function already exists at `accumulators.py:534`. Simplified to a dispatch-only change.
  - Task 3 had a conservation bug — target was receiving pre-clamp nominal amount, not actual amount moved. Fix now tracks `actual_amount = src_before - new_src`.
  - Task 9 duplicated `agent_ids` and `_next_id` onto `AgentPool`, but `Population` already owns them (line 27-34) and `compact()` already preserves them (line 160). Rewrote to simply change `log_step` to accept `Population` and use `population.agent_ids`.
  - Task 15 was realigned to the new `log_step(population)` signature from Task 9.
  - Task 16 replaced Roegner 2011 (Pacific salmon) with Liland 2024 (*Salmo salar*) and added a DO-field unit sanity check (mg/L vs mmol/m³).
- **v3 (2026-04-23, post-v2-verification)** — one more focused review pass found two residual gaps, both fixed:
  - Task 9 didn't name the three existing `test_output.py` callers (`test_logger_creates_file`, `test_logger_records_correct_columns`, `test_logger_accumulates_steps`) that would break from the signature change. Added explicit wrapping instructions with line numbers.
  - Task 2 left a stale `"IndividualLocations": "individual_locations"` string sentinel in the `_dispatch` dict. Added Step 4 to replace it with the actual callable and remove the special-case branch.
- **v4 (2026-04-23, security-depth pass)** — a focused security audit on Tasks 5-6 found four real attack-surface gaps and one sequencing hazard:
  - MEDIUM DoS via `_rng.random(10**9)` — sandbox allows any `ast.Constant` arg with no magnitude bound. Added `_RNG_ARG_MAX = 1_000_000` guard and test to Task 5.
  - MEDIUM DoS via `_compiled_expr_cache` flooding (bulk-clear at 10,000 keeps 9,999 entries indefinitely). Added new Task 5b: LRU-bounded `OrderedDict` cache capped at 256.
  - LOW silent corruption via `spatial_data` keys shadowing `_SAFE_MATH` functions (e.g., a spatial layer named `"sqrt"` could replace `np.sqrt`). Added a `key in _SAFE_MATH` skip guard to Task 6.
  - LOW silent corruption via `output_name` overwriting landscape structural keys (`"fields"`, `"mesh"`, etc.). Added a `_PROTECTED` set guard to Task 6.
  - Task 5's `accumulators.py:186-198` line reference is stable relative to Task 3's edit (at line 341-360), but added a grep-verify note for safety.
- **Dismissed (review-loop signal filtering):** An overlap audit claimed Tasks 2-11 duplicate `2026-03-29-codebase-review-fixes.md`. Verified false — that agent matched by task NUMBER across plans rather than topic. The March 29 plan covers UI/app-layer fixes (already shipped per `git log`: commits a8722df, b324204, c731057, etc.). Zero real overlap with this plan's substantive work.
- **v5 (2026-04-23, execution dry-run)** — executed Task 4 end-to-end in an isolated worktree to validate the plan against reality:
  - Task 4 worked **exactly as written**: failing test failed for the predicted reason, fix applied cleanly, test passed, full `test_events.py` + `test_events_hexsim.py` had zero regressions (49 passed vs 48 baseline).
  - **Caught:** the plan's `conda run -n shiny` commands fail on machines where only `micromamba` is installed. Added env note at the top of the plan.
  - **Caught:** `test_events.py::TestMovementEvent::test_calls_execute_movement` errors with "fixture 'mocker' not found" — pre-existing env gap (pytest-mock not installed in `shiny`). Documented at the top so implementers don't chase a phantom regression.

---

## File Structure (what will change)

**Modified files:**
- `salmon_ibm/bioenergetics.py` — fix starvation physics, mortality ordering
- `salmon_ibm/accumulators.py` — fix IndividualLocations fallback, accumulator_transfer clamp, tighten AST sandbox
- `salmon_ibm/events.py` — call `clear_combo_mask_cache()` from single-pop sequencer
- `salmon_ibm/events_phase3.py` — replace substring-match landscape injection with explicit allowlist
- `salmon_ibm/events_hexsim.py` — fix IndividualLocations dispatch string, temperature-push gating
- `salmon_ibm/simulation.py` — gate `push_temperature` to alive agents
- `salmon_ibm/agents.py` — assert ARRAY_FIELDS coverage in `__init__`
- `salmon_ibm/output.py` — use population.agent_ids for stable IDs; preallocate log buffers
- `salmon_ibm/movement.py` — cache contiguous centroids
- `tests/test_bioenergetics.py` — add starvation-mortality regression test
- `tests/test_events_phase3.py` — add landscape-injection allowlist test
- `tests/test_accumulators.py` — add IndividualLocations correctness test, AST sandbox test
- `tests/test_events.py` — add combo-mask cache-clear test for single-pop sequencer
- `tests/test_numba_fallback.py` — add Numba↔NumPy parity test
- `tests/test_playwright.py`, `tests/test_map_visualization.py` — UI test polish

**New files:** none (all fixes live in existing modules).

---

# Phase 1 — CRITICAL Science Correctness

## Task 1: Fix starvation mortality in Wisconsin bioenergetics

> **⚠️ BLOCKING: Modeler sign-off required before coding this task.**
> The direction of ED change under starvation depends critically on whether catabolized tissue density is *above* or *below* whole-body ED. The current `BioParams.ED_TISSUE = 5.0 kJ/g` represents whole-fish wet-mass energy density (Penney & Moffitt 2014), NOT the preferentially-burned lipid. With `ED_TISSUE < ed_kJ_g` starting state, the new formula makes ED *rise*, not fall — the opposite of the intended fix.
>
> **Options to decide before coding:**
> **(A) Lipid-first catabolism (recommended):** set `ED_TISSUE ≈ 36 kJ/g` (pure lipid, Brett 1995; Breck 2008). Formula below is correct; ED declines monotonically toward `ED_MORTAL`.
> **(B) Mixed lipid+protein catabolism:** set `ED_TISSUE ≈ 25-30 kJ/g` with explicit citation; same formula, slightly slower ED decline.
> **(C) Keep proportional-loss + remove mortality-hiding floor:** retain current `new_mass = mass * (new_e/old_e)` but drop `MASS_FLOOR_FRACTION`; ED stays constant but mass→0 triggers mortality through a `new_mass <= 0` check.
>
> The plan below assumes **Option A**. Before starting the task, the implementing engineer must either (i) confirm Option A with the modeler, OR (ii) rewrite the task for the chosen option. Do NOT proceed without sign-off.

**Problem:** `bioenergetics.py:64-85` computes proportional mass loss (`new_mass = mass * (new_e/old_e)`), which keeps energy density (ED) constant until `MASS_FLOOR_FRACTION=0.5` kicks in. Starvation mortality is hidden behind a two-regime behavior.

**Math check (for Option A, ED_TISSUE = 36 kJ/g):**
- M=1000g, e=6kJ/g, E=6·10⁶J, r=10,000J.
- mass_lost = 10,000/(36·1000) = 0.278g → M' = 999.72g.
- new_E = 5.99·10⁶J → new_ed = 5,990,000/(999,720) = 5.99 kJ/g. ✓ ED declines.

**Math check (for ED_TISSUE = 5, which is WRONG):**
- Same inputs: mass_lost = 2g → M' = 998g.
- new_ed = 5,990,000/998,000 = 6.002 kJ/g. ✗ ED rises.

**Files:**
- Modify: `salmon_ibm/bioenergetics.py:17-46` (BioParams.ED_TISSUE default) + `:64-85` (update_energy)
- Test: `tests/test_bioenergetics.py` (add new tests AND update existing tests — see Step 0)

- [ ] **Step 0: Inventory existing bioenergetics tests that will need updating**

Before writing new tests, identify tests that baked in old (ED-constant) physics:

```bash
conda run -n shiny grep -n "def test_" tests/test_bioenergetics.py
```

Specifically expect these to fail after the fix and need adjustment:
- `test_energy_conservation_single_step` — asserts `e_before - e_after == r_hourly` exactly. After the fix, `e_after` may differ because the mass floor (applied after mortality) is still in effect. Change to: `assert e_before - e_after <= r_hourly + 1e-6` OR exclude floor-triggered rows.
- `test_energy_density_decreases_monotonically` — will now actually verify the property; no change needed if the new formula is correct.

Make a note of which existing tests need updating in Step 4.

- [ ] **Step 1: Write a failing test that asserts ED declines under starvation**

Add to `tests/test_bioenergetics.py`:

```python
def test_energy_density_declines_under_starvation():
    """ED must decline monotonically when no feeding occurs.

    With lipid-first catabolism (ED_TISSUE > whole-body ED), burning tissue at
    a higher density than the body mean concentrates mass loss into the lipid
    pool, so remaining body ED declines. If ED_TISSUE < body ED, ED would RISE
    under starvation — test explicitly guards against that inversion.
    """
    from salmon_ibm.bioenergetics import update_energy, BioParams

    params = BioParams()
    # Guard: confirm physics direction is correct.
    assert params.ED_TISSUE > 6.0, (
        f"ED_TISSUE={params.ED_TISSUE} must exceed typical whole-body ED (~6) "
        f"for starvation to reduce ED. See Task 1 blocking note."
    )

    ed = np.array([6.0])          # start above ED_MORTAL (4.0) and below ED_TISSUE
    mass = np.array([1000.0])     # 1 kg
    temp = np.array([20.0])       # warm water, high respiration
    activity = np.array([1.5])
    salinity_cost = np.array([1.0])

    ed_trace = [ed[0]]
    for _ in range(200):
        ed, _dead, mass = update_energy(ed, mass, temp, activity, salinity_cost, params)
        ed_trace.append(ed[0])

    diffs = np.diff(ed_trace)
    assert np.all(diffs <= 1e-9), (
        f"ED must never increase under starvation; saw rises: {diffs[diffs > 0]}"
    )
    assert ed_trace[-1] < ed_trace[0] - 0.1, (
        f"ED must decline over 200h; got {ed_trace[0]} -> {ed_trace[-1]}"
    )


def test_starvation_triggers_mortality_when_energy_depleted():
    """Agents at very low energy must die (new_ed < ED_MORTAL)."""
    from salmon_ibm.bioenergetics import update_energy, BioParams

    params = BioParams()
    ed = np.array([4.1])          # just above mortal threshold
    mass = np.array([100.0])      # small fish
    temp = np.array([25.0])       # near T_MAX, high respiration
    activity = np.array([1.5])
    salinity_cost = np.array([1.2])

    died = False
    for _ in range(500):
        ed, dead, mass = update_energy(ed, mass, temp, activity, salinity_cost, params)
        if dead[0]:
            died = True
            break
    assert died, "Agent with starting ED near ED_MORTAL must eventually die under starvation"
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
conda run -n shiny python -m pytest tests/test_bioenergetics.py::test_energy_density_declines_under_starvation tests/test_bioenergetics.py::test_starvation_triggers_mortality_when_energy_depleted -v
```

Expected: FAIL — current proportional-loss logic keeps ED flat, so the "ED must decline meaningfully" assertion fails; the mortality test likely also fails.

- [ ] **Step 3: Update `BioParams.ED_TISSUE` default to match lipid catabolism**

In `salmon_ibm/bioenergetics.py`, change line 20 default:

```python
    ED_TISSUE: float = 36.0  # lipid catabolism density (Brett 1995; Breck 2008)
```

- [ ] **Step 4: Rewrite `update_energy` to catabolize at `ED_TISSUE`**

Replace `salmon_ibm/bioenergetics.py:64-85` with:

```python
def update_energy(
    ed_kJ_g: np.ndarray,
    mass_g: np.ndarray,
    temperature_c: np.ndarray,
    activity_mult: np.ndarray,
    salinity_cost: np.ndarray,
    params: BioParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_hourly = (
        hourly_respiration(mass_g, temperature_c, activity_mult, params) * salinity_cost
    )
    e_total_j = ed_kJ_g * 1000.0 * mass_g
    new_e_total_j = np.maximum(e_total_j - r_hourly, 0.0)

    # Non-feeding migrants catabolize lipid-rich tissue at ED_TISSUE kJ/g.
    # Mass declines as tissue is burned: dm = -r_hourly / (ED_TISSUE * 1000).
    # For ED_TISSUE > whole-body ED (lipid-first catabolism), ED declines
    # smoothly toward ED_MORTAL. See Task 1 physics note above.
    tissue_ed_j_per_g = params.ED_TISSUE * 1000.0
    mass_lost = np.where(tissue_ed_j_per_g > 0, r_hourly / tissue_ed_j_per_g, 0.0)
    new_mass = mass_g - mass_lost

    # Mortality is determined BEFORE the mass floor, from true energy state.
    # (Applying the floor first would artificially inflate ED above ED_MORTAL.)
    safe_mass = np.where(new_mass > 0, new_mass, 1e-9)
    new_ed_raw = np.where(new_mass > 0, new_e_total_j / (safe_mass * 1000.0), 0.0)
    dead = (new_ed_raw < params.ED_MORTAL) | (new_mass <= 0)

    # Numerical floor to prevent mass -> 0 blowups in downstream code.
    # Applied after mortality decision, so it does not mask starvation.
    new_mass = np.maximum(new_mass, mass_g * params.MASS_FLOOR_FRACTION)
    new_ed = np.where(new_mass > 0, new_e_total_j / (new_mass * 1000.0), 0.0)
    return new_ed, dead, new_mass
```

- [ ] **Step 5: Run the new tests — expect PASS**

```bash
conda run -n shiny python -m pytest tests/test_bioenergetics.py::test_energy_density_declines_under_starvation tests/test_bioenergetics.py::test_starvation_triggers_mortality_when_energy_depleted -v
```

- [ ] **Step 6: Run the full bioenergetics test file — expect SOME PRE-EXISTING TESTS TO FAIL**

```bash
conda run -n shiny python -m pytest tests/test_bioenergetics.py -v
```

Tests known to likely fail and how to fix:

**If `test_energy_conservation_single_step` fails:** The assertion `e_before - e_after == r_hourly` no longer holds exactly because the mass floor, when active, can trap energy on the mass axis. Change the test to allow for floor-trapping:

```python
# Old: assert (e_before - e_after) == pytest.approx(r_hourly)
# New: energy loss must be <= respiration (floor can trap some)
assert (e_before - e_after) <= r_hourly + 1e-6
# Additionally, when far from the floor, conservation holds exactly:
if new_mass > mass_g * params.MASS_FLOOR_FRACTION + 1.0:
    assert (e_before - e_after) == pytest.approx(r_hourly, rel=1e-6)
```

**If any other pre-existing test fails:** inspect and decide whether the test baked in old physics. Update the assertion to match new correct physics, not the other way round.

- [ ] **Step 7: Run the Snyder reference tests — expect PASS**

```bash
conda run -n shiny python -m pytest tests/test_snyder_reference.py -v
```

`test_parameter_identity` only checks RA/RB/RQ/ED_MORTAL, not starvation dynamics.

- [ ] **Step 8: Run the full suite to spot integration regressions**

```bash
conda run -n shiny python -m pytest tests/ -v
```

Downstream tests that depend on ED trajectory (e.g., `test_simulation.py`, ensemble parity) may need baseline refresh.

- [ ] **Step 9: Commit**

```bash
git add salmon_ibm/bioenergetics.py tests/test_bioenergetics.py
git commit -m "fix(bioenergetics): lipid-first catabolism (ED_TISSUE=36) so ED declines under starvation"
```

---

# Phase 2 — Accumulator Correctness

## Task 2: Fix IndividualLocations zero-fallback

**Problem:** `events_hexsim.py:354-363` — when the spatial layer for `IndividualLocations` is missing, it calls `updater_quantify_location` with `hex_map=np.zeros(1)`, silently writing `0.0` for all agents. The correct behavior is to write each agent's `tri_idx`.

**Good news:** `updater_individual_locations` **already exists** at `salmon_ibm/accumulators.py:534-543` with the exact signature needed. This task is just a dispatch change.

**Files:**
- Test: `tests/test_accumulators.py`
- Modify: `salmon_ibm/events_hexsim.py:354-363`

- [ ] **Step 1: Write the regression test (guards against regression of the zero-fallback bug)**

Add to `tests/test_accumulators.py`:

```python
def test_individual_locations_writes_cell_indices():
    """IndividualLocations must write each agent's tri_idx to its accumulator."""
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_individual_locations

    defs = [AccumulatorDef(name="pos", min_val=0.0, max_val=None)]
    mgr = AccumulatorManager(n_agents=4, definitions=defs)
    cell_indices = np.array([10, 20, 30, 40], dtype=np.int64)
    mask = np.ones(4, dtype=bool)

    updater_individual_locations(mgr, "pos", mask, cell_indices=cell_indices)

    np.testing.assert_array_equal(mgr.data[:, 0], [10.0, 20.0, 30.0, 40.0])
```

- [ ] **Step 2: Run — expect PASS (`updater_individual_locations` already exists)**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py::test_individual_locations_writes_cell_indices -v
```

If this FAILs, something else is wrong — investigate before proceeding.

- [ ] **Step 3: Replace the zero-fallback call in `events_hexsim.py:354-363`**

Current (note: exact indentation must match existing dispatch block):

```python
                else:
                    # Spatial data updaters (IndividualLocations, QuantifyLocation, ExploredRunningSum)
                    if spatial and spatial in spatial_data:
                        layer = spatial_data[spatial]
                        self._dispatch.get("QuantifyLocation", handler)(
                            acc_mgr,
                            acc_name,
                            uf_mask,
                            hex_map=layer,
                            cell_indices=population.tri_idx,
                        )
                    elif func_name == "IndividualLocations":
                        from salmon_ibm.accumulators import updater_quantify_location

                        updater_quantify_location(
                            acc_mgr,
                            acc_name,
                            uf_mask,
                            hex_map=np.zeros(1),
                            cell_indices=population.tri_idx,
                        )
```

Replace with:

```python
                else:
                    # Spatial data updaters (IndividualLocations, QuantifyLocation, ExploredRunningSum)
                    if spatial and spatial in spatial_data:
                        layer = spatial_data[spatial]
                        self._dispatch.get("QuantifyLocation", handler)(
                            acc_mgr,
                            acc_name,
                            uf_mask,
                            hex_map=layer,
                            cell_indices=population.tri_idx,
                        )
                    elif func_name == "IndividualLocations":
                        from salmon_ibm.accumulators import updater_individual_locations

                        updater_individual_locations(
                            acc_mgr,
                            acc_name,
                            uf_mask,
                            cell_indices=population.tri_idx,
                        )
```

- [ ] **Step 4: Remove the stale dispatch-dict sentinel entry**

At `events_hexsim.py:260`, the `_dispatch` dict still has `"IndividualLocations": "individual_locations"` (a string sentinel, not a callable). After Step 3, this entry is dead — the `elif func_name == "IndividualLocations"` branch bypasses the dispatch dict entirely. Leaving the string entry will confuse future maintainers.

Change the entry to the actual callable:

```python
self._dispatch["IndividualLocations"] = updater_individual_locations
```

(You'll need to import `updater_individual_locations` from `salmon_ibm.accumulators` at the top of `events_hexsim.py` if it isn't already.)

Then remove the `elif func_name == "IndividualLocations"` special-case branch from Step 3 — the dispatch table now handles it uniformly.

- [ ] **Step 5: Run full test suite — expect PASS**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py tests/test_events_hexsim.py -v
```

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/events_hexsim.py tests/test_accumulators.py
git commit -m "fix(events_hexsim): IndividualLocations fallback uses existing updater instead of zero hex_map"
```

---

## Task 3: Clamp source accumulator to `min_val` after `Transfer`

**Problem:** `accumulators.py:341-360` `updater_accumulator_transfer` subtracts `amount` from source without clamping to `src_defn.min_val`; target is clamped but source is not.

**Files:**
- Test: `tests/test_accumulators.py`
- Modify: `salmon_ibm/accumulators.py:341-360`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_accumulators.py`:

```python
def test_accumulator_transfer_clamps_source_to_min_val():
    """Source accumulator must not fall below its own min_val after transfer."""
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_accumulator_transfer

    defs = [
        AccumulatorDef(name="src", min_val=1.0, max_val=100.0),
        AccumulatorDef(name="tgt", min_val=0.0, max_val=100.0),
    ]
    mgr = AccumulatorManager(n_agents=2, definitions=defs)
    mgr.data[:, 0] = 5.0  # src = 5, min_val = 1.0
    mask = np.ones(2, dtype=bool)

    # Fraction=1.0 would drive src to 0, below min_val=1.0
    updater_accumulator_transfer(mgr, "src", "tgt", mask, fraction=1.0)

    assert np.all(mgr.data[:, 0] >= 1.0), f"src must clamp to min_val=1.0, got {mgr.data[:, 0]}"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py::test_accumulator_transfer_clamps_source_to_min_val -v
```

- [ ] **Step 3: Also write a conservation test (source clamp must not create or destroy mass)**

Add to `tests/test_accumulators.py`:

```python
def test_accumulator_transfer_conserves_mass_under_clamp():
    """When source clamp reduces the actual subtracted amount, target must
    receive the actual amount — not the pre-clamp nominal amount.
    Conservation: delta_src + delta_tgt == 0 (modulo target clamping)."""
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef, updater_accumulator_transfer

    defs = [
        AccumulatorDef(name="src", min_val=1.0, max_val=100.0),
        AccumulatorDef(name="tgt", min_val=0.0, max_val=100.0),
    ]
    mgr = AccumulatorManager(n_agents=2, definitions=defs)
    mgr.data[:, 0] = 5.0  # src = 5, clamp floor 1.0
    mgr.data[:, 1] = 0.0
    mask = np.ones(2, dtype=bool)

    updater_accumulator_transfer(mgr, "src", "tgt", mask, fraction=1.0)

    # src is clamped from 0 back up to 1.0 → actual amount moved was 4.0, not 5.0.
    # Target must receive exactly 4.0 (no phantom mass).
    np.testing.assert_array_almost_equal(mgr.data[:, 0], [1.0, 1.0])
    np.testing.assert_array_almost_equal(mgr.data[:, 1], [4.0, 4.0])
```

- [ ] **Step 4: Run both tests — expect FAIL**

- [ ] **Step 5: Clamp source AND pass the ACTUAL amount moved to the target**

Replace `salmon_ibm/accumulators.py:341-360` with:

```python
def updater_accumulator_transfer(
    manager,
    source_name: str,
    target_name: str,
    mask,
    *,
    fraction: float = 1.0,
):
    """Transfer a fraction of one accumulator's value to another.

    Conservation: the actual amount moved from source (which may differ from
    `fraction * src_value` if the source clamps at min_val) is what the target
    receives. The pre-clamp nominal amount is never added to the target.
    """
    src_idx = manager._resolve_idx(source_name)
    tgt_idx = manager._resolve_idx(target_name)

    src_before = manager.data[mask, src_idx].copy()
    nominal_amount = src_before * fraction
    src_defn = manager.definitions[src_idx]
    new_src = src_before - nominal_amount
    if src_defn.min_val is not None:
        new_src = np.maximum(new_src, src_defn.min_val)
    if src_defn.max_val is not None:
        new_src = np.minimum(new_src, src_defn.max_val)
    manager.data[mask, src_idx] = new_src

    # Actual mass moved = how much source actually changed. Preserves conservation.
    actual_amount = src_before - new_src

    tgt_defn = manager.definitions[tgt_idx]
    new_tgt = manager.data[mask, tgt_idx] + actual_amount
    if tgt_defn.min_val is not None:
        new_tgt = np.maximum(new_tgt, tgt_defn.min_val)
    if tgt_defn.max_val is not None:
        new_tgt = np.minimum(new_tgt, tgt_defn.max_val)
    manager.data[mask, tgt_idx] = new_tgt
```

- [ ] **Step 6: Run both tests — expect PASS**

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/accumulators.py tests/test_accumulators.py
git commit -m "fix(accumulators): Transfer clamps source and conserves actual amount moved"
```

---

## Task 4: Clear combo-mask cache in single-pop `EventSequencer.step()`

**Problem:** `events.py:97` — `EventSequencer.step()` does not call `clear_combo_mask_cache()`. `MultiPopEventSequencer.step()` (line 141) does. Trait-combo masks leak stale data across steps in the default single-pop sequencer.

**Files:**
- Test: `tests/test_events.py`
- Modify: `salmon_ibm/events.py:97-102`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_events.py`:

```python
def test_single_pop_sequencer_clears_combo_mask_cache_each_step():
    """EventSequencer.step() must clear per-step combo mask cache (parity with MultiPopEventSequencer)."""
    from salmon_ibm.events import EventSequencer
    from salmon_ibm.events_hexsim import _combo_mask_cache
    from unittest.mock import MagicMock

    # Populate cache directly to simulate a leak from a prior step
    _combo_mask_cache[("sentinel",)] = np.array([True])

    population = MagicMock()
    population.alive = np.array([True])
    population.arrived = np.array([False])
    landscape = {}

    seq = EventSequencer(events=[])
    seq.step(population, landscape, t=0)

    assert ("sentinel",) not in _combo_mask_cache, (
        "EventSequencer.step() must clear the combo mask cache"
    )
```

- [ ] **Step 2: Run — expect FAIL**

```bash
conda run -n shiny python -m pytest tests/test_events.py::test_single_pop_sequencer_clears_combo_mask_cache_each_step -v
```

- [ ] **Step 3: Add the cache clear to `EventSequencer.step()`**

Replace `salmon_ibm/events.py:97-102` with:

```python
    def step(self, population, landscape, t: int) -> None:
        landscape["step_alive_mask"] = population.alive & ~population.arrived
        # Clear per-step caches (parity with MultiPopEventSequencer)
        from salmon_ibm.events_hexsim import clear_combo_mask_cache

        clear_combo_mask_cache()
        for event in self.events:
            if event.trigger.should_fire(t):
                mask = self._compute_mask(population, event.trait_filter)
                event.execute(population, landscape, t, mask)
```

- [ ] **Step 4: Run the test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/events.py tests/test_events.py
git commit -m "fix(events): clear combo mask cache in single-pop EventSequencer.step()"
```

---

# Phase 3 — Security Hardening

## Task 5: Tighten AST sandbox — allowlist attribute method names on `_rng`

**Problem:** `accumulators.py:186-193` allows any `_rng.<attr>`, and `ast.Call` with `ast.Attribute` func is not allowlisted. A scenario expression can call `_rng.seed(123)` (reproducibility DoS) or `_rng.bytes(10**9)` (memory DoS). Defense-in-depth: restrict to the four methods the HexSim DSL actually uses.

**Files:**
- Test: `tests/test_accumulators.py`
- Modify: `salmon_ibm/accumulators.py:186-198`

> **Note on line numbers:** if Task 3 (`updater_accumulator_transfer` rewrite) has already been applied, line 341-360 grows by ~15 lines. The `186-198` range here is stable (it's well above Task 3's edit point), but always verify with `grep -n "isinstance(node, ast.Attribute)" salmon_ibm/accumulators.py` before patching.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_accumulators.py`:

```python
def test_ast_sandbox_allows_rng_random():
    """Legitimate _rng.random() call must pass validation."""
    from salmon_ibm.accumulators import _validate_expression, _HEXSIM_FUNCTIONS

    # Should not raise
    _validate_expression("_rng.random(1)", extra_names=_HEXSIM_FUNCTIONS)
    _validate_expression("_rng.uniform(0, 1)", extra_names=_HEXSIM_FUNCTIONS)
    _validate_expression("_rng.normal(0, 1)", extra_names=_HEXSIM_FUNCTIONS)
    _validate_expression("_rng.integers(0, 10)", extra_names=_HEXSIM_FUNCTIONS)


def test_ast_sandbox_rejects_rng_seed():
    """_rng.seed / _rng.bytes / arbitrary methods must be rejected (DoS surface)."""
    import pytest
    from salmon_ibm.accumulators import _validate_expression, _HEXSIM_FUNCTIONS

    for malicious in [
        "_rng.seed(42)",
        "_rng.bytes(1000000)",
        "_rng.__class__()",
        "_rng.permutation(10)",
    ]:
        with pytest.raises(ValueError, match="attribute|method"):
            _validate_expression(malicious, extra_names=_HEXSIM_FUNCTIONS)


def test_ast_sandbox_rejects_method_call_on_non_rng_attribute():
    """Method calls via attribute access on anything other than _rng must fail."""
    import pytest
    from salmon_ibm.accumulators import _validate_expression, _HEXSIM_FUNCTIONS

    with pytest.raises(ValueError):
        _validate_expression("foo.bar(1)", extra_names=_HEXSIM_FUNCTIONS)


def test_ast_sandbox_rejects_huge_rng_arg_dos():
    """_rng.random(10**9) must fail — prevents GB-scale allocation DoS from scenario XML."""
    import pytest
    from salmon_ibm.accumulators import _validate_expression, _HEXSIM_FUNCTIONS, _RNG_ARG_MAX

    with pytest.raises(ValueError, match="too large"):
        _validate_expression("_rng.random(10000000000)", extra_names=_HEXSIM_FUNCTIONS)

    # A normal-sized arg still passes.
    _validate_expression(f"_rng.random({_RNG_ARG_MAX - 1})", extra_names=_HEXSIM_FUNCTIONS)
```

- [ ] **Step 2: Run — expect FAIL (`_rng.seed` currently passes validation)**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py -k "ast_sandbox" -v
```

- [ ] **Step 3: Add an allowlist and tighten `_validate_expression`**

Replace `salmon_ibm/accumulators.py:186-198` with:

```python
        if isinstance(node, ast.Attribute):
            # Only allow attribute access on _rng, and only for whitelisted methods
            if not (isinstance(node.value, ast.Name) and node.value.id == "_rng"):
                raise ValueError(
                    f"Disallowed attribute access in expression: "
                    f"only '_rng.<method>' is permitted, got "
                    f"'{ast.dump(node.value)}.{node.attr}'"
                )
            if node.attr not in _ALLOWED_RNG_METHODS:
                raise ValueError(
                    f"Disallowed _rng method '{node.attr}'. "
                    f"Allowed: {sorted(_ALLOWED_RNG_METHODS)}"
                )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_names:
                    raise ValueError(f"Unknown function in expression: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                # Attribute-form calls are only allowed through the _rng path above.
                # The ast.Attribute branch has already validated the target + method.
                # DoS guard: reject large literal numeric args (prevents e.g. _rng.random(10**9)).
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(
                        arg.value, (int, float)
                    ):
                        if arg.value > _RNG_ARG_MAX:
                            raise ValueError(
                                f"Argument {arg.value} too large in _rng call "
                                f"(exceeds _RNG_ARG_MAX={_RNG_ARG_MAX})"
                            )
            else:
                raise ValueError(
                    f"Disallowed call target: {type(node.func).__name__}"
                )
```

Then, above `_ALLOWED_NODE_TYPES` (around line 129), add:

```python
# Methods on _rng that scenario expressions may call (numpy.random.Generator).
# Extend only if a real HexSim scenario uses it AND the method is non-mutating.
_ALLOWED_RNG_METHODS = frozenset({"random", "uniform", "normal", "integers"})

# DoS guard: reject _rng.<method>(N) when N exceeds this bound.
# Prevents scenario XML from allocating GB-scale arrays via e.g. _rng.random(10**9).
# 1M floats = 8 MB; typical legitimate usage is <= n_agents (<100k).
_RNG_ARG_MAX = 1_000_000
```

- [ ] **Step 4: Run the tests — expect PASS**

- [ ] **Step 5: Run the full DSL-using tests to catch any regression**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py tests/test_hexsim_expr.py tests/test_events_hexsim.py -v
```

If any pre-existing scenario uses e.g. `_rng.choice` or `_rng.permutation`, either widen the allowlist (if non-mutating) or migrate the scenario. Do NOT add `seed` or `bytes`.

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/accumulators.py tests/test_accumulators.py
git commit -m "fix(accumulators): tighten AST sandbox — _rng method allowlist + DoS arg guard"
```

---

## Task 5b: Replace unbounded `_compiled_expr_cache` with LRU eviction

**Problem:** `accumulators.py:261-266` and `hexsim_expr.py:47` use `dict` caches that only clear when they exceed 10,000 entries (bulk clear). A scenario submitting 9,999 distinct expressions (e.g., `1+1+1+...`) keeps the cache full indefinitely until the 10,001st insert triggers a full clear. Memory-sensitive; a MEDIUM DoS if attacker-controlled scenario XML submits many variants.

**Fix strategy:** Replace the plain `dict` + bulk-clear with `OrderedDict` + LRU eviction capped at 256 entries.

**Files:**
- Modify: `salmon_ibm/accumulators.py:261-266` (the `_compiled_expr_cache` block)
- Modify: `salmon_ibm/hexsim_expr.py:47` (the `_translate_cache` block — same pattern)
- Test: `tests/test_accumulators.py`

- [ ] **Step 1: Inspect both cache sites**

```bash
conda run -n shiny grep -n "_compiled_expr_cache\|_translate_cache" salmon_ibm/accumulators.py salmon_ibm/hexsim_expr.py
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_accumulators.py`:

```python
def test_compiled_expr_cache_bounded_to_lru_size():
    """Cache must evict oldest entries when over capacity; should not grow unbounded."""
    from salmon_ibm.accumulators import _compiled_expr_cache, _EXPR_CACHE_MAX

    _compiled_expr_cache.clear()
    # Submit MAX + 10 distinct expressions; cache must stay at MAX, not grow to MAX+10.
    for i in range(_EXPR_CACHE_MAX + 10):
        expr = f"1 + {i}"
        import ast
        _compiled_expr_cache[expr] = compile(expr, "<test>", "eval")
        # Simulate the eviction guard the fix should apply:
    assert len(_compiled_expr_cache) <= _EXPR_CACHE_MAX, (
        f"Cache grew to {len(_compiled_expr_cache)}, expected <= {_EXPR_CACHE_MAX}"
    )
```

- [ ] **Step 3: Run — expect FAIL (`_EXPR_CACHE_MAX` doesn't exist yet)**

- [ ] **Step 4: Replace the cache with an LRU-bounded `OrderedDict`**

In `salmon_ibm/accumulators.py`, at the top-level cache definition:

```python
from collections import OrderedDict

_EXPR_CACHE_MAX = 256
_compiled_expr_cache: OrderedDict[str, object] = OrderedDict()
```

Then replace the insertion block (currently at lines 261-266):

```python
if translated not in _compiled_expr_cache:
    if len(_compiled_expr_cache) >= _EXPR_CACHE_MAX:
        _compiled_expr_cache.popitem(last=False)  # evict oldest (LRU)
    _compiled_expr_cache[translated] = compile(translated, "<hexsim-expr>", "eval")
else:
    # Move to end to mark as recently used
    _compiled_expr_cache.move_to_end(translated)
```

- [ ] **Step 5: Apply the same pattern to `hexsim_expr.py:_translate_cache`**

Use an identical LRU pattern with its own `_TRANSLATE_CACHE_MAX = 256`.

- [ ] **Step 6: Run tests — expect PASS**

```bash
conda run -n shiny python -m pytest tests/test_accumulators.py tests/test_hexsim_expr.py -v
```

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/accumulators.py salmon_ibm/hexsim_expr.py tests/test_accumulators.py
git commit -m "fix(accumulators): LRU-bounded expression caches (prevents cache-flooding DoS)"
```

---

## Task 6: Replace substring-match landscape injection with an explicit allowlist

**Problem:** `events_phase3.py:104-110` injects any `landscape[key]` that is an ndarray and whose `key` appears as a substring in the expression string. Substring matching is exploitable and unsafe — a landscape key `"x"` will inject for *any* expression containing `x`.

**Files:**
- Test: `tests/test_events_phase3.py`
- Modify: `salmon_ibm/events_phase3.py:76-112`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_events_phase3.py`:

```python
def test_generated_hexmap_injects_only_allowlisted_landscape_keys():
    """GeneratedHexmapEvent must not inject arbitrary landscape ndarrays by substring match."""
    from salmon_ibm.events_phase3 import GeneratedHexmapEvent
    from unittest.mock import MagicMock

    evt = GeneratedHexmapEvent(
        name="test",
        expression="temperature + 1",
        output_name="heat",
    )
    # landscape has a 'secret' ndarray whose name is a substring of nothing sensible,
    # but also 'temperature' which the expression legitimately uses.
    landscape = {
        "n_cells": 3,
        "spatial_data": {},
        "temperature": np.array([10.0, 11.0, 12.0]),
        # 'e' is a substring of 'temperature' — old logic would inject this.
        "e": np.array([1.0, 2.0, 3.0]),
    }
    population = MagicMock()
    population.alive = np.array([True])
    population.tri_idx = np.array([0])
    mask = np.array([True])

    evt.execute(population, landscape, t=0, mask=mask)

    result = landscape["heat"]
    np.testing.assert_array_almost_equal(result, [11.0, 12.0, 13.0])


def test_generated_hexmap_uses_spatial_data_allowlist():
    """Only keys explicitly listed in spatial_data OR in allowed_landscape_keys should be injected."""
    from salmon_ibm.events_phase3 import GeneratedHexmapEvent
    from unittest.mock import MagicMock

    evt = GeneratedHexmapEvent(
        name="test",
        expression="my_layer * 2",
        output_name="doubled",
    )
    landscape = {
        "n_cells": 3,
        "spatial_data": {"my_layer": np.array([1.0, 2.0, 3.0])},
    }
    population = MagicMock()
    population.alive = np.array([True])
    population.tri_idx = np.array([0])
    mask = np.array([True])

    evt.execute(population, landscape, t=0, mask=mask)
    np.testing.assert_array_almost_equal(landscape["doubled"], [2.0, 4.0, 6.0])
```

- [ ] **Step 2: Run — expect FAIL (the substring-match case injects `e` which is the math constant, and the expression `temperature + 1` currently relies on substring-matching top-level landscape to inject temperature correctly, which is the behavior under review)**

```bash
conda run -n shiny python -m pytest tests/test_events_phase3.py::test_generated_hexmap_injects_only_allowlisted_landscape_keys tests/test_events_phase3.py::test_generated_hexmap_uses_spatial_data_allowlist -v
```

- [ ] **Step 3: Replace the injection loop with an explicit allowlist on the event itself**

Modify `salmon_ibm/events_phase3.py:76-112`. Replace the `GeneratedHexmapEvent` class with:

```python
@register_event("generated_hexmap")
@dataclass
class GeneratedHexmapEvent(Event):
    """Create or update a hex map using an algebraic expression.

    Expression namespace:
      - _SAFE_MATH (math functions)
      - t (current timestep)
      - density (if 'density' appears literally in the expression)
      - All keys from landscape['spatial_data'] that are ndarrays
      - Any key listed in `allowed_landscape_keys` that resolves to an ndarray in landscape

    Note: landscape keys are NOT injected by substring match on the expression;
    this avoids ambiguity and foot-guns (e.g., 'e' matching 'temperature').
    Use `allowed_landscape_keys` to opt a specific landscape variable into scope.
    """

    expression: str = ""
    output_name: str = ""
    allowed_landscape_keys: tuple[str, ...] = ()

    def execute(self, population, landscape, t, mask):
        from salmon_ibm.accumulators import _validate_expression, _SAFE_MATH

        _validate_expression(self.expression)
        namespace = dict(_SAFE_MATH)
        namespace["t"] = t
        if "density" in self.expression:
            n_cells = landscape.get("n_cells", 0)
            if n_cells > 0:
                alive = population.alive if hasattr(population, "alive") else mask
                positions = population.tri_idx[alive & mask]
                density = np.bincount(positions, minlength=n_cells).astype(np.float64)
                namespace["density"] = density
        # Inject spatial_data arrays (scenario-declared spatial layers).
        # Reject keys that would shadow _SAFE_MATH names (e.g. 'sqrt', 'clip')
        # — a scenario could otherwise corrupt safe-math functions silently.
        spatial_data = landscape.get("spatial_data", {})
        from salmon_ibm.accumulators import _SAFE_MATH
        for key, value in spatial_data.items():
            if key in _SAFE_MATH:
                continue  # never let a spatial layer shadow a math function
            if isinstance(value, np.ndarray) and key not in namespace:
                namespace[key] = value
        # Explicit allowlist for non-spatial_data landscape ndarrays.
        for key in self.allowed_landscape_keys:
            if key in _SAFE_MATH:
                continue
            value = landscape.get(key)
            if isinstance(value, np.ndarray) and key not in namespace:
                namespace[key] = value
        # Protect landscape's own structural keys from being overwritten by output.
        _PROTECTED = {"fields", "mesh", "spatial_data", "n_cells", "multi_pop_mgr",
                      "step_alive_mask"}
        if self.output_name in _PROTECTED:
            raise ValueError(
                f"GeneratedHexmapEvent.output_name={self.output_name!r} would "
                f"overwrite a protected landscape key."
            )
        result = eval(self.expression, {"__builtins__": {}}, namespace)
        landscape[self.output_name] = np.asarray(result, dtype=np.float64)
```

Update the first test to pass `allowed_landscape_keys=("temperature",)`:

```python
    evt = GeneratedHexmapEvent(
        name="test",
        expression="temperature + 1",
        output_name="heat",
        allowed_landscape_keys=("temperature",),
    )
```

- [ ] **Step 4: Run the tests — expect PASS**

- [ ] **Step 5: Check scenario loader for callers that need `allowed_landscape_keys`**

```bash
conda run -n shiny grep -rn "GeneratedHexmapEvent\|generated_hexmap" salmon_ibm/ tests/
```

If any YAML/XML scenario loader constructs `GeneratedHexmapEvent`, add `allowed_landscape_keys` parsing there. If none exists yet, leave as is — fail-closed is safer than fail-open.

- [ ] **Step 6: Run affected tests — expect PASS**

```bash
conda run -n shiny python -m pytest tests/test_events_phase3.py tests/test_scenario_loader.py -v
```

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/events_phase3.py tests/test_events_phase3.py
git commit -m "fix(events_phase3): replace substring-match injection with explicit allowlist"
```

---

# Phase 4 — Robustness

## Task 7: Gate `push_temperature` to alive agents

**Problem:** `push_temperature` in `agents.py:82-84` unconditionally writes `temps` into `temp_history[:, -1]` for all rows. Dead agents' stale `tri_idx` means the sampled temperature is arbitrary; after a `compact()`, stale entries survive into the new index layout and contaminate surviving agents' t3h_mean.

**Current code** (`salmon_ibm/agents.py:82-84`):
```python
def push_temperature(self, temps: np.ndarray):
    self.temp_history[:, :-1] = self.temp_history[:, 1:]
    self.temp_history[:, -1] = temps
```

**Fix strategy:** Give `push_temperature` an optional `alive_mask` that lets dead agents retain their previous last-slot value (so the ring-buffer shift never injects a stale sample). Update the call site in `simulation.py` / `events_builtin.py` to pass the mask.

**Files:**
- Modify: `salmon_ibm/agents.py:82-84`
- Modify: `salmon_ibm/simulation.py` and/or `salmon_ibm/events_builtin.py` (call site)
- Test: `tests/test_agents.py`

- [ ] **Step 1: Find the call site**

```bash
conda run -n shiny grep -n "push_temperature" salmon_ibm/
```

Record the file:line. The task assumes call sites exist in `simulation.py` and/or `events_builtin.py`.

- [ ] **Step 2: Write the failing test**

Add to `tests/test_agents.py`:

```python
def test_push_temperature_preserves_dead_agents_history():
    """With alive_mask, dead agents retain their last-slot value; ring shift drops the oldest."""
    import numpy as np
    from salmon_ibm.agents import AgentPool

    pool = AgentPool(n=3, start_tri=0, rng_seed=0)
    pool.temp_history[:] = np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    pool.alive[1] = False  # agent 1 dead

    pool.push_temperature(np.array([99.0, 99.0, 99.0]), alive_mask=pool.alive)

    # Alive agents (0 and 2): last slot is 99.0 (fresh write).
    assert pool.temp_history[0, -1] == 99.0
    assert pool.temp_history[2, -1] == 99.0
    # Dead agent (1): last slot preserved at 7.0 (ring shift was NOT applied).
    assert pool.temp_history[1, -1] == 7.0, (
        f"Dead agent's temp_history must be frozen; got {pool.temp_history[1, -1]}"
    )
```

- [ ] **Step 3: Run — expect FAIL**

```bash
conda run -n shiny python -m pytest tests/test_agents.py::test_push_temperature_preserves_dead_agents_history -v
```

- [ ] **Step 4: Add `alive_mask` parameter to `AgentPool.push_temperature`**

Replace `salmon_ibm/agents.py:82-84` with:

```python
def push_temperature(self, temps: np.ndarray, alive_mask: np.ndarray | None = None):
    if alive_mask is None:
        self.temp_history[:, :-1] = self.temp_history[:, 1:]
        self.temp_history[:, -1] = temps
        return
    # For alive agents: ring shift + new sample.
    # For dead agents: freeze history so compacted indices don't inherit stale data.
    alive_mask = np.asarray(alive_mask, dtype=bool)
    self.temp_history[alive_mask, :-1] = self.temp_history[alive_mask, 1:]
    self.temp_history[alive_mask, -1] = temps[alive_mask]
```

- [ ] **Step 5: Update call site(s) in `simulation.py` / `events_builtin.py`**

Find each `push_temperature(temps)` call and change it to:

```python
alive_mask = population.alive & ~population.arrived
population.pool.push_temperature(temps, alive_mask=alive_mask)
```

(Use `population.pool` if calling on the Population wrapper; use `pool` directly if the caller already has the pool.)

- [ ] **Step 6: Run the test + full agents/simulation tests**

```bash
conda run -n shiny python -m pytest tests/test_agents.py tests/test_simulation.py -v
```

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/agents.py salmon_ibm/simulation.py salmon_ibm/events_builtin.py tests/test_agents.py
git commit -m "fix(agents): gate push_temperature to alive agents to prevent stale-index contamination"
```

---

## Task 8: Assert `ARRAY_FIELDS` coverage in `AgentPool.__init__`

**Problem:** `agents.py:35-67` hardcodes 11 field inits. Adding a new field to `ARRAY_FIELDS` without updating `__init__` fails silently — `add_agents()` catches the drift later, but `__init__` should catch it at construction.

**Files:**
- Test: `tests/test_agents.py`
- Modify: `salmon_ibm/agents.py:35-67`

- [ ] **Step 1: Inspect current shape**

```bash
conda run -n shiny python -c "from salmon_ibm.agents import AgentPool; p = AgentPool(); print(AgentPool.ARRAY_FIELDS); print([a for a in dir(p) if not a.startswith('_') and isinstance(getattr(p,a), __import__('numpy').ndarray)])"
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_agents.py`:

```python
def test_agent_pool_init_covers_all_array_fields():
    """Every field in ARRAY_FIELDS must be set to an ndarray by __init__.

    Guards against adding a field to ARRAY_FIELDS but forgetting to initialize it.
    """
    from salmon_ibm.agents import AgentPool

    pool = AgentPool()
    missing = []
    for field in AgentPool.ARRAY_FIELDS:
        attr = getattr(pool, field, None)
        if not isinstance(attr, np.ndarray):
            missing.append(field)
    assert not missing, (
        f"AgentPool.__init__ did not initialize these ARRAY_FIELDS as ndarrays: {missing}"
    )
```

- [ ] **Step 3: Run — should PASS today, but locks in the invariant**

If it fails, there's a pre-existing drift to fix before proceeding.

- [ ] **Step 4: Add a defensive assertion at the end of `AgentPool.__init__`**

Locate the end of `AgentPool.__init__` in `salmon_ibm/agents.py` and add, just before the method returns:

```python
        # Invariant: every declared array field must be initialized as an ndarray.
        missing = [
            f for f in self.ARRAY_FIELDS if not isinstance(getattr(self, f, None), np.ndarray)
        ]
        assert not missing, (
            f"AgentPool.__init__ missed ARRAY_FIELDS: {missing}. "
            f"Add initialization or update ARRAY_FIELDS."
        )
```

- [ ] **Step 5: Run — expect PASS**

```bash
conda run -n shiny python -m pytest tests/test_agents.py -v
```

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/agents.py tests/test_agents.py
git commit -m "feat(agents): assert ARRAY_FIELDS fully initialized in AgentPool.__init__"
```

---

## Task 9: Use existing `Population.agent_ids` in `OutputLogger`

**Problem:** `output.py:28` logs `np.arange(n, dtype=np.int32)` as agent IDs. After `Population.compact()` removes dead agents, `n` shrinks and `arange(n)` relabels survivors from 0 — cross-timestep agent tracking breaks.

**Good news:** `Population` already has `agent_ids` (`population.py:28, 33`) and `_next_id` (`population.py:27, 34`). `Population.compact()` already preserves `agent_ids` (`population.py:160`), and `add_agents` already extends it (`population.py:240-244`). **No new infrastructure needed.**

**Fix strategy:** Change `OutputLogger.log_step` to accept a `Population` (not `AgentPool`) and use `population.agent_ids`. The current call site `simulation.py:251` already passes `population` — the type hint is wrong but the runtime is correct. We just align the hint and body.

**Files:**
- Modify: `salmon_ibm/output.py:8, 12, 25, 70` (signature + body)
- Test: `tests/test_output.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_output.py`:

```python
def test_logger_uses_stable_agent_ids_across_compact():
    """Agent IDs must survive Population.compact() so cross-step tracking works."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.output import OutputLogger

    pool = AgentPool(n=3, start_tri=0, rng_seed=0)
    pop = Population(name="test", pool=pool)
    centroids = np.zeros((10, 2))
    logger = OutputLogger(path="/tmp/unused.csv", centroids=centroids)

    logger.log_step(0, pop)
    ids_t0 = logger._agent_ids[-1].copy()
    assert list(ids_t0) == [0, 1, 2]

    # Kill agent 1 and use the canonical compact path.
    pop.pool.alive[1] = False
    pop.compact()

    logger.log_step(1, pop)
    ids_t1 = logger._agent_ids[-1]
    # Surviving agents keep their ORIGINAL IDs, not relabeled to [0, 1].
    assert list(ids_t1) == [0, 2], f"Expected [0, 2], got {list(ids_t1)}"
```

- [ ] **Step 2: Run — expect FAIL (`log_step` uses `np.arange` which becomes `[0, 1]` after compact)**

```bash
conda run -n shiny python -m pytest tests/test_output.py::test_logger_uses_stable_agent_ids_across_compact -v
```

- [ ] **Step 3: Change `OutputLogger.log_step` to accept `Population` and use its `agent_ids`**

In `salmon_ibm/output.py`:

1. Update the import (line 8):

```python
from salmon_ibm.agents import AgentPool
from salmon_ibm.population import Population
```

2. Change `log_step` signature and body (line 25-35):

```python
def log_step(self, t: int, population: Population):
    pool = population.pool
    n = pool.n
    self._times.append(np.full(n, t, dtype=np.int32))
    self._agent_ids.append(population.agent_ids.astype(np.int32).copy())
    self._tri_idxs.append(pool.tri_idx.copy())
    self._lats.append(self.centroids[pool.tri_idx, 0].copy())
    self._lons.append(self.centroids[pool.tri_idx, 1].copy())
    self._eds.append(pool.ed_kJ_g.copy())
    self._behaviors.append(pool.behavior.copy())
    self._alive.append(pool.alive.copy())
    self._arrived.append(pool.arrived.copy())
```

3. Update `summary` signature too (line 70) — it currently takes `pool: AgentPool`; if any caller passes `population`, align; otherwise leave for now.

- [ ] **Step 4: Update the three existing `test_output.py` callers that pass a bare `AgentPool`**

The existing tests in `tests/test_output.py` construct a bare `AgentPool` and call `logger.log_step(0, pool)` directly. After the signature change they will raise `AttributeError: 'AgentPool' object has no attribute 'agent_ids'`. Wrap each:

**`tests/test_output.py:9-15` — `test_logger_creates_file`:**
```python
def test_logger_creates_file(tmp_path):
    centroids = np.array([[55.0, 21.0], [55.1, 21.1]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=3, start_tri=0)
    pop = Population(name="test", pool=pool)   # NEW
    logger.log_step(0, pop)                     # CHANGED
    logger.close()
    assert os.path.exists(tmp_path / "tracks.csv")
```

**`tests/test_output.py:18-26` — `test_logger_records_correct_columns`:**
```python
def test_logger_records_correct_columns(tmp_path):
    centroids = np.array([[55.0, 21.0], [55.1, 21.1]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=2, start_tri=0)
    pop = Population(name="test", pool=pool)   # NEW
    logger.log_step(0, pop)                     # CHANGED
    logger.close()
    df = pd.read_csv(tmp_path / "tracks.csv")
    ...
```

**`tests/test_output.py:29-37` — `test_logger_accumulates_steps`:**
```python
def test_logger_accumulates_steps(tmp_path):
    centroids = np.array([[55.0, 21.0]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=2, start_tri=0)
    pop = Population(name="test", pool=pool)   # NEW
    logger.log_step(0, pop)                     # CHANGED
    logger.log_step(1, pop)                     # CHANGED
    logger.close()
    df = pd.read_csv(tmp_path / "tracks.csv")
    assert len(df) == 4
```

Also add the Population import at the top of `tests/test_output.py`:

```python
from salmon_ibm.population import Population
```

Finally, verify no other callers exist:

```bash
conda run -n shiny grep -rn "log_step(" salmon_ibm/ tests/
```

Expected: `simulation.py:251` (already passes `population` — correct) + the three updated tests + the new test from Step 1. No others.

- [ ] **Step 5: Run the test — expect PASS**

- [ ] **Step 6: Run the output + reporting + simulation test suites**

```bash
conda run -n shiny python -m pytest tests/test_output.py tests/test_reporting.py tests/test_simulation.py -v
```

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/output.py tests/test_output.py
git commit -m "fix(output): use Population.agent_ids so IDs survive compact()"
```

---

## Task 10: Numba↔NumPy fallback parity test

**Problem:** Both paths exist (`HAS_NUMBA` flag) but no test asserts they produce identical results. Silent numeric drift is possible if Numba compilation breaks.

**Files:**
- Test: `tests/test_numba_fallback.py`

- [ ] **Step 1: Inspect the fallback toggle**

```bash
conda run -n shiny grep -n "HAS_NUMBA\|FORCE_NUMPY" salmon_ibm/*.py
```

- [ ] **Step 2: Write a parity test for the movement kernel**

Add to `tests/test_numba_fallback.py`:

```python
def test_move_gradient_numba_matches_numpy_fallback():
    """Numba JIT and pure-NumPy fallback must produce identical results."""
    from salmon_ibm import events_hexsim

    rng = np.random.default_rng(42)
    n_cells = 100
    n_agents = 30
    water_nbrs = rng.integers(0, n_cells, size=(n_cells, 6), dtype=np.int64)
    water_nbr_count = np.full(n_cells, 6, dtype=np.int64)
    gradient = rng.random(n_cells)

    positions_a = rng.integers(0, n_cells, size=n_agents, dtype=np.int64).copy()
    positions_b = positions_a.copy()

    # Numba path
    pa, da = events_hexsim._move_gradient_numba(
        positions_a, water_nbrs, water_nbr_count, gradient, 5, True, 0.0
    )
    # Numpy path: force fallback by calling the underlying Python function.
    # If the codebase exposes _move_gradient_numpy, use it; else dispatch via a flag.
    if hasattr(events_hexsim, "_move_gradient_numpy"):
        pb, db = events_hexsim._move_gradient_numpy(
            positions_b, water_nbrs, water_nbr_count, gradient, 5, True, 0.0
        )
    else:
        import pytest
        pytest.skip("No explicit NumPy fallback exposed; parity test skipped")

    np.testing.assert_array_equal(pa, pb, err_msg="Positions diverge between Numba and NumPy")
    np.testing.assert_allclose(da, db, rtol=1e-12, err_msg="Distances diverge")
```

- [ ] **Step 3: Run — expect PASS or SKIP**

```bash
conda run -n shiny python -m pytest tests/test_numba_fallback.py -v
```

If the test SKIPs because no explicit fallback exists, that's valuable — document the finding in the plan follow-up instead of expanding the scope here.

- [ ] **Step 4: Commit**

```bash
git add tests/test_numba_fallback.py
git commit -m "test(numba): add Numba↔NumPy parity test for move_gradient"
```

---

# Phase 5 — Performance Wins

## Task 11: Cache contiguous centroids on the mesh

**Problem:** `movement.py:~341-365` — `_apply_current_advection_vec` calls `np.ascontiguousarray(mesh.centroids)` every step. `mesh.centroids` is static.

**Files:**
- Modify: `salmon_ibm/mesh.py`, `salmon_ibm/movement.py`

- [ ] **Step 1: Locate the callsite**

```bash
conda run -n shiny grep -n "ascontiguousarray\|_apply_current_advection" salmon_ibm/movement.py
```

- [ ] **Step 2: Cache on the mesh**

Add to `salmon_ibm/mesh.py` (at the end of the HexMesh/TriMesh `__init__` or as a `@property` with `functools.cached_property`):

```python
from functools import cached_property

# ... inside the mesh class:
@cached_property
def centroids_c(self) -> np.ndarray:
    """Contiguous view of centroids for Numba kernels. Computed once."""
    return np.ascontiguousarray(self.centroids)
```

- [ ] **Step 3: Use the cached property in `movement.py`**

Replace the per-step `np.ascontiguousarray(mesh.centroids)` call with `mesh.centroids_c`.

- [ ] **Step 4: Run tests — expect PASS**

```bash
conda run -n shiny python -m pytest tests/test_movement.py tests/test_mesh.py -v
```

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/mesh.py salmon_ibm/movement.py
git commit -m "perf(mesh): cache contiguous centroids instead of per-step conversion"
```

---

## Task 12: Transpose `AccumulatorManager.data` to `(n_acc, n_agents)` — DEFERRED

**Problem:** Row-major `(n_agents, n_acc)` layout means `data[:, idx]` is a strided gather across rows. Transposing gives column-wise contiguous access, expected 1.5-3× on accumulator-heavy scenarios.

**Why deferred:** The change touches ~30 updater call sites (each uses `data[mask, col_idx]`). A full transpose requires flipping every indexing, which is surface-area-large and risks subtle regressions. This should be a dedicated plan with:
1. Benchmark baseline captured
2. Full audit of every `data[...]` access site
3. Transposed layout implemented behind a feature flag
4. Benchmark re-measured
5. Flag flipped, old code removed

- [ ] **Action:** Open a follow-up plan: `docs/superpowers/plans/2026-04-24-accumulator-layout-transpose.md` using the `superpowers:writing-plans` skill. Do NOT attempt inline.

---

## Task 13: Vectorize `InteractionEvent.execute` — DEFERRED

**Problem:** `interactions.py:125-145` nested Python loops over agents — violates project convention.

**Why deferred:** Rewriting to `rng.random((|A|, |B|)) < p` + `np.nonzero` semantics requires reconciling deterministic ordering (currently implicit via loop order) with vectorized semantics. This touches genetics/disease/predation behavior — needs biological review, not just a mechanical rewrite.

- [ ] **Action:** Open a follow-up plan: `docs/superpowers/plans/2026-04-24-vectorize-interactions.md`. Deterministic ordering must be defined (e.g., `np.argsort` by `(b_idx, random_tiebreaker)`) before coding.

---

## Task 14: Precompute per-step `temps_at_agents`, `sal_at_agents`, `activity` in `landscape`

**Problem:** `events_builtin.py` (bioenergetics fallback) and `simulation.py` `SurvivalEvent` both recompute `fields["temperature"][population.tri_idx]` and friends. Duplicated gathers are O(N) per step, wasted.

**Files:**
- Modify: `salmon_ibm/simulation.py` (the per-step setup), `salmon_ibm/events_builtin.py`

- [ ] **Step 1: Identify the current callers**

```bash
conda run -n shiny grep -n 'temperature"\]\[\|fields\["temperature"\]\|sal\[' salmon_ibm/events_builtin.py salmon_ibm/simulation.py
```

- [ ] **Step 2: Add a per-step precompute to the sequencer-entry point**

Find the method that builds `landscape` each step (likely `Simulation.step` or `EventSequencer.step` setup). Insert:

```python
        # Per-step gathers used by multiple events — compute once.
        tri = population.tri_idx
        fields = landscape.get("fields", {})
        if "temperature" in fields:
            landscape["temps_at_agents"] = fields["temperature"][tri]
        if "salinity" in fields:
            landscape["sal_at_agents"] = fields["salinity"][tri]
```

- [ ] **Step 3: Update bioenergetics and survival events to use the precomputed values**

Find the two call sites and change:

```python
temps = landscape["fields"]["temperature"][population.tri_idx]
```

to:

```python
temps = landscape.get("temps_at_agents")
if temps is None:
    temps = landscape["fields"]["temperature"][population.tri_idx]
```

(The fallback keeps existing tests green even if a scenario bypasses the new precompute.)

- [ ] **Step 4: Run tests — expect PASS**

```bash
conda run -n shiny python -m pytest tests/test_events.py tests/test_simulation.py tests/test_bioenergetics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/simulation.py salmon_ibm/events_builtin.py
git commit -m "perf(events): precompute per-step temps/sal gathers, share via landscape"
```

---

## Task 15: Preallocated `OutputLogger` buffers (optional opt-in)

**Problem:** `output.py:25-35` makes 8× `.copy()` calls per step into Python lists, then `np.concatenate` at close. For 50k agents × 2928 steps this allocates ~1-8 GB of temporaries. Preallocation is a clear win when the caller knows `max_steps` and `max_agents` upfront.

**Fix strategy:** Opt-in preallocation — if the caller passes `max_steps` and `max_agents` to `__init__`, allocate 2-D arrays once and write row `t`. If not passed, keep existing list-append path.

**Depends on:** Task 9 (log_step accepts Population). Do Task 9 first.

**Current fields** (from `output.py`): `_times, _agent_ids, _tri_idxs, _lats, _lons, _eds, _behaviors, _alive, _arrived`.

**Files:**
- Modify: `salmon_ibm/output.py`
- Test: `tests/test_output.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_output.py`:

```python
def test_logger_preallocated_path_matches_list_path():
    """Preallocated and list-append paths must produce identical DataFrames."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.output import OutputLogger

    pool = AgentPool(n=5, start_tri=0, rng_seed=0)
    pop = Population(name="test", pool=pool)
    centroids = np.random.RandomState(0).rand(10, 2)

    list_logger = OutputLogger(path="/tmp/a.csv", centroids=centroids)
    prealloc_logger = OutputLogger(
        path="/tmp/b.csv", centroids=centroids, max_steps=3, max_agents=5
    )
    for t in range(3):
        list_logger.log_step(t, pop)
        prealloc_logger.log_step(t, pop)

    df_list = list_logger.to_dataframe().sort_values(["time", "agent_id"]).reset_index(drop=True)
    df_pre = prealloc_logger.to_dataframe().sort_values(["time", "agent_id"]).reset_index(drop=True)
    assert len(df_list) == len(df_pre) == 15
    for col in ["time", "agent_id", "tri_idx", "behavior", "alive", "arrived"]:
        np.testing.assert_array_equal(df_list[col].values, df_pre[col].values, err_msg=col)
    for col in ["lat", "lon", "ed_kJ_g"]:
        np.testing.assert_allclose(df_list[col].values, df_pre[col].values, rtol=1e-12, err_msg=col)
```

- [ ] **Step 2: Run — expect FAIL (`max_steps`/`max_agents` not accepted yet)**

```bash
conda run -n shiny python -m pytest tests/test_output.py::test_logger_preallocated_path_matches_list_path -v
```

- [ ] **Step 3: Extend `OutputLogger.__init__` and `log_step`**

Replace `salmon_ibm/output.py` `__init__` and `log_step` with:

```python
def __init__(
    self,
    path: str,
    centroids: np.ndarray,
    max_steps: int | None = None,
    max_agents: int | None = None,
):
    self.path = path
    self.centroids = centroids
    self._preallocated = max_steps is not None and max_agents is not None
    if self._preallocated:
        self._max_steps = max_steps
        self._max_agents = max_agents
        self._n_rows = 0
        self._step_counts = np.zeros(max_steps, dtype=np.int32)
        self._times_arr = np.empty((max_steps, max_agents), dtype=np.int32)
        self._agent_ids_arr = np.empty((max_steps, max_agents), dtype=np.int32)
        self._tri_idxs_arr = np.empty((max_steps, max_agents), dtype=np.int64)
        self._lats_arr = np.empty((max_steps, max_agents), dtype=np.float64)
        self._lons_arr = np.empty((max_steps, max_agents), dtype=np.float64)
        self._eds_arr = np.empty((max_steps, max_agents), dtype=np.float64)
        self._behaviors_arr = np.empty((max_steps, max_agents), dtype=np.int32)
        self._alive_arr = np.empty((max_steps, max_agents), dtype=bool)
        self._arrived_arr = np.empty((max_steps, max_agents), dtype=bool)
    else:
        self._times: list[np.ndarray] = []
        self._agent_ids: list[np.ndarray] = []
        self._tri_idxs: list[np.ndarray] = []
        self._lats: list[np.ndarray] = []
        self._lons: list[np.ndarray] = []
        self._eds: list[np.ndarray] = []
        self._behaviors: list[np.ndarray] = []
        self._alive: list[np.ndarray] = []
        self._arrived: list[np.ndarray] = []

def log_step(self, t: int, population: Population):
    pool = population.pool
    n = pool.n
    if self._preallocated:
        if self._n_rows >= self._max_steps:
            raise ValueError(
                f"OutputLogger: exceeded max_steps={self._max_steps}"
            )
        if n > self._max_agents:
            raise ValueError(
                f"OutputLogger: agent count {n} > max_agents={self._max_agents}"
            )
        r = self._n_rows
        self._step_counts[r] = n
        self._times_arr[r, :n] = t
        self._agent_ids_arr[r, :n] = population.agent_ids[:n].astype(np.int32)
        self._tri_idxs_arr[r, :n] = pool.tri_idx[:n]
        self._lats_arr[r, :n] = self.centroids[pool.tri_idx[:n], 0]
        self._lons_arr[r, :n] = self.centroids[pool.tri_idx[:n], 1]
        self._eds_arr[r, :n] = pool.ed_kJ_g[:n]
        self._behaviors_arr[r, :n] = pool.behavior[:n]
        self._alive_arr[r, :n] = pool.alive[:n]
        self._arrived_arr[r, :n] = pool.arrived[:n]
        self._n_rows += 1
    else:
        self._times.append(np.full(n, t, dtype=np.int32))
        self._agent_ids.append(population.agent_ids.astype(np.int32).copy())
        self._tri_idxs.append(pool.tri_idx.copy())
        self._lats.append(self.centroids[pool.tri_idx, 0].copy())
        self._lons.append(self.centroids[pool.tri_idx, 1].copy())
        self._eds.append(pool.ed_kJ_g.copy())
        self._behaviors.append(pool.behavior.copy())
        self._alive.append(pool.alive.copy())
        self._arrived.append(pool.arrived.copy())
```

And update `to_dataframe()` to branch on `self._preallocated`:

```python
def to_dataframe(self) -> pd.DataFrame:
    if self._preallocated:
        if self._n_rows == 0:
            return pd.DataFrame(columns=[
                "time", "agent_id", "tri_idx", "lat", "lon",
                "ed_kJ_g", "behavior", "alive", "arrived",
            ])
        parts = []
        for r in range(self._n_rows):
            n = int(self._step_counts[r])
            parts.append(pd.DataFrame({
                "time": self._times_arr[r, :n],
                "agent_id": self._agent_ids_arr[r, :n],
                "tri_idx": self._tri_idxs_arr[r, :n],
                "lat": self._lats_arr[r, :n],
                "lon": self._lons_arr[r, :n],
                "ed_kJ_g": self._eds_arr[r, :n],
                "behavior": self._behaviors_arr[r, :n],
                "alive": self._alive_arr[r, :n],
                "arrived": self._arrived_arr[r, :n],
            }))
        return pd.concat(parts, ignore_index=True)
    # list-append path (unchanged)
    if not self._times:
        return pd.DataFrame(columns=[
            "time", "agent_id", "tri_idx", "lat", "lon",
            "ed_kJ_g", "behavior", "alive", "arrived",
        ])
    return pd.DataFrame({
        "time": np.concatenate(self._times),
        "agent_id": np.concatenate(self._agent_ids),
        "tri_idx": np.concatenate(self._tri_idxs),
        "lat": np.concatenate(self._lats),
        "lon": np.concatenate(self._lons),
        "ed_kJ_g": np.concatenate(self._eds),
        "behavior": np.concatenate(self._behaviors),
        "alive": np.concatenate(self._alive),
        "arrived": np.concatenate(self._arrived),
    })
```

- [ ] **Step 4: Run the test — expect PASS**

```bash
conda run -n shiny python -m pytest tests/test_output.py -v
```

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/output.py tests/test_output.py
git commit -m "perf(output): optional preallocated OutputLogger buffers"
```

---

# Phase 6 — Scientific Plausibility Updates

## Task 16: Update DO thresholds with Baltic-specific citation + add unit check

**Problem:** `estuary.py:34-49` DO thresholds `lethal=2, high=4` mg/L are below published *Salmo salar*-specific avoidance thresholds (Liland et al. 2024: reduced performance below ~5.5 mg/L at 15°C). Roegner et al. 2011 is about Columbia River *Oncorhynchus* and should not be the primary citation for Baltic salmon work.

**Additional concern:** DO fields loaded from NetCDF (CMEMS Baltic, HBM) are commonly in **mmol/m³** (values ~200-350), not mg/L. If raw mmol/m³ enters with `lethal=3`, every cell registers DO_OK forever — a silent failure. Conversion: `mg/L = mmol/m³ × 32 / 1000 × 1.0` (density-adjusted).

**Files:**
- Modify: `salmon_ibm/estuary.py`
- Modify: `salmon_ibm/simulation.py:~348` (DO field load — add unit check)
- Test: `tests/test_estuary.py`

- [ ] **Step 1: Inspect current code**

```bash
conda run -n shiny grep -n "lethal\|DO_\|do_\|dissolved_oxygen" salmon_ibm/estuary.py salmon_ibm/simulation.py
```

- [ ] **Step 2: Promote thresholds to an `EstuaryParams` dataclass**

In `salmon_ibm/estuary.py`, replace the module-level DO constants with:

```python
@dataclass
class EstuaryParams:
    """Parameters for estuarine stressors.

    DO defaults cite Liland et al. (2024) for *Salmo salar* performance:
    50% saturation (~4.5 mg/L at 15°C) reduces growth; 60% (~5.5 mg/L)
    is the sub-optimal threshold. Acute mortality approaches ~3 mg/L
    (see Davis 1975 criteria).
    """
    do_lethal: float = 3.0       # mg/L — below this, acute mortality
    do_high: float = 5.5         # mg/L — below this, sub-optimal (avoidance)
    s_opt: float = 0.5           # PSU — osmotic optimum
    s_tol: float = 6.0           # PSU — tolerance bound
    seiche_threshold_m_per_s: float = 0.02  # |dSSH/dt|, explicit m/s
```

- [ ] **Step 3: Add a unit sanity check at DO-field load time**

In `simulation.py` (or wherever `fields["do"]` is populated), add:

```python
do_field = self.env.fields.get("do")
if do_field is not None:
    # Sanity: mg/L values should be in [0, 20] range.
    # mmol/m³ values are ~150-400 — catch that.
    finite = do_field[np.isfinite(do_field)]
    if finite.size and finite.max() > 30.0:
        raise ValueError(
            f"DO field max={finite.max():.1f} suggests mmol/m³ input. "
            f"Expected mg/L (typical range 0-20). "
            f"Convert via: mg/L = mmol/m³ * 32 / 1000."
        )
```

- [ ] **Step 4: Write tests for the unit check and the new defaults**

Add to `tests/test_estuary.py`:

```python
def test_do_field_rejects_mmol_m3_units():
    """Loading a DO field in mmol/m³ units must fail loudly, not silently."""
    import pytest
    # Simulate the check directly; adapt import path to where it lives.
    do_field = np.array([250.0, 300.0, 280.0])  # mmol/m³
    with pytest.raises(ValueError, match="mmol/m"):
        finite = do_field[np.isfinite(do_field)]
        if finite.size and finite.max() > 30.0:
            raise ValueError(
                f"DO field max={finite.max():.1f} suggests mmol/m³ input..."
            )


def test_estuary_params_defaults_match_liland_2024():
    from salmon_ibm.estuary import EstuaryParams
    p = EstuaryParams()
    assert p.do_lethal == 3.0
    assert p.do_high == 5.5
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
conda run -n shiny python -m pytest tests/test_estuary.py tests/test_simulation.py -v
```

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/estuary.py salmon_ibm/simulation.py tests/test_estuary.py
git commit -m "fix(estuary): Baltic salmon-specific DO thresholds (Liland 2024) + unit check"
```

**Reference citations for the commit message body:**
- Liland, N. S., Rønnestad, I., & de Azevedo, M. L. V. (2024). *Data in Brief*, 57, 110983. https://doi.org/10.1016/j.dib.2024.110983
- Davis, J. C. (1975). Minimal dissolved oxygen requirements of aquatic life. *J. Fish. Res. Board Can.*, 32(12), 2295-2332.

---

## Task 17: Activity-by-temperature for Wisconsin bioenergetics — DEFERRED (modeler input required)

**Problem:** `bioenergetics.py:22-30` uses fixed per-behavior activity multipliers (1.0-1.5). Snyder et al. 2019 uses temperature-dependent `activity = exp(RTO·ACT·W^RK4·exp(BACT·T))`. Current code under-estimates respiration ~3-5× at Baltic summer temps.

**Why deferred:** The choice between "keep fixed multipliers + recalibrate RA" vs "adopt Snyder activity-by-temperature" is a modeling decision that needs the modeler's call. The `tests/test_snyder_reference.py:200-219` code already knows R at 25°C is ~13% of Snyder's — this is a deliberate simplification that should be an explicit decision.

- [ ] **Action:** Open an issue / decision document asking the modeler: do we switch to Snyder activity-by-temperature, or calibrate RA upward for the fixed-multiplier model? Tag `@razinkele`.

---

# Phase 7 — Test Suite Polish

## Task 18: Replace Playwright hard-coded timeouts with explicit waits

**Problem:** `test_playwright.py:191, 476` — `page.wait_for_timeout(1500)` / `(2000)` are brittle.

**Files:**
- Modify: `tests/test_playwright.py`

- [ ] **Step 1: Inspect the call sites**

```bash
conda run -n shiny grep -n "wait_for_timeout" tests/test_playwright.py tests/test_map_visualization.py
```

- [ ] **Step 2: Replace each with `expect().to_be_visible(timeout=...)` or a similar condition**

For each occurrence, find the element the test is actually waiting for and use:

```python
from playwright.sync_api import expect
expect(page.locator("#some-map-canvas")).to_be_visible(timeout=5000)
```

instead of `page.wait_for_timeout(1500)`.

- [ ] **Step 3: Run the UI tests manually (Playwright tests skip in headless CI)**

```bash
conda run -n shiny python -m pytest tests/test_playwright.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_playwright.py tests/test_map_visualization.py
git commit -m "test(playwright): replace wait_for_timeout with explicit expect() waits"
```

---

## Task 19: Resolve stale `xfail` on `test_map_visualization.py:424`

**Problem:** `@pytest.mark.xfail(reason="...selectize binding...")` has no ticket, no scheduled re-enable.

**Files:**
- Modify: `tests/test_map_visualization.py:424`

- [ ] **Step 1: Try re-enabling by removing the xfail**

Delete the `@pytest.mark.xfail(...)` decorator and run the test.

- [ ] **Step 2: If it passes, commit the removal**

```bash
git add tests/test_map_visualization.py
git commit -m "test(ui): re-enable selectize binding test — Playwright upgrade fixed it"
```

- [ ] **Step 3: If it still fails, replace with `@pytest.mark.skip` and a ticket reference**

Open an issue in the repo first, then:

```python
@pytest.mark.skip(reason="Issue #NN: Playwright + Shiny selectize interaction broken")
```

Commit:

```bash
git add tests/test_map_visualization.py
git commit -m "test(ui): convert stale xfail to skip with issue reference"
```

---

## Task 20: Remove or clean up tautology assertions in `test_snyder_reference.py`

**Problem:** `test_snyder_reference.py:230, 238` — `assert survival == pytest.approx(0.5)` where `0.5` is a hard-coded constant that was just computed into `survival` from the same constants. The assertions add no signal.

**Files:**
- Modify: `tests/test_snyder_reference.py`

- [ ] **Step 1: Either cite the Snyder paper value in a comment**

```python
# Source: Snyder et al. (2019) Table S2, row for T=20°C, W=500g: survival = 0.5
assert survival == pytest.approx(0.5)
```

- [ ] **Step 2: OR delete the assertion if the value was never from an external source**

- [ ] **Step 3: Commit**

```bash
git add tests/test_snyder_reference.py
git commit -m "test(snyder): document source of reference values or remove tautology"
```

---

# Phase 8 — Deferred / Out-of-Scope (separate plans)

| Finding | Why deferred | Plan file to create |
|---|---|---|
| **Task 12:** Transpose `AccumulatorManager.data` | Surface-area across ~30 updaters; requires benchmark before+after | `docs/superpowers/plans/2026-04-24-accumulator-layout-transpose.md` |
| **Task 13:** Vectorize `InteractionEvent` | Needs deterministic-ordering design decision | `docs/superpowers/plans/2026-04-24-vectorize-interactions.md` |
| **Task 17:** Activity-by-temperature bioenergetics | Modeler's call; not a code bug | Decision document (not a plan) |
| **Vectorize group/range updaters** (review PERF #21) | `accumulators.py:363-419` Python `for i in np.where(mask)[0]`. Replace with `np.bincount` / `np.add.reduceat`. Touches several updaters; batch as one plan. | `docs/superpowers/plans/2026-04-24-vectorize-range-updaters.md` |
| **`scenario_loader.py` n=0 placeholder hazard** (review MED #14) | Shape-alignment risk between accumulator manager and placeholder pool. Fix: defer accumulator/trait manager construction until after `add_agents`. | Batch with broader scenario-loader rewrite. |
| **`hexsim_viewer.py` split** | 1414-line UI module; refactor is a project on its own | `docs/superpowers/plans/2026-04-24-viewer-split.md` |
| **`scenario_loader.py` implicit event-module imports** | Low-priority code organization | Batch with viewer refactor |
| **Extract `events_base.py` to break import cycle** | Cosmetic; existing cycle is managed | Only if a future change breaks it |
| **`hexsimlab/` prototype guard consistency** | Move `NotImplementedError` to `hexsimlab/__init__.py` with unified warning. | Trivial; fold into any hexsimlab-adjacent change. |
| **`Landscape` TypedDict runtime typo guard** | TypedDict is compile-time only; runtime typos silently accepted. Replace with dataclass or custom `__setitem__`. | Low priority; do when extending landscape schema next. |

---

# Verification at end of plan

After Tasks 1-11, 14-16, 18-20 are merged:

- [ ] **Run the full test suite**

```bash
conda run -n shiny python -m pytest tests/ -v
```

Expected: all 557+N tests pass (N = new tests added here). ~4 min runtime.

- [ ] **Run the Columbia parity benchmark smoke test**

```bash
conda run -n shiny python -m pytest tests/test_parity_test.py -v
```

If the Columbia scenario XML is present, this verifies the HexSim parity hasn't regressed.

- [ ] **Spot-check benchmark numbers**

The README claims "~1.20s/step, 380 MB peak RAM." If you changed `OutputLogger` or added precomputed landscape fields, re-run a small benchmark and confirm numbers have not regressed:

```bash
conda run -n shiny python run.py --config config_columbia.yaml --agents 500 --steps 50
```

Record wall-clock and RSS. If regression is >10%, investigate before merging.

- [ ] **Open PR**

Title: `fix: address deep codebase review (CRITICAL science + security + robustness)`

Body: link each task to the corresponding issue/finding in the review synthesis.
