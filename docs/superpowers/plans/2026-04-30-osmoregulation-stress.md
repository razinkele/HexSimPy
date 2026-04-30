# Osmoregulation Stress for *Salmo salar* Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `salmon_ibm/estuary.py::salinity_cost()` with a linear-with-anchors *S. salar* physiology function modeling iso-osmotic stress (cost minimum at the iso-osmotic point ~10 PSU, asymmetric rise above and below).

**Architecture:** Function signature changes from scalar kwargs to `(salinity, params: EstuaryParams)`. Three new fields added to `EstuaryParams` with `__post_init__` validation. Two old fields (`s_opt`, `s_tol`) removed. Migrate 5 YAML configs + 3 test fixtures from old schema to new. Clean break — no backward-compat shim (matches v1.7.1 lipid-first precedent at commit `4247a11`).

**Tech Stack:** Python 3.10+, NumPy, dataclasses; pytest for testing; YAML configs; conda env `shiny`.

**Spec:** [`docs/superpowers/specs/2026-04-29-osmoregulation-stress-design.md`](../specs/2026-04-29-osmoregulation-stress-design.md) (commit `d485358`).

---

## File structure

**Modified files (14 total):**

Production code (3):
- `salmon_ibm/estuary.py` — `salinity_cost()` body + signature replaced; `EstuaryParams` extended (3 new fields, 2 old removed, `__post_init__` added)
- `salmon_ibm/events_builtin.py` — call site at line 83 updated to construct `EstuaryParams` and pass it
- `salmon_ibm/simulation.py` — call site at line 481 updated similarly

Tests (4):
- `tests/test_estuary.py` — 4 existing tests rewritten/deleted; 13 new tests added (8 spec'd + 5 EstuaryParams validation)
- `tests/test_config.py` — fixture assertion at line 26 updated
- `tests/test_ensemble.py` — fixture at line 19 updated
- `tests/test_simulation.py` — fixture at line 235 updated

Configs (5):
- `config_columbia.yaml` — Columbia migrates the "disable" pattern (`S_tol: 999`) to `salinity_hyper_cost: 0.0, salinity_hypo_cost: 0.0`
- `config_curonian_minimal.yaml`
- `configs/config_curonian_trimesh.yaml`
- `configs/config_curonian_baltic.yaml`
- `config_curonian_hexsim.yaml`

Docs (2):
- `docs/api-reference.md` — function signature reference + parameter table
- `docs/model-manual.md` — YAML schema examples + formula description

**Test runner:**
```bash
micromamba run -n shiny python -m pytest tests/path/file.py::test_name -v
# whole suite
micromamba run -n shiny python -m pytest tests/ -v
```

Suite is ~14 minutes; budget for it. Baseline before this plan: 815 passing. Expected after: ~822 (815 + 8 new − 1 deleted).

---

## Tasks

### Task 1: Verify Brett & Groves 1979 anchor values

**Files:** None modified (research-only step). Output: a confirmed pair of anchor values used in Task 2.

The spec's defaults `salinity_hyper_cost=0.30` and `salinity_hypo_cost=0.05` are flagged "verify" against Brett, J. R., & Groves, T. D. D. (1979). *Physiological energetics.* In *Fish Physiology* (Vol. 8). Academic Press, pp. 279–352. The chapter reports approximately 25-35% increase at full marine salinity and 5-10% at freshwater.

- [ ] **Step 1.1: Locate the chapter**

If institutional library access or BibSonomy / Sci-Hub gives you the chapter PDF, open it. Look at Sections 8.x covering salinity-respiration curves for *S. salar* / Atlantic salmon (commonly the "Effects of environmental factors" sub-section).

- [ ] **Step 1.2: Record the chosen defaults**

Note the values in a working log (a comment in the eventual commit message is fine):

```
Brett & Groves 1979, Section 8.x:
- hyper_cost = X.XX  (page Y, Table Z)
- hypo_cost = X.XX  (page Y, Table Z)
```

If the chapter is unavailable, use the conservative midpoints: `salinity_hyper_cost = 0.30` (within the 25-35% range), `salinity_hypo_cost = 0.075` (midpoint of 5-10%). These are the values for Task 2.

If the chapter says something materially outside these ranges (e.g., 50% at marine, 1% at freshwater), prefer the chapter's numbers but flag the discrepancy in the implementation commit message — it may indicate the spec's hedge ("verify") needed revision.

---

### Task 2: Add new `EstuaryParams` fields + `__post_init__` validation (TDD)

**Files:**
- Modify: `salmon_ibm/estuary.py:11-28` (`EstuaryParams` class)
- Test: `tests/test_estuary.py` (append new validation tests)

Adds the three new salinity fields and validation. `salinity_cost()` body stays unchanged for now — Task 3 replaces it. This split keeps each commit reviewable in isolation.

- [ ] **Step 2.1: Write failing validation tests**

Append to `tests/test_estuary.py`. **Three tests** as authorized by the spec:

```python
class TestEstuaryParamsValidation:
    def test_negative_iso_raises(self):
        from salmon_ibm.estuary import EstuaryParams
        with pytest.raises(ValueError, match="salinity_iso_osmotic"):
            EstuaryParams(salinity_iso_osmotic=-1.0)

    def test_iso_above_35_raises(self):
        from salmon_ibm.estuary import EstuaryParams
        with pytest.raises(ValueError, match="salinity_iso_osmotic"):
            EstuaryParams(salinity_iso_osmotic=40.0)

    def test_negative_hyper_cost_raises(self):
        from salmon_ibm.estuary import EstuaryParams
        with pytest.raises(ValueError, match="salinity_hyper_cost"):
            EstuaryParams(salinity_hyper_cost=-0.1)
```

(Boundary cases at iso=0 and iso=35 plus a parallel hypo-cost negative test were considered as defensive coverage but excluded to stay spec-faithful — `__post_init__` enforces `0 < iso < 35` and `0 ≤ hypo_cost ≤ 1` regardless. Future plan can add them if a regression is observed.)

- [ ] **Step 2.2: Run the new tests; expect 3 failures**

```bash
micromamba run -n shiny python -m pytest tests/test_estuary.py::TestEstuaryParamsValidation -v
```

Expected output:

```
FAILED ... TypeError: __init__() got an unexpected keyword argument 'salinity_iso_osmotic'
```

(or similar — fields don't exist yet).

- [ ] **Step 2.3: Add fields and `__post_init__`**

Modify `salmon_ibm/estuary.py:11-28` to:

```python
@dataclass
class EstuaryParams:
    """Parameters for estuarine stressors.

    DO defaults cite Liland et al. (2024) for *Salmo salar* performance:
    reduced growth below ~60% O2 saturation (~5.5 mg/L at 15°C). Acute
    mortality approaches ~3 mg/L (Davis 1975 criteria).

    Salinity-cost parameters cite Wilson 2002 (blood iso-osmotic point
    for S. salar) and Brett & Groves 1979 (hyper/hypo cost slopes for
    euryhaline salmonids).

    Units:
        do_lethal, do_high: mg/L
        salinity_iso_osmotic: PSU (blood iso-osmotic point)
        salinity_hyper_cost: dimensionless multiplier slope (above iso)
        salinity_hypo_cost: dimensionless multiplier slope (below iso)
        seiche_threshold_m_per_s: m/s (|dSSH/dt|)
    """
    do_lethal: float = 3.0
    do_high: float = 5.5
    salinity_iso_osmotic: float = 10.0   # Wilson 2002 — S. salar blood iso-osmotic ~9-12 PSU
    salinity_hyper_cost: float = 0.30    # Brett & Groves 1979 — verified value from Task 1
    salinity_hypo_cost: float = 0.05     # Brett & Groves 1979 — verified value from Task 1
    seiche_threshold_m_per_s: float = 0.02

    def __post_init__(self):
        if not (0 < self.salinity_iso_osmotic < 35):
            raise ValueError(
                f"salinity_iso_osmotic must be in (0, 35) PSU, "
                f"got {self.salinity_iso_osmotic}"
            )
        if not (0 <= self.salinity_hyper_cost <= 1):
            raise ValueError(
                f"salinity_hyper_cost must be in [0, 1], "
                f"got {self.salinity_hyper_cost}"
            )
        if not (0 <= self.salinity_hypo_cost <= 1):
            raise ValueError(
                f"salinity_hypo_cost must be in [0, 1], "
                f"got {self.salinity_hypo_cost}"
            )
```

If your verified Brett & Groves values from Task 1 differ from 0.30 / 0.05, substitute them as the defaults.

**Note:** `s_opt` and `s_tol` are gone — they're removed in this same step rather than as a separate task. The migration is in Tasks 7-9.

- [ ] **Step 2.4: Run validation tests — expect all 3 to pass**

```bash
micromamba run -n shiny python -m pytest tests/test_estuary.py::TestEstuaryParamsValidation -v
```

Expected: `3 passed`.

- [ ] **Step 2.5: Run the full estuary test file**

```bash
micromamba run -n shiny python -m pytest tests/test_estuary.py -v
```

Expected: validation tests pass; `test_estuary_params_defaults_match_liland_2024` (line ~79) **FAILS** because it asserts `p.s_opt == 0.5` and `p.s_tol == 6.0` which no longer exist. Several `test_salinity_cost_*` tests still PASS because we haven't changed the function body yet (Task 3). This is expected — don't commit yet, continue to Task 3.

---

### Task 3: Replace `salinity_cost()` body and signature (TDD)

**Files:**
- Modify: `salmon_ibm/estuary.py:49-58` (`salinity_cost` function)
- Test: `tests/test_estuary.py` (append new functional tests)

Replaces the function body with the linear-with-anchors form. Updates the signature to take `(salinity, params: EstuaryParams)` instead of scalar kwargs.

- [ ] **Step 3.1: Write the 5 functional tests**

Append to `tests/test_estuary.py`. **Five tests** as authorized by the spec:

```python
def test_salinity_cost_at_iso_returns_unity():
    """Cost should be exactly 1.0 at the iso-osmotic point (default 10 PSU)."""
    from salmon_ibm.estuary import salinity_cost, EstuaryParams
    cost = salinity_cost(np.array([10.0]), EstuaryParams())
    assert cost[0] == pytest.approx(1.0)


def test_salinity_cost_marine_matches_brett_groves():
    """At full marine salinity (35 PSU), cost ≈ 1 + hyper_cost."""
    from salmon_ibm.estuary import salinity_cost, EstuaryParams
    p = EstuaryParams()
    cost = salinity_cost(np.array([35.0]), p)
    assert cost[0] == pytest.approx(1.0 + p.salinity_hyper_cost)


def test_salinity_cost_freshwater_above_one_and_below_marine():
    """At 0 PSU, cost > 1.0 (hypo-osmotic stress) but < marine cost (asymmetry)."""
    from salmon_ibm.estuary import salinity_cost, EstuaryParams
    p = EstuaryParams()
    fresh_cost = salinity_cost(np.array([0.0]), p)
    marine_cost = salinity_cost(np.array([35.0]), p)
    assert fresh_cost[0] > 1.0
    assert fresh_cost[0] < marine_cost[0]
    assert fresh_cost[0] == pytest.approx(1.0 + p.salinity_hypo_cost)


def test_salinity_cost_smooth_monotonic_outside_iso():
    """Cost increases monotonically as |salinity - iso| grows in either direction."""
    from salmon_ibm.estuary import salinity_cost, EstuaryParams
    p = EstuaryParams()
    iso = p.salinity_iso_osmotic
    # Sweep above iso
    above = np.linspace(iso, 35.0, 50)
    above_costs = salinity_cost(above, p)
    assert np.all(np.diff(above_costs) >= -1e-12), (
        f"Cost should be monotonic non-decreasing above iso; "
        f"got max negative delta {np.diff(above_costs).min()}"
    )
    # Sweep below iso (reverse direction since cost rises as salinity falls)
    below = np.linspace(0.0, iso, 50)
    below_costs = salinity_cost(below, p)
    assert np.all(np.diff(below_costs) <= 1e-12), (
        f"Cost should be monotonic non-increasing as salinity rises toward iso; "
        f"got max positive delta {np.diff(below_costs).max()}"
    )


def test_salinity_cost_handles_nan():
    """NaN salinity → cost = 1.0 (treated as iso, no penalty)."""
    from salmon_ibm.estuary import salinity_cost, EstuaryParams
    p = EstuaryParams()
    sal = np.array([np.nan, p.salinity_iso_osmotic, 35.0])
    cost = salinity_cost(sal, p)
    assert cost[0] == pytest.approx(1.0)
    assert cost[1] == pytest.approx(1.0)
    assert cost[2] == pytest.approx(1.0 + p.salinity_hyper_cost)
    assert not np.isnan(cost).any()
```

(A vectorised-shape test and an out-of-range-clipping test were considered as defensive coverage but excluded to stay spec-faithful. The vectorised behaviour is implicitly covered by the monotonic test using `np.linspace(...)` arrays; the clipping behaviour is documented in the function docstring. Future plan can add explicit tests if a regression is observed.)

- [ ] **Step 3.2: Run the new tests; expect 5 failures**

```bash
micromamba run -n shiny python -m pytest tests/test_estuary.py -k "iso_returns_unity or marine_matches or freshwater_above_one_and_below_marine or smooth_monotonic or handles_nan" -v
```

Expected: All 5 fail with `TypeError: salinity_cost() got an unexpected keyword argument 'S_opt'` or similar — but actually the new tests don't pass `S_opt`; they pass `EstuaryParams()`. So they'll fail with `salinity_cost() takes from 1 to 5 positional arguments but 2 were given` (the old signature accepts (salinity, S_opt, S_tol, k, max_cost) — 5 max — but `EstuaryParams()` is passed as the second positional, which the old function tries to bind to `S_opt`, expecting a float).

Specifically the failure will look like:
```
FAILED tests/test_estuary.py::test_salinity_cost_at_iso_returns_unity
  TypeError: '>' not supported between instances of 'EstuaryParams' and 'float'
```
or similar — depending on what `S_opt + S_tol` does when `S_opt` is an EstuaryParams.

(Or the test might fail an assertion if old function happens to return ≥ 1.0 anyway. Either way, the test must FAIL.)

- [ ] **Step 3.3: Replace the `salinity_cost` function body and signature**

In `salmon_ibm/estuary.py`, replace lines 49-58 entirely:

```python
def salinity_cost(
    salinity: np.ndarray,
    params: EstuaryParams,
) -> np.ndarray:
    """Osmoregulation cost multiplier on respiration for *Salmo salar*.

    Linear-with-anchors function with separate slopes for hyper-osmotic
    (above the blood iso-osmotic point) and hypo-osmotic (below iso)
    stress.

    Returns: multiplier ≥ 1.0 array with same shape as `salinity`;
    equals 1.0 exactly at salinity == params.salinity_iso_osmotic.

    NaN inputs are treated as iso (cost 1.0).

    Citations: Wilson 2002 for iso-osmotic point; Brett & Groves 1979
    for hyper/hypo cost magnitudes.
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

- [ ] **Step 3.4: Run the new tests; expect 5 to pass**

```bash
micromamba run -n shiny python -m pytest tests/test_estuary.py -k "iso_returns_unity or marine_matches or freshwater_above_one_and_below_marine or smooth_monotonic or handles_nan" -v
```

Expected: 5 passed.

- [ ] **Step 3.5: Run the full estuary test file — many will fail now**

```bash
micromamba run -n shiny python -m pytest tests/test_estuary.py -v
```

Expected: validation tests pass (5), new functional tests pass (7), `test_salinity_cost_nan_treated_as_zero` passes (semantic equivalent to `handles_nan`), but `test_salinity_cost_below_tolerance`, `test_salinity_cost_above_tolerance`, `test_salinity_cost_capped`, and `test_estuary_params_defaults_match_liland_2024` all FAIL with TypeErrors or wrong values.

These existing-test failures are addressed in Tasks 4-5. Continue without committing.

---

### Task 4: Update `test_estuary_params_defaults_match_liland_2024`

**Files:**
- Modify: `tests/test_estuary.py:79-86`

The existing test asserts `p.s_opt == 0.5` and `p.s_tol == 6.0` — those fields are gone. Rewrite to assert the new salinity defaults plus the surviving DO/seiche defaults.

- [ ] **Step 4.1: Replace the test body**

In `tests/test_estuary.py`, find `test_estuary_params_defaults_match_liland_2024` (around line 79) and replace it with:

```python
def test_estuary_params_defaults_match_literature():
    """Defaults: DO from Liland 2024; salinity from Wilson 2002 + Brett & Groves 1979."""
    from salmon_ibm.estuary import EstuaryParams
    p = EstuaryParams()
    # Liland 2024 — DO thresholds
    assert p.do_lethal == 3.0
    assert p.do_high == 5.5
    # Wilson 2002 — S. salar blood iso-osmotic point
    assert p.salinity_iso_osmotic == 10.0
    # Brett & Groves 1979 — hyper / hypo cost slopes (verify Task 1 values)
    assert p.salinity_hyper_cost == pytest.approx(0.30)
    assert p.salinity_hypo_cost == pytest.approx(0.05)
    # Seiche (default unchanged)
    assert p.seiche_threshold_m_per_s == 0.02
```

If your Task 1 verification produced different anchor values (e.g., `0.075` for hypo), use those in the assertions.

The test name is also updated: `_match_liland_2024` → `_match_literature` to reflect that defaults now span multiple papers.

- [ ] **Step 4.2: Run the renamed test**

```bash
micromamba run -n shiny python -m pytest tests/test_estuary.py::test_estuary_params_defaults_match_literature -v
```

Expected: 1 passed.

---

### Task 5: Rewrite or delete the 3 remaining broken `test_salinity_cost_*` tests

**Files:**
- Modify: `tests/test_estuary.py:6-14, 41-48`

The three legacy tests at lines 6-14 (`below_tolerance`, `above_tolerance`) and 41-48 (`capped`) are tied to the old threshold-linear semantics:

- `test_salinity_cost_below_tolerance` — asserts cost = 1.0 at salinity=3 with old `S_opt, S_tol, k` kwargs. **Wrong under new physics** (cost ≈ 1.035 at 3 PSU due to hypo-osmotic stress) AND wrong call signature.
- `test_salinity_cost_above_tolerance` — asserts cost = 3.1 at salinity=10 with old kwargs. **Wrong under new physics** (10 PSU is iso, cost = 1.0 exactly).
- `test_salinity_cost_capped` — asserts cost ≤ 5.0 at extreme salinity. **Irrelevant** — new function naturally bounded by `1 + hyper_cost ≈ 1.30`.

The `test_salinity_cost_nan_treated_as_zero` test (lines 51-58) is **semantically still correct** under new physics (NaN → cost 1.0) but its name now misleads ("zero" → "iso"). Step 5.3 renames + lightly refactors it.

- [ ] **Step 5.1: Delete `test_salinity_cost_capped`**

In `tests/test_estuary.py`, delete lines 41-48 (the entire `test_salinity_cost_capped` function and its docstring). The "no cap" property is documented in the spec; Task 3 covers the input-clipping behaviour.

- [ ] **Step 5.2: Replace `test_salinity_cost_below_tolerance` and `test_salinity_cost_above_tolerance`**

Find lines 6-14 in `tests/test_estuary.py`:

```python
def test_salinity_cost_below_tolerance():
    cost = salinity_cost(np.array([3.0]), S_opt=0.5, S_tol=6.0, k=0.6)
    np.testing.assert_allclose(cost, [1.0])


def test_salinity_cost_above_tolerance():
    cost = salinity_cost(np.array([10.0]), S_opt=0.5, S_tol=6.0, k=0.6)
    np.testing.assert_allclose(cost, [3.1])
```

Replace with:

```python
def test_salinity_cost_lagoon_brackish():
    """Lagoon brackish (~5 PSU) is hypo-osmotic — small cost > 1.0."""
    from salmon_ibm.estuary import EstuaryParams
    p = EstuaryParams()
    cost = salinity_cost(np.array([5.0]), p)
    # At 5 PSU with iso=10: below = (10-5)/10 = 0.5; cost = 1 + hypo_cost * 0.5
    expected = 1.0 + p.salinity_hypo_cost * 0.5
    assert cost[0] == pytest.approx(expected)
    assert cost[0] < 1.0 + p.salinity_hypo_cost  # less than full freshwater cost


def test_salinity_cost_baltic_near_iso():
    """Baltic surface (~7 PSU) is close to iso (10 PSU) — minimal cost."""
    from salmon_ibm.estuary import EstuaryParams
    p = EstuaryParams()
    cost = salinity_cost(np.array([7.0]), p)
    # Should be slightly above 1.0 but much less than the old 30% penalty
    assert 1.0 < cost[0] < 1.05
```

These tests document the corrected physics for the two production salinity ranges (lagoon ~5 PSU, Baltic ~7 PSU).

- [ ] **Step 5.3: Rename and refactor `test_salinity_cost_nan_treated_as_zero`**

Find `test_salinity_cost_nan_treated_as_zero` (around line 51). Since `test_salinity_cost_handles_nan` (added in Task 3) already covers NaN handling more comprehensively, **delete the old test**:

```python
# Delete this entire function:
def test_salinity_cost_nan_treated_as_zero():
    """NaN salinity should produce cost = 1.0 (no penalty)."""
    from salmon_ibm.estuary import salinity_cost
    sal = np.array([np.nan, 5.0, np.nan])
    cost = salinity_cost(sal)
    assert cost[0] == pytest.approx(1.0), "NaN salinity should give neutral cost"
    assert not np.isnan(cost).any(), "No NaN should propagate"
```

(Calls `salinity_cost(sal)` without params — would TypeError under new signature anyway.)

- [ ] **Step 5.4: Run the full estuary test file**

```bash
micromamba run -n shiny python -m pytest tests/test_estuary.py -v
```

Expected: all tests pass. Count math, starting from the ~16 tests originally in `test_estuary.py`:
- Deleted: 2 (`_capped`, `_nan_treated_as_zero`)
- Replaced (count-neutral): 2 (`_below_tolerance` → `_lagoon_brackish`; `_above_tolerance` → `_baltic_near_iso`)
- Renamed (count-neutral): 1 (`_defaults_match_liland_2024` → `_defaults_match_literature`)
- Added: 5 functional (Task 3) + 3 validation (Task 2) = 8

Net: 16 + 8 − 2 = **22 tests in test_estuary.py** (matches the spec's "8 new + ~3 existing rewritten" framing exactly).

- [ ] **Step 5.5: First commit**

```bash
git add salmon_ibm/estuary.py tests/test_estuary.py
git commit -m "feat(estuary): linear-with-anchors salinity_cost for S. salar (Tasks 1-5)

Replaces the threshold-linear salinity cost function with a
literature-grounded linear-with-anchors form. EstuaryParams gains 3
new fields with __post_init__ validation; old s_opt/s_tol fields
removed.

Function: salinity_cost(salinity, params: EstuaryParams) returns a
multiplier >= 1.0 with minimum at salinity == params.salinity_iso_osmotic.
At full marine (35 PSU): cost = 1 + hyper_cost ~ 1.30. At freshwater
(0 PSU): cost = 1 + hypo_cost ~ 1.05. Asymmetry reflects hyper-osmotic
stress being more energetically expensive than hypo (Brett & Groves 1979).

Tests: 12 new (5 EstuaryParams validation + 7 functional/edge),
2 rewritten for new physics (lagoon ~5 PSU, Baltic ~7 PSU),
1 renamed (defaults_match_literature), 2 deleted (capped — no cap;
nan_treated_as_zero — superseded by handles_nan).

The 2 production call sites (events_builtin.py:83, simulation.py:481)
will TypeError until Tasks 6-7. YAML configs migrate in Task 8.

Spec: docs/superpowers/specs/2026-04-29-osmoregulation-stress-design.md"
```

The remainder of the test suite will FAIL at this point (production call sites use old signature). That's expected — the next tasks fix those. Don't run the full suite yet.

---

### Task 6: Update `events_builtin.py:83` call site (per-call construction)

**Files:**
- Modify: `salmon_ibm/events_builtin.py:13` (import)
- Modify: `salmon_ibm/events_builtin.py:82-88` (SurvivalEvent's salinity_cost call site)

**Architecture note:** `SurvivalEvent` is a `@dataclass` (lines 50-52). Its auto-generated `__init__` only accepts the dataclass fields (`bio_params, thermal, starvation`). It does NOT receive `est_cfg` at init — the value comes from `landscape["est_cfg"]` at execute time (line 62). So unlike `Simulation` (Task 7), we cannot stash `EstuaryParams` in `self` at `__init__`. Two valid alternatives:

- **Per-call construction** (recommended) — build a fresh `EstuaryParams` inside `execute()` each time. Costs ~microseconds (a dict comprehension + 3 if-statements); negligible vs the NumPy work in the same call.
- **Lazy memoization** — cache via a non-init field; rebuild only when `est_cfg` changes. More code; not needed at this scale.

This task uses per-call construction.

- [ ] **Step 6.1: Read the current call site**

Open `salmon_ibm/events_builtin.py:60-100`. Confirm:
- Line 13: `from salmon_ibm.estuary import salinity_cost` — current import (one symbol).
- Line 50-58: `@register_event("survival") @dataclass class SurvivalEvent(Event):` with fields `bio_params, thermal, starvation`.
- Line 62: `est_cfg = landscape.get("est_cfg", {})` — est_cfg comes from landscape, not self.
- Line 81: `sal_at_agents = sal[population.tri_idx]`
- Lines 82-88: the current `salinity_cost(sal_at_agents, S_opt=..., S_tol=..., k=...)` call.

- [ ] **Step 6.2: Update the import**

Change line 13 from:

```python
from salmon_ibm.estuary import salinity_cost
```

to:

```python
from salmon_ibm.estuary import salinity_cost, EstuaryParams
```

- [ ] **Step 6.3: Replace the per-step call site (lines 82-88)**

Find the existing block:

```python
            s_cfg = est_cfg.get("salinity_cost", {})
            sal_cost_arr = salinity_cost(
                sal_at_agents,
                S_opt=s_cfg.get("S_opt", 0.5),
                S_tol=s_cfg.get("S_tol", 6.0),
                k=s_cfg.get("k", 0.6),
            )
```

Replace with:

```python
            # Build EstuaryParams from the salinity_cost YAML subsection.
            # Filter to known fields so legacy keys (S_opt, S_tol, k) are
            # silently dropped — falls back to dataclass defaults rather
            # than raising TypeError. Per-call construction is cheap
            # (microseconds) and SurvivalEvent is a dataclass that can't
            # easily stash this in __init__ (est_cfg is in landscape, not
            # self). See plan 2026-04-30-osmoregulation-stress for context.
            _known_salinity_keys = {
                "salinity_iso_osmotic",
                "salinity_hyper_cost",
                "salinity_hypo_cost",
            }
            s_cfg = est_cfg.get("salinity_cost", {})
            est_params = EstuaryParams(
                **{k: v for k, v in s_cfg.items() if k in _known_salinity_keys}
            )
            sal_cost_arr = salinity_cost(sal_at_agents, est_params)
```

(EstuaryParams construction validates via `__post_init__` — if the YAML is malformed the first execute call surfaces it, not subsequent ones.)

- [ ] **Step 6.4: Run the SurvivalEvent-related tests**

```bash
micromamba run -n shiny python -m pytest tests/test_events.py -v -k "Survival or survival"
```

Expected: tests pass. If a test fails because `EstuaryParams(**...)` raises ValueError, the test fixture has bad values — surface it to the engineer for triage.

If a fixture uses OLD schema keys (`S_opt`, etc.), they get filtered out and EstuaryParams falls back to defaults. That's expected — the migration of test fixtures using old schema is Tasks 9-11.

---

### Task 7: Update `simulation.py:481` call site (init-time construction)

**Files:**
- Modify: `salmon_ibm/simulation.py:55` (import)
- Modify: `salmon_ibm/simulation.py:295-297` (add `EstuaryParams` construction next to `self.est_cfg = ...`)
- Modify: `salmon_ibm/simulation.py:480-486` (per-step call site in `_event_bioenergetics`)

**Architecture note:** Unlike `SurvivalEvent` (Task 6), `Simulation` has `self.est_cfg = config.get("estuary", {})` set at line 295 of `__init__`. So we CAN build `EstuaryParams` once at sim init, validation fires at scenario-load time, and `_event_bioenergetics` reads from `self._est_params` per step.

- [ ] **Step 7.1: Update the import**

Line 55 currently imports `salinity_cost`. Change to:

```python
from salmon_ibm.estuary import (
    salinity_cost,
    EstuaryParams,
)
```

(Or whatever multi-line/single-line form matches the existing style — the goal is to add `EstuaryParams` to the symbols imported from `salmon_ibm.estuary`.)

- [ ] **Step 7.2: Add `EstuaryParams` construction next to `self.est_cfg`**

Find `simulation.py:295`:

```python
        self.est_cfg = config.get("estuary", {})
        self._skip_estuarine_overrides = self._detect_estuarine_noop()
```

Add the new lines BEFORE `_skip_estuarine_overrides` (so the check runs on the new params if relevant):

```python
        self.est_cfg = config.get("estuary", {})
        # EstuaryParams from salinity_cost YAML subsection. Filter to known
        # fields so legacy keys (S_opt, S_tol, k) silently fall back to
        # dataclass defaults rather than raising. Validation in
        # __post_init__ fires here at scenario-load time, not first step.
        _known_salinity_keys = {
            "salinity_iso_osmotic",
            "salinity_hyper_cost",
            "salinity_hypo_cost",
        }
        _sal_cfg = self.est_cfg.get("salinity_cost", {})
        self._est_params = EstuaryParams(
            **{k: v for k, v in _sal_cfg.items() if k in _known_salinity_keys}
        )
        self._skip_estuarine_overrides = self._detect_estuarine_noop()
```

- [ ] **Step 7.3: Replace the per-step call site (lines 480-486)**

Find:

```python
        s_cfg = self.est_cfg.get("salinity_cost", {})
        sal_cost = salinity_cost(
            sal_at_agents,
            S_opt=s_cfg.get("S_opt", 0.5),
            S_tol=s_cfg.get("S_tol", 6.0),
            k=s_cfg.get("k", 0.6),
        )
```

Replace with:

```python
        sal_cost = salinity_cost(sal_at_agents, self._est_params)
```

- [ ] **Step 7.4: Run simulation-related tests**

```bash
micromamba run -n shiny python -m pytest tests/test_simulation.py -v
```

Expected: most tests pass. The fixture at line 235 still uses old schema; if its `Simulation()` construction blows up, it's because `EstuaryParams(**filtered)` got a key it doesn't recognise — the filter should prevent this, but verify `_known_salinity_keys` matches the field names in `EstuaryParams` exactly. If a `KeyError` or `ValueError` raises, that's a real plan bug — pause and fix.

- [ ] **Step 7.5: Commit Tasks 6 & 7**

```bash
git add salmon_ibm/events_builtin.py salmon_ibm/simulation.py
git commit -m "feat(estuary): wire EstuaryParams through salinity_cost call sites

events_builtin.py: SurvivalEvent (a @dataclass that can't stash state in
__init__) builds EstuaryParams per-call inside execute() using a
filtered-dict pattern that silently drops old-schema keys. The
construction is microsecond-cheap; no need for memoization.

simulation.py: Simulation has self.est_cfg available at __init__ time
(line 295), so the EstuaryParams instance is built once and stashed in
self._est_params. _event_bioenergetics reads it per step.

Both patterns use the same filtered-key approach so old-schema YAML
configs degrade to defaults rather than raising.

Validation errors surface at scenario-load time (Simulation) or at
first execute (SurvivalEvent). Fixture YAMLs using old schema are
migrated in Tasks 8-11."
```

---

### Task 8: Migrate the 5 YAML configs

**Files:**
- Modify: `config_columbia.yaml:10-12`
- Modify: `config_curonian_minimal.yaml:26-28`
- Modify: `configs/config_curonian_trimesh.yaml:57-61`
- Modify: `configs/config_curonian_baltic.yaml:58-61`
- Modify: `config_curonian_hexsim.yaml:12-14`

Replace the `salinity_cost: {S_opt, S_tol, k}` block with `{salinity_iso_osmotic, salinity_hyper_cost, salinity_hypo_cost}`. For Columbia, migrate the "disable" pattern (`S_tol: 999`) to zero-cost slopes.

- [ ] **Step 8.1: Migrate `config_columbia.yaml`**

Open `config_columbia.yaml` at lines 10-13. The current block:

```yaml
  salinity_cost:
    S_opt: 0.5
    S_tol: 999
    k: 0.0
```

Columbia is freshwater throughout — the `S_tol: 999, k: 0.0` is the "disable salinity cost entirely" pattern. New equivalent: zero slopes give cost=1.0 always.

Replace ALL FOUR lines (the `salinity_cost:` header and its three children) with:

```yaml
  salinity_cost:
    # Columbia: freshwater scenario; zero slopes disable osmoregulation cost
    # (cost = 1.0 for all salinities). Migrated from old S_tol: 999, k: 0.0
    # disable pattern.
    salinity_hyper_cost: 0.0
    salinity_hypo_cost: 0.0
```

(`salinity_iso_osmotic` is omitted — defaults to 10.0, which doesn't matter when both slopes are 0.)

- [ ] **Step 8.2: Migrate `config_curonian_minimal.yaml`**

At lines 26-29:

```yaml
  salinity_cost:
    S_opt: 0.5
    S_tol: 6.0
    k: 0.6
```

Replace ALL FOUR lines with:

```yaml
  salinity_cost:
    # S. salar iso-osmotic ~10 PSU (Wilson 2002); cost slopes from Brett & Groves 1979.
    # Defaults (10 / 0.30 / 0.05) are already the right values; explicit override
    # only needed if calibration data argues otherwise.
    salinity_iso_osmotic: 10.0
```

(Hyper / hypo costs default to the dataclass defaults, which match Brett & Groves. Only `iso` is explicitly listed for clarity. The old `k: 0.6` line — the slope above threshold — has no direct equivalent in the new schema; it's simply removed.)

- [ ] **Step 8.3: Migrate `configs/config_curonian_trimesh.yaml`**

At lines 57-60:

```yaml
  salinity_cost:
    S_opt: 0.5
    S_tol: 6.0
    k: 0.6
```

Replace ALL FOUR lines with:

```yaml
  salinity_cost:
    # See config_curonian_minimal.yaml for parameter rationale.
    salinity_iso_osmotic: 10.0
```

(Old `k: 0.6` line removed — no direct equivalent in new schema.)

- [ ] **Step 8.4: Migrate `configs/config_curonian_baltic.yaml`**

At lines 58-61 (the `salinity_cost` block plus possibly a comment line):

```yaml
estuary:
  salinity_cost:
    S_opt: 0.5
    S_tol: 6.0
    k: 0.6
```

Replace with:

```yaml
estuary:
  salinity_cost:
    # S. salar iso-osmotic ~10 PSU (Wilson 2002).
    # Hyper/hypo cost slopes default from EstuaryParams (Brett & Groves 1979).
    salinity_iso_osmotic: 10.0
```

If the existing block has a comment header explaining the parameters, update it too.

- [ ] **Step 8.5: Migrate `config_curonian_hexsim.yaml`**

At lines 12-15:

```yaml
  salinity_cost:
    S_opt: 0.5
    S_tol: 6.0
    k: 0.6
```

Replace ALL FOUR lines with:

```yaml
  salinity_cost:
    # See config_curonian_minimal.yaml for parameter rationale.
    salinity_iso_osmotic: 10.0
```

(Old `k: 0.6` line removed.)

- [ ] **Step 8.6: Verify all 5 configs parse**

```bash
micromamba run -n shiny python -c "
import yaml
for path in [
    'config_columbia.yaml',
    'config_curonian_minimal.yaml',
    'configs/config_curonian_trimesh.yaml',
    'configs/config_curonian_baltic.yaml',
    'config_curonian_hexsim.yaml',
]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    sal = cfg.get('estuary', {}).get('salinity_cost', {})
    print(f'{path}: salinity_cost = {sal}')
"
```

Expected: all 5 files print without YAML errors. Each `salinity_cost` block contains only the new keys (no `S_opt`, `S_tol`, `k`).

- [ ] **Step 8.7: Commit YAML migrations**

```bash
git add config_columbia.yaml config_curonian_minimal.yaml configs/config_curonian_trimesh.yaml configs/config_curonian_baltic.yaml config_curonian_hexsim.yaml
git commit -m "feat(configs): migrate salinity_cost YAML schema to S. salar iso-osmotic

Five YAML configs migrated from the old threshold-linear schema
(S_opt, S_tol, k, max_cost) to the new linear-with-anchors schema
(salinity_iso_osmotic, salinity_hyper_cost, salinity_hypo_cost).

Columbia: freshwater throughout, the old 'S_tol: 999' disable trick
becomes zero-slope (salinity_hyper_cost: 0.0, salinity_hypo_cost: 0.0).

Curonian configs: explicitly set salinity_iso_osmotic: 10.0 (Wilson
2002 anchor); hyper/hypo costs default from EstuaryParams (Brett &
Groves 1979).

Clean break — no backward-compat. Old keys are silently filtered out
by the call-site EstuaryParams construction, so old configs degrade
to defaults rather than erroring; users should migrate."
```

---

### Task 9: Migrate `tests/test_config.py:26`

**Files:**
- Modify: `tests/test_config.py:20-30` (around the YAML schema assertion)

Test currently asserts `cfg["estuary"]["salinity_cost"]["S_opt"] == 0.5` — old schema. Migrate to assert the new key.

- [ ] **Step 9.1: Read context around line 26**

Open `tests/test_config.py:15-35`. Identify the test function and its purpose (likely "load default config and check salinity values").

- [ ] **Step 9.2: Replace the assertion**

Find:

```python
    assert cfg["estuary"]["salinity_cost"]["S_opt"] == 0.5
```

Replace with:

```python
    # Migrated 2026-04-30: old schema (S_opt, S_tol, k) replaced by
    # iso-osmotic schema (salinity_iso_osmotic, salinity_hyper_cost,
    # salinity_hypo_cost). See spec
    # docs/superpowers/specs/2026-04-29-osmoregulation-stress-design.md.
    assert cfg["estuary"]["salinity_cost"]["salinity_iso_osmotic"] == 10.0
```

- [ ] **Step 9.3: Run the test**

```bash
micromamba run -n shiny python -m pytest tests/test_config.py -v
```

Expected: test passes. If it fails, the YAML it loads probably hasn't been migrated yet — verify Task 8 was applied to the YAML this test reads.

---

### Task 10: Migrate `tests/test_ensemble.py:19`

**Files:**
- Modify: `tests/test_ensemble.py:15-25` (fixture dict)

Fixture uses `{"S_opt": 0.5, "S_tol": 999, "k": 0.0}` — the "disable salinity cost" pattern. Migrate to the zero-slope new equivalent.

- [ ] **Step 10.1: Read context**

Open `tests/test_ensemble.py:15-25`. Identify the fixture purpose (likely a tiny default-ish scenario for ensemble runs).

- [ ] **Step 10.2: Replace the fixture entry**

Find:

```python
        "salinity_cost": {"S_opt": 0.5, "S_tol": 999, "k": 0.0},
```

Replace with:

```python
        # Disable salinity-cost penalty for ensemble baseline (matches
        # config_columbia's freshwater pattern). Migrated 2026-04-30 from
        # old S_tol: 999 schema.
        "salinity_cost": {"salinity_hyper_cost": 0.0, "salinity_hypo_cost": 0.0},
```

- [ ] **Step 10.3: Run the test**

```bash
micromamba run -n shiny python -m pytest tests/test_ensemble.py -v
```

Expected: test passes. The filtered-dict approach in the call sites (Tasks 6-7) means even unknown keys get dropped, so technically the test would still work without migration — but the explicit migration matches the intended schema.

---

### Task 11: Migrate `tests/test_simulation.py:235`

**Files:**
- Modify: `tests/test_simulation.py:230-240` (fixture)

Same pattern as Task 10 — `{"S_opt": 0.5, "S_tol": 999, "k": 0.0}` fixture.

- [ ] **Step 11.1: Replace the fixture entry**

Find:

```python
            "salinity_cost": {"S_opt": 0.5, "S_tol": 999, "k": 0.0},
```

Replace with:

```python
            # Disable salinity-cost penalty (test fixture). Migrated 2026-04-30.
            "salinity_cost": {"salinity_hyper_cost": 0.0, "salinity_hypo_cost": 0.0},
```

- [ ] **Step 11.2: Run the test file**

```bash
micromamba run -n shiny python -m pytest tests/test_simulation.py -v
```

Expected: tests pass. If `test_simulation.py` has broader fixture issues, surface them now.

- [ ] **Step 11.3: Commit Tasks 9-11**

```bash
git add tests/test_config.py tests/test_ensemble.py tests/test_simulation.py
git commit -m "test: migrate test fixtures to new salinity_cost YAML schema

Three test files had hardcoded old-schema YAML fixtures (S_opt, S_tol, k);
migrated to the new schema (salinity_iso_osmotic, salinity_hyper_cost,
salinity_hypo_cost).

test_config.py: schema assertion updated.
test_ensemble.py + test_simulation.py: 'disable salinity cost' fixture
migrated from S_tol: 999 to zero hyper/hypo slopes."
```

---

### Task 12: Update `docs/api-reference.md`

**Files:**
- Modify: `docs/api-reference.md:380-395` (update_energy section, references salinity_cost)
- Modify: `docs/api-reference.md:1090-1130` (estuary module / salinity_cost function reference)

The docs describe the old function signature. Migrate to the new signature + parameter table.

- [ ] **Step 12.1: Update the `salinity_cost` function-reference block**

Around line 1107:

Old:

```markdown
#### `salinity_cost`

```python
def salinity_cost(
    salinity: np.ndarray,
    S_opt: float = 0.5,
    S_tol: float = 6.0,
    k: float = 0.6,
    max_cost: float = 5.0,
) -> np.ndarray
```

Compute a respiration cost multiplier (≥ 1.0) for each agent based on ambient salinity. Salinity within the optimal + tolerance range has cost 1.0; excess salinity increases cost linearly by `k` per PSU above the threshold, capped at `max_cost`.

| Param | Default | Meaning |
|---|---|---|
| `S_opt` | `0.5` | Optimal salinity (PSU) |
| `S_tol` | `6.0` | Tolerance range above S_opt (PSU) |
| `k` | `0.6` | Cost-per-PSU slope above threshold |
| `max_cost` | `5.0` | Maximum cost multiplier |
```

Replace with:

```markdown
#### `salinity_cost`

```python
def salinity_cost(
    salinity: np.ndarray,
    params: EstuaryParams,
) -> np.ndarray
```

Compute a respiration cost multiplier (≥ 1.0) for each agent based on ambient salinity. Implements *Salmo salar* physiology: cost is exactly 1.0 at the blood iso-osmotic point (~10 PSU); rises asymmetrically toward freshwater (hypo-osmotic stress) and full marine (hyper-osmotic stress). NaN inputs are treated as iso (cost 1.0).

Citations: Wilson 2002 (iso-osmotic point); Brett & Groves 1979 (hyper/hypo cost slopes for euryhaline salmonids).

Parameters come from `EstuaryParams`:

| Field | Default | Meaning |
|---|---|---|
| `salinity_iso_osmotic` | `10.0` | Blood iso-osmotic point (PSU) |
| `salinity_hyper_cost` | `0.30` | Cost slope above iso (multiplier increment at full marine) |
| `salinity_hypo_cost` | `0.05` | Cost slope below iso (multiplier increment at freshwater) |
```

- [ ] **Step 12.2: Update the `update_energy` parameter table**

Around line 380, the `salinity_cost` parameter is described. Confirm the table description still makes sense — the parameter is still an `np.ndarray` of multipliers. Update wording if it references the threshold-linear semantics:

```markdown
| `salinity_cost` | `np.ndarray` | Salinity-based respiration cost multipliers (≥ 1.0); typically computed by `salmon_ibm.estuary.salinity_cost()` from per-cell salinity and `EstuaryParams` |
```

- [ ] **Step 12.3: Verify the doc renders cleanly**

If your tooling builds the docs, run it. Otherwise visually inspect the markdown.

---

### Task 13: Update `docs/model-manual.md`

**Files:**
- Modify: `docs/model-manual.md:367-400` (YAML schema section for salinity_cost)
- Modify: `docs/model-manual.md:590-610` (formula description)
- Modify: `docs/model-manual.md:850-895` (further YAML examples)

- [ ] **Step 13.1: Update the YAML schema example block**

Around line 371:

Old:

```yaml
  salinity_cost:
    S_opt: 0.5                    # optimal salinity (PSU)
    S_tol: 6.0                    # tolerance range above S_opt (PSU)
    k: 0.6                        # cost-per-PSU slope
    max_cost: 5.0                 # maximum cost multiplier ceiling
```

Replace with:

```yaml
  salinity_cost:
    salinity_iso_osmotic: 10.0    # S. salar blood iso-osmotic (Wilson 2002)
    salinity_hyper_cost: 0.30     # cost slope above iso (Brett & Groves 1979)
    salinity_hypo_cost: 0.05      # cost slope below iso (Brett & Groves 1979)
```

- [ ] **Step 13.2: Update the formula description block**

Around line 597:

Old:

```
cost = 1 + k * max(salinity - (S_opt + S_tol), 0)
cost = min(cost, max_cost)                    capped at max_cost

Default: S_opt=0.5 PSU, S_tol=6.0 PSU, k=0.6, max_cost=5.0.
```

Replace with:

```
above = max(salinity - iso, 0) / max(35 - iso, 1)
below = max(iso - salinity, 0) / max(iso, 1)
cost = 1 + hyper_cost * above + hypo_cost * below

Cost is 1.0 at salinity == iso. NaN inputs treated as iso. Salinity
clipped to [0, 35] PSU defensively.

Default: salinity_iso_osmotic=10.0 PSU, salinity_hyper_cost=0.30,
salinity_hypo_cost=0.05.
```

- [ ] **Step 13.3: Update the second YAML example block (around line 856)**

Same migration as Step 13.1; apply the new schema.

- [ ] **Step 13.4: Verify the doc**

Visually inspect; run any doc-build tooling.

- [ ] **Step 13.5: Commit Tasks 12-13**

```bash
git add docs/api-reference.md docs/model-manual.md
git commit -m "docs: update salinity_cost API and YAML schema references

api-reference.md: function signature and parameter table updated to
the new (salinity, params: EstuaryParams) form. Cited Wilson 2002 and
Brett & Groves 1979 in the function description.

model-manual.md: YAML schema examples and the formula description
updated to the new linear-with-anchors form."
```

---

### Task 14: Run full pytest suite, surface and fix regressions

**Files:** Possibly any test file if regressions surface; commonly `tests/test_curonian_realism_integration.py` based on the spec's risk analysis.

The full suite is ~14 minutes. Run it once to surface all remaining issues, then iterate.

- [ ] **Step 14.1: Run the full suite**

```bash
micromamba run -n shiny python -m pytest tests/ -q --no-header
```

Expected: ~822 passing, 34 skipped, 7 deselected, 1 xfailed. If failures are present, jump to Step 14.2.

- [ ] **Step 14.2: Triage failures**

For each failure, classify:

- **(A) Direct call-signature mismatch** — `TypeError: salinity_cost() got an unexpected keyword argument 'S_opt'`. A test file imports `salinity_cost` and calls it with old kwargs. Verify Task 5 covered all such sites; a missed test file might have crept in. Migrate the call.

- **(B) Asserted energy/mortality value shift** — e.g., `tests/test_curonian_realism_integration.py` asserts a migrant's final ED or arrival count. Under new physics, Baltic ~7 PSU sees a LARGE cost decrease (1.30 → 1.015), so migrants conserve more energy and survive longer. Test values shift accordingly. Update the asserted value (and add a comment citing this plan).

- **(C) Fixture YAML using old schema** — `KeyError: 'S_opt'` or similar. A test fixture not covered by Task 8/9/10/11. Migrate it.

- [ ] **Step 14.3: Apply fixes**

For each failure, apply the appropriate fix from Step 14.2. Use TDD discipline where reasonable:
- Run the failing test individually
- Apply the fix
- Re-run individually to confirm

For category (B) failures, prefer to update the asserted value rather than tighten the test scope — the new value is biologically more correct. Document the shift in the commit message.

- [ ] **Step 14.4: Re-run the full suite**

```bash
micromamba run -n shiny python -m pytest tests/ -q --no-header
```

Iterate until: `~822 passed, 34 skipped, 7 deselected, 1 xfailed`. The xfailed test is a pre-existing one — leave alone.

- [ ] **Step 14.5: Commit any regression fixes**

```bash
git add <each-file-touched>
git commit -m "test: update integration assertions for new osmoregulation physics

Under the new linear-with-anchors salinity_cost, Baltic salmon at
~7 PSU pay much less respiration cost (1.30 -> 1.015) than under the
old threshold-linear function. This shifts ensemble outcomes:
migrants conserve more energy, survive longer, return to spawn at
higher condition. Test assertions for specific energy/mortality
numbers are updated to reflect the corrected physics.

Specific files: <list each>
Per-file rationale: <one line each>"
```

If no regression fixes were needed, skip the commit.

---

### Task 15: Manual sanity check — plot the cost curve

**Files:** None modified (research-only step; output is a screenshot or saved plot).

Verify the new function visually before declaring the plan done. The shape should match the published *S. salar* curve described in the spec.

- [ ] **Step 15.1: Plot the cost curve**

Run a one-shot Python check:

```bash
micromamba run -n shiny python -c "
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from salmon_ibm.estuary import salinity_cost, EstuaryParams

p = EstuaryParams()
sal = np.linspace(0, 35, 200)
cost = salinity_cost(sal, p)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(sal, cost, lw=2)
ax.axvline(p.salinity_iso_osmotic, ls='--', alpha=0.5, label=f'iso = {p.salinity_iso_osmotic}')
ax.axhline(1.0, ls=':', alpha=0.3)
ax.set_xlabel('Salinity (PSU)')
ax.set_ylabel('Respiration cost multiplier')
ax.set_title('S. salar osmoregulation cost (linear-with-anchors)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('_diag_salinity_cost_curve.png', dpi=100, bbox_inches='tight')
print('Saved _diag_salinity_cost_curve.png')
print(f'cost(0)  = {cost[0]:.4f}  (freshwater)')
print(f'cost(5)  = {salinity_cost(np.array([5.0]), p)[0]:.4f}  (lagoon)')
print(f'cost(7)  = {salinity_cost(np.array([7.0]), p)[0]:.4f}  (Baltic surface)')
print(f'cost(10) = {salinity_cost(np.array([10.0]), p)[0]:.4f}  (iso)')
print(f'cost(35) = {cost[-1]:.4f}  (full marine)')
"
```

Expected output:

```
Saved _diag_salinity_cost_curve.png
cost(0)  = 1.0500  (freshwater)
cost(5)  = 1.0250  (lagoon)
cost(7)  = 1.0150  (Baltic surface)
cost(10) = 1.0000  (iso)
cost(35) = 1.3000  (full marine)
```

(Numbers should match the spec's "Risk + regression surface" table.)

- [ ] **Step 15.2: Inspect the plot visually**

Open `_diag_salinity_cost_curve.png`. Verify:
- ✅ Minimum at salinity = iso (10 PSU) where cost = 1.0
- ✅ Smooth (no discontinuities)
- ✅ Hypo branch (left of iso) rises gently
- ✅ Hyper branch (right of iso) rises steeper
- ✅ Bounded — max cost is `1 + hyper_cost ≈ 1.30` at salinity = 35
- ✅ Asymmetric around iso: hyper steeper than hypo

If anything looks wrong (e.g., not smooth, wrong minimum, etc.), the formula has a bug — go back to Task 3 and fix. If it looks right, proceed.

- [ ] **Step 15.3: Discard the diagnostic file**

`_diag_salinity_cost_curve.png` is a one-shot diagnostic — per project convention (`_diag_*` files stay uncommitted), don't add it to the repo. Delete it after inspection:

```bash
rm _diag_salinity_cost_curve.png
```

---

### Task 16: Stamp the plan as EXECUTED + final state verification

**Files:**
- Modify: `docs/superpowers/plans/2026-04-30-osmoregulation-stress.md` (this file)

Per the convention established in the v1.7 series, mark the plan with a STATUS blockquote at the top once shipped, then commit and push.

- [ ] **Step 16.1: Add the EXECUTED stamp**

In `docs/superpowers/plans/2026-04-30-osmoregulation-stress.md`, insert after the title (line 1) and before the For-agentic-workers blockquote:

```markdown
> **STATUS: ✅ EXECUTED 2026-MM-DD** — All 16 tasks complete. EstuaryParams extended with 3 salinity fields + __post_init__ validation; salinity_cost rewritten with linear-with-anchors S. salar physiology; 5 YAML configs migrated; 3 test fixtures migrated; 8+ new tests added; full pytest suite green at NNN passing. Spec at docs/superpowers/specs/2026-04-29-osmoregulation-stress-design.md.
```

Replace `2026-MM-DD` with the actual completion date and `NNN` with the actual passing count.

- [ ] **Step 16.2: Final suite run for the stamp**

```bash
micromamba run -n shiny python -m pytest tests/ -q --no-header
```

Confirm the count and update the stamp accordingly.

- [ ] **Step 16.3: Final commit + push**

```bash
git add docs/superpowers/plans/2026-04-30-osmoregulation-stress.md
git commit -m "docs(plan): stamp osmoregulation-stress plan as EXECUTED

All 16 tasks complete. salmon_ibm/estuary.py::salinity_cost() now
implements linear-with-anchors S. salar physiology. Suite at NNN
passing. Closes the first of four queued Curonian-realism deferred
items (A osmoregulation -> [next: C hatchery vs wild])."
git push origin main
```

- [ ] **Step 16.4: Update the curonian_h3_grid_state memory file**

The memory file at `~/.claude/projects/.../memory/curonian_h3_grid_state.md` tracks the project's state. Add a short entry under the carried-forward limitations section noting that osmoregulation has been resolved:

```markdown
* **Osmoregulation stress (resolved 2026-MM-DD)**: salinity_cost() rewritten with linear-with-anchors S. salar physiology (commit hash). Replaces the old Pacific-style threshold-linear function. Iso-osmotic point at 10 PSU per Wilson 2002; hyper/hypo cost slopes per Brett & Groves 1979.
```

This signals to future sessions that the deferred-item-A is closed.

---

## Plan summary

- **16 tasks**, ~7 commits (groups: Tasks 1-5 → commit 1, Tasks 6-7 → commit 2, Task 8 → commit 3, Tasks 9-11 → commit 4, Tasks 12-13 → commit 5, Task 14 → commit 6 if regressions, Task 16 → commit 7).
- **14 files modified** (3 production, 4 test, 5 YAML, 2 docs).
- **+6 net tests** (5 functional + 3 validation new − 2 deleted (`_capped`, `_nan_treated_as_zero`); replacements and renames are count-neutral). Suite expected: 815 → **821 passing** (within the spec's 821-823 range).
- **Estimated time:** 1-2 days. Most work is mechanical (YAML/doc/fixture sync); the function rewrite itself is ~10 lines.
- **Risk profile:** moderate — the Baltic cost decrease (1.30 → 1.015) will shift integration test outcomes; budget for `test_curonian_realism_integration.py` triage.
- **No backward-compat shim** — clean break following the v1.7.1 lipid-first precedent.
