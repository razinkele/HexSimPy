# Hatchery vs Wild C3.1 — Pre-Spawn Skip Probability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hatchery reproducers skip pre-spawn with Bernoulli probability `p_skip = 0.3` (configurable via YAML); wild reproducers always proceed (`p_skip = 0.0`). Single new BioParams field; reuses C2's HatcheryDispatch bundle pattern; one new dispatch site in `ReproductionEvent.execute`. Closes the first sub-tier of C3 (third in the C1 → C2 → C3 sequence).

**Architecture:** Add `pre_spawn_skip_prob: float = 0.0` field to `BalticBioParams`; extend `_apply_hatchery_overrides` `ALLOWED_OVERRIDE_KEYS` and add a `SCALAR_OVERRIDE_FIELDS` set for forward-compat; insert a Bernoulli skip filter in `ReproductionEvent.execute` between `reproducer_idx` computation and Poisson clutch sampling. Filter is gated by `landscape.get("hatchery_dispatch") is not None and hd.params.pre_spawn_skip_prob > 0` so pre-C3.1 scenarios are unaffected.

**Tech Stack:** Python 3.10+, NumPy, dataclasses, pytest; conda env `shiny`.

**Spec:** [`docs/superpowers/specs/2026-05-02-hatchery-c3-spawn-design.md`](../specs/2026-05-02-hatchery-c3-spawn-design.md) (commit `88c3a40`).

---

## File structure

**Modified files (4 total: 1 new test file + 3 modified):**

Production code (3):
- `salmon_ibm/baltic_params.py` — new `pre_spawn_skip_prob: float = 0.0` field on `BalticBioParams` + `__post_init__` range validation; `_apply_hatchery_overrides` extended with `SCALAR_OVERRIDE_FIELDS` set and `dataclasses.replace(..., **scalar_kwargs)` mechanism.
- `salmon_ibm/events_builtin.py` — top import gains `ORIGIN_HATCHERY`; `ReproductionEvent.execute` gains a Bernoulli skip filter immediately after `reproducer_idx = np.where(can_reproduce)[0]` (around line 314).
- `configs/baltic_salmon_species.yaml` — `pre_spawn_skip_prob: 0.0` at wild level (explicit baseline) and `pre_spawn_skip_prob: 0.3` inside `hatchery_overrides`, plus full provenance comment block citing Bouchard 2022, Christie 2014, Jonsson 2019.

Tests (1 new file):
- `tests/test_hatchery_c3_spawn.py` — 6 new tests.

**Test runner:**
```bash
micromamba run -n shiny python -m pytest tests/path/file.py::test_name -v
# whole suite
micromamba run -n shiny python -m pytest tests/ -v
```

Suite is ~14-26 minutes. Baseline before this plan: **842 passing on `hatchery-c2-bioparams` branch (post-C2)**. Expected after: **848 passing** (842 + 6 new tests, 0 regressions).

**Commit cadence (4 commits):**
- Task 1 → commit 1 (new field + __post_init__ validation)
- Task 2 → commit 2 (extended loader + scalar override mechanism)
- Task 3 → commit 3 (ReproductionEvent dispatch + YAML)
- Task 4 → commit 4 (full pytest + plan stamp + push)

**Branch:** `hatchery-c3-spawn` (created from `main` after C2 PR merges; if C2 still unmerged at start, branch from `hatchery-c2-bioparams` instead and rebase later — same convention as the C2→C1 stacking).

---

## Tasks

### Task 1: New field + `__post_init__` validation (TDD)

**Files:**
- Create branch
- Modify: `salmon_ibm/baltic_params.py` (append field to `BalticBioParams` + extend `__post_init__`)
- Create: `tests/test_hatchery_c3_spawn.py` (1 test for now: `test_pre_spawn_skip_prob_rejects_out_of_range`)

This task ships the new dataclass field with default 0.0 and the range-validation invariant.

- [ ] **Step 1.1: Create the branch**

```bash
# If C2 PR has merged into main:
git switch main
git pull origin main
git checkout -b hatchery-c3-spawn

# If C2 PR still open (likely at C3.1 implementation start):
git switch hatchery-c2-bioparams
git checkout -b hatchery-c3-spawn
```

(After C2 PR merges, Task 4's push step will rebase this branch onto main per the same convention as C2→C1 stacking.)

- [ ] **Step 1.2: Create `tests/test_hatchery_c3_spawn.py` with the failing validation test**

Create `tests/test_hatchery_c3_spawn.py`:

```python
"""Tests for the C3.1 hatchery vs wild pre-spawn skip probability.

Spec: docs/superpowers/specs/2026-05-02-hatchery-c3-spawn-design.md
"""
import pytest


def test_pre_spawn_skip_prob_rejects_out_of_range():
    """BalticBioParams.__post_init__ rejects pre_spawn_skip_prob outside
    [0, 1]. Locks the validation contract; covers both negative and
    >1.0 boundary cases. C3.1 spec mandates 0.0 <= p <= 1.0."""
    from salmon_ibm.baltic_params import BalticBioParams
    with pytest.raises(ValueError, match=r"pre_spawn_skip_prob must be"):
        BalticBioParams(pre_spawn_skip_prob=-0.1)
    with pytest.raises(ValueError, match=r"pre_spawn_skip_prob must be"):
        BalticBioParams(pre_spawn_skip_prob=1.5)
```

- [ ] **Step 1.3: Run the test; expect failure**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_c3_spawn.py::test_pre_spawn_skip_prob_rejects_out_of_range -v
```

Expected: FAILED — `BalticBioParams.__init__()` got an unexpected keyword argument `pre_spawn_skip_prob`.

- [ ] **Step 1.4: Add the `pre_spawn_skip_prob` field to `BalticBioParams`**

In `salmon_ibm/baltic_params.py`, locate the `BalticBioParams` dataclass (around lines 41-110). Append the new field IMMEDIATELY AFTER the existing `activity_by_behavior` field (currently the last field on the class before `__post_init__`):

```python
    # Pre-spawn skip probability (C3.1). Bernoulli gate on reproducers
    # before Poisson clutch sampling; wild=0.0 (always spawns), hatchery
    # may divert via hatchery_overrides.pre_spawn_skip_prob.
    # Empirical anchor: Bouchard et al. 2022 (doi:10.1111/eva.13374)
    # — captive-bred Atlantic salmon RRS 0.65-0.80, "fewer mating events,
    # not smaller clutches" → skip-rate model intervention is the
    # right shape (matches Bouchard's mechanistic finding).
    pre_spawn_skip_prob: float = 0.0
```

- [ ] **Step 1.5: Extend `BalticBioParams.__post_init__` with range validation**

In the same file, locate `__post_init__` (the method that contains the existing `activity_by_behavior` validation block from C2). Append at the END of the method body:

```python
        if not (0.0 <= self.pre_spawn_skip_prob <= 1.0):
            raise ValueError(
                f"pre_spawn_skip_prob must be in [0, 1], got "
                f"{self.pre_spawn_skip_prob!r}"
            )
```

- [ ] **Step 1.6: Run the test; expect pass**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_c3_spawn.py::test_pre_spawn_skip_prob_rejects_out_of_range -v
```

Expected: 1 passed.

- [ ] **Step 1.7: Run baltic_params regression check**

```bash
micromamba run -n shiny python -m pytest tests/test_baltic_params.py tests/test_hatchery_params.py -v
```

Expected: all green. The default value `0.0` is in range, so existing construction paths are unaffected.

- [ ] **Step 1.8: Commit Task 1**

```bash
git add salmon_ibm/baltic_params.py tests/test_hatchery_c3_spawn.py
git commit -m "feat(hatchery-c3.1): pre_spawn_skip_prob field + validation (Task 1)

New BalticBioParams field 'pre_spawn_skip_prob: float = 0.0' for
Bernoulli pre-spawn gate. Wild default 0.0 (always spawn). Hatchery
overrides via species YAML hatchery_overrides.pre_spawn_skip_prob.

__post_init__ extended with range validation (0.0 <= p <= 1.0); raises
ValueError on out-of-range. Required because dataclasses.replace re-runs
__post_init__ when applying hatchery overrides (Task 2).

Test: test_pre_spawn_skip_prob_rejects_out_of_range covers both
negative (-0.1) and above-1 (1.5) boundary cases.

Spec: docs/superpowers/specs/2026-05-02-hatchery-c3-spawn-design.md"
```

---

### Task 2: Extend `_apply_hatchery_overrides` for scalar fields (TDD)

**Files:**
- Modify: `salmon_ibm/baltic_params.py` (extend `_apply_hatchery_overrides` with `SCALAR_OVERRIDE_FIELDS` set + `dataclasses.replace(..., **scalar_kwargs)` mechanism)
- Modify: `tests/test_hatchery_c3_spawn.py` (append 2 tests: loader happy path with identity check + extended-set-still-rejects-unknown)

The C2 `_apply_hatchery_overrides` only handles the nested-dict `activity_by_behavior` field. C3.1 introduces the first SCALAR override field, so the loader must support both override semantics: shallow-merge (existing) for dicts, full replacement (new) for scalars. The existing `ALLOWED_OVERRIDE_KEYS` set extends to include the new scalar field.

- [ ] **Step 2.1: Append 2 failing tests**

Append to `tests/test_hatchery_c3_spawn.py`:

```python
def _yaml_with_hatchery_skip(tmp_path, p_skip: float = 0.3) -> str:
    """Helper: write a YAML body with a hatchery pre_spawn_skip_prob
    override and return the path. Includes the C2 activity_by_behavior
    block so the loader's existing path is exercised too."""
    p = tmp_path / "species.yaml"
    p.write_text(f"""
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior:
      0: 1.0
      1: 1.2
      2: 0.8
      3: 1.5
      4: 1.0
    pre_spawn_skip_prob: 0.0
    hatchery_overrides:
      activity_by_behavior:
        1: 1.5
        3: 1.875
      pre_spawn_skip_prob: {p_skip}
""")
    return str(p)


def test_pre_spawn_skip_prob_loads_from_yaml(tmp_path):
    """YAML hatchery_overrides.pre_spawn_skip_prob is applied on top of
    wild via dataclasses.replace; identity check ensures wild instance
    is not mutated. Verifies both the new scalar field flows through
    and the existing C2 activity_by_behavior dict-merge still works
    when both override types are present in the same YAML."""
    from salmon_ibm.baltic_params import (
        load_baltic_species_config, BalticBioParams, BalticSpeciesConfig,
    )
    path = _yaml_with_hatchery_skip(tmp_path, p_skip=0.3)
    cfg = load_baltic_species_config(path)
    assert isinstance(cfg, BalticSpeciesConfig)
    assert isinstance(cfg.wild, BalticBioParams)
    assert isinstance(cfg.hatchery, BalticBioParams)
    # Wild value preserved (0.0 from YAML)
    assert cfg.wild.pre_spawn_skip_prob == 0.0
    # Hatchery value overridden (0.3 from hatchery_overrides)
    assert cfg.hatchery.pre_spawn_skip_prob == 0.3
    # C2 activity_by_behavior shallow-merge still works
    assert cfg.wild.activity_by_behavior == {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    assert cfg.hatchery.activity_by_behavior == {0: 1.0, 1: 1.5, 2: 0.8, 3: 1.875, 4: 1.0}
    # Identity check: hatchery is a NEW instance, not wild mutated
    assert cfg.hatchery is not cfg.wild


def test_extended_overrides_still_reject_unknown_keys(tmp_path):
    """C3.1 extended ALLOWED_OVERRIDE_KEYS to include
    pre_spawn_skip_prob. Verify the strict-loader contract still
    holds: unknown keys (typos, unsupported fields) raise ValueError.
    Locks against a regression where extending the set accidentally
    relaxes the strict check."""
    from salmon_ibm.baltic_params import load_baltic_species_config
    p = tmp_path / "species.yaml"
    p.write_text("""
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    hatchery_overrides:
      unknown_field: 1.0
""")
    with pytest.raises(ValueError, match="hatchery_overrides supports only"):
        load_baltic_species_config(str(p))
```

- [ ] **Step 2.2: Run the new tests; expect 2 failures**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_c3_spawn.py -v
```

Expected: 1 passed (Task 1's validation test) + 2 failed:
- `test_pre_spawn_skip_prob_loads_from_yaml`: ValueError "hatchery_overrides supports only ['activity_by_behavior']" — `pre_spawn_skip_prob` is not yet in `ALLOWED_OVERRIDE_KEYS`.
- `test_extended_overrides_still_reject_unknown_keys`: this might already pass since the existing strict loader rejects all non-`activity_by_behavior` keys. Verify after Step 2.3.

- [ ] **Step 2.3: Extend `_apply_hatchery_overrides` for scalar fields**

In `salmon_ibm/baltic_params.py`, locate `_apply_hatchery_overrides` (added in C2). The current function structure:

```python
def _apply_hatchery_overrides(
    wild_params: BalticBioParams,
    overrides: dict,
) -> BalticBioParams:
    """..."""
    from salmon_ibm.agents import Behavior
    ALLOWED_OVERRIDE_KEYS = {"activity_by_behavior"}
    VALID_BEHAVIORS = {int(b) for b in Behavior}

    unknown = set(overrides) - ALLOWED_OVERRIDE_KEYS
    if unknown:
        raise ValueError(...)

    activity_overrides_raw = overrides.get("activity_by_behavior", {})
    try:
        activity_overrides = {
            int(k): float(v) for k, v in activity_overrides_raw.items()
        }
    except (ValueError, TypeError) as exc:
        raise ValueError(...) from exc

    invalid_keys = set(activity_overrides) - VALID_BEHAVIORS
    if invalid_keys:
        raise ValueError(...)

    merged_dict = {**wild_params.activity_by_behavior, **activity_overrides}
    return dataclasses.replace(wild_params, activity_by_behavior=merged_dict)
```

Modify the function as follows. Two changes: (1) extend `ALLOWED_OVERRIDE_KEYS`; (2) collect scalar overrides into a `**scalar_kwargs` dict that flows into `dataclasses.replace`:

```python
def _apply_hatchery_overrides(
    wild_params: BalticBioParams,
    overrides: dict,
) -> BalticBioParams:
    """..."""
    from salmon_ibm.agents import Behavior
    ALLOWED_OVERRIDE_KEYS = {"activity_by_behavior", "pre_spawn_skip_prob"}
    SCALAR_OVERRIDE_FIELDS = {"pre_spawn_skip_prob"}
    VALID_BEHAVIORS = {int(b) for b in Behavior}

    unknown = set(overrides) - ALLOWED_OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            f"hatchery_overrides supports only "
            f"{sorted(ALLOWED_OVERRIDE_KEYS)} in C3.1; unsupported keys: "
            f"{sorted(unknown)}"
        )

    activity_overrides_raw = overrides.get("activity_by_behavior", {})
    try:
        activity_overrides = {
            int(k): float(v) for k, v in activity_overrides_raw.items()
        }
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"hatchery_overrides.activity_by_behavior keys must be integers "
            f"(Behavior enum values 0-4); got non-integer key in "
            f"{activity_overrides_raw!r}"
        ) from exc

    invalid_keys = set(activity_overrides) - VALID_BEHAVIORS
    if invalid_keys:
        raise ValueError(
            f"hatchery_overrides.activity_by_behavior keys must be valid "
            f"Behavior enum values {sorted(VALID_BEHAVIORS)}; got invalid: "
            f"{sorted(invalid_keys)}"
        )

    # Shallow-merge over wild base for activity_by_behavior dict
    merged_dict = {**wild_params.activity_by_behavior, **activity_overrides}

    # Scalar field overrides (C3.1+): full replacement, no merge needed.
    # Add new C3.x scalar fields to SCALAR_OVERRIDE_FIELDS as they ship.
    scalar_kwargs = {
        k: v for k, v in overrides.items() if k in SCALAR_OVERRIDE_FIELDS
    }

    # dataclasses.replace re-runs __post_init__ validation on the new
    # instance, catching out-of-range scalar values (e.g. negative skip
    # probability) at load time.
    return dataclasses.replace(
        wild_params,
        activity_by_behavior=merged_dict,
        **scalar_kwargs,
    )
```

(The error-message string for unknown keys updates from "C2" to "C3.1" to keep the diagnostic accurate as the allowed-set evolves.)

- [ ] **Step 2.4: Run the 2 new tests; expect both pass**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_c3_spawn.py -v
```

Expected: 3 passed (1 from Task 1 + 2 from this task).

- [ ] **Step 2.5: Run the full hatchery test files for regression**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py tests/test_hatchery_c3_spawn.py tests/test_baltic_params.py -v
```

Expected: green. The C2 tests that exercise `_apply_hatchery_overrides` (test 2 / `test_hatchery_overrides_unsupported_key_raises`) still pass because `T_OPT` remains outside `ALLOWED_OVERRIDE_KEYS`.

- [ ] **Step 2.6: Commit Task 2**

```bash
git add salmon_ibm/baltic_params.py tests/test_hatchery_c3_spawn.py
git commit -m "feat(hatchery-c3.1): scalar override mechanism in loader (Task 2)

_apply_hatchery_overrides() extended to support scalar override fields
alongside the existing activity_by_behavior dict-merge:

- ALLOWED_OVERRIDE_KEYS extended to {'activity_by_behavior',
  'pre_spawn_skip_prob'}
- New SCALAR_OVERRIDE_FIELDS = {'pre_spawn_skip_prob'} set; future
  C3.x scalar fields just add to this set
- scalar_kwargs collected from overrides, applied via
  dataclasses.replace(..., **scalar_kwargs) — re-runs __post_init__
  validation (Task 1's range check) on the new instance
- Unknown-key error message updated from 'C2' to 'C3.1' to keep the
  diagnostic accurate as the allowed-set evolves

Tests 1 + 6 of 6: happy-path merge with identity check (cfg.hatchery
is not cfg.wild); strict loader still rejects unknown keys after
ALLOWED_OVERRIDE_KEYS expansion.

Spec: docs/superpowers/specs/2026-05-02-hatchery-c3-spawn-design.md"
```

---

### Task 3: `ReproductionEvent.execute` dispatch + YAML provenance (TDD)

**Files:**
- Modify: `salmon_ibm/events_builtin.py` (top import gains `ORIGIN_HATCHERY`; `ReproductionEvent.execute` gains the Bernoulli skip filter)
- Modify: `configs/baltic_salmon_species.yaml` (`pre_spawn_skip_prob: 0.0` at wild + `pre_spawn_skip_prob: 0.3` in `hatchery_overrides` + provenance comments)
- Modify: `tests/test_hatchery_c3_spawn.py` (append 3 tests: dispatch at p=1.0, dispatch at p=0.0, graceful without hatchery_dispatch)

This is the user-visible behavioural change. The Bernoulli skip filter sits between `reproducer_idx` computation and Poisson clutch sampling. Gated by `hatchery_dispatch is not None and pre_spawn_skip_prob > 0` so pre-C3.1 scenarios bypass the new path entirely.

- [ ] **Step 3.1: Append 3 failing tests**

Append to `tests/test_hatchery_c3_spawn.py`:

```python
def test_reproduction_skips_hatchery_at_p_skip_one():
    """Set p_skip=1.0 → ALL hatchery reproducers skip; wild always
    proceed. Deterministic test — no RNG dependence on outcome.
    Verifies the Bernoulli dispatch correctly reads parent origin."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.baltic_params import BalticBioParams, HatcheryDispatch
    from salmon_ibm.events_builtin import ReproductionEvent
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY

    pool = AgentPool(n=4, start_tri=0, rng_seed=42)
    pool.origin[:2] = ORIGIN_WILD       # 2 wild
    pool.origin[2:] = ORIGIN_HATCHERY   # 2 hatchery
    pop = Population(name="test", pool=pool)
    pop.group_id = np.array([0, 0, 0, 0], dtype=np.int32)  # all in same group

    hd = HatcheryDispatch(
        params=BalticBioParams(pre_spawn_skip_prob=1.0),
        activity_lut=np.ones(5),  # unused for reproduction test
    )
    landscape = {
        "rng": np.random.default_rng(0),
        "hatchery_dispatch": hd,
    }
    evt = ReproductionEvent(name="r", clutch_mean=10.0, min_group_size=1)
    n_before = pool.n
    evt.execute(pop, landscape, t=0, mask=np.ones(4, dtype=bool))
    n_offspring = pool.n - n_before
    # Only the 2 wild reproducers spawned (clutch_mean=10 each ≈ 20 expected).
    # No offspring from hatchery parents (skipped at p=1.0).
    # Cannot easily check parent origin per-offspring; instead check count
    # is in the wild-only-Poisson range, not the all-4-Poisson range.
    # 2 reproducers × Poisson(10) → mean 20, 99% CI ~10-30
    # 4 reproducers × Poisson(10) → mean 40, 99% CI ~26-54
    # Window 25 is safely below the 4-reproducer lower bound.
    assert n_offspring < 30, (
        f"Expected wild-only Poisson(10) × 2 ≈ 20 offspring; got {n_offspring}. "
        f"Hatchery skip not applied?"
    )


def test_reproduction_no_skip_at_p_zero():
    """When pre_spawn_skip_prob=0.0, hatchery reproducers behave
    identically to wild. Lock-in for the explicit 0-value path
    (covers sensitivity-sweep null point and confirms the
    `pre_spawn_skip_prob > 0` guard short-circuits)."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.baltic_params import BalticBioParams, HatcheryDispatch
    from salmon_ibm.events_builtin import ReproductionEvent
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY

    pool = AgentPool(n=4, start_tri=0, rng_seed=42)
    pool.origin[:2] = ORIGIN_WILD
    pool.origin[2:] = ORIGIN_HATCHERY
    pop = Population(name="test", pool=pool)
    pop.group_id = np.array([0, 0, 0, 0], dtype=np.int32)

    # p=0.0 → no skipping; all 4 reproducers proceed
    hd = HatcheryDispatch(
        params=BalticBioParams(pre_spawn_skip_prob=0.0),
        activity_lut=np.ones(5),
    )
    landscape = {
        "rng": np.random.default_rng(0),
        "hatchery_dispatch": hd,
    }
    evt = ReproductionEvent(name="r", clutch_mean=10.0, min_group_size=1)
    n_before = pool.n
    evt.execute(pop, landscape, t=0, mask=np.ones(4, dtype=bool))
    n_offspring = pool.n - n_before
    # All 4 reproducers spawned → Poisson(10) × 4 ≈ 40, 99% CI ~26-54.
    # Verify count is in the 4-reproducer range, not the 2-reproducer range.
    assert n_offspring >= 25, (
        f"Expected all-reproducers Poisson(10) × 4 ≈ 40 offspring; got "
        f"{n_offspring}. Skip applied at p=0?"
    )


def test_reproduction_graceful_without_hatchery_dispatch():
    """When landscape has no 'hatchery_dispatch' key (pre-C3.1 scenarios
    or wild-only configs), ReproductionEvent.execute proceeds without
    ANY skip logic. Hatchery-tagged agents (if any) reproduce as
    normal. Locks the graceful-fallback semantics; no regression on
    pre-C2 reproductive scenarios."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.events_builtin import ReproductionEvent
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY

    pool = AgentPool(n=4, start_tri=0, rng_seed=42)
    pool.origin[:2] = ORIGIN_WILD
    pool.origin[2:] = ORIGIN_HATCHERY
    pop = Population(name="test", pool=pool)
    pop.group_id = np.array([0, 0, 0, 0], dtype=np.int32)

    # Landscape with NO hatchery_dispatch key — graceful fallback
    landscape = {"rng": np.random.default_rng(0)}
    evt = ReproductionEvent(name="r", clutch_mean=10.0, min_group_size=1)
    n_before = pool.n
    evt.execute(pop, landscape, t=0, mask=np.ones(4, dtype=bool))
    n_offspring = pool.n - n_before
    # No skip path executes; all 4 reproducers spawn → Poisson(10) × 4 ≈ 40.
    assert n_offspring >= 25, (
        f"Expected all-reproducers Poisson(10) × 4 ≈ 40 offspring; got "
        f"{n_offspring}. Skip applied without hatchery_dispatch key?"
    )
```

- [ ] **Step 3.2: Run the 3 new tests; expect 3 failures**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_c3_spawn.py -v
```

Expected: 3 passed (Tasks 1 + 2) + 3 failed:
- `test_reproduction_skips_hatchery_at_p_skip_one`: skip filter doesn't exist yet → all 4 reproducers fire → ~40 offspring (>30 trips assertion).
- `test_reproduction_no_skip_at_p_zero`: should already pass (no skip behaviour expected at p=0); verify after Step 3.3.
- `test_reproduction_graceful_without_hatchery_dispatch`: should already pass; verify after Step 3.3.

(Tests 4 and 5 may pass even before the implementation if the existing logic already produces full-clutch counts. The important verification is test 3 fails until Step 3.3 lands.)

- [ ] **Step 3.3: Update top import in `events_builtin.py`**

In `salmon_ibm/events_builtin.py`, locate the existing `from salmon_ibm.origin import ORIGIN_WILD` import (added in C1, around the top of the file). Update to also import `ORIGIN_HATCHERY`:

```python
from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY
```

- [ ] **Step 3.4: Insert the Bernoulli skip filter in `ReproductionEvent.execute`**

In `salmon_ibm/events_builtin.py`, locate `ReproductionEvent.execute` (the class starts at line 291; method body starts around line 302). Find the existing line:

```python
        reproducer_idx = np.where(can_reproduce)[0]
```

(this is around line 314 in the C2 branch state). Immediately AFTER this line and BEFORE the existing `if len(reproducer_idx) == 0: return`, insert the C3.1 skip filter:

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

(The existing `if len(reproducer_idx) == 0: return` early-return on the line below now correctly handles the case where all hatchery reproducers were skipped and no wild remain.)

- [ ] **Step 3.5: Run the 3 new tests; expect all pass**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_c3_spawn.py -v
```

Expected: 6 passed (3 from Tasks 1+2 + 3 from this task).

- [ ] **Step 3.6: Update `configs/baltic_salmon_species.yaml`**

Open `configs/baltic_salmon_species.yaml`. The current state (post-C2) has:
- A wild-level `activity_by_behavior:` block with provenance comments above
- A `hatchery_overrides:` block containing only `activity_by_behavior:` overrides

C3.1 adds two things:
1. `pre_spawn_skip_prob: 0.0` at the wild level (alongside other scalar fields like `cmax_A`, `T_OPT`)
2. `pre_spawn_skip_prob: 0.3` inside `hatchery_overrides:` AFTER the existing `activity_by_behavior:` block, with a full provenance comment block above

**Keep ALL existing fields and comments unchanged.** Only add the new lines.

The wild-level addition (find a sensible spot after the existing `activity_by_behavior:` block, around the same indentation level as other scalar fields):

```yaml
    # C3.1: wild reproducers always proceed (skip probability 0.0)
    pre_spawn_skip_prob: 0.0
```

The `hatchery_overrides:` addition (after the existing `activity_by_behavior:` block, at the same indentation level):

```yaml
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

- [ ] **Step 3.7: Verify the YAML loads correctly**

```bash
micromamba run -n shiny python -c "
from salmon_ibm.baltic_params import load_baltic_species_config
cfg = load_baltic_species_config('configs/baltic_salmon_species.yaml')
print('wild p_skip:', cfg.wild.pre_spawn_skip_prob)
print('hatch p_skip:', cfg.hatchery.pre_spawn_skip_prob if cfg.hatchery else None)
print('wild activity:', cfg.wild.activity_by_behavior)
print('hatch activity:', cfg.hatchery.activity_by_behavior if cfg.hatchery else None)
"
```

Expected output:
```
wild p_skip: 0.0
hatch p_skip: 0.3
wild activity: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
hatch activity: {0: 1.0, 1: 1.5, 2: 0.8, 3: 1.875, 4: 1.0}
```

If the YAML fails to load, double-check indentation (YAML is whitespace-sensitive). The wild-level `pre_spawn_skip_prob: 0.0` should be at the same indent as `cmax_A`, `T_OPT`, etc.; the hatchery-overrides `pre_spawn_skip_prob: 0.3` should be at the same indent as `activity_by_behavior:` inside `hatchery_overrides:`.

- [ ] **Step 3.8: Run wider regression check**

```bash
micromamba run -n shiny python -m pytest tests/test_events.py tests/test_hatchery_params.py tests/test_hatchery_c3_spawn.py -v
```

Expected: green. Existing reproduction tests in `test_events.py` construct WILD-only populations or no `hatchery_dispatch` landscape → bypass the new code path entirely.

- [ ] **Step 3.9: Commit Task 3**

```bash
git add salmon_ibm/events_builtin.py configs/baltic_salmon_species.yaml tests/test_hatchery_c3_spawn.py
git commit -m "feat(hatchery-c3.1): ReproductionEvent dispatch + YAML (Task 3)

ReproductionEvent.execute gains a Bernoulli skip filter inserted
between reproducer_idx computation (line 314) and the existing
empty-check / Poisson clutch sampling. Filter is gated by
landscape.get('hatchery_dispatch') is not None AND
hd.params.pre_spawn_skip_prob > 0 — pre-C3.1 scenarios bypass the
new code path entirely.

Top import extended: 'from salmon_ibm.origin import ORIGIN_WILD,
ORIGIN_HATCHERY' (was only ORIGIN_WILD from C1).

configs/baltic_salmon_species.yaml gains pre_spawn_skip_prob: 0.0
at wild level (explicit baseline) and pre_spawn_skip_prob: 0.3 in
hatchery_overrides with full provenance comment block citing
Bouchard 2022 (primary, mechanistic shape), Christie 2014 (cross-
species meta-analytic baseline), Jonsson 2019 (population-scale).

Tests 3, 4, 5 of 6 lock in:
- Skip-all at p=1.0 (deterministic, only wild reproducers spawn)
- No-skip at p=0.0 (sensitivity-sweep null point)
- Graceful fallback when landscape has no hatchery_dispatch key

Spec: docs/superpowers/specs/2026-05-02-hatchery-c3-spawn-design.md"
```

---

### Task 4: Full pytest + plan stamp + final commit + push

**Files:**
- Modify: `docs/superpowers/plans/2026-05-02-hatchery-c3-spawn.md` (add ✅ EXECUTED stamp)

Run the whole suite to surface regressions; if green, stamp and push.

- [ ] **Step 4.1: Run full pytest suite**

```bash
micromamba run -n shiny python -m pytest tests/ -q --no-header
```

Expected (~14-26 minutes): `848 passed, 34 skipped, 7 deselected, 1 xfailed`. Zero failures.

- [ ] **Step 4.2: If failures, triage**

Most likely failure modes:
- **(A) Pre-existing reproduction test that constructs hatchery agents** → would not happen since C2's runtime guards forbid HATCHERY agents without `hatchery_dispatch`. Confirm C2 behavior unchanged.
- **(B) Test fixture using `BalticBioParams(...)` with explicit unrecognized kwargs** → the new `pre_spawn_skip_prob` field has a default, so existing call sites are unaffected.
- **(C) Performance-flaky test (e.g., `test_full_step_time_within_one_percent_of_baseline`)** → a flaky timing benchmark, not a real regression. Re-run that specific test to confirm.
- **(D) Unexpected regression in unrelated test** → should not happen since C3.1 is gated by `hatchery_dispatch is not None` and `pre_spawn_skip_prob > 0`.

Re-run the suite after each fix; commit fixes as separate small commits with `fix(hatchery-c3.1): ...` messages.

- [ ] **Step 4.3: Stamp the plan as ✅ EXECUTED**

In `docs/superpowers/plans/2026-05-02-hatchery-c3-spawn.md`, replace the line:

```markdown
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
```

With:

```markdown
> **STATUS: ✅ EXECUTED 2026-MM-DD** — All 4 tasks complete. salmon_ibm/baltic_params.py gains pre_spawn_skip_prob field + range validation + scalar override mechanism in _apply_hatchery_overrides; salmon_ibm/events_builtin.py inserts Bernoulli skip filter in ReproductionEvent.execute (gated by hatchery_dispatch presence + p>0); configs/baltic_salmon_species.yaml gains pre_spawn_skip_prob fields + provenance block. Full pytest suite green at NNN passing (842 baseline + 6 new C3.1 tests). Spec at docs/superpowers/specs/2026-05-02-hatchery-c3-spawn-design.md. Closes the first sub-tier of C3 (third in the C1 → C2 → C3 sequence); next: C3.2 sea-age sampling.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
```

Replace `2026-MM-DD` with today's date and `NNN` with the verified passing count from Step 4.1.

- [ ] **Step 4.4: Final commit**

**Do NOT create this commit unless Step 4.1 reports zero failures.** A red commit poisons the branch history and complicates revert.

```bash
git add docs/superpowers/plans/2026-05-02-hatchery-c3-spawn.md
git commit -m "docs(plan): stamp hatchery-c3.1-spawn plan as EXECUTED (Task 4)

All 4 tasks complete. Suite at NNN passing (842 baseline + 6 new).
Closes the first sub-tier of C3 (third in the C1 -> C2 -> C3 sequence);
next plan will be C3.2 sea-age sampling."
```

(Update `NNN` to match the actual count.)

- [ ] **Step 4.5: Rebase onto main if branched from `hatchery-c2-bioparams`**

First check whether the branch needs rebasing:

```bash
git -C "$(pwd)" log --oneline origin/main..HEAD | head -20
```

If the output includes commits like `feat(hatchery-c2): ...` (Task 1-5 of C2), the branch was stacked on `hatchery-c2-bioparams`. Rebase onto main once C2 PR has merged:

```bash
git fetch origin main
git rebase origin/main
```

Resolve any conflicts (none expected — C3.1 doesn't overlap C2 except in `events_builtin.py` imports and `baltic_params.py` field declarations, both additive). Verify after rebase:

```bash
git log --oneline origin/main..HEAD
```

Expected: only the 4 C3.1 commits (Tasks 1-4), no C2 or C1 commits in the list.

If the branch was created from main directly (clean case), skip this step.

- [ ] **Step 4.6: Push the branch and open PR**

```bash
git push -u origin hatchery-c3-spawn
```

Then open a PR via `gh pr create`. Use this template (mirrors the C2 PR pattern):

```bash
gh pr create --title "Hatchery vs wild C3.1: pre-spawn skip probability" --body "$(cat <<'EOF'
## Summary

Tier C3.1 of hatchery-vs-wild distinction — first sub-task of C3 (third
in the C1 → C2 → C3 sequence). Hatchery reproducers skip pre-spawn with
Bernoulli probability \`p_skip = 0.3\` (RRS ≈ 0.7, mid-range of
Bouchard 2022 Atlantic salmon RRS); wild reproducers always proceed.

- New \`BalticBioParams.pre_spawn_skip_prob: float = 0.0\` field with
  range validation in \`__post_init__\`
- \`_apply_hatchery_overrides\` extended: \`ALLOWED_OVERRIDE_KEYS\` includes
  \`pre_spawn_skip_prob\`; new \`SCALAR_OVERRIDE_FIELDS\` set; scalar
  overrides flow through \`dataclasses.replace(..., **scalar_kwargs)\`
- \`ReproductionEvent.execute\` gains a Bernoulli skip filter inserted
  between \`reproducer_idx\` computation and Poisson clutch sampling.
  Gated by \`hatchery_dispatch is not None\` AND \`pre_spawn_skip_prob > 0\`
  → pre-C3.1 scenarios bypass the new code path entirely
- \`configs/baltic_salmon_species.yaml\` gains \`pre_spawn_skip_prob: 0.0\`
  at wild level + \`pre_spawn_skip_prob: 0.3\` in hatchery_overrides +
  provenance comment block citing Bouchard 2022 + Christie 2014 +
  Jonsson 2019

## Files changed (4 total)

| File | Touches |
|---|---|
| \`salmon_ibm/baltic_params.py\` | +pre_spawn_skip_prob field + __post_init__ validation + extended _apply_hatchery_overrides (ALLOWED_OVERRIDE_KEYS + SCALAR_OVERRIDE_FIELDS) |
| \`salmon_ibm/events_builtin.py\` | top import gains ORIGIN_HATCHERY; ReproductionEvent.execute Bernoulli skip filter |
| \`configs/baltic_salmon_species.yaml\` | pre_spawn_skip_prob fields + provenance comment block |
| \`tests/test_hatchery_c3_spawn.py\` | NEW — 6 tests |

## Test plan

- [x] 6 new tests in tests/test_hatchery_c3_spawn.py (validation, loader, dispatch p=1.0, dispatch p=0.0, graceful fallback, strict-loader extension)
- [x] Full pytest suite: **NNN passed** (was 842; +6 new; 0 regressions)
- [x] Calibration sensitivity sweep planned: {0.0, 0.15, 0.30, 0.50} (cap at 0.6)

## Spec & plan

- Spec: docs/superpowers/specs/2026-05-02-hatchery-c3-spawn-design.md
- Plan: docs/superpowers/plans/2026-05-02-hatchery-c3-spawn.md (stamped EXECUTED YYYY-MM-DD)

## Deferred

- **C3.2:** Sea-age sampling divergence (new sea_age field + sampling event)
- **C3.3:** Homing precision divergence (modifies migration / delta-routing logic)
- **Origin inheritance on reproduction:** offspring of hatchery parents stay tagged ORIGIN_WILD per C1 scope-OUT
- **Sex-specific skip rates:** Bouchard 2022 found males more affected (1SW males ~65% RRS vs MSW ~80%); current model doesn't track sex
- **Year-to-year RRS stochasticity:** single constant skip probability per scenario

## Backward compatibility

Fully backward-compatible. \`pre_spawn_skip_prob\` defaults to 0.0
everywhere; existing reproduction tests construct WILD-only
populations or no \`hatchery_dispatch\` landscape → bypass the new code
path entirely. Same opt-in semantics as C2.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Update `NNN` and `YYYY-MM-DD` in the body before running.

- [ ] **Step 4.7: Update memory after merge + deploy**

Once the PR is merged + tagged + deployed, update memory files:
- `~/.claude/projects/.../memory/curonian_h3_grid_state.md`: bump deployed version label, add a "vNEW — Hatchery vs wild C3.1" entry mirroring the C2 entry's structure.
- `~/.claude/projects/.../memory/curonian_deferred.md`: update item #7 (hatchery vs wild) to mark "C3.1 RESOLVED, C3.2 + C3.3 still queued".

---

## Plan summary

- **4 tasks**, **4 commits** (one per task; Task 4 is the stamp + final commit).
- **4 files modified** (1 new test file + 3 modified production files).
- **+6 net tests** (matches spec exactly).
- **Estimated time:** ~2 days. All work is mechanical (one new dataclass field, one extended loader function, one new dispatch site, one YAML edit).
- **Risk profile:** very low — `pre_spawn_skip_prob > 0` guard means pre-C3.1 scenarios continue with zero behaviour change; no new agent fields; no Numba kernel changes.
- **No backward-compat shim needed** — `pre_spawn_skip_prob` defaults to 0.0; YAMLs without it inherit the default.

## Spec coverage check

| Spec section | Implementing task |
|---|---|
| `BalticBioParams.pre_spawn_skip_prob` field | Task 1 |
| `BalticBioParams.__post_init__` range validation | Task 1 |
| `_apply_hatchery_overrides` extension (ALLOWED_OVERRIDE_KEYS + SCALAR_OVERRIDE_FIELDS) | Task 2 |
| `ReproductionEvent.execute` Bernoulli skip filter | Task 3 |
| `events_builtin.py` import update (ORIGIN_HATCHERY) | Task 3 |
| `configs/baltic_salmon_species.yaml` field additions + provenance | Task 3 |
| 6 new tests (test_hatchery_c3_spawn.py) | Tasks 1, 2, 3 |
| Plan stamp on completion | Task 4.3 |
