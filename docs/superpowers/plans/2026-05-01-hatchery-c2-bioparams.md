# Hatchery vs Wild C2 — `activity_by_behavior` Divergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hatchery agents (origin=1) pay +25% on RANDOM (key 1) and UPSTREAM (key 3) Wisconsin activity multipliers; wild (origin=0) unchanged. Single divergent field; dispatch hides behind existing per-agent activity_mult array. Closes the 2nd of 3 hatchery-vs-wild tiers.

**Architecture:** New `HatcheryDispatch` frozen dataclass bundling params + LUT atomically; new `origin_aware_activity_mult` helper in `bioenergetics.py` graceful on `lut_hatch=None`; `Simulation` holds optional `hatchery_dispatch`; both wild and hatchery LUTs derived from cached `BalticSpeciesConfig` (no per-slider disk I/O). YAML loader strict on unknown keys, validates Behavior enum sub-keys, rejects non-numeric and non-positive values.

**Tech Stack:** Python 3.10+, NumPy, dataclasses (frozen + NamedTuple), IntEnum, pytest; conda env `shiny`.

**Spec:** [`docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md`](../specs/2026-05-01-hatchery-c2-bioparams-design.md) (commits `7b8ff92` + `003f643` + `5e998cd` + `f6df326`).

---

## File structure

**Modified files (8 total: 1 new test file + 7 modified):**

Production code (7):
- `salmon_ibm/baltic_params.py` — NEW `HatcheryDispatch` dataclass + NEW `BalticSpeciesConfig` NamedTuple + extended `__post_init__` validation + NEW `_apply_hatchery_overrides()` + extended `load_baltic_species_config()` (return type changed)
- `salmon_ibm/bioenergetics.py` — NEW `origin_aware_activity_mult()` helper
- `salmon_ibm/config.py` — `load_bio_params_from_config()` returns `BalticSpeciesConfig` always
- `salmon_ibm/simulation.py` — `Landscape` TypedDict + 5 touch points (init branching, `_species_config` cache, `rebuild_luts()` method, `step()` landscape injection, `_event_bioenergetics` dispatch)
- `salmon_ibm/events_builtin.py` — `SurvivalEvent.execute` dispatch + `IntroductionEvent.execute` runtime guard
- `salmon_ibm/events_hexsim.py` — `PatchIntroductionEvent.execute` runtime guard
- `app.py` — sidebar block at lines 1493-1501 calls `sim.rebuild_luts()` instead of `sim._build_activity_lut()`

Tests (1 new file):
- `tests/test_hatchery_params.py` — 13 new tests

**Test runner:**
```bash
micromamba run -n shiny python -m pytest tests/path/file.py::test_name -v
# whole suite
micromamba run -n shiny python -m pytest tests/ -v
```

Suite is ~14-23 minutes. Baseline before this plan: **829 passing on `hatchery-origin-c1` branch (post-C1)**. Expected after: **842 passing** (829 + 13 new tests, 0 regressions).

**Commit cadence (6 commits):**
- Task 1 → commit 1 (types + post_init validation)
- Task 2 → commit 2 (override loader)
- Task 3 → commit 3 (unified config return + helper)
- Task 4 → commit 4 (Simulation init + rebuild_luts + step + dispatch)
- Task 5 → commit 5 (event dispatches + runtime guards + app.py)
- Task 6 → commit 6 (full pytest + plan stamp + push)

**Branch:** `hatchery-c2-bioparams` (created from `main` after PR #3 merges). If PR #3 still open at implementation start, branch from `hatchery-origin-c1` instead and rebase later.

---

## Tasks

### Task 1: New types + `__post_init__` validation in `baltic_params.py` (TDD)

**Files:**
- Create branch
- Modify: `salmon_ibm/baltic_params.py` (top imports + `BalticBioParams.__post_init__` + 2 new top-level types)
- Create: `tests/test_hatchery_params.py` (1 test for now: `test_activity_by_behavior_rejects_nonpositive_value`)

This task ships the `HatcheryDispatch` frozen dataclass and `BalticSpeciesConfig` NamedTuple as scaffolding for everything else, plus the `__post_init__` validation that catches non-positive activity values. No external behaviour change yet; subsequent tasks consume these types.

- [ ] **Step 1.1: Create the branch**

```bash
# If PR #3 (hatchery-origin-c1) has merged into main:
git switch main
git pull origin main
git checkout -b hatchery-c2-bioparams

# If PR #3 still open:
git switch hatchery-origin-c1
git checkout -b hatchery-c2-bioparams
```

- [ ] **Step 1.2: Create `tests/test_hatchery_params.py` with the failing validation test**

Create `tests/test_hatchery_params.py`:

```python
"""Tests for the C2 hatchery vs wild parameter divergence (Tier C2).

Spec: docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md
"""
import pytest


def test_activity_by_behavior_rejects_nonpositive_value():
    """BalticBioParams.__post_init__ rejects activity_by_behavior with
    non-positive values (e.g., 0.0 or -0.5). Locks in the C2 validation
    that the override merge depends on."""
    from salmon_ibm.baltic_params import BalticBioParams
    with pytest.raises(ValueError, match="positive floats"):
        BalticBioParams(activity_by_behavior={0: 1.0, 1: -0.5, 2: 0.8, 3: 1.5, 4: 1.0})
```

- [ ] **Step 1.3: Run the test; expect failure**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py::test_activity_by_behavior_rejects_nonpositive_value -v
```

Expected: FAILED — current `__post_init__` doesn't validate `activity_by_behavior`.

- [ ] **Step 1.4: Extend `BalticBioParams.__post_init__`**

In `salmon_ibm/baltic_params.py`, locate `__post_init__` (around lines 81-101). Append the following at the END of the method body:

```python
        if not self.activity_by_behavior:
            raise ValueError("activity_by_behavior must be non-empty")
        for k, v in self.activity_by_behavior.items():
            if not isinstance(k, int) or k < 0:
                raise ValueError(
                    f"activity_by_behavior keys must be non-negative ints, got {k!r}"
                )
            if not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(
                    f"activity_by_behavior values must be positive floats, "
                    f"got {k}: {v!r}"
                )
```

- [ ] **Step 1.5: Run the test; expect pass**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py::test_activity_by_behavior_rejects_nonpositive_value -v
```

Expected: 1 passed.

- [ ] **Step 1.6: Run `tests/test_baltic_params.py` to confirm no regression**

```bash
micromamba run -n shiny python -m pytest tests/test_baltic_params.py -v
```

Expected: all green. The default `activity_by_behavior` dict in `BalticBioParams` (`{0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}`) is all positive floats; existing tests pass.

- [ ] **Step 1.7: Add `HatcheryDispatch` frozen dataclass and `BalticSpeciesConfig` NamedTuple**

In `salmon_ibm/baltic_params.py`, modify the top imports to include:

```python
from typing import NamedTuple
import numpy as np
```

(`numpy` should already be imported transitively via the rest of the codebase, but add it directly for typing.)

Then, immediately AFTER the `BalticBioParams` class definition (after the closing of the class, around line 110), add:

```python
@dataclass(frozen=True)
class HatcheryDispatch:
    """Bundle hatchery params + their derived activity LUT atomically.

    Holding params and LUT separately as paired nullables on Simulation
    invites desync (one rebuilt, one stale). This bundle makes the
    invariant `params is None ↔ lut is None` structurally impossible
    to violate, and gives callers a single nullable to guard on
    (`if landscape.get('hatchery_dispatch') is None: ...`).
    """
    params: BalticBioParams
    activity_lut: np.ndarray


class BalticSpeciesConfig(NamedTuple):
    """Loaded species config — wild + optional hatchery override.

    Always returned by load_baltic_species_config(); legacy non-Baltic
    path returns BalticSpeciesConfig(wild=plain_BioParams, hatchery=None)
    so callers don't need isinstance branching.
    """
    wild: "BalticBioParams"  # forward-ref to avoid moving the class
    hatchery: BalticBioParams | None
```

(`from salmon_ibm.bioenergetics import BioParams` may also need to be imported if `wild` is later widened to accept plain `BioParams`. For this task we keep the type as `BalticBioParams`; widening happens in Task 3.)

- [ ] **Step 1.8: Run baltic_params tests to confirm types load cleanly**

```bash
micromamba run -n shiny python -c "from salmon_ibm.baltic_params import HatcheryDispatch, BalticSpeciesConfig, BalticBioParams; print('ok')"
```

Expected: `ok` (no ImportError or syntax error).

- [ ] **Step 1.9: Commit Task 1**

```bash
git add salmon_ibm/baltic_params.py tests/test_hatchery_params.py
git commit -m "feat(hatchery-c2): types scaffold + activity validation (Task 1)

New HatcheryDispatch frozen dataclass bundles hatchery params + their
derived activity LUT atomically — invariant 'params is None iff lut
is None' becomes structurally impossible to violate. New
BalticSpeciesConfig NamedTuple wraps the (wild, hatchery_or_None)
loader return.

BalticBioParams.__post_init__ extended to validate
activity_by_behavior is non-empty, keys are non-negative ints, values
are positive floats. Required for the override-merge path that ships
in Task 2.

Test 10 of 13 (test_activity_by_behavior_rejects_nonpositive_value)
locks in the validation contract.

Spec: docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md"
```

---

### Task 2: `_apply_hatchery_overrides` + extended `load_baltic_species_config` (TDD)

**Files:**
- Modify: `salmon_ibm/baltic_params.py` (extend `load_baltic_species_config` + new `_apply_hatchery_overrides`)
- Modify: `tests/test_hatchery_params.py` (append 5 tests: 1, 2, 3, 8, 9)

The loader takes the current path (returns `BalticBioParams`) and extends it to optionally apply `hatchery_overrides:`. Strict on unknown top-level override keys, strict on invalid Behavior sub-keys, strict on non-numeric sub-keys.

- [ ] **Step 2.1: Append 5 failing tests**

Append to `tests/test_hatchery_params.py`:

```python
def _write_yaml(tmp_path, body: str) -> str:
    """Helper: write a YAML body to tmp_path/species.yaml and return the path."""
    p = tmp_path / "species.yaml"
    p.write_text(body)
    return str(p)


def test_hatchery_overrides_activity_by_behavior_loads(tmp_path):
    """Happy path: hatchery_overrides.activity_by_behavior is shallow-merged
    over the wild base, producing a BalticSpeciesConfig with both wild
    and hatchery objects. Identity check ensures no in-place mutation
    of the wild dict."""
    from salmon_ibm.baltic_params import (
        load_baltic_species_config, BalticBioParams, BalticSpeciesConfig,
    )
    yaml_body = """
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior:
      0: 1.0
      1: 1.2
      2: 0.8
      3: 1.5
      4: 1.0
    hatchery_overrides:
      activity_by_behavior:
        1: 1.5
        3: 1.875
"""
    path = _write_yaml(tmp_path, yaml_body)
    cfg = load_baltic_species_config(path)
    assert isinstance(cfg, BalticSpeciesConfig)
    assert isinstance(cfg.wild, BalticBioParams)
    assert isinstance(cfg.hatchery, BalticBioParams)
    # Wild dict unchanged
    assert cfg.wild.activity_by_behavior == {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    # Hatchery dict shallow-merged
    assert cfg.hatchery.activity_by_behavior == {0: 1.0, 1: 1.5, 2: 0.8, 3: 1.875, 4: 1.0}
    # Identity check: hatchery is a NEW instance, not wild mutated
    assert cfg.hatchery is not cfg.wild


def test_hatchery_overrides_unsupported_key_raises(tmp_path):
    """Strict loader: hatchery_overrides containing fields other than
    activity_by_behavior raises ValueError. C2 only supports
    activity_by_behavior; other dataclass fields would be biologically
    inert under C2's dispatch."""
    from salmon_ibm.baltic_params import load_baltic_species_config
    yaml_body = """
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    hatchery_overrides:
      T_OPT: 14.0
"""
    path = _write_yaml(tmp_path, yaml_body)
    with pytest.raises(ValueError, match="hatchery_overrides supports only"):
        load_baltic_species_config(path)


def test_hatchery_overrides_typo_raises(tmp_path):
    """Strict loader: typo in top-level override key (e.g.
    'activity_for_behavior' instead of 'activity_by_behavior') raises."""
    from salmon_ibm.baltic_params import load_baltic_species_config
    yaml_body = """
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    hatchery_overrides:
      activity_for_behavior:
        1: 1.5
"""
    path = _write_yaml(tmp_path, yaml_body)
    with pytest.raises(ValueError, match="hatchery_overrides supports only"):
        load_baltic_species_config(path)


def test_hatchery_overrides_invalid_behavior_key_raises(tmp_path):
    """Strict loader: hatchery_overrides.activity_by_behavior keys must
    be valid Behavior enum values (0-4). A typo like '999' raises;
    without this check, the LUT would silently grow to 1000 elements
    and the override would have zero behavioural effect."""
    from salmon_ibm.baltic_params import load_baltic_species_config
    yaml_body = """
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    hatchery_overrides:
      activity_by_behavior:
        999: 5.0
"""
    path = _write_yaml(tmp_path, yaml_body)
    with pytest.raises(ValueError, match="valid Behavior enum values"):
        load_baltic_species_config(path)


def test_hatchery_overrides_nonnumeric_behavior_key_raises(tmp_path):
    """Strict loader: hatchery_overrides.activity_by_behavior keys must
    be coercible to int. A YAML key like 'hold' (likely a user typo
    intending HOLD=0) raises with an actionable error message."""
    from salmon_ibm.baltic_params import load_baltic_species_config
    yaml_body = """
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    hatchery_overrides:
      activity_by_behavior:
        hold: 1.5
"""
    path = _write_yaml(tmp_path, yaml_body)
    with pytest.raises(ValueError, match="keys must be integers"):
        load_baltic_species_config(path)
```

- [ ] **Step 2.2: Run the new tests; expect 5 failures**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py -v
```

Expected: 5 failed (the validation logic doesn't exist yet) + 1 passed (Task 1's test 10).

- [ ] **Step 2.3: Add `_apply_hatchery_overrides` to `baltic_params.py`**

In `salmon_ibm/baltic_params.py`, add to top imports:

```python
import dataclasses
```

(or alongside the existing `from dataclasses import dataclass, field`).

Then, AFTER the `BalticSpeciesConfig` NamedTuple (after Task 1's additions), add:

```python
def _apply_hatchery_overrides(
    wild_params: BalticBioParams,
    overrides: dict,
) -> BalticBioParams:
    """Build a hatchery BalticBioParams by overlaying overrides on wild.

    For C2, the only allowed override key is `activity_by_behavior`.
    Sub-keys must be valid Behavior enum values (0-4). Non-numeric or
    out-of-range sub-keys raise ValueError at load time.

    See docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md.
    """
    from salmon_ibm.agents import Behavior  # local import to avoid cycle

    ALLOWED_OVERRIDE_KEYS = {"activity_by_behavior"}
    VALID_BEHAVIORS = {int(b) for b in Behavior}  # {0, 1, 2, 3, 4}

    unknown = set(overrides) - ALLOWED_OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            f"hatchery_overrides supports only "
            f"{sorted(ALLOWED_OVERRIDE_KEYS)} in C2; unsupported keys: "
            f"{sorted(unknown)}"
        )

    activity_overrides_raw = overrides.get("activity_by_behavior", {})
    # Coerce YAML string keys to int (PyYAML may emit '1' rather than 1).
    # Wrap in try/except so non-numeric keys produce an actionable message
    # rather than a bare 'invalid literal for int()' traceback.
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

    # Shallow-merge over wild base: missing keys keep wild values.
    merged_dict = {**wild_params.activity_by_behavior, **activity_overrides}
    # dataclasses.replace re-runs __post_init__ validation on the merged
    # instance, catching e.g. negative override values.
    return dataclasses.replace(wild_params, activity_by_behavior=merged_dict)
```

- [ ] **Step 2.4: Extend `load_baltic_species_config` to return `BalticSpeciesConfig`**

In `salmon_ibm/baltic_params.py`, modify `load_baltic_species_config` (currently at lines 113-137) to return `BalticSpeciesConfig` with optional hatchery:

```python
def load_baltic_species_config(path: str | Path) -> BalticSpeciesConfig:
    """Load the canonical baltic_salmon_species.yaml into BalticSpeciesConfig.

    Always returns a BalticSpeciesConfig NamedTuple. If the YAML's
    species.BalticAtlanticSalmon block contains a hatchery_overrides:
    sub-block, .hatchery is populated; otherwise .hatchery is None.

    YAML schema (minimum):

        species:
          BalticAtlanticSalmon:
            cmax_A: 0.303
            activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
            # optional:
            hatchery_overrides:
              activity_by_behavior:
                1: 1.5
                3: 1.875

    Unknown top-level keys in BalticAtlanticSalmon are silently filtered
    (legacy tolerance). Unknown keys in hatchery_overrides RAISE
    ValueError (strict — typos there are scientifically dangerous).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("species", {}).get("BalticAtlanticSalmon")
    if block is None:
        raise ValueError(
            f"{path}: missing 'species.BalticAtlanticSalmon' block"
        )
    # Pop hatchery_overrides BEFORE the known-field filter so it doesn't
    # get silently dropped.
    hatchery_overrides = block.pop("hatchery_overrides", None)
    # Filter to fields BalticBioParams knows about; extra keys tolerated.
    known = {f.name for f in BalticBioParams.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in block.items() if k in known}
    wild = BalticBioParams(**kwargs)

    if hatchery_overrides is None:
        return BalticSpeciesConfig(wild=wild, hatchery=None)
    hatchery = _apply_hatchery_overrides(wild, hatchery_overrides)
    return BalticSpeciesConfig(wild=wild, hatchery=hatchery)
```

- [ ] **Step 2.5: Run the 5 new tests; expect all pass**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py -v
```

Expected: 6 passed (5 from this task + 1 from Task 1).

- [ ] **Step 2.6: Run baltic_params regression check**

```bash
micromamba run -n shiny python -m pytest tests/test_baltic_params.py -v
```

Expected: pre-existing tests still pass. The function signature change (return type `BalticBioParams` → `BalticSpeciesConfig`) breaks the existing tests at `tests/test_baltic_params.py:107,120` that do `params = load_baltic_species_config(path)` and access `.cmax_A` directly. **Fix those callers as part of this step:** update them to do `cfg = load_baltic_species_config(path); params = cfg.wild; assert params.cmax_A == ...`.

After fixing, re-run:

```bash
micromamba run -n shiny python -m pytest tests/test_baltic_params.py -v
```

Expected: all pass.

- [ ] **Step 2.7: Commit Task 2**

```bash
git add salmon_ibm/baltic_params.py tests/test_hatchery_params.py tests/test_baltic_params.py
git commit -m "feat(hatchery-c2): override loader + strict validation (Task 2)

New _apply_hatchery_overrides() applies a hatchery_overrides: dict
over a BalticBioParams instance via dataclasses.replace, with strict
validation: unknown top-level keys raise (only activity_by_behavior
allowed in C2), invalid Behavior enum sub-keys raise, non-numeric
sub-keys raise with actionable message.

load_baltic_species_config() now returns BalticSpeciesConfig instead
of bare BalticBioParams. Existing test_baltic_params.py callers
updated to cfg.wild access pattern.

Tests 1, 2, 3, 8, 9 of 13 lock in: happy-path merge with identity
check; unsupported override key; YAML typo; invalid Behavior key;
non-numeric sub-key.

Spec: docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md"
```

---

### Task 3: Unified `load_bio_params_from_config` + `origin_aware_activity_mult` helper (TDD)

**Files:**
- Modify: `salmon_ibm/config.py` (`load_bio_params_from_config` returns `BalticSpeciesConfig` always)
- Modify: `salmon_ibm/bioenergetics.py` (new `origin_aware_activity_mult` helper)
- Modify: `tests/test_hatchery_params.py` (append 1 test: 4)

`load_bio_params_from_config` is currently a router: returns `BalticBioParams` if `species_config:` present, else `BioParams`. C2 unifies: it always returns `BalticSpeciesConfig`. Legacy non-Baltic path wraps `BioParams` into `BalticSpeciesConfig(wild=plain_BioParams, hatchery=None)`. This eliminates the isinstance branch at simulation.py:294 caller.

- [ ] **Step 3.1: Append the dispatch helper test**

Append to `tests/test_hatchery_params.py`:

```python
def test_origin_aware_activity_mult_dispatch():
    """Helper dispatches per-agent: WILD origin reads from lut_wild,
    HATCHERY from lut_hatch. Graceful path (lut_hatch is None) returns
    lut_wild for all agents (covers pre-C2 callers and test fixtures).
    Element-by-element equality check, not just shape (a stub returning
    wrong values must not pass)."""
    import numpy as np
    from salmon_ibm.bioenergetics import origin_aware_activity_mult
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY

    behavior = np.array([0, 1, 3, 1, 3], dtype=int)
    origin = np.array(
        [ORIGIN_WILD, ORIGIN_HATCHERY, ORIGIN_HATCHERY, ORIGIN_WILD, ORIGIN_WILD],
        dtype=np.int8,
    )
    lut_wild = np.array([1.0, 1.2, 0.8, 1.5, 1.0])
    lut_hatch = np.array([1.0, 1.5, 0.8, 1.875, 1.0])

    # Mixed dispatch
    out = origin_aware_activity_mult(behavior, origin, lut_wild, lut_hatch)
    expected = np.array([1.0, 1.5, 1.875, 1.2, 1.5])
    np.testing.assert_array_equal(out, expected)

    # Graceful: lut_hatch=None returns lut_wild[behavior] for all agents
    out_graceful = origin_aware_activity_mult(behavior, origin, lut_wild, None)
    np.testing.assert_array_equal(out_graceful, lut_wild[behavior])
```

- [ ] **Step 3.2: Run the new test; expect failure**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py::test_origin_aware_activity_mult_dispatch -v
```

Expected: FAILED with `ImportError: cannot import name 'origin_aware_activity_mult'`.

- [ ] **Step 3.3: Add `origin_aware_activity_mult` to `bioenergetics.py`**

In `salmon_ibm/bioenergetics.py`, add to top imports:

```python
from salmon_ibm.origin import ORIGIN_HATCHERY
```

At the END of the file (after `update_energy`), add:

```python
def origin_aware_activity_mult(
    behavior: np.ndarray,
    origin: np.ndarray,
    lut_wild: np.ndarray,
    lut_hatch: np.ndarray | None,
) -> np.ndarray:
    """Per-agent activity multiplier with origin-aware dispatch.

    Returns lut_wild[behavior] when lut_hatch is None — graceful for
    pre-C2 paths, test fixtures, and scenarios without hatchery
    overrides. Otherwise dispatches per-agent via origin column:
    HATCHERY agents read from lut_hatch, WILD from lut_wild.

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

- [ ] **Step 3.4: Run the helper test; expect pass**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py::test_origin_aware_activity_mult_dispatch -v
```

Expected: 1 passed.

- [ ] **Step 3.5: Append non-Baltic test 12**

Append to `tests/test_hatchery_params.py`:

```python
def test_simulation_init_non_baltic_has_no_hatchery_dispatch(tmp_path):
    """Non-Baltic config (no species_config: key) returns
    BalticSpeciesConfig(wild=plain_BioParams, hatchery=None) from the
    unified loader. Simulation.hatchery_dispatch is None; no isinstance
    branching needed. Locks in the C2 unified-return semantics."""
    from salmon_ibm.config import load_bio_params_from_config
    from salmon_ibm.baltic_params import BalticSpeciesConfig
    from salmon_ibm.bioenergetics import BioParams

    cfg_dict = {}  # No species_config: key — legacy path
    loaded = load_bio_params_from_config(cfg_dict)
    assert isinstance(loaded, BalticSpeciesConfig)
    assert loaded.hatchery is None
    # Legacy path returns plain BioParams as wild, NOT BalticBioParams
    assert isinstance(loaded.wild, BioParams)
```

- [ ] **Step 3.6: Run the test; expect failure**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py::test_simulation_init_non_baltic_has_no_hatchery_dispatch -v
```

Expected: FAILED — `load_bio_params_from_config` currently returns `BioParams`, not `BalticSpeciesConfig`.

- [ ] **Step 3.7: Update `salmon_ibm/config.py`**

In `salmon_ibm/config.py`, locate `load_bio_params_from_config` (around line 47-66). Modify to always return `BalticSpeciesConfig`:

```python
from salmon_ibm.baltic_params import BalticSpeciesConfig, load_baltic_species_config


def load_bio_params_from_config(cfg: dict) -> BalticSpeciesConfig:
    """Route to BalticSpeciesConfig from species_config if present,
    else wrap a plain BioParams in BalticSpeciesConfig(wild=BioParams, hatchery=None).

    Always returns BalticSpeciesConfig — the unified return type
    eliminates isinstance branching at the caller (simulation.py:294)
    and the AttributeError failure mode.
    """
    species_config = cfg.get("species_config")
    if species_config is not None:
        return load_baltic_species_config(species_config)
    # Legacy non-Baltic path: wrap plain BioParams
    plain = bio_params_from_config(cfg)
    return BalticSpeciesConfig(wild=plain, hatchery=None)
```

(Adjust if `bio_params_from_config` is the existing helper that builds a plain `BioParams` from the config dict. Check the file before editing.)

- [ ] **Step 3.8: Run test 12; expect pass**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py::test_simulation_init_non_baltic_has_no_hatchery_dispatch -v
```

Expected: 1 passed.

- [ ] **Step 3.9: Run config + baltic_params regressions**

```bash
micromamba run -n shiny python -m pytest tests/test_config.py tests/test_baltic_params.py -v
```

Expected: green. Likely failure: any test/caller still expecting bare `BioParams` from `load_bio_params_from_config`. Fix them analogously to Step 2.6 (`cfg = load_bio_params_from_config(...); params = cfg.wild`).

- [ ] **Step 3.10: Commit Task 3**

```bash
git add salmon_ibm/config.py salmon_ibm/bioenergetics.py tests/test_hatchery_params.py tests/test_config.py
git commit -m "feat(hatchery-c2): unified loader return + dispatch helper (Task 3)

load_bio_params_from_config() now always returns BalticSpeciesConfig.
Legacy non-Baltic path wraps plain BioParams into
BalticSpeciesConfig(wild=BioParams, hatchery=None). Eliminates the
isinstance branch at simulation.py:294 and the AttributeError failure
mode entirely.

New origin_aware_activity_mult() helper in bioenergetics.py: graceful
fallback to lut_wild when lut_hatch is None; np.where dispatch by
origin column otherwise. Wisconsin kernel signatures unchanged.

Tests 4, 12 of 13 lock in: helper dispatch with mixed origin and
graceful path; non-Baltic legacy returns BalticSpeciesConfig with
hatchery=None and wild as plain BioParams.

Spec: docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md"
```

---

### Task 4: `Simulation` init + cache + `rebuild_luts` + `step` injection + `_event_bioenergetics` (TDD)

**Files:**
- Modify: `salmon_ibm/simulation.py` (5 touch points: `Landscape` TypedDict, imports, `__init__`, `rebuild_luts`, `step`, `_event_bioenergetics`)
- Modify: `tests/test_hatchery_params.py` (append 3 tests: 6, 7, 13)

This task ships the simulation-side wiring. Simulation holds an optional `hatchery_dispatch`; init builds it from the cached `BalticSpeciesConfig`; `rebuild_luts()` re-derives without disk I/O; `step()` injects the dispatch into the per-step landscape dict; `_event_bioenergetics` reads `self.hatchery_dispatch` directly.

- [ ] **Step 4.1: Append 3 tests (6, 7, 13)**

Append to `tests/test_hatchery_params.py`:

```python
def _baltic_yaml_with_hatchery(tmp_path) -> dict:
    """Build a tmp_path-based config dict pointing at a Baltic species
    YAML with hatchery overrides."""
    species_yaml = tmp_path / "species.yaml"
    species_yaml.write_text("""
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior:
      0: 1.0
      1: 1.2
      2: 0.8
      3: 1.5
      4: 1.0
    hatchery_overrides:
      activity_by_behavior:
        1: 1.5
        3: 1.875
""")
    return {"species_config": str(species_yaml)}


def test_rebuild_luts_resilient_to_slider_replacement(tmp_path):
    """app.py:1493 sidebar replaces sim.bio_params with a plain
    BioParams from slider values. rebuild_luts() must NOT anchor the
    hatchery LUT to the plain-BioParams Chinook defaults — instead
    re-derive from the cached BalticSpeciesConfig. Locks in the
    cache-and-re-derive semantics."""
    from salmon_ibm.bioenergetics import BioParams
    from salmon_ibm.simulation import Simulation
    cfg = _baltic_yaml_with_hatchery(tmp_path)
    # NOTE: This test requires Simulation to be constructible from a minimal cfg.
    # If the project's Simulation needs more setup (mesh, env, etc.), use the
    # smallest existing Baltic test fixture and apply the slider replacement.
    sim = Simulation(cfg)  # adjust per existing Simulation signature
    # Mimic app.py:1493 sidebar: replace bio_params with plain BioParams
    sim.bio_params = BioParams(RA=0.005, RB=-0.2, RQ=0.07, ED_MORTAL=4.0,
                               T_OPT=15.0, T_MAX=24.0)
    sim.rebuild_luts()
    # Wild LUT reflects Baltic species-config (NOT plain BioParams Chinook)
    assert sim._activity_lut[3] == pytest.approx(1.5)  # UPSTREAM Baltic
    # Hatchery LUT reflects merged YAML overrides
    assert sim.hatchery_dispatch is not None
    assert sim.hatchery_dispatch.activity_lut[1] == pytest.approx(1.5)  # RANDOM hatchery
    assert sim.hatchery_dispatch.activity_lut[3] == pytest.approx(1.875)  # UPSTREAM hatchery


def test_step_injects_hatchery_dispatch_landscape_key(tmp_path, monkeypatch):
    """Simulation.step() injects hatchery_dispatch into the landscape
    dict. Without this test, a missing dict-key insertion in step()
    would silently degrade dispatch to wild-only without breaking any
    other test."""
    import numpy as np
    from salmon_ibm.simulation import Simulation
    from salmon_ibm.baltic_params import HatcheryDispatch
    cfg = _baltic_yaml_with_hatchery(tmp_path)
    sim = Simulation(cfg)
    captured: dict = {}

    def spy(population, landscape, t):
        captured["landscape"] = landscape

    monkeypatch.setattr(sim._sequencer, "step", spy)
    sim.step()  # advances simulation by one timestep; sequencer.step is monkey-patched

    assert "hatchery_dispatch" in captured["landscape"]
    assert captured["landscape"]["hatchery_dispatch"] is not None
    assert isinstance(captured["landscape"]["hatchery_dispatch"], HatcheryDispatch)


def test_rebuild_luts_noop_on_non_baltic_sim():
    """Non-Baltic sim (no species_config in config) calls rebuild_luts()
    without exception; sim.hatchery_dispatch remains None. Without this,
    a future change could throw KeyError on missing _species_config."""
    from salmon_ibm.simulation import Simulation
    cfg = {}  # No species_config — legacy path
    sim = Simulation(cfg)
    sim.rebuild_luts()  # must not raise
    assert sim.hatchery_dispatch is None
```

- [ ] **Step 4.2: Run the new tests; expect 3 failures**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py -v
```

Expected: 3 failed (one or more attribute / method missing on `Simulation`).

- [ ] **Step 4.3: Update `Landscape` TypedDict in `salmon_ibm/simulation.py`**

In `salmon_ibm/simulation.py`, locate the `Landscape` TypedDict (around line 9). Add new key:

```python
class Landscape(TypedDict, total=False):
    # ... existing keys ...
    activity_lut: np.ndarray
    # ... existing keys ...
    hatchery_dispatch: "HatcheryDispatch | None"  # NEW (C2)
```

Add to top imports:

```python
from salmon_ibm.baltic_params import BalticSpeciesConfig, HatcheryDispatch
```

- [ ] **Step 4.4: Update `Simulation.__init__` to use unified loader + cache**

In `salmon_ibm/simulation.py`, locate `Simulation.__init__` (around line 292-295). Replace the existing line `self.bio_params = load_bio_params_from_config(config)` with:

```python
        loaded = load_bio_params_from_config(config)
        self._species_config = loaded  # cached for rebuild_luts() — no per-slider disk I/O
        self.bio_params = loaded.wild
        if loaded.hatchery is not None:
            from salmon_ibm.bioenergetics import _build_lut as build_lut  # if helper exists
            hatch_lut = self._build_activity_lut_for(loaded.hatchery.activity_by_behavior)
            self.hatchery_dispatch = HatcheryDispatch(
                params=loaded.hatchery,
                activity_lut=hatch_lut,
            )
        else:
            self.hatchery_dispatch = None
```

(Where `_build_activity_lut_for(activity_dict)` is a small helper that mirrors the existing `_build_activity_lut` logic but takes a dict argument. If `_build_activity_lut` already exists at line 575, factor out a private `_build_activity_lut_for(self, activity_dict)` that the existing method delegates to.)

If `hatchery_dispatch` is set, log the overridden keys:

```python
        if self.hatchery_dispatch is not None:
            wild_dict = self.bio_params.activity_by_behavior
            hatch_dict = self.hatchery_dispatch.params.activity_by_behavior
            overridden = sorted({
                k for k in hatch_dict if wild_dict.get(k) != hatch_dict[k]
            })
            logger.info(
                "C2 hatchery dispatch active: activity_by_behavior keys overridden: %s",
                overridden,
            )
```

(Use the project's logger; the existing simulation.py likely has one. If not, `import logging; logger = logging.getLogger(__name__)`.)

- [ ] **Step 4.5: Add `Simulation.rebuild_luts()` method**

In `salmon_ibm/simulation.py`, AFTER `_build_activity_lut` (around line 575-581), add:

```python
    def rebuild_luts(self):
        """Rebuild wild + hatchery activity LUTs from cached species-config.

        Reads from self._species_config (cached at __init__) — does NOT
        re-read disk on every sidebar event. Slider adjustments to
        self.bio_params via app.py:1493 do NOT affect activity LUTs;
        activity is locked to the species-config baseline.

        No-ops on non-Baltic configs (no _species_config cached or
        species_config.hatchery is None).
        """
        if not getattr(self, "_species_config", None):
            return
        species_cfg = self._species_config
        self._activity_lut = self._build_activity_lut_for(
            species_cfg.wild.activity_by_behavior
        )
        if species_cfg.hatchery is not None:
            hatch_lut = self._build_activity_lut_for(
                species_cfg.hatchery.activity_by_behavior
            )
            self.hatchery_dispatch = HatcheryDispatch(
                params=species_cfg.hatchery,
                activity_lut=hatch_lut,
            )
        else:
            self.hatchery_dispatch = None
```

- [ ] **Step 4.6: Update `Simulation.step()` to inject `hatchery_dispatch`**

In `salmon_ibm/simulation.py`, locate the `step()` method (around line 557+) and the landscape dict construction (around line 560-571). Add a new key:

```python
        landscape: Landscape = {
            "mesh": self.mesh,
            "fields": self.env.fields,
            "rng": self._rng,
            "activity_lut": self._activity_lut,
            "hatchery_dispatch": self.hatchery_dispatch,  # NEW (C2)
            # ... existing keys ...
        }
```

- [ ] **Step 4.7: Update `_event_bioenergetics` to dispatch by origin**

In `salmon_ibm/simulation.py`, locate `_event_bioenergetics` (around line 488-510). Find the line that does `np.take(self._activity_lut, ...)` (around line 491) and replace:

```python
        # Was: activity = np.take(self._activity_lut, population.behavior, mode="clip")
        from salmon_ibm.bioenergetics import origin_aware_activity_mult
        hatch_lut = (
            self.hatchery_dispatch.activity_lut
            if self.hatchery_dispatch is not None else None
        )
        activity = origin_aware_activity_mult(
            self.pool.behavior,
            self.pool.origin,
            self._activity_lut,
            hatch_lut,
        )
```

(Adjust to match existing dispatch shape; the helper is graceful so existing code paths still work when hatchery_dispatch is None.)

- [ ] **Step 4.8: Run the 3 new tests; expect all pass**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py -v
```

Expected: 9 passed (1 + 5 + 1 + 2 + ... cumulative).

- [ ] **Step 4.9: Run simulation regression check**

```bash
micromamba run -n shiny python -m pytest tests/test_simulation.py tests/test_integration.py -v
```

Expected: green. The graceful fallback in `origin_aware_activity_mult` means existing tests that don't construct hatchery scenarios are unaffected.

- [ ] **Step 4.10: Commit Task 4**

```bash
git add salmon_ibm/simulation.py tests/test_hatchery_params.py
git commit -m "feat(hatchery-c2): Simulation init + rebuild_luts + step dispatch (Task 4)

Simulation.__init__ caches BalticSpeciesConfig at self._species_config
(eliminates per-slider disk I/O), constructs HatcheryDispatch when
hatchery overrides supplied. New rebuild_luts() reads from cache.
step() injects hatchery_dispatch into landscape dict (single key).
_event_bioenergetics now reads self.hatchery_dispatch directly and
dispatches via origin_aware_activity_mult helper.

Landscape TypedDict gains hatchery_dispatch key (total=False, so
existing consumers unaffected).

Tests 6, 7, 13 of 13 lock in: rebuild_luts resilient to sidebar
slider replacement; step() injects HatcheryDispatch into landscape;
rebuild_luts no-ops on non-Baltic sim.

Spec: docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md"
```

---

### Task 5: `SurvivalEvent` dispatch + Introduction guards + `app.py` sidebar (TDD)

**Files:**
- Modify: `salmon_ibm/events_builtin.py` (`SurvivalEvent.execute` dispatch + `IntroductionEvent.execute` runtime guard + comment)
- Modify: `salmon_ibm/events_hexsim.py` (`PatchIntroductionEvent.execute` runtime guard)
- Modify: `app.py` (sidebar block at line 1493-1501)
- Modify: `tests/test_hatchery_params.py` (append 2 tests: 5, 11)

This task ships the event-side wiring: SurvivalEvent dispatches via landscape-supplied LUT, both Introduction events guard against HATCHERY-without-overrides scenarios, and the Shiny sidebar uses `rebuild_luts()`.

- [ ] **Step 5.1: Append 2 tests (5, 11)**

Append to `tests/test_hatchery_params.py`:

```python
def test_introduction_event_runtime_guard_no_hatchery_params():
    """IntroductionEvent.execute() raises when origin=HATCHERY but
    landscape.get('hatchery_dispatch') is None. Catches the case where
    a runtime IntroductionEvent at step N tries to add hatchery agents
    but no overrides are configured. Init-time guard alone wouldn't
    catch this because Simulation starts with all-WILD agents."""
    import numpy as np
    from salmon_ibm.events_builtin import IntroductionEvent
    from salmon_ibm.origin import ORIGIN_HATCHERY
    # Minimal setup: empty landscape (so hatchery_dispatch is None)
    evt = IntroductionEvent(
        name="bad_intro",
        n_agents=2,
        positions=[0],
        origin=ORIGIN_HATCHERY,
    )
    with pytest.raises(ValueError, match=r"HATCHERY.*hatchery_dispatch"):
        evt.execute(population=None, landscape={}, t=0, mask=None)


def test_patch_introduction_event_runtime_guard_no_hatchery_params():
    """PatchIntroductionEvent.execute() mirror of the IntroductionEvent
    guard. Without this test, the hexsim-mode introduction guard could
    be silently omitted; existing test_events_hexsim.py doesn't
    construct hatchery scenarios."""
    from salmon_ibm.events_hexsim import PatchIntroductionEvent
    from salmon_ibm.origin import ORIGIN_HATCHERY
    evt = PatchIntroductionEvent(
        name="bad_patch",
        patch_spatial_data="some_layer",
        origin=ORIGIN_HATCHERY,
    )
    with pytest.raises(ValueError, match=r"HATCHERY.*hatchery_dispatch"):
        evt.execute(population=None, landscape={}, t=0, mask=None)
```

- [ ] **Step 5.2: Run the new tests; expect 2 failures**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py::test_introduction_event_runtime_guard_no_hatchery_params tests/test_hatchery_params.py::test_patch_introduction_event_runtime_guard_no_hatchery_params -v
```

Expected: 2 failed (no guard yet).

- [ ] **Step 5.3: Add runtime guard + dispatch update to `SurvivalEvent`**

In `salmon_ibm/events_builtin.py`, locate `SurvivalEvent.execute()` (around lines 56-124).

Add module-top import (top of file):

```python
from salmon_ibm.bioenergetics import origin_aware_activity_mult
```

Add a comment at line 56 above `bio_params: BioParams = field(...)`:

```python
    # NOTE: bio_params.activity_by_behavior on this event is dead weight
    # — activity dispatch goes through landscape["activity_lut"] and
    # landscape["hatchery_dispatch"]. Local activity_by_behavior values
    # here are silently ignored. bio_params is still used for
    # T_ACUTE_LETHAL / T_MAX (line 122). See C2 spec.
```

Replace the line 69 dispatch:

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

- [ ] **Step 5.4: Add `IntroductionEvent.execute` runtime guard**

In `salmon_ibm/events_builtin.py`, locate `IntroductionEvent.execute()` (around line 247-249, just before `population.add_agents(...)`). Add at the very top of the method body (right after `def execute(...)`):

```python
        from salmon_ibm.origin import ORIGIN_HATCHERY  # local to avoid module-init cost
        if self.origin == ORIGIN_HATCHERY and landscape.get("hatchery_dispatch") is None:
            raise ValueError(
                f"IntroductionEvent '{self.name}' tags new agents as HATCHERY, "
                f"but the simulation has no hatchery_dispatch configured. "
                f"Add a 'hatchery_overrides:' block under "
                f"species.BalticAtlanticSalmon in the species YAML."
            )
```

- [ ] **Step 5.5: Add `PatchIntroductionEvent.execute` runtime guard**

In `salmon_ibm/events_hexsim.py`, locate `PatchIntroductionEvent.execute()` (around line 396-427). Add at the very top of the method body:

```python
        from salmon_ibm.origin import ORIGIN_HATCHERY
        if self.origin == ORIGIN_HATCHERY and landscape.get("hatchery_dispatch") is None:
            raise ValueError(
                f"PatchIntroductionEvent '{self.name}' tags new agents as HATCHERY, "
                f"but the simulation has no hatchery_dispatch configured. "
                f"Add a 'hatchery_overrides:' block under "
                f"species.BalticAtlanticSalmon in the species YAML."
            )
```

- [ ] **Step 5.6: Update `app.py` sidebar to call `rebuild_luts()`**

In `app.py`, locate the sidebar slider re-init block (around lines 1493-1501). Find the line `sim._activity_lut = sim._build_activity_lut()` and replace with:

```python
        sim.rebuild_luts()
```

(Remove the standalone `_build_activity_lut()` call — `rebuild_luts()` handles both wild and hatchery LUTs in one call.)

- [ ] **Step 5.7: Update `configs/baltic_salmon_species.yaml` with `hatchery_overrides:` block + provenance comments**

Locate the deployed `configs/baltic_salmon_species.yaml` (or the active species config used by deployed scenarios). Add the `hatchery_overrides:` block with the full provenance comment block. Use the exact text from the spec's "YAML schema documentation" section:

```yaml
species:
  BalticAtlanticSalmon:
    # ... existing fields ...
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

Verify the YAML parses correctly:

```bash
micromamba run -n shiny python -c "
from salmon_ibm.baltic_params import load_baltic_species_config
cfg = load_baltic_species_config('configs/baltic_salmon_species.yaml')
print('wild:', cfg.wild.activity_by_behavior)
print('hatch:', cfg.hatchery.activity_by_behavior if cfg.hatchery else None)
"
```

Expected output:
```
wild: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
hatch: {0: 1.0, 1: 1.5, 2: 0.8, 3: 1.875, 4: 1.0}
```

- [ ] **Step 5.8: Run the 2 new guard tests; expect pass**

```bash
micromamba run -n shiny python -m pytest tests/test_hatchery_params.py::test_introduction_event_runtime_guard_no_hatchery_params tests/test_hatchery_params.py::test_patch_introduction_event_runtime_guard_no_hatchery_params -v
```

Expected: 2 passed.

- [ ] **Step 5.9: Run wider regression check**

```bash
micromamba run -n shiny python -m pytest tests/test_events.py tests/test_hatchery_params.py -v
```

Expected: green. Existing IntroductionEvent / PatchIntroductionEvent / SurvivalEvent tests should still pass — they construct WILD agents (default), so the runtime guards don't fire.

- [ ] **Step 5.10: Commit Task 5**

```bash
git add salmon_ibm/events_builtin.py salmon_ibm/events_hexsim.py app.py configs/baltic_salmon_species.yaml tests/test_hatchery_params.py
git commit -m "feat(hatchery-c2): event dispatches + Introduction guards + sidebar (Task 5)

SurvivalEvent.execute now reads landscape['hatchery_dispatch'] and
dispatches via origin_aware_activity_mult. Module-top comment notes
that SurvivalEvent.bio_params.activity_by_behavior is dead weight
(landscape LUT governs); bio_params is still used for thermal
mortality fields.

IntroductionEvent.execute and PatchIntroductionEvent.execute both
gain runtime guards: if self.origin == ORIGIN_HATCHERY and
landscape.get('hatchery_dispatch') is None, raise ValueError with
actionable message. Catches scenarios where a runtime introduction
adds hatchery agents but no overrides are configured.

app.py sidebar at line 1493-1501 now calls sim.rebuild_luts() instead
of sim._build_activity_lut() — rebuilds both wild and hatchery LUTs
from cached species-config.

Tests 5, 11 of 13 lock in both runtime guards.

Spec: docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md"
```

---

### Task 6: Full pytest + plan stamp + final commit + push

**Files:**
- Modify: `docs/superpowers/plans/2026-05-01-hatchery-c2-bioparams.md` (add ✅ EXECUTED stamp)

Run the whole suite to surface regressions; if green, stamp and push.

- [ ] **Step 6.1: Run full pytest suite**

```bash
micromamba run -n shiny python -m pytest tests/ -q --no-header
```

Expected (~14-23 minutes): `842 passed, 34 skipped, 7 deselected, 1 xfailed`. Zero failures.

- [ ] **Step 6.2: If failures, triage**

- **(A) `AttributeError: 'BioParams' object has no attribute 'wild'`** — a caller of `load_bio_params_from_config` or `load_baltic_species_config` wasn't updated to use `cfg.wild`. Search and fix.
- **(B) `ValueError: activity_by_behavior values must be positive floats`** — an existing fixture or YAML has 0.0 or negative activity values. Either fix the fixture or relax the validation if the value is intentional (zero is questionable but not strictly invalid biologically).
- **(C) Test 6 / 7 / 13 fail with `Simulation` constructor errors** — the test fixtures may need adjustment to match the actual `Simulation()` signature. Fix the test setup, not the production code.
- **(D) Unexpected regression in unrelated test** — should not happen since C2's dispatch is graceful when `lut_hatch is None`. If it does, escalate.

Re-run the suite after each fix; commit fixes as separate small commits with `fix(hatchery-c2): ...` messages.

- [ ] **Step 6.3: Stamp the plan as ✅ EXECUTED**

In `docs/superpowers/plans/2026-05-01-hatchery-c2-bioparams.md`, replace the line:

```markdown
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
```

With:

```markdown
> **STATUS: ✅ EXECUTED 2026-MM-DD** — All 6 tasks complete. salmon_ibm/baltic_params.py gains HatcheryDispatch + BalticSpeciesConfig types and __post_init__ activity validation; salmon_ibm/bioenergetics.py adds origin_aware_activity_mult helper; salmon_ibm/config.py loader unified to BalticSpeciesConfig; salmon_ibm/simulation.py wires init/cache/rebuild_luts/step/_event_bioenergetics; salmon_ibm/events_builtin.py + events_hexsim.py add SurvivalEvent dispatch + Introduction guards; app.py sidebar uses rebuild_luts. Full pytest suite green at NNN passing. Spec at docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md. Closes the second of three planned tiers (C1 → C2 → C3); next: C3 behaviour divergence.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
```

Replace `2026-MM-DD` with the actual completion date and `NNN` with the verified passing count from Step 6.1.

- [ ] **Step 6.4: Final commit**

```bash
git add docs/superpowers/plans/2026-05-01-hatchery-c2-bioparams.md
git commit -m "docs(plan): stamp hatchery-c2-bioparams plan as EXECUTED (Task 6)

All 6 tasks complete. Suite at NNN passing (829 baseline + 13 new).
Closes the second of three planned tiers (C1 -> C2 -> C3); next plan
will be C3 behaviour divergence (spawn-success, homing precision,
sea-age sampling)."
```

(Update `NNN` to match the actual count.)

- [ ] **Step 6.5: Push the branch and open PR**

```bash
git push -u origin hatchery-c2-bioparams
```

Then open a PR via `gh pr create` (matching the v1.7.3 osmoregulation + hatchery-origin-c1 PR pattern). Title: `Hatchery vs wild C2: activity_by_behavior parameter divergence`. Body should reference the spec, list the 8 modified files, summarize the test count delta (+13 → 842), include the calibration sensitivity sweep table {0%, +12.5%, +25%, +37.5%}, and note that C3 is deferred.

- [ ] **Step 6.6: Update memory after merge + deploy**

Once the PR is merged + tagged + deployed, update memory files:
- `~/.claude/projects/.../memory/curonian_h3_grid_state.md`: bump deployed version label, replace the "in-flight" hatchery-origin-c1 bullet with a new "vNEW — Hatchery vs wild C2" entry.
- `~/.claude/projects/.../memory/curonian_deferred.md`: mark hatchery-vs-wild item #7 as "C2 RESOLVED, C3 still queued" with link to the C3 future plan.

---

## Plan summary

- **6 tasks**, **6 commits** (one per task; Task 6 is the stamp + final commit).
- **8 files modified** (1 new test file + 7 modified production files).
- **+13 net tests** (matches spec exactly).
- **Estimated time:** ~2 days. All work is mechanical (one new dataclass, one new helper, one new method, two runtime guards, one sidebar one-liner).
- **Risk profile:** very low — graceful fallback in `origin_aware_activity_mult` means pre-C2 scenarios continue to use the wild path unchanged; behaviour change surface for non-hatchery scenarios is exactly zero.
- **No backward-compat shim needed** — `hatchery_overrides:` is opt-in; YAMLs without it get `hatchery_dispatch=None` and the dispatch helper short-circuits.

## Spec coverage check

| Spec section | Implementing task |
|---|---|
| `HatcheryDispatch` frozen dataclass | Task 1 |
| `BalticSpeciesConfig` NamedTuple | Task 1 |
| `BalticBioParams.__post_init__` activity validation | Task 1 |
| `_apply_hatchery_overrides()` strict loader | Task 2 |
| `load_baltic_species_config()` extended return | Task 2 |
| `load_bio_params_from_config()` unified return | Task 3 |
| `origin_aware_activity_mult()` helper | Task 3 |
| `Simulation.__init__` branching + cache | Task 4 |
| `Simulation.rebuild_luts()` method | Task 4 |
| `Simulation.step()` landscape injection | Task 4 |
| `_event_bioenergetics` dispatch | Task 4 |
| `Landscape` TypedDict extension | Task 4 |
| Startup log message | Task 4 |
| `SurvivalEvent.execute` dispatch + dead-weight comment | Task 5 |
| `IntroductionEvent.execute` runtime guard | Task 5 |
| `PatchIntroductionEvent.execute` runtime guard | Task 5 |
| `app.py` sidebar `rebuild_luts()` call | Task 5 |
| `configs/baltic_salmon_species.yaml` `hatchery_overrides:` + provenance comments | Task 5 (Step 5.7) |
| 13 new tests (test_hatchery_params.py) | Tasks 1, 2, 3, 4, 5 |
| Plan stamp on completion | Task 6.3 |
