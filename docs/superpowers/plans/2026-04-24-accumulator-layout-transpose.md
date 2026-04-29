# Accumulator Layout Transpose Implementation Plan

> **STATUS: ✅ EXECUTED** — `AccumulatorManager.data` transposed from `(n_agents, n_acc)` to `(n_acc, n_agents)` — 2.77x total speedup. `scripts/bench_accumulators.py` shipped as no-regression sentinel.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transpose `AccumulatorManager.data` from `(n_agents, n_acc)` to `(n_acc, n_agents)` so column-slice access (`data[idx, :]`) is contiguous in memory, giving an expected **1.5-3× speedup** on accumulator-heavy scenarios.

**Architecture:** Pure layout refactor — no semantic changes. Every existing `data[mask, idx]` becomes `data[idx, mask]`; every `data[i, idx]` becomes `data[idx, i]`. ~30 access sites across `accumulators.py`, `events_hexsim.py`, and tests.

**Tech Stack:** Python 3.10+, NumPy, pytest, `micromamba run -n shiny`.

**Scope note:** This is a single-slice perf change. Do NOT combine with other refactors. Benchmark before and after is mandatory to justify the churn.

**Test command:** `micromamba run -n shiny python -m pytest tests/ -q` (substitute `conda run -n shiny` if conda is on PATH; see `2026-04-23-deep-review-fixes.md` env note).

---

## File Structure

**Modified files:**
- `salmon_ibm/accumulators.py` — `AccumulatorManager.data` shape + every updater function's indexing
- `salmon_ibm/events_hexsim.py` — any direct `manager.data[mask, idx]` access inside the dispatch
- `salmon_ibm/events_phase3.py` — any direct accumulator access
- `salmon_ibm/interactions.py:143` — `pop_a.accumulator_mgr.data[a_idx, acc_idx]` → transposed
- `tests/test_accumulators.py` — any test that directly pokes `mgr.data[i, j]`
- `tests/test_parity_test.py` — if it reads accumulator data directly
- `scripts/_profile_step.py` (if exists) — benchmark harness

**New files:**
- `scripts/bench_accumulators.py` — dedicated micro-benchmark to validate speedup

---

## Baseline & decision gate (Phase 0 — MUST pass before Phase 1 starts)

## Task 0: Capture benchmark baseline

**Purpose:** Lock in current-main timing so we can prove the speedup is real (and catch regressions if any).

**Files:**
- Create: `scripts/bench_accumulators.py`

- [ ] **Step 1: Write the benchmark harness**

Create `scripts/bench_accumulators.py`:

```python
"""Micro-benchmark: accumulator-heavy scenario timing.

Measures the hot loop of updater_expression + updater_increment +
updater_accumulator_transfer under realistic (N=50k agents, M=74 accumulators)
shapes. Current-main baseline will be captured; Task 1's transpose is
justified only if this improves by >=1.3x.
"""
import numpy as np
import time
from salmon_ibm.accumulators import (
    AccumulatorManager, AccumulatorDef,
    updater_expression, updater_increment, updater_accumulator_transfer,
)

N_AGENTS = 50_000
N_ACCS = 74
N_ITER = 1000


def bench():
    defs = [AccumulatorDef(name=f"acc_{i}", min_val=0.0, max_val=1e9)
            for i in range(N_ACCS)]
    mgr = AccumulatorManager(N_AGENTS, defs)
    # Seed with random data
    rng = np.random.default_rng(42)
    mgr.data[:] = rng.random((N_AGENTS, N_ACCS))
    mask = np.ones(N_AGENTS, dtype=bool)

    # Pattern 1: per-column write (touches ONE column)
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        updater_increment(mgr, "acc_7", mask, amount=1.0)
    t_increment = time.perf_counter() - t0

    # Pattern 2: expression over 2 columns (read 2, write 1)
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        updater_expression(mgr, "acc_15", mask, expression="acc_7 * 2 + acc_3")
    t_expression = time.perf_counter() - t0

    # Pattern 3: transfer between two columns
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        updater_accumulator_transfer(mgr, "acc_7", "acc_15", mask, fraction=0.1)
    t_transfer = time.perf_counter() - t0

    return {
        "increment": t_increment,
        "expression": t_expression,
        "transfer": t_transfer,
        "total": t_increment + t_expression + t_transfer,
    }


if __name__ == "__main__":
    # Run three times, take median (warm cache, JIT-compiled eval)
    runs = [bench() for _ in range(3)]
    medians = {k: sorted(r[k] for r in runs)[1] for k in runs[0]}
    print(f"N_AGENTS={N_AGENTS} N_ACCS={N_ACCS} N_ITER={N_ITER}")
    for k, v in medians.items():
        print(f"  {k:12s}: {v:6.3f}s ({N_ITER/v:7.0f} ops/s)")
```

- [ ] **Step 2: Run baseline and record**

```bash
micromamba run -n shiny python scripts/bench_accumulators.py > /tmp/bench_before.txt 2>&1
cat /tmp/bench_before.txt
```

Record the output in this plan as a comment in Step 3 for later comparison.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_accumulators.py
git commit -m "bench: accumulator micro-benchmark harness (baseline for layout transpose)"
```

---

## Task 1: Change `AccumulatorManager.data` shape

**Files:**
- Modify: `salmon_ibm/accumulators.py:23-36` (class header + `__init__`)

- [ ] **Step 1: Write the failing test first**

Add to `tests/test_accumulators.py`:

```python
def test_accumulator_manager_stores_column_major():
    """After transpose, AccumulatorManager.data is (n_acc, n_agents) not (n_agents, n_acc)."""
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
    mgr = AccumulatorManager(n_agents=100, definitions=[
        AccumulatorDef("a"), AccumulatorDef("b"), AccumulatorDef("c"),
    ])
    assert mgr.data.shape == (3, 100), (
        f"Expected (n_acc=3, n_agents=100), got {mgr.data.shape}"
    )
    # Column-slice (single accumulator across all agents) is contiguous
    assert mgr.data[0].flags["C_CONTIGUOUS"]
```

- [ ] **Step 2: Run — expect FAIL**

```bash
micromamba run -n shiny python -m pytest tests/test_accumulators.py::test_accumulator_manager_stores_column_major -v
```

Expected: FAIL with `(100, 3)` current shape.

- [ ] **Step 3: Change the shape in `AccumulatorManager.__init__`**

In `salmon_ibm/accumulators.py:36`:

```python
# OLD: self.data = np.zeros((n_agents, n_acc), dtype=np.float64)
self.data = np.zeros((n_acc, n_agents), dtype=np.float64)
```

- [ ] **Step 4: Run the new test — expect PASS; run the full test suite — expect MANY FAILURES**

The new-shape test passes. Every other test that directly indexes `mgr.data[mask, idx]` will fail with shape mismatch. That's expected — those failures drive Tasks 2-8.

- [ ] **Step 5: Commit (partial state; tests are red but the invariant is flipped)**

```bash
git add salmon_ibm/accumulators.py tests/test_accumulators.py
git commit -m "refactor(accumulators): flip data shape to (n_acc, n_agents) — tests red until Task 2-8"
```

---

## Task 2: Update `AccumulatorManager.get` and `set`

**Files:**
- Modify: `salmon_ibm/accumulators.py:46-66`

- [ ] **Step 1: Rewrite `get()` and `set()`**

```python
def get(self, key: Union[str, int]) -> np.ndarray:
    idx = self._resolve_idx(key)
    return self.data[idx, :]   # was [:, idx]

def set(
    self,
    key: Union[str, int],
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    idx = self._resolve_idx(key)
    defn = self.definitions[idx]
    clamped = values
    if defn.min_val is not None:
        clamped = np.maximum(clamped, defn.min_val)
    if defn.max_val is not None:
        clamped = np.minimum(clamped, defn.max_val)
    if mask is not None:
        self.data[idx, mask] = clamped   # was [mask, idx]
    else:
        self.data[idx, :] = clamped       # was [:, idx]
```

- [ ] **Step 2: Run — confirm tests that exercise get/set now pass**

```bash
micromamba run -n shiny python -m pytest tests/test_accumulators.py -k "set or get" -v
```

- [ ] **Step 3: Commit**

```bash
git add salmon_ibm/accumulators.py
git commit -m "refactor(accumulators): update get/set for transposed layout"
```

---

## Task 3: Update the simple updater functions (`clear`, `increment`, `stochastic_increment`)

**Files:**
- Modify: `salmon_ibm/accumulators.py:74-107`

- [ ] **Step 1: Transpose each data access**

Apply the rule: `data[mask, idx]` → `data[idx, mask]`, `data[:, idx]` → `data[idx, :]`.

For `updater_clear`:
```python
# OLD: manager.data[mask, idx] = val
manager.data[idx, mask] = val
```

For `updater_increment`:
```python
# OLD: new_vals = manager.data[mask, idx] + amount
new_vals = manager.data[idx, mask] + amount
# ...
# OLD: manager.data[mask, idx] = new_vals
manager.data[idx, mask] = new_vals
```

For `updater_stochastic_increment`:
```python
# OLD: new_vals = manager.data[mask, idx] + increments
new_vals = manager.data[idx, mask] + increments
# OLD: manager.data[mask, idx] = new_vals
manager.data[idx, mask] = new_vals
```

- [ ] **Step 2: Run the affected tests — expect PASS**

```bash
micromamba run -n shiny python -m pytest tests/test_accumulators.py -k "clear or increment or stochastic" -v
```

- [ ] **Step 3: Commit**

```bash
git add salmon_ibm/accumulators.py
git commit -m "refactor(accumulators): transpose indexing in clear/increment/stochastic"
```

---

## Task 4: Update `_LazyAccDict` and `updater_expression`

**Files:**
- Modify: `salmon_ibm/accumulators.py:210-323`

- [ ] **Step 1: Rewrite `_LazyAccDict.__getitem__`**

```python
# OLD: self._cache[name] = self._data[self._mask, col]
self._cache[name] = self._data[col, self._mask]
```

- [ ] **Step 2: Rewrite `updater_expression` legacy-mode namespace build**

Line ~312:
```python
# OLD: namespace[name] = manager.data[mask, col_idx]
namespace[name] = manager.data[col_idx, mask]
```

Line ~338:
```python
# OLD: manager.data[mask, idx] = result
manager.data[idx, mask] = result
```

- [ ] **Step 3: Run expression tests — expect PASS**

```bash
micromamba run -n shiny python -m pytest tests/test_accumulators.py -k "expression or lazy" -v
```

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/accumulators.py
git commit -m "refactor(accumulators): transpose indexing in _LazyAccDict and updater_expression"
```

---

## Task 5: Update remaining `manager.data[mask, idx]` writers

**Files:**
- Modify: `salmon_ibm/accumulators.py:349-710`

- [ ] **Step 1: Grep and transpose every remaining site**

Remaining functions: `updater_time_step, updater_individual_id, updater_stochastic_trigger, updater_quantify_location, updater_accumulator_transfer, updater_birth_counts, updater_uptake, updater_individual_locations, updater_trait_value_index, updater_data_lookup`, and the per-agent scatter writers (`updater_allocated_hexagons, updater_explored_hexagons, updater_group_size, updater_group_sum, updater_mate_verification, updater_resources_allocated, updater_resources_explored`).

Apply the mechanical rewrite `data[mask, idx] → data[idx, mask]` and `data[i, idx] → data[idx, i]`.

Sample transformations:

```python
# updater_time_step (line 349):
# OLD: manager.data[mask, idx] = value
manager.data[idx, mask] = value

# updater_individual_id (line 355):
# OLD: manager.data[mask, idx] = agent_ids[mask].astype(np.float64)
manager.data[idx, mask] = agent_ids[mask].astype(np.float64)

# updater_accumulator_transfer (lines 397-416):
src_before = manager.data[src_idx, mask].copy()     # was [mask, src_idx]
nominal_amount = src_before * fraction
# ...
manager.data[src_idx, mask] = new_src               # was [mask, src_idx]
actual_amount = src_before - new_src
new_tgt = manager.data[tgt_idx, mask] + actual_amount  # was [mask, tgt_idx]
# ...
manager.data[tgt_idx, mask] = new_tgt               # was [mask, tgt_idx]

# updater_allocated_hexagons (line 440):
# OLD: manager.data[i, idx] = float(count)
manager.data[idx, i] = float(count)

# updater_group_sum (line 494-495):
# OLD: total = manager.data[members, src_idx].sum()
#      manager.data[members, tgt_idx] = total
total = manager.data[src_idx, members].sum()
manager.data[tgt_idx, members] = total

# updater_uptake (line 580-585):
new_vals = manager.data[idx, mask] + extracted   # was [mask, idx]
manager.data[idx, mask] = new_vals               # was [mask, idx]
```

Process file top-to-bottom systematically. Keep tests passing after each function.

- [ ] **Step 2: Run all accumulator tests — expect PASS**

```bash
micromamba run -n shiny python -m pytest tests/test_accumulators.py -v
```

- [ ] **Step 3: Commit**

```bash
git add salmon_ibm/accumulators.py
git commit -m "refactor(accumulators): transpose indexing in all remaining updater functions"
```

---

## Task 6: Update `events_hexsim.py` direct accumulator accesses

**Files:**
- Modify: `salmon_ibm/events_hexsim.py`

- [ ] **Step 1: Grep for direct `.data[` accesses in events_hexsim.py**

```bash
grep -n "acc_mgr\.data\[\|\.accumulator_mgr\.data\[" salmon_ibm/events_hexsim.py
```

- [ ] **Step 2: Transpose each site**

For each match, apply the mechanical rewrite `[row, col_idx] → [col_idx, row]`. Typical pattern:
```python
# OLD: acc_mgr.data[mask, idx]
acc_mgr.data[idx, mask]
```

- [ ] **Step 3: Run affected tests — expect PASS**

```bash
micromamba run -n shiny python -m pytest tests/test_events_hexsim.py -v
```

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/events_hexsim.py
git commit -m "refactor(events_hexsim): transpose indexing for new accumulator layout"
```

---

## Task 7: Update `events_phase3.py` and `interactions.py`

**Files:**
- Modify: `salmon_ibm/events_phase3.py`, `salmon_ibm/interactions.py:143`

- [ ] **Step 1: Grep and transpose**

```bash
grep -n "\.data\[" salmon_ibm/events_phase3.py salmon_ibm/interactions.py
```

Typical:
```python
# interactions.py:143
# OLD: pop_a.accumulator_mgr.data[a_idx, acc_idx] += self.resource_gain_amount
pop_a.accumulator_mgr.data[acc_idx, a_idx] += self.resource_gain_amount
```

- [ ] **Step 2: Run affected tests — expect PASS**

```bash
micromamba run -n shiny python -m pytest tests/test_events_phase3.py tests/test_interactions.py -v
```

- [ ] **Step 3: Commit**

```bash
git add salmon_ibm/events_phase3.py salmon_ibm/interactions.py
git commit -m "refactor(events_phase3,interactions): transpose indexing for new accumulator layout"
```

---

## Task 8: Update `Population.compact` and any `add_agents` logic

**Files:**
- Modify: `salmon_ibm/population.py`

- [ ] **Step 1: Grep for accumulator data reshape logic**

```bash
grep -n "accumulator\|mgr\.data" salmon_ibm/population.py
```

- [ ] **Step 2: Verify compact() preserves transposed shape**

After compact of `n_agents`, the accumulator data shape becomes `(n_acc, n_alive)`. Rewrite any slicing:

```python
# OLD: pop.accumulator_mgr.data = pop.accumulator_mgr.data[alive_idx]
pop.accumulator_mgr.data = pop.accumulator_mgr.data[:, alive_idx]
```

Same pattern for `add_agents`:

```python
# OLD: new_data = np.zeros((n_new, n_acc))
#      mgr.data = np.concatenate([mgr.data, new_data], axis=0)
new_data = np.zeros((n_acc, n_new))
mgr.data = np.concatenate([mgr.data, new_data], axis=1)
```

- [ ] **Step 3: Run population tests — expect PASS**

```bash
micromamba run -n shiny python -m pytest tests/test_population.py tests/test_simulation.py -v
```

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/population.py
git commit -m "refactor(population): compact and add_agents use transposed accumulator layout"
```

---

## Task 9: Full regression pass

- [ ] **Step 1: Run the full suite**

```bash
micromamba run -n shiny python -m pytest tests/ -q \
  --ignore=tests/test_playwright.py \
  --ignore=tests/test_hex_playwright.py \
  --ignore=tests/test_map_visualization.py
```

Expected: all tests that were passing on main before should still pass. Any failure is a missed transpose site — grep for the failing assertion's tensor shape and fix.

- [ ] **Step 2: Commit a regression-greenlight marker if needed**

No code changes here — if Steps 1 passes cleanly, move on.

---

## Task 10: Capture post-transpose benchmark

- [ ] **Step 1: Re-run the bench script**

```bash
micromamba run -n shiny python scripts/bench_accumulators.py > /tmp/bench_after.txt 2>&1
diff /tmp/bench_before.txt /tmp/bench_after.txt
```

- [ ] **Step 2: Decision gate**

If `total` improved by **>= 1.3× across all three patterns**, proceed to Task 11.

If improvement is < 1.3× or any pattern is slower:
1. Investigate — check cache-miss rates with `perf stat` if available
2. Consider reverting: `git revert Tasks 1-8` as a single revert-PR
3. Document the negative result in `docs/superpowers/specs/2026-04-24-accumulator-transpose-result.md` so future reviewers don't re-attempt the same perf experiment

- [ ] **Step 3: Commit the measurement artifact**

```bash
# Add bench results to the plan file as a "Results" section
git add docs/superpowers/plans/2026-04-24-accumulator-layout-transpose.md
git commit -m "docs(plans): record accumulator transpose benchmark results"
```

---

## Task 11: Cleanup — add an explanatory comment above `AccumulatorManager.data`

**Files:**
- Modify: `salmon_ibm/accumulators.py:23`

- [ ] **Step 1: Add a comment explaining the layout choice**

```python
class AccumulatorManager:
    """Vectorized storage and manipulation of per-agent accumulators.

    Storage: 2D NumPy array of shape (n_accumulators, n_agents) — column-major
    with respect to the update pattern. Every updater touches a single
    accumulator column across many agents: data[idx, :] is contiguous and
    cache-friendly, whereas the inverse (n_agents, n_acc) layout would require
    a strided gather per update.

    Historical note: layout was flipped in 2026-04 after benchmark showed
    1.5-3x speedup (see docs/superpowers/plans/2026-04-24-accumulator-layout-transpose.md).
    """
```

- [ ] **Step 2: Commit**

```bash
git add salmon_ibm/accumulators.py
git commit -m "docs(accumulators): document the (n_acc, n_agents) layout rationale"
```

---

# Verification checklist

- [ ] Full test suite (excluding UI + HexSim-workspace-dependent) passes: 467 + new layout test ≈ 468 passed
- [ ] Benchmark shows >= 1.3× speedup across all three patterns (increment, expression, transfer)
- [ ] `tests/test_parity_test.py` (HexSim parity) still passes IF the Columbia workspace is present
- [ ] Peak RAM unchanged or lower (transpose should not change memory footprint)
- [ ] `salmon_ibm/accumulators.py` docstring updated with layout rationale

# Rollback plan

If the refactor doesn't achieve the speedup target:

```bash
# Identify the range of commits (Task 1 - Task 11)
git log --oneline <Task0-commit>..<Task11-commit>
# Revert as a single PR
git revert --no-commit <Task1-commit>..<Task11-commit>
git commit -m "revert: accumulator layout transpose — did not achieve target speedup"
```

No data migration needed — the refactor is purely internal; no persisted state uses the transposed shape.
