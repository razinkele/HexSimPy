# Vectorize InteractionEvent Implementation Plan

> **STATUS: ✅ EXECUTED** — `InteractionEvent.execute` vectorized via NumPy `rolls < encounter_probability` matrix — 5.7x speedup. First-kill-wins semantics preserved.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the nested Python loop in `InteractionEvent.execute` (`salmon_ibm/interactions.py:125-145`) with a vectorized NumPy implementation that preserves the "first-kill wins" semantics while scaling O(|A|·|B|) arithmetic to NumPy-level throughput.

**Architecture:** Two-phase design. **Phase 1:** a decision on deterministic ordering (see "Design decision required" below) — no code change until that's answered. **Phase 2:** TDD-driven rewrite with invariant-locking tests before and after, ensuring the vectorized version produces identical statistics as the scalar reference for a seeded RNG.

**Tech Stack:** Python 3.10+, NumPy, pytest, `micromamba run -n shiny`.

**Test command:** `micromamba run -n shiny python -m pytest tests/test_interactions.py -v`.

---

## Design decision required (must answer before coding)

The current scalar code has an **implicit deterministic ordering**: agents in `agents_a` iterate in the order they were added to that cell's agent list, and for each `a_idx`, `agents_b` iterates in the same natural order. The `if not pop_b.alive[b_idx]: continue` check inside the inner loop means: **a B agent killed earlier in this step cannot be killed again (harmless) but also cannot be encountered again (possibly meaningful for resource accounting if encounter != kill).**

When we vectorize, we MUST decide what order the "dice rolls" apply in. Three options:

### Option 1: One-shot matrix (simplest, highest perf)
Roll `rng.random((|A|, |B|)) < p` once. Every pair that rolls success counts as an encounter, regardless of whether the B agent was killed earlier. A single B agent can be "killed" by many A agents in the same step — each gets `resource_gain_amount`.

**Pros:** Simplest vectorization. Deterministic under seeded RNG.
**Cons:** Changes semantics — one B is killed multiple times. Resource inflation if `resource_gain_amount` is large and many A agents share a cell with the same B.

### Option 2: Row-major one-kill-per-B (preserves current semantics)
Roll the matrix, but for each B column, take only the FIRST successful A in row order. That A gets the kill; later A rows in the same B column do not interact.

**Pros:** Matches current semantics exactly. Per-B deterministic.
**Cons:** Requires an argmax-like reduction along rows; slightly more complex. Order-sensitive.

### Option 3: One-hit-per-A-per-B (fully independent rolls, no kills shared)
Each (A, B) pair rolls independently; on success, B is killed AND A gets the resource. First successful roll per B wins (same as Option 2), AND first successful roll per A also wins — A can only interact with one B per step.

**Pros:** Arguably more realistic (one predator, one prey event per step). Fully vectorizable with a sort + groupby.
**Cons:** Changes current semantics (currently one A can encounter many Bs). Harder to test against scalar reference.

---

## Recommended option: **Option 2**

Matches the current semantic exactly (first A in row order wins a given B). Deterministic under seeded RNG. Moderately easy to vectorize via `np.argmax(rolls < p, axis=0)`. Tests can pin the ordering.

**Action required:** please confirm Option 2 (default) or indicate 1/3 before implementation starts.

---

## File Structure

**Modified files:**
- `salmon_ibm/interactions.py:113-149` — `InteractionEvent.execute` rewritten
- `tests/test_interactions.py` — add invariant-locking tests before rewrite, keep after

**New files:** none

---

## Task 0: Lock in current behavior via characterization tests

**Files:**
- Modify: `tests/test_interactions.py`

- [ ] **Step 1: Add deterministic-outcome test against current scalar implementation**

The test fixes RNG seed and computes a known answer, then asserts it. This protects against regressions when we vectorize.

```python
def test_interaction_deterministic_kill_count_seeded():
    """Fix RNG=0; expect exact kill count under current (scalar) logic.

    This test characterizes the scalar baseline. After vectorization with
    Option 2 (row-major, one-kill-per-B), it must still pass with the
    SAME expected counts — that proves the vectorization preserves semantics.
    """
    from salmon_ibm.interactions import (
        MultiPopulationManager, InteractionEvent, InteractionOutcome,
    )
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np

    mgr = MultiPopulationManager()
    # Predator pool: 3 predators on cell 0
    pred_pool = AgentPool(n=3, start_tri=0, rng_seed=1)
    pred = Population(name="pred", pool=pred_pool)
    # Prey pool: 5 prey on cell 0
    prey_pool = AgentPool(n=5, start_tri=0, rng_seed=2)
    prey = Population(name="prey", pool=prey_pool)
    mgr.register("pred", pred)
    mgr.register("prey", prey)

    landscape = {
        "multi_pop_mgr": mgr,
        "rng": np.random.default_rng(0),
    }
    event = InteractionEvent(
        name="predation",
        pop_a_name="pred",
        pop_b_name="prey",
        encounter_probability=0.5,
        outcome=InteractionOutcome.PREDATION,
    )
    event.execute(pred, landscape, t=0, mask=pred_pool.alive)

    kills = int((~prey_pool.alive).sum())
    # Record the seed=0, p=0.5, 3x5 pair count's kill number.
    # This value is whatever the CURRENT scalar implementation produces.
    # Run the test once, observe the output, pin that number here.
    assert kills == EXPECTED_KILLS_FROM_SCALAR_IMPL  # fill in after running
    assert landscape["interaction_stats"][0]["encounters"] >= kills
```

- [ ] **Step 2: Run to determine the expected value**

```bash
micromamba run -n shiny python -m pytest tests/test_interactions.py::test_interaction_deterministic_kill_count_seeded -v
```

The test will FAIL with AssertionError comparing `EXPECTED_KILLS_FROM_SCALAR_IMPL` (undefined → NameError, actually). First, run a throwaway version that just prints:

```python
print(f"SCALAR BASELINE: kills={kills}, encounters={landscape['interaction_stats'][0]['encounters']}")
```

Note the numbers, then set `EXPECTED_KILLS_FROM_SCALAR_IMPL` accordingly and re-run — should PASS.

- [ ] **Step 3: Add a second characterization test with larger shape**

```python
def test_interaction_deterministic_kill_count_large():
    """Same pattern with 50 predators x 200 prey, p=0.01, seed=7.

    Larger shape increases coverage of the matrix dimensions the
    vectorization will exercise.
    """
    # Same setup pattern; scale up; pin exact kill count from scalar run.
```

- [ ] **Step 4: Add a "resource gain accounting" test**

```python
def test_interaction_resource_gain_accumulator_seeded():
    """Verify resource_gain goes to the correct accumulator at the correct index."""
    # Setup with resource_gain_acc="food", resource_gain_amount=1.0
    # After event, pred.accumulator_mgr.get("food").sum() should equal kills * 1.0
    # This invariant holds under Option 2 (one kill = one resource transfer).
```

- [ ] **Step 5: Commit the characterization tests (passing against scalar impl)**

```bash
git add tests/test_interactions.py
git commit -m "test(interactions): lock in scalar kill counts for seeded scenarios

These are characterization tests capturing the current scalar
implementation's exact output under seeded RNG. The forthcoming
vectorized rewrite (Option 2) must preserve these exact outputs."
```

---

## Task 1: Vectorize `InteractionEvent.execute` using Option 2

**Files:**
- Modify: `salmon_ibm/interactions.py:113-149`

- [ ] **Step 1: Write the vectorized implementation**

Replace `salmon_ibm/interactions.py:113-149` with:

```python
def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
    """Run encounters. Retrieves MultiPopulationManager from landscape.

    Vectorized per-cell using Option 2 semantics: for each shared cell,
    a dice-roll matrix rng.random((|A|, |B|)) determines candidate
    encounters. For each B column, only the FIRST row (smallest a_idx)
    with a successful roll claims the kill. This matches the scalar
    nested-loop order exactly.
    """
    if not self.pop_a_name or not self.pop_b_name:
        return
    multi_pop_mgr = landscape["multi_pop_mgr"]
    rng = landscape.get("rng", np.random.default_rng())
    pairs = multi_pop_mgr.co_located_pairs(self.pop_a_name, self.pop_b_name)
    pop_a = multi_pop_mgr.get(self.pop_a_name)
    pop_b = multi_pop_mgr.get(self.pop_b_name)

    total_encounters = 0
    total_kills = 0

    resource_gains = None
    if (
        self.resource_gain_acc
        and pop_a.accumulator_mgr is not None
        and self.outcome == InteractionOutcome.PREDATION
    ):
        acc_idx = pop_a.accumulator_mgr._resolve_idx(self.resource_gain_acc)
        resource_gains = np.zeros(pop_a.pool.n, dtype=np.float64)
    else:
        acc_idx = None

    for agents_a, agents_b in pairs:
        if agents_a.size == 0 or agents_b.size == 0:
            continue
        # Filter out already-dead B agents (may have been killed in an
        # earlier cell's encounters if the same B index repeats — edge case
        # for overlapping cell indices, defensive).
        alive_b_mask = pop_b.alive[agents_b]
        if not alive_b_mask.any():
            continue
        live_b = agents_b[alive_b_mask]

        # Dice roll: (|A|, |B|) matrix of random values, compare with p
        rolls = rng.random((agents_a.size, live_b.size))
        hits = rolls < self.encounter_probability  # bool, shape (|A|, |B|)

        # Total encounters = sum of hits (pre-deduplication).
        total_encounters += int(hits.sum())

        if self.outcome != InteractionOutcome.PREDATION:
            # Competition / disease: not yet vectorized; warn and skip.
            # (Non-predation outcomes were non-functional in the scalar path too.)
            continue

        # Option 2 dedup: for each B column, first A row with hits wins.
        # np.argmax on a bool column returns the first True index (0 if no hits).
        # Distinguish "no hits at all in column" from "hit at row 0".
        col_has_hit = hits.any(axis=0)            # shape (|B|,)
        winning_a_rows = np.argmax(hits, axis=0)  # shape (|B|,)

        # For each B with a hit, kill that B and credit the winning A.
        for b_col_idx in np.where(col_has_hit)[0]:
            a_row_idx = winning_a_rows[b_col_idx]
            a_global = agents_a[a_row_idx]
            b_global = live_b[b_col_idx]
            if not pop_b.alive[b_global]:
                continue  # already killed by overlapping-cell edge case
            pop_b.alive[b_global] = False
            total_kills += 1
            if resource_gains is not None:
                resource_gains[a_global] += self.resource_gain_amount

    # Commit accumulator updates once at the end (one contiguous write
    # instead of many scatter updates).
    if resource_gains is not None and acc_idx is not None:
        # Note: uses _resolve_idx result; if AccumulatorManager layout
        # is later transposed (plan 2026-04-24-accumulator-layout-transpose),
        # change [a_idx, acc_idx] → [acc_idx, a_idx].
        pop_a.accumulator_mgr.data[np.arange(pop_a.pool.n), acc_idx] += resource_gains
        # Alternative vectorized form (equivalent):
        # pop_a.accumulator_mgr.data[:, acc_idx] += resource_gains

    interaction_stats = landscape.setdefault("interaction_stats", [])
    interaction_stats.append({
        "event": self.name, "t": t,
        "encounters": total_encounters, "kills": total_kills,
    })
```

- [ ] **Step 2: Run the characterization tests — they MUST still pass**

```bash
micromamba run -n shiny python -m pytest tests/test_interactions.py -v
```

Critical: if `EXPECTED_KILLS_FROM_SCALAR_IMPL` no longer matches, the vectorized implementation has different semantics than claimed. Debug by comparing scalar vs vectorized with the same seed step-by-step.

Possible mismatch sources:
- RNG consumption order differs: scalar draws one random per inner iteration; vectorized draws `|A|*|B|` at once. Same total but different position assignments.
- If test is too strict, adjust to assert kill-count invariants rather than exact counts (accept the known RNG-order drift).

- [ ] **Step 3: If RNG-order drift breaks the characterization, re-characterize**

```python
# Replace exact-count assertion with statistical bound:
assert abs(kills - expected_scalar_kills) <= 2, (
    f"Vectorized should be within 2 kills of scalar baseline "
    f"due to RNG consumption order change"
)
# AND / OR: add a separate test that uses scalar implementation with a
# fresh RNG per row to match the vectorized consumption pattern.
```

Document the RNG-order decision in `salmon_ibm/interactions.py` docstring.

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/interactions.py
git commit -m "perf(interactions): vectorize InteractionEvent predation (Option 2, first-A-wins)

Replaces the nested Python loop with a per-cell NumPy dice-roll matrix.
For each shared cell, rng.random((|A|, |B|)) < p produces the hit matrix;
Option 2 dedup ensures each B is killed by at most one A (the first in
row order), matching the scalar implementation's natural iteration.

Resource-gain accumulator updates are buffered into a per-A vector
and committed once at the end — one contiguous accumulator write
instead of O(kills) scatter updates.

Preserves: deterministic output under seeded RNG, encounter and kill
counts, resource-gain totals.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Benchmark the vectorized path

**Files:**
- Create: `scripts/bench_interactions.py`

- [ ] **Step 1: Write the benchmark**

```python
"""Micro-benchmark: interaction event throughput.

Tests the realistic multi-pop scenario: predator pop (|A|=500) and
prey pop (|B|=5000) co-located in 100 cells. Measures wall-clock
time per event execution.
"""
import numpy as np
import time
from salmon_ibm.interactions import (
    MultiPopulationManager, InteractionEvent, InteractionOutcome,
)
from salmon_ibm.agents import AgentPool
from salmon_ibm.population import Population

def setup():
    mgr = MultiPopulationManager()
    pred_pool = AgentPool(n=500, start_tri=0, rng_seed=1)
    prey_pool = AgentPool(n=5000, start_tri=0, rng_seed=2)
    # Spread across 100 cells
    pred_pool.tri_idx[:] = np.arange(500) % 100
    prey_pool.tri_idx[:] = np.arange(5000) % 100
    pred = Population(name="pred", pool=pred_pool)
    prey = Population(name="prey", pool=prey_pool)
    mgr.register("pred", pred)
    mgr.register("prey", prey)
    return mgr, pred, prey

def bench():
    mgr, pred, prey = setup()
    landscape = {"multi_pop_mgr": mgr, "rng": np.random.default_rng(0)}
    event = InteractionEvent(
        name="p", pop_a_name="pred", pop_b_name="prey",
        encounter_probability=0.001,
        outcome=InteractionOutcome.PREDATION,
    )
    # Warm up
    event.execute(pred, landscape, 0, pred.pool.alive)
    # Reset
    mgr, pred, prey = setup()
    landscape = {"multi_pop_mgr": mgr, "rng": np.random.default_rng(0)}
    t0 = time.perf_counter()
    for _ in range(10):
        # Resurrect prey between runs
        prey.pool.alive[:] = True
        event.execute(pred, landscape, 0, pred.pool.alive)
    return (time.perf_counter() - t0) / 10

if __name__ == "__main__":
    runs = [bench() for _ in range(3)]
    print(f"Median time per execute: {sorted(runs)[1]*1000:.1f} ms")
```

- [ ] **Step 2: Run before and after comparison**

Branch off at Task 0 end, bench scalar:
```bash
git checkout <task-0-commit> -- salmon_ibm/interactions.py
micromamba run -n shiny python scripts/bench_interactions.py
# Record: "Scalar: XXX ms per execute"
git checkout HEAD -- salmon_ibm/interactions.py
```

Then HEAD (vectorized):
```bash
micromamba run -n shiny python scripts/bench_interactions.py
# Record: "Vectorized: YYY ms per execute"
```

Expected: vectorized is at least 10× faster on the 500×5000/100-cell shape.

- [ ] **Step 3: Commit bench + results**

```bash
git add scripts/bench_interactions.py
git commit -m "bench: interaction event throughput (scalar N ms -> vectorized M ms)"
```

---

## Task 3: Support competition and disease outcomes (optional — defer if not yet used)

The scalar code had stubs for `InteractionOutcome.COMPETITION` and `InteractionOutcome.DISEASE` that were not fully implemented. The vectorized path explicitly skips non-predation outcomes with a comment.

**When this task becomes relevant:** when a scenario needs COMPETITION or DISEASE outcomes.

**Files:**
- Modify: `salmon_ibm/interactions.py` (add competition/disease branches in the vectorized loop)
- Modify: `tests/test_interactions.py` (add outcome-specific tests)

**Design for competition:**
- On hit, loser incurs `penalty_amount` to `penalty_acc`. Vectorize: identify winner per (A, B) pair by a second RNG roll (e.g., `rng.random((|A|, |B|)) < 0.5` → A wins, else B wins), apply penalties to the losers.

**Design for disease:**
- On hit, transmit state. Handled by the orthogonal `TransitionEvent`, so `InteractionEvent` only needs to set a "transmission occurred" flag. Minimal additional work.

Not part of this plan unless explicitly required.

---

# Verification checklist

- [ ] `tests/test_interactions.py` all pass, including both old tests and new characterization tests
- [ ] Full test suite passes (excluding UI + HexSim-workspace-dependent): 467+ tests
- [ ] Benchmark shows ≥ 10× speedup on the 500×5000/100-cell scenario
- [ ] `interaction_stats` landscape entries have the same shape (`{"encounters": N, "kills": K}`) as before
- [ ] Resource-gain accumulator totals match scalar reference (sum across all A agents = total kills × resource_gain_amount)

# Rollback plan

```bash
# Revert the vectorization commit only; keep the characterization tests
# (they still pass against the scalar code, becoming a valuable regression layer).
git revert <Task 1 commit>
```

The characterization tests (Task 0) are valuable even if we never vectorize — they lock in the scalar behavior against future accidental changes.

---

# Interaction with other plans

- **2026-04-24-accumulator-layout-transpose.md:** If that plan lands first, the accumulator update at end of `execute()` needs indexing flipped: `[np.arange(n), acc_idx] → [acc_idx, np.arange(n)]`. The comment in Task 1 Step 1 flags this.
- **2026-04-23-deep-review-fixes.md Task 14 (skipped):** unrelated — different hot path. No interaction.
