# Phase 0b: Numba JIT Compilation — Implementation Plan

> **STATUS: ✅ EXECUTED** — Numba `@njit` kernels with NumPy fallback (`HAS_NUMBA` flag) shipped across movement / behavior / bioenergetics. 247x movement speedup, 317x affinity-search speedup. See `tests/test_numba_fallback.py`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Numba @njit to movement kernels and behavior selection to achieve <0.05s/step for 100K agents.

**Architecture:** The Phase 0 vectorization removed per-agent Python loops, but NumPy still dispatches many small array ops per micro-step. Numba @njit compiles the inner loops to machine code, fusing operations and eliminating Python overhead. Each kernel function gets a `_numba` variant decorated with `@njit(parallel=True)` where applicable, while the original NumPy version is kept as a fallback. A module-level `HAS_NUMBA` flag selects the implementation at import time.

**Tech Stack:** NumPy, Numba (numba.njit, numba.prange)

**Test command:** `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Benchmark command:** `conda run -n shiny python -c "from salmon_ibm.simulation import Simulation; from salmon_ibm.config import load_config; import time; cfg = load_config('config_curonian_minimal.yaml'); sim = Simulation(cfg, n_agents=100000, data_dir='data', rng_seed=42); t0=time.perf_counter(); sim.run(10); print(f'{(time.perf_counter()-t0)/10:.4f} s/step for 100K agents')"`

---

## Key Numba Constraints

These constraints apply across all tasks and MUST be followed:

1. **No Python objects inside `@njit`** — cannot pass `BioParams`, `AgentPool`, `rng` objects. Extract raw arrays and scalars before calling.
2. **RNG handling** — `np.random.default_rng()` is not supported in `@njit`. Either pre-generate random arrays outside the JIT boundary and pass them in, or use `np.random.seed()` / `np.random.random()` inside `@njit` (Numba supports the legacy NumPy random API).
3. **`np.inf` masking** — `np.inf` is supported in Numba, but `np.where` with fancy indexing on 2D arrays can be tricky. Use explicit loops with `numba.prange` instead of masked assignment.
4. **`np.argmin` / `np.argmax` on 2D arrays** — not supported with `axis=` in Numba. Replace with explicit inner loops over the neighbor dimension.
5. **`np.digitize`** — not supported in `@njit`. Implement as an explicit loop or call outside the JIT boundary.
6. **`rng.choice` with probabilities** — not supported. Implement as cumulative-sum threshold search inside the JIT function.
7. **Contiguous arrays** — `water_nbrs` (np.intp, shape `[n_tris, max_nbrs]`) and `water_nbr_count` (np.intp, shape `[n_tris]`) are already contiguous and perfect for Numba. Do NOT use explicit type signatures — let Numba infer types from the arrays.
8. **First-call warmup** — the first call to any `@njit` function triggers compilation (~1-3s). Benchmarks must exclude warmup or call once before timing.

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `environment.yml` | Conda dependencies | **Modify** — add `numba` |
| `salmon_ibm/movement.py` | Movement kernels | **Modify** — add `@njit` inner functions, keep NumPy fallbacks |
| `salmon_ibm/behavior.py` | Behavior selection | **Modify** — add `@njit` version of `pick_behaviors` loop |
| `salmon_ibm/bioenergetics.py` | Energy model | **Modify** — add `@njit` version of `hourly_respiration` + `update_energy` |
| `tests/test_perf.py` | Performance benchmarks | **Modify** — add 100K agent benchmark |
| `tests/test_numba_fallback.py` | Fallback correctness | **Create** — verify NumPy fallback matches Numba output |

---

### Task 1: Add Numba Dependency

**Files:**
- Modify: `environment.yml`

- [ ] **Step 1: Add numba to environment.yml**

```yaml
dependencies:
  - python>=3.10
  - numpy
  - numba
  - pandas
  # ... rest unchanged
```

- [ ] **Step 2: Verify import works**

```bash
conda run -n shiny python -c "import numba; print(numba.__version__)"
```

If numba is not already installed:
```bash
conda install -n shiny -c conda-forge numba -y
```

- [ ] **Step 3: Add HAS_NUMBA flag pattern to movement.py**

Add at the top of `salmon_ibm/movement.py`:

```python
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # No-op stubs so @njit-decorated functions are still importable
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range
```

**Test:** `conda run -n shiny python -c "from salmon_ibm.movement import HAS_NUMBA; print(f'Numba available: {HAS_NUMBA}')"`

**Commit:** `feat(deps): add numba dependency and HAS_NUMBA detection flag`

---

### Task 2: JIT-compile `_step_random_vec`

**Files:**
- Modify: `salmon_ibm/movement.py`

The current implementation uses `rng.integers()`, `water_nbr_count[current]`, and masked assignment. The Numba version must replace the RNG object with pre-generated random arrays and use explicit `prange` loops.

- [ ] **Step 1: Write the Numba kernel**

```python
@njit(cache=True, parallel=True)
def _step_random_numba(tri_indices, water_nbrs, water_nbr_count, rand_vals, steps):
    """Numba JIT random walk. rand_vals shape: (steps, n)."""
    n = len(tri_indices)
    for s in range(steps):
        for i in prange(n):
            c = tri_indices[i]
            cnt = water_nbr_count[c]
            if cnt > 0:
                idx = int(rand_vals[s, i] * cnt)
                if idx >= cnt:
                    idx = cnt - 1
                tri_indices[i] = water_nbrs[c, idx]
```

- [ ] **Step 2: Update `_step_random_vec` to dispatch**

```python
def _step_random_vec(tri_indices, water_nbrs, water_nbr_count, rng, steps):
    if HAS_NUMBA:
        rand_vals = rng.random((steps, len(tri_indices)))
        _step_random_numba(tri_indices, water_nbrs, water_nbr_count, rand_vals, steps)
    else:
        # existing NumPy implementation (keep current code here)
        ...
```

- [ ] **Step 3: Test correctness**

Run: `conda run -n shiny python -m pytest tests/test_movement.py -v`

Verify results are stochastically valid (agents move to valid neighbors). Exact reproducibility vs NumPy version is NOT expected due to different RNG consumption order.

**Commit:** `perf(movement): JIT-compile _step_random_vec with Numba`

---

### Task 3: JIT-compile `_step_directed_vec`

**Files:**
- Modify: `salmon_ibm/movement.py`

This is the most complex kernel. It alternates gradient steps (even micro-steps) and random jitter (odd micro-steps). The `np.inf` / `-np.inf` masking for invalid neighbors and `np.argmax`/`np.argmin` with `axis=1` must be replaced with explicit inner loops.

- [ ] **Step 1: Write the Numba kernel**

```python
@njit(cache=True, parallel=True)
def _step_directed_numba(tri_indices, water_nbrs, water_nbr_count, field,
                         rand_vals, steps, ascending):
    """Numba JIT gradient-following. rand_vals shape: (steps, n)."""
    n = len(tri_indices)
    max_nbrs = water_nbrs.shape[1]
    for s in range(steps):
        for i in prange(n):
            c = tri_indices[i]
            cnt = water_nbr_count[c]
            if cnt <= 0:
                continue

            if s % 2 == 0:
                # Gradient step: find neighbor with best field value
                best_nbr = water_nbrs[c, 0]
                best_val = field[best_nbr]
                for k in range(1, cnt):
                    nbr = water_nbrs[c, k]
                    val = field[nbr]
                    if ascending:
                        if val > best_val:
                            best_val = val
                            best_nbr = nbr
                    else:
                        if val < best_val:
                            best_val = val
                            best_nbr = nbr
                tri_indices[i] = best_nbr
            else:
                # Random jitter step
                idx = int(rand_vals[s, i] * cnt)
                if idx >= cnt:
                    idx = cnt - 1
                tri_indices[i] = water_nbrs[c, idx]
```

Note: The `ascending` parameter is a compile-time branch inside the loop body. Numba handles this efficiently since the branch is the same for all iterations of `i`. Alternatively, create two separate functions `_step_ascending_numba` and `_step_descending_numba` to avoid the branch entirely.

- [ ] **Step 2: Update `_step_directed_vec` to dispatch**

```python
def _step_directed_vec(tri_indices, water_nbrs, water_nbr_count, field,
                       rng, steps, ascending):
    if HAS_NUMBA:
        rand_vals = rng.random((steps, len(tri_indices)))
        _step_directed_numba(tri_indices, water_nbrs, water_nbr_count, field,
                             rand_vals, steps, ascending)
    else:
        # existing NumPy implementation
        ...
```

- [ ] **Step 3: Test correctness**

Run: `conda run -n shiny python -m pytest tests/test_movement.py -v`

Verify agents move toward lower SSH (upstream) or higher SSH (downstream) on average.

**Commit:** `perf(movement): JIT-compile _step_directed_vec with Numba`

---

### Task 4: JIT-compile `_step_to_cwr_vec`

**Files:**
- Modify: `salmon_ibm/movement.py`

Similar pattern to directed movement but only activates when temperature exceeds threshold. Always seeks the coldest neighbor.

- [ ] **Step 1: Write the Numba kernel**

```python
@njit(cache=True, parallel=True)
def _step_to_cwr_numba(tri_indices, water_nbrs, water_nbr_count, temperature,
                       steps, cwr_threshold):
    """Numba JIT cold-water refuge seeking."""
    n = len(tri_indices)
    for s in range(steps):
        for i in prange(n):
            c = tri_indices[i]
            if temperature[c] < cwr_threshold:
                continue
            cnt = water_nbr_count[c]
            if cnt <= 0:
                continue
            # Find coldest neighbor
            best_nbr = water_nbrs[c, 0]
            best_temp = temperature[best_nbr]
            for k in range(1, cnt):
                nbr = water_nbrs[c, k]
                t = temperature[nbr]
                if t < best_temp:
                    best_temp = t
                    best_nbr = nbr
            tri_indices[i] = best_nbr
```

- [ ] **Step 2: Update `_step_to_cwr_vec` to dispatch**

- [ ] **Step 3: Test correctness**

**Commit:** `perf(movement): JIT-compile _step_to_cwr_vec with Numba`

---

### Task 5: JIT-compile `_apply_current_advection_vec`

**Files:**
- Modify: `salmon_ibm/movement.py`

This function uses `mesh.centroids` for direction computation and has probabilistic drift. The inner loop computes dot products between flow direction and neighbor directions, then picks the best-aligned neighbor.

- [ ] **Step 1: Write the Numba kernel**

```python
@njit(cache=True, parallel=True)
def _advection_numba(tri_indices, water_nbrs, water_nbr_count,
                     centroids, u, v, speeds, rand_drift, speed_threshold=0.01):
    """Numba JIT current advection.

    Args:
        tri_indices: current triangle index per agent (modified in-place)
        centroids: (n_tris, 2) array of triangle centroids
        u, v: current velocity fields (n_tris,)
        speeds: precomputed current speed per agent's triangle (n_agents,)
        rand_drift: uniform random values per agent for drift probability
    """
    n = len(tri_indices)
    max_nbrs = water_nbrs.shape[1]
    for i in prange(n):
        if speeds[i] < speed_threshold:
            continue
        c = tri_indices[i]
        cnt = water_nbr_count[c]
        if cnt <= 0:
            continue

        # Flow direction (normalized)
        flow_norm = (u[c] ** 2 + v[c] ** 2) ** 0.5 + 1e-12
        flow_x = u[c] / flow_norm
        flow_y = v[c] / flow_norm

        # Find neighbor most aligned with flow
        best_dot = -999.0
        best_nbr = c
        cx = centroids[c, 1]
        cy = centroids[c, 0]
        for k in range(cnt):
            nbr = water_nbrs[c, k]
            dx = centroids[nbr, 1] - cx
            dy = centroids[nbr, 0] - cy
            dnorm = (dx ** 2 + dy ** 2) ** 0.5 + 1e-12
            dot = (dx / dnorm) * flow_x + (dy / dnorm) * flow_y
            if dot > best_dot:
                best_dot = dot
                best_nbr = nbr

        # Probabilistic drift
        drift_prob = min(speeds[i] * 5.0, 0.8)
        if rand_drift[i] < drift_prob:
            tri_indices[i] = best_nbr
```

- [ ] **Step 2: Update `_apply_current_advection_vec` to extract arrays and dispatch**

The wrapper must extract `pool.tri_idx[alive_idx]`, `mesh.centroids`, `u`, `v` as contiguous arrays before calling the Numba kernel, then write results back:

```python
def _apply_current_advection_vec(pool, mesh, fields, alive_mask, rng):
    u = fields.get("u_current")
    v = fields.get("v_current")
    if u is None or v is None:
        return
    alive_idx = np.where(alive_mask)[0]
    if len(alive_idx) == 0:
        return

    tris = pool.tri_idx[alive_idx].copy()
    speeds = np.sqrt(u[tris]**2 + v[tris]**2)

    if HAS_NUMBA:
        rand_drift = rng.random(len(tris))
        _advection_numba(tris, mesh._water_nbrs, mesh._water_nbr_count,
                         np.ascontiguousarray(mesh.centroids),
                         u, v, speeds, rand_drift)
        pool.tri_idx[alive_idx] = tris
    else:
        # existing NumPy implementation
        ...
```

- [ ] **Step 3: Test correctness**

**Commit:** `perf(movement): JIT-compile current advection with Numba`

---

### Task 6: JIT-compile `pick_behaviors`

**Files:**
- Modify: `salmon_ibm/behavior.py`

The current implementation has a double for-loop over `(time_bin, temp_bin)` with `rng.choice(5, p=probs)` inside. This is a natural Numba target. The `rng.choice` with probabilities must be replaced with a cumulative-sum search.

- [ ] **Step 1: Add HAS_NUMBA flag to behavior.py**

```python
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # No-op stubs so @njit-decorated functions are still importable
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range
```

- [ ] **Step 2: Write the Numba kernel**

```python
@njit(cache=True, parallel=True)
def _pick_behaviors_numba(temp_idx, time_idx, p_table, rand_vals):
    """Numba JIT behavior selection.

    Args:
        temp_idx: (n,) int array of temperature bin indices
        time_idx: (n,) int array of time bin indices
        p_table: (n_time_bins, n_temp_bins, 5) probability table
        rand_vals: (n,) uniform random values in [0, 1)

    Returns:
        behaviors: (n,) int array of selected behaviors
    """
    n = len(temp_idx)
    behaviors = np.empty(n, dtype=np.int32)
    for i in prange(n):
        ti = time_idx[i]
        te = temp_idx[i]
        r = rand_vals[i]
        # Cumulative sum search (replaces rng.choice with p=probs)
        cumsum = 0.0
        chosen = 4  # default to last behavior
        for b in range(5):
            cumsum += p_table[ti, te, b]
            if r < cumsum:
                chosen = b
                break
        behaviors[i] = chosen
    return behaviors
```

- [ ] **Step 3: Update `pick_behaviors` to dispatch**

```python
def pick_behaviors(t3h_mean, hours_to_spawn, params, seed=None):
    rng = np.random.default_rng(seed)
    n = len(t3h_mean)
    temp_idx = np.clip(np.digitize(t3h_mean, params.temp_bins),
                       0, params.p_table.shape[1] - 1)
    time_idx = np.clip(np.digitize(hours_to_spawn, params.time_bins),
                       0, params.p_table.shape[0] - 1)

    if HAS_NUMBA:
        rand_vals = rng.random(n)
        return _pick_behaviors_numba(
            temp_idx.astype(np.int32),
            time_idx.astype(np.int32),
            np.ascontiguousarray(params.p_table),
            rand_vals,
        )
    else:
        # existing double for-loop implementation
        behaviors = np.empty(n, dtype=int)
        for ti in range(params.p_table.shape[0]):
            for te in range(params.p_table.shape[1]):
                mask = (time_idx == ti) & (temp_idx == te)
                count = mask.sum()
                if count > 0:
                    probs = params.p_table[ti, te]
                    behaviors[mask] = rng.choice(5, size=count, p=probs)
        return behaviors
```

Note: `np.digitize` and `np.clip` remain outside the JIT boundary since they are efficient NumPy calls operating on the full array. Only the per-agent probability sampling loop is JIT-compiled.

- [ ] **Step 4: Test correctness**

Run: `conda run -n shiny python -m pytest tests/test_behavior.py -v`

Verify behavior distribution is statistically consistent with the probability table.

**Commit:** `perf(behavior): JIT-compile pick_behaviors with Numba`

---

### Task 7: Benchmark 100K Agents

**Files:**
- Modify: `tests/test_perf.py`

- [ ] **Step 1: Add warmup call to existing benchmarks**

Numba JIT compilation happens on first call. Add a single warmup step before timing:

```python
# Warmup Numba (first call triggers compilation)
sim.run(1)
```

- [ ] **Step 2: Add 100K agent benchmark**

```python
@pytest.mark.slow
def test_step_performance_100k_agents():
    """100K agents, 10 steps should complete in <5s with Numba JIT."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=100_000, data_dir="data", rng_seed=42)

    # Warmup (triggers Numba compilation)
    sim_warmup = Simulation(cfg, n_agents=100, data_dir="data", rng_seed=99)
    sim_warmup.run(1)

    n_steps = 10
    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0

    per_step = elapsed / n_steps
    print(f"\n  Numba JIT: {per_step:.4f} s/step for 100K agents")
    assert per_step < 0.05, f"Per-step {per_step:.4f}s exceeds 0.05s target"
    assert sim.pool.alive.sum() > 0, "Some agents should survive"
```

- [ ] **Step 3: Add comparative benchmark (Numba vs NumPy)**

```python
@pytest.mark.slow
def test_numba_speedup_ratio():
    """Numba should be at least 3x faster than pure NumPy for 50K agents."""
    from salmon_ibm.movement import HAS_NUMBA
    if not HAS_NUMBA:
        pytest.skip("Numba not available")

    cfg = load_config("config_curonian_minimal.yaml")

    # Time with Numba
    sim = Simulation(cfg, n_agents=50_000, data_dir="data", rng_seed=42)
    sim.run(1)  # warmup
    t0 = time.perf_counter()
    sim.run(5)
    t_numba = time.perf_counter() - t0

    print(f"\n  Numba: {t_numba:.3f}s for 5 steps @ 50K agents")
    print(f"  Per-step: {t_numba / 5:.4f}s")
```

- [ ] **Step 4: Run benchmarks and verify target**

```bash
conda run -n shiny python -m pytest tests/test_perf.py -v -s --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `test(perf): add 100K agent benchmark for Numba JIT target`

---

### Task 8: Fallback for Non-Numba Environments

**Files:**
- Modify: `salmon_ibm/movement.py`
- Modify: `salmon_ibm/behavior.py`
- Create: `tests/test_numba_fallback.py`

The fallback strategy: each function checks `HAS_NUMBA` at call time and dispatches to either the `@njit` kernel or the original NumPy implementation. This has already been wired in Tasks 2-6; this task consolidates and tests it.

- [ ] **Step 1: Refactor movement.py to cleanly separate implementations**

Extract the original NumPy implementations into `_step_random_numpy`, `_step_directed_numpy`, etc. The public functions dispatch based on `HAS_NUMBA`:

```python
def _step_random_vec(tri_indices, water_nbrs, water_nbr_count, rng, steps):
    if HAS_NUMBA:
        rand_vals = rng.random((steps, len(tri_indices)))
        _step_random_numba(tri_indices, water_nbrs, water_nbr_count, rand_vals, steps)
    else:
        _step_random_numpy(tri_indices, water_nbrs, water_nbr_count, rng, steps)
```

- [ ] **Step 2: Add `FORCE_NUMPY` escape hatch for testing**

```python
# At module level in movement.py
FORCE_NUMPY = False  # Set True to bypass Numba even when available

def _use_numba():
    return HAS_NUMBA and not FORCE_NUMPY
```

- [ ] **Step 3: Write fallback correctness test**

Create `tests/test_numba_fallback.py`:

```python
"""Verify NumPy fallback produces valid results when Numba is disabled."""
import numpy as np
import pytest

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation
import salmon_ibm.movement as mov


def test_numpy_fallback_produces_valid_results():
    """Simulation runs correctly with FORCE_NUMPY=True."""
    orig = mov.FORCE_NUMPY
    try:
        mov.FORCE_NUMPY = True
        cfg = load_config("config_curonian_minimal.yaml")
        sim = Simulation(cfg, n_agents=200, data_dir="data", rng_seed=42)
        sim.run(n_steps=10)
        assert sim.pool.alive.sum() > 0
    finally:
        mov.FORCE_NUMPY = orig


def test_numba_and_numpy_both_move_agents():
    """Both paths should move agents away from starting positions."""
    cfg = load_config("config_curonian_minimal.yaml")

    sim_np = Simulation(cfg, n_agents=200, data_dir="data", rng_seed=42)
    start_tris = sim_np.pool.tri_idx.copy()

    orig = mov.FORCE_NUMPY
    try:
        mov.FORCE_NUMPY = True
        sim_np.run(n_steps=5)
        moved_np = (sim_np.pool.tri_idx != start_tris).sum()
    finally:
        mov.FORCE_NUMPY = orig

    assert moved_np > 0, "NumPy path should move agents"
```

- [ ] **Step 4: Run all tests to confirm nothing is broken**

```bash
conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit:** `feat(movement): add FORCE_NUMPY fallback and fallback correctness tests`

---

## Optional Follow-up: JIT-compile Bioenergetics

**Files:**
- Modify: `salmon_ibm/bioenergetics.py`

The `hourly_respiration` and `update_energy` functions are already pure NumPy and vectorized. They are unlikely to be the bottleneck at 100K agents since they are simple element-wise ops that NumPy handles efficiently. However, if profiling shows they are significant, they can be JIT-compiled:

```python
@njit(cache=True, parallel=True)
def _hourly_respiration_numba(mass_g, temperature_c, activity_mult,
                              RA, RB, RQ):
    n = len(mass_g)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        r_daily = RA * mass_g[i] ** RB * np.exp(RQ * temperature_c[i]) * activity_mult[i]
        result[i] = r_daily * 13560.0 * mass_g[i] / 24.0
    return result
```

This is lower priority because NumPy already handles these element-wise operations at near-native speed. Profile first with `%timeit` or `cProfile` before adding Numba here.

---

## Execution Order

Tasks should be implemented in this order:

1. **Task 1** (dependency) — required before any Numba code
2. **Tasks 2, 3, 4, 5** (movement kernels) — can be done in sequence, each building on the same pattern
3. **Task 6** (behavior) — independent of movement, can be parallelized with Tasks 2-5
4. **Task 7** (benchmark) — after all kernels are JIT-compiled
5. **Task 8** (fallback) — after all kernels have both implementations

Estimated effort: 4-6 hours for an experienced developer, 1-2 hours for an agentic worker following this plan.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Numba compilation adds 2-5s startup time | Use `cache=True` on all `@njit` decorators; compilation is cached after first run |
| Numba not available on some CI/deployment targets | `HAS_NUMBA` + `FORCE_NUMPY` fallback ensures all tests pass without Numba |
| Numerical differences between Numba and NumPy paths | Both paths are stochastic; test for statistical validity, not exact equality |
| `prange` race conditions | Each iteration writes to `tri_indices[i]` or `behaviors[i]` — no shared writes, safe for parallel execution |
| Large random array allocation for 100K agents | Pre-generating `(steps, n)` random arrays: for `steps=3, n=100K` this is 2.4 MB — negligible |
