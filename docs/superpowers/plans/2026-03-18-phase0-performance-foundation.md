# Phase 0: Performance Foundation — Implementation Plan

> **STATUS: ✅ EXECUTED** — Vectorized movement kernels (`_step_random_vec`, `_step_directed_vec`) in `salmon_ibm/movement.py`; perf benchmark shipped.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the salmon IBM handle 100K agents at <0.5s per timestep by vectorizing movement kernels and eliminating per-agent Python loops.

**Architecture:** The current `execute_movement()` iterates over every alive agent in a Python `for` loop. Each iteration calls `mesh.water_neighbors()` which returns a Python list. We replace this with group-vectorized NumPy operations using the existing precomputed `_water_nbrs` / `_water_nbr_count` padded arrays. Movement functions become array-native: all agents of the same behavior type are processed as a batch. Advection is similarly vectorized.

**Tech Stack:** NumPy (vectorized array ops), existing SoA AgentPool, existing precomputed neighbor arrays. No new dependencies.

**Test command:** `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Benchmark command:** `conda run -n shiny python -c "from salmon_ibm.simulation import Simulation; from salmon_ibm.config import load_config; import time; cfg = load_config('config_curonian_minimal.yaml'); sim = Simulation(cfg, n_agents=5000, data_dir='data', rng_seed=42); t0=time.perf_counter(); sim.run(10); print(f'{(time.perf_counter()-t0)/10:.4f} s/step for 5000 agents')"`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `salmon_ibm/movement.py` | Movement kernels (RANDOM, DIRECTED, CWR, advection) | **Rewrite** — replace per-agent loops with vectorized ops |
| `tests/test_movement.py` | Movement correctness tests | **Modify** — add benchmark test, keep existing behavioral tests |
| `tests/test_perf.py` | Performance regression tests | **Create** — ensure vectorized is faster and correct |

---

### Task 1: Add Performance Benchmark Baseline

**Files:**
- Create: `tests/test_perf.py`

Capture current per-step timing so we can measure improvement.

- [ ] **Step 1: Write the benchmark test**

```python
"""Performance regression tests for the simulation engine."""
import time

import numpy as np
import pytest

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


@pytest.mark.slow
def test_step_performance_5000_agents():
    """Benchmark: 10 steps with 5000 agents should complete in <10s.

    This is a loose bound for the un-optimized baseline.
    After vectorization, tighten to <2s.
    """
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=5000, data_dir="data", rng_seed=42)

    n_steps = 10
    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0

    per_step = elapsed / n_steps
    print(f"\n  Baseline: {per_step:.4f} s/step for 5000 agents")
    assert elapsed < 10.0, f"10 steps took {elapsed:.1f}s, expected <10s"
```

- [ ] **Step 2: Run test to capture baseline**

Run: `conda run -n shiny python -m pytest tests/test_perf.py -v -s --ignore=tests/test_playwright.py`
Expected: PASS, prints baseline timing (likely 0.1-0.5 s/step)

- [ ] **Step 3: Commit**

```bash
git add tests/test_perf.py
git commit -m "test: add performance benchmark baseline for 5000 agents"
```

---

### Task 2: Vectorize RANDOM Movement

**Files:**
- Modify: `salmon_ibm/movement.py` — replace `_step_random` and its call site
- Test: `tests/test_movement.py::test_random_moves_to_neighbor`

The key insight: for N agents doing RANDOM walk with `n_micro_steps`, we can process all N simultaneously per micro-step. Each agent picks a random neighbor from the precomputed `_water_nbrs` array.

- [ ] **Step 1: Write failing test for vectorized random**

Add to `tests/test_movement.py`:

```python
def test_random_movement_vectorized_lands_on_water(mesh):
    """Vectorized random movement should land on valid water cells."""
    water_ids = np.where(mesh.water_mask)[0]
    for start in water_ids:
        if len(mesh.water_neighbors(start)) > 0:
            break

    # Run scalar (current implementation) with fixed seed
    pool1 = AgentPool(n=50, start_tri=start)
    pool1.behavior[:] = Behavior.RANDOM
    fields = {"ssh": np.zeros(mesh.n_triangles), "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool1, mesh, fields, seed=99)

    # All agents should be on water cells
    assert np.all(mesh.water_mask[pool1.tri_idx])
    # At least some should have moved
    assert (pool1.tri_idx != start).sum() > 0
```

- [ ] **Step 2: Run test to verify it passes with current code**

Run: `conda run -n shiny python -m pytest tests/test_movement.py::test_random_movement_vectorized_lands_on_water -v`
Expected: PASS (this is a correctness baseline)

- [ ] **Step 3: Implement vectorized `_step_random_vec`**

Replace the body of `_step_random` in `salmon_ibm/movement.py` with a vectorized batch function. Replace the per-agent dispatch loop for RANDOM in `execute_movement`:

```python
def _step_random_vec(tri_indices, water_nbrs, water_nbr_count, rng, steps):
    """Vectorized random walk for a batch of agents.

    Parameters
    ----------
    tri_indices : int array (n,) — current cell index per agent (modified in-place)
    water_nbrs : int array (n_cells, max_nbrs) — precomputed neighbor array
    water_nbr_count : int array (n_cells,) — valid neighbor count per cell
    rng : numpy Generator
    steps : int — number of micro-steps
    """
    n = len(tri_indices)
    for _ in range(steps):
        current = tri_indices
        counts = water_nbr_count[current]             # (n,)
        has_nbrs = counts > 0
        if not has_nbrs.any():
            break
        # For agents with neighbors: pick a random neighbor index [0, count)
        rand_idx = rng.integers(0, np.maximum(counts, 1))  # (n,)
        chosen = water_nbrs[current, rand_idx]         # (n,)
        # Only move agents that have neighbors
        tri_indices[has_nbrs] = chosen[has_nbrs]
```

Update `execute_movement` to dispatch RANDOM agents in batch:

```python
def execute_movement(pool, mesh, fields, seed=None, n_micro_steps=3, cwr_threshold=16.0):
    rng = np.random.default_rng(seed)
    alive = pool.alive & ~pool.arrived

    # Get precomputed neighbor arrays (works for both TriMesh and HexMesh)
    water_nbrs = mesh._water_nbrs
    water_nbr_count = mesh._water_nbr_count

    # --- RANDOM ---
    mask_random = alive & (pool.behavior == Behavior.RANDOM)
    if mask_random.any():
        idx = np.where(mask_random)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_random_vec(tri_buf, water_nbrs, water_nbr_count, rng, n_micro_steps)
        pool.tri_idx[idx] = tri_buf

    # ... (other behaviors remain per-agent for now, vectorized in later tasks)
    # --- Per-agent fallback for UPSTREAM, DOWNSTREAM, TO_CWR ---
    for i in np.where(alive & (pool.behavior >= Behavior.TO_CWR))[0]:
        beh = pool.behavior[i]
        tri = pool.tri_idx[i]
        if beh == Behavior.UPSTREAM:
            pool.tri_idx[i] = _step_directed(tri, mesh, fields["ssh"], rng, n_micro_steps, ascending=False)
        elif beh == Behavior.DOWNSTREAM:
            pool.tri_idx[i] = _step_directed(tri, mesh, fields["ssh"], rng, n_micro_steps, ascending=True)
        elif beh == Behavior.TO_CWR:
            pool.tri_idx[i] = _step_to_cwr(tri, mesh, fields["temperature"], rng, n_micro_steps, cwr_threshold=cwr_threshold)

    _apply_current_advection(pool, mesh, fields, np.where(alive)[0], rng)
```

- [ ] **Step 4: Run all movement tests**

Run: `conda run -n shiny python -m pytest tests/test_movement.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/movement.py tests/test_movement.py
git commit -m "perf: vectorize RANDOM movement kernel"
```

---

### Task 3: Vectorize DIRECTED Movement (UPSTREAM / DOWNSTREAM)

**Files:**
- Modify: `salmon_ibm/movement.py` — add `_step_directed_vec`, update dispatch
- Test: `tests/test_movement.py::test_upstream_net_movement_follows_gradient`, `test_downstream_follows_ascending_ssh`

For gradient-following: all agents of the same direction gather neighbor field values via fancy indexing, then select argmin/argmax.

- [ ] **Step 1: Write failing test for vectorized directed movement**

Add to `tests/test_movement.py`:

```python
def test_directed_movement_vectorized_follows_gradient(mesh):
    """Vectorized UPSTREAM should move agents toward lower SSH values."""
    water_ids = np.where(mesh.water_mask)[0]
    ssh = mesh.centroids[:, 0].copy()  # SSH increases with latitude

    # Pick a start cell near median SSH (room to move both directions)
    water_ssh = ssh[water_ids]
    median_ssh = np.median(water_ssh)
    order = np.argsort(np.abs(water_ssh - median_ssh))
    for idx in order:
        start = water_ids[idx]
        if len(mesh.water_neighbors(start)) > 0:
            break

    pool = AgentPool(n=30, start_tri=start)
    pool.behavior[:] = Behavior.UPSTREAM
    pool.steps[:] = 10  # not first move
    initial_ssh = ssh[start]
    fields = {"ssh": ssh, "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=77)

    # All should be on water cells
    assert np.all(mesh.water_mask[pool.tri_idx])
    # Most should have moved toward lower SSH (upstream)
    final_ssh = ssh[pool.tri_idx]
    moved_lower = (final_ssh < initial_ssh).mean()
    assert moved_lower > 0.4, (
        f"UPSTREAM fish should tend toward lower SSH, but only {moved_lower:.0%} did"
    )
```

- [ ] **Step 2: Run test to verify reference works**

Run: `conda run -n shiny python -m pytest tests/test_movement.py::test_directed_movement_vectorized_follows_gradient -v`
Expected: PASS

- [ ] **Step 3: Implement `_step_directed_vec`**

```python
def _step_directed_vec(tri_indices, water_nbrs, water_nbr_count, field,
                       rng, steps, ascending):
    """Vectorized gradient-following for a batch of agents.

    Even micro-steps: move to neighbor with best field value.
    Odd micro-steps: random jitter (move to random neighbor).
    """
    n = len(tri_indices)
    max_nbrs = water_nbrs.shape[1]

    for s in range(steps):
        current = tri_indices
        counts = water_nbr_count[current]                    # (n,)
        has_nbrs = counts > 0
        if not has_nbrs.any():
            break

        if s % 2 == 0:
            # Gradient step: gather field values at all neighbors
            # Shape: (n, max_nbrs) — use -1 neighbors masked out
            nbr_matrix = water_nbrs[current]                 # (n, max_nbrs)
            # Clamp -1 to 0 for safe indexing, then mask
            safe_idx = np.where(nbr_matrix >= 0, nbr_matrix, 0)
            nbr_vals = field[safe_idx]                       # (n, max_nbrs)
            # Mask invalid neighbors
            invalid = nbr_matrix < 0
            if ascending:
                nbr_vals[invalid] = -np.inf
                best_local = np.argmax(nbr_vals, axis=1)     # (n,)
            else:
                nbr_vals[invalid] = np.inf
                best_local = np.argmin(nbr_vals, axis=1)     # (n,)
            chosen = nbr_matrix[np.arange(n), best_local]    # (n,)
            tri_indices[has_nbrs] = chosen[has_nbrs]
        else:
            # Random jitter step
            rand_idx = rng.integers(0, np.maximum(counts, 1))
            chosen = water_nbrs[current, rand_idx]
            tri_indices[has_nbrs] = chosen[has_nbrs]
```

Update `execute_movement` to dispatch UPSTREAM and DOWNSTREAM in batch:

```python
    # --- UPSTREAM ---
    mask_up = alive & (pool.behavior == Behavior.UPSTREAM)
    if mask_up.any():
        idx = np.where(mask_up)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_directed_vec(tri_buf, water_nbrs, water_nbr_count,
                           fields["ssh"], rng, n_micro_steps, ascending=False)
        pool.tri_idx[idx] = tri_buf

    # --- DOWNSTREAM ---
    mask_down = alive & (pool.behavior == Behavior.DOWNSTREAM)
    if mask_down.any():
        idx = np.where(mask_down)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_directed_vec(tri_buf, water_nbrs, water_nbr_count,
                           fields["ssh"], rng, n_micro_steps, ascending=True)
        pool.tri_idx[idx] = tri_buf
```

- [ ] **Step 4: Run all movement tests**

Run: `conda run -n shiny python -m pytest tests/test_movement.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/movement.py tests/test_movement.py
git commit -m "perf: vectorize UPSTREAM/DOWNSTREAM movement kernels"
```

---

### Task 4: Vectorize TO_CWR Movement

**Files:**
- Modify: `salmon_ibm/movement.py` — add `_step_to_cwr_vec`, update dispatch
- Test: `tests/test_movement.py::test_cwr_stops_at_threshold`, `test_to_cwr_seeks_cooler_water`

CWR movement: agents move to coldest neighbor until below threshold. Vectorize by processing all TO_CWR agents per micro-step.

- [ ] **Step 1: Implement `_step_to_cwr_vec`**

```python
def _step_to_cwr_vec(tri_indices, water_nbrs, water_nbr_count, temperature,
                     steps, cwr_threshold):
    """Vectorized cold-water refuge seeking for a batch of agents.

    Agents move to coldest neighbor each step. Stop if already below threshold.
    """
    n = len(tri_indices)
    max_nbrs = water_nbrs.shape[1]

    for _ in range(steps):
        current = tri_indices
        # Stop condition: already at CWR
        above_thresh = temperature[current] >= cwr_threshold
        counts = water_nbr_count[current]
        active = above_thresh & (counts > 0)
        if not active.any():
            break

        # Gather neighbor temps, pick coldest
        nbr_matrix = water_nbrs[current]
        safe_idx = np.where(nbr_matrix >= 0, nbr_matrix, 0)
        nbr_temps = temperature[safe_idx]
        nbr_temps[nbr_matrix < 0] = np.inf   # mask invalid
        best_local = np.argmin(nbr_temps, axis=1)
        chosen = nbr_matrix[np.arange(n), best_local]
        tri_indices[active] = chosen[active]
```

Update `execute_movement` to dispatch TO_CWR in batch:

```python
    # --- TO_CWR ---
    mask_cwr = alive & (pool.behavior == Behavior.TO_CWR)
    if mask_cwr.any():
        idx = np.where(mask_cwr)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_to_cwr_vec(tri_buf, water_nbrs, water_nbr_count,
                         fields["temperature"], n_micro_steps, cwr_threshold)
        pool.tri_idx[idx] = tri_buf
```

- [ ] **Step 2: Remove per-agent fallback loop**

The per-agent loop for `TO_CWR`, `UPSTREAM`, `DOWNSTREAM` in `execute_movement` should now be completely removed since all behaviors are vectorized. Keep `_step_directed` and `_step_to_cwr` as private functions for reference/testing but they should no longer be called from `execute_movement`.

The final `execute_movement` should be:

```python
def execute_movement(pool, mesh, fields, seed=None, n_micro_steps=3, cwr_threshold=16.0):
    rng = np.random.default_rng(seed)
    alive = pool.alive & ~pool.arrived

    water_nbrs = mesh._water_nbrs
    water_nbr_count = mesh._water_nbr_count

    # --- HOLD: no movement (skip) ---

    # --- RANDOM ---
    mask_random = alive & (pool.behavior == Behavior.RANDOM)
    if mask_random.any():
        idx = np.where(mask_random)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_random_vec(tri_buf, water_nbrs, water_nbr_count, rng, n_micro_steps)
        pool.tri_idx[idx] = tri_buf

    # --- UPSTREAM ---
    mask_up = alive & (pool.behavior == Behavior.UPSTREAM)
    if mask_up.any():
        idx = np.where(mask_up)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_directed_vec(tri_buf, water_nbrs, water_nbr_count,
                           fields["ssh"], rng, n_micro_steps, ascending=False)
        pool.tri_idx[idx] = tri_buf

    # --- DOWNSTREAM ---
    mask_down = alive & (pool.behavior == Behavior.DOWNSTREAM)
    if mask_down.any():
        idx = np.where(mask_down)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_directed_vec(tri_buf, water_nbrs, water_nbr_count,
                           fields["ssh"], rng, n_micro_steps, ascending=True)
        pool.tri_idx[idx] = tri_buf

    # --- TO_CWR ---
    mask_cwr = alive & (pool.behavior == Behavior.TO_CWR)
    if mask_cwr.any():
        idx = np.where(mask_cwr)[0]
        tri_buf = pool.tri_idx[idx].copy()
        _step_to_cwr_vec(tri_buf, water_nbrs, water_nbr_count,
                         fields["temperature"], n_micro_steps, cwr_threshold)
        pool.tri_idx[idx] = tri_buf

    # --- Current advection (still scalar until Task 5) ---
    _apply_current_advection(pool, mesh, fields, np.where(alive)[0], rng)
```

> **Note:** `_apply_current_advection` (scalar) is used here because the vectorized
> version is not implemented until Task 5. Task 5 will replace this call with
> `_apply_current_advection_vec`.

> **RNG behavior change:** After vectorization, the same seed will produce different
> trajectories because RNG calls happen in batch rather than per-agent. Existing tests
> check behavioral properties (gradient-following, water-cell validity), not exact positions.

- [ ] **Step 3: Run all movement tests**

Run: `conda run -n shiny python -m pytest tests/test_movement.py -v`
Expected: ALL PASS

- [ ] **Step 4: Run full test suite**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/movement.py
git commit -m "perf: vectorize TO_CWR movement kernel, remove per-agent loop"
```

---

### Task 5: Vectorize Current Advection

**Files:**
- Modify: `salmon_ibm/movement.py` — replace `_apply_current_advection` with `_apply_current_advection_vec`

The advection loop iterates per-agent to find the neighbor most aligned with current flow direction. This can be vectorized by computing dot products for all agents × all neighbors simultaneously.

- [ ] **Step 1: Implement `_apply_current_advection_vec`**

```python
def _apply_current_advection_vec(pool, mesh, fields, alive_mask, rng):
    """Vectorized current advection for all alive agents."""
    u = fields.get("u_current")
    v = fields.get("v_current")
    if u is None or v is None:
        return

    alive_idx = np.where(alive_mask)[0]
    if len(alive_idx) == 0:
        return

    tris = pool.tri_idx[alive_idx]
    speeds = np.sqrt(u[tris]**2 + v[tris]**2)

    # Filter to agents with meaningful current
    moving = speeds >= 0.01
    if not moving.any():
        return

    mov_idx = alive_idx[moving]
    mov_tris = pool.tri_idx[mov_idx]
    mov_speeds = speeds[moving]
    water_nbrs = mesh._water_nbrs
    water_nbr_count = mesh._water_nbr_count
    max_nbrs = water_nbrs.shape[1]

    counts = water_nbr_count[mov_tris]
    has_nbrs = counts > 0
    if not has_nbrs.any():
        return

    # Flow direction for each agent: (v, u) normalized
    flow_y = v[mov_tris]
    flow_x = u[mov_tris]
    flow_norm = np.sqrt(flow_y**2 + flow_x**2) + 1e-12
    flow_y /= flow_norm
    flow_x /= flow_norm

    # Gather neighbor centroids
    nbr_matrix = water_nbrs[mov_tris]                        # (n, max_nbrs)
    safe_idx = np.where(nbr_matrix >= 0, nbr_matrix, 0)

    # Direction vectors to each neighbor
    c0 = mesh.centroids[mov_tris]                            # (n, 2)
    cn = mesh.centroids[safe_idx]                            # (n, max_nbrs, 2)
    dy = cn[:, :, 0] - c0[:, np.newaxis, 0]                 # (n, max_nbrs)
    dx = cn[:, :, 1] - c0[:, np.newaxis, 1]                 # (n, max_nbrs)
    dnorm = np.sqrt(dy**2 + dx**2) + 1e-12
    dy /= dnorm
    dx /= dnorm

    # Dot product with flow direction
    dots = dy * flow_y[:, np.newaxis] + dx * flow_x[:, np.newaxis]
    dots[nbr_matrix < 0] = -999.0

    # Best neighbor per agent
    best_local = np.argmax(dots, axis=1)
    best_nbr = nbr_matrix[np.arange(len(mov_tris)), best_local]

    # Probabilistic drift
    drift_prob = np.minimum(mov_speeds * 5.0, 0.8)
    drift = rng.random(len(mov_tris)) < drift_prob
    update = has_nbrs & drift

    pool.tri_idx[mov_idx[update]] = best_nbr[update]
```

- [ ] **Step 2: Run all movement tests**

Run: `conda run -n shiny python -m pytest tests/test_movement.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/movement.py
git commit -m "perf: vectorize current advection"
```

---

### Task 6: Add HexMesh Precomputed Neighbor Count Array

**Files:**
- Modify: `salmon_ibm/hexsim.py:84-87` — add `_water_nbr_count` for TriMesh duck-typing

`HexMesh` has `neighbors` (N, 6) with -1 for missing, but does NOT have `_water_nbrs` / `_water_nbr_count` that `TriMesh` has. The vectorized movement kernels rely on these. Add them.

- [ ] **Step 1: Write failing test**

Add to `tests/test_movement.py`:

```python
def test_hexmesh_has_water_nbr_arrays():
    """HexMesh should have _water_nbrs and _water_nbr_count for vectorized movement."""
    pytest.importorskip("heximpy")
    from salmon_ibm.hexsim import HexMesh
    import os
    ws = "Columbia River Migration Model/Columbia [small]"
    if not os.path.exists(ws):
        pytest.skip("Columbia workspace not found")
    mesh = HexMesh.from_hexsim(ws, species="chinook")
    assert hasattr(mesh, '_water_nbrs')
    assert hasattr(mesh, '_water_nbr_count')
    assert mesh._water_nbrs.shape[0] == mesh.n_cells
    assert mesh._water_nbr_count.shape[0] == mesh.n_cells
    # Counts should match actual neighbor count
    idx = mesh.n_cells // 2
    expected_count = sum(1 for n in mesh.neighbors[idx] if n >= 0)
    assert mesh._water_nbr_count[idx] == expected_count
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_movement.py::test_hexmesh_has_water_nbr_arrays -v`
Expected: FAIL with `AttributeError: 'HexMesh' object has no attribute '_water_nbrs'`

- [ ] **Step 3: Add `_water_nbrs` and `_water_nbr_count` to HexMesh**

In `salmon_ibm/hexsim.py`, at the end of `__init__` (after `self._tree = cKDTree(centroids)`):

```python
        # Precompute padded water-neighbor arrays for vectorized movement
        # For HexMesh all stored cells are water, so _water_nbrs == neighbors
        self._water_nbrs = neighbors.copy()
        self._water_nbr_count = np.sum(neighbors >= 0, axis=1).astype(np.intp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n shiny python -m pytest tests/test_movement.py::test_hexmesh_has_water_nbr_arrays -v`
Expected: PASS

- [ ] **Step 5: Run full test suite (including hexsim tests)**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_playwright.py`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/hexsim.py tests/test_movement.py
git commit -m "perf: add precomputed water-neighbor arrays to HexMesh"
```

---

### Task 7: Tighten Performance Benchmark

**Files:**
- Modify: `tests/test_perf.py`

After vectorization, the per-step time should be significantly lower. Tighten the benchmark and add a 10K agent test.

- [ ] **Step 1: Update benchmark test**

Replace the test in `tests/test_perf.py`:

```python
"""Performance regression tests for the simulation engine."""
import time

import numpy as np
import pytest

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


@pytest.mark.slow
def test_step_performance_5000_agents():
    """5000 agents, 10 steps should complete in <5s after vectorization."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=5000, data_dir="data", rng_seed=42)

    n_steps = 10
    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0

    per_step = elapsed / n_steps
    print(f"\n  Vectorized: {per_step:.4f} s/step for 5000 agents")
    assert elapsed < 5.0, f"10 steps took {elapsed:.1f}s, expected <5s"


@pytest.mark.slow
def test_step_performance_10000_agents():
    """10000 agents, 10 steps should complete in <10s after vectorization."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10000, data_dir="data", rng_seed=42)

    n_steps = 10
    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0

    per_step = elapsed / n_steps
    print(f"\n  Vectorized: {per_step:.4f} s/step for 10000 agents")
    assert elapsed < 10.0, f"10 steps took {elapsed:.1f}s, expected <10s"


def test_vectorized_movement_correctness():
    """Full simulation should still produce valid, reproducible results."""
    cfg = load_config("config_curonian_minimal.yaml")

    sim1 = Simulation(cfg, n_agents=100, data_dir="data", rng_seed=42)
    sim1.run(n_steps=20)

    cfg2 = load_config("config_curonian_minimal.yaml")
    sim2 = Simulation(cfg2, n_agents=100, data_dir="data", rng_seed=42)
    sim2.run(n_steps=20)

    np.testing.assert_array_equal(sim1.pool.tri_idx, sim2.pool.tri_idx)
    np.testing.assert_array_almost_equal(sim1.pool.ed_kJ_g, sim2.pool.ed_kJ_g)
    assert sim1.pool.alive.sum() > 0, "Some agents should survive 20 steps"
```

- [ ] **Step 2: Run benchmarks**

Run: `conda run -n shiny python -m pytest tests/test_perf.py -v -s`
Expected: ALL PASS, prints vectorized timing

- [ ] **Step 3: Commit**

```bash
git add tests/test_perf.py
git commit -m "test: tighten performance benchmarks after vectorization"
```

---

### Task 8: Clean Up — Remove Dead Scalar Functions

**Files:**
- Modify: `salmon_ibm/movement.py` — remove unused scalar functions

After confirming all tests pass with vectorized movement, remove the now-unused scalar helper functions: `_step_random`, `_step_directed`, `_step_to_cwr`, `_apply_current_advection`.

- [ ] **Step 1: Remove dead scalar functions**

Delete the following functions from `salmon_ibm/movement.py` (search by name, line numbers will have shifted):
- `_step_random` — scalar single-agent random walk
- `_step_directed` — scalar single-agent gradient following
- `_step_to_cwr` — scalar single-agent CWR seeking
- `_apply_current_advection` — scalar per-agent advection loop

Keep only: `execute_movement`, `_step_random_vec`, `_step_directed_vec`, `_step_to_cwr_vec`, `_apply_current_advection_vec`.

- [ ] **Step 2: Check for external references**

Run: `grep -r "_step_random\|_step_directed\|_step_to_cwr\|_apply_current_advection" tests/ salmon_ibm/ --include="*.py" | grep -v "_vec"`

If any test imports the scalar functions directly (e.g. `test_directed_movement_batch_correctness`), update those tests to use the vectorized versions or remove the reference.

- [ ] **Step 3: Run full test suite**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add salmon_ibm/movement.py tests/
git commit -m "refactor: remove scalar movement functions replaced by vectorized versions"
```

---

## Summary of Expected Improvements

| Metric | Before | After (Expected) |
|--------|--------|-------------------|
| `execute_movement` inner loop | Per-agent Python `for` | Group-vectorized NumPy |
| `_apply_current_advection` | Per-agent Python `for` | Batch dot product |
| `water_neighbors()` call overhead | Python list per call | Direct array slicing |
| 5K agents/step | ~0.2-0.5s | ~0.02-0.05s |
| 10K agents/step | ~0.5-1.0s | ~0.05-0.1s |
| Scaling | O(N) Python iterations | O(1) NumPy overhead + O(N) array ops |

## What This Plan Does NOT Cover

These are separate plans to be written after Phase 0:

1. **Phase 0b: Numba JIT** — Further 10-100x speedup on vectorized kernels (separate plan)
2. **Phase 0c: CuPy GPU** — GPU acceleration for bioenergetics/movement (separate plan)
3. **Phase 1: Event Engine** — Configurable event sequencer (separate plan)
4. **Phase 2: Population Management** — Floaters, groups, territories (separate plan)
5. **Phase 3: Genetics & Interactions** — Diploid genetics, predation (separate plan)
