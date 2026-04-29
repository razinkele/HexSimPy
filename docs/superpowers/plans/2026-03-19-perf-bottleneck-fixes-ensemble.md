# Phase 0 Performance: Bottleneck Fixes + Ensemble Runner

> **STATUS: ✅ EXECUTED** — `HexSimEnvironment.advance()` redundant-copy elimination, `dSSH_dt` short-circuit, and `salmon_ibm/ensemble.py` multiprocessing runner all shipped. Subsequent 12.8x perf gain landed on top.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the measured bottlenecks in `HexSimEnvironment` and estuarine overrides (62.5% + 16% of runtime), then add a multiprocessing ensemble runner for parallel replicate execution.

**Architecture:** Three targeted fixes: (1) eliminate redundant array copies in static-SSH HexSim mode, (2) short-circuit estuarine overrides when config disables them, (3) add a `run_ensemble()` function using `multiprocessing.Pool`. Each fix is independent and testable in isolation.

**Tech Stack:** NumPy, multiprocessing (stdlib), pytest-benchmark for regression tests.

**Profiling baseline (Columbia River, 1000 agents, 20 steps):**
- `hexsim_env.advance()`: 62.5% of runtime (unnecessary `.copy()` and `.astype()`)
- `dSSH_dt_array()` + estuarine overrides: 16% (always zero for static SSH)
- Movement (Numba JIT): 10% — already fast
- 50K agents: 0.0225 s/step — already under 0.5s target

---

## Task 1: Eliminate Redundant Array Copies in HexSimEnvironment

**Files:**
- Modify: `salmon_ibm/hexsim_env.py:45-88`
- Test: `tests/test_hexsim.py` (add new tests)

The `advance()` method copies the static SSH array twice per step and creates a new float64 temperature array every step. Since SSH never changes, `dSSH_dt` is always zero — skip the copy. Pre-allocate a float64 buffer and write temperature into it each step (implicit float32→float64 conversion during assignment, avoids doubling table memory).

- [ ] **Step 1: Write the failing test for cached temperature**

Add to `tests/test_hexsim.py`:

```python
def test_hexsim_env_advance_reuses_temp_array_dtype():
    """Temperature field should be float64 without per-step conversion."""
    from salmon_ibm.hexsim import HexMesh
    from salmon_ibm.hexsim_env import HexSimEnvironment
    WS = "Columbia River Migration Model/Columbia [small]"
    if not os.path.exists(WS):
        pytest.skip("Columbia workspace not found")
    mesh = HexMesh.from_hexsim(WS)
    env = HexSimEnvironment(WS, mesh)
    env.advance(0)
    assert env.fields["temperature"].dtype == np.float64
    # Advance again — should reuse the same buffer (identity check)
    env.advance(1)
    assert env.fields["temperature"] is env._temp_buf
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_hexsim.py::test_hexsim_env_advance_reuses_temp_array_dtype -v`
Expected: FAIL — `advance()` currently creates a new array each call via `.astype(np.float64)`

- [ ] **Step 3: Write the failing test for static SSH zero-copy**

```python
def test_hexsim_env_ssh_is_static_no_copy():
    """SSH field should not be re-copied each step for static gradient."""
    from salmon_ibm.hexsim import HexMesh
    from salmon_ibm.hexsim_env import HexSimEnvironment
    WS = "Columbia River Migration Model/Columbia [small]"
    if not os.path.exists(WS):
        pytest.skip("Columbia workspace not found")
    mesh = HexMesh.from_hexsim(WS)
    env = HexSimEnvironment(WS, mesh)
    env.advance(0)
    ssh0 = env.fields["ssh"]
    env.advance(1)
    ssh1 = env.fields["ssh"]
    # SSH should be the same object (no copy) since it's static
    assert ssh0 is ssh1
```

- [ ] **Step 4: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_hexsim.py::test_hexsim_env_ssh_is_static_no_copy -v`
Expected: FAIL — `advance()` currently does `self._ssh_static.copy()`

- [ ] **Step 5: Write the failing test for dSSH_dt_array short-circuit**

```python
def test_hexsim_env_dssh_dt_always_zero():
    """dSSH_dt_array should return zeros for static SSH (no computation)."""
    from salmon_ibm.hexsim import HexMesh
    from salmon_ibm.hexsim_env import HexSimEnvironment
    WS = "Columbia River Migration Model/Columbia [small]"
    if not os.path.exists(WS):
        pytest.skip("Columbia workspace not found")
    mesh = HexMesh.from_hexsim(WS)
    env = HexSimEnvironment(WS, mesh)
    env.advance(0)
    env.advance(1)
    dssh = env.dSSH_dt_array()
    assert (dssh == 0.0).all()
    # Should return a cached zeros array (same object each call)
    dssh2 = env.dSSH_dt_array()
    assert dssh is dssh2
```

- [ ] **Step 6: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_hexsim.py::test_hexsim_env_dssh_dt_always_zero -v`
Expected: FAIL on `dssh is dssh2` — currently creates a new array each call

- [ ] **Step 7: Implement optimized HexSimEnvironment**

Modify `salmon_ibm/hexsim_env.py`:

```python
def __init__(self, workspace_dir, mesh, temperature_csv="River Temperature.csv"):
    # ... existing code up to _temp_table load ...

    # Keep float32 to save memory; conversion happens during buffer write
    self._temp_table = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    self.n_timesteps = self._temp_table.shape[1]

    # ... existing SSH gradient code ...

    # Pre-allocate fields — these buffers are reused every step
    n = mesh.n_cells
    self._temp_buf = np.zeros(n, dtype=np.float64)
    self._zero_buf = np.zeros(n, dtype=np.float64)  # cached dSSH/dt = 0
    self._zero_buf.flags.writeable = False  # prevent accidental mutation
    self.fields: dict[str, np.ndarray] = {
        "temperature": self._temp_buf,
        "salinity": np.zeros(n, dtype=np.float64),
        "ssh": self._ssh_static,  # direct reference, no copy
        "u_current": np.zeros(n, dtype=np.float64),
        "v_current": np.zeros(n, dtype=np.float64),
    }
    self._prev_ssh = None  # not needed for static SSH
    self.current_t = -1

def advance(self, t: int) -> None:
    """Update fields for timestep t (wraps around)."""
    self.current_t = t
    t_idx = t % self.n_timesteps
    # Write temperature directly into pre-allocated buffer (implicit float32→64)
    self._temp_buf[:] = self._temp_table[self._zone_ids, t_idx]
    # SSH is static — no copy needed, fields["ssh"] already points to _ssh_static

def dSSH_dt_array(self) -> np.ndarray:
    """Always zero for static SSH gradient."""
    return self._zero_buf

def dSSH_dt(self, cell_idx: int) -> float:
    """Always zero for static SSH gradient."""
    return 0.0
```

- [ ] **Step 8: Run all three new tests + existing tests**

Run: `conda run -n shiny python -m pytest tests/test_hexsim.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add salmon_ibm/hexsim_env.py tests/test_hexsim.py
git commit -m "perf(hexsim_env): eliminate redundant SSH copies and per-step astype in advance()"
```

---

## Task 2: Short-Circuit Estuarine Overrides in Simulation

**Files:**
- Modify: `salmon_ibm/simulation.py:225-248`
- Test: `tests/test_simulation.py` (add new test)

When the config effectively disables estuarine overrides (seiche threshold = 999, DO thresholds = 0), detect this at init and skip the entire `_apply_estuarine_overrides()` call path — including the `dSSH_dt_array()` call that was 16% of runtime. Note: salinity cost is handled separately in `_event_bioenergetics`, not in estuarine overrides.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_simulation.py`:

```python
def test_estuarine_overrides_skipped_when_disabled():
    """Simulation should detect disabled estuary config and skip overrides."""
    from salmon_ibm.simulation import Simulation
    cfg = {
        "grid": {"type": "hexsim"},
        "hexsim": {
            "workspace": "Columbia River Migration Model/Columbia [small]",
            "species": "chinook",
            "temperature_csv": "River Temperature.csv",
        },
        "estuary": {
            "salinity_cost": {"S_opt": 0.5, "S_tol": 999, "k": 0.0},
            "do_avoidance": {"lethal": 0.0, "high": 0.0},
            "seiche_pause": {"dSSHdt_thresh_m_per_15min": 999},
        },
    }
    import os
    if not os.path.exists(cfg["hexsim"]["workspace"]):
        pytest.skip("Columbia workspace not found")
    sim = Simulation(cfg, n_agents=10, rng_seed=42)
    assert sim._skip_estuarine_overrides is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py::test_estuarine_overrides_skipped_when_disabled -v`
Expected: FAIL — `_skip_estuarine_overrides` attribute doesn't exist

- [ ] **Step 3: Implement the short-circuit detection**

In `salmon_ibm/simulation.py`, add after `self.est_cfg = config.get("estuary", {})` in `__init__`:

```python
self._skip_estuarine_overrides = self._detect_estuarine_noop()
```

Add new method:

```python
def _detect_estuarine_noop(self) -> bool:
    """Return True if estuarine overrides are effectively disabled."""
    est = self.est_cfg
    if not est:
        return True
    seiche = est.get("seiche_pause", {})
    thresh = seiche.get("dSSHdt_thresh_m_per_hour")
    if thresh is None:
        thresh = seiche.get("dSSHdt_thresh_m_per_15min", 0.02) * 4.0
    seiche_noop = thresh >= 100.0  # threshold so high it never triggers
    do_cfg = est.get("do_avoidance", {})
    do_noop = do_cfg.get("lethal", 0.0) <= 0 and do_cfg.get("high", 0.0) <= 0
    return seiche_noop and do_noop
```

Add early-return guard to existing `_event_estuarine_overrides` (only change is the guard clause):

```python
def _event_estuarine_overrides(self, population, landscape, t, mask):
    if self._skip_estuarine_overrides:
        return
    self._apply_estuarine_overrides()
```

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/simulation.py tests/test_simulation.py
git commit -m "perf(simulation): short-circuit estuarine overrides when config disables them"
```

---

## Task 3: Numba-Accelerated Barrier Resolution

**Files:**
- Modify: `salmon_ibm/movement.py:360-392`
- Test: `tests/test_barriers.py` (add new test)

The current `_resolve_barriers_vec` uses pure NumPy. Add a Numba kernel for the inner barrier-check loop, with NumPy fallback. Note: `barrier_trans` is accepted in the function signature for API compatibility with `execute_movement` callers, but is unused in both the NumPy and Numba paths (transmission is the implicit default when mortality and deflection don't trigger).

- [ ] **Step 1: Write the failing performance test**

Add to `tests/test_barriers.py`:

```python
def test_barrier_resolution_numba_produces_same_result():
    """Numba and NumPy barrier resolution should produce identical results."""
    import salmon_ibm.movement as mov
    rng = np.random.default_rng(42)
    n = 1000
    n_cells = 500
    max_nbrs = 6
    neighbors = np.random.randint(0, n_cells, (n_cells, max_nbrs))
    current = rng.integers(0, n_cells, n)
    proposed = rng.integers(0, n_cells, n)
    # Make some proposed == a neighbor of current (realistic moves)
    for i in range(n):
        if rng.random() < 0.7:
            proposed[i] = neighbors[current[i], rng.integers(0, max_nbrs)]
    mort = rng.random((n_cells, max_nbrs)) * 0.1
    defl = rng.random((n_cells, max_nbrs)) * 0.2
    trans = np.ones((n_cells, max_nbrs))

    # NumPy path
    orig = mov.FORCE_NUMPY
    try:
        mov.FORCE_NUMPY = True
        final_np, died_np = mov._resolve_barriers_vec(
            current.copy(), proposed.copy(), mort, defl, trans, neighbors,
            np.random.default_rng(99))
        mov.FORCE_NUMPY = False
        final_nb, died_nb = mov._resolve_barriers_vec(
            current.copy(), proposed.copy(), mort, defl, trans, neighbors,
            np.random.default_rng(99))
    finally:
        mov.FORCE_NUMPY = orig
    np.testing.assert_array_equal(final_np, final_nb)
    np.testing.assert_array_equal(died_np, died_nb)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_barriers.py::test_barrier_resolution_numba_produces_same_result -v`
Expected: FAIL — `_resolve_barriers_vec` doesn't have a Numba branch yet

- [ ] **Step 3: Implement Numba barrier resolution kernel**

Add to `salmon_ibm/movement.py`:

```python
@njit(cache=True, parallel=True)
def _resolve_barriers_numba(current, proposed, barrier_mort, barrier_defl,
                             neighbors, rand_vals):
    """Numba kernel for barrier resolution."""
    n = len(current)
    final = proposed.copy()
    died = np.zeros(n, dtype=np.bool_)
    max_nbrs = neighbors.shape[1]
    for i in prange(n):
        if current[i] == proposed[i]:
            continue
        # Find which neighbor slot matches proposed
        slot = -1
        for k in range(max_nbrs):
            if neighbors[current[i], k] == proposed[i]:
                slot = k
                break
        if slot < 0:
            continue
        p_mort = barrier_mort[current[i], slot]
        p_defl = barrier_defl[current[i], slot]
        if p_mort <= 0.0 and p_defl <= 0.0:
            continue
        r = rand_vals[i]
        if r < p_mort:
            died[i] = True
        elif r < p_mort + p_defl:
            final[i] = current[i]
    return final, died
```

Modify `_resolve_barriers_vec` to dispatch:

```python
def _resolve_barriers_vec(current, proposed, barrier_mort, barrier_defl,
                          barrier_trans, neighbors, rng):
    """Resolve barrier outcomes for a batch of proposed moves."""
    n = len(current)
    if n == 0:
        return proposed.copy(), np.zeros(n, dtype=bool)

    if _use_numba():
        rand_vals = rng.random(n)
        return _resolve_barriers_numba(
            current, proposed, barrier_mort, barrier_defl, neighbors, rand_vals)

    # Original NumPy implementation (existing code)
    final = proposed.copy()
    died = np.zeros(n, dtype=bool)
    moving = current != proposed
    if not moving.any():
        return final, died
    nbr_matrix = neighbors[current]
    match = (nbr_matrix == proposed[:, np.newaxis])
    has_match = match.any(axis=1) & moving
    if not has_match.any():
        return final, died
    slot = np.argmax(match, axis=1)
    p_mort = barrier_mort[current, slot]
    p_defl = barrier_defl[current, slot]
    has_barrier = has_match & ((p_mort > 0) | (p_defl > 0))
    if not has_barrier.any():
        return final, died
    rolls = rng.random(n)
    kill = has_barrier & (rolls < p_mort)
    died[kill] = True
    deflect = has_barrier & ~kill & (rolls < p_mort + p_defl)
    final[deflect] = current[deflect]
    return final, died
```

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_barriers.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/movement.py tests/test_barriers.py
git commit -m "perf(barriers): add Numba kernel for barrier resolution"
```

---

## Task 4: Multiprocessing Ensemble Runner

**Files:**
- Create: `salmon_ibm/ensemble.py`
- Test: `tests/test_ensemble.py`

Add a function that runs N simulation replicates in parallel using `multiprocessing.Pool`, returning collected history from all runs.

- [ ] **Step 1: Write the failing test for basic ensemble**

Create `tests/test_ensemble.py`:

```python
"""Tests for multiprocessing ensemble runner."""
import numpy as np
import pytest
from salmon_ibm.ensemble import run_ensemble


def test_ensemble_returns_correct_number_of_results():
    """run_ensemble with 4 replicates should return 4 result dicts."""
    cfg = {
        "grid": {"type": "hexsim"},
        "hexsim": {
            "workspace": "Columbia River Migration Model/Columbia [small]",
            "species": "chinook",
            "temperature_csv": "River Temperature.csv",
        },
        "estuary": {
            "salinity_cost": {"S_opt": 0.5, "S_tol": 999, "k": 0.0},
            "do_avoidance": {"lethal": 0.0, "high": 0.0},
            "seiche_pause": {"dSSHdt_thresh_m_per_15min": 999},
        },
    }
    import os
    if not os.path.exists(cfg["hexsim"]["workspace"]):
        pytest.skip("Columbia workspace not found")

    results = run_ensemble(cfg, n_replicates=4, n_agents=20, n_steps=5, n_workers=2)
    assert len(results) == 4
    for r in results:
        assert "seed" in r
        assert "history" in r
        assert len(r["history"]) == 5
        assert r["history"][-1]["n_alive"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_ensemble.py::test_ensemble_returns_correct_number_of_results -v`
Expected: FAIL — `salmon_ibm.ensemble` module doesn't exist

- [ ] **Step 3: Write the failing test for reproducibility**

Add to `tests/test_ensemble.py`:

```python
def test_ensemble_replicates_are_different():
    """Different seeds should produce different outcomes."""
    cfg = {
        "grid": {"type": "hexsim"},
        "hexsim": {
            "workspace": "Columbia River Migration Model/Columbia [small]",
            "species": "chinook",
            "temperature_csv": "River Temperature.csv",
        },
        "estuary": {
            "salinity_cost": {"S_opt": 0.5, "S_tol": 999, "k": 0.0},
            "do_avoidance": {"lethal": 0.0, "high": 0.0},
            "seiche_pause": {"dSSHdt_thresh_m_per_15min": 999},
        },
    }
    import os
    if not os.path.exists(cfg["hexsim"]["workspace"]):
        pytest.skip("Columbia workspace not found")

    results = run_ensemble(cfg, n_replicates=3, n_agents=50, n_steps=10, n_workers=1)
    # Different seeds => different final mean_ed (probabilistic behavior)
    mean_eds = [r["history"][-1]["mean_ed"] for r in results]
    assert len(set(round(e, 6) for e in mean_eds)) > 1, "Replicates should differ"


def test_ensemble_deterministic_with_base_seed():
    """Same base_seed should produce identical ensemble results.

    Uses n_workers=1 (sequential) because FORCE_NUMPY is a module-level
    flag that doesn't propagate to child processes spawned by Pool.
    """
    cfg = {
        "grid": {"type": "hexsim"},
        "hexsim": {
            "workspace": "Columbia River Migration Model/Columbia [small]",
            "species": "chinook",
            "temperature_csv": "River Temperature.csv",
        },
        "estuary": {
            "salinity_cost": {"S_opt": 0.5, "S_tol": 999, "k": 0.0},
            "do_avoidance": {"lethal": 0.0, "high": 0.0},
            "seiche_pause": {"dSSHdt_thresh_m_per_15min": 999},
        },
    }
    import os
    if not os.path.exists(cfg["hexsim"]["workspace"]):
        pytest.skip("Columbia workspace not found")

    import salmon_ibm.movement as mov
    orig = mov.FORCE_NUMPY
    try:
        mov.FORCE_NUMPY = True
        r1 = run_ensemble(cfg, n_replicates=2, n_agents=20, n_steps=5,
                          n_workers=1, base_seed=42)
        r2 = run_ensemble(cfg, n_replicates=2, n_agents=20, n_steps=5,
                          n_workers=1, base_seed=42)
    finally:
        mov.FORCE_NUMPY = orig

    for a, b in zip(r1, r2):
        assert a["seed"] == b["seed"]
        for ha, hb in zip(a["history"], b["history"]):
            assert ha["n_alive"] == hb["n_alive"]
            assert abs(ha["mean_ed"] - hb["mean_ed"]) < 1e-10
```

- [ ] **Step 4: Implement ensemble runner**

Create `salmon_ibm/ensemble.py`:

```python
"""Multiprocessing ensemble runner for replicate simulations."""
from __future__ import annotations

from multiprocessing import Pool
from typing import Any

import numpy as np


def _run_single_replicate(args: tuple) -> dict[str, Any]:
    """Run one replicate — must be a top-level function for pickling."""
    config, n_agents, n_steps, seed = args
    from salmon_ibm.simulation import Simulation
    sim = Simulation(config, n_agents=n_agents, rng_seed=seed)
    sim.run(n_steps)
    return {
        "seed": seed,
        "history": sim.history,
        "n_alive": int(sim.pool.alive.sum()),
        "n_arrived": int(sim.pool.arrived.sum()),
    }


def run_ensemble(
    config: dict,
    n_replicates: int = 10,
    n_agents: int = 1000,
    n_steps: int = 100,
    n_workers: int | None = None,
    base_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run multiple simulation replicates in parallel.

    Parameters
    ----------
    config : simulation config dict (same as Simulation.__init__ expects).
    n_replicates : number of independent replicates.
    n_agents : agents per replicate.
    n_steps : timesteps per replicate.
    n_workers : number of parallel processes (None = os.cpu_count()).
    base_seed : deterministic seed generator. If None, uses random seeds.

    Returns
    -------
    List of result dicts, each with keys: seed, history, n_alive, n_arrived.
    """
    rng = np.random.default_rng(base_seed)
    seeds = [int(rng.integers(2**31)) for _ in range(n_replicates)]

    args_list = [(config, n_agents, n_steps, s) for s in seeds]

    if n_workers == 1:
        # Sequential — useful for debugging and deterministic tests
        return [_run_single_replicate(a) for a in args_list]

    with Pool(processes=n_workers) as pool:
        results = pool.map(_run_single_replicate, args_list)

    return results
```

- [ ] **Step 5: Run ensemble tests**

Run: `conda run -n shiny python -m pytest tests/test_ensemble.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/ensemble.py tests/test_ensemble.py
git commit -m "feat(ensemble): add multiprocessing ensemble runner for parallel replicates"
```

---

## Task 5: Performance Regression Test Update

**Files:**
- Modify: `tests/test_perf.py`
- Create: `scripts/profile_perf.py` (already created, update)

Update the performance regression tests to tighten thresholds based on the new optimizations, and add a 100K-agent HexSim-specific test.

- [ ] **Step 1: Add 100K HexSim perf test**

Add to `tests/test_perf.py`:

```python
@pytest.mark.slow
def test_hexsim_100k_agents_under_half_second_per_step():
    """100K agents on Columbia workspace should run < 0.5s/step.

    Measures end-to-end performance with config_columbia.yaml settings
    (estuary overrides disabled via high thresholds).
    """
    import os
    if not os.path.exists("Columbia River Migration Model/Columbia [small]"):
        pytest.skip("Columbia workspace not found")
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation

    cfg = load_config("config_columbia.yaml")
    # Warmup Numba
    sim_warmup = Simulation(cfg, n_agents=10, rng_seed=99)
    sim_warmup.run(1)

    cfg2 = load_config("config_columbia.yaml")
    sim = Simulation(cfg2, n_agents=100_000, rng_seed=42)
    n_steps = 10
    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0
    per_step = elapsed / n_steps
    print(f"\n  HexSim 100K: {per_step:.4f} s/step")
    assert per_step < 0.5, f"{per_step:.3f}s/step exceeds 0.5s target"
    assert sim.pool.alive.sum() > 0
```

- [ ] **Step 2: Run perf tests**

Run: `conda run -n shiny python -m pytest tests/test_perf.py -v -m slow --tb=short`
Expected: All PASS (may be skipped if data files not present)

- [ ] **Step 3: Run the full profiling script to confirm improvements**

Run: `conda run -n shiny python scripts/profile_perf.py`
Expected: `advance()` should no longer dominate; movement should be the primary cost.

- [ ] **Step 4: Commit**

```bash
git add tests/test_perf.py scripts/profile_perf.py
git commit -m "test(perf): add 100K HexSim perf test, update profiling script"
```

---

## Task 6: Final Integration Verification

- [ ] **Step 1: Run full test suite**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_playwright.py`
Expected: All 382+ tests PASS (new tests added)

- [ ] **Step 2: Run profiling to confirm improvement**

Run: `conda run -n shiny python scripts/profile_perf.py`
Expected output should show:
- `advance()` no longer dominant (< 10% of runtime)
- `estuarine_overrides` near zero for Columbia config
- 100K agents < 0.5s/step

- [ ] **Step 3: Commit all if not already committed**

```bash
git status
# Commit any remaining changes
```
