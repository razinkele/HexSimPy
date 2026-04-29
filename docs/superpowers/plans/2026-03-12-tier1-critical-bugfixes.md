# Tier 1 Critical Bug Fixes Implementation Plan

> **STATUS: ✅ EXECUTED** — Bioenergetics dome removed, energy double-counting fixed, time-to-spawn axis corrected, CWR counter + RNG seed + vectorized dSSH/dt all shipped. Scope absorbed by `2026-03-21-tier1-3-codebase-fixes.md`.

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 6 critical bugs that affect scientific validity of the Baltic salmon IBM simulation results.

**Architecture:** Each fix is isolated to 1-2 files with targeted test-first development. Fixes are ordered so earlier fixes don't break later ones. All fixes maintain backward compatibility with existing tests.

**Tech Stack:** Python 3, NumPy, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `salmon_ibm/simulation.py` | Modify | Add CWR counter updates, propagate RNG seed, scope bioenergetics to alive agents |
| `salmon_ibm/bioenergetics.py` | Modify | Fix dome modifier, fix energy double-counting, fix return type annotation |
| `salmon_ibm/behavior.py` | Modify | Fix time-to-spawn axis inversion |
| `salmon_ibm/environment.py` | Modify | Vectorize dSSH_dt for seiche threshold fix |
| `tests/test_bioenergetics.py` | Modify | Add regression tests for dome fix and energy accounting |
| `tests/test_behavior.py` | Modify | Add time-to-spawn sensitivity test |
| `tests/test_simulation.py` | Modify | Add CWR counter test, reproducibility test |

---

## Chunk 1: Bioenergetics Fixes

### Task 1: Fix respiration dome to not suppress R above T_OPT

The current dome `1 - ((T - T_OPT)/(T_MAX - T_OPT))^2` is symmetric and suppresses respiration both above T_OPT (biologically backwards for non-feeding fish) and below T=6C (biologically wrong — fish still respire in cold water). The fix: remove the dome entirely. For non-feeding migrants, respiration should increase monotonically with temperature via the exponential `exp(RQ * T)`. Thermal mortality at T > T_MAX handles the lethal endpoint independently.

**Files:**
- Modify: `salmon_ibm/bioenergetics.py:31-33`
- Modify: `tests/test_bioenergetics.py`

- [ ] **Step 1: Write failing test — respiration increases monotonically with temperature**

Add to `tests/test_bioenergetics.py`:

```python
def test_respiration_monotonically_increases_with_temperature():
    """Respiration should increase with temperature across the full range.
    For non-feeding migrants, there is no consumption dome — only the
    exponential R(T) from the Wisconsin model applies."""
    p = BioParams()
    mass = np.array([3000.0])
    temps = [5.0, 10.0, 16.0, 20.0, 24.0, 26.0]
    resps = [float(hourly_respiration(mass, np.array([t]), np.array([1.0]), p)) for t in temps]
    for i in range(len(resps) - 1):
        assert resps[i + 1] > resps[i], (
            f"R({temps[i+1]}) = {resps[i+1]:.4f} should exceed R({temps[i]}) = {resps[i]:.4f}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_bioenergetics.py::test_respiration_monotonically_increases_with_temperature -v`

Expected: FAIL — current dome suppresses R above T_OPT and below 6C.

- [ ] **Step 3: Remove dome from respiration**

In `salmon_ibm/bioenergetics.py`, replace lines 31-33:

```python
# OLD:
    r_daily = params.RA * np.power(mass_g, params.RB) * np.exp(params.RQ * temperature_c) * activity_mult
    dome = np.clip(1.0 - ((temperature_c - params.T_OPT) / (params.T_MAX - params.T_OPT)) ** 2, 0.0, 1.0)
    return r_daily * dome * OXY_CAL_J_PER_GO2 * mass_g / 24.0

# NEW:
    r_daily = params.RA * np.power(mass_g, params.RB) * np.exp(params.RQ * temperature_c) * activity_mult
    return r_daily * OXY_CAL_J_PER_GO2 * mass_g / 24.0
```

- [ ] **Step 4: Update existing tests that assumed dome behavior**

The following tests assumed the dome and must be updated:

In `tests/test_bioenergetics.py`, **remove** `test_respiration_peaks_near_t_opt` and `test_respiration_zero_at_t_max` — these tested dome behavior that is now removed. The new monotonicity test replaces them.

- [ ] **Step 5: Run all bioenergetics tests**

Run: `conda run -n shiny python -m pytest tests/test_bioenergetics.py -v`

Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/bioenergetics.py tests/test_bioenergetics.py
git commit -m "fix: remove respiration dome — R(T) now monotonically increases with temperature

The parabolic dome suppressed respiration above T_OPT and below 6C,
which is biologically incorrect for non-feeding migrants. Metabolic
cost should increase with temperature; thermal mortality at T > T_MAX
handles the lethal endpoint independently.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Fix energy double-counting in update_energy

Currently, respiration cost R is subtracted from the energy pool AND used to reduce mass. This removes ~2R joules from the system per timestep. The correct approach: subtract R from total energy, then recompute mass from the new total energy and the tissue energy density.

**Files:**
- Modify: `salmon_ibm/bioenergetics.py:36-51`
- Modify: `tests/test_bioenergetics.py`

- [ ] **Step 1: Write failing test — energy conservation**

Add to `tests/test_bioenergetics.py`:

```python
def test_energy_conservation_single_step():
    """Total energy lost should equal respiration cost (no double-counting).
    energy_before - energy_after == R_hourly (within floating point tolerance)."""
    p = BioParams()
    ed = np.array([6.5])
    mass = np.array([3000.0])
    temps = np.array([15.0])
    activity = np.array([1.0])
    sal_cost = np.array([1.0])

    e_before = ed[0] * 1000.0 * mass[0]  # total energy in J
    r_hourly = float(hourly_respiration(mass, temps, activity, p) * sal_cost)

    new_ed, dead, new_mass = update_energy(ed, mass, temps, activity, sal_cost, p)
    e_after = new_ed[0] * 1000.0 * new_mass[0]  # total energy in J

    energy_lost = e_before - e_after
    assert energy_lost == pytest.approx(r_hourly, rel=1e-6), (
        f"Energy lost ({energy_lost:.2f} J) should equal respiration ({r_hourly:.2f} J)"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_bioenergetics.py::test_energy_conservation_single_step -v`

Expected: FAIL — double-counting causes energy_lost > r_hourly.

- [ ] **Step 3: Fix update_energy to use consistent energy accounting**

In `salmon_ibm/bioenergetics.py`, replace the `update_energy` function (lines 36-51):

```python
def update_energy(
    ed_kJ_g: np.ndarray,
    mass_g: np.ndarray,
    temperature_c: np.ndarray,
    activity_mult: np.ndarray,
    salinity_cost: np.ndarray,
    params: BioParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_hourly = hourly_respiration(mass_g, temperature_c, activity_mult, params) * salinity_cost
    e_total_j = ed_kJ_g * 1000.0 * mass_g
    e_total_j = np.maximum(e_total_j - r_hourly, 0.0)
    # Mass lost = energy respired / energy density of catabolized tissue
    mass_loss_g = r_hourly / (params.ED_TISSUE * 1000.0)
    new_mass = np.maximum(mass_g - mass_loss_g, mass_g * 0.5)
    # Recompute ED using new mass (consistent accounting)
    new_ed = np.where(new_mass > 0, e_total_j / (new_mass * 1000.0), 0.0)
    dead = new_ed < params.ED_MORTAL
    return new_ed, dead, new_mass
```

Key changes:
1. Return type annotation fixed: `tuple[np.ndarray, np.ndarray, np.ndarray]` (3-tuple, not 2-tuple)
2. `new_ed` now divides by `new_mass` (not stale `mass_g`)
3. Guard against zero mass with `np.where`

- [ ] **Step 4: Run all bioenergetics tests**

Run: `conda run -n shiny python -m pytest tests/test_bioenergetics.py -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/bioenergetics.py tests/test_bioenergetics.py
git commit -m "fix: correct energy double-counting and stale mass in update_energy

Energy density now computed with new_mass (not stale mass_g), and the
total energy accounting is consistent: energy_lost == respiration_cost.
Also fixes return type annotation (3-tuple, not 2-tuple) and adds a
zero-mass guard.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Fix T_MAX boundary — thermal mortality at >= T_MAX

At exactly T = T_MAX (26C), respiration is nonzero (after dome removal) but thermal mortality uses strict `>`. Change to `>=` so fish at exactly T_MAX are killed.

**Files:**
- Modify: `salmon_ibm/simulation.py:112`
- Modify: `tests/test_simulation.py`

- [ ] **Step 1: Write failing test — mortality at exactly T_MAX**

Add to `tests/test_simulation.py`:

```python
def test_thermal_mortality_at_exact_t_max():
    """Fish at exactly T_MAX should die (>= not just >)."""
    from salmon_ibm.bioenergetics import BioParams
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim.bio_params = BioParams(T_MAX=26.0)
    original_advance = sim.env.advance
    def exact_tmax_advance(t):
        original_advance(t)
        sim.env.fields["temperature"][:] = 26.0
    sim.env.advance = exact_tmax_advance
    sim.step()
    assert not sim.pool.alive.any(), "All fish should die at exactly T_MAX"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py::test_thermal_mortality_at_exact_t_max -v`

Expected: FAIL — strict `>` misses T == T_MAX.

- [ ] **Step 3: Change strict > to >= in simulation.py**

In `salmon_ibm/simulation.py`, line 112:

```python
# OLD:
        thermal_kill = temps_at_agents > self.bio_params.T_MAX
# NEW:
        thermal_kill = temps_at_agents >= self.bio_params.T_MAX
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py::test_thermal_mortality_at_exact_t_max -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/simulation.py tests/test_simulation.py
git commit -m "fix: thermal mortality triggers at T >= T_MAX (was strict >)

Fish at exactly T_MAX were previously immortal — zero respiration (now
fixed) and no thermal kill. Changed to >= for consistent boundary.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Chunk 2: Behavior and Simulation Fixes

### Task 4: Fix time-to-spawn axis inversion in behavior table

`np.digitize(hours_to_spawn, [360, 720])` returns index 0 for <360h (<15 days, urgent) and index 2 for >=720h (>30 days, relaxed). But p_table[0] is commented as ">30 days" and has passive behavior (high HOLD), while p_table[2] is commented as "<15 days" and has aggressive upstream behavior. The comments match biological expectation but the indexing is inverted.

**Fix:** Reverse the sub-array order so p_table[0] = most urgent (<15 days), p_table[2] = most relaxed (>30 days). This way `np.digitize` index 0 (urgent) maps to the urgent behavior profile.

**Files:**
- Modify: `salmon_ibm/behavior.py:21-37`
- Modify: `tests/test_behavior.py`

- [ ] **Step 1: Write failing test — urgent fish should prefer UPSTREAM**

Add to `tests/test_behavior.py`:

```python
def test_urgent_fish_prefer_upstream(params):
    """Fish with < 15 days to spawn at cool temperature should
    strongly prefer UPSTREAM over HOLD."""
    n = 1000
    t3h = np.full(n, 14.0)  # cool water (below temp_bins[0]=16)
    hours = np.full(n, 100.0)  # < 360 hours = urgent
    behaviors = pick_behaviors(t3h, hours, params, seed=42)
    upstream_frac = (behaviors == Behavior.UPSTREAM).mean()
    hold_frac = (behaviors == Behavior.HOLD).mean()
    assert upstream_frac > 0.5, (
        f"Urgent fish in cool water: UPSTREAM fraction {upstream_frac:.2f} should exceed 0.5"
    )
    assert upstream_frac > hold_frac, (
        f"Urgent fish: UPSTREAM ({upstream_frac:.2f}) should exceed HOLD ({hold_frac:.2f})"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_behavior.py::test_urgent_fish_prefer_upstream -v`

Expected: FAIL — currently urgent fish map to the passive (high HOLD) sub-array.

- [ ] **Step 3: Reverse the p_table sub-array order**

In `salmon_ibm/behavior.py`, replace lines 21-37:

```python
        p = np.array([
            # <15 days to spawn — urgent: strongly UPSTREAM
            [[0.00, 0.20, 0.00, 0.80, 0.00],
             [0.00, 0.10, 0.20, 0.70, 0.00],
             [0.00, 0.00, 0.50, 0.50, 0.00],
             [0.00, 0.00, 0.60, 0.40, 0.00]],
            # 15-30 days — moderate urgency
            [[0.20, 0.20, 0.00, 0.60, 0.00],
             [0.00, 0.20, 0.30, 0.50, 0.00],
             [0.00, 0.00, 0.40, 0.40, 0.20],
             [0.00, 0.00, 0.70, 0.00, 0.30]],
            # >30 days — relaxed: more HOLD and RANDOM
            [[0.60, 0.10, 0.00, 0.30, 0.00],
             [0.40, 0.00, 0.20, 0.40, 0.00],
             [0.30, 0.00, 0.50, 0.20, 0.00],
             [0.20, 0.00, 0.80, 0.00, 0.00]],
        ])
```

This reversal aligns the array so that `np.digitize` index 0 (<15 days) maps to the urgent sub-array.

- [ ] **Step 4: Run all behavior tests**

Run: `conda run -n shiny python -m pytest tests/test_behavior.py -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/behavior.py tests/test_behavior.py
git commit -m "fix: correct time-to-spawn axis inversion in behavior probability table

np.digitize returns index 0 for <15 days (urgent) and 2 for >30 days
(relaxed). The p_table sub-arrays were in reverse order, causing urgent
fish to hold/rest and relaxed fish to aggressively migrate upstream.
Reversed the array order so urgent fish strongly prefer UPSTREAM.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Add CWR counter updates and propagate RNG seed

Two fixes in simulation.py:
1. Update `cwr_hours` and `hours_since_cwr` each step (currently dead code).
2. Pass deterministic seeds to `pick_behaviors` and `execute_movement`.

**Files:**
- Modify: `salmon_ibm/simulation.py:57-92`
- Modify: `tests/test_simulation.py`

- [ ] **Step 1: Write failing test — CWR counters increment**

Add to `tests/test_simulation.py`:

```python
def test_cwr_counters_update():
    """cwr_hours should increment for fish in TO_CWR state,
    and hours_since_cwr should reset when leaving CWR."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=5, data_dir="data", rng_seed=42)
    sim.env.advance(0)

    # Manually set agents to TO_CWR
    sim.pool.behavior[:] = Behavior.TO_CWR
    sim.pool.steps[:] = 10  # not first move
    sim.pool.cwr_hours[:] = 0
    sim.pool.hours_since_cwr[:] = 999

    # Run one step (step will reassign behaviors, but we check counter logic)
    sim.step()

    # After step, counters should have been updated at some point.
    # We directly test the counter update logic:
    # Force CWR state and call the counter update part
    sim.pool.behavior[:] = Behavior.TO_CWR
    sim.pool.cwr_hours[:] = 0
    sim.pool.hours_since_cwr[:] = 999
    sim._update_cwr_counters()
    assert np.all(sim.pool.cwr_hours == 1), "cwr_hours should increment for TO_CWR fish"
    assert np.all(sim.pool.hours_since_cwr == 0), (
        "hours_since_cwr should reset to 0 while in CWR"
    )

    # Now switch away from CWR
    sim.pool.behavior[:] = Behavior.UPSTREAM
    sim._update_cwr_counters()
    assert np.all(sim.pool.cwr_hours == 0), (
        "cwr_hours should reset to 0 when fish leave CWR"
    )
    assert np.all(sim.pool.hours_since_cwr == 1), (
        "hours_since_cwr should increment to 1 after leaving CWR"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py::test_cwr_counters_update -v`

Expected: FAIL — `_update_cwr_counters` does not exist.

- [ ] **Step 3: Write failing test — simulation reproducibility**

Add to `tests/test_simulation.py`:

```python
def test_simulation_reproducibility():
    """Same rng_seed should produce identical results across two runs."""
    cfg = load_config("config_curonian_minimal.yaml")

    sim1 = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim1.run(n_steps=5)
    ed1 = sim1.pool.ed_kJ_g.copy()
    tri1 = sim1.pool.tri_idx.copy()

    sim2 = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim2.run(n_steps=5)
    ed2 = sim2.pool.ed_kJ_g.copy()
    tri2 = sim2.pool.tri_idx.copy()

    np.testing.assert_array_equal(tri1, tri2, "Positions should be identical with same seed")
    np.testing.assert_array_almost_equal(ed1, ed2, decimal=10,
        err_msg="Energy should be identical with same seed")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py::test_simulation_reproducibility -v`

Expected: FAIL — `seed=None` in pick_behaviors and execute_movement.

- [ ] **Step 5: Implement CWR counter updates and RNG seed propagation**

In `salmon_ibm/simulation.py`, add a `_update_cwr_counters` method and modify `step()`:

```python
class Simulation:
    def __init__(self, config, n_agents=100, data_dir="data", rng_seed=None, output_path=None):
        # ... existing code ...
        self._rng = np.random.default_rng(rng_seed)
        # ... rest unchanged ...

    def _update_cwr_counters(self):
        """Update CWR tracking counters based on current behavior state."""
        in_cwr = self.pool.behavior == Behavior.TO_CWR
        not_in_cwr = ~in_cwr

        # Fish in CWR: increment cwr_hours, reset hours_since_cwr
        self.pool.cwr_hours[in_cwr] += 1
        self.pool.hours_since_cwr[in_cwr] = 0

        # Fish not in CWR: reset cwr_hours, increment hours_since_cwr
        # (only reset cwr_hours if they were previously in CWR, i.e., cwr_hours > 0)
        was_in_cwr = not_in_cwr & (self.pool.cwr_hours > 0)
        self.pool.cwr_hours[was_in_cwr] = 0
        self.pool.hours_since_cwr[not_in_cwr] += 1
```

In `step()`, make three changes:

1. Replace `seed=None` on line 70 with `seed=int(self._rng.integers(2**31))`
2. Replace `seed=None` on line 80 with `seed=int(self._rng.integers(2**31))`
3. After the overrides (after line 77), add `self._update_cwr_counters()`

The updated step method section (lines 66-80) becomes:

```python
        # 1. Pick behaviors
        t3h = self.pool.t3h_mean()
        self.pool.behavior[alive_mask] = pick_behaviors(
            t3h[alive_mask], self.pool.target_spawn_hour[alive_mask],
            self.beh_params, seed=int(self._rng.integers(2**31)),
        )

        # 2. Apply overrides
        self.pool.behavior = apply_overrides(self.pool, self.beh_params)

        # 3. Estuarine overrides
        self._apply_estuarine_overrides()

        # 4. Update CWR counters (after all behavior decisions are final)
        self._update_cwr_counters()

        # 5. Movement
        execute_movement(self.pool, self.mesh, self.env.fields,
                         seed=int(self._rng.integers(2**31)))
```

Also in `__init__`, replace the RNG setup. Change lines 43-44 to derive both the pool RNG and simulation RNG from a single seed sequence:

```python
        base_rng = np.random.default_rng(rng_seed)
        # Derive independent RNG streams to avoid correlated sequences
        rng = np.random.default_rng(base_rng.integers(2**63))
        self._rng = np.random.default_rng(base_rng.integers(2**63))
        start_tris = rng.choice(water_ids, size=n_agents)
```

This replaces the old `rng = np.random.default_rng(rng_seed)` on line 43.

- [ ] **Step 6: Run both new tests**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py::test_cwr_counters_update tests/test_simulation.py::test_simulation_reproducibility -v`

Expected: PASS

- [ ] **Step 7: Run all simulation tests**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py -v`

Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add salmon_ibm/simulation.py tests/test_simulation.py
git commit -m "fix: add CWR counter updates and propagate RNG seed for reproducibility

CWR counters (cwr_hours, hours_since_cwr) were initialized but never
updated, making the CWR override logic dead code. Now cwr_hours
increments while in TO_CWR state, hours_since_cwr resets on exit.

Also propagates deterministic RNG seeds from the simulation-level RNG
to pick_behaviors and execute_movement, ensuring reproducible results.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Fix seiche threshold units — vectorize dSSH_dt

The config key says "m per 15 min" but `dSSH_dt()` computes m/hour (raw difference between hourly timesteps). Fix by renaming the config key and vectorizing the dSSH lookup (currently a Python loop).

**Files:**
- Modify: `salmon_ibm/environment.py:62-65`
- Modify: `salmon_ibm/simulation.py:137-139`
- Modify: `tests/test_simulation.py`

- [ ] **Step 1: Write failing test — dSSH_dt_array matches per-element dSSH_dt**

Add to `tests/test_simulation.py`:

```python
def test_dssh_dt_array_matches_scalar():
    """Vectorized dSSH_dt_array should match per-element dSSH_dt calls."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=5, data_dir="data", rng_seed=42)
    sim.env.advance(0)
    sim.env.advance(1)  # need two steps for dSSH_dt to be nonzero
    arr = sim.env.dSSH_dt_array()
    for i in range(min(10, sim.mesh.n_triangles)):
        scalar = sim.env.dSSH_dt(i)
        assert arr[i] == pytest.approx(scalar), f"Mismatch at triangle {i}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py::test_dssh_dt_array_matches_scalar -v`

Expected: FAIL — `dSSH_dt_array` does not exist yet.

- [ ] **Step 3: Vectorize dSSH_dt to return full array**

In `salmon_ibm/environment.py`, add a new method (keep old `dSSH_dt` for backward compat):

```python
    def dSSH_dt_array(self) -> np.ndarray:
        """Rate of SSH change (m/timestep) for all triangles."""
        if self._prev_ssh is None:
            return np.zeros(self.mesh.n_triangles)
        return self.fields["ssh"] - self._prev_ssh
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py::test_dssh_dt_array_matches_scalar -v`

Expected: PASS

- [ ] **Step 5: Update simulation.py to use vectorized dSSH and rename config key**

In `salmon_ibm/simulation.py`, replace lines 137-139:

```python
# OLD:
        dSSH = np.array([self.env.dSSH_dt(int(tri)) for tri in self.pool.tri_idx])

# NEW:
        dSSH_all = self.env.dSSH_dt_array()
        dSSH = dSSH_all[self.pool.tri_idx] if len(dSSH_all) > 0 else np.zeros(self.pool.n)
```

Also update the config key lookup on line 137:

```python
# OLD:
        thresh = seiche_cfg.get("dSSHdt_thresh_m_per_15min", 0.02)
# NEW:
        thresh = seiche_cfg.get("dSSHdt_thresh_m_per_hour",
                                seiche_cfg.get("dSSHdt_thresh_m_per_15min", 0.02))
```

This accepts the new key name but falls back to the old key for backward compatibility.

- [ ] **Step 6: Run all simulation tests**

Run: `conda run -n shiny python -m pytest tests/test_simulation.py -v`

Expected: ALL PASS

- [ ] **Step 7: Run the full test suite**

Run: `conda run -n shiny python -m pytest tests/ -v`

Expected: ALL PASS (confirming no regressions across the entire codebase)

- [ ] **Step 8: Commit**

```bash
git add salmon_ibm/environment.py salmon_ibm/simulation.py
git commit -m "fix: vectorize dSSH_dt and rename seiche threshold config key

Replaced per-agent Python loop for dSSH_dt with vectorized array
operation. Renamed config key from dSSHdt_thresh_m_per_15min to
dSSHdt_thresh_m_per_hour to match actual units (difference between
hourly SSH snapshots). Old key name still accepted for backward compat.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Final Verification

- [ ] **Run entire test suite to confirm no regressions**

```bash
conda run -n shiny python -m pytest tests/ -v
```

Expected: ALL PASS

- [ ] **Quick smoke test — run simulation for 24 steps**

```bash
conda run -n shiny python -c "
from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation
cfg = load_config('config_curonian_minimal.yaml')
sim = Simulation(cfg, n_agents=50, data_dir='data', rng_seed=42)
sim.run(24)
print(f'After 24h: {sim.pool.alive.sum()}/50 alive, mean ED={sim.pool.ed_kJ_g[sim.pool.alive].mean():.2f} kJ/g')
print(f'CWR hours range: {sim.pool.cwr_hours.min()}-{sim.pool.cwr_hours.max()}')
"
```

Expected: Simulation completes without error, alive count reasonable, CWR hours may be > 0 if any fish entered CWR.
