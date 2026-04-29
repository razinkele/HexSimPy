# H3 Grid Rebuild — Connectivity-First, Three-Tier Resolution

> **STATUS: ✅ EXECUTED** — Three-tier H3 rebuild (rivers res 11 / lagoon res 10 / Baltic res 9) shipped as **v1.5.0**. Polygon-dilation tessellation, bridge-cell pass, resolution-aware `n_micro_steps`, area-weighted IntroductionEvent. Three carry-forward limits documented in memory `curonian_h3_grid_state.md` (find_cross_res spurious links, Atmata 2 components, OpenBaltic Swiss-cheese) — accepted, not unfinished.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the H3 multi-resolution landscape so that (a) every inSTREAM reach forms a single connected H3 cell graph (no isolated islands), (b) cross-reach transitions have enough cell-edges to be ecologically plausible (e.g., Nemunas ↔ Atmata ≥ 200 links, not the current 4), (c) cell sizes match the physical scale hierarchy: rivers res 11 (28 m), Curonian Lagoon res 10 (76 m), Baltic res 9 (201 m).

**Diagnostic baseline (v1.4.0, before rebuild):**

```
Reach           cells   cell_area  poly_area   gap   #components
Nemunas         1,319    16.94      16.71     -1.4%   5 / 1319
Atmata            152     1.94       1.97     +1.4%   1 / 152   ✓
Minija            219     2.78       3.02     +7.9%   135 / 219  ☠
Sysa              153     1.96       1.83     -6.6%   60 / 153   ☠
Skirvyte          201     2.57       2.50     -2.7%   9 / 201
Leite             102     1.31       1.32     +1.0%   58 / 102   ☠
Gilija            557     7.17       7.30     +1.9%   8 / 557
CuronianLagoon  17,264  1551.5    1559.2     +0.5%   2 / 17 264
BalticCoast      9,663   852.8      864.0     +1.3%   2 / 9 663
OpenBaltic       6,788     —          —        —     6 / 6 788
```

The problem isn't *area*: cell-area sums match polygon-area within 8%. The problem is *topology*: a thin meandering river polygon at H3 resolution 10 (~75 m cells) yields a sparse cell set whose members aren't each other's H3 grid-ring neighbours. Minija's 219 cells fragment into 135 islands. Cross-reach links (`BalticCoast ↔ CuronianLagoon = 34`) are too few for the geographic adjacency they represent.

**Architecture decision — three-tier resolution scheme (user-chosen):**

| Reach group | Resolution | Edge length | Approx cells | Rationale |
|---|---:|---:|---:|---|
| **Rivers** (7 reaches) | **11** | ~28 m | ~19 000 | Channel widths 50–200 m → 1–7 cells across. Self-tessellates connectedly at this scale. |
| **CuronianLagoon** | **10** | ~76 m | ~120 000 | Resolves spatial heterogeneity in mortality, temperature, and salinity gradients across the lagoon's 1 600 km². |
| **BalticCoast + OpenBaltic** | **9** | ~201 m | ~60 000 | Uniform open-water resolution removes the v1.4.0 res-9/res-8 boundary discontinuity. Mesoscale circulation doesn't need finer cells. |
| **Total** | | | **~200 000** | (was ~36 k in v1.4.0; 5.5× growth) |

This scheme matches the physical scale hierarchy: rivers (10s of m channel widths) → lagoon (km-scale circulation) → Baltic (10s of km mesoscale). Each domain gets the cell size appropriate to its dominant processes.

Connectivity fixes (independent of resolution choice):
* **Polygon dilation** by half a cell edge (14 m at res 11, 38 m at res 10, 100 m at res 9) before `h3.polygon_to_cells` so thin rivers tessellate as a connected swath.
* **Bridge-cell pass**: after tessellation, find disconnected components per reach and add the shortest H3 path between them (via `h3.grid_path_cells`) so each reach forms a single connected graph.
* **Cross-reach link audit**: at the boundary between two reaches, ensure ≥1 same-resolution cell-edge exists; if not, log a warning (cross-resolution algorithm in `find_cross_res_neighbours` should still bridge them, but a same-res link is more robust).
* **Movement-kernel calibration** (Task 0.5): make `n_micro_steps` per-cell so finer river cells get more cell-hops per simulation step, holding physical swim distance constant.
* **Area-weighted introduction** (Task 0.6): one-line fix so initial agent placement is proportional to cell area, not uniform-over-cells.

**Tech stack:** Python 3.13, h3-py 4.4.2, geopandas, shapely (especially `polygon.buffer()` and `is_valid`), numpy, pytest.

---

## Cross-resolution adjacencies

Three resolutions in use means three drop=1 boundary types:

| From → To | Where | Drop |
|---|---|---:|
| Rivers (res 11) → CuronianLagoon (res 10) | River mouths into lagoon (Nemunas delta, Minija, Sysa, Skirvytė, Atmata, Leitė, Gilija) | 1 |
| CuronianLagoon (res 10) → BalticCoast (res 9) | Klaipėda strait (~400 m wide opening) | 1 |
| BalticCoast (res 9) → OpenBaltic (res 9) | Same resolution — direct H3 adjacency | 0 |

No drop=2 transitions exist geographically (no river flows directly into the open Baltic — all 7 inSTREAM rivers terminate in the lagoon). The existing `find_cross_res_neighbours` algorithm handles drop=1 already; no code change needed there.

The Klaipėda strait at ~400 m width is the connectivity bottleneck. At res 10 (76 m edges) the lagoon side has ~5 cells across the strait; at res 9 (201 m edges) the Baltic side has ~2 cells. Cross-res linking should produce ≥ 10 lagoon↔Baltic connections at this bottleneck — Step 4.2b validates this explicitly.

---

## Pre-flight checks

- [ ] **Step 0.1: Verify the current state.**

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" log --oneline -5
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" status
```

Expected: working tree may have several uncommitted hexsim/scripts artefacts — that's fine, but **do not** start the rebuild with uncommitted edits to `salmon_ibm/` or `scripts/build_h3_multires_landscape.py` (the files this plan modifies).  If those are dirty, stash them or commit before proceeding.

- [ ] **Step 0.2: Snapshot diagnostic numbers for the rebuild commit message.**

```bash
micromamba run -n shiny python "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/_diag_grid_quality.py" 2>&1 | tee /tmp/grid_baseline.txt
```

Expected: shows the pre-rebuild metrics from this plan's introduction (5/1319, 135/219, etc.).  Save the output for the v1.5.0 commit message.

(If `_diag_grid_quality.py` was deleted, recreate it from the snippet in `docs/h3-multires-roadmap.md` — Phase 2 left a copy.)

---

## Task 0.5: Resolution-aware `n_micro_steps` (movement-kernel calibration)

**Why this is here, not deferred to v1.6.0:**

Scientific-validator audit (2026-04-26) flagged that the movement kernel hard-codes `n_micro_steps = 3` cell-hops per simulation step regardless of cell size.  At res 11 (28 m edges), max realisable speed = `4 × 28 / 3600 ≈ 0.031 m/s`.  Migrating smolt cruise speed is 0.5–2 m/s, so river transit is **16–64× too slow** — Nemunas main-stem (~110 km) at 0.031 m/s implies 41 days vs literature 3–5 days.  The artefact already exists at res 10 (5–25× slow) and res 9 (2–9× slow) but res 11 makes it 2.7× worse.

The fix decouples per-cell hop count from the resolution by computing `n_micro_steps[c] = round(SWIM_SPEED_M_S × dt_s / cell_edge_m[c])` once at simulation init.  Movement kernels then use this per-cell budget instead of a single scalar.

**Files:**
- Modify: `salmon_ibm/movement.py:34-133` (`execute_movement` signature + per-call dispatch)
- Modify: `salmon_ibm/movement.py:136-` (`_step_random_numba`, `_step_directed_numba`, `_step_to_cwr_numba`) — add `n_micro_per_cell` parameter and per-iteration budget check
- Modify: `salmon_ibm/movement.py:257-` (the `_vec` wrappers + NumPy fallback paths) — same change
- Modify: `salmon_ibm/simulation.py:299-303` — compute `_n_micro_steps_per_cell` at init in `_build_event_sequence` (or wherever the mesh is first accessible), inject into landscape dict at line ~466 where the dict is constructed
- Modify: `salmon_ibm/simulation.py` `Landscape` TypedDict (lines 9-26 region) — add `n_micro_steps_per_cell: np.ndarray` key
- Modify: `salmon_ibm/events_builtin.py:16-37` — `MovementEvent.execute` reads `n_micro_steps_per_cell` from landscape, falls back to scalar broadcast
- Test: `tests/test_movement_resolution_aware.py` (new file)

**Naming convention used throughout this task:**
- `self._n_micro_steps_per_cell` — private cache on the `Simulation` instance (Step 0.5.1)
- `landscape["n_micro_steps_per_cell"]` — landscape-dict key that flows to events (Step 0.5.1)
- `n_micro_steps_per_cell` — public parameter name on `execute_movement` (Step 0.5.4)
- `n_micro_per_cell` — short alias for the parameter inside Numba kernels (Step 0.5.3)

- [ ] **Step 0.5.1: Compute `_n_micro_steps_per_cell` at sim init.**

In `salmon_ibm/simulation.py`, add the computation in `Simulation.__init__` immediately after `self.population = Population(...)` (current line 189) — by then all four mesh-construction branches (lines 76, 127, 146, 159) have run and `self.mesh` exists.

Add module-level constants near the top of the file (after the existing imports, around line 50; module-level so the test file can import them):

```python
SWIM_SPEED_M_S = 1.0   # midpoint of 0.5-2 m/s migrating-smolt range
DT_S = 3600.0          # one hour per simulation step
MAX_N_MICRO_STEPS = 256  # ceiling for budget — at res 11 (~28 m) this
                         # equals ≈ 7.2 km/step which exceeds any
                         # plausible smolt swim speed, so it's an
                         # effective no-op upper bound.
```

(Don't import `h3` at module level — the existing `__init__` already does `import h3 as _h3` lazily inside the H3-backend branches at lines 70 and 93.  The block below uses a local lazy import so non-H3 backends never pull in the dependency.)

Inside `__init__`, after `self.population` is assigned (around line 190):

```python
# Per-cell number of cell-hops budget.  Equal to physical distance
# travelled (SWIM_SPEED_M_S × DT_S) divided by cell edge length.
# Falls back to 3 (legacy default) if mesh has no per-cell resolution.
if hasattr(self.mesh, "resolutions"):
    import h3 as _h3   # local; module-level not desired (see above)
    edge_m = np.array(
        [_h3.average_hexagon_edge_length(int(r), unit="m")
         for r in self.mesh.resolutions],
        dtype=np.float32,
    )
    n_micro = np.round(SWIM_SPEED_M_S * DT_S / np.maximum(edge_m, 1.0))
    self._n_micro_steps_per_cell = np.clip(
        n_micro, 1, MAX_N_MICRO_STEPS
    ).astype(np.int32)
else:
    self._n_micro_steps_per_cell = np.full(
        self.mesh.n_cells, 3, dtype=np.int32,
    )
```

Then wire it into the landscape dict at `simulation.py:464` (inside the `step()` method, in the dict literal):

```python
landscape: Landscape = {
    "mesh": self.mesh,
    "fields": self.env.fields,
    "rng": self._rng,
    "activity_lut": self._activity_lut,
    "est_cfg": self.est_cfg,
    "barrier_arrays": self._barrier_arrays,
    "genome": self._genome,
    "multi_pop_mgr": self._multi_pop_mgr,
    "network": self._network,
    "n_micro_steps_per_cell": self._n_micro_steps_per_cell,   # NEW
}
```

Also update the `Landscape` TypedDict at `simulation.py:9-26` to declare the new key:

```python
class Landscape(TypedDict, total=False):
    """Typed dict passed to every event. All keys are optional (total=False)."""

    mesh: object  # TriMesh | HexMesh
    fields: dict[str, np.ndarray]
    rng: np.random.Generator
    # ...existing keys...
    n_micro_steps_per_cell: np.ndarray   # NEW: per-cell hop budget
```

- [ ] **Step 0.5.2: Plumb the budget through `MovementEvent`.**

`salmon_ibm/events_builtin.py:16-37` — modify `MovementEvent.execute` to read the per-cell array from landscape.  **Preserve** the `@register_event("movement")` decorator and the `cwr_threshold: float = 16.0` field.  Replace ONLY the `execute` method body:

```python
@register_event("movement")
@dataclass
class MovementEvent(Event):
    """Wraps execute_movement() as an event."""

    n_micro_steps: int = 3   # legacy fallback for uniform-res meshes
    cwr_threshold: float = 16.0

    def execute(self, population, landscape, t, mask):
        mesh = landscape["mesh"]
        fields = landscape["fields"]
        rng = landscape["rng"]
        barrier_arrays = landscape.get("barrier_arrays")
        n_micro = landscape.get("n_micro_steps_per_cell")
        if n_micro is None:
            # Legacy path: scalar broadcast to per-cell so the kernel
            # signature stays uniform.
            n_micro = np.full(
                mesh.n_cells, self.n_micro_steps, dtype=np.int32,
            )
        execute_movement(
            population,
            mesh,
            fields,
            seed=int(rng.integers(2**31)),
            n_micro_steps_per_cell=n_micro,
            cwr_threshold=self.cwr_threshold,
            barrier_arrays=barrier_arrays,
        )
```

- [ ] **Step 0.5.3: Modify the three Numba kernels with a per-agent fractional swim-budget.**

**Algorithm — per-agent fractional budget (NOT per-iteration cell check).**

The naive "if `s >= n_micro_per_cell[current_cell]` skip" pattern is **physically wrong** at resolution boundaries.  An agent that spends 9 hops in res-9 cells (each consuming 1/18 of its swim-distance budget) and then crosses into a res-11 cell (budget 128 hops) would compare its now-current iteration `s=9` against the new cell's budget 128 — `9 < 128` → keeps moving — and would re-spend its already-consumed swim time at the finer resolution.  Net result: agent over-swims by ~40 % at every coarse→fine boundary crossing.

The correct approach: each agent has a **fraction-remaining** scalar `f[i] ∈ [0, 1]` initialised to 1.0 (full budget).  Every hop in cell `c` consumes `1 / n_micro_per_cell[c]` of that budget.  The agent stops when `f[i] ≤ 0`, regardless of which iteration `s` we're in.  This conserves physical swim distance across resolution transitions:

* Agent stays in res-9 only: 18 hops × (1/18) = 1.0 → stops at hop 18. Distance = 18 × 201 m = 3.6 km. ✓
* Agent stays in res-11 only: 128 × (1/128) = 1.0 → stops at hop 128. Distance = 128 × 28 m = 3.6 km. ✓
* Agent does 9 res-9 hops (used 9/18 = 0.5), then crosses to res-11 (each costs 1/128): runs for 0.5 / (1/128) = 64 more hops. Total distance = 9 × 201 + 64 × 28 = 1.81 + 1.79 = 3.60 km. ✓

`max_steps` is still `int(n_micro_steps_per_cell.max())` — the upper bound on iterations so agents that stay in fine cells get to use their full budget.  But now each agent stops when `f[i] ≤ 0`, not when `s` exceeds some cell's budget.

Replace `_step_random_numba` at `salmon_ibm/movement.py:136-148`:

```python
@njit(cache=True, parallel=True)
def _step_random_numba(
    tri_indices, water_nbrs, water_nbr_count, rand_vals,
    max_steps, n_micro_per_cell, fraction_remaining,
):
    """Random walk with per-agent fractional swim budget.

    fraction_remaining[i] starts at 1.0 (full timestep budget).  Each
    hop in cell c subtracts 1.0/n_micro_per_cell[c] — i.e. the fraction
    of the timestep that one cell-hop in this resolution represents.
    Agent stops when fraction_remaining[i] ≤ 0.

    This conserves physical swim distance across resolution boundaries:
    an agent that uses half its budget in coarse cells and crosses into
    fine cells will use the *remaining* half-budget at the fine cell's
    cost, not get a fresh full budget at the new resolution.
    """
    n = len(tri_indices)
    for s in range(max_steps):
        for i in prange(n):
            if fraction_remaining[i] <= 0.0:
                continue
            c = tri_indices[i]
            cnt = water_nbr_count[c]
            if cnt > 0:
                idx = int(rand_vals[s, i] * cnt)
                if idx >= cnt:
                    idx = cnt - 1
                tri_indices[i] = water_nbrs[c, idx]
                fraction_remaining[i] -= 1.0 / n_micro_per_cell[c]
```

Replace `_step_directed_numba` at `salmon_ibm/movement.py:150-180`:

```python
@njit(cache=True, parallel=True)
def _step_directed_numba(
    tri_indices, water_nbrs, water_nbr_count, field, rand_vals,
    max_steps, n_micro_per_cell, fraction_remaining, ascending,
):
    n = len(tri_indices)
    for s in range(max_steps):
        for i in prange(n):
            if fraction_remaining[i] <= 0.0:
                continue
            c = tri_indices[i]
            cnt = water_nbr_count[c]
            if cnt <= 0:
                continue
            if s % 2 == 0:
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
                idx = int(rand_vals[s, i] * cnt)
                if idx >= cnt:
                    idx = cnt - 1
                tri_indices[i] = water_nbrs[c, idx]
            fraction_remaining[i] -= 1.0 / n_micro_per_cell[c]
```

Replace `_step_to_cwr_numba` at `salmon_ibm/movement.py:183-204`:

```python
@njit(cache=True, parallel=True)
def _step_to_cwr_numba(
    tri_indices, water_nbrs, water_nbr_count, temperature,
    max_steps, n_micro_per_cell, fraction_remaining, cwr_threshold,
):
    n = len(tri_indices)
    for s in range(max_steps):
        for i in prange(n):
            if fraction_remaining[i] <= 0.0:
                continue
            c = tri_indices[i]
            if temperature[c] < cwr_threshold:
                continue
            cnt = water_nbr_count[c]
            if cnt <= 0:
                continue
            best_nbr = water_nbrs[c, 0]
            best_temp = temperature[best_nbr]
            for k in range(1, cnt):
                nbr = water_nbrs[c, k]
                t = temperature[nbr]
                if t < best_temp:
                    best_temp = t
                    best_nbr = nbr
            tri_indices[i] = best_nbr
            fraction_remaining[i] -= 1.0 / n_micro_per_cell[c]
```

`fraction_remaining` is allocated once per `_step_*_vec` call in `execute_movement` (Step 0.5.4) using `np.ones(len(tri_buf), dtype=np.float32)` — each behaviour bucket gets its own fresh budget.

Update the NumPy fallback paths in `_step_random_vec`, `_step_directed_vec`, `_step_to_cwr_vec` (around lines 257-330) using the same pattern.  In NumPy form:

```python
# In each _step_*_vec NumPy fallback, the per-iteration loop becomes:
fraction_remaining = np.ones(len(tri_indices), dtype=np.float32)
for s in range(max_steps):
    active_mask = fraction_remaining > 0.0
    if not active_mask.any():
        break
    current = tri_indices
    counts = water_nbr_count[current]
    has_nbrs = counts > 0
    move_mask = active_mask & has_nbrs
    # ... compute chosen[move_mask] using existing logic ...
    tri_indices[move_mask] = chosen[move_mask]
    # Decrement budget by per-cell cost for agents that moved.
    fraction_remaining[move_mask] -= 1.0 / n_micro_per_cell[current[move_mask]]
```

- [ ] **Step 0.5.4: Update `execute_movement` signature and dispatch.**

Replace `salmon_ibm/movement.py:34-133`.  The four `_step_*_vec` calls (lines 73, 80-88, 95-103, 109-117 in the v1.4.0 file) each need their `n_micro_steps` int parameter replaced with `(max_steps, n_micro_steps_per_cell)`:

```python
def execute_movement(
    pool,
    mesh,
    fields,
    seed=None,
    n_micro_steps_per_cell=None,   # NEW: per-cell hop budget
    cwr_threshold=16.0,
    barrier_arrays=None,
):
    """Execute movement with optional barrier enforcement.

    Parameters
    ----------
    n_micro_steps_per_cell : np.ndarray | None
        Per-cell array of cell-hop budgets.  If None, defaults to a
        uniform 3 hops/step (legacy behaviour).
    barrier_arrays : tuple (mort, defl, trans) from BarrierMap.to_arrays(), or None
    """
    if n_micro_steps_per_cell is None:
        n_micro_steps_per_cell = np.full(
            mesh.n_cells, 3, dtype=np.int32,
        )
    max_steps = int(n_micro_steps_per_cell.max())
    rng = np.random.default_rng(seed)
    alive = pool.alive & ~pool.arrived

    water_nbrs = mesh._water_nbrs
    water_nbr_count = mesh._water_nbr_count

    pre_move = pool.tri_idx.copy() if barrier_arrays is not None else None

    alive_idx = np.where(alive)[0]
    if len(alive_idx) == 0:
        _apply_current_advection_vec(pool, mesh, fields, alive, rng)
        return
    alive_beh = pool.behavior[alive_idx]
    buckets = {}
    for b in np.unique(alive_beh):
        buckets[int(b)] = alive_idx[alive_beh == b]

    # Each behaviour bucket gets its own fresh fraction-remaining
    # budget (one entry per agent in that bucket, all initialised to
    # 1.0 = full timestep).  Passed by reference so kernels can
    # mutate in place; allocated locally so buckets don't share state.

    # --- RANDOM ---
    idx = buckets.get(int(Behavior.RANDOM))
    if idx is not None and len(idx) > 0:
        tri_buf = pool.tri_idx[idx].copy()
        fraction_remaining = np.ones(len(tri_buf), dtype=np.float32)
        _step_random_vec(
            tri_buf, water_nbrs, water_nbr_count, rng,
            max_steps, n_micro_steps_per_cell, fraction_remaining,
        )
        pool.tri_idx[idx] = tri_buf

    # --- UPSTREAM ---
    idx = buckets.get(int(Behavior.UPSTREAM))
    if idx is not None and len(idx) > 0:
        tri_buf = pool.tri_idx[idx].copy()
        fraction_remaining = np.ones(len(tri_buf), dtype=np.float32)
        _step_directed_vec(
            tri_buf, water_nbrs, water_nbr_count, fields["ssh"], rng,
            max_steps, n_micro_steps_per_cell, fraction_remaining,
            ascending=False,
        )
        pool.tri_idx[idx] = tri_buf

    # --- DOWNSTREAM ---
    idx = buckets.get(int(Behavior.DOWNSTREAM))
    if idx is not None and len(idx) > 0:
        tri_buf = pool.tri_idx[idx].copy()
        fraction_remaining = np.ones(len(tri_buf), dtype=np.float32)
        _step_directed_vec(
            tri_buf, water_nbrs, water_nbr_count, fields["ssh"], rng,
            max_steps, n_micro_steps_per_cell, fraction_remaining,
            ascending=True,
        )
        pool.tri_idx[idx] = tri_buf

    # --- TO_CWR ---
    idx = buckets.get(int(Behavior.TO_CWR))
    if idx is not None and len(idx) > 0:
        tri_buf = pool.tri_idx[idx].copy()
        fraction_remaining = np.ones(len(tri_buf), dtype=np.float32)
        _step_to_cwr_vec(
            tri_buf, water_nbrs, water_nbr_count, fields["temperature"],
            max_steps, n_micro_steps_per_cell, fraction_remaining,
            cwr_threshold,
        )
        pool.tri_idx[idx] = tri_buf

    # --- Barrier enforcement (after all movement, before advection) ---
    if barrier_arrays is not None:
        alive_idx = np.where(alive)[0]
        if len(alive_idx) > 0:
            mort, defl, trans = barrier_arrays
            current = pre_move[alive_idx]
            proposed = pool.tri_idx[alive_idx]
            final, died = _resolve_barriers_vec(
                current, proposed, mort, defl, trans, water_nbrs, rng
            )
            pool.tri_idx[alive_idx] = final
            pool.alive[alive_idx[died]] = False

    _apply_current_advection_vec(pool, mesh, fields, alive, rng)
```

Each `_step_*_vec` wrapper (e.g., `_step_random_vec` at line 257) will need its signature widened too.  The pattern: replace the existing `steps` int parameter with `(max_steps, n_micro_per_cell)` and pass them through to the Numba kernel and the NumPy fallback.

- [ ] **Step 0.5.5: Write a regression test.**

`tests/test_movement_resolution_aware.py`:

```python
"""Resolution-aware movement: agents in finer cells should hop more
cell-edges per simulation step, producing equal physical displacement
across mixed-resolution meshes."""
from __future__ import annotations
import numpy as np
import pytest


def test_n_micro_steps_per_cell_proportional_to_inverse_edge_length():
    """n_micro_steps[res11] / n_micro_steps[res9] should be ~7
    (the H3 area ratio per resolution drop is 7×, edge ratio is √7 ≈ 2.65).

    Actually the relevant ratio is edge-length-inverse: edge(res9) /
    edge(res11) ≈ 201 / 28 ≈ 7.18.  So n_micro[res11] ≈ 7 × n_micro[res9].
    """
    import h3
    SWIM_SPEED = 1.0
    DT = 3600.0
    edge_9 = h3.average_hexagon_edge_length(9, unit="m")
    edge_11 = h3.average_hexagon_edge_length(11, unit="m")
    n9 = round(SWIM_SPEED * DT / edge_9)
    n11 = round(SWIM_SPEED * DT / edge_11)
    ratio = n11 / n9
    assert 6.5 < ratio < 7.5, (
        f"n_micro ratio res11:res9 = {ratio:.2f}, expected ~7.18"
    )


def test_budget_uniform_resolution_stops_after_n_hops():
    """In a ring graph at uniform resolution, an agent with
    n_micro_per_cell=2 should stop after exactly 2 hops, regardless
    of how many iterations max_steps allows."""
    from salmon_ibm.movement import _step_random_numba

    n_cells = 10
    water_nbrs = np.zeros((n_cells, 1), dtype=np.int32)
    water_nbr_count = np.ones(n_cells, dtype=np.int32)
    for i in range(n_cells):
        water_nbrs[i, 0] = (i + 1) % n_cells

    n_agents = 4
    tri_indices = np.zeros(n_agents, dtype=np.int32)
    rand_vals = np.zeros((10, n_agents), dtype=np.float64)
    n_micro_per_cell = np.full(n_cells, 2, dtype=np.int32)
    fraction_remaining = np.ones(n_agents, dtype=np.float32)

    _step_random_numba(
        tri_indices, water_nbrs, water_nbr_count, rand_vals,
        max_steps=10,
        n_micro_per_cell=n_micro_per_cell,
        fraction_remaining=fraction_remaining,
    )
    # Each agent: hop 1 (0→1) consumes 0.5, hop 2 (1→2) consumes 0.5,
    # fraction reaches 0, agent stops.  Final position: cell 2.
    assert (tri_indices == 2).all(), (
        f"agents took {tri_indices.tolist()} hops, expected 2 each"
    )
    assert (fraction_remaining <= 0.0).all()


def test_budget_distance_conserved_across_resolution_boundary():
    """An agent crossing from coarse-budget cells into fine-budget
    cells must NOT get its swim-distance budget refreshed at the
    boundary.

    Setup: ring graph of 10 cells.  Cells [0,1] have budget=4 (cost
    0.25 per hop = "coarse"); cells [2..9] have budget=8 (cost 0.125
    per hop = "fine").  Agent starts at cell 0.

    Correct trace (cost charged at *source* cell):
        0→1: cost 0.25, used=0.25, at cell 1
        1→2: cost 0.25, used=0.50, at cell 2
        2→3: cost 0.125, used=0.625, at cell 3
        3→4: cost 0.125, used=0.750, at cell 4
        4→5: cost 0.125, used=0.875, at cell 5
        5→6: cost 0.125, used=1.000, at cell 6, fraction_remaining ≤ 0
        agent stops at cell 6.  Total hops = 6.

    Buggy trace (budget refreshed when entering fine cell):
        2 coarse hops + 8 fine hops = 10 hops, ending at cell 10%10=0.

    The assertion `tri_indices == 6` discriminates correct from buggy."""
    from salmon_ibm.movement import _step_random_numba

    n_cells = 10
    water_nbrs = np.zeros((n_cells, 1), dtype=np.int32)
    water_nbr_count = np.ones(n_cells, dtype=np.int32)
    for i in range(n_cells):
        water_nbrs[i, 0] = (i + 1) % n_cells

    n_micro_per_cell = np.array(
        [4, 4, 8, 8, 8, 8, 8, 8, 8, 8], dtype=np.int32,
    )
    n_agents = 4
    tri_indices = np.zeros(n_agents, dtype=np.int32)
    rand_vals = np.zeros((16, n_agents), dtype=np.float64)
    fraction_remaining = np.ones(n_agents, dtype=np.float32)

    _step_random_numba(
        tri_indices, water_nbrs, water_nbr_count, rand_vals,
        max_steps=16,  # max(n_micro_per_cell)
        n_micro_per_cell=n_micro_per_cell,
        fraction_remaining=fraction_remaining,
    )
    # Correct algorithm: 2 coarse hops (at cost 0.25 each = 0.5 used)
    # + 4 fine hops (at cost 0.125 each = 0.5 used) → 6 total, at cell 6.
    # Buggy algorithm (resource refresh): would reach cell 10%10 = 0.
    assert (tri_indices == 6).all(), (
        f"agents reached {tri_indices.tolist()}, expected 6.  Anything "
        f"else (especially 0 or 10) suggests the budget-refresh bug "
        f"described in plan Task 0.5 — agent re-spent its swim distance "
        f"after crossing into the finer-resolution cells."
    )
```

Both tests run immediately and exercise the budget logic without any H3 mesh dependency.

- [ ] **Step 0.5.6: Verify legacy path still works.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_movement.py" -v --tb=short
```

Expected: the existing 30+ movement tests all pass — when `n_micro_steps_per_cell` isn't provided, behaviour is identical to the previous `n_micro_steps=3` scalar.

- [ ] **Step 0.5.7: Commit.**

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" add salmon_ibm/movement.py salmon_ibm/simulation.py salmon_ibm/events_builtin.py tests/test_movement_resolution_aware.py
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" commit -m "feat(movement): resolution-aware n_micro_steps per cell"
```

Commit body should note: this is a calibration fix prompted by the v1.5.0 res-11 rebuild; legacy behaviour preserved when `n_micro_steps_per_cell` not in landscape.

---

## Task 0.6: Area-weighted `IntroductionEvent` placement

**Why:** at res 11 cells (~730 m²) vs res 10 (~5 460 m²) vs res 9 (~38 300 m²), uniform-over-cells choice over-places agents in rivers by ~7.5× / ~52×.  One-line fix.

**Files:**
- Modify: `salmon_ibm/events_builtin.py:198-205` (`IntroductionEvent.execute`)
- Modify: `salmon_ibm/simulation.py` — expose `mesh.areas` via landscape dict (likely already exposed; verify)
- Test: `tests/test_introduction_area_weighted.py` (new file, ~30 LOC)

- [ ] **Step 0.6.1: Read mesh from landscape, weight choice by area.**

`salmon_ibm/events_builtin.py:198-205` — replace:

```python
nonzero = np.where(layer != 0)[0]
if len(nonzero) > 0:
    pos_arr = rng.choice(nonzero, size=n, replace=True)
```

with:

```python
nonzero = np.where(layer != 0)[0]
if len(nonzero) > 0:
    mesh = landscape.get("mesh")
    if mesh is not None and hasattr(mesh, "areas"):
        # Area-weighted: cells get drawn proportional to their
        # m².  Without this, three-tier H3 meshes over-place agents
        # in fine-resolution river cells by ~7-50×.
        weights = mesh.areas[nonzero].astype(np.float64)
        weights = weights / weights.sum()
        pos_arr = rng.choice(nonzero, size=n, replace=True, p=weights)
    else:
        # Legacy mesh without per-cell areas: uniform-over-cells.
        pos_arr = rng.choice(nonzero, size=n, replace=True)
```

- [ ] **Step 0.6.2: Verify `mesh.areas` is exposed in landscape dict.**

`H3MultiResMesh.__init__` (`salmon_ibm/h3_multires.py:285`) sets `self.areas`.  `simulation.py` should already pass the mesh object via `landscape["mesh"]`; the `IntroductionEvent` accesses it via `landscape.get("mesh")`.  Sanity-check by reading the lines around `landscape: Landscape = {` in `simulation.py` and confirming `"mesh": self.mesh` is present.  If not, add it.

- [ ] **Step 0.6.3: Write a focused test.**

`tests/test_introduction_area_weighted.py`:

```python
"""IntroductionEvent must place agents proportional to cell area, not
uniformly per cell.  Otherwise three-tier H3 meshes over-place in fine
river cells by ~7-50×.
"""
from __future__ import annotations
import numpy as np
from unittest.mock import MagicMock
from salmon_ibm.events_builtin import IntroductionEvent


def test_introduction_weights_by_area():
    """In a 2-cell mesh where cell 0 has area 100 and cell 1 has area 1,
    placing 10 000 agents from a uniform spatial-data layer should put
    ~9 901 in cell 0 and ~99 in cell 1."""
    rng = np.random.default_rng(42)
    mesh = MagicMock()
    mesh.areas = np.array([100.0, 1.0], dtype=np.float32)
    mesh.n_cells = 2
    layer = np.array([1, 1])  # both cells eligible

    pop = MagicMock()
    pop.add_agents.return_value = np.arange(10_000)
    pop.trait_mgr = None
    pop.accumulator_mgr = None

    event = IntroductionEvent(
        n_agents=10_000,
        initialization_spatial_data="placement_layer",
    )
    landscape = {
        "rng": rng,
        "spatial_data": {"placement_layer": layer},
        "mesh": mesh,
    }
    event.execute(pop, landscape, t=0, mask=None)

    pos_arr = pop.add_agents.call_args[0][1]   # second positional arg
    n_in_cell0 = int((pos_arr == 0).sum())
    expected = 10_000 * 100 / 101  # ≈ 9900.99
    # σ for binomial(10000, 0.99) ≈ 9.95.  Use 3σ tolerance (<30).  A
    # uniform-over-cells implementation would put ~5000 in cell 0 — a
    # ~4900 miss, far outside this bound.  3σ also catches a ~1%
    # weight-normalisation bug (e.g., forgetting to normalise by sum).
    assert abs(n_in_cell0 - expected) < 30, (
        f"area-weighted placement off: {n_in_cell0} in cell 0, "
        f"expected {expected:.0f} ± 30"
    )


def test_introduction_falls_back_to_uniform_without_mesh():
    """When mesh is missing or has no .areas, fall back to uniform."""
    rng = np.random.default_rng(42)
    layer = np.array([1, 1])
    pop = MagicMock()
    pop.add_agents.return_value = np.arange(10_000)
    pop.trait_mgr = None
    pop.accumulator_mgr = None
    event = IntroductionEvent(
        n_agents=10_000, initialization_spatial_data="layer"
    )
    landscape = {
        "rng": rng, "spatial_data": {"layer": layer},
    }
    event.execute(pop, landscape, t=0, mask=None)
    pos_arr = pop.add_agents.call_args[0][1]
    n_in_cell0 = int((pos_arr == 0).sum())
    # Uniform: ~5000 in each cell, ±200.
    assert 4_700 < n_in_cell0 < 5_300
```

Run:

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_introduction_area_weighted.py" -v --tb=short
```

Expected: 2 passed.

- [ ] **Step 0.6.4: Commit.**

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" add salmon_ibm/events_builtin.py tests/test_introduction_area_weighted.py
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" commit -m "feat(introduction): area-weighted spatial-data placement"
```

---

## Task 1: Polygon-dilation tessellation helper

**Files:**
- Modify: `scripts/build_h3_multires_landscape.py:192-222` (`tessellate_reach` function)
- Test: `tests/test_h3_grid_quality.py` (new file)

The current `tessellate_reach` calls `h3.polygon_to_cells(polygon, resolution)` without buffering. `polygon_to_cells` uses a strict centroid-in-polygon test — cells whose centroid sits even slightly outside the polygon edge are excluded. For a meandering river polygon at res 10 (~75 m cells), this fragmented Minija into 135 components in 219 cells (see baseline). At res 11 (rivers' new resolution, ~28 m cells) the fragmentation is much milder because cells are smaller relative to typical 50–200 m channel widths, but the bug still produces ragged edges. Dilation makes the tessellation robust at all three resolutions.

The fix: dilate the polygon by half a cell edge before tessellation.  Half-edge buffer per resolution (using `h3.average_hexagon_edge_length(r, unit="m")`):
* res 11 → 28.0 m / 2 ≈ 14 m → buffer ≈ 0.00013° at lat 55
* res 10 → 75.9 m / 2 ≈ 38 m → buffer ≈ 0.00034° at lat 55
* res  9 → 200.8 m / 2 ≈ 100 m → buffer ≈ 0.00090° at lat 55

- [ ] **Step 1.1: Write a failing connectivity test.**

`tests/test_h3_grid_quality.py`:

```python
"""Grid-quality regressions: connectivity and area-fidelity per reach.

Loads `data/curonian_h3_multires_landscape.nc` and asserts:
1. Each reach forms ≤ N_MAX_COMPONENTS connected H3 components.
2. Cell-area sum matches inSTREAM polygon area within ±10%.
3. Cross-reach link count meets minimum thresholds for ecological
   adjacency (e.g., Nemunas↔Atmata ≥ 5 links).

Tests skip cleanly when the NC isn't built locally.
"""
from __future__ import annotations
from pathlib import Path
import pytest
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
NC = PROJECT / "data" / "curonian_h3_multires_landscape.nc"

# Per-reach maximum allowed component count.  Single-component is
# the goal everywhere; the lagoon and Baltic are allowed 1 because
# Curonian Spit splits them (we treat them as separate reaches).
N_MAX_COMPONENTS = {
    "Nemunas":         1,
    "Atmata":          1,
    "Minija":          1,
    "Sysa":            1,
    "Skirvyte":        1,
    "Leite":           1,
    "Gilija":          1,
    "CuronianLagoon":  1,
    "BalticCoast":     1,
    "OpenBaltic":      1,
}

# Minimum cross-reach links — geographically adjacent reaches must
# have at least this many same-resolution OR cross-resolution links
# in the neighbour table.  Numbers calibrated for the three-tier
# scheme (rivers res 11, lagoon res 10, Baltic res 9): finer cells
# multiply boundary linkage relative to the v1.4.0 baseline.
MIN_CROSS_REACH_LINKS = {
    # Same-resolution Baltic↔Baltic — should be plentiful at res 9.
    ("BalticCoast",    "OpenBaltic"):  2_000,
    # Klaipėda strait: lagoon res 10 (~5 cells across) ↔ Baltic res 9
    # (~2 cells across).  Cross-res, but the bottleneck has many
    # lagoon children abutting each Baltic cell.
    ("CuronianLagoon", "BalticCoast"):    500,    # was 34
    # River mouths into lagoon: rivers res 11 (~7 cells across) ↔
    # lagoon res 10.  Each river-mouth cell links to ~7 lagoon cells.
    ("Atmata",         "CuronianLagoon"): 200,
    ("Minija",         "CuronianLagoon"): 200,
    ("Sysa",           "CuronianLagoon"): 100,
    ("Skirvyte",       "CuronianLagoon"): 200,
    ("Leite",          "CuronianLagoon"): 100,
    ("Gilija",         "CuronianLagoon"): 200,
    ("Nemunas",        "CuronianLagoon"): 200,
    # Inter-river junctions in the Nemunas delta (rivers branching
    # off the main Nemunas).  Same resolution res 11.
    ("Nemunas",        "Atmata"):         200,    # was 4
    ("Nemunas",        "Skirvyte"):       100,    # was 6
    ("Atmata",         "Skirvyte"):        50,    # was 4
}


def _load_nc():
    if not NC.exists():
        pytest.skip(f"{NC.name} missing — rebuild via scripts/build_h3_multires_landscape.py")
    import xarray as xr
    return xr.open_dataset(NC, engine="h5netcdf")


def _components_per_reach(ds, reach_name: str) -> int:
    """BFS within reach cells using the CSR neighbour table."""
    reach_names = ds.attrs["reach_names"].split(",")
    if reach_name not in reach_names:
        return 0
    rid = reach_names.index(reach_name)
    reach_id = ds["reach_id"].values
    nbr_starts = ds["nbr_starts"].values
    nbr_idx = ds["nbr_idx"].values
    cells_in = set(np.where(reach_id == rid)[0].tolist())
    if not cells_in:
        return 0
    seen: set[int] = set()
    n_components = 0
    for start in cells_in:
        if start in seen:
            continue
        n_components += 1
        stack = [start]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            for nb in nbr_idx[nbr_starts[c]:nbr_starts[c+1]]:
                nb = int(nb)
                if nb in cells_in and nb not in seen:
                    stack.append(nb)
    return n_components


def test_each_reach_is_connected():
    ds = _load_nc()
    failed = []
    for reach, max_components in N_MAX_COMPONENTS.items():
        n = _components_per_reach(ds, reach)
        if n == 0:
            continue  # reach not present in this NC
        if n > max_components:
            failed.append((reach, n, max_components))
    ds.close()
    assert not failed, (
        "reaches with too many disconnected components:\n  "
        + "\n  ".join(f"{r}: {got} > {max_}" for r, got, max_ in failed)
    )


def test_cross_reach_link_thresholds():
    ds = _load_nc()
    reach_names = ds.attrs["reach_names"].split(",")
    reach_id = ds["reach_id"].values
    nbr_starts = ds["nbr_starts"].values
    nbr_idx = ds["nbr_idx"].values

    # Count links per (sorted) reach pair.
    pairs: dict[tuple[str, str], int] = {}
    for i in range(len(reach_id)):
        rid_i = int(reach_id[i])
        if rid_i < 0:
            continue
        for j in nbr_idx[nbr_starts[i]:nbr_starts[i+1]]:
            rid_j = int(reach_id[j])
            if rid_j < 0 or rid_j == rid_i:
                continue
            key = tuple(sorted([reach_names[rid_i], reach_names[rid_j]]))
            pairs[key] = pairs.get(key, 0) + 1
    ds.close()

    failed = []
    for (a, b), threshold in MIN_CROSS_REACH_LINKS.items():
        key = tuple(sorted([a, b]))
        got = pairs.get(key, 0)
        if got < threshold:
            failed.append((a, b, got, threshold))
    assert not failed, (
        "cross-reach link counts below minimum:\n  "
        + "\n  ".join(f"{a}↔{b}: {got} < {th}" for a, b, got, th in failed)
    )
```

Run it (expected to fail against the v1.4.0 NC):

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_grid_quality.py" -v --tb=short
```

Expected: both tests FAIL.  `test_each_reach_is_connected` reports `Minija: 135 > 1`, `Sysa: 60 > 1`, etc.  `test_cross_reach_link_thresholds` reports `CuronianLagoon↔BalticCoast: 34 < 100`, `Nemunas↔Atmata: 4 < 50`, etc.

- [ ] **Step 1.2: Replace `tessellate_reach` with the dilated version.**

In `scripts/build_h3_multires_landscape.py`, replace the existing function with:

```python
def tessellate_reach(polygon, resolution: int) -> list[str]:
    """Return the H3 cells tessellating a buffered version of polygon.

    Buffer = half the H3 edge length at the target resolution.  This
    ensures cells whose centroid sits within half-a-cell of the
    polygon edge are included — without it, a thin meandering river
    polygon at res 10 (~75 m cells) tessellates as a sparse, often
    disconnected, cell set because polygon_to_cells uses a strict
    centroid-in-polygon test.

    Buffer is applied in degrees: convert metres → degrees at lat 55°
    (the bbox centre) using 1 deg ≈ 111 km.

    For multi-polygon inputs, each part is buffered + tessellated
    independently and the cell sets are unioned.
    """
    from shapely.geometry import MultiPolygon

    edge_m = h3.average_hexagon_edge_length(resolution, unit="m")
    # Buffer in degrees: half a cell edge at lat 55°.
    buffer_deg = (edge_m / 2.0) / 111_000.0

    if isinstance(polygon, MultiPolygon):
        parts = list(polygon.geoms)
    else:
        parts = [polygon]

    cells: set[str] = set()
    for part in parts:
        if part.is_empty:
            continue
        buffered = part.buffer(buffer_deg)
        # Buffer can return a MultiPolygon when the original was thin
        # (the dilation merges previously-separate pieces).  Normalise
        # to a list of single Polygons.
        if hasattr(buffered, "geoms"):
            inner_parts = list(buffered.geoms)
        else:
            inner_parts = [buffered]
        for ip in inner_parts:
            if ip.is_empty:
                continue
            ext = [(y, x) for x, y in ip.exterior.coords]
            holes = [
                [(y, x) for x, y in interior.coords]
                for interior in ip.interiors
            ]
            try:
                poly = h3.LatLngPoly(ext, *holes)
            except Exception as e:
                print(f"  ! skipping polygon part: {e}")
                continue
            for c in h3.polygon_to_cells(poly, resolution):
                cells.add(c)
    return sorted(cells)
```

The change: `buffered = part.buffer(buffer_deg)` before the `LatLngPoly` construction.  Same behaviour for any width-≥-2-edges polygon (buffering doesn't move the cells); fixes connectivity for thin polygons.

- [ ] **Step 1.3: Verify the dilation helper at unit-test scale.**

Quick smoke check (no commit yet):

```bash
micromamba run -n shiny python -c "
import sys
sys.path.insert(0, r'C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/scripts')
sys.path.insert(0, r'C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim')
from build_h3_multires_landscape import tessellate_reach
from _water_polygons import fetch_instream_polygons
gdf = fetch_instream_polygons()
minija = gdf[gdf['REACH_NAME'] == 'Minija'].geometry.unary_union
cells_old = []  # call the unbuffered path mentally
cells_new = tessellate_reach(minija, 10)
print(f'Minija buffered tessellation at res 10: {len(cells_new)} cells')
"
```

Expected: should print > 219 cells (the v1.4.0 count) — the buffer adds bank-adjacent cells.  If still 219, the buffer didn't apply — debug.

---

## Task 2: Bridge-cell post-processing pass

Even with dilation, some reach polygons fragment into multiple H3 components when the polygon itself is multi-part (e.g., the inSTREAM Curonian Lagoon polygon has the main body + a small detached piece near Klaipėda).  After tessellation, walk each reach's cells, identify components, and connect them via `h3.grid_path_cells` shortest paths — adding the path cells to the reach's cell set.

**Files:**
- Modify: `scripts/build_h3_multires_landscape.py:build_cell_list` — call a new `bridge_components` helper after each per-reach tessellation.

- [ ] **Step 2.1: Add bridge-cell helper.**

In `scripts/build_h3_multires_landscape.py`, add this function after `tessellate_reach`:

```python
def bridge_components(
    cells: list[str], resolution: int, max_bridge_len: int = 5
) -> list[str]:
    """Merge disconnected H3 components into a single graph by adding
    shortest-path cells between adjacent components.

    **Precondition: all input cells must be at the same H3 resolution**
    (passed as the ``resolution`` argument).  ``h3.grid_ring`` and
    ``h3.grid_distance`` only work between same-resolution cells; mixing
    resolutions would silently produce spurious component splits and
    nonsensical bridge paths.  In `build_cell_list` (Step 2.2) this is
    guaranteed because each call sees only one reach's cells, all at
    that reach's resolution.

    For each pair of components A and B, find the cell pair (a ∈ A,
    b ∈ B) with the shortest H3 grid distance.  If that distance is
    ≤ ``max_bridge_len`` cells, add the intermediate cells from
    `h3.grid_path_cells(a, b)` to the cell set.  This connects
    near-adjacent fragments without grafting on huge unrelated zones.

    Returns the augmented cell list, sorted.
    """
    if not cells or len(cells) < 2:
        return cells
    cell_set = set(cells)

    # BFS within cell_set using grid_ring(1) to find components.
    components: list[set[str]] = []
    seen: set[str] = set()
    for start in cell_set:
        if start in seen:
            continue
        comp: set[str] = set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            comp.add(c)
            for nb in h3.grid_ring(c, 1):
                if nb in cell_set and nb not in seen:
                    stack.append(nb)
        components.append(comp)

    if len(components) <= 1:
        return cells

    # Pick the largest component as the "anchor".  For each smaller
    # component, find the nearest cell to the anchor and bridge.
    components.sort(key=len, reverse=True)
    anchor = components[0]
    bridges_added = 0
    for orphan in components[1:]:
        # Find the (anchor_cell, orphan_cell) pair with shortest grid
        # distance.  O(|anchor|·|orphan|) — fine for the small
        # components (typically <100 cells each).
        best_dist = max_bridge_len + 1
        best_pair = None
        for a in orphan:
            for b in anchor:
                d = h3.grid_distance(a, b)
                if d < best_dist:
                    best_dist = d
                    best_pair = (a, b)
                    if d == 1:
                        break
            if best_pair and best_dist == 1:
                break
        if best_pair is None or best_dist > max_bridge_len:
            print(
                f"  ! orphan component of {len(orphan)} cells too far "
                f"from anchor (>{max_bridge_len} cells) — leaving "
                f"disconnected"
            )
            continue
        # Add intermediate path cells to bridge.
        path = list(h3.grid_path_cells(best_pair[0], best_pair[1]))
        for pc in path:
            if pc not in cell_set:
                cell_set.add(pc)
                bridges_added += 1

    if bridges_added:
        print(f"  bridge-cell pass: added {bridges_added} cells across "
              f"{len(components) - 1} component gaps")
    return sorted(cell_set)
```

- [ ] **Step 2.2: Wire into `build_cell_list`.**

The existing per-reach loop in `build_cell_list` is at `scripts/build_h3_multires_landscape.py:239-245`.  Insert one new line (`cells = bridge_components(cells, res)`) between `tessellate_reach` and the `for c in cells:` deduplication:

```python
for reach_name, poly in reach_polygons.items():
    res = reach_res.get(reach_name, 9)
    cells = tessellate_reach(poly, res)
    cells = bridge_components(cells, res)   # NEW
    for c in cells:
        ci = int(h3.str_to_int(c))
        if ci not in seen_id_to_reach:
            seen_id_to_reach[ci] = reach_name
```

Note: the `seen_id_to_reach` dict still claims-the-first-wins, so a bridge cell added by an earlier reach won't be re-tagged by a later reach — fine for our purposes (the rivers are processed before the lagoon, so river-bridge cells stay tagged as river).

---

## Task 3: Resolution scheme — three-tier (rivers 11 / lagoon 10 / Baltic 9)

Switch the default resolution map to the user-chosen three-tier scheme.

**Current state of `DEFAULT_RES` in the file (verified at plan-time):** rivers already at res 11 (set in v1.4.0 work), but `CuronianLagoon: 9` (target 10) and `OpenBaltic: 8` (target 9) are stale.  Only those two values change in this task; the river entries already match target.

**Files:**
- Modify: `scripts/build_h3_multires_landscape.py:103-114` (`DEFAULT_RES` dict)

- [ ] **Step 3.1: Update `DEFAULT_RES`.**

Replace the existing dict with:

```python
DEFAULT_RES = {
    # Rivers — res 11 (~28 m edge) to resolve 50–200 m channel
    # widths as 1–7 cells across.  At this scale, polygon_to_cells
    # tessellates rivers as a connected swath naturally; bridge-cell
    # logic (Task 2) is a safety net for multi-polygon cases.
    "Nemunas":        11,
    "Atmata":         11,
    "Minija":         11,
    "Sysa":           11,
    "Skirvyte":       11,
    "Leite":          11,
    "Gilija":         11,
    # Lagoon — res 10 (~76 m edge, ~120 k cells) for spatial
    # heterogeneity in mortality, temperature, salinity gradients
    # across the lagoon's 1 600 km².  Was res 9 in v1.4.0.
    "CuronianLagoon": 10,
    # Baltic — uniform res 9 (~201 m edge) eliminates the v1.4.0
    # res-9 / res-8 boundary discontinuity at the BalticCoast↔
    # OpenBaltic interface.  Mesoscale circulation does not need
    # finer cells.  OpenBaltic gains ~7× cells (50 k vs 7 k); the
    # visible "scattered hexagons" boundary disappears.
    "BalticCoast":     9,
    "OpenBaltic":      9,
}
```

The `--resolutions` CLI override still works for one-off experiments.

---

## Task 4: Rebuild + validate

- [ ] **Step 4.1: Run the new builder.**

```bash
PYTHONUNBUFFERED=1 micromamba run -n shiny python -u "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/scripts/build_h3_multires_landscape.py"
```

(No `--resolutions` flag — uses the new default.)

Expected: build runs ~10–15 minutes (slower than v1.4.0 because cell count grows 5.5× and per-reach BFS for component detection scales with N).  Output:

```
Resolutions in use:
       Nemunas: res 11  (28 m edge)
        Atmata: res 11  (28 m edge)
        ...
CuronianLagoon: res 10  (76 m edge)
   BalticCoast: res  9  (201 m edge)
    OpenBaltic: res  9  (201 m edge)

[2/4] Tessellating per-reach…
  total cells: ~200 000 (estimate)
       Nemunas (res 11): ~9 000 cells
        Atmata (res 11): ~1 100 cells
        ...
CuronianLagoon (res 10): ~120 000 cells
   BalticCoast (res  9): ~9 700 cells
    OpenBaltic (res  9): ~50 000 cells

  bridge-cell pass: added N cells across M component gaps  (per reach)
neighbour table: ~1 300 000 edges, avg ~6.5 per cell
wrote ~80–100 MB
```

Cross-check: the v1.4.0 build had 36 k cells and 210 k edges; this one has ~200 k cells and ~1.3 M edges.  NetCDF on disk grows from ~16 MB to ~80–100 MB.

- [ ] **Step 4.2: Run the grid-quality test from Step 1.1.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_grid_quality.py" -v --tb=short
```

Expected: BOTH tests PASS.

If `test_each_reach_is_connected` still fails for some river (e.g., Sysa), inspect — likely the river polygon has a fragment too far from the main reach (>5-cell bridge length).  Either bump `max_bridge_len` to 10 in `bridge_components`, or accept the failure and lower `N_MAX_COMPONENTS[reach]` for that specific reach with a comment explaining why.

If `test_cross_reach_link_thresholds` fails for some pair, check whether it is a cross-resolution-only adjacency: in the three-tier scheme, rivers↔lagoon are cross-res (drop=1), lagoon↔BalticCoast is cross-res (drop=1), BalticCoast↔OpenBaltic is same-res (no drop).  If a pair shows zero cross-res links where it should have many, debug `find_cross_res_neighbours` for that resolution drop.

- [ ] **Step 4.2b: Klaipėda strait cross-resolution link audit.**

The strait is the only same-water-body cross-resolution boundary (lagoon res 10 ↔ Baltic res 9).  Confirm `find_cross_res_neighbours` actually built links here:

```bash
micromamba run -n shiny python -c "
import numpy as np, xarray as xr
ds = xr.open_dataset(r'C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/data/curonian_h3_multires_landscape.nc', engine='h5netcdf')
reach_id = ds['reach_id'].values
res = ds['resolution'].values
nbr_starts = ds['nbr_starts'].values
nbr_idx = ds['nbr_idx'].values
names = ds.attrs['reach_names'].split(',')
lagoon = names.index('CuronianLagoon')
baltic = names.index('BalticCoast')
# Count directed-edge pairs.  Each undirected lagoon↔Baltic edge is
# visited TWICE (once when iterating from each endpoint), so the
# directed-count threshold is 2× the desired undirected-count.
directed_count = 0
for i in range(len(reach_id)):
    ri = int(reach_id[i])
    if ri not in (lagoon, baltic):
        continue
    for j in nbr_idx[nbr_starts[i]:nbr_starts[i+1]]:
        rj = int(reach_id[j])
        if rj == ri:
            continue
        if {ri, rj} == {lagoon, baltic} and res[i] != res[j]:
            directed_count += 1
undirected = directed_count // 2
print(f'Strait cross-res links: {undirected} undirected ({directed_count} directed)')
ds.close()
"
```

Expected: ≥ 10 undirected (≈ 20 directed) cross-res links at the strait.  If 0, `find_cross_res_neighbours` failed to bridge the resolution drop — debug there.  Note: the formal gate in `test_cross_reach_link_thresholds` (Step 1.1) already requires `Lagoon↔BalticCoast ≥ 500 directed` (≥ 250 undirected) — this audit is a quicker eyeball check after the build completes.

- [ ] **Step 4.3: Re-run the H3 regression suite.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires.py" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_builder.py" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py" --tb=short
```

Expected: every test passes; no new failures vs the v1.4.0 baseline.  These integration tests assert *behaviour* (movement, mortality contrast, salinity gradient, agent-count invariants), which is unchanged by the geometric rebuild.  Specific cell-count assertions live in the new `test_h3_grid_quality.py`, not here.

If the slow tests are run, the slow ones in this suite should also pass — confirm with:

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py" -m slow --tb=short
```

- [ ] **Step 4.4: Re-run the v1.4.0 per-reach mortality test.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_nemunas_h3_integration.py::test_open_baltic_kills_more_than_river" -m slow -v
```

Expected: PASS.  The mortality contrast is a function of `mortality_per_reach` rates, not cell topology.

---

## Task 5: Commit + tag + deploy + visual check

- [ ] **Step 5.1: Commit the geometric-rebuild changes.**

(Tasks 0.5 and 0.6 each committed their own changes already.  This commit covers Tasks 1–4: tessellation dilation, bridge-cell helper, `DEFAULT_RES`, and the new grid-quality test.  The rebuilt `data/curonian_h3_multires_landscape.nc` is excluded by `.gitignore:31` — it is shipped to laguna via SCP in Step 5.3, not committed.)

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" add scripts/build_h3_multires_landscape.py tests/test_h3_grid_quality.py
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" commit -m "feat(h3-grid): three-tier connectivity-first rebuild"
```

Commit body should include the v1.4.0 → v1.5.0 before/after connectivity table from Step 0.2 so the rebuild is reproducible from git history alone (the NC itself isn't tracked).

- [ ] **Step 5.2: Tag + push + deploy + SCP.**

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" tag -a v1.5.0 -m "v1.5.0 — connectivity-first H3 grid rebuild"
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" push origin main
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" push origin v1.5.0
bash "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/scripts/deploy_laguna.sh" apply
scp "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/data/curonian_h3_multires_landscape.nc" razinka@laguna.ku.lt:/srv/shiny-server/HexSimPy/data/
ssh razinka@laguna.ku.lt 'cd /srv/shiny-server/HexSimPy && md5sum data/curonian_h3_multires_landscape.nc && touch restart.txt'
```

- [ ] **Step 5.3: Browser sanity check.**

Hard-refresh `http://laguna.ku.lt/HexSimPy/`, pick "Curonian Lagoon H3 (multi-res)", Reset → Run.  Expected:
* River channels render as visibly continuous lines (was: scattered dots).
* No visible BalticCoast↔OpenBaltic boundary (was: cell-size discontinuity).
* Each reach renders as a contiguous blob, not a confetti spread.

The connectivity tests in Step 4.2 are the formal gate; this visual step is confirmation only.  If something looks wrong visually but Step 4.2 passed, capture a screenshot and add a follow-up issue rather than blocking the v1.5.0 release.

---

## Risks and mitigations

* **Bridge-cell pass leaks into other reaches.** A river-bridging path could pass through lagoon territory and tag those cells as river.  Mitigation: `seen_id_to_reach` is first-wins; rivers process before lagoon, so bridge cells stay rivers.  But a bridge that crosses the *spit* would tag spit-land cells as river — log a warning and visually inspect post-rebuild.  At res 11 (rivers) → res 10 (lagoon), bridges within rivers stay at res 11, so a bridge "leaking" into lagoon territory would still be a res 11 cell sitting inside a region tagged res 10 — unusual but tolerated by `find_cross_res_neighbours`.
* **Viewer payload grows 5×.** Total cells go from ~36 k to ~200 k.  deck.gl JSON payload grows from ~3 MB to ~10–15 MB.  Should still render in <1 s on the desktop laguna browser, but on tablets the H3HexagonLayer may lag visibly.  Acceptable for the project's research-tool audience.
* **Build time grows 5–7×.** v1.4.0 was ~2 min; expect **~10–15 min** for v1.5.0.  Major drivers: per-reach BFS for component detection (O(N)), `find_cross_res_neighbours` (O(N·avg_drops)), neighbour-table construction.  Run as a manual one-off; do not put on CI without caching.
* **CMEMS regridding slows ~5×.** Environment field interpolation from CMEMS grid to H3 cells happens at sim startup.  Was ~30 s for 36 k cells; expect **~3 min** for 200 k cells.  One-time cost per simulation run, but noticeable for short interactive experiments.
* **NetCDF on disk grows 5×.** v1.4.0 NC was ~16 MB; v1.5.0 will be **~80–100 MB**.  Still fits in the laguna deploy SCP step (~30 s upload over campus network); just a larger artefact to manage.
* **MAX_NBRS sufficiency.**  Drop=1 between any two adjacent resolutions (rivers↔lagoon, lagoon↔Baltic) produces at most ~14 neighbours per cell.  `MAX_NBRS=64` (set in v1.3.0) remains comfortable.  No bump needed.
* **Polygon dilation buffer interacts with res 11 edge length (~28 m).**  Half-edge buffer = 14 m, which is below the spatial precision of the inSTREAM polygons (digitised at ~5–10 m precision per the example_baltic shapefile).  If digitisation noise pushes a cell centroid outside the dilated polygon, the bridge-cell pass catches it.

---

## Out of scope (intentionally)

* **Alternative resolution schemes.**  Earlier drafts of this plan considered Option A (uniform res 9 everywhere — ~50 k cells), Option B (lagoon res 9 / everything else res 10 — ~420 k cells), and Option C (rivers res 10 / everything else res 9 — ~80 k cells).  The user-chosen three-tier scheme (rivers 11 / lagoon 10 / Baltic 9) sits at ~200 k cells.  If post-rebuild benchmarks show the laguna server can't comfortably host the larger NetCDF, drop rivers to res 10 and the lagoon to res 9 (~80 k cells, equivalent to Option C).
* **Multi-res integration test updates.**  `test_resolutions_are_mixed` (Phase 2) asserts ≥ 2 unique resolutions in the water mask.  Three-tier has res 9 + res 10 + res 11 → passes.  No update needed.
* **Per-reach buffering tuning.**  The current half-edge buffer is uniform.  For very thin rivers (Šyša ~80 m at res 11 ≈ 3 cells across the polygon), a full-edge buffer might still give better connectivity.  Defer; revisit if `test_each_reach_is_connected` fails for specific reaches after the rebuild.
* **Further calibration.**  Tasks 0.5 (resolution-aware `n_micro_steps`) and 0.6 (area-weighted introduction) are the **only** calibration changes in this plan — they are the validator-flagged blockers for res 11.  Other ecological calibration knobs (per-reach drift food, salinity tolerance, density-dependent crowding) remain Phase 5 work and are out of scope here.
* **Numba kernel tuning for 200k cells.**  Existing kernels are O(n_agents) per step, not O(n_cells), so the larger mesh shouldn't slow the hot path.  But environment-cache memory grows ~5× — if the laptop OOMs, lazy-load environment fields per-day instead of pre-loading the full 30-day cube.  Defer until measured.
