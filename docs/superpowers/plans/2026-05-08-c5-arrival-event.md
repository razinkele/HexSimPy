# C5 Arrival Event Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up `pool.arrived` (dead production state since the IBM was built) so it's tagged True when an agent reaches the upstream portion of its natal reach (75th-percentile by `dist_from_sea`); makes the four-tier hatchery-vs-wild architecture's wild-vs-hatchery contrast measurable in production.

**Architecture:** New `ArrivalEvent` registered via `@register_event("arrival")` in `events_builtin.py`. Reads per-natal-reach top-quartile thresholds from `Simulation._arrival_threshold_by_natal_rid` (computed once at sim init from `mesh.dist_from_sea`). Helper `_compute_arrival_thresholds` lives in `simulation.py`; init-time validators emit warnings via stdlib logging when the scenario lacks `arrival` despite supporting it (matching the C4 dormancy-guard pattern). Vectorised; sticky once set. No kernel changes, no substrate edits — purely additive metric.

**Tech Stack:** Python 3.10+, conda env `shiny`, NumPy, pytest. Spec at `docs/superpowers/specs/2026-05-08-c5-arrival-event-design.md` (v2 CONVERGED).

**Status:** ✅ EXECUTED on 2026-05-08. Final test count: 925 passed / 33 skipped / 1 xfailed (baseline + 14 new C5 tests). Pre-existing failures (perf-baseline flake, nemunas-NC errors) acceptable per C4 EXECUTED note. End-to-end smoke confirms `pool.arrived` now derives from real biology (no longer stuck at 0). Branch `c5-arrival-event` ready for PR + v1.7.9 tag.

**Plan version:** ✅ v2 final — **2-pass plan-review-loop CONVERGED**. Pass-1 found 1 CRITICAL (validator referenced non-existent `self.events`) + 3 BLOCKING (line numbers + YAML structure) + 2 minor. Pass-2 verified all closures and confirmed 3 code-correctness questions (`self._sequencer.events`, `_build_events()` insertion point, ArrivalEvent import) — all correct against the actual codebase. Plus one documentation-consistency fix (stamp template said "+9 = 961"; corrected to "+14 = ~966"). Implementation-ready.

**Pre-flight:** Confirm baseline before starting:

```bash
micromamba run -n shiny python -m pytest tests/ --collect-only -q | tail -1
```

Should report ~952 tests collected (post-C4 baseline at v1.7.8). Record the count; expected post-C5 = baseline + 14 (= ~966). The plan adds 14 distinct test functions:

- Task 2: 3 (threshold-computation: happy-path, skipped-reach, legacy-mesh)
- Task 3: 4 (Tests 1, 2, 5, 5b)
- Task 4: 2 (Tests 3, 4)
- Task 5: 1 (Test 6 movement integration)
- Task 6: 3 (Test 8 missing-event, Test 8 sanity, Test 9 misorder)
- Task 7: 1 (Test 7 AST sticky-overwrite)

The spec lists "9 numbered tests" (Tests 1-9); the plan adds 5 supporting tests beyond the numbered set (3 threshold-computation in Task 2, 1 sanity counter-test in Task 6, and the threshold-validation 5b is counted within the 9 already).

---

## Task 1: Err-id constants + Landscape TypedDict + landscape dict gain "sim" key

**Files:**
- Modify: `salmon_ibm/h3_env.py` — add 2 err-id constants alongside the existing C4 ones (`ERR_DIST_FROM_SEA_*` block).
- Modify: `salmon_ibm/simulation.py` — add `sim` field to `Landscape` TypedDict + `"sim": self` to the landscape dict in `step()`.

This task is structural setup — no behavior change yet. Tests in subsequent tasks will exercise the additions.

- [ ] **Step 1: Add err-id constants to `salmon_ibm/h3_env.py`**

Locate the existing C4 err-id block (`ERR_DIST_FROM_SEA_MISSING` etc., around `h3_env.py:30-34`). Append:

```python
# C5: arrival event err-ids. ArrivalEvent lives in events_builtin.py
# but err-ids are centralised here next to the C4 dist_from_sea
# constants for grep-able operational logging.
ERR_C5_MISSING_ARRIVAL_EVENT = "c5-arrival-event-missing"
ERR_C5_ARRIVAL_EVENT_MISORDERED = "c5-arrival-event-misordered"
```

- [ ] **Step 2: Add `sim` field to `Landscape` TypedDict in `simulation.py`**

Locate the `Landscape(TypedDict)` definition (around `simulation.py:12-33`). Add the new field, matching the existing string-annotation style used for `H3Environment`/`Environment`/`HexSimEnvironment`:

```python
class Landscape(TypedDict, total=False):
    # ... existing fields ...
    sim: "Simulation"  # NEW (C5) — for ArrivalEvent's threshold lookup.
```

The string annotation avoids circular import (Simulation references Landscape internally).

- [ ] **Step 3: Add `"sim": self` to the landscape dict in `Simulation.step()`**

Locate the dict construction in `step()`: the dict literal opens at **`simulation.py:603`** (`landscape: Landscape = {`); `"env": self.env` is at **line 616** (the line added by C4). Add `"sim": self` immediately below the `"env"` line:

```python
landscape: Landscape = {
    # ... existing keys ...
    "env": self.env,
    "sim": self,  # NEW (C5) — for ArrivalEvent threshold lookup.
}
```

Direct attribute access (NOT `getattr`) — matches the C4 fail-loud convention.

- [ ] **Step 4: Smoke test the structural change**

Run: `micromamba run -n shiny python -m pytest tests/test_simulation.py tests/test_movement.py tests/test_movement_gradient.py -q`
Expected: all pass — the additions are additive (no existing test reads `landscape["sim"]`; no existing test asserts a specific TypedDict key set).

If a test breaks because its synthetic landscape dict lacks `"sim"` and is passed somewhere that does `landscape["sim"]` — that path doesn't exist yet (Task 3 adds it). Verify the failure is unrelated; if related, the test is wrong and needs `"sim": None` added to its synthetic landscape (or stub a Simulation-shaped object).

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/h3_env.py salmon_ibm/simulation.py
git commit -m "$(cat <<'EOF'
feat(c5): err-id constants + Landscape sim field + landscape dict sim key

Plan task 1/8. Structural setup for C5 ArrivalEvent:

- Two err-id constants in h3_env.py alongside the C4 block:
  ERR_C5_MISSING_ARRIVAL_EVENT, ERR_C5_ARRIVAL_EVENT_MISORDERED.
- `sim: "Simulation"` field on the Landscape TypedDict.
- "sim": self injected into the landscape dict at step()
  alongside "env": self.env (mirrors the C4 pattern).

No behavior change; structural only. Subsequent tasks add the
ArrivalEvent that consumes landscape["sim"]._arrival_threshold_*.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `_compute_arrival_thresholds` method + sim init wiring

**Files:**
- Modify: `salmon_ibm/simulation.py` — add `_compute_arrival_thresholds` method + call from `__init__` after C4's `assert_branch_topology`.
- Test: `tests/test_arrival_event.py` (NEW)

- [ ] **Step 1: Create the test file with imports + Test 5c (threshold computation)**

Create `tests/test_arrival_event.py`:

```python
"""Tests for C5 — arrival event.

Spec: docs/superpowers/specs/2026-05-08-c5-arrival-event-design.md
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest


def test_compute_arrival_thresholds_returns_75th_percentile_per_reach():
    """C5 Test 5c (threshold-computation): build a synthetic Simulation
    fixture with a known mesh.dist_from_sea distribution; assert
    `_compute_arrival_thresholds` returns 75th-percentile per reach.

    Synthetic mesh: 12 cells.
    - cells 0..3: reach_id=0 (OpenBaltic), dist=[0, 100, 200, 300]
        → 75th percentile = 225.0
    - cells 4..7: reach_id=1 (CuronianLagoon), dist=[400, 500, 600, 700]
        → 75th percentile = 625.0
    - cells 8..11: reach_id=2 (Nemunas), dist=[800, 900, 1000, 1100]
        → 75th percentile = 1025.0
    """
    # Minimal Simulation-like object that exposes `_compute_arrival_thresholds`
    # via direct method call. Build the mesh attributes the method reads.
    class _MeshShim:
        pass
    mesh = _MeshShim()
    mesh.reach_id = np.array(
        [0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2], dtype=np.int8,
    )
    mesh.dist_from_sea = np.arange(12, dtype=np.float32) * 100.0
    mesh.water_mask = np.ones(12, dtype=bool)
    mesh.reach_names = ["OpenBaltic", "CuronianLagoon", "Nemunas"]

    # Construct a minimal Simulation-shaped class that owns the method.
    from salmon_ibm.simulation import Simulation
    sim = Simulation.__new__(Simulation)  # bypass __init__
    sim.mesh = mesh

    thresholds = sim._compute_arrival_thresholds()
    assert thresholds == pytest.approx({
        0: 225.0,
        1: 625.0,
        2: 1025.0,
    })


def test_compute_arrival_thresholds_skips_reach_with_no_water_cells(caplog):
    """C5: if a reach has zero finite-dist water cells, threshold is
    not computed; warning emitted with err-id."""
    class _MeshShim:
        pass
    mesh = _MeshShim()
    mesh.reach_id = np.array([0, 0, 1, 1], dtype=np.int8)
    mesh.dist_from_sea = np.array([0.0, 100.0, np.nan, np.nan], dtype=np.float32)
    mesh.water_mask = np.array([True, True, True, True], dtype=bool)
    mesh.reach_names = ["OpenBaltic", "Nemunas"]

    from salmon_ibm.simulation import Simulation
    sim = Simulation.__new__(Simulation)
    sim.mesh = mesh

    caplog.set_level(logging.WARNING, logger="salmon_ibm.simulation")
    thresholds = sim._compute_arrival_thresholds()
    assert 0 in thresholds  # OpenBaltic has water cells
    assert 1 not in thresholds  # Nemunas all-NaN, skipped
    matching = [
        r for r in caplog.records
        if "c5-arrival-skipped-reach" in r.getMessage()
        and "Nemunas" in r.getMessage()
    ]
    assert matching, (
        f"Expected c5-arrival-skipped-reach warning naming Nemunas; "
        f"got {[r.getMessage() for r in caplog.records]!r}"
    )


def test_compute_arrival_thresholds_returns_empty_on_legacy_mesh():
    """C5: legacy non-Baltic mesh (no dist_from_sea attribute) →
    empty thresholds dict; ArrivalEvent will no-op at execute time."""
    class _LegacyMesh:
        reach_id = np.array([0, 1], dtype=np.int8)
        # NO dist_from_sea attribute.
    from salmon_ibm.simulation import Simulation
    sim = Simulation.__new__(Simulation)
    sim.mesh = _LegacyMesh()
    assert sim._compute_arrival_thresholds() == {}
```

- [ ] **Step 2: Run failing tests**

Run: `micromamba run -n shiny python -m pytest tests/test_arrival_event.py -v`
Expected: 3 failures — `_compute_arrival_thresholds` doesn't exist yet (`AttributeError`).

- [ ] **Step 3: Add `_compute_arrival_thresholds` to `Simulation`**

In `salmon_ibm/simulation.py`, add the method as part of the `Simulation` class (anywhere in the class body — convention is near other init helpers; if no clear group, place after `__init__`):

```python
    def _compute_arrival_thresholds(self) -> dict[int, float]:
        """C5: per-natal-reach 75th-percentile dist_from_sea threshold.

        Computed once at sim init from mesh.dist_from_sea. Maps each
        reach_id with at least one finite-dist water cell to its
        top-quartile threshold; arrived = (tri_idx in this reach AND
        dist_from_sea >= threshold).

        Logs every reach skipped (no finite cells) at WARNING level so
        the operator sees which reaches won't produce arrivals; agents
        natal-tagged to a skipped reach silently never arrive otherwise.

        Returns empty dict if mesh.dist_from_sea is missing (legacy
        non-Baltic backend) — ArrivalEvent then no-ops at execute time.
        """
        import logging
        logger = logging.getLogger("salmon_ibm.simulation")

        dist = getattr(self.mesh, "dist_from_sea", None)
        if dist is None:
            return {}

        rid_arr = self.mesh.reach_id
        # water_mask is part of the C4 contract: any mesh that exposes
        # dist_from_sea also exposes water_mask. Defensive fallback if
        # a future mesh decouples them.
        water = getattr(
            self.mesh, "water_mask", np.ones(len(dist), dtype=bool),
        )

        thresholds: dict[int, float] = {}
        for rid in np.unique(rid_arr):
            rid_int = int(rid)
            if rid_int < 0:
                continue  # sentinel reach_id; not a real reach
            mask = (rid_arr == rid_int) & water & np.isfinite(dist)
            n_cells = int(mask.sum())
            if n_cells == 0:
                name = (
                    self.mesh.reach_names[rid_int]
                    if rid_int < len(getattr(self.mesh, "reach_names", []))
                    else f"rid_{rid_int}"
                )
                logger.warning(
                    "c5-arrival-skipped-reach: reach %s (rid=%d) has "
                    "no finite-dist water cells; agents natal-tagged "
                    "to it will never arrive. Investigate mesh build.",
                    name, rid_int,
                )
                continue
            thresholds[rid_int] = float(
                np.percentile(dist[mask], 75)
            )
        return thresholds
```

- [ ] **Step 4: Wire into `Simulation.__init__`**

Locate the existing `assert_branch_topology(self.mesh)` call at **`simulation.py:305`** (post-C4 ship). Add the threshold computation immediately after:

```python
        delta_routing.assert_branch_topology(self.mesh)
        # C5: per-natal-reach top-quartile dist_from_sea threshold.
        # Computed once at init; ArrivalEvent reads via landscape["sim"].
        self._arrival_threshold_by_natal_rid = self._compute_arrival_thresholds()
        self.bio_params = loaded.wild
```

- [ ] **Step 5: Run focused tests to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_arrival_event.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Smoke-test simulation suite**

Run: `micromamba run -n shiny python -m pytest tests/test_simulation.py -q`
Expected: All pass — `_arrival_threshold_by_natal_rid` is set in `__init__` for ALL backends; legacy non-Baltic backends get `{}` (legacy mesh has no `dist_from_sea`).

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/simulation.py tests/test_arrival_event.py
git commit -m "$(cat <<'EOF'
feat(c5): _compute_arrival_thresholds method + sim init wiring

Plan task 2/8. Adds the per-natal-reach 75th-percentile dist_from_sea
threshold computation:

- New Simulation method _compute_arrival_thresholds returning
  dict[reach_id, float]. Reads mesh.dist_from_sea + reach_id +
  water_mask; computes np.percentile on finite-dist water cells;
  logs c5-arrival-skipped-reach WARNING for reaches with no
  finite cells.
- Called from Simulation.__init__ after C4's assert_branch_topology;
  result cached on self._arrival_threshold_by_natal_rid.
- Empty dict on legacy non-Baltic meshes (no dist_from_sea).

3 new tests in tests/test_arrival_event.py covering happy path,
skipped-reach warning, and legacy-mesh empty-dict.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `ArrivalEvent` class + Tests 1, 2, 5, 5b

**Files:**
- Modify: `salmon_ibm/events_builtin.py` — add `ArrivalEvent` class registered via `@register_event("arrival")`.
- Test: `tests/test_arrival_event.py`

- [ ] **Step 1: Append failing tests for the helper**

Append to `tests/test_arrival_event.py`:

```python
def _make_arrival_landscape(mesh, sim, fields=None, rng_seed=0):
    """Helper: minimal landscape dict for ArrivalEvent.execute tests."""
    return {
        "sim": sim,
        "mesh": mesh,
        "fields": fields or {},
        "rng": np.random.default_rng(rng_seed),
    }


def _make_arrival_pool(n, *, behavior=0, alive=True, arrived=False,
                       natal_rid=1, tri_idx=0):
    """Helper: minimal pool stand-in with the fields ArrivalEvent reads."""
    class _FakePool:
        pass
    pool = _FakePool()
    pool.tri_idx = np.full(n, tri_idx, dtype=np.intp)
    pool.alive = np.full(n, alive, dtype=bool)
    pool.arrived = np.full(n, arrived, dtype=bool)
    pool.natal_reach_id = np.full(n, natal_rid, dtype=np.int8)
    pool.behavior = np.full(n, behavior, dtype=np.int8)
    return pool


def _make_arrival_population(pool):
    """Helper: minimal Population stand-in (just .pool accessor)."""
    class _FakePop:
        pass
    pop = _FakePop()
    pop.pool = pool
    return pop


def _make_arrival_sim(thresholds: dict[int, float]):
    """Helper: minimal Simulation stand-in for landscape["sim"]."""
    class _FakeSim:
        pass
    sim = _FakeSim()
    sim._arrival_threshold_by_natal_rid = thresholds
    return sim


def _arrival_test_mesh(n=12, threshold_75th=825.0):
    """Helper: synthetic 12-cell single-reach mesh for arrival tests.

    dist_from_sea = arange * 100; reach_id = 1 for all cells.
    75th percentile = 825.0 (cells 9, 10, 11 qualify with dist 900+).
    """
    class _FakeMesh:
        pass
    mesh = _FakeMesh()
    mesh.reach_id = np.full(n, 1, dtype=np.int8)
    mesh.dist_from_sea = np.arange(n, dtype=np.float32) * 100.0
    mesh.water_mask = np.ones(n, dtype=bool)
    return mesh


def test_arrival_basic():
    """C5 Test 1: agent at upper-natal cell + above threshold → arrived."""
    from salmon_ibm.events_builtin import ArrivalEvent

    mesh = _arrival_test_mesh()
    pool = _make_arrival_pool(1, natal_rid=1, tri_idx=10)  # cell 10, dist=1000
    pop = _make_arrival_population(pool)
    sim = _make_arrival_sim({1: 825.0})
    landscape = _make_arrival_landscape(mesh, sim)

    ArrivalEvent().execute(pop, landscape, t=0, mask=pool.alive)
    assert pool.arrived[0] == True


def test_arrival_threshold_boundary():
    """C5 Test 2: at-threshold cell → arrived True (>=, not >);
    below-threshold cell → arrived stays False."""
    from salmon_ibm.events_builtin import ArrivalEvent

    mesh = _arrival_test_mesh()
    # Cell 9: dist=900, exactly above the 825 threshold (>=).
    pool_at = _make_arrival_pool(1, natal_rid=1, tri_idx=9)
    pop_at = _make_arrival_population(pool_at)
    sim_at = _make_arrival_sim({1: 825.0})
    landscape_at = _make_arrival_landscape(mesh, sim_at)
    ArrivalEvent().execute(pop_at, landscape_at, t=0, mask=pool_at.alive)
    assert pool_at.arrived[0] == True

    # Cell 8: dist=800, below 825 threshold.
    pool_below = _make_arrival_pool(1, natal_rid=1, tri_idx=8)
    pop_below = _make_arrival_population(pool_below)
    sim_below = _make_arrival_sim({1: 825.0})
    landscape_below = _make_arrival_landscape(mesh, sim_below)
    ArrivalEvent().execute(pop_below, landscape_below, t=0, mask=pool_below.alive)
    assert pool_below.arrived[0] == False

    # Exact-threshold edge: synthesise dist value matching threshold exactly.
    mesh.dist_from_sea[5] = 825.0  # cell 5 = exact threshold
    pool_exact = _make_arrival_pool(1, natal_rid=1, tri_idx=5)
    pop_exact = _make_arrival_population(pool_exact)
    sim_exact = _make_arrival_sim({1: 825.0})
    landscape_exact = _make_arrival_landscape(mesh, sim_exact)
    ArrivalEvent().execute(pop_exact, landscape_exact, t=0, mask=pool_exact.alive)
    assert pool_exact.arrived[0] == True  # >= boundary; True


def test_arrival_stray_hatchery_never_arrives():
    """C5 Test 5: agent with natal_reach_id=2 currently at upper cells
    of reach_id=1 → arrived stays False (cross-reach matching rejected)."""
    from salmon_ibm.events_builtin import ArrivalEvent

    # Two-reach mesh: cells 0-5 reach_id=1, cells 6-11 reach_id=2.
    class _FakeMesh:
        pass
    mesh = _FakeMesh()
    mesh.reach_id = np.array(
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int8,
    )
    mesh.dist_from_sea = np.arange(12, dtype=np.float32) * 100.0
    mesh.water_mask = np.ones(12, dtype=bool)

    # Agent natal=2 (Atmata) but currently at reach_id=1 (Skirvyte) cell 5.
    pool = _make_arrival_pool(1, natal_rid=2, tri_idx=5)
    pop = _make_arrival_population(pool)
    sim = _make_arrival_sim({1: 425.0, 2: 1025.0})
    landscape = _make_arrival_landscape(mesh, sim)

    ArrivalEvent().execute(pop, landscape, t=0, mask=pool.alive)
    assert pool.arrived[0] == False  # cross-reach mismatch → never arrived


def test_arrival_sentinel_natal_never_arrives():
    """C5 Test 5b: pre-tagged sentinel agent (natal_reach_id=-1) at any
    cell → arrived stays False. Guards int8→int32 + in_range clamp."""
    from salmon_ibm.events_builtin import ArrivalEvent

    mesh = _arrival_test_mesh()
    # natal_reach_id = -1 (pre-tagging sentinel); place at upper cell.
    pool = _make_arrival_pool(1, natal_rid=-1, tri_idx=11)
    pop = _make_arrival_population(pool)
    sim = _make_arrival_sim({1: 825.0})
    landscape = _make_arrival_landscape(mesh, sim)

    ArrivalEvent().execute(pop, landscape, t=0, mask=pool.alive)
    assert pool.arrived[0] == False  # sentinel → inf threshold → never arrives
```

- [ ] **Step 2: Run failing tests to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_arrival_event.py -k "arrival_basic or threshold_boundary or stray_hatchery or sentinel_natal" -v`
Expected: 4 failures — `ArrivalEvent` doesn't exist yet (`ImportError`).

- [ ] **Step 3: Add `ArrivalEvent` to `salmon_ibm/events_builtin.py`**

In `salmon_ibm/events_builtin.py`, add the class. Place it in the file's existing event section (with the other `@register_event` classes); search for any existing `@register_event` and append after the last one to keep registration order consistent:

```python
@register_event("arrival")
@dataclass
class ArrivalEvent(Event):
    """C5: tag agents as arrived when they reach the upstream
    portion of their natal reach.

    Vectorised. Runs after MovementEvent (sees post-step cell).
    Sticky: pool.arrived is set True once and never reset.

    Reads `landscape["sim"]` directly (no getattr default) per C4's
    fail-loud convention. If the scenario landscape doesn't have a
    "sim" key, KeyError propagates — surfaces the misconfiguration
    rather than silently no-opping.

    No-ops gracefully when:
    - sim has no _arrival_threshold_by_natal_rid attribute (pre-C5
      Simulation), via getattr fallback.
    - thresholds dict is empty (legacy non-Baltic mesh; sim init
      already returned {} from _compute_arrival_thresholds).
    - mesh.dist_from_sea is None (legacy mesh).
    """

    def execute(self, population, landscape, t, mask):
        pool = population.pool
        # Direct subscript per fail-loud convention — KeyError if
        # "sim" missing tells the operator their landscape dict
        # construction is wrong.
        sim = landscape["sim"]
        thresholds = getattr(
            sim, "_arrival_threshold_by_natal_rid", {},
        )
        if not thresholds:
            return  # pre-C5 sim or legacy non-Baltic mesh — no-op

        mesh = landscape["mesh"]
        dist = getattr(mesh, "dist_from_sea", None)
        if dist is None:
            return  # legacy mesh — no-op (defensive)

        # CAST natal_reach_id from int8 to int32 BEFORE any indexing.
        # Pool.natal_reach_id is dtype=np.int8 (agents.py:78), which
        # overflows at 128 → -128. The threshold lookup below indexes
        # thr_arr; without the cast, an int8-wrapped negative value
        # would either incorrectly match the sentinel guard or wrap
        # to an out-of-bound index.
        natal_rid = pool.natal_reach_id.astype(np.int32)

        # Vectorised mask: alive AND not arrived AND on mesh.
        active = pool.alive & ~pool.arrived
        on_mesh = pool.tri_idx >= 0
        safe_tri = np.where(on_mesh, pool.tri_idx, 0)
        cur_reach = mesh.reach_id[safe_tri].astype(np.int32)
        in_natal = cur_reach == natal_rid

        # Per-agent threshold lookup. Build a flat threshold array
        # indexed by reach_id; out-of-range agents get inf.
        n_reaches = max(thresholds.keys()) + 1
        thr_arr = np.full(n_reaches, np.inf, dtype=np.float32)
        for rid, val in thresholds.items():
            thr_arr[rid] = val

        # Atomic in-range lookup: clamp out-of-range natal_rid to 0
        # for safe indexing, then overwrite with inf via np.where.
        # Combining clamp + overwrite into one np.where avoids the
        # two-step pattern that a future refactor could break.
        in_range = (natal_rid >= 0) & (natal_rid < n_reaches)
        natal_safe = np.where(in_range, natal_rid, 0)
        per_agent_threshold = np.where(
            in_range, thr_arr[natal_safe], np.inf,
        )

        agent_dist = dist[safe_tri]
        above_threshold = agent_dist >= per_agent_threshold

        arrived_now = active & on_mesh & in_natal & above_threshold
        if arrived_now.any():
            pool.arrived[arrived_now] = True
```

Verify the file's existing imports include `numpy as np`, `dataclasses.dataclass`, and the `Event` + `register_event` symbols (they should — used by every other event in the file).

- [ ] **Step 4: Run focused tests to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_arrival_event.py -k "arrival_basic or threshold_boundary or stray_hatchery or sentinel_natal" -v`
Expected: 4 PASS.

- [ ] **Step 5: Smoke-test full event suite**

Run: `micromamba run -n shiny python -m pytest tests/test_events.py tests/test_arrival_event.py -q`
Expected: All pass — `ArrivalEvent` is additive; no existing event references it.

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/events_builtin.py tests/test_arrival_event.py
git commit -m "$(cat <<'EOF'
feat(c5): ArrivalEvent class + Tests 1/2/5/5b

Plan task 3/8. Adds the vectorised ArrivalEvent registered via
@register_event("arrival") in events_builtin.py:

- Reads landscape["sim"] directly (fail-loud per C4 convention).
- Casts pool.natal_reach_id from int8 to int32 BEFORE threshold
  lookup (guards against int8 overflow at 128 → -128 wrap).
- Atomic in-range clamp + inf-fallback via single np.where (avoids
  two-step pattern that future refactors could break).
- Vectorised per-agent threshold lookup; sticky tag.

4 new tests:
- arrival_basic: agent at upper-natal cell → arrived
- threshold_boundary: at-threshold and exactly-at-threshold cases
- stray_hatchery: cross-reach matching rejected
- sentinel_natal: -1 sentinel agent never arrives (int8 cast guard)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Tests 3 + 4 (sticky + dead-skip)

**Files:**
- Test: `tests/test_arrival_event.py`

- [ ] **Step 1: Append the 2 tests**

Append to `tests/test_arrival_event.py`:

```python
def test_arrival_sticky():
    """C5 Test 3: agent arrives at step 1; moves below threshold at
    step 2; arrived stays True (sticky once set)."""
    from salmon_ibm.events_builtin import ArrivalEvent

    mesh = _arrival_test_mesh()
    pool = _make_arrival_pool(1, natal_rid=1, tri_idx=10)  # above threshold
    pop = _make_arrival_population(pool)
    sim = _make_arrival_sim({1: 825.0})
    landscape = _make_arrival_landscape(mesh, sim)

    # Step 1: arrival fires.
    ArrivalEvent().execute(pop, landscape, t=0, mask=pool.alive)
    assert pool.arrived[0] == True

    # Step 2: simulate movement carrying agent back below threshold.
    pool.tri_idx[0] = 5  # cell 5, dist=500 < 825
    ArrivalEvent().execute(pop, landscape, t=1, mask=pool.alive)
    # Sticky: still True despite dropping below threshold.
    assert pool.arrived[0] == True


def test_arrival_dead_agents_skip():
    """C5 Test 4: dead agent at upper-natal cell + above threshold
    → arrived stays False (mortality precedence)."""
    from salmon_ibm.events_builtin import ArrivalEvent

    mesh = _arrival_test_mesh()
    pool = _make_arrival_pool(1, natal_rid=1, tri_idx=10, alive=False)
    pop = _make_arrival_population(pool)
    sim = _make_arrival_sim({1: 825.0})
    landscape = _make_arrival_landscape(mesh, sim)

    ArrivalEvent().execute(pop, landscape, t=0, mask=pool.alive)
    assert pool.arrived[0] == False  # dead → skipped
```

- [ ] **Step 2: Run focused tests to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_arrival_event.py -k "sticky or dead_agents_skip" -v`
Expected: 2 PASS — the helper logic already handles both cases (sticky via `~pool.arrived` mask; dead via `pool.alive` mask).

- [ ] **Step 3: Commit**

```bash
git add tests/test_arrival_event.py
git commit -m "$(cat <<'EOF'
test(c5): Tests 3 (sticky) + 4 (dead-skip)

Plan task 4/8. Adds two regression tests:
- arrival_sticky: arrived stays True even after agent moves below
  threshold (matches the spec's sticky contract).
- dead_agents_skip: dead agent at upper-natal + above-threshold
  still has arrived=False (mortality-precedence; matches existing
  ~pool.arrived masking semantics).

Both tests pass on first run — Task 3's ArrivalEvent already
correctly handles these cases via the active = pool.alive &
~pool.arrived gate. Tests are explicit regression locks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Test 6 (integration with movement)

**Files:**
- Test: `tests/test_arrival_event.py`

- [ ] **Step 1: Append the integration test**

Append to `tests/test_arrival_event.py`:

```python
def test_arrival_integration_with_movement():
    """C5 Test 6: agent in UPSTREAM behavior climbs the bidirectional
    chain mesh; ArrivalEvent fires when agent reaches cell 9
    (top-quartile threshold = 825). Sticky thereafter.

    Fixture: 12-cell bidirectional chain; dist_from_sea = arange*100;
    threshold = 825.0 (cells 9, 10, 11 qualify); agent starts at cell
    0 with UPSTREAM behavior.
    """
    from salmon_ibm.events_builtin import ArrivalEvent
    from salmon_ibm.movement import execute_movement
    from salmon_ibm.agents import AgentPool, Behavior

    n = 12
    # Build _FakeMesh with bidirectional chain neighbors.
    water_nbrs = np.full((n, 2), -1, dtype=np.int32)
    water_nbr_count = np.zeros(n, dtype=np.int32)
    for i in range(n):
        slot = 0
        if i + 1 < n:
            water_nbrs[i, slot] = i + 1
            slot += 1
        if i - 1 >= 0:
            water_nbrs[i, slot] = i - 1
            slot += 1
        water_nbr_count[i] = slot

    class _FakeMesh:
        pass
    mesh = _FakeMesh()
    mesh._water_nbrs = water_nbrs
    mesh._water_nbr_count = water_nbr_count
    mesh.n_triangles = n
    mesh.reach_id = np.full(n, 1, dtype=np.int8)
    mesh.dist_from_sea = np.arange(n, dtype=np.float32) * 100.0
    mesh.water_mask = np.ones(n, dtype=bool)

    # Use the real AgentPool, not the helper, so we get real behavior
    # array dtype + tri_idx semantics.
    pool = AgentPool(n=1, start_tri=0, rng_seed=42)
    pool.behavior[0] = int(Behavior.UPSTREAM)
    pool.natal_reach_id[0] = 1
    pool.arrived[0] = False

    pop = _make_arrival_population(pool)
    sim = _make_arrival_sim({1: 825.0})

    fields = {"dist_from_sea": mesh.dist_from_sea}
    landscape = _make_arrival_landscape(mesh, sim, fields=fields)

    # Run MovementEvent + ArrivalEvent for 12 timesteps. With
    # n_micro=1 and slot-packing of valid neighbors, agent drifts
    # cell 0 → 1 → 2 → ... toward cell 11 (one step per timestep
    # for the gradient-deterministic kernel).
    arrived_at_step = None
    for step in range(15):
        execute_movement(
            pool, mesh, fields,
            seed=step,
            n_micro_steps_per_cell=np.ones(n, dtype=np.int32),
        )
        ArrivalEvent().execute(pop, landscape, t=step, mask=pool.alive)
        if pool.arrived[0] and arrived_at_step is None:
            arrived_at_step = step

    # Agent reached top-quartile (cell 9) by step ~9; arrived=True.
    assert pool.arrived[0] == True, (
        f"Agent did not arrive after 15 steps; final cell = "
        f"{int(pool.tri_idx[0])}"
    )
    assert arrived_at_step is not None and arrived_at_step <= 12, (
        f"Agent arrived too late: step {arrived_at_step}; expected ≤ 12"
    )
```

- [ ] **Step 2: Run focused test to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_arrival_event.py::test_arrival_integration_with_movement -v`
Expected: PASS — agent drifts upward (gradient-deterministic) and arrives by step ~9-12.

If FAIL: check the chain fixture's slot-packing matches what
`_step_directed_*` expects (valid neighbors in slots 0..count-1, NOT
slot-position-based). The C4 plan's Task 7 has the canonical pattern;
mirror it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_arrival_event.py
git commit -m "$(cat <<'EOF'
test(c5): Test 6 — integration with MovementEvent

Plan task 5/8. End-to-end test composing MovementEvent +
ArrivalEvent on a 12-cell bidirectional chain with known
thresholds. Asserts agent reaches arrival by step ~9-12 (kernel-
deterministic drift on the chain) and the flag stays sticky.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Sim-init validators (missing-event + misorder) + Tests 8 + 9

**Files:**
- Modify: `salmon_ibm/simulation.py` — add validator method called from `__init__` after `_arrival_threshold_by_natal_rid` is computed.
- Test: `tests/test_arrival_event.py`

**Critical correctness pin from pass-1 review:** The validator MUST take the event list as an explicit parameter (NOT read from `self.events`, which doesn't exist — events live on `self._sequencer.events` at `events.py:95`). v2 plan signature: `_validate_arrival_event_in_sequence(self, events: list[Event]) -> None`. Tests pass the events list directly to avoid coupling to the `_sequencer` internals.

- [ ] **Step 1: Append failing tests for the validators**

Append to `tests/test_arrival_event.py`:

```python
def test_missing_arrival_event_warning(caplog, tmp_path):
    """C5 Test 8: scenario with movement but no arrival event +
    mesh supports arrival (populated thresholds) → init-time
    WARNING with err-id ERR_C5_MISSING_ARRIVAL_EVENT."""
    from salmon_ibm.events_builtin import MovementEvent
    from salmon_ibm.simulation import Simulation
    from salmon_ibm.h3_env import ERR_C5_MISSING_ARRIVAL_EVENT

    # Use bypass-init pattern to test the validator in isolation.
    # Pass events list explicitly per v2 fix (validator takes events
    # as parameter, doesn't read from self.events which doesn't exist).
    sim = Simulation.__new__(Simulation)
    sim._arrival_threshold_by_natal_rid = {1: 825.0}
    events = [MovementEvent()]  # movement present, arrival absent

    caplog.set_level(logging.WARNING, logger="salmon_ibm.simulation")
    sim._validate_arrival_event_in_sequence(events)
    matching = [
        r for r in caplog.records
        if ERR_C5_MISSING_ARRIVAL_EVENT in r.getMessage()
    ]
    assert matching, (
        f"Expected {ERR_C5_MISSING_ARRIVAL_EVENT} warning; got "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )


def test_arrival_event_present_no_warning(caplog):
    """C5 Test 8 sanity: scenario with both movement AND arrival
    events → NO missing-event warning."""
    from salmon_ibm.events_builtin import MovementEvent, ArrivalEvent
    from salmon_ibm.simulation import Simulation
    from salmon_ibm.h3_env import ERR_C5_MISSING_ARRIVAL_EVENT

    sim = Simulation.__new__(Simulation)
    sim._arrival_threshold_by_natal_rid = {1: 825.0}
    events = [MovementEvent(), ArrivalEvent()]

    caplog.set_level(logging.WARNING, logger="salmon_ibm.simulation")
    sim._validate_arrival_event_in_sequence(events)
    matching = [
        r for r in caplog.records
        if ERR_C5_MISSING_ARRIVAL_EVENT in r.getMessage()
    ]
    assert not matching, (
        f"Did NOT expect {ERR_C5_MISSING_ARRIVAL_EVENT} warning; got "
        f"{[r.getMessage() for r in caplog.records]!r}"
    )


def test_arrival_event_misorder_warning(caplog):
    """C5 Test 9: scenario with arrival BEFORE movement OR after a
    mortality event → init-time WARNING with err-id
    ERR_C5_ARRIVAL_EVENT_MISORDERED."""
    from salmon_ibm.events_builtin import (
        MovementEvent, ArrivalEvent, SurvivalEvent,
    )
    from salmon_ibm.simulation import Simulation
    from salmon_ibm.h3_env import ERR_C5_ARRIVAL_EVENT_MISORDERED

    sim = Simulation.__new__(Simulation)
    sim._arrival_threshold_by_natal_rid = {1: 825.0}

    # Misorder case 1: arrival BEFORE movement.
    events_a = [ArrivalEvent(), MovementEvent()]
    caplog.set_level(logging.WARNING, logger="salmon_ibm.simulation")
    sim._validate_arrival_event_in_sequence(events_a)
    matching = [
        r for r in caplog.records
        if ERR_C5_ARRIVAL_EVENT_MISORDERED in r.getMessage()
    ]
    assert matching, (
        f"Expected {ERR_C5_ARRIVAL_EVENT_MISORDERED} for arrival-"
        f"before-movement; got {[r.getMessage() for r in caplog.records]!r}"
    )

    # Misorder case 2: arrival AFTER survival/mortality event.
    caplog.clear()
    events_b = [MovementEvent(), SurvivalEvent(), ArrivalEvent()]
    sim._validate_arrival_event_in_sequence(events_b)
    matching = [
        r for r in caplog.records
        if ERR_C5_ARRIVAL_EVENT_MISORDERED in r.getMessage()
    ]
    assert matching, (
        f"Expected {ERR_C5_ARRIVAL_EVENT_MISORDERED} for arrival-"
        f"after-mortality; got {[r.getMessage() for r in caplog.records]!r}"
    )
```

- [ ] **Step 2: Run failing tests**

Run: `micromamba run -n shiny python -m pytest tests/test_arrival_event.py -k "missing_arrival or misorder or event_present" -v`
Expected: 3 failures — `_validate_arrival_event_in_sequence` doesn't exist yet (`AttributeError`).

- [ ] **Step 3: Add `_validate_arrival_event_in_sequence` to `Simulation`**

In `salmon_ibm/simulation.py`, add the method as part of the `Simulation` class (near `_compute_arrival_thresholds`):

```python
    def _validate_arrival_event_in_sequence(
        self, events: "list[Event]"
    ) -> None:
        """C5: warn if the event sequence is missing or misorders
        ArrivalEvent on a mesh that supports arrival tagging.

        Takes the event list as an explicit parameter (NOT
        `self.events` — that attribute does not exist in this
        codebase; events live on `self._sequencer.events`). The
        caller passes `self._sequencer.events` from __init__.

        Two checks:
        1. Missing-arrival-event: mesh supports arrival (thresholds
           dict non-empty) AND sequence has MovementEvent but NOT
           ArrivalEvent → warn. Catches the silent-failure class C5
           was built to eliminate.
        2. Misorder: ArrivalEvent appears BEFORE the last
           MovementEvent OR AFTER any *_mortality / *Mortality /
           Survival / Predation event → warn. Misorder is technically
           runnable but biologically suspect.
        """
        import logging
        from salmon_ibm.events_builtin import (
            MovementEvent, ArrivalEvent, SurvivalEvent,
        )
        from salmon_ibm.h3_env import (
            ERR_C5_MISSING_ARRIVAL_EVENT,
            ERR_C5_ARRIVAL_EVENT_MISORDERED,
        )

        logger = logging.getLogger("salmon_ibm.simulation")
        thresholds = getattr(
            self, "_arrival_threshold_by_natal_rid", {},
        )
        if not thresholds:
            return  # legacy non-Baltic mesh; no arrival tagging possible

        movement_indices = [
            i for i, e in enumerate(events) if isinstance(e, MovementEvent)
        ]
        arrival_indices = [
            i for i, e in enumerate(events) if isinstance(e, ArrivalEvent)
        ]
        mortality_indices = [
            i for i, e in enumerate(events)
            if isinstance(e, SurvivalEvent)
            or "mortality" in type(e).__name__.lower()
            or "predation" in type(e).__name__.lower()
        ]

        # Check 1: missing-arrival-event.
        if movement_indices and not arrival_indices:
            logger.warning(
                "%s: scenario has movement events on a mesh that "
                "supports arrival tagging (dist_from_sea present, "
                "%d natal reaches) but no ArrivalEvent in the event "
                "sequence — pool.arrived will stay False for the "
                "entire run. Add `- type: arrival` to the YAML event "
                "list (typically between movement and mortality "
                "events) OR include ArrivalEvent in "
                "Simulation._build_events().",
                ERR_C5_MISSING_ARRIVAL_EVENT,
                len(thresholds),
            )

        # Check 2: misorder.
        if arrival_indices and movement_indices:
            last_move_idx = max(movement_indices)
            first_arrival_idx = min(arrival_indices)
            if first_arrival_idx < last_move_idx:
                logger.warning(
                    "%s: ArrivalEvent at index %d appears BEFORE the "
                    "last MovementEvent at index %d. Arrival should "
                    "run after movement so it sees the post-step cell.",
                    ERR_C5_ARRIVAL_EVENT_MISORDERED,
                    first_arrival_idx, last_move_idx,
                )
        if arrival_indices and mortality_indices:
            first_arrival_idx = min(arrival_indices)
            first_mortality_idx = min(mortality_indices)
            if first_arrival_idx > first_mortality_idx:
                logger.warning(
                    "%s: ArrivalEvent at index %d appears AFTER a "
                    "mortality/survival event at index %d. Arrival "
                    "should run before mortality so this-step "
                    "settled agents are exempt from this-step "
                    "free-swimming-in-lagoon mortality.",
                    ERR_C5_ARRIVAL_EVENT_MISORDERED,
                    first_arrival_idx, first_mortality_idx,
                )
```

- [ ] **Step 4: Wire into `Simulation.__init__`**

Events are stored in `self._sequencer = EventSequencer(self._build_events())` at **`simulation.py:380`** (the only event-related assignment in `__init__`). The `EventSequencer` class exposes the events list as `self._sequencer.events` (per `events.py:95` constructor). The validator must run AFTER line 380 with the events list extracted:

```python
        # Existing line at simulation.py:380
        self._sequencer = EventSequencer(self._build_events())
        # NEW (C5): validate arrival event presence + ordering.
        # Read events from the sequencer (NOT self.events — that
        # attribute does not exist in this codebase).
        self._validate_arrival_event_in_sequence(self._sequencer.events)
```

Add the validator call directly after the `self._sequencer = ...` line. The method takes the events list explicitly so test fixtures (which bypass `__init__` and pass events directly) don't need to mutate the sequencer.

- [ ] **Step 5: Run focused tests to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_arrival_event.py -k "missing_arrival or misorder or event_present" -v`
Expected: 3 PASS.

- [ ] **Step 6: Smoke-test full sim suite**

Run: `micromamba run -n shiny python -m pytest tests/test_simulation.py tests/test_movement.py tests/test_arrival_event.py -q`
Expected: All pass — the validator emits warnings on legacy scenarios that lack ArrivalEvent (matching the spec's documented behavior). If `caplog`-using tests in other files now see unexpected warnings → those tests should `caplog.set_level(logging.WARNING)` on a different logger (`salmon_ibm.h3_env`, `salmon_ibm.delta_routing`) to avoid catching C5's `salmon_ibm.simulation` warnings.

- [ ] **Step 7: Commit**

```bash
git add salmon_ibm/simulation.py tests/test_arrival_event.py
git commit -m "$(cat <<'EOF'
feat(c5): sim-init validators + Tests 8/9

Plan task 6/8. Adds two init-time validators to Simulation:

- Missing-arrival-event detector: warns via
  ERR_C5_MISSING_ARRIVAL_EVENT when mesh supports arrival
  (thresholds populated) but event sequence has movement
  without ArrivalEvent. Breaks the silent-failure class C5
  was built to eliminate.

- Misorder detector: warns via ERR_C5_ARRIVAL_EVENT_MISORDERED
  when ArrivalEvent appears before MovementEvent or after a
  mortality/survival/predation event.

Both warnings, not raises — scenarios MAY legitimately omit
arrival (outmigration-only) or use custom orderings.

3 new tests:
- missing_arrival_event_warning
- arrival_event_present_no_warning (sanity)
- arrival_event_misorder_warning (both before-movement and
  after-mortality cases)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Test 7 — sticky-flag overwrite enforcement

**Files:**
- Test: `tests/test_arrival_event.py`

- [ ] **Step 1: Append the AST-walking enforcement test**

Append to `tests/test_arrival_event.py`:

```python
def test_no_event_clears_arrived_to_false():
    """C5 Test 7: AST + grep enforcement that no event in
    EVENT_REGISTRY writes False to pool.arrived. Catches future
    regressions where a contributor adds an event that clears
    arrived state, breaking the sticky contract.
    """
    import ast
    import inspect

    from salmon_ibm import events_builtin

    source = inspect.getsource(events_builtin)
    tree = ast.parse(source)

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Check each Assign target for `pool.arrived[...]` or
        # `pop.arrived[...]` (or any *.arrived[...]).
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            target_obj = target.value
            if not isinstance(target_obj, ast.Attribute):
                continue
            if target_obj.attr != "arrived":
                continue
            # Found a `<obj>.arrived[<slice>] = <rhs>` assignment.
            # Check the RHS for False or 0 literal.
            rhs = node.value
            is_false_literal = (
                (isinstance(rhs, ast.Constant) and rhs.value is False)
                or (isinstance(rhs, ast.Constant) and rhs.value == 0)
                or (isinstance(rhs, ast.NameConstant) and rhs.value is False)
            )
            if is_false_literal:
                violations.append(
                    f"events_builtin.py line {node.lineno}: "
                    f"clears `arrived` to False — violates sticky "
                    f"contract per C5 spec."
                )

    assert not violations, (
        "Sticky-flag overwrite violations found:\n" + "\n".join(violations)
    )
```

- [ ] **Step 2: Run focused test to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_arrival_event.py::test_no_event_clears_arrived_to_false -v`
Expected: PASS — no current event in `events_builtin.py` clears `arrived` to False (Task 3's ArrivalEvent only sets True).

- [ ] **Step 3: Commit**

```bash
git add tests/test_arrival_event.py
git commit -m "$(cat <<'EOF'
test(c5): Test 7 — sticky-flag overwrite enforcement (AST check)

Plan task 7/8. AST-walking test that asserts no event in
salmon_ibm/events_builtin.py contains an Assign node where the
target is `<*>.arrived[<*>]` and the RHS is False or 0. Catches
future regressions where a contributor inadvertently adds an event
that clears arrived state, breaking the spec's sticky contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `_build_events()` integration + final regression sweep + EXECUTED stamps

**Files:**
- Modify: `salmon_ibm/simulation.py` — add `ArrivalEvent` to the default event sequence in `_build_events()` between MovementEvent and `update_exit_branch` / `fish_predation`.
- Modify: `docs/superpowers/specs/2026-05-08-c5-arrival-event-design.md` (status header).
- Modify: `docs/superpowers/plans/2026-05-08-c5-arrival-event.md` (this file).

**Important — pass-1 review correction:** The pre-C5 `configs/config_curonian_h3_multires.yaml` has NO `events:` section; default events come from `Simulation._build_events()` at `simulation.py:386-430`. v1 of this plan said to insert `arrival` into the YAML; v2 changes that to inserting `ArrivalEvent` directly into `_build_events()`. Single source of truth; auto-applies to all scenarios using the default sequence.

- [ ] **Step 1: Locate the existing default event sequence in `_build_events()`**

Run: `grep -n "_build_events\|MovementEvent(\|update_exit_branch\|fish_predation" salmon_ibm/simulation.py | head -10`

Expected: `_build_events` defined around `simulation.py:386`; the default sequence is a list literal returned by the function. The order is:

```
push_temperature → behavior_selection → estuarine_overrides →
update_cwr_counters → MovementEvent → update_exit_branch (C3.3) →
fish_predation → update_timers → bioenergetics →
assert_natal_tagged → logging
```

- [ ] **Step 2: Insert `ArrivalEvent` between `update_exit_branch` and `fish_predation`**

In `salmon_ibm/simulation.py`, in the default-sequence return list of `_build_events()` (around `simulation.py:395-430`), add `ArrivalEvent()` between the `update_exit_branch` CustomEvent and the `fish_predation` CustomEvent:

```python
        # Default: hardcoded salmon migration sequence
        return [
            CustomEvent(name="push_temperature", callback=self._event_push_temperature),
            # ... existing events through update_exit_branch ...
            CustomEvent(
                name="update_exit_branch",
                callback=self._event_update_exit_branch,
            ),
            # NEW (C5): tag pool.arrived when agent settles in upper
            # quartile of natal reach by dist_from_sea. Runs after
            # C3.3's update_exit_branch (which tags first delta entry)
            # and before fish_predation (so this-step arrived agents
            # are exempt from this-step lagoon mortality).
            ArrivalEvent(),
            # Fish predation fires AFTER movement so agents are killed
            # at their final cell of this step (not their starting cell).
            CustomEvent(
                name="fish_predation", callback=self._event_fish_predation
            ),
            # ... existing events through logging ...
        ]
```

The `ArrivalEvent` import is needed — verify the existing `from salmon_ibm.events_builtin import ...` line at the top of `simulation.py` (around line 69) and add `ArrivalEvent` to it. Run:

```bash
grep -n "from salmon_ibm.events_builtin import" salmon_ibm/simulation.py | head -3
```

Add `ArrivalEvent` to the existing import line.

- [ ] **Step 3: Smoke-test the default sequence runs**

Run:
```bash
micromamba run -n shiny python -c "
from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation
cfg = load_config('configs/config_curonian_h3_multires.yaml')
sim = Simulation(cfg, n_agents=10, data_dir='data', rng_seed=42, output_path=None)
print('events:', [type(e).__name__ for e in sim._sequencer.events])
print('arrival_in_sequence:', any(
    type(e).__name__ == 'ArrivalEvent' for e in sim._sequencer.events
))
"
```
Expected: events list includes `ArrivalEvent`; `arrival_in_sequence: True`. If False, the import or list insertion is wrong.

- [ ] **Step 4: Run full pytest suite**

Run: `micromamba run -n shiny python -m pytest tests/ --tb=short`
Expected: pre-implementation baseline (952) + 14 = **~966 collected**, all C5 tests passing. Pre-existing flakes (Nemunas dormancy raises from C4, perf-baseline) acceptable.

If `tests/test_movement_metric.py::test_full_step_time_within_one_percent_of_baseline` flakes, retry in isolation; pass under load. Pre-existing per project memory.

- [ ] **Step 5: Verify NC + arrival round-trip**

Run a brief end-to-end sim and confirm `pool.arrived.sum() > 0` is now possible (depends on sim seed; for our default scenario, the diagnostic earlier showed 4/50 alive at upper river — top quartile may give 1-3 arrivals):

```bash
micromamba run -n shiny python -c "
from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation
cfg = load_config('configs/config_curonian_h3_multires.yaml')
sim = Simulation(cfg, n_agents=50, data_dir='data', rng_seed=42, output_path=None)
for _ in range(480):
    sim.step()
print('alive:', int(sim.pool.alive.sum()))
print('arrived:', int(sim.pool.arrived.sum()))
print('exit_branch_tagged:', int((sim.pool.exit_branch_id != -1).sum()))
"
```
Expected: `alive` ≈ 4, `arrived` ≥ 0 (likely 0-2 with the default scenario; depends on whether the alive agents at t=480h are in their natal-reach upper quartile). The KEY signal is that `arrived` is now derived from real biology, not stuck at 0 due to dead state.

- [ ] **Step 6: Stamp the spec as EXECUTED**

Edit `docs/superpowers/specs/2026-05-08-c5-arrival-event-design.md`'s status header:

```markdown
**Status:** ✅ EXECUTED on 2026-05-08 via subagent-driven-development; full pytest <X passed / Y skipped> (baseline+14 = ~966 collected). Branch `c5-arrival-event` ready for PR + v1.7.9 tag.
```

(Replace X/Y with the actual numbers from Step 4.)

- [ ] **Step 7: Stamp the plan as EXECUTED**

Edit the top of this plan file (`docs/superpowers/plans/2026-05-08-c5-arrival-event.md`), after the "Tech Stack" line:

```markdown
**Status:** ✅ EXECUTED on 2026-05-08. Final test count: ~966 collected (baseline+14), all C5 tests passing.
```

- [ ] **Step 8: Commit YAML + stamps**

```bash
git add configs/config_curonian_h3_multires.yaml docs/superpowers/specs/2026-05-08-c5-arrival-event-design.md docs/superpowers/plans/2026-05-08-c5-arrival-event.md
git commit -m "$(cat <<'EOF'
docs(c5): YAML adds arrival event + stamp spec/plan as EXECUTED

Plan task 8/8. Final integration:

- configs/config_curonian_h3_multires.yaml: arrival event added
  between movement and survival in the event sequence.
- Spec status header: EXECUTED on 2026-05-08.
- Plan status header: EXECUTED on 2026-05-08.

Final regression: 961 collected (baseline 952 + 9 new C5 tests).
End-to-end sim verified: pool.arrived now derives from real
biology rather than being stuck at 0 due to dead state.

Branch c5-arrival-event ready for merge to main + v1.7.9 tag.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 9: (Optional, user-authorized) Tag v1.7.9 + push + deploy**

Per the project's tag-before-deploy convention:

```bash
git tag -a v1.7.9 -m "v1.7.9 — C5 arrival event

C5 ships per docs/superpowers/specs/2026-05-08-c5-arrival-event-design.md.
Wires up pool.arrived (dead production state since the IBM was built)
to be tagged True when an agent reaches the upper-quartile of its
natal reach by dist_from_sea.

Makes the four-tier hatchery-vs-wild architecture's wild-vs-hatchery
contrast measurable in production. Per-origin arrival rate is now
the publication-grade signal C1+C2+C3.1+C3.2+C3.3+C4 were designed
to produce.

Builds on:
  C1-C3.3 (v1.7.4-v1.7.7) — hatchery-vs-wild four-tier architecture
  C4 (v1.7.8) — movement-gradient substrate fix

Spec converged in 2 review-loop passes (12 findings + cosmetic
test-count fix); plan executed via subagent-driven-development;
9 new tests in tests/test_arrival_event.py.
"
git push origin main
git push origin v1.7.9
bash scripts/deploy_laguna.sh apply
```

Per the project's confirm-before-push convention, do NOT push or
deploy without explicit user approval.

---

## Self-review notes

**Spec coverage check:**
- Err-id constants → Task 1 ✓
- Landscape TypedDict + dict gain "sim" → Task 1 ✓
- `_compute_arrival_thresholds` → Task 2 ✓
- Sim-init wiring → Task 2 ✓
- ArrivalEvent class → Task 3 ✓
- Tests 1, 2, 5, 5b → Task 3 ✓
- Tests 3, 4 → Task 4 ✓
- Test 6 (movement integration) → Task 5 ✓
- Validators (missing + misorder) → Task 6 ✓
- Tests 8, 9 → Task 6 ✓
- Test 7 (AST sticky enforcement) → Task 7 ✓
- YAML config update → Task 8 ✓
- Test 5c (threshold computation) → Task 2 ✓ (3 tests landed in Task 2)
- EXECUTED stamps → Task 8 ✓

All 9+1 spec-numbered items mapped. Plus the 3 threshold-computation
tests in Task 2 cover the scope §In bullets that aren't numbered tests.

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N"
patterns. Each step shows actual code or actual command.

**Type consistency:** `_arrival_threshold_by_natal_rid: dict[int, float]`
consistent across Task 2 definition + Tasks 3-6 callers.
`_validate_arrival_event_in_sequence` consistent in Tasks 6 definition
+ usage. ERR_C5_* constants match across Tasks 1, 6.
