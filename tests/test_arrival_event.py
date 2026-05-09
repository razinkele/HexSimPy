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


def _make_arrival_landscape(mesh, sim, fields=None, rng_seed=0):
    """Helper: minimal landscape dict for ArrivalEvent.execute tests."""
    return {
        "sim": sim,
        "mesh": mesh,
        "fields": fields or {},
        "rng": np.random.default_rng(rng_seed),
    }


def _make_arrival_pool(n, *, behavior=0, alive=True, arrived=False,
                       natal_rid=1, tri_idx=0, been_to_sea=True):
    """Helper: minimal pool stand-in with the fields ArrivalEvent reads.

    `been_to_sea` defaults True for these pre-C5.1 tests: their intent
    is "agent at upper-natal cell + threshold met → arrived True", and
    the round-trip flag is incidental to that intent (D1 disposition
    per Task 5 plan). D3 tests (stray-natal, sentinel, dead-agent) are
    blocked elsewhere in the mask (in_natal mismatch, inf threshold,
    or alive=False) and remain correct with been_to_sea=True.
    """
    class _FakePool:
        pass
    pool = _FakePool()
    pool.tri_idx = np.full(n, tri_idx, dtype=np.intp)
    pool.alive = np.full(n, alive, dtype=bool)
    pool.arrived = np.full(n, arrived, dtype=bool)
    pool.natal_reach_id = np.full(n, natal_rid, dtype=np.int8)
    pool.behavior = np.full(n, behavior, dtype=np.int8)
    pool.been_to_sea = np.full(n, been_to_sea, dtype=bool)
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
    # C5.1 (D1): pre-C5.1 test where the round-trip flag is incidental
    # to the test's intent (UPSTREAM agent climbs gradient → arrives).
    # The fixture is a single-reach mesh (no at-sea cells), so
    # BeenToSeaEvent would never set this flag during the run; fake the
    # prior at-sea visit here so the mask passes.
    pool.been_to_sea[0] = True

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
    # MovementEvent inherits Event.name (no default) — must pass explicitly.
    events = [MovementEvent(name="movement")]  # movement present, arrival absent

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
    events = [MovementEvent(name="movement"), ArrivalEvent()]

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
    events_a = [ArrivalEvent(), MovementEvent(name="movement")]
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
    events_b = [
        MovementEvent(name="movement"),
        SurvivalEvent(name="survival"),
        ArrivalEvent(),
    ]
    sim._validate_arrival_event_in_sequence(events_b)
    matching = [
        r for r in caplog.records
        if ERR_C5_ARRIVAL_EVENT_MISORDERED in r.getMessage()
    ]
    assert matching, (
        f"Expected {ERR_C5_ARRIVAL_EVENT_MISORDERED} for arrival-"
        f"after-mortality; got {[r.getMessage() for r in caplog.records]!r}"
    )


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


# ---------------------------------------------------------------------------
# C5.1 — _is_baltic discriminator + _at_sea_rid_set resolution (Task 2/9)
# ---------------------------------------------------------------------------

from salmon_ibm.h3_env import ERR_C5_1_AT_SEA_REACHES_MISSING


def test_is_baltic_true_on_curonian_h3_multires(tmp_path):
    """Default Curonian H3 multires config triggers _is_baltic=True."""
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("configs/config_curonian_h3_multires.yaml")
    sim = Simulation(cfg, n_agents=5, data_dir="data", rng_seed=42,
                     output_path=None)
    assert sim._is_baltic is True
    assert len(sim._at_sea_rid_set) >= 1
    assert isinstance(sim._at_sea_rid_set, frozenset)


def test_is_baltic_false_on_columbia(tmp_path):
    """Columbia config has non-Baltic BioParams + no Baltic branches."""
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("config_columbia.yaml")
    sim = Simulation(cfg, n_agents=5, rng_seed=42, output_path=None)
    assert sim._is_baltic is False
    assert sim._at_sea_rid_set == frozenset()


def test_at_sea_reaches_missing_raises(monkeypatch):
    """A Baltic-detected mesh whose reach_names is missing both
    'BalticCoast' and 'OpenBaltic' must raise RuntimeError at sim init."""
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("configs/config_curonian_h3_multires.yaml")
    import salmon_ibm.h3_multires as h3m
    orig_init = h3m.H3MultiResMesh.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.reach_names = [
            r for r in self.reach_names
            if r not in ("BalticCoast", "OpenBaltic")
        ]

    monkeypatch.setattr(h3m.H3MultiResMesh, "__init__", patched_init)
    with pytest.raises(RuntimeError, match=ERR_C5_1_AT_SEA_REACHES_MISSING):
        Simulation(cfg, n_agents=5, data_dir="data", rng_seed=42,
                   output_path=None)


def test_at_sea_partial_miss_warns(monkeypatch, caplog):
    """Baltic-detected mesh with only ONE of {BalticCoast, OpenBaltic}
    in reach_names → warning logged, sim init succeeds with narrower
    _at_sea_rid_set."""
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("configs/config_curonian_h3_multires.yaml")
    import salmon_ibm.h3_multires as h3m
    orig_init = h3m.H3MultiResMesh.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.reach_names = [
            r for r in self.reach_names if r != "OpenBaltic"
        ]

    monkeypatch.setattr(h3m.H3MultiResMesh, "__init__", patched_init)
    with caplog.at_level(logging.WARNING, logger="salmon_ibm.simulation"):
        sim = Simulation(cfg, n_agents=5, data_dir="data", rng_seed=42,
                         output_path=None)
    assert "OpenBaltic" not in sim.mesh.reach_names, (
        "Monkeypatch on H3MultiResMesh.__init__ did not take effect "
        "— test is pass-only-if-patched. Verify Simulation.__init__ "
        "still constructs mesh internally via H3MultiResMesh()."
    )
    assert sim._is_baltic is True
    assert len(sim._at_sea_rid_set) == 1  # only BalticCoast resolved
    assert any(
        "c5.1-at-sea-reach-partial-miss" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# C5.1 — sim-init validators for BeenToSeaEvent + sea-pen detector (Task 6/9)
# ---------------------------------------------------------------------------


def test_missing_been_to_sea_event_raises(monkeypatch):
    """Baltic scenario with ArrivalEvent but no BeenToSeaEvent → raise."""
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("configs/config_curonian_h3_multires.yaml")
    orig_build = Simulation._build_events

    def patched_build(self):
        events = orig_build(self)
        from salmon_ibm.events_builtin import BeenToSeaEvent
        return [e for e in events if not isinstance(e, BeenToSeaEvent)]

    monkeypatch.setattr(Simulation, "_build_events", patched_build)
    with pytest.raises(RuntimeError, match="c5.1-been-to-sea-missing"):
        Simulation(cfg, n_agents=5, data_dir="data", rng_seed=42,
                   output_path=None)


def test_been_to_sea_misorder_raises(monkeypatch):
    """BeenToSeaEvent placed AFTER ArrivalEvent → raise."""
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("configs/config_curonian_h3_multires.yaml")
    orig_build = Simulation._build_events

    def patched_build(self):
        events = orig_build(self)
        from salmon_ibm.events_builtin import BeenToSeaEvent, ArrivalEvent
        bts_idx = next(i for i, e in enumerate(events) if isinstance(e, BeenToSeaEvent))
        ae_idx = next(i for i, e in enumerate(events) if isinstance(e, ArrivalEvent))
        if bts_idx < ae_idx:
            bts_evt = events.pop(bts_idx)
            events.insert(ae_idx + 1, bts_evt)
        return events

    monkeypatch.setattr(Simulation, "_build_events", patched_build)
    with pytest.raises(RuntimeError, match="c5.1-been-to-sea-misordered"):
        Simulation(cfg, n_agents=5, data_dir="data", rng_seed=42,
                   output_path=None)


def test_non_baltic_validator_no_complaint(caplog):
    """Columbia/TriMesh scenario without BeenToSeaEvent → no warning,
    no raise. _is_baltic=False suppresses C5.1 checks."""
    import logging
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("config_columbia.yaml")
    with caplog.at_level(logging.WARNING, logger="salmon_ibm.simulation"):
        sim = Simulation(cfg, n_agents=5, rng_seed=42, output_path=None)
    c5_1_warnings = [
        r for r in caplog.records
        if "c5.1-been-to-sea" in r.message
    ]
    assert not c5_1_warnings


def test_sea_pen_hatchery_degenerate_warning(caplog):
    """Baltic scenario where any agent's natal_reach_id ∈ _at_sea_rid_set
    → warning at c5.1-sea-pen-hatchery-degenerate-arrival; no raise.
    """
    import logging
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config("configs/config_curonian_h3_multires.yaml")
    sim = Simulation(cfg, n_agents=5, data_dir="data", rng_seed=42,
                     output_path=None)
    at_sea_rid = next(iter(sim._at_sea_rid_set))
    # Inject degenerate state.
    for pop in sim._multi_pop_mgr.populations.values():
        pop.pool.natal_reach_id[0] = at_sea_rid
        break
    with caplog.at_level(logging.WARNING, logger="salmon_ibm.simulation"):
        sim._validate_arrival_event_in_sequence(sim._sequencer.events)
    sea_pen_warnings = [
        r for r in caplog.records
        if "c5.1-sea-pen-hatchery-degenerate-arrival" in r.message
    ]
    assert len(sea_pen_warnings) >= 1


# ---------------------------------------------------------------------------
# C5.1 — AST sticky-overwrite contract for pool.been_to_sea (Task 7/9)
# ---------------------------------------------------------------------------


def _walk_been_to_sea_violations(tree, attr_name="been_to_sea"):
    """Walk an AST and return list of violation messages for any
    write to <obj>.<attr_name>[...] that doesn't match a sticky-safe
    AST shape:
      (a) Constant True RHS to subscript assign
      (b) AugAssign with BitOr op
      (c) Read-modify-write OR (BinOp BitOr with left matching target)
    Used by both the production-code AST scan and the counter-tests.
    """
    import ast as _ast

    def is_target(node):
        if isinstance(node, _ast.Subscript):
            return is_target(node.value)
        if isinstance(node, _ast.Attribute) and node.attr == attr_name:
            return True
        return False

    def attr_chain(node):
        chain = []
        cur = node
        if isinstance(cur, _ast.Subscript):
            cur = cur.value
        while isinstance(cur, _ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, _ast.Name):
            chain.append(cur.id)
            return tuple(reversed(chain))
        return None

    violations = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.AugAssign):
            if not is_target(node.target):
                continue
            if not isinstance(node.op, _ast.BitOr):
                violations.append(
                    f"line {node.lineno}: AugAssign on {attr_name} "
                    f"uses non-BitOr op {type(node.op).__name__}"
                )
        elif isinstance(node, _ast.Assign):
            for tgt in node.targets:
                # Only flag SUBSCRIPTED writes to <obj>.been_to_sea[...].
                # Bare-attribute Assign (e.g. `self.been_to_sea = np.zeros(n)`
                # in AgentPool.__init__) is field declaration/allocation,
                # not a sticky-contract write — exclude per spec intent.
                # Shapes (a) and (c) both target Subscript; shape (b) is
                # the bare-Attribute case but is an AugAssign, handled above.
                if not isinstance(tgt, _ast.Subscript):
                    continue
                if not is_target(tgt):
                    continue
                rhs = node.value
                # Shape (a): Constant True.
                if isinstance(rhs, _ast.Constant) and rhs.value is True:
                    continue
                # Shape (c): BinOp BitOr with left == target.
                if isinstance(rhs, _ast.BinOp) and isinstance(rhs.op, _ast.BitOr):
                    target_chain = attr_chain(tgt)
                    left_chain = attr_chain(rhs.left)
                    if target_chain and target_chain == left_chain:
                        continue
                    violations.append(
                        f"line {node.lineno}: BinOp on {attr_name} "
                        f"left chain {left_chain} != target chain "
                        f"{target_chain}"
                    )
                    continue
                rhs_repr = (
                    _ast.unparse(rhs) if hasattr(_ast, "unparse")
                    else str(rhs)
                )
                violations.append(
                    f"line {node.lineno}: Assign to {attr_name} has "
                    f"non-True/non-OR RHS: {rhs_repr}"
                )
    return violations


def test_no_event_clears_been_to_sea_to_false():
    """C5.1 sticky-overwrite contract: any write to pool.been_to_sea[...]
    in salmon_ibm/*.py must match one of three sticky-safe shapes:

      (a) Constant True literal RHS to subscript assign
      (b) AugAssign with BitOr op
      (c) Read-modify-write OR with matching target
    """
    import ast
    salmon_ibm_dir = Path(__file__).parent.parent / "salmon_ibm"
    all_violations = []
    for py_file in sorted(salmon_ibm_dir.rglob("*.py")):
        try:
            src = py_file.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (UnicodeDecodeError, SyntaxError):
            continue
        file_violations = _walk_been_to_sea_violations(tree)
        for v in file_violations:
            all_violations.append(f"{py_file.name}: {v}")
    assert not all_violations, (
        "salmon_ibm/ contains writes to been_to_sea that violate the "
        "sticky contract (only constant True, |=, or BinOp(BitOr, "
        "same target) RHS allowed):\n"
        + "\n".join(f"  - {v}" for v in all_violations)
    )


def test_ast_walker_counter_examples():
    """Counter-tests: synthesize AST for forbidden patterns and confirm
    the SAME walker (via _walk_been_to_sea_violations helper) rejects
    them. Eliminates drift risk between production scan and counter-
    test predicates per pass-1 H-3.
    """
    import ast as _ast

    def violations_for(src):
        return _walk_been_to_sea_violations(_ast.parse(src))

    # Accepted patterns (no violations expected):
    assert violations_for("pool.been_to_sea[mask] = True") == []
    assert violations_for("pool.been_to_sea |= mask") == []
    assert violations_for(
        "pool.been_to_sea[mask] = pool.been_to_sea[mask] | new_mask"
    ) == []

    # Rejected patterns (violations expected):
    assert violations_for("pool.been_to_sea[mask] = False") != []
    assert violations_for("pool.been_to_sea[mask] = 0") != []
    assert violations_for("pool.been_to_sea[mask] = arr") != []
    # Different left target than assign target → reject.
    assert violations_for(
        "pool.been_to_sea[mask] = other_pool.been_to_sea[mask] | new_mask"
    ) != []
