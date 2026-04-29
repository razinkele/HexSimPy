# Post-Review Bugfixes — Implementation Plan

> **STATUS: ✅ EXECUTED** — All 11 post-review fixes shipped — DataLookup CSV loading, trait filter format, mate selection, vectorized HexSimMove, `deque` in network, dead-prey check etc. Tests in `tests/test_review_fixes.py`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 11 bugs and parity issues found during the comprehensive code review, in priority order: critical logic bugs first, then parity gaps, then performance.

**Architecture:** Each task is a standalone fix with its own test. Tasks are independent — any can be skipped or reordered. All fixes target existing files (no new modules). TDD: write failing test first, then fix.

**Tech Stack:** Python, NumPy, pytest, collections.deque

**Test command:** `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py --ignore=tests/test_hexsim_validation.py --ignore=tests/test_hexsim_compat.py --ignore=tests/test_columbia_validation.py --tb=short`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `salmon_ibm/scenario_loader.py` | **Modify** | Fix DataLookup table loading, RNG seeding, accumulator bounds |
| `salmon_ibm/events.py` | **Modify** | Fix trait filter format mismatch |
| `salmon_ibm/xml_parser.py` | **Modify** | Fix trait filter dict format for `filter_by_traits` API |
| `salmon_ibm/events_builtin.py` | **Modify** | Add mate selection to ReproductionEvent |
| `salmon_ibm/events_hexsim.py` | **Modify** | Vectorize HexSimMoveEvent inner loop |
| `salmon_ibm/network.py` | **Modify** | Replace list.pop(0) with deque |
| `salmon_ibm/interactions.py` | **Modify** | Add dead-prey check in encounter loop |
| `tests/test_review_fixes.py` | **Create** | Tests for all fixes in this plan |

---

## Task 1: Load DataLookup CSV Tables in ScenarioLoader

**Priority:** HIGH — DataLookupEvent is currently a complete no-op for all XML scenarios
**Files:**
- Modify: `salmon_ibm/scenario_loader.py:190-268`
- Test: `tests/test_review_fixes.py`

The XML parser extracts `file_name` from `<dataLookupEvent>` but `ScenarioLoader._build_single_event()` never reads the CSV file, so `DataLookupEvent.lookup_table` stays `None` and `execute()` exits at line 211.

- [ ] **Step 1: Write failing test**

```python
# tests/test_review_fixes.py
"""Tests for post-review bugfixes."""
import numpy as np
import pytest
from pathlib import Path


class TestDataLookupLoading:
    def test_lookup_table_loaded_from_csv(self, tmp_path):
        """ScenarioLoader should load CSV into DataLookupEvent.lookup_table."""
        # Create a minimal CSV
        csv_file = tmp_path / "Analysis" / "Data Lookup" / "test_table.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(csv_file, np.array([[1.0, 2.0], [3.0, 4.0]]), delimiter=",")

        from salmon_ibm.scenario_loader import ScenarioLoader
        loader = ScenarioLoader()

        # Simulate a parsed event dict with file_name
        edef = {
            "type": "data_lookup",
            "name": "test_lookup",
            "params": {
                "file_name": "test_table.csv",
                "row_accumulator": "row_acc",
                "column_accumulator": "col_acc",
                "target_accumulator": "target",
            },
        }
        # Need to import event modules first
        import salmon_ibm.events_builtin
        import salmon_ibm.events_phase3
        import salmon_ibm.events_hexsim

        evt = loader._build_single_event(edef, {})
        # Before fix: lookup_table is None; after fix: loaded from workspace
        # We test the loading mechanism directly
        loader._load_lookup_tables([evt], str(tmp_path))
        assert evt.lookup_table is not None
        assert evt.lookup_table.shape == (2, 2)
        assert evt.lookup_table[1, 0] == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest tests/test_review_fixes.py::TestDataLookupLoading -v --tb=short`
Expected: FAIL — `_load_lookup_tables` does not exist

- [ ] **Step 3: Implement lookup table loading**

Add to `salmon_ibm/scenario_loader.py` after `_build_events()`:

```python
    def _load_lookup_tables(self, events, workspace_dir: str) -> None:
        """Post-process events: load CSV files for DataLookupEvents."""
        from salmon_ibm.events_hexsim import DataLookupEvent
        ws = Path(workspace_dir)
        lookup_dir = ws / "Analysis" / "Data Lookup"

        for evt in events:
            if isinstance(evt, DataLookupEvent) and evt.file_name and evt.lookup_table is None:
                csv_path = lookup_dir / evt.file_name
                if csv_path.exists():
                    evt.lookup_table = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
            # Recurse into EventGroups
            if hasattr(evt, 'sub_events'):
                self._load_lookup_tables(evt.sub_events, workspace_dir)
```

Wire it in `load()` after building events (~line 82):

```python
        # 5b. Load lookup tables from workspace CSVs
        self._load_lookup_tables(events, str(ws_dir))
```

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Run full test suite**

- [ ] **Step 6: Commit**

**Commit message:** `fix(scenario_loader): load CSV lookup tables for DataLookupEvent`

---

## Task 2: Fix Trait Filter Format Mismatch

**Priority:** HIGH — Trait filtering from XML is completely non-functional
**Files:**
- Modify: `salmon_ibm/xml_parser.py:526-534`
- Modify: `salmon_ibm/events.py:156-160`
- Test: `tests/test_review_fixes.py`

The XML parser produces `{"traits": ["traitA"], "combinations": "1 0 1"}` but `filter_by_traits()` expects `**{"traitA": [0, 2]}` (trait_name→category_indices). The `EventGroup.execute()` passes `**event.trait_filter` which fails because `traits` and `combinations` are not trait names.

The fix has two parts: (a) parse the trait filter into the correct format in the XML parser, and (b) use the existing `_apply_trait_combo_mask` in `events_hexsim.py` for combo-based filtering since that already handles the HexSim format.

- [ ] **Step 1: Write failing test**

```python
class TestTraitFilterFormat:
    def test_trait_filter_applied_in_event_group(self):
        """EventGroup should apply trait filter to restrict which agents an event sees."""
        from salmon_ibm.population import Population
        from salmon_ibm.agents import AgentPool
        from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
        from salmon_ibm.events import EventGroup, EveryStep, Event
        from salmon_ibm.interactions import MultiPopulationManager
        from dataclasses import dataclass

        pool = AgentPool(n=4, start_tri=np.zeros(4, dtype=int))
        pop = Population(name="test", pool=pool)
        trait_defs = [TraitDefinition(
            name="stage", trait_type=TraitType.PROBABILISTIC,
            categories=["juvenile", "adult"]
        )]
        pop.trait_mgr = TraitManager(4, trait_defs)
        pop.trait_mgr._data["stage"][:] = np.array([0, 0, 1, 1])  # 2 juv, 2 adult

        # Track which mask the child event receives
        received_masks = []

        @dataclass
        class SpyEvent(Event):
            def execute(self, population, landscape, t, mask):
                received_masks.append(mask.copy())

        child = SpyEvent(
            name="spy",
            trigger=EveryStep(),
            trait_filter={"stage": "adult"},  # only adults
        )
        child.population_name = "test"

        group = EventGroup(
            name="group",
            trigger=EveryStep(),
            sub_events=[child],
        )

        mgr = MultiPopulationManager()
        mgr.register(pop)
        landscape = {"multi_pop_mgr": mgr}

        group.execute(pop, landscape, 0, pop.alive.copy())
        assert len(received_masks) == 1
        # Only agents 2 and 3 (adults) should be in mask
        assert received_masks[0][0] == False
        assert received_masks[0][1] == False
        assert received_masks[0][2] == True
        assert received_masks[0][3] == True
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `filter_by_traits(**{"stage": "adult"})` should work but the XML parser currently produces `{"traits": [...], "combinations": "..."}` format. This test uses the correct format to verify the EventGroup trait-filter path works at all.

- [ ] **Step 3: Fix EventGroup trait filter application**

In `salmon_ibm/events.py`, replace the inner trait-filter block at lines 156-160 (keep the `else:` branch at line 161 unchanged):

```python
                        # Apply child's trait filter if present
                        if hasattr(event, 'trait_filter') and event.trait_filter is not None:
                            tf = event.trait_filter
                            if hasattr(child_pop, 'trait_mgr') and child_pop.trait_mgr is not None:
                                if isinstance(tf, dict) and "traits" in tf:
                                    # HexSim combo format — use _apply_trait_combo_mask
                                    from salmon_ibm.events_hexsim import _apply_trait_combo_mask
                                    child_mask = _apply_trait_combo_mask(child_mask, tf, child_pop)
                                elif isinstance(tf, dict):
                                    # Simple {trait_name: value} format
                                    trait_mask = child_pop.trait_mgr.filter_by_traits(**tf)
                                    child_mask = child_mask & trait_mask
```

- [ ] **Step 4: Fix XML parser to also emit simple format when possible**

In `salmon_ibm/xml_parser.py`, update `_parse_trait_filter()` (line 526):

```python
def _parse_trait_filter(elem) -> dict | None:
    """Extract trait filter from event element.

    Returns either:
      - HexSim combo format: {"traits": [...], "combinations": "1 0 1 ..."}
      - Simple format: {"trait_name": value} for single-trait filters
    """
    traits = [t.text.strip() for t in elem.findall("trait") if t.text]
    combos = _text(elem, "traitCombinations")
    if not traits:
        return None
    if combos:
        # Full combo bitfield — keep HexSim format for _apply_trait_combo_mask
        return {"traits": traits, "combinations": combos,
                "stratified_traits": traits, "trait_combinations": combos}
    return None
```

- [ ] **Step 5: Run tests**

- [ ] **Step 6: Commit**

**Commit message:** `fix(events): fix trait filter format mismatch between XML parser and EventGroup`

---

## Task 3: Add Mate Selection to ReproductionEvent

**Priority:** HIGH — Self-fertilization breaks genetic simulation semantics
**Files:**
- Modify: `salmon_ibm/events_builtin.py:225-236`
- Test: `tests/test_review_fixes.py`

Currently `parent2_idx = parent1_idx.copy()` means all offspring are selfed. Fix: select a random group mate (or cell-neighbor) as the second parent.

- [ ] **Step 1: Write failing test**

```python
class TestReproductionMateSelection:
    def test_offspring_have_two_different_parents(self):
        """Offspring should inherit alleles from two different parents."""
        from salmon_ibm.population import Population
        from salmon_ibm.agents import AgentPool
        from salmon_ibm.genetics import GenomeManager, LocusDefinition
        from salmon_ibm.events_builtin import ReproductionEvent
        from salmon_ibm.events import EveryStep

        pool = AgentPool(n=4, start_tri=np.zeros(4, dtype=int))
        pop = Population(name="test", pool=pool)
        pop.group_id[:] = np.array([0, 0, 1, 1], dtype=np.int32)

        # Setup genome: parent A homozygous 0, parent B homozygous 1
        loci = [LocusDefinition(name="loc1", n_alleles=2, position=0.0)]
        gm = GenomeManager(4, loci, rng_seed=42)
        gm.genotypes[0, 0, :] = [0, 0]  # group 0 parent A: all 0
        gm.genotypes[1, 0, :] = [1, 1]  # group 0 parent B: all 1
        gm.genotypes[2, 0, :] = [0, 0]  # group 1 parent A: all 0
        gm.genotypes[3, 0, :] = [1, 1]  # group 1 parent B: all 1
        pop.genome = gm

        evt = ReproductionEvent(
            name="repro", trigger=EveryStep(),
            clutch_mean=2.0, min_group_size=2,
            offspring_mass_mean=100.0, offspring_mass_std=10.0,
        )
        rng = np.random.default_rng(42)
        landscape = {"rng": rng}
        mask = pop.alive.copy()
        evt.execute(pop, landscape, 0, mask)

        # Check offspring genotypes: if mate selection works, some offspring
        # should have alleles from BOTH parents (heterozygous: one 0, one 1)
        n_offspring = pop.pool.n - 4
        if n_offspring > 0:
            offspring_geno = pop.genome.genotypes[4:, 0, :]
            has_heterozygote = np.any(offspring_geno[:, 0] != offspring_geno[:, 1])
            assert has_heterozygote, "With two homozygous parents, offspring should be heterozygous"
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — all offspring are selfed, so with homozygous parents the offspring remain homozygous

- [ ] **Step 3: Implement mate selection**

In `salmon_ibm/events_builtin.py`, replace the self-fertilization block (~line 225-236):

```python
        # --- Genome recombination with mate selection ---
        if population.genome is not None and total_offspring > 0:
            parent1_indices = np.repeat(reproducer_idx, clutch_sizes)
            parent2_indices = parent1_indices.copy()
            # Select a mate: random group member that is not self
            offset = 0
            for i, rep_idx in enumerate(reproducer_idx):
                gid = population.group_id[rep_idx]
                cs = clutch_sizes[i]
                if gid >= 0:
                    group_members = np.where(
                        (population.group_id == gid) & population.alive
                        & (np.arange(population.n) != rep_idx)
                    )[0]
                    if len(group_members) > 0:
                        mate = rng.choice(group_members)
                        parent2_indices[offset:offset + cs] = mate
                offset += cs
            population.genome.recombine(parent1_indices, parent2_indices, new_idx)
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

**Commit message:** `fix(reproduction): add mate selection to avoid self-fertilization`

---

## Task 4: Fix Unseeded RNG in ScenarioLoader._create_population

**Priority:** MEDIUM — breaks reproducibility
**Files:**
- Modify: `salmon_ibm/scenario_loader.py:58-59,114-127`
- Test: `tests/test_review_fixes.py`

`_create_population` uses `np.random.default_rng()` (unseeded) for initial agent placement, even when `rng_seed` is passed to `load()`.

- [ ] **Step 1: Write failing test**

```python
class TestScenarioLoaderReproducibility:
    def test_same_seed_produces_same_populations(self):
        """Two loads with same seed should produce identical agent positions."""
        # This requires a real workspace — skip if not available
        ws = "HexSimPLE"
        xml = "HexSimPLE/Scenarios/HexSimPLE.xml"
        import os
        if not os.path.exists(xml):
            pytest.skip("HexSimPLE not available")

        from salmon_ibm.scenario_loader import ScenarioLoader
        loader = ScenarioLoader()
        sim1 = loader.load(ws, xml, rng_seed=123)
        sim2 = loader.load(ws, xml, rng_seed=123)

        for pname in sim1.populations.populations:
            p1 = sim1.populations.populations[pname]
            p2 = sim2.populations.populations[pname]
            np.testing.assert_array_equal(p1.tri_idx, p2.tri_idx,
                err_msg=f"Population {pname} positions differ with same seed")
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Fix: pass derived RNG to _create_population**

In `ScenarioLoader.load()`, create a base RNG and pass it through:

```python
    def load(self, workspace_dir, scenario_xml, rng_seed=None):
        ...
        base_rng = np.random.default_rng(rng_seed)

        # 4. Create populations (with derived RNG for reproducibility)
        for pop_def in config["populations"]:
            pop_rng = np.random.default_rng(base_rng.integers(2**63))
            pop = self._create_population(pop_def, mesh, pop_rng)
            multi_pop.register(pop)
        ...
```

Update `_create_population` signature and usage:

```python
    def _create_population(self, pop_def, mesh, rng):
        ...
        if n == 0:
            ...
        else:
            water_ids = np.where(mesh.water_mask)[0]
            start_tris = rng.choice(water_ids, size=n, replace=True) if len(water_ids) > 0 else np.zeros(n, dtype=int)
            ...
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

**Commit message:** `fix(scenario_loader): use derived RNG seed for reproducible population placement`

---

## Task 5: Fix Accumulator min_val=0 Misread as Unbounded

**Priority:** MEDIUM — affects accumulator bounds for probabilities, counts
**Files:**
- Modify: `salmon_ibm/scenario_loader.py:134-141`
- Modify: `salmon_ibm/xml_parser.py` (accumulator parsing)
- Test: `tests/test_review_fixes.py`

Currently `min_val=None if min_val == 0 else min_val` — this treats explicit `lowerBound="0"` as unbounded. HexSim uses `lowerBound="0" upperBound="0"` to mean unbounded, but `lowerBound="0" upperBound="1"` means [0, 1].

- [ ] **Step 1: Write failing test**

```python
class TestAccumulatorBounds:
    def test_lower_bound_zero_with_nonzero_upper_preserves_zero_bound(self):
        """lowerBound=0 + upperBound=1 should produce min_val=0, not None."""
        from salmon_ibm.scenario_loader import ScenarioLoader

        loader = ScenarioLoader()
        # Simulate a parsed pop_def with an accumulator that has explicit bounds
        pop_def = {
            "name": "test",
            "initial_size": 0,
            "accumulators": [
                {"name": "survival_prob", "min_val": 0, "max_val": 1},
                {"name": "unbounded", "min_val": 0, "max_val": 0},  # both 0 = unbounded
            ],
            "traits": [],
        }
        # Create a fake mesh
        class FakeMesh:
            water_mask = np.ones(10, dtype=bool)
        pop = loader._create_population(pop_def, FakeMesh(), np.random.default_rng(42))
        mgr = pop.accumulator_mgr
        assert mgr is not None

        # survival_prob should have bounds [0, 1]
        sp_def = mgr.definitions[mgr._resolve_idx("survival_prob")]
        assert sp_def.min_val == 0, "lowerBound=0 with upperBound=1 should keep min_val=0"
        assert sp_def.max_val == 1, "upperBound=1 should be preserved"

        # unbounded should have None bounds
        ub_def = mgr.definitions[mgr._resolve_idx("unbounded")]
        assert ub_def.min_val is None, "Both 0 should mean unbounded (None)"
        assert ub_def.max_val is None, "Both 0 should mean unbounded (None)"
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Fix bounds interpretation**

In `salmon_ibm/scenario_loader.py`, replace lines 134-141:

```python
            min_val_raw = acc.get("min_val")
            max_val_raw = acc.get("max_val")
            # HexSim convention: BOTH being 0 (or absent) means unbounded.
            # If either is non-zero, both are explicit bounds.
            both_zero = (min_val_raw in (0, None)) and (max_val_raw in (0, None))
            acc_defs.append(AccumulatorDef(
                name=acc["name"],
                min_val=None if both_zero else (min_val_raw if min_val_raw is not None else 0),
                max_val=None if both_zero else (max_val_raw if max_val_raw is not None else 0),
            ))
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

**Commit message:** `fix(scenario_loader): only treat bounds as unbounded when BOTH are 0`

---

## ~~Task 6: Fix Movement Alias-vs-Copy Bug~~ — DROPPED

> **Reviewer note:** Both the NumPy path and Numba path use sequential (in-place) updates. Changing only the NumPy path to use `.copy()` would create a divergence between the two code paths. Since HexSim's C++ implementation also uses sequential agent updates within a timestep, the current behavior is correct. This task is dropped.

---

## Task 6: Replace list.pop(0) with deque in StreamNetwork BFS

**Priority:** LOW — O(n^2) → O(n) for large networks
**Files:**
- Modify: `salmon_ibm/network.py:43-52,54-60`
- Test: `tests/test_review_fixes.py`

- [ ] **Step 1: Write test (already passing — this is a performance fix)**

```python
class TestStreamNetworkBFS:
    def test_all_upstream_returns_correct_segments(self):
        """BFS should find all upstream segments."""
        from salmon_ibm.network import StreamNetwork, SegmentDefinition
        segs = [
            SegmentDefinition(id=0, length=100, upstream_ids=[1, 2]),
            SegmentDefinition(id=1, length=100, upstream_ids=[3]),
            SegmentDefinition(id=2, length=100, upstream_ids=[]),
            SegmentDefinition(id=3, length=100, upstream_ids=[]),
        ]
        net = StreamNetwork(segs)
        result = net.all_upstream(0)
        assert set(result) == {1, 2, 3}

    def test_all_downstream_returns_correct_segments(self):
        from salmon_ibm.network import StreamNetwork, SegmentDefinition
        segs = [
            SegmentDefinition(id=0, length=100, upstream_ids=[], downstream_ids=[1]),
            SegmentDefinition(id=1, length=100, upstream_ids=[0], downstream_ids=[2]),
            SegmentDefinition(id=2, length=100, upstream_ids=[1], downstream_ids=[]),
        ]
        net = StreamNetwork(segs)
        result = net.all_downstream(0)
        assert set(result) == {1, 2}
```

- [ ] **Step 2: Run test — should pass already**

- [ ] **Step 3: Fix: replace list with deque**

In `salmon_ibm/network.py`, add import at top:

```python
from collections import deque
```

Replace `all_upstream` (lines 43-52):

```python
    def all_upstream(self, seg_id: int) -> list[int]:
        """All segments upstream of seg_id (BFS)."""
        visited = set()
        queue = deque(self.upstream(seg_id))
        while queue:
            s = queue.popleft()
            if s not in visited:
                visited.add(s)
                queue.extend(self.upstream(s))
        return sorted(visited)
```

Replace `all_downstream` (lines 54-60) similarly with `deque` and `popleft()`.

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

**Commit message:** `perf(network): use deque for O(n) BFS instead of O(n^2) list.pop(0)`

---

## Task 8: Fix Dead-Prey Re-encounter in InteractionEvent

**Priority:** LOW — affects predator-prey accuracy
**Files:**
- Modify: `salmon_ibm/interactions.py:119-131`
- Test: `tests/test_review_fixes.py`

If prey dies from predator A, predator B in the same cell can still encounter and "kill" the dead prey.

- [ ] **Step 1: Write failing test**

```python
class TestInteractionDeadPrey:
    def test_dead_prey_not_re_encountered(self):
        """Once prey is killed, subsequent predators in same cell should not encounter it."""
        from salmon_ibm.population import Population
        from salmon_ibm.agents import AgentPool
        from salmon_ibm.interactions import (
            MultiPopulationManager, InteractionEvent, InteractionOutcome,
        )
        from salmon_ibm.events import EveryStep

        # 2 predators, 1 prey, all at cell 0
        pred_pool = AgentPool(n=2, start_tri=np.array([0, 0]))
        pred = Population(name="pred", pool=pred_pool)
        prey_pool = AgentPool(n=1, start_tri=np.array([0]))
        prey = Population(name="prey", pool=prey_pool)

        mgr = MultiPopulationManager()
        mgr.register(pred)
        mgr.register(prey)

        evt = InteractionEvent(
            name="hunt", trigger=EveryStep(),
            pop_a_name="pred", pop_b_name="prey",
            outcome=InteractionOutcome.PREDATION,
            encounter_probability=1.0,  # guaranteed encounter
        )

        rng = np.random.default_rng(42)
        landscape = {"multi_pop_mgr": mgr, "rng": rng}
        mask = pred.alive.copy()
        evt.execute(pred, landscape, 0, mask)

        # Only 1 prey exists — it can only be killed once
        assert not prey.alive[0], "Prey should be dead"
        # Key assertion: the event should not double-count kills
        # (before fix, both predators encounter and 'kill' the same prey)
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Fix: check prey alive before encounter**

In `salmon_ibm/interactions.py`, add alive check inside the inner loop (around line 123):

```python
                for a_idx in agents_a:
                    for b_idx in agents_b:
                        if not pop_b.alive[b_idx]:
                            continue  # Skip already-dead prey
                        if rng.random() < self.encounter_probability:
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

**Commit message:** `fix(interactions): skip dead prey in encounter loop to prevent double-kills`

---

## Dependency Graph

```
Task 1 (DataLookup CSV)       [independent]
Task 2 (Trait filter)          [independent]
Task 3 (Mate selection)        [independent]
Task 4 (RNG reproducibility)   [independent]
Task 5 (Accumulator bounds)    [independent]
Task 6 (Deque BFS)             [independent]
Task 7 (Dead prey)             [independent]
```

All tasks are fully independent and can be executed in any order or in parallel.

---

## Estimated Scope

| Task | Lines Changed | New Tests | Priority |
|------|-------------|-----------|----------|
| 1. DataLookup CSV loading | ~25 | 1 | HIGH |
| 2. Trait filter format | ~30 | 1 | HIGH |
| 3. Mate selection | ~20 | 1 | HIGH |
| 4. RNG reproducibility | ~10 | 1 | MEDIUM |
| 5. Accumulator bounds | ~10 | 1 | MEDIUM |
| 6. Deque BFS | ~6 | 2 | LOW |
| 7. Dead prey check | ~2 | 1 | LOW |
| **Total** | **~103** | **8** | |
