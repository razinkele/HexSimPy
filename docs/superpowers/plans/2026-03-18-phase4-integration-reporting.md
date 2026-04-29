# Phase 4: Integration, Reporting, and Compatibility — Implementation Plan

> **STATUS: 🟡 PARTIAL** — Core wiring shipped: barriers in `execute_movement`, genome on Population, all 25 accumulator updaters, `reporting.py` exists. **Indefinitely deferred**: the planned `CensusEvent` class (zero refs in `reporting.py`) and the four prescribed test files (`test_reporting.py`, `test_xml_loader.py`, `test_hexsim_basic.py`, `test_integration_wiring.py`). The XML loader scope was absorbed by `2026-03-19-phase-a-xml-parser.md` and shipped as `xml_parser.py` + `scenario_loader.py`. The reporting use case is covered differently by `OutputLogger` columnar arrays. **Do not retry this plan as written** — open a fresh plan if CensusEvent-style reporting is needed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire all Phase 2-3 components together, add reporting/census events, remaining accumulator updaters, and XML scenario loading — making the system usable for real HexSim scenarios.

**Architecture:** Phase 4 is primarily a wiring and completion phase. The barrier arrays, genome manager, multi-population manager, and stream network all exist as standalone modules but are not connected into the movement engine, reproduction pipeline, simulation loop, or YAML config loader. This phase closes those gaps, then adds a reporting layer (census CSV, binary snapshots, summary statistics) and an XML scenario parser to load native HexSim `.xml` files. No new data structures are invented; instead, existing interfaces are connected through the `landscape` dict, `Population` attributes, and `config.py` loaders.

**Tech Stack:** NumPy, xml.etree, dataclasses, existing event engine

**Test command:** `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `salmon_ibm/movement.py` | Wire `_resolve_barriers_vec` into `execute_movement` | **Modify** |
| `salmon_ibm/events_builtin.py` | Pass barrier_arrays through MovementEvent; wire ReproductionEvent to genetics | **Modify** |
| `salmon_ibm/population.py` | Add `genome: GenomeManager \| None` attribute | **Modify** |
| `salmon_ibm/simulation.py` | Expand landscape dict with barriers, genome, multi_pop_mgr, network | **Modify** |
| `salmon_ibm/config.py` | Add genetics, barriers, network, reporting YAML config sections | **Modify** |
| `salmon_ibm/accumulators.py` | Add 17 remaining updater functions | **Modify** |
| `salmon_ibm/reporting.py` | CensusEvent, LogWriter, SummaryReport | **Create** |
| `salmon_ibm/xml_loader.py` | Parse HexSim .xml scenario files into event sequences | **Create** |
| `tests/test_integration_wiring.py` | Integration tests for barrier+movement, genetics+reproduction, config loading | **Create** |
| `tests/test_accumulators_remaining.py` | Tests for the 17 new updater functions | **Create** |
| `tests/test_reporting.py` | Census, log writer, summary report tests | **Create** |
| `tests/test_xml_loader.py` | XML scenario parser tests | **Create** |
| `tests/test_hexsim_basic.py` | HexSim "Basic" example scenario compatibility test | **Create** |

---

## Task 1: Wire Barriers into execute_movement

**Priority:** Highest (integration wiring)
**Files:**
- Modify: `salmon_ibm/movement.py`
- Modify: `salmon_ibm/events_builtin.py`
- Create: `tests/test_integration_wiring.py`

Currently `_resolve_barriers_vec` exists in `movement.py` (lines 338-370) but `execute_movement` (line 26) never calls it. `MovementEvent.execute` (events_builtin.py line 23) doesn't accept or pass barrier arrays.

- [ ] **Step 1: Add `barrier_arrays` parameter to `execute_movement`**

```python
# salmon_ibm/movement.py — modify signature at line 26
def execute_movement(pool, mesh, fields, seed=None, n_micro_steps=3,
                     cwr_threshold=16.0, barrier_arrays=None):
```

After each movement kernel (RANDOM, UPSTREAM, DOWNSTREAM, TO_CWR) stores its result in `pool.tri_idx[idx]`, insert a barrier resolution call if `barrier_arrays is not None`. The pattern for each kernel block is:

```python
    # After: pool.tri_idx[idx] = tri_buf
    if barrier_arrays is not None:
        b_mort, b_defl, b_trans = barrier_arrays
        old_pos = pool.tri_idx[idx].copy()  # save pre-move positions
        # ... (existing movement code writes to tri_buf) ...
        pool.tri_idx[idx] = tri_buf
        final, died = _resolve_barriers_vec(
            old_pos, pool.tri_idx[idx], b_mort, b_defl, b_trans,
            mesh._water_nbrs, rng,
        )
        pool.tri_idx[idx] = final
        pool.alive[idx[died]] = False
```

Implementation detail: save `old_pos = pool.tri_idx[idx].copy()` **before** writing `tri_buf` to `pool.tri_idx[idx]`, then call `_resolve_barriers_vec(old_pos, tri_buf, ...)` and apply the result.

- [ ] **Step 2: Pass barrier_arrays through MovementEvent**

```python
# salmon_ibm/events_builtin.py — modify MovementEvent.execute
def execute(self, population, landscape, t, mask):
    mesh = landscape["mesh"]
    fields = landscape["fields"]
    rng = landscape["rng"]
    barrier_arrays = landscape.get("barrier_arrays")  # NEW
    execute_movement(
        population, mesh, fields,
        seed=int(rng.integers(2**31)),
        n_micro_steps=self.n_micro_steps,
        cwr_threshold=self.cwr_threshold,
        barrier_arrays=barrier_arrays,  # NEW
    )
```

- [ ] **Step 3: Write tests**

```python
# tests/test_integration_wiring.py
"""Integration tests: Phase 2-3 components wired into simulation pipeline."""
import numpy as np
import pytest
from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.movement import execute_movement, _resolve_barriers_vec


class TestBarrierMovementIntegration:
    def test_execute_movement_with_barriers_deflects(self):
        """Agents moving into a fully-deflecting barrier are bounced back."""
        # Create a small 4-cell mesh, barrier on edge 0->1
        # barrier_arrays = (mort, defl, trans) shaped (n_cells, max_nbrs)
        # Set defl[0, slot_for_nbr_1] = 1.0
        # Place agent at cell 0, behavior=UPSTREAM with gradient toward cell 1
        # After execute_movement, agent should remain at cell 0
        ...

    def test_execute_movement_with_barriers_kills(self):
        """Agents moving into a lethal barrier die."""
        # Same setup but mort[0, slot] = 1.0 instead of defl
        # After execute_movement, agent.alive should be False
        ...

    def test_execute_movement_without_barriers_unchanged(self):
        """When barrier_arrays is None, behavior is identical to before."""
        ...
```

**Test:** `conda run -n shiny python -m pytest tests/test_integration_wiring.py::TestBarrierMovementIntegration -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `feat(movement): wire _resolve_barriers_vec into execute_movement pipeline`

---

## Task 2: Add genome Attribute to Population

**Priority:** Highest (integration wiring)
**Files:**
- Modify: `salmon_ibm/population.py`
- Modify: `tests/test_population.py`

Currently `Population` has `accumulator_mgr` and `trait_mgr` but no `genome` attribute. Phase 3 events (`MutationEvent` in `events_phase3.py` line 18) already do `getattr(population, 'genome', None)` but it's never set.

- [ ] **Step 1: Add genome field to Population**

```python
# salmon_ibm/population.py — add import at top
from salmon_ibm.genetics import GenomeManager

# Add field to Population dataclass (after trait_mgr)
    genome: GenomeManager | None = None
```

- [ ] **Step 2: Expand add_agents to resize genome arrays**

In `Population.add_agents()`, after the trait_mgr resizing block (line 184-188), add:

```python
        if self.genome is not None:
            new_geno = np.zeros((n, self.genome.n_loci, 2), dtype=np.int32)
            self.genome.genotypes = np.concatenate([self.genome.genotypes, new_geno])
            self.genome.n_agents = new_n
```

- [ ] **Step 3: Expand compact to shrink genome arrays**

In `Population.compact()`, after the trait_mgr compaction block (line 149-152), add:

```python
        if self.genome is not None:
            self.genome.genotypes = self.genome.genotypes[alive_idx].copy()
            self.genome.n_agents = n_new
```

- [ ] **Step 4: Write tests**

```python
# tests/test_population.py — add tests
class TestPopulationGenome:
    def test_population_genome_none_by_default(self):
        pop = Population(name="test", pool=make_pool(10))
        assert pop.genome is None

    def test_population_with_genome(self):
        pop = Population(name="test", pool=make_pool(10),
                         genome=make_genome(10))
        assert pop.genome.n_agents == 10

    def test_add_agents_resizes_genome(self):
        pop = Population(name="test", pool=make_pool(5),
                         genome=make_genome(5))
        pop.add_agents(3, np.array([0, 0, 0]))
        assert pop.genome.genotypes.shape[0] == 8
        assert pop.genome.n_agents == 8

    def test_compact_shrinks_genome(self):
        pop = Population(name="test", pool=make_pool(5),
                         genome=make_genome(5))
        pop.alive[2] = False
        pop.compact()
        assert pop.genome.genotypes.shape[0] == 4
```

**Test:** `conda run -n shiny python -m pytest tests/test_population.py::TestPopulationGenome -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `feat(population): add genome attribute with add_agents/compact support`

---

## Task 3: Wire ReproductionEvent to Genetics

**Priority:** Highest (integration wiring)
**Files:**
- Modify: `salmon_ibm/events_builtin.py`
- Add to: `tests/test_integration_wiring.py`

Currently `ReproductionEvent.execute` (events_builtin.py line 187) creates offspring positions and traits but never calls `genome.recombine()`. Offspring should inherit recombined genomes from their parents.

- [ ] **Step 1: Add genome recombination to ReproductionEvent**

After the existing offspring creation code (line 207-213), before the trait assignment block:

```python
        # --- Genome recombination ---
        if population.genome is not None and total_offspring > 0:
            # parent1 = reproducer, parent2 = random group mate (or self if no mate)
            parent1_indices = np.repeat(reproducer_idx, clutch_sizes)
            # Find a mate: another alive agent in the same group
            parent2_indices = parent1_indices.copy()  # default: self
            for i, rep_idx in enumerate(reproducer_idx):
                gid = population.group_id[rep_idx]
                if gid >= 0:
                    group_members = np.where(
                        (population.group_id == gid) & population.alive
                        & (np.arange(population.n) != rep_idx)
                    )[0]
                    if len(group_members) > 0:
                        mate = rng.choice(group_members)
                        start = clutch_sizes[:i].sum()
                        end = start + clutch_sizes[i]
                        parent2_indices[start:end] = mate
            population.genome.recombine(parent1_indices, parent2_indices, new_idx)
```

- [ ] **Step 2: Write tests**

```python
# tests/test_integration_wiring.py
class TestReproductionGeneticsIntegration:
    def test_offspring_inherit_recombined_genomes(self):
        """Offspring genotypes are recombinations of parent genotypes."""
        # Create population with 2 agents in same group, genome with 3 loci
        # Set parent genotypes to known values
        # Fire ReproductionEvent
        # Assert offspring genotypes are valid recombinations (alleles come from parents)
        ...

    def test_reproduction_without_genome_unchanged(self):
        """ReproductionEvent works normally when genome is None."""
        ...
```

**Test:** `conda run -n shiny python -m pytest tests/test_integration_wiring.py::TestReproductionGeneticsIntegration -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `feat(reproduction): wire genome recombination into ReproductionEvent`

---

## Task 4: Add Genetics, Barriers, Network Config Sections

**Priority:** Highest (integration wiring)
**Files:**
- Modify: `salmon_ibm/config.py`
- Modify: `tests/test_config.py`

Currently `config.py` only loads `bioenergetics` and `behavior` sections from YAML. The genetics, barriers, and network modules have no config support.

- [ ] **Step 1: Add genetics config parser**

```python
# salmon_ibm/config.py
from salmon_ibm.genetics import GenomeManager, LocusDefinition

def genome_from_config(cfg: dict, n_agents: int) -> GenomeManager | None:
    """Create GenomeManager from optional ``genetics`` YAML section.

    Example YAML:
        genetics:
          loci:
            - name: run_timing
              n_alleles: 4
              position: 0.0
            - name: growth_rate
              n_alleles: 3
              position: 50.0
          rng_seed: 42
    """
    gen_cfg = cfg.get("genetics")
    if not gen_cfg:
        return None
    loci = [LocusDefinition(**loc) for loc in gen_cfg["loci"]]
    seed = gen_cfg.get("rng_seed")
    gm = GenomeManager(n_agents, loci, rng_seed=seed)
    if gen_cfg.get("initialize_random", True):
        gm.initialize_random()
    return gm
```

- [ ] **Step 2: Add barriers config parser**

```python
def barrier_map_from_config(cfg: dict, mesh) -> tuple | None:
    """Load barriers from config and return (BarrierMap, barrier_arrays) or None.

    Example YAML:
        barriers:
          file: barriers.hbf
          classes:
            dam:
              forward: {mortality: 0.1, deflection: 0.8, transmission: 0.1}
              reverse: {mortality: 0.0, deflection: 1.0, transmission: 0.0}
    """
    bar_cfg = cfg.get("barriers")
    if not bar_cfg:
        return None
    from salmon_ibm.barriers import BarrierMap, BarrierClass, BarrierOutcome
    class_config = {}
    for name, cls_def in bar_cfg.get("classes", {}).items():
        fwd = cls_def.get("forward", {})
        rev = cls_def.get("reverse", {})
        class_config[name] = BarrierClass(
            name=name,
            forward=BarrierOutcome(fwd.get("mortality", 0), fwd.get("deflection", 1), fwd.get("transmission", 0)),
            reverse=BarrierOutcome(rev.get("mortality", 0), rev.get("deflection", 1), rev.get("transmission", 0)),
        )
    bmap = BarrierMap.from_hbf(bar_cfg["file"], mesh, class_config=class_config)
    arrays = bmap.to_arrays(mesh)
    return bmap, arrays
```

- [ ] **Step 3: Add network config parser**

```python
def network_from_config(cfg: dict):
    """Create StreamNetwork from optional ``network`` YAML section.

    Example YAML:
        network:
          segments:
            - id: 0
              length: 1000.0
              upstream_ids: []
              downstream_ids: [1]
            - id: 1
              length: 2000.0
              upstream_ids: [0]
              downstream_ids: []
    """
    net_cfg = cfg.get("network")
    if not net_cfg:
        return None
    from salmon_ibm.network import StreamNetwork, SegmentDefinition
    segs = [SegmentDefinition(**s) for s in net_cfg["segments"]]
    return StreamNetwork(segs)
```

- [ ] **Step 4: Add validation for new sections**

```python
# In validate_config(), add validation for genetics, barriers, network sections
    gen = cfg.get("genetics")
    if gen:
        loci = gen.get("loci")
        if not loci or not isinstance(loci, list):
            raise ValueError("genetics.loci must be a non-empty list")
        for loc in loci:
            if "name" not in loc or "n_alleles" not in loc:
                raise ValueError("Each locus must have 'name' and 'n_alleles'")
            if loc["n_alleles"] < 2:
                raise ValueError(f"n_alleles must be >= 2, got {loc['n_alleles']}")

    bar = cfg.get("barriers")
    if bar:
        if "file" not in bar:
            raise ValueError("barriers section must have a 'file' key")
```

- [ ] **Step 5: Write tests**

```python
# tests/test_config.py — add tests
class TestGeneticsConfig:
    def test_genome_from_config_creates_manager(self):
        cfg = {"genetics": {"loci": [{"name": "a", "n_alleles": 3, "position": 0.0}]}}
        gm = genome_from_config(cfg, n_agents=10)
        assert gm is not None
        assert gm.n_loci == 1

    def test_genome_from_config_none_when_absent(self):
        assert genome_from_config({}, n_agents=10) is None

    def test_validate_genetics_bad_n_alleles(self):
        cfg = {"grid": {"type": "hexsim"}, "genetics": {"loci": [{"name": "a", "n_alleles": 1}]}}
        with pytest.raises(ValueError):
            validate_config(cfg)

class TestBarriersConfig:
    def test_barrier_config_requires_file(self):
        cfg = {"grid": {"type": "hexsim"}, "barriers": {"classes": {}}}
        with pytest.raises(ValueError):
            validate_config(cfg)

class TestNetworkConfig:
    def test_network_from_config_creates_network(self):
        cfg = {"network": {"segments": [{"id": 0, "length": 100.0}]}}
        net = network_from_config(cfg)
        assert net is not None
        assert net.n_segments == 1
```

**Test:** `conda run -n shiny python -m pytest tests/test_config.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `feat(config): add YAML config loaders for genetics, barriers, and network`

---

## Task 5: Expand Landscape Dict in Simulation

**Priority:** Highest (integration wiring)
**Files:**
- Modify: `salmon_ibm/simulation.py`
- Add to: `tests/test_simulation.py`

Currently `Simulation.step()` builds a landscape dict with only `mesh`, `fields`, `rng`, `activity_lut`, `est_cfg`. The barrier arrays, genome, multi_pop_mgr, and network are never included.

- [ ] **Step 1: Import new config loaders and build optional components in `__init__`**

```python
# salmon_ibm/simulation.py — add imports
from salmon_ibm.config import genome_from_config, barrier_map_from_config, network_from_config
from salmon_ibm.interactions import MultiPopulationManager

# In __init__, after self.population is created (~line 53):
        # --- Optional Phase 2-3 components ---
        self._genome = genome_from_config(config, n_agents)
        if self._genome is not None:
            self.population.genome = self._genome

        self._barrier_map = None
        self._barrier_arrays = None
        bar_result = barrier_map_from_config(config, self.mesh) if config.get("barriers") else None
        if bar_result is not None:
            self._barrier_map, self._barrier_arrays = bar_result

        self._network = network_from_config(config)

        self._multi_pop_mgr = MultiPopulationManager()
        self._multi_pop_mgr.register(self.population.name, self.population)
```

- [ ] **Step 2: Expand landscape dict in step()**

```python
# In Simulation.step(), modify the landscape dict (~line 181):
        landscape = {
            "mesh": self.mesh,
            "fields": self.env.fields,
            "rng": self._rng,
            "activity_lut": self._activity_lut,
            "est_cfg": self.est_cfg,
            # Phase 2-3 integration
            "barrier_arrays": self._barrier_arrays,
            "genome": self._genome,
            "multi_pop_mgr": self._multi_pop_mgr,
            "network": self._network,
        }
```

- [ ] **Step 3: Write tests**

```python
# tests/test_simulation.py — add tests (or tests/test_integration_wiring.py)
class TestSimulationLandscapeExpansion:
    def test_landscape_contains_barrier_arrays_when_configured(self):
        """When barriers config is present, landscape["barrier_arrays"] is a 3-tuple."""
        ...

    def test_landscape_barrier_arrays_none_when_not_configured(self):
        """When no barriers config, landscape["barrier_arrays"] is None."""
        ...

    def test_landscape_contains_multi_pop_mgr(self):
        """landscape["multi_pop_mgr"] always present and contains primary population."""
        ...

    def test_population_genome_set_when_configured(self):
        """When genetics config present, population.genome is a GenomeManager."""
        ...
```

**Test:** `conda run -n shiny python -m pytest tests/test_integration_wiring.py::TestSimulationLandscapeExpansion -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `feat(simulation): expand landscape dict with barriers, genome, multi_pop_mgr, network`

---

## Task 6: Remaining Accumulator Updaters (17 of 25)

**Priority:** Medium (completion)
**Files:**
- Modify: `salmon_ibm/accumulators.py`
- Create: `tests/test_accumulators_remaining.py`

Currently 8 updaters are implemented: `clear`, `increment`, `stochastic_increment`, `expression`, `time_step`, `individual_id`, `stochastic_trigger`, `quantify_location`. The remaining 17 HexSim updaters need implementation.

- [ ] **Step 1: Resource updaters — Allocated and Explored**

```python
def updater_resources_allocated(manager, acc_name, mask, *, range_alloc, resource_map):
    """Sum resource values over each agent's allocated range cells."""
    idx = manager._resolve_idx(acc_name)
    masked_indices = np.where(mask)[0]
    for i in masked_indices:
        agent_range = range_alloc.get_range(i)
        if agent_range is not None:
            total = resource_map[list(agent_range.cells)].sum()
            manager.data[i, idx] = total

def updater_resources_explored(manager, acc_name, mask, *, positions, resource_map):
    """Write resource value at each agent's current position."""
    idx = manager._resolve_idx(acc_name)
    manager.data[mask, idx] = resource_map[positions[mask]]
```

- [ ] **Step 2: Group Size and Group Sum**

```python
def updater_group_size(manager, acc_name, mask, *, group_ids):
    """Write the number of alive members in each agent's group."""
    idx = manager._resolve_idx(acc_name)
    masked_indices = np.where(mask)[0]
    unique_groups, counts = np.unique(group_ids[masked_indices], return_counts=True)
    group_count_map = dict(zip(unique_groups, counts))
    for i in masked_indices:
        gid = group_ids[i]
        manager.data[i, idx] = float(group_count_map.get(gid, 0))

def updater_group_sum(manager, acc_name, mask, *, group_ids, source_acc_name):
    """Sum a source accumulator across all members of each agent's group."""
    idx = manager._resolve_idx(acc_name)
    src_idx = manager._resolve_idx(source_acc_name)
    masked_indices = np.where(mask)[0]
    unique_groups = np.unique(group_ids[masked_indices])
    for gid in unique_groups:
        if gid < 0:
            continue
        members = masked_indices[group_ids[masked_indices] == gid]
        total = manager.data[members, src_idx].sum()
        manager.data[members, idx] = total
```

- [ ] **Step 3: Births counter**

```python
def updater_births(manager, acc_name, mask, *, n_offspring_per_agent):
    """Record the number of offspring produced by each agent this timestep."""
    idx = manager._resolve_idx(acc_name)
    manager.data[mask, idx] = n_offspring_per_agent[mask].astype(np.float64)
```

- [ ] **Step 4: Mate Verification**

```python
def updater_mate_verification(manager, acc_name, mask, *, group_ids, min_group_size=2):
    """Write 1.0 if agent's group has >= min_group_size members, else 0.0."""
    idx = manager._resolve_idx(acc_name)
    masked_indices = np.where(mask)[0]
    unique_groups, counts = np.unique(group_ids[masked_indices], return_counts=True)
    group_count_map = dict(zip(unique_groups, counts))
    for i in masked_indices:
        gid = group_ids[i]
        has_mate = 1.0 if gid >= 0 and group_count_map.get(gid, 0) >= min_group_size else 0.0
        manager.data[i, idx] = has_mate
```

- [ ] **Step 5: Hexagon-based updaters — Allocated, Explored, Presence**

```python
def updater_allocated_hexagons(manager, acc_name, mask, *, range_alloc):
    """Write the number of hexagons in each agent's allocated range."""
    idx = manager._resolve_idx(acc_name)
    masked_indices = np.where(mask)[0]
    for i in masked_indices:
        agent_range = range_alloc.get_range(i)
        n_cells = len(agent_range.cells) if agent_range is not None else 0
        manager.data[i, idx] = float(n_cells)

def updater_explored_hexagons(manager, acc_name, mask, *, positions, exploration_tracker):
    """Write number of unique cells each agent has visited."""
    idx = manager._resolve_idx(acc_name)
    masked_indices = np.where(mask)[0]
    for i in masked_indices:
        exploration_tracker.setdefault(i, set()).add(int(positions[i]))
        manager.data[i, idx] = float(len(exploration_tracker[i]))

def updater_hexagon_presence(manager, acc_name, mask, *, positions, cell_id):
    """Write 1.0 if agent is at specified cell, else 0.0."""
    idx = manager._resolve_idx(acc_name)
    manager.data[mask, idx] = (positions[mask] == cell_id).astype(np.float64)
```

- [ ] **Step 6: Uptake**

```python
def updater_uptake(manager, acc_name, mask, *, resource_map, positions, uptake_rate=1.0):
    """Agents consume resources at their position; accumulator stores consumed amount."""
    idx = manager._resolve_idx(acc_name)
    defn = manager.definitions[idx]
    masked_indices = np.where(mask)[0]
    available = resource_map[positions[masked_indices]]
    consumed = np.minimum(available, uptake_rate)
    resource_map[positions[masked_indices]] -= consumed  # mutates resource_map in-place
    new_vals = manager.data[masked_indices, idx] + consumed
    if defn.max_val is not None:
        new_vals = np.minimum(new_vals, defn.max_val)
    manager.data[masked_indices, idx] = new_vals
```

- [ ] **Step 7: Quantify Extremes**

```python
def updater_quantify_extremes(manager, acc_name, mask, *, source_acc_name, mode="max"):
    """Track running min or max of a source accumulator.

    mode: "max" or "min"
    """
    idx = manager._resolve_idx(acc_name)
    src_idx = manager._resolve_idx(source_acc_name)
    current_extremes = manager.data[mask, idx]
    source_vals = manager.data[mask, src_idx]
    if mode == "max":
        manager.data[mask, idx] = np.maximum(current_extremes, source_vals)
    elif mode == "min":
        manager.data[mask, idx] = np.minimum(current_extremes, source_vals)
```

- [ ] **Step 8: Trait Value Index**

```python
def updater_trait_value_index(manager, acc_name, mask, *, trait_mgr, trait_name):
    """Write the integer category index of a trait as a float accumulator value."""
    idx = manager._resolve_idx(acc_name)
    trait_vals = trait_mgr.get(trait_name)
    manager.data[mask, idx] = trait_vals[mask].astype(np.float64)
```

- [ ] **Step 9: Data Lookup**

```python
def updater_data_lookup(manager, acc_name, mask, *, lookup_table, key_acc_name):
    """Look up values from a table using another accumulator as the key.

    lookup_table: dict[int, float] or np.ndarray indexed by int key
    """
    idx = manager._resolve_idx(acc_name)
    key_idx = manager._resolve_idx(key_acc_name)
    keys = manager.data[mask, key_idx].astype(int)
    if isinstance(lookup_table, np.ndarray):
        keys = np.clip(keys, 0, len(lookup_table) - 1)
        manager.data[mask, idx] = lookup_table[keys]
    else:
        masked_indices = np.where(mask)[0]
        for i, k in zip(masked_indices, keys):
            manager.data[i, idx] = float(lookup_table.get(int(k), 0.0))
```

- [ ] **Step 10: Subpopulation Assign and Selector**

```python
def updater_subpopulation_assign(manager, acc_name, mask, *, trait_mgr, trait_name, category_name):
    """Write 1.0 if agent belongs to the named trait category, else 0.0."""
    idx = manager._resolve_idx(acc_name)
    defn = trait_mgr.definitions[trait_name]
    cat_idx = defn.categories.index(category_name)
    trait_vals = trait_mgr.get(trait_name)
    manager.data[mask, idx] = (trait_vals[mask] == cat_idx).astype(np.float64)

def updater_subpopulation_selector(manager, acc_name, mask, *, selector_acc_name, threshold=0.5):
    """Write 1.0 if selector accumulator >= threshold, else 0.0.

    Used to create derived sub-population masks from accumulator values.
    """
    idx = manager._resolve_idx(acc_name)
    sel_idx = manager._resolve_idx(selector_acc_name)
    sel_vals = manager.data[mask, sel_idx]
    manager.data[mask, idx] = (sel_vals >= threshold).astype(np.float64)
```

- [ ] **Step 11: Write tests for all 17 updaters**

Each updater gets at least 2 tests: one positive case verifying correct output, one edge case (empty mask, no group members, etc.). Tests follow the pattern of existing `tests/test_accumulators.py`.

```python
# tests/test_accumulators_remaining.py
"""Tests for the remaining 17 accumulator updater functions."""
import numpy as np
import pytest
from salmon_ibm.accumulators import (
    AccumulatorManager, AccumulatorDef,
    updater_resources_allocated, updater_resources_explored,
    updater_group_size, updater_group_sum,
    updater_births, updater_mate_verification,
    updater_allocated_hexagons, updater_explored_hexagons,
    updater_hexagon_presence, updater_uptake,
    updater_quantify_extremes, updater_trait_value_index,
    updater_data_lookup, updater_subpopulation_assign,
    updater_subpopulation_selector,
)

# Example test:
class TestUpdaterGroupSize:
    def test_correct_group_sizes(self):
        mgr = AccumulatorManager(5, [AccumulatorDef("gsize")])
        mask = np.ones(5, dtype=bool)
        group_ids = np.array([0, 0, 0, 1, 1])
        updater_group_size(mgr, "gsize", mask, group_ids=group_ids)
        assert mgr.get("gsize")[0] == 3.0
        assert mgr.get("gsize")[3] == 2.0

    def test_floaters_get_zero(self):
        mgr = AccumulatorManager(3, [AccumulatorDef("gsize")])
        mask = np.ones(3, dtype=bool)
        group_ids = np.array([-1, 0, 0])
        updater_group_size(mgr, "gsize", mask, group_ids=group_ids)
        # floaters (gid=-1) still get a count of 1 (themselves)
        ...
```

**Test:** `conda run -n shiny python -m pytest tests/test_accumulators_remaining.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `feat(accumulators): implement remaining 17 HexSim updater functions`

---

## Task 7: Census Event

**Priority:** Medium (reporting)
**Files:**
- Create: `salmon_ibm/reporting.py`
- Create: `tests/test_reporting.py`

A registered event that counts the alive population broken down by a trait and writes a CSV row each time it fires.

- [ ] **Step 1: Implement CensusEvent**

```python
# salmon_ibm/reporting.py
"""Reporting events: census, log writer, summary reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv

import numpy as np

from salmon_ibm.events import Event, register_event


@register_event("census")
@dataclass
class CensusEvent(Event):
    """Count population by trait categories, write one CSV row per firing.

    Output columns: timestep, total_alive, <category_1>, <category_2>, ...
    """
    output_path: str = "census.csv"
    trait_name: str | None = None
    _writer: csv.writer | None = field(init=False, default=None, repr=False)
    _file: object = field(init=False, default=None, repr=False)
    _header_written: bool = field(init=False, default=False, repr=False)

    def execute(self, population, landscape, t, mask):
        alive = population.alive
        n_alive = int(alive.sum())
        row = {"timestep": t, "total_alive": n_alive}

        if self.trait_name and population.trait_mgr is not None:
            defn = population.trait_mgr.definitions[self.trait_name]
            trait_vals = population.trait_mgr.get(self.trait_name)
            for i, cat_name in enumerate(defn.categories):
                count = int(((trait_vals == i) & alive).sum())
                row[cat_name] = count

        if self._file is None:
            self._file = open(self.output_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
            self._header_written = True

        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
```

- [ ] **Step 2: Write tests**

```python
# tests/test_reporting.py
class TestCensusEvent:
    def test_census_writes_csv(self, tmp_path):
        """CensusEvent produces a valid CSV with correct counts."""
        ...

    def test_census_with_trait_breakdown(self, tmp_path):
        """CSV columns include trait categories when trait_name is set."""
        ...

    def test_census_without_traits(self, tmp_path):
        """Works with trait_name=None, only timestep and total_alive columns."""
        ...
```

**Test:** `conda run -n shiny python -m pytest tests/test_reporting.py::TestCensusEvent -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `feat(reporting): add CensusEvent for per-timestep population counts`

---

## Task 8: Log Writer — Binary Per-Timestep Snapshots

**Priority:** Medium (reporting)
**Files:**
- Modify: `salmon_ibm/reporting.py`
- Add to: `tests/test_reporting.py`

A compact binary log that writes full agent state snapshots each timestep, suitable for post-hoc analysis and replay.

- [ ] **Step 1: Implement LogWriter**

```python
# salmon_ibm/reporting.py — append

@register_event("log_snapshot")
@dataclass
class LogSnapshotEvent(Event):
    """Write binary per-timestep agent state snapshots using NumPy .npz format.

    Each snapshot file: {output_dir}/step_{t:06d}.npz
    Contains: tri_idx, alive, mass_g, ed_kJ_g, behavior, group_id
    """
    output_dir: str = "logs"
    include_accumulators: bool = False
    include_traits: bool = False

    def execute(self, population, landscape, t, mask):
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        data = {
            "tri_idx": population.tri_idx,
            "alive": population.alive,
            "mass_g": population.mass_g,
            "ed_kJ_g": population.ed_kJ_g,
            "behavior": population.behavior,
            "group_id": population.group_id,
            "agent_ids": population.agent_ids,
        }
        if self.include_accumulators and population.accumulator_mgr is not None:
            data["accumulators"] = population.accumulator_mgr.data
        if self.include_traits and population.trait_mgr is not None:
            for name in population.trait_mgr._data:
                data[f"trait_{name}"] = population.trait_mgr._data[name]
        np.savez_compressed(out / f"step_{t:06d}.npz", **data)
```

- [ ] **Step 2: Write tests**

```python
class TestLogSnapshotEvent:
    def test_creates_npz_files(self, tmp_path):
        """LogSnapshotEvent writes .npz files to output_dir."""
        ...

    def test_npz_contains_expected_arrays(self, tmp_path):
        """Snapshot files contain tri_idx, alive, mass_g, etc."""
        ...

    def test_includes_accumulators_when_requested(self, tmp_path):
        ...
```

**Test:** `conda run -n shiny python -m pytest tests/test_reporting.py::TestLogSnapshotEvent -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `feat(reporting): add LogSnapshotEvent for binary per-timestep state dumps`

---

## Task 9: Summary Reports — Births, Deaths, Lambda

**Priority:** Medium (reporting)
**Files:**
- Modify: `salmon_ibm/reporting.py`
- Add to: `tests/test_reporting.py`

An event that tracks births, deaths, and population growth rate (lambda) over time and writes a summary CSV at the end.

- [ ] **Step 1: Implement SummaryReportEvent**

```python
@register_event("summary_report")
@dataclass
class SummaryReportEvent(Event):
    """Track per-timestep population statistics; write summary CSV on close().

    Columns: timestep, n_alive, n_births, n_deaths, lambda
    """
    output_path: str = "summary.csv"
    _records: list[dict] = field(init=False, default_factory=list, repr=False)
    _prev_alive: int = field(init=False, default=0, repr=False)

    def execute(self, population, landscape, t, mask):
        n_alive = int(population.alive.sum())
        # Births/deaths estimated by delta: n_alive - _prev_alive = births - deaths
        # For more accurate tracking, check landscape for birth/death counters
        births = int(landscape.get("births_this_step", 0))
        deaths = int(landscape.get("deaths_this_step", 0))
        lam = n_alive / self._prev_alive if self._prev_alive > 0 else float("nan")
        self._records.append({
            "timestep": t,
            "n_alive": n_alive,
            "n_births": births,
            "n_deaths": deaths,
            "lambda": round(lam, 6),
        })
        self._prev_alive = n_alive

    def close(self):
        if not self._records:
            return
        import csv
        with open(self.output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._records[0].keys()))
            writer.writeheader()
            writer.writerows(self._records)
```

- [ ] **Step 2: Write tests**

```python
class TestSummaryReportEvent:
    def test_records_population_size(self):
        ...

    def test_lambda_calculation(self):
        """lambda = n_alive(t) / n_alive(t-1)."""
        ...

    def test_close_writes_csv(self, tmp_path):
        ...
```

**Test:** `conda run -n shiny python -m pytest tests/test_reporting.py::TestSummaryReportEvent -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `feat(reporting): add SummaryReportEvent with births, deaths, and lambda`

---

## Task 10: XML Scenario Parser

**Priority:** Low (compatibility)
**Files:**
- Create: `salmon_ibm/xml_loader.py`
- Create: `tests/test_xml_loader.py`

Parse HexSim `.xml` scenario files into Python event sequences compatible with `EventSequencer`. HexSim scenarios encode their event sequences in XML under `<EventSequence>` elements with `<Event>` children specifying type, parameters, and trigger timing.

- [ ] **Step 1: Implement core XML parser**

```python
# salmon_ibm/xml_loader.py
"""Parse HexSim .xml scenario files into event sequences."""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Any

from salmon_ibm.events import Event, EVENT_REGISTRY, EveryStep, Periodic, Once, Window


def load_scenario_xml(path: str | Path) -> dict:
    """Parse a HexSim scenario .xml file.

    Returns a dict with:
        - "events": list[Event] ready for EventSequencer
        - "populations": list of population config dicts
        - "parameters": global scenario parameters
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    result = {
        "events": [],
        "populations": [],
        "parameters": _parse_parameters(root),
    }

    for pop_elem in root.findall(".//Population"):
        result["populations"].append(_parse_population(pop_elem))

    for seq_elem in root.findall(".//EventSequence"):
        for event_elem in seq_elem.findall("Event"):
            event = _parse_event(event_elem)
            if event is not None:
                result["events"].append(event)

    return result


# HexSim event type name -> our registered event type name
_HEXSIM_EVENT_MAP = {
    "Movement": "movement",
    "Survival": "survival",
    "Reproduction": "reproduction",
    "Accumulate": "accumulate",
    "Transition": "transition",
    "Introduction": "introduction",
    "Census": "census",
    "RangeDynamics": "range_dynamics",
    "GeneratedHexmap": "generated_hexmap",
    "Mutation": "mutation",
    "FloaterCreation": "floater_creation",
    "Interaction": "interaction",
    "SwitchPopulation": "switch_population",
    "SetAffinity": "set_affinity",
    "StageSpecificSurvival": "stage_survival",
}


def _parse_event(elem: ET.Element) -> Event | None:
    hs_type = elem.get("type", elem.get("Type", ""))
    our_type = _HEXSIM_EVENT_MAP.get(hs_type)
    if our_type is None or our_type not in EVENT_REGISTRY:
        return None  # skip unknown events

    cls = EVENT_REGISTRY[our_type]
    name = elem.get("name", elem.get("Name", hs_type))
    trigger = _parse_trigger_xml(elem)
    params = _parse_params_xml(elem)
    trait_filter = _parse_trait_filter_xml(elem)

    try:
        return cls(name=name, trigger=trigger, trait_filter=trait_filter, **params)
    except TypeError:
        # If params don't match constructor, create with just name/trigger
        return cls(name=name, trigger=trigger)


def _parse_trigger_xml(elem: ET.Element):
    """Parse trigger from XML element attributes/children."""
    trigger_elem = elem.find("Trigger")
    if trigger_elem is None:
        return EveryStep()
    kind = trigger_elem.get("type", "every_step")
    if kind == "periodic":
        return Periodic(interval=int(trigger_elem.get("interval", "1")))
    elif kind == "once":
        return Once(at=int(trigger_elem.get("at", "0")))
    elif kind == "window":
        return Window(
            start=int(trigger_elem.get("start", "0")),
            end=int(trigger_elem.get("end", "100")),
        )
    return EveryStep()


def _parse_params_xml(elem: ET.Element) -> dict:
    """Extract event parameters from XML element."""
    params = {}
    for param_elem in elem.findall("Parameter"):
        key = param_elem.get("name", param_elem.get("Name", ""))
        val = param_elem.text or param_elem.get("value", "")
        params[key] = _auto_cast(val)
    return params


def _parse_trait_filter_xml(elem: ET.Element) -> dict | None:
    """Parse trait filter from XML."""
    filter_elem = elem.find("TraitFilter")
    if filter_elem is None:
        return None
    return {
        filter_elem.get("trait", ""): filter_elem.get("value", "")
    }


def _parse_population(elem: ET.Element) -> dict:
    return {
        "name": elem.get("name", elem.get("Name", "default")),
        "size": int(elem.get("size", elem.get("Size", "100"))),
    }


def _parse_parameters(root: ET.Element) -> dict:
    params = {}
    for p in root.findall(".//GlobalParameter"):
        params[p.get("name", "")] = _auto_cast(p.text or "")
    return params


def _auto_cast(val: str) -> Any:
    """Try to cast a string to int, float, or leave as string."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    return val
```

- [ ] **Step 2: Write tests with minimal XML fixtures**

```python
# tests/test_xml_loader.py
"""Tests for HexSim XML scenario parser."""
import pytest
from pathlib import Path
from salmon_ibm.xml_loader import load_scenario_xml


MINIMAL_SCENARIO_XML = """\
<?xml version="1.0"?>
<Scenario>
  <GlobalParameter name="timesteps">100</GlobalParameter>
  <Population name="salmon" size="50"/>
  <EventSequence>
    <Event type="Movement" name="move_step">
      <Trigger type="every_step"/>
    </Event>
    <Event type="Survival" name="survive">
      <Trigger type="every_step"/>
    </Event>
    <Event type="Reproduction" name="breed">
      <Trigger type="periodic" interval="10"/>
      <Parameter name="clutch_mean" value="3.0"/>
    </Event>
  </EventSequence>
</Scenario>
"""


class TestXMLLoader:
    def test_load_minimal_scenario(self, tmp_path):
        xml_file = tmp_path / "test_scenario.xml"
        xml_file.write_text(MINIMAL_SCENARIO_XML)
        result = load_scenario_xml(xml_file)
        assert len(result["events"]) == 3
        assert result["events"][0].name == "move_step"
        assert result["parameters"]["timesteps"] == 100
        assert result["populations"][0]["name"] == "salmon"

    def test_periodic_trigger_parsed(self, tmp_path):
        xml_file = tmp_path / "test_scenario.xml"
        xml_file.write_text(MINIMAL_SCENARIO_XML)
        result = load_scenario_xml(xml_file)
        breed_event = result["events"][2]
        assert breed_event.trigger.interval == 10

    def test_unknown_event_type_skipped(self, tmp_path):
        xml = '<Scenario><EventSequence><Event type="UnknownThing"/></EventSequence></Scenario>'
        xml_file = tmp_path / "unknown.xml"
        xml_file.write_text(xml)
        result = load_scenario_xml(xml_file)
        assert len(result["events"]) == 0

    def test_empty_scenario(self, tmp_path):
        xml_file = tmp_path / "empty.xml"
        xml_file.write_text("<Scenario/>")
        result = load_scenario_xml(xml_file)
        assert result["events"] == []
        assert result["populations"] == []
```

**Test:** `conda run -n shiny python -m pytest tests/test_xml_loader.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `feat(xml_loader): parse HexSim .xml scenario files into event sequences`

---

## Task 11: HexSim "Basic" Example Scenario — Load and Run

**Priority:** Low (compatibility testing)
**Files:**
- Create: `tests/test_hexsim_basic.py`
- Possibly modify: `salmon_ibm/xml_loader.py`

Load the official HexSim "Basic" example scenario using the XML parser and run it for a few timesteps to verify no crashes.

- [ ] **Step 1: Locate and examine the Basic scenario XML**

The HexSim examples ship with the workspace. Look in `HexSim Examples/` or `HexSim 4.0.20/` for a "Basic" scenario. Identify the XML structure and any event types it uses that we may not have mapped yet.

- [ ] **Step 2: Write a smoke test**

```python
# tests/test_hexsim_basic.py
"""Compatibility test: load and run HexSim 'Basic' example scenario."""
import pytest
from pathlib import Path
from salmon_ibm.xml_loader import load_scenario_xml


# Skip if the HexSim examples directory is not present
HEXSIM_EXAMPLES = Path(__file__).parent.parent / "HexSim Examples"
BASIC_SCENARIO = HEXSIM_EXAMPLES / "Basic" / "Scenarios" / "Basic.xml"

pytestmark = pytest.mark.skipif(
    not BASIC_SCENARIO.exists(),
    reason="HexSim Basic example not found"
)


class TestHexSimBasicScenario:
    def test_xml_loads_without_error(self):
        result = load_scenario_xml(BASIC_SCENARIO)
        assert len(result["events"]) > 0

    def test_all_events_have_known_types(self):
        result = load_scenario_xml(BASIC_SCENARIO)
        for event in result["events"]:
            assert event is not None, "All events should parse to known types"

    def test_run_10_steps(self):
        """Smoke test: build and run 10 timesteps without crash."""
        # Build a minimal simulation from the parsed scenario
        # This may require a helper function to convert XML result -> Simulation
        ...
```

- [ ] **Step 3: Map any missing HexSim event types discovered in the Basic scenario**

Update `_HEXSIM_EVENT_MAP` in `xml_loader.py` as needed. For event types we cannot support yet, log a warning and skip.

**Test:** `conda run -n shiny python -m pytest tests/test_hexsim_basic.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `test(compat): add HexSim Basic example scenario smoke test`

---

## Task 12: HexSim Output Comparison

**Priority:** Low (compatibility testing)
**Files:**
- Add to: `tests/test_hexsim_basic.py`

Compare our simulation output against reference results from the HexSim C++ implementation. This requires running the Basic scenario in both engines and checking key metrics match within tolerance.

- [ ] **Step 1: Generate reference outputs from HexSim C++**

Run the Basic scenario in HexSim 4.0.20 and save:
- Census CSV (population counts per timestep)
- Final population size at timestep 100

Store reference data in `tests/fixtures/hexsim_basic_reference/`.

- [ ] **Step 2: Write comparison tests**

```python
# tests/test_hexsim_basic.py — add to existing file
class TestHexSimBasicComparison:
    """Compare Python output against HexSim C++ reference."""

    REFERENCE_DIR = Path(__file__).parent / "fixtures" / "hexsim_basic_reference"

    @pytest.mark.skipif(
        not (REFERENCE_DIR / "census.csv").exists(),
        reason="Reference data not available"
    )
    def test_population_trajectory_within_tolerance(self):
        """Population counts at each timestep should match within 15%.

        HexSim uses stochastic events, so exact match is not expected.
        We compare mean population size over the run.
        """
        import csv
        ref_path = self.REFERENCE_DIR / "census.csv"
        # Load reference census
        with open(ref_path) as f:
            reader = csv.DictReader(f)
            ref_sizes = [int(row["total_alive"]) for row in reader]

        # Run our simulation
        # ... (load scenario, run 100 steps, collect census)
        our_sizes = [...]

        ref_mean = sum(ref_sizes) / len(ref_sizes)
        our_mean = sum(our_sizes) / len(our_sizes)
        assert abs(our_mean - ref_mean) / ref_mean < 0.15, (
            f"Mean pop size {our_mean:.0f} vs reference {ref_mean:.0f}"
        )

    def test_final_population_order_of_magnitude(self):
        """Final population should be within an order of magnitude of reference."""
        ...
```

- [ ] **Step 3: Document known deviations**

Any systematic differences between our output and HexSim C++ should be documented in the test file docstrings, noting the likely cause (e.g., different RNG sequence, movement kernel simplifications, etc.).

**Test:** `conda run -n shiny python -m pytest tests/test_hexsim_basic.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

**Commit message:** `test(compat): add HexSim C++ output comparison tests for Basic scenario`

---

## Dependency Graph

```
Task 1 (barriers -> movement)  ─┐
Task 2 (genome -> population)   ├──> Task 5 (expand landscape dict)
Task 3 (genetics -> reproduction)┘         │
Task 4 (config sections) ────────────────> Task 5
                                            │
Task 6 (remaining updaters)    [independent]│
Task 7 (census event)         ─┐            │
Task 8 (log writer)            ├─ [independent, after Task 5 for full integration]
Task 9 (summary reports)      ─┘
Task 10 (XML parser)          ─────────> Task 11 (Basic scenario load)
                                                   │
                                          Task 12 (output comparison)
```

**Recommended execution order:** Tasks 1-5 (sequential, wiring), then Tasks 6-9 (parallel, independent), then Tasks 10-12 (sequential, compatibility).

---

## Estimated Scope

| Task | Estimated Lines | New Tests |
|------|----------------|-----------|
| 1. Wire barriers into movement | ~40 | ~6 |
| 2. Add genome to Population | ~25 | ~6 |
| 3. Wire reproduction to genetics | ~30 | ~4 |
| 4. Config sections | ~80 | ~8 |
| 5. Expand landscape dict | ~25 | ~6 |
| 6. Remaining 17 updaters | ~200 | ~34 |
| 7. Census event | ~50 | ~4 |
| 8. Log writer | ~35 | ~4 |
| 9. Summary reports | ~40 | ~4 |
| 10. XML parser | ~120 | ~6 |
| 11. Basic scenario load | ~30 | ~4 |
| 12. Output comparison | ~40 | ~4 |
| **Total** | **~715** | **~86** |
