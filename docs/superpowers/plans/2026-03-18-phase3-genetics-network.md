# Phase 3: Genetics, Interactions, and Network Topology — Implementation Plan

> **STATUS: ✅ EXECUTED** — `genetics.py`, `interactions.py`, `network.py`, `events_phase3.py` all shipped. Tests in `tests/test_genetics.py`, `tests/test_interactions.py`, `tests/test_network.py`. Population now carries genome.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add diploid genetics with recombination/mutation, multi-species interaction events, 1D stream network topology, and remaining HexSim event types — completing the core HexSim feature set.

**Architecture:** Phase 3 builds on the event engine (`events.py`), accumulator system (`accumulators.py`), and trait system (`traits.py`) from Phases 1a-1b, plus the Population class, barriers, reproduction, and survival from Phase 2. New modules are added as peers: `genetics.py` for the diploid genome manager, `interactions.py` for multi-population encounters, `network.py` for the 1D stream topology, and individual event files registered via `@register_event`. Each sub-system is independently testable and integrates through the existing `Event.execute()` signature and `TraitType` enum extension.

**Tech Stack:** NumPy, Numba (for recombination), dataclasses, existing event engine

**Test command:** `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `salmon_ibm/genetics.py` | GenomeManager, recombination, mutation | **Create** |
| `salmon_ibm/traits.py` | Add GENETIC / GENETIC_ACCUMULATED trait types | **Modify** |
| `salmon_ibm/interactions.py` | MultiPopulationManager, InteractionEvent | **Create** |
| `salmon_ibm/network.py` | StreamNetwork, network movement, network range dynamics | **Create** |
| `salmon_ibm/events_phase3.py` | TransitionEvent, GeneratedHexmapEvent, RangeDynamicsEvent, SetAffinityEvent, SwitchPopulationEvent | **Create** |
| `salmon_ibm/events.py` | Register new event types (if needed) | **Modify** |
| `tests/test_genetics.py` | GenomeManager, recombination, mutation tests | **Create** |
| `tests/test_interactions.py` | Multi-population and interaction event tests | **Create** |
| `tests/test_network.py` | Stream network and network movement tests | **Create** |
| `tests/test_events_phase3.py` | Transition, generated hexmap, range dynamics, affinity tests | **Create** |

---

## Task 1: GenomeManager Class — Storage and Loci Definitions

**Files:**
- Create: `salmon_ibm/genetics.py`
- Create: `tests/test_genetics.py`

Implement the core diploid genotype storage. Each individual has a genotype array of shape `(n_loci, 2)` representing two alleles per locus. The manager stores all genotypes in a single 3D array `int[n_agents, n_loci, 2]` for vectorized access.

- [ ] **Step 1: Define LocusDefinition and GenomeManager**

```python
# salmon_ibm/genetics.py
"""Diploid genetic system: genotype storage, recombination, mutation."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class LocusDefinition:
    """Definition for a single genetic locus."""
    name: str
    n_alleles: int  # number of distinct alleles at this locus
    position: float = 0.0  # position on chromosome (cM) for linkage


class GenomeManager:
    """Diploid genotype storage and operations for all agents.

    Storage: 3D NumPy array of shape (n_agents, n_loci, 2).
    Alleles are stored as integer indices [0, n_alleles).
    """

    def __init__(self, n_agents: int, loci: list[LocusDefinition],
                 rng_seed: int | None = None):
        self.n_agents = n_agents
        self.loci = list(loci)
        self.n_loci = len(loci)
        self._name_to_idx: dict[str, int] = {
            loc.name: i for i, loc in enumerate(loci)
        }
        self.rng = np.random.default_rng(rng_seed)

        # Diploid genotypes: int[n_agents, n_loci, 2]
        self.genotypes = np.zeros((n_agents, self.n_loci, 2), dtype=np.int32)

        # Precompute linkage distances between consecutive loci (in cM)
        positions = np.array([loc.position for loc in loci], dtype=np.float64)
        self.linkage_distances = np.diff(positions) if len(positions) > 1 else np.array([], dtype=np.float64)

    def locus_index(self, name: str) -> int:
        return self._name_to_idx[name]

    def get_locus(self, name: str) -> np.ndarray:
        """Return alleles at a named locus: shape (n_agents, 2)."""
        idx = self._name_to_idx[name]
        return self.genotypes[:, idx, :]

    def initialize_random(self, mask: np.ndarray | None = None) -> None:
        """Assign random alleles to all (or masked) agents."""
        for i, loc in enumerate(self.loci):
            if mask is not None:
                n = mask.sum()
                self.genotypes[mask, i, :] = self.rng.integers(
                    0, loc.n_alleles, size=(n, 2)
                )
            else:
                self.genotypes[:, i, :] = self.rng.integers(
                    0, loc.n_alleles, size=(self.n_agents, 2)
                )

    def homozygosity(self, locus_name: str | None = None,
                     mask: np.ndarray | None = None) -> np.ndarray:
        """Fraction of homozygous loci per individual (or at a single locus).

        Returns float[n_agents] (or float[mask.sum()] if masked).
        """
        if locus_name is not None:
            idx = self._name_to_idx[locus_name]
            alleles = self.genotypes[:, idx, :]
            if mask is not None:
                alleles = alleles[mask]
            return (alleles[:, 0] == alleles[:, 1]).astype(np.float64)
        else:
            geno = self.genotypes if mask is None else self.genotypes[mask]
            return (geno[:, :, 0] == geno[:, :, 1]).mean(axis=1)
```

- [ ] **Step 2: Write unit tests for GenomeManager**

```python
# tests/test_genetics.py
"""Unit tests for the genetics sub-model."""
import numpy as np
import pytest

from salmon_ibm.genetics import LocusDefinition, GenomeManager


class TestGenomeManager:
    def setup_method(self):
        self.loci = [
            LocusDefinition("color", n_alleles=4, position=0.0),
            LocusDefinition("size", n_alleles=3, position=50.0),
            LocusDefinition("speed", n_alleles=5, position=120.0),
        ]
        self.gm = GenomeManager(n_agents=100, loci=self.loci, rng_seed=42)

    def test_shape(self):
        assert self.gm.genotypes.shape == (100, 3, 2)

    def test_initialize_random_fills_valid_alleles(self):
        self.gm.initialize_random()
        for i, loc in enumerate(self.loci):
            assert self.gm.genotypes[:, i, :].max() < loc.n_alleles
            assert self.gm.genotypes[:, i, :].min() >= 0

    def test_get_locus_by_name(self):
        self.gm.initialize_random()
        color = self.gm.get_locus("color")
        assert color.shape == (100, 2)

    def test_homozygosity_single_locus(self):
        self.gm.genotypes[:, 0, 0] = 1
        self.gm.genotypes[:, 0, 1] = 1  # all homozygous
        h = self.gm.homozygosity("color")
        assert np.all(h == 1.0)

    def test_homozygosity_all_loci(self):
        self.gm.genotypes[:, :, 0] = 0
        self.gm.genotypes[:, :, 1] = 1  # all heterozygous
        h = self.gm.homozygosity()
        assert np.all(h == 0.0)

    def test_linkage_distances(self):
        np.testing.assert_array_almost_equal(
            self.gm.linkage_distances, [50.0, 70.0]
        )

    def test_masked_initialize(self):
        mask = np.zeros(100, dtype=bool)
        mask[:10] = True
        self.gm.initialize_random(mask=mask)
        # Unmasked agents should still be zero
        assert np.all(self.gm.genotypes[10:] == 0)
        # Masked agents should have non-trivial values (probabilistically)
        assert self.gm.genotypes[:10].sum() > 0
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_genetics.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(genetics): add GenomeManager with diploid storage and homozygosity`

---

## Task 2: Recombination — Crossover with Linkage Distances

**Files:**
- Modify: `salmon_ibm/genetics.py`
- Modify: `tests/test_genetics.py`

Implement meiotic recombination. When two parents produce offspring, each parent contributes one haploid gamete produced by crossover. Crossover probability between adjacent loci is derived from linkage distance using the Haldane mapping function: `P(crossover) = 0.5 * (1 - exp(-2d/100))` where `d` is distance in centiMorgans.

- [ ] **Step 1: Implement _haldane_crossover_probs helper**

```python
# In salmon_ibm/genetics.py

def _haldane_crossover_probs(linkage_distances_cM: np.ndarray) -> np.ndarray:
    """Convert linkage distances (cM) to crossover probabilities via Haldane."""
    return 0.5 * (1.0 - np.exp(-2.0 * linkage_distances_cM / 100.0))
```

- [ ] **Step 2: Implement _produce_gamete with Numba**

```python
# In salmon_ibm/genetics.py
import numba

@numba.njit
def _produce_gamete_numba(parent_genotype, crossover_probs, rng_draws):
    """Produce a haploid gamete from a diploid parent via crossover.

    Args:
        parent_genotype: int[n_loci, 2] — diploid genotype
        crossover_probs: float[n_loci - 1] — P(crossover) between adjacent loci
        rng_draws: float[n_loci - 1] — pre-drawn uniform random values

    Returns:
        int[n_loci] — haploid gamete
    """
    n_loci = parent_genotype.shape[0]
    gamete = np.empty(n_loci, dtype=np.int32)
    strand = 0 if rng_draws[0] < 0.5 else 1  # random initial strand (unbiased)
    gamete[0] = parent_genotype[0, strand]
    for i in range(1, n_loci):
        if rng_draws[i - 1] < crossover_probs[i - 1]:
            strand = 1 - strand  # crossover
        gamete[i] = parent_genotype[i, strand]
    return gamete
```

- [ ] **Step 3: Add vectorized recombine method to GenomeManager**

```python
# In GenomeManager class

def recombine(self, parent1_indices: np.ndarray, parent2_indices: np.ndarray,
              offspring_indices: np.ndarray) -> None:
    """Create offspring genotypes by recombining two parents.

    Each parent contributes one gamete (with crossover) to each offspring.
    parent1_indices, parent2_indices, offspring_indices are aligned arrays.
    """
    n_offspring = len(offspring_indices)
    crossover_probs = _haldane_crossover_probs(self.linkage_distances)

    for k in range(n_offspring):
        p1 = self.genotypes[parent1_indices[k]]
        p2 = self.genotypes[parent2_indices[k]]

        draws1 = self.rng.random(len(crossover_probs))
        draws2 = self.rng.random(len(crossover_probs))

        gamete1 = _produce_gamete_numba(p1, crossover_probs, draws1)
        gamete2 = _produce_gamete_numba(p2, crossover_probs, draws2)

        self.genotypes[offspring_indices[k], :, 0] = gamete1
        self.genotypes[offspring_indices[k], :, 1] = gamete2
```

- [ ] **Step 4: Write tests for recombination**

```python
# In tests/test_genetics.py

class TestRecombination:
    def test_offspring_alleles_come_from_parents(self):
        """Each offspring allele must exist in the corresponding parent locus."""
        loci = [
            LocusDefinition("A", n_alleles=10, position=0.0),
            LocusDefinition("B", n_alleles=10, position=50.0),
            LocusDefinition("C", n_alleles=10, position=100.0),
        ]
        gm = GenomeManager(n_agents=10, loci=loci, rng_seed=42)
        gm.initialize_random()

        parent1 = np.array([0, 1])
        parent2 = np.array([2, 3])
        offspring = np.array([4, 5])
        gm.recombine(parent1, parent2, offspring)

        for k in range(2):
            for locus in range(3):
                allele0 = gm.genotypes[offspring[k], locus, 0]
                allele1 = gm.genotypes[offspring[k], locus, 1]
                p1_alleles = set(gm.genotypes[parent1[k], locus, :])
                p2_alleles = set(gm.genotypes[parent2[k], locus, :])
                assert allele0 in p1_alleles  # gamete1 from parent1
                assert allele1 in p2_alleles  # gamete2 from parent2

    def test_tight_linkage_preserves_haplotype(self):
        """Loci at distance 0 cM should never recombine."""
        loci = [
            LocusDefinition("A", n_alleles=4, position=0.0),
            LocusDefinition("B", n_alleles=4, position=0.0),  # same position
        ]
        gm = GenomeManager(n_agents=4, loci=loci, rng_seed=99)
        # Manually set parent genotypes
        gm.genotypes[0] = [[0, 1], [2, 3]]  # parent1: strand0=[0,2], strand1=[1,3]
        gm.genotypes[1] = [[0, 1], [2, 3]]  # parent2
        parent1 = np.array([0])
        parent2 = np.array([1])
        offspring = np.array([2])
        # Run many times — should always get [0,2] or [1,3] from each parent
        for _ in range(50):
            gm.recombine(parent1, parent2, offspring)
            g = gm.genotypes[2]
            assert (g[0, 0], g[1, 0]) in [(0, 2), (1, 3)]
            assert (g[0, 1], g[1, 1]) in [(0, 2), (1, 3)]

    def test_haldane_crossover_probs(self):
        from salmon_ibm.genetics import _haldane_crossover_probs
        # At 50 cM, probability should be ~0.316
        probs = _haldane_crossover_probs(np.array([50.0]))
        assert 0.30 < probs[0] < 0.33
        # At 0 cM, probability should be 0
        probs = _haldane_crossover_probs(np.array([0.0]))
        assert probs[0] == 0.0
```

- [ ] **Step 5: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_genetics.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(genetics): add recombination with Haldane crossover and Numba gamete production`

---

## Task 3: MutationEvent — Allele Transition Matrices

**Files:**
- Modify: `salmon_ibm/genetics.py`
- Create: `salmon_ibm/events_phase3.py` (start with MutationEvent)
- Modify: `tests/test_genetics.py`

Each locus has an allele transition matrix `T[n_alleles, n_alleles]` where `T[i, j]` is the probability that allele `i` mutates to allele `j` in one timestep. Rows must sum to 1.0. The diagonal holds the probability of no mutation.

- [ ] **Step 1: Add mutate method to GenomeManager**

```python
# In GenomeManager class

def mutate(self, locus_name: str, transition_matrix: np.ndarray,
           mask: np.ndarray | None = None) -> int:
    """Apply allele mutations at a single locus using a transition matrix.

    Args:
        locus_name: which locus to mutate
        transition_matrix: float[n_alleles, n_alleles], rows sum to 1.0
        mask: optional bool[n_agents] — only mutate these agents

    Returns:
        Number of mutations that occurred.
    """
    idx = self._name_to_idx[locus_name]
    loc = self.loci[idx]
    assert transition_matrix.shape == (loc.n_alleles, loc.n_alleles)

    agents = np.arange(self.n_agents) if mask is None else np.where(mask)[0]
    n_mutations = 0

    for ploidy in range(2):  # both allele copies
        current_alleles = self.genotypes[agents, idx, ploidy]
        new_alleles = np.empty_like(current_alleles)
        for allele_val in range(loc.n_alleles):
            allele_mask = current_alleles == allele_val
            if not allele_mask.any():
                continue
            n = allele_mask.sum()
            probs = transition_matrix[allele_val]
            drawn = self.rng.choice(loc.n_alleles, size=n, p=probs)
            new_alleles[allele_mask] = drawn
        changed = new_alleles != current_alleles
        n_mutations += changed.sum()
        self.genotypes[agents, idx, ploidy] = new_alleles

    return n_mutations
```

- [ ] **Step 2: Create MutationEvent**

```python
# salmon_ibm/events_phase3.py
"""Phase 3 event types: genetics, interactions, network, and remaining events."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from salmon_ibm.events import Event, register_event


@register_event("mutation")
@dataclass
class MutationEvent(Event):
    """Apply allele mutations at specified loci each timestep."""
    locus_name: str = ""
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
        if population.genome is None:
            return
        population.genome.mutate(
            self.locus_name, self.transition_matrix, mask=mask
        )
```

- [ ] **Step 3: Write tests for mutation**

```python
# In tests/test_genetics.py

class TestMutation:
    def test_identity_matrix_no_change(self):
        """Identity transition matrix should produce zero mutations."""
        loci = [LocusDefinition("A", n_alleles=3, position=0.0)]
        gm = GenomeManager(n_agents=50, loci=loci, rng_seed=42)
        gm.initialize_random()
        original = gm.genotypes.copy()
        T = np.eye(3)
        n_mut = gm.mutate("A", T)
        assert n_mut == 0
        np.testing.assert_array_equal(gm.genotypes, original)

    def test_forced_mutation(self):
        """Transition matrix that always mutates allele 0 -> 1."""
        loci = [LocusDefinition("A", n_alleles=3, position=0.0)]
        gm = GenomeManager(n_agents=20, loci=loci, rng_seed=42)
        gm.genotypes[:, 0, :] = 0  # all agents have allele 0
        T = np.array([
            [0.0, 1.0, 0.0],  # allele 0 always becomes 1
            [0.0, 1.0, 0.0],  # allele 1 stays
            [0.0, 0.0, 1.0],  # allele 2 stays
        ])
        gm.mutate("A", T)
        assert np.all(gm.genotypes[:, 0, :] == 1)

    def test_mutation_respects_mask(self):
        loci = [LocusDefinition("A", n_alleles=2, position=0.0)]
        gm = GenomeManager(n_agents=20, loci=loci, rng_seed=42)
        gm.genotypes[:, 0, :] = 0
        T = np.array([[0.0, 1.0], [1.0, 0.0]])  # always flip
        mask = np.zeros(20, dtype=bool)
        mask[:5] = True
        gm.mutate("A", T, mask=mask)
        assert np.all(gm.genotypes[:5, 0, :] == 1)
        assert np.all(gm.genotypes[5:, 0, :] == 0)
```

- [ ] **Step 4: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_genetics.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(genetics): add MutationEvent with allele transition matrices`

---

## Task 4: GENETIC Trait Type — Single-Locus Phenotype Derivation

**Files:**
- Modify: `salmon_ibm/traits.py`
- Modify: `tests/test_genetics.py`

Add `TraitType.GENETIC` to the existing `TraitType` enum. A GENETIC trait derives its categorical value from a single locus: `phenotype = mapping_table[allele0, allele1]`. The mapping table encodes dominance relationships (e.g., allele 0 dominant over allele 1).

- [ ] **Step 1: Extend TraitType enum**

```python
# In salmon_ibm/traits.py — modify TraitType
class TraitType(Enum):
    PROBABILISTIC = "probabilistic"
    ACCUMULATED = "accumulated"
    GENETIC = "genetic"
    GENETIC_ACCUMULATED = "genetic_accumulated"
```

- [ ] **Step 2: Extend TraitDefinition with genetic fields**

```python
# In salmon_ibm/traits.py — modify TraitDefinition
@dataclass
class TraitDefinition:
    name: str
    trait_type: TraitType
    categories: list[str]
    accumulator_name: str | None = None
    thresholds: np.ndarray | None = None
    # Genetic fields (Phase 3)
    locus_name: str | None = None  # for GENETIC type
    phenotype_map: np.ndarray | None = None  # int[n_alleles, n_alleles] -> category index
    locus_names: list[str] | None = None  # for GENETIC_ACCUMULATED (multi-locus)
    locus_weights: np.ndarray | None = None  # float[n_loci] weights for multi-locus
```

- [ ] **Step 3: Add evaluate_genetic method to TraitManager**

```python
# In TraitManager class

def evaluate_genetic(self, name: str, genome_manager,
                     mask: np.ndarray | None = None) -> None:
    """Evaluate a GENETIC trait from a single locus using the phenotype map.

    phenotype_map[allele0, allele1] -> category index
    """
    defn = self.definitions[name]
    if defn.trait_type != TraitType.GENETIC:
        raise ValueError(f"Trait {name!r} is {defn.trait_type.value}, not genetic")
    alleles = genome_manager.get_locus(defn.locus_name)  # (n_agents, 2)
    categories = defn.phenotype_map[alleles[:, 0], alleles[:, 1]].astype(np.int32)
    if mask is not None:
        self._data[name][mask] = categories[mask]
    else:
        self._data[name][:] = categories
```

- [ ] **Step 4: Write tests**

```python
# In tests/test_genetics.py

class TestGeneticTrait:
    def test_simple_dominance(self):
        """Allele 0 dominant over allele 1: heterozygotes express category 0."""
        from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
        from salmon_ibm.genetics import LocusDefinition, GenomeManager

        loci = [LocusDefinition("color", n_alleles=2, position=0.0)]
        gm = GenomeManager(n_agents=4, loci=loci, rng_seed=42)
        # Agent 0: (0,0), Agent 1: (0,1), Agent 2: (1,0), Agent 3: (1,1)
        gm.genotypes[0, 0] = [0, 0]
        gm.genotypes[1, 0] = [0, 1]
        gm.genotypes[2, 0] = [1, 0]
        gm.genotypes[3, 0] = [1, 1]

        phenotype_map = np.array([[0, 0], [0, 1]])  # 0 dominant over 1
        trait_def = TraitDefinition(
            name="color_trait", trait_type=TraitType.GENETIC,
            categories=["dark", "light"],
            locus_name="color", phenotype_map=phenotype_map,
        )
        tm = TraitManager(n_agents=4, definitions=[trait_def])
        tm.evaluate_genetic("color_trait", gm)

        vals = tm.get("color_trait")
        assert list(vals) == [0, 0, 0, 1]  # dark, dark, dark, light
```

- [ ] **Step 5: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_genetics.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(traits): add GENETIC trait type with single-locus phenotype derivation`

---

## Task 5: GENETIC_ACCUMULATED Trait Type — Multi-Locus Phenotype

**Files:**
- Modify: `salmon_ibm/traits.py`
- Modify: `tests/test_genetics.py`

A GENETIC_ACCUMULATED trait computes a continuous value from multiple loci (weighted sum of allele values), then bins it into categories using thresholds — like ACCUMULATED but the source is genetic rather than an accumulator.

- [ ] **Step 1: Add evaluate_genetic_accumulated method to TraitManager**

```python
# In TraitManager class

def evaluate_genetic_accumulated(self, name: str, genome_manager,
                                  mask: np.ndarray | None = None) -> None:
    """Evaluate a GENETIC_ACCUMULATED trait from multiple loci.

    Computes weighted sum of allele values across specified loci,
    then bins into categories using thresholds.
    """
    defn = self.definitions[name]
    if defn.trait_type != TraitType.GENETIC_ACCUMULATED:
        raise ValueError(
            f"Trait {name!r} is {defn.trait_type.value}, not genetic_accumulated"
        )
    # Sum weighted allele values across loci
    total = np.zeros(self.n_agents, dtype=np.float64)
    for i, locus_name in enumerate(defn.locus_names):
        alleles = genome_manager.get_locus(locus_name)  # (n_agents, 2)
        locus_value = alleles.sum(axis=1).astype(np.float64)  # additive model
        weight = defn.locus_weights[i] if defn.locus_weights is not None else 1.0
        total += weight * locus_value

    categories = np.digitize(total, defn.thresholds).astype(np.int32)
    if mask is not None:
        self._data[name][mask] = categories[mask]
    else:
        self._data[name][:] = categories
```

- [ ] **Step 2: Write tests**

```python
# In tests/test_genetics.py

class TestGeneticAccumulatedTrait:
    def test_two_locus_additive(self):
        """Two loci with equal weight, thresholds at [2, 4]."""
        from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
        from salmon_ibm.genetics import LocusDefinition, GenomeManager

        loci = [
            LocusDefinition("A", n_alleles=3, position=0.0),
            LocusDefinition("B", n_alleles=3, position=50.0),
        ]
        gm = GenomeManager(n_agents=3, loci=loci, rng_seed=42)
        # Agent 0: A=(0,0), B=(0,0) -> total=0 -> category 0
        gm.genotypes[0] = [[0, 0], [0, 0]]
        # Agent 1: A=(1,1), B=(0,0) -> total=2 -> category 1
        gm.genotypes[1] = [[1, 1], [0, 0]]
        # Agent 2: A=(2,2), B=(1,1) -> total=6 -> category 2
        gm.genotypes[2] = [[2, 2], [1, 1]]

        trait_def = TraitDefinition(
            name="polygenic", trait_type=TraitType.GENETIC_ACCUMULATED,
            categories=["low", "medium", "high"],
            locus_names=["A", "B"],
            locus_weights=np.array([1.0, 1.0]),
            thresholds=np.array([2.0, 4.0]),
        )
        tm = TraitManager(n_agents=3, definitions=[trait_def])
        tm.evaluate_genetic_accumulated("polygenic", gm)

        vals = tm.get("polygenic")
        assert list(vals) == [0, 1, 2]
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_genetics.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(traits): add GENETIC_ACCUMULATED trait type for multi-locus polygenic phenotypes`

---

## Task 6: MultiPopulationManager

**Files:**
- Create: `salmon_ibm/interactions.py`
- Create: `tests/test_interactions.py`

Manages multiple `Population` instances sharing the same spatial landscape. Provides cross-population spatial queries (which agents from population B are co-located with agents from population A).

- [ ] **Step 1: Implement MultiPopulationManager**

```python
# salmon_ibm/interactions.py
"""Multi-species interaction system: population management and encounter events."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


class MultiPopulationManager:
    """Manages multiple named populations on a shared landscape.

    Provides cross-population spatial queries using cell-based hashing.
    """

    def __init__(self):
        self.populations: dict[str, Any] = {}  # name -> Population
        self._cell_index: dict[str, dict[int, np.ndarray]] = {}  # pop_name -> {cell_id -> agent_indices}

    def register(self, name: str, population) -> None:
        self.populations[name] = population

    def get(self, name: str):
        return self.populations[name]

    def build_cell_index(self, name: str) -> None:
        """Build cell-based spatial hash for a population.

        Maps cell_id -> array of alive agent indices at that cell.
        """
        pop = self.populations[name]
        alive = pop.alive
        positions = pop.tri_idx  # cell indices
        index: dict[int, list[int]] = {}
        alive_indices = np.where(alive)[0]
        for i in alive_indices:
            cell = int(positions[i])
            if cell not in index:
                index[cell] = []
            index[cell].append(i)
        self._cell_index[name] = {
            k: np.array(v, dtype=np.int64) for k, v in index.items()
        }

    def agents_at_cell(self, name: str, cell_id: int) -> np.ndarray:
        """Return indices of alive agents from population `name` at `cell_id`."""
        idx = self._cell_index.get(name, {})
        return idx.get(cell_id, np.array([], dtype=np.int64))

    def co_located_pairs(self, pop_a: str, pop_b: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Find cells where both populations have agents.

        Returns list of (agents_a, agents_b) for each shared cell.
        """
        self.build_cell_index(pop_a)
        self.build_cell_index(pop_b)
        idx_a = self._cell_index[pop_a]
        idx_b = self._cell_index[pop_b]
        shared_cells = set(idx_a.keys()) & set(idx_b.keys())
        pairs = []
        for cell in shared_cells:
            pairs.append((idx_a[cell], idx_b[cell]))
        return pairs
```

- [ ] **Step 2: Write tests**

```python
# tests/test_interactions.py
"""Unit tests for multi-species interactions."""
import numpy as np
import pytest

from salmon_ibm.interactions import MultiPopulationManager


class MockPopulation:
    """Minimal population mock for testing."""
    def __init__(self, n, positions, alive=None):
        self.n = n
        self.tri_idx = np.array(positions, dtype=np.int64)
        self.alive = np.ones(n, dtype=bool) if alive is None else np.array(alive, dtype=bool)


class TestMultiPopulationManager:
    def test_register_and_get(self):
        mgr = MultiPopulationManager()
        pop = MockPopulation(5, [0, 1, 2, 3, 4])
        mgr.register("prey", pop)
        assert mgr.get("prey") is pop

    def test_cell_index(self):
        mgr = MultiPopulationManager()
        pop = MockPopulation(5, [10, 10, 20, 20, 20])
        mgr.register("prey", pop)
        mgr.build_cell_index("prey")
        assert len(mgr.agents_at_cell("prey", 10)) == 2
        assert len(mgr.agents_at_cell("prey", 20)) == 3
        assert len(mgr.agents_at_cell("prey", 99)) == 0

    def test_co_located_pairs(self):
        mgr = MultiPopulationManager()
        predators = MockPopulation(3, [10, 20, 30])
        prey = MockPopulation(4, [20, 20, 40, 10])
        mgr.register("predator", predators)
        mgr.register("prey", prey)
        pairs = mgr.co_located_pairs("predator", "prey")
        # Cells 10 and 20 are shared
        assert len(pairs) == 2

    def test_dead_agents_excluded(self):
        mgr = MultiPopulationManager()
        alive = [True, False, True]
        pop = MockPopulation(3, [10, 10, 10], alive=alive)
        mgr.register("a", pop)
        mgr.build_cell_index("a")
        assert len(mgr.agents_at_cell("a", 10)) == 2
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_interactions.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(interactions): add MultiPopulationManager with cell-based spatial hashing`

---

## Task 7: InteractionEvent — Probabilistic Encounters

**Files:**
- Modify: `salmon_ibm/interactions.py`
- Modify: `salmon_ibm/events_phase3.py`
- Modify: `tests/test_interactions.py`

Implement probabilistic encounters between co-located individuals from different populations. Supports predation (prey dies, predator gains resources) and competition (loser incurs accumulator penalty).

- [ ] **Step 1: Define InteractionOutcome enum and InteractionEvent**

```python
# In salmon_ibm/interactions.py
from enum import Enum

class InteractionOutcome(Enum):
    PREDATION = "predation"       # prey dies, predator gains resource
    COMPETITION = "competition"   # loser incurs penalty
    DISEASE = "disease"           # pathogen transmission (handled by TransitionEvent)


@register_event("interaction")
@dataclass
class InteractionEvent(Event):
    """Probabilistic encounter between two co-located populations.

    Follows the Event ABC. Receives the MultiPopulationManager via
    ``landscape["multi_pop_mgr"]`` and the RNG via ``landscape["rng"]``.
    The ``population`` parameter in execute() is the primary population;
    the partner population is retrieved from the multi-pop manager.
    """
    pop_a_name: str = ""  # e.g., "predator"
    pop_b_name: str = ""  # e.g., "prey"
    encounter_probability: float = 0.1  # P(encounter) per co-located pair
    outcome: InteractionOutcome = InteractionOutcome.PREDATION
    resource_gain_acc: str | None = None
    resource_gain_amount: float = 0.0
    penalty_acc: str | None = None  # accumulator to decrement on loser
    penalty_amount: float = 0.0

    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
        """Run encounters. Retrieves MultiPopulationManager from landscape."""
        multi_pop_mgr = landscape["multi_pop_mgr"]
        rng = landscape.get("rng", np.random.default_rng())
        pairs = multi_pop_mgr.co_located_pairs(self.pop_a_name, self.pop_b_name)
        pop_a = multi_pop_mgr.get(self.pop_a_name)
        pop_b = multi_pop_mgr.get(self.pop_b_name)

        stats = {"encounters": 0, "kills": 0}

        for agents_a, agents_b in pairs:
            # Each agent in A encounters each agent in B with probability p
            for a_idx in agents_a:
                for b_idx in agents_b:
                    if rng.random() < self.encounter_probability:
                        stats["encounters"] += 1
                        if self.outcome == InteractionOutcome.PREDATION:
                            pop_b.alive[b_idx] = False
                            stats["kills"] += 1
                            if self.resource_gain_acc and pop_a.accumulators is not None:
                                acc_idx = pop_a.accumulators._resolve_idx(self.resource_gain_acc)
                                pop_a.accumulators.data[a_idx, acc_idx] += self.resource_gain_amount

        return stats
```

- [ ] **Step 2: Write tests**

```python
# In tests/test_interactions.py

class TestInteractionEvent:
    def test_predation_kills_prey(self):
        from salmon_ibm.interactions import InteractionEvent, InteractionOutcome
        mgr = MultiPopulationManager()
        predator = MockPopulation(1, [10])
        prey = MockPopulation(1, [10])
        mgr.register("predator", predator)
        mgr.register("prey", prey)

        event = InteractionEvent(
            name="hunt", pop_a_name="predator", pop_b_name="prey",
            encounter_probability=1.0,  # guaranteed encounter
            outcome=InteractionOutcome.PREDATION,
        )
        rng = np.random.default_rng(42)
        stats = event.execute(mgr, rng)
        assert stats["kills"] == 1
        assert prey.alive[0] == False
        assert predator.alive[0] == True

    def test_no_encounter_when_different_cells(self):
        from salmon_ibm.interactions import InteractionEvent, InteractionOutcome
        mgr = MultiPopulationManager()
        predator = MockPopulation(1, [10])
        prey = MockPopulation(1, [20])  # different cell
        mgr.register("predator", predator)
        mgr.register("prey", prey)

        event = InteractionEvent(
            name="hunt", pop_a_name="predator", pop_b_name="prey",
            encounter_probability=1.0,
            outcome=InteractionOutcome.PREDATION,
        )
        rng = np.random.default_rng(42)
        stats = event.execute(mgr, rng)
        assert stats["encounters"] == 0
        assert prey.alive[0] == True

    def test_probabilistic_encounter(self):
        from salmon_ibm.interactions import InteractionEvent, InteractionOutcome
        mgr = MultiPopulationManager()
        predator = MockPopulation(1, [10])
        prey = MockPopulation(100, [10] * 100)
        mgr.register("predator", predator)
        mgr.register("prey", prey)

        event = InteractionEvent(
            name="hunt", pop_a_name="predator", pop_b_name="prey",
            encounter_probability=0.5,
            outcome=InteractionOutcome.PREDATION,
        )
        rng = np.random.default_rng(42)
        stats = event.execute(mgr, rng)
        # With p=0.5, expect roughly 50 kills (not 0 and not 100)
        assert 20 < stats["kills"] < 80
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_interactions.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(interactions): add InteractionEvent with predation and probabilistic encounters`

---

## Task 8: TransitionEvent — Probabilistic Trait Modification (SEIR Disease)

**Files:**
- Modify: `salmon_ibm/events_phase3.py`
- Create: `tests/test_events_phase3.py`

A TransitionEvent modifies a probabilistic trait's category based on a transition matrix. This is the core mechanism for SEIR disease modeling: `S -> E -> I -> R` with per-timestep transition probabilities.

- [ ] **Step 1: Implement TransitionEvent**

```python
# In salmon_ibm/events_phase3.py

@register_event("transition")
@dataclass
class TransitionEvent(Event):
    """Modify a probabilistic trait using a transition matrix.

    transition_matrix[i, j] = P(category i -> category j) per timestep.
    Rows must sum to 1.0. Used for SEIR disease states, life stages, etc.
    """
    trait_name: str = ""
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
        traits = population.traits
        if traits is None:
            return
        defn = traits.definitions[self.trait_name]
        n_categories = len(defn.categories)
        current = traits.get(self.trait_name)

        masked_indices = np.where(mask)[0]
        if len(masked_indices) == 0:
            return

        rng = landscape.get("rng", np.random.default_rng())
        current_cats = current[masked_indices]
        new_cats = np.empty_like(current_cats)

        for cat_val in range(n_categories):
            cat_mask = current_cats == cat_val
            if not cat_mask.any():
                continue
            n = cat_mask.sum()
            probs = self.transition_matrix[cat_val]
            drawn = rng.choice(n_categories, size=n, p=probs)
            new_cats[cat_mask] = drawn

        traits.set(self.trait_name, new_cats, mask=mask)
```

- [ ] **Step 2: Write tests**

```python
# tests/test_events_phase3.py
"""Unit tests for Phase 3 events."""
import numpy as np
import pytest

from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType


class MockPopulation:
    """Minimal population mock for event testing."""
    def __init__(self, n):
        self.n = n
        self.alive = np.ones(n, dtype=bool)
        self.arrived = np.zeros(n, dtype=bool)
        self.traits = None
        self.accumulators = None
        self.genome = None
        self.tri_idx = np.zeros(n, dtype=np.int64)


class TestTransitionEvent:
    def test_seir_forced_transitions(self):
        """S->E with probability 1.0 should move all susceptible to exposed."""
        from salmon_ibm.events_phase3 import TransitionEvent

        pop = MockPopulation(10)
        trait_def = TraitDefinition(
            name="disease", trait_type=TraitType.PROBABILISTIC,
            categories=["S", "E", "I", "R"],
        )
        pop.traits = TraitManager(n_agents=10, definitions=[trait_def])
        # All start as S (category 0)
        pop.traits.set("disease", np.zeros(10, dtype=np.int32))

        T = np.array([
            [0.0, 1.0, 0.0, 0.0],  # S -> E always
            [0.0, 0.0, 1.0, 0.0],  # E -> I always
            [0.0, 0.0, 0.0, 1.0],  # I -> R always
            [0.0, 0.0, 0.0, 1.0],  # R stays
        ])
        event = TransitionEvent(name="infect", trait_name="disease", transition_matrix=T)
        mask = np.ones(10, dtype=bool)

        event.execute(pop, {}, t=0, mask=mask)
        assert np.all(pop.traits.get("disease") == 1)  # all E now

        event.execute(pop, {}, t=1, mask=mask)
        assert np.all(pop.traits.get("disease") == 2)  # all I now

    def test_identity_matrix_no_change(self):
        from salmon_ibm.events_phase3 import TransitionEvent

        pop = MockPopulation(10)
        trait_def = TraitDefinition(
            name="state", trait_type=TraitType.PROBABILISTIC,
            categories=["A", "B"],
        )
        pop.traits = TraitManager(n_agents=10, definitions=[trait_def])
        pop.traits.set("state", np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32))
        original = pop.traits.get("state").copy()

        T = np.eye(2)
        event = TransitionEvent(name="noop", trait_name="state", transition_matrix=T)
        mask = np.ones(10, dtype=bool)
        event.execute(pop, {}, t=0, mask=mask)
        np.testing.assert_array_equal(pop.traits.get("state"), original)
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_events_phase3.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(events): add TransitionEvent for SEIR disease modeling and trait state transitions`

---

## Task 9: StreamNetwork Class — Segments, Nodes, Connectivity

**Files:**
- Create: `salmon_ibm/network.py`
- Create: `tests/test_network.py`

Implement the 1D branching stream network as a directed graph of segments connected at nodes. Each segment has properties (length, stream order, flow direction). The network supports upstream/downstream traversal.

- [ ] **Step 1: Define SegmentDef and StreamNetwork**

```python
# salmon_ibm/network.py
"""1D branching stream network for aquatic species."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SegmentDef:
    """A single stream segment."""
    id: int
    length: float  # segment length in spatial units
    order: int = 1  # Strahler stream order
    upstream_node: int = -1  # node ID at upstream end
    downstream_node: int = -1  # node ID at downstream end
    capacity: int = -1  # max agents, -1 for unlimited


class StreamNetwork:
    """1D branching directional network for aquatic species.

    Segments are connected at nodes. Each segment has a direction
    (upstream_node -> downstream_node following flow). Agents occupy
    positions along segments defined as (segment_id, offset) where
    offset is in [0, segment_length].
    """

    def __init__(self, segments: list[SegmentDef]):
        self.segments = {s.id: s for s in segments}
        self.n_segments = len(segments)

        # Build connectivity: node_id -> list of segment_ids
        self._node_to_segments: dict[int, list[int]] = {}
        for s in segments:
            for node in (s.upstream_node, s.downstream_node):
                if node >= 0:
                    self._node_to_segments.setdefault(node, []).append(s.id)

        # Precompute upstream/downstream neighbors per segment
        self._upstream: dict[int, list[int]] = {}  # seg_id -> upstream segment IDs
        self._downstream: dict[int, list[int]] = {}  # seg_id -> downstream segment IDs
        for s in segments:
            self._upstream[s.id] = []
            self._downstream[s.id] = []

        for s in segments:
            # Upstream neighbors: other segments whose downstream_node == this segment's upstream_node
            if s.upstream_node >= 0:
                for other_id in self._node_to_segments.get(s.upstream_node, []):
                    if other_id != s.id:
                        other = self.segments[other_id]
                        if other.downstream_node == s.upstream_node:
                            self._upstream[s.id].append(other_id)
            # Downstream neighbors: other segments whose upstream_node == this segment's downstream_node
            if s.downstream_node >= 0:
                for other_id in self._node_to_segments.get(s.downstream_node, []):
                    if other_id != s.id:
                        other = self.segments[other_id]
                        if other.upstream_node == s.downstream_node:
                            self._downstream[s.id].append(other_id)

    def upstream_of(self, segment_id: int) -> list[int]:
        """Return segment IDs directly upstream of the given segment."""
        return self._upstream.get(segment_id, [])

    def downstream_of(self, segment_id: int) -> list[int]:
        """Return segment IDs directly downstream of the given segment."""
        return self._downstream.get(segment_id, [])

    def segment_length(self, segment_id: int) -> float:
        return self.segments[segment_id].length

    def path_downstream(self, segment_id: int, max_depth: int = 100) -> list[int]:
        """Return ordered list of segment IDs following downstream path.

        At confluences, follows the first downstream segment.
        """
        path = []
        current = segment_id
        for _ in range(max_depth):
            ds = self.downstream_of(current)
            if not ds:
                break
            current = ds[0]
            path.append(current)
        return path

    def path_upstream(self, segment_id: int, max_depth: int = 100) -> list[int]:
        """Return ordered list of segment IDs following upstream mainstem path.

        At bifurcations, follows the highest-order upstream segment.
        """
        path = []
        current = segment_id
        for _ in range(max_depth):
            us = self.upstream_of(current)
            if not us:
                break
            # Pick highest order
            current = max(us, key=lambda sid: self.segments[sid].order)
            path.append(current)
        return path
```

- [ ] **Step 2: Write tests**

```python
# tests/test_network.py
"""Unit tests for the stream network module."""
import numpy as np
import pytest

from salmon_ibm.network import SegmentDef, StreamNetwork


def _simple_Y_network():
    """Create a simple Y-shaped network:

        seg0 (trib1)
              \\
               node1 --- seg2 (mainstem) --- node3 (outlet)
              /
        seg1 (trib2)

    seg0: node0 -> node1
    seg1: node2 -> node1
    seg2: node1 -> node3
    """
    return StreamNetwork([
        SegmentDef(id=0, length=100.0, order=1, upstream_node=0, downstream_node=1),
        SegmentDef(id=1, length=80.0, order=1, upstream_node=2, downstream_node=1),
        SegmentDef(id=2, length=200.0, order=2, upstream_node=1, downstream_node=3),
    ])


class TestStreamNetwork:
    def test_downstream_of_tributaries(self):
        net = _simple_Y_network()
        assert net.downstream_of(0) == [2]
        assert net.downstream_of(1) == [2]

    def test_upstream_of_mainstem(self):
        net = _simple_Y_network()
        us = net.upstream_of(2)
        assert set(us) == {0, 1}

    def test_outlet_has_no_downstream(self):
        net = _simple_Y_network()
        assert net.downstream_of(2) == []

    def test_headwaters_have_no_upstream(self):
        net = _simple_Y_network()
        assert net.upstream_of(0) == []
        assert net.upstream_of(1) == []

    def test_path_downstream(self):
        net = _simple_Y_network()
        path = net.path_downstream(0)
        assert path == [2]

    def test_path_upstream_follows_highest_order(self):
        net = _simple_Y_network()
        # From outlet segment 2, upstream should pick one of the tributaries
        path = net.path_upstream(2)
        assert len(path) == 1
        assert path[0] in [0, 1]

    def test_segment_length(self):
        net = _simple_Y_network()
        assert net.segment_length(0) == 100.0
        assert net.segment_length(2) == 200.0
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_network.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(network): add StreamNetwork class with segment connectivity and path traversal`

---

## Task 10: Network Movement — Upstream/Downstream Along Branches

**Files:**
- Modify: `salmon_ibm/network.py`
- Modify: `tests/test_network.py`

Agents on the network have positions `(segment_id, offset)`. Movement advances the offset along the segment. When an agent reaches a segment boundary, it transitions to the adjacent segment (upstream or downstream).

- [ ] **Step 1: Implement NetworkAgentState and movement functions**

```python
# In salmon_ibm/network.py

class NetworkAgentState:
    """Position state for agents on a stream network.

    Each agent's position is (segment_id, offset) where offset is the
    distance from the upstream end of the segment.
    """

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.segment_id = np.zeros(n_agents, dtype=np.int64)
        self.offset = np.zeros(n_agents, dtype=np.float64)

    def place(self, agent_indices: np.ndarray, segment_ids: np.ndarray,
              offsets: np.ndarray) -> None:
        self.segment_id[agent_indices] = segment_ids
        self.offset[agent_indices] = offsets


def move_downstream(network: StreamNetwork, state: NetworkAgentState,
                    distances: np.ndarray, mask: np.ndarray,
                    rng: np.random.Generator) -> None:
    """Move masked agents downstream by given distances.

    When an agent overshoots a segment boundary, it transitions to the
    downstream segment. At confluences with multiple downstream options,
    one is chosen randomly.
    """
    indices = np.where(mask)[0]
    for i in indices:
        remaining = distances[i]
        seg = state.segment_id[i]
        off = state.offset[i]

        while remaining > 0:
            seg_len = network.segment_length(seg)
            space_left = seg_len - off  # distance to downstream end

            if remaining <= space_left:
                off += remaining
                remaining = 0
            else:
                remaining -= space_left
                ds = network.downstream_of(seg)
                if not ds:
                    off = seg_len  # stuck at outlet
                    remaining = 0
                else:
                    seg = ds[0] if len(ds) == 1 else rng.choice(ds)
                    off = 0.0

        state.segment_id[i] = seg
        state.offset[i] = off


def move_upstream(network: StreamNetwork, state: NetworkAgentState,
                  distances: np.ndarray, mask: np.ndarray,
                  rng: np.random.Generator) -> None:
    """Move masked agents upstream by given distances.

    At bifurcations with multiple upstream branches, one is chosen randomly.
    """
    indices = np.where(mask)[0]
    for i in indices:
        remaining = distances[i]
        seg = state.segment_id[i]
        off = state.offset[i]

        while remaining > 0:
            if remaining <= off:
                off -= remaining
                remaining = 0
            else:
                remaining -= off
                us = network.upstream_of(seg)
                if not us:
                    off = 0.0  # stuck at headwater
                    remaining = 0
                else:
                    seg = us[0] if len(us) == 1 else rng.choice(us)
                    off = network.segment_length(seg)

        state.segment_id[i] = seg
        state.offset[i] = off
```

- [ ] **Step 2: Write tests**

```python
# In tests/test_network.py

from salmon_ibm.network import NetworkAgentState, move_downstream, move_upstream


class TestNetworkMovement:
    def test_move_downstream_within_segment(self):
        net = _simple_Y_network()
        state = NetworkAgentState(1)
        state.place(np.array([0]), np.array([2]), np.array([0.0]))
        mask = np.array([True])
        distances = np.array([50.0])
        rng = np.random.default_rng(42)
        move_downstream(net, state, distances, mask, rng)
        assert state.segment_id[0] == 2
        assert state.offset[0] == 50.0

    def test_move_downstream_crosses_segment_boundary(self):
        net = _simple_Y_network()
        state = NetworkAgentState(1)
        state.place(np.array([0]), np.array([0]), np.array([50.0]))
        mask = np.array([True])
        distances = np.array([80.0])  # 50 remaining in seg0 + 30 into seg2
        rng = np.random.default_rng(42)
        move_downstream(net, state, distances, mask, rng)
        assert state.segment_id[0] == 2
        assert abs(state.offset[0] - 30.0) < 1e-10

    def test_move_downstream_stops_at_outlet(self):
        net = _simple_Y_network()
        state = NetworkAgentState(1)
        state.place(np.array([0]), np.array([2]), np.array([100.0]))
        mask = np.array([True])
        distances = np.array([500.0])  # more than total remaining
        rng = np.random.default_rng(42)
        move_downstream(net, state, distances, mask, rng)
        assert state.segment_id[0] == 2
        assert state.offset[0] == 200.0  # clamped at outlet

    def test_move_upstream_within_segment(self):
        net = _simple_Y_network()
        state = NetworkAgentState(1)
        state.place(np.array([0]), np.array([2]), np.array([150.0]))
        mask = np.array([True])
        distances = np.array([50.0])
        rng = np.random.default_rng(42)
        move_upstream(net, state, distances, mask, rng)
        assert state.segment_id[0] == 2
        assert abs(state.offset[0] - 100.0) < 1e-10

    def test_move_upstream_crosses_into_tributary(self):
        net = _simple_Y_network()
        state = NetworkAgentState(1)
        state.place(np.array([0]), np.array([2]), np.array([30.0]))
        mask = np.array([True])
        distances = np.array([50.0])  # 30 back to node1, then 20 into a tributary
        rng = np.random.default_rng(42)
        move_upstream(net, state, distances, mask, rng)
        # Should be in one of the tributaries (seg 0 or 1)
        assert state.segment_id[0] in [0, 1]

    def test_move_upstream_stops_at_headwater(self):
        net = _simple_Y_network()
        state = NetworkAgentState(1)
        state.place(np.array([0]), np.array([0]), np.array([10.0]))
        mask = np.array([True])
        distances = np.array([500.0])
        rng = np.random.default_rng(42)
        move_upstream(net, state, distances, mask, rng)
        assert state.segment_id[0] == 0
        assert state.offset[0] == 0.0
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_network.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(network): add upstream/downstream movement with segment boundary crossing`

---

## Task 11: NetworkRangeDynamics — Territory on Network Segments

**Files:**
- Modify: `salmon_ibm/network.py`
- Modify: `tests/test_network.py`

Agents can claim territory on network segments. A range is a contiguous stretch of one or more segments. Range dynamics allow territory expansion, contraction, and reallocation based on resource thresholds.

- [ ] **Step 1: Implement NetworkRange and NetworkRangeManager**

```python
# In salmon_ibm/network.py

@dataclass
class NetworkRange:
    """A contiguous territory on the stream network."""
    owner: int  # agent index
    segments: list[int]  # ordered list of segment IDs in this range
    start_offset: float  # offset on first segment
    end_offset: float  # offset on last segment

    def total_length(self, network: StreamNetwork) -> float:
        """Total length of the range across all spanned segments."""
        if len(self.segments) == 1:
            return self.end_offset - self.start_offset
        # First segment: from start_offset to segment end
        total = network.segment_length(self.segments[0]) - self.start_offset
        # Middle segments: full length
        for seg in self.segments[1:-1]:
            total += network.segment_length(seg)
        # Last segment: from start to end_offset
        total += self.end_offset
        return total


class NetworkRangeManager:
    """Manages territory allocation on a stream network."""

    def __init__(self, network: StreamNetwork):
        self.network = network
        self.ranges: dict[int, NetworkRange] = {}  # agent_index -> range
        # Track occupancy: segment_id -> set of agent indices claiming it
        self._segment_occupancy: dict[int, set[int]] = {
            sid: set() for sid in network.segments
        }

    def allocate(self, agent_idx: int, segment_id: int,
                 start_offset: float, end_offset: float) -> bool:
        """Attempt to allocate a range on a single segment.

        Returns True if successful, False if the space is occupied.
        """
        # Simple non-overlapping check (simplified: whole segments only in v1)
        if self._segment_occupancy[segment_id]:
            return False
        r = NetworkRange(owner=agent_idx, segments=[segment_id],
                         start_offset=start_offset, end_offset=end_offset)
        self.ranges[agent_idx] = r
        self._segment_occupancy[segment_id].add(agent_idx)
        return True

    def release(self, agent_idx: int) -> None:
        """Release an agent's territory."""
        if agent_idx in self.ranges:
            for seg in self.ranges[agent_idx].segments:
                self._segment_occupancy[seg].discard(agent_idx)
            del self.ranges[agent_idx]

    def is_occupied(self, segment_id: int) -> bool:
        return bool(self._segment_occupancy.get(segment_id))
```

- [ ] **Step 2: Write tests**

```python
# In tests/test_network.py

from salmon_ibm.network import NetworkRangeManager


class TestNetworkRangeManager:
    def test_allocate_and_query(self):
        net = _simple_Y_network()
        mgr = NetworkRangeManager(net)
        assert mgr.allocate(agent_idx=0, segment_id=0, start_offset=0, end_offset=100)
        assert mgr.is_occupied(0)
        assert not mgr.is_occupied(1)

    def test_cannot_double_allocate(self):
        net = _simple_Y_network()
        mgr = NetworkRangeManager(net)
        assert mgr.allocate(0, segment_id=0, start_offset=0, end_offset=100)
        assert not mgr.allocate(1, segment_id=0, start_offset=0, end_offset=100)

    def test_release_frees_segment(self):
        net = _simple_Y_network()
        mgr = NetworkRangeManager(net)
        mgr.allocate(0, segment_id=0, start_offset=0, end_offset=100)
        mgr.release(0)
        assert not mgr.is_occupied(0)
        assert mgr.allocate(1, segment_id=0, start_offset=0, end_offset=100)
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_network.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(network): add NetworkRangeManager for territory allocation on stream segments`

---

## Task 12: SwitchPopulation Event — Grid to Network Transfer

**Files:**
- Modify: `salmon_ibm/events_phase3.py`
- Modify: `tests/test_events_phase3.py`

The SwitchPopulation event transfers agents between the hex grid and the stream network (or vice versa). This supports life stages that transition between terrestrial/grid and aquatic/network habitats (e.g., salmon smolts entering a river network).

- [ ] **Step 1: Implement SwitchPopulationEvent**

```python
# In salmon_ibm/events_phase3.py

@register_event("switch_population")
@dataclass
class SwitchPopulationEvent(Event):
    """Transfer agents between grid and network spatial representations.

    mapping: dict mapping grid cell IDs to (segment_id, offset) tuples.
    direction: "grid_to_network" or "network_to_grid".
    """
    direction: str = "grid_to_network"
    cell_to_segment: dict = field(default_factory=dict)  # cell_id -> (seg_id, offset)
    segment_to_cell: dict = field(default_factory=dict)  # seg_id -> cell_id

    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return

        if self.direction == "grid_to_network":
            network_state = landscape.get("network_state")
            if network_state is None:
                return
            for i in indices:
                cell = int(population.tri_idx[i])
                if cell in self.cell_to_segment:
                    seg_id, offset = self.cell_to_segment[cell]
                    network_state.segment_id[i] = seg_id
                    network_state.offset[i] = offset

        elif self.direction == "network_to_grid":
            network_state = landscape.get("network_state")
            if network_state is None:
                return
            for i in indices:
                seg = int(network_state.segment_id[i])
                if seg in self.segment_to_cell:
                    population.tri_idx[i] = self.segment_to_cell[seg]
```

- [ ] **Step 2: Write tests**

```python
# In tests/test_events_phase3.py

class TestSwitchPopulationEvent:
    def test_grid_to_network(self):
        from salmon_ibm.events_phase3 import SwitchPopulationEvent
        from salmon_ibm.network import NetworkAgentState

        pop = MockPopulation(3)
        pop.tri_idx = np.array([10, 20, 30], dtype=np.int64)
        net_state = NetworkAgentState(3)

        mapping = {10: (0, 0.0), 20: (1, 50.0)}  # cell 30 has no mapping
        landscape = {"network_state": net_state}

        event = SwitchPopulationEvent(
            name="to_river", direction="grid_to_network",
            cell_to_segment=mapping,
        )
        mask = np.ones(3, dtype=bool)
        event.execute(pop, landscape, t=0, mask=mask)

        assert net_state.segment_id[0] == 0
        assert net_state.offset[1] == 50.0
        assert net_state.segment_id[2] == 0  # unmapped, unchanged

    def test_network_to_grid(self):
        from salmon_ibm.events_phase3 import SwitchPopulationEvent
        from salmon_ibm.network import NetworkAgentState

        pop = MockPopulation(2)
        pop.tri_idx = np.array([0, 0], dtype=np.int64)
        net_state = NetworkAgentState(2)
        net_state.segment_id[:] = [5, 6]

        mapping = {5: 100, 6: 200}
        landscape = {"network_state": net_state}

        event = SwitchPopulationEvent(
            name="to_grid", direction="network_to_grid",
            segment_to_cell=mapping,
        )
        mask = np.ones(2, dtype=bool)
        event.execute(pop, landscape, t=0, mask=mask)

        assert pop.tri_idx[0] == 100
        assert pop.tri_idx[1] == 200
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_events_phase3.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(events): add SwitchPopulationEvent for grid-network transfer`

---

## Task 13: GeneratedHexmapEvent — Runtime Map Algebra

**Files:**
- Modify: `salmon_ibm/events_phase3.py`
- Modify: `tests/test_events_phase3.py`

A GeneratedHexmapEvent creates or modifies a hex map at runtime using algebraic expressions over existing maps and population data. Examples: density maps, resource depletion, environmental change.

- [ ] **Step 1: Implement GeneratedHexmapEvent**

```python
# In salmon_ibm/events_phase3.py

@register_event("generated_hexmap")
@dataclass
class GeneratedHexmapEvent(Event):
    """Create or update a hex map using an algebraic expression.

    The expression can reference:
      - Named hex maps in the landscape (e.g., "temperature", "habitat")
      - "density" — per-cell agent count
      - NumPy math functions (sqrt, exp, log, clip, etc.)

    Result is stored in landscape[output_name].
    """
    expression: str = ""
    output_name: str = ""

    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
        from salmon_ibm.accumulators import _validate_expression, _SAFE_MATH

        _validate_expression(self.expression)

        # Build namespace from landscape hex maps
        namespace = dict(_SAFE_MATH)
        namespace["t"] = t

        # Add density map
        if "density" in self.expression:
            n_cells = landscape.get("n_cells", 0)
            if n_cells > 0:
                alive = population.alive if hasattr(population, 'alive') else mask
                positions = population.tri_idx[alive & mask]
                density = np.bincount(positions, minlength=n_cells).astype(np.float64)
                namespace["density"] = density

        # Add hex maps from landscape
        for key, value in landscape.items():
            if isinstance(value, np.ndarray) and key not in namespace:
                namespace[key] = value

        result = eval(self.expression, {"__builtins__": {}}, namespace)
        landscape[self.output_name] = np.asarray(result, dtype=np.float64)
```

- [ ] **Step 2: Write tests**

```python
# In tests/test_events_phase3.py

class TestGeneratedHexmapEvent:
    def test_simple_expression(self):
        from salmon_ibm.events_phase3 import GeneratedHexmapEvent

        pop = MockPopulation(5)
        landscape = {
            "temperature": np.array([10.0, 15.0, 20.0, 25.0]),
            "n_cells": 4,
        }

        event = GeneratedHexmapEvent(
            name="thermal_stress",
            expression="maximum(temperature - 18.0, 0.0)",
            output_name="stress_map",
        )
        mask = np.ones(5, dtype=bool)
        event.execute(pop, landscape, t=0, mask=mask)

        expected = np.array([0.0, 0.0, 2.0, 7.0])
        np.testing.assert_array_almost_equal(landscape["stress_map"], expected)

    def test_time_varying_expression(self):
        from salmon_ibm.events_phase3 import GeneratedHexmapEvent

        pop = MockPopulation(5)
        base = np.array([10.0, 12.0, 14.0])
        landscape = {"base_temp": base, "n_cells": 3}

        event = GeneratedHexmapEvent(
            name="seasonal_temp",
            expression="base_temp + 5.0 * sin(t * pi / 180)",
            output_name="current_temp",
        )
        mask = np.ones(5, dtype=bool)
        event.execute(pop, landscape, t=90, mask=mask)

        # sin(90 * pi / 180) = 1.0
        expected = base + 5.0
        np.testing.assert_array_almost_equal(landscape["current_temp"], expected)
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_events_phase3.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(events): add GeneratedHexmapEvent for runtime map algebra`

---

## Task 14: RangeDynamicsEvent — Territory Resize on Grid

**Files:**
- Modify: `salmon_ibm/events_phase3.py`
- Modify: `tests/test_events_phase3.py`

RangeDynamicsEvent adjusts agent territories on the hex grid. Groups can expand (claim adjacent cells), contract (release peripheral cells), or shift based on resource availability. This operates on the Population's `RangeAllocator` (from Phase 2).

- [ ] **Step 1: Implement RangeDynamicsEvent**

```python
# In salmon_ibm/events_phase3.py

@register_event("range_dynamics")
@dataclass
class RangeDynamicsEvent(Event):
    """Adjust agent territories based on resource availability.

    mode: "expand", "contract", or "shift"
    resource_map_name: landscape key for the resource hex map
    resource_threshold: minimum resource value per cell to be viable
    max_range_size: maximum number of cells in a range
    """
    mode: str = "expand"
    resource_map_name: str = "resources"
    resource_threshold: float = 0.0
    max_range_size: int = 50

    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
        range_alloc = getattr(population, 'ranges', None)
        if range_alloc is None:
            return

        resource_map = landscape.get(self.resource_map_name)
        if resource_map is None:
            return

        indices = np.where(mask)[0]
        for i in indices:
            agent_range = range_alloc.get_range(i)
            if agent_range is None:
                continue

            if self.mode == "expand":
                self._expand(i, agent_range, resource_map, range_alloc, landscape)
            elif self.mode == "contract":
                self._contract(i, agent_range, resource_map, range_alloc)
            elif self.mode == "shift":
                self._contract(i, agent_range, resource_map, range_alloc)
                self._expand(i, agent_range, resource_map, range_alloc, landscape)

    def _expand(self, agent_idx, agent_range, resource_map, range_alloc, landscape):
        """Try to add adjacent cells meeting the resource threshold."""
        mesh = landscape.get("mesh")
        if mesh is None:
            return
        current_cells = set(agent_range.cells)
        if len(current_cells) >= self.max_range_size:
            return
        for cell in list(current_cells):
            neighbors = mesh._water_nbrs[cell, :mesh._water_nbr_count[cell]]
            for nb in neighbors:
                if nb not in current_cells and resource_map[nb] >= self.resource_threshold:
                    if range_alloc.try_add_cell(agent_idx, nb):
                        current_cells.add(nb)
                        if len(current_cells) >= self.max_range_size:
                            return

    def _contract(self, agent_idx, agent_range, resource_map, range_alloc):
        """Release cells below the resource threshold."""
        for cell in list(agent_range.cells):
            if resource_map[cell] < self.resource_threshold:
                range_alloc.release_cell(agent_idx, cell)
```

- [ ] **Step 2: Write tests**

```python
# In tests/test_events_phase3.py

class TestRangeDynamicsEvent:
    def test_placeholder_no_crash_without_ranges(self):
        """Event should silently do nothing if population has no RangeAllocator."""
        from salmon_ibm.events_phase3 import RangeDynamicsEvent

        pop = MockPopulation(5)
        landscape = {"resources": np.array([1.0, 2.0, 3.0])}
        event = RangeDynamicsEvent(name="range_expand", mode="expand")
        mask = np.ones(5, dtype=bool)
        # Should not raise
        event.execute(pop, landscape, t=0, mask=mask)
```

*Note: Full integration tests for RangeDynamicsEvent depend on Phase 2's `RangeAllocator` being implemented. The unit test here validates graceful degradation.*

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_events_phase3.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(events): add RangeDynamicsEvent for grid territory expansion and contraction`

---

## Task 15: SetAffinityEvent — Group and Spatial Affinity

**Files:**
- Modify: `salmon_ibm/events_phase3.py`
- Modify: `tests/test_events_phase3.py`

SetAffinityEvent modifies agent movement affinities. Group affinity biases movement toward group members. Spatial affinity biases movement toward specific map values. These affinities are read by the movement kernel to influence direction choices.

- [ ] **Step 1: Implement SetAffinityEvent**

```python
# In salmon_ibm/events_phase3.py

@register_event("set_affinity")
@dataclass
class SetAffinityEvent(Event):
    """Set movement affinity for agents.

    affinity_type: "group" or "spatial"
    For spatial: affinity_map_name is the landscape key for the affinity map.
    For group: strength controls attraction to group centroid.
    """
    affinity_type: str = "spatial"  # "group" or "spatial"
    affinity_map_name: str | None = None
    strength: float = 1.0

    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None:
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return

        if self.affinity_type == "spatial" and self.affinity_map_name:
            affinity_map = landscape.get(self.affinity_map_name)
            if affinity_map is None:
                return
            # Store per-agent spatial affinity reference
            if not hasattr(population, 'spatial_affinity'):
                return
            population.spatial_affinity[indices] = self.strength
            population.spatial_affinity_map_name = self.affinity_map_name

        elif self.affinity_type == "group":
            if not hasattr(population, 'group_affinity'):
                return
            population.group_affinity[indices] = self.strength
```

- [ ] **Step 2: Write tests**

```python
# In tests/test_events_phase3.py

class TestSetAffinityEvent:
    def test_spatial_affinity_sets_strength(self):
        from salmon_ibm.events_phase3 import SetAffinityEvent

        pop = MockPopulation(5)
        pop.spatial_affinity = np.zeros(5, dtype=np.float64)
        pop.spatial_affinity_map_name = None

        landscape = {"habitat_quality": np.array([1.0, 2.0, 3.0])}

        event = SetAffinityEvent(
            name="attract_to_habitat",
            affinity_type="spatial",
            affinity_map_name="habitat_quality",
            strength=0.8,
        )
        mask = np.array([True, True, False, False, True])
        event.execute(pop, landscape, t=0, mask=mask)

        assert pop.spatial_affinity[0] == 0.8
        assert pop.spatial_affinity[1] == 0.8
        assert pop.spatial_affinity[2] == 0.0  # unmasked, unchanged
        assert pop.spatial_affinity[4] == 0.8

    def test_group_affinity_sets_strength(self):
        from salmon_ibm.events_phase3 import SetAffinityEvent

        pop = MockPopulation(3)
        pop.group_affinity = np.zeros(3, dtype=np.float64)

        event = SetAffinityEvent(
            name="group_cohesion",
            affinity_type="group",
            strength=1.5,
        )
        mask = np.ones(3, dtype=bool)
        event.execute(pop, {}, t=0, mask=mask)

        assert np.all(pop.group_affinity == 1.5)

    def test_no_crash_without_affinity_attributes(self):
        from salmon_ibm.events_phase3 import SetAffinityEvent

        pop = MockPopulation(3)
        # No spatial_affinity or group_affinity attributes
        event = SetAffinityEvent(name="test", affinity_type="spatial", strength=1.0)
        mask = np.ones(3, dtype=bool)
        event.execute(pop, {}, t=0, mask=mask)  # should not raise
```

- [ ] **Step 3: Run tests**

```bash
conda run -n shiny python -m pytest tests/test_events_phase3.py -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

**Commit message:** `feat(events): add SetAffinityEvent for group and spatial movement affinity`

---

## Integration Checklist

After all 15 tasks are complete, run the full test suite to confirm nothing is broken:

- [ ] **Run full test suite**

```bash
conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_hexsim.py --ignore=tests/test_playwright.py
```

- [ ] **Verify new modules import cleanly**

```bash
conda run -n shiny python -c "from salmon_ibm.genetics import GenomeManager; from salmon_ibm.interactions import MultiPopulationManager; from salmon_ibm.network import StreamNetwork; print('All Phase 3 modules import OK')"
```

- [ ] **Verify event registration**

```bash
conda run -n shiny python -c "from salmon_ibm.events import EVENT_REGISTRY; import salmon_ibm.events_phase3; print('Registered events:', list(EVENT_REGISTRY.keys()))"
```

---

## Dependency Graph

```
Task 1 (GenomeManager)
  └── Task 2 (Recombination)
  └── Task 3 (MutationEvent)
  └── Task 4 (GENETIC trait) ── depends on Task 1
  └── Task 5 (GENETIC_ACCUMULATED trait) ── depends on Task 1

Task 6 (MultiPopulationManager)
  └── Task 7 (InteractionEvent) ── depends on Task 6

Task 8 (TransitionEvent) ── independent

Task 9 (StreamNetwork)
  └── Task 10 (Network movement) ── depends on Task 9
  └── Task 11 (NetworkRangeDynamics) ── depends on Task 9
  └── Task 12 (SwitchPopulation) ── depends on Task 9

Task 13 (GeneratedHexmapEvent) ── independent
Task 14 (RangeDynamicsEvent) ── depends on Phase 2 RangeAllocator
Task 15 (SetAffinityEvent) ── independent
```

**Parallelizable work streams:**
- Genetics (Tasks 1-5) can proceed independently of Network (Tasks 9-12)
- TransitionEvent (Task 8), GeneratedHexmapEvent (Task 13), and SetAffinityEvent (Task 15) are fully independent
- Multi-species (Tasks 6-7) is independent of both genetics and network

---

## Estimated Effort

| Group | Tasks | Est. LOC (impl + tests) | Effort |
|-------|-------|------------------------|--------|
| Genetics | 1-5 | ~2,600-4,100 | 3-5 days |
| Multi-species | 6-8 | ~1,800-3,300 | 2-3 days |
| Network | 9-12 | ~2,500-4,400 | 3-5 days |
| Remaining events | 13-15 | ~1,500-2,500 | 2-3 days |
| Integration testing | — | ~500-800 | 1 day |
| **Total** | **15 tasks** | **~8,900-15,100** | **~11-17 days** |
