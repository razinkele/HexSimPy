"""Unit tests for the genetics sub-model."""
import numpy as np
import pytest

from salmon_ibm.genetics import LocusDefinition, GenomeManager, _haldane_crossover_probs


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
        # Run many times -- should always get [0,2] or [1,3] from each parent
        for _ in range(50):
            gm.recombine(parent1, parent2, offspring)
            g = gm.genotypes[2]
            assert (g[0, 0], g[1, 0]) in [(0, 2), (1, 3)]
            assert (g[0, 1], g[1, 1]) in [(0, 2), (1, 3)]

    def test_haldane_crossover_probs(self):
        # At 50 cM, probability should be ~0.316
        probs = _haldane_crossover_probs(np.array([50.0]))
        assert 0.30 < probs[0] < 0.33
        # At 0 cM, probability should be 0
        probs = _haldane_crossover_probs(np.array([0.0]))
        assert probs[0] == 0.0


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


class TestGeneticTrait:
    def test_simple_dominance(self):
        """Allele 0 dominant over allele 1: heterozygotes express category 0."""
        from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType

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


class TestGeneticAccumulatedTrait:
    def test_two_locus_additive(self):
        """Two loci with equal weight, thresholds at [2, 4]."""
        from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType

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
