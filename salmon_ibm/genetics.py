"""Diploid genetic system: genotype storage, recombination, mutation."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


def _haldane_crossover_probs(linkage_distances_cM: np.ndarray) -> np.ndarray:
    """Convert linkage distances (cM) to crossover probabilities via Haldane."""
    return 0.5 * (1.0 - np.exp(-2.0 * linkage_distances_cM / 100.0))


@njit
def _produce_gamete_numba(parent_genotype, crossover_probs, rng_draws):
    """Produce a haploid gamete from a diploid parent via crossover.

    Args:
        parent_genotype: int[n_loci, 2] -- diploid genotype
        crossover_probs: float[n_loci - 1] -- P(crossover) between adjacent loci
        rng_draws: float[n_loci] -- pre-drawn uniform random values
            rng_draws[0] selects the initial strand;
            rng_draws[1:] are used for crossover decisions.

    Returns:
        int[n_loci] -- haploid gamete
    """
    n_loci = parent_genotype.shape[0]
    gamete = np.empty(n_loci, dtype=np.int32)
    strand = 0 if rng_draws[0] < 0.5 else 1  # random initial strand (unbiased)
    gamete[0] = parent_genotype[0, strand]
    for i in range(1, n_loci):
        if rng_draws[i] < crossover_probs[i - 1]:
            strand = 1 - strand  # crossover
        gamete[i] = parent_genotype[i, strand]
    return gamete


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

    def recombine(self, parent1_indices: np.ndarray, parent2_indices: np.ndarray,
                  offspring_indices: np.ndarray) -> None:
        """Create offspring genotypes by recombining two parents.

        Each parent contributes one gamete (with crossover) to each offspring.
        parent1_indices, parent2_indices, offspring_indices are aligned arrays.
        """
        n_offspring = len(offspring_indices)
        crossover_probs = _haldane_crossover_probs(self.linkage_distances)
        n_draws = max(self.n_loci, 1)  # need at least 1 draw for initial strand

        for k in range(n_offspring):
            p1 = self.genotypes[parent1_indices[k]]
            p2 = self.genotypes[parent2_indices[k]]

            draws1 = self.rng.random(n_draws)
            draws2 = self.rng.random(n_draws)

            gamete1 = _produce_gamete_numba(p1, crossover_probs, draws1)
            gamete2 = _produce_gamete_numba(p2, crossover_probs, draws2)

            self.genotypes[offspring_indices[k], :, 0] = gamete1
            self.genotypes[offspring_indices[k], :, 1] = gamete2

    def mutate(self, locus_name: str, transition_matrix: np.ndarray,
               mask: np.ndarray | None = None) -> int:
        """Apply allele mutations at a single locus using a transition matrix.

        Args:
            locus_name: which locus to mutate
            transition_matrix: float[n_alleles, n_alleles], rows sum to 1.0
            mask: optional bool[n_agents] -- only mutate these agents

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
