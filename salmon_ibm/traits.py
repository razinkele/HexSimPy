"""Trait system: categorical per-agent state with auto-evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union
import numpy as np


class TraitType(Enum):
    PROBABILISTIC = "probabilistic"
    ACCUMULATED = "accumulated"
    GENETIC = "genetic"
    GENETIC_ACCUMULATED = "genetic_accumulated"


@dataclass
class TraitDefinition:
    name: str
    trait_type: TraitType
    categories: list[str]
    accumulator_name: str | None = None
    thresholds: np.ndarray | None = None  # ascending; len = len(categories) - 1
    # Genetic fields (Phase 3)
    locus_name: str | None = None  # for GENETIC type
    phenotype_map: np.ndarray | None = None  # int[n_alleles, n_alleles] -> category index
    locus_names: list[str] | None = None  # for GENETIC_ACCUMULATED (multi-locus)
    locus_weights: np.ndarray | None = None  # float[n_loci] weights for multi-locus


class TraitManager:
    """Vectorized storage and evaluation of per-agent categorical traits."""

    def __init__(self, n_agents: int, definitions: list[TraitDefinition]):
        self.n_agents = n_agents
        self.definitions: dict[str, TraitDefinition] = {d.name: d for d in definitions}
        self._data: dict[str, np.ndarray] = {
            d.name: np.zeros(n_agents, dtype=np.int32) for d in definitions
        }

    def get(self, name: str) -> np.ndarray:
        if name not in self._data:
            raise KeyError(f"Unknown trait: {name!r}")
        return self._data[name]

    def set(self, name: str, values: np.ndarray, mask: np.ndarray | None = None) -> None:
        if name not in self._data:
            raise KeyError(f"Unknown trait: {name!r}")
        if mask is not None:
            self._data[name][mask] = values
        else:
            self._data[name][:] = values

    def category_names(self, name: str) -> list[str]:
        defn = self.definitions[name]
        indices = self._data[name]
        return [defn.categories[i] for i in indices]

    def evaluate_accumulated(self, name: str, acc_manager, mask: np.ndarray | None = None) -> None:
        """Re-evaluate an accumulated trait by binning its linked accumulator."""
        defn = self.definitions[name]
        if defn.trait_type != TraitType.ACCUMULATED:
            raise ValueError(f"Trait {name!r} is {defn.trait_type.value}, not accumulated")
        acc_values = acc_manager.get(defn.accumulator_name)
        categories = np.digitize(acc_values, defn.thresholds).astype(np.int32)
        if mask is not None:
            self._data[name][mask] = categories[mask]
        else:
            self._data[name][:] = categories

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

    def _resolve_category(self, trait_name: str, value) -> list[int]:
        defn = self.definitions[trait_name]
        if isinstance(value, (list, tuple)):
            return [self._resolve_single_category(trait_name, v, defn) for v in value]
        return [self._resolve_single_category(trait_name, value, defn)]

    def _resolve_single_category(self, trait_name: str, value, defn: TraitDefinition) -> int:
        if isinstance(value, str):
            return defn.categories.index(value)
        return int(value)

    def filter_by_traits(self, **criteria) -> np.ndarray:
        """Return boolean mask for agents matching all criteria (AND across traits, OR within)."""
        mask = np.ones(self.n_agents, dtype=bool)
        for trait_name, value in criteria.items():
            indices = self._resolve_category(trait_name, value)
            trait_vals = self._data[trait_name]
            trait_mask = np.zeros(self.n_agents, dtype=bool)
            for idx in indices:
                trait_mask |= (trait_vals == idx)
            mask &= trait_mask
        return mask
