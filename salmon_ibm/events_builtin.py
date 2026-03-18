"""Built-in event types that wrap existing salmon IBM logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from salmon_ibm.events import Event, EveryStep, EventTrigger, register_event
from salmon_ibm.movement import execute_movement
from salmon_ibm.bioenergetics import BioParams, update_energy
from salmon_ibm.estuary import salinity_cost
from salmon_ibm.population import Population


@register_event("movement")
@dataclass
class MovementEvent(Event):
    """Wraps execute_movement() as an event."""
    n_micro_steps: int = 3
    cwr_threshold: float = 16.0

    def execute(self, population, landscape, t, mask):
        mesh = landscape["mesh"]
        fields = landscape["fields"]
        rng = landscape["rng"]
        barrier_arrays = landscape.get("barrier_arrays")
        execute_movement(
            population, mesh, fields,
            seed=int(rng.integers(2**31)),
            n_micro_steps=self.n_micro_steps,
            cwr_threshold=self.cwr_threshold,
            barrier_arrays=barrier_arrays,
        )


@register_event("survival")
@dataclass
class SurvivalEvent(Event):
    """Bioenergetics energy update + thermal/starvation mortality."""
    bio_params: BioParams = field(default_factory=BioParams)
    thermal: bool = True
    starvation: bool = True

    def execute(self, population, landscape, t, mask):
        fields = landscape["fields"]
        activity_lut = landscape["activity_lut"]
        est_cfg = landscape.get("est_cfg", {})

        temps = fields["temperature"][population.tri_idx]
        alive = mask

        if self.starvation and alive.any():
            activity = activity_lut[population.behavior]
            sal = fields.get("salinity", np.zeros(len(fields["temperature"])))
            sal_at_agents = sal[population.tri_idx]
            s_cfg = est_cfg.get("salinity_cost", {})
            sal_cost_arr = salinity_cost(
                sal_at_agents,
                S_opt=s_cfg.get("S_opt", 0.5),
                S_tol=s_cfg.get("S_tol", 6.0),
                k=s_cfg.get("k", 0.6),
            )
            new_ed, dead, new_mass = update_energy(
                population.ed_kJ_g[alive], population.mass_g[alive],
                temps[alive], activity[alive], sal_cost_arr[alive],
                self.bio_params,
            )
            population.ed_kJ_g[alive] = new_ed
            population.mass_g[alive] = new_mass
            dead_indices = np.where(alive)[0][dead]
            population.alive[dead_indices] = False

        if self.thermal:
            thermal_kill = alive & (temps >= self.bio_params.T_MAX)
            population.alive[thermal_kill] = False


@register_event("accumulate")
@dataclass
class AccumulateEvent(Event):
    """Runs updater functions that modify agent state."""
    updaters: list[Callable] = field(default_factory=list)

    def execute(self, population, landscape, t, mask):
        for updater in self.updaters:
            updater(population, landscape, t, mask)


@dataclass
class CustomEvent(Event):
    """Generic event that delegates to a Python callback."""
    callback: Callable = field(default=lambda pop, land, t, mask: None)

    def execute(self, population, landscape, t, mask):
        self.callback(population, landscape, t, mask)


@register_event("stage_survival")
@dataclass
class StageSpecificSurvivalEvent(Event):
    """Trait-filtered survival with stage-specific mortality rates."""
    trait_name: str = "stage"
    mortality_rates: dict[str, float] = field(default_factory=dict)
    density_dependent: bool = False
    density_scale: float = 1.0

    def execute(self, population, landscape, t, mask):
        if not mask.any():
            return
        rng = landscape.get("rng", np.random.default_rng())
        trait_mgr = getattr(population, 'trait_mgr', None) or getattr(population, 'traits', None)
        if trait_mgr is None:
            default_rate = self.mortality_rates.get("default", 0.0)
            if default_rate > 0:
                rolls = rng.random(mask.sum())
                die_local = rolls < default_rate
                die_indices = np.where(mask)[0][die_local]
                population.alive[die_indices] = False
            return
        trait_vals = trait_mgr.get(self.trait_name)
        defn = trait_mgr.definitions[self.trait_name]
        alive_idx = np.where(mask)[0]
        mort_prob = np.zeros(len(alive_idx), dtype=np.float64)
        for cat_name, rate in self.mortality_rates.items():
            cat_idx = defn.categories.index(cat_name)
            cat_mask = trait_vals[alive_idx] == cat_idx
            mort_prob[cat_mask] = rate
        if self.density_dependent:
            mesh = landscape.get("mesh")
            if mesh is not None:
                positions = population.tri_idx[alive_idx]
                n_cells = mesh.n_cells if hasattr(mesh, 'n_cells') else mesh.n_triangles
                cell_counts = np.bincount(positions, minlength=n_cells)
                local_density = cell_counts[positions].astype(np.float64)
                density_factor = 1.0 + self.density_scale * np.maximum(local_density - 1.0, 0.0)
                mort_prob = np.minimum(mort_prob * density_factor, 1.0)
        rolls = rng.random(len(alive_idx))
        die_local = rolls < mort_prob
        die_indices = alive_idx[die_local]
        population.alive[die_indices] = False


@register_event("introduction")
@dataclass
class IntroductionEvent(Event):
    """Add new individuals to the population."""
    n_agents: int = 10
    positions: list[int] = field(default_factory=lambda: [0])
    initial_mass_mean: float = 3500.0
    initial_mass_std: float = 500.0
    initial_ed: float = 6.5
    initial_traits: dict[str, str] = field(default_factory=dict)
    initial_accumulators: dict[str, float] = field(default_factory=dict)

    def execute(self, population, landscape, t, mask):
        rng = landscape.get("rng", np.random.default_rng())
        n = self.n_agents
        pos_arr = np.array(self.positions, dtype=int)
        if len(pos_arr) < n:
            pos_arr = np.tile(pos_arr, (n // len(pos_arr)) + 1)[:n]
        mass = np.clip(
            rng.normal(self.initial_mass_mean, self.initial_mass_std, n),
            self.initial_mass_mean * 0.5, self.initial_mass_mean * 1.5)
        new_idx = population.add_agents(n, pos_arr, mass_g=mass, ed_kJ_g=self.initial_ed)
        if population.trait_mgr is not None:
            for trait_name, cat_name in self.initial_traits.items():
                defn = population.trait_mgr.definitions[trait_name]
                cat_idx = defn.categories.index(cat_name)
                population.trait_mgr._data[trait_name][new_idx] = cat_idx
        if population.accumulator_mgr is not None:
            for acc_name, value in self.initial_accumulators.items():
                idx = population.accumulator_mgr.index_of(acc_name)
                population.accumulator_mgr.data[new_idx, idx] = value


@register_event("reproduction")
@dataclass
class ReproductionEvent(Event):
    """Group-based reproduction with Poisson clutch sizes."""
    clutch_mean: float = 4.0
    offspring_trait_name: str = "stage"
    offspring_trait_value: str = "juvenile"
    min_group_size: int = 1
    offspring_mass_mean: float = 100.0
    offspring_mass_std: float = 20.0
    offspring_ed: float = 6.5

    def execute(self, population, landscape, t, mask):
        rng = landscape.get("rng", np.random.default_rng())
        can_reproduce = mask & (population.group_id >= 0)
        if self.min_group_size > 1:
            group_ids = population.group_id[can_reproduce]
            unique_groups, counts = np.unique(group_ids[group_ids >= 0], return_counts=True)
            valid_groups = set(unique_groups[counts >= self.min_group_size])
            can_reproduce = can_reproduce & np.isin(population.group_id, list(valid_groups))
        reproducer_idx = np.where(can_reproduce)[0]
        if len(reproducer_idx) == 0:
            return
        clutch_sizes = rng.poisson(self.clutch_mean, size=len(reproducer_idx))
        total_offspring = clutch_sizes.sum()
        if total_offspring == 0:
            return
        parent_positions = population.tri_idx[reproducer_idx]
        offspring_positions = np.repeat(parent_positions, clutch_sizes)
        offspring_mass = np.clip(
            rng.normal(self.offspring_mass_mean, self.offspring_mass_std, total_offspring),
            self.offspring_mass_mean * 0.5, self.offspring_mass_mean * 1.5)
        new_idx = population.add_agents(
            total_offspring, offspring_positions,
            mass_g=offspring_mass, ed_kJ_g=self.offspring_ed)
        if population.trait_mgr is not None and self.offspring_trait_name:
            defn = population.trait_mgr.definitions[self.offspring_trait_name]
            cat_idx = defn.categories.index(self.offspring_trait_value)
            population.trait_mgr._data[self.offspring_trait_name][new_idx] = cat_idx

        # Genetic recombination: offspring inherit from parents
        if population.genome is not None:
            parent1_idx = np.repeat(reproducer_idx, clutch_sizes)
            # Simple model: self-fertilization (same parent for both gametes)
            # For sexual reproduction, pair selection should be added later
            parent2_idx = parent1_idx.copy()
            population.genome.recombine(parent1_idx, parent2_idx, new_idx)


@register_event("floater_creation")
@dataclass
class FloaterCreationEvent(Event):
    """Release agents from groups to become floaters."""
    probability: float = 0.1

    def execute(self, population, landscape, t, mask):
        rng = landscape.get("rng", np.random.default_rng())
        candidates = mask & (population.group_id >= 0)
        cand_idx = np.where(candidates)[0]
        if len(cand_idx) == 0:
            return
        rolls = rng.random(len(cand_idx))
        release = rolls < self.probability
        release_idx = cand_idx[release]
        population.group_id[release_idx] = -1


@register_event("census")
@dataclass
class CensusEvent(Event):
    """Record population counts stratified by trait combinations.

    Results are stored in landscape["census_records"] as a list of dicts.
    Each dict has: time, trait counts, total alive, total dead.
    """
    trait_names: list[str] = field(default_factory=list)

    def execute(self, population, landscape, t, mask):
        record = {
            "time": t,
            "n_alive": int(population.alive.sum()),
            "n_dead": int((~population.alive).sum()),
            "n_total": population.n,
        }

        # Count by trait if trait manager exists
        trait_mgr = getattr(population, 'trait_mgr', None) or getattr(population, 'traits', None)
        if trait_mgr is not None and self.trait_names:
            for trait_name in self.trait_names:
                if trait_name in trait_mgr.definitions:
                    defn = trait_mgr.definitions[trait_name]
                    vals = trait_mgr.get(trait_name)
                    alive = population.alive
                    counts = {}
                    for i, cat_name in enumerate(defn.categories):
                        counts[cat_name] = int(((vals == i) & alive).sum())
                    record[f"trait_{trait_name}"] = counts

        # Append to census records in landscape
        if "census_records" not in landscape:
            landscape["census_records"] = []
        landscape["census_records"].append(record)


import tempfile
from pathlib import Path


@register_event("log_snapshot")
@dataclass
class LogSnapshotEvent(Event):
    """Save a binary snapshot of agent state each timestep.

    Saves to landscape["log_dir"]/<prefix>_t<timestep>.npz
    containing: tri_idx, alive, behavior, ed_kJ_g, mass_g arrays.
    """
    prefix: str = "snapshot"

    def execute(self, population, landscape, t, mask):
        log_dir = landscape.get("log_dir")
        if log_dir is None:
            return
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{self.prefix}_t{t:06d}.npz"
        np.savez_compressed(
            path,
            tri_idx=population.tri_idx,
            alive=population.alive,
            behavior=population.behavior,
            ed_kJ_g=population.ed_kJ_g,
            mass_g=population.mass_g,
        )


@register_event("summary_report")
@dataclass
class SummaryReportEvent(Event):
    """Compute and store per-timestep summary statistics.

    Appends to landscape["summary_reports"] a dict with:
    time, n_alive, n_dead, births (if tracked), mean_ed, mean_mass.
    """

    def execute(self, population, landscape, t, mask):
        alive = population.alive
        record = {
            "time": t,
            "n_alive": int(alive.sum()),
            "n_dead": int((~alive).sum()),
            "n_total": population.n,
            "mean_ed": float(population.ed_kJ_g[alive].mean()) if alive.any() else 0.0,
            "mean_mass": float(population.mass_g[alive].mean()) if alive.any() else 0.0,
        }
        if "summary_reports" not in landscape:
            landscape["summary_reports"] = []
        landscape["summary_reports"].append(record)
