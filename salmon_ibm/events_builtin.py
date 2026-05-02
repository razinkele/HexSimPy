"""Built-in event types that wrap existing salmon IBM logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from salmon_ibm.events import Event, register_event
from salmon_ibm.movement import execute_movement
from salmon_ibm.bioenergetics import BioParams, update_energy, origin_aware_activity_mult
from salmon_ibm.estuary import salinity_cost, EstuaryParams
from salmon_ibm.origin import ORIGIN_WILD


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
        n_micro = landscape.get("n_micro_steps_per_cell")
        if n_micro is None:
            # Legacy path: scalar broadcast to per-cell so the kernel
            # signature stays uniform.  ``getattr`` keeps mock-mesh
            # tests (object() stand-ins) working — execute_movement
            # itself only inspects n_micro.max().
            n_cells_legacy = getattr(mesh, "n_triangles", 1)
            n_micro = np.full(
                n_cells_legacy, self.n_micro_steps, dtype=np.int32,
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


@register_event("survival")
@dataclass
class SurvivalEvent(Event):
    """Bioenergetics energy update + thermal/starvation mortality."""

    # NOTE: bio_params.activity_by_behavior on this event is dead weight
    # — activity dispatch goes through landscape["activity_lut"] and
    # landscape["hatchery_dispatch"]. Local activity_by_behavior values
    # here are silently ignored. bio_params is still used for
    # T_ACUTE_LETHAL / T_MAX (line 122). See C2 spec.
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
            # Was: activity = activity_lut[population.behavior]
            hd = landscape.get("hatchery_dispatch")
            hatch_lut = hd.activity_lut if hd is not None else None
            activity = origin_aware_activity_mult(
                population.behavior,
                population.origin,
                activity_lut,
                hatch_lut,
            )
            sal = fields.get("salinity")
            if sal is None:
                sal = np.zeros(len(fields["temperature"]))
                if not getattr(self, "_salinity_warned", False):
                    import warnings

                    warnings.warn(
                        "No 'salinity' field in environment — using zeros (no salinity cost).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._salinity_warned = True
            sal_at_agents = sal[population.tri_idx]
            # Build EstuaryParams from the salinity_cost YAML subsection.
            # Filter to known fields so legacy keys (S_opt, S_tol, k) are
            # silently dropped — falls back to dataclass defaults rather
            # than raising TypeError. Per-call construction is cheap
            # (microseconds) and SurvivalEvent is a dataclass that can't
            # easily stash this in __init__ (est_cfg is in landscape, not
            # self). See plan 2026-04-30-osmoregulation-stress for context.
            _known_salinity_keys = {
                "salinity_iso_osmotic",
                "salinity_hyper_cost",
                "salinity_hypo_cost",
            }
            s_cfg = est_cfg.get("salinity_cost", {})
            est_params = EstuaryParams(
                **{k: v for k, v in s_cfg.items() if k in _known_salinity_keys}
            )
            sal_cost_arr = salinity_cost(sal_at_agents, est_params)
            new_ed, dead, new_mass = update_energy(
                population.ed_kJ_g[alive],
                population.mass_g[alive],
                temps[alive],
                activity[alive],
                sal_cost_arr[alive],
                self.bio_params,
            )
            population.ed_kJ_g[alive] = new_ed
            population.mass_g[alive] = new_mass
            dead_indices = np.where(alive)[0][dead]
            population.alive[dead_indices] = False

        if self.thermal:
            # Recompute alive mask after starvation kills. Prefer T_ACUTE_LETHAL
            # when available (BalticBioParams): T_MAX on Baltic returns T_AVOID
            # (20°C behavioral threshold) — not the acute-lethal gate.
            current_alive = population.alive & ~getattr(
                population, "arrived", np.zeros(population.n, dtype=bool)
            )
            lethal_T = getattr(
                self.bio_params, "T_ACUTE_LETHAL", self.bio_params.T_MAX
            )
            thermal_kill = current_alive & (temps >= lethal_T)
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
        trait_mgr = getattr(population, "trait_mgr", None) or getattr(
            population, "traits", None
        )
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
                n_cells = mesh.n_cells if hasattr(mesh, "n_cells") else mesh.n_triangles
                cell_counts = np.bincount(positions, minlength=n_cells)
                local_density = cell_counts[positions].astype(np.float64)
                density_factor = 1.0 + self.density_scale * np.maximum(
                    local_density - 1.0, 0.0
                )
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
    initialization_spatial_data: str = ""
    origin: int = ORIGIN_WILD

    def execute(self, population, landscape, t, mask):
        from salmon_ibm.origin import ORIGIN_HATCHERY  # local to avoid module-init cost
        if self.origin == ORIGIN_HATCHERY and landscape.get("hatchery_dispatch") is None:
            raise ValueError(
                f"IntroductionEvent '{self.name}' tags new agents as HATCHERY, "
                f"but the simulation has no hatchery_dispatch configured. "
                f"Add a 'hatchery_overrides:' block under "
                f"species.BalticAtlanticSalmon in the species YAML."
            )
        rng = landscape.get("rng", np.random.default_rng())
        n = self.n_agents
        if self.initialization_spatial_data:
            spatial_data = landscape.get("spatial_data", {})
            layer = spatial_data.get(self.initialization_spatial_data)
            if layer is not None:
                nonzero = np.where(layer != 0)[0]
                if len(nonzero) > 0:
                    mesh = landscape.get("mesh")
                    if mesh is not None and hasattr(mesh, "areas"):
                        # Area-weighted: cells get drawn proportional to their
                        # m².  Without this, three-tier H3 meshes over-place
                        # agents in fine-resolution river cells by ~7-50×.
                        weights = mesh.areas[nonzero].astype(np.float64)
                        weights = weights / weights.sum()
                        pos_arr = rng.choice(
                            nonzero, size=n, replace=True, p=weights
                        )
                    else:
                        # Legacy mesh without per-cell areas: uniform-over-cells.
                        pos_arr = rng.choice(nonzero, size=n, replace=True)
                else:
                    pos_arr = np.zeros(n, dtype=int)
            else:
                pos_arr = np.zeros(n, dtype=int)
        else:
            pos_arr = np.array(self.positions, dtype=int)
            if len(pos_arr) < n:
                pos_arr = np.tile(pos_arr, (n // len(pos_arr)) + 1)[:n]
        mass = np.clip(
            rng.normal(self.initial_mass_mean, self.initial_mass_std, n),
            self.initial_mass_mean * 0.5,
            self.initial_mass_mean * 1.5,
        )
        new_idx = population.add_agents(
            n, pos_arr, mass_g=mass, ed_kJ_g=self.initial_ed,
            origin=self.origin,
        )
        # Tag natal_reach_id from current cell — see salmon_ibm/delta_routing.py.
        mesh = landscape.get("mesh")
        if mesh is not None:
            population.set_natal_reach_from_cells(new_idx, mesh)
        if population.trait_mgr is not None:
            for trait_name, cat_name in self.initial_traits.items():
                defn = population.trait_mgr.definitions[trait_name]
                cat_idx = defn.categories.index(cat_name)
                population.trait_mgr._data[trait_name][new_idx] = cat_idx
        if population.accumulator_mgr is not None:
            for acc_name, value in self.initial_accumulators.items():
                idx = population.accumulator_mgr.index_of(acc_name)
                population.accumulator_mgr.data[idx, new_idx] = value


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
            unique_groups, counts = np.unique(
                group_ids[group_ids >= 0], return_counts=True
            )
            valid_groups = set(unique_groups[counts >= self.min_group_size])
            can_reproduce = can_reproduce & np.isin(
                population.group_id, list(valid_groups)
            )
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
            rng.normal(
                self.offspring_mass_mean, self.offspring_mass_std, total_offspring
            ),
            self.offspring_mass_mean * 0.5,
            self.offspring_mass_mean * 1.5,
        )
        # NOTE: origin is intentionally NOT propagated here per C1
        # spec scope-OUT (docs/superpowers/specs/2026-04-30-hatchery-
        # origin-c1-design.md). Offspring of hatchery parents default
        # to ORIGIN_WILD via Population.add_agents's default kwarg.
        # The biological case for inheritance (genetic + epigenetic
        # carryover from hatchery parents) is real but unsettled for
        # Atlantic salmon; deferring to a future tier (post-C3.x)
        # rather than baking in an arbitrary inheritance probability.
        new_idx = population.add_agents(
            total_offspring,
            offspring_positions,
            mass_g=offspring_mass,
            ed_kJ_g=self.offspring_ed,
        )
        mesh = landscape.get("mesh")
        if mesh is not None:
            population.set_natal_reach_from_cells(new_idx, mesh)
        if population.trait_mgr is not None and self.offspring_trait_name:
            defn = population.trait_mgr.definitions[self.offspring_trait_name]
            cat_idx = defn.categories.index(self.offspring_trait_value)
            population.trait_mgr._data[self.offspring_trait_name][new_idx] = cat_idx

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
                        (population.group_id == gid)
                        & population.alive
                        & (np.arange(population.n) != rep_idx)
                    )[0]
                    if len(group_members) > 0:
                        mate = rng.choice(group_members)
                        parent2_indices[offset : offset + cs] = mate
                    else:
                        import warnings

                        warnings.warn(
                            f"Agent {rep_idx} (group {gid}) has no available mates — self-mating.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                offset += cs
            population.genome.recombine(parent1_indices, parent2_indices, new_idx)


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
        trait_mgr = getattr(population, "trait_mgr", None) or getattr(
            population, "traits", None
        )
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
