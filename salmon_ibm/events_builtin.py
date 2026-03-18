"""Built-in event types that wrap existing salmon IBM logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from salmon_ibm.events import Event, EveryStep, EventTrigger, register_event
from salmon_ibm.movement import execute_movement
from salmon_ibm.bioenergetics import BioParams, update_energy
from salmon_ibm.estuary import salinity_cost


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
        execute_movement(
            population, mesh, fields,
            seed=int(rng.integers(2**31)),
            n_micro_steps=self.n_micro_steps,
            cwr_threshold=self.cwr_threshold,
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
