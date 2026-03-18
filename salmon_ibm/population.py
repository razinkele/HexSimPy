"""Population: unified lifecycle manager for a named agent collection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from salmon_ibm.agents import AgentPool
from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
from salmon_ibm.traits import TraitManager, TraitDefinition


@dataclass
class Population:
    """Wraps AgentPool + AccumulatorManager + TraitManager + group tracking."""
    name: str
    pool: AgentPool
    accumulator_mgr: AccumulatorManager | None = None
    trait_mgr: TraitManager | None = None
    genome: Any = None  # GenomeManager | None (Phase 3)

    group_id: np.ndarray = field(init=False)
    _next_id: int = field(init=False, default=0)
    agent_ids: np.ndarray = field(init=False)

    def __post_init__(self):
        n = self.pool.n
        self.group_id = np.full(n, -1, dtype=np.int32)
        self.agent_ids = np.arange(n, dtype=np.int64)
        self._next_id = n

    # --- Core properties ---
    @property
    def n(self) -> int:
        return self.pool.n

    @property
    def n_alive(self) -> int:
        return int(self.pool.alive.sum())

    @property
    def alive(self) -> np.ndarray:
        return self.pool.alive

    @property
    def arrived(self) -> np.ndarray:
        return self.pool.arrived

    @property
    def tri_idx(self) -> np.ndarray:
        return self.pool.tri_idx
    @tri_idx.setter
    def tri_idx(self, v):
        self.pool.tri_idx = v

    # --- Proxies for AgentPool attributes used by event callbacks ---
    @property
    def behavior(self) -> np.ndarray:
        return self.pool.behavior
    @behavior.setter
    def behavior(self, v):
        self.pool.behavior = v

    @property
    def ed_kJ_g(self) -> np.ndarray:
        return self.pool.ed_kJ_g
    @ed_kJ_g.setter
    def ed_kJ_g(self, v):
        self.pool.ed_kJ_g = v

    @property
    def mass_g(self) -> np.ndarray:
        return self.pool.mass_g
    @mass_g.setter
    def mass_g(self, v):
        self.pool.mass_g = v

    @property
    def steps(self) -> np.ndarray:
        return self.pool.steps
    @steps.setter
    def steps(self, v):
        self.pool.steps = v

    @property
    def target_spawn_hour(self) -> np.ndarray:
        return self.pool.target_spawn_hour
    @target_spawn_hour.setter
    def target_spawn_hour(self, v):
        self.pool.target_spawn_hour = v

    @property
    def cwr_hours(self) -> np.ndarray:
        return self.pool.cwr_hours
    @cwr_hours.setter
    def cwr_hours(self, v):
        self.pool.cwr_hours = v

    @property
    def hours_since_cwr(self) -> np.ndarray:
        return self.pool.hours_since_cwr
    @hours_since_cwr.setter
    def hours_since_cwr(self, v):
        self.pool.hours_since_cwr = v

    @property
    def temp_history(self) -> np.ndarray:
        return self.pool.temp_history
    @temp_history.setter
    def temp_history(self, v):
        self.pool.temp_history = v

    def t3h_mean(self) -> np.ndarray:
        return self.pool.t3h_mean()

    def push_temperature(self, temps):
        self.pool.push_temperature(temps)

    @property
    def floaters(self) -> np.ndarray:
        return self.pool.alive & (self.group_id == -1)

    @property
    def grouped(self) -> np.ndarray:
        return self.pool.alive & (self.group_id >= 0)

    # --- Dynamic resizing ---
    def remove_agents(self, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.intp)
        self.pool.alive[indices] = False

    def compact(self) -> None:
        alive_mask = self.pool.alive
        if alive_mask.all():
            return
        alive_idx = np.where(alive_mask)[0]
        n_new = len(alive_idx)
        for attr in ["tri_idx", "mass_g", "ed_kJ_g", "target_spawn_hour",
                     "behavior", "cwr_hours", "hours_since_cwr", "steps",
                     "alive", "arrived", "temp_history"]:
            arr = getattr(self.pool, attr)
            setattr(self.pool, attr, arr[alive_idx].copy())
        self.pool.n = n_new
        self.group_id = self.group_id[alive_idx].copy()
        self.agent_ids = self.agent_ids[alive_idx].copy()
        if self.accumulator_mgr is not None:
            self.accumulator_mgr.data = self.accumulator_mgr.data[alive_idx].copy()
            self.accumulator_mgr.n_agents = n_new
        if self.trait_mgr is not None:
            for name in self.trait_mgr._data:
                self.trait_mgr._data[name] = self.trait_mgr._data[name][alive_idx].copy()
            self.trait_mgr.n_agents = n_new
        if self.genome is not None:
            self.genome.genotypes = self.genome.genotypes[alive_idx].copy()
            self.genome.n_agents = n_new

    def add_agents(self, n: int, positions: np.ndarray, *, mass_g=None,
                   ed_kJ_g: float = 6.5, group_id: int = -1) -> np.ndarray:
        old_n = self.pool.n
        new_n = old_n + n
        self.pool.tri_idx = np.concatenate([self.pool.tri_idx, positions])
        self.pool.mass_g = np.concatenate([
            self.pool.mass_g, mass_g if mass_g is not None else np.full(n, 3500.0)])
        self.pool.ed_kJ_g = np.concatenate([self.pool.ed_kJ_g, np.full(n, ed_kJ_g)])
        self.pool.target_spawn_hour = np.concatenate([
            self.pool.target_spawn_hour, np.full(n, 720, dtype=int)])
        self.pool.behavior = np.concatenate([self.pool.behavior, np.zeros(n, dtype=int)])
        self.pool.cwr_hours = np.concatenate([self.pool.cwr_hours, np.zeros(n, dtype=int)])
        self.pool.hours_since_cwr = np.concatenate([
            self.pool.hours_since_cwr, np.full(n, 999, dtype=int)])
        self.pool.steps = np.concatenate([self.pool.steps, np.zeros(n, dtype=int)])
        self.pool.alive = np.concatenate([self.pool.alive, np.ones(n, dtype=bool)])
        self.pool.arrived = np.concatenate([self.pool.arrived, np.zeros(n, dtype=bool)])
        self.pool.temp_history = np.concatenate([
            self.pool.temp_history, np.full((n, 3), 15.0)])
        self.pool.n = new_n
        self.group_id = np.concatenate([
            self.group_id, np.full(n, group_id, dtype=np.int32)])
        new_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self.agent_ids = np.concatenate([self.agent_ids, new_ids])
        self._next_id += n
        if self.accumulator_mgr is not None:
            n_acc = self.accumulator_mgr.data.shape[1]
            self.accumulator_mgr.data = np.concatenate([
                self.accumulator_mgr.data, np.zeros((n, n_acc), dtype=np.float64)])
            self.accumulator_mgr.n_agents = new_n
        if self.trait_mgr is not None:
            for name in self.trait_mgr._data:
                self.trait_mgr._data[name] = np.concatenate([
                    self.trait_mgr._data[name], np.zeros(n, dtype=np.int32)])
            self.trait_mgr.n_agents = new_n
        if self.genome is not None:
            n_loci = self.genome.n_loci
            self.genome.genotypes = np.concatenate([
                self.genome.genotypes,
                np.zeros((n, n_loci, 2), dtype=np.int32)])
            self.genome.n_agents = new_n
        return np.arange(old_n, new_n)
