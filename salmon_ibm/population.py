"""Population: unified lifecycle manager for a named agent collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from salmon_ibm.agents import AgentPool
from salmon_ibm.accumulators import AccumulatorManager
from salmon_ibm.traits import TraitManager
from salmon_ibm.origin import ORIGIN_WILD


@dataclass
class Population:
    """Wraps AgentPool + AccumulatorManager + TraitManager + group tracking."""

    name: str
    pool: AgentPool
    accumulator_mgr: AccumulatorManager | None = None
    trait_mgr: TraitManager | None = None
    genome: Any = None  # GenomeManager | None (Phase 3)
    ranges: Any = None  # RangeAllocator | None (Phase 4)

    group_id: np.ndarray = field(init=False)
    _next_id: int = field(init=False, default=0)
    agent_ids: np.ndarray = field(init=False)

    def __post_init__(self):
        n = self.pool.n
        self.group_id = np.full(n, -1, dtype=np.int32)
        self.agent_ids = np.arange(n, dtype=np.int64)
        self._next_id = n
        self.affinity_targets = np.full(
            n, -1, dtype=np.intp
        )  # target cell for affinity
        self.spatial_affinity = np.zeros(n, dtype=np.float64)  # affinity strength

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
    def natal_reach_id(self) -> np.ndarray:
        return self.pool.natal_reach_id

    @natal_reach_id.setter
    def natal_reach_id(self, v):
        self.pool.natal_reach_id = v

    @property
    def exit_branch_id(self) -> np.ndarray:
        return self.pool.exit_branch_id

    @exit_branch_id.setter
    def exit_branch_id(self, v):
        self.pool.exit_branch_id = v

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

    def push_temperature(self, temps, alive_mask=None):
        self.pool.push_temperature(temps, alive_mask=alive_mask)

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
        for attr in AgentPool.ARRAY_FIELDS:
            arr = getattr(self.pool, attr)
            setattr(self.pool, attr, arr[alive_idx].copy())
        self.pool.n = n_new
        self.group_id = self.group_id[alive_idx].copy()
        self.agent_ids = self.agent_ids[alive_idx].copy()
        self.affinity_targets = self.affinity_targets[alive_idx].copy()
        self.spatial_affinity = self.spatial_affinity[alive_idx].copy()
        if self.accumulator_mgr is not None:
            self.accumulator_mgr.data = self.accumulator_mgr.data[:, alive_idx].copy()
            self.accumulator_mgr.n_agents = n_new
        if self.trait_mgr is not None:
            for name in self.trait_mgr._data:
                self.trait_mgr._data[name] = self.trait_mgr._data[name][
                    alive_idx
                ].copy()
            self.trait_mgr.n_agents = n_new
        if self.genome is not None:
            self.genome.genotypes = self.genome.genotypes[alive_idx].copy()
            self.genome.n_agents = n_new

    def add_agents(
        self,
        n: int,
        positions: np.ndarray,
        *,
        mass_g=None,
        ed_kJ_g: float = 6.5,
        group_id: int = -1,
        origin: int = ORIGIN_WILD,
    ) -> np.ndarray:
        old_n = self.pool.n
        new_n = old_n + n

        # --- Bulk pre-allocate all AgentPool arrays at once ---
        # Uses np.empty + slice assignment instead of 11 np.concatenate calls.
        # Avoids creating temporary arrays; ~2x faster due to fewer allocations.
        new_arrays = {}
        for attr in AgentPool.ARRAY_FIELDS:
            old = getattr(self.pool, attr)
            if old.ndim == 1:
                new_arr = np.empty(new_n, dtype=old.dtype)
                new_arr[:old_n] = old
            else:
                new_arr = np.empty((new_n,) + old.shape[1:], dtype=old.dtype)
                new_arr[:old_n] = old
            new_arrays[attr] = new_arr

        # Fill new agent slots with defaults
        new_arrays["tri_idx"][old_n:] = positions
        new_arrays["mass_g"][old_n:] = mass_g if mass_g is not None else 3500.0
        new_arrays["ed_kJ_g"][old_n:] = ed_kJ_g
        new_arrays["target_spawn_hour"][old_n:] = 720
        new_arrays["behavior"][old_n:] = 0
        new_arrays["cwr_hours"][old_n:] = 0
        new_arrays["hours_since_cwr"][old_n:] = 999
        new_arrays["steps"][old_n:] = 0
        new_arrays["alive"][old_n:] = True
        new_arrays["arrived"][old_n:] = False
        new_arrays["temp_history"][old_n:] = 15.0
        new_arrays["natal_reach_id"][old_n:] = -1
        new_arrays["exit_branch_id"][old_n:] = -1
        new_arrays["origin"][old_n:] = origin

        # Assign all AgentPool arrays at once
        for attr, arr in new_arrays.items():
            setattr(self.pool, attr, arr)
        self.pool.n = new_n

        # Verify all AgentPool fields were extended
        assert self.pool.n == new_n
        for attr in AgentPool.ARRAY_FIELDS:
            arr = getattr(self.pool, attr)
            assert len(arr) == new_n, (
                f"AgentPool.{attr} has length {len(arr)}, expected {new_n}. "
                f"Did you forget to update add_agents() for a new field?"
            )

        # --- Population-level arrays (same bulk pattern) ---
        def _extend_1d(old_arr, fill_value, dtype):
            new_arr = np.empty(new_n, dtype=dtype)
            new_arr[:old_n] = old_arr
            new_arr[old_n:] = fill_value
            return new_arr

        self.group_id = _extend_1d(self.group_id, group_id, np.int32)
        self.affinity_targets = _extend_1d(self.affinity_targets, -1, np.intp)
        self.spatial_affinity = _extend_1d(self.spatial_affinity, 0.0, np.float64)

        new_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        ids_arr = np.empty(new_n, dtype=np.int64)
        ids_arr[:old_n] = self.agent_ids
        ids_arr[old_n:] = new_ids
        self.agent_ids = ids_arr
        self._next_id += n

        # --- Optional managers ---
        if self.accumulator_mgr is not None:
            n_acc = self.accumulator_mgr.data.shape[0]
            new_data = np.empty((n_acc, new_n), dtype=np.float64)
            new_data[:, :old_n] = self.accumulator_mgr.data
            new_data[:, old_n:] = 0.0
            self.accumulator_mgr.data = new_data
            self.accumulator_mgr.n_agents = new_n
        if self.trait_mgr is not None:
            for name in self.trait_mgr._data:
                old_arr = self.trait_mgr._data[name]
                new_arr = np.empty(new_n, dtype=old_arr.dtype)
                new_arr[:old_n] = old_arr
                new_arr[old_n:] = 0
                self.trait_mgr._data[name] = new_arr
            self.trait_mgr.n_agents = new_n
        if self.genome is not None:
            n_loci = self.genome.n_loci
            new_geno = np.empty((new_n, n_loci, 2), dtype=np.int32)
            new_geno[:old_n] = self.genome.genotypes
            new_geno[old_n:] = 0
            self.genome.genotypes = new_geno
            self.genome.n_agents = new_n
        return np.arange(old_n, new_n)

    def set_natal_reach_from_cells(self, new_idx, mesh) -> None:
        """Tag new agents' natal_reach_id by looking up the mesh reach_id at
        their current cell. Called by every add_agents call site.

        Truth-checks reach_names (a list) rather than reach_id (an ndarray)
        to avoid `if not arr` raising on multi-element numpy arrays. Same
        defensive pattern as delta_routing.update_exit_branch_id.

        Off-mesh agents (tri_idx < 0) are skipped — without the guard,
        `mesh.reach_id[-1]` would silently return the last cell's reach_id
        and tag the agent with a wrong reach.
        """
        if not getattr(mesh, "reach_names", None):
            return
        new_idx = np.asarray(new_idx)
        tri = self.pool.tri_idx[new_idx]
        valid = tri >= 0
        if valid.any():
            self.pool.natal_reach_id[new_idx[valid]] = mesh.reach_id[tri[valid]]

    def assert_natal_tagged(self) -> None:
        """Fail loudly if any alive on-mesh agent lacks natal_reach_id tagging.

        Called once per simulation step from Simulation.step() before logging,
        so any failure surfaces in the same step that introduced the
        un-tagged agents. Suppressed under Simulation(resume=True).
        """
        pool = self.pool
        on_mesh = pool.alive & (pool.tri_idx >= 0)
        untagged = pool.natal_reach_id == -1
        bad = on_mesh & untagged
        assert not bad.any(), (
            "Agents introduced without natal_reach_id tagging — every "
            "code path calling add_agents() must follow up with "
            "set_natal_reach_from_cells() or equivalent. "
            f"{int(bad.sum())} agents affected."
        )
