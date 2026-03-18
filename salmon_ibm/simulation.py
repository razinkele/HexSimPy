"""Main simulation loop."""
from __future__ import annotations

import numpy as np

from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.behavior import BehaviorParams, pick_behaviors, apply_overrides
from salmon_ibm.population import Population
from salmon_ibm.bioenergetics import BioParams, update_energy
from salmon_ibm.config import load_config, bio_params_from_config, behavior_params_from_config
from salmon_ibm.environment import Environment
from salmon_ibm.estuary import salinity_cost, do_override, seiche_pause, DO_ESCAPE, DO_LETHAL
from salmon_ibm.events import EventSequencer
from salmon_ibm.events_builtin import MovementEvent, CustomEvent
from salmon_ibm.mesh import TriMesh
from salmon_ibm.movement import execute_movement
from salmon_ibm.output import OutputLogger


class Simulation:
    def __init__(self, config, n_agents=100, data_dir="data", rng_seed=None, output_path=None):
        self.config = config
        self.rng_seed = rng_seed
        self.current_t = 0

        grid_type = config.get("grid", {}).get("type", "netcdf")

        if grid_type == "hexsim":
            from salmon_ibm.hexsim import HexMesh
            from salmon_ibm.hexsim_env import HexSimEnvironment

            hs = config["hexsim"]
            self.mesh = HexMesh.from_hexsim(
                hs["workspace"], species=hs.get("species", "chinook"),
                extent_layer=hs.get("extent_layer"),
                depth_layer=hs.get("depth_layer"),
            )
            self.env = HexSimEnvironment(
                hs["workspace"], self.mesh,
                temperature_csv=hs.get("temperature_csv", "River Temperature.csv"),
            )
        else:
            grid_file = f"{data_dir}/{config['grid']['file']}"
            self.mesh = TriMesh.from_netcdf(grid_file)
            self.env = Environment(config, self.mesh, data_dir=data_dir)

        water_ids = np.where(self.mesh.water_mask)[0]
        base_rng = np.random.default_rng(rng_seed)
        rng = np.random.default_rng(base_rng.integers(2**63))
        self._rng = np.random.default_rng(base_rng.integers(2**63))
        start_tris = rng.choice(water_ids, size=n_agents)
        self.pool = AgentPool(n=n_agents, start_tri=start_tris, rng_seed=rng_seed)
        self.population = Population(name="salmon", pool=self.pool)

        # Load barriers if configured
        barrier_cfg = self.config.get("barriers")
        self._barrier_arrays = None
        if barrier_cfg and grid_type == "hexsim":
            from salmon_ibm.barriers import BarrierMap
            hbf_path = barrier_cfg.get("file")
            if hbf_path:
                bmap = BarrierMap.from_hbf(hbf_path, self.mesh)
                if bmap.has_barriers():
                    self._barrier_arrays = bmap.to_arrays(self.mesh)

        self.beh_params = behavior_params_from_config(config)
        self.bio_params = bio_params_from_config(config)
        self._activity_lut = self._build_activity_lut()
        self.est_cfg = config.get("estuary", {})

        self.logger = None
        if output_path:
            self.logger = OutputLogger(output_path, self.mesh.centroids)

        self.history = []
        self._sequencer = EventSequencer(self._build_events())

    # ------------------------------------------------------------------
    # Event sequencer setup
    # ------------------------------------------------------------------

    def _build_events(self):
        """Build event sequence from config or use default salmon sequence."""
        event_defs = self.config.get("events")
        if event_defs is not None:
            from salmon_ibm.events import load_events_from_config
            return load_events_from_config(event_defs, self._build_callback_registry())

        # Default: hardcoded salmon migration sequence
        return [
            CustomEvent(name="push_temperature", callback=self._event_push_temperature),
            CustomEvent(name="behavior_selection", callback=self._event_behavior_selection),
            CustomEvent(name="estuarine_overrides", callback=self._event_estuarine_overrides),
            CustomEvent(name="update_cwr_counters", callback=self._event_update_cwr_counters),
            MovementEvent(name="movement", n_micro_steps=3, cwr_threshold=self.beh_params.temp_bins[0]),
            CustomEvent(name="update_timers", callback=self._event_update_timers),
            CustomEvent(name="bioenergetics", callback=self._event_bioenergetics),
            CustomEvent(name="logging", callback=self._event_logging),
        ]

    def _build_callback_registry(self):
        """Map custom event names to their callback methods."""
        return {
            "push_temperature": self._event_push_temperature,
            "behavior_selection": self._event_behavior_selection,
            "estuarine_overrides": self._event_estuarine_overrides,
            "update_cwr_counters": self._event_update_cwr_counters,
            "update_timers": self._event_update_timers,
            "bioenergetics": self._event_bioenergetics,
            "logging": self._event_logging,
        }

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def _event_push_temperature(self, population, landscape, t, mask):
        temps = landscape["fields"]["temperature"][population.tri_idx]
        population.push_temperature(temps)

    def _event_behavior_selection(self, population, landscape, t, mask):
        step_mask = landscape["step_alive_mask"]
        t3h = population.t3h_mean()
        population.behavior[step_mask] = pick_behaviors(
            t3h[step_mask], population.target_spawn_hour[step_mask],
            self.beh_params, seed=int(self._rng.integers(2**31)),
        )
        population.behavior = apply_overrides(population, self.beh_params)

    def _event_estuarine_overrides(self, population, landscape, t, mask):
        self._apply_estuarine_overrides()

    def _event_update_cwr_counters(self, population, landscape, t, mask):
        self._update_cwr_counters()

    def _event_update_timers(self, population, landscape, t, mask):
        step_mask = landscape["step_alive_mask"]
        population.steps[step_mask] += 1
        population.target_spawn_hour[step_mask] = np.maximum(
            population.target_spawn_hour[step_mask] - 1, 0
        )

    def _event_bioenergetics(self, population, landscape, t, mask):
        fields = landscape["fields"]
        temps_at_agents = fields["temperature"][population.tri_idx]
        activity = self._activity_lut[population.behavior]
        sal = fields.get("salinity", np.zeros(self.mesh.n_triangles))
        sal_at_agents = sal[population.tri_idx]
        s_cfg = self.est_cfg.get("salinity_cost", {})
        sal_cost = salinity_cost(
            sal_at_agents,
            S_opt=s_cfg.get("S_opt", 0.5),
            S_tol=s_cfg.get("S_tol", 6.0),
            k=s_cfg.get("k", 0.6),
        )
        alive = population.alive & ~population.arrived
        if alive.any():
            new_ed, dead, new_mass = update_energy(
                population.ed_kJ_g[alive], population.mass_g[alive],
                temps_at_agents[alive], activity[alive], sal_cost[alive],
                self.bio_params,
            )
            population.ed_kJ_g[alive] = new_ed
            population.mass_g[alive] = new_mass
            dead_indices = np.where(alive)[0][dead]
            population.alive[dead_indices] = False
        thermal_kill = alive & (temps_at_agents >= self.bio_params.T_MAX)
        population.alive[thermal_kill] = False

    def _event_logging(self, population, landscape, t, mask):
        if self.logger:
            self.logger.log_step(t, population)
        summary = {
            "time": t,
            "n_alive": int(population.alive.sum()),
            "n_arrived": int(population.arrived.sum()),
            "mean_ed": float(population.ed_kJ_g[population.alive].mean()) if population.alive.any() else 0.0,
            "mean_mass": float(population.mass_g[population.alive].mean()) if population.alive.any() else 0.0,
            "behavior_counts": {
                int(b): int((population.behavior[population.alive] == b).sum()) for b in range(5)
            },
        }
        self.history.append(summary)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self):
        t = self.current_t
        self.env.advance(t)
        landscape = {
            "mesh": self.mesh,
            "fields": self.env.fields,
            "rng": self._rng,
            "activity_lut": self._activity_lut,
            "est_cfg": self.est_cfg,
            "barrier_arrays": self._barrier_arrays,
        }
        self._sequencer.step(self.population, landscape, t)
        self.current_t += 1

    def _build_activity_lut(self):
        """Build vectorized activity multiplier lookup table."""
        max_beh = max(self.bio_params.activity_by_behavior.keys())
        lut = np.ones(max_beh + 1)
        for k, v in self.bio_params.activity_by_behavior.items():
            lut[k] = v
        return lut

    def _update_cwr_counters(self):
        """Update CWR tracking counters based on current behavior state."""
        in_cwr = self.pool.behavior == Behavior.TO_CWR
        not_in_cwr = ~in_cwr

        # Fish in CWR: increment cwr_hours, reset hours_since_cwr
        self.pool.cwr_hours[in_cwr] += 1
        self.pool.hours_since_cwr[in_cwr] = 0

        # Fish not in CWR: reset cwr_hours if they were in CWR, increment hours_since_cwr
        was_in_cwr = not_in_cwr & (self.pool.cwr_hours > 0)
        self.pool.cwr_hours[was_in_cwr] = 0
        self.pool.hours_since_cwr[not_in_cwr] += 1

    def _apply_estuarine_overrides(self):
        alive = self.pool.alive

        # Seiche pause
        seiche_cfg = self.est_cfg.get("seiche_pause", {})
        thresh = seiche_cfg.get("dSSHdt_thresh_m_per_hour")
        if thresh is None:
            # Convert per-15-min threshold to per-hour (dSSH_dt_array returns m/hour)
            thresh = seiche_cfg.get("dSSHdt_thresh_m_per_15min", 0.02) * 4.0
        dSSH_all = self.env.dSSH_dt_array()
        dSSH = dSSH_all[self.pool.tri_idx] if len(dSSH_all) > 0 else np.zeros(self.pool.n)
        paused = seiche_pause(dSSH, thresh=thresh)
        self.pool.behavior[paused & alive] = Behavior.HOLD

        # Dissolved oxygen avoidance
        do_cfg = self.est_cfg.get("do_avoidance", {})
        do_lethal = do_cfg.get("lethal", 0.0)
        do_high = do_cfg.get("high", 0.0)
        do_field = self.env.fields.get("do")
        if do_field is not None and (do_lethal > 0 or do_high > 0):
            do_at_agents = do_field[self.pool.tri_idx]
            do_state = do_override(do_at_agents, lethal=do_lethal, high=do_high)
            self.pool.behavior[(do_state == DO_ESCAPE) & alive] = Behavior.DOWNSTREAM
            self.pool.alive[(do_state == DO_LETHAL) & alive] = False

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()

    def close(self):
        if self.logger:
            self.logger.close()
        self.env.close()
