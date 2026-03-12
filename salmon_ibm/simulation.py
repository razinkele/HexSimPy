"""Main simulation loop."""
from __future__ import annotations

import numpy as np

from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.behavior import BehaviorParams, pick_behaviors, apply_overrides
from salmon_ibm.bioenergetics import BioParams, update_energy
from salmon_ibm.config import load_config, bio_params_from_config, behavior_params_from_config
from salmon_ibm.environment import Environment
from salmon_ibm.estuary import salinity_cost, do_override, seiche_pause, DO_ESCAPE, DO_LETHAL
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

        self.beh_params = behavior_params_from_config(config)
        self.bio_params = bio_params_from_config(config)
        self._activity_lut = self._build_activity_lut()
        self.est_cfg = config.get("estuary", {})

        self.logger = None
        if output_path:
            self.logger = OutputLogger(output_path, self.mesh.centroids)

        self.history = []

    def step(self):
        t = self.current_t
        self.env.advance(t)

        alive_mask = self.pool.alive & ~self.pool.arrived

        temps_at_agents = self.env.fields["temperature"][self.pool.tri_idx]
        self.pool.push_temperature(temps_at_agents)

        # 1. Pick behaviors
        t3h = self.pool.t3h_mean()
        self.pool.behavior[alive_mask] = pick_behaviors(
            t3h[alive_mask], self.pool.target_spawn_hour[alive_mask],
            self.beh_params, seed=int(self._rng.integers(2**31)),
        )

        # 2. Apply overrides
        self.pool.behavior = apply_overrides(self.pool, self.beh_params)

        # 3. Estuarine overrides
        self._apply_estuarine_overrides()

        # 4. Update CWR counters (after all behavior decisions are final)
        self._update_cwr_counters()

        # 5. Movement
        execute_movement(self.pool, self.mesh, self.env.fields,
                         seed=int(self._rng.integers(2**31)),
                         cwr_threshold=self.beh_params.temp_bins[0])

        # 5. Update timers
        self.pool.steps[alive_mask] += 1
        self.pool.target_spawn_hour[alive_mask] = np.maximum(
            self.pool.target_spawn_hour[alive_mask] - 1, 0
        )

        # 6. Bioenergetics
        activity = self._activity_lut[self.pool.behavior]
        sal = self.env.fields.get("salinity", np.zeros(self.mesh.n_triangles))
        sal_at_agents = sal[self.pool.tri_idx]
        s_cfg = self.est_cfg.get("salinity_cost", {})
        sal_cost = salinity_cost(
            sal_at_agents,
            S_opt=s_cfg.get("S_opt", 0.5),
            S_tol=s_cfg.get("S_tol", 6.0),
            k=s_cfg.get("k", 0.6),
        )

        # 7. Bioenergetics (alive agents only)
        alive = self.pool.alive & ~self.pool.arrived
        if alive.any():
            new_ed, dead, new_mass = update_energy(
                self.pool.ed_kJ_g[alive], self.pool.mass_g[alive],
                temps_at_agents[alive], activity[alive], sal_cost[alive],
                self.bio_params,
            )
            self.pool.ed_kJ_g[alive] = new_ed
            self.pool.mass_g[alive] = new_mass
            # dead is relative to the alive subset
            dead_indices = np.where(alive)[0][dead]
            self.pool.alive[dead_indices] = False

        # 7b. Thermal mortality (alive agents only)
        thermal_kill = alive & (temps_at_agents >= self.bio_params.T_MAX)
        self.pool.alive[thermal_kill] = False

        # 7. Logging
        if self.logger:
            self.logger.log_step(t, self.pool)

        summary = {
            "time": t,
            "n_alive": int(self.pool.alive.sum()),
            "n_arrived": int(self.pool.arrived.sum()),
            "mean_ed": float(self.pool.ed_kJ_g[self.pool.alive].mean()) if self.pool.alive.any() else 0.0,
            "mean_mass": float(self.pool.mass_g[self.pool.alive].mean()) if self.pool.alive.any() else 0.0,
            "behavior_counts": {
                int(b): int((self.pool.behavior[self.pool.alive] == b).sum()) for b in range(5)
            },
        }
        self.history.append(summary)
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
        thresh = seiche_cfg.get("dSSHdt_thresh_m_per_hour",
                                seiche_cfg.get("dSSHdt_thresh_m_per_15min", 0.02))
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
