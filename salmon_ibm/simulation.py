"""Main simulation loop."""
from __future__ import annotations

import numpy as np

from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.behavior import BehaviorParams, pick_behaviors, apply_overrides
from salmon_ibm.bioenergetics import BioParams, update_energy
from salmon_ibm.config import load_config
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

        grid_file = f"{data_dir}/{config['grid']['file']}"
        self.mesh = TriMesh.from_netcdf(grid_file)
        self.env = Environment(config, self.mesh, data_dir=data_dir)

        water_ids = np.where(self.mesh.water_mask)[0]
        rng = np.random.default_rng(rng_seed)
        start_tris = rng.choice(water_ids, size=n_agents)
        self.pool = AgentPool(n=n_agents, start_tri=start_tris, rng_seed=rng_seed)

        self.beh_params = BehaviorParams.defaults()
        self.bio_params = BioParams()
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
            self.beh_params, seed=None,
        )

        # 2. Apply overrides
        self.pool.behavior = apply_overrides(self.pool, self.beh_params)

        # 3. Estuarine overrides
        self._apply_estuarine_overrides()

        # 4. Movement
        execute_movement(self.pool, self.mesh, self.env.fields, seed=None)

        # 5. Update timers
        self.pool.steps[alive_mask] += 1
        self.pool.target_spawn_hour[alive_mask] = np.maximum(
            self.pool.target_spawn_hour[alive_mask] - 1, 0
        )

        # 6. Bioenergetics
        activity = np.array([
            self.bio_params.activity_by_behavior.get(int(b), 1.0)
            for b in self.pool.behavior
        ])
        sal = self.env.fields.get("salinity", np.zeros(self.mesh.n_triangles))
        sal_at_agents = sal[self.pool.tri_idx]
        s_cfg = self.est_cfg.get("salinity_cost", {})
        sal_cost = salinity_cost(
            sal_at_agents,
            S_opt=s_cfg.get("S_opt", 0.5),
            S_tol=s_cfg.get("S_tol", 6.0),
            k=s_cfg.get("k", 0.6),
        )

        new_ed, dead = update_energy(
            self.pool.ed_kJ_g, self.pool.mass_g,
            temps_at_agents, activity, sal_cost, self.bio_params,
        )
        self.pool.ed_kJ_g = new_ed
        self.pool.alive[dead] = False

        # 7. Logging
        if self.logger:
            self.logger.log_step(t, self.pool)

        summary = {
            "time": t,
            "n_alive": int(self.pool.alive.sum()),
            "n_arrived": int(self.pool.arrived.sum()),
            "mean_ed": float(self.pool.ed_kJ_g[self.pool.alive].mean()) if self.pool.alive.any() else 0.0,
            "behavior_counts": {
                int(b): int((self.pool.behavior[self.pool.alive] == b).sum()) for b in range(5)
            },
        }
        self.history.append(summary)
        self.current_t += 1

    def _apply_estuarine_overrides(self):
        seiche_cfg = self.est_cfg.get("seiche_pause", {})
        thresh = seiche_cfg.get("dSSHdt_thresh_m_per_15min", 0.02)
        dSSH = np.array([self.env.dSSH_dt(int(tri)) for tri in self.pool.tri_idx])
        paused = seiche_pause(dSSH, thresh=thresh)
        self.pool.behavior[paused & self.pool.alive] = Behavior.HOLD

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()

    def close(self):
        if self.logger:
            self.logger.close()
        self.env.close()
