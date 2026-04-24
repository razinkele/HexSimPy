"""Run the given config; record key outputs for comparison.

Used twice in the Curonian realism plan:
  - Phase 0: baseline of config_curonian_minimal.yaml (stub)
  - Phase 5.2: post-upgrade on config_curonian_baltic.yaml (realistic)

Both invocations use the same script via --config, so outputs are
directly comparable.
"""
import argparse

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


def run(config_path: str, n_agents: int = 500, n_steps: int = 720):
    cfg = load_config(config_path)
    sim = Simulation(cfg, n_agents=n_agents, data_dir="data", rng_seed=42)
    sim.run(n_steps=n_steps)
    alive_mask = sim.pool.alive
    alive = int(alive_mask.sum())
    arrived = int(sim.pool.arrived.sum())
    mean_ed = float(sim.pool.ed_kJ_g[alive_mask].mean()) if alive else 0.0
    mean_mass = float(sim.pool.mass_g[alive_mask].mean()) if alive else 0.0
    result = {
        "config": config_path,
        "n_agents": n_agents,
        "n_steps": n_steps,
        "alive": alive,
        "arrived": arrived,
        "mean_ed_kJ_g": mean_ed,
        "mean_mass_g": mean_mass,
        "temp_range": (
            float(sim.env.fields["temperature"].min()),
            float(sim.env.fields["temperature"].max()),
        ),
    }
    if "salinity" in sim.env.fields:
        result["salinity_range"] = (
            float(sim.env.fields["salinity"].min()),
            float(sim.env.fields["salinity"].max()),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_curonian_minimal.yaml")
    parser.add_argument("--n-agents", type=int, default=500)
    parser.add_argument("--n-steps", type=int, default=720)
    args = parser.parse_args()
    for k, v in run(args.config, args.n_agents, args.n_steps).items():
        print(f"{k}: {v}")
