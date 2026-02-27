"""Command-line interface for running the salmon IBM."""
import argparse
import os

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description="Baltic Salmon IBM")
    parser.add_argument("--config", default="config_curonian_minimal.yaml")
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="output/tracks.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cfg = load_config(args.config)
    sim = Simulation(
        cfg, n_agents=args.agents, data_dir=args.data_dir,
        rng_seed=args.seed, output_path=args.output,
    )

    print(f"Running {args.steps} hourly steps with {args.agents} agents...")
    sim.run(n_steps=args.steps)
    sim.close()

    alive = sim.pool.alive.sum()
    arrived = sim.pool.arrived.sum()
    print(f"Done. Alive: {alive}/{args.agents}, Arrived: {arrived}")
    print(f"Tracks saved to {args.output}")


if __name__ == "__main__":
    main()
