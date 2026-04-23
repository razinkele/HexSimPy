"""Profile a simulation step after the 2026-04 perf wins.

Warms JIT, then profiles 50 Columbia hex-grid steps with 2000 agents —
the same shape as bench_e2e.py. Output: top 30 cumulative-time callers.
"""
import cProfile
import pstats
import io

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


def run():
    cfg = load_config("config_columbia.yaml")
    sim = Simulation(cfg, n_agents=2000, data_dir="data", rng_seed=42)
    # Warm JIT
    sim.run(n_steps=5)

    pr = cProfile.Profile()
    pr.enable()
    sim.run(n_steps=50)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    run()
