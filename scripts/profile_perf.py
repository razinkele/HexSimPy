"""Profile simulation performance at different agent counts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import cProfile
import pstats
import io
import numpy as np

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


def profile_scaling():
    """Measure wall time per step at different agent counts."""
    print("=== Scaling Profile (Columbia River HexSim workspace) ===\n")

    # Warmup Numba JIT
    print("Warming up Numba JIT...")
    cfg = load_config("config_columbia.yaml")
    sim = Simulation(cfg, n_agents=10, rng_seed=99)
    sim.run(2)
    print("  Done.\n")

    print(f"{'Agents':>8} | {'s/step':>8} | {'Alive':>6} | {'Notes'}")
    print("-" * 50)

    for n_agents in [100, 1000, 5000, 10000, 50000]:
        cfg = load_config("config_columbia.yaml")
        try:
            sim = Simulation(cfg, n_agents=n_agents, rng_seed=42)
        except Exception as e:
            print(f"{n_agents:>8} | FAILED   |        | {e}")
            continue

        n_steps = 10
        t0 = time.perf_counter()
        sim.run(n_steps)
        elapsed = time.perf_counter() - t0
        alive = int(sim.pool.alive.sum())
        per_step = elapsed / n_steps
        print(f"{n_agents:>8} | {per_step:>8.4f} | {alive:>6} | {elapsed:.2f}s total")


def profile_cprofile():
    """Detailed cProfile of a 1000-agent, 20-step run."""
    print("\n=== cProfile (1000 agents, 20 steps) ===\n")

    cfg = load_config("config_columbia.yaml")
    sim = Simulation(cfg, n_agents=1000, rng_seed=42)

    pr = cProfile.Profile()
    pr.enable()
    sim.run(20)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


def profile_step_breakdown():
    """Time each phase of a single step."""
    print("\n=== Step Breakdown (5000 agents, 1 step) ===\n")

    cfg = load_config("config_columbia.yaml")
    sim = Simulation(cfg, n_agents=5000, rng_seed=42)
    # Run 1 warmup step
    sim.run(1)

    # Now time individual phases of step 2
    t = sim.current_t
    sim.env.advance(t)

    landscape = {
        "mesh": sim.mesh,
        "fields": sim.env.fields,
        "rng": sim._rng,
        "activity_lut": sim._activity_lut,
        "est_cfg": sim.est_cfg,
        "barrier_arrays": sim._barrier_arrays,
    }
    landscape["step_alive_mask"] = sim.population.alive & ~sim.population.arrived

    timings = {}
    for event in sim._sequencer.events:
        mask = sim._sequencer._compute_mask(sim.population, event.trait_filter)
        t0 = time.perf_counter()
        event.execute(sim.population, landscape, t, mask)
        timings[event.name] = time.perf_counter() - t0

    total = sum(timings.values())
    print(f"{'Event':>25} | {'Time (ms)':>10} | {'%':>6}")
    print("-" * 50)
    for name, dt in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"{name:>25} | {dt*1000:>10.3f} | {dt/total*100:>5.1f}%")
    print(f"{'TOTAL':>25} | {total*1000:>10.3f} |")


if __name__ == "__main__":
    profile_scaling()
    profile_step_breakdown()
    profile_cprofile()
