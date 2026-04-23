"""End-to-end timing probe: full simulation step cost.

Warms Numba JIT with 5 steps, then times 100 steps. Prints per-step
ms so the current hot-loop cost can be compared against the README's
claim (~1.20s/step on Columbia, 2000 agents).
"""
import time

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


def bench(n_agents, n_steps, config="config_curonian_minimal.yaml"):
    cfg = load_config(config)
    sim = Simulation(cfg, n_agents=n_agents, data_dir="data", rng_seed=42)
    # Warm JIT
    sim.run(n_steps=5)

    t0 = time.perf_counter()
    sim.run(n_steps=n_steps)
    dt = time.perf_counter() - t0

    return dt, int(sim.pool.alive.sum()), int(sim.pool.arrived.sum())


if __name__ == "__main__":
    print("=== Curonian Lagoon (NetCDF triangular mesh) ===")
    for n in [500, 2000, 5000]:
        dt, alive, arrived = bench(n_agents=n, n_steps=50,
                                    config="config_curonian_minimal.yaml")
        print(f"N={n:5d} agents x 50 steps: {dt:6.2f}s  "
              f"({dt/50*1000:6.1f} ms/step)  "
              f"alive={alive} arrived={arrived}")

    print("\n=== Columbia River (HexSim hex grid, 16M-cell) ===")
    # Columbia [small] workspace; 2000 agents matches README benchmark.
    for n in [500, 2000]:
        try:
            dt, alive, arrived = bench(n_agents=n, n_steps=50,
                                        config="config_columbia.yaml")
            print(f"N={n:5d} agents x 50 steps: {dt:6.2f}s  "
                  f"({dt/50*1000:6.1f} ms/step)  "
                  f"alive={alive} arrived={arrived}")
        except Exception as e:
            print(f"N={n:5d}: SKIP ({type(e).__name__}: {e})")
