"""Multiprocessing ensemble runner for replicate simulations."""
from __future__ import annotations

from multiprocessing import Pool
from typing import Any

import numpy as np


def _run_single_replicate(args: tuple) -> dict[str, Any]:
    """Run one replicate — must be a top-level function for pickling."""
    config, n_agents, n_steps, seed = args
    from salmon_ibm.simulation import Simulation
    sim = Simulation(config, n_agents=n_agents, rng_seed=seed)
    sim.run(n_steps)
    return {
        "seed": seed,
        "history": sim.history,
        "n_alive": int(sim.pool.alive.sum()),
        "n_arrived": int(sim.pool.arrived.sum()),
    }


def run_ensemble(
    config: dict,
    n_replicates: int = 10,
    n_agents: int = 1000,
    n_steps: int = 100,
    n_workers: int | None = None,
    base_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run multiple simulation replicates in parallel.

    Parameters
    ----------
    config : simulation config dict (same as Simulation.__init__ expects).
    n_replicates : number of independent replicates.
    n_agents : agents per replicate.
    n_steps : timesteps per replicate.
    n_workers : number of parallel processes (None = os.cpu_count()).
    base_seed : deterministic seed generator. If None, uses random seeds.

    Returns
    -------
    List of result dicts, each with keys: seed, history, n_alive, n_arrived.
    """
    rng = np.random.default_rng(base_seed)
    seeds = [int(rng.integers(2**31)) for _ in range(n_replicates)]
    args_list = [(config, n_agents, n_steps, s) for s in seeds]

    if n_workers == 1:
        return [_run_single_replicate(a) for a in args_list]

    with Pool(processes=n_workers) as pool:
        results = pool.map(_run_single_replicate, args_list)
    return results
