"""Micro-benchmark: InteractionEvent throughput.

Scenario: 500 predators + 5000 prey distributed across 100 cells.
Measures wall-clock time per event execution.
"""
import time

import numpy as np

from salmon_ibm.interactions import (
    MultiPopulationManager,
    InteractionEvent,
    InteractionOutcome,
)
from salmon_ibm.agents import AgentPool
from salmon_ibm.population import Population


def setup():
    mgr = MultiPopulationManager()
    pred_pool = AgentPool(n=500, start_tri=0, rng_seed=1)
    prey_pool = AgentPool(n=5000, start_tri=0, rng_seed=2)
    # Spread across 100 cells
    pred_pool.tri_idx[:] = np.arange(500) % 100
    prey_pool.tri_idx[:] = np.arange(5000) % 100
    pred = Population(name="pred", pool=pred_pool)
    prey = Population(name="prey", pool=prey_pool)
    mgr.register("pred", pred)
    mgr.register("prey", prey)
    return mgr, pred, prey


def bench_run():
    mgr, pred, prey = setup()
    landscape = {"multi_pop_mgr": mgr, "rng": np.random.default_rng(0)}
    event = InteractionEvent(
        name="p", pop_a_name="pred", pop_b_name="prey",
        encounter_probability=0.001,
        outcome=InteractionOutcome.PREDATION,
    )
    # Warm-up run
    event.execute(pred, landscape, 0, pred.pool.alive)
    # Timed runs — resurrect prey each time so we exercise the full matrix
    t0 = time.perf_counter()
    n_runs = 10
    for _ in range(n_runs):
        prey.pool.alive[:] = True
        event.execute(pred, landscape, 0, pred.pool.alive)
    return (time.perf_counter() - t0) / n_runs


if __name__ == "__main__":
    # Three trials, report median
    runs = [bench_run() for _ in range(3)]
    median = sorted(runs)[1]
    print(f"500 predators x 5000 prey x 100 cells, p=0.001")
    print(f"  per-execute time (median of 3): {median*1000:6.1f} ms")
    print(f"  ({runs[0]*1000:.1f}, {runs[1]*1000:.1f}, {runs[2]*1000:.1f})")
