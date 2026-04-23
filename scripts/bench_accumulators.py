"""Micro-benchmark: accumulator-heavy scenario timing.

Measures the hot loop of updater_expression + updater_increment +
updater_accumulator_transfer under realistic (N=50k agents, M=74 accumulators)
shapes. Current-main baseline will be captured; the transpose refactor
(plan 2026-04-24-accumulator-layout-transpose) is justified only if this
improves by >=1.3x.
"""
import time

import numpy as np

from salmon_ibm.accumulators import (
    AccumulatorManager,
    AccumulatorDef,
    updater_expression,
    updater_increment,
    updater_accumulator_transfer,
)


N_AGENTS = 50_000
N_ACCS = 74
N_ITER = 200


def bench():
    defs = [AccumulatorDef(name=f"acc_{i}", min_val=0.0, max_val=1e9)
            for i in range(N_ACCS)]
    mgr = AccumulatorManager(N_AGENTS, defs)
    rng = np.random.default_rng(42)
    # Seed with random data in whatever the current layout expects.
    mgr.data[:] = rng.random(mgr.data.shape)
    mask = np.ones(N_AGENTS, dtype=bool)

    # Pattern 1: per-column write (touches ONE column)
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        updater_increment(mgr, "acc_7", mask, amount=1.0)
    t_increment = time.perf_counter() - t0

    # Pattern 2: expression over 2 columns (read 2, write 1)
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        updater_expression(mgr, "acc_15", mask, expression="acc_7 * 2 + acc_3")
    t_expression = time.perf_counter() - t0

    # Pattern 3: transfer between two columns
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        updater_accumulator_transfer(mgr, "acc_7", "acc_15", mask, fraction=0.1)
    t_transfer = time.perf_counter() - t0

    return {
        "increment": t_increment,
        "expression": t_expression,
        "transfer": t_transfer,
        "total": t_increment + t_expression + t_transfer,
    }


if __name__ == "__main__":
    runs = [bench() for _ in range(3)]
    medians = {k: sorted(r[k] for r in runs)[1] for k in runs[0]}
    print(f"N_AGENTS={N_AGENTS} N_ACCS={N_ACCS} N_ITER={N_ITER}")
    for k, v in medians.items():
        print(f"  {k:12s}: {v:6.3f}s ({N_ITER/v:9.0f} ops/s)")
