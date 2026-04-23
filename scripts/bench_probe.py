"""Isolated probe: does transposing (n_agents, n_acc) -> (n_acc, n_agents)
actually speed up the strided-gather pattern used by _LazyAccDict?
"""
import time
import numpy as np


N_AGENTS = 50_000
N_ACCS = 74
N_ITER = 5000  # more iters since this is pure numpy


def bench(layout):
    if layout == "row_major":
        data = np.random.rand(N_AGENTS, N_ACCS).astype(np.float64)
    else:
        data = np.random.rand(N_ACCS, N_AGENTS).astype(np.float64)

    mask = np.ones(N_AGENTS, dtype=bool)
    col = 7

    # Pattern: same as _LazyAccDict.__getitem__
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        if layout == "row_major":
            x = data[mask, col]  # strided gather
        else:
            x = data[col, mask]  # contiguous gather
    gather_time = time.perf_counter() - t0

    # Pattern: write back (last step of updater_expression)
    result = np.random.rand(int(mask.sum()))
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        if layout == "row_major":
            data[mask, col] = result
        else:
            data[col, mask] = result
    write_time = time.perf_counter() - t0

    # Expression pattern: gather 2 cols + arithmetic + write 1
    col2 = 15
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        if layout == "row_major":
            a = data[mask, col]
            b = data[mask, col2]
            data[mask, col2] = a * 2 + b
        else:
            a = data[col, mask]
            b = data[col2, mask]
            data[col2, mask] = a * 2 + b
    expr_time = time.perf_counter() - t0

    return gather_time, write_time, expr_time


if __name__ == "__main__":
    g_r, w_r, e_r = bench("row_major")
    g_c, w_c, e_c = bench("col_major")
    print(f"N_AGENTS={N_AGENTS} N_ACCS={N_ACCS} N_ITER={N_ITER}")
    print(f"{'':12s}  {'row(n_a,n_c)':>14s}  {'col(n_c,n_a)':>14s}  {'speedup':>8s}")
    print(f"{'gather':12s}  {g_r:>14.3f}  {g_c:>14.3f}  {g_r/g_c:>7.2f}x")
    print(f"{'write':12s}  {w_r:>14.3f}  {w_c:>14.3f}  {w_r/w_c:>7.2f}x")
    print(f"{'expression':12s}  {e_r:>14.3f}  {e_c:>14.3f}  {e_r/e_c:>7.2f}x")
