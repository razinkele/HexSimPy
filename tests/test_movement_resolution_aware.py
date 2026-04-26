"""Resolution-aware movement: agents in finer cells should hop more
cell-edges per simulation step, producing equal physical displacement
across mixed-resolution meshes."""
from __future__ import annotations
import numpy as np
import pytest


def test_n_micro_steps_per_cell_proportional_to_inverse_edge_length():
    """n_micro_steps[res11] / n_micro_steps[res9] should be ~7
    (the H3 area ratio per resolution drop is 7x, edge ratio is sqrt(7) ~ 2.65).

    Actually the relevant ratio is edge-length-inverse: edge(res9) /
    edge(res11) ~ 201 / 28 ~ 7.18.  So n_micro[res11] ~ 7 x n_micro[res9].
    """
    import h3
    SWIM_SPEED = 1.0
    DT = 3600.0
    edge_9 = h3.average_hexagon_edge_length(9, unit="m")
    edge_11 = h3.average_hexagon_edge_length(11, unit="m")
    n9 = round(SWIM_SPEED * DT / edge_9)
    n11 = round(SWIM_SPEED * DT / edge_11)
    ratio = n11 / n9
    assert 6.5 < ratio < 7.5, (
        f"n_micro ratio res11:res9 = {ratio:.2f}, expected ~7.18"
    )


def test_budget_uniform_resolution_stops_after_n_hops():
    """In a ring graph at uniform resolution, an agent with
    n_micro_per_cell=2 should stop after exactly 2 hops, regardless
    of how many iterations max_steps allows."""
    from salmon_ibm.movement import _step_random_numba

    n_cells = 10
    water_nbrs = np.zeros((n_cells, 1), dtype=np.int32)
    water_nbr_count = np.ones(n_cells, dtype=np.int32)
    for i in range(n_cells):
        water_nbrs[i, 0] = (i + 1) % n_cells

    n_agents = 4
    tri_indices = np.zeros(n_agents, dtype=np.int32)
    rand_vals = np.zeros((10, n_agents), dtype=np.float64)
    n_micro_per_cell = np.full(n_cells, 2, dtype=np.int32)
    fraction_remaining = np.ones(n_agents, dtype=np.float32)

    _step_random_numba(
        tri_indices, water_nbrs, water_nbr_count, rand_vals,
        10,                # max_steps
        n_micro_per_cell,
        fraction_remaining,
    )
    # Each agent: hop 1 (0->1) consumes 0.5, hop 2 (1->2) consumes 0.5,
    # fraction reaches 0, agent stops.  Final position: cell 2.
    assert (tri_indices == 2).all(), (
        f"agents took {tri_indices.tolist()} hops, expected 2 each"
    )
    assert (fraction_remaining <= 0.0).all()


def test_budget_distance_conserved_across_resolution_boundary():
    """An agent crossing from coarse-budget cells into fine-budget
    cells must NOT get its swim-distance budget refreshed at the
    boundary.

    Setup: ring graph of 10 cells.  Cells [0,1] have budget=4 (cost
    0.25 per hop = "coarse"); cells [2..9] have budget=8 (cost 0.125
    per hop = "fine").  Agent starts at cell 0.

    Correct trace (cost charged at *source* cell):
        0->1: cost 0.25, used=0.25, at cell 1
        1->2: cost 0.25, used=0.50, at cell 2
        2->3: cost 0.125, used=0.625, at cell 3
        3->4: cost 0.125, used=0.750, at cell 4
        4->5: cost 0.125, used=0.875, at cell 5
        5->6: cost 0.125, used=1.000, at cell 6, fraction_remaining <= 0
        agent stops at cell 6.  Total hops = 6.

    Buggy trace (budget refreshed when entering fine cell):
        2 coarse hops + 8 fine hops = 10 hops, ending at cell 10%10=0.

    The assertion `tri_indices == 6` discriminates correct from buggy."""
    from salmon_ibm.movement import _step_random_numba

    n_cells = 10
    water_nbrs = np.zeros((n_cells, 1), dtype=np.int32)
    water_nbr_count = np.ones(n_cells, dtype=np.int32)
    for i in range(n_cells):
        water_nbrs[i, 0] = (i + 1) % n_cells

    n_micro_per_cell = np.array(
        [4, 4, 8, 8, 8, 8, 8, 8, 8, 8], dtype=np.int32,
    )
    n_agents = 4
    tri_indices = np.zeros(n_agents, dtype=np.int32)
    rand_vals = np.zeros((16, n_agents), dtype=np.float64)
    fraction_remaining = np.ones(n_agents, dtype=np.float32)

    _step_random_numba(
        tri_indices, water_nbrs, water_nbr_count, rand_vals,
        16,                # max_steps = max(n_micro_per_cell)
        n_micro_per_cell,
        fraction_remaining,
    )
    # Correct algorithm: 2 coarse hops (at cost 0.25 each = 0.5 used)
    # + 4 fine hops (at cost 0.125 each = 0.5 used) -> 6 total, at cell 6.
    # Buggy algorithm (resource refresh): would reach cell 10%10 = 0.
    assert (tri_indices == 6).all(), (
        f"agents reached {tri_indices.tolist()}, expected 6.  Anything "
        f"else (especially 0 or 10) suggests the budget-refresh bug "
        f"described in plan Task 0.5 - agent re-spent its swim distance "
        f"after crossing into the finer-resolution cells."
    )
