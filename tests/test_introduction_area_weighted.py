"""IntroductionEvent must place agents proportional to cell area, not
uniformly per cell.  Otherwise three-tier H3 meshes over-place in fine
river cells by ~7-50×.
"""
from __future__ import annotations
import numpy as np
from unittest.mock import MagicMock
from salmon_ibm.events_builtin import IntroductionEvent


def test_introduction_weights_by_area():
    """In a 2-cell mesh where cell 0 has area 100 and cell 1 has area 1,
    placing 10 000 agents from a uniform spatial-data layer should put
    ~9 901 in cell 0 and ~99 in cell 1."""
    rng = np.random.default_rng(42)
    mesh = MagicMock()
    mesh.areas = np.array([100.0, 1.0], dtype=np.float32)
    mesh.n_cells = 2
    layer = np.array([1, 1])  # both cells eligible

    pop = MagicMock()
    pop.add_agents.return_value = np.arange(10_000)
    pop.trait_mgr = None
    pop.accumulator_mgr = None

    event = IntroductionEvent(
        name="intro_test",
        n_agents=10_000,
        initialization_spatial_data="placement_layer",
    )
    landscape = {
        "rng": rng,
        "spatial_data": {"placement_layer": layer},
        "mesh": mesh,
    }
    event.execute(pop, landscape, t=0, mask=None)

    pos_arr = pop.add_agents.call_args[0][1]   # second positional arg
    n_in_cell0 = int((pos_arr == 0).sum())
    expected = 10_000 * 100 / 101  # ≈ 9900.99
    # σ for binomial(10000, 0.99) ≈ 9.95.  Use 3σ tolerance (<30).  A
    # uniform-over-cells implementation would put ~5000 in cell 0 — a
    # ~4900 miss, far outside this bound.  3σ also catches a ~1%
    # weight-normalisation bug (e.g., forgetting to normalise by sum).
    assert abs(n_in_cell0 - expected) < 30, (
        f"area-weighted placement off: {n_in_cell0} in cell 0, "
        f"expected {expected:.0f} ± 30"
    )


def test_introduction_falls_back_to_uniform_without_mesh():
    """When mesh is missing or has no .areas, fall back to uniform."""
    rng = np.random.default_rng(42)
    layer = np.array([1, 1])
    pop = MagicMock()
    pop.add_agents.return_value = np.arange(10_000)
    pop.trait_mgr = None
    pop.accumulator_mgr = None
    event = IntroductionEvent(
        name="intro_test", n_agents=10_000, initialization_spatial_data="layer"
    )
    landscape = {
        "rng": rng, "spatial_data": {"layer": layer},
    }
    event.execute(pop, landscape, t=0, mask=None)
    pos_arr = pop.add_agents.call_args[0][1]
    n_in_cell0 = int((pos_arr == 0).sum())
    # Uniform: ~5000 in each cell, ±200.
    assert 4_700 < n_in_cell0 < 5_300
