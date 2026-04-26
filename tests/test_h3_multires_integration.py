"""End-to-end test for the multi-res H3 backend."""
from __future__ import annotations
from pathlib import Path
import pytest

PROJECT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT / "configs" / "config_curonian_h3_multires.yaml"
LANDSCAPE_NC = PROJECT / "data" / "curonian_h3_multires_landscape.nc"


@pytest.fixture(scope="module")
def multires_sim():
    if not CONFIG.exists() or not LANDSCAPE_NC.exists():
        pytest.skip("h3_multires config or NC missing — build first.")
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config(str(CONFIG))
    return Simulation(
        cfg, n_agents=50,
        data_dir=str(PROJECT / "data"), rng_seed=42,
    )


def test_mesh_is_h3multires(multires_sim):
    from salmon_ibm.h3_multires import H3MultiResMesh
    assert isinstance(multires_sim.mesh, H3MultiResMesh)


def test_resolutions_are_mixed(multires_sim):
    """The whole point of multi-res is mixed resolutions per cell."""
    import numpy as np
    res = multires_sim.mesh.resolutions
    unique_res = np.unique(res[multires_sim.mesh.water_mask])
    assert len(unique_res) >= 2, (
        f"expected multiple H3 resolutions; got {unique_res.tolist()}"
    )


def test_initial_placement_lands_on_water(multires_sim):
    placed = multires_sim.pool.tri_idx
    assert multires_sim.mesh.water_mask[placed].all()


def test_one_step_runs_without_error(multires_sim):
    """The single step exercises the full event sequencer including the
    movement kernels — proves the (N, 12) padded neighbour view is
    consumed correctly.  Order-independent: asserts a delta of 1, not
    an absolute current_t.  The module-scope fixture is shared, so any
    earlier test that called step() bumps current_t before we get here."""
    t_before = multires_sim.current_t
    multires_sim.step()
    assert multires_sim.current_t == t_before + 1


def test_at_least_one_agent_moves(multires_sim):
    """Cross-resolution neighbour links should let agents move just
    like the uniform-res mesh does."""
    initial = multires_sim.pool.tri_idx.copy()
    for _ in range(10):
        multires_sim.step()
    moved = (multires_sim.pool.tri_idx != initial).any()
    assert moved


def test_h3_multires_in_sidebar_choices():
    """The Study area dropdown must include 'curonian_h3_multires' so
    a user can select it without typing URL params.  Walks the
    sidebar UI panel object tree and looks for the choice key in the
    rendered HTML.

    ``Sidebar.__repr__`` returns the default object repr — we have to
    call ``tagify()`` to materialise the sidebar's TagList and render
    it to HTML before substring-matching.
    """
    from ui.sidebar import sidebar_panel
    panel = sidebar_panel()
    html = str(panel.tagify())
    assert "curonian_h3_multires" in html, (
        "h3_multires not registered in sidebar choices; users can't "
        "select it"
    )
