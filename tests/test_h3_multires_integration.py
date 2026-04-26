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


@pytest.mark.slow
def test_agents_cross_resolution_boundaries():
    """Over 3 days an agent that starts in a fine river cell should
    end up in a coarser zone (lagoon or Baltic) at least once.

    Builds its OWN Simulation (not the module-scope fixture) so the
    forced position-overwrite below is the only mutation in flight.
    """
    if not CONFIG.exists() or not LANDSCAPE_NC.exists():
        pytest.skip("h3_multires config or NC missing — build first.")
    import numpy as np
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config(str(CONFIG))
    sim = Simulation(
        cfg, n_agents=50,
        data_dir=str(PROJECT / "data"), rng_seed=42,
    )
    mesh = sim.mesh

    # Force-place all 50 agents in fine cells (rivers are res 10 in
    # the test config; lagoon res 9, Baltic res 8).  AgentPool stores
    # positions as a plain SoA array — direct assignment is safe; no
    # derived caches need invalidating because env / movement re-read
    # ``pool.tri_idx`` every step.
    fine_water = np.where(
        (mesh.resolutions == 10) & mesh.water_mask
    )[0]
    if len(fine_water) < 50:
        pytest.skip(f"only {len(fine_water)} fine water cells — need ≥50")
    sim.pool.tri_idx[:] = fine_water[:50]

    for _ in range(3 * 24):  # 3 days; reduced from 30 to keep CI fast
        sim.step()

    final_res = mesh.resolutions[sim.pool.tri_idx]
    crossed = (final_res < 10).any()
    assert crossed, (
        "no agent crossed from fine to coarse zone in 3 days; the "
        "cross-resolution neighbour table may not be wired into "
        "movement kernels"
    )


def test_salinity_correlates_with_resolution_zone(multires_sim):
    """Agents in res-10 river cells should experience low salinity
    (≲3 PSU); agents in res-8 OpenBaltic should be high (≳4 PSU).
    Decouples the salinity-gradient test from cell-count percentiles
    (the v1.2.7 N-S test breaks under skewed cell areas, see
    Known Limitations doc)."""
    import numpy as np
    sal = multires_sim.env.current()["salinity"]
    res = multires_sim.mesh.resolutions[multires_sim.pool.tri_idx]
    in_river = res == 10
    in_baltic = res == 8
    if in_river.any():
        assert np.nanmean(sal[multires_sim.pool.tri_idx[in_river]]) < 3.0, (
            "river-resolution agents should see fresh water"
        )
    if in_baltic.any():
        assert np.nanmean(sal[multires_sim.pool.tri_idx[in_baltic]]) > 4.0, (
            "Baltic-resolution agents should see brackish-to-saline water"
        )


@pytest.mark.slow
def test_no_one_way_trap_at_resolution_boundary():
    """Force-place agents in coarse OpenBaltic cells and run 3 days;
    at least one should reach a finer-resolution cell (lagoon res 9
    or river res 10).  The cross-res neighbour table must be
    symmetric: a fine→coarse hop must be reversible."""
    import numpy as np
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    if not CONFIG.exists() or not LANDSCAPE_NC.exists():
        pytest.skip("h3_multires NC missing")
    cfg = load_config(str(CONFIG))
    sim = Simulation(cfg, n_agents=50, data_dir=str(PROJECT / "data"), rng_seed=43)
    # Filter coarse cells to those with at least one finer neighbour —
    # i.e., the boundary between OpenBaltic (res-8) and BalticCoast
    # (res-9).  Without this filter, agents placed deep in OpenBaltic
    # take many random-walk steps to even reach the boundary; we want
    # to test the cross-res link itself, not random-walk diffusion.
    coarse = np.where(
        (sim.mesh.resolutions == 8) & sim.mesh.water_mask
    )[0]
    boundary = []
    for c in coarse:
        nbrs = sim.mesh.neighbours_of(int(c))
        if len(nbrs) and (sim.mesh.resolutions[nbrs] > 8).any():
            boundary.append(int(c))
    if len(boundary) < 50:
        pytest.skip(
            f"only {len(boundary)} OpenBaltic boundary cells — need ≥50"
        )
    sim.pool.tri_idx[:] = np.array(boundary[:50], dtype=sim.pool.tri_idx.dtype)
    for _ in range(3 * 24):
        sim.step()
    final_res = sim.mesh.resolutions[sim.pool.tri_idx]
    came_back = (final_res > 8).any()
    assert came_back, (
        "no agent moved from res-8 OpenBaltic boundary to a finer zone "
        "in 3 days — the cross-res neighbour table is asymmetric "
        "(fine→coarse works but coarse→fine is broken)."
    )


def test_per_step_displacement_bounded(multires_sim):
    """Every link in the cross-res neighbour table must connect cells
    whose centroids are no farther apart than 3 × max edge of the
    two cells.  Tested directly on the static neighbour table — not
    via a sim step — because ``step()`` runs multiple movement
    substeps inside the numba kernel, so per-step displacement isn't
    a single-link bound.  This is a geometric sanity check on the
    cross-res adjacency: catches malformed table entries that would
    let agents teleport across the mesh."""
    import numpy as np
    import h3 as _h3
    mesh = multires_sim.mesh
    centroids = mesh.centroids  # (N, 2) lat, lon

    # Build origin/dest index pairs from the CSR table.
    n = mesh.n_cells
    # Repeat origin index per neighbour count so origin[k] pairs
    # with mesh.nbr_idx[k].
    counts = np.diff(mesh.nbr_starts).astype(np.int64)
    origin = np.repeat(np.arange(n, dtype=np.int64), counts)
    dest = mesh.nbr_idx.astype(np.int64)
    if len(origin) == 0:
        pytest.skip("empty neighbour table")

    # Vectorised haversine over all links.
    lat0 = centroids[origin, 0]; lon0 = centroids[origin, 1]
    lat1 = centroids[dest, 0];   lon1 = centroids[dest, 1]
    dlat = np.radians(lat1 - lat0)
    dlon = np.radians(lon1 - lon0)
    mid = np.radians(0.5 * (lat0 + lat1))
    dy = dlat * 6_371_000
    dx = dlon * 6_371_000 * np.cos(mid)
    dist = np.sqrt(dx * dx + dy * dy)

    # Bound: 3 × max(origin edge, dest edge) — generous to absorb
    # cross-res cases where a fine cell's link to a coarse parent
    # crosses up to one coarse edge length.
    edge_by_res = {
        r: _h3.average_hexagon_edge_length(int(r), unit="m")
        for r in np.unique(mesh.resolutions)
    }
    edge_o = np.array([edge_by_res[int(r)] for r in mesh.resolutions[origin]])
    edge_d = np.array([edge_by_res[int(r)] for r in mesh.resolutions[dest]])
    max_dist = 3.0 * np.maximum(edge_o, edge_d)

    violations = dist > max_dist
    assert not violations.any(), (
        f"{int(violations.sum())} cross-res neighbour links exceed "
        f"3 × max(origin edge, dest edge).  Worst case: "
        f"{dist[violations].max():.0f} m vs bound "
        f"{max_dist[violations].max():.0f} m at link "
        f"({origin[violations][0]} → {dest[violations][0]})."
    )


def test_mass_loss_rate_independent_of_cell_resolution():
    """Pin two cohorts of identical agents in cells at the same
    temperature but different resolutions (one in a river, one in
    the lagoon).  After 24 hourly steps their mean mass-loss
    fraction should agree to within 5 %.

    Catches accidental area-coupling in any future bioenergetics
    change — e.g. a `food = drift_density * mesh.areas[c]`
    formula that a per-reach ecology refactor might introduce.
    """
    pytest.skip(
        "Pending Phase B per-reach ecology — useful regression test "
        "to add when `mortality_per_reach` lands.  See "
        "docs/per-reach-ecology-plan.md."
    )
