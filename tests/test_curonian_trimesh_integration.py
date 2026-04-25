"""End-to-end smoke test for the Curonian Lagoon TriMesh scenario.

Mirrors tests/test_nemunas_h3_integration.py but for the TriMesh
backend.  Skipped when the gitignored landscape NC isn't built locally
(re-run scripts/build_curonian_trimesh.py to produce it).
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
LANDSCAPE_NC = PROJECT / "data" / "curonian_trimesh_landscape.nc"
FORCING_NC = PROJECT / "data" / "curonian_trimesh_forcing_stub.nc"
CONFIG = PROJECT / "configs" / "config_curonian_trimesh.yaml"


@pytest.fixture(scope="module")
def sim():
    """Build the simulation once for the whole module."""
    if not LANDSCAPE_NC.exists() or not FORCING_NC.exists():
        pytest.skip(
            "Curonian TriMesh NCs not built — run "
            "`scripts/build_curonian_trimesh.py` first."
        )
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config(str(CONFIG))
    return Simulation(cfg, n_agents=50, data_dir=str(PROJECT / "data"), rng_seed=42)


def test_mesh_shape(sim):
    """The mesh should be a TriMesh with sensible bounds and water cell count."""
    from salmon_ibm.mesh import TriMesh
    assert isinstance(sim.mesh, TriMesh)
    # 100 m grid over the H3 bbox: ~948 k nodes → ~1.9 M triangles via Delaunay.
    assert 1_500_000 < sim.mesh.n_triangles < 2_500_000, (
        f"unexpected n_triangles={sim.mesh.n_triangles:,}"
    )
    # Water cells should be a substantial fraction (~half) of total triangles.
    n_water = int(sim.mesh.water_mask.sum())
    assert 500_000 < n_water < 1_500_000, f"n_water={n_water:,} outside expected range"


def test_centroids_in_bbox(sim):
    """Mesh centroids should sit inside the published bbox."""
    lats = sim.mesh.centroids[:, 0]
    lons = sim.mesh.centroids[:, 1]
    assert 54.89 < lats.min() < 55.81
    assert 54.89 < lats.max() < 55.81
    assert 20.39 < lons.min() < 21.91
    assert 20.39 < lons.max() < 21.91


def test_initial_placement(sim):
    """All agents land on water cells."""
    placed = sim.pool.tri_idx
    assert len(placed) == 50
    assert sim.mesh.water_mask[placed].all(), (
        f"{(~sim.mesh.water_mask[placed]).sum()} agents placed on land"
    )


def test_step_advances_time(sim):
    """A step bumps the simulation clock and at least one agent moves."""
    t0 = sim.current_t
    initial = sim.pool.tri_idx.copy()
    sim.step()
    assert sim.current_t == t0 + 1
    moved = (sim.pool.tri_idx != initial).any()
    assert moved, "no agent moved after one step"
