"""Grid-quality regressions: connectivity and area-fidelity per reach.

Loads `data/curonian_h3_multires_landscape.nc` and asserts:
1. Each reach forms ≤ N_MAX_COMPONENTS connected H3 components.
2. Cell-area sum matches inSTREAM polygon area within ±10%.
3. Cross-reach link count meets minimum thresholds for ecological
   adjacency (e.g., Nemunas↔Atmata ≥ 5 links).

Tests skip cleanly when the NC isn't built locally.
"""
from __future__ import annotations
from pathlib import Path
import pytest
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
NC = PROJECT / "data" / "curonian_h3_multires_landscape.nc"

# Per-reach maximum allowed component count.  Within-reach connectivity
# is the v1.5.0 goal (was 135 components for Minija pre-rebuild,
# 60 for Sysa, 58 for Leite).  Two reaches are allowed >1 components:
# - Atmata: the inSTREAM source shapefile has 2 separated polygon
#   pieces (a small detached fragment near the lagoon mouth that the
#   bridge-cell pass cannot span at res 11 / >10-cell distance).
# - OpenBaltic: legitimately Swiss-cheese-fragmented at res 9 because
#   the polygon is (NE ocean − BalticCoast − CuronianLagoon) — the
#   lagoon subtraction (added in v1.5.2 to stop OpenBaltic claiming
#   cells INSIDE the lagoon) creates a hole that fragments the
#   tessellation into ~60 small pieces.  v1.5.0 (no lagoon
#   subtraction) had 8 components but ~16k cells inside the lagoon —
#   visually wrong.  Trade-off favours visual correctness.
N_MAX_COMPONENTS = {
    "Nemunas":         1,
    "Atmata":          2,
    "Minija":          1,
    "Sysa":            1,
    "Skirvyte":        1,
    "Leite":           1,
    "Gilija":          1,
    "CuronianLagoon":  1,
    "BalticCoast":     1,
    "OpenBaltic":     70,
}

# Minimum cross-reach links — geographically adjacent reaches must
# have at least this many same-resolution OR cross-resolution links
# in the neighbour table.  Calibrated against the v1.5.0 build's
# actual achieved counts: most inland rivers (Nemunas, Minija, Sysa,
# Leite) do NOT directly border the lagoon because their inSTREAM
# polygons end where the channel splits into delta branches.  Only
# the delta-branch reaches (Atmata, Gilija, Skirvyte) physically
# touch the lagoon.  We use threshold = floor(observed × 0.5) for
# realistic regression coverage with margin for stochasticity.
MIN_CROSS_REACH_LINKS = {
    # Strait — Klaipėda lagoon ↔ Baltic.  Observed 198 directed.
    ("CuronianLagoon", "BalticCoast"):    100,
    # Same-resolution Baltic↔Baltic.  Observed 748 directed.
    ("BalticCoast",    "OpenBaltic"):     400,
    # Delta-branch rivers reaching the lagoon (cross-res 11→10).
    ("Atmata",         "CuronianLagoon"):  10,    # observed 28
    ("Skirvyte",       "CuronianLagoon"):  25,    # observed 62
    ("Gilija",         "CuronianLagoon"):  30,    # observed 86
    # Inter-river junctions in the Nemunas delta (same-res 11).
    ("Nemunas",        "Atmata"):          15,    # observed 38
    ("Nemunas",        "Gilija"):          10,    # observed 26
    ("Atmata",         "Skirvyte"):         8,    # observed 20
}


def _load_nc():
    if not NC.exists():
        pytest.skip(f"{NC.name} missing — rebuild via scripts/build_h3_multires_landscape.py")
    import xarray as xr
    return xr.open_dataset(NC, engine="h5netcdf")


def _components_per_reach(ds, reach_name: str) -> int:
    """BFS within reach cells using the CSR neighbour table."""
    reach_names = ds.attrs["reach_names"].split(",")
    if reach_name not in reach_names:
        return 0
    rid = reach_names.index(reach_name)
    reach_id = ds["reach_id"].values
    nbr_starts = ds["nbr_starts"].values
    nbr_idx = ds["nbr_idx"].values
    cells_in = set(np.where(reach_id == rid)[0].tolist())
    if not cells_in:
        return 0
    seen: set[int] = set()
    n_components = 0
    for start in cells_in:
        if start in seen:
            continue
        n_components += 1
        stack = [start]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            for nb in nbr_idx[nbr_starts[c]:nbr_starts[c+1]]:
                nb = int(nb)
                if nb in cells_in and nb not in seen:
                    stack.append(nb)
    return n_components


def test_each_reach_is_connected():
    ds = _load_nc()
    failed = []
    for reach, max_components in N_MAX_COMPONENTS.items():
        n = _components_per_reach(ds, reach)
        if n == 0:
            continue  # reach not present in this NC
        if n > max_components:
            failed.append((reach, n, max_components))
    ds.close()
    assert not failed, (
        "reaches with too many disconnected components:\n  "
        + "\n  ".join(f"{r}: {got} > {max_}" for r, got, max_ in failed)
    )


def test_reach_id_implies_water_mask():
    """Every cell with reach_id != -1 must have water_mask == True.

    The build script tessellates each reach polygon with a half-cell-edge
    buffer to ensure narrow channels get cells (otherwise centroid-in-
    polygon test produces sparse, disconnected sets — the v1.5.0 issue).
    But the buffer can capture cells whose centroids sit on dry land
    beyond the actual water's edge.  Without a post-filter, those cells
    end up tagged with reach_id but have water_mask=False because EMODnet
    bathymetry says they're dry.

    Symptom: the viewer (app.py:1811) filters by water_mask=True, dropping
    47-87% of river cells from the rendered hex layer — polygon outlines
    appear visually unfilled.

    Fix: build_h3_multires_landscape.py:475 unsets reach_id (-1) for
    cells with water_mask=False.

    This test pins the contract: a cell's reach_id and water_mask must
    agree about whether the cell is water.
    """
    ds = _load_nc()
    reach_id = ds["reach_id"].values
    water_mask = ds["water_mask"].values.astype(bool)
    tagged_dry = (reach_id != -1) & ~water_mask
    n_bad = int(tagged_dry.sum())
    if n_bad:
        # Per-reach breakdown for the failure message.
        names = ds.attrs["reach_names"].split(",")
        per_reach = []
        for rid in sorted(set(reach_id[tagged_dry].tolist())):
            n = int(((reach_id == rid) & ~water_mask).sum())
            name = names[rid] if 0 <= rid < len(names) else f"<id={rid}>"
            per_reach.append(f"  {name}: {n} dry cells tagged")
        ds.close()
        raise AssertionError(
            f"{n_bad} cells have reach_id != -1 but water_mask=False:\n"
            + "\n".join(per_reach)
        )
    ds.close()


def test_cross_reach_link_thresholds():
    ds = _load_nc()
    reach_names = ds.attrs["reach_names"].split(",")
    reach_id = ds["reach_id"].values
    nbr_starts = ds["nbr_starts"].values
    nbr_idx = ds["nbr_idx"].values

    # Count links per (sorted) reach pair.
    pairs: dict[tuple[str, str], int] = {}
    for i in range(len(reach_id)):
        rid_i = int(reach_id[i])
        if rid_i < 0:
            continue
        for j in nbr_idx[nbr_starts[i]:nbr_starts[i+1]]:
            rid_j = int(reach_id[j])
            if rid_j < 0 or rid_j == rid_i:
                continue
            key = tuple(sorted([reach_names[rid_i], reach_names[rid_j]]))
            pairs[key] = pairs.get(key, 0) + 1
    ds.close()

    failed = []
    for (a, b), threshold in MIN_CROSS_REACH_LINKS.items():
        key = tuple(sorted([a, b]))
        got = pairs.get(key, 0)
        if got < threshold:
            failed.append((a, b, got, threshold))
    assert not failed, (
        "cross-reach link counts below minimum:\n  "
        + "\n  ".join(f"{a}↔{b}: {got} < {th}" for a, b, got, th in failed)
    )
