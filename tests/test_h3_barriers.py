"""Unit tests for the H3 barrier loader + line-to-edges helper.

Phase 4 of ``docs/superpowers/plans/2026-04-24-h3mesh-backend.md``.
"""
from __future__ import annotations

import csv as _csv
from pathlib import Path

import h3
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Task 4.1 — line_barrier_to_h3_edges
# ---------------------------------------------------------------------------


def test_line_barrier_emits_one_edge_per_cell_crossing():
    """A 2-km easterly line at 55°N res 9 (~200 m cells) should cross
    ~10 cell boundaries — bidirectional doubles to ~20 edges.
    """
    from salmon_ibm.h3_barriers import line_barrier_to_h3_edges

    edges = line_barrier_to_h3_edges(
        55.30, 21.10, 55.30, 21.13,  # ~2 km eastward at 55°N
        resolution=9,
    )
    assert 5 < len(edges) < 40, (
        f"expected ~10-30 bidirectional edges, got {len(edges)}"
    )


def test_line_barrier_pairs_are_bidirectional():
    from salmon_ibm.h3_barriers import line_barrier_to_h3_edges

    edges = line_barrier_to_h3_edges(
        55.30, 21.10, 55.30, 21.13, resolution=9, bidirectional=True
    )
    pairs = set(edges)
    for a, b in list(pairs):
        assert (b, a) in pairs, f"reverse edge missing for {a} -> {b}"


def test_line_barrier_unidirectional_is_half():
    from salmon_ibm.h3_barriers import line_barrier_to_h3_edges

    bi = line_barrier_to_h3_edges(
        55.30, 21.10, 55.30, 21.13, resolution=9, bidirectional=True
    )
    uni = line_barrier_to_h3_edges(
        55.30, 21.10, 55.30, 21.13, resolution=9, bidirectional=False
    )
    assert len(uni) == len(bi) // 2


def test_line_barrier_edges_are_h3_neighbours():
    from salmon_ibm.h3_barriers import line_barrier_to_h3_edges

    edges = line_barrier_to_h3_edges(
        55.30, 21.10, 55.30, 21.13, resolution=9
    )
    for a, b in edges:
        nbrs = set(h3.grid_ring(a, 1))
        assert b in nbrs, f"{b} is not an H3 neighbour of {a}"


def test_line_barrier_zero_length_returns_empty():
    from salmon_ibm.h3_barriers import line_barrier_to_h3_edges

    edges = line_barrier_to_h3_edges(55.3, 21.1, 55.3, 21.1, resolution=9)
    assert edges == []


def test_line_barrier_picks_n_samples_proportional_to_length():
    """1° (~70 km) line at res 9 (~200 m cells) → ~350 cell crossings.
    Helper should auto-scale n_samples so it doesn't under-sample."""
    from salmon_ibm.h3_barriers import line_barrier_to_h3_edges

    edges = line_barrier_to_h3_edges(
        55.0, 21.0, 55.0, 22.0,   # ~64 km at 55°N
        resolution=9,
    )
    # Bidirectional → 2 edges per crossing.  At 200 m cells over 64 km
    # we expect ~300 crossings → ~600 edges.  Generous bounds.
    assert 200 < len(edges) < 1500


# ---------------------------------------------------------------------------
# Task 4.2 — BarrierMap.from_csv_h3
# ---------------------------------------------------------------------------


def _write_barrier_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Helper: write a small barrier CSV from row dicts."""
    out = tmp_path / "barriers.csv"
    fields = ["from_h3", "to_h3", "mortality", "deflection", "transmission", "note"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return out


def _make_h3_mesh_centred_at(lat: float, lon: float, ring_k: int = 2):
    """Convenience: small H3 mesh around (lat, lon) at res 9."""
    from salmon_ibm.h3mesh import H3Mesh
    centre = h3.latlng_to_cell(lat, lon, 9)
    cells = list(h3.grid_disk(centre, ring_k))
    return H3Mesh.from_h3_cells(cells)


def test_from_csv_h3_round_trips_through_line_helper(tmp_path):
    """End-to-end: line_barrier_to_h3_edges output → CSV → BarrierMap
    with every edge faithfully loaded."""
    from salmon_ibm.h3_barriers import line_barrier_to_h3_edges
    from salmon_ibm.barriers import BarrierMap

    edges = line_barrier_to_h3_edges(
        55.30, 21.10, 55.30, 21.13, resolution=9
    )
    # Build a mesh that includes every cell touched by the line.
    touched = set()
    for a, b in edges:
        touched.add(a)
        touched.add(b)
    from salmon_ibm.h3mesh import H3Mesh
    mesh = H3Mesh.from_h3_cells(list(touched))

    rows = [
        {"from_h3": a, "to_h3": b,
         "mortality": "0.10", "deflection": "0.85", "transmission": "0.05",
         "note": "test"}
        for a, b in edges
    ]
    csv_path = _write_barrier_csv(tmp_path, rows)
    bmap = BarrierMap.from_csv_h3(csv_path, mesh)
    # Every input edge survived as a barrier.
    assert bmap.n_edges == len(edges), (
        f"loaded {bmap.n_edges} edges, expected {len(edges)}"
    )


def test_from_csv_h3_rejects_nonneighbour_edge(tmp_path):
    """Two cells > 1 ring apart can't be a single-step barrier."""
    from salmon_ibm.barriers import BarrierMap
    from salmon_ibm.h3mesh import H3Mesh

    centre = h3.latlng_to_cell(55.3, 21.1, 9)
    far = h3.latlng_to_cell(55.4, 21.3, 9)  # ~20 km away
    mesh = H3Mesh.from_h3_cells(list(h3.grid_disk(centre, 2)) + [far])

    csv_path = _write_barrier_csv(tmp_path, [{
        "from_h3": centre, "to_h3": far,
        "mortality": "0.5", "deflection": "0.4", "transmission": "0.1",
    }])
    with pytest.raises(ValueError, match="not H3 neighbours"):
        BarrierMap.from_csv_h3(csv_path, mesh)


def test_from_csv_h3_rejects_bad_probability_sum(tmp_path):
    from salmon_ibm.barriers import BarrierMap

    centre = h3.latlng_to_cell(55.3, 21.1, 9)
    mesh = _make_h3_mesh_centred_at(55.3, 21.1)
    nbr = list(h3.grid_ring(centre, 1))[0]

    csv_path = _write_barrier_csv(tmp_path, [{
        "from_h3": centre, "to_h3": nbr,
        # 0.5 + 0.5 + 0.5 = 1.5 — must reject.
        "mortality": "0.5", "deflection": "0.5", "transmission": "0.5",
    }])
    with pytest.raises(ValueError, match=r"1\.0"):
        BarrierMap.from_csv_h3(csv_path, mesh)


def test_from_csv_h3_skips_off_mesh_rows(tmp_path):
    """Edges whose cells aren't in the mesh are skipped, not raised —
    lets a single CSV cover multiple landscapes."""
    from salmon_ibm.barriers import BarrierMap
    from salmon_ibm.h3mesh import H3Mesh

    centre = h3.latlng_to_cell(55.3, 21.1, 9)
    nbr = list(h3.grid_ring(centre, 1))[0]
    mesh = H3Mesh.from_h3_cells([centre, nbr])

    far_a = h3.latlng_to_cell(20.0, 30.0, 9)  # not in mesh
    far_b = list(h3.grid_ring(far_a, 1))[0]   # also not in mesh

    csv_path = _write_barrier_csv(tmp_path, [
        {"from_h3": centre, "to_h3": nbr,
         "mortality": "0.1", "deflection": "0.85", "transmission": "0.05"},
        {"from_h3": far_a, "to_h3": far_b,
         "mortality": "0.1", "deflection": "0.85", "transmission": "0.05"},
    ])
    bmap = BarrierMap.from_csv_h3(csv_path, mesh)
    assert bmap.n_edges == 1, (
        f"expected 1 in-mesh edge, got {bmap.n_edges}"
    )


def test_from_csv_h3_rejects_mesh_without_h3_ids(tmp_path):
    """Type guard: the loader requires an H3Mesh."""
    from salmon_ibm.barriers import BarrierMap

    centre = h3.latlng_to_cell(55.3, 21.1, 9)
    nbr = list(h3.grid_ring(centre, 1))[0]
    csv_path = _write_barrier_csv(tmp_path, [{
        "from_h3": centre, "to_h3": nbr,
        "mortality": "0.1", "deflection": "0.85", "transmission": "0.05",
    }])

    class FakeNonH3Mesh:
        n_cells = 10
    with pytest.raises(TypeError, match="H3Mesh"):
        BarrierMap.from_csv_h3(csv_path, FakeNonH3Mesh())


def test_from_csv_h3_missing_columns_raises(tmp_path):
    from salmon_ibm.barriers import BarrierMap
    from salmon_ibm.h3mesh import H3Mesh

    centre = h3.latlng_to_cell(55.3, 21.1, 9)
    nbr = list(h3.grid_ring(centre, 1))[0]
    mesh = H3Mesh.from_h3_cells([centre, nbr])

    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("from_h3,to_h3\n" f"{centre},{nbr}\n")
    with pytest.raises(ValueError, match="missing columns"):
        BarrierMap.from_csv_h3(csv_path, mesh)
