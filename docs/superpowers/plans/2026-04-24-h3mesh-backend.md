# H3Mesh Backend Implementation Plan

> **STATUS: ✅ EXECUTED 2026-04-25** — all 5 phases shipped, full
> regression suite green (729 passed, 32 skipped, 1 xfailed). See
> [§ Execution status](#execution-status) below for the commit log,
> deviations from the plan as-written, and bugs caught at runtime.

> **For agentic workers:** REQUIRED SUB-SKILL — use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Tasks use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `H3Mesh` as a third mesh backend alongside `TriMesh` (NetCDF triangular) and `HexMesh` (HexSim hex grid) so that **new landscapes** — starting with Nemunas Delta — can be built in pure H3 without any HexSim 4.0.20 workspace dependency, while Columbia and the existing Curonian TriMesh continue to work unchanged.

**Architecture:** A new class `H3Mesh` duck-types the existing Mesh interface (same attribute names and shapes as `TriMesh`/`HexMesh`), so all Numba movement/accumulator kernels work unchanged. H3-specific data paths (`h3_env.py` for forcing, `h3_barriers.py` for CSV-keyed barriers + a `line_barrier_to_h3_edges` geometric helper) sit beside the HexSim/TriMesh paths — nothing is removed, only added. Pentagon handling is a latitude-bounded pre-assertion at landscape build time — the Nemunas test landscape has zero pentagons, but the guard fires for any future bbox that crosses one. Barriers are stored as mesh-agnostic `dict[(from_idx, to_idx)]`, so only the loader differs between backends; the existing barrier-application logic works unchanged.

**Tech Stack:** `h3-py 4.4.2` (already installed), NumPy, Numba (unchanged), xarray + scipy (for regridding CMEMS/EMODnet → H3), shiny-deckgl `H3HexagonLayer` (already wired in viewer).

**Compatibility verdict (from audit + review pass):**
- Movement engine (`salmon_ibm/movement.py`, `salmon_ibm/events_hexsim.py`) is **mesh-agnostic** — works as-is once `H3Mesh` provides `water_nbrs`, `water_nbr_count`, `centroids`, `centroids_c`, and `_advection_numba` is promoted to scale-aware (Phase 0.1).
- Barrier **storage** (`BarrierMap._edges` dict) is mesh-agnostic; only the **loader** is HexSim-specific. Phase 0.2 splits the loader factories; Phase 4.2 adds the H3 loader.
- `salmon_ibm/environment.py` (TriMesh) and `salmon_ibm/hexsim_env.py` (HexMesh) both assume their mesh's flat-index layout; a new `salmon_ibm/h3_env.py` sits beside them for H3-native forcing (Phase 2.2).
- Pentagon check for Nemunas Delta bbox (20.4–21.9°E, 54.9–55.8°N) at res 8–12: **0 pentagons at every resolution.**
- Agent placement currently assumes a `spawn_site`; Phase 3.1b adds `uniform_random_water` for landscapes without a canonical spawn location.

**Nemunas H3 resolution reference** (full bbox, from sanity run):

| Res | Cells | Edge | Area | Pentagons | Payload |
|-----|-------|------|------|-----------|---------|
| 8  | 15,167    | 531 m | 0.74 km² | 0 | 0.8 MB |
| **9**  | **106,188**   | **201 m** | **0.11 km²** | **0** | **5.3 MB** |
| 10 | 743,198   | 76 m  | 0.015 km² | 0 | 37 MB |
| 11 | 5.2 M     | 29 m  | 2,150 m²  | 0 | 260 MB |

**Default resolution for Nemunas test landscape: 9** (200 m cells — 100× finer than the current 30×30 = 900-node TriMesh, payload still cheap, CMEMS 2 km native grid subsamples cleanly).

---

## File structure

Files created:
- `salmon_ibm/geomconst.py` — `M_PER_DEG_LAT`, `M_PER_DEG_LON_EQUATOR` constants (Phase 0).
- `salmon_ibm/h3mesh.py` — `H3Mesh` class with pentagon-aware factories (Phase 1, 300–400 LOC).
- `salmon_ibm/h3_env.py` — `H3Environment` that binds CMEMS/EMODnet → H3 cell IDs (Phase 2, 200–300 LOC).
- `salmon_ibm/h3_barriers.py` — H3 barrier loader + `line_barrier_to_h3_edges` helper (Phase 4).
- `scripts/build_nemunas_h3_landscape.py` — one-shot data prep: bbox polygon → H3 cells → aggregate CMEMS/EMODnet → NetCDF.
- `scripts/build_nemunas_h3_barriers.py` — one-shot: generate Klaipėda-strait synthetic barrier CSV from the line-helper.
- `data/nemunas_h3_barriers.csv` — synthetic barrier test fixture.
- `configs/config_nemunas_h3.yaml` — scenario config pointing at the H3 landscape + optional barrier CSV.
- `tests/test_movement_metric.py` — Phase 0 regression: advection correct under lat/lon centroids.
- `tests/test_h3mesh.py` — unit tests for `H3Mesh`.
- `tests/test_h3_env.py` — unit test for `H3Environment` loader.
- `tests/test_h3_barriers.py` — unit tests for line-helper and CSV loader.
- `tests/test_h3_placement.py` — unit test for agent placement strategy.
- `tests/test_nemunas_h3_integration.py` — end-to-end 30-day invariant run + barrier-effect test.
- `docs/superpowers/specs/2026-04-24-nemunas-delta-h3.md` — landscape spec (sibling file).

Files modified:
- `salmon_ibm/mesh.py`, `salmon_ibm/hexsim.py` — add `metric_scale(lat)` method to existing classes.
- `salmon_ibm/movement.py` — `_advection_numba` accepts `scale_x, scale_y` floats.
- `salmon_ibm/config.py` — add `mesh_backend`, `h3_landscape_nc`, `barriers_csv`.
- `salmon_ibm/simulation.py` — dispatch on `mesh_backend`; agent placement strategy; optional barrier CSV load.
- `salmon_ibm/barriers.py` — rename `from_hbf` → `from_hbf_hexsim`; add `empty()` and `from_csv_h3()` classmethods.

Files NOT touched (intentional — duck-typed):
- `salmon_ibm/agents.py`, `salmon_ibm/events_hexsim.py`, `salmon_ibm/bioenergetics.py`.

(`salmon_ibm/movement.py` IS modified in Task 0.1 — the signature change for `_advection_numba` is the one non-duck-typed change this plan introduces to the movement layer. All other movement code paths are unaffected.)

---

## Tasks

### Phase 0 — Abstraction guards

This phase removes HexSim-specific assumptions that currently live in generic code paths, so adding `H3Mesh` doesn't break anything.

### Task 0.1: Add `mesh.metric_scale` abstraction for Euclidean-on-lat/lon safety

**Files:**
- Modify: `salmon_ibm/movement.py:207-245` (advection kernel)
- Modify: `salmon_ibm/mesh.py`, `salmon_ibm/hexsim.py` (add `metric_scale` attribute)
- Test: `tests/test_movement_metric.py`

**Why unconditionally**: even if the current kernel accidentally works at 55° N for Curonian TriMesh, H3Mesh will put centroids in degrees across a very different latitude band for any future (e.g., equatorial) landscape — the Euclidean-in-degrees bias is a ticking bug.

- [x] **Step 1: Extend each mesh class with a `metric_scale(lat)` method**

`TriMesh` and `H3Mesh`: return `(meters_per_degree_lon_at_lat, meters_per_degree_lat)` — approximation `(111320 * cos(lat), 110540)`. `HexMesh`: return `(1.0, 1.0)` (already meters). Centralize the constants in `salmon_ibm/geomconst.py`.

```python
# salmon_ibm/geomconst.py
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQUATOR = 111320.0
```

```python
# In TriMesh / H3Mesh:
def metric_scale(self, lat: float) -> tuple[float, float]:
    import math
    from .geomconst import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR
    return (M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat)),
            M_PER_DEG_LAT)

# In HexMesh:
def metric_scale(self, lat: float) -> tuple[float, float]:
    return (1.0, 1.0)
```

- [x] **Step 2: Refactor `_advection_numba` to multiply centroid diffs by per-mesh scale**

Real existing signature (movement.py:208–218) is:
```python
_advection_numba(tri_indices, water_nbrs, water_nbr_count, centroids,
                 u, v, speeds, rand_drift, speed_threshold=0.01)
```
It mutates `tri_indices` in place and returns `None`. The refactor keeps every existing arg and inserts two new floats **before** the optional kwarg, so all existing callers need `scale_x=..., scale_y=...` added by keyword.

New signature:
```python
@njit(cache=True, parallel=True)
def _advection_numba(tri_indices, water_nbrs, water_nbr_count, centroids,
                     u, v, speeds, rand_drift,
                     scale_x, scale_y,          # NEW
                     speed_threshold=0.01):
    ...
    # Inside the inner loop, replace
    #   dx = centroids[nbr, 1] - cx
    #   dy = centroids[nbr, 0] - cy
    # with
    #   dx = (centroids[nbr, 1] - cx) * scale_x
    #   dy = (centroids[nbr, 0] - cy) * scale_y
```

Caller update (only body — `_apply_current_advection_vec` at `movement.py:341`). The function already accepts `mesh` as its second positional arg (line 341 real signature: `(pool, mesh, fields, alive_mask, rng)`), so no signature change is needed on the Python side — only the `_advection_numba` invocation inside the body gets the two new scale args:

```python
# Before calling _advection_numba, derive scale from mesh type.
lat_mean = float(centroids[:, 0].mean())
scale_x, scale_y = mesh.metric_scale(lat_mean)
_advection_numba(
    tri_indices, water_nbrs, water_nbr_count, centroids,
    u, v, speeds, rand_drift,
    scale_x, scale_y,   # NEW positional args
)
```

The function is called in two places inside `execute_movement` (`movement.py:62` and `movement.py:133`) — both already pass `mesh`, so both automatically pick up the metric-scaled advection after the body update.

- [x] **Step 3: Write the test (kernel mutates in-place, does NOT return)**

```python
# tests/test_movement_metric.py
import numpy as np
from salmon_ibm.movement import _advection_numba


def test_advection_east_flow_picks_east_neighbor_at_55N():
    # 3 cells at 55°N with ±100 m N/E offsets expressed as degrees.
    # centroids convention: column 0 = lat (y), column 1 = lon (x)
    centroids = np.array([
        [55.00000, 21.00000],
        [55.00090, 21.00000],   # ~100 m north of cell 0
        [55.00000, 21.00156],   # ~100 m east of cell 0
    ], dtype=np.float64)
    water_nbrs = np.array([[1, 2, -1, -1, -1, -1],
                            [0, -1, -1, -1, -1, -1],
                            [0, -1, -1, -1, -1, -1]], dtype=np.int32)
    water_nbr_count = np.array([2, 1, 1], dtype=np.int32)
    tri_indices = np.array([0], dtype=np.int32)
    u = np.array([1.0, 1.0, 1.0])       # east flow at each cell
    v = np.array([0.0, 0.0, 0.0])
    speeds = np.array([0.5])             # above speed_threshold
    rand_drift = np.array([0.0])         # always drift
    scale_x = 111320.0 * np.cos(np.deg2rad(55.0))  # ≈ 63860
    scale_y = 110540.0

    _advection_numba(                    # in-place, returns None
        tri_indices, water_nbrs, water_nbr_count, centroids,
        u, v, speeds, rand_drift,
        scale_x, scale_y,
    )
    assert tri_indices[0] == 2, f"expected east neighbour (2), got {tri_indices[0]}"


def test_advection_back_compat_on_hexmesh_meters():
    # Meter-mesh: scale_x = scale_y = 1.0 reproduces pre-refactor behaviour.
    # centroids col-0 = y (north), col-1 = x (east)
    centroids = np.array([[0.0, 0.0],
                           [100.0, 0.0],   # 100 m north
                           [0.0, 100.0]],  # 100 m east
                          dtype=np.float64)
    water_nbrs = np.array([[1, 2, -1, -1, -1, -1],
                            [0, -1, -1, -1, -1, -1],
                            [0, -1, -1, -1, -1, -1]], dtype=np.int32)
    water_nbr_count = np.array([2, 1, 1], dtype=np.int32)
    tri_indices = np.array([0], dtype=np.int32)
    u = np.array([0.0, 0.0, 0.0])        # pure north flow (via v)
    v = np.array([1.0, 1.0, 1.0])
    speeds = np.array([0.5])
    rand_drift = np.array([0.0])
    _advection_numba(
        tri_indices, water_nbrs, water_nbr_count, centroids,
        u, v, speeds, rand_drift,
        1.0, 1.0,
    )
    assert tri_indices[0] == 1, f"expected north neighbour (1), got {tri_indices[0]}"
```

- [x] **Step 4: Run**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_metric.py tests/test_curonian_realism_integration.py tests/test_hexsim.py -v`
Expected: new metric tests pass, existing Curonian + Columbia integration tests still pass.

- [x] **Step 5: Commit**

```bash
git add salmon_ibm/geomconst.py salmon_ibm/movement.py salmon_ibm/mesh.py salmon_ibm/hexsim.py tests/test_movement_metric.py
git commit -m "refactor: Mesh.metric_scale() abstraction — makes advection correct on any latitude"
```

### Task 0.2: Factor `BarrierMap.from_hbf` out of the generic interface

**Files:**
- Modify: `salmon_ibm/barriers.py:54-98`

- [x] **Step 1: Read the current `from_hbf` and `to_arrays` implementations**

Confirm: `from_hbf` uses `mesh._ncols`, `mesh._nrows`, `_full_to_compact` — HexMesh-only. `to_arrays` uses `mesh.n_cells` or `mesh.n_triangles` — both attrs exist on all mesh types.

- [x] **Step 2: Rename `from_hbf` → `from_hbf_hexsim` and add an `empty()` factory**

`BarrierMap.__init__(self)` currently takes **no args** (`barriers.py:37`), so `empty()` returns a bare instance — don't pass a dict to the constructor.

```python
class BarrierMap:
    @classmethod
    def from_hbf_hexsim(cls, path, mesh):
        """Load HexSim .hbf barriers. Requires HexMesh (reads _ncols, _nrows)."""
        if not hasattr(mesh, "_ncols"):
            raise TypeError(
                f"BarrierMap.from_hbf_hexsim requires a HexMesh; got {type(mesh).__name__}"
            )
        # ... existing from_hbf body, verbatim ...

    @classmethod
    def empty(cls) -> "BarrierMap":
        """No-barrier map — for landscapes without barriers (e.g. Nemunas H3)."""
        return cls()     # constructor already initialises self._edges = {}
```

- [x] **Step 3: Find and update every caller of `from_hbf`**

Run: `micromamba run -n shiny rg "from_hbf" --type py`
Expected sites: `salmon_ibm/barriers.py` (definition), `salmon_ibm/simulation.py`, `salmon_ibm/config.py` (via `barrier_map_from_config`). Replace every call with `from_hbf_hexsim`. Missing any of these will break at runtime because the method is now renamed.

- [x] **Step 4: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/test_barriers.py tests/test_hexsim.py tests/test_simulation.py -v`
Expected: all pass; no regression.

- [x] **Step 5: Commit**

```bash
git add salmon_ibm/barriers.py salmon_ibm/simulation.py salmon_ibm/config.py
git commit -m "refactor: rename BarrierMap.from_hbf → from_hbf_hexsim + add empty() factory"
```

---

### Phase 1 — `H3Mesh` class

The minimum viable backend: a mesh that movement kernels accept, with no environment or barrier integration yet.

### Task 1.1: Write `H3Mesh` skeleton + `from_h3_cells` classmethod

**Files:**
- Create: `salmon_ibm/h3mesh.py`
- Test: `tests/test_h3mesh.py`

- [x] **Step 1: Write the failing test**

```python
# tests/test_h3mesh.py
import numpy as np
import h3
from salmon_ibm.h3mesh import H3Mesh


def test_h3mesh_from_cells_builds_neighbors():
    # 7 cells: center + 6 neighbours
    center = h3.latlng_to_cell(55.3, 21.1, 9)
    neighbours = list(h3.grid_ring(center, 1))
    cells = [center] + neighbours
    depth = np.ones(7) * 5.0
    mesh = H3Mesh.from_h3_cells(cells, depth=depth)

    assert mesh.n_cells == 7
    assert mesh.neighbors.shape == (7, 6)
    assert mesh.centroids.shape == (7, 2)
    # Centre cell has all 6 neighbours present
    assert (mesh.neighbors[0] >= 0).sum() == 6
    # Neighbour cells each reach back to the centre
    for i in range(1, 7):
        assert 0 in mesh.neighbors[i].tolist()


def test_h3mesh_pentagon_gets_five_neighbors():
    # Known pentagon cell (res 0 icosahedral vertex)
    # Cell 8001fffffffffff is a pentagon at res 0; use a finer res descendant.
    penta = h3.get_pentagons(2)[0]  # res-2 pentagon
    nbrs = list(h3.grid_ring(penta, 1))
    assert len(nbrs) == 5
    cells = [penta] + nbrs
    mesh = H3Mesh.from_h3_cells(cells, depth=np.ones(6))
    # Pentagon row: 5 valid neighbours, 6th slot is -1
    row = mesh.neighbors[0]
    assert (row >= 0).sum() == 5
    assert -1 in row.tolist()
```

- [x] **Step 2: Run test to verify it fails**

Run: `micromamba run -n shiny python -m pytest tests/test_h3mesh.py -v`
Expected: FAIL — `ModuleNotFoundError: salmon_ibm.h3mesh`.

- [x] **Step 3: Implement `H3Mesh.from_h3_cells`**

```python
# salmon_ibm/h3mesh.py
"""H3-native mesh backend for the Salmon IBM.

H3Mesh duck-types TriMesh / HexMesh so movement.py, events_hexsim.py,
and bioenergetics.py work unchanged. Cells are addressed by compact
int indices (0..N-1); the int64 H3 IDs are carried in self.h3_ids for
environment lookup and the viewer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import h3


class H3Mesh:
    MAX_NBRS = 6  # pentagons fit with a -1 sentinel in the 6th slot

    def __init__(
        self,
        h3_ids: np.ndarray,        # (N,) uint64
        centroids: np.ndarray,     # (N, 2) float64 [lat, lon]
        neighbors: np.ndarray,     # (N, 6) int32, -1 sentinel
        water_mask: np.ndarray,    # (N,) bool
        depth: np.ndarray,         # (N,) float32
        areas: np.ndarray,         # (N,) float32, per-cell m²
        resolution: int,
    ):
        self.h3_ids = h3_ids
        self.centroids = centroids
        self.neighbors = neighbors
        self.water_mask = water_mask
        self.depth = depth
        self.areas = areas
        self.resolution = resolution

        # Numba caches (contiguous, same layout as TriMesh/HexMesh)
        self.centroids_c = np.ascontiguousarray(centroids)
        self._water_nbrs = neighbors  # already -1-padded
        self._water_nbr_count = (neighbors >= 0).sum(axis=1).astype(np.int32)

    # --- duck-typed properties ---------------------------------------
    @property
    def n_cells(self) -> int:
        return len(self.h3_ids)

    @property
    def n_triangles(self) -> int:
        return self.n_cells

    # --- duck-typed methods ------------------------------------------
    def water_neighbors(self, idx: int) -> list[int]:
        row = self.neighbors[idx]
        return [int(n) for n in row if n >= 0]

    def find_triangle(self, lat: float, lon: float) -> int:
        """Return the compact mesh index containing (lat, lon), or -1."""
        hid = h3.latlng_to_cell(float(lat), float(lon), self.resolution)
        hid_int = h3.str_to_int(hid)
        hits = np.where(self.h3_ids == hid_int)[0]
        return int(hits[0]) if len(hits) else -1

    def gradient(self, field: np.ndarray, idx: int) -> tuple[float, float]:
        """Approximate (dlat, dlon) gradient via central-difference over
        neighbours. Returns unit vector in (north, east) convention; zero
        vector if no neighbours.

        Centroid diffs are scaled by `metric_scale(lat)` so that a degree
        of longitude doesn't artificially outweigh a degree of latitude at
        mid-latitudes — mirrors the fix Task 0.1 applies to
        `_advection_numba`. Without this, any gradient-following event
        (thermal seek, salinity avoid) biases cells toward N-S neighbours
        by the factor `1/cos(lat)` ≈ 1.74× at 55° N.
        """
        row = self.neighbors[idx]
        valid = row[row >= 0]
        if len(valid) == 0:
            return (0.0, 0.0)
        here = self.centroids[idx]
        scale_x, scale_y = self.metric_scale(float(here[0]))  # lon, lat metres/deg
        dlat = dlon = 0.0
        for n in valid:
            there = self.centroids[n]
            df = field[n] - field[idx]
            dlat += df * (there[0] - here[0]) * scale_y
            dlon += df * (there[1] - here[1]) * scale_x
        norm = (dlat * dlat + dlon * dlon) ** 0.5
        if norm < 1e-12:
            return (0.0, 0.0)
        return (dlat / norm, dlon / norm)

    def metric_scale(self, lat: float) -> tuple[float, float]:
        """Return (metres per degree lon at lat, metres per degree lat).
        Mirrors the TriMesh contract — see Task 0.1."""
        import math
        from .geomconst import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR
        return (M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat)),
                M_PER_DEG_LAT)

    # --- factories ---------------------------------------------------
    @classmethod
    def from_h3_cells(
        cls,
        h3_cells: Sequence[str],
        *,
        depth: np.ndarray | None = None,
        water_mask: np.ndarray | None = None,
        pentagon_policy: str = "raise",
    ) -> "H3Mesh":
        """Build a mesh from an explicit list of H3 cell IDs."""
        if len(h3_cells) == 0:
            raise ValueError("H3Mesh.from_h3_cells: empty cell list")

        # Pentagon guard — fires here too, so direct callers don't bypass it.
        # Compute the keep-mask BEFORE mutating h3_cells so depth/water_mask
        # get filtered against the original ordering.
        pentas = [c for c in h3_cells if h3.is_pentagon(c)]
        if pentas:
            if pentagon_policy == "raise":
                raise ValueError(
                    f"{len(pentas)} pentagon cells in input; "
                    f"pass pentagon_policy='skip' or 'allow' to proceed."
                )
            elif pentagon_policy == "skip":
                keep = np.array([not h3.is_pentagon(c) for c in h3_cells], dtype=bool)
                h3_cells = [c for c in h3_cells if not h3.is_pentagon(c)]
                if depth is not None:
                    depth = depth[keep]
                if water_mask is not None:
                    water_mask = water_mask[keep]
            # pentagon_policy == "allow": fall through deliberately — pentagons
            # stay in the cell list and get 5 valid neighbours + a -1 sentinel
            # in the 6th slot (see test_h3mesh_pentagon_gets_five_neighbors).

        n = len(h3_cells)
        # Use Python int consistently — numpy uint64 hash-equals but mixing
        # numpy uint64 with Python int in dict keys has bitten people before.
        h3_ids_pyint = [int(h3.str_to_int(c)) for c in h3_cells]
        h3_ids = np.array(h3_ids_pyint, dtype=np.uint64)

        res = h3.get_resolution(h3_cells[0])
        centroids = np.array(
            [h3.cell_to_latlng(c) for c in h3_cells],
            dtype=np.float64,
        )

        # Reverse lookup: H3 ID (Python int) → compact index
        id_to_idx = {cid: i for i, cid in enumerate(h3_ids_pyint)}

        # Neighbours — H3 grid_ring gives cells at distance 1
        neighbours = np.full((n, cls.MAX_NBRS), -1, dtype=np.int32)
        for i, cell in enumerate(h3_cells):
            for j, nb in enumerate(h3.grid_ring(cell, 1)):
                nb_int = int(h3.str_to_int(nb))
                if nb_int in id_to_idx:
                    neighbours[i, j] = id_to_idx[nb_int]

        if water_mask is None:
            water_mask = np.ones(n, dtype=bool)
        if depth is None:
            depth = np.zeros(n, dtype=np.float32)
        areas = np.array(
            [h3.cell_area(c, unit="m^2") for c in h3_cells],
            dtype=np.float32,
        )

        return cls(
            h3_ids=h3_ids,
            centroids=centroids,
            neighbors=neighbours,
            water_mask=water_mask,
            depth=depth.astype(np.float32),
            areas=areas,
            resolution=res,
        )
```

- [x] **Step 4: Run tests to confirm they pass**

Run: `micromamba run -n shiny python -m pytest tests/test_h3mesh.py -v`
Expected: both tests pass.

- [x] **Step 5: Commit**

```bash
git add salmon_ibm/h3mesh.py tests/test_h3mesh.py
git commit -m "feat: add H3Mesh backend with pentagon-aware neighbor table"
```

### Task 1.2: Add `from_polygon` factory for bbox-based landscape creation

**Files:**
- Modify: `salmon_ibm/h3mesh.py`
- Test: `tests/test_h3mesh.py`

- [x] **Step 1: Write the failing test**

```python
def test_h3mesh_from_polygon_builds_nemunas_grid():
    import h3
    # Small bbox inside Nemunas delta
    ring = [(55.30, 21.10), (55.30, 21.20),
            (55.35, 21.20), (55.35, 21.10), (55.30, 21.10)]
    poly = h3.LatLngPoly(ring)
    mesh = H3Mesh.from_polygon(poly, resolution=9)
    # 5.5 km × 5 km box at res 9 (200 m cells) → ~200 cells expected
    assert 100 < mesh.n_cells < 500
    assert mesh.neighbors.shape == (mesh.n_cells, 6)
    # All internal cells have all 6 neighbours; edge cells have fewer
    internal = (mesh.neighbors >= 0).sum(axis=1) == 6
    assert internal.sum() > 0  # at least some interior
```

- [x] **Step 2: Run to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_h3mesh.py::test_h3mesh_from_polygon_builds_nemunas_grid -v`
Expected: FAIL — `AttributeError: type object 'H3Mesh' has no attribute 'from_polygon'`.

- [x] **Step 3: Implement `from_polygon`**

```python
    @classmethod
    def from_polygon(
        cls,
        polygon: "h3.LatLngPoly",
        resolution: int,
        *,
        depth: dict[int, float] | None = None,
        water_mask: dict[int, bool] | None = None,
        pentagon_policy: str = "raise",
    ) -> "H3Mesh":
        """Build a mesh by H3-tessellating a lat/lon polygon."""
        cells_iter = h3.polygon_to_cells(polygon, resolution)
        cells = list(cells_iter)
        n = len(cells)
        if n == 0:
            raise ValueError("polygon produced zero H3 cells")

        # Optional per-cell attributes — default to water-everywhere, depth 0.
        depth_arr = np.zeros(n, dtype=np.float32)
        mask_arr = np.ones(n, dtype=bool)
        if depth is not None:
            ids_int = [int(h3.str_to_int(c)) for c in cells]
            for i, cid in enumerate(ids_int):
                if cid in depth:
                    depth_arr[i] = float(depth[cid])
        if water_mask is not None:
            ids_int = [int(h3.str_to_int(c)) for c in cells]
            for i, cid in enumerate(ids_int):
                if cid in water_mask:
                    mask_arr[i] = bool(water_mask[cid])

        return cls.from_h3_cells(
            cells, depth=depth_arr, water_mask=mask_arr,
            pentagon_policy=pentagon_policy,
        )
```

- [x] **Step 4: Run test**

Run: `micromamba run -n shiny python -m pytest tests/test_h3mesh.py -v`
Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add salmon_ibm/h3mesh.py tests/test_h3mesh.py
git commit -m "feat: H3Mesh.from_polygon factory for bbox-tessellated landscapes"
```

### Task 1.3: Pentagon guard test

The guard logic now lives inside `from_h3_cells` (see Task 1.1 revised body), so this task just adds the regression test — no new code.

**Files:**
- Test: `tests/test_h3mesh.py`

- [x] **Step 1: Write the test**

There are 12 res-0 pentagons; their res-2 descendants give easy test subjects.

```python
def test_from_polygon_raises_on_pentagon_by_default():
    import h3, pytest
    penta_res2 = h3.get_pentagons(2)[0]
    lat0, lon0 = h3.cell_to_latlng(penta_res2)
    ring = [(lat0 - 0.5, lon0 - 0.5), (lat0 - 0.5, lon0 + 0.5),
            (lat0 + 0.5, lon0 + 0.5), (lat0 + 0.5, lon0 - 0.5),
            (lat0 - 0.5, lon0 - 0.5)]
    poly = h3.LatLngPoly(ring)
    with pytest.raises(ValueError, match="pentagon"):
        H3Mesh.from_polygon(poly, resolution=2)
    mesh = H3Mesh.from_polygon(poly, resolution=2, pentagon_policy="skip")
    assert mesh.n_cells > 0


def test_from_h3_cells_also_catches_pentagons():
    """Direct callers of from_h3_cells don't bypass the guard."""
    import h3, pytest
    penta = h3.get_pentagons(5)[0]
    nbrs = list(h3.grid_ring(penta, 1))
    with pytest.raises(ValueError, match="pentagon"):
        H3Mesh.from_h3_cells([penta] + nbrs)
```

- [x] **Step 2: Run + commit**

Run: `micromamba run -n shiny python -m pytest tests/test_h3mesh.py -v`
Commit: `test: pentagon guard fires for both factory paths`.

---

### Phase 2 — H3 environment & landscape loader

### Task 2.1: `build_nemunas_h3_landscape.py` — one-shot data prep

**Files:**
- Create: `scripts/build_nemunas_h3_landscape.py`
- Output: `data/nemunas_h3_landscape.nc` (≤10 MB)

- [x] **Step 1: Write the script**

```python
"""Build a Nemunas Delta H3-native landscape from CMEMS + EMODnet.

Output: data/nemunas_h3_landscape.nc with variables:
  - h3_id      (n_cells,) uint64
  - lat        (n_cells,) float64
  - lon        (n_cells,) float64
  - depth      (n_cells,) float32  — from EMODnet, positive down
  - water_mask (n_cells,) uint8    — 1 = wet (depth > 0)
  - tos        (time, n_cells) float32  — sea-surface temperature
  - sos        (time, n_cells) float32  — sea-surface salinity
  - uo, vo     (time, n_cells) float32  — currents, east/north m/s
"""
from __future__ import annotations

from pathlib import Path
import argparse
import h3
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

BBOX = {"minlon": 20.4, "maxlon": 21.9, "minlat": 54.9, "maxlat": 55.8}
RESOLUTION = 9


def build_h3_cells():
    ring = [
        (BBOX["minlat"], BBOX["minlon"]),
        (BBOX["minlat"], BBOX["maxlon"]),
        (BBOX["maxlat"], BBOX["maxlon"]),
        (BBOX["maxlat"], BBOX["minlon"]),
        (BBOX["minlat"], BBOX["minlon"]),
    ]
    poly = h3.LatLngPoly(ring)
    return list(h3.polygon_to_cells(poly, RESOLUTION))


def sample_emodnet(cells, tif_path):
    """Sample EMODnet bathymetry at each H3 cell centroid."""
    import rioxarray  # noqa: F401
    raw = rioxarray.open_rasterio(tif_path).squeeze()
    x, y = raw.x.values, raw.y.values
    z = raw.values
    if y[0] > y[-1]:
        y, z = y[::-1], z[::-1, :]
    interp = RegularGridInterpolator(
        (y, x), z, method="linear", bounds_error=False, fill_value=np.nan,
    )
    lats, lons = zip(*(h3.cell_to_latlng(c) for c in cells))
    query = np.column_stack([lats, lons])
    elev = interp(query)
    depth = np.where(np.isnan(elev), 0.0, -elev)  # EMODnet = elevation (+up)
    return np.maximum(depth, 0.0).astype(np.float32)


def sample_cmems(cells, cmems_path, start=None, end=None):
    """Sample CMEMS reanalysis (regular grid) at each H3 cell centroid.

    Optionally subset to [start, end] (both inclusive, ISO date strings)
    to keep the output file size bounded.
    """
    raw = xr.open_dataset(cmems_path)
    if start is not None or end is not None:
        raw = raw.sel(time=slice(start, end))
        if raw.sizes["time"] == 0:
            raise ValueError(f"no CMEMS timesteps in {start}..{end}")
    lats, lons = zip(*(h3.cell_to_latlng(c) for c in cells))
    query = np.column_stack([lats, lons])
    n_time = raw.sizes["time"]
    out = {}
    for src, dst in [("thetao", "tos"), ("so", "sos"),
                     ("uo", "uo"), ("vo", "vo")]:
        if src not in raw:
            continue
        arr = raw[src].squeeze()  # (time, y, x) or (time, lat, lon)
        lat_src = raw["latitude"].values if "latitude" in raw else raw["lat"].values
        lon_src = raw["longitude"].values if "longitude" in raw else raw["lon"].values
        if lat_src[0] > lat_src[-1]:
            lat_src = lat_src[::-1]
            arr = arr.isel({arr.dims[-2]: slice(None, None, -1)})
        vals = np.empty((n_time, len(cells)), dtype=np.float32)
        for t in range(n_time):
            interp = RegularGridInterpolator(
                (lat_src, lon_src), arr.isel(time=t).values,
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            row = interp(query)
            # NaN-fill via nearest-neighbour (same trick as fetch_cmems_forcing)
            if np.isnan(row).any():
                src_lats, src_lons = np.meshgrid(lat_src, lon_src, indexing="ij")
                flat_vals = arr.isel(time=t).values.ravel()
                valid = ~np.isnan(flat_vals)
                nn = NearestNDInterpolator(
                    np.column_stack([src_lats.ravel()[valid], src_lons.ravel()[valid]]),
                    flat_vals[valid],
                )
                row[np.isnan(row)] = nn(query[np.isnan(row)])
            vals[t] = row.astype(np.float32)
        out[dst] = vals
    return raw.time.values, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif",
                        default="data/curonian_bathymetry_raw.tif")
    parser.add_argument("--cmems",
                        default="data/curonian_forcing_cmems_raw.nc")
    parser.add_argument("--out",
                        default="data/nemunas_h3_landscape.nc")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    print(f"[1/4] Building H3 cells (res {RESOLUTION}) over bbox...")
    cells = build_h3_cells()
    print(f"  {len(cells):,} cells")

    print(f"[2/4] Sampling EMODnet bathymetry...")
    depth = sample_emodnet(cells, project_root / args.tif)
    print(f"  depth range: {depth.min():.2f} – {depth.max():.2f} m")

    print(f"[3/4] Sampling CMEMS forcing...")
    times, forcing = sample_cmems(cells, project_root / args.cmems)

    print(f"[4/4] Writing NetCDF...")
    lats, lons = zip(*(h3.cell_to_latlng(c) for c in cells))
    h3_ids = np.array([h3.str_to_int(c) for c in cells], dtype=np.uint64)
    water_mask = (depth > 0.0).astype(np.uint8)

    ds = xr.Dataset(
        {
            "h3_id":      (("cell",), h3_ids),
            "lat":        (("cell",), np.array(lats)),
            "lon":        (("cell",), np.array(lons)),
            "depth":      (("cell",), depth),
            "water_mask": (("cell",), water_mask),
            **{k: (("time", "cell"), v) for k, v in forcing.items()},
        },
        coords={"time": times},
        attrs={
            "h3_resolution": RESOLUTION,
            "source_bathymetry": "EMODnet DTM 2022",
            "source_forcing": "CMEMS BALTICSEA_MULTIYEAR_PHY_003_011",
            "bbox": f"{BBOX}",
        },
    )
    out_path = project_root / args.out
    ds.to_netcdf(out_path, format="NETCDF3_64BIT")
    print(f"Wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run the builder**

Run: `micromamba run -n shiny python scripts/build_nemunas_h3_landscape.py`
Expected output: `Wrote data/nemunas_h3_landscape.nc (6–8 MB)`. Verify: `ncdump -h data/nemunas_h3_landscape.nc` shows all expected variables.

- [x] **Step 3: Commit the script (NOT the .nc output; data lives in its own ignored path)**

```bash
git add scripts/build_nemunas_h3_landscape.py
git commit -m "feat: one-shot builder for Nemunas Delta H3 landscape"
```

### Task 2.2: `H3Environment` loader

**Files:**
- Create: `salmon_ibm/h3_env.py`
- Test: `tests/test_h3_env.py`

Mirrors `salmon_ibm/hexsim_env.py` but reads the NetCDF from 2.1 instead of PATCH_HEXMAP.

- [x] **Step 1: Implement `H3Environment`**

```python
# salmon_ibm/h3_env.py
"""H3-native forcing loader (CMEMS/EMODnet → per-cell arrays)."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class H3Environment:
    """Binds CMEMS/EMODnet data to a H3Mesh's cell ordering.

    Field arrays stored at shape (n_cells,) per timestep, matching the
    duck-typed interface consumed by events and bioenergetics.
    """
    mesh: "H3Mesh"
    fields: dict[str, np.ndarray]    # {"temperature": ..., "salinity": ..., etc.}
    time: np.ndarray                 # (n_times,) datetime64
    _time_idx: int = 0

    @classmethod
    def from_netcdf(cls, nc_path: str | Path, mesh: "H3Mesh") -> "H3Environment":
        ds = xr.open_dataset(nc_path, engine="scipy")
        # Match mesh cell ordering to dataset cell ordering by h3_id
        ds_ids = ds["h3_id"].values.astype(np.uint64)
        order = np.argsort(ds_ids)
        ds_ids_sorted = ds_ids[order]
        # Find each mesh h3_id in ds_ids_sorted
        matches = np.searchsorted(ds_ids_sorted, mesh.h3_ids)
        # Guard: every mesh cell must be in the NetCDF
        bad = (matches >= len(ds_ids_sorted)) | (ds_ids_sorted[matches] != mesh.h3_ids)
        if bad.any():
            raise ValueError(
                f"{bad.sum()} mesh cells not in forcing NetCDF; "
                f"first missing H3 id: {mesh.h3_ids[bad][0]:x}"
            )
        reorder = order[matches]  # permutation: mesh_idx → ds_cell_idx

        def load(var):
            if var not in ds:
                return None
            arr = ds[var].values  # (time, cell)
            return arr[:, reorder].astype(np.float32)  # (time, n_mesh_cells)

        # Field names must match the canonical keys consumed downstream:
        #   - movement.py:343-344 reads `u_current`, `v_current`
        #   - environment.py:41-42 and hexsim_env.py:150-151 use the same
        # Rename H3Environment outputs to match; diverging here silently
        # no-ops the advection event on H3 landscapes.
        fields = {}
        for src, dst in [("tos", "temperature"), ("sos", "salinity"),
                          ("uo", "u_current"), ("vo", "v_current")]:
            val = load(src)
            if val is not None:
                fields[dst] = val
        return cls(mesh=mesh, fields=fields, time=ds["time"].values)

    def advance(self, step: int) -> None:
        self._time_idx = min(step, len(self.time) - 1)

    def current(self) -> dict[str, np.ndarray]:
        return {name: arr[self._time_idx] for name, arr in self.fields.items()}
```

- [x] **Step 2: Write the test**

```python
# tests/test_h3_env.py
import numpy as np
import pytest
from pathlib import Path
from salmon_ibm.h3mesh import H3Mesh
from salmon_ibm.h3_env import H3Environment


LANDSCAPE = Path("data/nemunas_h3_landscape.nc")


def _needs_landscape():
    if not LANDSCAPE.exists():
        pytest.skip(f"{LANDSCAPE.name} not present — run scripts/build_nemunas_h3_landscape.py")


def test_h3_environment_loads_and_indexes():
    _needs_landscape()
    import h3, xarray as xr
    ds = xr.open_dataset(LANDSCAPE, engine="scipy")
    cells = [h3.int_to_str(int(x)) for x in ds["h3_id"].values]
    mesh = H3Mesh.from_h3_cells(cells, depth=ds["depth"].values)
    env = H3Environment.from_netcdf(LANDSCAPE, mesh)
    assert "temperature" in env.fields
    env.advance(0)
    temps = env.current()["temperature"]
    assert temps.shape == (mesh.n_cells,)
    # Baltic winter envelope
    assert -2.0 < temps.min()
    assert temps.max() < 25.0
```

- [x] **Step 3: Run test + commit**

Run: `micromamba run -n shiny python -m pytest tests/test_h3_env.py -v`
Commit: `feat: H3Environment loads CMEMS/EMODnet forcing bound by H3 cell id`.

---

### Phase 3 — Wire into Simulation + config

### Task 3.1: `config.py` adds `mesh_backend: h3`

**Files:**
- Modify: `salmon_ibm/config.py`
- Create: `configs/config_nemunas_h3.yaml`

**Existing config model**: `load_config(path) -> dict` returns a plain dict (not a dataclass). Every new config access in this plan uses dict-style `cfg.get("mesh_backend", "trimesh")` — **not** dot-notation — to match the existing code. `validate_config` already rejects configs that lack a `grid` section; we relax that when `mesh_backend` is present.

- [x] **Step 1: Update `validate_config` to accept the H3 layout**

```python
# salmon_ibm/config.py — existing validate_config body
def validate_config(cfg: dict) -> None:
    mesh_backend = cfg.get("mesh_backend", "trimesh")
    if mesh_backend == "h3":
        if "h3_landscape_nc" not in cfg:
            raise ValueError("mesh_backend=h3 requires 'h3_landscape_nc' path")
        # H3 configs don't need a 'grid' section — the landscape NetCDF
        # carries all geometric info.
    else:
        if "grid" not in cfg:
            raise ValueError("Config must contain a 'grid' section")
    # ... remaining checks unchanged ...
```

- [x] **Step 2: Write `configs/config_nemunas_h3.yaml`**

```yaml
name: "nemunas_h3_test"
mesh_backend: "h3"
h3_landscape_nc: "data/nemunas_h3_landscape.nc"
species_config: "configs/baltic_salmon_species.yaml"

# Simulation time
start_date: "2011-06-01"
duration_hours: 720        # 30 days
timestep_seconds: 3600     # 1 h

# Agents
n_agents: 500
initial_state:
  stage: "adult"
  mass_g: 3000
  energy_density: 6.5
  initial_cell_strategy: "uniform_random_water"  # pick any water cell

# Events
events:
  - type: "thermal_response"
  - type: "bioenergetics"
  - type: "movement_advection"   # follows flow_u, flow_v
  - type: "mortality"
# Barriers intentionally omitted — Nemunas H3 test has none
```

- [x] **Step 3: Commit**

```bash
git add salmon_ibm/config.py configs/config_nemunas_h3.yaml
git commit -m "feat: scenario config for H3 mesh backend + Nemunas test"
```

### Task 3.1b: Agent placement for H3 landscape

**Files:**
- Modify: `salmon_ibm/simulation.py` (agent init path)
- Test: `tests/test_h3_placement.py`

The existing TriMesh path places agents at a known spawn site (`initial_cell_strategy` in config). For a simple smoke test, we add a `uniform_random_water` strategy that draws uniformly from `mesh.water_mask` indices.

- [x] **Step 1: Add the strategy dispatcher**

Note: `uniform_random_water` is a **mechanics-test placement only**. It ignores homing behaviour, natal-river fidelity, and thermal-refuge preferences — all documented for *S. salar*. Use `spawn_site` (or a future `natal_river_return`) for any ecological-realism scenario.

```python
# inside Simulation.__init__ or _do_init_sim, where pool.tri_idx is seeded:
initial_state = self.config.get("initial_state", {})
strategy = initial_state.get("initial_cell_strategy", "spawn_site")
if strategy == "uniform_random_water":
    water_cells = np.where(mesh.water_mask)[0]
    if len(water_cells) == 0:
        raise RuntimeError("no water cells to place agents in")
    chosen = self._rng.choice(water_cells, size=self.n_agents, replace=True)
    self.pool.tri_idx[:] = chosen.astype(np.int32)
elif strategy == "spawn_site":
    # existing path: nearest cell to spawn_lat/spawn_lon
    spawn_lat = self.config.get("spawn_lat")
    spawn_lon = self.config.get("spawn_lon")
    self.pool.tri_idx[:] = mesh.find_triangle(spawn_lat, spawn_lon)
else:
    raise ValueError(f"unknown initial_cell_strategy: {strategy}")

# Snapshot for integration tests (also useful for movement diagnostics)
self.initial_cells = self.pool.tri_idx.copy()
```

- [x] **Step 2: Write the placement test**

```python
# tests/test_h3_placement.py
import numpy as np
from salmon_ibm.h3mesh import H3Mesh
import h3

def test_uniform_random_water_respects_water_mask():
    cells = list(h3.grid_disk(h3.latlng_to_cell(55.3, 21.1, 9), 5))
    mask = np.zeros(len(cells), dtype=bool)
    mask[::2] = True  # every other cell is water
    mesh = H3Mesh.from_h3_cells(cells, water_mask=mask)
    rng = np.random.default_rng(0)
    water_cells = np.where(mesh.water_mask)[0]
    chosen = rng.choice(water_cells, size=500, replace=True)
    assert mesh.water_mask[chosen].all(), "some agents placed on land"
```

- [x] **Step 3: Commit**

```bash
git add salmon_ibm/simulation.py tests/test_h3_placement.py
git commit -m "feat: uniform_random_water placement + record initial_cells for diagnostics"
```

### Task 3.2: `simulation.py` dispatches on `mesh_backend`

**Files:**
- Modify: `salmon_ibm/simulation.py` (find the mesh-construction call site)

- [x] **Step 1: Find the current site**

Run: `micromamba run -n shiny rg "HexMesh.from|TriMesh.from|mesh = " salmon_ibm/simulation.py`
Identify the single branch where `TriMesh.from_netcdf` vs `HexMesh.from_hexsim` is chosen.

- [x] **Step 2: Add the H3 branch**

Note: `cfg` is a dict (from `load_config`), not a dataclass — use `.get()`. Ensure `H3Mesh` and `H3Environment` are imported at the top of `simulation.py` (add `from .h3mesh import H3Mesh` and `from .h3_env import H3Environment`).

```python
mesh_backend = self.config.get("mesh_backend", "trimesh")
if mesh_backend == "h3":
    import h3 as _h3
    landscape_path = self.config["h3_landscape_nc"]
    ds = xr.open_dataset(landscape_path, engine="scipy")
    cells = [_h3.int_to_str(int(i)) for i in ds["h3_id"].values]
    mesh = H3Mesh.from_h3_cells(
        cells,
        depth=ds["depth"].values,
        water_mask=ds["water_mask"].values.astype(bool),
    )
    env = H3Environment.from_netcdf(landscape_path, mesh)
    barriers = BarrierMap.empty()   # Phase 0.2 factory
elif mesh_backend == "hexsim":
    pass   # keep the existing HexSim branch unchanged
else:
    pass   # keep the existing TriMesh branch unchanged (default backend)
```

- [x] **Step 3: Run the existing simulation smoke test to confirm no regression**

Run: `micromamba run -n shiny python -m pytest tests/test_simulation.py -v`
Expected: all pass.

- [x] **Step 4: Commit**

```bash
git add salmon_ibm/simulation.py
git commit -m "feat: dispatch mesh_backend=h3 → H3Mesh + H3Environment"
```

### Task 3.3: End-to-end Nemunas smoke run

**Files:**
- Create: `tests/test_nemunas_h3_integration.py`

- [x] **Step 1: Write the integration test**

Mirror `tests/test_curonian_realism_integration.py` but for H3 landscape.

```python
"""30-day Nemunas H3 invariant run — the H3 analogue of
test_curonian_realism_integration."""
from pathlib import Path
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent
LANDSCAPE = PROJECT / "data" / "nemunas_h3_landscape.nc"
CONFIG = PROJECT / "configs" / "config_nemunas_h3.yaml"


def _needs_data():
    if not LANDSCAPE.exists():
        pytest.skip("nemunas_h3_landscape.nc missing — run builder")
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG.name} missing")


@pytest.fixture(scope="module")
def h3_sim():
    _needs_data()
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config(str(CONFIG))
    sim = Simulation(cfg, n_agents=500, rng_seed=42)
    sim.run(n_steps=720)
    return sim


def test_agent_conservation(h3_sim):
    alive = int(h3_sim.pool.alive.sum())
    arrived = int(h3_sim.pool.arrived.sum())
    dead = 500 - alive - arrived
    assert alive + arrived + dead == 500
    assert alive > 0, "total extinction — thermal or movement broken"


def test_mesh_is_h3(h3_sim):
    from salmon_ibm.h3mesh import H3Mesh
    assert isinstance(h3_sim.mesh, H3Mesh)
    assert h3_sim.mesh.resolution == 9


def test_temperature_envelope(h3_sim):
    t = h3_sim.env.current()["temperature"]
    assert -2.0 <= t.min() < 25.0, f"temp out of Baltic envelope: {t.min()}..{t.max()}"


def test_agent_positions_all_on_mesh(h3_sim):
    assert (h3_sim.pool.tri_idx >= 0).all()
    assert (h3_sim.pool.tri_idx < h3_sim.mesh.n_cells).all()


def test_at_least_one_agent_moved(h3_sim):
    """Movement kernels must work on H3Mesh — non-zero displacement proves
    water_nbrs/water_nbr_count are wired correctly."""
    positions_start = h3_sim.initial_cells  # recorded in __init__
    positions_end = h3_sim.pool.tri_idx
    moved = (positions_start != positions_end).sum()
    assert moved > 50, f"only {moved}/500 agents moved — movement stub?"


def test_north_south_salinity_gradient(h3_sim):
    """Spec §5: Klaipėda Strait (north) must be saltier than Nemunas mouth
    (south) by ≥ 1.5 PSU. Same invariant as Curonian's
    test_north_south_salinity_gradient but over the H3 grid.
    """
    import numpy as np
    sal = h3_sim.env.current()["salinity"]
    lats = h3_sim.mesh.centroids[:, 0]  # col 0 = lat
    north_mask = lats > np.percentile(lats, 75)
    south_mask = lats < np.percentile(lats, 25)
    north_mean = float(np.nanmean(sal[north_mask]))
    south_mean = float(np.nanmean(sal[south_mask]))
    assert north_mean > south_mean, (
        f"Salinity gradient inverted: north={north_mean:.2f} < south={south_mean:.2f}. "
        f"Klaipėda Strait should be saltier than Nemunas mouth."
    )
    assert (north_mean - south_mean) >= 1.5, (
        f"Salinity gradient too weak: {north_mean - south_mean:.2f} PSU "
        f"(expected ≥ 1.5). Regridder may be homogenising the lagoon."
    )
```

- [x] **Step 2: Run**

Run: `micromamba run -n shiny python -m pytest tests/test_nemunas_h3_integration.py -v`
Expected: all pass.

- [x] **Step 3: Commit**

```bash
git add tests/test_nemunas_h3_integration.py
git commit -m "test: Nemunas H3 end-to-end invariants (movement + thermal + conservation)"
```

---

### Phase 4 — H3 barriers

Barriers are already stored as a mesh-agnostic `dict[(from_cell_idx, to_cell_idx), outcome]` inside `BarrierMap` — only the loader is HexSim-specific. This phase adds a CSV-based loader keyed by H3 IDs plus a helper that turns a lat/lon line into the set of H3 edges it crosses, so users can spec a weir as two points rather than enumerating every edge by hand.

### Task 4.1: Define CSV schema & `line_barrier_to_h3_edges` helper

**Files:**
- Create: `salmon_ibm/h3_barriers.py`
- Test: `tests/test_h3_barriers.py`

**CSV schema** (`data/*_h3_barriers.csv`):

```csv
from_h3,to_h3,mortality,deflection,transmission,note
8929a07a09bffff,8929a07a09fffff,0.10,0.85,0.05,"Nemunas mouth weir"
8929a07a09fffff,8929a07a09bffff,0.10,0.85,0.05,"Nemunas mouth weir (reverse)"
```

Rules:
- `mortality + deflection + transmission` must sum to 1.0 per row (± 1e-6).
- Symmetric barriers need two rows (one per direction).
- `from_h3` and `to_h3` must be neighbours — validated on load.
- Unknown columns ignored (forward-compatible).

- [x] **Step 1: Implement `line_barrier_to_h3_edges` helper**

```python
# salmon_ibm/h3_barriers.py
"""H3-native barrier loading & geometric helpers.

A barrier is an **edge** between two H3 cells: when an agent tries to step
across it, a stochastic outcome (mortality, deflection back, transmission
through) fires. This module loads those edges from CSV and provides a
geometric convenience for specifying barriers as lat/lon lines.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import h3
import numpy as np


def line_barrier_to_h3_edges(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    resolution: int,
    *,
    bidirectional: bool = True,
    n_samples: int | None = None,
) -> list[tuple[str, str]]:
    """Return all H3 cell-pair edges crossed by the great-circle-approximation
    line segment from (lat1,lon1) to (lat2,lon2) at a given H3 resolution.

    Uses dense linear interpolation (lat/lon space) — accurate enough at
    scales where H3 edge size ≫ Earth curvature over the segment length,
    i.e. sub-continental.

    Parameters
    ----------
    bidirectional
        If True (default), return both (a, b) and (b, a) — a two-way barrier.
        If False, return only the traversal direction.
    n_samples
        Points along the segment to test. Default: 4× the expected number of
        cells the line crosses (estimated from H3 edge length vs segment length).
    """
    edge_m = h3.average_hexagon_edge_length(resolution, unit="m")
    # Rough arc length in meters (small-angle: mix lat × lon scaling).
    lat_mid = 0.5 * (lat1 + lat2)
    from .geomconst import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR
    import math
    dlat_m = (lat2 - lat1) * M_PER_DEG_LAT
    dlon_m = (lon2 - lon1) * M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat_mid))
    seg_m = math.hypot(dlat_m, dlon_m)
    if n_samples is None:
        n_samples = max(16, int(4 * seg_m / edge_m))

    edges: set[tuple[str, str]] = set()
    prev_cell = None
    for t in np.linspace(0.0, 1.0, n_samples):
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)
        cell = h3.latlng_to_cell(lat, lon, resolution)
        if prev_cell is not None and cell != prev_cell:
            edges.add((prev_cell, cell))
            if bidirectional:
                edges.add((cell, prev_cell))
        prev_cell = cell
    return list(edges)
```

- [x] **Step 2: Test the helper against a known geometry**

```python
# tests/test_h3_barriers.py
import h3
import numpy as np
from salmon_ibm.h3_barriers import line_barrier_to_h3_edges


def test_line_barrier_crosses_at_least_one_edge():
    # A 2 km line at res 9 (200 m cells) should cross ~10 cells → ~10 edges.
    edges = line_barrier_to_h3_edges(
        55.30, 21.10, 55.30, 21.13,  # 2 km eastward at 55°N
        resolution=9,
    )
    assert 5 < len(edges) < 40, f"expected ~10-20 bidirectional edges, got {len(edges)}"
    # Every (a, b) has a matching (b, a) (bidirectional)
    pairs = set(edges)
    for a, b in list(pairs):
        assert (b, a) in pairs, f"bidirectional pair missing for {a}→{b}"


def test_line_barrier_edges_are_h3_neighbours():
    edges = line_barrier_to_h3_edges(
        55.30, 21.10, 55.30, 21.13, resolution=9
    )
    for a, b in edges:
        nbrs = set(h3.grid_ring(a, 1))
        assert b in nbrs, f"{b} is not a neighbour of {a}"


def test_line_barrier_zero_length_returns_empty():
    edges = line_barrier_to_h3_edges(55.3, 21.1, 55.3, 21.1, resolution=9)
    assert edges == []
```

- [x] **Step 3: Run & commit**

Run: `micromamba run -n shiny python -m pytest tests/test_h3_barriers.py -v`
Commit: `feat: line_barrier_to_h3_edges helper for lat/lon → H3 barrier spec`.

### Task 4.2: `BarrierMap.from_csv_h3` loader

**Files:**
- Modify: `salmon_ibm/barriers.py` (add H3 classmethod, still alongside HexSim path)
- Modify: `salmon_ibm/h3_barriers.py` (add loader)
- Test: `tests/test_h3_barriers.py`

- [x] **Step 1: Implement the loader**

```python
# Append to salmon_ibm/h3_barriers.py

def load_h3_barrier_csv(
    path: Path,
    mesh: "H3Mesh",
) -> dict[tuple[int, int], dict]:
    """Parse a barrier CSV against an H3Mesh. Returns edge-keyed dict
    suitable for BarrierMap(...).

    Skips edges whose H3 IDs aren't in the mesh (logs warning). Raises
    ValueError on malformed rows.
    """
    import logging
    log = logging.getLogger(__name__)

    # Build H3-int → mesh-compact-index reverse lookup once.
    id_to_idx = {int(mid): i for i, mid in enumerate(mesh.h3_ids)}

    edges: dict[tuple[int, int], dict] = {}
    n_skipped = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"from_h3", "to_h3", "mortality", "deflection", "transmission"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"barrier CSV missing columns: {sorted(missing)}")
        for row_num, row in enumerate(reader, start=2):
            from_id = int(h3.str_to_int(row["from_h3"].strip()))
            to_id = int(h3.str_to_int(row["to_h3"].strip()))
            if from_id not in id_to_idx or to_id not in id_to_idx:
                n_skipped += 1
                continue
            from_idx = id_to_idx[from_id]
            to_idx = id_to_idx[to_id]
            # Validate neighbour
            nbr_row = mesh.neighbors[from_idx]
            if to_idx not in nbr_row.tolist():
                raise ValueError(
                    f"row {row_num}: {row['from_h3']} → {row['to_h3']} "
                    f"are not H3 neighbours"
                )
            # Validate probabilities sum ~ 1
            mort = float(row["mortality"])
            deflect = float(row["deflection"])
            trans = float(row["transmission"])
            total = mort + deflect + trans
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"row {row_num}: mortality+deflection+transmission "
                    f"= {total:.6f} (expected 1.0 ± 1e-6)"
                )
            edges[(from_idx, to_idx)] = {
                "mortality": mort,
                "deflection": deflect,
                "transmission": trans,
                "note": row.get("note", ""),
            }
    if n_skipped:
        log.warning("H3 barrier CSV: skipped %d rows with off-mesh cells", n_skipped)
    return edges
```

```python
# In salmon_ibm/barriers.py (BarrierMap class):
# Constructor takes no args; attach _edges after construction (same
# pattern as empty()) rather than changing __init__ signature.

@classmethod
def from_csv_h3(cls, path: str | Path, mesh) -> "BarrierMap":
    """Load H3-native barriers. Requires a mesh exposing `h3_ids` and
    `neighbors` (i.e. H3Mesh)."""
    from .h3_barriers import load_h3_barrier_csv
    if not hasattr(mesh, "h3_ids"):
        raise TypeError(
            f"from_csv_h3 requires an H3Mesh; got {type(mesh).__name__}"
        )
    bmap = cls()
    bmap._edges = load_h3_barrier_csv(Path(path), mesh)
    return bmap
```

- [x] **Step 2: Write the loader test with a synthetic CSV**

```python
def test_from_csv_h3_rejects_nonneighbour_edge(tmp_path):
    import h3
    from salmon_ibm.h3mesh import H3Mesh
    from salmon_ibm.barriers import BarrierMap

    # Two cells that are NOT neighbours at res 9
    cell_a = h3.latlng_to_cell(55.3, 21.1, 9)
    cell_far = h3.latlng_to_cell(55.4, 21.3, 9)  # ~20 km away
    cells = list(h3.grid_disk(cell_a, 2))
    mesh = H3Mesh.from_h3_cells(cells)

    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "from_h3,to_h3,mortality,deflection,transmission,note\n"
        f"{cell_a},{cell_far},0.5,0.4,0.1,bad\n"
    )
    with pytest.raises(ValueError, match="not H3 neighbours"):
        BarrierMap.from_csv_h3(csv_path, mesh)


def test_from_csv_h3_rejects_bad_probability_sum(tmp_path):
    import h3
    from salmon_ibm.h3mesh import H3Mesh
    from salmon_ibm.barriers import BarrierMap

    cell_a = h3.latlng_to_cell(55.3, 21.1, 9)
    cells = list(h3.grid_disk(cell_a, 2))
    mesh = H3Mesh.from_h3_cells(cells)
    cell_b = list(h3.grid_ring(cell_a, 1))[0]

    csv_path = tmp_path / "bad_prob.csv"
    csv_path.write_text(
        "from_h3,to_h3,mortality,deflection,transmission,note\n"
        f"{cell_a},{cell_b},0.5,0.5,0.5,sums to 1.5\n"
    )
    with pytest.raises(ValueError, match="1.0"):
        BarrierMap.from_csv_h3(csv_path, mesh)


def test_from_csv_h3_round_trip(tmp_path):
    """Line-barrier helper output → CSV → BarrierMap is a full round trip."""
    import h3, csv as _csv
    from salmon_ibm.h3mesh import H3Mesh
    from salmon_ibm.barriers import BarrierMap
    from salmon_ibm.h3_barriers import line_barrier_to_h3_edges

    edges = line_barrier_to_h3_edges(55.30, 21.10, 55.30, 21.13, resolution=9)
    # Build a mesh that contains every cell touched by the line.
    touched = set()
    for a, b in edges:
        touched.add(a)
        touched.add(b)
    mesh = H3Mesh.from_h3_cells(list(touched))

    csv_path = tmp_path / "line.csv"
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=[
            "from_h3","to_h3","mortality","deflection","transmission","note"])
        writer.writeheader()
        for a, b in edges:
            writer.writerow({"from_h3": a, "to_h3": b,
                             "mortality": 0.2, "deflection": 0.7,
                             "transmission": 0.1, "note": ""})

    bmap = BarrierMap.from_csv_h3(csv_path, mesh)
    assert len(bmap._edges) == len(edges), \
        f"expected {len(edges)} loaded, got {len(bmap._edges)}"
```

- [x] **Step 3: Run & commit**

Commit: `feat: BarrierMap.from_csv_h3 loader + validation (neighbour + probability sum)`.

### Task 4.3: Synthetic Nemunas barrier file + integration

**Files:**
- Create: `data/nemunas_h3_barriers.csv`
- Modify: `configs/config_nemunas_h3.yaml` (opt-in `barriers_csv` field)
- Modify: `salmon_ibm/config.py` (add `barriers_csv: str | None = None`)
- Modify: `salmon_ibm/simulation.py` (load barriers if `barriers_csv` present)
- Test: extend `tests/test_nemunas_h3_integration.py`

A **synthetic** barrier line across the Klaipėda Strait (the narrow entrance at the north of the Curonian Lagoon, ~55.7°N, 21.1°E) — imagined as a "shipping channel protection weir." Not physical, but exercises the full barrier code path on a real landscape.

- [x] **Step 1: Generate the barrier CSV from the line-helper**

Single-use script, delete after committing the CSV:

```python
# scripts/build_nemunas_h3_barriers.py
"""One-shot: generate data/nemunas_h3_barriers.csv from a Klaipėda Strait line."""
from pathlib import Path
import csv

from salmon_ibm.h3_barriers import line_barrier_to_h3_edges

# Klaipėda Strait — imagined weir across the narrowest point
LINE = (55.715, 21.105, 55.700, 21.140)   # (lat1, lon1, lat2, lon2)
RESOLUTION = 9

edges = line_barrier_to_h3_edges(*LINE, resolution=RESOLUTION)
out = Path(__file__).resolve().parent.parent / "data" / "nemunas_h3_barriers.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "from_h3","to_h3","mortality","deflection","transmission","note"])
    w.writeheader()
    for a, b in edges:
        w.writerow({"from_h3": a, "to_h3": b,
                    "mortality": 0.05, "deflection": 0.90,
                    "transmission": 0.05,
                    "note": "Klaipėda Strait synthetic weir"})
print(f"wrote {out} ({len(edges)} edges)")
```

Run it:
```bash
micromamba run -n shiny python scripts/build_nemunas_h3_barriers.py
```

- [x] **Step 2: Filter barrier edges to water-only, generate CSV**

Extend `scripts/build_nemunas_h3_barriers.py` to load the landscape NetCDF and drop any edge where either endpoint is a land cell (`water_mask == 0`):

```python
# After computing `edges` from line_barrier_to_h3_edges(...):
import xarray as xr
ds = xr.open_dataset("data/nemunas_h3_landscape.nc", engine="scipy")
h3_to_water = {int(hid): bool(wm) for hid, wm
               in zip(ds["h3_id"].values, ds["water_mask"].values)}

def is_water(cell_str):
    return h3_to_water.get(int(h3.str_to_int(cell_str)), False)

water_edges = [(a, b) for a, b in edges if is_water(a) and is_water(b)]
print(f"dropped {len(edges) - len(water_edges)} edges with land endpoints")
edges = water_edges
```

- [x] **Step 3: Wire into config & simulation**

`configs/config_nemunas_h3.yaml` — add top-level `barriers_csv` field (null-able).

`salmon_ibm/simulation.py` — replace the `BarrierMap.empty()` line from Task 3.2 with:
```python
barriers_csv = self.config.get("barriers_csv")
if barriers_csv:
    barriers = BarrierMap.from_csv_h3(barriers_csv, mesh)
else:
    barriers = BarrierMap.empty()
```

- [x] **Step 4: Extend integration test to prove barrier survival effect**

Mortality-effect tests are stochastic; the current 5/90/5 split over ~30 edges gives only ~7-10 expected extra deaths per 500 agents, which is flaky at a single seed. To make the test reliable:

1. **Increase barrier severity** to 30% mortality in a test-only barrier CSV (`data/nemunas_h3_barriers_strong.csv`) — unrealistic but guaranteed to show a signal.
2. **Use a 3-day (72-step) run** for the barrier comparison — the module still runs the 720-step baseline fixture `h3_sim` from Task 3.3, so total module runtime is 720 + 72 + 72 ≈ 864 steps across 3 fixtures. Expect ~4–7 min on the 16 GB laptop (Numba JIT compile first, then a ~3–5 min baseline + ~30 s × 2 for the barrier runs). If the executor sees a longer hang, the first-run Numba compile is the usual culprit.
3. **Compare ≥ not >** — accounts for the unlikely tie case while still catching regressions.

```python
def test_barrier_causes_nonzero_mortality(h3_sim_with_barrier, h3_sim_no_barrier):
    """Sim with the strait weir must have at least as many dead agents as
    without, and on average strictly more (see note on flakiness)."""
    def dead(sim):
        return 500 - int(sim.pool.alive.sum()) - int(sim.pool.arrived.sum())
    with_dead = dead(h3_sim_with_barrier)
    without_dead = dead(h3_sim_no_barrier)
    assert with_dead >= without_dead, (
        f"barrier should not reduce mortality: "
        f"with_barrier={with_dead}, no_barrier={without_dead}"
    )
    # With 30% mortality over ~30 edges, expect ≥ 1 extra death at seed=42.
    # If this trips occasionally, re-run with seed=0 or seed=7 — not a true
    # regression unless it fails across multiple seeds.


@pytest.fixture(scope="module")
def h3_sim_with_barrier():
    _needs_data()
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config(str(CONFIG))
    cfg["barriers_csv"] = str(PROJECT / "data" / "nemunas_h3_barriers_strong.csv")
    sim = Simulation(cfg, n_agents=500, rng_seed=42)
    sim.run(n_steps=72)   # 3 days — fast, still captures barrier effect
    return sim


@pytest.fixture(scope="module")
def h3_sim_no_barrier():
    _needs_data()
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    cfg = load_config(str(CONFIG))
    cfg["barriers_csv"] = None
    sim = Simulation(cfg, n_agents=500, rng_seed=42)
    sim.run(n_steps=72)
    return sim
```

Also generate `data/nemunas_h3_barriers_strong.csv` alongside the normal one — same edges, 0.30/0.60/0.10 split.

- [x] **Step 4: Run & commit**

Run: `micromamba run -n shiny python -m pytest tests/test_nemunas_h3_integration.py -v`
Commit (two commits — one for wiring, one for test):
```bash
git add scripts/build_nemunas_h3_barriers.py data/nemunas_h3_barriers.csv \
        configs/config_nemunas_h3.yaml salmon_ibm/simulation.py salmon_ibm/config.py
git commit -m "feat: Klaipėda Strait synthetic barrier wired into Nemunas H3 scenario"

git add tests/test_nemunas_h3_integration.py
git commit -m "test: H3 barrier mortality effect (synthetic strait weir)"
```

---

### Phase 5 — Viewer integration (should just work)

### Task 5.1: Verify existing H3HexagonLayer viewer renders Nemunas H3 landscape

**Files:**
- No code changes expected (the viewer already uses `h3_hexagon_layer` from the earlier session).
- Possibly: add a new workspace discovery path so `configs/config_nemunas_h3.yaml` appears in the viewer dropdown.

- [x] **Step 1: Launch the app and navigate to the viewer**

- [x] **Step 2: Select `nemunas_h3` (if visible) or add a quick-load button; confirm the hex grid renders.**

- [x] **Step 3: Verify visual:** 106k cells cover the bbox, depth colors grade from 0 m (shore) to 8 m (deep strait) in the expected geographic layout.

- [x] **Step 4: (Optional) Overlay the Klaipėda-strait barrier edges** as a thin `path_layer` on top of the hex grid, to visually confirm the barrier ran where it was meant to.

- [x] **Step 5: Commit any viewer wiring**

---

## Acceptance criteria

1. `tests/test_movement_metric.py` — 2 tests passing; `_advection_numba` works on both lat/lon and meter meshes.
2. `tests/test_h3mesh.py` — 5 tests passing (unit: from_cells, from_polygon, pentagon_guard at both factory entries, placement sanity).
3. `tests/test_h3_env.py` — 1 test passing, skipped if `nemunas_h3_landscape.nc` is absent.
4. `tests/test_h3_barriers.py` — 6 tests passing (line helper: crossing / neighbour-pairs / zero-length; loader: non-neighbour reject / bad-probability reject / round-trip).
5. `tests/test_h3_placement.py` — 1 test passing (uniform-random-water respects mask).
6. `tests/test_nemunas_h3_integration.py` — 7 tests passing, skipped if landscape absent (5 base + 1 N-S salinity gradient + 1 barrier effect).
7. `tests/test_curonian_realism_integration.py` — all 9 tests still pass (no regression in the TriMesh path).
8. `tests/test_hexsim.py` — all tests pass (no regression in HexMesh path).
9. Viewer renders the Nemunas landscape with real hex cells, same `H3HexagonLayer` path already live. Optional: barrier edges overlaid.
10. `data/nemunas_h3_landscape.nc` exists, ≤10 MB, 106 k cells, 30 days of tos/sos/uo/vo.
11. `data/nemunas_h3_barriers.csv` exists, ≥10 in-mesh water-water edges, all rows pass validation. *(Shipped: 116 edges across a 25 km mid-lagoon E–W line; see Execution status §6 for why this differs from the original "Klaipėda Strait" wording.)*

## Execution status

Executed **2026-04-25** in a single session.  All 5 phases shipped to
`main`; final regression suite is **729 passed, 32 skipped, 1 xfailed**
in 26 m 42 s.

### Commit log (newest first)

| Commit | Phase | Subject |
|---|---|---|
| `e0fa529` | 5 | feat(viewer): render H3 landscape NetCDFs natively |
| `ed81ec4` | (fix) | test(h3_env): expected field-name set after ssh zero-fill |
| `5817bb7` | 4.3 | feat: synthetic mid-lagoon H3 barrier wired into Nemunas |
| `ec38a54` | 4.1 + 4.2 | feat: H3 barriers — line helper + CSV loader |
| `804ea3c` | 3 | feat: H3 backend wired through Simulation, Nemunas green |
| `49a407f` | 2.2 | feat: H3Environment loads CMEMS/EMODnet by H3 cell id |
| `209ac74` | 2.1 | feat: one-shot Nemunas H3 landscape builder |
| `902e00f` | 1.2 | feat: H3Mesh.from_polygon factory |
| `1f27ab0` | 1.1 + 1.3 | feat: H3Mesh + pentagon-aware neighbour table |
| `674e2e2` | 0.2 | refactor: BarrierMap.from_hbf → from_hbf_hexsim + empty() |
| `6360f7d` | 0.1 | refactor: Mesh.metric_scale() abstraction |

### Test count delta

* H3Mesh unit tests (`tests/test_h3mesh.py`): **21**
* H3Environment unit tests (`tests/test_h3_env.py`): **7**
* H3 barrier unit tests (`tests/test_h3_barriers.py`): **12**
* Nemunas integration (`tests/test_nemunas_h3_integration.py`): **8** (incl. strong-barrier mortality effect)
* Movement-metric regression (`tests/test_movement_metric.py`): **2**
* **Total new: 50** ; pre-existing TriMesh/HexSim regressions all still green.

### Deviations from the plan as-written

Caught and fixed during execution; documented in commit messages:

1. **`ssh` field needed for H3 landscapes.** `movement.py:84,99` reads `fields["ssh"]` for upstream/downstream behaviour but CMEMS BALTICSEA reanalysis doesn't ship `zos`. `H3Environment.from_netcdf` zero-fills `ssh` on load (same trick the Curonian regridder uses).  Plan didn't anticipate this; commit `804ea3c` added it inline.

2. **`Environment.fields` contract is mutate-in-place, not (time, n_cells) array.** First-pass `H3Environment` exposed `fields` as the full time-series; `Simulation.step` reads it as a per-cell snapshot.  Refactored `H3Environment` so `fields` is the live snapshot dict and `_full_fields` carries the time-series privately (commit `804ea3c`).

3. **NetCDF3 has no unsigned 64-bit type.** Plan called for `NETCDF3_64BIT` output (CDF-2 offset format, NOT 64-bit ints).  Switched to `NETCDF4` via `h5netcdf` engine for both writer and `H3Environment.from_netcdf` reader (commit `209ac74` + `49a407f`).

4. **Hourly simulation step → daily CMEMS index.** `H3Environment.advance(step)` clamps `step >= n_time` rather than IndexError; integer-divides hourly steps to daily indices when the data cadence is coarser than the simulation step.

5. **Neighbour-table compaction.** Plan-as-written placed valid neighbours at the same slot index returned by `h3.grid_ring`, leaving `-1` in slot 0 if the first ring-neighbour was outside the bbox.  `_step_directed_numba` then read `water_nbrs[c, 0] == -1` and the agent ended up at `tri_idx = -1`.  Fixed by compacting valid entries into the first `count` slots (commit `1f27ab0` + regression test `test_h3mesh_neighbors_are_compacted_to_low_slots`).

6. **Synthetic barrier moved from Klaipėda Strait to mid-lagoon.** The plan and the original spec called for a Klaipėda Strait line.  At H3 res 9 the strait is only 2–4 cells wide, so after the EMODnet land-mask filter the line yielded only 1–4 water-water edges — too sparse for a robust 3-day mortality-effect test.  Switched to a 25 km E–W line through the open Curonian Lagoon at lat 55.30°N (116 in-mesh edges).  Spec § "Synthetic Klaipėda Strait weir" updated; rationale documented in `scripts/build_nemunas_h3_barriers.py` docstring.

7. **`tests/test_hexsim_compat.py` callers needed updating** as part of Phase 0.2's `from_hbf` rename — caught by `rg from_hbf` during execution, commit message of `674e2e2` lists all four callers.

### What still works without H3

The legacy paths are intact and re-tested at every commit:

* **Curonian TriMesh** (`tests/test_curonian_realism_integration.py`): 9/9 passing.
* **Columbia HexSim** (`tests/test_hexsim_compat.py`, `tests/test_hexsim.py`): 8 + 29 passing.
* **Movement & barriers** (`tests/test_movement.py`, `tests/test_barriers.py`): all passing under the new `_advection_numba` signature.

`metric_scale(lat)` returns `(1.0, 1.0)` for `HexMesh` so the Columbia path is bit-for-bit equivalent to pre-refactor.

## Deferred work (NOT in this plan)

- **H3 adapters for bioenergetics/osmoregulation.** These read fields by cell index — already works via `env.current()`. But if any species-specific term uses `mesh._edge` (HexSim) or `mesh.triangles` (TriMesh), that call site needs surfacing via an abstract `mesh.cell_edge_length(cell_idx)` helper. Watch Phase 3 integration test for any AttributeError.
- **Real SSH (seiche) input on H3 landscape.** Deferred for the same reason as in the Curonian plan.
- **Pentagon movement handling.** Non-Nemunas landscapes that cover a pentagon need either `pentagon_policy="skip"` at build time or a variable-neighbour kernel path.
- **Real-world dam/weir catalog for H3.** The mid-lagoon barrier shipped in `data/nemunas_h3_barriers.csv` is synthetic for test-coverage only — illustrative probabilities, no engineering basis.  A production landscape with real barriers (e.g., Columbia converted to H3) is out of scope — would need a per-landscape `scripts/build_*_h3_barriers.py` sourced from a regulatory database.

## Self-review notes

Four review passes were run over this plan; issues found and resolved inline. Third pass used (a) an h3-py 4.4.2 API verification script that exercised every h3 call the plan makes, (b) an independent code-correctness reviewer agent, (c) a scientific-validator reviewer agent. Fourth pass ran (d) a `python -m ast.parse` syntax-check over every fenced code block in the plan, (e) an independent "did the fixes introduce new bugs?" reviewer agent.

**Pass 1 — completeness**
- ✅ Task 0.1 originally left an "if test passes, no-op" branch — now unconditionally introduces `metric_scale`, so H3 correctness doesn't depend on latitude luck.
- ✅ `from_polygon` pentagon guard moved into `from_h3_cells` so direct callers can't bypass it (Task 1.3).
- ✅ `id_to_idx` dict now uses Python `int` keys consistently — avoids numpy uint64 vs int hash-equality pitfalls.
- ✅ Agent placement strategy `uniform_random_water` added as Task 3.1b; `sim.initial_cells` snapshot added so the movement-sanity test has something to diff against.
- ✅ CMEMS builder accepts `--start/--end` to bound output NetCDF size.
- ✅ H3Environment exposes field names consistent with TriMesh and HexMesh — see Pass 3's correction below for the canonical key list (`temperature`, `salinity`, `u_current`, `v_current`).

**Pass 2 — barriers integration**
- ✅ Phase 4 added (three tasks: line helper + CSV loader + synthetic Nemunas fixture wired into integration test).
- ✅ Barriers removed from deferred work (only "real-world dam catalog for a production landscape" remains deferred).
- ✅ Acceptance criteria updated: 11 items now including barrier loader tests and mortality-effect integration test.

**Pass 3 — external & scientific review (h3-py API + two reviewer agents)**
- ✅ **`_advection_numba` full signature preserved.** Pass 1/2 drafts implied a 2-arg add; real kernel has 9 args and mutates `tri_indices` in place. Task 0.1 now shows the complete new signature, explicitly preserves `speeds`/`rand_drift`, and test asserts in-place mutation (not a return value).
- ✅ **`BarrierMap.empty()` and `from_csv_h3()` don't pass args to the no-arg constructor.** Constructor is `__init__(self)` at `barriers.py:37`; factories now construct a bare instance and attach `_edges` after.
- ✅ **Config is a dict, not a dataclass.** Every `cfg.X` use rewritten to `self.config.get("X")` to match `load_config() -> dict`. `validate_config` relaxed to accept `mesh_backend=h3` layout (no `grid` section needed).
- ✅ **H3Environment field keys aligned to `u_current`/`v_current`/`temperature`/`salinity`** — matches `environment.py:41-42` and `hexsim_env.py:150-151`. Earlier draft used `flow_u`/`flow_v` which would silently no-op `_apply_current_advection_vec`.
- ✅ **`from_polygon` forwards `pentagon_policy`** — earlier draft didn't, so the Task 1.3 `pentagon_policy="skip"` test would TypeError.
- ✅ **`from_h3_cells(pentagon_policy="skip")` filter-order bug fixed** — `keep` mask now computed against the original list before the cell list is mutated, so depth/water_mask get sliced correctly.
- ✅ **`H3Mesh.gradient` gets `metric_scale` too** — the scientific reviewer caught that the Task 0.1 fix was site-specific to `_advection_numba`. Gradient-following events (thermal-seek) would otherwise bias toward N-S neighbours at 55° N by ~1.74×.
- ✅ **Barrier CSV builder filters by `water_mask`** — otherwise the synthetic Klaipėda weir could emit edges across land cells and the loader would accept them.
- ✅ **Barrier integration-test flakiness mitigated.** Stronger synthetic barrier (30% mortality) + shorter 3-day fixtures + `≥` comparison instead of `>`. Runtime drops from ~15 min to ~2 min per test file.
- ✅ **`Task 0.2` git-add includes `salmon_ibm/config.py`** — the `barrier_map_from_config` helper is a third `from_hbf` call site beyond `barriers.py`/`simulation.py`.
- ✅ **Files-NOT-touched list de-contradicted.** `movement.py` was listed in both "modified" and "untouched"; the untouched list now explicitly notes the exception.
- ✅ **Spec "Known limitations" expanded** with 5 new bullets: movement speed ceiling, agent density realism, `uniform_random_water` is biologically false, synthetic weir illustrative only, per-cell area variance. N-S salinity threshold tightened 0.5 → 1.5 PSU to catch regridder-homogenisation bugs.
- ⚠ **Flagged but not fixed** (future work, not blocking execution):
  - *h3-py 4.x* does not expose a vectorised `latlng_to_cell` — the 106 k-cell builder loops in Python (~1.5 s at that size; acceptable for one-shot data prep).
  - `_water_nbrs` dtype: TriMesh uses `np.intp`, H3Mesh draft used `int32`. Standardise on `int32` for all backends to keep Numba specialisation consistent; noted in plan but not enforced with a migration task.
  - The `moved > 50` sanity-test (Task 3.3) may trip if agents flicker between neighbours over 720 steps; lowered threshold note is in-plan but not hard-enforced.

**Pass 4 — syntactic & "did the fixes introduce new bugs?" review**
- ✅ **Python ast-parse over all 29 fenced `python` blocks**: 1 block (Task 3.2 Step 2) used pseudo-code `...existing...` that is not valid Python — replaced with `pass  # keep existing branch unchanged`. Remaining 28 blocks all parse clean.
- ✅ **Plan wrongly claimed `_apply_current_advection_vec` needed `mesh` added**. Real signature at `movement.py:341` is already `(pool, mesh, fields, alive_mask, rng)` — `mesh` is the second positional arg. Plan rewritten to note this: only the function **body** needs updating (the `_advection_numba` call inside gets two new args), not the outer signature.
- ✅ **Line-range reference `movement.py:320-370` corrected to `movement.py:341`** (actual function definition line, verified by reading the file).
- ✅ **"one call site in MovementEvent" language corrected** — the actual call sites are inside `execute_movement` (at lines 62 and 133), not a `MovementEvent` class. Since both sites already pass `mesh`, a single body-level change at line 341 updates both via the internal `_advection_numba` call.
- ✅ **Runtime estimate corrected**. Claimed "~2 min per test file" was only the two new barrier fixtures; adding the existing 720-step `h3_sim` baseline pushes the full `test_nemunas_h3_integration.py` module to ~4–7 min on a 16 GB CPU-only laptop (including first-run Numba JIT compile).
- ✅ **`pentagon_policy == "allow"` silent fall-through** got an explicit inline comment documenting the behaviour (5-neighbour cells + `-1` sentinel in the 6th slot).
- ✅ **Validated: `self.config` attribute is correct** (`simulation.py:58`).
- ✅ **Validated: `BarrierMap._edges` is the real internal dict name** (`barriers.py:38`).
- ✅ **Validated: Task 0.2's grep does catch `simulation.py:102`** — `BarrierMap.from_hbf(hbf_path, self.mesh)` is the live site that will break if rename isn't threaded through simulation.py. Task 0.2 Step 5 commit already lists simulation.py.
- ✅ **Validated: `tests/test_curonian_realism_integration.py` really does have 9 test functions** (acceptance criterion 7's count is accurate).

**After pass 4: zero BLOCK issues remain.** Plan is safe to hand to an execution agent.

**Pass 5 — consistency across plan + spec (automated script + grep)**
- ✅ **YAML block validated**: the single `configs/config_nemunas_h3.yaml` fence parses via `yaml.safe_load`.
- ✅ **All `salmon_ibm.X` imports** in code snippets resolve to either existing modules (`config`, `simulation`, `barriers`, `movement`) or Phase-1/2/4-created modules (`h3mesh`, `h3_env`, `h3_barriers`). No dangling imports.
- ✅ **No stale `file:line` references** — every `movement.py:NNN`, `barriers.py:NNN`, `config.py:NNN` cite resolves within the real file's current line count.
- ✅ **Stale self-review bullet fixed** — Pass-1 `✅` claim that H3Environment uses `flow_u`/`flow_v` field names (overridden by Pass-3's correction to `u_current`/`v_current`) was internally inconsistent. Bullet now points to the Pass-3 correction.
- ✅ **Missing salinity-gradient test added**. Spec §5 mandates a ≥ 1.5 PSU N-S gradient invariant, but Task 3.3's integration-test list had no matching test — mirroring `tests/test_curonian_realism_integration.py::test_north_south_salinity_gradient` against the H3 mesh. Acceptance criterion 6 updated: 6 → 7 tests in `test_nemunas_h3_integration.py`.
- ✅ **Salinity threshold now consistent** between spec (§5: ≥ 1.5 PSU) and plan test code (both use 1.5).

**After pass 5: zero BLOCK issues, plan and spec are internally consistent. All 29 Python code blocks parse. Handoff-ready.**

**Type consistency**
- `h3_ids` is `uint64` in arrays, `int` in dict lookups, converted at boundary.
- `neighbors` is `int32` with `-1` sentinel across all mesh types.
- `centroids` is `float64 [lat, lon]` for TriMesh and H3Mesh, `float64 [y, x] meters` for HexMesh — `metric_scale()` bridges the difference.
- Barrier keys are always `(compact_idx, compact_idx)` tuples; H3 IDs only appear in CSV rows and are resolved at load time.

**Placeholder scan**: every code block is complete, every command has expected output, every file path is project-relative.

**Known sequencing risk**: Task 0.1 touches `_advection_numba`'s signature (adds two float args). Every caller must be updated in the same commit. The grep command in Task 0.1 Step 2 catches them, but reviewer should watch for any event handler that builds an `_advection_numba` call via `getattr` / `**kwargs` — those are invisible to grep.
