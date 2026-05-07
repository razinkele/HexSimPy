# C4 Movement Gradient Implementation Plan

**Plan version:** ✅ v3 final — **3-pass plan-review-loop CONVERGED**. Pass-1 (3 parallel reviewers: code-reviewer, pr-test-analyzer, silent-failure-hunter) found 21 issues including a CRITICAL signature mismatch (`execute_movement(pool, mesh, fields, ...)` not `landscape`). Pass-2 caught two new issues v2 introduced (Test 7b signature, Step 1a `--collect-only` skipif blindness). Pass-3 verified all closures + flagged one optional NIT (`tail -20` for warning-line resilience) — applied. **Plan ready for execution.** Total findings across 3 plan-review passes: 24.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat-zero `ssh` field that powers UPSTREAM/DOWNSTREAM movement with a `dist_from_sea` scalar (multi-source edge-length-weighted Dijkstra at landscape build), so directed-movement actually produces displacement and the four-tier hatchery-vs-wild architecture (C1-C3.3) becomes biologically active in production.

**Architecture:** Single global gradient computed at landscape-build time. Loaded into `H3Environment` post-construction (NOT via `full_fields`, to avoid `advance()` overwriting). Movement.py's existing `_step_directed_*` kernels read `fields["dist_from_sea"]` instead of `fields["ssh"]` with the `ascending` flag flipped (since `dist_from_sea` is higher upstream while `ssh` was assumed lower upstream). Per-env latched dormancy raise catches deployed apps that load a degraded NC with logging suppressed. Two backward-compat tiers: Case A (NC missing variable → warn + zero-fill + sim-time raise) and Case B (NC structurally invalid → raise at sim init).

**Tech Stack:** Python 3.10+, conda env `shiny`, NumPy, xarray, pytest, h5netcdf, Numba (existing kernels — kernel logic unchanged in C4). Spec at `docs/superpowers/specs/2026-05-07-c4-movement-gradient-design.md` (v11 final, commit `9cc201c`).

**Pre-flight:** Confirm baseline before starting.

```bash
micromamba run -n shiny python -m pytest tests/ --collect-only -q | tail -1
```

Should report 928 tests collected (post-C3.3 baseline at v1.7.7). Record the count; expected post-C4 = baseline + 17 (16 new tests in `test_movement_gradient.py` + at least 1 strengthened test in `test_h3_env.py`).

---

## Task 1: `compute_dist_from_sea` function (build-script-side)

**Files:**
- Modify: `scripts/build_h3_multires_landscape.py` — add the function near the top of the module (with other utility helpers).
- Test: `tests/test_movement_gradient.py` (NEW)

- [ ] **Step 1: Create the new test file with imports + Test 5 (disconnected graph)**

Create `tests/test_movement_gradient.py`:

```python
"""Tests for C4 — movement gradient (substrate fix).

Spec: docs/superpowers/specs/2026-05-07-c4-movement-gradient-design.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Import the build-script function. The script lives outside the
# salmon_ibm package, so add the scripts directory to sys.path.
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class _SyntheticMesh:
    """Minimal mesh for compute_dist_from_sea unit tests.

    Provides the attributes the function reads: nbr_starts, nbr_idx,
    centroids, water_mask, reach_id, reach_names. Adds N_cells convenience
    via len(reach_id).
    """
    def __init__(
        self,
        nbr_starts: np.ndarray,
        nbr_idx: np.ndarray,
        centroids: np.ndarray,
        water_mask: np.ndarray,
        reach_id: np.ndarray,
        reach_names: list[str],
    ):
        self.nbr_starts = nbr_starts
        self.nbr_idx = nbr_idx
        self.centroids = centroids
        self.water_mask = water_mask
        self.reach_id = reach_id
        self.reach_names = reach_names

    @property
    def N_cells(self) -> int:
        return len(self.reach_id)


def test_compute_dist_from_sea_raises_on_disconnected_component():
    """C4 Test 5: synthetic mesh with two disconnected components
    (one with sea, one without). compute_dist_from_sea must raise
    RuntimeError naming the unreachable reach."""
    from build_h3_multires_landscape import compute_dist_from_sea

    # Component A: cells 0,1 (sea + adjacent), reach=OpenBaltic.
    # Component B: cells 2,3 (river, no path to sea), reach=Nemunas.
    nbr_starts = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    nbr_idx = np.array([1, 0, 3, 2], dtype=np.int32)
    centroids = np.array([
        [55.0, 21.0],  # cell 0
        [55.0, 21.001],  # cell 1
        [55.5, 21.5],  # cell 2
        [55.5, 21.501],  # cell 3
    ], dtype=np.float64)
    water_mask = np.ones(4, dtype=bool)
    reach_id = np.array([0, 0, 1, 1], dtype=np.int8)
    reach_names = ["OpenBaltic", "Nemunas"]

    mesh = _SyntheticMesh(nbr_starts, nbr_idx, centroids, water_mask,
                          reach_id, reach_names)
    with pytest.raises(RuntimeError, match=r"Nemunas"):
        compute_dist_from_sea(mesh)
```

- [ ] **Step 2: Run test to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_gradient.py::test_compute_dist_from_sea_raises_on_disconnected_component -v`
Expected: ImportError or "no module named build_h3_multires_landscape" — function doesn't exist yet.

- [ ] **Step 3: Add `compute_dist_from_sea` to build script**

In `scripts/build_h3_multires_landscape.py`, ensure the following imports exist at module top (add any missing):

```python
import heapq
import math
import numpy as np
```

Then, near the top of the module (after imports, before existing helpers), add:

```python
def compute_dist_from_sea(mesh) -> np.ndarray:
    """Multi-source edge-length-weighted Dijkstra from OpenBaltic
    water cells. Returns float32[N_cells] distance-in-meters.

    Pure: does NOT mutate the input mesh. The caller (this build
    script) writes the returned array to the landscape NC as the
    `dist_from_sea` variable; H3Environment.from_netcdf attaches
    it to mesh.dist_from_sea post-construction (see
    docs/superpowers/specs/2026-05-07-c4-movement-gradient-design.md
    "Where it lives" → mesh-attribute post-hoc subsection).

    Determinism: source set is np.sort()-ed before pushing to heap;
    heap tuples are (distance, cell_index) so ties break by cell
    index. See spec §Determinism.

    Raises RuntimeError if any water cell has no path to the
    OpenBaltic source set (disconnected sub-graph). Error message
    includes the reach name(s) of unreachable cells so the build
    operator can locate the topology defect.
    """
    # Imports are at module top per v2 reviewer feedback.
    N = mesh.N_cells
    # Identify OpenBaltic water cells (the source set).
    if "OpenBaltic" not in mesh.reach_names:
        raise RuntimeError(
            "compute_dist_from_sea: 'OpenBaltic' not in mesh.reach_names "
            f"({mesh.reach_names!r}); cannot compute distance-from-sea "
            "without a sea reach."
        )
    open_baltic_id = mesh.reach_names.index("OpenBaltic")
    sources = np.where(
        (mesh.reach_id == open_baltic_id) & mesh.water_mask
    )[0]
    sources = np.sort(sources)  # determinism: source-set order
    if len(sources) == 0:
        raise RuntimeError(
            "compute_dist_from_sea: no OpenBaltic water cells; mesh has "
            "the reach but every cell is water_mask=False."
        )

    # Initialise distances. Land cells stay NaN (per spec data structure).
    dist = np.full(N, np.inf, dtype=np.float64)
    dist[sources] = 0.0

    # Min-heap: (distance, cell_index). Cell_index is the deterministic
    # tie-break per spec §Determinism.
    heap: list[tuple[float, int]] = [(0.0, int(c)) for c in sources]
    heapq.heapify(heap)

    while heap:
        d, c = heapq.heappop(heap)
        if d > dist[c]:
            continue  # stale entry
        # Iterate water-mask neighbors via CSR neighbor table.
        s, e = int(mesh.nbr_starts[c]), int(mesh.nbr_starts[c + 1])
        for k in range(s, e):
            n = int(mesh.nbr_idx[k])
            if n < 0 or not mesh.water_mask[n]:
                continue
            # Edge weight = haversine in meters between centroids.
            lat1, lon1 = mesh.centroids[c]
            lat2, lon2 = mesh.centroids[n]
            edge_m = _haversine_m(lat1, lon1, lat2, lon2)
            nd = d + edge_m
            if nd < dist[n]:
                dist[n] = nd
                heapq.heappush(heap, (nd, n))

    # Validate: every water cell must be reachable.
    unreachable = np.where(
        mesh.water_mask & ~np.isfinite(dist)
    )[0]
    if len(unreachable) > 0:
        # Group by reach for the error message.
        unreached_reaches: dict[str, int] = {}
        for c in unreachable:
            rid = int(mesh.reach_id[c])
            name = (
                mesh.reach_names[rid]
                if 0 <= rid < len(mesh.reach_names)
                else f"rid_{rid}"
            )
            unreached_reaches[name] = unreached_reaches.get(name, 0) + 1
        raise RuntimeError(
            "compute_dist_from_sea: water cells unreachable from "
            "OpenBaltic — disconnected mesh sub-graph. Per-reach cell "
            f"counts: {sorted(unreached_reaches.items())!r}. Fix the "
            "mesh's neighbor table or remove the orphan cluster."
        )

    # Land cells get NaN (spec data structure).
    out = dist.astype(np.float32)
    out[~mesh.water_mask] = np.nan
    return out


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in meters."""
    R_EARTH_M = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R_EARTH_M * c
```

- [ ] **Step 4: Run focused test to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_gradient.py::test_compute_dist_from_sea_raises_on_disconnected_component -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_h3_multires_landscape.py tests/test_movement_gradient.py
git commit -m "feat(c4): compute_dist_from_sea function + Test 5 disconnected-graph check"
```

---

## Task 2: Determinism + Y-junction tests (Tests 5b, 5c)

**Files:**
- Test: `tests/test_movement_gradient.py`

- [ ] **Step 1: Append failing tests for determinism**

Append to `tests/test_movement_gradient.py`:

```python
def _make_chain_mesh(n: int = 10, sea_at_zero: bool = True) -> _SyntheticMesh:
    """10-cell bidirectional chain. Cell 0 = OpenBaltic source.
    Cells 1..n-1 = Nemunas (river). Used by Tests 1, 2, 2b, 3, 5b."""
    # Bidirectional CSR: cell i has neighbors {i-1, i+1} where they exist.
    nbr_starts = np.zeros(n + 1, dtype=np.int32)
    nbrs = []
    for i in range(n):
        if i > 0:
            nbrs.append(i - 1)
        if i < n - 1:
            nbrs.append(i + 1)
        nbr_starts[i + 1] = len(nbrs)
    nbr_idx = np.array(nbrs, dtype=np.int32)
    # Centroids spaced 100m apart along a meridian (uniform haversine).
    centroids = np.array(
        [[55.0 + i * 0.0009, 21.0] for i in range(n)],
        dtype=np.float64,
    )
    water_mask = np.ones(n, dtype=bool)
    reach_id = np.zeros(n, dtype=np.int8)
    reach_id[1:] = 1  # cell 0 = OpenBaltic; cells 1..n-1 = Nemunas
    reach_names = ["OpenBaltic", "Nemunas"]
    return _SyntheticMesh(
        nbr_starts, nbr_idx, centroids, water_mask, reach_id, reach_names,
    )


def test_compute_dist_from_sea_deterministic_synthetic():
    """C4 Test 5b (synthetic part): two runs on the same mesh produce
    NaN-aware-equal output."""
    from build_h3_multires_landscape import compute_dist_from_sea

    mesh = _make_chain_mesh(n=10)
    out1 = compute_dist_from_sea(mesh)
    out2 = compute_dist_from_sea(mesh)
    assert np.array_equal(out1, out2, equal_nan=True), (
        "compute_dist_from_sea is non-deterministic on a 10-cell chain"
    )


def test_compute_dist_from_sea_y_junction_tie_break():
    """C4 Test 5c: 4-cell Y-junction with three equidistant non-source
    cells. Asserts byte-equal recompute."""
    from build_h3_multires_landscape import compute_dist_from_sea

    # Cell 0 = OpenBaltic source. Cells 1,2,3 each connected to cell 0
    # only — three equidistant arms. Bidirectional edges.
    nbr_starts = np.array([0, 3, 4, 5, 6], dtype=np.int32)
    nbr_idx = np.array([1, 2, 3,  0,  0,  0], dtype=np.int32)
    # All three arm-tips at the same lat offset but different lons
    # (cells 1, 2, 3 each differ from cell 0 by ~100m along distinct
    # bearings — haversine values are bit-identical because we use
    # identical math.cos/math.sin inputs).
    centroids = np.array([
        [55.0, 21.0],         # cell 0 (source)
        [55.0009, 21.0],      # cell 1 (north)
        [55.0, 21.00157],     # cell 2 (east; 0.00157° lon ≈ 100m at 55°N)
        [54.9991, 21.0],      # cell 3 (south)
    ], dtype=np.float64)
    water_mask = np.ones(4, dtype=bool)
    reach_id = np.array([0, 1, 1, 1], dtype=np.int8)  # 0=OpenBaltic,1=Nemunas
    reach_names = ["OpenBaltic", "Nemunas"]

    mesh = _SyntheticMesh(
        nbr_starts, nbr_idx, centroids, water_mask, reach_id, reach_names,
    )
    out1 = compute_dist_from_sea(mesh)
    out2 = compute_dist_from_sea(mesh)
    assert np.array_equal(out1, out2, equal_nan=True), (
        "Y-junction tie-break is non-deterministic"
    )
    # Cell 0 = source, distance 0.
    assert out1[0] == 0.0
    # Cells 1, 2, 3 all reachable, finite, > 0.
    assert np.all(np.isfinite(out1[1:]))
    assert np.all(out1[1:] > 0)
```

- [ ] **Step 2: Run focused tests to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_gradient.py::test_compute_dist_from_sea_deterministic_synthetic tests/test_movement_gradient.py::test_compute_dist_from_sea_y_junction_tie_break -v`
Expected: PASS — `compute_dist_from_sea` already deterministic per the explicit `np.sort` on sources + `(distance, cell_index)` heap tuple.

- [ ] **Step 3: Commit**

```bash
git add tests/test_movement_gradient.py
git commit -m "test(c4): Tests 5b (determinism) + 5c (Y-junction tie-break)"
```

---

## Task 3: `H3Environment.from_netcdf` Case A path + err-id constants

**Files:**
- Modify: `salmon_ibm/h3_env.py` — add err-id constants at module top; add Case A logic after the existing `cls(...)` construction.
- Test: `tests/test_h3_env.py`

- [ ] **Step 1: Append failing test for Case A**

Append to `tests/test_h3_env.py`:

```python
def test_from_netcdf_case_a_dist_from_sea_missing(tmp_path, caplog):
    """C4 Test 4: NC missing dist_from_sea variable triggers warn +
    zero-fill + flag init."""
    import logging
    import xarray as xr
    import numpy as np
    from salmon_ibm.h3_env import H3Environment, ERR_DIST_FROM_SEA_MISSING

    # Build a minimal NC matching the existing schema MINUS dist_from_sea.
    # Reuse the existing fixture if available; otherwise construct a 4-cell
    # synthetic NC with the exact variables H3Environment.from_netcdf
    # expects (h3_id, resolution, lat, lon, depth, water_mask, reach_id,
    # nbr_starts, nbr_idx, plus the time-indexed forcing fields).
    nc_path = tmp_path / "test_minimal.nc"
    _build_minimal_h3_nc(nc_path, include_dist_from_sea=False)

    # Build a minimal H3MultiResMesh via the same NC.
    from salmon_ibm.h3_multires import H3MultiResMesh
    mesh = _build_mesh_from_nc(nc_path)

    caplog.set_level(logging.WARNING, logger="salmon_ibm.h3_env")
    env = H3Environment.from_netcdf(str(nc_path), mesh)

    # (a) Warning emitted with the err-id.
    matching = [
        r for r in caplog.records
        if r.name == "salmon_ibm.h3_env"
        and ERR_DIST_FROM_SEA_MISSING in r.getMessage()
    ]
    assert matching, (
        f"Expected warning with err-id {ERR_DIST_FROM_SEA_MISSING}; "
        f"got {[(r.name, r.getMessage()) for r in caplog.records]!r}"
    )

    # (b) fields["dist_from_sea"] exists and is all-zeros.
    assert "dist_from_sea" in env.fields
    assert env.fields["dist_from_sea"].shape == (mesh.N_cells,)
    assert np.all(env.fields["dist_from_sea"] == 0.0)
    assert env.fields["dist_from_sea"].dtype == np.float32

    # (c) Per-env latch flag initialised.
    assert env._dormant_gradient_check_done is False

    # (d) mesh.dist_from_sea attached (same array reference).
    assert mesh.dist_from_sea is env.fields["dist_from_sea"]
```

The test depends on two helpers `_build_minimal_h3_nc` and
`_build_mesh_from_nc`. If they don't already exist in
`tests/test_h3_env.py`, add them at the top of the file:

```python
def _build_minimal_h3_nc(
    path,
    *,
    include_dist_from_sea: bool = True,
    dist_from_sea_arr: "np.ndarray | None" = None,
) -> None:
    """Construct a 4-cell synthetic H3 multires NC for env tests.

    Cells: 0 (OpenBaltic, water), 1 (CuronianLagoon, water),
           2 (Nemunas, water), 3 (Nemunas, land).
    """
    import xarray as xr
    import numpy as np

    n = 4
    h3_id = np.arange(n, dtype=np.uint64)
    resolution = np.full(n, 9, dtype=np.int8)
    lat = np.array([55.0, 55.3, 55.5, 55.51], dtype=np.float64)
    lon = np.array([21.0, 21.3, 21.5, 21.51], dtype=np.float64)
    depth = np.array([10.0, 3.0, 2.0, 0.0], dtype=np.float32)
    water_mask = np.array([True, True, True, False], dtype=bool)
    reach_id = np.array([0, 1, 2, 2], dtype=np.int8)
    # Bidirectional chain: 0-1-2; cell 3 is land (no neighbors).
    nbr_starts = np.array([0, 1, 3, 4, 4], dtype=np.int32)
    nbr_idx = np.array([1, 0, 2, 1], dtype=np.int32)
    # Time-indexed forcing fields (1 timestep, 4 cells).
    tos = np.full((1, n), 12.0, dtype=np.float32)
    sos = np.full((1, n), 7.0, dtype=np.float32)
    uo = np.zeros((1, n), dtype=np.float32)
    vo = np.zeros((1, n), dtype=np.float32)
    time = np.array(["2026-01-01"], dtype="datetime64[D]")

    data_vars = {
        "h3_id": ("cell", h3_id),
        "resolution": ("cell", resolution),
        "lat": ("cell", lat),
        "lon": ("cell", lon),
        "depth": ("cell", depth),
        "water_mask": ("cell", water_mask),
        "reach_id": ("cell", reach_id),
        "nbr_starts": ("cell_p1", nbr_starts),
        "nbr_idx": ("edge", nbr_idx),
        "tos": (("time", "cell"), tos),
        "sos": (("time", "cell"), sos),
        "uo": (("time", "cell"), uo),
        "vo": (("time", "cell"), vo),
    }
    if include_dist_from_sea:
        if dist_from_sea_arr is None:
            # Default valid: cell 0 source (0.0), distances increasing.
            dist_from_sea_arr = np.array(
                [0.0, 100.0, 200.0, np.nan], dtype=np.float32,
            )
        data_vars["dist_from_sea"] = ("cell", dist_from_sea_arr)

    coords = {"time": time}
    attrs = {
        "reach_names": "OpenBaltic,CuronianLagoon,Nemunas",
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    ds.to_netcdf(path, engine="h5netcdf")


def _build_mesh_from_nc(nc_path):
    """Build an H3MultiResMesh from the test NC path."""
    import xarray as xr
    import numpy as np
    from salmon_ibm.h3_multires import H3MultiResMesh

    ds = xr.open_dataset(str(nc_path), engine="h5netcdf")
    names_attr = ds.attrs.get("reach_names", "")
    reach_names = names_attr.split(",") if names_attr else []
    mesh = H3MultiResMesh(
        h3_ids=ds["h3_id"].values.astype(np.uint64),
        resolutions=ds["resolution"].values.astype(np.int8),
        centroids=np.column_stack([ds["lat"].values, ds["lon"].values]),
        nbr_starts=ds["nbr_starts"].values.astype(np.int32),
        nbr_idx=ds["nbr_idx"].values.astype(np.int32),
        water_mask=ds["water_mask"].values.astype(bool),
        depth=ds["depth"].values.astype(np.float32),
        areas=np.zeros(len(ds["h3_id"]), dtype=np.float32),
        reach_id=ds["reach_id"].values.astype(np.int8),
        reach_names=reach_names,
    )
    ds.close()
    return mesh
```

- [ ] **Step 2: Run test to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_h3_env.py::test_from_netcdf_case_a_dist_from_sea_missing -v`
Expected: ImportError on `ERR_DIST_FROM_SEA_MISSING` or AttributeError on `env._dormant_gradient_check_done`.

- [ ] **Step 3: Add err-id constants + Case A logic to h3_env.py**

In `salmon_ibm/h3_env.py`, add at module top (after imports, before any class):

```python
# C4: err-id constants for grep-able operational logging.
# Mirrors ERR_HOMING_HATCHERY_NO_DISPATCH in delta_routing.py:43.
ERR_DIST_FROM_SEA_MISSING = "dist-from-sea-missing"
ERR_DIST_FROM_SEA_SHAPE_MISMATCH = "dist-from-sea-shape-mismatch"
ERR_DIST_FROM_SEA_NAN_ON_WATER = "dist-from-sea-nan-on-water"
ERR_DIST_FROM_SEA_ALL_ZERO = "dist-from-sea-all-zero"
ERR_DIST_FROM_SEA_NO_SOURCES = "dist-from-sea-no-sources"
```

Then in `H3Environment.from_netcdf`, AFTER the existing `return cls(mesh=mesh, full_fields=full_fields, time=ds["time"].values)` line: refactor to capture the env first, run the dist_from_sea load step, then return. Replace the existing return with:

```python
        env = cls(mesh=mesh, full_fields=full_fields, time=ds["time"].values)
        _load_dist_from_sea(env, ds, mesh)
        ds.close()
        return env
```

**Important — `ds.close()` ordering and ownership.** `_load_dist_from_sea` MUST NOT close `ds` itself; it only reads from `ds`. The single `ds.close()` belongs in `from_netcdf` AFTER `_load_dist_from_sea` returns. Do NOT add a `ds.close()` inside the helper; if you do, the second close from `from_netcdf` will raise. Verify the existing `from_netcdf` body: locate the current `return cls(...)` (around `h3_env.py:127`), check whether there's any `ds.close()` already (there is not in v1.7.7); if absent, the new line you add IS the only close — do not add a second one. Verify line ordering: open → fields → ssh zero-fill → cls() → load_dist_from_sea(env, ds, mesh) → ds.close() → return env.

Add the helper function at module level (after the H3Environment class):

```python
def _load_dist_from_sea(env, ds, mesh) -> None:
    """C4: load `dist_from_sea` from NC into env.fields and mesh.

    Two cases per spec §"Where it lives → Validation discipline →
    Sim-init":

    Case A — variable absent: warn + zero-fill + flag init.
    Case B — variable present but structurally invalid: raise.

    On all-checks-pass: inject env.fields["dist_from_sea"] +
    mesh.dist_from_sea (same array reference) + flag init.
    """
    import logging
    import numpy as np

    logger = logging.getLogger("salmon_ibm.h3_env")
    n = mesh.N_cells if hasattr(mesh, "N_cells") else len(mesh.reach_id)

    if "dist_from_sea" not in ds.variables:
        # Case A: backward-compat for pre-C4 NCs.
        logger.warning(
            "%s: dist_from_sea missing from NC; movement gradient will "
            "be flat — agents will not migrate. Rebuild landscape with "
            "build_h3_multires_landscape.py.",
            ERR_DIST_FROM_SEA_MISSING,
        )
        zero = np.zeros(n, dtype=np.float32)
        env.fields["dist_from_sea"] = zero
        mesh.dist_from_sea = zero
        env._dormant_gradient_check_done = False
        return

    # Case B placeholder (Task 4 fills this in).
    arr = ds["dist_from_sea"].values
    arr32 = arr.astype(np.float32)
    env.fields["dist_from_sea"] = arr32
    mesh.dist_from_sea = arr32
    env._dormant_gradient_check_done = False
```

- [ ] **Step 4: Run focused test to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_h3_env.py::test_from_netcdf_case_a_dist_from_sea_missing -v`
Expected: PASS.

- [ ] **Step 5: Smoke-test h3_env**

Run: `micromamba run -n shiny python -m pytest tests/test_h3_env.py -q`
Expected: All existing h3_env tests pass — the change is additive.

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/h3_env.py tests/test_h3_env.py
git commit -m "feat(c4): h3_env Case A path + err-id constants + Test 4"
```

---

## Task 4: Case B structural validation (4 raise paths) + Tests 4a-4d

**Files:**
- Modify: `salmon_ibm/h3_env.py` — extend `_load_dist_from_sea` with the 4 structural checks.
- Test: `tests/test_h3_env.py`

- [ ] **Step 1: Append failing tests for Case B raise paths**

Append to `tests/test_h3_env.py`:

```python
def test_from_netcdf_case_b_shape_mismatch_raises(tmp_path):
    """C4 Test 4a: NC has dist_from_sea but wrong shape → raise."""
    import numpy as np
    from salmon_ibm.h3_env import H3Environment

    nc_path = tmp_path / "shape_mismatch.nc"
    # 4-cell mesh, but dist_from_sea has 9 elements.
    bad = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    _build_minimal_h3_nc(nc_path, dist_from_sea_arr=bad)
    mesh = _build_mesh_from_nc(nc_path)

    with pytest.raises(RuntimeError, match=r"dist-from-sea-shape-mismatch"):
        H3Environment.from_netcdf(str(nc_path), mesh)


def test_from_netcdf_case_b_nan_on_water_raises(tmp_path):
    """C4 Test 4b: dist_from_sea has NaN at a water cell → raise."""
    import numpy as np
    from salmon_ibm.h3_env import H3Environment

    nc_path = tmp_path / "nan_on_water.nc"
    # Cells 0,1,2 are water; cell 3 is land. Inject NaN at cell 1.
    bad = np.array([0.0, np.nan, 200.0, np.nan], dtype=np.float32)
    _build_minimal_h3_nc(nc_path, dist_from_sea_arr=bad)
    mesh = _build_mesh_from_nc(nc_path)

    with pytest.raises(RuntimeError, match=r"dist-from-sea-nan-on-water"):
        H3Environment.from_netcdf(str(nc_path), mesh)


def test_from_netcdf_case_b_nan_on_land_does_not_raise(tmp_path):
    """C4 Test 4b sanity: NaN at land cell (water_mask=False) does
    NOT trigger the water-NaN raise."""
    import numpy as np
    from salmon_ibm.h3_env import H3Environment

    nc_path = tmp_path / "nan_on_land.nc"
    # Land cell 3 has NaN; water cells 0,1,2 are valid.
    ok = np.array([0.0, 100.0, 200.0, np.nan], dtype=np.float32)
    _build_minimal_h3_nc(nc_path, dist_from_sea_arr=ok)
    mesh = _build_mesh_from_nc(nc_path)

    # No raise.
    env = H3Environment.from_netcdf(str(nc_path), mesh)
    assert np.isnan(env.fields["dist_from_sea"][3])  # land cell stays NaN
    assert env.fields["dist_from_sea"][0] == 0.0
    # Identity assertion: env.fields and mesh point to the SAME array
    # (shared reference, not independent copies). Pass-1 silent-failure
    # finding: a future refactor that copies via arr.copy() would create
    # divergence invisible to the value-equality tests.
    assert mesh.dist_from_sea is env.fields["dist_from_sea"]
    # Per-env latch flag initialized on the Case B happy path.
    assert env._dormant_gradient_check_done is False


def test_from_netcdf_case_b_all_zero_raises(tmp_path):
    """C4 Test 4c: dist_from_sea is all zeros → raise."""
    import numpy as np
    from salmon_ibm.h3_env import H3Environment

    nc_path = tmp_path / "all_zero.nc"
    bad = np.zeros(4, dtype=np.float32)
    _build_minimal_h3_nc(nc_path, dist_from_sea_arr=bad)
    mesh = _build_mesh_from_nc(nc_path)

    with pytest.raises(RuntimeError, match=r"dist-from-sea-all-zero"):
        H3Environment.from_netcdf(str(nc_path), mesh)


def test_from_netcdf_case_b_no_sources_raises(tmp_path):
    """C4 Test 4d: no OpenBaltic cell at distance 0 → raise."""
    import numpy as np
    from salmon_ibm.h3_env import H3Environment

    nc_path = tmp_path / "no_sources.nc"
    # All distances >= 1, even the OpenBaltic cell (cell 0).
    bad = np.array([5.0, 100.0, 200.0, np.nan], dtype=np.float32)
    _build_minimal_h3_nc(nc_path, dist_from_sea_arr=bad)
    mesh = _build_mesh_from_nc(nc_path)

    with pytest.raises(RuntimeError, match=r"dist-from-sea-no-sources"):
        H3Environment.from_netcdf(str(nc_path), mesh)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_h3_env.py -k case_b -v`
Expected: 4 failures + 1 pass (the nan-on-land sanity test passes via the existing minimal Case A→Case B placeholder).

- [ ] **Step 3: Extend `_load_dist_from_sea` with the 4 structural checks**

In `salmon_ibm/h3_env.py`, replace the Case B placeholder block (the section after `if "dist_from_sea" not in ds.variables:` early-return) with the full validation:

```python
    # Case B: variable present — run 4 structural checks. Each
    # failure raises RuntimeError with the matching err-id; no
    # zero-fill, no graceful continuation.
    arr = ds["dist_from_sea"].values

    # (a) Shape match.
    if arr.shape != (n,):
        raise RuntimeError(
            f"{ERR_DIST_FROM_SEA_SHAPE_MISMATCH}: dist_from_sea shape "
            f"{arr.shape} != expected ({n},). Stale NC built against a "
            "different mesh? Rebuild with "
            "build_h3_multires_landscape.py."
        )

    # (b) No NaN/Inf on water cells.
    water_arr = arr[mesh.water_mask]
    if not np.all(np.isfinite(water_arr)):
        n_bad = int(np.sum(~np.isfinite(water_arr)))
        raise RuntimeError(
            f"{ERR_DIST_FROM_SEA_NAN_ON_WATER}: dist_from_sea has "
            f"{n_bad} NaN/Inf value(s) on water cells. NC build is "
            "corrupt; rebuild required."
        )

    # (c) max > 0 — catches all-zero from a build that crashed
    # mid-Dijkstra but still wrote the file.
    if float(arr.max()) <= 0.0:
        raise RuntimeError(
            f"{ERR_DIST_FROM_SEA_ALL_ZERO}: dist_from_sea has max "
            f"{float(arr.max())}, expected > 0. NC was written but "
            "Dijkstra didn't run; rebuild required."
        )

    # (d) Sources exist — at least one OpenBaltic water cell at
    # distance 0. Skip the entire check if "OpenBaltic" not in
    # reach_names (non-Baltic mesh — no validation possible).
    if "OpenBaltic" in mesh.reach_names:
        ob_id = mesh.reach_names.index("OpenBaltic")
        ob_mask = (mesh.reach_id == ob_id) & mesh.water_mask
        if not ob_mask.any():
            # Mesh declares OpenBaltic reach but has zero water cells
            # in it. Degenerate mesh — raise (not a silent skip, per
            # pass-1 review-loop finding).
            raise RuntimeError(
                f"{ERR_DIST_FROM_SEA_NO_SOURCES}: mesh has reach "
                "'OpenBaltic' but no water_mask=True cells in it. "
                "Cannot validate sources; rebuild mesh."
            )
        if not np.any(arr[ob_mask] == 0.0):
            raise RuntimeError(
                f"{ERR_DIST_FROM_SEA_NO_SOURCES}: no OpenBaltic water "
                "cell has dist_from_sea == 0; the source set is empty "
                "or the build used a different source definition."
            )

    # All checks pass: inject (single cast, shared reference).
    arr32 = arr.astype(np.float32)
    env.fields["dist_from_sea"] = arr32
    mesh.dist_from_sea = arr32
    env._dormant_gradient_check_done = False
```

- [ ] **Step 4: Run focused tests to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_h3_env.py -k case_b -v`
Expected: 5 PASS (4 raise tests + nan-on-land sanity).

- [ ] **Step 5: Smoke-test full h3_env suite**

Run: `micromamba run -n shiny python -m pytest tests/test_h3_env.py -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/h3_env.py tests/test_h3_env.py
git commit -m "feat(c4): h3_env Case B structural validation + Tests 4a-4d"
```

---

## Task 5: `simulation.py` Landscape TypedDict + dict construction

**Files:**
- Modify: `salmon_ibm/simulation.py` — add `env` field to `Landscape` TypedDict; add `"env": ...` to landscape dict construction at the existing `step()` site.

- [ ] **Step 1: Locate the existing Landscape TypedDict + the landscape-dict construction**

Run: `grep -n "Landscape\|class Landscape\|TypedDict" salmon_ibm/simulation.py | head -10`
Run: `grep -n '"fields":\|"rng":\|"mesh":' salmon_ibm/simulation.py | head -10`

Note the line numbers. The TypedDict definition is around `simulation.py:12-33`. The landscape construction site is the `step()` method's dict literal — find the line that builds the dict passed to `EventSequencer.step` or to events. Record the path: do NOT proceed until you can name both line ranges.

- [ ] **Step 2: Identify the env attribute on `Simulation`**

Run: `grep -n "self.env\s*=\|self\.env\b" salmon_ibm/simulation.py | head -10`

Expected: `self.env = H3Environment.from_netcdf(...)` is set in the H3-multires init branch (around `simulation.py:181`). For non-H3 paths (TriMesh, HexMesh, hexsim), `self.env` may be a different class (`Environment`, `HexSimEnvironment`). Confirm — the dormancy check only fires when `env` is an `H3Environment`-shaped object with `fields["dist_from_sea"]` and `_dormant_gradient_check_done`. For non-H3 envs, the check is a no-op.

- [ ] **Step 3: Update the TypedDict**

In `salmon_ibm/simulation.py`, locate the `Landscape(TypedDict)` definition. Add the new field. Match the existing convention (probably `total=False` since the dict has many optional keys):

```python
class Landscape(TypedDict, total=False):
    # ... existing fields ...
    env: "H3Environment | Environment | HexSimEnvironment | None"  # NEW (C4)
```

If the existing TypedDict uses string annotations, keep that style. If `H3Environment` isn't already imported at the top via `TYPE_CHECKING`, add:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from salmon_ibm.h3_env import H3Environment
```

- [ ] **Step 4: Update the landscape dict construction**

In `simulation.py`'s `step()` method, locate the dict literal. Add `"env": self.env` (direct attribute access, NOT `getattr(...)`):

```python
landscape: Landscape = {
    # ... existing keys ...
    "env": self.env,  # NEW (C4) — for the dormancy check in
                      # MovementEvent.execute. self.env is set in
                      # __init__ for ALL backends (H3 / TriMesh /
                      # HexMesh / HexSim) — direct access, not
                      # getattr, so a future refactor that renames
                      # self.env fails LOUD instead of silently
                      # leaving env=None.
}
```

(Pass-1 silent-failure finding: `getattr(self, "env", None)` would silently mask a renamed-attribute regression. Direct access fails fast — preferred per the project's fail-fast convention.)

- [ ] **Step 5: Smoke-test simulation**

Run: `micromamba run -n shiny python -m pytest tests/test_simulation.py -q`
Expected: All pass — the change is additive.

If a test breaks because its synthetic `landscape` dict lacks the `env` key, that's a fixture issue. Per the TypedDict's `total=False`, the key is optional — but tests that build a landscape and then look up `landscape["env"]` would KeyError. Fix in the test fixture.

Run: `micromamba run -n shiny python -m pytest tests/test_movement.py tests/test_delta_routing.py -q`
Expected: All pass — these tests don't touch `landscape["env"]` directly (the dormancy check is gated on the helper, which is added in Task 6).

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/simulation.py
git commit -m "feat(c4): simulation.py Landscape TypedDict + dict gain env key"
```

---

## Task 6: `_check_dormant_gradient` helper + Tests 4e, 4f

**Architectural correction (v2):** Pass-1 review-loop confirmed
`execute_movement(pool, mesh, fields, ...)` takes `fields` as the
third positional arg, NOT a `landscape` dict. The dormancy check
therefore CANNOT live inside `execute_movement` (which has no `env`
or `pool.behavior` access through `fields`). The check belongs at
the **`MovementEvent.execute` layer** in `salmon_ibm/events_builtin.py`
(line 25-48), which already receives `landscape` and `population`.
Helper signature: `_check_dormant_gradient(landscape, pool)` — the
helper computes `has_directed` from `pool.behavior` directly (no
buckets dependency).

**Files:**
- Modify: `salmon_ibm/movement.py` — add the helper at module level (still in movement.py for cohesion with the kernel it guards).
- Modify: `salmon_ibm/events_builtin.py` — call the helper from `MovementEvent.execute` BEFORE `execute_movement`.
- Test: `tests/test_movement_gradient.py`

- [ ] **Step 1: Append failing tests for the latched check**

Helper signature: `_check_dormant_gradient(landscape, pool)`. The helper
reads `pool.behavior` to compute whether any agent is in
UPSTREAM/DOWNSTREAM. No buckets parameter.

Append to `tests/test_movement_gradient.py`:

```python
def _make_landscape(env, dist_from_sea_arr=None):
    """Helper: minimal landscape dict for dormancy-check tests.

    `dist_from_sea_arr` overrides what the helper inspects; if None,
    uses env.fields["dist_from_sea"] as-is.
    """
    fields = dict(env.fields) if hasattr(env, "fields") else {}
    if dist_from_sea_arr is not None:
        fields["dist_from_sea"] = dist_from_sea_arr
    return {
        "env": env,
        "fields": fields,
        "rng": np.random.default_rng(0),
    }


def _make_pool(behaviors):
    """Helper: minimal pool stand-in with .behavior array."""
    class _FakePool:
        pass
    pool = _FakePool()
    pool.behavior = np.asarray(behaviors, dtype=np.int8)
    return pool


def test_check_dormant_gradient_raises_on_flat_zero_with_directed_agents():
    """C4 Test 4e (part 1): env-A loaded via Case A path → flat-zero →
    directed agent → raise containing the err-id."""
    from salmon_ibm.movement import _check_dormant_gradient
    from salmon_ibm.agents import Behavior
    from salmon_ibm.h3_env import ERR_DIST_FROM_SEA_MISSING

    class _FakeEnv:
        pass
    env_a = _FakeEnv()
    env_a.fields = {"dist_from_sea": np.zeros(10, dtype=np.float32)}
    env_a._dormant_gradient_check_done = False

    pool = _make_pool([int(Behavior.UPSTREAM), int(Behavior.HOLD)])
    landscape = _make_landscape(env_a)

    with pytest.raises(RuntimeError, match=ERR_DIST_FROM_SEA_MISSING):
        _check_dormant_gradient(landscape, pool)


def test_check_dormant_gradient_per_env_isolation():
    """C4 Test 4e (part 2): env-A's latch does NOT affect env-B's
    independent check. Regression test for the pass-7 module-global
    → per-env-instance refactor."""
    from salmon_ibm.movement import _check_dormant_gradient
    from salmon_ibm.agents import Behavior

    class _FakeEnv:
        pass

    env_a = _FakeEnv()
    env_a.fields = {"dist_from_sea": np.zeros(10, dtype=np.float32)}
    env_a._dormant_gradient_check_done = False

    pool = _make_pool([int(Behavior.UPSTREAM)])

    with pytest.raises(RuntimeError):
        _check_dormant_gradient(_make_landscape(env_a), pool)
    assert env_a._dormant_gradient_check_done is True

    env_b = _FakeEnv()
    env_b.fields = {"dist_from_sea": np.zeros(10, dtype=np.float32)}
    env_b._dormant_gradient_check_done = False
    with pytest.raises(RuntimeError):
        _check_dormant_gradient(_make_landscape(env_b), pool)
    assert env_b._dormant_gradient_check_done is True


def test_check_dormant_gradient_happy_path_latches(tmp_path, caplog):
    """C4 Test 4f: load env via H3Environment.from_netcdf with a
    valid Case-B NC; check fires no raise and latches True. Tests the
    end-to-end Case-B init → helper-call sequence."""
    from salmon_ibm.movement import _check_dormant_gradient
    from salmon_ibm.h3_env import H3Environment
    from salmon_ibm.agents import Behavior

    nc_path = tmp_path / "happy_path.nc"
    valid = np.array([0.0, 100.0, 200.0, np.nan], dtype=np.float32)
    _build_minimal_h3_nc(nc_path, dist_from_sea_arr=valid)
    mesh = _build_mesh_from_nc(nc_path)
    env = H3Environment.from_netcdf(str(nc_path), mesh)
    assert env._dormant_gradient_check_done is False  # Case B init OK

    pool = _make_pool([int(Behavior.UPSTREAM)])
    # No raise — gradient has positive values.
    _check_dormant_gradient(_make_landscape(env), pool)
    assert env._dormant_gradient_check_done is True

    # Second call: latched. Even with the gradient now-zeroed, no raise.
    env.fields["dist_from_sea"][:] = 0.0
    _check_dormant_gradient(_make_landscape(env), pool)


def test_check_dormant_gradient_no_directed_agents_no_raise():
    """C4 sanity: flat-zero gradient + only HOLD/RANDOM/TO_CWR agents
    → no raise (the check is gated on UPSTREAM/DOWNSTREAM presence)."""
    from salmon_ibm.movement import _check_dormant_gradient
    from salmon_ibm.agents import Behavior

    class _FakeEnv:
        pass
    env = _FakeEnv()
    env.fields = {"dist_from_sea": np.zeros(10, dtype=np.float32)}
    env._dormant_gradient_check_done = False

    pool = _make_pool([
        int(Behavior.HOLD),
        int(Behavior.RANDOM),
        int(Behavior.TO_CWR),
    ])
    _check_dormant_gradient(_make_landscape(env), pool)
    assert env._dormant_gradient_check_done is True


def test_check_dormant_gradient_no_env_in_landscape_no_raise():
    """C4 sanity: legacy non-Baltic landscape (no env key) → no-op."""
    from salmon_ibm.movement import _check_dormant_gradient
    from salmon_ibm.agents import Behavior

    landscape = {
        "fields": {"dist_from_sea": np.zeros(10, dtype=np.float32)},
        "rng": np.random.default_rng(0),
        # NO "env" key.
    }
    pool = _make_pool([int(Behavior.UPSTREAM)])
    _check_dormant_gradient(landscape, pool)  # must not raise
```

- [ ] **Step 2: Run tests to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_gradient.py -k check_dormant -v`
Expected: ImportError on `_check_dormant_gradient`.

- [ ] **Step 3: Verify Behavior import in movement.py is at module level**

Run: `grep -n "from salmon_ibm.agents import Behavior\|^from salmon_ibm" salmon_ibm/movement.py`
Expected: a top-level `from salmon_ibm.agents import Behavior` exists at line ~31.

If the import is inside a function (lazy), promote it to module level — the helper added in Step 4 references `Behavior` at module scope and will fail otherwise.

- [ ] **Step 4: Add `_check_dormant_gradient` to `movement.py`**

In `salmon_ibm/movement.py`, add after the existing imports (after line 31's `from salmon_ibm.agents import Behavior`), before any kernel:

```python
from salmon_ibm.h3_env import ERR_DIST_FROM_SEA_MISSING


def _check_dormant_gradient(landscape, pool) -> None:
    """C4: raise once per env-instance if `dist_from_sea` is flat-zero
    AND any agent is in UPSTREAM/DOWNSTREAM behavior.

    Signature: takes `landscape` (dict) + `pool` (AgentPool stand-in
    with .behavior int8 array). Computes `has_directed` from
    pool.behavior directly — does not depend on the kernel's bucket
    construction. This decouples the dormancy check from
    execute_movement's internals so the check can be called from
    MovementEvent.execute (which does not have buckets).

    Latch is per-env-instance via `env._dormant_gradient_check_done`,
    NOT module-global — pass-7 review-loop pinned this to avoid
    cross-test landscape-swap leakage. Initialised to False by
    H3Environment.from_netcdf in both Case A and Case B paths.

    No-op if landscape has no `env` key (legacy non-Baltic) OR if
    the env's flag is already True (latched after first call).
    """
    env = landscape.get("env")
    if env is None or getattr(env, "_dormant_gradient_check_done", True):
        return  # legacy env, or check already run

    has_directed = bool(np.any(
        (pool.behavior == int(Behavior.UPSTREAM))
        | (pool.behavior == int(Behavior.DOWNSTREAM))
    ))
    if has_directed and not np.any(landscape["fields"]["dist_from_sea"]):
        env._dormant_gradient_check_done = True  # latch BEFORE raise
        raise RuntimeError(
            f"{ERR_DIST_FROM_SEA_MISSING}: dist_from_sea is flat-zero "
            "AND agents are in UPSTREAM/DOWNSTREAM behavior. Movement "
            "will not progress (legacy SSH=0 dormant state). Rebuild "
            "the landscape NC with build_h3_multires_landscape.py to "
            "populate dist_from_sea."
        )
    env._dormant_gradient_check_done = True
```

- [ ] **Step 5: Wire the helper into `MovementEvent.execute`**

In `salmon_ibm/events_builtin.py`, modify `MovementEvent.execute`. The current body (lines 25-48) starts:

```python
def execute(self, population, landscape, t, mask):
    mesh = landscape["mesh"]
    fields = landscape["fields"]
    rng = landscape["rng"]
    barrier_arrays = landscape.get("barrier_arrays")
    n_micro = landscape.get("n_micro_steps_per_cell")
    ...
```

Add the dormancy check at the top of the method body, immediately after the docstring (if any) but BEFORE the `landscape["mesh"]` access:

```python
def execute(self, population, landscape, t, mask):
    # C4: latched dormancy check. No-ops on legacy non-Baltic landscapes
    # (no "env" key) or after first successful call (latched).
    from salmon_ibm.movement import _check_dormant_gradient
    _check_dormant_gradient(landscape, population.pool)

    mesh = landscape["mesh"]
    ...
```

Local import (`from salmon_ibm.movement import ...` inside the method) avoids any circular-import risk if events_builtin is imported before movement during Python's module load — defensive.

- [ ] **Step 6: Run focused tests to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_gradient.py -k check_dormant -v`
Expected: 5 PASS.

- [ ] **Step 7: Smoke-test movement + simulation suites**

Run: `micromamba run -n shiny python -m pytest tests/test_movement.py tests/test_delta_routing.py tests/test_simulation.py tests/test_h3_env.py -q`
Expected: All pass. The MovementEvent.execute change adds one line at the top; on legacy fixtures (no "env" in landscape), `_check_dormant_gradient` no-ops.

If a test fails because its synthetic `population` doesn't have `.pool` (some legacy tests pass a bare `Pool`-shaped object), inspect: `population.pool` is the standard interface via `Population` wrapper. If the test passes `pool` directly as `population`, the helper line above access `population.pool` and AttributeErrors. Fix: make the helper resilient by trying `population.pool` first then `population` as fallback, OR fix the test fixture.

```python
# Defensive variant for the helper call site:
pool = getattr(population, "pool", population)
_check_dormant_gradient(landscape, pool)
```

Use the defensive variant — broadens compat without weakening the check.

- [ ] **Step 8: Commit**

```bash
git add salmon_ibm/movement.py salmon_ibm/events_builtin.py tests/test_movement_gradient.py
git commit -m "feat(c4): _check_dormant_gradient at MovementEvent.execute layer + Tests 4e/4f"
```

---

## Task 7: Movement kernel UPSTREAM/DOWNSTREAM swap + Tests 1, 2, 2b, 3

**Files:**
- Modify: `salmon_ibm/movement.py` — UPSTREAM/DOWNSTREAM dispatch (lines 94-127): `fields["ssh"]` → `fields["dist_from_sea"]`; flip `ascending` flag at lines 107 (False→True) and 125 (True→False).
- Test: `tests/test_movement_gradient.py`

**Important — `execute_movement` signature.** The kernel takes
`(pool, mesh, fields, seed=None, n_micro_steps_per_cell=None, ...)`.
The third positional argument is `fields` (a `dict[str, np.ndarray]`),
NOT a landscape dict. All Task 7 tests call:

```python
execute_movement(
    pool, mesh, fields,
    seed=42,
    n_micro_steps_per_cell=np.ones(n_cells, dtype=np.int32),
)
```

The dormancy check happens at the `MovementEvent.execute` layer
(Task 6), NOT inside `execute_movement` itself. Tests 1/2/2b/3 do
NOT exercise the dormancy check; they exercise only the kernel
field swap.

`_FakeMesh` for these tests must include `n_triangles = N` because
`execute_movement` line 53-54 reads `mesh.n_triangles` when
`n_micro_steps_per_cell is None`. Pass `n_micro_steps_per_cell`
explicitly to bypass that path, but also set `n_triangles` defensively.

- [ ] **Step 1: Append failing tests 1, 2, 2b, 3 — with shared chain-mesh fixture**

Append to `tests/test_movement_gradient.py`. First, the shared fixture:

```python
def _make_chain_mesh_fake(n: int = 10):
    """Build a `_FakeMesh` for kernel-direct tests (Tests 1, 2, 2b, 3).

    Bidirectional chain: cell `i` has neighbor i+1 (slot 0) and i-1
    (slot 1) where they exist. Endpoints (0 and n-1) have one neighbor.

    Returns a fresh mesh per call so tests don't share state.
    """
    water_nbrs = np.full((n, 2), -1, dtype=np.int32)
    for i in range(n - 1):
        water_nbrs[i, 0] = i + 1
    for i in range(1, n):
        water_nbrs[i, 1] = i - 1
    water_nbr_count = np.array(
        [1] + [2] * (n - 2) + [1], dtype=np.int32,
    )

    class _FakeMesh:
        pass
    mesh = _FakeMesh()
    mesh._water_nbrs = water_nbrs
    mesh._water_nbr_count = water_nbr_count
    mesh.n_triangles = n  # required by execute_movement when
                          # n_micro_steps_per_cell defaults to None
    return mesh


def _make_chain_fields(n: int = 10, *, gradient: bool = True):
    """Returns the `fields` dict for the kernel: dist_from_sea +
    sentinel ssh (for safety against accidental SSH coupling tests).

    `gradient=True`: dist_from_sea[i] = i * 100.0 (monotonic).
    `gradient=False`: all zeros (dormant state).
    """
    if gradient:
        dist = np.arange(n, dtype=np.float32) * 100.0
    else:
        dist = np.zeros(n, dtype=np.float32)
    return {"dist_from_sea": dist}


def test_upstream_climbs_chain_one_step():
    """C4 Test 1 part 1: UPSTREAM agent at cell 0 climbs to cell 1
    after 1 timestep with n_micro=1."""
    from salmon_ibm.movement import execute_movement
    from salmon_ibm.agents import AgentPool, Behavior

    n = 10
    mesh = _make_chain_mesh_fake(n)
    fields = _make_chain_fields(n, gradient=True)

    # Silent-failure guard: assert ssh is NOT in fields, so a kernel
    # that still reads ssh would KeyError instead of silently passing.
    assert "ssh" not in fields

    pool = AgentPool(n=1, start_tri=0, rng_seed=42)
    pool.behavior[0] = int(Behavior.UPSTREAM)

    execute_movement(
        pool, mesh, fields,
        seed=42,
        n_micro_steps_per_cell=np.ones(n, dtype=np.int32),
    )
    assert pool.tri_idx[0] == 1  # climbed one step


def test_upstream_reaches_far_cell_after_5_steps():
    """C4 Test 1 part 2: 5 timesteps → agent at cell ≥ 5."""
    from salmon_ibm.movement import execute_movement
    from salmon_ibm.agents import AgentPool, Behavior

    n = 10
    mesh = _make_chain_mesh_fake(n)
    fields = _make_chain_fields(n, gradient=True)

    pool = AgentPool(n=1, start_tri=0, rng_seed=42)
    pool.behavior[0] = int(Behavior.UPSTREAM)

    for _ in range(5):
        execute_movement(
            pool, mesh, fields,
            seed=42,
            n_micro_steps_per_cell=np.ones(n, dtype=np.int32),
        )
    assert pool.tri_idx[0] >= 5


def test_downstream_descends_chain():
    """C4 Test 2: DOWNSTREAM agent at cell 9 ends at cell ≤ 4 after
    5 timesteps."""
    from salmon_ibm.movement import execute_movement
    from salmon_ibm.agents import AgentPool, Behavior

    n = 10
    mesh = _make_chain_mesh_fake(n)
    fields = _make_chain_fields(n, gradient=True)

    pool = AgentPool(n=1, start_tri=9, rng_seed=42)
    pool.behavior[0] = int(Behavior.DOWNSTREAM)

    for _ in range(5):
        execute_movement(
            pool, mesh, fields,
            seed=42,
            n_micro_steps_per_cell=np.ones(n, dtype=np.int32),
        )
    assert pool.tri_idx[0] <= 4


def test_upstream_at_mesh_edge_does_not_crash():
    """C4 Test 2b: UPSTREAM agent at cell 9 (mesh edge / max-gradient)
    has no higher neighbor; falls back to slot-0 / random. Stays
    within {8, 9} for 5 timesteps."""
    from salmon_ibm.movement import execute_movement
    from salmon_ibm.agents import AgentPool, Behavior

    n = 10
    mesh = _make_chain_mesh_fake(n)
    fields = _make_chain_fields(n, gradient=True)

    pool = AgentPool(n=1, start_tri=9, rng_seed=42)
    pool.behavior[0] = int(Behavior.UPSTREAM)

    for _ in range(5):
        execute_movement(
            pool, mesh, fields,
            seed=42,
            n_micro_steps_per_cell=np.ones(n, dtype=np.int32),
        )
    # Did not crash; oscillates near edge.
    assert pool.tri_idx[0] in {8, 9}


def test_zero_gradient_fallback_behavioral():
    """C4 Test 3: zero gradient + UPSTREAM at cell 5. Calls
    execute_movement directly (not via MovementEvent) so the dormancy
    raise doesn't fire — this test is about the kernel's degenerate
    behavior, not the dormancy check.

    Agent oscillates within {4, 5, 6}.
    """
    from salmon_ibm.movement import execute_movement
    from salmon_ibm.agents import AgentPool, Behavior

    n = 10
    mesh = _make_chain_mesh_fake(n)
    fields = _make_chain_fields(n, gradient=False)  # all-zero gradient

    pool = AgentPool(n=1, start_tri=5, rng_seed=42)
    pool.behavior[0] = int(Behavior.UPSTREAM)

    for _ in range(10):
        execute_movement(
            pool, mesh, fields,
            seed=42,
            n_micro_steps_per_cell=np.ones(n, dtype=np.int32),
        )
    # Stayed within immediate neighborhood (oscillation).
    assert pool.tri_idx[0] in {4, 5, 6}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_gradient.py -k "upstream or downstream or fallback" -v`
Expected: 5 failures — kernel still reads `fields["ssh"]` (which the test doesn't populate); the agent doesn't move toward dist_from_sea.

- [ ] **Step 3: Swap field + flip `ascending` in movement.py**

In `salmon_ibm/movement.py`, locate the UPSTREAM dispatch block at lines 93-109. Change two lines:
- Line 102: `fields["ssh"]` → `fields["dist_from_sea"]`
- Line 107: `ascending=False` → `ascending=True`

Locate the DOWNSTREAM dispatch block at lines 111-127. Change two lines:
- Line 120: `fields["ssh"]` → `fields["dist_from_sea"]`
- Line 125: `ascending=True` → `ascending=False`

Resulting blocks (UPSTREAM):

```python
    # --- UPSTREAM ---
    idx = buckets.get(int(Behavior.UPSTREAM))
    if idx is not None and len(idx) > 0:
        tri_buf = pool.tri_idx[idx].copy()
        fraction_remaining = np.ones(len(tri_buf), dtype=np.float32)
        _step_directed_vec(
            tri_buf,
            water_nbrs,
            water_nbr_count,
            fields["dist_from_sea"],   # was fields["ssh"]
            rng,
            max_steps,
            n_micro_steps_per_cell,
            fraction_remaining,
            ascending=True,            # was ascending=False
        )
        pool.tri_idx[idx] = tri_buf
```

Resulting block (DOWNSTREAM):

```python
    # --- DOWNSTREAM ---
    idx = buckets.get(int(Behavior.DOWNSTREAM))
    if idx is not None and len(idx) > 0:
        tri_buf = pool.tri_idx[idx].copy()
        fraction_remaining = np.ones(len(tri_buf), dtype=np.float32)
        _step_directed_vec(
            tri_buf,
            water_nbrs,
            water_nbr_count,
            fields["dist_from_sea"],   # was fields["ssh"]
            rng,
            max_steps,
            n_micro_steps_per_cell,
            fraction_remaining,
            ascending=False,           # was ascending=True
        )
        pool.tri_idx[idx] = tri_buf
```

- [ ] **Step 4: Run focused tests to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_gradient.py -k "upstream or downstream or fallback" -v`
Expected: 5 PASS.

- [ ] **Step 5: Smoke-test broader movement suite**

Run: `micromamba run -n shiny python -m pytest tests/test_movement.py tests/test_delta_routing.py tests/test_simulation.py tests/test_h3_env.py -q`
Expected: All pass.

If any existing test references `fields["ssh"]` and breaks: those test fixtures need a `dist_from_sea` entry. Update inline (search-and-replace pattern: a test fixture that builds a landscape dict and passes it to `execute_movement` with UPSTREAM/DOWNSTREAM agents needs `fields["dist_from_sea"]`).

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/movement.py tests/test_movement_gradient.py
git commit -m "feat(c4): movement.py UPSTREAM/DOWNSTREAM use dist_from_sea + ascending flip + Tests 1/2/2b/3"
```

---

## Task 8: Build script integration + NC rebuild + Test 6

**Files:**
- Modify: `scripts/build_h3_multires_landscape.py` — wire the `compute_dist_from_sea` call into the main build flow + write to NC + per-reach sanity output.
- Test: `tests/test_movement_gradient.py`
- Rebuild: `data/curonian_h3_multires_landscape.nc`

- [ ] **Step 1: Locate the existing NC-writing block in the build script**

Run: `grep -n "to_netcdf\|nbr_starts\|dist_from_sea" scripts/build_h3_multires_landscape.py`
Expected: a `Dataset(...)` construction or `to_netcdf(...)` call near the bottom of the script.

- [ ] **Step 2: Wire `compute_dist_from_sea` into the build flow**

In `scripts/build_h3_multires_landscape.py`, AFTER the H3MultiResMesh is constructed and BEFORE `to_netcdf`, add:

```python
    print("Computing dist_from_sea (multi-source Dijkstra)...")
    dist_from_sea = compute_dist_from_sea(mesh)
    print(f"  shape: {dist_from_sea.shape}, dtype: {dist_from_sea.dtype}")

    # Per-reach sanity output.
    print("  per-reach distribution:")
    for rid, name in enumerate(mesh.reach_names):
        cells = (mesh.reach_id == rid) & mesh.water_mask
        n_cells = int(cells.sum())
        if n_cells == 0:
            continue
        valid = dist_from_sea[cells]
        valid = valid[np.isfinite(valid)]
        if len(valid) == 0:
            continue
        print(
            f"    {name:<18} cells={n_cells:>6}  "
            f"min={float(valid.min()):>10.0f}m  "
            f"max={float(valid.max()):>10.0f}m  "
            f"mean={float(valid.mean()):>10.0f}m"
        )
        # Pass-1 silent-failure finding: surface degenerate reaches
        # explicitly rather than silently `continue` past them.
        if len(valid) < n_cells:
            print(
                f"    WARNING: {name} has {n_cells - len(valid)} "
                f"non-finite cells (out of {n_cells})"
            )
```

Add `dist_from_sea` to the `Dataset` `data_vars` or the `to_netcdf` call:

```python
    ds_out["dist_from_sea"] = (("cell",), dist_from_sea)
```

(Adapt to the actual style used in the script — variable assignment vs Dataset constructor.)

- [ ] **Step 3: Rebuild the production NC**

Run: `micromamba run -n shiny python scripts/build_h3_multires_landscape.py`
Expected: build completes (exit 0); final output prints per-reach distribution; OpenBaltic mean < BalticCoast/CuronianLagoon mean < Atmata/Skirvyte/Gilija mean < Nemunas mean (roughly — CuronianLagoon may be bimodal).

If the script exits non-zero (e.g., raises mid-run), the NC may be partially-written. STOP and investigate before proceeding to Step 4.

- [ ] **Step 3.5: Programmatic post-build verification**

Run:
```bash
micromamba run -n shiny python -c "
import xarray as xr
import numpy as np
ds = xr.open_dataset('data/curonian_h3_multires_landscape.nc', engine='h5netcdf')
arr = ds['dist_from_sea'].values
wm = ds['water_mask'].values
print('shape:', arr.shape)
print('dtype:', arr.dtype)
print('water-cell finite:', bool(np.all(np.isfinite(arr[wm]))))
print('max:', float(arr.max()))
print('OpenBaltic-source-cells-at-zero:',
      bool(np.any(arr[(ds['reach_id'].values == ds.attrs['reach_names'].split(',').index('OpenBaltic')) & wm] == 0)))
ds.close()
"
```
Expected output (all four lines):
```
shape: (185428,)
dtype: float32
water-cell finite: True
max: <some positive value, expected > 10000>
OpenBaltic-source-cells-at-zero: True
```

If any check fails, the NC is corrupt — re-run Step 3 after diagnosing.

(Also implicitly verifies the per-reach print above didn't silently skip reaches via the `if len(valid) == 0: continue` path. If a reach was all-NaN, `water-cell finite` would be False here.)

- [ ] **Step 4: Append failing test 6**

Append to `tests/test_movement_gradient.py`:

```python
import xarray as xr


PRODUCTION_NC = Path("data/curonian_h3_multires_landscape.nc")


@pytest.mark.skipif(
    not PRODUCTION_NC.exists(),
    reason="production NC missing; run `python scripts/build_h3_multires_landscape.py`",
)
def test_production_mesh_gradient_sanity():
    """C4 Test 6: end-to-end production-mesh gradient sanity."""
    ds = xr.open_dataset(str(PRODUCTION_NC), engine="h5netcdf")
    assert "dist_from_sea" in ds.variables, (
        "production NC missing dist_from_sea; rebuild via "
        "scripts/build_h3_multires_landscape.py"
    )
    dist = ds["dist_from_sea"].values
    water_mask = ds["water_mask"].values
    reach_id = ds["reach_id"].values
    reach_names = ds.attrs.get("reach_names", "").split(",")
    ds.close()

    # Precondition: not all-zero (NC was rebuilt with working compute).
    assert dist.max() > 0, (
        "dist_from_sea is all-zeros; rebuild via build script"
    )

    # No NaN/Inf on water cells.
    assert np.all(np.isfinite(dist[water_mask])), (
        "dist_from_sea has NaN/Inf on water cells — NC corrupt"
    )

    # (a) mean(OpenBaltic) < mean(Nemunas).
    ob_id = reach_names.index("OpenBaltic")
    ne_id = reach_names.index("Nemunas")
    ob_mean = float(dist[(reach_id == ob_id) & water_mask].mean())
    ne_mean = float(dist[(reach_id == ne_id) & water_mask].mean())
    assert ob_mean < ne_mean, (
        f"gradient inverted: mean(OpenBaltic)={ob_mean:.1f}m >= "
        f"mean(Nemunas)={ne_mean:.1f}m"
    )

    # (b) min(Nemunas) > mean(OpenBaltic).
    ne_min = float(dist[(reach_id == ne_id) & water_mask].min())
    assert ne_min > ob_mean, (
        f"gross inversion: min(Nemunas)={ne_min:.1f}m <= "
        f"mean(OpenBaltic)={ob_mean:.1f}m"
    )

    # (c) every delta-branch cell has dist_from_sea > mean(BalticCoast).
    bc_id = reach_names.index("BalticCoast")
    bc_mean = float(dist[(reach_id == bc_id) & water_mask].mean())
    for branch in ("Atmata", "Skirvyte", "Gilija"):
        rid = reach_names.index(branch)
        branch_min = float(dist[(reach_id == rid) & water_mask].min())
        assert branch_min > bc_mean, (
            f"{branch}: min={branch_min:.1f}m <= mean(BalticCoast)="
            f"{bc_mean:.1f}m — delta cells should be inland of coast"
        )

    # (d) per-delta-branch inversion check.
    for branch in ("Atmata", "Skirvyte", "Gilija"):
        rid = reach_names.index(branch)
        branch_min = float(dist[(reach_id == rid) & water_mask].min())
        assert branch_min > ob_mean, (
            f"{branch} inversion: min={branch_min:.1f}m <= "
            f"mean(OpenBaltic)={ob_mean:.1f}m"
        )
```

- [ ] **Step 5: Run focused test to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_gradient.py::test_production_mesh_gradient_sanity -v`
Expected: PASS.

- [ ] **Step 6: Commit (build script + test only — NC rebuild is a separate commit per project convention)**

```bash
git add scripts/build_h3_multires_landscape.py tests/test_movement_gradient.py
git commit -m "feat(c4): build script wires compute_dist_from_sea + Test 6 production-mesh sanity"
```

- [ ] **Step 7: Commit the rebuilt NC separately**

```bash
git add data/curonian_h3_multires_landscape.nc
git commit -m "data(c4): rebuild Curonian H3 multires NC with dist_from_sea"
```

---

## Task 9: Test 7 (post-C3.3-teleport topology invariant)

**Files:**
- Test: `tests/test_movement_gradient.py`

- [ ] **Step 1: Append Test 7**

Append to `tests/test_movement_gradient.py`:

```python
@pytest.mark.skipif(
    not PRODUCTION_NC.exists(),
    reason="production NC missing",
)
def test_post_c33_teleport_invariant():
    """C4 Test 7: for each delta-branch reach, the C3.3 teleport
    target cell has at least one strictly-higher dist_from_sea
    water neighbor — guarantees a teleported strayer can progress
    inland on the next UPSTREAM step."""
    from salmon_ibm.delta_routing import _branch_entry_cell

    ds = xr.open_dataset(str(PRODUCTION_NC), engine="h5netcdf")
    dist = ds["dist_from_sea"].values
    water_mask = ds["water_mask"].values
    reach_id = ds["reach_id"].values
    reach_names = ds.attrs.get("reach_names", "").split(",")
    nbr_starts = ds["nbr_starts"].values
    nbr_idx = ds["nbr_idx"].values
    ds.close()

    # Build a minimal mesh that satisfies _branch_entry_cell's interface.
    # Per pass-1 review: must build BOTH _water_nbrs AND _water_nbr_count
    # (the C3.3 swimmable-cell preference reads _water_nbr_count, but
    # other helpers in the dispatch chain may read _water_nbrs).
    class _MeshShim:
        pass
    mesh = _MeshShim()
    mesh.reach_id = reach_id
    mesh.reach_names = reach_names
    mesh.water_mask = water_mask

    max_deg = int(np.diff(nbr_starts).max())
    mesh._water_nbrs = np.full(
        (len(reach_id), max_deg), -1, dtype=np.int32,
    )
    mesh._water_nbr_count = np.zeros(len(reach_id), dtype=np.int32)
    for i in range(len(reach_id)):
        s, e = int(nbr_starts[i]), int(nbr_starts[i + 1])
        slot = 0
        for k in range(s, e):
            n_idx = int(nbr_idx[k])
            if n_idx >= 0 and water_mask[n_idx]:
                mesh._water_nbrs[i, slot] = n_idx
                slot += 1
        mesh._water_nbr_count[i] = slot

    for branch in ("Atmata", "Skirvyte", "Gilija"):
        if branch not in reach_names:
            continue
        rid = reach_names.index(branch)
        entry = _branch_entry_cell(mesh, rid)
        assert entry >= 0, f"{branch}: no entry cell"

        # First, NaN guard on the entry cell.
        entry_dist = dist[entry]
        assert np.isfinite(entry_dist), (
            f"{branch}: entry cell {entry} has non-finite "
            f"dist_from_sea ({entry_dist}); NC build is corrupt"
        )

        # Then, find a strictly-higher water neighbor.
        s, e = int(nbr_starts[entry]), int(nbr_starts[entry + 1])
        higher_neighbors = []
        for k in range(s, e):
            n = int(nbr_idx[k])
            if n < 0 or not water_mask[n]:
                continue
            if dist[n] > entry_dist:
                higher_neighbors.append(n)
        assert higher_neighbors, (
            f"{branch}: entry cell {entry} (dist={entry_dist:.1f}m) has "
            "no strictly-higher water neighbor — a teleported strayer "
            "would oscillate at the branch mouth instead of progressing "
            "inland. Topology defect at the production mesh's branch "
            "entry."
        )
```

- [ ] **Step 2: Run focused test to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_gradient.py::test_post_c33_teleport_invariant -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_movement_gradient.py
git commit -m "test(c4): Test 7 post-C3.3-teleport topology invariant"
```

---

## Task 10: Test 7b end-to-end behavioral + Test 5b production-mesh determinism

**Files:**
- Test: `tests/test_movement_gradient.py`

- [ ] **Step 1: Append Test 7b (teleport-then-step end-to-end)**

Append to `tests/test_movement_gradient.py`:

```python
@pytest.mark.skipif(
    not PRODUCTION_NC.exists(),
    reason="production NC missing",
)
def test_teleport_then_upstream_advances_inland():
    """C4 Test 7b: composes _event_update_exit_branch (C3.3 stray
    teleport) + MovementEvent (UPSTREAM step). Asserts post-teleport
    UPSTREAM advances inland, not back to the lagoon."""
    import xarray as xr
    from salmon_ibm.delta_routing import (
        _branch_entry_cell,
        update_exit_branch_id,
    )
    from salmon_ibm.movement import execute_movement
    from salmon_ibm.agents import AgentPool, Behavior
    from salmon_ibm.origin import ORIGIN_HATCHERY
    from salmon_ibm.baltic_params import (
        load_baltic_species_config,
        HatcheryDispatch,
    )

    ds = xr.open_dataset(str(PRODUCTION_NC), engine="h5netcdf")
    dist = ds["dist_from_sea"].values
    water_mask = ds["water_mask"].values
    reach_id = ds["reach_id"].values
    reach_names = ds.attrs.get("reach_names", "").split(",")
    nbr_starts = ds["nbr_starts"].values
    nbr_idx = ds["nbr_idx"].values
    ds.close()

    # Reload cfg per-call to avoid state leak across iterations and
    # across tests that share the same cached YAML. Pass-1 silent-
    # failure finding: mutating cfg.hatchery.homing_precision in a
    # test would leak to subsequent tests if cfg is module-cached.
    cfg = load_baltic_species_config("configs/baltic_salmon_species.yaml")
    saved_homing = cfg.hatchery.homing_precision
    hd = HatcheryDispatch(
        params=cfg.hatchery,
        activity_lut=np.ones(5, dtype=np.float64),
    )

    class _MeshShim:
        pass
    mesh = _MeshShim()
    mesh.reach_id = reach_id
    mesh.reach_names = reach_names
    mesh.water_mask = water_mask
    mesh._water_nbrs = np.full(
        (len(reach_id), int(np.diff(nbr_starts).max())), -1, dtype=np.int32,
    )
    mesh._water_nbr_count = np.zeros(len(reach_id), dtype=np.int32)
    for i in range(len(reach_id)):
        s, e = int(nbr_starts[i]), int(nbr_starts[i + 1])
        slot = 0
        for k in range(s, e):
            n = int(nbr_idx[k])
            if n >= 0 and water_mask[n]:
                mesh._water_nbrs[i, slot] = n
                slot += 1
        mesh._water_nbr_count[i] = slot
    mesh.reach_id = reach_id

    # For each branch: place a hatchery agent on a different branch,
    # configure C3.3 to home them (homing_precision=1.0 forces the
    # natal branch). Then step UPSTREAM and assert progress.
    for natal_branch in ("Atmata", "Skirvyte", "Gilija"):
        natal_rid = reach_names.index(natal_branch)
        # Place agent on a different branch's lagoon side. Pick the
        # first branch that's not natal.
        other_branches = [b for b in ("Atmata", "Skirvyte", "Gilija")
                          if b != natal_branch]
        other_rid = reach_names.index(other_branches[0])
        other_cells = np.where((reach_id == other_rid) & water_mask)[0]
        start_cell = int(other_cells[0])

        pool = AgentPool(n=1, start_tri=start_cell, rng_seed=12345)
        pool.behavior[0] = int(Behavior.UPSTREAM)
        pool.natal_reach_id[0] = natal_rid
        pool.origin[0] = ORIGIN_HATCHERY
        pool.exit_branch_id[0] = -1

        # Force homing to natal: monkeypatch homing_precision to 1.0.
        cfg.hatchery.homing_precision = 1.0

        landscape = {
            "rng": np.random.default_rng(12345),
            "species_config": cfg,
            "hatchery_dispatch": hd,
            "fields": {"dist_from_sea": dist},
            "n_micro_steps_per_cell": np.full(len(reach_id), 5, dtype=np.int32),
        }

        # Phase 1: trigger C3.3 teleport.
        update_exit_branch_id(pool, mesh, landscape=landscape)
        post_teleport_cell = int(pool.tri_idx[0])
        post_teleport_dist = float(dist[post_teleport_cell])
        assert reach_id[post_teleport_cell] == natal_rid, (
            f"{natal_branch}: agent did not teleport to natal branch; "
            f"landed on rid={reach_id[post_teleport_cell]}"
        )

        # Phase 2: one UPSTREAM step. Call execute_movement with the
        # CORRECT signature: (pool, mesh, fields, seed=..., n_micro_...).
        # The dormancy check fires at MovementEvent.execute layer in
        # production; this test bypasses that layer (we're testing
        # the kernel field-swap + post-teleport progress, not the
        # dormancy guard).
        execute_movement(
            pool, mesh, landscape["fields"],
            seed=12345,
            n_micro_steps_per_cell=landscape["n_micro_steps_per_cell"],
        )
        final_cell = int(pool.tri_idx[0])
        final_dist = float(dist[final_cell])

        assert final_dist > post_teleport_dist, (
            f"{natal_branch}: UPSTREAM step did not advance inland. "
            f"post-teleport={post_teleport_dist:.1f}m, "
            f"final={final_dist:.1f}m"
        )

    # Restore cfg (defensive — even though we reload per-test, restore
    # in case the module-level cache wraps this object).
    cfg.hatchery.homing_precision = saved_homing
```

- [ ] **Step 2: Append Test 5b production-mesh determinism**

Append to `tests/test_movement_gradient.py`:

```python
@pytest.mark.skipif(
    not PRODUCTION_NC.exists(),
    reason="production NC missing",
)
def test_compute_dist_from_sea_matches_saved_nc():
    """C4 Test 5b (production part): the saved NC's dist_from_sea
    must equal a fresh recompute via compute_dist_from_sea(mesh).
    Pins down 'committed NC == current build script'."""
    from build_h3_multires_landscape import compute_dist_from_sea
    import xarray as xr

    ds = xr.open_dataset(str(PRODUCTION_NC), engine="h5netcdf")
    # Cast to float32 to match compute_dist_from_sea's return dtype.
    # xarray may upcast NC float32 to float64 on read depending on
    # engine — explicit cast normalizes for byte-equal comparison.
    saved = ds["dist_from_sea"].values.astype(np.float32)

    # Build mesh from NC (same logic as simulation.py:130-180).
    names_attr = ds.attrs.get("reach_names", "")
    reach_names = names_attr.split(",")
    from salmon_ibm.h3_multires import H3MultiResMesh
    mesh = H3MultiResMesh(
        h3_ids=ds["h3_id"].values.astype(np.uint64),
        resolutions=ds["resolution"].values.astype(np.int8),
        centroids=np.column_stack([ds["lat"].values, ds["lon"].values]),
        nbr_starts=ds["nbr_starts"].values.astype(np.int32),
        nbr_idx=ds["nbr_idx"].values.astype(np.int32),
        water_mask=ds["water_mask"].values.astype(bool),
        depth=ds["depth"].values.astype(np.float32),
        areas=np.zeros(len(ds["h3_id"]), dtype=np.float32),
        reach_id=ds["reach_id"].values.astype(np.int8),
        reach_names=reach_names,
    )
    ds.close()

    recomputed = compute_dist_from_sea(mesh)
    assert np.array_equal(saved, recomputed, equal_nan=True), (
        "saved NC's dist_from_sea differs from fresh recompute. "
        "Either the build script changed without rebuilding the NC, "
        "or compute_dist_from_sea is non-deterministic."
    )
```

- [ ] **Step 3: Run focused tests to verify pass**

Run: `micromamba run -n shiny python -m pytest tests/test_movement_gradient.py -k "teleport or matches_saved" -v`
Expected: 2 PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_movement_gradient.py
git commit -m "test(c4): Test 7b teleport-then-step + Test 5b production-NC determinism"
```

---

## Task 11: Final regression sweep + EXECUTED stamp

**Files:**
- Modify: `docs/superpowers/specs/2026-05-07-c4-movement-gradient-design.md` (status header)
- Modify: `docs/superpowers/plans/2026-05-07-c4-movement-gradient.md` (EXECUTED note at top)

- [ ] **Step 1a: Pre-flight — verify production NC exists AND production-dependent tests are collected**

Skipped tests count as "passing" in pytest's summary, so a missing
production NC could silently zero out C4's most important
acceptance tests. Verify before the regression sweep:

Run:
```bash
test -f data/curonian_h3_multires_landscape.nc || \
    { echo "FAIL: production NC missing"; exit 1; }
```

Then **actually run** the 4 production-dependent tests (not
`--collect-only` — `@pytest.mark.skipif` with a runtime condition
like `not PRODUCTION_NC.exists()` is evaluated at *run* time, not
collection time, so `--collect-only` cannot detect skips):

```bash
micromamba run -n shiny python -m pytest \
    tests/test_movement_gradient.py \
    -k "production_mesh or post_c33_teleport or teleport_then or matches_saved" \
    -v 2>&1 | tail -20
```
Expected output: each of the 4 tests reports `PASSED`; none report
`SKIPPED`. If any are SKIPPED, the production NC is missing or
incomplete despite Step 1a's `test -f` check (e.g., a zero-byte
file from a crashed build) — STOP and re-run Task 8 Step 3.

- [ ] **Step 1b: Run full pytest suite**

Run: `micromamba run -n shiny python -m pytest tests/ --tb=short`
Expected: 928 (pre-implementation baseline at v1.7.7 commit `b18ecaf`)
+ 19 (C4 new tests) = **947 collected; 945-946 passing** (depending
on how skip-counts vs xfail interact with the report). Spec mandates
17 numbered tests but the plan adds 2 sanity tests (Task 6's
`test_check_dormant_gradient_no_directed_agents_no_raise` and
`test_check_dormant_gradient_no_env_in_landscape_no_raise`) → 19
test functions.

If `tests/test_movement_metric.py::test_full_step_time_within_one_percent_of_baseline` flakes, retry it in isolation:
Run: `micromamba run -n shiny python -m pytest tests/test_movement_metric.py::test_full_step_time_within_one_percent_of_baseline -v`
Expected: PASS in isolation (Windows-load flakiness, not C4-introduced).

- [ ] **Step 2: Verify production NC is loadable + dist_from_sea is present**

```bash
micromamba run -n shiny python -c "import xarray as xr; ds=xr.open_dataset('data/curonian_h3_multires_landscape.nc'); print('dist_from_sea' in ds.variables, ds['dist_from_sea'].shape, ds['dist_from_sea'].values.max()); ds.close()"
```
Expected: `True (185428,) <some positive value, expected ~50000>`.

- [ ] **Step 3: Stamp the spec as EXECUTED**

In `docs/superpowers/specs/2026-05-07-c4-movement-gradient-design.md`, change the status header:

```
**Status:** ✅ EXECUTED on 2026-05-07 via subagent-driven-development; full pytest <baseline+17> passing. Branch `c4-movement-gradient` ready for PR + v1.7.8 tag.
```

- [ ] **Step 4: Add EXECUTED stamp to plan**

In `docs/superpowers/plans/2026-05-07-c4-movement-gradient.md` (this file), at the top after the goal/architecture/tech-stack header, add:

```markdown
**Status:** ✅ EXECUTED on 2026-05-07. Final test count: <baseline+17> passing.
```

- [ ] **Step 5: Commit the stamps**

```bash
git add docs/superpowers/specs/2026-05-07-c4-movement-gradient-design.md docs/superpowers/plans/2026-05-07-c4-movement-gradient.md
git commit -m "docs(c4): stamp spec + plan as EXECUTED"
```

- [ ] **Step 6: (Optional) Tag v1.7.8 + push + deploy**

Per the project's tag-before-deploy convention (`feedback_collaboration.md`), C4 warrants a v1.7.8 tag once on `main`:

```bash
git tag -a v1.7.8 -m "v1.7.8 — C4 movement gradient (substrate fix)

C4 ships per docs/superpowers/specs/2026-05-07-c4-movement-gradient-design.md.
Replaces flat-zero ssh field with dist_from_sea (multi-source Dijkstra
at landscape build) so directed-movement actually produces displacement.

Activates the four-tier hatchery-vs-wild architecture (C1-C3.3) which
was structurally complete at v1.7.7 but biologically dormant: returning
adults could not reach delta branches due to the flat-gradient
substrate. C4 is the substrate fix.

Builds on:
  C1 (PR #3, v1.7.4) — origin tag
  C2 (PR #4, v1.7.4) — activity multiplier divergence
  C3.1 (PR #5, v1.7.5) — pre-spawn skip probability
  C3.2 (PR #6, v1.7.6) — sea-age trinomial sampling
  C3.3 (v1.7.7) — homing precision divergence

12-pass review-loop; +17 tests; +1 NC rebuild.
"
```

DO NOT push the tag without user approval (per the confirm-before-push convention).

---

## Self-review notes

**Spec coverage check:**
- compute_dist_from_sea pure function → Task 1 ✓
- Determinism (sort + heap tie-break) → Task 1 (in code) + Tests 5b/5c (Task 2) ✓
- Build script wiring → Task 8 ✓
- NC rebuild → Task 8 ✓
- h3_env Case A → Task 3 ✓
- h3_env Case B 4 raise paths → Task 4 ✓
- err-id constants → Task 3 ✓
- Landscape TypedDict + dict construction → Task 5 ✓
- _check_dormant_gradient helper → Task 6 ✓
- UPSTREAM/DOWNSTREAM field swap + ascending flip → Task 7 ✓
- Tests 1, 2, 2b, 3 → Task 7 ✓
- Tests 4, 4a-4f → Tasks 3, 4, 6 ✓
- Tests 5, 5b, 5c → Tasks 1, 2, 10 ✓
- Test 6 production sanity → Task 8 ✓
- Test 7 topology invariant → Task 9 ✓
- Test 7b end-to-end → Task 10 ✓
- Live-test (Playwright) → mentioned in Task 11 step 6 (deploy gate); Playwright invocation is operator-driven, not in the test suite (matches existing project convention).

**Placeholder scan:** No "TBD"/"TODO"/"add appropriate" patterns. Each step shows actual code or actual command.

**Type consistency:** `compute_dist_from_sea(mesh) -> np.ndarray` consistent across Task 1 definition + Tasks 2, 8, 10 callers. `ERR_DIST_FROM_SEA_MISSING` consistent in Tasks 3, 4, 6.
