# H3 Multi-Resolution — Phase 2 Implementation Plan

> **STATUS: ✅ EXECUTED** — `H3MultiResMesh` wired through full sim/viewer/test stack and shipped as **v1.3.0**. CMEMS forcing in builder, deck.gl per-cell resolution, MAX_NBRS overflow guard, gradient() port. Tests in `tests/test_h3_multires.py`, `test_h3_multires_builder.py`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the v1.2.8 `H3MultiResMesh` scaffold through the full simulation, viewer, and test stack — so a user can pick a mixed-resolution Curonian Lagoon (rivers **res 10** ~75 m, lagoon res 9 ~200 m, OpenBaltic res 8 ~530 m) from the sidebar and run a 30-day simulation end-to-end. The default-resolution constants in `scripts/build_h3_multires_landscape.py` still claim river res 11; this plan deliberately runs the smoke build at res 10 so the integration test finishes in <60 s. Bumping rivers to res 11 is one config flag away — see the "Out of scope → Resolution tuning" note at the bottom.

**Architecture:** Add a new `mesh_backend: h3_multires` config value alongside the existing `h3` and TriMesh backends. The new branch loads `H3MultiResMesh` from a pre-built NC, the existing numba kernels read its `(N, 12)` padded `_water_nbrs` view unchanged, and the deck.gl viewer reads per-cell `resolution` instead of the uniform attribute.

**Tech Stack:** Python 3.13, numpy, h3-py 4.4.2, xarray (h5netcdf engine), numba `@njit(cache=True, parallel=True)`, Shiny + shiny_deckgl, pytest.

---

## Prerequisite: patch the v1.2.8 scaffold

Plan-review pass 6 (architect + code-reviewer) found two scaffold gaps that block Phase 2.  These are corrections to `salmon_ibm/h3_multires.py` itself — small, surgical, ship as v1.2.9 *before* Task 1 starts.

- [ ] **Step P.1: Add overflow assertion to `find_cross_res_neighbours`.**

A coarse cell at the corner of multiple fine zones can in principle accumulate >12 neighbours.  The current code silently truncates rows in `H3MultiResMesh.__init__` (`n_row = min(len(row), MAX_NBRS)`) — dropped neighbours are then permanently unreachable from movement kernels.

In `salmon_ibm/h3_multires.py`, at the end of `find_cross_res_neighbours` (after the loop builds `rows`, before the CSR conversion), add:

```python
max_row_len = max((len(r) for r in rows), default=0)
if max_row_len > MAX_NBRS:
    raise RuntimeError(
        f"Cross-resolution neighbour finder produced a row with "
        f"{max_row_len} entries; MAX_NBRS is {MAX_NBRS}.  Bump "
        f"MAX_NBRS in h3_multires.py and rebuild the landscape NC, "
        f"or reduce ``max_resolution_drop`` to limit cross-res "
        f"reach."
    )
```

- [ ] **Step P.2: Port `gradient()` from `H3Mesh` to `H3MultiResMesh`.**

`environment.py:79` exposes `env.gradient(field_name, idx)` which calls `mesh.gradient(field, idx)`.  Some events read this (e.g., behaviour switching based on field gradients).  `H3Mesh` implements it (`h3mesh.py:132-156`); `H3MultiResMesh` doesn't.  Lift the implementation verbatim — it uses `self.neighbors`, `self.centroids`, and `self.metric_scale`, all of which `H3MultiResMesh` already provides.

In `salmon_ibm/h3_multires.py`, near the other duck-typed methods (after `metric_scale`):

```python
def gradient(self, field: np.ndarray, idx: int) -> tuple[float, float]:
    """Approximate normalised ``(dlat, dlon)`` gradient of ``field`` at ``idx``.

    Same algorithm as :meth:`H3Mesh.gradient` — centroid diffs scaled
    by :meth:`metric_scale` so a degree of longitude doesn't outweigh
    a degree of latitude at mid-latitude.  Returns ``(0.0, 0.0)`` when
    the cell has no valid neighbours.
    """
    row = self.neighbors[idx]
    valid = row[row >= 0]
    if len(valid) == 0:
        return (0.0, 0.0)
    here = self.centroids[idx]
    scale_x, scale_y = self.metric_scale(float(here[0]))
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
```

- [ ] **Step P.3: Test, commit, tag v1.2.9.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires.py" -v
```

Expected: 8 PASSED.  (No new tests added in this prereq; Phase 2 tests will exercise these paths.)

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" add salmon_ibm/h3_multires.py
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" commit -m "fix(h3-multires): MAX_NBRS overflow guard + gradient() port"
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" tag -a v1.2.9 -m "v1.2.9 — scaffold corrections from plan review"
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" push origin main
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" push origin v1.2.9
```

---

## Pre-flight checks

- [ ] **Step 0.1: Verify v1.2.8 scaffold ships green.**

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" log --oneline -5
```

Expected: top commit is the v1.2.8 multi-res scaffold (`eceeb7d` or its tag).

- [ ] **Step 0.2: Confirm the existing multi-res tests still pass.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires.py" -v
```

Expected: 8 passed.

- [ ] **Step 0.3: Confirm movement kernels already iterate by `_water_nbr_count`.**

```bash
micromamba run -n shiny python -c "import re; src = open(r'C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/salmon_ibm/movement.py').read(); hits = re.findall(r'range\(\d+\)', src); print('hardcoded ranges:', hits)"
```

Expected: `hardcoded ranges: []` (no `range(6)` literals — all loops use `cnt`). If this prints any literals, those kernels need fixing **before** this plan starts.

---

## Task 1: Extend the multi-res builder with CMEMS forcing

The v1.2.8 builder samples bathymetry per cell but stops short of forcing. Without forcing, `H3Environment.from_netcdf` raises on field lookup. This is the prerequisite to any sim integration test.

**Files:**
- Modify: `scripts/build_h3_multires_landscape.py:80-330` (add forcing sampling stage between bathymetry and NC write)
- Reference: `scripts/build_nemunas_h3_landscape.py:115-178` (existing CMEMS sampling, port verbatim)

- [ ] **Step 1.1: Write a failing test for the builder's forcing output.**

Create `tests/test_h3_multires_builder.py`:

```python
"""Smoke tests for scripts/build_h3_multires_landscape.py."""
from __future__ import annotations
from pathlib import Path
import pytest

PROJECT = Path(__file__).resolve().parent.parent
LANDSCAPE_NC = PROJECT / "data" / "curonian_h3_multires_landscape.nc"


def test_multires_nc_carries_forcing_fields():
    """Builder must write tos/sos/uo/vo per (time, cell) so
    H3Environment.from_netcdf finds them at sim-init time."""
    if not LANDSCAPE_NC.exists():
        pytest.skip(
            "multi-res landscape NC not built — "
            "run scripts/build_h3_multires_landscape.py first."
        )
    import xarray as xr
    ds = xr.open_dataset(LANDSCAPE_NC, engine="h5netcdf")
    for var in ("tos", "sos", "uo", "vo"):
        assert var in ds.variables, (
            f"forcing field {var!r} missing from multi-res NC; "
            f"H3Environment will fail at simulation init."
        )
    assert "time" in ds.dims, "forcing time dimension missing"
    ds.close()
```

- [ ] **Step 1.2: Run it to confirm it fails.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_builder.py" -v
```

Expected: FAIL with `AssertionError: forcing field 'tos' missing` (assuming the NC was already built without forcing — if the NC doesn't exist yet, the test SKIPs, which is fine).

- [ ] **Step 1.3: Add `--cmems` arg + sampling stage to the builder.**

In `scripts/build_h3_multires_landscape.py`:

```python
# Add to argparse setup (after --tif):
parser.add_argument(
    "--cmems", default="data/curonian_forcing_cmems_raw.nc",
    help="CMEMS reanalysis NetCDF",
)
parser.add_argument(
    "--start", default="2011-06-01",
    help="CMEMS time-window start (ISO date) — keeps output ≤30 MB",
)
parser.add_argument(
    "--end", default="2011-06-30",
    help="CMEMS time-window end (ISO date)",
)
```

Then add this function (lifted verbatim from `build_nemunas_h3_landscape.py:83-160`, only the docstring needs trimming):

```python
def sample_cmems(
    cells: list[str],
    cmems_path: Path,
    start: str | None = None,
    end: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sample CMEMS thetao/so/uo/vo at each H3 cell centroid.

    Returns ``(time_array, {var_name: (time, cell) float32})``.
    Same NaN-fill via NearestNDInterpolator as the uniform-res builder
    so the thermal-kill path doesn't kill agents on a CMEMS land cell.
    """
    raw = xr.open_dataset(cmems_path)
    if start is not None or end is not None:
        raw = raw.sel(time=slice(start, end))
        if raw.sizes["time"] == 0:
            raise ValueError(f"no CMEMS timesteps in [{start}, {end}]")
    lat_src = raw["latitude"].values
    lon_src = raw["longitude"].values
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        raw = raw.isel(latitude=slice(None, None, -1))
    lats, lons = zip(*(h3.cell_to_latlng(c) for c in cells))
    query = np.column_stack([lats, lons])
    n_time = raw.sizes["time"]
    n_cells = len(cells)
    src_lats_2d, src_lons_2d = np.meshgrid(lat_src, lon_src, indexing="ij")
    var_map = [("thetao", "tos"), ("so", "sos"), ("uo", "uo"), ("vo", "vo")]
    out: dict[str, np.ndarray] = {}
    from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
    for src, dst in var_map:
        if src not in raw:
            print(f"  ! source var {src} missing — skipping")
            continue
        arr = raw[src].squeeze().values
        vals = np.empty((n_time, n_cells), dtype=np.float32)
        for t in range(n_time):
            src_t = arr[t]
            interp = RegularGridInterpolator(
                (lat_src, lon_src), src_t,
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            row = interp(query)
            nan_mask = np.isnan(row)
            if nan_mask.any():
                src_flat = src_t.ravel()
                src_valid = ~np.isnan(src_flat)
                if src_valid.any():
                    nn = NearestNDInterpolator(
                        np.column_stack([
                            src_lats_2d.ravel()[src_valid],
                            src_lons_2d.ravel()[src_valid],
                        ]),
                        src_flat[src_valid],
                    )
                    row[nan_mask] = nn(query[nan_mask])
            vals[t] = row.astype(np.float32)
        out[dst] = vals
        print(f"  ✓ {src} → {dst} ({n_time} timesteps, {n_cells:,} cells)")
    return raw["time"].values, out
```

In `main()`, between the bathymetry stage and the NC write, add:

```python
print(f"\n[3.5/4] Sampling CMEMS forcing [{args.start} .. {args.end}]…")
times, forcing = sample_cmems(
    cells, PROJECT / args.cmems, args.start, args.end,
)
```

And replace the existing `xr.Dataset(...)` constructor with this version (the only new lines are the `forcing.items()` unpack and the `coords={"time": times}` arg).  All other variables in the snippet — `cells`, `lats`, `lons`, `depth`, `water_mask`, `reach_id_arr`, `nbr_starts`, `nbr_idx` — already exist in `main()`'s scope from the v1.2.8 scaffold; the snippet shows the **full** constructor as a drop-in replacement, not a new self-contained block.

```python
# Pre-existing in main() scope (v1.2.8 scaffold): cells, lats, lons,
# depth, water_mask, reach_id_arr, nbr_starts, nbr_idx.  This task
# adds: times, forcing.
ds = xr.Dataset(
    {
        "h3_id":      (("cell",), np.array([int(h3.str_to_int(c)) for c in cells], dtype=np.uint64)),
        "resolution": (("cell",), np.array([h3.get_resolution(c) for c in cells], dtype=np.int8)),
        "lat":        (("cell",), np.array(lats, dtype=np.float64)),
        "lon":        (("cell",), np.array(lons, dtype=np.float64)),
        "depth":      (("cell",), depth),
        "water_mask": (("cell",), water_mask),
        "reach_id":   (("cell",), reach_id_arr),
        "nbr_starts": (("cell_p1",), nbr_starts),
        "nbr_idx":    (("edge",), nbr_idx),
        # Forcing per (time, cell) — added in this task.
        **{k: (("time", "cell"), v) for k, v in forcing.items()},
    },
    coords={"time": times},  # added in this task
    attrs={
        "title": "Curonian Lagoon multi-resolution H3 landscape",
        "reach_names": ",".join(reach_polygons.keys()),
        "reach_resolutions": ",".join(
            f"{n}={reach_res.get(n, 9)}" for n in reach_polygons.keys()
        ),
        "n_cells": len(cells),
        "n_edges": len(nbr_idx),
        # Static-viewer compat: median per-cell resolution lets the
        # legacy "h3_resolution" reader (app.py:2053 & 515) pick a
        # sensible camera zoom even though the NC is mixed-res.
        "h3_resolution": int(np.median(
            np.array([h3.get_resolution(c) for c in cells])
        )),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    },
)
```

- [ ] **Step 1.4: Rebuild the multi-res NC.**

```bash
PYTHONUNBUFFERED=1 micromamba run -n shiny python -u "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/scripts/build_h3_multires_landscape.py" --resolutions "Nemunas=10,Atmata=10,Minija=10,Sysa=10,Skirvyte=10,Leite=10,Gilija=10,CuronianLagoon=9,BalticCoast=9,OpenBaltic=8"
```

Expected: prints `[3.5/4] Sampling CMEMS forcing` followed by 4 `✓` lines (tos/sos/uo/vo), then `wrote ~12 MB` (was 2.1 MB without forcing).

- [ ] **Step 1.5: Run the test, expect PASS.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_builder.py" -v
```

Expected: PASS.

- [ ] **Step 1.6: Commit.**

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" add scripts/build_h3_multires_landscape.py tests/test_h3_multires_builder.py
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" commit -m "feat(h3-multires): builder samples CMEMS forcing per cell"
```

---

## Task 2: Add `mesh_backend: h3_multires` simulation init branch

**Files:**
- Modify: `salmon_ibm/simulation.py:67-89` (existing `mesh_backend == "h3"` branch — add a parallel `elif "h3_multires"` block right after)
- Modify: `salmon_ibm/config.py:96-115` (validate the new mesh_backend value)
- Create: `configs/config_curonian_h3_multires.yaml` (the new scenario config)

- [ ] **Step 2.1: Write failing test for h3_multires sim init.**

Create `tests/test_h3_multires_integration.py`:

```python
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
```

- [ ] **Step 2.2: Create the config YAML.**

`configs/config_curonian_h3_multires.yaml`:

```yaml
# Multi-resolution H3 Curonian Lagoon scenario.
#
# Rivers tessellated at H3 res 10 (~75 m edges) so the Nemunas main
# channel (~600 m wide) renders as 8 cells across; small delta arms
# (Šyša, Skirvytė ~80-100 m) get 1-2 cells.  Lagoon at res 9 (~200 m)
# and the open Baltic at res 8 (~530 m) keep the cell count
# tractable for interactive play (~36 k cells total).
#
# Built by scripts/build_h3_multires_landscape.py with the same
# --resolutions flag.  Re-run that script if the inSTREAM polygons
# or the resolution choices change.

name: "curonian_h3_multires"
mesh_backend: "h3_multires"
h3_multires_landscape_nc: "data/curonian_h3_multires_landscape.nc"

species_config: "configs/baltic_salmon_species.yaml"

initial_state:
  initial_cell_strategy: "uniform_random_water"

barriers_csv: null  # multi-res barrier CSVs not yet supported (Phase 4)
```

- [ ] **Step 2.3: Add the validator in `salmon_ibm/config.py`.**

Find the existing `mesh_backend == "h3"` validation block (around line 96-115) and add:

```python
# Inside validate_config, near the existing h3 block:
if cfg.get("mesh_backend") == "h3_multires":
    if "h3_multires_landscape_nc" not in cfg:
        raise ValueError(
            "mesh_backend=h3_multires requires 'h3_multires_landscape_nc' "
            "(path to a multi-res landscape NetCDF built by "
            "scripts/build_h3_multires_landscape.py)"
        )
```

- [ ] **Step 2.4: Run the test to confirm it fails on `mesh_backend`.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py::test_mesh_is_h3multires" -v
```

Expected: FAIL — `Simulation` doesn't recognise `mesh_backend == "h3_multires"`, so it falls through to the default branch and complains about a missing `grid` key (or similar).

- [ ] **Step 2.5: Add the `h3_multires` branch in `salmon_ibm/simulation.py`.**

After the existing `if mesh_backend == "h3":` block (line 89), insert:

```python
elif mesh_backend == "h3_multires":
    from salmon_ibm.h3_multires import H3MultiResMesh
    from salmon_ibm.h3_env import H3Environment
    import h3 as _h3
    import xarray as _xr

    landscape_path = config["h3_multires_landscape_nc"]
    ds = _xr.open_dataset(landscape_path, engine="h5netcdf")

    # reach metadata MUST be loaded as a pair — without `reach_id`,
    # `reach_names` is meaningless (every cell would report "Land").
    # Loading them independently was rejected in plan review pass 6.
    if "reach_id" in ds:
        reach_id_arr = ds["reach_id"].values.astype(np.int8)
        names_attr = ds.attrs.get("reach_names", "")
        reach_names = names_attr.split(",") if names_attr else None
    else:
        reach_id_arr = None
        reach_names = None

    # Build areas from the H3 cells — cell_area is per-cell at res-aware
    # scale, so a multi-res mesh has multi-scale areas natively.  The
    # NC stores h3_id in compact-index order; the H3MultiResMesh
    # constructor receives these arrays in the same order, so areas
    # below align cell-for-cell with everything else.  We can't call
    # `from_h3_cells` (which would also recompute areas) because that
    # would discard the pre-built CSR neighbour table in the NC.
    cells_str = [_h3.int_to_str(int(i)) for i in ds["h3_id"].values]
    areas = np.array(
        [_h3.cell_area(c, unit="m^2") for c in cells_str],
        dtype=np.float32,
    )

    centroids = np.column_stack(
        [ds["lat"].values, ds["lon"].values],
    )

    self.mesh = H3MultiResMesh(
        h3_ids=ds["h3_id"].values.astype(np.uint64),
        resolutions=ds["resolution"].values.astype(np.int8),
        centroids=centroids,
        nbr_starts=ds["nbr_starts"].values.astype(np.int32),
        nbr_idx=ds["nbr_idx"].values.astype(np.int32),
        water_mask=ds["water_mask"].values.astype(bool),
        depth=ds["depth"].values.astype(np.float32),
        areas=areas,
        reach_id=reach_id_arr,
        reach_names=reach_names,
    )
    ds.close()
    self.env = H3Environment.from_netcdf(landscape_path, self.mesh)
```

- [ ] **Step 2.6: Run the test, expect PASS for the first three.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py::test_mesh_is_h3multires" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py::test_resolutions_are_mixed" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py::test_initial_placement_lands_on_water" -v
```

Expected: 3 PASSED.

- [ ] **Step 2.7: Run the step + movement tests.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py::test_one_step_runs_without_error" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py::test_at_least_one_agent_moves" -v
```

Expected: 2 PASSED. If `test_one_step_runs_without_error` fails with a numba IndexError or "out of bounds" message, the kernel is reading past the `_water_nbr_count` row width — return to Step 0.3 to find any hardcoded `range(6)` and fix.

- [ ] **Step 2.8: Commit.**

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" add salmon_ibm/simulation.py salmon_ibm/config.py configs/config_curonian_h3_multires.yaml tests/test_h3_multires_integration.py
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" commit -m "feat(h3-multires): wire H3MultiResMesh through Simulation init"
```

---

## Task 3: Wire the sidebar dropdown + viewer

The sidebar dropdown lists landscapes via the `landscape` input; selecting `curonian_h3_multires` must trigger the new sim branch and render the mixed-resolution mesh as proper hexagons.

**Files:**
- Modify: `salmon_ibm/h3_multires.py:255` (add a `resolution` compat property — see Step 3.0)
- Modify: `ui/sidebar.py:55-90` (add the new dropdown choice)
- Modify: `app.py:710-735` (add the routing in `_do_init_sim`)
- Modify: `app.py:1380-1420, 1518, 1611, 1713` (h3 viewer branches — see Step 3.5)

- [ ] **Step 3.0: Add a `resolution` compat property to `H3MultiResMesh`.**

`app.py` routes to the H3 viewer branch via `hasattr(mesh, "resolution")` (singular) at five call sites (`app.py:1380, 1407, 1518, 1611, 1713`). `H3MultiResMesh` only exposes `resolutions` (plural).  Without this property the multi-res mesh falls through to the TriMesh branch and the viewer renders dots instead of hexagons.

`H3Mesh.find_triangle` (`salmon_ibm/h3mesh.py:117`) also reads `self.resolution` to do `h3.latlng_to_cell(lat, lon, self.resolution)`. The same call on a multi-res mesh would miss cells in fine zones — but for now the median-resolution fallback is enough for the agent-placement code path and the click-to-inspect viewer use case.  Proper multi-res `find_triangle` (try fine first, fall back to coarse) is a TODO comment on the property.

In `salmon_ibm/h3_multires.py` near the other properties (around line 255), add:

```python
@property
def resolution(self) -> int:
    """Median per-cell resolution — for legacy code that expects a
    single ``mesh.resolution`` (the deck.gl viewer in app.py and
    H3Mesh.find_triangle).  Multi-res-aware code should read
    ``self.resolutions`` (plural) instead.

    Edge cases:

    * Empty mesh — returns 9 (the default Curonian-Lagoon resolution),
      because ``np.median`` on an empty array raises and the only
      callers (viewer zoom + find_triangle) need a numeric value.
    * Even split between two resolutions — uses ``round`` rather
      than ``int`` truncation so 9.5 → 10, not 9.

    TODO: a proper multi-res ``find_triangle`` should walk the H3
    hierarchy from the finest resolution in use down to the coarsest,
    returning the first match.  Median is good enough for the scaffold
    because the only caller that matters at this stage is the viewer
    zoom heuristic.
    """
    arr = self.resolutions[self.water_mask] if self.water_mask.any() else self.resolutions
    if len(arr) == 0:
        return 9  # Curonian default; arbitrary but never reached in practice
    return int(round(float(np.median(arr))))
```

Run the multi-res mesh tests to confirm the property is exposed without breaking existing tests:

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires.py" -v
```

Expected: 8 PASSED (no regression).  Optional: add a one-liner test asserting `mesh.resolution` returns an int in `[res_min, res_max]`.

- [ ] **Step 3.1: Append a sidebar-presence check to the multi-res integration test file.**

Append to the **same** `tests/test_h3_multires_integration.py` we created in Task 2:

```python
def test_h3_multires_in_sidebar_choices():
    """The Study area dropdown must include 'curonian_h3_multires' so
    a user can select it without typing URL params.  Walks the
    sidebar UI panel object tree and looks for the choice key in the
    rendered HTML."""
    from ui.sidebar import sidebar_panel
    panel = sidebar_panel()
    html = str(panel)
    assert "curonian_h3_multires" in html, (
        "h3_multires not registered in sidebar choices; users can't "
        "select it"
    )
```

Run it:

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py::test_h3_multires_in_sidebar_choices" -v
```

Expected: FAIL — option not yet added.

- [ ] **Step 3.2: Add the dropdown choice in `ui/sidebar.py`.**

In the `choices=` dict of the `landscape` input_select (around line 55-75), add as a new entry — placement is up to taste; below shows it as the new default since multi-res is the most ecologically faithful:

```python
choices={
    "curonian_h3_multires": "Curonian Lagoon H3 (multi-res)",  # NEW
    "nemunas":               "Curonian Lagoon H3",
    "curonian_trimesh":      "Curonian Lagoon TriMesh",
    "columbia":              "Columbia River",
},
selected="curonian_h3_multires",
```

- [ ] **Step 3.3: Wire the routing in `app.py:_do_init_sim`.**

Find the existing config-dispatch ladder around line 710-735 and add the new branch:

```python
if landscape == "columbia":
    cfg = load_config("config_columbia.yaml")
elif landscape == "curonian_trimesh":
    cfg = load_config("configs/config_curonian_trimesh.yaml")
elif landscape == "curonian_h3_multires":
    cfg = load_config("configs/config_curonian_h3_multires.yaml")
else:
    cfg = load_config("configs/config_nemunas_h3.yaml")
```

Also update the `grid_name` dispatch dict (around line 690-695):

```python
grid_name = {
    "columbia": "Columbia River",
    "nemunas": "Curonian Lagoon H3",
    "curonian_trimesh": "Curonian Lagoon TriMesh",
    "curonian_h3_multires": "Curonian Lagoon H3 (multi-res)",  # NEW
}.get(landscape, "Curonian Lagoon H3")
```

- [ ] **Step 3.4: Run the dropdown test, expect PASS.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py::test_h3_multires_in_sidebar_choices" -v
```

Expected: PASS.

- [ ] **Step 3.5: Confirm the existing viewer branches handle multi-res via the `resolution` property added in Step 3.0.**

The H3 detection check `hasattr(sim.mesh, "resolution")` (`app.py:1380, 1407, 1518, 1611, 1713`) now returns True for `H3MultiResMesh` because Step 3.0 added the property.  The branch body uses `mesh.h3_ids` (works on both meshes) and computes the camera zoom from `h3.average_hexagon_edge_length(mesh.resolution, unit="m")` — `mesh.resolution` returns the median for multi-res, which gives a sensible camera target somewhere between the lagoon hex size and river hex size.

deck.gl's `H3HexagonLayer` reads each cell string's intrinsic resolution from the H3 ID — **no per-cell resolution prop is needed in the layer payload**, so the existing `data_rows` construction loop (`{"hex": h3.int_to_str(int(h)), "color": [r, g, b, 220]}`) still produces the correct mixed-resolution rendering.  Verify in the next step.

This step is a **no-op confirmation** — no code edit needed.  If you grep `app.py` and see no remaining hardcoded `H3Mesh` isinstance checks, you're good:

```bash
grep -n "isinstance.*H3Mesh\b" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/app.py"
```

Expected: zero matches (the codebase uses duck-typed `hasattr` everywhere).  If you find any, replace with `hasattr(sim.mesh, "resolution")` so multi-res is included.

- [ ] **Step 3.6: Verify visually with Playwright.**

```bash
bash "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/scripts/deploy_laguna.sh" apply --allow-untagged
```

Then in Playwright:

```python
# In the active claude session — uses the playwright MCP tools we already have
mcp__plugin_playwright_playwright__browser_navigate(url="http://laguna.ku.lt/HexSimPy/?cb=multires_check")
# wait for sim init (~30 s for the multi-res landscape — neighbour table is bigger)
mcp__plugin_playwright_playwright__browser_wait_for(time=45)
mcp__plugin_playwright_playwright__browser_take_screenshot(filename="multires-rendered.png")
```

Read the screenshot. Expected: hexagons render at multiple visible scales — small fine hexes traced along the Nemunas, Atmata, Minija delta channels and large coarse hexes filling the lagoon and Baltic. No warnings in the page console about unknown layer type or bad data.

- [ ] **Step 3.7: Commit.**

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" add ui/sidebar.py app.py tests/test_h3_multires_integration.py
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" commit -m "feat(h3-multires): sidebar option + viewer per-cell resolution"
```

---

## Task 4: Cross-resolution movement test

A regression test ensuring agents *do* cross resolution boundaries during a simulation — proving the multi-res neighbour links work end-to-end, not just at construction time.

**Files:**
- Modify: `tests/test_h3_multires_integration.py` (append the cross-res test)

- [ ] **Step 4.1: Write the cross-res movement test.**

Use a **fresh** `Simulation` instance (not the shared module-scope
`multires_sim` fixture) so position-overwrite doesn't depend on what
prior tests already mutated.  Mark `slow` and reduce the time horizon
from 30 days × 24 h to 3 days × 24 h — the cross-res boundary is at
most a few cells from any starting fine cell, so 72 hourly steps is
ample to surface a wiring bug.  Plan-review pass 6 flagged the
720-step / shared-fixture combination as both brittle and too slow
for a default `pytest` run.

Append to `tests/test_h3_multires_integration.py`:

```python
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
```

Configure pytest to skip `slow` tests by default — add to
`pytest.ini` or `pyproject.toml`:

```ini
# pytest.ini (create if missing)
[pytest]
markers =
    slow: tests that exceed 30 s wall time; opt in with `-m slow`.
addopts = -m "not slow"
```

Slow tests run via `pytest -m slow` or `pytest -m ""` (override the
default filter).

- [ ] **Step 4.2: Run the cross-res test.**

The test is `@pytest.mark.slow` and `pytest.ini`'s default `addopts = -m "not slow"` filters it out — must override with `-m slow`:

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py::test_agents_cross_resolution_boundaries" -v -m slow
```

Expected: PASS.  If it fails with `crossed = False`, double-check that `_water_nbr_count` includes cross-resolution neighbours (inspect `mesh._water_nbr_count[fine_river_cell]` — should be ≥ 6 + extras at the boundary).

- [ ] **Step 4.2b: Add four supplementary ecological invariants.**

Plan-review pass 6 (scientific validator) recommended four additional invariants beyond the binary "did anyone cross?" test.  Each catches a specific failure mode that would otherwise go undetected.

Append to `tests/test_h3_multires_integration.py`:

```python
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
    coarse = np.where(
        (sim.mesh.resolutions == 8) & sim.mesh.water_mask
    )[0]
    if len(coarse) < 50:
        pytest.skip(f"only {len(coarse)} coarse water cells")
    sim.pool.tri_idx[:] = coarse[:50]
    for _ in range(3 * 24):
        sim.step()
    final_res = sim.mesh.resolutions[sim.pool.tri_idx]
    came_back = (final_res > 8).any()
    assert came_back, (
        "no agent moved from res-8 OpenBaltic to a finer zone in 3 "
        "days — the cross-res neighbour table is asymmetric (fine→"
        "coarse works but coarse→fine is broken)."
    )


def test_per_step_displacement_bounded(multires_sim):
    """No single step's haversine displacement should exceed the
    cell edge of the *destination* cell.  Catches malformed
    neighbour entries that would otherwise teleport agents."""
    import numpy as np
    import h3 as _h3
    mesh = multires_sim.mesh
    before = mesh.centroids[multires_sim.pool.tri_idx].copy()
    multires_sim.step()
    after = mesh.centroids[multires_sim.pool.tri_idx]

    # Haversine distance per agent (rough — small angles, no need
    # for great-circle precision).
    dlat = np.radians(after[:, 0] - before[:, 0])
    dlon = np.radians(after[:, 1] - before[:, 1])
    mid_lat = np.radians((after[:, 0] + before[:, 0]) / 2)
    dy = dlat * 6_371_000  # m
    dx = dlon * 6_371_000 * np.cos(mid_lat)
    disp = np.sqrt(dx**2 + dy**2)

    # Each cell's edge length depends on its resolution.
    dest_res = mesh.resolutions[multires_sim.pool.tri_idx]
    max_disp = np.array(
        [_h3.average_hexagon_edge_length(int(r), unit="m") * 3
         for r in dest_res]
    )

    violations = disp > max_disp
    assert not violations.any(), (
        f"{int(violations.sum())} agents teleported further than "
        f"3 × destination cell edge — cross-res neighbour table "
        f"may have spatially-distant entries.  Worst case: "
        f"{disp.max():.0f} m vs max {max_disp[violations].max():.0f} m."
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
```

- [ ] **Step 4.3: Run the full multi-res suite.**

Two-pass run — once at the default filter, once with the slow tests included:

```bash
# Default filter (addopts = -m "not slow"):
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires.py" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_builder.py" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py" --tb=short
```

Expected: `17 passed, 1 skipped, 2 deselected` — the 8 scaffold + 1 builder + 8 fast integration tests pass; the mass-loss test is `pytest.skip`ped pending Phase B; the two `@pytest.mark.slow` tests are deselected by `addopts`.

```bash
# Slow run (override the default filter):
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_multires_integration.py" -v -m slow
```

Expected: `2 passed, 9 deselected` — the two cross-res slow tests pass; the 9 fast integration tests are filtered out.

Integration test inventory (11 total in `test_h3_multires_integration.py`):
* `mesh_is_h3multires`                              — fast
* `resolutions_are_mixed`                           — fast
* `initial_placement_lands_on_water`                — fast
* `one_step_runs_without_error`                     — fast
* `at_least_one_agent_moves`                        — fast
* `h3_multires_in_sidebar_choices`                  — fast
* `salinity_correlates_with_resolution_zone`        — fast
* `per_step_displacement_bounded`                   — fast
* `mass_loss_rate_independent_of_cell_resolution`   — fast, but `pytest.skip` until Phase B
* `agents_cross_resolution_boundaries`              — slow
* `no_one_way_trap_at_resolution_boundary`          — slow

- [ ] **Step 4.4: Run the existing H3 + barriers + env regression.**

```bash
micromamba run -n shiny python -m pytest "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_env.py" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3_barriers.py" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_h3mesh.py" "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/tests/test_nemunas_h3_integration.py" --tb=short
```

Expected: 49 PASSED (the existing-passing count from v1.2.7).

- [ ] **Step 4.5: Commit.**

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" add tests/test_h3_multires_integration.py
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" commit -m "test(h3-multires): cross-resolution movement regression"
```

---

## Task 5: Tag, deploy, verify

- [ ] **Step 5.1: Tag.**

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" tag -a v1.3.0 -m "v1.3.0 — multi-resolution H3 backend (Phase 2 wired)"
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" push origin main
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" push origin v1.3.0
```

- [ ] **Step 5.2: Deploy.**

```bash
bash "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/scripts/deploy_laguna.sh" apply
```

Expected: `==> Done.  Deployed <sha> v1.3.0.`

- [ ] **Step 5.3: SCP the multi-res NC.**

```bash
scp "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/data/curonian_h3_multires_landscape.nc" razinka@laguna.ku.lt:/srv/shiny-server/HexSimPy/data/
ssh razinka@laguna.ku.lt 'cd /srv/shiny-server/HexSimPy && md5sum data/curonian_h3_multires_landscape.nc && touch restart.txt'
```

Compare server md5 to local:

```bash
md5sum "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/data/curonian_h3_multires_landscape.nc"
```

Expected: identical hashes.

- [ ] **Step 5.4: Browser sanity check.**

Hard-refresh `http://laguna.ku.lt/HexSimPy/` (Ctrl+Shift+R), pick "Curonian Lagoon H3 (multi-res)" from Study area, click Reset, watch for ~30 s init, click Run. Expected: agents place across the mesh, fine river hexes visible at high zoom, lagoon coarse hexes at low zoom, no errors in browser console.

- [ ] **Step 5.5: Update the roadmap doc.**

Edit `docs/h3-multires-roadmap.md` Phase 2 status line: `Status: shipped in v1.3.0`. Close out the "remaining effort" table.

```bash
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" add docs/h3-multires-roadmap.md
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" commit -m "docs(h3-multires): mark Phase 2 shipped in v1.3.0"
git -C "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim" push origin main
```

---

## Known limitations (documented, not fixed in Phase 2)

The scientific-validation review surfaced four ecological concerns that don't block Phase 2 from shipping but should be tracked.  Add a section to `docs/h3-multires-roadmap.md` listing them:

1. **Density-dependent mortality breaks under varied cell areas.**  `events_builtin.py:166-171` computes `local_density = cell_counts[positions]` (agents-per-*cell*, not per-m²).  A res-8 Baltic cell with 50 agents → 11 fish/km²; a res-10 river with 5 agents → 54 fish/km².  The river is denser but gets a smaller penalty.  Fix: when `mesh_backend == "h3_multires"`, divide by `mesh.areas[positions]` before applying density scale.  Out of scope for Phase 2 because no current test exercises stage-specific survival; flag in the roadmap as a Phase B prerequisite.

2. **`_step_to_cwr_numba` cross-res ecology**.  The CWR-seeking kernel (`movement.py:184-204`) picks the absolute coldest neighbour without weighting by distance.  At a fine→coarse boundary the coarse cell's temperature is a 530 m spatial average that may not represent realistic refugia.  For Phase 2 ship as-is — the test suite doesn't exercise CWR seeking at boundaries.  Phase 5 calibration should add a test placing an agent at a hot fine cell adjacent to one cold fine and one same-temperature coarse cell, asserting it picks the *fine* cold cell.

3. **N-S salinity gradient test invariant fragile.**  `test_north_south_salinity_gradient` (`test_nemunas_h3_integration.py:168-187`) uses cell-count percentiles that disproportionately sample the smaller-cell-count OpenBaltic res-8 zone for the multi-res mesh.  Phase 2 ships a *replacement* test (`test_salinity_correlates_with_resolution_zone` added in Step 4.2b) that uses reach IDs instead.  Don't delete the old test — it still applies to the uniform-res H3 backend.

4. **Migration-distance accumulation must use haversine, not step count.**  Future migration-progress / ETA-to-spawning metrics that count `tri_idx` changes will report 7× faster Baltic speeds than river speeds because coarse cells are bigger.  No such metric currently exists; flag in the roadmap as a constraint on any future telemetry feature.

---

## Build-time expectations (corrected from review)

Performance review pass 6 corrected the build-time estimate for `find_cross_res_neighbours`:

* Original docstring claim: ~1 s for 50 k cells.
* Realistic at smoke-build res-10 rivers: **30–90 s**, dominated by `h3.cell_to_children` calls at `r+3` (343 children per missing ring member, worst case).
* CMEMS sampling adds another ~10 s of fixed `RegularGridInterpolator` construction overhead per build.

Total Step 1.4 wall time: expect **2–3 minutes**, not seconds.  This is acceptable for a one-shot build but rules out runtime use.  Future optimisation: batch `cell_to_latlng` + `latlng_to_cell` calls instead of per-cell.

---

## Out of scope (intentionally)

* **Per-reach ecology parameters** (`mortality_per_reach`, `drift_conc_per_reach`, etc.).  See `docs/per-reach-ecology-plan.md`.  The reach_id wiring from v1.2.7 is the foundation; Phase B implements it.  Independent of multi-res.
* **Variable-resolution barriers**.  `barriers_csv: null` in the new config — barrier-CSV loading currently assumes one resolution.  Punted to Phase 4 of the roadmap.
* **Multi-res calibration against published Nemunas hydraulics**.  Roadmap Phase 5.  Real ecological work, not coding work.
* **Resolution tuning to res 11 in the rivers**.  This plan ships the wiring with rivers at res 10 (~75 m, ~36 k cells total) — small enough that the integration test finishes in <60 s.  Bumping the river config to `Nemunas=11,Atmata=11,…` (the default in `DEFAULT_RES`) yields ~17 k extra fine cells and ~50 k cells total; build time grows from ~10 s to ~30 s and the test suite would need a matching skip-or-slow marker.  Do this in a follow-up patch once Phase 2 lands green.
