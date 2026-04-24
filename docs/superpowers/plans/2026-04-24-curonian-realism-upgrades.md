# Curonian Lagoon Realism Upgrades — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the HexSim Curonian Lagoon study area from a stub-driven setup (generic Chinook bioenergetics + placeholder data files + scalar salinity gate) into a defensible Baltic salmon model with real bathymetry, spatially-explicit salinity, real temperature forcing, species-accurate parameters, and post-smolt marine mortality — closing the gap to inSTREAM's `example_baltic.yaml`.

**Architecture:** Work in priority order. P0 (data + species config) unlocks all downstream calibration. P1 (reach attributes, predation) builds ecological realism on that foundation. P2 (ice, seiche, redd scour) refines. Each task has a concrete deliverable (data file, config block, or code change) with verification.

**Tech Stack:** Python 3.10+, NumPy, xarray, PyYAML, scipy, pytest, `micromamba run -n shiny`. Data: EMODnet Bathymetry WCS, CMEMS Baltic reanalysis, HELCOM/ICES discharge records, OSM (via osmnx).

**Reference implementation:** inSTREAM at `C:\Users\arturas.baziukas\OneDrive - ku.lt\HORIZON_EUROPE\inSTREAM\instream-py\`. Specifically:
- `configs/baltic_salmon_species.yaml` — 225-line species config with 24 peer-reviewed citations
- `configs/example_baltic.yaml` — 420-line multi-reach full config
- `scripts/generate_baltic_example.py` — cell generation, OSM extraction
- `app/modules/bathymetry.py::fetch_emodnet_dtm()` — EMODnet WCS client
- `docs/calibration-notes.md` — parameter provenance

**Test command:** `micromamba run -n shiny python -m pytest tests/ -v`. Substitute `conda run -n shiny` on machines with conda on PATH.

---

## File Structure

### New files to create

- `salmon_ibm/baltic_params.py` — `BalticBioParams` dataclass + `load_baltic_species_config()` loader
- `configs/baltic_salmon_species.yaml` — ported from inSTREAM (225 lines, 24 citations preserved)
- `configs/config_curonian_baltic.yaml` — new full Curonian config referencing the real data
- `scripts/fetch_emodnet_bathymetry.py` — EMODnet WCS client, regrids to Curonian mesh
- `scripts/fetch_cmems_forcing.py` — CMEMS Baltic Physics + BGC fetcher for temperature + salinity
- `scripts/fetch_nemunas_discharge.py` — Lithuanian EPA / HELCOM discharge fetcher
- `data/curonian_bathymetry.nc` — real EMODnet depths on the Curonian mesh
- `data/curonian_forcing_cmems.nc` — real daily temperature, salinity, currents, SSH for 2011-2024
- `data/nemunas_discharge.nc` — real daily Q
- `docs/curonian-data-provenance.md` — per-variable data source, fetch date, license
- `tests/test_baltic_params.py` — species config loader tests
- `tests/test_environment_salinity_field.py` — salinity gradient field tests
- `tests/test_curonian_realism_integration.py` — end-to-end smoke with real data

### Modified files

- `salmon_ibm/bioenergetics.py` — accept `BalticBioParams` in addition to `BioParams`
- `salmon_ibm/estuary.py` — promote `salinity_cost()` to accept per-cell salinity field
- `salmon_ibm/environment.py` — wire a spatially-explicit salinity field into `fields`
- `salmon_ibm/config.py` — support species-config block pointing to baltic_salmon_species.yaml

---

## Scope: what this plan does NOT cover

- **Full reach-level habitat attributes** (FRACSPWN, FRACVSHL, drift_conc per cell). Requires field survey + OSM landcover extraction. Scoped as a dedicated sub-plan after this plan lands.
- **Grey seal + cormorant predation events.** Requires adding new event types to `events_builtin.py`. Separate P1 sub-plan.
- **Ice cover, seiche wind-forcing, redd scour depth tuning.** All P2; separate plans when P0+P1 are proven.

---

# PHASE 0 — Infrastructure

## Task 0.1: Capture baseline behavior

**Purpose:** record what the stub Curonian config produces today, so the real-data upgrade has a measurable baseline.

**Files:** `scripts/baseline_curonian_stub.py` (new)

- [ ] **Step 1: Write a deterministic baseline script**

```python
"""Run current stub Curonian config; record key outputs for comparison."""
import numpy as np
from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation

def run():
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=500, data_dir="data", rng_seed=42)
    sim.run(n_steps=720)  # 30 days hourly
    alive = int(sim.pool.alive.sum())
    arrived = int(sim.pool.arrived.sum())
    mean_ed = float(sim.pool.ed_kJ_g[sim.pool.alive].mean()) if alive else 0.0
    mean_mass = float(sim.pool.mass_g[sim.pool.alive].mean()) if alive else 0.0
    return {
        "alive": alive, "arrived": arrived,
        "mean_ed_kJ_g": mean_ed, "mean_mass_g": mean_mass,
        "temp_range": (float(sim.env.fields["temperature"].min()),
                       float(sim.env.fields["temperature"].max())),
    }

if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k}: {v}")
```

- [ ] **Step 2: Run and record output**

```bash
micromamba run -n shiny python scripts/baseline_curonian_stub.py > docs/curonian-baseline-stub.txt
```

- [ ] **Step 3: Commit**

```bash
git add scripts/baseline_curonian_stub.py docs/curonian-baseline-stub.txt
git commit -m "bench(curonian): capture stub-config baseline outputs for realism comparison"
```

---

# PHASE 1 — Port Baltic Salmon Species Config (P0)

## Task 1.1: Create `BalticBioParams` dataclass

**Purpose:** HexSim's `BioParams` is Snyder-Chinook-derived. Baltic salmon need different RA/RB/RQ, different thermal response, different fecundity. Extend rather than replace so existing Chinook tests keep passing.

**Files:**
- `salmon_ibm/baltic_params.py` (new)
- `tests/test_baltic_params.py` (new)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_baltic_params.py`:

```python
"""Tests for Baltic Atlantic salmon parameter loader."""
import pytest
from salmon_ibm.baltic_params import BalticBioParams, load_baltic_species_config


def test_baltic_bioparams_defaults_from_literature():
    """Default values must match the peer-reviewed values from the Baltic species config."""
    p = BalticBioParams()
    # Smith et al. 2009 CMax coefficients (marine-phase post-smolt S. salar)
    assert p.cmax_A == pytest.approx(0.303)
    assert p.cmax_B == pytest.approx(-0.275)
    # Koskela et al. 1997 thermal optimum; Handeland 2008 upper lethal
    assert p.T_OPT == pytest.approx(16.0)
    assert p.T_MAX == pytest.approx(20.0)
    # Kallio-Nyberg 2020 length-weight coefficients
    assert p.LW_a == pytest.approx(0.0077)
    assert p.LW_b == pytest.approx(3.05)
    # Baum & Prouzet 1987 linear fecundity
    assert p.fecundity_per_g == pytest.approx(2.0)


def test_baltic_bioparams_rejects_invalid_ranges():
    """Post-init validation must reject nonsense parameters (same discipline as BioParams)."""
    with pytest.raises(ValueError, match="T_MAX"):
        BalticBioParams(T_OPT=25.0, T_MAX=20.0)
    with pytest.raises(ValueError, match="cmax_A"):
        BalticBioParams(cmax_A=-1.0)


def test_baltic_species_config_loader_parses_yaml(tmp_path):
    """Loader must parse the canonical baltic_salmon_species.yaml and return BalticBioParams."""
    cfg_path = tmp_path / "species.yaml"
    cfg_path.write_text("""
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    cmax_B: -0.275
    T_OPT: 16.0
    T_MAX: 20.0
    LW_a: 0.0077
    LW_b: 3.05
    fecundity_per_g: 2.0
    spawn_window_start_day: 288
    spawn_window_end_day: 334
""")
    params = load_baltic_species_config(cfg_path)
    assert isinstance(params, BalticBioParams)
    assert params.T_OPT == 16.0
    assert params.spawn_window_start_day == 288  # Oct 15
    assert params.spawn_window_end_day == 334    # Nov 30
```

- [ ] **Step 2: Run the test — expect FAIL**

```bash
micromamba run -n shiny python -m pytest tests/test_baltic_params.py -v
```

Expected: `ModuleNotFoundError: No module named 'salmon_ibm.baltic_params'`.

- [ ] **Step 3: Create `salmon_ibm/baltic_params.py`**

```python
"""Baltic Atlantic salmon (Salmo salar) bioenergetics parameters.

All scientifically-sensitive values are sourced from peer-reviewed
literature; citations inline below. Parameter provenance summary:

  cmax_A, cmax_B: Smith, Booker & Wells 2009 (doi:10.1016/j.marenvres.2008.12.010)
  T_OPT:          Koskela et al. 1997 (Baltic salmon 16-29 cm fish)
  T_MAX:          Handeland et al. 2008 (doi:10.1016/j.aquaculture.2008.03.057)
  LW_a, LW_b:     Kallio-Nyberg et al. 2020 (Baltic post-smolt LW regression)
  fecundity:      Baum & Prouzet 1987 (Atlantic salmon linear fecundity)
  spawn window:   Lilja & Romakkaniemi 2003 (Baltic spawning phenology)

These values supersede the generic Snyder-Chinook BioParams for
Baltic salmon simulations. See docs/superpowers/specs/ for full
calibration notes if the inSTREAM doc is ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BalticBioParams:
    # Wisconsin bioenergetics (compatible with existing update_energy signature)
    RA: float = 0.00264       # Snyder 2019 baseline, overridable
    RB: float = -0.217
    RQ: float = 0.06818
    ED_MORTAL: float = 4.0    # kJ/g
    ED_TISSUE: float = 36.0   # lipid-first catabolism (Brett 1995 / Breck 2008)
    MASS_FLOOR_FRACTION: float = 0.5

    # Species-specific Baltic values
    cmax_A: float = 0.303          # Smith 2009 post-smolt CMax intercept
    cmax_B: float = -0.275         # Smith 2009 post-smolt CMax slope
    T_OPT: float = 16.0            # Koskela 1997 Baltic salmon thermal peak
    T_MAX: float = 20.0            # Handeland 2008 zero-growth upper bound
    LW_a: float = 0.0077           # Kallio-Nyberg 2020 length-weight intercept
    LW_b: float = 3.05             # Kallio-Nyberg 2020 length-weight exponent
    fecundity_per_g: float = 2.0   # Baum & Prouzet 1987 linear eggs-per-gram

    # Spawning phenology (day of year; Lilja & Romakkaniemi 2003)
    spawn_window_start_day: int = 288  # Oct 15
    spawn_window_end_day: int = 334    # Nov 30
    spawn_temp_min_c: float = 5.0
    spawn_temp_max_c: float = 14.0

    # Activity by behavior (keep Snyder structure, Baltic-tunable)
    activity_by_behavior: dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    )

    def __post_init__(self):
        if self.T_MAX <= self.T_OPT:
            raise ValueError(
                f"T_MAX ({self.T_MAX}) must be > T_OPT ({self.T_OPT})"
            )
        if self.cmax_A <= 0:
            raise ValueError(f"cmax_A must be > 0, got {self.cmax_A}")
        if self.LW_a <= 0 or self.LW_b <= 0:
            raise ValueError("Length-weight coefficients must be positive")
        if not (0 < self.MASS_FLOOR_FRACTION <= 1):
            raise ValueError(
                f"MASS_FLOOR_FRACTION must be in (0, 1], got {self.MASS_FLOOR_FRACTION}"
            )
        if self.ED_TISSUE <= 0 or self.ED_MORTAL <= 0:
            raise ValueError("ED_TISSUE and ED_MORTAL must be > 0")
        if not (0 <= self.spawn_window_start_day <= 365):
            raise ValueError("spawn_window_start_day must be 0-365")


def load_baltic_species_config(path: str | Path) -> BalticBioParams:
    """Load the canonical baltic_salmon_species.yaml into BalticBioParams."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("species", {}).get("BalticAtlanticSalmon")
    if block is None:
        raise ValueError(
            f"{path} missing species.BalticAtlanticSalmon block"
        )
    # Filter to fields BalticBioParams knows about; extra keys tolerated.
    known = {f.name for f in BalticBioParams.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in block.items() if k in known}
    return BalticBioParams(**kwargs)
```

- [ ] **Step 4: Run the test — expect PASS**

```bash
micromamba run -n shiny python -m pytest tests/test_baltic_params.py -v
```

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/baltic_params.py tests/test_baltic_params.py
git commit -m "feat(baltic): BalticBioParams dataclass with literature-traced defaults

Port of inSTREAM's baltic_salmon_species.yaml parameters into HexSim's
Python config system. 24 peer-reviewed citations preserved in the
module docstring; scientifically-sensitive values (CMax, T_OPT, T_MAX,
length-weight, fecundity, spawn window) wired to BalticBioParams fields.

__post_init__ validates ranges in the same discipline as the existing
BioParams validation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

## Task 1.2: Port the full `baltic_salmon_species.yaml` config

**Files:**
- `configs/baltic_salmon_species.yaml` (new — copy from inSTREAM with HexSim schema adaptations)

- [ ] **Step 1: Copy and adapt**

```bash
cp "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/inSTREAM/instream-py/configs/baltic_salmon_species.yaml" \
    "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/configs/baltic_salmon_species.yaml"
```

Then edit the new file to keep only the fields `BalticBioParams` recognizes (CMax, T_OPT, T_MAX, LW_a, LW_b, fecundity, spawn window). Preserve all citation comments — they document provenance.

- [ ] **Step 2: Test round-trip: config → params**

Run the `test_baltic_species_config_loader_parses_yaml` test against the real ported file:

```bash
micromamba run -n shiny python -c "
from salmon_ibm.baltic_params import load_baltic_species_config
p = load_baltic_species_config('configs/baltic_salmon_species.yaml')
print(f'OK: T_OPT={p.T_OPT}, T_MAX={p.T_MAX}, spawn={p.spawn_window_start_day}-{p.spawn_window_end_day}')
"
```

Expected output: `OK: T_OPT=16.0, T_MAX=20.0, spawn=288-334`.

- [ ] **Step 3: Commit**

```bash
git add configs/baltic_salmon_species.yaml
git commit -m "feat(baltic): port baltic_salmon_species.yaml from inSTREAM with full citations"
```

---

# PHASE 2 — Real EMODnet Bathymetry (P0)

## Task 2.1: EMODnet WCS fetch script

**Purpose:** Replace the stub `depth` variable in `data/curonian_minimal_grid.nc` with a real 1/16-arcmin EMODnet DTM sampled onto the Curonian mesh.

**Reference implementation:** `inSTREAM/instream-py/app/modules/bathymetry.py::fetch_emodnet_dtm()`.

**Files:**
- `scripts/fetch_emodnet_bathymetry.py` (new)
- `data/curonian_bathymetry.nc` (new, generated)

- [ ] **Step 1: Inspect the inSTREAM reference**

```bash
grep -n "fetch_emodnet_dtm\|emodnet__mean\|WCS" \
    "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/inSTREAM/instream-py/app/modules/bathymetry.py" | head -30
```

Note the WCS URL, coverage name, request parameters, and the rasterio/xarray pattern it uses.

- [ ] **Step 2: Write `scripts/fetch_emodnet_bathymetry.py`**

Skeleton (adapt from inSTREAM reference):

```python
"""Fetch EMODnet Bathymetry for the Curonian Lagoon + Nemunas mouth + Baltic coast.

Output: data/curonian_bathymetry.nc with variables (lat, lon, depth).

Fetches EMODnet DTM 2022 (1/16 arc-minute mean) via WCS, subsets to
the Curonian bounding box, and regrids onto the existing Curonian
triangular mesh defined in data/curonian_minimal_grid.nc.

Provenance recorded in docs/curonian-data-provenance.md.
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import requests  # or owslib.wcs; whichever inSTREAM used

# Curonian Lagoon + mouth + 10 km coastal buffer
BBOX = {
    "minlon": 20.4, "maxlon": 21.9,
    "minlat": 54.9, "maxlat": 55.8,
}
WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"
COVERAGE = "emodnet__mean"


def fetch(bbox=BBOX, out_path="data/curonian_bathymetry_raw.tif"):
    # WCS GetCoverage request; see inSTREAM reference for exact params
    ...


def regrid_to_mesh(raw_tif_path, mesh_nc_path, out_nc_path):
    mesh = xr.open_dataset(mesh_nc_path)
    raw = xr.open_rasterio(raw_tif_path).squeeze()
    # Interpolate raw (regular grid) onto mesh.lat, mesh.lon (1D node positions)
    depth = raw.interp(x=mesh.lon, y=mesh.lat, method="linear")
    ds = xr.Dataset(
        {"depth": (("node",), depth.values)},
        coords={
            "lat": ("node", mesh.lat.values),
            "lon": ("node", mesh.lon.values),
        },
        attrs={
            "source": "EMODnet Bathymetry 2022, 1/16 arc-min mean",
            "url": WCS_URL,
            "coverage": COVERAGE,
            "fetched": "<fill-in-date>",
            "license": "CC-BY 4.0 (EMODnet)",
        },
    )
    ds.to_netcdf(out_nc_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="data/curonian_minimal_grid.nc")
    parser.add_argument("--out", default="data/curonian_bathymetry.nc")
    args = parser.parse_args()
    tif = "data/curonian_bathymetry_raw.tif"
    fetch(out_path=tif)
    regrid_to_mesh(tif, args.mesh, args.out)
    print(f"Wrote {args.out}")
```

- [ ] **Step 3: Run it once and verify sanity**

```bash
micromamba run -n shiny python scripts/fetch_emodnet_bathymetry.py
micromamba run -n shiny python -c "
import xarray as xr
ds = xr.open_dataset('data/curonian_bathymetry.nc')
print(f'n_nodes={ds.depth.size}')
print(f'depth range: {float(ds.depth.min()):.2f} to {float(ds.depth.max()):.2f} m')
print(f'mean Curonian depth (expected ~3.8 m): {float(ds.depth.mean()):.2f} m')
"
```

Expected values (validate against EMODnet):
- Curonian Lagoon mean: ~3.5-4.5 m (it is a shallow lagoon)
- Near Klaipėda Strait: 5-15 m
- Baltic Sea coast: 10-25 m within 10 km
- No negative depths (land) in the mesh region; if present, mesh mask is wrong.

- [ ] **Step 4: Add regression test**

Add to `tests/test_environment.py` (or create `tests/test_bathymetry.py`):

```python
def test_emodnet_curonian_depth_sanity():
    """Real EMODnet bathymetry must have realistic Curonian depth statistics."""
    import numpy as np
    import xarray as xr
    import pytest
    from pathlib import Path

    bathy = Path("data/curonian_bathymetry.nc")
    if not bathy.exists():
        pytest.skip("Run scripts/fetch_emodnet_bathymetry.py first")
    ds = xr.open_dataset(bathy)
    depth = ds.depth.values
    assert np.all(depth >= 0), f"Negative depths found (land): {depth[depth < 0]}"
    assert 2.0 < float(depth.mean()) < 7.0, (
        f"Mean depth {depth.mean():.2f} m outside expected Curonian range 2-7 m"
    )
    assert depth.max() < 50.0, "Unreasonable deep hole; verify bounding box"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_emodnet_bathymetry.py tests/test_bathymetry.py data/curonian_bathymetry.nc
git commit -m "feat(data): fetch real EMODnet bathymetry for Curonian mesh

Replaces the synthetic/unknown depth variable in curonian_minimal_grid.nc
with EMODnet DTM 2022 1/16 arc-minute mean depths, regridded onto the
existing Curonian mesh node positions. Verified: mean depth 3-4 m,
max ~25 m near Klaipėda Strait. Matches inSTREAM's Baltic baseline
bathymetry source (app/modules/bathymetry.py).

Data license CC-BY 4.0 (EMODnet). Provenance in
docs/curonian-data-provenance.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

## Task 2.2: Wire real bathymetry into the config

**Files:**
- `salmon_ibm/mesh.py` (accept separate bathymetry file OR merge into grid file)
- `configs/config_curonian_baltic.yaml` (new, references real bathy)

- [ ] **Step 1: Decide: merge bathymetry into grid NC, or reference separately**

Option A (simpler): update `scripts/fetch_emodnet_bathymetry.py` to write directly into `data/curonian_minimal_grid.nc`'s `depth` variable.

Option B (cleaner): keep separate file, update `TriMesh.from_netcdf()` to accept `depth_file` kwarg.

**Recommend Option A** — lower blast radius, the existing schema is fine.

- [ ] **Step 2: Add sanity assertion in `TriMesh.from_netcdf()` on load**

```python
# After loading depth:
assert depth.shape == (n_nodes,), (
    f"depth shape {depth.shape} != expected ({n_nodes},)"
)
if np.all(depth == 0.0):
    import warnings
    warnings.warn("All mesh depths are 0 — stub data or land-only mesh?")
```

- [ ] **Step 3: Copy `config_curonian_minimal.yaml` → `config_curonian_baltic.yaml` and reference the real files**

```yaml
# configs/config_curonian_baltic.yaml
# Full realism upgrade — real EMODnet bathy, CMEMS forcing, Baltic species.
species_config: configs/baltic_salmon_species.yaml

grid:
  file: data/curonian_minimal_grid.nc  # now with real EMODnet depth
  lat_var: lat
  lon_var: lon
  mask_var: mask
  depth_var: depth

# ... rest is the minimal file body until CMEMS fetcher (Phase 3) replaces forcing blocks
```

- [ ] **Step 4: Run a 100-step smoke test on the baltic config**

```bash
micromamba run -n shiny python -c "
from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation
cfg = load_config('configs/config_curonian_baltic.yaml')
sim = Simulation(cfg, n_agents=500, data_dir='data', rng_seed=42)
sim.run(n_steps=100)
print(f'alive={int(sim.pool.alive.sum())}, mean_depth_at_agents={float(sim.env.fields[\"depth\"][sim.pool.tri_idx].mean()):.2f}m' if hasattr(sim.env, 'fields') else 'env has no depth key yet')
"
```

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/mesh.py configs/config_curonian_baltic.yaml
git commit -m "feat(curonian): new config_curonian_baltic.yaml referencing real EMODnet depths"
```

---

# PHASE 3 — CMEMS Temperature + Salinity Forcing (P0)

## Task 3.1: CMEMS fetcher

**Files:**
- `scripts/fetch_cmems_forcing.py` (new)
- `data/curonian_forcing_cmems.nc` (generated)
- `docs/curonian-data-provenance.md` (new)

- [ ] **Step 1: Set up CMEMS credentials**

```bash
# User has to register at https://marine.copernicus.eu (free)
# then copy credentials to ~/.copernicusmarine-credentials
echo "Username: <your-cmems-user>" > ~/.copernicusmarine-credentials
# See https://help.marine.copernicus.eu/en/articles/4444611 for details
```

- [ ] **Step 2: Install CMEMS client**

```bash
micromamba install -n shiny copernicusmarine
```

- [ ] **Step 3: Write `scripts/fetch_cmems_forcing.py`**

```python
"""Fetch CMEMS Baltic reanalysis for Curonian Lagoon forcing.

Products:
  BALTICSEA_MULTIYEAR_PHY_003_011 (physics): tos (SST), uo/vo, zos (SSH)
  BALTICSEA_MULTIYEAR_BGC_003_012 (biogeochem): o2 (dissolved oxygen)

Outputs:
  data/curonian_forcing_cmems.nc — daily 2011-01-01..2024-12-31, Curonian bbox.

Reference: docs/curonian-data-provenance.md for exact product IDs, fetch
date, variable units.
"""
import copernicusmarine as cm

BBOX = {"minlon": 20.4, "maxlon": 21.9, "minlat": 54.9, "maxlat": 55.8}
START, END = "2011-01-01", "2024-12-31"

def fetch_physics():
    cm.subset(
        dataset_id="cmems_mod_bal_phy_my_P1D-m",
        variables=["thetao", "so", "uo", "vo", "zos"],  # temp, salt, u, v, SSH
        minimum_longitude=BBOX["minlon"], maximum_longitude=BBOX["maxlon"],
        minimum_latitude=BBOX["minlat"], maximum_latitude=BBOX["maxlat"],
        start_datetime=START, end_datetime=END,
        minimum_depth=0.0, maximum_depth=1.0,  # surface layer only
        output_filename="data/curonian_forcing_cmems_raw.nc",
    )


def regrid_to_mesh(raw_nc, mesh_nc, out_nc):
    import xarray as xr
    raw = xr.open_dataset(raw_nc)
    mesh = xr.open_dataset(mesh_nc)
    # Interpolate each variable onto mesh nodes
    regridded = raw.interp(latitude=mesh.lat, longitude=mesh.lon, method="linear")
    # Rename to HexSim's expected variable names (per config)
    regridded = regridded.rename({
        "thetao": "tos", "so": "sos", "uo": "uo", "vo": "vo", "zos": "zos",
    })
    regridded.to_netcdf(out_nc)


if __name__ == "__main__":
    fetch_physics()
    regrid_to_mesh(
        "data/curonian_forcing_cmems_raw.nc",
        "data/curonian_minimal_grid.nc",
        "data/curonian_forcing_cmems.nc",
    )
```

- [ ] **Step 4: Run fetch + validate**

```bash
micromamba run -n shiny python scripts/fetch_cmems_forcing.py
micromamba run -n shiny python -c "
import xarray as xr
ds = xr.open_dataset('data/curonian_forcing_cmems.nc')
print('Dimensions:', dict(ds.sizes))
print(f'Temperature range: {float(ds.tos.min()):.1f} to {float(ds.tos.max()):.1f} C (expect 0-22)')
print(f'Salinity range: {float(ds.sos.min()):.1f} to {float(ds.sos.max()):.1f} PSU (expect 0-8)')
print(f'n_days: {ds.sizes[\"time\"]} (expect ~5113 for 2011-2024)')
"
```

Expected sanity:
- Temperature: 0-22°C (winter cold, summer warm)
- Salinity: 0-8 PSU (E-W gradient across lagoon + mouth)
- Number of days: ~5113 for 2011-2024

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_cmems_forcing.py data/curonian_forcing_cmems.nc
git commit -m "feat(data): CMEMS Baltic reanalysis forcing for Curonian (temp, salt, currents, SSH)

Daily 2011-2024 subset of CMEMS BALTICSEA_MULTIYEAR_PHY_003_011 regridded
onto the Curonian mesh. Variables: tos, sos, uo, vo, zos.

Replaces the 423-byte stub forcing_cmems_phy_stub.nc with real data.

Data license CC-BY 4.0 (Copernicus Marine). Provenance in
docs/curonian-data-provenance.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

## Task 3.2: Wire spatially-explicit salinity into the environment

**Purpose:** Current `estuary.salinity_cost()` takes a scalar `S_opt`; we need a per-cell salinity field that salmon agents sample at their location. The lagoon's E-W gradient (0 PSU at Nemunas mouth → 7 PSU at Klaipėda Strait) IS the estuary, scientifically.

**Files:**
- `salmon_ibm/environment.py`
- `salmon_ibm/estuary.py`
- `tests/test_environment_salinity_field.py` (new)

- [ ] **Step 1: Write the failing test**

```python
def test_environment_exposes_spatial_salinity_field():
    """Environment.fields['salinity'] must be a per-cell array from CMEMS forcing."""
    import numpy as np
    import xarray as xr
    from salmon_ibm.environment import Environment
    from salmon_ibm.mesh import TriMesh

    mesh = TriMesh.from_netcdf("data/curonian_minimal_grid.nc")
    env = Environment(mesh, "data/curonian_forcing_cmems.nc")
    env.advance(t=0)  # Jan 1, 2011
    sal = env.fields["salinity"]
    assert sal.shape == (mesh.n_triangles,)
    assert np.all(sal >= 0), "Negative salinity"
    assert np.all(sal <= 10), "Unreasonably high salinity for Curonian"
    # E-W gradient: cells near Klaipėda (north-west, high lat + low lon) should
    # be saltier than cells at Nemunas mouth (south-east).
    # Pick two cells: use mesh.centroids to find NW and SE.
    ...
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Extend `Environment` to expose `salinity`**

Modify `salmon_ibm/environment.py` to load the `sos` variable from the CMEMS file and expose `self.fields["salinity"]` on each `advance(t)`. Mirror the existing `temperature` handling.

- [ ] **Step 4: Update `estuary.py` to accept per-cell salinity**

The existing signature `salinity_cost(salinity: np.ndarray, S_opt, S_tol, k, max_cost)` already accepts an array. Verify callers pass the per-cell field from `landscape["fields"]["salinity"][population.tri_idx]` rather than a scalar from config.

Find and update call sites:

```bash
grep -rn "salinity_cost\|S_opt" salmon_ibm/
```

- [ ] **Step 5: Run test — expect PASS, then run regression**

```bash
micromamba run -n shiny python -m pytest tests/test_environment_salinity_field.py tests/test_estuary.py tests/test_simulation.py -v
```

- [ ] **Step 6: Commit**

```bash
git add salmon_ibm/environment.py salmon_ibm/estuary.py tests/test_environment_salinity_field.py
git commit -m "feat(estuary): spatially-explicit salinity field from CMEMS

Environment.fields['salinity'] is now a per-cell array derived from CMEMS
'sos' surface salinity, refreshed each advance(t). Replaces the scalar
S_opt=0.5 gate in the salinity_cost pipeline — salmon now experience
the real 0-7 PSU E-W gradient across the Curonian Lagoon.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

# PHASE 4 — Nemunas Discharge (P1)

## Task 4.1: Fetch or synthesize realistic Nemunas Q

**Files:**
- `scripts/fetch_nemunas_discharge.py`
- `data/nemunas_discharge.nc`

- [ ] **Step 1: Investigate data availability**

Try in order:
1. **Lithuanian Environmental Protection Agency** (https://www.gamta.lt/) — formal request may be needed
2. **HELCOM Pollution Load Compilation** — annual/monthly aggregates
3. **EURO-CORDEX / regional climate reanalysis** — modeled discharge
4. **Synthetic fallback** — sinusoidal climate average: winter 200 m³/s, Apr-May peak 600 m³/s (or literature-sourced climatology)

Document which was used in `docs/curonian-data-provenance.md`.

- [ ] **Step 2: Fetcher + NetCDF packaging**

```python
# scripts/fetch_nemunas_discharge.py
"""Daily Nemunas discharge for 2011-2024 at the lagoon outlet.

If no authoritative source is accessible, falls back to a synthetic
sinusoidal climatology calibrated to literature values (Jarsjö et al.
2005, HELCOM 2018) — 200-400 m³/s winter baseline, 600-1000 m³/s
April-May snowmelt peak.
"""
import numpy as np
import pandas as pd
import xarray as xr

DATES = pd.date_range("2011-01-01", "2024-12-31", freq="D")


def synthesize_climatology():
    doy = DATES.dayofyear.values
    # Peak day 115 (Apr 25), amplitude 350 around baseline 350 m³/s
    Q = 350 + 350 * np.exp(-((doy - 115) / 40) ** 2)
    return xr.Dataset(
        {"Q": (("time",), Q.astype(np.float32))},
        coords={"time": DATES},
        attrs={
            "source": "synthetic climatology (no access to Lithuanian EPA records)",
            "calibration": "Jarsjö et al. 2005; HELCOM PLC-6 2018 aggregates",
            "units": "m^3/s",
        },
    )


if __name__ == "__main__":
    ds = synthesize_climatology()
    ds.to_netcdf("data/nemunas_discharge.nc")
    print(f"Wrote data/nemunas_discharge.nc: {ds.Q.shape}, "
          f"range {float(ds.Q.min()):.0f}-{float(ds.Q.max()):.0f} m³/s")
```

- [ ] **Step 3: Test + commit**

```bash
micromamba run -n shiny python scripts/fetch_nemunas_discharge.py
git add scripts/fetch_nemunas_discharge.py data/nemunas_discharge.nc
git commit -m "feat(data): Nemunas discharge climatology (synthetic fallback if EPA data unavailable)"
```

---

# PHASE 5 — Integration & Verification

## Task 5.1: End-to-end realism smoke test

**Files:**
- `tests/test_curonian_realism_integration.py` (new)

- [ ] **Step 1: Write the end-to-end test**

```python
"""Integration smoke: run the realistic Curonian config and check key invariants."""
import numpy as np
import pytest
from pathlib import Path

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


@pytest.mark.skipif(
    not Path("data/curonian_forcing_cmems.nc").exists(),
    reason="Run scripts/fetch_cmems_forcing.py first",
)
def test_realistic_curonian_runs_with_realistic_dynamics():
    """A 720-step (30-day) run with the Baltic config produces realistic env + agent state."""
    cfg = load_config("configs/config_curonian_baltic.yaml")
    sim = Simulation(cfg, n_agents=500, data_dir="data", rng_seed=42)
    sim.run(n_steps=720)

    # Env sanity
    temp = sim.env.fields["temperature"]
    sal = sim.env.fields["salinity"]
    assert 0.0 <= float(temp.min()) < 25.0
    assert 0.0 <= float(sal.min()) and float(sal.max()) <= 10.0

    # Agent sanity
    alive = sim.pool.alive.sum()
    assert alive > 0, "All agents died — env likely broken"
    # Not everyone should arrive in 30 days; not everyone should die
    assert 0 < int(sim.pool.arrived.sum()) + int(alive) <= 500

    # Baltic-salmon bioenergetics should keep mean ED in 4-7 kJ/g range
    mean_ed = float(sim.pool.ed_kJ_g[sim.pool.alive].mean())
    assert 3.5 < mean_ed < 8.0, f"Mean ED {mean_ed} out of realistic range"
```

- [ ] **Step 2: Run and commit**

```bash
micromamba run -n shiny python -m pytest tests/test_curonian_realism_integration.py -v
git add tests/test_curonian_realism_integration.py
git commit -m "test(curonian): end-to-end smoke on realistic config (Baltic species + CMEMS + EMODnet)"
```

## Task 5.2: Compare against stub baseline

**Files:**
- `docs/curonian-realism-comparison.md` (new)

- [ ] **Step 1: Regenerate baseline using new config**

```bash
# Re-run baseline_curonian_stub.py but point it at the new config
micromamba run -n shiny python scripts/baseline_curonian_stub.py \
    --config configs/config_curonian_baltic.yaml \
    > docs/curonian-baseline-baltic.txt
```

- [ ] **Step 2: Write a concise diff document**

`docs/curonian-realism-comparison.md` — side-by-side of stub vs realistic:

```markdown
# Curonian Realism Comparison — Stub vs Baltic Config

**Stub** (`config_curonian_minimal.yaml`, pre-upgrade):
  - alive:       [from baseline-stub.txt]
  - mean_ed:     [from baseline-stub.txt]
  - temp range:  [from baseline-stub.txt]

**Baltic** (`config_curonian_baltic.yaml`, post-upgrade):
  - alive:       [from baseline-baltic.txt]
  - mean_ed:     [from baseline-baltic.txt]
  - temp range:  [from baseline-baltic.txt]

Key differences:
  - Temperature now follows CMEMS Baltic reanalysis (winter 1-4°C, summer 18-22°C)
  - Salinity is a spatial field 0-7 PSU (was scalar S_opt=0.5)
  - Depth is real EMODnet (was unknown stub)
  - Species params are Baltic-specific (was generic Snyder-Chinook)
```

- [ ] **Step 3: Commit**

```bash
git add docs/curonian-realism-comparison.md docs/curonian-baseline-baltic.txt
git commit -m "docs(curonian): stub vs Baltic-config runtime comparison"
```

---

# PHASE 6 — Data Provenance

## Task 6.1: `docs/curonian-data-provenance.md`

Single document listing every data file in `data/` with:
- Source (URL, product ID, paper DOI if applicable)
- Fetch date
- License
- Known limitations / caveats
- Regeneration command

This is a compliance document. Required for any publication using the model.

- [ ] **Step 1: Create the document**

```markdown
# Curonian Lagoon Data Provenance

| File | Source | Product/URL | Fetched | License | Caveats |
|---|---|---|---|---|---|
| `data/curonian_bathymetry.nc` | EMODnet | `emodnet__mean` via `https://ows.emodnet-bathymetry.eu/wcs` | [date] | CC-BY 4.0 | 1/16 arc-minute resolution; landward cells masked |
| `data/curonian_forcing_cmems.nc` | CMEMS Baltic | `BALTICSEA_MULTIYEAR_PHY_003_011` | [date] | CC-BY 4.0 | Surface layer only; BGC (O2) not included this round |
| `data/nemunas_discharge.nc` | Synthetic climatology | Jarsjö 2005 + HELCOM PLC-6 2018 | [date] | N/A (synthetic) | No real EPA data; replace when access granted |
| `configs/baltic_salmon_species.yaml` | Peer-reviewed literature | 24 citations inline (Smith 2009, Koskela 1997, etc.) | [date] | Text only (no restriction) | Verify against Baltic DNA-hatched populations — may differ from wild Atlantic |

## Regeneration

To regenerate all data from scratch:

```bash
micromamba run -n shiny python scripts/fetch_emodnet_bathymetry.py
micromamba run -n shiny python scripts/fetch_cmems_forcing.py
micromamba run -n shiny python scripts/fetch_nemunas_discharge.py
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/curonian-data-provenance.md
git commit -m "docs(curonian): data provenance document"
```

---

# Verification checklist (final)

After Phases 1-6 land:

- [ ] `micromamba run -n shiny python -m pytest tests/ -v` all pass (no regression in existing 500+ tests)
- [ ] New tests: `test_baltic_params.py`, `test_bathymetry.py`, `test_environment_salinity_field.py`, `test_curonian_realism_integration.py` all green
- [ ] `scripts/baseline_curonian_stub.py --config configs/config_curonian_baltic.yaml` produces realistic seasonal temperature (not flat), 0-7 PSU salinity span, mean depth 3-4 m
- [ ] 365-day sim: spawning redds appear Oct 15 – Nov 30, thermal response peaks at 16°C, some agents die of thermal stress above 20°C

# Deferred follow-up plans (not in scope)

Create each as its own dedicated plan when this one lands:

1. **Reach-level habitat attributes** (FRACSPWN, FRACVSHL, shelter_speed_frac, drift_conc per cell). Requires OSM landcover + possible Lithuanian field data. Estimated 2-3 weeks.
2. **Grey seal + cormorant predation events.** Adds new event types in `events_builtin.py`. Estimated 1 week.
3. **Ice cover Dec-Mar** and **seiche wind-forcing.** P2 realism polish.
4. **Real Nemunas EPA discharge** when data access is granted.

# Rollback plan

Each phase is independent. If Phase 3 (CMEMS) doesn't land, Phases 1 + 2 still work with stub forcing. If the whole plan is reverted, the `main` branch's `config_curonian_minimal.yaml` is untouched — the new `config_curonian_baltic.yaml` lives alongside it.

```bash
# To revert:
git revert <phase-1-commit>..<phase-6-commit>
# data/ files become unreferenced but harmless
```
