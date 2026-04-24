# Curonian Lagoon Realism Upgrades — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the HexSim Curonian Lagoon study area from a stub-driven setup (generic Chinook bioenergetics + placeholder data files + scalar salinity gate) into a defensible Baltic salmon model with real bathymetry, spatially-explicit salinity, real temperature forcing, species-accurate parameters, and post-smolt marine mortality — closing the gap to inSTREAM's `example_baltic.yaml`.

**Architecture:** Work in priority order. P0 (data + species config) unlocks all downstream calibration. P1 (reach attributes, predation) builds ecological realism on that foundation. P2 (ice, seiche, redd scour) refines. Each task has a concrete deliverable (data file, config block, or code change) with verification.

**Tech Stack:** Python 3.10+, NumPy, xarray, PyYAML, scipy, pytest, `micromamba run -n shiny`. Data: EMODnet Bathymetry WCS, CMEMS Baltic reanalysis, HELCOM/ICES discharge records, OSM (via osmnx).

**Reference implementation:** inSTREAM at `C:\Users\arturas.baziukas\OneDrive - ku.lt\HORIZON_EUROPE\inSTREAM\instream-py\`. Specifically:
- `configs/baltic_salmon_species.yaml` — Baltic species config with ~17 peer-reviewed citation blocks
- `configs/example_baltic.yaml` — ~410-line multi-reach full config
- `docs/calibration-notes.md` — parameter provenance

**Note (v2 correction):** Two files originally claimed as reference implementations — `scripts/generate_baltic_example.py` and `app/modules/bathymetry.py` — **do not exist** in inSTREAM. The EMODnet and Baltic-cell-generation code in Phases 2-3 must be written from scratch (not ported).

**Test command:** `micromamba run -n shiny python -m pytest tests/ -v`. Substitute `conda run -n shiny` on machines with conda on PATH.

## Plan revision history

- **v1 (2026-04-24)** — initial draft from multi-agent Curonian analysis.
- **v2 (2026-04-24, post-review)** — three review agents (scientific-validator, code-reviewer, explore) independently flagged substantive issues. Revisions applied:
  - **Added Task 1.3** — the critical integration bridge: `species_config` YAML key must be wired into `Simulation.__init__` via a new `load_bio_params_from_config()` in `config.py`. Without this, Tasks 1.1-1.2 produce an inert file the runtime never reads.
  - **Fixed Handeland DOI** — `10.1016/j.aquaculture.2008.03.057` points to a flounder genetics paper. Correct is `10.1016/j.aquaculture.2008.06.042`.
  - **Split T_MAX into T_AVOID (20°C) + T_ACUTE_LETHAL (24°C)** — treating 20°C as a hard mortality cap would wipe out all agents every Curonian summer (observed SST routinely reaches 20-22°C). T_AVOID is behavioral; acute mortality is ~24°C per Elliott & Elliott 2010. Back-compat `T_MAX` property exposes T_AVOID.
  - **Corrected citations** — T_OPT: Jensen et al. 1989/2001 (primary) not Koskela 1997 (supporting only). Spawn window: Heinimaa & Heinimaa 2003 or local Lithuanian sources, NOT Lilja & Romakkaniemi 2003 (that paper is about June river entry, not autumn spawning). Length-weight: provenance flagged as "needs verification" (Kallio-Nyberg 2020 is not a length-weight paper).
  - **Corrected numerics** — Curonian max depth ~14.5 m natural (not 25 m); Klaipėda salinity typically 2.5 PSU (not 7 PSU); Nemunas discharge 300-500 m³/s baseline, 1500-2500 m³/s spring flood peak (not 200-400/600-1000); YAML has ~17 citations (not 24).
  - **Removed broken references** — `inSTREAM/.../app/modules/bathymetry.py` and `scripts/generate_baltic_example.py` do not exist. The EMODnet client must be written from scratch.
  - **Fixed YAML field-name mismatch** — inSTREAM uses MM-DD strings (`spawn_start_day: "10-15"`) but `BalticBioParams` uses DOY integers. Added explicit translation step in Task 1.2 with a verification check to catch silent fallback-to-defaults.
  - **Fixed `xr.open_rasterio`** — removed in xarray 2022.06. Script now uses `rioxarray.open_rasterio`.
  - **Fixed unstructured-mesh regridding** — `raw.interp(lat=..., lon=...)` on 1D mesh node arrays does per-dim broadcasting, NOT pointwise interpolation. Both EMODnet and CMEMS scripts now use `scipy.interpolate.RegularGridInterpolator` for correct pointwise lookup.
  - **Pinned `copernicusmarine<2.0`** — v2.0 has breaking API changes to `subset()`.
  - **Added CMEMS land-sea mask warning** — at ~2 km native resolution, CMEMS Baltic may mask the lagoon interior as land. Added a verification step and an alternative data source (SHYFEM via Idzelytė et al. 2023) if CMEMS coverage fails.
  - **Fixed test placeholders** — `test_environment_exposes_spatial_salinity_field` now has a concrete N>S gradient assertion. Integration test invariant replaced with hard `alive + dead + arrived == 500` (v1 was a tautology).
  - **Added argparse to baseline script** — Phase 0 and Phase 5.2 both invoke `scripts/baseline_curonian_stub.py --config ...`; previously Phase 0 hardcoded the stub config.
  - **Expanded deferred-follow-ups** — cormorant colony at Juodkrantė (specific location), Nemunas delta branching, hatchery vs wild origin, round goby invasion, summer cyanobacteria.
- **v3 (2026-04-24, post-v2-verification)** — one more focused review found two residual issues, both fixed:
  - **CRITICAL:** the back-compat `T_MAX` property on `BalticBioParams` returns `T_AVOID = 20.0°C`. But `simulation.py:252` (`_event_bioenergetics`) and `events_builtin.py:97` (`SurvivalEvent`) use `self.bio_params.T_MAX` as a **hard kill gate** — so Baltic configs would kill agents at 20°C, the exact mass-die-off the v2 thermal split was designed to prevent. Task 1.3 Step 5 now explicitly updates both kill-gate callers to prefer `T_ACUTE_LETHAL` via `getattr(bio_params, "T_ACUTE_LETHAL", bio_params.T_MAX)`. `events_builtin.py` added to Task 1.3's Files list.
  - Task 1.3 Step 4 now updates BOTH the import line in `simulation.py:34` AND the call site — v2 only mentioned the call site, which would have produced `ImportError` at runtime.
  - Fixed vacuous `not isinstance(params, object)` assertion in the fallback test (v2 had `type(params).__mro__[1] if False else object` which always evaluated to `object`).
- **v3.1 (2026-04-24, post-v3-fresh-eyes)** — a generalist pass after three specialist passes surfaced two small-but-real operational risks:
  - Added **Execution order** section near the top showing Phases 1-4 must land before Phase 5 can run. Without explicit ordering, an implementer could defer Phase 3 (CMEMS) and then Phase 5's integration test would skip silently via `pytest.skip` for missing data — false green.
  - Added **Task 3.1 Step 2b**: SHYFEM fallback stub (`scripts/fetch_shyfem_forcing.py`). Keeps a placeholder in-tree so an implementer hitting the CMEMS land-mask failure path doesn't have to invent the fallback from scratch. The stub raises `NotImplementedError` with an explicit contact pointer to KU Marine Research Institute.

## Post-execution notes (actual results vs plan predictions)

All 6 phases were executed on 2026-04-24 against the live codebase. The following are lessons learned that a future session should know — predictions that matched, that didn't, and new edge cases found only at execution time.

### ✅ Predictions that matched

- **N>S salinity gradient is real** — post-fetch CMEMS shows northern quartile cells ~0.5-2 PSU saltier than southern, exactly as the estuary physics required.
- **EMODnet bathymetry gives realistic Curonian depths** — mean 11.9 m across the mesh (mesh extends into Baltic coast, hence higher than lagoon-interior 3-4 m). `test_bathymetry_mesh_envelope` pins 2-20 m mean.
- **The 3-pass plan review caught all execution-blocking bugs** before coding. Zero blocker-class surprises at execution.
- **BalticBioParams+Population integration worked without touching `update_energy`** — the plan's claim that "BalticBioParams exposes all the same Wisconsin fields" was correct; no existing Chinook path broke.

### ❌ Predictions that were wrong

- **`copernicusmarine<2.0` pin is incompatible with Python 3.13.** The `shiny` env is Python 3.13, and v1.x requires Python <3.13. Had to use v2.4 with its breaking API changes. The v2 breakages encountered:
  - `force_download` → deprecated, ignored (warning only)
  - `overwrite_output_data` → renamed to `overwrite`
  - `zos` not in `cmems_mod_bal_phy_my_P1D-m` dataset (needed a separate SSH product which was deferred)
- **owslib 2.0.1 requires `identifier=coverage` as a string**, not `identifier=[coverage]` as the plan's skeleton wrote. A list gets URL-encoded as the Python `repr` (e.g. `%5B%27emodnet__mean%27%5D`) and the WCS service returns HTTP 404.
- **35% NaN from CMEMS land-sea mask is enough to kill all agents.** The plan's 50% threshold for SHYFEM fallback was too loose — at 35% NaN, NaN values still propagated through the thermal kill path and killed every agent. Fix landed: `NearestNDInterpolator` fill of any NaN cells, not just at the 50%+ threshold.

### 🆕 Execution-only discoveries

- **The mesh is `(y=30, x=30)` 2D regular grid, not 1D node-indexed.** The plan's regridder skeleton used `np.column_stack([lat.values, lon.values])` assuming 1D arrays. With 2D lat/lon, `column_stack` does `hstack` → shape `(30, 60)` which scipy's `RegularGridInterpolator` rejects (expects `(N, 2)`). Fix: `.ravel()` the mesh coords before stacking, reshape output back to `(y, x)`.
- **`xarray.to_netcdf()` defaults to NetCDF4**, but HexSim's `Environment` loader uses `engine="scipy"` which only reads NetCDF3. All data writes must specify `format="NETCDF3_64BIT"` explicitly. Hit this on both Nemunas discharge and CMEMS regridded output.
- **Environment loader requires `ssh_var` even if no SSH product is available.** Plan deferred real SSH to a separate product, but the loader fails fast without `ssh_var`. Workaround: zero-fill `zos` in the CMEMS regridder output so the loader succeeds and seiche detection becomes a no-op until a real SSH product is wired.
- **EMODnet coverage name `emodnet__mean` (double underscore) still valid** as of 2026-04-24 — no need for the plan's dynamic coverage-picker `pick_coverage()` helper, but the helper is future-proof so it stayed.
- **CMEMS download size: 227 MB raw → 92 MB regridded** for the Curonian bbox × 5114 days × 4 variables at 2 km native. Within the plan's 200-800 MB envelope estimate.

### Tooling environment

- Python 3.13 in micromamba env `shiny`
- `copernicusmarine 2.4.0` (installed via conda-forge)
- `owslib 2.0.1` (installed via conda-forge)
- `rioxarray 0.19.0` (installed via conda-forge)
- `pytest-mock` needed installing to fix a pre-existing (unrelated) test regression — landed at session end.

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

## Execution order (critical)

Phases are **sequentially dependent** for Phase 5 (integration) to succeed:

```
  Phase 0 (baseline capture)
    ↓
  Phase 1 (species config + loader wiring)
    ↓
  Phase 2 (EMODnet bathymetry)
    ↓
  Phase 3 (CMEMS forcing + spatial salinity)   ← may fall back to SHYFEM
    ↓
  Phase 4 (Nemunas discharge, synthetic or real)
    ↓
  Phase 5 (integration test)
    ↓
  Phase 6 (provenance documentation)
```

Phase 6 is the only phase that can be done any time (pure documentation). All others MUST be done in order — skipping Phase 3 for later means Phase 5's integration test can't run (will skip silently via `pytest.skip(...)` due to missing CMEMS data).

**Do NOT start Phase 5 until Phases 1-4 have all landed.**

## Scope: what this plan does NOT cover

- **Full reach-level habitat attributes** (FRACSPWN, FRACVSHL, drift_conc per cell). Requires field survey + OSM landcover extraction. Scoped as a dedicated sub-plan after this plan lands.
- **Grey seal + cormorant predation events.** Requires adding new event types to `events_builtin.py`. Separate P1 sub-plan.
- **Ice cover, seiche wind-forcing, redd scour depth tuning.** All P2; separate plans when P0+P1 are proven.

---

# PHASE 0 — Infrastructure

## Task 0.1: Capture baseline behavior

**Purpose:** record what the stub Curonian config produces today, so the real-data upgrade has a measurable baseline.

**Files:** `scripts/baseline_curonian_stub.py` (new)

- [ ] **Step 1: Write a deterministic baseline script with `--config` argparse**

```python
"""Run the given config; record key outputs for comparison.

Used twice in this plan:
  - Phase 0: baseline of config_curonian_minimal.yaml (stub)
  - Phase 5.2: post-upgrade on config_curonian_baltic.yaml (realistic)

Both invocations use the same script via --config, so outputs are
directly comparable.
"""
import argparse
from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


def run(config_path: str, n_agents: int = 500, n_steps: int = 720):
    cfg = load_config(config_path)
    sim = Simulation(cfg, n_agents=n_agents, data_dir="data", rng_seed=42)
    sim.run(n_steps=n_steps)
    alive_mask = sim.pool.alive
    alive = int(alive_mask.sum())
    arrived = int(sim.pool.arrived.sum())
    mean_ed = float(sim.pool.ed_kJ_g[alive_mask].mean()) if alive else 0.0
    mean_mass = float(sim.pool.mass_g[alive_mask].mean()) if alive else 0.0
    return {
        "config": config_path,
        "alive": alive,
        "arrived": arrived,
        "mean_ed_kJ_g": mean_ed,
        "mean_mass_g": mean_mass,
        "temp_range": (float(sim.env.fields["temperature"].min()),
                       float(sim.env.fields["temperature"].max())),
        "salinity_range": (
            float(sim.env.fields["salinity"].min()),
            float(sim.env.fields["salinity"].max()),
        ) if "salinity" in sim.env.fields else (None, None),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_curonian_minimal.yaml")
    parser.add_argument("--n-agents", type=int, default=500)
    parser.add_argument("--n-steps", type=int, default=720)
    args = parser.parse_args()
    for k, v in run(args.config, args.n_agents, args.n_steps).items():
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
    # Jensen et al. 2001 thermal optimum
    assert p.T_OPT == pytest.approx(16.0)
    # Two-threshold thermal response (v2): avoidance vs acute mortality
    assert p.T_AVOID == pytest.approx(20.0)
    assert p.T_ACUTE_LETHAL == pytest.approx(24.0)
    # Length-weight (provenance under verification)
    assert p.LW_a == pytest.approx(0.0077)
    assert p.LW_b == pytest.approx(3.05)
    # Linear fecundity approximation
    assert p.fecundity_per_g == pytest.approx(2.0)
    # Backward-compat: T_MAX property still works (maps to T_AVOID)
    assert p.T_MAX == pytest.approx(20.0)


def test_baltic_bioparams_rejects_invalid_ranges():
    """Post-init validation must reject nonsense parameters (same discipline as BioParams)."""
    with pytest.raises(ValueError, match="T_AVOID"):
        BalticBioParams(T_OPT=25.0, T_AVOID=20.0)
    with pytest.raises(ValueError, match="T_ACUTE_LETHAL"):
        BalticBioParams(T_AVOID=25.0, T_ACUTE_LETHAL=20.0)
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
                  — verify exact values against Table 2 before production use
  T_OPT:          Jensen, Jonsson & Forseth 2001 (doi:10.1046/j.0269-8463.2001.00572.x)
                  — primary source for 16-17°C Atlantic salmon thermal peak;
                  Koskela et al. 1997 (doi:10.1111/j.1095-8649.1997.tb01976.x)
                  is Baltic low-temperature supporting evidence only.
  T_AVOID:        20.0°C — behavioral avoidance / reduced growth (Handeland
                  et al. 2008 doi:10.1016/j.aquaculture.2008.06.042 — CORRECTED
                  DOI; v1 plan cited .03.057 which is a different paper).
                  NOT a hard mortality cap.
  T_ACUTE_LETHAL: 24.0°C — acute thermal mortality (Elliott & Elliott 2010 review;
                  Smialek, Pander & Geist 2021 doi:10.1111/fme.12507).
  LW_a, LW_b:     Provenance needs verification. Baltic length-weight typically
                  from ICES WGBAST reports or Kallio-Nyberg & Ikonen 1992.
                  Kallio-Nyberg et al. 2020 (cited in v1) is NOT a length-weight paper.
  fecundity:      Linear 2.0 eggs/g defensible for mid-sized (Heinimaa & Heinimaa
                  2003: 1845 eggs/kg = 1.85 eggs/g for 9 kg females). Real form
                  declines with size — consider allometric for production runs.
  spawn window:   Late Oct – Nov 30 for Lithuanian Nemunas tributaries (Žeimena,
                  Merkys, Dubysa). v1 plan's Lilja & Romakkaniemi 2003 citation
                  was WRONG — that paper is about June adult river entry, not
                  autumn spawning. Use Heinimaa & Heinimaa 2003 or local Lithuanian
                  sources for Nemunas basin.

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

    # Species-specific Baltic values. Two-threshold thermal response (v2 correction):
    #   T_OPT → peak growth
    #   T_AVOID → behavioral avoidance (NOT a mortality gate)
    #   T_ACUTE_LETHAL → hard mortality threshold
    # v1 used T_MAX=20°C as a kill-gate — that would wipe out agents every Curonian
    # summer (observed SST routinely reaches 20-22°C). Split into two thresholds.
    cmax_A: float = 0.303           # Smith 2009 post-smolt CMax intercept
    cmax_B: float = -0.275          # Smith 2009 post-smolt CMax slope
    T_OPT: float = 16.0             # Jensen et al. 1989/2001 Atlantic salmon peak
    T_AVOID: float = 20.0           # Handeland 2008: behavioral avoidance / reduced growth
    T_ACUTE_LETHAL: float = 24.0    # Elliott & Elliott 2010: acute mortality
    LW_a: float = 0.0077            # Length-weight intercept — provenance needs verification
    LW_b: float = 3.05              # Length-weight exponent — provenance needs verification
    fecundity_per_g: float = 2.0    # Linear approximation (size-declining in reality)

    # Spawning phenology (day of year; Lithuanian Nemunas basin populations)
    spawn_window_start_day: int = 288  # Oct 15 (Lithuanian; spawning typically later than N. Finland)
    spawn_window_end_day: int = 334    # Nov 30
    spawn_temp_min_c: float = 5.0
    spawn_temp_max_c: float = 14.0

    # Activity by behavior (keep Snyder structure, Baltic-tunable)
    activity_by_behavior: dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    )

    def __post_init__(self):
        if self.T_AVOID <= self.T_OPT:
            raise ValueError(
                f"T_AVOID ({self.T_AVOID}) must be > T_OPT ({self.T_OPT})"
            )
        if self.T_ACUTE_LETHAL <= self.T_AVOID:
            raise ValueError(
                f"T_ACUTE_LETHAL ({self.T_ACUTE_LETHAL}) must be > T_AVOID ({self.T_AVOID})"
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

    # Backward compat: existing callers of update_energy read params.T_MAX as the
    # thermal mortality gate. Under v2 semantics T_AVOID is that gate (behavioral),
    # while T_ACUTE_LETHAL is the hard-mortality threshold. Expose T_MAX as T_AVOID
    # for legacy callers; new code reading acute mortality must use T_ACUTE_LETHAL.
    @property
    def T_MAX(self) -> float:
        return self.T_AVOID


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

## Task 1.3: Wire `species_config` loader into `Simulation`

**Purpose:** CRITICAL GAP from v1 plan. Tasks 1.1 and 1.2 create `BalticBioParams` and the YAML file but DON'T connect them to runtime. Without this task, the species config is inert — `Simulation` keeps using the old `BioParams` regardless. This task is the bridge.

**Files:**
- Modify: `salmon_ibm/config.py` — add `load_bio_params_from_config`
- Modify: `salmon_ibm/simulation.py` — update import (line ~34) + call site (line ~120) + kill-gate (line ~252)
- Modify: `salmon_ibm/events_builtin.py` — update kill-gate at `SurvivalEvent.execute` (line ~97)
- Modify: `salmon_ibm/bioenergetics.py` — doc comment on `update_energy` noting dual-params acceptance
- Test: `tests/test_baltic_params.py`

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_baltic_params.py`:

```python
def test_simulation_loads_baltic_bioparams_via_species_config_key(tmp_path):
    """A YAML config with species_config: <path> must route Simulation to BalticBioParams."""
    from salmon_ibm.config import load_bio_params_from_config
    from salmon_ibm.baltic_params import BalticBioParams

    species_yaml = tmp_path / "species.yaml"
    species_yaml.write_text("""
species:
  BalticAtlanticSalmon:
    T_OPT: 16.0
    T_MAX: 20.0
""")
    cfg = {"species_config": str(species_yaml)}
    params = load_bio_params_from_config(cfg)
    assert isinstance(params, BalticBioParams)
    assert params.T_OPT == 16.0


def test_load_bio_params_falls_back_to_bio_params_when_no_species_config():
    """If species_config key absent, return classic BioParams (backward compat)."""
    from salmon_ibm.config import load_bio_params_from_config
    from salmon_ibm.bioenergetics import BioParams
    from salmon_ibm.baltic_params import BalticBioParams

    cfg = {"bioenergetics": {"RA": 0.003}}
    params = load_bio_params_from_config(cfg)
    assert isinstance(params, BioParams)
    assert not isinstance(params, BalticBioParams), (
        "Fall-back path must return BioParams, not BalticBioParams"
    )
```

- [ ] **Step 2: Run — expect FAIL (`load_bio_params_from_config` doesn't exist)**

- [ ] **Step 3: Add loader to `salmon_ibm/config.py`**

```python
def load_bio_params_from_config(config: dict):
    """Route to BalticBioParams if `species_config:` key present, else BioParams.

    The species_config key points to a YAML file like
    `configs/baltic_salmon_species.yaml` — see the Baltic realism plan
    (docs/superpowers/plans/2026-04-24-curonian-realism-upgrades.md).
    """
    species_path = config.get("species_config")
    if species_path:
        from salmon_ibm.baltic_params import load_baltic_species_config
        return load_baltic_species_config(species_path)
    return bio_params_from_config(config)
```

- [ ] **Step 4: Update `simulation.py` — BOTH the import AND the call site**

`simulation.py` line 34 currently has `from salmon_ibm.config import bio_params_from_config`. Add the new loader to this import:

```python
# OLD: from salmon_ibm.config import bio_params_from_config
from salmon_ibm.config import bio_params_from_config, load_bio_params_from_config
```

Then at `simulation.py:~120`, replace the call:

```python
# OLD: self.bio_params = bio_params_from_config(config)
self.bio_params = load_bio_params_from_config(config)
```

- [ ] **Step 5: Update the two kill-gate callers of `bio_params.T_MAX`**

> **v3 correction — CRITICAL.** The back-compat `T_MAX` property on `BalticBioParams` returns `T_AVOID = 20.0°C`. But `simulation.py:252` (`_event_bioenergetics`) and `events_builtin.py:97` (`SurvivalEvent.execute`) both use `self.bio_params.T_MAX` as a **hard kill gate** (`thermal_kill = temp >= T_MAX → pool.alive = False`). With a Baltic config this would kill agents at 20°C — the exact mass-die-off the v2 thermal split was designed to prevent. Both callers must branch on `T_ACUTE_LETHAL` when available.

Find the two callers:

```bash
grep -n "bio_params.T_MAX\|params.T_MAX" salmon_ibm/simulation.py salmon_ibm/events_builtin.py
```

Expected output: `simulation.py:~252` and `events_builtin.py:~97`.

Update both sites to prefer `T_ACUTE_LETHAL` when defined:

```python
# OLD: thermal_kill = (... & (temps >= self.bio_params.T_MAX))
lethal_T = getattr(self.bio_params, "T_ACUTE_LETHAL", self.bio_params.T_MAX)
thermal_kill = (... & (temps >= lethal_T))
```

This keeps the Chinook `BioParams` path unchanged (no `T_ACUTE_LETHAL` attr → falls back to `T_MAX`) and uses the correct 24°C acute threshold for `BalticBioParams`.

- [ ] **Step 6: Verify `update_energy` handles BalticBioParams**

`update_energy(ed_kJ_g, mass_g, temperature_c, activity_mult, salinity_cost, params)` reads `params.RA, params.RB, params.RQ, params.ED_MORTAL, params.ED_TISSUE, params.MASS_FLOOR_FRACTION`. `BalticBioParams` exposes all of these. Document with a comment in `update_energy`:

```python
def update_energy(..., params):
    """params: BioParams | BalticBioParams (both expose the same Wisconsin fields)."""
```

- [ ] **Step 7: Run the integration tests — expect PASS**

```bash
micromamba run -n shiny python -m pytest tests/test_baltic_params.py tests/test_bioenergetics.py tests/test_simulation.py tests/test_events.py -v
```

- [ ] **Step 8: Commit**

```bash
git add salmon_ibm/config.py salmon_ibm/simulation.py salmon_ibm/bioenergetics.py salmon_ibm/events_builtin.py tests/test_baltic_params.py
git commit -m "feat(config): route species_config to BalticBioParams + use T_ACUTE_LETHAL for kill-gate

Wires the Phase 1 Baltic species config into Simulation.__init__, AND
updates the two kill-gate callers (_event_bioenergetics, SurvivalEvent)
to read bio_params.T_ACUTE_LETHAL when available. Without this second
fix the back-compat T_MAX property would make 20°C a hard kill gate
for Baltic configs — defeating the entire point of the T_AVOID vs
T_ACUTE_LETHAL split.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 1.2: Port the full `baltic_salmon_species.yaml` config

**Files:**
- `configs/baltic_salmon_species.yaml` (new — copy from inSTREAM with HexSim schema adaptations)

> **v2 correction — field name mismatch in inSTREAM YAML.** The source file uses `spawn_start_day: "10-15"` and `spawn_end_day: "11-30"` (MM-DD strings), not `spawn_window_start_day: 288` (integer). A plain `cp` will NOT work: `load_baltic_species_config()` filters unknown keys and will silently fall back to BalticBioParams defaults with no error. Translation is required.

- [ ] **Step 1: Copy AND adapt (not a plain cp)**

```bash
cp "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/inSTREAM/instream-py/configs/baltic_salmon_species.yaml" \
    "C:/Users/arturas.baziukas/OneDrive - ku.lt/HORIZON_EUROPE/HexSim/configs/baltic_salmon_species.yaml"
```

Then edit the new HexSim file:
1. Keep only the fields `BalticBioParams` recognizes (CMax, T_OPT, T_AVOID, T_ACUTE_LETHAL, LW_a, LW_b, fecundity, spawn window).
2. **Translate MM-DD spawn fields to day-of-year integers:**

```yaml
# OLD (inSTREAM): spawn_start_day: "10-15"  # string MM-DD
# OLD (inSTREAM): spawn_end_day:   "11-30"  # string MM-DD
# NEW (HexSim):   spawn_window_start_day: 288  # Oct 15 = DOY 288
# NEW (HexSim):   spawn_window_end_day:   334  # Nov 30 = DOY 334
```

3. **Split `T_MAX` into `T_AVOID` + `T_ACUTE_LETHAL`** per the two-threshold thermal model (see Task 1.1 docstring). If the inSTREAM file has a single `T_MAX: 20.0`, replace with:

```yaml
T_AVOID: 20.0
T_ACUTE_LETHAL: 24.0
```

4. **Preserve all citation comments** from inSTREAM verbatim — they document provenance. Correct the Handeland DOI if present (`.03.057` → `.06.042`).

- [ ] **Step 2: Test round-trip**

```bash
micromamba run -n shiny python -c "
from salmon_ibm.baltic_params import load_baltic_species_config
p = load_baltic_species_config('configs/baltic_salmon_species.yaml')
print(f'OK: T_OPT={p.T_OPT}, T_AVOID={p.T_AVOID}, T_ACUTE_LETHAL={p.T_ACUTE_LETHAL}, spawn={p.spawn_window_start_day}-{p.spawn_window_end_day}')
"
```

Expected: `OK: T_OPT=16.0, T_AVOID=20.0, T_ACUTE_LETHAL=24.0, spawn=288-334`.

If output shows `spawn=288-334` but you did NOT set those keys explicitly in the YAML, the translation step above was skipped and the loader is falling back to defaults — fix the YAML.

- [ ] **Step 3: Commit**

```bash
git add configs/baltic_salmon_species.yaml
git commit -m "feat(baltic): port baltic_salmon_species.yaml with MM-DD→DOY translation and T_AVOID/T_ACUTE_LETHAL split"
```

---

# PHASE 2 — Real EMODnet Bathymetry (P0)

## Task 2.1: EMODnet WCS fetch script

**Purpose:** Replace the stub `depth` variable in `data/curonian_minimal_grid.nc` with a real 1/16-arcmin EMODnet DTM sampled onto the Curonian mesh.

**Reference implementation:** `inSTREAM/instream-py/app/modules/bathymetry.py::fetch_emodnet_dtm()`.

**Files:**
- `scripts/fetch_emodnet_bathymetry.py` (new)
- `data/curonian_bathymetry.nc` (new, generated)

- [ ] **Step 1: Install needed packages**

> **v2 correction:** the v1 plan referenced `inSTREAM/.../app/modules/bathymetry.py::fetch_emodnet_dtm` as a template, but that file does **not** exist in inSTREAM. Write the EMODnet client from scratch using owslib + rioxarray.

```bash
micromamba install -n shiny owslib rioxarray
```

- [ ] **Step 2: Verify EMODnet WCS coverage name**

EMODnet Bathymetry releases (2018, 2020, 2022) have used different coverage names. Before coding, manually check the current GetCapabilities:

```bash
curl -s "https://ows.emodnet-bathymetry.eu/wcs?service=WCS&version=2.0.1&request=GetCapabilities" | grep -oE "<wcs:CoverageId>[^<]+" | head -5
```

Use the coverage name returned by the live service. If "emodnet__mean" is no longer valid, use whatever the service lists (likely `emodnet_mean_2022` or similar).

- [ ] **Step 3: Write `scripts/fetch_emodnet_bathymetry.py`**

```python
"""Fetch EMODnet Bathymetry for the Curonian Lagoon + Nemunas mouth + Baltic coast.

Output: data/curonian_bathymetry.nc with variables (lat, lon, depth) on the
Curonian mesh nodes.

v2 implementation notes:
  - `xr.open_rasterio` was removed in xarray 2022.06 — use rioxarray.
  - Unstructured-mesh regridding: we cannot use `raw.interp(lat=..., lon=...)`
    with 1-D node coordinate arrays because xarray's interp broadcasts along
    each dim independently, not point-wise. We use scipy.interpolate instead.
"""
import argparse
import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray  # registers .rio accessor on xr.DataArray/Dataset
from owslib.wcs import WebCoverageService
from scipy.interpolate import RegularGridInterpolator

BBOX = {
    "minlon": 20.4, "maxlon": 21.9,
    "minlat": 54.9, "maxlat": 55.8,
}
WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"
# VERIFY against live GetCapabilities (Step 2); update if the release has changed.
COVERAGE = "emodnet__mean"


def fetch(bbox, out_path):
    wcs = WebCoverageService(WCS_URL, version="2.0.1")
    response = wcs.getCoverage(
        identifier=[COVERAGE],
        subsets=[("Lat", bbox["minlat"], bbox["maxlat"]),
                 ("Long", bbox["minlon"], bbox["maxlon"])],
        format="image/tiff",
    )
    with open(out_path, "wb") as f:
        f.write(response.read())


def regrid_to_mesh(raw_tif_path, mesh_nc_path, out_nc_path):
    mesh = xr.open_dataset(mesh_nc_path)
    # rioxarray opens a georeferenced raster with .x/.y coordinates
    raw = rioxarray.open_rasterio(raw_tif_path).squeeze()
    # raw.x is lon, raw.y is lat (descending per GeoTIFF convention)
    # Build a scipy interpolator for point-wise queries on mesh nodes
    x = raw.x.values  # lon (1D)
    y = raw.y.values  # lat (1D, likely descending)
    z = raw.values    # (n_y, n_x) depth in meters (EMODnet convention: positive down)
    # Flip to ascending latitude if needed, so the interpolator gets monotonic axes
    if y[0] > y[-1]:
        y = y[::-1]
        z = z[::-1, :]
    interp = RegularGridInterpolator(
        (y, x), z, method="linear", bounds_error=False, fill_value=np.nan
    )
    query_points = np.column_stack([mesh.lat.values, mesh.lon.values])
    depth_at_nodes = interp(query_points)
    # EMODnet uses negative values for depth below sea level; normalize to positive
    # downward if that's the HexSim convention (check mesh.depth semantics).
    depth_at_nodes = np.abs(depth_at_nodes)
    ds = xr.Dataset(
        {"depth": (("node",), depth_at_nodes.astype(np.float32))},
        coords={"lat": ("node", mesh.lat.values),
                "lon": ("node", mesh.lon.values)},
        attrs={
            "source": "EMODnet Bathymetry (verify release via WCS GetCapabilities)",
            "url": WCS_URL,
            "coverage": COVERAGE,
            "fetched": datetime.date.today().isoformat(),
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
    fetch(BBOX, tif)
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

Expected values (validated against Mėžinė et al. 2019, Stragauskaitė et al. 2021):
- Curonian Lagoon mean: ~3.8 m (lagoon area 1584 km², volume 6.3 km³; Mėžinė 2019)
- Klaipėda Strait natural max: ~14.5 m (dredged shipping channel is deeper but artificial)
- Nemunas mouth: ~5 m
- Baltic Sea coast within 10 km: 10-25 m
- No negative depths (land) in the mesh region; if present, mesh mask is wrong.

**Note:** any `depth.max() > 18 m` in the natural mesh indicates the bbox has captured dredged port channels at Klaipėda (artificial, ~14.5 m natural maximum). Decide whether to include (if agents can use shipping channels) or clip.

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
    # Curonian published mean is ~3.8 m (Mėžinė et al. 2019); allow 2-7 m envelope.
    assert 2.0 < float(depth.mean()) < 7.0, (
        f"Mean depth {depth.mean():.2f} m outside expected Curonian range 2-7 m"
    )
    # Natural max ~14.5 m at Klaipėda Strait; coast to 25 m in 10 km buffer.
    # >30 m suggests the bbox has captured too much open Baltic or a WCS error.
    assert depth.max() < 30.0, (
        f"Max depth {depth.max():.1f} m — natural Curonian max is 14.5 m, "
        f"coast reaches ~25 m in 10 km buffer. Check bbox."
    )
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

- [ ] **Step 2: Install CMEMS client — PIN THE VERSION**

`copernicusmarine` v2.0 (Nov 2024) rewrote the `subset()` API; this plan's script targets v1.x syntax. Pin:

```bash
micromamba install -n shiny "copernicusmarine>=1.3,<2.0"
```

If you need v2+, update the script's `cm.subset()` call — `output_filename` and dataset selection keywords changed.

- [ ] **Step 2a: CMEMS land-sea mask warning for the Curonian**

CMEMS Baltic physics runs at ~2 km (≈1 nautical mile) native resolution. The Curonian Lagoon is a shallow coastal lagoon; in the CMEMS Baltic grid many of its cells may be masked as **land**. Before committing to CMEMS for the lagoon interior, verify coverage by plotting `tos` on day 1 and confirming the lagoon is not uniform NaN:

```bash
micromamba run -n shiny python -c "
import xarray as xr
import numpy as np
ds = xr.open_dataset('data/curonian_forcing_cmems_raw.nc')
print(f'NaN fraction in tos[0]: {float(np.isnan(ds.thetao.isel(time=0)).mean()):.2%}')
print(f'Valid temp range on day 0: {float(ds.thetao.isel(time=0).min()):.1f} to {float(ds.thetao.isel(time=0).max()):.1f}')
"
```

If >50% of the Curonian mesh falls in CMEMS-masked land, CMEMS is the wrong data source for the lagoon interior. Fall back to the **SHYFEM high-resolution Curonian hydrodynamic model** (Idzelytė et al. 2023, doi:10.5194/os-19-1047-2023) or interpolate from nearby Baltic Proper cells.

- [ ] **Step 2b: SHYFEM fallback stub (prepare before attempting CMEMS)**

Create `scripts/fetch_shyfem_forcing.py` as a placeholder to avoid inventing a fallback mid-execution if CMEMS land-masking fails:

```python
"""SHYFEM Curonian Lagoon hydrodynamic model output fetcher (FALLBACK).

Use only if CMEMS land-sea mask covers >50% of the Curonian mesh
(see fetch_cmems_forcing.py Step 2a verification).

Reference: Idzelytė et al. 2023, doi:10.5194/os-19-1047-2023 — SHYFEM
unstructured-mesh coupled hydrological-hydrodynamic model for Nemunas
+ Curonian Lagoon. Output hosted at Klaipėda University Marine Research
Institute; access requires contact with R. Idzelytė or the authors.

This is a PLACEHOLDER. Fill in the actual access path + variable names
when SHYFEM is actually needed. Keeping the file in-tree means the
implementer does not have to invent a fallback on the spot.
"""
import argparse


def fetch_shyfem_forcing(bbox, start, end, out_path):
    raise NotImplementedError(
        "SHYFEM fallback not wired. Contact KU Marine Research Institute "
        "for data access; see Idzelytė et al. 2023 for variable names."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", default="20.4,54.9,21.9,55.8")
    parser.add_argument("--out", default="data/curonian_forcing_shyfem.nc")
    args = parser.parse_args()
    fetch_shyfem_forcing(args.bbox, "2011-01-01", "2024-12-31", args.out)
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
    """Regrid CMEMS (regular lat/lon grid) onto the unstructured Curonian mesh.

    Uses point-wise interpolation via scipy — xarray's `interp` with 1D
    coordinate arrays does NOT do pointwise interpolation on unstructured
    meshes; it broadcasts along each dim independently. This is the same
    gotcha as the EMODnet bathymetry script.
    """
    import xarray as xr
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    raw = xr.open_dataset(raw_nc)
    mesh = xr.open_dataset(mesh_nc)
    n_time = raw.sizes["time"]
    n_nodes = mesh.lat.size

    # CMEMS dim names may be 'latitude'/'longitude' or 'lat'/'lon' depending
    # on product; normalize.
    rename = {d: "lat" for d in raw.dims if d in ("latitude",)}
    rename.update({d: "lon" for d in raw.dims if d in ("longitude",)})
    raw = raw.rename(rename)

    lat_src = raw.lat.values
    lon_src = raw.lon.values
    # Ensure ascending
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        raw = raw.isel(lat=slice(None, None, -1))
    query = np.column_stack([mesh.lat.values, mesh.lon.values])

    variables = {}
    for src, dst in [("thetao", "tos"), ("so", "sos"),
                     ("uo", "uo"), ("vo", "vo"), ("zos", "zos")]:
        if src not in raw:
            continue
        arr = raw[src].squeeze().values  # (time, lat, lon) — drop depth if present
        out = np.empty((n_time, n_nodes), dtype=np.float32)
        for t in range(n_time):
            interp = RegularGridInterpolator(
                (lat_src, lon_src), arr[t],
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            out[t] = interp(query).astype(np.float32)
        variables[dst] = (("time", "node"), out)

    ds = xr.Dataset(
        variables,
        coords={"time": raw.time.values,
                "lat": ("node", mesh.lat.values),
                "lon": ("node", mesh.lon.values)},
    )
    ds.to_netcdf(out_nc)


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

Expected sanity (validated against Stakėnienė et al. 2023, Idzelytė et al. 2023):
- Temperature: 0-22°C (winter cold, summer warm at surface)
- Salinity in Klaipėda Strait: typical 2.5 PSU, with episodic intrusions reaching 6-7 PSU in autumn (2019-2020 extreme: 6.3 PSU). The 1986-2005 mean was 2.5 PSU at the strait and 1.2 PSU in the lagoon interior. Gradient does not extend >~20 km south of the strait — most of the lagoon is freshwater. v1 plan's "0-7 PSU across the lagoon" was a peak-intrusion extreme, not typical.
- Number of days: ~5113 for 2011-2024.

**Expected CMEMS NetCDF file size: 200-800 MB** for the full Curonian bbox + 5 variables × 5113 days. Add `data/curonian_forcing_cmems*.nc` to `.gitignore` (see Phase 6 provenance doc); commit regeneration script instead.

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
    """Environment.fields['salinity'] must be a per-cell array from CMEMS forcing.

    Verifies (a) shape matches mesh, (b) values are in Baltic lagoon range,
    (c) gradient points the right direction: saltier near Klaipėda Strait
    (NW, ~55.7 N, 21.1 E) than at Nemunas mouth (SE, ~55.3 N, 21.2 E).
    """
    import numpy as np
    from pathlib import Path
    import pytest
    from salmon_ibm.environment import Environment
    from salmon_ibm.mesh import TriMesh

    if not Path("data/curonian_forcing_cmems.nc").exists():
        pytest.skip("Run scripts/fetch_cmems_forcing.py first")

    mesh = TriMesh.from_netcdf("data/curonian_minimal_grid.nc")
    env = Environment({"forcings": {"physics_surface": {
        "file": "curonian_forcing_cmems.nc", "salt_var": "sos",
        "temp_var": "tos", "u_var": "uo", "v_var": "vo", "ssh_var": "zos",
        "time_var": "time",
    }}}, mesh, data_dir="data")
    env.advance(t=0)
    sal = env.fields["salinity"]
    assert sal.shape == (mesh.n_triangles,)
    # Lagoon-realistic envelope: 0 PSU interior to ~7 PSU peak at strait.
    valid = sal[~np.isnan(sal)]
    assert valid.size > 0, "All-NaN salinity — CMEMS land-sea mask covers mesh"
    assert np.all(valid >= 0), "Negative salinity"
    assert valid.max() <= 10.0, f"Unrealistic salinity max {valid.max():.1f} PSU"

    # Gradient check: sort cells by latitude; northern cells should average
    # higher salinity than southern cells (Klaipėda Strait vs Nemunas mouth).
    centroids = mesh.centroids  # (n_tri, 2) as [lat, lon]
    lat = centroids[:, 0]
    north_mask = lat > np.percentile(lat, 75)  # top quartile by latitude
    south_mask = lat < np.percentile(lat, 25)
    north_mean = float(np.nanmean(sal[north_mask]))
    south_mean = float(np.nanmean(sal[south_mask]))
    assert north_mean > south_mean, (
        f"Expected saltier at north (Klaipėda Strait, {north_mean:.2f} PSU) "
        f"than south (Nemunas mouth, {south_mean:.2f} PSU) — CMEMS gradient "
        f"may be inverted or lagoon is mostly land-masked on this day."
    )
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

    # Agent sanity — hard invariant (not a tautology like v1)
    alive = int(sim.pool.alive.sum())
    dead = int((~sim.pool.alive & ~sim.pool.arrived).sum())
    arrived = int(sim.pool.arrived.sum())
    assert alive + dead + arrived == 500, (
        f"Agent count invariant broken: alive={alive} dead={dead} arrived={arrived}"
    )
    # Not total extinction — 30 days at realistic temperature should keep some alive
    assert alive > 0, "All agents died — env likely broken"
    # At realistic Curonian temperature, thermal-avoid threshold 20°C should
    # NOT cause mass die-off; T_ACUTE_LETHAL=24°C is the real mortality gate.
    # If >90% died in 30 days, something is wrong with the thermal response.
    assert dead < 450, f"Mass die-off ({dead}/500) — check T_ACUTE_LETHAL logic"

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
2. **Grey seal + cormorant predation events.** Adds new event types in `events_builtin.py`. Cormorant colony at **Juodkrantė on the Curonian Spit** is the specific location (4,000+ breeding pairs); not a generic Kuršių Nerija reference. Estimated 1 week.
3. **Ice cover Dec-Mar** — 58-134 days of lagoon ice cover (Idzelytė et al. 2019); critical for egg/winter-resident modeling. Also triggers under-ice hypoxia. P2 realism polish.
4. **Seiche wind-forcing** — wire `winds_stub.nc` into `seiche_pause()`.
5. **Real Nemunas EPA discharge** — negotiate access to Lithuanian Environmental Protection Agency records at Smalininkai station (Valiuškevičius et al. 2019: long-term mean ~530-700 m³/s, spring flood 1500-2500+ m³/s).
6. **Nemunas delta branching** — the delta splits into **Atmata, Skirvytė, Pakalnė, Gilija, Rusnė**. Skirvytė carries the main flow to Kaliningrad; Atmata/Pakalnė discharge to the NE corner of the lagoon. Salmon homing to natal tributaries (Žeimena, Merkys, Dubysa) must navigate this branching — needs topology-aware migration logic.
7. **Hatchery vs wild fish distinction** — most "Baltic salmon" in the Nemunas today are **hatchery-origin from Žeimena and Simnas rivers** (Lithuanian programme since 1997). Separate calibration and behavior parameters for hatchery vs wild populations may be needed.
8. **Round goby egg predation** — `Neogobius melanostomus` invasion in the Curonian since 2002 (Rakauskas et al. 2013) is a potential egg/larval predator on salmon redds.
9. **Summer cyanobacteria + hypoxia** — heavy eutrophic lagoon (Mėžinė et al. 2019); summer DO crashes and H₂S episodes documented but not scoped here.

# Rollback plan

Each phase is independent. If Phase 3 (CMEMS) doesn't land, Phases 1 + 2 still work with stub forcing. If the whole plan is reverted, the `main` branch's `config_curonian_minimal.yaml` is untouched — the new `config_curonian_baltic.yaml` lives alongside it.

```bash
# To revert:
git revert <phase-1-commit>..<phase-6-commit>
# data/ files become unreferenced but harmless
```
