# Baltic Salmon IBM for Curonian Lagoon — Design Document

**Date:** 2026-02-27
**Status:** Approved

## Overview

A Python Individual-Based Model (IBM) simulating Baltic salmon (*Salmo salar*) migration through the Curonian Lagoon to Nemunas River spawning grounds. Based on Snyder et al. (2019) HexSim migration corridor model, extended with full estuarine features (salinity stress, DO avoidance, seiche-pause).

## Key Decisions

| Decision | Choice |
|---|---|
| Target scenario | Curonian Lagoon, Baltic salmon |
| Core engine | Custom lightweight (numpy/xarray/scipy) |
| UI framework | Shiny for Python |
| Map visualization | Plotly (interactive triangular mesh + agents) |
| Scope | Full estuarine features in v0.1 |
| Grid | Unstructured triangular (Delaunay from regular grid) |
| Architecture | Hybrid OOP + Vectorized (AgentPool pattern) |
| Bioenergetics | Wisconsin model, Forseth et al. 2001 params |
| Output | Parquet tracks + NetCDF mesh snapshots |

## Project Structure

```
salmon/
├── config_curonian_minimal.yaml
├── data/
│   ├── curonian_minimal_grid.nc
│   ├── forcing_cmems_phy_stub.nc
│   ├── nemunas_discharge_stub.nc
│   └── winds_stub.nc
├── salmon_ibm/                      # core engine (no UI dependency)
│   ├── __init__.py
│   ├── config.py                    # YAML config loader + validation
│   ├── mesh.py                      # Triangular mesh from regular grid
│   ├── environment.py               # Time-varying fields on mesh
│   ├── agents.py                    # FishAgent dataclass + AgentPool
│   ├── behavior.py                  # 5-state decision table + overrides
│   ├── movement.py                  # Movement kernels on triangular mesh
│   ├── bioenergetics.py             # Wisconsin hourly energy budget
│   ├── estuary.py                   # Salinity cost, DO avoidance, seiche-pause
│   ├── simulation.py                # Main loop / scheduler
│   └── output.py                    # Track logging, diagnostics
├── app.py                           # Shiny for Python entry point
├── ui/
│   ├── __init__.py
│   ├── sidebar.py                   # Parameter controls
│   ├── map_view.py                  # Plotly mesh + agent viz
│   ├── charts.py                    # Time series plots
│   └── run_controls.py              # Start/stop/step/reset
├── tests/
│   ├── test_mesh.py
│   ├── test_bioenergetics.py
│   ├── test_behavior.py
│   └── test_simulation.py
└── run.py                           # CLI entry point (headless)
```

## Module Details

### 1. Mesh (`salmon_ibm/mesh.py`)

Converts 30x30 regular lat/lon grid into unstructured triangular mesh via `scipy.spatial.Delaunay`.

```
TriMesh:
  nodes: ndarray[N, 2]          # (lat, lon) of grid nodes
  triangles: ndarray[M, 3]      # node indices per triangle
  centroids: ndarray[M, 2]      # triangle centers
  neighbors: ndarray[M, 3]      # adjacent triangle indices (-1 = boundary)
  mask: ndarray[M]              # 1=water, 0=land
  depth: ndarray[M]             # bathymetry at centroids
  areas: ndarray[M]             # triangle areas in m^2
```

Key methods:
- `from_regular_grid(grid_nc)` — construct mesh from NetCDF grid
- `find_triangle(lat, lon)` — locate containing triangle
- `water_neighbors(tri_idx)` — masked neighbor list
- `gradient(field, tri_idx)` — field gradient from neighbor differences

### 2. Environment (`salmon_ibm/environment.py`)

Loads time-varying NetCDF forcing and serves per-triangle values.

Fields at each hourly timestep:
- temperature (C) from `tos`
- salinity (PSU) from `sos`
- u_current, v_current (m/s) from `uo`, `vo`
- ssh (m) from `zos`
- do (mg/L) — derived or placeholder
- discharge (m3/s) from `Q`
- wind_u, wind_v (m/s) from `u10`, `v10`

Key methods:
- `advance(t)` — load fields for timestep t
- `sample(tri_idx)` — return all fields at a triangle
- `gradient(field, tri_idx)` — spatial gradient for directed movement
- `nearest_cwr(tri_idx)` — closest cold-water refuge cell and distance
- `dSSH_dt(tri_idx)` — SSH rate of change for seiche detection

### 3. Agents (`salmon_ibm/agents.py`)

**FishAgent** (dataclass, view into AgentPool):
- id, tri_idx, mass_g, ed_kJ_g
- target_spawn_hour, behavior (enum: HOLD/RANDOM/TO_CWR/UPSTREAM/DOWNSTREAM)
- cwr_hours, hours_since_left_cwr, steps
- dead, arrived
- temp_history (last 3 hourly temperatures for T3h mean)

**AgentPool** (vectorized batch):
- All agent state as parallel numpy arrays (structure-of-arrays)
- FishAgent is a zero-copy view into pool arrays
- Vectorized methods: `update_bioenergetics()`, `pick_behaviors()`, `apply_overrides()`

### 4. Behavior (`salmon_ibm/behavior.py`)

5-state probabilistic decision table based on (time_to_spawn, T3h_mean).
Temperature bins and probability weights configurable via YAML.

Overrides (in priority order):
1. First move always UPSTREAM
2. In CWR and under max residence -> stay
3. Exceeded CWR residence -> force UPSTREAM
4. TO_CWR but nearest CWR too far -> UPSTREAM
5. TO_CWR but under avoid cooldown -> UPSTREAM
6. Arrived at top of reach -> HOLD

### 5. Estuary Extensions (`salmon_ibm/estuary.py`)

Beyond Snyder's original model:

**Salinity stress** (from config):
- S_opt = 0.5 PSU, S_tol = 6.0 PSU, k = 0.6
- Activity cost multiplier: `1 + k * max(0, S - S_opt - S_tol)`

**DO avoidance** (from config):
- lethal threshold: 2.0 mg/L (additional mortality probability)
- high threshold: 4.0 mg/L (force escape movement)

**Seiche pause** (from config):
- If |dSSH/dt| > 0.02 m/15min, switch behavior to HOLD
- Prevents migration during dangerous flow reversals

### 6. Movement Kernels (`salmon_ibm/movement.py`)

All movement operates via triangle neighbor traversal on the mesh:

- **HOLD**: Stay in current triangle
- **RANDOM**: Auto-correlated random walk through water neighbors
- **UPSTREAM**: Follow negative SSH gradient + random jitter to avoid stalling
- **DOWNSTREAM**: Follow positive SSH gradient + random jitter
- **TO_CWR**: Follow negative temperature gradient until cold cell reached
- **Current-assisted**: Flow advection (u,v) translated to neighbor moves, added to behavioral movement

### 7. Bioenergetics (`salmon_ibm/bioenergetics.py`)

Wisconsin Bioenergetics, hourly. Non-feeding migrants (C=0).

Core equation:
```
R_base = RA * mass^(RB-1) * exp(RQ * T) * OXY_CAL   [J/g/day]
R_hourly = R_base * mass * activity_mult * salinity_cost / 24   [J/fish/hour]
E_total -= R_hourly
ed_kJ_g = E_total / (mass * 1000)
dead = (ed_kJ_g < 4.0)
```

Parameters for Salmo salar (Forseth et al. 2001 / FB4):
- RA = 0.00264 g O2/g/day
- RB = -0.217
- RQ = 0.06818 1/C
- OXY_CAL = 13,560 J/gO2
- Mortality threshold: 4.0 kJ/g (Snyder et al. 2019)

Activity multipliers by behavior:
- CWR: 0.8, HOLD: 1.0, DOWNSTREAM: 1.0, RANDOM: 1.2, UPSTREAM: 1.5

### 8. Simulation Loop (`salmon_ibm/simulation.py`)

```
for each hour t in simulation_period:
    env.advance(t)
    pool.pick_behaviors(env, params)       # vectorized
    pool.apply_overrides(env, params)      # vectorized + estuarine
    execute_movement(pool, env, mesh)      # per-agent neighbor traversal
    pool.update_cwr_state(env)             # update timers
    pool.check_arrival(env, spawn_zone)    # check spawning ground reached
    pool.update_bioenergetics(env)         # vectorized
    pool.apply_mortality()                 # vectorized
    output.log_step(t, pool)
```

### 9. Output (`salmon_ibm/output.py`)

- Agent tracks: Parquet with [time, agent_id, tri_idx, lat, lon, ed_kJ_g, behavior, alive, arrived]
- Summary stats: per-timestep survival, mean energy, behavior distribution, CWR occupancy
- Mesh snapshots: optional NetCDF with environmental fields + agent density per triangle

### 10. Shiny UI

**app.py** — main Shiny application with 4 panels:

| Panel | Content |
|---|---|
| Sidebar | Species params (RA, RB, RQ), estuary thresholds, agent count, spawn timing, config selector |
| Map View | Plotly triangular mesh colored by field (T/S/DO/depth) + agent scatter colored by behavior |
| Charts | Survival curve, energy density over time, behavior stacked area, CWR occupancy |
| Run Controls | Start/Pause/Step/Reset buttons, speed slider, progress bar |

Simulation runs in asyncio background task; each timestep triggers reactive plot updates.

UI modules in `ui/`:
- `sidebar.py` — parameter input controls
- `map_view.py` — Plotly mesh + agent rendering
- `charts.py` — time series and distribution plots
- `run_controls.py` — simulation control buttons and progress

## Dependencies

Core: numpy, scipy, xarray, pyyaml, pandas, pyarrow
Viz: plotly
UI: shiny (for Python)
Data: h5netcdf or netcdf4 (for NetCDF I/O)

## References

- Snyder, M.N. et al. (2019). Migration corridor simulation model. Landscape Ecology.
  GitHub: snydermn/migration_corridor_simulation_model
- Forseth, T. et al. (2001). Bioenergetics model for Atlantic salmon. Can. J. Fish. Aquat. Sci. 58:419-432.
- Deslauriers, D. et al. (2017). Fish Bioenergetics 4.0. Fisheries 42(11):586-596.
- Umgiesser, G. et al. (2016). Seasonal renewal time variability in the Curonian Lagoon. Ocean Science 12:391-402.
- Crossin, G.T. et al. (2004). Energy-density mortality threshold reference.
