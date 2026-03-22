# HexSimPy Model Manual — Data Requirements and Configuration Guide

This manual describes the data and configuration required to set up and run a HexSimPy salmon individual-based model (IBM) simulation. It covers both operational modes (NetCDF and HexSim), all configuration parameters, and optional subsystems.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Grid Modes](#2-grid-modes)
3. [NetCDF Grid Mode](#3-netcdf-grid-mode)
4. [HexSim Grid Mode](#4-hexsim-grid-mode)
5. [YAML Configuration Schema](#5-yaml-configuration-schema)
6. [Agent Initialization](#6-agent-initialization)
7. [Bioenergetics Parameters](#7-bioenergetics-parameters)
8. [Behavioral Decision Table](#8-behavioral-decision-table)
9. [Estuary Parameters](#9-estuary-parameters)
10. [Optional Subsystems](#10-optional-subsystems)
11. [Event System](#11-event-system)
12. [Output and Reporting](#12-output-and-reporting)
13. [Ensemble Runs](#13-ensemble-runs)
14. [Example Configurations](#14-example-configurations)

---

## 1. Overview

HexSimPy simulates the migration of Baltic salmon (or other fish species) through a spatial domain using an individual-based modeling approach. Each agent (fish) maintains its own state: position, mass, energy density, behavior, and spawn timing.

The simulation advances in hourly timesteps. Each timestep:

1. The environment updates (temperature, salinity, currents, SSH)
2. Agents select a behavior based on temperature and urgency
3. Estuarine overrides are applied (salinity stress, low DO, seiche)
4. Agents move according to their behavior (with barrier enforcement)
5. The Wisconsin bioenergetics model computes energy expenditure
6. Agents below the lethal energy density threshold die
7. Output is logged

Two operational modes are supported:

| Mode | Mesh | Environment | Config key |
|------|------|-------------|------------|
| **NetCDF** | `TriMesh` (Delaunay triangulation) | `Environment` (xarray forcing) | `grid.file` |
| **HexSim** | `HexMesh` (hex grid from workspace) | `HexSimEnvironment` (zone lookup) | `grid.type: hexsim` |

---

## 2. Grid Modes

### NetCDF Mode

Uses a triangular mesh built from a regular lat/lon grid via scipy Delaunay triangulation. Environmental forcing (temperature, salinity, currents, SSH) is loaded from NetCDF files and interpolated onto mesh triangles.

Best for: oceanographic/lagoon domains with spatially-explicit forcing from operational models (e.g., CMEMS).

### HexSim Mode

Uses a hexagonal grid loaded from an EPA HexSim workspace directory. Temperature comes from a zone-based lookup table. The hex grid uses pointy-top hexagons in an odd-row offset layout.

Best for: river systems with HexSim workspace data, or compatibility with existing HexSim scenarios.

---

## 3. NetCDF Grid Mode

### 3.1 Grid File

A single NetCDF file defining the spatial domain. Required variables:

| Variable | Dimensions | Type | Description |
|----------|-----------|------|-------------|
| `lat` | `(n_lat,)` | `float` | Latitude coordinates (degrees north) |
| `lon` | `(n_lon,)` | `float` | Longitude coordinates (degrees east) |
| `mask` | `(n_lat, n_lon)` | `int` or `bool` | Water mask (1 = water, 0 = land) |
| `depth` | `(n_lat, n_lon)` | `float` | Bathymetric depth (metres, positive down) |

Variable names are configurable in YAML:

```yaml
grid:
  file: curonian_lagoon.nc
  lat_var: lat       # default: lat
  lon_var: lon       # default: lon
  mask_var: mask     # default: mask
  depth_var: depth   # default: depth
```

The mesh is constructed by:
1. Creating a node grid from `lat` x `lon`
2. Building Delaunay triangulation
3. Marking triangles as water if all 3 nodes have `mask = 1`
4. Computing per-triangle centroids, areas, depths, and neighbor lists

### 3.2 Forcing Files

Time-varying environmental forcing is loaded from NetCDF files specified in the `forcings` section. Three forcing types are supported:

#### Physics Surface Forcing (required)

Provides temperature, salinity, current velocity, and sea surface height.

```yaml
forcings:
  physics_surface:
    file: forcing_cmems_phy.nc
    time_var: time        # time dimension name
    temp_var: tos         # sea surface temperature (°C)
    salt_var: sos         # sea surface salinity (PSU)
    u_var: uo             # eastward current velocity (m/s)
    v_var: vo             # northward current velocity (m/s)
    ssh_var: zos          # sea surface height (m)
```

All variables must share the same spatial grid as the mesh NetCDF file and have a time dimension. Data is preloaded at initialization and indexed by timestep modulo the number of available time steps.

#### Wind Forcing (optional)

```yaml
forcings:
  winds:
    file: winds_era5.nc
    time_var: time
    u10_var: u10          # 10m eastward wind (m/s)
    v10_var: v10          # 10m northward wind (m/s)
```

#### River Discharge Forcing (optional)

```yaml
forcings:
  river_discharge:
    file: nemunas_discharge.nc
    time_var: time
    Q_var: Q              # discharge (m³/s)
```

### 3.3 Data Interpolation

Raw forcing data is interpolated onto the triangular mesh at initialization:
- Node values are averaged over each triangle's 3 vertices
- The resulting per-triangle arrays are stored for O(1) access per timestep
- Environment advances by index: `fields["temperature"] = preloaded_data[t % n_timesteps]`

---

## 4. HexSim Grid Mode

### 4.1 Workspace Directory Structure

A HexSim workspace is a directory containing the following structure:

```
Workspace/
├── GridMeta.xml                        — Grid dimensions and metadata
├── Spatial Data/
│   └── Hexagons/
│       ├── River [ extent ]/           — Water extent mask layer
│       │   └── River [ extent ].1.hxn  — Binary hex-map file
│       ├── River [ depth ]/            — Depth layer
│       │   └── River [ depth ].1.hxn
│       ├── Temperature Zones/          — Temperature zone IDs
│       │   └── Temperature Zones.1.hxn
│       └── Gradient [ upstream ]/      — Upstream gradient (for SSH proxy)
│           └── Gradient [ upstream ].1.hxn
├── Spatial Data/
│   └── barriers/                       — Movement barrier files (optional)
│       └── dams.hbf
└── Analysis/
    └── Data Lookup/
        └── River Temperature.csv       — Temperature lookup table
```

### 4.2 .hxn File Format

HexSim hex-map binary files store per-cell data. Two formats are supported:

**PATCH_HEXMAP format** (37-byte header):
- Magic: `PATCH_HEXMAP` (13 bytes)
- Width, height (int32 each)
- Narrow flag (int32)
- Classification count (int32)
- Data width, data height (int32 each)
- Data: `float32[data_height * data_width]`

**Plain format** (no header):
- Raw `float32` values, one per cell
- Cell count = `nrows * ncols` from GridMeta.xml

### 4.3 GridMeta.xml

Required XML elements:

| Element | Type | Description |
|---------|------|-------------|
| `<hexCount>` | int | Total hexagons in grid |
| `<rows>` | int | Grid rows |
| `<columns>` | int | Grid columns |
| `<narrow>` | bool | Whether grid uses narrow layout |
| `<hexagonWidth>` | float | Cell width in metres |

### 4.4 Temperature CSV

Located at `Analysis/Data Lookup/<temperature_csv>`:

- **Format**: CSV with no header
- **Rows**: temperature zones (0-indexed internally; HexSim uses 1-based zones, subtract 1)
- **Columns**: timesteps (hourly)
- **Values**: temperature in °C

Example (3 zones, 4 timesteps):

```
12.5,13.0,13.5,14.0
10.0,10.5,11.0,11.5
8.0,8.5,9.0,9.5
```

The zone for each cell is read from the "Temperature Zones" hex-map layer.

### 4.5 .hbf Barrier File Format

Text-based file listing barrier edges between cells:

```
barrier_class_name
from_cell_id to_cell_id
from_cell_id to_cell_id
...
```

Barrier outcomes (mortality, deflection, transmission probabilities) are configured in YAML.

### 4.6 HexSim Grid Geometry

- **Hex type**: pointy-top
- **Layout**: odd-row offset
- **Edge length**: derived from `cell_width` in GridMeta.xml
- **Spacing**: dx = `sqrt(3) * edge`, dy = `1.5 * edge`
- **Neighbors**: up to 6 per cell
- **Coordinate system**: metric (metres), not geographic

### 4.7 HexSim YAML Configuration

```yaml
grid:
  type: hexsim

hexsim:
  workspace: "Columbia River Migration Model/Columbia [small]"
  species: chinook                     # species identifier
  extent_layer: "River [ extent ]"     # water mask layer name (auto-detected if omitted)
  depth_layer: "River [ depth ]"       # depth layer name (auto-detected if omitted)
  temperature_csv: River Temperature.csv
```

### 4.8 Fields Provided by HexSimEnvironment

| Field | Source | Notes |
|-------|--------|-------|
| `temperature` | Zone lookup CSV | Updated each timestep |
| `salinity` | (all zeros) | Not available in zone model |
| `ssh` | Gradient [ upstream ] layer | Static proxy, normalized and negated |
| `u_current` | (all zeros) | Not available |
| `v_current` | (all zeros) | Not available |

---

## 5. YAML Configuration Schema

The complete YAML configuration file consists of the following top-level sections. Only `grid` is required; all other sections have defaults.

### 5.1 `grid` (required)

```yaml
grid:
  # NetCDF mode:
  file: path/to/grid.nc        # path to NetCDF grid file
  lat_var: lat                 # latitude variable name (default: lat)
  lon_var: lon                 # longitude variable name (default: lon)
  mask_var: mask               # water mask variable name (default: mask)
  depth_var: depth             # depth variable name (default: depth)

  # HexSim mode:
  type: hexsim                 # set to "hexsim" to use HexSim workspace
```

### 5.2 `hexsim` (required for HexSim mode)

```yaml
hexsim:
  workspace: "path/to/workspace"     # relative to config file directory
  species: chinook                   # species name (for auto-detection)
  extent_layer: "River [ extent ]"   # optional, auto-detected
  depth_layer: "River [ depth ]"     # optional, auto-detected
  temperature_csv: River Temperature.csv
```

### 5.3 `forcings` (required for NetCDF mode)

```yaml
forcings:
  physics_surface:
    file: forcing_phy.nc
    time_var: time
    temp_var: tos
    salt_var: sos
    u_var: uo
    v_var: vo
    ssh_var: zos
  winds:                          # optional
    file: winds.nc
    time_var: time
    u10_var: u10
    v10_var: v10
  river_discharge:                # optional
    file: discharge.nc
    time_var: time
    Q_var: Q
```

### 5.4 `bioenergetics` (optional)

Wisconsin bioenergetics model parameters. All have defaults suitable for Atlantic salmon.

```yaml
bioenergetics:
  RA: 0.00264                   # respiration scaling coefficient
  RB: -0.217                    # mass allometry exponent (must be < 0)
  RQ: 0.06818                   # temperature Q10 exponent (must be > 0)
  ED_MORTAL: 4.0                # lethal energy density (kJ/g, must be > 0)
  T_OPT: 16.0                   # optimal temperature (°C)
  T_MAX: 26.0                   # upper lethal temperature (°C, must be > T_OPT)
  ED_TISSUE: 5.0                # energy density of catabolized tissue (kJ/g)
  MASS_FLOOR_FRACTION: 0.5      # minimum mass as fraction of initial (prevents collapse)
  activity_by_behavior:          # activity multiplier per behavior state
    0: 1.0                       # HOLD
    1: 1.2                       # RANDOM
    2: 0.8                       # TO_CWR
    3: 1.5                       # UPSTREAM
    4: 1.0                       # DOWNSTREAM
```

**Validation rules**: `RA > 0`, `RB < 0`, `RQ > 0`, `ED_MORTAL > 0`, `T_MAX > T_OPT`.

### 5.5 `behavior` (optional)

Behavioral decision table parameters (Snyder et al. 2019).

```yaml
behavior:
  temp_bins: [16.0, 18.0, 20.0]    # temperature bin edges (°C)
  time_bins: [360, 720]             # hours-to-spawn bin edges
  max_cwr_hours: 48                 # max consecutive hours in CWR before forced upstream
  avoid_cwr_cooldown_h: 12          # cooldown hours after leaving CWR
  max_dist_to_cwr: 5000.0           # max distance (m) to consider CWR reachable
  p_table:                           # 3D probability table [time_idx, temp_idx, 5 behaviors]
    # ... (optional, defaults to standard salmon table)
```

The probability table has dimensions `(n_time_bins, n_temp_bins+1, 5)` where the 5 columns correspond to `[HOLD, RANDOM, TO_CWR, UPSTREAM, DOWNSTREAM]` and rows sum to 1.0.

If `p_table` is omitted, the standard Chinook salmon probability table from Snyder et al. (2019) is used.

### 5.6 `estuary` (optional)

Estuarine stress parameters.

```yaml
estuary:
  salinity_cost:
    S_opt: 0.5                    # optimal salinity (PSU)
    S_tol: 6.0                    # tolerance range above S_opt (PSU)
    k: 0.6                        # cost slope per PSU excess
    max_cost: 5.0                 # maximum cost multiplier ceiling
  do_avoidance:
    lethal: 2.0                   # DO below this = death (mg/L)
    high: 4.0                     # DO below this = escape behavior (mg/L)
  seiche_pause:
    dSSHdt_thresh_m_per_15min: 0.02   # SSH rate threshold for movement pause
```

To disable estuary effects (e.g., for riverine simulations), set extreme values:

```yaml
estuary:
  salinity_cost:
    S_opt: 0.5
    S_tol: 999       # effectively unlimited tolerance
    k: 0.0            # no cost
  do_avoidance:
    lethal: 0.0
    high: 0.0
  seiche_pause:
    dSSHdt_thresh_m_per_15min: 999   # never triggers
```

### 5.7 `genetics` (optional)

Diploid genetic system configuration.

```yaml
genetics:
  loci:
    - name: run_timing
      n_alleles: 4          # number of distinct alleles (must be >= 2)
      position: 0.0         # chromosomal position in centiMorgans
    - name: growth_rate
      n_alleles: 3
      position: 50.0
  rng_seed: 42              # optional seed for genetic initialization
  initialize_random: true   # randomly assign alleles at start (default: true)
```

Linkage between loci is computed using Haldane's mapping function: positions closer in cM have lower crossover probability during recombination.

### 5.8 `barriers` (optional)

Edge-based movement barriers.

```yaml
barriers:
  file: barriers.hbf               # path to .hbf barrier file
  classes:
    dam:                            # barrier class name (must match .hbf file)
      forward:                      # outcomes when crossing in forward direction
        mortality: 0.1              # probability of death
        deflection: 0.8             # probability of being turned back
        transmission: 0.1           # probability of passing through
      reverse:                      # outcomes when crossing in reverse direction
        mortality: 0.0
        deflection: 1.0
        transmission: 0.0
```

For each barrier class, `mortality + deflection + transmission` must equal 1.0 for both directions.

### 5.9 `network` (optional)

1D stream network topology.

```yaml
network:
  segments:
    - id: 0
      length: 1000.0               # segment length in metres
      upstream_ids: []              # connected upstream segments
      downstream_ids: [1]           # connected downstream segments
      order: 1                      # Strahler stream order
    - id: 1
      length: 2000.0
      upstream_ids: [0]
      downstream_ids: []
      order: 2
```

### 5.10 `population` (optional)

Initial population configuration.

```yaml
population:
  name: salmon                     # population identifier
  n_agents: 1000                   # initial population size
  mass_mean: 3500.0                # mean initial mass (g)
  mass_std: 500.0                  # mass standard deviation (g)
  ed_init: 6.5                     # initial energy density (kJ/g)
  spawn_hours_mean: 720.0          # mean hours to target spawn (30 days)
  spawn_hours_std: 168.0           # std dev of spawn timing (7 days)
```

### 5.11 `events` (optional)

Custom event sequence. If omitted, the default salmon pipeline is used. Events are specified as a list:

```yaml
events:
  - type: movement
    trigger: every_step
    params:
      n_micro_steps: 3
      cwr_threshold: 16.0
  - type: survival
    trigger: every_step
    params:
      thermal: true
      starvation: true
  - type: custom
    trigger:
      periodic:
        interval: 24
    params:
      callback: my_module.my_function
```

---

## 6. Agent Initialization

Each agent starts with the following state, drawn from configurable distributions:

| State variable | Default distribution | Description |
|---------------|---------------------|-------------|
| `tri_idx` | Uniform from water cells | Starting mesh cell index |
| `mass_g` | Normal(3500, 500) g | Initial wet body mass |
| `ed_kJ_g` | Fixed 6.5 kJ/g | Initial energy density |
| `target_spawn_hour` | Normal(720, 168) h | Target spawn time (hours from start) |
| `behavior` | `UPSTREAM` (0 step) | Initial behavior |
| `cwr_hours` | 0 | CWR occupancy counter |
| `hours_since_cwr` | 0 | Time since last CWR visit |
| `steps` | 0 | Movement step counter |
| `alive` | True | Alive status |
| `arrived` | False | Arrival at spawning grounds |
| `temp_history` | zeros(3) | 3-hour temperature rolling buffer |

All agent state is stored as Structure-of-Arrays (SoA) in `AgentPool` for vectorized operations.

---

## 7. Bioenergetics Parameters

The Wisconsin bioenergetics model computes hourly energy expenditure for non-feeding migrants:

```
R_daily = RA * mass^RB * exp(RQ * T) * activity_mult    [gO₂/day]
R_hourly = R_daily / 24                                   [gO₂/hour]
R_joules = R_hourly * 13560                               [J/hour]   (oxycalorific coefficient)
```

Energy loss is subtracted from the total energy pool (`ED * mass`). Mass decreases proportionally:

```
energy_lost = R_joules * salinity_cost
total_energy = ED * mass - energy_lost
new_mass = max(mass * MASS_FLOOR_FRACTION, total_energy / ED_TISSUE)
new_ED = total_energy / new_mass
```

If `new_ED < ED_MORTAL`, the agent dies (starvation mortality).
If temperature `T >= T_MAX`, the agent dies (thermal mortality).

### Parameter Defaults and Ranges

| Parameter | Default | Typical Range | Source |
|-----------|---------|--------------|--------|
| `RA` | 0.00264 | 0.001–0.01 | Forseth et al. (2001) |
| `RB` | -0.217 | -0.3 to -0.1 | Forseth et al. (2001) |
| `RQ` | 0.06818 | 0.04–0.08 | Forseth et al. (2001) |
| `ED_MORTAL` | 4.0 kJ/g | 3.0–5.0 | Species-specific calibration |
| `T_OPT` | 16.0 °C | 12–20 | Species-specific |
| `T_MAX` | 26.0 °C | 24–28 | Species-specific |
| `ED_TISSUE` | 5.0 kJ/g | 4.0–6.0 | Hanson et al. (1997) |
| `MASS_FLOOR_FRACTION` | 0.5 | 0.3–0.7 | Prevents numerical mass collapse |

---

## 8. Behavioral Decision Table

Agent behavior is selected stochastically from a probability table indexed by:
- **Temperature**: 3-hour mean temperature (°C), binned by `temp_bins`
- **Urgency**: hours remaining to target spawn time, binned by `time_bins`

### Default Bins

- Temperature edges: `[16.0, 18.0, 20.0]` -> 4 bins: `<16`, `16-18`, `18-20`, `>20`
- Time edges: `[360, 720]` -> 3 bins: `<360h` (urgent), `360-720h` (moderate), `>720h` (relaxed)

### Behavior States

| ID | Name | Movement |
|----|------|----------|
| 0 | `HOLD` | No movement |
| 1 | `RANDOM` | Random walk to any water neighbor |
| 2 | `TO_CWR` | Move toward coldest water below threshold |
| 3 | `UPSTREAM` | Move toward highest SSH (ascending gradient) |
| 4 | `DOWNSTREAM` | Move toward lowest SSH (descending gradient) |

### Deterministic Overrides (applied after stochastic selection)

1. **First step**: All agents forced to `UPSTREAM`
2. **CWR timeout**: Agents exceeding `max_cwr_hours` (48h) in cold-water refuge forced `UPSTREAM`
3. **CWR cooldown**: Within `avoid_cwr_cooldown_h` (12h) of leaving CWR, `TO_CWR` overridden to `UPSTREAM`

---

## 9. Estuary Parameters

Three estuarine stress mechanisms modify agent behavior and energetics:

### 9.1 Salinity Cost

Increases metabolic cost when salinity exceeds the tolerance range:

```
cost = 1.0                                    if S <= S_opt + S_tol
cost = 1.0 + k * (S - S_opt - S_tol)         if S > S_opt + S_tol
cost = min(cost, max_cost)                    capped at max_cost
```

Default: `S_opt=0.5 PSU`, `S_tol=6.0 PSU`, `k=0.6`, `max_cost=5.0`.

### 9.2 Dissolved Oxygen Avoidance

Classifies DO levels into three states:

| State | Condition | Effect |
|-------|-----------|--------|
| `OK` | DO >= `high` | Normal behavior |
| `ESCAPE` | `lethal` < DO < `high` | Force downstream movement |
| `LETHAL` | DO <= `lethal` | Agent dies |

Default: `lethal=2.0 mg/L`, `high=4.0 mg/L`. Validation: `lethal <= high`.

### 9.3 Seiche Pause

Pauses movement when the rate of sea surface height change exceeds a threshold:

```
pause = |dSSH/dt| > thresh
```

Default: `thresh=0.02 m/15min`.

---

## 10. Optional Subsystems

### 10.1 Accumulators

Per-agent floating-point state variables managed by `AccumulatorManager`. Shape: `(n_agents, n_accumulators)`.

Each accumulator is defined with optional bounds:

```python
AccumulatorDef(name="age", min_val=0.0, max_val=None, linked_trait=None)
```

24 built-in updater functions are available (see [API Reference](api-reference.md#salmon_ibmaccumulators) for the full list), including:
- `clear`, `increment`, `stochastic_increment`, `expression`
- `time_step`, `individual_id`, `individual_locations`
- `quantify_location`, `quantify_extremes`, `hexagon_presence`
- `uptake`, `accumulator_transfer`
- `allocated_hexagons`, `explored_hexagons`, `resources_allocated`, `resources_explored`
- `group_size`, `group_sum`, `births`, `mate_verification`
- `subpopulation_assign`, `subpopulation_selector`
- `trait_value_index`, `data_lookup`, `stochastic_trigger`

### 10.2 Traits

Categorical per-agent state (e.g., life stage, sex). Four derivation modes:

| Type | Source | Example |
|------|--------|---------|
| `PROBABILISTIC` | Random assignment | Initial sex ratio |
| `ACCUMULATED` | Binned from accumulator + thresholds | Age class from age accumulator |
| `GENETIC` | Single-locus diploid genotype + phenotype map | Run timing from genotype |
| `GENETIC_ACCUMULATED` | Multi-locus weighted sum + thresholds | Growth phenotype from QTL |

Traits can be used as event filters: events fire only for agents matching specific trait values.

### 10.3 Genetics

Diploid genome system. Each agent has genotype array `(n_loci, 2)` with integer allele indices.

Operations:
- **Recombination**: Haldane-mapped crossover during reproduction
- **Mutation**: Per-allele transition matrix
- **Homozygosity**: Per-individual or per-locus measurement
- **Trait mapping**: Genotype -> phenotype via phenotype map or weighted sum

### 10.4 Barriers

Edge-based movement restrictions between adjacent cells. Each barrier edge has directional outcomes:
- **Mortality**: agent dies attempting to cross
- **Deflection**: agent bounces back to origin cell
- **Transmission**: agent successfully crosses

Barriers are precomputed into arrays for Numba-accelerated resolution during movement.

### 10.5 Stream Network

1D segment-based network for river topology. Agents can move upstream or downstream along segments, crossing segment boundaries when step length exceeds remaining distance.

### 10.6 Territorial Ranges

Non-overlapping cell ownership via `RangeAllocator`. Agents can expand territories by BFS to adjacent available cells with resources above a threshold, or contract by releasing low-resource cells.

### 10.7 Multi-Population Interactions

`MultiPopulationManager` tracks multiple named populations on the same mesh. Interaction events (predation, competition, disease) act on co-located agent pairs.

---

## 11. Event System

The simulation is driven by an ordered sequence of events. Each event has:
- **Name**: human-readable identifier
- **Trigger**: when the event fires (every step, periodic, window, random, once)
- **Trait filter**: optional filter to restrict to agents matching trait criteria
- **Population name**: target population (for multi-population scenarios)

### 11.1 Triggers

| Trigger | Parameters | Fires when |
|---------|-----------|------------|
| `EveryStep` | — | Every timestep |
| `Once` | `at` | Timestep `t == at` |
| `Periodic` | `interval`, `offset` | `(t - offset) % interval == 0` |
| `Window` | `start`, `end` | `start <= t < end` |
| `RandomTrigger` | `p` | With probability `p` per step |

### 11.2 Default Event Sequence (Salmon Pipeline)

If no custom events are configured, the following sequence runs each timestep:

1. **push_temperature** — Update 3-hour temperature history
2. **behavior_selection** — Stochastic behavior from decision table
3. **estuarine_overrides** — Salinity, DO, and seiche modifications
4. **update_cwr_counters** — Track CWR occupancy
5. **movement** — 3 micro-steps of behavior-specific movement
6. **update_timers** — Decrement spawn timer, increment step counter
7. **bioenergetics** — Wisconsin model energy expenditure + mortality
8. **logging** — Record output

### 11.3 Registered Event Types

| Type name | Class | Description |
|-----------|-------|-------------|
| `movement` | `MovementEvent` | Behavioral movement with barrier resolution |
| `survival` | `SurvivalEvent` | Wisconsin bioenergetics + thermal mortality |
| `stage_survival` | `StageSpecificSurvivalEvent` | Trait-filtered stage mortality |
| `introduction` | `IntroductionEvent` | Add new agents |
| `reproduction` | `ReproductionEvent` | Pair mating with genetic recombination |
| `custom` | `CustomEvent` | Arbitrary callback |
| `move` | `HexSimMoveEvent` | HexSim gradient/affinity movement |
| `hexsim_survival` | `HexSimSurvivalEvent` | Expression-based survival |
| `accumulate` | `HexSimAccumulateEvent` | Run updater functions |
| `patch_introduction` | `PatchIntroductionEvent` | Spatial data-driven introduction |
| `data_lookup` | `DataLookupEvent` | 2D CSV table lookup |
| `set_spatial_affinity` | `SetSpatialAffinityEvent` | Set movement affinity targets |
| `mutation` | `MutationEvent` | Allele mutations |
| `transition` | `TransitionEvent` | Probabilistic state transitions |
| `generated_hexmap` | `GeneratedHexmapEvent` | Expression-generated spatial data |
| `range_dynamics` | `RangeDynamicsEvent` | Territory expansion/contraction |
| `set_affinity` | `SetAffinityEvent` | Set spatial affinity from map |
| `plant_dynamics` | `PlantDynamicsEvent` | Seed dispersal and establishment |
| `interaction` | `InteractionEvent` | Multi-population interactions |
| `switch_population` | `SwitchPopulationEvent` | Transfer agents between populations |

---

## 12. Output and Reporting

### 12.1 CSV Output (OutputLogger)

When `output_path` is provided to `Simulation`, a CSV file is written with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `time` | int | Timestep |
| `agent_id` | int | Agent index |
| `tri_idx` | int | Current mesh cell |
| `lat` | float | Latitude (from centroid) |
| `lon` | float | Longitude (from centroid) |
| `ed_kJ_g` | float | Energy density (kJ/g) |
| `behavior` | int | Current behavior (0-4) |
| `alive` | bool | Alive status |
| `arrived` | bool | Arrival status |

### 12.2 Per-Step Summary

Each timestep produces a summary dict appended to `Simulation.history`:

```python
{"time": t, "n_alive": int, "n_arrived": int, "mean_ed": float, "behavior_counts": dict}
```

### 12.3 Reports (ReportManager)

| Report | Output |
|--------|--------|
| `ProductivityReport` | births, deaths, growth rate (lambda) per timestep |
| `DemographicReport` | n_alive, n_dead, mean_mass, mean_ed per timestep |
| `DispersalReport` | per-agent displacement from origin |
| `GeneticReport` | allele frequencies and heterozygosity per locus |

### 12.4 Spatial Tallies

| Tally | Output |
|-------|--------|
| `OccupancyTally` | Timesteps each cell was occupied |
| `DensityTally` | Cumulative agent count per cell |
| `DispersalFluxTally` | Agent movements between cells |
| `BarrierTally` | Barrier encounter outcomes per cell |

All reports export to CSV; tallies export as NumPy arrays via `ReportManager.save_all()`.

---

## 13. Ensemble Runs

Run multiple independent replicates in parallel:

```python
from salmon_ibm.config import load_config
from salmon_ibm.ensemble import run_ensemble

config = load_config("config_curonian_minimal.yaml")
results = run_ensemble(
    config,
    n_replicates=20,       # number of independent runs
    n_agents=1000,         # agents per replicate
    n_steps=720,           # timesteps (30 days)
    n_workers=8,           # parallel processes (None = all CPUs)
    base_seed=42,          # master seed for reproducibility
)

# Each result: {"seed": int, "history": list[dict], "n_alive": int, "n_arrived": int}
```

---

## 14. Example Configurations

### 14.1 NetCDF Mode — Curonian Lagoon

```yaml
# config_curonian_minimal.yaml
grid:
  file: curonian_minimal_grid.nc
  lat_var: lat
  lon_var: lon
  mask_var: mask
  depth_var: depth

forcings:
  physics_surface:
    file: forcing_cmems_phy_stub.nc
    time_var: time
    u_var: uo
    v_var: vo
    ssh_var: zos
    temp_var: tos
    salt_var: sos
  river_discharge:
    file: nemunas_discharge_stub.nc
    time_var: time
    Q_var: Q
  winds:
    file: winds_stub.nc
    time_var: time
    u10_var: u10
    v10_var: v10

estuary:
  salinity_cost:
    S_opt: 0.5
    S_tol: 6.0
    k: 0.6
  do_avoidance:
    lethal: 2.0
    high: 4.0
  seiche_pause:
    dSSHdt_thresh_m_per_15min: 0.02
```

Run: `python run.py --config config_curonian_minimal.yaml --agents 100 --steps 24`

### 14.2 HexSim Mode — Columbia River

```yaml
# config_columbia.yaml
grid:
  type: hexsim

hexsim:
  workspace: "Columbia River Migration Model/Columbia [small]"
  species: chinook
  temperature_csv: River Temperature.csv

estuary:
  salinity_cost:
    S_opt: 0.5
    S_tol: 999          # disable salinity cost
    k: 0.0
  do_avoidance:
    lethal: 0.0
    high: 0.0
  seiche_pause:
    dSSHdt_thresh_m_per_15min: 999   # disable seiche pause
```

Run: `python run.py --config config_columbia.yaml --agents 500 --steps 720`

### 14.3 HexSim Mode — Curonian Lagoon (hex grid)

```yaml
# config_curonian_hexsim.yaml
grid:
  type: hexsim

hexsim:
  workspace: "Curonian Lagoon"
  species: salmon
  extent_layer: "Lagoon [ extent ]"
  depth_layer: "Lagoon [ depth ]"
  temperature_csv: Lagoon Temperature.csv

estuary:
  salinity_cost:
    S_opt: 0.5
    S_tol: 6.0
    k: 0.6
  do_avoidance:
    lethal: 2.0
    high: 4.0
  seiche_pause:
    dSSHdt_thresh_m_per_15min: 0.02
```

### 14.4 Full Configuration with Optional Subsystems

```yaml
grid:
  type: hexsim

hexsim:
  workspace: "My Workspace"
  species: salmon
  temperature_csv: Temperature.csv

bioenergetics:
  RA: 0.00264
  RB: -0.217
  RQ: 0.06818
  ED_MORTAL: 4.0
  T_OPT: 16.0
  T_MAX: 26.0
  MASS_FLOOR_FRACTION: 0.5

behavior:
  temp_bins: [16.0, 18.0, 20.0]
  time_bins: [360, 720]
  max_cwr_hours: 48
  avoid_cwr_cooldown_h: 12

estuary:
  salinity_cost:
    S_opt: 0.5
    S_tol: 6.0
    k: 0.6
  do_avoidance:
    lethal: 2.0
    high: 4.0
  seiche_pause:
    dSSHdt_thresh_m_per_15min: 0.02

genetics:
  loci:
    - name: run_timing
      n_alleles: 4
      position: 0.0
    - name: growth_rate
      n_alleles: 3
      position: 50.0
  rng_seed: 42
  initialize_random: true

barriers:
  file: dams.hbf
  classes:
    dam:
      forward:
        mortality: 0.1
        deflection: 0.8
        transmission: 0.1
      reverse:
        mortality: 0.0
        deflection: 1.0
        transmission: 0.0

network:
  segments:
    - id: 0
      length: 5000.0
      upstream_ids: []
      downstream_ids: [1, 2]
      order: 1
    - id: 1
      length: 3000.0
      upstream_ids: [0]
      downstream_ids: []
      order: 2
    - id: 2
      length: 4000.0
      upstream_ids: [0]
      downstream_ids: []
      order: 2
```
