# salmon_ibm API Reference

This document covers the public API for 30 modules in the `salmon_ibm` package. Private functions and methods (those starting with `_`) are omitted. Two internal modules (`__init__`, `hexsim_viewer`) are excluded.

---

## Table of Contents

1. [`salmon_ibm.simulation`](#salmon_ibmsimulation)
2. [`salmon_ibm.agents`](#salmon_ibmagents)
3. [`salmon_ibm.population`](#salmon_ibmpopulation)
4. [`salmon_ibm.bioenergetics`](#salmon_ibmbioenergetics)
5. [`salmon_ibm.behavior`](#salmon_ibmbehavior)
6. [`salmon_ibm.events`](#salmon_ibmevents)
7. [`salmon_ibm.environment`](#salmon_ibmenvironment)
8. [`salmon_ibm.hexsim_env`](#salmon_ibmhexsim_env)
9. [`salmon_ibm.mesh`](#salmon_ibmmesh)
10. [`salmon_ibm.hexsim`](#salmon_ibmhexsim)
11. [`salmon_ibm.config`](#salmon_ibmconfig)
12. [`salmon_ibm.accumulators`](#salmon_ibmaccumulators)
13. [`salmon_ibm.genetics`](#salmon_ibmgenetics)
14. [`salmon_ibm.estuary`](#salmon_ibmestuary)
15. [`salmon_ibm.ensemble`](#salmon_ibmensemble)
16. [`salmon_ibm.output`](#salmon_ibmoutput)
17. [`salmon_ibm.movement`](#salmon_ibmmovement)
18. [`salmon_ibm.events_builtin`](#salmon_ibmevents_builtin)
19. [`salmon_ibm.events_hexsim`](#salmon_ibmevents_hexsim)
20. [`salmon_ibm.events_phase3`](#salmon_ibmevents_phase3)
21. [`salmon_ibm.event_descriptors`](#salmon_ibmevent_descriptors)
22. [`salmon_ibm.traits`](#salmon_ibmtraits)
23. [`salmon_ibm.barriers`](#salmon_ibmbarriers)
24. [`salmon_ibm.network`](#salmon_ibmnetwork)
25. [`salmon_ibm.ranges`](#salmon_ibmranges)
26. [`salmon_ibm.interactions`](#salmon_ibminteractions)
27. [`salmon_ibm.reporting`](#salmon_ibmreporting)
28. [`salmon_ibm.xml_parser`](#salmon_ibmxml_parser)
29. [`salmon_ibm.scenario_loader`](#salmon_ibmscenario_loader)
30. [`salmon_ibm.hexsim_expr`](#salmon_ibmhexsim_expr)

---

### `salmon_ibm.simulation`

> Main simulation loop.

#### `Landscape`

A `TypedDict` (all keys optional) passed to every event callback. Provides a shared context for the mesh, environmental fields, and optional subsystem managers.

```python
class Landscape(TypedDict, total=False):
    mesh: object                    # TriMesh | HexMesh
    fields: dict[str, np.ndarray]
    rng: np.random.Generator
    activity_lut: np.ndarray
    est_cfg: dict
    barrier_arrays: tuple | None
    genome: object | None           # GenomeManager | None
    multi_pop_mgr: object | None    # MultiPopulationManager | None
    network: object | None          # StreamNetwork | None
    step_alive_mask: np.ndarray
    spatial_data: dict[str, np.ndarray]
    global_variables: dict[str, float]
    census_records: list
    summary_reports: list
    log_dir: str
```

| Key | Type | Description |
|-----|------|-------------|
| `mesh` | `TriMesh` or `HexMesh` | Spatial mesh for the simulation domain |
| `fields` | `dict[str, np.ndarray]` | Environmental fields keyed by name (e.g. `"temperature"`) |
| `rng` | `np.random.Generator` | Shared RNG for stochastic processes |
| `activity_lut` | `np.ndarray` | Activity multiplier lookup table indexed by behavior integer |
| `est_cfg` | `dict` | Estuary configuration sub-dict from YAML |
| `barrier_arrays` | `tuple` or `None` | Pre-built barrier arrays from `BarrierMap.to_arrays()` |
| `genome` | `GenomeManager` or `None` | Genetics manager (Phase 3) |
| `multi_pop_mgr` | `MultiPopulationManager` or `None` | Multi-population interaction manager |
| `network` | `StreamNetwork` or `None` | Stream network topology (Phase 3) |
| `step_alive_mask` | `np.ndarray` | Boolean mask of living, non-arrived agents at step start |

---

#### `Simulation`

Top-level orchestrator for a single simulation run. Owns the mesh, environment, agent pool, population, event sequencer, and optional subsystems.

```python
class Simulation:
    def __init__(
        self,
        config: dict,
        n_agents: int = 100,
        data_dir: str = "data",
        rng_seed: int | None = None,
        output_path: str | None = None,
    ): ...

    def step(self) -> None: ...
    def run(self, n_steps: int) -> None: ...
    def close(self) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `dict` | — | Simulation configuration dict, typically from `load_config()` |
| `n_agents` | `int` | `100` | Number of fish agents to initialise |
| `data_dir` | `str` | `"data"` | Directory containing NetCDF forcing files (NetCDF grid only) |
| `rng_seed` | `int` or `None` | `None` | Seed for reproducible runs |
| `output_path` | `str` or `None` | `None` | If given, CSV output is written here via `OutputLogger` |

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `config` | `dict` | The configuration dict passed at construction |
| `mesh` | `TriMesh` or `HexMesh` | The spatial mesh |
| `env` | `Environment` or `HexSimEnvironment` | Environmental forcing adapter |
| `pool` | `AgentPool` | Vectorised agent state storage |
| `population` | `Population` | Lifecycle-aware population wrapper |
| `history` | `list[dict]` | Per-step summary records appended by the logging event |
| `current_t` | `int` | Current timestep counter |
| `logger` | `OutputLogger` or `None` | Output logger (present when `output_path` is set) |

**Methods**

- **`step()`** — Advance the simulation by one timestep. Calls `env.advance(t)`, builds the `Landscape` dict, then runs the event sequencer.
- **`run(n_steps)`** — Call `step()` `n_steps` times.
- **`close()`** — Flush the output logger and close the environment. Must be called when output logging is enabled.

---

### `salmon_ibm.agents`

> Fish agent state: `FishAgent` (OOP view) and `AgentPool` (vectorised arrays).

#### `Behavior`

Integer enum of salmon behavioural states.

```python
class Behavior(IntEnum):
    HOLD = 0
    RANDOM = 1
    TO_CWR = 2
    UPSTREAM = 3
    DOWNSTREAM = 4
```

| Member | Value | Description |
|--------|-------|-------------|
| `HOLD` | `0` | Stationary holding behaviour |
| `RANDOM` | `1` | Random walk |
| `TO_CWR` | `2` | Moving toward cold-water refuge |
| `UPSTREAM` | `3` | Active upstream migration |
| `DOWNSTREAM` | `4` | Downstream movement |

---

#### `ARRAY_FIELDS`

A tuple of all per-agent array attribute names on `AgentPool`. Used by `Population.compact()` and `Population.add_agents()` to resize all agent arrays together.

```python
ARRAY_FIELDS = (
    "tri_idx", "mass_g", "ed_kJ_g", "target_spawn_hour",
    "behavior", "cwr_hours", "hours_since_cwr", "steps",
    "alive", "arrived", "temp_history",
)
```

---

#### `AgentPool`

Vectorised structure-of-arrays storage for all fish agents.

```python
class AgentPool:
    ARRAY_FIELDS: tuple[str, ...]

    def __init__(
        self,
        n: int,
        start_tri: int | np.ndarray,
        rng_seed: int | None = None,
        mass_mean: float = 3500.0,
        mass_std: float = 500.0,
        ed_init: float = 6.5,
        spawn_hours_mean: float = 720.0,
        spawn_hours_std: float = 168.0,
    ): ...

    def get_agent(self, idx: int) -> FishAgent: ...
    def t3h_mean(self) -> np.ndarray: ...
    def push_temperature(self, temps: np.ndarray) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | — | Number of agents |
| `start_tri` | `int` or `np.ndarray` | — | Starting triangle/cell index (scalar or per-agent array) |
| `rng_seed` | `int` or `None` | `None` | RNG seed for initial state generation |
| `mass_mean` | `float` | `3500.0` | Mean initial body mass (g) |
| `mass_std` | `float` | `500.0` | Standard deviation of initial body mass (g) |
| `ed_init` | `float` | `6.5` | Initial energy density (kJ/g) |
| `spawn_hours_mean` | `float` | `720.0` | Mean hours remaining to target spawning time |
| `spawn_hours_std` | `float` | `168.0` | Standard deviation of hours to spawning |

**Array attributes** (all shape `(n,)` unless noted)

| Attribute | dtype | Description |
|-----------|-------|-------------|
| `tri_idx` | `int` | Current mesh cell index |
| `mass_g` | `float64` | Body mass (g) |
| `ed_kJ_g` | `float64` | Energy density (kJ/g) |
| `target_spawn_hour` | `int` | Countdown to target spawning hour |
| `behavior` | `int` | Current `Behavior` integer |
| `cwr_hours` | `int` | Consecutive hours spent in cold-water refuge |
| `hours_since_cwr` | `int` | Hours since last cold-water refuge visit |
| `steps` | `int` | Total steps taken |
| `alive` | `bool` | Whether agent is alive |
| `arrived` | `bool` | Whether agent has reached spawning grounds |
| `temp_history` | `float64` | Shape `(n, 3)` — last 3 hourly temperatures |

**Methods**

- **`get_agent(idx)`** — Return a `FishAgent` OOP view into agent `idx` (zero-copy).
- **`t3h_mean()`** — Return 3-hour mean temperature for each agent; shape `(n,)`.
- **`push_temperature(temps)`** — Shift `temp_history` left and append `temps` as the newest column.

---

### `salmon_ibm.population`

> Unified lifecycle manager for a named agent collection.

#### `Population`

Dataclass wrapping `AgentPool` with dynamic resizing, group tracking, and optional subsystem managers. All per-agent array access is proxied through the underlying pool.

```python
@dataclass
class Population:
    name: str
    pool: AgentPool
    accumulator_mgr: AccumulatorManager | None = None
    trait_mgr: TraitManager | None = None
    genome: Any = None
    ranges: Any = None
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Identifier for this population |
| `pool` | `AgentPool` | Underlying vectorised state storage |
| `accumulator_mgr` | `AccumulatorManager` or `None` | Optional accumulator subsystem |
| `trait_mgr` | `TraitManager` or `None` | Optional trait subsystem |
| `genome` | `GenomeManager` or `None` | Optional genetics subsystem |
| `ranges` | `RangeAllocator` or `None` | Optional territorial range allocator |

**Read-only properties**

| Property | Type | Description |
|----------|------|-------------|
| `n` | `int` | Total number of agents (alive + dead) |
| `n_alive` | `int` | Count of alive agents |
| `alive` | `np.ndarray[bool]` | Per-agent alive flags |
| `arrived` | `np.ndarray[bool]` | Per-agent arrival flags |
| `tri_idx` | `np.ndarray[int]` | Current cell indices |
| `floaters` | `np.ndarray[bool]` | Alive agents with no group assignment |
| `grouped` | `np.ndarray[bool]` | Alive agents assigned to a group |

**Proxied array properties** (readable and writable, delegating to `pool`)

`behavior`, `ed_kJ_g`, `mass_g`, `steps`, `target_spawn_hour`, `cwr_hours`, `hours_since_cwr`, `temp_history`

**Methods**

```python
def t3h_mean(self) -> np.ndarray: ...
def push_temperature(self, temps: np.ndarray) -> None: ...
def remove_agents(self, indices: np.ndarray) -> None: ...
def compact(self) -> None: ...
def add_agents(
    self,
    n: int,
    positions: np.ndarray,
    *,
    mass_g=None,
    ed_kJ_g: float = 6.5,
    group_id: int = -1,
) -> np.ndarray: ...
```

- **`t3h_mean()`** — Delegates to `pool.t3h_mean()`.
- **`push_temperature(temps)`** — Delegates to `pool.push_temperature(temps)`.
- **`remove_agents(indices)`** — Set `alive[indices] = False`. Does not compact arrays.
- **`compact()`** — Remove dead agents in-place by compacting all arrays. Updates pool, group tracking arrays, and optional managers.
- **`add_agents(n, positions, *, mass_g, ed_kJ_g, group_id)`** — Extend all arrays to accommodate `n` new agents starting at `positions`. Returns an index array of the newly added rows.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | — | Number of agents to add |
| `positions` | `np.ndarray[int]` | — | Cell indices for the new agents |
| `mass_g` | `float` or `None` | `3500.0` | Initial body mass (g) |
| `ed_kJ_g` | `float` | `6.5` | Initial energy density (kJ/g) |
| `group_id` | `int` | `-1` | Group assignment (`-1` = no group) |

---

### `salmon_ibm.bioenergetics`

> Wisconsin Bioenergetics Model — hourly energy budget for non-feeding migrants.

#### `BioParams`

Dataclass holding all bioenergetics parameters. Instantiate directly or via `bio_params_from_config()`.

```python
@dataclass
class BioParams:
    RA: float = 0.00264
    RB: float = -0.217
    RQ: float = 0.06818
    ED_MORTAL: float = 4.0
    T_OPT: float = 16.0
    T_MAX: float = 26.0
    ED_TISSUE: float = 5.0
    MASS_FLOOR_FRACTION: float = 0.5
    activity_by_behavior: dict[int, float] = ...  # {0:1.0, 1:1.2, 2:0.8, 3:1.5, 4:1.0}
```

| Field | Default | Description |
|-------|---------|-------------|
| `RA` | `0.00264` | Respiration intercept (Wisconsin model) |
| `RB` | `-0.217` | Respiration mass exponent |
| `RQ` | `0.06818` | Respiration temperature coefficient |
| `ED_MORTAL` | `4.0` | Energy density below which an agent dies (kJ/g) |
| `T_OPT` | `16.0` | Optimal temperature for respiration (°C) |
| `T_MAX` | `26.0` | Upper lethal temperature (°C) |
| `ED_TISSUE` | `5.0` | Energy density of catabolised tissue (kJ/g) |
| `MASS_FLOOR_FRACTION` | `0.5` | Minimum mass as fraction of current mass (prevents collapse) |
| `activity_by_behavior` | `dict` | Activity multiplier per `Behavior` integer |

---

#### `hourly_respiration`

```python
def hourly_respiration(
    mass_g: np.ndarray,
    temperature_c: np.ndarray,
    activity_mult: np.ndarray,
    params: BioParams,
) -> np.ndarray:
```

Compute hourly respiratory energy expenditure (J) for a cohort of agents using the Wisconsin model: `RA × mass^RB × exp(RQ × T) × activity`, scaled from daily to hourly and converted from gO₂ to Joules.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mass_g` | `np.ndarray` | Body masses (g) |
| `temperature_c` | `np.ndarray` | Water temperatures (°C) |
| `activity_mult` | `np.ndarray` | Activity multipliers from `BioParams.activity_by_behavior` |
| `params` | `BioParams` | Model parameter set |

Returns `np.ndarray` of hourly respiration in Joules per agent.

---

#### `update_energy`

```python
def update_energy(
    ed_kJ_g: np.ndarray,
    mass_g: np.ndarray,
    temperature_c: np.ndarray,
    activity_mult: np.ndarray,
    salinity_cost: np.ndarray,
    params: BioParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

Apply one hourly energy budget step. Respiration (scaled by salinity cost) is subtracted from total energy. Mass shrinks proportionally; energy density is recalculated. Agents below `ED_MORTAL` are marked dead.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ed_kJ_g` | `np.ndarray` | Current energy densities (kJ/g) |
| `mass_g` | `np.ndarray` | Current body masses (g) |
| `temperature_c` | `np.ndarray` | Water temperatures (°C) |
| `activity_mult` | `np.ndarray` | Activity multipliers |
| `salinity_cost` | `np.ndarray` | Salinity-based respiration cost multipliers (≥ 1.0) |
| `params` | `BioParams` | Model parameter set |

Returns `(new_ed_kJ_g, dead_mask, new_mass_g)` — all shape `(n,)`.

---

### `salmon_ibm.behavior`

> Behavioural decision table and overrides (Snyder et al. 2019).

#### `BehaviorParams`

Dataclass holding behavioral decision parameters. Use `BehaviorParams.defaults()` for the standard salmon probability table.

```python
@dataclass
class BehaviorParams:
    temp_bins: list[float] = field(default_factory=lambda: [16.0, 18.0, 20.0])
    time_bins: list[float] = field(default_factory=lambda: [360, 720])
    p_table: np.ndarray | None = None
    max_cwr_hours: int = 48
    avoid_cwr_cooldown_h: int = 12
    max_dist_to_cwr: float = 5000.0

    @classmethod
    def defaults(cls) -> BehaviorParams: ...
```

| Field | Default | Description |
|-------|---------|-------------|
| `temp_bins` | `[16.0, 18.0, 20.0]` | Temperature bin edges (°C) for behaviour lookup |
| `time_bins` | `[360, 720]` | Hours-to-spawn bin edges for behaviour lookup |
| `p_table` | `None` | Shape `(n_time_bins, n_temp_bins, 5)` probability table over `Behavior` values |
| `max_cwr_hours` | `48` | Maximum consecutive hours in cold-water refuge before forced upstream movement |
| `avoid_cwr_cooldown_h` | `12` | Hours after leaving cold-water refuge during which `TO_CWR` is suppressed |
| `max_dist_to_cwr` | `5000.0` | Maximum distance (m) to consider cold-water refuge reachable |

- **`defaults()`** — Return a `BehaviorParams` instance with the standard 3×4×5 probability table calibrated for Chinook salmon migration.

---

#### `pick_behaviors`

```python
def pick_behaviors(
    t3h_mean: np.ndarray,
    hours_to_spawn: np.ndarray,
    params: BehaviorParams,
    seed: int | None = None,
) -> np.ndarray:
```

Sample one behaviour per agent from the probability table, conditioned on 3-hour mean temperature and hours remaining to spawning. Uses Numba JIT if available.

| Parameter | Type | Description |
|-----------|------|-------------|
| `t3h_mean` | `np.ndarray` | 3-hour mean temperatures, shape `(n,)` |
| `hours_to_spawn` | `np.ndarray` | Hours remaining to target spawn, shape `(n,)` |
| `params` | `BehaviorParams` | Behavioural parameter set |
| `seed` | `int` or `None` | RNG seed |

Returns `np.ndarray[int]` of `Behavior` values, shape `(n,)`.

---

#### `apply_overrides`

```python
def apply_overrides(pool: AgentPool, params: BehaviorParams) -> np.ndarray:
```

Apply deterministic overrides on top of stochastic behaviour assignments:

1. Agents on their first step (`steps == 0`) are forced `UPSTREAM`.
2. Agents exceeding `max_cwr_hours` in cold-water refuge are forced `UPSTREAM`.
3. Agents within `avoid_cwr_cooldown_h` hours of leaving cold-water refuge have `TO_CWR` overridden to `UPSTREAM`.

Returns a copy of `pool.behavior` with overrides applied.

---

### `salmon_ibm.events`

> Event engine: base classes, triggers, and sequencer.

#### Trigger Classes

All triggers inherit from `EventTrigger` and implement `should_fire(t: int) -> bool`.

```python
class EveryStep(EventTrigger):
    def should_fire(self, t: int) -> bool: ...   # always True

@dataclass
class Once(EventTrigger):
    at: int
    def should_fire(self, t: int) -> bool: ...   # True only when t == at

@dataclass
class Periodic(EventTrigger):
    interval: int
    offset: int = 0
    def should_fire(self, t: int) -> bool: ...   # True when (t - offset) % interval == 0

@dataclass
class Window(EventTrigger):
    start: int
    end: int
    def should_fire(self, t: int) -> bool: ...   # True when start <= t < end

@dataclass
class RandomTrigger(EventTrigger):
    p: float
    def should_fire(self, t: int) -> bool: ...   # True with probability p
```

| Trigger | Key Parameters | Description |
|---------|---------------|-------------|
| `EveryStep` | — | Fires on every timestep |
| `Once` | `at` | Fires once at timestep `at` |
| `Periodic` | `interval`, `offset=0` | Fires every `interval` steps starting at `offset` |
| `Window` | `start`, `end` | Fires during `[start, end)` |
| `RandomTrigger` | `p` | Fires with probability `p` per step |

---

#### `Event`

Abstract base class for all events.

```python
@dataclass
class Event(ABC):
    name: str
    trigger: EventTrigger = field(default_factory=EveryStep)
    trait_filter: dict | None = None
    population_name: str | None = None
    enabled: bool = True

    @abstractmethod
    def execute(self, population, landscape: Landscape, t: int, mask: np.ndarray) -> None: ...
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | Human-readable event identifier |
| `trigger` | `EventTrigger` | `EveryStep()` | Controls when the event fires |
| `trait_filter` | `dict` or `None` | `None` | Filter agents by trait values before execution |
| `population_name` | `str` or `None` | `None` | Target population by name (multi-pop sequencer only) |
| `enabled` | `bool` | `True` | Disable without removing from sequence |

---

#### `EventSequencer`

Executes a list of events in order each timestep.

```python
class EventSequencer:
    def __init__(self, events: list[Event]): ...
    def step(self, population, landscape: Landscape, t: int) -> None: ...
```

- **`step(population, landscape, t)`** — Populates `landscape["step_alive_mask"]`, then for each enabled event whose trigger fires, computes the agent mask and calls `event.execute()`.

---

#### `EventGroup`

An `Event` subclass that wraps a list of sub-events and runs them for a configurable number of iterations per timestep.

```python
@dataclass
class EventGroup(Event):
    sub_events: list[Event] = field(default_factory=list)
    iterations: int = 1

    def execute(self, population, landscape, t, mask) -> None: ...
```

| Field | Default | Description |
|-------|---------|-------------|
| `sub_events` | `[]` | Ordered list of child events |
| `iterations` | `1` | Number of times to run the sub-event list per step |

---

#### `register_event`

```python
def register_event(type_name: str):
```

Decorator that registers an `Event` subclass in the global `EVENT_REGISTRY` under `type_name`. Registered types can be instantiated from YAML configuration via `load_events_from_config()`.

```python
@register_event("my_event")
class MyEvent(Event):
    ...
```

---

### `salmon_ibm.environment`

> Time-varying environmental fields interpolated onto the triangular mesh.

#### `Environment`

Manages hourly environmental forcing (temperature, salinity, currents, SSH) loaded from NetCDF files and averaged onto mesh triangles.

```python
class Environment:
    def __init__(self, config: dict, mesh: TriMesh, data_dir: str = "data"): ...
    def advance(self, t: int) -> None: ...
    def sample(self, tri_idx: int) -> dict[str, float]: ...
    def gradient(self, field_name: str, tri_idx: int) -> tuple[float, float]: ...
    def dSSH_dt(self, tri_idx: int) -> float: ...
    def dSSH_dt_array(self) -> np.ndarray: ...
    def close(self) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `dict` | — | Full simulation config with a `forcings` section |
| `mesh` | `TriMesh` | — | Triangular mesh |
| `data_dir` | `str` | `"data"` | Directory containing NetCDF forcing files |

**Attribute**

| Attribute | Type | Description |
|-----------|------|-------------|
| `fields` | `dict[str, np.ndarray]` | Current-step field arrays keyed by name (`"temperature"`, `"salinity"`, `"u_current"`, `"v_current"`, `"ssh"`) |

**Methods**

- **`advance(t)`** — Load timestep `t % n_timesteps` from preloaded arrays and write into `fields`. Saves previous SSH for `dSSH_dt` calculations.
- **`sample(tri_idx)`** — Return `{field_name: float}` for all fields at a single triangle.
- **`gradient(field_name, tri_idx)`** — Compute normalized spatial gradient of the named field at triangle `tri_idx` via `mesh.gradient()`. Returns `(dlat, dlon)`.
- **`dSSH_dt(tri_idx)`** — Rate of SSH change (m/timestep) at a single triangle.
- **`dSSH_dt_array()`** — Rate of SSH change for all triangles; shape `(n_triangles,)`.
- **`close()`** — No-op (datasets are closed after preloading in `__init__`).

---

### `salmon_ibm.hexsim_env`

> Zone-based environment adapter for HexSim workspaces. Serves the same `fields` dict interface as `Environment` but sources temperature from a zone lookup table.

#### `HexSimEnvironment`

```python
class HexSimEnvironment:
    def __init__(
        self,
        workspace_dir: str | Path,
        mesh: HexMesh,
        temperature_csv: str = "River Temperature.csv",
    ): ...

    def advance(self, t: int) -> None: ...
    def sample(self, cell_idx: int) -> dict[str, float]: ...
    def gradient(self, field_name: str, cell_idx: int) -> tuple[float, float]: ...
    def dSSH_dt(self, cell_idx: int) -> float: ...
    def dSSH_dt_array(self) -> np.ndarray: ...
    def close(self) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace_dir` | `str` or `Path` | — | Path to the HexSim workspace directory |
| `mesh` | `HexMesh` | — | Water-only compacted hex mesh |
| `temperature_csv` | `str` | `"River Temperature.csv"` | Filename of the River Temperature CSV, relative to `workspace/Analysis/Data Lookup/` |

**Attribute**

| Attribute | Type | Description |
|-----------|------|-------------|
| `fields` | `dict[str, np.ndarray]` | Current fields: `"temperature"`, `"salinity"`, `"ssh"`, `"u_current"`, `"v_current"` |

**Methods**

- **`advance(t)`** — Update `fields["temperature"]` by looking up zone temperatures at `t % n_timesteps`.
- **`sample(cell_idx)`** — All field values at a single cell; returns `{name: float}`.
- **`gradient(field_name, cell_idx)`** — Normalized spatial gradient at a cell via `mesh.gradient()`.
- **`dSSH_dt(cell_idx)`** — Always returns `0.0` (static river gradient).
- **`dSSH_dt_array()`** — Returns a zero array of length `n_cells` (static gradient).
- **`close()`** — No-op.

---

### `salmon_ibm.mesh`

> Triangular mesh constructed from a regular lat/lon grid.

#### `TriMesh`

Unstructured triangular mesh over a 2D domain (e.g. Curonian Lagoon). Nodes are arranged on a regular lat/lon grid; triangles are formed by Delaunay triangulation.

```python
class TriMesh:
    def __init__(
        self,
        nodes: np.ndarray,
        triangles: np.ndarray,
        mask_per_node: np.ndarray,
        depth_per_node: np.ndarray,
        delaunay=None,
    ): ...

    @classmethod
    def from_netcdf(cls, path: str) -> TriMesh: ...

    def water_neighbors(self, tri_idx: int) -> list[int]: ...
    def find_triangle(self, lat: float, lon: float) -> int: ...
    def gradient(self, field: np.ndarray, tri_idx: int) -> tuple[float, float]: ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `nodes` | `np.ndarray` | Shape `(n_nodes, 2)` — `[lat, lon]` coordinates |
| `triangles` | `np.ndarray` | Shape `(n_triangles, 3)` — node indices per triangle |
| `mask_per_node` | `np.ndarray` | Boolean water mask per node |
| `depth_per_node` | `np.ndarray` | Depth per node |
| `delaunay` | `scipy.spatial.Delaunay` or `None` | Pre-built Delaunay object (computed internally if `None`) |

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `nodes` | `np.ndarray` | Node coordinates |
| `triangles` | `np.ndarray` | Triangle node indices |
| `n_triangles` | `int` | Number of triangles |
| `centroids` | `np.ndarray` | Shape `(n_triangles, 2)` — triangle centroid coordinates |
| `areas` | `np.ndarray` | Shape `(n_triangles,)` — triangle areas |
| `water_mask` | `np.ndarray[bool]` | `True` if all 3 nodes of a triangle are water |
| `depth` | `np.ndarray` | Mean depth per triangle |
| `neighbors` | `np.ndarray` | Shape `(n_triangles, 3)` — neighbor triangle indices (`-1` = boundary) |

**Class methods**

- **`from_netcdf(path)`** — Load from a NetCDF file containing `lat`, `lon`, `mask`, and `depth` variables. Builds a Delaunay triangulation automatically.

**Instance methods**

- **`water_neighbors(tri_idx)`** — Return list of water-only neighbor indices for triangle `tri_idx`.
- **`find_triangle(lat, lon)`** — Return the index of the triangle containing `(lat, lon)`.
- **`gradient(field, tri_idx)`** — Compute a normalized spatial gradient of `field` at `tri_idx` using centroid finite differences with cos(lat) correction. Returns `(dlat, dlon)`.

---

### `salmon_ibm.hexsim`

> HexSim workspace loader and `HexMesh` class. Reads EPA HexSim workspaces and constructs a water-only hexagonal mesh that duck-types `TriMesh`.

#### `HexMesh`

Water-only hexagonal mesh. Only water cells are stored (compacted). All array indices refer to the compact water-cell numbering unless stated otherwise. Uses pointy-top hexagons in an odd-row offset grid.

```python
class HexMesh:
    def __init__(
        self,
        centroids: np.ndarray,
        depth: np.ndarray,
        neighbors: np.ndarray,
        areas: np.ndarray,
        water_full_idx: np.ndarray,
        full_to_compact: dict,
        ncols: int,
        nrows: int,
        n_data: int,
        *,
        edge: float = 1.0,
        workspace=None,
    ): ...

    @classmethod
    def from_hexsim(
        cls,
        workspace_dir: str | Path,
        species: str = "chinook",
        extent_layer: str | None = None,
        depth_layer: str | None = None,
    ) -> HexMesh: ...

    @property
    def n_triangles(self) -> int: ...

    def water_neighbors(self, idx: int) -> list[int]: ...
    def find_triangle(self, y: float, x: float) -> int: ...
    def gradient(self, field: np.ndarray, idx: int) -> tuple[float, float]: ...
```

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `centroids` | `np.ndarray` | Shape `(N_water, 2)` — `[y, x]` centroid coordinates in metres |
| `depth` | `np.ndarray` | Shape `(N_water,)` — depth per water cell |
| `neighbors` | `np.ndarray` | Shape `(N_water, 6)` — compact neighbor indices (`-1` = absent) |
| `areas` | `np.ndarray` | Shape `(N_water,)` — hex area in m² |
| `n_cells` | `int` | Number of water cells |
| `water_mask` | `np.ndarray[bool]` | All `True` (every stored cell is water) |
| `n_triangles` | `int` | Alias for `n_cells` (TriMesh compatibility) |

**Class methods**

- **`from_hexsim(workspace_dir, species, extent_layer, depth_layer)`** — Load a HexSim workspace directory and return a `HexMesh`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace_dir` | `str` or `Path` | — | Path to HexSim workspace root |
| `species` | `str` | `"chinook"` | Selects the depth layer by name (`"chinook"` or `"steelhead"`) |
| `extent_layer` | `str` or `None` | `None` | Name of the water extent layer (auto-detected if `None`) |
| `depth_layer` | `str` or `None` | `None` | Name of the depth layer (auto-detected if `None`) |

**Instance methods**

- **`water_neighbors(idx)`** — Water-only neighbors of compact cell `idx`.
- **`find_triangle(y, x)`** — Find nearest water cell to `(y, x)` metric coordinates (KD-tree lookup).
- **`gradient(field, idx)`** — Normalized spatial gradient of `field` at cell `idx` using up to 6 hex neighbors. Returns `(dy, dx)`.

---

### `salmon_ibm.config`

> YAML configuration loader and factory functions.

#### `load_config`

```python
def load_config(path: str | Path) -> dict:
```

Load and validate a simulation configuration from a YAML file. For HexSim configs (`grid.type == "hexsim"`), the workspace path is resolved relative to the config file's directory.

Raises `ValueError` if the config fails validation (see `validate_config`).

---

#### `bio_params_from_config`

```python
def bio_params_from_config(cfg: dict) -> BioParams:
```

Create a `BioParams` instance from the optional `bioenergetics` section of the config dict. Keys present in YAML override dataclass defaults; absent keys keep defaults.

---

#### `behavior_params_from_config`

```python
def behavior_params_from_config(cfg: dict) -> BehaviorParams:
```

Create a `BehaviorParams` instance from the optional `behavior` section. If the section is absent or empty, returns `BehaviorParams.defaults()` (standard salmon probability table).

---

#### `genome_from_config`

```python
def genome_from_config(cfg: dict, n_agents: int) -> GenomeManager | None:
```

Create a `GenomeManager` from the optional `genetics` YAML section. Calls `initialize_random()` unless `initialize_random: false` is set. Returns `None` if no genetics section is present.

---

#### `barrier_map_from_config`

```python
def barrier_map_from_config(cfg: dict, mesh) -> tuple | None:
```

Create a `BarrierMap` and its precomputed arrays from the optional `barriers` YAML section. Returns `(BarrierMap, barrier_arrays)` or `None` if no barriers are configured.

---

#### `network_from_config`

```python
def network_from_config(cfg: dict) -> StreamNetwork | None:
```

Create a `StreamNetwork` from the optional `network` YAML section. Returns `None` if no network section is present.

---

#### `population_config_from_yaml`

```python
def population_config_from_yaml(cfg: dict) -> dict:
```

Extract the `population` sub-dict from a config dict.

---

#### `barrier_config_from_yaml`

```python
def barrier_config_from_yaml(cfg: dict) -> dict | None:
```

Extract the `barriers` sub-dict from a config dict, or `None`.

---

#### `genetics_config_from_yaml`

```python
def genetics_config_from_yaml(cfg: dict) -> dict | None:
```

Extract the `genetics` sub-dict from a config dict, or `None`.

---

### `salmon_ibm.accumulators`

> Accumulator system: general-purpose per-agent floating-point state.

#### `AccumulatorDef`

Definition for a single named accumulator column.

```python
@dataclass
class AccumulatorDef:
    name: str
    min_val: float | None = None
    max_val: float | None = None
    linked_trait: str | None = None
```

| Field | Default | Description |
|-------|---------|-------------|
| `name` | — | Column identifier |
| `min_val` | `None` | Clamp floor applied on every write |
| `max_val` | `None` | Clamp ceiling applied on every write |
| `linked_trait` | `None` | Optional trait name this accumulator mirrors |

---

#### `AccumulatorManager`

Vectorised storage for per-agent accumulators. Data is a 2D NumPy array of shape `(n_agents, n_accumulators)`.

```python
class AccumulatorManager:
    def __init__(self, n_agents: int, definitions: list[AccumulatorDef]): ...
    def index_of(self, name: str) -> int: ...
    def get(self, key: str | int) -> np.ndarray: ...
    def set(
        self,
        key: str | int,
        values: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None: ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_agents` | `int` | Number of agents |
| `definitions` | `list[AccumulatorDef]` | Ordered list of accumulator definitions |

**Methods**

- **`index_of(name)`** — Return the column index for accumulator `name`.
- **`get(key)`** — Return the full column array for accumulator `key` (name or integer index); shape `(n_agents,)`.
- **`set(key, values, mask=None)`** — Write `values` to accumulator `key`, applying `min_val`/`max_val` clamping. If `mask` is given, only writes to masked rows.

---

#### Updater Functions

All updater functions operate in-place on an `AccumulatorManager`. The `mask` parameter is a boolean `np.ndarray` selecting the agents to update.

| Function | Signature | Description |
|----------|-----------|-------------|
| `updater_clear` | `(manager, acc_name, mask)` | Reset to zero (or `min_val`) for masked agents |
| `updater_increment` | `(manager, acc_name, mask, *, amount)` | Add fixed `amount` to accumulator |
| `updater_stochastic_increment` | `(manager, acc_name, mask, *, low, high, rng)` | Add uniform random value in `[low, high)` |
| `updater_expression` | `(manager, acc_name, mask, *, expression, globals_dict=None, rng=None)` | Evaluate algebraic expression with AST safety validation |
| `updater_time_step` | `(manager, acc_name, mask, *, timestep, modulus=None)` | Write current timestep (optionally `% modulus`) |
| `updater_individual_id` | `(manager, acc_name, mask, *, agent_ids)` | Write each agent's unique ID |
| `updater_stochastic_trigger` | `(manager, acc_name, mask, *, probability, rng)` | Write 1.0 with probability `p`, else 0.0 |
| `updater_quantify_location` | `(manager, acc_name, mask, *, hex_map, cell_indices)` | Sample hex-map values at agent cell positions |
| `updater_accumulator_transfer` | `(manager, source_name, target_name, mask, *, fraction=1.0)` | Transfer fraction of one accumulator to another |
| `updater_allocated_hexagons` | `(manager, acc_name, mask, *, range_allocator, agent_indices)` | Count hexagons in each agent's territory |
| `updater_explored_hexagons` | `(manager, acc_name, mask, *, explored_sets, agent_indices)` | Count hexagons in each agent's explored area |
| `updater_group_size` | `(manager, acc_name, mask, *, group_ids)` | Write the size of each agent's group |
| `updater_group_sum` | `(manager, acc_name, source_name, mask, *, group_ids)` | Sum source accumulator across group members |
| `updater_births` | `(manager, acc_name, mask, *, birth_counts)` | Write offspring count per agent |
| `updater_mate_verification` | `(manager, acc_name, mask, *, mate_ids, alive)` | Clear mate accumulator if mate has died |
| `updater_quantify_extremes` | `(manager, acc_name, mask, *, hex_map, cell_indices, mode="max")` | Write min or max hex-map value at agent position |
| `updater_hexagon_presence` | `(manager, acc_name, mask, *, hex_map, cell_indices, threshold=0.0)` | Write 1.0 if hex-map value exceeds threshold |
| `updater_uptake` | `(manager, acc_name, mask, *, hex_map, cell_indices, rate=1.0)` | Extract resource from hex-map into accumulator |
| `updater_individual_locations` | `(manager, acc_name, mask, *, cell_indices)` | Write each agent's current cell index |
| `updater_resources_allocated` | `(manager, acc_name, mask, *, resource_map, range_allocator)` | Resource total in allocated territory |
| `updater_resources_explored` | `(manager, acc_name, mask, *, resource_map, explored_sets)` | Resource total in explored area |
| `updater_subpopulation_assign` | `(manager, acc_name, mask, *, n_select, value, rng)` | Randomly select N agents and assign value |
| `updater_subpopulation_selector` | `(manager, acc_name, mask, *, group_ids, n_per_group, value)` | Select first N agents per group and assign value |
| `updater_trait_value_index` | `(manager, acc_name, mask, *, trait_mgr, trait_name)` | Write each agent's trait category index |
| `updater_data_lookup` | `(manager, acc_name, mask, *, lookup_table, key_acc_name)` | Table lookup keyed by another accumulator's value |

---

### `salmon_ibm.genetics`

> Diploid genetic system: genotype storage, recombination, and mutation.

#### `LocusDefinition`

```python
@dataclass
class LocusDefinition:
    name: str
    n_alleles: int
    position: float = 0.0
```

| Field | Default | Description |
|-------|---------|-------------|
| `name` | — | Locus identifier |
| `n_alleles` | — | Number of distinct alleles at this locus |
| `position` | `0.0` | Chromosomal position (cM) used for linkage calculations |

---

#### `GenomeManager`

Diploid genotype storage and operations for all agents. Genotypes are stored as `int32` array of shape `(n_agents, n_loci, 2)`. Alleles are integer indices in `[0, n_alleles)`.

```python
class GenomeManager:
    def __init__(
        self,
        n_agents: int,
        loci: list[LocusDefinition],
        rng_seed: int | None = None,
    ): ...

    def locus_index(self, name: str) -> int: ...
    def get_locus(self, name: str) -> np.ndarray: ...
    def initialize_random(self, mask: np.ndarray | None = None) -> None: ...
    def homozygosity(
        self,
        locus_name: str | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray: ...
    def recombine(
        self,
        parent1_indices: np.ndarray,
        parent2_indices: np.ndarray,
        offspring_indices: np.ndarray,
    ) -> None: ...
    def mutate(
        self,
        locus_name: str,
        transition_matrix: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> int: ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_agents` | `int` | Number of agents |
| `loci` | `list[LocusDefinition]` | Ordered locus definitions |
| `rng_seed` | `int` or `None` | RNG seed |

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_agents` | `int` | Number of agents |
| `n_loci` | `int` | Number of loci |
| `loci` | `list[LocusDefinition]` | Locus definitions |
| `genotypes` | `np.ndarray[int32]` | Shape `(n_agents, n_loci, 2)` |

**Methods**

- **`locus_index(name)`** — Return the integer index of a named locus.
- **`get_locus(name)`** — Return diploid alleles at locus `name`; shape `(n_agents, 2)`.
- **`initialize_random(mask=None)`** — Draw random alleles uniformly from `[0, n_alleles)` for all (or masked) agents.
- **`homozygosity(locus_name=None, mask=None)`** — Fraction of homozygous loci per individual. If `locus_name` is given, restricts to that locus only. Returns shape `(n_agents,)` or `(mask.sum(),)`.
- **`recombine(parent1_indices, parent2_indices, offspring_indices)`** — Fill offspring genotypes by crossing over gametes from each parent pair. Uses Haldane's mapping function to convert cM distances to crossover probabilities.
- **`mutate(locus_name, transition_matrix, mask=None)`** — Apply per-allele mutation at `locus_name` using a row-stochastic `transition_matrix` of shape `(n_alleles, n_alleles)`. Returns the number of mutations that occurred.

---

### `salmon_ibm.estuary`

> Estuarine extensions: salinity cost, dissolved oxygen avoidance, and seiche pause.

#### `DOState`

Integer enum for dissolved oxygen states.

```python
class DOState(IntEnum):
    OK = 0
    ESCAPE = 1
    LETHAL = 2
```

Module-level aliases: `DO_OK`, `DO_ESCAPE`, `DO_LETHAL`.

---

#### `salinity_cost`

```python
def salinity_cost(
    salinity: np.ndarray,
    S_opt: float = 0.5,
    S_tol: float = 6.0,
    k: float = 0.6,
    max_cost: float = 5.0,
) -> np.ndarray:
```

Compute a respiration cost multiplier (≥ 1.0) for each agent based on ambient salinity. Salinity within the optimal + tolerance range has cost 1.0; excess salinity increases cost linearly by `k` per PSU above the threshold, capped at `max_cost`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `salinity` | — | Salinity values (PSU), shape `(n,)` |
| `S_opt` | `0.5` | Optimal salinity (PSU) |
| `S_tol` | `6.0` | Salinity tolerance range (PSU) above `S_opt` |
| `k` | `0.6` | Cost slope (per PSU excess) |
| `max_cost` | `5.0` | Maximum cost multiplier |

Returns `np.ndarray` of cost multipliers, shape `(n,)`.

---

#### `do_override`

```python
def do_override(
    do_mg_l: np.ndarray,
    lethal: float = 2.0,
    high: float = 4.0,
) -> np.ndarray:
```

Classify each agent's dissolved oxygen level into a `DOState`. NaN values receive `DO_OK`. Requires `lethal <= high`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `do_mg_l` | — | Dissolved oxygen (mg/L), shape `(n,)` |
| `lethal` | `2.0` | Below this level: `DO_LETHAL` |
| `high` | `4.0` | Below this level (and above `lethal`): `DO_ESCAPE` |

Returns `np.ndarray[int]` of `DOState` values, shape `(n,)`.

---

#### `seiche_pause`

```python
def seiche_pause(
    dSSH_dt: np.ndarray,
    thresh: float = 0.02,
) -> np.ndarray:
```

Return a boolean mask indicating which agents experience seiche-forced pausing: `True` where `|dSSH_dt| > thresh`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dSSH_dt` | — | Rate of SSH change (m/timestep), shape `(n,)` |
| `thresh` | `0.02` | Threshold above which movement is paused |

Returns `np.ndarray[bool]`, shape `(n,)`.

---

### `salmon_ibm.ensemble`

> Multiprocessing ensemble runner for replicate simulations.

#### `run_ensemble`

```python
def run_ensemble(
    config: dict,
    n_replicates: int = 10,
    n_agents: int = 1000,
    n_steps: int = 100,
    n_workers: int | None = None,
    base_seed: int | None = None,
) -> list[dict[str, Any]]:
```

Run multiple independent simulation replicates in parallel using `multiprocessing.Pool`. Each replicate is seeded deterministically from `base_seed`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `dict` | — | Simulation configuration dict |
| `n_replicates` | `int` | `10` | Number of independent replicates |
| `n_agents` | `int` | `1000` | Agents per replicate |
| `n_steps` | `int` | `100` | Timesteps per replicate |
| `n_workers` | `int` or `None` | `None` | Parallel processes (`None` = `os.cpu_count()`). Pass `1` to run serially. |
| `base_seed` | `int` or `None` | `None` | Master seed for deterministic replica seeds. `None` = random. |

Returns a `list` of result dicts, one per replicate, each containing:

| Key | Type | Description |
|-----|------|-------------|
| `seed` | `int` | RNG seed used for this replicate |
| `history` | `list[dict]` | Per-step summary records from `Simulation.history` |
| `n_alive` | `int` | Number of surviving agents at end of run |
| `n_arrived` | `int` | Number of agents that reached spawning grounds |

---

### `salmon_ibm.output`

> Track logging and diagnostics output.

#### `OutputLogger`

Accumulates per-agent state snapshots each timestep and writes them to a CSV file on `close()`.

```python
class OutputLogger:
    def __init__(self, path: str, centroids: np.ndarray): ...
    def log_step(self, t: int, pool: AgentPool) -> None: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    def summary(self, t: int, pool: AgentPool) -> dict: ...
    def close(self) -> None: ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Output CSV file path |
| `centroids` | `np.ndarray` | Mesh centroid coordinates, shape `(n_cells, 2)` |

**Methods**

- **`log_step(t, pool)`** — Append state snapshot for all agents at timestep `t`. Records: `time`, `agent_id`, `tri_idx`, `lat`, `lon`, `ed_kJ_g`, `behavior`, `alive`, `arrived`.
- **`to_dataframe()`** — Concatenate all logged steps into a `pd.DataFrame` with the columns above.
- **`summary(t, pool)`** — Return a summary dict for timestep `t` (does not persist). Keys: `time`, `n_alive`, `n_arrived`, `mean_ed`, `behavior_counts`.
- **`close()`** — Write all logged data to the CSV at `path` via `to_dataframe()`.

---

### `salmon_ibm.movement`

> Movement kernels: random walk, directed (upstream/downstream), CWR-seeking, current advection, and barrier resolution.

#### `execute_movement`

```python
def execute_movement(
    pool,
    mesh,
    fields,
    seed=None,
    n_micro_steps=3,
    cwr_threshold=16.0,
    barrier_arrays=None,
) -> None:
```

Execute one timestep of agent movement. Each timestep consists of `n_micro_steps` micro-steps. Each micro-step dispatches agents to the appropriate kernel based on their current `Behavior`:

| Behavior | Kernel | Description |
|----------|--------|-------------|
| `HOLD` | (no movement) | Agent stays in place |
| `RANDOM` | `_step_random_numba` | Move to a random water neighbor |
| `UPSTREAM` | `_step_directed_numba` (descending SSH) | Move toward highest elevation |
| `DOWNSTREAM` | `_step_directed_numba` (ascending SSH) | Move toward lowest elevation |
| `TO_CWR` | `_step_to_cwr_numba` | Move toward coldest water below `cwr_threshold` |

After behavioral movement, current advection is applied via `_advection_numba`. If `barrier_arrays` is provided, barrier resolution follows via `_resolve_barriers_numba`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pool` | `AgentPool` | — | Agent state |
| `mesh` | `TriMesh` or `HexMesh` | — | Spatial mesh |
| `fields` | `dict[str, np.ndarray]` | — | Environmental fields (`"temperature"`, `"ssh"`, `"u_current"`, `"v_current"`) |
| `seed` | `int` or `None` | `None` | RNG seed for stochastic movement |
| `n_micro_steps` | `int` | `3` | Number of sub-steps per timestep |
| `cwr_threshold` | `float` | `16.0` | Temperature threshold (°C) defining cold-water refuge |
| `barrier_arrays` | `tuple` or `None` | `None` | `(mortality, deflection, transmission)` arrays from `BarrierMap.to_arrays()` |

All Numba kernels are compiled with `@njit(cache=True, parallel=True)` using `prange` for parallel execution.

---

### `salmon_ibm.events_builtin`

> Built-in event types for the default salmon simulation pipeline.

#### `MovementEvent`

```python
@register_event("movement")
@dataclass
class MovementEvent(Event):
    n_micro_steps: int = 3
    cwr_threshold: float = 16.0
```

Wraps `execute_movement()` as an event. Reads `barrier_arrays` from the landscape.

---

#### `SurvivalEvent`

```python
@register_event("survival")
@dataclass
class SurvivalEvent(Event):
    bio_params: BioParams = field(default_factory=BioParams)
    thermal: bool = True
    starvation: bool = True
```

Applies the Wisconsin bioenergetics model (`update_energy`) plus optional thermal mortality (`T >= T_MAX`). Marks dead agents via `alive = False`.

---

#### `StageSpecificSurvivalEvent`

```python
@register_event("stage_survival")
@dataclass
class StageSpecificSurvivalEvent(Event):
    trait_name: str = ""
    mortality_rates: dict = field(default_factory=dict)
    density_dependent: bool = False
```

Apply per-stage mortality rates based on a categorical trait. Optionally density-dependent.

---

#### `IntroductionEvent`

```python
@register_event("introduction")
@dataclass
class IntroductionEvent(Event):
    n_agents: int = 0
    positions: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    initial_mass_mean: float = 3500.0
    initial_mass_std: float = 500.0
    initial_ed: float = 6.5
    initial_traits: dict = field(default_factory=dict)
    initial_accumulators: dict = field(default_factory=dict)
```

Introduce new agents into the population at specified positions.

---

#### `ReproductionEvent`

```python
@register_event("reproduction")
@dataclass
class ReproductionEvent(Event):
    offspring_per_pair: int = 2
    mate_distance: int = 1
```

Pair adjacent agents and produce offspring with genetic recombination if a `GenomeManager` is present.

---

#### `CustomEvent`

```python
@register_event("custom")
@dataclass
class CustomEvent(Event):
    callback: Callable = field(default_factory=lambda: lambda *a: None)
```

Wraps an arbitrary callable as an event.

---

### `salmon_ibm.events_hexsim`

> HexSim-compatible events with Numba JIT kernels for performance-critical operations.

#### `HexSimAccumulateEvent`

```python
@register_event("accumulate")
@dataclass
class HexSimAccumulateEvent(Event):
    updater_functions: list = field(default_factory=list)
```

Run a sequence of accumulator updater functions. Each entry in `updater_functions` specifies the updater name and parameters.

---

#### `HexSimSurvivalEvent`

```python
@register_event("hexsim_survival")
@dataclass
class HexSimSurvivalEvent(Event):
    survival_accumulator: str = ""
```

Kill agents whose survival accumulator value indicates death. Uses HexSim expression evaluation.

---

#### `HexSimMoveEvent`

```python
@register_event("move")
@dataclass
class HexSimMoveEvent(Event):
    move_strategy: str = "onlyDisperse"
    dispersal_spatial_data: str = ""
    walk_up_gradient: bool = False
    dispersal_accumulator: str = ""
    distance_accumulator: str = ""
    dispersal_use_affinity: object = None
    dispersal_halt_target: float = 0.0
    resource_threshold: float = 0.0
```

HexSim movement with gradient-following, affinity targets, and configurable dispersal strategies. Caches mesh/gradient references for performance. Uses Numba JIT kernels `_move_gradient_numba` and `_move_affinity_numba`.

| Strategy | Description |
|----------|-------------|
| `"onlyDisperse"` | Random dispersal movement |
| `"walkUpGradient"` | Follow ascending spatial data gradient |
| `"walkDownGradient"` | Follow descending spatial data gradient |
| `"moveToAffinityTarget"` | Move toward an affinity target cell |

---

#### `PatchIntroductionEvent`

```python
@register_event("patch_introduction")
@dataclass
class PatchIntroductionEvent(Event):
    patch_spatial_data: str = ""
```

Introduce agents at cells selected by a spatial data layer.

---

#### `DataLookupEvent`

```python
@register_event("data_lookup")
@dataclass
class DataLookupEvent(Event):
    file_name: str = ""
    row_accumulator: str = ""
    column_accumulator: str = ""
    target_accumulator: str = ""
    lookup_table: np.ndarray | None = field(default=None, repr=False)
```

Look up values in a 2D CSV table using two accumulator values as row/column keys, and write the result to a target accumulator.

---

#### `SetSpatialAffinityEvent`

```python
@register_event("set_spatial_affinity")
@dataclass
class SetSpatialAffinityEvent(Event):
    affinity_name: str = ""
    strategy: str = "better"
    spatial_series: str = ""
    error_accumulator: str = ""
    min_accumulator: str = ""
    max_accumulator: str = ""
```

Set per-agent spatial affinity targets by evaluating neighbor cells against a strategy. Uses Numba JIT kernel `_set_affinity_numba`.

---

#### `clear_combo_mask_cache`

```python
def clear_combo_mask_cache() -> None:
```

Clear the cached trait-combination masks. Should be called when trait values change between steps.

---

### `salmon_ibm.events_phase3`

> Phase 3 event types: mutation, trait transition, generated hex maps, range dynamics.

#### `MutationEvent`

```python
@register_event("mutation")
@dataclass
class MutationEvent(Event):
    locus_name: str = ""
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
```

Apply allele mutations at a named locus using a row-stochastic transition matrix of shape `(n_alleles, n_alleles)`.

---

#### `TransitionEvent`

```python
@register_event("transition")
@dataclass
class TransitionEvent(Event):
    trait_name: str = ""
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
```

Apply probabilistic state transitions on a categorical trait (e.g., SEIR disease states or life stages). Transition matrix shape: `(n_categories, n_categories)`.

---

#### `GeneratedHexmapEvent`

```python
@register_event("generated_hexmap")
@dataclass
class GeneratedHexmapEvent(Event):
    expression: str = ""
    output_name: str = ""
```

Create a per-cell spatial data layer by evaluating an algebraic expression referencing global variables, accumulators, and existing spatial data.

---

#### `RangeDynamicsEvent`

```python
@register_event("range_dynamics")
@dataclass
class RangeDynamicsEvent(Event):
    mode: str = "expand"
    resource_map_name: str = "resources"
    resource_threshold: float = 0.0
    max_range_size: int = 50
```

Expand or contract agent territories via BFS based on resource availability.

| Mode | Description |
|------|-------------|
| `"expand"` | Add neighboring cells to territory (up to `max_range_size`) |
| `"contract"` | Remove low-resource cells from territory |

---

#### `SetAffinityEvent`

```python
@register_event("set_affinity")
@dataclass
class SetAffinityEvent(Event):
    affinity_type: str = "spatial"
    affinity_map_name: str | None = None
    strength: float = 1.0
```

Set spatial affinity targets for agents from a named spatial data layer.

---

#### `PlantDynamicsEvent`

```python
@register_event("plant_dynamics")
@dataclass
class PlantDynamicsEvent(Event):
    seed_production_rate: float = 5.0
    dispersal_radius: int = 3
    establishment_threshold: float = 0.5
```

Simulate seed production, dispersal, and establishment for plant populations.

---

### `salmon_ibm.event_descriptors`

> Typed descriptor dataclasses for XML event parameter extraction and registry-driven loading.

#### `EventDescriptor`

```python
@dataclass
class EventDescriptor:
    name: str
    event_type: str
    timestep: int
    population_name: str
    enabled: bool = True
    params: dict = field(default_factory=dict)
```

Base descriptor for all event types. Specialized subclasses extract type-specific parameters from XML.

---

#### Specialized Descriptors

| Descriptor | Event Type | Key Fields |
|-----------|------------|------------|
| `MoveEventDescriptor` | `"move"` | `move_type`, `max_steps`, `affinity_name` |
| `SurvivalEventDescriptor` | `"hexsim_survival"` | `survival_expression`, `accumulator_refs` |
| `AccumulateEventDescriptor` | `"accumulate"` | `updater_functions` (list of dicts) |
| `TransitionEventDescriptor` | `"transition"` | `trait_name`, `transition_matrix` |
| `CensusEventDescriptor` | `"census"` | (no extra fields) |
| `IntroductionEventDescriptor` | `"introduction"` | `n_agents`, `initialization_spatial_data` |
| `PatchIntroductionEventDescriptor` | `"patch_introduction"` | `n_agents`, `spatial_data` |
| `DataProbeEventDescriptor` | `"data_probe"` | (no extra fields) |
| `DataLookupEventDescriptor` | `"data_lookup"` | `lookup_file`, `accumulator_name` |
| `SetSpatialAffinityEventDescriptor` | `"set_spatial_affinity"` | `affinity_name`, `spatial_data` |
| `InteractionEventDescriptor` | `"interaction"` | `interaction_type` |
| `ReanimationEventDescriptor` | `"reanimation"` | (no extra fields) |

---

#### `DESCRIPTOR_REGISTRY`

```python
DESCRIPTOR_REGISTRY: dict[str, type[EventDescriptor]]
```

Maps event type names to their descriptor classes. Used by `from_descriptor()` for registry-driven event loading.

---

### `salmon_ibm.traits`

> Categorical per-agent trait system with four derivation modes.

#### `TraitType`

```python
class TraitType(Enum):
    PROBABILISTIC = "probabilistic"
    ACCUMULATED = "accumulated"
    GENETIC = "genetic"
    GENETIC_ACCUMULATED = "genetic_accumulated"
```

| Type | Description |
|------|-------------|
| `PROBABILISTIC` | Random categorical assignment |
| `ACCUMULATED` | Binned from an accumulator value using thresholds |
| `GENETIC` | Mapped from a single-locus diploid genotype via phenotype map |
| `GENETIC_ACCUMULATED` | Weighted sum across multiple loci, then binned |

---

#### `TraitDefinition`

```python
@dataclass
class TraitDefinition:
    name: str
    trait_type: TraitType
    categories: list[str]
    accumulator_name: str | None = None
    thresholds: np.ndarray | None = None
    locus_name: str | None = None
    phenotype_map: np.ndarray | None = None
    locus_names: list[str] | None = None
    locus_weights: np.ndarray | None = None
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Trait identifier |
| `categories` | `list[str]` | Category names (e.g., `["juvenile", "adult"]`) |
| `accumulator_name` | `str` or `None` | Source accumulator (for `ACCUMULATED` type) |
| `thresholds` | `np.ndarray` or `None` | Ascending bin edges; length = `len(categories) - 1` |
| `locus_name` | `str` or `None` | Genetic locus (for `GENETIC` type) |
| `phenotype_map` | `np.ndarray` or `None` | `int[n_alleles, n_alleles]` -> category index |
| `locus_names` | `list[str]` or `None` | Multiple loci (for `GENETIC_ACCUMULATED`) |
| `locus_weights` | `np.ndarray` or `None` | Per-locus weights for weighted sum |

---

#### `TraitManager`

```python
class TraitManager:
    def __init__(self, n_agents: int, definitions: list[TraitDefinition]): ...

    def get(self, name: str) -> np.ndarray: ...
    def set(self, name: str, values: np.ndarray, mask: np.ndarray | None = None) -> None: ...
    def category_names(self, name: str) -> list[str]: ...
    def evaluate_accumulated(self, name: str, acc_manager, mask: np.ndarray | None = None) -> None: ...
    def evaluate_genetic(self, name: str, genome_manager, mask: np.ndarray | None = None) -> None: ...
    def evaluate_genetic_accumulated(self, name: str, genome_manager, mask: np.ndarray | None = None) -> None: ...
    def filter_by_traits(self, **criteria) -> np.ndarray: ...
```

**Methods**

- **`get(name)`** — Return category indices for all agents; shape `(n_agents,)`.
- **`set(name, values, mask=None)`** — Write category indices. If `mask` is given, only writes to masked rows.
- **`category_names(name)`** — Return the list of category names for trait `name`.
- **`evaluate_accumulated(name, acc_manager, mask=None)`** — Rebin agents into categories using accumulator values and thresholds.
- **`evaluate_genetic(name, genome_manager, mask=None)`** — Map diploid genotype to category using the phenotype map.
- **`evaluate_genetic_accumulated(name, genome_manager, mask=None)`** — Compute weighted multi-locus sum, then bin into categories.
- **`filter_by_traits(**criteria)`** — Return a boolean mask of agents matching all trait criteria. Example: `filter_by_traits(stage=0, sex=1)`.

---

### `salmon_ibm.barriers`

> Edge-based movement barriers with probabilistic outcomes.

#### `BarrierOutcome`

```python
class BarrierOutcome(NamedTuple):
    p_mortality: float
    p_deflection: float
    p_transmission: float
```

Probabilities for a single crossing attempt. Must satisfy `p_mortality + p_deflection + p_transmission = 1.0` (transmission is the implicit complement).

**Static constructors**

- **`BarrierOutcome.impassable()`** — Returns `(0.0, 1.0, 0.0)` — always deflected.
- **`BarrierOutcome.lethal()`** — Returns `(1.0, 0.0, 0.0)` — always killed.

---

#### `BarrierClass`

```python
@dataclass
class BarrierClass:
    name: str
    forward: BarrierOutcome
    reverse: BarrierOutcome
```

Defines directional outcomes for a named barrier type (e.g., `"dam"`).

---

#### `BarrierMap`

```python
class BarrierMap:
    def __init__(self): ...

    def add_edge(self, from_cell: int, to_cell: int, outcome: BarrierOutcome) -> None: ...
    def check(self, from_cell: int, to_cell: int) -> BarrierOutcome | None: ...
    def has_barriers(self) -> bool: ...

    @property
    def n_edges(self) -> int: ...

    @classmethod
    def from_hbf(cls, path, mesh, class_config=None): ...

    def to_arrays(self, mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
```

**Methods**

- **`add_edge(from_cell, to_cell, outcome)`** — Register a barrier on the edge from `from_cell` to `to_cell`.
- **`check(from_cell, to_cell)`** — Return the `BarrierOutcome` for this edge, or `None` if no barrier.
- **`has_barriers()`** — True if any barriers are registered.
- **`from_hbf(path, mesh, class_config=None)`** — Load barriers from a HexSim `.hbf` text file. Maps barrier classes to outcomes using `class_config`.
- **`to_arrays(mesh)`** — Precompute `(mortality, deflection, transmission)` arrays of shape `(n_cells, max_neighbors)` for use in Numba movement kernels.

---

### `salmon_ibm.network`

> 1D stream network topology with segment-based movement.

#### `SegmentDefinition`

```python
@dataclass
class SegmentDefinition:
    id: int
    length: float
    upstream_ids: list[int] = field(default_factory=list)
    downstream_ids: list[int] = field(default_factory=list)
    order: int = 1
```

| Field | Description |
|-------|-------------|
| `id` | Unique segment identifier |
| `length` | Segment length in metres |
| `upstream_ids` | IDs of upstream-connected segments |
| `downstream_ids` | IDs of downstream-connected segments |
| `order` | Strahler stream order |

---

#### `StreamNetwork`

```python
class StreamNetwork:
    def __init__(self, segments: list[SegmentDefinition]): ...

    def segment_length(self, seg_id: int) -> float: ...
    def upstream(self, seg_id: int) -> list[int]: ...
    def downstream(self, seg_id: int) -> list[int]: ...
    def is_headwater(self, seg_id: int) -> bool: ...
    def is_outlet(self, seg_id: int) -> bool: ...
    def all_upstream(self, seg_id: int) -> list[int]: ...
    def all_downstream(self, seg_id: int) -> list[int]: ...
```

- **`all_upstream(seg_id)`** / **`all_downstream(seg_id)`** — BFS traversal returning all reachable segments in the specified direction.

---

#### `NetworkPosition`

```python
@dataclass
class NetworkPosition:
    segment_id: int
    offset: float
```

Position within a segment: `offset` is the distance from the start of the segment.

---

#### `NetworkMovement`

```python
class NetworkMovement:
    def __init__(self, network: StreamNetwork, rng_seed: int | None = None): ...

    def move_upstream(self, positions: list[NetworkPosition],
                      step_lengths: np.ndarray) -> list[NetworkPosition]: ...
    def move_downstream(self, positions: list[NetworkPosition],
                        step_lengths: np.ndarray) -> list[NetworkPosition]: ...
```

Handles segment boundary crossings when agents move beyond the end of a segment.

---

#### `SwitchPopulationEvent`

```python
@register_event("switch_population")
@dataclass
class SwitchPopulationEvent(Event):
    source_pop: str = ""
    target_pop: str = ""
    transfer_probability: float = 0.1
```

Transfer agents between named populations with a given probability.

---

### `salmon_ibm.ranges`

> Non-overlapping territorial range allocator.

#### `AgentRange`

```python
@dataclass
class AgentRange:
    owner: int
    cells: set[int] = field(default_factory=set)
    resource_total: float = 0.0
```

A single agent's territory: the set of owned cells and total resources within them.

---

#### `RangeAllocator`

```python
class RangeAllocator:
    def __init__(self, mesh): ...

    @property
    def n_occupied(self) -> int: ...

    def get_range(self, agent_idx: int) -> AgentRange | None: ...
    def owner_of(self, cell_id: int) -> int: ...
    def is_available(self, cell_id: int) -> bool: ...
    def allocate_cell(self, agent_idx: int, cell_id: int) -> bool: ...
    def release_cell(self, agent_idx: int, cell_id: int) -> None: ...
    def release_all(self, agent_idx: int) -> None: ...
    def expand_range(self, agent_idx: int, resource_map: np.ndarray,
                     resource_threshold: float = 0.0,
                     max_cells: int = 50) -> int: ...
    def contract_range(self, agent_idx: int, resource_map: np.ndarray,
                       resource_threshold: float = 0.0) -> int: ...
    def compute_resources(self, agent_idx: int, resource_map: np.ndarray) -> float: ...
    def summary(self) -> dict: ...
```

**Methods**

- **`allocate_cell(agent_idx, cell_id)`** — Claim a cell for an agent. Returns `False` if already owned.
- **`expand_range(agent_idx, resource_map, threshold, max_cells)`** — BFS expansion adding available neighbor cells above `threshold`. Returns count of cells added.
- **`contract_range(agent_idx, resource_map, threshold)`** — Release cells with resource below `threshold`. Returns count of cells released.
- **`compute_resources(agent_idx, resource_map)`** — Sum resource values across the agent's territory.

---

### `salmon_ibm.interactions`

> Multi-population manager and interaction events.

#### `MultiPopulationManager`

```python
class MultiPopulationManager:
    def __init__(self): ...

    def register(self, name_or_population, population=None) -> None: ...
    def get(self, name: str): ...
    def build_cell_index(self, name: str) -> None: ...
    def agents_at_cell(self, name: str, cell_id: int) -> np.ndarray: ...
    def co_located_pairs(self, pop_a: str, pop_b: str) -> list[tuple[np.ndarray, np.ndarray]]: ...
```

- **`co_located_pairs(pop_a, pop_b)`** — Find all pairs of agents from two populations sharing the same cell. Returns list of `(indices_a, indices_b)` tuples per shared cell.

---

#### `InteractionOutcome`

```python
class InteractionOutcome(Enum):
    PREDATION = "predation"
    COMPETITION = "competition"
    DISEASE = "disease"
```

---

#### `InteractionEvent`

```python
@register_event("interaction")
@dataclass
class InteractionEvent(Event):
    pop_a_name: str = ""
    pop_b_name: str = ""
    encounter_probability: float = 0.1
    outcome: InteractionOutcome = InteractionOutcome.PREDATION
    resource_gain_acc: str | None = None
    resource_gain_amount: float = 0.0
    penalty_acc: str | None = None
    penalty_amount: float = 0.0
```

Probabilistic interaction between co-located agents from two populations. Depending on `outcome`, one agent may die (predation), lose resources (competition), or receive a penalty accumulator (disease).

---

### `salmon_ibm.reporting`

> Report and tally framework for simulation output.

#### `Report`

```python
@dataclass
class Report:
    name: str
    records: list[dict] = field(default_factory=list)

    def record(self, data: dict) -> None: ...
    def to_csv(self, path: str | Path) -> None: ...
    def clear(self) -> None: ...
```

Base class for time-series reports. Subclasses implement `update()` methods.

---

#### Report Subclasses

| Class | Key Fields Recorded |
|-------|-------------------|
| `ProductivityReport` | `time`, `n_alive`, `n_births`, `n_deaths`, `lambda` |
| `DemographicReport` | `time`, `n_alive`, `n_dead`, `mean_mass`, `mean_ed` |
| `DispersalReport` | `time`, per-agent displacement from origin |
| `GeneticReport` | `time`, allele frequencies, heterozygosity per locus |

---

#### `Tally`

```python
@dataclass
class Tally:
    name: str
    n_cells: int
    data: np.ndarray = field(init=False)

    def reset(self) -> None: ...
    def to_array(self) -> np.ndarray: ...
```

Base class for per-cell spatial tallies. Shape: `(n_cells,)`.

---

#### Tally Subclasses

| Class | What It Tallies |
|-------|----------------|
| `OccupancyTally` | Count of timesteps each cell was occupied |
| `DensityTally` | Cumulative agent count per cell |
| `DispersalFluxTally` | Agent movement between cells |
| `BarrierTally` | Barrier encounter outcomes per cell |

---

#### `ReportManager`

```python
class ReportManager:
    def __init__(self): ...

    def add_report(self, report: Report) -> None: ...
    def add_tally(self, tally: Tally) -> None: ...
    def get_report(self, name: str) -> Report | None: ...
    def get_tally(self, name: str) -> Tally | None: ...
    def save_all(self, output_dir: str | Path) -> None: ...
    def summary(self) -> dict: ...
```

Collects reports and tallies. `save_all()` writes all reports as CSVs and tallies as NumPy arrays to the output directory.

---

### `salmon_ibm.xml_parser`

> Parse real HexSim `.xml` scenario files into Python dicts.

#### `load_scenario_xml`

```python
def load_scenario_xml(path: str | Path) -> dict:
```

Parse a HexSim scenario XML file and return a dict with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `simulation` | `dict` | `n_timesteps`, `n_replicates`, `start_log_step` |
| `grid` | `dict` | `n_hexagons`, `rows`, `columns`, `narrow` (bool), `cell_width` |
| `workspace` | `str` | Workspace directory path |
| `global_variables` | `dict[str, float]` | Named global variables |
| `spatial_data_series` | `dict[str, dict]` | Named spatial data with `datatype`, `time_series`, `cycle_length` |
| `populations` | `list[dict]` | Population definitions with accumulators, traits, affinities, ranges |
| `events` | `list[dict]` | Recursive event tree with triggers, filters, and parameters |

---

#### `build_events_from_xml`

```python
def build_events_from_xml(xml_config: dict, callback_registry: dict | None = None) -> list:
```

Convert the `events` list from `load_scenario_xml` output into a list of `Event` instances using `EVENT_REGISTRY` and `DESCRIPTOR_REGISTRY`.

---

### `salmon_ibm.scenario_loader`

> End-to-end XML-driven simulation setup for real HexSim scenarios.

#### `ScenarioLoader`

```python
class ScenarioLoader:
    def load(
        self, workspace_dir: str, scenario_xml: str, rng_seed: int | None = None
    ) -> HexSimSimulation: ...
```

Orchestrates full scenario loading:
1. Parse XML scenario via `load_scenario_xml`
2. Create `HexMesh` from workspace
3. Create `HexSimEnvironment`
4. Build spatial data registry from workspace hex-maps
5. Create multi-population setup from XML population definitions
6. Build event tree from XML event definitions
7. Load lookup tables from workspace CSVs
8. Return fully configured `HexSimSimulation`

---

#### `HexSimSimulation`

```python
class HexSimSimulation:
    def __init__(self, populations, sequencer, environment, landscape, n_timesteps): ...

    def step(self) -> None: ...
    def run(self, n_steps=None) -> None: ...
```

Top-level simulation runner for XML-loaded HexSim scenarios. Supports multiple named populations with a shared event sequencer.

- **`step()`** — Advance environment by one timestep, then run the event sequencer.
- **`run(n_steps=None)`** — Run for `n_steps` (or `n_timesteps` from XML if `None`).

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `populations` | `dict[str, Population]` | Named populations |
| `environment` | `HexSimEnvironment` | Zone-based environment |
| `landscape` | `Landscape` | Shared event context |
| `history` | `list[dict]` | Per-step summary records |

---

### `salmon_ibm.hexsim_expr`

> HexSim expression DSL translator and safe evaluator.

#### `translate_hexsim_expr`

```python
def translate_hexsim_expr(expr: str) -> str:
```

Translate a HexSim DSL expression into safe Python. Transformations:

| HexSim syntax | Python equivalent | Description |
|--------------|------------------|-------------|
| `'single quoted'` | `_g["single quoted"]` | Global variable reference |
| `"double quoted"` | `_a["double quoted"]` | Accumulator reference |
| `Cond(a, b)` | `_cond(a, b)` | Sign-of-difference |
| `Floor(x)` | `floor(x)` | Floor function |
| `Pow(a, b)` | `pow(a, b)` | Power |
| `GasDev()` | `_gasdev()` | Standard normal random |
| `Rand()` | `_rand()` | Uniform random [0, 1) |

The result is AST-validated before use to prevent code injection.

---

#### `build_hexsim_namespace`

```python
def build_hexsim_namespace(globals_dict, acc_dict, rng, n_masked) -> dict:
```

Build a safe namespace for `eval()` with:
- `_g`: global variables dict
- `_a`: accumulator values dict
- HexSim functions: `_cond`, `Floor`, `Pow`, `Exp`, `Max`, `Min`, `GasDev`, `Rand`
- Math functions: `sqrt`, `abs`, `exp`, `log`, `sin`, `cos`, `tan`, `minimum`, `maximum`, `clip`, `where`, `pi`, `e`
