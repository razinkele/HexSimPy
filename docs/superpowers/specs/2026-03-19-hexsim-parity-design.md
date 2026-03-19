# HexSim Parity: Closing All Critical Gaps — Design Spec

**Date:** 2026-03-19
**Scope:** Fix all 18 critical gaps identified in the parity audit so the Python engine can load and run real HexSim scenario XML files (specifically the Columbia River `gr_Columbia2017B.xml`).

---

## Current State

The Python implementation has the architectural pieces (event engine, accumulators, traits, populations) but the XML parser cannot actually read real HexSim files due to wrong tag names, missing attribute parsing, and missing event types. The gap analysis identified 18 critical, 15 important, and 4 nice-to-have gaps.

**Real scenario complexity (gr_Columbia2017B.xml):**
- 4 populations (Chinook, Steelhead, Iterator, Refuges)
- ~65 accumulators per fish population
- ~30 traits (mix of probabilistic + accumulated)
- ~360 events using 13 distinct event types
- ~290 expression updater calls using a HexSim-specific DSL
- 42 global variable constants
- 17 spatial data series

---

## Phase Structure

| Phase | What | Critical Gaps Closed | Testable Milestone |
|-------|------|---------------------|-------------------|
| **A: XML Parser Rewrite** | Fix tag names, parse traits/accumulators/events from real XML | #1-4, #18, #22-24 | `load_scenario_xml()` returns correct counts for Columbia scenario |
| **B: Expression DSL Translator** | Translate HexSim expression syntax to evaluable Python | #5-6, #17 | All 290 expressions from Columbia XML evaluate without error |
| **C: Multi-Population Router** | Route events to named populations | #13-14 | EventSequencer dispatches to 4 named populations |
| **D: Missing Event Types** | `moveEvent`, `survivalEvent`, `dataLookupEvent`, `setSpatialAffinityEvent`, `patchIntroductionEvent` | #7-12, #15-16 | All 13 event types instantiate from parsed XML |
| **E: ScenarioLoader** | Wire everything into `load_scenario(workspace, xml)` | End-to-end | Columbia scenario loads, initializes, runs 10 steps |

---

## Phase A: XML Parser Rewrite

### Problem

The current `xml_parser.py` uses wrong tag names and misses most of the real HexSim XML structure:
- `<simulationParameters><timesteps>` parsed as `<simulation><timeSteps>`
- `<initialSize>` parsed as `<initialCount>`
- `<probabilisticTrait>` and `<accumulatedTrait>` never matched (parser uses `<trait>`)
- Accumulator attributes (`lowerBound`, `upperBound`, `birthLower`, `birthUpper`, `inherit`) read from child elements instead of XML attributes
- Event nesting (`<eventGroupEvent>` containing `<event>` children) flattened
- `<event timestep="N">` and `eventOff="True"` attributes ignored
- `<populationName>` inside events not extracted
- `<spatialDataSeries>` declarations not parsed
- `<globalVariables>` not parsed
- `<hexagonGrid>` metadata not parsed

### Design

Rewrite `xml_parser.py` as `salmon_ibm/xml_parser.py` (same file, new implementation). The output dict structure changes to support the richer data:

```python
{
    "simulation": {
        "n_timesteps": 2928,
        "n_replicates": 1,
        "start_log_step": 0,
    },
    "grid": {
        "n_hexagons": 880000,
        "cell_width": 27.752,
        "columns": 1100,
        "rows": 800,
    },
    "global_variables": {
        "Hexagon Area": 500.0,
        "Fish Respiration alpha": 0.00264,
        ...
    },
    "spatial_data_series": {
        "River [ extent ]": {"datatype": "HexMap", "time_series": False},
        "Fish Ladder Available": {"datatype": "Barrier", "time_series": True, "cycle_length": 24},
        ...
    },
    "populations": [
        {
            "name": "Chinook",
            "type": "terrestrial",
            "initial_size": 0,
            "initialization_spatial_data": "",
            "exclusion_layer": "River [ extent ]",
            "exclude_if_zero": True,
            "affinities": [...],
            "accumulators": [
                # Note: field names match AccumulatorDef: min_val, max_val
                # Phase E maps XML lower_bound→min_val, upper_bound→max_val
                # birth_lower/birth_upper/inherit are stored but not yet in AccumulatorDef
                {"name": "Fitness [ weight ]", "min_val": 0, "max_val": 0,
                 "birth_lower": 0, "birth_upper": 0, "holds_id": False, "inherit": False},
                ...
            ],
            "traits": [
                {"name": "Fish Status [ movement ]", "type": "probabilistic",
                 "categories": [
                     {"name": "Do Not Move", "init": 100, "birth": 100},
                     {"name": "Move Randomly", "init": 0, "birth": 0},
                     ...
                 ]},
                {"name": "Fish Status [ thermal ]", "type": "accumulated",
                 "accumulator": "Temperature [ mean ]",
                 "categories": [
                     {"name": "Below 16 Degrees", "threshold": float("-inf")},
                     {"name": "16 Degrees", "threshold": 16.0},
                     ...
                 ]},
                ...
            ],
            "range_parameters": {
                "resources_target": 0, "range_threshold": 0,
                "max_indiv_in_group": 1, "max_range_hectares": 0.05,
                "range_spatial_data": "River [ extent ]",
                ...
            },
        },
        ...  # Steelhead, Iterator, Refuges
    ],
    "events": [
        {
            "type": "event_group",
            "name": "Initialize Refuge Population",
            "timestep": 1,  # HexSim semantics: timestep=N means fire ONCE at step N (1-indexed)
            # Events WITHOUT a timestep attribute fire EVERY step.
            # <permanent>false</permanent> means one-shot (fire once then deregister).
            # Convention: timestep=1 + permanent=false → Once(at=0) in Python (0-indexed)
            # No timestep + permanent=true → EveryStep()
            "permanent": False,  # one-shot event
            "population": None,  # group-level; children specify population
            "iterations": 1,
            "enabled": True,
            "sub_events": [
                {
                    "type": "patch_introduction",
                    "name": "Add Refuge Population",
                    "population": "Refuges",
                    "params": {"patch_spatial_data": "Special Sites [ refuges ]"},
                },
                {
                    "type": "accumulate",
                    "name": "Set Refuge ID",
                    "population": "Refuges",
                    "updater_functions": [
                        {"accumulator": "Refuge ID",
                         "function": "IndividualLocations",  # type discriminator for Phase E dispatch
                         "spatial_data": "Special Sites [ refuges ]",
                         "parameters": []},
                    ],
                    # Note: each updater_function dict includes "function" as type discriminator
                    # Phase E uses this to dispatch: "Expression" → translate + eval,
                    # "IndividualLocations" → updater_quantify_location, etc.
                },
            ],
        },
        ...
    ],
}
```

### Key parsing functions to implement

1. `_parse_simulation_params(root)` — find `<simulationParameters>`, read `<timesteps>`, `<startLogStep>`
2. `_parse_grid_metadata(root)` — find `<hexagonGrid>`, read dimensions
3. `_parse_global_variables(root)` — find `<globalVariables>`, build name→value dict
4. `_parse_spatial_data_series(root)` — find all `<spatialDataSeries>`, catalog names and types
5. `_parse_population(elem)` — rewrite: read `<initialSize>`, parse `<probabilisticTrait>` and `<accumulatedTrait>` with correct attributes, parse `<accumulator>` attributes, parse `<affinities>`, `<rangeParameters>`
6. `_parse_events_recursive(root)` — find direct `<event>` children of `<scenario>`, recurse into `<eventGroupEvent>`, handle `timestep=`, `eventOff=`, extract `<populationName>`, parse `<updaterFunction>` elements with `<function>`, `<accumulator>`, `<parameter>`, `<accumulateSpatialData>`
7. `_parse_trait_filter(event_elem)` — extract `<trait>` + `<traitCombinations>` filter specs

### Tests

- `test_parses_simulation_params_from_real_xml()` — timesteps=2928
- `test_parses_four_populations()` — names match
- `test_parses_chinook_accumulators()` — count=~65, attributes present
- `test_parses_chinook_traits()` — probabilistic + accumulated, thresholds including -INF
- `test_parses_global_variables()` — 42 entries
- `test_parses_spatial_data_series()` — 17 entries
- `test_parses_nested_events()` — event groups with sub-events
- `test_parses_updater_functions()` — accumulate events have updater_functions list
- `test_parses_event_timestep_attribute()` — timestep=1 events marked correctly
- `test_parses_event_off_attribute()` — disabled events marked
- `test_parses_population_name_on_events()` — each event carries population name

---

## Phase B: HexSim Expression DSL Translator

### Problem

79% of updater calls use `ExpressionUpdaterFunction` with a HexSim-specific DSL that is incompatible with Python:

| HexSim Syntax | Python Equivalent |
|---------------|-------------------|
| `'Hexagon Area'` (single-quoted global var) | `_globals["Hexagon Area"]` |
| `"Fitness [ weight ]"` (double-quoted accumulator) | `_acc["Fitness [ weight ]"]` |
| `GasDev()` | `_rng.standard_normal()` |
| `Rand()` | `_rng.random()` |
| `Cond(test, true_val, false_val)` | `np.where(test > 0, true_val, false_val)` |
| `Floor(x)` | `np.floor(x)` |
| `Pow(base, exp)` | `np.power(base, exp)` |
| `Exp(x)` (capital E) | `np.exp(x)` |
| `%` modulus | `np.mod(a, b)` (already works) |
| `Min(a, b)` | `np.minimum(a, b)` |
| `Max(a, b)` | `np.maximum(a, b)` |

### Design

Create `salmon_ibm/hexsim_expr.py` — a translator that converts HexSim expression strings to Python-evaluable form.

```python
def translate_hexsim_expr(expr: str) -> str:
    """Translate HexSim expression DSL to Python-evaluable string.

    Transformations:
    1. 'single quoted' → _g["single quoted"]  (global variable lookup)
    2. "double quoted" → _a["double quoted"]   (accumulator lookup)
    3. GasDev() → _rng.standard_normal(_n)
    4. Rand() → _rng.random(_n)
    5. Cond(test, t, f) → np.where(test > 0, t, f)
    6. Floor(x) → np.floor(x)
    7. Pow(b, e) → np.power(b, e)
    8. Exp(x) → np.exp(x)
    """
```

Update `updater_expression()` in `accumulators.py` to:
1. Accept a `globals_dict` parameter (from parsed `<globalVariables>`)
2. Call `translate_hexsim_expr()` before evaluation
3. Inject `_g` (globals), `_a` (accumulators as dict of name→array), `_rng`, `_n` (`mask.sum()`, NOT total agent count — must match masked array lengths) into namespace
4. Extend `_ALLOWED_NODE_TYPES` to include `ast.Subscript`, `ast.Attribute`, `ast.Index` — required for translated `_g["name"]` and `_rng.random(_n)` syntax
5. Add `_rng` attribute access (`standard_normal`, `random`) to allowed function calls

### Tests

- `test_translate_single_quoted_global()` — `'Hexagon Area'` → `_g["Hexagon Area"]`
- `test_translate_double_quoted_accumulator()` — `"Fitness [ weight ]"` → `_a["Fitness [ weight ]"]`
- `test_translate_gasdev()` — `GasDev()` → vectorized normal
- `test_translate_rand()` — `Rand()` → vectorized uniform
- `test_translate_cond()` — `Cond(x - 5, 1, 0)` → where
- `test_translate_nested()` — complex real expression from XML
- `test_evaluate_real_expression()` — end-to-end with AccumulatorManager

---

## Phase C: Multi-Population Event Router

### Problem

Every event in real HexSim targets a specific named population via `<populationName>`. The current `EventSequencer.step()` takes a single population. Multi-population scenarios (4 populations in Columbia) are architecturally impossible.

### Design

**Reuse** the existing `MultiPopulationManager` in `salmon_ibm/interactions.py` (lines 13-66) — do NOT create a duplicate class. It already has `populations: dict`, `register()`, `get()`.

Add `MultiPopEventSequencer` to `salmon_ibm/events.py` (alongside existing `EventSequencer`):

```python
class MultiPopEventSequencer:
    """Executes events, routing each to its target population."""

    def __init__(self, events: list[Event], multi_pop: MultiPopulationManager):
        self.events = events
        self.multi_pop = multi_pop

    def step(self, landscape: dict, t: int) -> None:
        # Inject multi_pop into landscape so cross-population events can access it
        landscape["multi_pop_mgr"] = self.multi_pop
        for event in self.events:
            if not event.enabled:
                continue
            if not event.trigger.should_fire(t):
                continue
            pop_name = getattr(event, 'population_name', None)
            if pop_name:
                population = self.multi_pop.get(pop_name)
                mask = self._compute_mask(population, event.trait_filter)
            else:
                # Group-level event: no single population target.
                # EventGroup.execute() delegates to children, each with their own pop_name.
                population = None
                mask = np.ones(0, dtype=bool)  # unused; children compute their own masks
            event.execute(population, landscape, t, mask)
```

**EventGroup.execute() fix:** When `population is None` (group-level), the group must route each child event to its own named population via `landscape["multi_pop_mgr"]`:

```python
def execute(self, population, landscape, t, mask):
    multi_pop = landscape.get("multi_pop_mgr")
    for _ in range(self.iterations):
        for event in self.sub_events:
            if not event.trigger.should_fire(t):
                continue
            child_pop = population  # default: inherit parent's population
            if getattr(event, 'population_name', None) and multi_pop:
                child_pop = multi_pop.get(event.population_name)
            if child_pop is not None:
                child_mask = child_pop.alive & ~child_pop.arrived
            else:
                child_mask = np.ones(0, dtype=bool)
            event.execute(child_pop, landscape, t, child_mask)
```

**Cross-population events** (e.g., `InteractionEvent`): these events receive one `population` via the router but access the second population through `landscape["multi_pop_mgr"]`. This convention is already established by the existing `InteractionEvent` in `interactions.py` and must be documented as the standard pattern for multi-population events.

Add `population_name: str | None` and `enabled: bool = True` fields to `Event` base class.

### Tests

- `test_routes_event_to_named_population()`
- `test_event_group_routes_children_independently()`
- `test_disabled_events_skipped()`
- `test_timestep_trigger_fires_once()`

---

## Phase D: Missing Event Types

### 8 event types to implement (7 critical + 1 important)

#### D.1: `HexSimMoveEvent` (CRITICAL)

The real HexSim `<moveEvent>` is fundamentally different from our `MovementEvent`. Key parameters from the Columbia XML:
- `<moveStrategy>onlyDisperse</moveStrategy>` — note: the XML value is `onlyDisperse`, NOT `dispersal`
- `<dispersalSpatialData>Gradient [ upstream ]</dispersalSpatialData>` — named spatial gradient
- `<walkUpGradient>true/false</walkUpGradient>` — follow gradient up or down
- `<dispersalAccumulator>` — accumulator tracking distance moved this event
- `<barrierSeries>` — named barrier layer
- `<dispersalAutoCorrelation>` — directional persistence (0 to 1)
- `<dispersalHaltMinimum>`, `<dispersalHaltTarget>`, `<dispersalHaltMemory>` — stopping conditions
- **`<dispersalUseAffinity />`** — CRITICAL: when present (empty element), the move event uses the affinity target set by `setSpatialAffinityEvent` as the gradient direction instead of the raw spatial data gradient. This is the mechanism for upstream migration and CWR seeking.
- `<attractionCoefficients>` — four floats controlling directional bias (e.g., `0 0 100 100`)
- `<attractionMultiplier>` — overall attraction strength
- `<trendPeriod>` — autocorrelation memory window
- `<priorityExplore>` — boolean for exploration priority
- Trait filtering via `<trait>` + `<traitCombinations>`

**Design:** Create `salmon_ibm/events_hexsim.py` with `HexSimMoveEvent` registered as `"move"`. This event reads spatial data by name from landscape, supports both raw gradient following and affinity-targeted movement, uses the existing mesh neighbor arrays, and writes distance to a named accumulator. The `dispersalUseAffinity` flag switches between gradient-following and affinity-targeted modes.

#### D.2: `HexSimSurvivalEvent` (CRITICAL)

The real `<survivalEvent>` is trait-category-driven, NOT raw-accumulator-driven:
- `<useAccumulator>true</useAccumulator>` + `<survivalAccumulator>Survival [ switch ]</survivalAccumulator>`
- The linked accumulated trait has categories: category 0 = "Will Die", category 1 = "Will Live"
- Agents in trait category 0 die (deterministic, no probability)
- The mechanism: check the trait category derived from the accumulator value, NOT the raw accumulator value

**Design:** Register as `"hexsim_survival"`. Reads the named accumulator's linked trait, kills agents in category 0. This correctly handles any threshold arrangement, not just the ≤ 0 special case.

#### D.3: `TransitionEvent` (CRITICAL — was entirely absent from spec)

The Columbia scenario makes HEAVY use of `<transitionEvent>` for all fish behavioral state transitions (movement mode, thermal response, refuge history). Without this, fish behavior is frozen.

Structure from XML:
- `<transitionTrait>Fish Status [ movement ]</transitionTrait>` — trait being updated
- One or more `<trait>` elements — conditioning traits (row dimensions of the matrix)
- `<traitCombinations>` — flat bitfield for multi-trait conditioning
- `<matrixSet transposed= rows= columns= matrixCount=>` with inline probability matrix

**Design:** The existing `TransitionEvent` in `events_phase3.py` handles single-trait transitions. Extend it to support:
1. Multi-trait conditioning via `<trait>` + `<traitCombinations>` bitfield
2. `<matrixSet>` inline probability matrix parsing (from Phase A parser output)
3. Register as `"transition"` (already registered, but parser must extract the full structure)

#### D.4: `DataLookupEvent` (CRITICAL)

Looks up values from an external CSV using accumulator values as row/column keys.

**Design:** Register as `"data_lookup"`. **Reuse** the existing `updater_data_lookup()` in `accumulators.py` (lines 374-384) for 1D lookup; extend it for 2D lookup (row + column accumulators). Must handle:
- `<hasColumnHeader>` and `<hasRowHeader>` flags (whether first row/col is header)
- **CSV path remapping:** XML contains absolute Windows paths (e.g., `F:\Marcia\...`). ScenarioLoader must remap relative to workspace `Analysis/Data Lookup/` directory.

#### D.5: `SetSpatialAffinityEvent` (CRITICAL)

Sets a spatial affinity goal for agents — picks best cell within bounds based on a spatial gradient.

Key parameters from XML:
- `<affinity>Movement Goal</affinity>` — named affinity slot (must match population's `<affinity>` definitions)
- `<strategy>better</strategy>` — search only for cells that improve on current value
- `<spatialSeries>` — the gradient spatial data
- `<useBounds>true</useBounds>` with `<minAccumulator>` and `<maxAccumulator>` — bound search range
- **`<errorAccumulator>Affinity Switch</errorAccumulator>`** — set to 1 if no valid cell found within bounds. This drives a trait used in trait filtering throughout movement logic.

**Design:** Register as `"set_spatial_affinity"`. Must implement the error accumulator write on failure — without it, fish behavior after failed affinity searches will be wrong.

#### D.6: `PatchIntroductionEvent` (CRITICAL)

Places one agent on every non-zero cell of a named spatial data layer. Used for Refuges population.

**Design:** Register as `"patch_introduction"`. Reads hex-map by name, finds non-zero cells, calls `population.add_agents()`.

#### D.7: `IntroductionEvent` (CRITICAL — was conflated with PatchIntroduction)

Distinct from `PatchIntroductionEvent`. Used for Chinook and Steelhead fish initialization:
- `<initialSize>N</initialSize>` — total number of agents to add
- `<initializationSpatialData>Special Sites [ initialization ]</initializationSpatialData>` — place agents on non-zero cells of this layer

**Design:** The existing `IntroductionEvent` in `events_builtin.py` must be extended to support spatial data layer placement (resolve layer name → cell positions via `landscape["spatial_data"]`).

#### D.8: `InteractionEvent` (CRITICAL — semantics were unspecified)

The refuge density control mechanism — central ecological feature of the Columbia scenario:
- `<presence>explored</presence>` — agents must have explored their current cell
- `<colocation type="always" />` — populations must be co-located (same hex cell)
- `<encounterProbability>1</encounterProbability>` — certain encounter
- Multiple `<interaction>` sub-blocks, each naming a second population (e.g., Chinook, Steelhead)
- `<outcomes>` with `<p1Traits>` (Refuges traits) and `<p2Traits>` (fish traits): when p1ComboIndex and p2ComboIndex match, `<p1Changes>` and `<p2Changes>` modify accumulators via `updater="assign"` or `updater="increment"`

**Design:** The existing `InteractionEvent` in `interactions.py` provides a basic framework. Extend it to support:
1. p1/p2 trait-filtered encounter outcomes
2. `assign` and `increment` accumulator modification modes
3. Multi-interaction blocks within a single event
4. Access both populations through `landscape["multi_pop_mgr"]`

### Additional updater functions needed (Phase A/B)

Three updater types are used in the Columbia XML but missing from spec:

1. **`ExploredRunningSumUpdaterFunction`** — sums a spatial data layer across all hexes in the agent's explored range. Used for computing refuge effective volumes. Add to `accumulators.py`.

2. **`QuantifyLocationUpdaterFunction`** — returns the gradient value at the agent's current cell, multiplied by a scale parameter. Distinct from `IndividualLocationsUpdaterFunction` (which returns the spatial data value). Add to `accumulators.py`.

3. **`TraitIdUpdaterFunction`** — writes the integer category index of a named trait (via `<sourceTrait>` element) into the target accumulator. Add to `accumulators.py`.

### EventGroup trait filtering

`<eventGroupEvent>` elements can have their own `<trait>` + `<traitCombinations>` filters that gate the entire group. The spec's `EventGroup.execute()` must apply this group-level filter mask before iterating children.

### Tests

Each event type gets unit tests with synthetic data (no workspace dependency).

---

## Phase E: ScenarioLoader

### Design

Create `salmon_ibm/scenario_loader.py`:

```python
class ScenarioLoader:
    """Load a HexSim workspace + XML scenario → runnable simulation."""

    def load(self, workspace_dir: str, scenario_xml: str) -> HexSimSimulation:
        """
        1. Parse XML → config dict (Phase A)
        2. Load workspace → HexMesh, spatial data layers
        3. Create populations with accumulators + traits (from config)
        4. Build event list with population routing (Phase C)
        5. Translate expressions (Phase B)
        6. Return HexSimSimulation with MultiPopEventSequencer
        """

class HexSimSimulation:
    """Multi-population simulation driven by event sequencer.

    This is a NEW class that coexists with the existing `Simulation` class:
    - `Simulation` (simulation.py): single-population salmon migration with hardcoded event sequence
    - `HexSimSimulation` (scenario_loader.py): multi-population, XML-driven, general-purpose

    The existing `Simulation` class, `app.py`, and all current tests remain unchanged.
    `HexSimSimulation` is used exclusively by `ScenarioLoader.load()`.
    """

    def __init__(self, populations, sequencer, environment, landscape):
        ...

    def step(self):
        self.environment.advance(self.current_t)
        self.sequencer.step(self.landscape, self.current_t)
        self.current_t += 1

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()
```

### Spatial data layer registry

The loader must build a registry mapping layer names → numpy arrays:

```python
spatial_registry = {}
for name, series_def in config["spatial_data_series"].items():
    if series_def["datatype"] == "HexMap":
        hm = workspace.hexmaps.get(name)
        if hm is not None:
            spatial_registry[name] = hm.values[mesh._water_full_idx]
    elif series_def["datatype"] == "Barrier":
        # Load barrier files
        ...
```

This registry is injected into `landscape["spatial_data"]` so events can look up layers by name.

### Tests

- `test_load_columbia_scenario()` — loads XML + workspace, returns HexSimSimulation
- `test_columbia_populations_initialized()` — 4 populations with correct sizes
- `test_columbia_accumulators_created()` — Chinook has ~65 accumulators
- `test_columbia_traits_created()` — Chinook has ~30 traits with correct types
- `test_columbia_runs_10_steps()` — no crashes, agents alive

---

## Milestone Definition

**M1 (target):** The Columbia River `gr_Columbia2017B.xml` scenario:
1. Parses completely (no unknown tags/attributes)
2. Creates 4 populations with correct accumulators and traits
3. Builds nested event sequence with population routing
4. Evaluates expression updaters using HexSim DSL
5. Runs for 100 timesteps without crashes
6. Produces plausible agent behavior (agents move, energy changes, some die)

**M1 REQUIRES (minimum for plausible behavior):**
- `timestep="N"` correctly fires events once at step N (not periodically)
- `transitionEvent` working (fish behavioral state transitions)
- `introductionEvent` working (fish initialization from spatial data)
- `moveEvent` with `dispersalUseAffinity` working (correct migration direction)
- `interactionEvent` with p1/p2 outcomes working (refuge density control)

**M1 does NOT require:**
- Exact numerical match with HexSim C++ output (that's M6 per roadmap)
- `dataProbeEvent` output logging
- `reanimationEvent` (disabled in Columbia XML)
- Network mode or non-Columbia scenarios

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Expression DSL has edge cases not covered by Columbia XML | Some expressions fail at runtime | Parse all 290 expressions from XML as test cases |
| Multi-population event routing has ordering dependencies | Wrong simulation behavior | Preserve XML event order exactly |
| Spatial data loading is slow for large workspaces | Long startup time | Lazy-load layers on first access |
| Some event types need deep HexSim domain knowledge | Incorrect behavior | Focus on Columbia scenario specifically; warn on unsupported features |

---

## Implementation Order

```
Phase A (XML Parser) ──→ Phase C (Multi-Pop Router) ──→ Phase D (Missing Events)
                    \                                 ↗         ↓
Phase B (Expression DSL) ────────────────────────────          Phase E (ScenarioLoader)
```

Phases A and B are independent and can be parallelized. C depends on A (needs parsed event structure). D depends on A+B+C (accumulate events nested in event groups use expressions). E depends on all.

---

## Estimated Scope

| Phase | New/Modified Files | LOC | Tests |
|-------|-------------------|-----|-------|
| A: XML Parser | 1 modified | ~500 | ~12 |
| B: Expression DSL | 1 new + 1 modified | ~250 | ~12 |
| C: Multi-Pop Router | 2 modified (events.py, interactions.py) | ~200 | ~8 |
| D: Missing Events (8 types + 3 updaters) | 1 new + 3 modified | ~600 | ~25 |
| E: ScenarioLoader | 1 new | ~300 | ~10 |
| **Total** | **~8 files** | **~1,850** | **~67** |

## Parity Review Fixes Applied

This spec was reviewed twice. The second review (HexSim domain expert) identified these critical issues, all now fixed:

1. **`timestep="N"` semantics:** Fixed from "periodic" to "fire once at step N" with `permanent` flag
2. **`transitionEvent` added:** Was entirely absent; now D.3 with multi-trait conditioning
3. **`introductionEvent` separated from `patchIntroductionEvent`:** Now D.6 + D.7
4. **`dispersalUseAffinity` added to moveEvent:** Critical for upstream migration direction
5. **`interactionEvent` semantics specified:** p1/p2 trait-filtered outcomes with assign/increment
6. **3 missing updater types added:** ExploredRunningSum, QuantifyLocation, TraitId
7. **`setSpatialAffinityEvent` error accumulator added:** Affinity Switch on search failure
8. **CSV path remapping noted:** DataLookupEvent must remap absolute Windows paths
9. **EventGroup trait filtering added:** Group-level trait gates
10. **`HexSimSurvivalEvent` corrected:** Trait category 0 = die, not raw accumulator ≤ 0
