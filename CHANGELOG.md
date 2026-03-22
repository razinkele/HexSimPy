# Changelog

All notable changes to the HexSimPy Baltic Salmon IBM are documented in this file. Entries are organized by development phase and category.

---

## v1.0.0 (2026-03-22)

### Benchmark & Validation
- `f31ee9d` docs: add expected model behavior from Snyder 2019/2022 publications
- `fb39f77` fix: scenario_loader CSV path resolution + header handling, add parity section to README
- `2ecb662` bench: add Columbia steelhead benchmark — HexSimPy 2.1x faster than C++

### Documentation
- `ffcfc3f` docs: update README, API reference, add CHANGELOG and model manual

---

## Pre-release Changes

### Bug Fixes
- `d997335` fix: grid persistence on step + improved map contrast
- `fb797f8` fix: self-mating warning, interaction stats, salinity warning, cache bounds, float== docs
- `7a1c6e1` fix: guard broken hexsimlab prototypes with NotImplementedError + bounds check
- `2b9958d` fix: add bounds checks to updater functions + fix accumulator bounds logic
- `4e8bf08` fix: invalidate HexSimMoveEvent cache when mesh changes across runs
- `0a4dfc9` fix: use inspect.signature instead of except TypeError in scenario_loader
- `39712d5` fix: close unused datasets after preload + recompute behavior mask
- `83c99e4` fix: clamp mass to positive in hourly_respiration — prevent NaN/Inf
- `15ee6e9` fix: warn on unknown updater, missing accumulator, spatial data, and lookup CSV

### Performance
- `328f401` perf: inline t3h_mean, bincount logging, pre-bucket movement agents
- `1fe38ee` perf: use bulk pre-allocation in add_agents instead of 11 np.concatenate calls
- `b353ff0` perf: stack environment fields for single-pass advance
- `4e5f284` perf: cache alive mask in EventGroup + compile expression cache
- `b095695` perf: replace per-agent dict accumulation with columnar arrays in OutputLogger
- `7d2a822` perf: add parallel=True + prange to Numba movement kernels
- `2ae1262` perf: pre-load xarray data at init — avoid 5 dataset reads per step

### Refactoring
- `0184e89` refactor: add Landscape TypedDict for type-safe event context
- `6d795ef` refactor: replace hardcoded field list with AgentPool.ARRAY_FIELDS
- `23347a9` refactor: move mass floor constant to BioParams.MASS_FLOOR_FRACTION
- `65f2f2c` refactor: consolidate duplicate MockPopulation into tests/helpers.py

### Security
- `abae760` security: validate translated HexSim expressions before eval

### Tests
- `264f191` test: add 5 missing test cases — compact+genome, uptake negative, push_temp, trait_filter
- `ac36c50` feat: add zero-agent edge tests + convert DO states to IntEnum

### Documentation
- `309b861` docs: add README.md, API reference, and update CLAUDE.md

---

## UI: Map Pipeline and Streaming Charts

### Features
- `70ed0ba` feat: rewrite _update_map with 3-branch logic — hex grid stable during run
- `1369b4a` feat: replace _color_and_agent_update with _hex_color_update — no set_style call
- `b07113f` feat: always enable spring transitions on agent layer — smooth animation
- `f7851b6` feat: add _push_chart_data with backpressure — streaming chart updates
- `85a2ed3` feat: add _push_chart_reset to init — migration bins + JS init
- `127f4bb` feat: wire streaming charts panel into Map tab
- `34eb768` feat: add CSS styles for streaming charts panel
- `b39d9d2` feat: add streaming charts JS — Plotly extendTraces message handlers
- `d0f452d` feat: add charts_panel UI component for streaming charts

### Bug Fixes
- `495e13d` fix: convert BLANK_STYLE to data URL — MapLibre was treating JSON string as URL
- `6986ce2` fix: correct _should_transition inverted logic — enable spring animations
- `02bb441` fix: narrow exception handling + warn on trait-combo mask fallback
- `0ab244e` fix: recompute alive mask in timers + apply overrides only to alive agents
- `c7ad39a` fix: use proportional mass loss in update_energy — prevent ED inflation
- `3920208` fix: add warnings for missing temperature, gradient, and NaN in expressions
- `85129e6` fix: use np.subtract.at in updater_uptake for correct multi-agent depletion
- `c759760` fix: validate DO thresholds — reject lethal > high

### Refactoring
- `cb289f2` refactor: rename _agent_only_update -> _agent_trail_update

### Performance
- `b022699` perf: decouple status text from history — use lightweight step_stats

---

## HexSim I/O Improvements and Validation

### Features
- `3529224` feat: add from_descriptor() with fallback — registry-driven event loading
- `ddb81d2` feat: add per-type XML event parameter extraction to descriptors
- `af298cc` feat: add typed event descriptor dataclasses
- `e5fbdb3` feat: add world file and barrier file validation
- `ce55e9c` feat: add XML required element checks + event loading warnings
- `89c615b` feat: add temperature CSV shape validation
- `fe10ea4` feat: add GridMeta physical plausibility validation
- `d675ad2` feat: add HXN read/write validation for data length and dtype
- `32fbecc` feat: add GridMeta.data_height/data_width + dimension validation

### Bug Fixes
- `5e34945` fix: narrow-grid support in hxnparser neighbors, exports, and hex_to_xy
- `85ea0eb` fix: narrow-grid neighbor computation with pointy-top odd-row offsets

### Tests
- `b505657` test: add round-trip fidelity, cell count, and barrier fixture tests
- `4e100cd` test: add data layout verification for pointy-top odd-row convention

---

## UI: Hex Grid Display and Animation

### Features
- `4cd1f1c` feat(ui): add HexSim viewer tab, light theme, and hex grid rendering
- `7a1a831` feat(ui): add trail visualization with PathLayer + toggle
- `296fbdb` feat(ui): add live stats bar with real-time population counts
- `105a030` feat(ui): binary-encode agents + deck.gl transitions + adaptive sleep

### Bug Fixes
- `a49aad7` fix(ui): switch to pointy-top hex display matching HexSim 4.0.20
- `f4c29f5` fix(ui): compute zoom for visible hexagons + correct narrow-grid mapping
- `722e313` fix: proper narrow-grid row/col mapping for hex coordinates
- `7731290` fix(ui): revert to flat-top hex vertices (matches centroid spacing)
- `cddec57` fix(ui): change hex orientation from flat-top to pointy-top
- `39fbe57` fix(ui): add odd-column offset to viewer hex coordinates
- `934328d` fix(ui): add filled=true to SolidPolygonLayer — fixes black hexagons
- `909ac4d` fix(ui): add HexSim spatial layer selector + fix viewer zoom
- `6eb87a1` fix(ui): dynamic scale for HexSim grids + enhanced legend
- `5420fef` fix(ui): fix HexSim grid — revert to MapView with pseudo-geographic coords
- `050c401` fix(ui): fix HexSim grid subsampling — use stride instead of random
- `a195082` fix(ui): fix HexSim grid display — use CARTESIAN coordinates + OrthographicView

### Performance
- `2ad7285` perf(ui): throttle Plotly chart regeneration to every 5th step

### Tests
- `356a4a6` test: add hex grid rendering test suite with playwright screenshots

---

## HexSim Parity: Performance Optimization (247x-521x speedup)

### Performance
- `e01ba16` perf: pre-resolve trait filters + remove empty-mask skip bug (521x total)
- `210b4e5` perf: accumulate dispatch table + cleanup (479x total)
- `4569a5d` perf: cache mesh/gradient refs + n_alive fast path (476x total)
- `53b3833` perf: cache trait-combo masks per step (394x total speedup)
- `08336d7` perf: Numba JIT affinity search + cache trait-combo flags (317x total)
- `3959bf8` perf(move): add Numba JIT kernels for HexSim movement (247x total speedup)
- `392ae39` perf(events): optimize EventGroup dispatch with pre-filtering and n_alive fast path
- `326733b` perf(accumulators): add lazy accumulator dict for expression evaluation
- `3d28028` perf(events): vectorize HexSimMoveEvent and SetSpatialAffinityEvent (22x speedup)

### Bug Fixes
- `d8efa7f` fix: comprehensive code review bugfixes and Phase 4 completion
- `ffd922a` fix: address engine code review findings
- `cca42d4` fix(ui): address code review findings for animation features

### Tests
- `9f1ae8c` test: add deep tests for HexSim events, Numba kernels, and caching
- `a5311e6` test(perf): add 100K HexSim perf test and profiling script

---

## HexSim Parity: XML Scenarios, Events, and Subsystems

### Features — Event System
- `e5791e5` feat(scenario): add ScenarioLoader + HexSimSimulation for real HexSim XML scenarios
- `59905d0` feat(events): add HexSim event types (move, survival, data_lookup, affinity, patch_introduction)
- `817e47e` feat(events): add multi-population event router with EventGroup child routing
- `6ca6070` feat(expr): add HexSim expression DSL translator with function injection
- `69df8ee` feat(xml): rewrite parser — simulation params, grid, populations, events, updater functions

### Features — Subsystems
- `e47ef87` perf(barriers): add Numba kernel for barrier resolution
- `a24c6f1` perf(simulation): short-circuit estuarine overrides when config disables them
- `b0adec3` perf(hexsim_env): eliminate redundant SSH copies and per-step astype in advance()
- `66716a9` feat(ensemble): add multiprocessing ensemble runner for parallel replicates

### Tests
- `ef360f3` test: add HexSim compatibility tests for Columbia River workspace

---

## Phase 3: Genetics, Network, Ranges, Interactions

### Features
- `8a2521e` feat: add PlantDynamicsEvent and wire RangeAllocator into Population
- `4bb1978` feat(reporting): add report/tally framework with productivity, demographic, genetic, dispersal reports
- `42b16fd` feat(ranges): add RangeAllocator for non-overlapping territory management
- `83e6282` feat(xml): add XML scenario parser for HexSim .xml file loading
- `f82d3b5` feat(events): add LogSnapshotEvent and SummaryReportEvent for simulation output
- `1439e43` feat(accumulators): complete all 25 HexSim updater functions
- `4746b79` feat(config): add barrier, genetics, and population config loading
- `192dd50` feat(reproduction): wire genetics recombination into ReproductionEvent
- `0d86533` feat(events): add CensusEvent for population reporting by trait
- `b3005a3` feat: wire barriers into movement and add genome to Population
- `97c2ef2` feat(events): add GeneratedHexmap, RangeDynamics, and SetAffinity events
- `a694acd` feat(network): add StreamNetwork, movement, range management, and SwitchPopulationEvent
- `61bf99f` feat(traits): add GENETIC and GENETIC_ACCUMULATED trait types
- `58bc350` feat(genetics): add GenomeManager with diploid storage, recombination, and mutation
- `53c4adc` feat(events): add TransitionEvent for SEIR disease modeling
- `7473c55` feat(interactions): add MultiPopulationManager and InteractionEvent

---

## Phase 2: Population Lifecycle and Barriers

### Features
- `60bad03` test(phase2): add multi-generation integration tests for population lifecycle
- `e2268e9` feat(events): add StageSpecificSurvival, Introduction, Reproduction, FloaterCreation events
- `13496be` feat(barriers): add _resolve_barriers_vec for barrier enforcement in movement
- `a4b1aa4` feat(barriers): add BarrierMap class for edge-based barrier lookup
- `14698ea` refactor(simulation): pass Population instead of raw AgentPool to EventSequencer
- `4491446` feat(population): add Population class with dynamic array resizing

---

## Phase 1: Event Engine, Accumulators, Traits

### Features
- `dd0e0c6` perf: add Numba JIT compilation for movement kernels and behavior selection
- `dae69d1` feat(events): add YAML event loading with EventFactory and register_event
- `9e96c5d` refactor(simulation): replace hardcoded step() with EventSequencer
- `23a03fc` feat(events): add MovementEvent, SurvivalEvent, AccumulateEvent, CustomEvent
- `2541bbd` feat(events): add Event ABC, triggers, EventSequencer, and EventGroup
- `4ab5484` feat: integrate accumulators and traits into AgentPool
- `d4cddbf` feat(traits): add TraitManager with probabilistic, accumulated types and filtering
- `0854f0a` feat(accumulators): add all 8 priority updater functions
- `a8484c3` feat: add AccumulatorManager with storage, get/set, and bounds clamping

---

## Phase 0: Vectorization and Performance Baseline

### Performance
- `24ea2b7` refactor: remove scalar movement functions replaced by vectorized versions
- `859c2ef` test: tighten performance benchmarks after vectorization
- `80bfc6b` perf: add precomputed water-neighbor arrays to HexMesh
- `21eb4fb` perf: vectorize current advection
- `25e9da1` perf: vectorize TO_CWR movement kernel, remove per-agent loop
- `9cab93e` perf: vectorize UPSTREAM/DOWNSTREAM movement kernels
- `91cc4b2` perf: vectorize RANDOM movement kernel

### Tests
- `6466f54` test: add performance benchmark baseline for 5000 agents

---

## heximpy: HexSim Binary File Parser

### Features
- `635e219` feat: add Workspace loader for HexSim workspace directories
- `a02d5fe` feat: add GeoTIFF export to HexMap
- `3b3c402` feat: add CSV, GeoDataFrame, and shapefile export to HexMap
- `94829c3` feat: add HexMap.to_file write support and plain-format fields
- `a9d9e91` feat: add GridMeta.from_file and read_barriers for .grid/.hbf parsing
- `5b27e39` feat: add HexMap.from_file with PATCH_HEXMAP and plain format support
- `60b9f1e` feat: scaffold hxnparser dataclasses (HexMap, GridMeta, Barrier) with tests

### Bug Fixes
- `550c5d6` fix(hxnparser): align dataclass fields with spec
- `799b16d` refactor: remove unused Sequence import from hxnparser
- `febb0a0` fix(hxnparser): address final review issues

### Tests
- `8d159cd` test(hxnparser): add cross-validation and integration tests

---

## UI: Shiny + deck.gl Migration

### Features
- `88abd99` feat: replace matplotlib/iframe map with shiny-deckgl MapWidget
- `01359b6` feat: replace matplotlib/iframe renderers with shiny-deckgl widget.update()
- `05d5715` feat: add map legend as Shiny UI overlay
- `4ba687e` feat: add shiny-deckgl MapWidget to UI layout

### Refactoring
- `ae52091` refactor: remove DECK_TEMPLATE, matplotlib imports, and _hex_to_rgb_f

### Bug Fixes
- `0860666` fix: restore matplotlib imports temporarily for _render_mpl

---

## Core Engine: Bug Fixes and Calibration

### Bug Fixes
- `017766f` feat: YAML-configurable BioParams/BehaviorParams and config validation
- `789903e` fix: add parameter validation with user-visible warnings on init
- `984c3fe` fix: vectorize activity lookup and scope bioenergetics to alive agents
- `89c6bc5` fix: cap salinity cost at 5x and handle NaN in estuary functions
- `425a4c1` fix: apply cos(lat) correction to mesh gradient for accurate spatial distances
- `39c9e2a` fix: vectorize dSSH_dt, rename seiche config key, fix energy tests
- `6ef7bd7` fix: add CWR counter updates and propagate RNG seed for reproducibility
- `f89311d` fix: correct time-to-spawn axis inversion in behavior probability table
- `efa1d19` fix: thermal mortality triggers at T >= T_MAX (was strict >)
- `f4e2127` fix: correct energy double-counting and stale mass in update_energy
- `6fc3e01` fix: remove respiration dome — R(T) now monotonically increases with temperature
- `f6989c0` fix: configurable CWR threshold and invert directed movement jitter ratio

### Performance
- `47ccc3c` perf: vectorize behavior selection by (time_idx, temp_idx) group
- `abb0106` perf: use Delaunay.neighbors and precompute water neighbor arrays

### UI
- `9bd2f8a` fix: run sim.step in background thread, update sidebar hints
- `720c3e0` fix: connect ed_init UI control, cache water indices, remove dead code
- `76e914a` fix: add error handling with user-visible notifications in step/run
- `9688011` style(ui): Lagoon Field Station dark theme with scientific context

### Tests
- `1d42d3d` test: add DOWNSTREAM, TO_CWR movement and behavior sensitivity tests
- `8877ab2` chore: add tests/conftest.py with shared fixtures and skipif marks

---

## Initial Release: Core Simulation Engine

### Features
- `a045b5f` feat: integration tests — full 24h simulation end-to-end
- `caee200` feat(ui): add Shiny for Python interactive dashboard
- `4b4f2b5` feat: simulation loop and CLI entry point
- `74a19b7` feat: movement kernels — random, directed, CWR-seeking, current advection
- `5a078ac` feat: environment module — hourly fields on triangular mesh
- `a9807ea` feat: triangular mesh from regular grid with Delaunay
- `0a58bfc` feat: behavioral decision table and override rules
- `ca2935d` feat: output logger — CSV track recording and summaries
- `6a6c9c8` feat: Wisconsin bioenergetics — hourly energy budget and mortality
- `3bfd9af` feat: estuary extensions — salinity cost, DO avoidance, seiche pause
- `16f8e32` feat: FishAgent + AgentPool — hybrid OOP/vectorized agents
- `4e90a21` feat: project scaffolding and YAML config loader
- `b6860cd` data: add NetCDF stub files for smoke testing
- `b8abf16` initial: project files and design docs
