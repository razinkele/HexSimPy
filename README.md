# HexSimPy — Baltic Salmon Individual-Based Model

A Python reimplementation of EPA HexSim for simulating Baltic salmon migration through the Curonian Lagoon. Supports both NetCDF-based triangular meshes and HexSim workspace hex grids. Features a Shiny for Python web UI with deck.gl visualization.

---

## Features

- **Structure-of-Arrays (SoA) agent architecture** with Numba JIT acceleration
- **Wisconsin bioenergetics model** for non-feeding migrants (proportional mass loss)
- **Behavioral decision table** (Snyder et al. 2019) — temperature x urgency -> behavior probabilities
- **Full HexSim event system compatibility** — accumulators (24 updater functions), traits, genetics, barriers, network, ranges
- **Dual grid backends** — NetCDF triangular mesh (scipy Delaunay) or HexSim hex grid (.hxn files)
- **HexSim XML scenario loading** — parse real HexSim .xml scenarios with multi-population support
- **Estuarine extensions** — salinity cost, dissolved oxygen avoidance, seiche pause
- **Diploid genetics** — multi-locus genomes with crossover and mutation
- **Stream network topology** — 1D segment-based movement with upstream/downstream navigation
- **Edge-based barriers** — mortality, deflection, and transmission probabilities from .hbf files
- **Territorial ranges** — non-overlapping cell ownership with BFS expansion/contraction
- **Multi-population interactions** — predation, competition, and disease between co-located populations
- **Reporting framework** — productivity, demographic, genetic, and dispersal reports with spatial tallies
- **Ensemble runner** with multiprocessing for parallel replicates
- **Shiny web UI** with deck.gl map, streaming charts, trail visualization, and live stats
- **556 tests passing**, 14 skipped (~4.5 minute runtime)

## Performance

- 4,000+ agent-steps/ms at scale (50K agents)
- Numba `@njit(parallel=True)` for movement, behavior, barrier resolution, and affinity search
- Pre-loaded environment data into NumPy arrays (no per-step xarray reads)
- Columnar output logging with `np.bincount` aggregation
- Cached trait-combo masks, alive masks, and expression compilation

### Benchmark: HexSimPy vs HexSim 4.0.20 (C++)

Tested on the Columbia River `snake_Columbia2017B.xml` scenario (16M-cell hex grid, 2000 Chinook agents, 2928 hourly timesteps). Hardware: Intel i7-11800H 8C/16T, 64GB RAM, Windows 11.

| Metric | HexSim 4.0.20 (C++) | HexSimPy (Python+Numba) |
|--------|---------------------|-------------------------|
| Time per step | ~2.54s | ~1.20s |
| Full run (2928 steps) | ~124 min (extrapolated) | 59 min |
| Peak RAM | 7.6 GB | 380 MB |
| **Speed** | **1.0x** | **2.1x faster** |

HexSimPy achieves its speed advantage through water-cell compaction (storing ~40K water cells instead of 16M total), Numba-parallelized kernels across 16 threads, and SoA vectorized operations. Full results: [docs/benchmark_columbia_steelhead.md](docs/benchmark_columbia_steelhead.md).

---

## Installation

```bash
conda env create -f environment.yml
conda activate shiny
```

### Dependencies

- Python >= 3.10
- NumPy, Pandas, SciPy, Matplotlib, Plotly
- xarray (NetCDF grid mode)
- PyYAML
- Shiny for Python (web UI)
- Numba (optional, for JIT acceleration)

---

## Quick Start

```bash
# CLI — NetCDF grid mode (Curonian Lagoon)
python run.py --config config_curonian_minimal.yaml --agents 100 --steps 24

# CLI — HexSim grid mode (Columbia River)
python run.py --config config_columbia.yaml --agents 500 --steps 720

# Shiny web app
python app.py
```

---

## Project Structure

```
salmon_ibm/              — Core simulation engine (32 modules)
  simulation.py          — Simulation orchestrator + Landscape TypedDict
  agents.py              — AgentPool SoA + Behavior enum + ARRAY_FIELDS
  population.py          — Population lifecycle (add/compact/remove agents)
  movement.py            — Numba movement kernels (random, directed, CWR, advection)
  behavior.py            — Behavioral decision table (Snyder et al. 2019)
  bioenergetics.py       — Wisconsin energy model (hourly respiration + mortality)
  events.py              — Event engine (ABC, triggers, sequencer, event groups)
  events_builtin.py      — Movement, survival, reproduction, introduction events
  events_hexsim.py       — HexSim-compatible events + Numba kernels
  events_phase3.py       — Mutation, transition, generated hexmap, range dynamics
  event_descriptors.py   — Typed event descriptor dataclasses for XML loading
  accumulators.py        — Per-agent floating-point state (24 updater functions)
  traits.py              — Categorical trait system (probabilistic, genetic, accumulated)
  genetics.py            — Diploid genome with crossover and mutation
  barriers.py            — Edge-based movement barriers (.hbf format)
  network.py             — Stream network topology + 1D segment movement
  ranges.py              — Territorial range allocator (non-overlapping cells)
  interactions.py        — Multi-population manager + interaction events
  environment.py         — NetCDF environmental forcing (temperature, salinity, SSH, currents)
  hexsim_env.py          — HexSim zone-based temperature lookup
  hexsim.py              — HexMesh from HexSim workspace (.hxn files)
  mesh.py                — TriMesh from NetCDF (Delaunay triangulation)
  config.py              — YAML configuration loader + factory functions
  xml_parser.py          — HexSim XML scenario parser
  scenario_loader.py     — End-to-end XML-driven simulation setup
  hexsim_expr.py         — HexSim expression DSL translator + evaluator
  estuary.py             — Salinity cost, DO avoidance, seiche pause
  output.py              — OutputLogger (columnar CSV logging)
  reporting.py           — Reports (productivity, demographic, genetic, dispersal) + tallies
  ensemble.py            — Multiprocessing ensemble runner
  hexsim_viewer.py       — HexSim viewer integration
heximpy/                 — HexSim binary file parser (.hxn, .hbf, .grid)
hexsimlab/               — Prototype GPU/Numba grid tools (incomplete, guarded)
tests/                   — 556 pytest tests (14 skipped)
ui/                      — Shiny UI components
www/                     — HTML/CSS/JS assets (streaming charts, styling)
scripts/                 — Utility and debug scripts
docs/                    — Documentation, plans, and design specs
```

---

## Configuration

Configuration is YAML-based. Two grid modes are supported:

| Mode | Config key | Mesh class | Environment class | Data source |
|------|-----------|------------|-------------------|-------------|
| NetCDF | `grid.file: path.nc` | `TriMesh` | `Environment` | NetCDF forcing files |
| HexSim | `grid.type: hexsim` | `HexMesh` | `HexSimEnvironment` | HexSim workspace + CSV |

See `config_curonian_minimal.yaml` (NetCDF) and `config_columbia.yaml` (HexSim) for working examples.

For detailed data requirements and configuration schema, see **[docs/model-manual.md](docs/model-manual.md)**.

---

## Testing

```bash
conda run -n shiny python -m pytest tests/ -v
```

Test files mirror the source structure: `salmon_ibm/foo.py` -> `tests/test_foo.py`.

---

## Functional Parity with HexSim 4.0.20

HexSimPy loads and runs real HexSim XML scenarios (`.xml`) from EPA HexSim workspaces. Parity verification against the Snyder et al. (2019) Columbia River migration corridor model:

| Feature | HexSim 4.0.20 | HexSimPy | Status |
|---------|---------------|----------|--------|
| XML scenario loading | Native | `ScenarioLoader` + `xml_parser` | Verified |
| Multi-population events | 4 populations (Chinook, Steelhead, Refuges, Iterator) | Same 4 populations | Verified |
| Spatial data layers | 16 hex-map layers | 16 layers loaded via `heximpy` | Verified |
| Temperature zones | Zone-based CSV lookup | `HexSimEnvironment` zone lookup | Verified |
| Accumulator system | 25 updater functions | 24 updater functions | Verified |
| Trait system | Probabilistic + accumulated | Same types via `TraitManager` | Verified |
| Event sequencer | C++ virtual dispatch | `MultiPopEventSequencer` | Verified |
| Expression evaluator | Custom C++ DSL | AST-validated Python `eval()` | Verified |
| Barrier loading | `.hbf` native | `read_barriers()` via `heximpy` | Verified |
| Population size (1417 shared steps) | 2000 Chinook | 2000 Chinook | Matched |

**Known gaps**: `reanimation` event type not implemented (used for caching, not critical). Some data lookup CSV files reference absolute paths from the original machine — the scenario loader extracts basenames to resolve locally.

**Scientific reference**: The model replicates the Snyder et al. (2019) / EPA CWR study that simulates salmon migration from Bonneville Dam to Snake River confluence at hourly resolution, tracking thermal exposure, energy consumption, cold-water refuge use, and acute temperature mortality across 122 days (July-October). Four populations were modeled: Tucannon Summer Steelhead, Grande Ronde Summer Steelhead, Snake River Fall Chinook, and Hanford Reach Fall Chinook.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/model-manual.md](docs/model-manual.md) | Data requirements, configuration schema, and model setup guide |
| [docs/api-reference.md](docs/api-reference.md) | Public API reference for all 30 documented modules |
| [docs/benchmark_columbia_steelhead.md](docs/benchmark_columbia_steelhead.md) | Comparative benchmark vs HexSim 4.0.20 with publication validation |
| [docs/comparison-python-vs-original-hexsim.md](docs/comparison-python-vs-original-hexsim.md) | Feature-by-feature comparison with original HexSim |
| [CHANGELOG.md](CHANGELOG.md) | Version history organized by development phase |
| [heximpy/README.md](heximpy/README.md) | HexSim binary file parser quick start |
| [heximpy/API.md](heximpy/API.md) | heximpy API reference |
| [heximpy/FILE_FORMATS.md](heximpy/FILE_FORMATS.md) | HexSim binary format specifications |

---

## Key Conventions

| Convention | Description |
|---|---|
| `AgentPool.ARRAY_FIELDS` | Single source of truth for all pool array attributes |
| `Landscape` TypedDict | Typed event context defined in `simulation.py` |
| `BioParams` dataclass | All bioenergetics parameters, including `MASS_FLOOR_FRACTION` |
| `@register_event("type_name")` | Decorator for registering custom event types into `EVENT_REGISTRY` |
| `_SAFE_MATH` / `_HEXSIM_FUNCTIONS` | Whitelisted functions for expression evaluation |
| SoA architecture | Never iterate agents in Python — use vectorized NumPy or Numba kernels |

---

## Architecture

```
YAML Config ─────────┐
HexSim XML Scenario ─┤
                      ├──> Simulation / HexSimSimulation
                      │       ├── Mesh (TriMesh | HexMesh)
                      │       ├── Environment (Environment | HexSimEnvironment)
                      │       ├── Population
                      │       │     ├── AgentPool (SoA arrays)
                      │       │     ├── AccumulatorManager
                      │       │     ├── TraitManager
                      │       │     ├── GenomeManager
                      │       │     └── RangeAllocator
                      │       ├── EventSequencer
                      │       │     ├── MovementEvent
                      │       │     ├── SurvivalEvent
                      │       │     ├── HexSimMoveEvent
                      │       │     ├── HexSimSurvivalEvent
                      │       │     ├── InteractionEvent
                      │       │     └── ... (20+ event types)
                      │       ├── BarrierMap
                      │       ├── StreamNetwork
                      │       ├── MultiPopulationManager
                      │       └── OutputLogger / ReportManager
                      │
                      └──> EnsembleRunner (multiprocessing)
```

---

## License

---

## References

- Snyder, M.N., Schumaker, N.H., Ebersole, J.L., et al. (2019). Individual based modeling of fish migration in a 2-D river system: model description and case study. *Landscape Ecology* 34:737–754.
- Snyder, M.N., Schumaker, N.H., Dunham, J.B., et al. (2022). Tough places and safe spaces: Can refuges save salmon from a warming climate? *Ecosphere* 13(11):e4265.
- Snyder, M.N., Schumaker, N.H., & Ebersole, J.L. (2020). HexSim migration corridor simulation model results. EPA Technical Memorandum (CWR Plan Appendix 21).
- Fulford, R.S., Tolan, J.L., & Hagy, J.H. (2024). Simulating implications of fish behavioral response for managing hypoxia in estuaries. *Ecological Modelling* 490:110635.
- Schumaker, N.H. & Brookes, A. (2018). HexSim: a modeling environment for ecology and conservation. *Landscape Ecology* 33:197–211.
- Forseth, T., et al. (2001). Bioenergetics of Atlantic salmon. *Canadian Journal of Fisheries and Aquatic Sciences*.
- Hanson, P.C., et al. (1997). Fish Bioenergetics 3.0. *University of Wisconsin Sea Grant Institute*.
- Stewart, D.J. & Ibarra, M. (1991). Predation and production by salmonine fishes in Lake Michigan. *Canadian Journal of Fisheries and Aquatic Sciences* 48:909–922.
