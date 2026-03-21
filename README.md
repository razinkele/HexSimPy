# HexSimPy — Baltic Salmon Individual-Based Model

A Python reimplementation of EPA HexSim for simulating Baltic salmon migration through the Curonian Lagoon. Supports both NetCDF-based triangular meshes and HexSim workspace hex grids. Features a Shiny for Python web UI with deck.gl visualization.

---

## Features

- **Structure-of-Arrays (SoA) agent architecture** with Numba JIT acceleration
- **Wisconsin bioenergetics model** for non-feeding migrants (proportional mass loss)
- **Behavioral decision table** (Snyder et al. 2019) — temperature x urgency -> behavior probabilities
- **Full HexSim event system compatibility** — accumulators, traits, genetics, barriers, network
- **Dual grid backends** — NetCDF triangular mesh (scipy Delaunay) or HexSim hex grid (.hxn files)
- **Ensemble runner** with multiprocessing
- **557 tests passing**

## Performance

- 4,000+ agent-steps/ms at scale (50K agents)
- Numba `@njit(parallel=True)` for movement, behavior, and barrier resolution
- Pre-loaded environment data, columnar output logging

---

## Installation

```bash
conda env create -f environment.yml
conda activate shiny
```

---

## Quick Start

```bash
# CLI
python run.py --config config_curonian_minimal.yaml --agents 100 --steps 24

# Shiny web app
python app.py
```

---

## Project Structure

```
salmon_ibm/          — Core simulation engine (32 modules)
  agents.py          — AgentPool SoA + Behavior enum
  simulation.py      — Simulation orchestrator + Landscape TypedDict
  movement.py        — Numba movement kernels
  behavior.py        — Behavioral decision table
  bioenergetics.py   — Wisconsin energy model
  events.py          — Event engine (ABC, triggers, sequencer)
  events_builtin.py  — Movement, survival, reproduction events
  events_hexsim.py   — HexSim-compatible events + Numba kernels
  accumulators.py    — Per-agent floating-point state (24 updater functions)
  population.py      — Population lifecycle (add/compact agents)
  environment.py     — NetCDF environmental forcing
  hexsim_env.py      — HexSim zone-based temperature
  hexsim.py          — HexMesh from HexSim workspace
  mesh.py            — TriMesh from NetCDF
  config.py          — YAML configuration loader
  genetics.py        — Diploid genome with crossover
  barriers.py        — Edge-based movement barriers
  estuary.py         — Salinity cost, DO avoidance, seiche pause
  ...
heximpy/             — HexSim binary file parser (.hxn, .hbf, .grid)
hexsimlab/           — Prototype GPU/Numba grid tools (incomplete)
tests/               — 557 pytest tests
ui/                  — Shiny UI components
www/                 — HTML assets
scripts/             — Utility scripts
```

---

## Configuration

Configuration is YAML-based. See `config_curonian_minimal.yaml` for a working example.

Two grid modes are supported:

- `grid.type: netcdf` — uses `TriMesh` with xarray-based environmental forcing
- `grid.type: hexsim` — uses `HexMesh` with a HexSim workspace directory

---

## Testing

```bash
conda run -n shiny python -m pytest tests/ -v
```

---

## Key Conventions

| Convention | Description |
|---|---|
| `AgentPool.ARRAY_FIELDS` | Single source of truth for all pool array attributes |
| `Landscape` TypedDict | Typed event context defined in `simulation.py` |
| `BioParams` dataclass | All bioenergetics parameters, including `MASS_FLOOR_FRACTION` |
| `@register_event("type_name")` | Decorator for registering custom event types |

---

## License

---

## References

- Schumaker, N.H. (2024). HexSim. U.S. Environmental Protection Agency, Corvallis, Oregon.
- Snyder, M.N., et al. (2019). A behavioral decision framework for modeling Atlantic salmon smolt migration. *Ecological Modelling*.
- Forseth, T., et al. (2001). Bioenergetics of Atlantic salmon. *Canadian Journal of Fisheries and Aquatic Sciences*.
