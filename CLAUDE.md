# Shell commands
- Never use `cd <path> && git <cmd>`. Always use `git -C <path> <cmd>` instead.
- If you need to run multiple commands in a subdirectory, use separate tool calls rather than chaining with &&.

# Project Overview
- Baltic Salmon IBM (Individual-Based Model) for Curonian Lagoon migration simulation
- Python 3.10+, conda env: `shiny`
- Dual grid backends: NetCDF triangular mesh or HexSim hex grid workspace

# Testing
- Run tests: `conda run -n shiny python -m pytest tests/ -v`
- 557 tests, ~4 minute runtime
- Test files mirror source: `salmon_ibm/foo.py` -> `tests/test_foo.py`

# Key Conventions
- `AgentPool.ARRAY_FIELDS` is the single source of truth for pool array attributes. Update it when adding new fields, and update `Population.add_agents()` to handle the new field's default value.
- `Landscape` TypedDict in `simulation.py` defines the event context schema. Add new keys there when extending the landscape dict.
- `BioParams` dataclass holds all bioenergetics parameters including `MASS_FLOOR_FRACTION`.
- Events register via `@register_event("type_name")` decorator into `EVENT_REGISTRY`.
- Expression evaluator validates AST for both legacy and HexSim modes — any new namespace functions must be added to `_SAFE_MATH` or `_HEXSIM_FUNCTIONS`.

# Performance
- Hot loops use Numba `@njit(cache=True, parallel=True)` with `prange`
- Do NOT add `any_moved` early-exit patterns inside `prange` loops (data race)
- Environment data is pre-loaded into NumPy arrays at init (no per-step xarray reads)
- OutputLogger uses columnar arrays (not per-agent dicts)
- Use `np.bincount` instead of per-behavior comparison loops

# Architecture
- SoA (Structure-of-Arrays) agent model — never iterate agents in Python
- Event-driven pipeline: `EventSequencer.step()` fires events in order
- Dual-path execution: Numba JIT + NumPy fallback (controlled by `HAS_NUMBA`)
- `hexsimlab/` contains incomplete prototypes (guarded with NotImplementedError)
