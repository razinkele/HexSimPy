# Per-Reach Ecology — Implementation Plan

**Status:** Phase A (cell tagging) shipped in v1.2.7.  This doc plans
Phase B (per-reach simulation parameters).

## Background

The `docs/h3-multi-resolution-feasibility.md` analysis identified
**per-reach ecology parameters** as the higher-ROI next step after
the v1.2.6 inSTREAM-polygon water-mask refactor. The Curonian Lagoon,
Nemunas Delta channels, and open Baltic have ecologically distinct
predation pressure, food availability, and salinity regimes; modelling
them with one global set of parameters loses signal that the
inSTREAM example_baltic config encodes per reach.

v1.2.7 delivers the foundation: every H3 water cell now carries a
`reach_id` tag identifying which inSTREAM polygon (or Open Baltic) it
falls inside.  `H3Mesh.reach_id` is a `(N,) int8` array; `H3Mesh.
reach_names` is the `list[str]` decoder.

## The three parameters that vary the most across reaches

From `inSTREAM/instream-py/configs/example_baltic.yaml`:

| Parameter | Rivers | Lagoon | OpenBaltic |
|---|---:|---:|---:|
| `fish_pred_min` (daily fish-predation survival) | 0.97–0.985 | 0.80 | 0.65 |
| `drift_conc` (invertebrate-drift food availability) | 1.5e-8 to 2e-8 | 1.0e-9 | 0.2e-9 |
| Salinity (PSU) — modelled via the existing CMEMS field, not a config | 0 | 0–7 | 6–8 |

Predation is the dominant signal for HexSimPy's mortality calculation
— a juvenile that spends a month in the open Baltic faces ~70× the
daily mortality of one in a river reach.  This is the parameter to
prioritise.

## Phase B — minimal-effort change

### Step 1: Add `mortality_per_reach` block to the config

`configs/config_nemunas_h3.yaml`:

```yaml
mesh_backend: h3
h3_landscape_nc: data/nemunas_h3_landscape.nc

# Per-reach daily survival probability (from fish predation).
# Falls back to global default for reaches not listed and for
# OpenBaltic when no inSTREAM tag matched.
mortality_per_reach:
  Nemunas:        0.985
  Atmata:         0.970
  Minija:         0.980
  Sysa:           0.960
  Skirvyte:       0.950
  Leite:          0.955
  Gilija:         0.955
  CuronianLagoon: 0.800     # severe lagoon predation
  BalticCoast:    0.700     # nearshore Baltic
  OpenBaltic:     0.650     # seals + cod offshore
  default:        0.985     # used if reach_id missing
```

### Step 2: Look up per-cell mortality at simulation init

`salmon_ibm/simulation.py` after H3 mesh init:

```python
# Build a (n_cells,) array of daily survival probabilities indexed
# by reach.  Look-up is one numpy fancy-index per step; no per-cell
# Python loop in the hot path.
default_surv = config.get("mortality_per_reach", {}).get("default", 0.985)
self._cell_survival = np.full(
    self.mesh.n_cells, default_surv, dtype=np.float32,
)
for name, surv in config.get("mortality_per_reach", {}).items():
    if name == "default":
        continue
    cells = self.mesh.cells_in_reach(name)
    if len(cells):
        self._cell_survival[cells] = surv
```

### Step 3: Apply in the existing fish-predation event

`salmon_ibm/events_builtin.py` (or wherever the mortality lives —
trace from `simulation.py`'s event sequencer):

```python
def _event_fish_predation(self, population, landscape, t, mask):
    if not hasattr(self, "_cell_survival"):
        return  # legacy landscapes without reach_id
    p_surv = self._cell_survival[population.tri_idx]
    # Bernoulli mortality per agent per step
    rng = np.random.default_rng(self._rng.integers(2**31))
    survives = rng.random(len(p_surv)) < p_surv
    population.alive[~survives & population.alive] = False
```

(If a fish-predation event already exists, modify it in-place to use
`self._cell_survival[population.tri_idx]` instead of the global rate.)

### Step 4: Test

`tests/test_per_reach_mortality.py`:

```python
def test_open_baltic_kills_more_than_river(h3_sim):
    """Agents placed only in OpenBaltic die ~10× faster than agents
    placed only in a river reach over a 30-day window."""
    # Force-place 100 agents in OpenBaltic, run 30 days, count alive.
    # Force-place 100 agents in CuronianLagoon, run 30 days.
    # Assert open-Baltic mortality > lagoon mortality > river mortality.
```

## Phase C (later) — drift food + salinity tolerance

Once Phase B's per-reach mortality is in place, the same mechanism
extends to:

* **Drift food** — `inSTREAM`'s `drift_conc` and `search_prod` map
  to HexSimPy's existing `BioParams` (the Wisconsin-model `C`
  consumption term is currently 0 for non-feeding migrants; per-reach
  feeding could be a separate `Feeding` event that adds energy back
  scaled by `drift_conc[reach]`).
* **Salinity exposure** — already partially in the existing
  `osmoregulation` event via the global `S_opt`/`S_tol` params; could
  vary `k` per reach if the lagoon's brackish gradient warrants
  reach-specific cost coefficients.

These are smaller changes (each ~50-100 LOC) once the reach-aware
parameter-lookup pattern is established by Phase B.

## Estimated effort

* Phase B (per-reach mortality):  **half a day**, including the test.
* Phase C extras (drift food, salinity):  **1–2 days each** if the
  caller wants them.
