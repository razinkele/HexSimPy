# Nemunas Delta Branching — Design Spec

> **STATUS: DRAFT 2026-04-27.** Awaiting user review before handoff to
> `superpowers:writing-plans`. No code changes yet.

Sibling-of-shapefile to `docs/superpowers/specs/2026-04-24-nemunas-delta-h3.md`
(which delivered the H3 *test landscape*). This spec defines the **delta-branching
upgrade** itself — agent-side branch identity, per-branch discharge routing,
and the data-shape extensions that unlock future per-branch mortality and
homing studies.

## Purpose

The deployed Curonian salmon IBM (v1.5.4 at <http://laguna.ku.lt/HexSimPy/>)
already tessellates the three real flowing branches of the Nemunas delta
(Atmata, Skirvytė, Gilija) at H3 resolution 11 inside the multi-resolution
landscape, and `tests/test_h3_grid_quality.py` calibrates cross-reach
connectivity (Nemunas↔Atmata ≥ 15 links, Nemunas↔Gilija ≥ 10, etc.). What is
**absent** is *agent-side topology awareness*: agents have a `target_spawn_hour`
(when) but no natal reach (where), and behaviour is UPSTREAM/DOWNSTREAM/HOLD/
RANDOM/TO_CWR — direction comes from the velocity field, not from a homing
target. Today, at a delta junction, an upstream agent picks the branch with the
strongest current; there is no genetic or natal preference. Discharge is a
single Nemunas number (`data/nemunas_discharge.nc`); no per-branch fractions
exist anywhere in the codebase.

This spec is the **D+ slice** chosen during brainstorming on 2026-04-27 — the
"topological foundation plus the minimum agent-side hook" interpretation. It is
deliberately *not* full per-branch mortality or natal-river homing. Both are
listed below in "Deferred work".

## Scope decisions reached during brainstorming

| # | Decision | Rationale |
|---|---|---|
| 1 | Three branches (Atmata + Skirvytė + Gilija). Pakalnė dropped. | Pakalnė is a small distributary off Atmata, absent from the inSTREAM `BalticExample.shp` source; the Lithuanian sediment-budget literature lumps it into Atmata anyway. Geographic correction also flagged: "Rusnė" is the *island* between the channels, not a branch. |
| 2 | Two new agent fields: `natal_reach_id` and `exit_branch_id`. | One field alone limits future analyses. `natal_reach_id` enables future homing work; `exit_branch_id` enables per-branch passage analyses. ~16 bits per agent. |
| 3 | Both landscapes get the change (multi-res production + Nemunas test landscape). | Production landscape ships the feature to the deployed app; test landscape keeps the fast (~30 s rebuild) integration loop. Shared module ensures consistency. |
| 4 | "Hybrid" approach (Approach 3): per-branch discharge in NC schema, but agent-side update is a plain function call (not a registered Event). | Data shape extensible for future per-branch mortality; runtime stays minimal. |
| 5 | `BRANCH_FRACTIONS` lives in code (`delta_routing.py`), not YAML. | Geographic constants, not scenario-tunable. Sweeps are a research question for a follow-up plan. |

## Branch fractions

Source: **Ramsar Site 629 Information Sheet (Nemunas Delta), 2010**
([rsis.ramsar.org](https://rsis.ramsar.org/RISapp/files/41231939/documents/LT629_lit161122.pdf)).

The Ramsar sheet gives ranges; the spec uses interior-of-range midpoints
summing to 1.0:

| Branch | Range (Ramsar) | Adopted | Note |
|---|---|---|---|
| Skirvytė | 0.48 – 0.54 | **0.51** | Main flow to Kaliningrad / SW lagoon |
| Atmata | 0.23 – 0.30 | **0.27** | NE distributary; Pakalnė lumped in |
| Gilija | 0.16 – 0.29 | **0.22** | Easternmost; branches off upstream of Rusnė island |

**Geography note.** The Ramsar phrasing — *"Gilija and the rest of it goes to
the Rusnė branches: Skirvytė and Atmata"* — confirms a two-stage delta topology:
Gilija branches off the Nemunas *before* Rusnė island; the residual flow then
splits at Rusnė into Skirvytė (south) and Atmata (north). Gilija is therefore
not a "minor side-channel" but carries roughly the same magnitude as Atmata.

## Architecture

Three logical layers, one new module, no new registered Event subclasses.

```
                ┌────────────────────────────────────────┐
  build-time    │  scripts/build_h3_multires_landscape.py│
  (data shape)  │  scripts/build_nemunas_h3_landscape.py │
                │       │                                │
                │       │ writes new NC vars             │
                │       ▼                                │
                │  data/*.nc:                            │
                │    discharge_per_branch[branch, time]  │  ← NEW
                │    branch_names (global attr)          │  ← NEW
                │    branch_fractions_source (attr)      │  ← NEW
                │    + existing reach_id, reach_names    │
                └────────────────────────────────────────┘
                                │
                                │  (NC variable available for any future
                                │   event; no consumer wired today)
                                ▼
                ┌────────────────────────────────────────┐
  runtime       │  salmon_ibm/delta_routing.py  (NEW)    │
  (LUT +        │   ├── BRANCH_FRACTIONS                 │
  utilities)    │   ├── DELTA_BRANCH_REACHES (frozenset) │
                │   ├── split_discharge(q_total)         │
                │   └── update_exit_branch_id(pool, mesh)│
                └────────────────────────────────────────┘
                                │
                                │  called by Simulation.step()
                                │  after the movement event
                                ▼
                ┌────────────────────────────────────────┐
  agent state   │  salmon_ibm/agents.py                  │
                │   ARRAY_FIELDS += (                    │
                │     "natal_reach_id",   # int8         │
                │     "exit_branch_id",   # int8         │
                │   )                                    │
                └────────────────────────────────────────┘
```

**Three architectural calls:**

1. **No registered Event subclass for `update_exit_branch_id`.** Inserted as a
   `CustomEvent` in `Simulation._build_default_events()`, between `MovementEvent`
   and `fish_predation`. Same pattern as `_event_fish_predation`: a callback
   that no-ops on backends without `reach_id`. Reasoning: a 4-line vectorised
   bookkeeping operation that takes no YAML configuration shouldn't carry a
   class hierarchy.
2. **`delta_routing.py` is dependency-light.** Imports `numpy` only; takes
   `pool` and `mesh` by duck-type. Stays consumable by tests without spinning
   up a `Simulation`. Matches the pattern of `geomconst.py` and
   `baltic_params.py`.
3. **NC schema is *additive only*.** No existing variables removed or renamed.
   Simulations loading older NCs without `discharge_per_branch` keep working —
   `simulation.py` doesn't read it. Deliberate "ship schema first, wire later"
   design.

## Components

### 1. `salmon_ibm/delta_routing.py` (NEW, ~80 LoC)

```python
"""Nemunas delta branch identity, discharge fractions, and exit-tracking utility.

Three branches: Atmata, Skirvyte, Gilija. Pakalne deferred (see this spec).
Fractions follow Ramsar Site 629 Information Sheet (Nemunas Delta, 2010);
they are geographic constants of the delta, not scenario-tunable parameters."""

import numpy as np

BRANCH_FRACTIONS: dict[str, float] = {
    "Skirvyte": 0.51,
    "Atmata":   0.27,
    "Gilija":   0.22,
}
assert abs(sum(BRANCH_FRACTIONS.values()) - 1.0) < 1e-9

DELTA_BRANCH_REACHES = frozenset(BRANCH_FRACTIONS)

def split_discharge(q_total: np.ndarray) -> dict[str, np.ndarray]:
    """Apply BRANCH_FRACTIONS to a (T,)- or scalar-shaped Nemunas climatology."""
    return {br: q_total * f for br, f in BRANCH_FRACTIONS.items()}

def update_exit_branch_id(pool, mesh) -> None:
    """Mutate pool.exit_branch_id where the agent's current cell sits in a
    delta-branch reach. Sticky — only writes (never resets) so once an agent
    has crossed Skirvyte, that's their exit branch even if they later wander
    into Atmata in the lagoon. No-op when the mesh has no reach_names."""
    # Implementation:
    #   - early return if not getattr(mesh, "reach_names", None)
    #   - resolve DELTA_BRANCH_REACH_IDS from mesh.reach_names (cached per mesh)
    #   - cur_reach = mesh.reach_id[pool.tri_idx]
    #   - mask = is_branch & (exit_branch_id == -1) & alive
    #   - exit_branch_id[mask] = cur_reach[mask]
    # Vectorised; expected ~0.05 ms per call.
```

### 2. `salmon_ibm/agents.py` — `AgentPool.ARRAY_FIELDS` extension

Add two int8 fields, both default `-1`:

```python
ARRAY_FIELDS = (
    "tri_idx", "mass_g", "ed_kJ_g", "target_spawn_hour", "behavior",
    # ...existing fields...
    "natal_reach_id",   # int8: cell's reach_id at introduction; -1 if pre-tagging
    "exit_branch_id",   # int8: first delta-branch reach_id touched; sticky; -1 if never
)
```

Both initialised to `np.full(n, -1, dtype=np.int8)` in `AgentPool.__init__`.

### 3. `salmon_ibm/population.py` — `add_agents` defaults + helper

```python
new_arrays["natal_reach_id"][old_n:] = -1   # caller fills explicitly when known
new_arrays["exit_branch_id"][old_n:] = -1
```

New helper `Population.set_natal_reach_from_cells(idx_range, mesh)` writes
`natal_reach_id` from the agents' `tri_idx` lookup against `mesh.reach_id`.
Called from the introduction events (see Data Flow §1 below).

### 4. `salmon_ibm/simulation.py` — event-sequence insertion

```python
MovementEvent(name="movement", n_micro_steps=3, ...),
CustomEvent(name="update_exit_branch", callback=self._event_update_exit_branch),  # NEW
CustomEvent(name="fish_predation", callback=self._event_fish_predation),
```

The callback is a one-liner:
`delta_routing.update_exit_branch_id(population.pool, self.mesh)`.

### 5. Build scripts — NC schema additions

`scripts/build_h3_multires_landscape.py` and `scripts/build_nemunas_h3_landscape.py`
both gain, after the existing Nemunas discharge load:

```python
fractions = list(delta_routing.BRANCH_FRACTIONS.items())
branch_names = [br for br, _ in fractions]
discharge_per_branch = np.stack(
    [discharge_nemunas * f for _, f in fractions]
)  # shape: (n_branches, time)

ds["discharge_per_branch"] = (("branch", "time"), discharge_per_branch)
ds.attrs["branch_names"] = ",".join(branch_names)
ds.attrs["branch_fractions_source"] = (
    "Ramsar Site 629 Information Sheet (Nemunas Delta), 2010"
)
```

Variable is *written*, never read by the runtime. Forward-looking by design.

### 6. Output — no new columns this plan

`OutputLogger`'s existing per-agent dump path picks up the two new int8 fields
automatically (it serialises `ARRAY_FIELDS`). Per-branch entry/exit counts are
computed in post-processing:

```python
import xarray as xr
out = xr.open_dataset("run_output.nc")
exit_counts = (out.exit_branch_id.where(out.exit_branch_id >= 0)
                  .groupby(out.exit_branch_id).count())
```

If a Shiny dashboard panel ends up wanting per-branch counts live, *that*
follow-up plan adds a small accumulator. Not now.

## Data flow

### Build-time (one-shot, manual)

```
inSTREAM polygons + Nemunas discharge climatology
            │
            ▼
build_h3_multires_landscape.py / build_nemunas_h3_landscape.py
            │  (existing) tessellate, tag reach_id, sample bathy/CMEMS
            │  (existing) load Nemunas q[t] climatology
            │  (NEW)      apply delta_routing.split_discharge(q[t])
            ▼
data/*_landscape.nc
```

### Simulation init

```
Simulation.__init__
        │  loads NC → builds H3MultiResMesh (existing reach_id/reach_names)
        │  AgentPool created with NEW ARRAY_FIELDS, both = -1
        ▼
Sequencer event order:
    push_temperature → behavior_selection → estuarine_overrides
    → update_cwr_counters → movement
    → update_exit_branch  ← NEW
    → fish_predation → update_timers → bioenergetics → logging
```

### Agent introduction (two distinct paths)

**Path 1 — `IntroductionEvent` (`events_builtin.py`, registered `"introduction"`).**
After the existing `Population.add_agents(n_new, cells, ...)` call:

```python
new_idx = slice(prev_n, prev_n + n_new)
population.natal_reach_id[new_idx] = mesh.reach_id[population.tri_idx[new_idx]]
```

**Path 2 — `PatchIntroductionEvent` (`events_hexsim.py`, registered
`"patch_introduction"`).** Identical addition after the existing placement step.

**Future recruitment events** (any code path that calls `add_agents` mid-run)
must do the same. The pattern is documented in `delta_routing.py`'s module
docstring; the `Population.compact()` assertion (below) enforces it.

### Per-step

```
movement event:
    pool.tri_idx[i] ← new cell  (existing)
        │
        ▼
update_exit_branch event (NEW, ~3 lines):
    cur_reach = mesh.reach_id[pool.tri_idx]
    is_branch = np.isin(cur_reach, DELTA_BRANCH_REACH_IDS)
    untagged  = pool.exit_branch_id == -1
    target    = is_branch & untagged & pool.alive
    pool.exit_branch_id[target] = cur_reach[target]
```

**Sticky semantics** — once written, `exit_branch_id` never resets. The science
question is "which branch did this agent transit?", not "which branch is it
currently in?" First-touch is the correct semantics for future passage-fraction
analysis. If a returning adult later visits a different branch, *that*'s the
homing question, scoped out.

**Cost** — vectorised over `n_agents`; ~0.05 ms at typical agent counts.
Negligible vs. movement (~10 ms).

### Output

OutputLogger dumps `ARRAY_FIELDS` automatically — both new fields appear in
`data/run_output.nc` as int8 columns. No new aggregator. Post-processing user-side.

## Error handling and invariants

### Build-time invariants (asserted at NC creation)

| Invariant | Where | Failure mode caught |
|---|---|---|
| `sum(BRANCH_FRACTIONS.values()) ≈ 1.0` (tol 1e-9) | module-level assert in `delta_routing.py` | typo silently mis-allocating discharge |
| `set(BRANCH_FRACTIONS) ⊆ set(reach_names)` | build script after reach tagging | renamed reach in shapefile, LUT key out of date |
| `discharge_per_branch.sum(axis=0) ≈ discharge` (rtol 1e-6) | build script | floating-point bug in split |
| `discharge_per_branch.shape == (n_branches, n_time)` | build script | dim ordering slip |
| LUT and `branch_names` global attr in identical order | build script writes `branch_names` from `list(BRANCH_FRACTIONS)` | NC consumer indexing wrong branch |

### Runtime invariants

**Untagged-on-mesh assertion** — runs once per simulation step, called from
`Simulation.step()` *before* the logging event so any failure surfaces in the
same step that introduced the un-tagged agents (not a step later, when the
trace is harder to follow). Implemented as `Population.assert_natal_tagged()`:

```python
def assert_natal_tagged(self) -> None:
    if self._resume:                        # option α — see schema evolution
        return
    pool = self.pool
    on_mesh = pool.alive & (pool.tri_idx >= 0)
    untagged = pool.natal_reach_id == -1
    bad = on_mesh & untagged
    assert not bad.any(), (
        "Agents introduced without natal_reach_id tagging — every "
        "code path calling add_agents() must follow up with "
        "set_natal_reach_from_cells() or equivalent. "
        f"{int(bad.sum())} agents affected."
    )
```

**Why not in `Population.compact()`:** `compact()` early-returns when everyone
is alive, and overwrites `pool.alive` to all-True afterwards — checking there
either skips or reads the wrong state. A dedicated method called every step is
more robust and the cost is identical (one bitmask AND, ~µs).

Suppressed via `Simulation(..., resume=True)` (see "Schema evolution" — option α).

**`update_exit_branch_id` no-op safety:** early return when `mesh.reach_names`
is empty. Same defensive pattern as `_event_fish_predation`.

**Stickiness invariant:** `exit_branch_id` only flips from `-1 → {Atmata|
Skirvyte|Gilija reach_id}`. Enforced by construction (the update mask requires
`exit_branch_id == -1`). Verified by a unit test that runs an agent through
Atmata → Skirvyte → CuronianLagoon and asserts `exit_branch_id ==
reach_id_of("Atmata")` at end.

### Schema evolution (don't break old runs / old NCs)

| Concern | Mitigation |
|---|---|
| Old NC (no `discharge_per_branch`) loaded with new code | `Environment.load` doesn't read it — no KeyError |
| Old NC (no `branch_names` attr) loaded | Not consumed at runtime; missing attr → no error |
| Old run-output NC (no `natal_reach_id` column) re-opened | Existing code didn't expect the column; absence invisible. Post-processing scripts that read it must guard with `if "natal_reach_id" in out:` — documented |
| New NC loaded with old code (pre-this-plan) | Old `Environment.load` ignores unknown variables; runs as before |
| Resumed simulation from a pre-tagged checkpoint | Agents have both fields = `-1`. Compaction assertion **disabled for resume runs** via a `Simulation(..., resume=True)` flag — option **α** (suppress, don't lie). Resume runs are rare; user can re-run from scratch if natal-tagging is needed. |

### Documented limitations (intentionally not addressed)

1. **Mid-step branch transit between cells of different branches.** If an
   agent steps directly from an Atmata cell to a Skirvyte cell in one
   movement, `update_exit_branch_id` runs *after* movement and sees only the
   final cell. At res 11 (≈ 28 m) and 1–3 hops/step this is rare but real.
2. **Agents introduced into a delta-branch cell directly (not via Nemunas).**
   Their `natal_reach_id` and `exit_branch_id` end up identical from step 0.
   This is *correct* (they originated there) but means the "exit branch"
   semantic is degenerate for that cohort. Post-processing should filter on
   `natal_reach_id != exit_branch_id` if asking "which branch did
   Nemunas-natal smolts use?"

## Testing

### Unit tests — `tests/test_delta_routing.py` (NEW, ~12 tests)

```
test_branch_fractions_sum_to_one
test_branch_fractions_keys_are_real_reaches
test_split_discharge_preserves_total
test_split_discharge_handles_scalar_and_array
test_split_discharge_zero_input
test_split_discharge_negative_raises
test_update_exit_branch_id_first_touch
test_update_exit_branch_id_sticky
test_update_exit_branch_id_skips_dead_agents
test_update_exit_branch_id_skips_lagoon_only
test_update_exit_branch_id_no_op_without_reach_meta
test_update_exit_branch_id_no_op_on_land_cells
```

Hand-built fake `mesh` for `update_exit_branch_id` tests (no NC dependency,
runs in milliseconds). Same pattern as `tests/test_h3_barriers.py`.

### Agent-pool extension — `tests/test_agents.py` (+3)

```
test_array_fields_includes_natal_and_exit_ids
test_pool_init_defaults_natal_and_exit_to_minus_one
test_compact_preserves_natal_and_exit_ids
```

### Population extension — `tests/test_population.py` (+4)

```
test_add_agents_defaults_natal_and_exit_to_minus_one
test_set_natal_reach_from_cells_writes_correct_reach_ids
test_assert_natal_tagged_fires_on_untagged_alive_agents
test_assert_natal_tagged_silent_when_resume_flag_set
```

### Grid-quality extension — `tests/test_h3_grid_quality.py` (+1)

```
test_discharge_per_branch_present_and_consistent
    - "discharge_per_branch" variable exists
    - shape (n_branches, n_time)
    - branch_names attr matches BRANCH_FRACTIONS keys
    - sum(axis=0) ≈ Nemunas discharge total within rtol 1e-5
    - branch_fractions_source attr is non-empty string
```

Skips with a clear message when the NC is not built locally — matches
existing skip pattern.

### Integration extension — `tests/test_nemunas_h3_integration.py` (+1)

```
test_smolts_originating_in_nemunas_record_an_exit_branch
    Filter on natal_reach_id == reach_id_of("Nemunas").
    Assert exit_branch_id values are in {Atmata, Skirvyte, Gilija} reach_ids.
    Assert observed-exit fraction ≥ floor(observed_during_calibration × 0.6).
```

**Threshold caveat.** The 50% figure quoted during brainstorming is provisional.
The implementation plan calibrates against an actual run and sets
`threshold = floor(observed × 0.6)`, matching the discipline in
`MIN_CROSS_REACH_LINKS` (`tests/test_h3_grid_quality.py`).

### Performance regression — `tests/test_movement_metric.py` (+1)

Assert that the full step time (with new `update_exit_branch` event) hasn't
grown more than 1% vs. the existing recorded baseline. Catches O(n_agents²)
regressions if anyone replaces the vectorised update.

### Test count delta

| Bucket | Count |
|---|---|
| `test_delta_routing.py` (new) | +12 |
| `test_agents.py` extension | +3 |
| `test_population.py` extension | +4 |
| `test_h3_grid_quality.py` extension | +1 |
| `test_nemunas_h3_integration.py` extension | +1 |
| `test_movement_metric.py` extension | +1 |
| **Total** | **+22** |

Suite: 557 → 579 tests. Runtime impact: ~+10 s.

## Deferred work (carry-forward limitations)

This spec deliberately does **not** include the following — each becomes its
own future plan:

1. **Per-branch differential mortality fields.** Reading `discharge_per_branch`
   into events; per-branch survival probabilities. Needs Kaliningrad fisheries
   data we don't have for Skirvytė.
2. **Natal-tributary homing for Žeimena/Merkys/Dubysa.** Needs ~200 km
   eastward domain extension to reach the actual natal tributaries. The
   current spec treats "Nemunas" as the natal target; a homing model needs
   subreach granularity.
3. **Pakalnė as a fourth branch.** Requires fresh polygon (digitised from
   OSM hydrography or satellite imagery). Lumped into Atmata for this plan.
4. **Dynamic branch fractions.** Current spec uses static climatological
   midpoints. Real fractions vary with stage and discharge. Needs per-branch
   gauges that don't exist in EPA Smalininkai records.
5. **`exit_branch_id` mid-step trajectory awareness.** The first-touch
   semantics (above) miss agents that traverse two branches in one step. A
   sub-step trajectory log would resolve this; deferred until any analysis
   actually depends on the distinction.
6. **Live per-branch counts in the Shiny dashboard.** Post-processing only
   for now.
7. **Branch-specific habitat attributes.** `BalticExample.shp` already
   carries `FRACSPWN`, `FRACVSHL`, `NUM_HIDING`, `M_TO_ESC` per polygon;
   none consumed by the IBM today. Separate deferred-realism item (see
   `curonian_deferred.md` item 1).

When picked up, each item lands as a dedicated plan with its own spec.

## References

- Ramsar Site 629 Information Sheet — Nemunas Delta (2010).
  <https://rsis.ramsar.org/RISapp/files/41231939/documents/LT629_lit161122.pdf>
- Sibling spec — `docs/superpowers/specs/2026-04-24-nemunas-delta-h3.md`.
- inSTREAM source polygons — `data/instream_baltic_polygons/BalticExample.shp`
  (9 reaches, EPSG:3035, 1591 features).
- Existing delta-aware tests — `tests/test_h3_grid_quality.py`
  (`MIN_CROSS_REACH_LINKS` for Nemunas↔Atmata, Atmata↔Skirvyte, etc.).
