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
| 3 | Both landscapes get the change. (Realised during self-review pass 3: discharge is *not* in the landscape NC — it lives in `data/nemunas_discharge.nc`, loaded separately. So the landscape build scripts get no changes; only the discharge fetch script does. Both landscapes share the discharge file, so this is one-NC-modification, not two.) | Production landscape ships the feature to the deployed app; test landscape keeps the fast (~30 s rebuild) integration loop. Shared module ensures consistency. |
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
  build-time    │  scripts/fetch_nemunas_discharge.py    │
  (data shape)  │       │                                │
                │       │ writes new NC vars             │
                │       ▼                                │
                │  data/nemunas_discharge.nc:            │
                │    Q[time]                            ← existing
                │    Q_per_branch[branch, time]          │  ← NEW
                │    branch_names (global attr)          │  ← NEW
                │    branch_fractions_source (attr)      │  ← NEW
                │                                        │
                │  data/*_landscape.nc                   │
                │    (no change — discharge is a         │
                │     separate forcing file)             │
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
3. **NC schema is *additive only*.** No existing variables removed or renamed
   in `data/nemunas_discharge.nc`. Simulations loading older NCs without
   `Q_per_branch` keep working — the existing `forcings.river_discharge`
   loader (`environment.py:31`) only reads `Q`. Deliberate "ship schema
   first, wire later" design.

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

New helper `Population.set_natal_reach_from_cells(new_idx, mesh)` writes
`natal_reach_id` from the agents' `tri_idx` lookup against `mesh.reach_id`.
Called by every `add_agents` call site (see Data Flow → "Agent introduction
— all `add_agents` call sites").

### 4. `salmon_ibm/simulation.py` — event-sequence insertion

```python
MovementEvent(name="movement", n_micro_steps=3, ...),
CustomEvent(name="update_exit_branch", callback=self._event_update_exit_branch),  # NEW
CustomEvent(name="fish_predation", callback=self._event_fish_predation),
```

The callback is a one-liner:
`delta_routing.update_exit_branch_id(population.pool, self.mesh)`.

### 5. Discharge fetch script — `scripts/fetch_nemunas_discharge.py`

The Nemunas discharge climatology is a *separate* NetCDF (not part of the
landscape NC), built by `scripts/fetch_nemunas_discharge.py`. The script
synthesises `Q(t)` (variable name `Q`, shape `(time,)`, 5114 daily values
2011–2024). After the existing `synthesize_climatology()` returns the
dataset, extend it:

```python
from salmon_ibm import delta_routing

ds = synthesize_climatology()                                    # existing
fractions = list(delta_routing.BRANCH_FRACTIONS.items())
branch_names = [br for br, _ in fractions]
Q_per_branch = np.stack(
    [ds["Q"].values * f for _, f in fractions]
).astype(np.float32)                                              # (n_branches, n_time)

ds["Q_per_branch"] = (("branch", "time"), Q_per_branch)
ds.attrs["branch_names"] = ",".join(branch_names)
ds.attrs["branch_fractions_source"] = (
    "Ramsar Site 629 Information Sheet (Nemunas Delta), 2010"
)
```

Both landscape build scripts (`build_h3_multires_landscape.py` and
`build_nemunas_h3_landscape.py`) get **no changes** — discharge is a separate
forcing file in this codebase. After re-running the fetch script once, both
the deployed multi-res scenarios and the test-landscape integration runs use
the new variable (or, more precisely, *will* use it once a future event reads
it).

The new variable is *written*, never read by the runtime in this plan. The
runtime keeps using `forcings.river_discharge` → `Q[t]` only.

### 6. Output — explicit additions to `OutputLogger`

`OutputLogger.append_step` (`salmon_ibm/output.py:64–85`) *cherry-picks* fields
it serialises (`tri_idx`, `ed_kJ_g`, `behavior`, `alive`, `arrived`) — it does
**not** iterate `ARRAY_FIELDS`. The method has **two parallel branches**: a
columnar pre-allocated array path (`_max_agents` set, lines 64–75) and a
list-append fallback (lines 76–85). Both must be extended.

1. `OutputLogger.__init__` allocates **paired storage in both modes**:
   columnar: `self._natal_reach_id_arr` and `self._exit_branch_id_arr` (int8,
   shape `(max_steps, max_agents)`); list-append: `self._natal_reach_id` and
   `self._exit_branch_id` (lists of int8 arrays).
2. `OutputLogger.append_step` writes both branches (columnar slice assignment
   in the first; `.copy()` append in the second).
3. `OutputLogger.to_dataframe` and the empty-output column list (line 88) gain
   two int8 columns.

Per-branch entry/exit counts are then computed in post-processing:

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
Nemunas discharge climatology (synthesised in fetch script)
            │
            ▼
fetch_nemunas_discharge.py
            │  (existing) synthesise daily Q(t) climatology
            │  (NEW)      Q_per_branch = stack([Q * f for f in FRACS])
            ▼
data/nemunas_discharge.nc:
    Q[time]                    ← existing
    Q_per_branch[branch, time] ← NEW
    branch_names attr          ← NEW
    branch_fractions_source    ← NEW

(Landscape build scripts unchanged — discharge is a separate forcing file.)
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

### Agent introduction — all `add_agents` call sites

`Population.add_agents(n, positions, *, mass_g=None, ed_kJ_g=6.5, group_id=-1)`
is invoked from **five places** in the codebase. The contract: every call must
be followed by `population.set_natal_reach_from_cells(new_idx, mesh)` *unless*
the call site explicitly preserves `natal_reach_id` from a source agent (the
transfer case).

| Call site | Event / context | Tagging contract |
|---|---|---|
| `events_builtin.py:237` | `IntroductionEvent.step` (`@register_event("introduction")`) | **Set natal from cell.** New agents start at scenario-defined positions. |
| `events_hexsim.py:418` | `PatchIntroductionEvent.step` (`@register_event("patch_introduction")`) | **Set natal from cell.** Same as above for HexSim-style scenarios. |
| `events_builtin.py:292` | `ReproductionEvent.step` (`@register_event("reproduction")`) | **Set natal from cell.** Offspring inherit parent's `tri_idx`; natal becomes the parent's spawn cell — biologically the right answer. Currently unused by deployed salmon scenarios but the assertion fires if enabled without tagging. |
| `events_phase3.py:295` | Phase-3 vegetation seedling event (non-fish) | **Set natal from cell.** Out-of-domain for the salmon IBM, but the assertion still applies; tagged for consistency. |
| `network.py:194` | `TransferEvent.step` (multi-population transfer) | **Preserve from source.** Transferred agents inherit `natal_reach_id` from the source population — natal is fixed at birth, not at transfer. Implementation: the `TransferEvent` copies the field from `source.natal_reach_id[transfer]` to `target.natal_reach_id[new_idx]`. **Note:** if the source and target populations use different meshes, the reach_id encoding may not match — documented limitation. |

The canonical idiom for the four "set natal from cell" cases:

```python
new_idx = population.add_agents(n, positions, ...)   # returns the new-agent slice
population.set_natal_reach_from_cells(new_idx, mesh)
```

`set_natal_reach_from_cells` is implemented as:

```python
def set_natal_reach_from_cells(self, new_idx, mesh) -> None:
    # Use reach_names (a list) for the truthiness check, NOT reach_id (an
    # ndarray — `if not arr` raises on multi-element arrays). Matches the
    # pattern in delta_routing.update_exit_branch_id().
    if not getattr(mesh, "reach_names", None):
        return                                         # TriMesh / HexMesh — no-op
    self.pool.natal_reach_id[new_idx] = mesh.reach_id[self.pool.tri_idx[new_idx]]
```

The pattern is documented in `delta_routing.py`'s module docstring;
`Population.assert_natal_tagged()` (below) enforces it at runtime.

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

`OutputLogger` requires explicit additions (see Component §6) — both new
int8 fields are appended in `append_step` (both columnar and list-append
branches) and emitted in `to_dataframe`. No new aggregator. Per-branch
counts are computed post-processing, user-side.

## Error handling and invariants

### Build-time invariants (asserted at NC creation)

| Invariant | Where | Failure mode caught |
|---|---|---|
| `sum(BRANCH_FRACTIONS.values()) ≈ 1.0` (tol 1e-9) | module-level assert in `delta_routing.py` | typo silently mis-allocating discharge |
| `Q_per_branch.sum(axis=0) ≈ Q` (rtol 1e-6) | `fetch_nemunas_discharge.py` | floating-point bug in split |
| `Q_per_branch.shape == (n_branches, n_time)` | `fetch_nemunas_discharge.py` | dim ordering slip |
| LUT and `branch_names` global attr in identical order | fetch script writes `branch_names` from `list(BRANCH_FRACTIONS)` | NC consumer indexing wrong branch |

### Init-time invariants (Simulation startup)

| Invariant | Where | Failure mode caught |
|---|---|---|
| `set(BRANCH_FRACTIONS) ⊆ set(mesh.reach_names)` | `Simulation.__init__` after mesh load, raises `ValueError` if missing | renamed reach in inSTREAM source / mesh + LUT key drift |
| `Q_per_branch` axis-1 length matches `Q` length, *if both present* | discharge loader (deferred — only matters when a future event reads `Q_per_branch`) | discharge file modified by hand in a way that breaks consistency |

### Runtime invariants

**Untagged-on-mesh assertion** — runs once per simulation step, called from
`Simulation.step()` *before* the logging event so any failure surfaces in the
same step that introduced the un-tagged agents (not a step later, when the
trace is harder to follow). Implemented as `Population.assert_natal_tagged()`:

```python
# In Population:
def assert_natal_tagged(self) -> None:
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

# In Simulation.step(), called before _event_logging:
if not self.resume:                          # option α — see schema evolution
    self.population.assert_natal_tagged()
```

The `resume` flag lives on `Simulation` (where session state belongs); Population
stays stateless about replay. Simulation already has constructor surface for
flags like `seed`; `resume: bool = False` is added alongside.

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
| Old discharge NC (no `Q_per_branch`) loaded with new code | `forcings.river_discharge` only reads `Q` — no KeyError |
| Old discharge NC (no `branch_names` attr) loaded | Not consumed at runtime; missing attr → no error |
| Old run-output NC (no `natal_reach_id` column) re-opened | Existing code didn't expect the column; absence invisible. Post-processing scripts that read it must guard with `if "natal_reach_id" in out:` — documented |
| New discharge NC loaded with old code (pre-this-plan) | Old loader ignores unknown variables; runs as before |
| Resumed simulation from a pre-tagged checkpoint | Agents have both fields = `-1`. The natal-tagging assertion (`Population.assert_natal_tagged`) **disabled for resume runs** via a `Simulation(..., resume=True)` flag — option **α** (suppress, don't lie). Resume runs are rare; user can re-run from scratch if natal-tagging is needed. |

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
3. **Multi-mesh `TransferEvent` reach-id encoding drift.** When source and
   target populations use *different* meshes, `TransferEvent` copies
   `natal_reach_id` byte-for-byte from source to target — but the integer
   encoding indexes into different `reach_names` lists, so the natal label
   becomes meaningless on the target side. No deployed salmon scenario uses
   multi-mesh transfers, so the assertion does not fire today. If multi-mesh
   transfers ever ship, the transfer logic must remap via
   `source.mesh.reach_names[natal_id]` → `target.mesh.reach_names.index(name)`.

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

### Population extension — `tests/test_population.py` (+3)

```
test_add_agents_defaults_natal_and_exit_to_minus_one
test_set_natal_reach_from_cells_writes_correct_reach_ids
test_assert_natal_tagged_fires_on_untagged_alive_on_mesh
```

(The resume-flag bypass is tested in `tests/test_simulation.py` since the
flag lives on `Simulation`, not `Population`.)

### Discharge schema — `tests/test_nemunas_discharge.py` (NEW, +1)

```
test_q_per_branch_present_and_consistent
    Loads data/nemunas_discharge.nc.
    Asserts:
      - "Q_per_branch" variable exists
      - shape (n_branches, n_time) where n_time == len(Q)
      - branch_names global attr matches list(BRANCH_FRACTIONS) order
      - Q_per_branch.sum(axis=0) ≈ Q within rtol 1e-5
      - branch_fractions_source attr is non-empty string
```

Skips with a clear message when `data/nemunas_discharge.nc` lacks the new
variable (i.e., user hasn't re-run `fetch_nemunas_discharge.py` since this
plan landed) — matches the existing skip-on-missing-data pattern in
`tests/test_h3_grid_quality.py:71`.

### `Simulation` extension — `tests/test_simulation.py` (+2)

```
test_init_raises_when_branch_fractions_keys_missing_from_mesh
    Build a fake mesh with reach_names = ["Nemunas", "CuronianLagoon"]
    (missing all three branches). Construct Simulation with this mesh.
    Assert ValueError raised with a message naming the missing branch.

test_step_skips_natal_assertion_when_resume_flag_set
    Construct Simulation(resume=True), inject an alive agent with
    natal_reach_id = -1 and tri_idx >= 0, run one step. Assert no
    AssertionError raised. (Repeat with resume=False — assertion fires.)
```

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

### Output extension — `tests/test_output.py` (+2)

```
test_outputlogger_serialises_natal_reach_id
test_outputlogger_serialises_exit_branch_id
```

### Performance regression — `tests/test_movement_metric.py` (+1)

Assert that the full step time (with new `update_exit_branch` event) hasn't
grown more than 1% vs. the existing recorded baseline. Catches O(n_agents²)
regressions if anyone replaces the vectorised update.

### Test count delta

| Bucket | Count |
|---|---|
| `test_delta_routing.py` (new) | +12 |
| `test_agents.py` extension | +3 |
| `test_population.py` extension | +3 |
| `test_nemunas_discharge.py` (new) | +1 |
| `test_simulation.py` extension | +2 |
| `test_nemunas_h3_integration.py` extension | +1 |
| `test_output.py` extension | +2 |
| `test_movement_metric.py` extension | +1 |
| **Total** | **+25** |

Suite: 557 → 582 tests. Runtime impact: ~+10 s.

## Deferred work (carry-forward limitations)

This spec deliberately does **not** include the following — each becomes its
own future plan:

1. **Per-branch differential mortality fields.** Reading `Q_per_branch` into
   events; per-branch survival probabilities. Needs Kaliningrad fisheries
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
5. **Live per-branch counts in the Shiny dashboard.** Post-processing only
   for now.
6. **Branch-specific habitat attributes.** `BalticExample.shp` already
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
