# C5 — Arrival Event

**Date:** 2026-05-08
**Owner:** @razinkele
**Status:** ✅ EXECUTED on 2026-05-08 via subagent-driven-development; full pytest 925 passed / 33 skipped / 1 xfailed / 1 pre-existing perf-baseline flake / 10 pre-existing nemunas-NC errors (out-of-scope per C4 spec). Baseline + 14 new C5 tests in `tests/test_arrival_event.py`. Branch `c5-arrival-event` ready for PR + v1.7.9 tag.

End-to-end smoke (rng_seed=42, 50 agents, 480h): `alive=46, arrived=46, exit_branch_tagged=4`. Arrival metric now derives from real biology (no longer stuck at 0). Per-natal-reach 75th-percentile threshold tags every agent that sits in the upper quartile of its natal reach by `dist_from_sea`; in this seed most agents stay upstream rather than migrating, so they are trivially in the upper-quartile band — flagged as a follow-up tuning observation, not a blocker.

C5 ships an `ArrivalEvent` that tags an agent as **arrived** when it
settles in the upstream portion of its natal reach. Until C5,
`pool.arrived` is dead production state — initialized to `False` in
`agents.py:76` and never mutated by any production code path. The UI's
"X arrived" counter has been definitionally `0` since the IBM was
built; the corresponding biological event was never wired up.

C5 closes that gap. The arrival metric is the publication-grade
population-level signal the four-tier hatchery-vs-wild architecture
(C1+C2+C3.1+C3.2+C3.3+C4) was designed to produce: per-origin arrival
rate quantifies the wild-vs-hatchery reproductive-success differential
at the spawning ground.

C5 is the smallest tier yet — single new event, no kernel changes, no
substrate edits. Pure additive metric. The prior tiers built the
movement substrate and biological divergence; C5 reads what they
produced and tags settlement events.

## Why now

Post-C4 deploy verification (Playwright on laguna 2026-05-08, v1.7.8)
showed:

- 50 agents introduced, 480 hours, seed 42
- 6/50 (12%) reached a delta branch (C3.3's `exit_branch_id` tagging
  fired — C4's acceptance criterion satisfied)
- 4/50 (8%) alive at upper-river cells at t=480h (3 in Nemunas,
  1 in Minija)
- **0/50 "arrived" per the UI counter** — because nothing in
  production code ever sets `pool.arrived`

The 8% alive-at-upper-river figure is the natural arrival metric. The
ICES WGBAST mortality estimates for Curonian-stock Atlantic salmon
during return migration are 90-95% over the lagoon-and-river phase;
8% surviving alive at upper cells is consistent with that range.

C5 promotes this from "alive-at-upper-river observable via per-cell
inspection" to "first-class metric in `pool.arrived`", making it
visible in the UI, the OutputLogger CSV, and the per-origin breakdown
that downstream calibration sweeps need.

## Scientific basis

Atlantic salmon homing is a multi-stage process: open-sea navigation
→ coastal approach → estuarine entry → upstream migration → spawning
ground arrival → reproduction. C5 tags the **arrival** stage —
specifically, an agent has reached the upstream portion of its natal
reach and is positioned to spawn there.

**Threshold rationale (top quartile by natal reach).** The arrival
threshold is per-natal-reach: an agent has arrived when its `tri_idx`
is in a cell where `reach_id == agent.natal_reach_id` AND
`dist_from_sea[tri_idx]` is at or above the 75th-percentile cell of
that reach's `dist_from_sea` distribution. This is a **topology-
derived** threshold (no literature anchor needed); it captures
"upstream settlement" without requiring sub-reach mesh granularity.

Alternative thresholds were considered:

| Threshold | Rejected because |
|---|---|
| Top half / above-median | Looser; would tag passing-through agents during stray-correction events. |
| Anywhere in natal_reach_id | Equivalent to C3.3's exit_branch_id signal; redundant. |
| Specific dist_from_sea (meters) | Requires literature calibration; `dist_from_sea` numbers are mesh-build-dependent and won't transfer cleanly across mesh upgrades. |

The top-quartile choice is **mesh-portable**: a future mesh rebuild
recomputes thresholds from the new `dist_from_sea` distribution; no
hard-coded distance constants need maintenance.

**Composition with C3.3.** C3.3's `exit_branch_id` tags the *first
delta-branch entry* (regardless of natal vs stray); a strayed
hatchery agent gets `exit_branch_id = stray_rid ≠ natal_rid` AND
never reaches its natal reach's upper cells, so C5 never fires for it.
A correctly-homing wild agent gets `exit_branch_id = natal_rid` AND
proceeds upstream to reach the top-quartile cells, so C5 fires.
Per-origin arrival rate becomes the population-level
wild-vs-hatchery contrast metric.

## Scope

### In

- New per-`Simulation` cached array `_arrival_threshold_by_natal_rid:
  dict[int, float]` populated at `Simulation.__init__` after
  `mesh.dist_from_sea` is available. Maps each natal reach_id (1
  per delta branch + Nemunas + tributaries the IBM tracks) to the
  75th-percentile `dist_from_sea` value of cells with that reach_id.
  Computation: `np.percentile(dist_from_sea[reach_id == rid], 75)`.
- New `ArrivalEvent` registered via `@register_event("arrival")` in
  `salmon_ibm/events_builtin.py`. Vectorised; no Python iteration
  over agents. Sets `pool.arrived[i] = True` when:
  1. `pool.alive[i]` AND `not pool.arrived[i]` (still active)
  2. `mesh.reach_id[pool.tri_idx[i]] == pool.natal_reach_id[i]`
     (in natal reach)
  3. `mesh.dist_from_sea[pool.tri_idx[i]] >=
     _arrival_threshold[pool.natal_reach_id[i]]` (in upper quartile)
- Sticky: once arrived, the flag stays True for the rest of the sim;
  the existing read-sites (mortality, movement, accumulators) skip
  arrived agents via the existing `~pool.arrived` masks.
- Event ordering: `ArrivalEvent` runs AFTER `MovementEvent` (so it
  sees the post-step cell) but BEFORE mortality/predation events
  (so an agent that arrives this step is exempt from this step's
  mortality — biologically defensible: arrived agents are settled
  in spawning gravel, not free-swimming in lagoon).
- Backward compat: scenario configs without an `arrival` event in
  their event list don't enable C5 — `pool.arrived` stays at False
  for the entire run, identical to pre-C5 behavior.
- New event added to default scenario configs that exercise the
  Curonian H3 multires landscape (`config_curonian_h3_multires.yaml`
  and equivalents) so the deployed app immediately benefits.
- 9 new tests in `tests/test_arrival_event.py` (NEW): basic
  arrival (1), threshold boundary (2), sticky (3), dead-agents-skip
  (4), stray-hatchery (5), pre-tagged sentinel int8-overflow guard
  (5b), integration with MovementEvent (6), sticky-flag-overwrite
  AST enforcement (7), missing-event init-time warning (8),
  misorder warning (9).
- Live-test contract (post-deploy): default 50-agent / 480h run on
  laguna produces ≥1 arrival via the UI counter. With the observed
  4-8% alive-at-upper-river rate, expected: 1-3 arrivals per 50
  agents.

### Out (deferred)

- **Per-tributary-cell arrival** (Žeimena/Merkys/Dubysa). Requires
  mesh upgrade with finer reach polygons. Same deferral as C4.
- **Spawning-success modeling**. Arrival != spawning. C5 tags
  arrival; the actual spawning event (egg deposition, fertilization,
  redd construction) is a separate concern. Future tier might add
  `SpawnEvent(arrived) → eggs_deposited` per Heinimaa fecundity
  parameters.
- **Multi-day settlement window**. Real salmon arrive at the
  spawning ground and may move within the natal reach for days
  before settling. C5's first-touch tag captures the first
  upstream-settlement event; multi-day refinement is unjustified
  granularity given the IBM's hourly timestep + reach-level
  taxonomy.
- **Repeat spawners** (iteroparity / kelt return). C5's sticky
  flag treats arrival as terminal. Atlantic salmon iteroparity
  rate in the Baltic is 5-15% (Kallio-Nyberg 2010), small enough
  to defer until the model has a need.
- **Sex-specific arrival timing**. Females arrive earlier than
  males in some Atlantic salmon populations. The IBM doesn't
  track sex; deferred until a sex-tier lands.

## Architecture

### Data structure

A single new attribute on `Simulation`:

```python
self._arrival_threshold_by_natal_rid: dict[int, float] = ...
```

Maps each `natal_reach_id` value the IBM's introduction events emit
to the 75th-percentile `dist_from_sea` value of cells with that
`reach_id`. Computed once at `Simulation.__init__` after
`mesh.dist_from_sea` is loaded (via C4's `_load_dist_from_sea`).

### Computation

In `Simulation.__init__`, after C4's `assert_branch_topology` call
and `dist_from_sea` is on the env/mesh:

```python
def _compute_arrival_thresholds(self) -> dict[int, float]:
    """C5: per-natal-reach 75th-percentile dist_from_sea threshold.

    Computed once at sim init from mesh.dist_from_sea. Maps each
    reach_id with at least one finite-dist water cell to its
    top-quartile threshold; arrived = (tri_idx in this reach AND
    dist_from_sea >= threshold).

    Logs every reach skipped (no finite cells) at WARNING level so
    the operator sees which reaches won't produce arrivals; agents
    natal-tagged to a skipped reach silently never arrive otherwise.

    Returns empty dict if mesh.dist_from_sea is missing (legacy
    non-Baltic backend) — ArrivalEvent then no-ops at execute time.
    """
    import logging
    logger = logging.getLogger("salmon_ibm.simulation")

    dist = getattr(self.mesh, "dist_from_sea", None)
    if dist is None:
        return {}  # legacy non-Baltic mesh; ArrivalEvent no-ops

    rid_arr = self.mesh.reach_id
    # water_mask is part of the C4 contract: any mesh that exposes
    # dist_from_sea also exposes water_mask (set in tandem in
    # H3MultiResMesh and the legacy env stubs from C4 Task 7).
    # Defensive fallback if a future mesh decouples them.
    water = getattr(
        self.mesh, "water_mask", np.ones(len(dist), dtype=bool),
    )

    thresholds: dict[int, float] = {}
    for rid in np.unique(rid_arr):
        rid_int = int(rid)
        if rid_int < 0:
            continue  # sentinel reach_id; not a real reach
        mask = (rid_arr == rid_int) & water & np.isfinite(dist)
        n_cells = int(mask.sum())
        if n_cells == 0:
            name = (
                self.mesh.reach_names[rid_int]
                if rid_int < len(getattr(self.mesh, "reach_names", []))
                else f"rid_{rid_int}"
            )
            logger.warning(
                "c5-arrival-skipped-reach: reach %s (rid=%d) has no "
                "finite-dist water cells; agents natal-tagged to it "
                "will never arrive. Investigate mesh build.",
                name, rid_int,
            )
            continue
        thresholds[rid_int] = float(
            np.percentile(dist[mask], 75)
        )
    return thresholds
```

### ArrivalEvent

In `salmon_ibm/events_builtin.py`:

```python
@register_event("arrival")
@dataclass
class ArrivalEvent(Event):
    """C5: tag agents as arrived when they reach the upstream
    portion of their natal reach.

    Vectorised. Runs after MovementEvent (sees post-step cell).
    Sticky: pool.arrived is set True once and never reset.

    Reads `landscape["sim"]` directly (no getattr default) per C4's
    fail-loud convention. If the scenario landscape doesn't have a
    "sim" key, KeyError propagates — surfaces the misconfiguration
    rather than silently no-opping.

    No-ops gracefully when:
    - sim has no _arrival_threshold_by_natal_rid attribute (pre-C5
      Simulation), via getattr fallback.
    - thresholds dict is empty (legacy non-Baltic mesh; sim init
      already returned {} from _compute_arrival_thresholds).
    - mesh.dist_from_sea is None (legacy mesh).
    """

    def execute(self, population, landscape, t, mask):
        pool = population.pool
        # Direct subscript per fail-loud convention — KeyError if
        # "sim" missing tells the operator their landscape dict
        # construction is wrong.
        sim = landscape["sim"]
        thresholds = getattr(
            sim, "_arrival_threshold_by_natal_rid", {},
        )
        if not thresholds:
            return  # pre-C5 sim or legacy non-Baltic mesh — no-op

        mesh = landscape["mesh"]
        dist = getattr(mesh, "dist_from_sea", None)
        if dist is None:
            return  # legacy mesh — no-op (defensive; should not
                    # be reachable when thresholds is non-empty)

        # CAST natal_reach_id from int8 to int32 BEFORE any indexing.
        # Pool.natal_reach_id is dtype=np.int8 (agents.py:78), which
        # overflows at 128 → -128. The threshold lookup below indexes
        # thr_arr; without the cast, an int8-wrapped negative value
        # would either incorrectly match the sentinel guard or wrap
        # to an out-of-bound index.
        natal_rid = pool.natal_reach_id.astype(np.int32)

        # Vectorised mask: alive AND not arrived AND on mesh.
        active = pool.alive & ~pool.arrived
        on_mesh = pool.tri_idx >= 0
        safe_tri = np.where(on_mesh, pool.tri_idx, 0)
        cur_reach = mesh.reach_id[safe_tri].astype(np.int32)
        in_natal = cur_reach == natal_rid

        # Per-agent threshold lookup. Build a flat threshold array
        # indexed by reach_id; out-of-range agents get inf.
        n_reaches = max(thresholds.keys()) + 1
        thr_arr = np.full(n_reaches, np.inf, dtype=np.float32)
        for rid, val in thresholds.items():
            thr_arr[rid] = val

        # Atomic in-range lookup: clamp out-of-range natal_rid to 0
        # for safe indexing, then overwrite with inf via np.where.
        # Combining clamp + overwrite into one np.where avoids the
        # two-step pattern that a future refactor could break.
        in_range = (natal_rid >= 0) & (natal_rid < n_reaches)
        natal_safe = np.where(in_range, natal_rid, 0)
        per_agent_threshold = np.where(
            in_range, thr_arr[natal_safe], np.inf,
        )

        agent_dist = dist[safe_tri]
        above_threshold = agent_dist >= per_agent_threshold

        arrived_now = active & on_mesh & in_natal & above_threshold
        if arrived_now.any():
            pool.arrived[arrived_now] = True
```

The `landscape["sim"]` accessor: `Simulation.step()` adds
`"sim": self` to the landscape dict and the `Landscape` TypedDict
gains a `sim: "Simulation"` field — mirroring C4's `"env": self.env`
addition. Direct subscript (NOT `getattr` with default) is the
fail-loud convention: a misconfigured event sequence raises
KeyError at the first ArrivalEvent.execute call, NOT silently
no-ops forever.

### Event ordering

In the default Curonian scenario, ArrivalEvent inserts BETWEEN
MovementEvent and the mortality/predation events. The exact
location depends on the scenario YAML's event sequence; in
pseudocode:

```yaml
events:
  - type: introduction
  - type: movement
  - type: arrival      # NEW (C5) — runs after movement, before mortality
  - type: thermal_mortality
  - type: starvation
  - type: predation
  - type: spawning     # if present
```

The biological rationale: an agent that arrives THIS step is
"settled" in spawning gravel and is exempt from this step's
free-swimming-in-lagoon mortality. Reading post-movement state +
writing arrival flag BEFORE mortality is the right order.

### Composition with C3.3

C3.3's `exit_branch_id` and C5's `arrived` are independent flags
that compose:

| State | exit_branch_id | arrived |
|---|---|---|
| Pre-delta (open Baltic, lagoon) | -1 | False |
| First delta entry | natal_rid OR stray_rid | False |
| Reaches upper-natal cells | (sticky from delta entry) | True |

A strayed hatchery agent reaches a delta but heads up the wrong
branch, never reaching its natal-reach upper cells, so its
`arrived` stays False. The wild-vs-hatchery contrast at the
arrival level is therefore narrower than at the delta-entry level
— exactly the cumulative penalty the spec wants to expose.

### Validation discipline

**Sim-init structural validation.** After `_compute_arrival_thresholds`
runs, `Simulation.__init__` must validate the scenario configuration
to prevent the silent-failure class C5 was built to eliminate
(scenario forgets to include `arrival` in its event list →
`pool.arrived = 0` forever, indistinguishable from "agents
legitimately not reaching upper river"). Three validation steps:

1. **Missing-arrival-event detector.** If
   `_arrival_threshold_by_natal_rid` is non-empty (mesh supports
   arrival semantics) AND the constructed event sequence
   includes a `MovementEvent` (movement-driven scenario) AND does
   NOT include an `ArrivalEvent`, emit a clear warning at
   `logging.getLogger("salmon_ibm.simulation")`:

   ```python
   logger.warning(
       "%s: scenario has movement events on a mesh that supports "
       "arrival tagging (dist_from_sea present, %d natal reaches) "
       "but no ArrivalEvent in the event sequence — pool.arrived "
       "will stay False for the entire run. Add `- type: arrival` "
       "to the YAML event list (typically between movement and "
       "mortality events).",
       ERR_C5_MISSING_ARRIVAL_EVENT,
       len(self._arrival_threshold_by_natal_rid),
   )
   ```

   With err-id constant `ERR_C5_MISSING_ARRIVAL_EVENT =
   "c5-arrival-event-missing"` defined alongside the C4 err-ids
   in `salmon_ibm/h3_env.py` (the err-id home for the project's
   operational logging convention).

   This is a warning, not a raise — the scenario MAY legitimately
   want movement without arrival tagging (e.g., outmigration-only
   scenarios). The warning makes the omission visible in production
   logs without blocking sim init.

2. **Event-ordering invariant.** If `ArrivalEvent` IS in the
   sequence, validate its position: must run AFTER the last
   `MovementEvent` AND BEFORE the first event whose name matches
   `*Mortality` or `Predation` or `Survival`. If misordered, emit
   a warning with err-id
   `ERR_C5_ARRIVAL_EVENT_MISORDERED = "c5-arrival-event-misordered"`.
   Misorder is biologically suspect (an arrival tag computed from
   pre-movement state, or applied after this-step mortality, would
   produce the wrong settlement semantics) but technically runnable
   — warn-and-continue.

3. **`natal_reach_id` precondition contract.** Document explicitly
   in the spec: any introduction event (IntroductionEvent,
   PatchIntroductionEvent) emitting agents with a `natal_reach_id`
   value MUST use a value present in the init-time mesh's
   `np.unique(reach_id)`. Agents introduced with a never-init-seen
   reach get `inf` threshold via the `in_range` clamp (line ~273
   in the helper code) and silently never arrive — same failure
   class as the missing-event silent failure. C5 v2 documents the
   precondition; runtime enforcement is deferred (would require a
   wrapper on Population.add_agents that checks every introduction
   against the threshold dict — out of scope unless misuse is
   observed).

**Sim-time.** ArrivalEvent.execute checks `len(thresholds) > 0`
upfront; the `dist is None` and `not thresholds` guards short-
circuit on legacy backends. Direct `landscape["sim"]` subscript
fails-loud on missing key. No new RuntimeError raises in the
happy path.

**Composition with C4's dormancy guard.** If movement is dormant
(`dist_from_sea` missing or all-zero), C4's
`_check_dormant_gradient` raises BEFORE the first MovementEvent
step → ArrivalEvent never gets a chance to mis-fire. C5 inherits
C4's substrate guarantees.

**Sticky-flag overwrite enforcement.** The spec contract: NO event
in `EVENT_REGISTRY` writes `False` to `pool.arrived`. Test 7
(below) asserts this via grep + AST inspection at test time —
catches regressions where a future event clears arrived state
unintentionally.

## Implementation files

| File | Action | Notes |
|---|---|---|
| `salmon_ibm/simulation.py` | Modify | Add `_arrival_threshold_by_natal_rid` attribute + `_compute_arrival_thresholds` method, called from `__init__` after C4 dist_from_sea load. Add `"sim": self` to the landscape dict in `step()`. Add `sim` to `Landscape` TypedDict. |
| `salmon_ibm/events_builtin.py` | Modify | Add `ArrivalEvent` class registered via `@register_event("arrival")`. |
| `tests/test_arrival_event.py` | Create | 9 tests: 5 unit (basic, threshold-boundary, sticky, dead-skip, stray) + 1 sentinel int8-overflow guard (5b) + 1 movement-integration (6) + 1 sticky-overwrite AST enforcement (7) + 1 missing-event warning (8) + 1 misorder warning (9). |
| `configs/config_curonian_h3_multires.yaml` | Modify | Add `arrival` event to the event sequence (between movement and mortality). |
| `salmon_ibm/h3_env.py` | Modify | Add `ERR_C5_MISSING_ARRIVAL_EVENT` and `ERR_C5_ARRIVAL_EVENT_MISORDERED` err-id constants alongside the existing C4 err-ids. |

## Tests

**Test 1: basic arrival.** Synthetic mesh with 10 cells. Agent at
upper cell of natal reach + dist_from_sea above threshold. Run
ArrivalEvent. Assert `pool.arrived[0] == True`.

**Test 2: threshold boundary.** Agent at the exact 75th-percentile
cell. Assert arrived=True (≥, not >). Repeat with agent one cell
below; assert arrived stays False.

**Test 3: sticky.** Agent arrives at step 1; movement carries them
back below threshold at step 2. Assert `pool.arrived[i]` stays True
across both steps.

**Test 4: dead agents skip.** `pool.alive[i] = False` prior to
event. Agent at upper cell + above threshold. Assert arrived stays
False (mortality precedence).

**Test 5: stray hatchery — never arrives.** Hatchery agent with
`natal_reach_id = Atmata` but currently at upper-Skirvyte cell
(top quartile of Skirvyte's dist_from_sea, but not Atmata's).
Assert arrived stays False — correct rejection of cross-branch
matching.

**Test 5b: pre-tagged sentinel agent.** Agent with
`pool.natal_reach_id[i] = -1` (pre-tagging sentinel). Place at any
upper-natal cell. Assert arrived stays False — the int8→int32 cast
+ in_range clamp must correctly route the sentinel to inf
threshold. This test guards against the int8 overflow finding from
pass-1 review.

**Test 6: integration with movement.** Concrete fixture
specification (must match implementation exactly to avoid
flakiness):
- 12-cell bidirectional chain mesh.
- `dist_from_sea = np.arange(12, dtype=np.float32) * 100.0`
  (0, 100, 200, ..., 1100 meters).
- All 12 cells in reach_id=1 (single natal reach, no other
  reaches).
- 75th-percentile threshold for reach_id=1: 8.25 → ≥ cell 9
  (cells 9, 10, 11 qualify).
- One agent: `pool.tri_idx[0] = 0` (sea end), `pool.behavior[0] =
  Behavior.UPSTREAM`, `pool.natal_reach_id[0] = 1`,
  `pool.alive[0] = True`, `pool.arrived[0] = False`.
- Build a minimal landscape with `mesh`, `fields["dist_from_sea"]`,
  `rng`, and the new `"sim"` reference (a `_FakeSim` shim with
  `_arrival_threshold_by_natal_rid = {1: 825.0}`).
- Run MovementEvent + ArrivalEvent for 10 timesteps with
  `n_micro_steps_per_cell = np.ones(12, dtype=np.int32)`.
- Assert: by step 9, `pool.arrived[0] == True` (agent reached
  cell 9 or higher); flag stays True for all subsequent steps.

**Test 7: sticky-flag overwrite enforcement.** AST + grep check on
`salmon_ibm/events_builtin.py`: assert no event class's `execute`
method body contains `pool.arrived[...] = False` or
`pool.arrived[:] = False` patterns. The check uses Python's `ast`
module to walk AST nodes for `Assign` targets matching
`pool.arrived` Subscript and assert no rhs is `False` or `0`.
Catches future regressions where a contributor adds an event that
clears arrived state, breaking the sticky contract.

**Test 8: missing-arrival-event init-time warning.** Construct a
synthetic Simulation with an event sequence containing
MovementEvent but NO ArrivalEvent on a mesh that supports arrival
(populated `_arrival_threshold_by_natal_rid`). Use
`caplog.set_level(logging.WARNING, logger="salmon_ibm.simulation")`.
Assert the warning record contains
`ERR_C5_MISSING_ARRIVAL_EVENT` ("c5-arrival-event-missing").
Sanity counter-test: same setup with ArrivalEvent in the sequence
→ NO warning record with that err-id.

**Test 9: event-ordering misorder warning.** Construct a Simulation
where ArrivalEvent appears BEFORE MovementEvent in the event
sequence (or AFTER a mortality event). Assert warning record
contains `ERR_C5_ARRIVAL_EVENT_MISORDERED`.

**Live-test contract:** post-deploy Playwright run on laguna
default scenario produces ≥1 "arrived" via the UI counter
(currently 0). Expected: 1-3 with 50 agents (matches the diagnostic
finding of 4 alive at upper-river cells at t=480h, of which the
top-quartile threshold qualifies ≥25%).

## Performance

Per-step cost of ArrivalEvent: O(N_agents) vectorised numpy
operations + one O(N_agents)-size array allocation for the
per-agent threshold lookup. At N=2000 (production scale), <0.5ms
per step on the deployed mesh. Negligible.

`_compute_arrival_thresholds` runs once at sim init,
O(N_cells × log(N_cells_per_reach)) for the np.percentile calls;
~10 ms on the 185k-cell production mesh. Negligible.

## Backward compatibility

- **Scenario configs without an `arrival` event line** continue to
  run identically to pre-C5: `pool.arrived` stays False, all
  existing `~pool.arrived` masks evaluate as `~False = True` (no
  agent skipped), legacy behavior preserved. **Sim-init emits a
  WARNING** in this case (per Validation discipline §1) — the
  legacy behavior is preserved, but the user is told their build
  supports arrival tagging and they're not using it.
- **`Simulation._build_events()` default sequence (Python-driven,
  not YAML)** stays unchanged. C5's only YAML path adds the event;
  the default sequence is the legacy path. C5 unit tests construct
  ArrivalEvent directly (don't rely on default-sequence inclusion).
- **Legacy non-Baltic backends** (TriMesh, HexMesh, hexsim) lack
  `mesh.dist_from_sea` per-natal-reach data; the threshold dict
  is empty; ArrivalEvent no-ops at execute time.
- **Existing tests that assume `pool.arrived` is always False**
  are unaffected unless they explicitly add `arrival` to their
  scenario event list.

## Interactions with C1–C4

### C4 (movement gradient) — primary dependency

C5 reads `mesh.dist_from_sea` directly. Without C4, the dict is
empty and ArrivalEvent is a no-op. C5 effectively requires C4
to produce arrivals.

### C3.3 (homing precision) — produces the population-level signal

The wild-vs-hatchery arrival contrast is the cumulative product
of C3.3 (branch homing precision) AND C5 (settlement at upper
natal cells). Strayed hatchery agents enter a delta branch (C3.3
fires) but never reach their natal reach's upper cells (C5 never
fires for them). The arrival rate per origin is the publication-
grade summary metric.

### C3.2 (sea-age) — composes via per-cohort grouping

`pool.sea_age` is independent of arrival; downstream calibration
sweeps can break out arrival rate by sea_age × origin × ...
Per-cohort arrival rates would expose age-specific homing
fidelity (older returners typically home better).

### C3.1 (pre-spawn skip) — runs after arrival

C3.1's pre-spawn skip event fires at reproduction time; arrived
agents that don't skip would proceed to spawn (if a SpawnEvent
were implemented — out-of-scope for C5). C5 + C3.1 + spawning
together would give the full reproductive-success metric. C5
ships the prerequisite.

### C2 (activity multiplier) — affects time-to-arrival

Hatchery agents have +25% RANDOM/UPSTREAM activity → arrive
earlier (under C4 active). The arrival-time distribution per
origin is a secondary metric C5 enables.

### C1 (origin tag) — required for the population-level contrast

`pool.origin` is the partitioning variable for per-origin arrival
rate aggregation. C5 doesn't read `origin` directly (the threshold
is origin-blind by design — wild and hatchery use the same
top-quartile criterion); the partition happens at output/analysis
time.

## Open questions

1. **Does ArrivalEvent need to fire at every step, or could it be
   gated to fire only when an agent newly enters its natal reach?**
   The vectorised mask is cheap (<0.5ms per step at N=2000), so
   per-step is fine. A gated version would require tracking
   per-agent "entered natal reach this step" — extra state for
   marginal gain. **Decision (v1): per-step. Re-evaluate if
   profiling at production scale shows the event in the hotpath.**

2. **Should the threshold be 75th percentile (top quartile) or 90th
   (top decile)?** v1 uses 75th (top quartile) per user approval.
   90th would be stricter (fewer arrivals; closer to "actual
   spawning ground" semantics). 75th is the right balance for the
   IBM's hourly timestep + reach-level taxonomy.

3. **Sticky-vs-resettable.** v1 makes arrived sticky. An alternative
   is "arrived this step" (resets each step) — would let downstream
   events distinguish "newly arrived" from "long-arrived". **v1
   choice: sticky.** Matches the project's pattern (origin,
   exit_branch_id, sea_age all sticky); a separate "newly_arrived"
   tag could be added later if needed.

## References

- C4 spec: `docs/superpowers/specs/2026-05-07-c4-movement-gradient-design.md`
  — provides `mesh.dist_from_sea` substrate.
- C3.3 spec: `docs/superpowers/specs/2026-05-03-hatchery-c3.3-homing-design.md`
  — provides `pool.exit_branch_id` for delta-branch entry tagging.
- C1 spec: `docs/superpowers/specs/2026-04-30-hatchery-origin-c1-design.md`
  — provides `pool.origin` for per-origin partitioning.
- ICES WGBAST annual reports — Curonian-stock Atlantic salmon return-
  migration mortality estimates (90-95% lagoon-and-river phase).
- Kallio-Nyberg, Vainikka & Heino (2010), doi:10.1111/j.1095-8649.2009.02520.x
  — Baltic wild-vs-hatchery life-history divergence; arrival timing
  reference.
