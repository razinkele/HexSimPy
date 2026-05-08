# C5 — Arrival Event

**Date:** 2026-05-08
**Owner:** @razinkele
**Status:** 📋 DRAFT v1 — awaiting review-loop passes.

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
- 6 new tests in `tests/test_arrival_event.py` (NEW): synthetic
  fixture covering basic arrival, threshold-boundary, sticky-once,
  per-origin-stratification, dead-agents-skip, and integration
  with `MovementEvent` execution order.
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
def _compute_arrival_thresholds(self):
    """C5: per-natal-reach 75th-percentile dist_from_sea threshold.

    Computed once at sim init from mesh.dist_from_sea. Maps each
    reach_id with at least one cell to its top-quartile threshold;
    arrived = (tri_idx in this reach AND dist_from_sea >= threshold).
    """
    if not getattr(self.mesh, "dist_from_sea", None) is not None:
        return {}  # legacy non-Baltic mesh; ArrivalEvent no-ops
    dist = self.mesh.dist_from_sea
    rid_arr = self.mesh.reach_id
    water = self.mesh.water_mask
    thresholds: dict[int, float] = {}
    for rid in np.unique(rid_arr):
        rid_int = int(rid)
        mask = (rid_arr == rid_int) & water & np.isfinite(dist)
        if not mask.any():
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
    """

    def execute(self, population, landscape, t, mask):
        pool = population.pool
        sim = landscape.get("sim")
        thresholds = (
            sim._arrival_threshold_by_natal_rid
            if sim is not None
            else None
        )
        if not thresholds:
            return  # legacy non-Baltic landscape — no-op

        mesh = landscape["mesh"]
        dist = getattr(mesh, "dist_from_sea", None)
        if dist is None:
            return  # legacy mesh — no-op

        # Vectorised mask: alive AND not arrived AND in natal reach
        # AND at-or-above threshold.
        active = pool.alive & ~pool.arrived
        on_mesh = pool.tri_idx >= 0
        cur_reach = mesh.reach_id[np.where(on_mesh, pool.tri_idx, 0)]
        in_natal = cur_reach == pool.natal_reach_id

        # Per-agent threshold lookup. The agent's threshold depends
        # on its natal_reach_id; a vectorised lookup uses np.take.
        # Build a flat threshold array indexed by reach_id.
        n_reaches = max(thresholds.keys()) + 1 if thresholds else 0
        thr_arr = np.full(n_reaches, np.inf, dtype=np.float32)
        for rid, val in thresholds.items():
            thr_arr[rid] = val

        # Agents whose natal_reach_id is out of the threshold array
        # range (e.g., -1 sentinel for pre-tagging agents) get inf
        # threshold → never arrive.
        natal_safe = np.where(
            (pool.natal_reach_id >= 0)
            & (pool.natal_reach_id < n_reaches),
            pool.natal_reach_id,
            0,
        )
        per_agent_threshold = thr_arr[natal_safe]
        per_agent_threshold[
            (pool.natal_reach_id < 0) | (pool.natal_reach_id >= n_reaches)
        ] = np.inf

        agent_dist = dist[np.where(on_mesh, pool.tri_idx, 0)]
        above_threshold = agent_dist >= per_agent_threshold

        arrived_now = active & on_mesh & in_natal & above_threshold
        if arrived_now.any():
            pool.arrived[arrived_now] = True
```

The `landscape["sim"]` accessor: `Simulation.step()` already has
self-reference patterns; add `"sim": self` to the landscape dict,
mirroring C4's `"env": self.env` addition. Direct attribute access
(NOT `getattr` with default) — fail-loud per the C4 convention.

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

**Sim-init:** if `mesh.dist_from_sea` is present and any reach has
finite `dist_from_sea` cells, `_arrival_threshold_by_natal_rid` is
populated. If absent (legacy non-Baltic mesh), the dict is empty
and ArrivalEvent no-ops.

**Sim-time:** ArrivalEvent.execute checks `len(thresholds) > 0`
upfront; the `dist is None` and `not thresholds` guards short-
circuit on legacy backends.

No new RuntimeError raises in C5. Composition with C4's dormancy
guard: if movement is dormant (`dist_from_sea` missing), C4's
guard fires first; C5 never runs. If movement is active but
ArrivalEvent isn't in the scenario, `pool.arrived` stays False —
the legacy behavior. No silent failure mode.

## Implementation files

| File | Action | Notes |
|---|---|---|
| `salmon_ibm/simulation.py` | Modify | Add `_arrival_threshold_by_natal_rid` attribute + `_compute_arrival_thresholds` method, called from `__init__` after C4 dist_from_sea load. Add `"sim": self` to the landscape dict in `step()`. Add `sim` to `Landscape` TypedDict. |
| `salmon_ibm/events_builtin.py` | Modify | Add `ArrivalEvent` class registered via `@register_event("arrival")`. |
| `tests/test_arrival_event.py` | Create | 6 unit + integration tests. |
| `configs/config_curonian_h3_multires.yaml` | Modify | Add `arrival` event to the event sequence (between movement and mortality). |

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

**Test 6: integration with movement.** Set up bidirectional
chain mesh; agent in UPSTREAM behavior; cell 9 (chain end) is
top-quartile of natal reach. Run MovementEvent then ArrivalEvent
for K timesteps. Assert agent arrives by step ~5, sticky from
that point on.

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
  agent skipped), legacy behavior preserved.
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
