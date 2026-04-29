# Salinity-driven osmoregulation stress for *Salmo salar*

**Date:** 2026-04-29
**Owner:** @razinkele
**Status:** 📋 DRAFT — awaiting writing-plans

This is the first of four queued Curonian-realism deferred items
(ordered: A osmoregulation → C hatchery vs wild → B predation → E habitat
realism). Each gets its own spec → plan → ship cycle.

## Why now

The v1.7.1 memory note flagged this exact gap: *"plan wires salinity into
`salinity_cost()` only; ion-balance turnover / osmoregulation is still
Chinook defaults, not defensible for S. salar."* The wiring shipped in
the v1.7.0 Curonian-realism plan; the *parameters* did not. Today's
lipid-first catabolism fix (commit `4247a11`) makes physiology the most
recently-touched code surface — natural code-locality for this work.

## Scope

**In:** Replace the body of `salmon_ibm/bioenergetics.py::salinity_cost`
with a smooth physiology-grounded function parameterised from peer-reviewed
*S. salar* literature. Add three new fields to `BioParams` and
`BalticBioParams` to hold the parameters. Add five unit tests.

**Out:**
- Calibration to field data (V1: literature-only validation).
- Ion-balance state on agents (gill ATPase activity tracking, acute
  transition spikes, recovery dynamics) — that's the A3 ambition tier;
  rejected for this plan, deferred indefinitely unless a use case
  surfaces.
- Salinity data plumbing — already in place via CMEMS forcing →
  `H3Environment` → `landscape["salinity"]`.
- Comparison to other *S. salar* IBMs (inSTREAM-Baltic, Säterberg 2023)
  — possible future plan.

## Choice of approach

Three options were considered:

1. **Linear-with-anchors** — two slope parameters anchored to specific
   published values (selected).
2. **Asymmetric tanh-saturating** — four parameters; smoother edges but
   two of the four have no clear literature anchor.
3. **Lookup-table** — same shape as (1) precomputed at fixed bins;
   adds YAML maintenance overhead with no expressiveness gain.

**(1) selected** because every parameter has a single, named citation;
salinity is naturally bounded so saturation isn't required; smallest
defensible change.

## Functional form

```python
def salinity_cost(salinity_ppt: np.ndarray, params: BioParams) -> np.ndarray:
    """Osmoregulation cost multiplier on respiration.

    Linear with separate slopes for hyper- and hypo-osmotic stress,
    anchored to S. salar literature values.

    Returns: multiplier ≥ 1.0; 1.0 at the blood iso-osmotic point.
    """
    s = np.clip(salinity_ppt, 0.0, 35.0)
    iso = params.SALINITY_ISO_OSMOTIC
    above = np.maximum(s - iso, 0.0) / max(35.0 - iso, 1.0)
    below = np.maximum(iso - s, 0.0) / max(iso, 1.0)
    return (
        1.0
        + params.SALINITY_HYPER_COST * above
        + params.SALINITY_HYPO_COST * below
    )
```

The function is pure (no I/O), vectorised, and stable for the input
range CMEMS produces.

## Parameters

Three new fields on `BioParams` and `BalticBioParams`. Both species use
the same defaults — Chinook and Atlantic salmon overlap on these
quantities at the literature-anchor level. Per the lipid-first precedent
(`ED_TISSUE: 5.0 → 36.0`), we update both rather than fork.

| Field | Default | Citation | Validation |
|---|---|---|---|
| `SALINITY_ISO_OSMOTIC` | `10.0` ppt | Wilson 2002 — *S. salar* blood plasma iso-osmolality is in the 9-12 ppt range | `0 < iso < 35` |
| `SALINITY_HYPER_COST` | `0.30` | Brett & Groves 1979 — ~30% above iso-osmotic respiration at full marine salinity (35 ppt) for euryhaline salmonids | `0 ≤ x ≤ 1` |
| `SALINITY_HYPO_COST` | `0.05` | Brett & Groves 1979 — ~5% above iso-osmotic respiration at freshwater (0 ppt); the asymmetry reflects hyper-osmotic stress being more energetically expensive than hypo | `0 ≤ x ≤ 1` |

The validation rules go in each class's `__post_init__`, raising
`ValueError` on out-of-range values, matching the existing pattern for
`RA`, `RQ`, `T_OPT`, `T_MAX`.

### Citations

- **Wilson, J. M., & Laurent, P. (2002).** Fish gill morphology: inside
  out. *Journal of Experimental Zoology*, 293(3), 192–213.
  https://doi.org/10.1002/jez.10124 — reviews iso-osmolality; *S. salar*
  blood plasma is ~340 mOsm ≈ 10-12 ppt.
- **Brett, J. R., & Groves, T. D. D. (1979).** Physiological energetics.
  In *Fish Physiology* (Vol. 8). Academic Press, pp. 279–352. — chapter
  reports salinity effects on respiration for migratory salmonids;
  ~30% increase at marine vs iso, ~5-10% at freshwater.

## Architecture

The change is local: one function body, six new fields (3 each on two
dataclasses), five new tests. No new modules, no new files.

`salinity_cost()` is called once today from `update_energy()` in
`bioenergetics.py:73-74`. The new body has the same input/output
contract (NumPy array in, NumPy array out, multiplier ≥ 1) — no caller
changes.

The salinity field per agent already flows from CMEMS forcing through
the env model. No data wiring changes.

## Tests

Five new tests in `tests/test_bioenergetics.py`:

1. `test_salinity_cost_at_iso_returns_unity` — `salinity_cost(np.array([10.0]), BioParams())[0]
   == pytest.approx(1.0)`. Lock in the iso-osmotic anchor.

2. `test_salinity_cost_marine_matches_brett_groves` — at salinity=35 ppt,
   cost is approximately `1 + 0.30 * 1.0 == 1.30`. Lock in the marine
   anchor.

3. `test_salinity_cost_freshwater_above_one` — at salinity=0 ppt, cost
   is approximately `1 + 0.05 * 1.0 == 1.05` and strictly less than the
   marine cost. Locks the asymmetry.

4. `test_salinity_cost_smooth_monotonic_outside_iso` — generates a sweep
   from 0 to 35 ppt; asserts cost is monotonically non-decreasing as we
   move from iso to either extreme.

5. `test_salinity_cost_vectorised` — passes a `(100,)` salinity array;
   asserts output has same shape and all values ≥ 1.0.

Plus three validation tests in the existing `TestBioParamsValidation`
class:

6. `test_negative_iso_raises` — `BioParams(SALINITY_ISO_OSMOTIC=-1.0)`
   raises `ValueError`.
7. `test_iso_above_max_raises` — `BioParams(SALINITY_ISO_OSMOTIC=40.0)`
   raises `ValueError`.
8. `test_negative_hyper_cost_raises` — `BioParams(SALINITY_HYPER_COST=-0.1)`
   raises `ValueError`.

Total new tests: 8 (5 functional + 3 validation).

## Risk + regression surface

**Behavior change surface:** any test that exercises agents transiting a
salinity gradient. The Chinook lookup the new function replaces was
already in production; switching to the new function will shift the
energy budget for migrants in the lagoon (~5 ppt, brackish) and Baltic
(~7 ppt typical, brackish-marine). Likely to be small, since both
landscapes sit relatively close to iso (10 ppt).

**Tests that may need updating:** any pre-existing migration test
that asserts specific energy/mortality numbers post-transit. Per the
v1.7.1 lipid-first precedent — the spec there warned of similar
regression-baseline risk and the actual breakage was zero (the new
formula also conserved energy by construction). Same expectation here:
the new function preserves the cost = 1.0 case (when salinity = iso),
so any test running at iso-salinity won't shift.

**Mitigation:** run full pytest suite, surface regressions, fix or
update them. Same playbook as v1.7.1.

## Success criteria

- [ ] `salinity_cost()` body replaced with linear-with-anchors form.
- [ ] All 8 new tests pass.
- [ ] Full pytest suite stays green (current: 815 passing on main;
      expected post-change: 823 passing).
- [ ] Manual sanity check: cost curve plotted from 0 to 35 ppt looks
      smooth and matches the published *S. salar* shape (low at iso,
      gradual hypo rise, steeper hyper rise).
- [ ] Plan stamp on the eventual implementation plan: ✅ EXECUTED.

## Estimated implementation time

**1-2 days** including TDD cycle, regression sweep, and any test fixups.
Single function body; same code-locality as the lipid-first work shipped
2026-04-29; same regression-management playbook (run suite, fix what
breaks).

## Out-of-scope (deferred)

- **Ion-balance state on agents** (the A3 ambition tier from
  brainstorming). Deferred indefinitely. If a future use case requires
  per-agent acclimation history, that's a separate plan.
- **Field-data calibration** (V2/V3 from brainstorming). Deferred. The
  V1 literature-only path closes the headline critique without a
  calibration phase. If the model later needs to match observed
  Curonian/Baltic migrant energy trajectories, a follow-on plan can
  add calibration.
- **Acute transition spikes.** Modeled via Brett & Groves' chronic-cost
  framing only. Not modeled because IBM-level signals are dominated by
  movement and encounter, not individual transition events.
