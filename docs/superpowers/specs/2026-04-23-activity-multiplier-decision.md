# Activity-by-Temperature in Wisconsin Bioenergetics — Modeler Decision Required

**Status:** BLOCKED — awaiting sign-off before Task 17 of `2026-04-23-deep-review-fixes.md` can be implemented.
**Owner:** @razinkele
**Context:** Deep codebase review (2026-04-23) HIGH finding #6 — the current activity multiplier is a fixed 1.0-1.5 table vs Snyder et al. 2019's temperature-dependent exponential.

---

## The divergence in one sentence

Our respiration at 25 °C is approximately **13% of Snyder's** because we use fixed per-behavior activity multipliers (0.8-1.5) while Snyder scales activity as `exp(RTO · ACT · W^RK4 · exp(BACT · T))`, which grows exponentially with temperature and already exceeds ~5× at 20 °C.

## Why

Snyder et al. (2019) Wisconsin model (scenario XML lines 2528-2566):
```
activity = exp(RTO · ACT · W^RK4 · exp(BACT · T))
         ≈ 1.0 at T=10°C
         ≈ 2.5 at T=15°C
         ≈ 5.0 at T=20°C
         ≈ 9.7 at T=25°C     (per SNYDER_ACT = 9.7)
```

Our `salmon_ibm/bioenergetics.py` (lines 22-30):
```python
activity_by_behavior = {
    Behavior.HOLD: 1.0,
    Behavior.RANDOM: 1.2,
    Behavior.TO_CWR: 0.8,
    Behavior.UPSTREAM: 1.5,
    Behavior.DOWNSTREAM: 1.0,
}  # — temperature-INDEPENDENT
```

`tests/test_snyder_reference.py:200-219` already documents this: "Our R is the basal component only; Snyder's includes activity. At 25°C: Snyder R ≈ 68,721 J/h; Our R ≈ 9,242 J/h (basal only)."

## Why it matters for results

- **Survival inflated at warm temperatures.** At 20-25 °C (typical Baltic summer), we under-estimate respiration 3-5×. Energy reserves last longer than they should; fish appear to survive temperatures that should kill them by exhaustion.
- **Mortality pathway bypassed.** We compensate partially with `T_MAX = 26 °C` thermal mortality (hard kill switch), but Snyder's continuous temperature penalty is replaced with a discontinuous threshold.
- **Baltic salmon run through lagoon is at 12-22 °C.** This is exactly the range where the divergence matters most.
- **Any ensemble comparing "fraction arrived at spawning grounds" between scenarios will be inflated**, and the temperature sensitivity gradient will be too flat.

---

## The three options

### Option A — Adopt Snyder's activity-by-temperature formula (recommended)

Implement the full Snyder activity term, replacing the fixed per-behavior multiplier.

**Physics:**
```python
activity = np.exp(
    RTO * ACT * (mass_g ** RK4) * np.exp(BACT * temp_c)
)
r = RA * (mass_g ** RB) * np.exp(RQ * temp_c) * activity * OXY_CAL * mass_g / 24
```

with `RTO=0.0234, ACT=9.7, BACT=0.0405, RK4=0.13` (Snyder XML lines 9597-9604).

**Pros:**
- Matches Snyder 2019 exactly — parity with published Columbia parameterization
- Smooth temperature dependence; no `T_MAX=26 °C` discontinuity needed (could become a soft starvation path instead)
- `test_snyder_activity_exceeds_our_fixed_mult` and `test_our_basal_r_vs_snyder_full_r_at_high_temp` become "should match" instead of "documents divergence"

**Cons:**
- All ensemble baselines shift; any paper claiming "parity with Snyder" needs to be re-run and re-plotted
- Per-behavior modulation is lost unless added as a separate multiplier on top of Snyder's activity term (e.g., `activity = snyder_activity * behavior_mod` with `behavior_mod ∈ [0.8, 1.5]`)
- May interact unpredictably with the `T_MAX` thermal-kill switch — at T >= T_MAX the Snyder activity is already very high (~15+), so natural respiratory death would approach T_MAX; need to decide whether to keep T_MAX as a belt-and-suspenders or drop it

**Cites:** Snyder et al. (2019) *Science of the Total Environment* — HexSim scenario XML lines 2528-2566; test_snyder_reference.py lines 45-72 already implements the reference formula.

---

### Option B — Keep fixed multipliers but recalibrate RA upward

Leave the `activity_by_behavior` table as-is. Increase `RA` so our R at 15-20 °C (Baltic summer mean) matches Snyder's.

**How much to raise RA:** our R at 20 °C is ~20% of Snyder's (Snyder activity ≈ 5.0 at 20 °C). So `RA_new = RA_old × 5 = 0.0132` would match at 20 °C but overshoot at 10 °C (where Snyder activity ≈ 1.0, so we'd be 5× too hot at cold temps).

**Pros:**
- Minimal code change (single constant)
- Per-behavior activity multipliers keep biological intuition ("upstream = higher metabolic load")
- Parameter identity tests (`test_ra_matches`) break, which is an honest signal we've diverged

**Cons:**
- Scientifically inaccurate — no longer "Wisconsin model with Snyder parameterization"
- Recalibration is temperature-dependent by nature; a single scalar RA cannot match Snyder's exponential across the full range
- `test_parameter_identity::test_ra_matches` fails; every place that says "we match Snyder" needs a caveat

---

### Option C — Explicit documentation of the simplification, no code change

Keep the fixed multipliers. Add prominent documentation (in README and `bioenergetics.py` docstring) that explicitly states: "We deliberately use fixed per-behavior activity multipliers, not Snyder's temperature-dependent term. This under-estimates respiration at 20-25 °C by ~3-5×. Thermal mortality is handled discontinuously via T_MAX. Ensemble outputs are NOT directly comparable to Snyder 2019 at high temperatures."

**Pros:**
- No code change, no regression risk
- Honest about the simplification rather than hiding it

**Cons:**
- Leaves the scientific gap in place
- Future reviewers will re-flag this unless the docstring is very prominent

---

## Decision matrix

|                                 | A (Snyder activity) | B (recalibrate RA) | C (document only) |
|---------------------------------|---------------------|---------------------|-------------------|
| Matches Snyder 2019              | ✓                   | Partial (single T) | ✗                 |
| Temperature sensitivity correct   | ✓                   | Partial             | ✗                 |
| Per-behavior modulation preserved | With add-on         | ✓                   | ✓                 |
| Code change                       | ~15 lines + tests   | 1 line              | Docstring only    |
| Ensemble baseline shift           | Significant         | Moderate            | None              |
| Aligns with T_MAX hard kill        | Better (smooth)     | Unchanged           | Unchanged         |

---

## Interaction with Task 1 (bioenergetics starvation fix)

**Do A + Task 1 A together if possible.** Both change ED dynamics and both shift ensemble baselines. Combining them means one regression-baseline refresh, not two. If you pick Option A here and Option A there (lipid-first catabolism), the resulting physics is:

1. Respiration is Snyder's full formula (high at warm T)
2. Burned energy comes from lipid tissue at `ED_TISSUE=36 kJ/g`
3. ED declines smoothly as lipid is depleted
4. Mortality fires when ED drops below ED_MORTAL

This is the most defensible published physiology.

---

## Action required from modeler

Please indicate **A, B, or C** with rationale. If you prefer A with a per-behavior modulation add-on (e.g., `activity = snyder_activity * behavior_factor`), specify whether `behavior_factor` comes from the current 0.8-1.5 table or needs recalibration.

## Files that will change under Option A

- `salmon_ibm/bioenergetics.py:22-30` — replace `activity_by_behavior` dict with Snyder activity function
- `salmon_ibm/bioenergetics.py:49-62` — update `hourly_respiration` to apply temperature-dependent activity
- `salmon_ibm/bioenergetics.py:12-30` — add `ACT, BACT, RTO, RK4` to `BioParams` with defaults from Snyder
- `tests/test_snyder_reference.py:191-219` — convert "documents divergence" tests to "verifies match"
- `tests/test_bioenergetics.py` — add new tests for temperature-dependent activity
- Plan file: `docs/superpowers/plans/2026-04-23-deep-review-fixes.md` — move Task 17 from Phase 8 to Phase 1, mark as unblocked

Estimated implementation time once decision is made: **1-2 hours** (longer than Task 1 because of per-behavior modulation design + more extensive test refresh).
