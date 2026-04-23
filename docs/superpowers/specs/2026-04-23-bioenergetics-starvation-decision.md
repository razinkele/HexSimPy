# Bioenergetics Starvation Physics — Modeler Decision Required

**Status:** BLOCKED — awaiting sign-off before Task 1 of `2026-04-23-deep-review-fixes.md` can be implemented.
**Owner:** @razinkele
**Context:** Deep codebase review (2026-04-23) identified a starvation-mortality bug in `salmon_ibm/bioenergetics.py:64-85`.

---

## The bug in one sentence

Under the current physics, **energy density (ED) never declines for starving migrants** until `MASS_FLOOR_FRACTION=0.5` of initial mass is reached — at which point ED suddenly drops. Starvation mortality is hidden behind a two-regime step function.

## Why

`update_energy()` computes proportional mass loss:

```python
energy_fraction = new_e_total_j / e_total_j
new_mass = mass_g * energy_fraction
new_ed = new_e_total_j / (new_mass * 1000)  # = ed_before (constant!)
```

Because mass declines in the same ratio as energy, ED is invariant. Only when `MASS_FLOOR_FRACTION` caps `new_mass` does ED finally start to drop — but by then the agent has lost 50% of its initial mass already, and the ED drop is abrupt.

The existing test `test_energy_density_decreases_monotonically` only asserts ED doesn't *increase*, so the flatness slipped past regression checks.

## Why it matters for results

Non-feeding migrating salmon should show smooth ED decline toward `ED_MORTAL = 4.0 kJ/g` as they burn reserves on the spawning run. Current physics:

- Survives too long — mortality fires only after mass-floor engagement
- Behavior selection (Snyder et al. 2019 p_table) is keyed on ED, so flat ED → stale behavior probabilities
- Any ensemble output summarizing "mean ED over time" shows a plateau where biology predicts decline

---

## The three options

### Option A — Lipid-first catabolism (recommended)

`ED_TISSUE = 36 kJ/g` (pure lipid catabolism, Brett 1995 via Breck 2008).

**Physics:**
```python
mass_lost = r_hourly / (ED_TISSUE * 1000)
new_mass = mass_g - mass_lost  # mass declines SLOWER than energy
new_ed = (e_total - r_hourly) / (new_mass * 1000)  # DECLINES toward ED_MORTAL
```

Because `ED_TISSUE > ed_kJ_g` (36 > 6 typical), burning lipid concentrates the loss on the fat pool; remaining body's mean ED drops smoothly.

**Math check:**
- Start: M=1000 g, ED=6, E=6·10⁶ J
- One step, r=10,000 J: mass_lost=0.278 g, M'=999.72 g, ED'=5.99 kJ/g ✓ smooth decline

**Pros:** Matches published salmonid fasting physiology; ED declines monotonically to ED_MORTAL; floor becomes a genuine numerical guard (never biologically engaged).

**Cons:** Changes downstream ED trajectories in all ensemble results — regression baselines need refresh. `test_energy_conservation_single_step` may need loosening.

**Cites:** Brett (1995) *Physiological Ecology of Pacific Salmon*; Breck (2008) Trans Am Fish Soc 137(1), 340-356 DOI `10.1577/t05-240.1`.

---

### Option B — Mixed lipid + protein catabolism

`ED_TISSUE = 25-30 kJ/g` with explicit citation.

Same formula as A, but models realistic mixed-substrate burn (protein ~20 kJ/g + lipid ~36 kJ/g in species-dependent ratio). ED still declines (because 25 > whole-body 6), just faster than A.

**When to prefer B over A:** if there's Baltic-salmon-specific literature showing non-trivial protein catabolism during the spawning run. Otherwise A is simpler and equally defensible.

**Cites needed:** *Salmo salar* body composition studies during upstream migration.

---

### Option C — Retain proportional loss, remove mass floor

Keep `new_mass = mass_g * energy_fraction` (ED stays constant) but delete `MASS_FLOOR_FRACTION`; let mortality fire when `new_mass <= 0` or `new_e_total_j <= 0`.

**Pros:** Minimal code change; preserves any ensemble baselines that locked in the flat-ED regime.

**Cons:** Scientifically inaccurate. ED stays flat at initial value until instantaneous death — no gradient for behavior selection. Matches no published salmonid physiology.

---

## Decision matrix

|                          | A (lipid-first, 36) | B (mixed, ~28) | C (proportional) |
|--------------------------|---------------------|----------------|------------------|
| Physics direction        | ✓ declines          | ✓ declines     | ✗ flat           |
| Literature-grounded      | ✓ Brett/Breck       | ✓ if cite      | ✗                |
| Ensemble-baseline impact | High                | High           | None             |
| Implementation effort    | 1 line change       | 1 line change  | 3 line change    |
| Behavior-selection gradient | ✓ smooth         | ✓ smooth       | ✗ binary         |

---

## Action required from modeler

**Please indicate A, B, or C (with rationale, especially if B — we need the citation).**

- If **A**: implementation lands today. Test baselines will be refreshed and any downstream consumers of ED trajectories will see smoother curves.
- If **B**: please supply the citation for Baltic salmon mixed catabolism and the preferred `ED_TISSUE` value (25, 28, 30?).
- If **C**: we'll implement, but recommend documenting in the README that ED is flat-by-design to avoid future reviewers re-flagging this.

---

## Adjacent question (optional but related)

`MASS_FLOOR_FRACTION = 0.5` — under Option A, is the floor still needed as a numerical guard, or should it be `0.1` (closer to "truly pathological") or removed entirely? Under A, the floor should almost never engage biologically, so the specific value is mostly a division-by-zero safeguard.

---

## Files that will change under Option A

- `salmon_ibm/bioenergetics.py:20` — `ED_TISSUE: float = 5.0` → `36.0`
- `salmon_ibm/bioenergetics.py:65-85` — rewrite `update_energy()` with tissue-catabolism formula
- `tests/test_bioenergetics.py` — add `test_energy_density_declines_under_starvation`, `test_starvation_triggers_mortality_when_energy_depleted`; update `test_energy_conservation_single_step` if it fails
- The Task 1 plan step in `docs/superpowers/plans/2026-04-23-deep-review-fixes.md` has the exact TDD sequence

Estimated implementation time once decision is made: **30 minutes** (including test refresh).
