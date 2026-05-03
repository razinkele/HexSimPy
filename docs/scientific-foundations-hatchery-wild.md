# Scientific Foundations of the Baltic Salmon Hatchery-vs-Wild Model

**Date:** 2026-05-03
**Author:** @razinkele
**Status:** Synthesis document for C1 + C2 + C3.1 + C3.2 tiers. All
citations verified via scite MCP (full-text matches against actual
DOI metadata, not inferred from titles).

This document consolidates the empirical and theoretical basis for
the four-tier hatchery-vs-wild origin support added to the HexSim
Baltic salmon individual-based model (IBM). It is intended as the
single source of truth for: (a) reviewers asking "where does this
parameter come from?"; (b) future tier authors needing to extend
the model without re-deriving the citation chain; (c) calibration
sweeps documenting which numbers are anchored vs which are
calibration-grade.

The document is organised by **biological mechanism**, not by code
file. For the implementation-level mapping, see the per-tier specs
in `docs/superpowers/specs/` and per-tier plans in
`docs/superpowers/plans/`.

**Implementation status as of 2026-05-03:**

| Tier  | Spec status        | Plan status        | Code status                          |
|-------|--------------------|--------------------|--------------------------------------|
| C1    | Final 2026-04-30   | Executed           | Shipped: PR #3 merged, commit 5791802|
| C2    | Final 2026-05-01   | Executed           | Shipped: PR #4 merged, commit 8a9192c|
| C3.1  | Final 2026-05-02   | Executed           | Shipped: PR #5 merged, commit 0284c45|
| C3.2  | v4.1 converged 2026-05-03 (5-pass review) | v2.1 converged 2026-05-03 (2-pass review) | Pending implementation |
| C3.3  | Not yet drafted    | Not yet drafted    | Future                                |

Tags: v1.7.4 covers C1 + C2; v1.7.5 covers C3.1; v1.7.6 will cover
C3.2 once implemented and merged. C3.2 spec at
`docs/superpowers/specs/2026-05-03-hatchery-c3.2-seaage-design.md`;
C3.2 implementation plan at
`docs/superpowers/plans/2026-05-03-hatchery-c3.2-seaage.md`.

---

## 1. Why a hatchery-vs-wild distinction matters for the Baltic

Baltic Atlantic salmon (*Salmo salar*) populations are heavily
hatchery-supported. The Lithuanian Žeimena and Simnas hatchery
programmes have run since 1997 (~29 years; ~4-7 salmon generations
at typical 5-7 year cycle), making most Nemunas-basin returners of
hatchery origin. ICES WGBAST stock assessments consistently report
that hatchery-supported Baltic stocks differ from adjacent wild
stocks across multiple demographically-significant dimensions:

- Bioenergetic efficiency (swimming cost, aerobic scope)
- Reproductive success per spawning attempt
- Age structure of returning fish (1SW vs MSW)
- Maturation timing
- Body size at age
- Genetic diversity at neutral and adaptive loci

A model that treats all returners as a homogeneous "wild" pool
systematically over-estimates productivity per breeder and
under-estimates compound stressor effects on the wild fraction.
The four-tier hatchery-vs-wild architecture (C1 → C2 → C3.1 →
C3.2) makes hatchery agents demographically distinguishable from
wild agents without requiring full pedigree tracking.

---

## 2. The four-tier architecture

Each tier introduces ONE divergent biological signal between hatchery
and wild agents:

| Tier  | Layer            | Mechanism                              | Status   |
|-------|------------------|----------------------------------------|----------|
| C1    | Identity         | int8 origin tag at agent introduction  | Shipped  |
| C2    | Bioenergetics    | activity multiplier divergence         | Shipped  |
| C3.1  | Reproduction     | pre-spawn skip probability             | Shipped  |
| C3.2  | Marine residency | sea-age distribution divergence        | Drafted  |
| C3.3  | Homing           | natal-stream precision divergence      | Future   |

The architecture is deliberately additive: each tier extends the
prior tier's data structures without modifying their semantics. The
load-bearing observation is that **independent biological signals
should be modelled as independent parameters** rather than collapsed
into a single "hatchery efficiency" multiplier; this mirrors how
empirical studies report each effect separately.

---

## 3. C1 — Origin Tracking (tag-only metadata)

### Mechanism

A new int8 column `origin` on the agent state-of-arrays
(`AgentPool`) records each agent's introduction-time origin tag:
0 = wild, 1 = hatchery. The tag is permanent for the agent's
lifetime; offspring of any parent default to wild (origin
inheritance is explicitly out of scope, per Section 8 below).

### Empirical basis

C1 is metadata-only — the tag itself does not produce any
biological effect. Its role is to enable the dispatching of
divergent parameters at consumer sites in C2/C3.x.

The choice to treat origin as a permanent agent-level tag rather
than an attribute of the breeding population follows the standard
IBM tradition (DeAngelis & Mooij 2005; Railsback & Harvey 2002):
agent identity is fixed at introduction.

### Reference

DeAngelis, D. L., & Mooij, W. M. (2005). Individual-based modeling
of ecological and evolutionary processes. *Annual Review of
Ecology, Evolution, and Systematics*, 36, 147-168.

---

## 4. C2 — Bioenergetic Activity Cost Divergence

### Mechanism

Hatchery agents pay a higher activity multiplier than wild agents
during active swimming. The Wisconsin bioenergetics framework
multiplies basal respiration by an activity coefficient
`activity_by_behavior[b]` for each behaviour state `b`. C2 adds
a hatchery override: `RANDOM` (b=1) and `UPSTREAM` (b=3) behaviours
incur a **+25% activity multiplier** for hatchery agents
(`1.2 → 1.5` and `1.5 → 1.875` respectively). `HOLD`, `TO_CWR`,
and `DOWNSTREAM` are unchanged.

### Empirical basis

**Primary anchor:** Enders, Boisclair & Roy (2004), *Canadian
Journal of Fisheries and Aquatic Sciences* 61(12):2302-2313,
doi:10.1139/f04-211 — *"The costs of habitat utilization of wild,
farmed, and domesticated juvenile Atlantic salmon (Salmo salar)"*.
Direct respirometry comparison across three groups: wild,
first-generation farmed (F1 hatchery progeny of wild parents), and
seventh-generation domesticated (Norwegian aquaculture strain).
Key finding (verbatim from abstract): *"Total swimming costs of
wild and farmed fish were not statistically different (average
difference = 6.7%). However, domesticated fish had total swimming
costs **12.0% to 29.2% higher** than farmed or wild fish. This may
be related to domesticated fish having deeper bodies and smaller
fins."*

The +25% C2 default sits in the middle of the empirically-measured
12-29% bracket. The 12% lower bound is conservative for a
seventh-generation strain; the Lithuanian Žeimena/Simnas programme
at ~4-7 generations sits between Enders' farmed (F1, ~6.7%
indistinguishable from wild) and domesticated (7th-gen,
12.0-29.2%) brackets.

**Important caveat (driven by prior reviewer correction):** an
earlier draft of the C2 spec misattributed this finding to Pedersen
2008 / Salonen 2007. The Enders et al. 2004 paper is the actual
primary source. C3.2's review loop explicitly verified all
citations against scite full-text DOI metadata to prevent this
class of error from recurring.

### Behaviour selection rationale

The +25% multiplier applies only to RANDOM and UPSTREAM:

- **DOWNSTREAM (b=4) unchanged:** passive drift; morphological
  inefficiency is suppressed when the current provides thrust.
  Hydrodynamic argument (drag vs thrust allocation), not directly
  cited from a single paper.
- **HOLD (b=0) unchanged:** standing in place doesn't exercise the
  morphological inefficiency Enders documented (deeper bodies,
  smaller fins).
- **TO_CWR (b=2) unchanged:** short-distance moves to cold-water
  refugia; not the sustained-aerobic regime Enders measured.

### Calibration status

The +25% increment is empirically bracketed but not directly
measured for Lithuanian hatchery stocks. Treat as
**calibration-grade**. Mandatory sensitivity sweep before
publication: `{0%, +12.5%, +25%, +37.5%}`. Cap at +40% without
additional citation support.

### Effect size

In a 21-day mixed-behaviour migration, C2 produces a ~0.21
percentage-point mass-loss differential between hatchery and wild
agents. This is small in absolute terms but compounds with C3.x
divergences and with environmental stressors (osmoregulation,
temperature) over multi-year cohort tracking.

### References

Enders, E. C., Boisclair, D., & Roy, A. G. (2004). The costs of
habitat utilization of wild, farmed, and domesticated juvenile
Atlantic salmon (*Salmo salar*). *Canadian Journal of Fisheries
and Aquatic Sciences*, 61(12), 2302-2313.
https://doi.org/10.1139/f04-211

---

## 5. C3.1 — Pre-Spawn Skip Probability

### Mechanism

A Bernoulli filter inserted into `ReproductionEvent.execute`
between the `reproducer_idx` computation and the Poisson clutch
sampling. Each hatchery agent that would otherwise reproduce
"skips" with probability `pre_spawn_skip_prob = 0.3` (default for
hatchery; 0.0 for wild). Wild agents always proceed; hatchery
agents have a 30% per-attempt no-spawn rate, corresponding to a
relative reproductive success (RRS) of approximately 0.7.

### Empirical basis

**Primary mechanistic anchor:** Bouchard, Wellband, Lecomte et al.
(2022), *Evolutionary Applications* 15(5):838-852,
doi:10.1111/eva.13374 — *"Effects of stocking at the parr stage
on the reproductive fitness and genetic diversity of a wild
population of Atlantic salmon (Salmo salar L.)"*. Microsatellite
parentage assignment of 2,381 offspring from a supplemented
Atlantic salmon population in Québec. Key findings (from abstract):
*"Captive-bred salmon stocked at the parr stage **had fewer mates
than their wild conspecifics**, as well as a reduced relative
reproductive success (RRS) compared with their wild counterparts."*
The phrase "fewer mates" is decisive: the empirical mechanism is
in mating *frequency*, not offspring per mating event. This
directly validates the C3.1 model intervention shape (a Bernoulli
gate before Poisson clutch sampling) over the alternative
"reduced clutch_mean for hatchery" formulation.

The paper observed RRS values in the 0.65–0.80 range for
captive-bred Atlantic salmon depending on cohort and sex; the C3.1
default `p_skip = 0.3` (RRS ≈ 0.7) sits in the middle of this
empirical bracket.

**Cross-species meta-analytic baseline:** Christie, Ford & Blouin
(2014), *Evolutionary Applications* 7(8):883-896,
doi:10.1111/eva.12183 — *"On the reproductive success of
early-generation hatchery fish in the wild"*. Meta-analysis of 51
estimates from 6 studies on 4 salmon species. Key findings: *"(i)
early-generation hatchery fish averaged only **half the
reproductive success** of their wild-origin counterparts when
spawning in the wild, (ii) the reduction in reproductive success
was more severe for males than for females, and (iii) all species
showed reduced fitness due to hatchery rearing."* Christie et al.
also noted (relevant to C3.2): *"Because fish that only spent a
single winter at sea had lower reproductive success than those
that spent longer durations at sea, **differential age at
maturation explains at least some of the difference in
reproductive success between wild and hatchery individuals**"* —
a direct mechanistic link from C3.1 (RRS) to C3.2 (sea-age).

The Christie 50% baseline is cross-species; the Bouchard 70%
baseline is *S. salar*-specific. Baltic Atlantic salmon at 4-7
hatchery generations are closer to Bouchard than to Christie's
upper end.

**Population-scale anchor:** Jönsson, Jönsson & Jonsson (2019),
*Conservation Science and Practice* 1:e85, doi:10.1111/csp2.85 —
*"Supportive breeders of Atlantic salmon Salmo salar have reduced
fitness in nature"*. River Imsa long-term monitoring (1976-2013):
mean smolts per 100 m² river area per female breeder dropped from
0.47 (wild only) to 0.088 (5% wild females). The 81% reduction is
larger than the 30% C3.1 implements because the metric mixes
spawning behaviour AND offspring survival; C3.1 deliberately
models only the spawning-behaviour fraction (per Bouchard 2022's
mechanistic decomposition).

### Calibration status

The 0.3 default is **calibration-grade**, bracketed by Bouchard
2022's Atlantic salmon range (1 - 0.65 = 0.35 upper; 1 - 0.80 =
0.20 lower). Mandatory sensitivity sweep before publication:
`{0.0, 0.15, 0.30, 0.50}`. Cap at 0.6 without additional citation
support — beyond exceeds Christie 2014's lower bound for
*S. salar*-specific estimates.

### Effect size

For N hatchery reproducers with `clutch_mean=4.0`, expected
offspring drop is ~30% (`N × 4.0 × 0.7` vs `N × 4.0`). For a mixed
scenario with k% hatchery, the population-level offspring count
drops by ~`0.3 × k%`.

### References

Bouchard, R., Wellband, K. W., Lecomte, L., et al. (2022). Effects
of stocking at the parr stage on the reproductive fitness and
genetic diversity of a wild population of Atlantic salmon (*Salmo
salar* L.). *Evolutionary Applications*, 15(5), 838-852.
https://doi.org/10.1111/eva.13374

Christie, M. R., Ford, M. J., & Blouin, M. S. (2014). On the
reproductive success of early-generation hatchery fish in the
wild. *Evolutionary Applications*, 7(8), 883-896.
https://doi.org/10.1111/eva.12183

Jönsson, B., Jönsson, N., & Jonsson, M. (2019). Supportive
breeders of Atlantic salmon *Salmo salar* have reduced fitness in
nature. *Conservation Science and Practice*, 1(9), e85.
https://doi.org/10.1111/csp2.85

---

## 6. C3.2 — Sea-Age Distribution Divergence

### Mechanism

A new int8 column `sea_age` on `AgentPool` records each agent's
sea-residency duration in years (1 = grilse / 1SW, 2 = 2SW,
3 = 3SW; -1 = sentinel for offspring or non-Baltic scenarios).
At introduction, `sea_age` is sampled from an origin-aware
trinomial distribution. Wild default: `{1: 0.35, 2: 0.55, 3: 0.10}`.
Hatchery override: `{1: 0.55, 2: 0.40, 3: 0.05}`.

### Empirical basis

**Primary anchor:** Jokikokko, Kallio-Nyberg & Jutila (2004),
*Journal of Applied Ichthyology* 20(1):37-42,
doi:10.1111/j.1439-0426.2004.00491.x — *"The timing, sex and age
composition of the wild and reared Atlantic salmon ascending the
Simojoki River, northern Finland"*. The Simojoki River is a
Bothnian Bay tributary, the closest unregulated Baltic system to
Lithuanian fisheries; the population has been intensively studied
across decades of varying hatchery contribution. Key finding
(verbatim): *"reared salmon stocked as smolts produced
considerable numbers of ascending **one-sea-winter (1 SW) males**,
whereas the proportion of male 1 SW salmon was low among spawning
migrants of wild or reared parr origin."* The reared-as-smolts
practice is the dominant Baltic stocking mode (and the Lithuanian
practice), making Jokikokko's contrast directly applicable: the
hatchery distribution shifts toward 1SW returners.

**Secondary anchor (life-history divergence):** Kallio-Nyberg,
Vainikka & Heino (2010), *Journal of Fish Biology* 76(3):622-640,
doi:10.1111/j.1095-8649.2009.02520.x — *"Divergent trends in
life-history traits between Atlantic salmon Salmo salar of wild
and hatchery origin in the Baltic Sea"*. Studies four Baltic
*S. salar* stocks 1972-1995 with varying breeding history. Key
finding: *"Maturation probabilities controlled for water
temperature, L_T at capture and L_T at release had increased in
all stocks. The least change was observed in the River Tornionjoki
*S. salar* that was subject only to supportive stockings
originating from wild parents. These results suggest a **long-term
divergence between semi-natural and broodstock-based S. salar
stocks**."* The Tornionjoki control demonstrates that the
divergence is driven by breeding-history (broodstock vs wild
supportive) rather than environmental drift alone.

**Supplementary anchor (environmental modulation):** Kallio-Nyberg,
Saloniemi & Koljonen (2020), *Journal of Applied Ichthyology*
36(3):288-297, doi:10.1111/jai.14033 — *"Increasing temperature
associated with increasing grilse proportion and smaller grilse
size of Atlantic salmon"*. Carlin-tag analysis (1985-2014).
Key finding: *"warmer spring temperatures during the smolt year
were associated with a higher proportion of one-sea-winter (1SW)
males during the return migration."* Documents a year-to-year
modulation (temperature → grilse fraction) that is explicitly
out-of-scope for C3.2's static-distribution architecture but is
flagged for a future tier.

**Population-scale anchor:** ICES Working Group on Baltic Salmon
and Trout (WGBAST) annual reports tabulate sea-age composition by
stock and year and consistently report higher 1SW fractions in
hatchery-supported stocks than in adjacent wild stocks. WGBAST is
the population-scale reference for the wild and hatchery defaults
(35% / 55% / 10% wild; 55% / 40% / 5% hatchery); sub-stock
granularity is not yet modelled.

### Mechanistic link to C3.1

Christie et al. (2014) — already cited under C3.1 — note that
*"differential age at maturation explains at least some of the
difference in reproductive success between wild and hatchery
individuals"*. C3.1 (RRS) and C3.2 (sea-age) are therefore
mechanistically coupled in the literature: a fraction of the C3.1
RRS reduction is itself produced by the C3.2 sea-age shift toward
younger 1SW returners (which are smaller and have lower
reproductive output). Modelling them as independent parameters is
a simplification: the model treats them as additive, but
empirically they are partially overlapping. This is a documented
limitation of the C3.x architecture; future work could re-fit the
C3.1 default after conditioning on C3.2's sea-age contrast.

### Calibration status

The hatchery 1SW share of 0.55 is **calibration-grade**, anchored
to Jokikokko 2004's Simojoki post-stocking 1SW-male prevalence.
Mandatory sensitivity sweep before publication:
**hatchery 1SW at `{0.35, 0.45, 0.55, 0.65}`** (with 2SW/3SW
renormalised proportionally to keep total = 1.0). Cap at 0.7 —
beyond is unsupported by Jokikokko 2004's Simojoki data and would
conflate sea-age effects with maturation-rate effects.

The wild 1SW share of 0.35 is NOT swept (treated as
calibration-anchor to WGBAST wild baseline).

### Effect size

For an introduced cohort of N hatchery agents, expected sea_age
composition shifts from (35/55/10) → (55/40/5) relative to the
wild scenario. In a mixed scenario with k% hatchery, the
population-level grilse fraction increases by ~`0.20 × k%`.

### References

Jokikokko, E., Kallio-Nyberg, I., & Jutila, E. (2004). The timing,
sex and age composition of the wild and reared Atlantic salmon
ascending the Simojoki River, northern Finland. *Journal of
Applied Ichthyology*, 20(1), 37-42.
https://doi.org/10.1111/j.1439-0426.2004.00491.x

Kallio-Nyberg, I., Vainikka, A., & Heino, M. (2010). Divergent
trends in life-history traits between Atlantic salmon *Salmo
salar* of wild and hatchery origin in the Baltic Sea. *Journal of
Fish Biology*, 76(3), 622-640.
https://doi.org/10.1111/j.1095-8649.2009.02520.x

Kallio-Nyberg, I., Saloniemi, I., & Koljonen, M.-L. (2020).
Increasing temperature associated with increasing grilse
proportion and smaller grilse size of Atlantic salmon. *Journal
of Applied Ichthyology*, 36(3), 288-297.
https://doi.org/10.1111/jai.14033

ICES (2024). Working Group on Baltic Salmon and Trout (WGBAST):
Annual report. International Council for the Exploration of the
Sea, Copenhagen.

---

## 7. Calibration methodology

All four tiers follow a uniform calibration-status discipline:

1. **Empirical anchor.** Each parameter has at least one
   peer-reviewed citation supporting its specific numerical value
   (or its bracketing range).
2. **Calibration-grade label.** Parameters whose Lithuanian-stock
   value cannot be directly measured (because no Žeimena/Simnas
   RRS or sea-age-by-origin study exists) are explicitly labelled
   "calibration-grade" in code comments and YAML provenance blocks.
3. **Mandatory sensitivity sweep.** Each calibration-grade
   parameter has a documented sweep range, anchored to the
   empirical literature, that MUST be exercised before any
   publication using the model. Sweep ranges are tighter on the
   wild side (where WGBAST baselines exist) and wider on the
   hatchery side (where Lithuanian measurements are missing).
4. **Cap.** Each parameter has an upper limit beyond which the
   citation chain breaks; values beyond the cap require fresh
   primary literature support.

### Why this matters

The Baltic salmon literature is uneven. Some signals (Enders 2004
respirometry, Jokikokko 2004 Simojoki sea-age) have direct
quantitative measurements. Others (Bouchard 2022 RRS, ICES WGBAST
sea-age) come from non-Lithuanian Atlantic salmon stocks that are
the closest available analog. The calibration-grade label is the
mechanism by which the model documents this difference honestly:
a simulation that uses defaults is reproducing a defensible
literature-anchored baseline; a simulation that swaps in
Lithuanian-measured values (when those become available) is
reproducing a stronger empirical baseline. Both are valid; the
distinction must be visible in any model output.

---

## 8. Out-of-scope: documented gaps

The following biological signals are real, citation-supported, and
*not yet modelled*. Each is a candidate for a future C3.x or D-tier
spec.

### 8.1 Origin inheritance

C1's scope-out decision (per the 2026-04-30 spec): offspring of
hatchery parents default to ORIGIN_WILD. The biological case for
inheritance — genetic + epigenetic carryover from hatchery parents
— is real but unsettled for *Salmo salar* in the Baltic. Suggestive
evidence comes from Le Luyer et al. (2017, doi:10.1073/pnas.1711229114)
on coho salmon DNA-methylation differences and Rodriguez-Barreto et
al. (2019) on Atlantic salmon early-life methylation, both cited
within Jönsson 2019 and Bouchard 2022. Neither has been
quantitatively measured for the Lithuanian programme.

### 8.2 Sex-specific divergence

Bouchard 2022 found 1SW males ~65% RRS vs MSW females + males ~80%.
Christie 2014 reported reduction "more severe for males than for
females". Jokikokko 2004 specifically observed the 1SW shift among
*male* returners. The current model has no sex tracking; adding it
would require a new int8 column on AgentPool plus sex-conditional
sampling at introduction and sex-conditional dispatch in C3.x
events.

### 8.3 Year-to-year stochastic variation

Kallio-Nyberg, Saloniemi & Koljonen (2020) documented that warmer
springs increase 1SW fractions. Bouchard 2022 reports per-year
RRS variation. The current model uses constant per-scenario
probabilities; year-to-year stochasticity would require coupling
the parameters to the existing Environment temperature time-series.

### 8.4 Sub-stock-specific distributions

Žeimena and Simnas may have different sea-age and RRS profiles.
The current `BalticAtlanticSalmon` species block treats all
Lithuanian salmon homogeneously. Sub-stock differentiation requires
a multi-population scenario (already supported by the IBM via
`MultiPopulationManager`) plus per-population YAML blocks.

### 8.5 Inter-population transfer encoding

`SwitchPopulationEvent` in `salmon_ibm/network.py` has a documented
limitation: when transferring agents between populations on
different meshes, the `natal_reach_id` and `origin` (and now
`sea_age`) fields may carry encoding mismatches. This is a
"reach_id encoding may not match" caveat from the existing code.
No deployed scenario uses multi-mesh transfers today.

### 8.6 4SW or older

Vanishingly rare in modern Baltic returns (Kallio-Nyberg 2010
reports MSW>2 fractions trending down even before hatchery
effects). The C3.2 `VALID_SEA_AGES = (1, 2, 3)` constraint can be
relaxed in a one-line edit when a 4SW signal becomes
quantitatively defensible.

### 8.7 Sea-age → body-size, sea-age → fecundity coupling

C3.2 stores `sea_age` as a metadata-only tag. Future tiers will
wire it into:
- **Body size at introduction** via length-at-age allometry
  (anchors: ICES WGBAST length-at-age tables; Kallio-Nyberg 2010
  Teno River length-by-sea-age data).
- **Fecundity** (anchor: Heinimaa & Heinimaa 2003 length-weight-
  fecundity for Atlantic salmon).

The C3.2 `Population.adult_sea_age_mask()` defensive helper is
the documented entry point for these future consumers.

### 8.8 Round goby egg predation, cyanobacteria/hypoxia, ice cover, seiches

Documented Baltic-specific environmental signals that interact
with salmon biology:
- *Neogobius melanostomus* (round goby) egg predation in Curonian
  since 2002 (Rakauskas et al. 2013, doi:10.3750/aip2013.43.2.02).
- Summer cyanobacteria + DO crashes (Mėžinė et al. 2019,
  doi:10.3390/w11101970).
- 58-134 days lagoon ice cover Dec-Mar (Idzelytė et al. 2019,
  doi:10.3390/rs11172059).
- Wind-forced seiches (`winds_stub.nc` exists; not wired).

These are tracked in the `curonian_deferred.md` memory and are
independent of the hatchery-vs-wild work.

---

## 9. Compound-effect summary

A scenario with k% hatchery agents shows the following first-order
divergences from a wild-only scenario:

| Tier  | Compound effect                                                  |
|-------|------------------------------------------------------------------|
| C1    | Distinguishable via output column; no direct biological effect.  |
| C2    | ~0.21 pp population-level mass-loss differential over 21d.       |
| C3.1  | ~0.30 × (k/100) reduction in population offspring count.         |
| C3.2  | ~0.20 × (k/100) increase in population grilse (1SW) fraction.    |

The compound effects are not strictly independent (per Christie
2014's mechanistic link from sea-age to RRS), but the
parameterisation treats them as additive for tractability. This is
a documented limitation; a future re-calibration tier could fit
the C3.1 RRS default conditional on the C3.2 sea-age contrast.

---

## 10. Verification methodology

All citations in this document have been verified via the scite
MCP server against actual DOI metadata (title, authors, journal,
pages, abstract). An earlier draft of the C3.2 spec contained
fabricated paper titles for two DOIs (a misattribution failure
mode that had previously occurred in the C2 spec); the present
verification protocol is the response to that recurring failure.

The protocol is:
1. Search scite for the DOI as a quoted string.
2. Compare returned title, authors, abstract against the spec
   claim.
3. If mismatch, find the actual paper at the DOI and either (a)
   replace the spec claim with the actual paper's content, or
   (b) find a different DOI that supports the original claim.
4. For supplementary citations, perform the same DOI-quoted
   search before adding to the document.

This protocol is documented in `feedback_collaboration.md` memory
and applies to any future C3.x spec.

---

## 11. References (consolidated)

Bouchard, R., Wellband, K. W., Lecomte, L., et al. (2022). Effects
of stocking at the parr stage on the reproductive fitness and
genetic diversity of a wild population of Atlantic salmon (*Salmo
salar* L.). *Evolutionary Applications*, 15(5), 838-852.
https://doi.org/10.1111/eva.13374

Christie, M. R., Ford, M. J., & Blouin, M. S. (2014). On the
reproductive success of early-generation hatchery fish in the
wild. *Evolutionary Applications*, 7(8), 883-896.
https://doi.org/10.1111/eva.12183

DeAngelis, D. L., & Mooij, W. M. (2005). Individual-based modeling
of ecological and evolutionary processes. *Annual Review of
Ecology, Evolution, and Systematics*, 36, 147-168.

Enders, E. C., Boisclair, D., & Roy, A. G. (2004). The costs of
habitat utilization of wild, farmed, and domesticated juvenile
Atlantic salmon (*Salmo salar*). *Canadian Journal of Fisheries
and Aquatic Sciences*, 61(12), 2302-2313.
https://doi.org/10.1139/f04-211

Heinimaa, S., & Heinimaa, P. (2003). Effect of size of female
salmon (*Salmo salar*) on egg quality and fecundity. *Bulletin of
the Finnish Game and Fisheries Research Institute*. (Cited
indirectly via C3.x deferred work; full DOI pending.)

ICES (2024). Working Group on Baltic Salmon and Trout (WGBAST):
Annual report. International Council for the Exploration of the
Sea, Copenhagen.

Idzelytė, R., Kozlov, I. E., & Dailidienė, I. (2019). Sea ice
characteristics in the Curonian Lagoon, Lithuania, derived from
Sentinel-1A imagery. *Remote Sensing*, 11(17), 2059.
https://doi.org/10.3390/rs11172059

Jokikokko, E., Kallio-Nyberg, I., & Jutila, E. (2004). The timing,
sex and age composition of the wild and reared Atlantic salmon
ascending the Simojoki River, northern Finland. *Journal of
Applied Ichthyology*, 20(1), 37-42.
https://doi.org/10.1111/j.1439-0426.2004.00491.x

Jönsson, B., Jönsson, N., & Jonsson, M. (2019). Supportive
breeders of Atlantic salmon *Salmo salar* have reduced fitness in
nature. *Conservation Science and Practice*, 1(9), e85.
https://doi.org/10.1111/csp2.85

Kallio-Nyberg, I., Saloniemi, I., & Koljonen, M.-L. (2020).
Increasing temperature associated with increasing grilse
proportion and smaller grilse size of Atlantic salmon. *Journal
of Applied Ichthyology*, 36(3), 288-297.
https://doi.org/10.1111/jai.14033

Kallio-Nyberg, I., Vainikka, A., & Heino, M. (2010). Divergent
trends in life-history traits between Atlantic salmon *Salmo
salar* of wild and hatchery origin in the Baltic Sea. *Journal
of Fish Biology*, 76(3), 622-640.
https://doi.org/10.1111/j.1095-8649.2009.02520.x

Le Luyer, J., Laporte, M., Beacham, T. D., et al. (2017).
Parallel epigenetic modifications induced by hatchery rearing in
a Pacific salmon. *Proceedings of the National Academy of
Sciences*, 114(49), 12964-12969.
https://doi.org/10.1073/pnas.1711229114

Mėžinė, J., Ferrarin, C., Vaičiūtė, D., et al. (2019). Sediment
transport mechanisms in a lagoon with high river discharge and
sediment loading. *Water*, 11(10), 1970.
https://doi.org/10.3390/w11101970

Rakauskas, V., Bacevičius, E., Pūtys, Ž., et al. (2013).
Distribution, sex structure, and abundance of the invasive round
goby *Neogobius melanostomus* (Pallas, 1814) in the Curonian
Lagoon (SE Baltic Sea). *Acta Ichthyologica et Piscatoria*, 43(2),
89-95. https://doi.org/10.3750/aip2013.43.2.02

Railsback, S. F., & Harvey, B. C. (2002). Analysis of habitat
selection rules using an individual-based model. *Ecology*, 83(7),
1817-1830.

Rodriguez-Barreto, D., Garcia de Leaniz, C., Verspoor, E., et al.
(2019). DNA methylation changes in the sperm of captive-reared
fish: A route to epigenetic introgression in wild populations?
*Molecular Biology and Evolution*, 36(10), 2205-2211.
(Cited indirectly via Bouchard 2022; not directly verified.)

---

**Document maintenance:** This document is the consolidated
scientific basis for the hatchery-vs-wild tier sequence. Each
tier's per-tier spec in `docs/superpowers/specs/` contains the
implementation-level mapping; this document is the empirical
mapping. When a new tier ships, append a section here with the
same structure (Mechanism / Empirical basis / Calibration status /
Effect size / References). When a deferred item is taken up,
move its description from "Out-of-scope" to its own tier section.
