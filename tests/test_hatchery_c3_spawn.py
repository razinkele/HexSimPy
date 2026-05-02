"""Tests for the C3.1 hatchery vs wild pre-spawn skip probability.

Spec: docs/superpowers/specs/2026-05-02-hatchery-c3-spawn-design.md
"""
import pytest


def test_pre_spawn_skip_prob_rejects_out_of_range():
    """BalticBioParams.__post_init__ rejects pre_spawn_skip_prob outside
    [0, 1]. Locks the validation contract; covers both negative and
    >1.0 boundary cases. C3.1 spec mandates 0.0 <= p <= 1.0."""
    from salmon_ibm.baltic_params import BalticBioParams
    with pytest.raises(ValueError, match=r"pre_spawn_skip_prob must be"):
        BalticBioParams(pre_spawn_skip_prob=-0.1)
    with pytest.raises(ValueError, match=r"pre_spawn_skip_prob must be"):
        BalticBioParams(pre_spawn_skip_prob=1.5)


def _yaml_with_hatchery_skip(tmp_path, p_skip: float = 0.3) -> str:
    """Helper: write a YAML body with a hatchery pre_spawn_skip_prob
    override and return the path. Includes the C2 activity_by_behavior
    block so the loader's existing path is exercised too."""
    p = tmp_path / "species.yaml"
    p.write_text(f"""
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior:
      0: 1.0
      1: 1.2
      2: 0.8
      3: 1.5
      4: 1.0
    pre_spawn_skip_prob: 0.0
    hatchery_overrides:
      activity_by_behavior:
        1: 1.5
        3: 1.875
      pre_spawn_skip_prob: {p_skip}
""")
    return str(p)


def test_pre_spawn_skip_prob_loads_from_yaml(tmp_path):
    """YAML hatchery_overrides.pre_spawn_skip_prob is applied on top of
    wild via dataclasses.replace; identity check ensures wild instance
    is not mutated. Verifies both the new scalar field flows through
    and the existing C2 activity_by_behavior dict-merge still works
    when both override types are present in the same YAML."""
    from salmon_ibm.baltic_params import (
        load_baltic_species_config, BalticBioParams, BalticSpeciesConfig,
    )
    path = _yaml_with_hatchery_skip(tmp_path, p_skip=0.3)
    cfg = load_baltic_species_config(path)
    assert isinstance(cfg, BalticSpeciesConfig)
    assert isinstance(cfg.wild, BalticBioParams)
    assert isinstance(cfg.hatchery, BalticBioParams)
    # Wild value preserved (0.0 from YAML)
    assert cfg.wild.pre_spawn_skip_prob == 0.0
    # Hatchery value overridden (0.3 from hatchery_overrides)
    assert cfg.hatchery.pre_spawn_skip_prob == 0.3
    # C2 activity_by_behavior shallow-merge still works
    assert cfg.wild.activity_by_behavior == {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    assert cfg.hatchery.activity_by_behavior == {0: 1.0, 1: 1.5, 2: 0.8, 3: 1.875, 4: 1.0}
    # Identity check: hatchery is a NEW instance, not wild mutated
    assert cfg.hatchery is not cfg.wild


def test_extended_overrides_still_reject_unknown_keys(tmp_path):
    """C3.1 extended ALLOWED_OVERRIDE_KEYS to include
    pre_spawn_skip_prob. Verify the strict-loader contract still
    holds: unknown keys (typos, unsupported fields) raise ValueError.
    Locks against a regression where extending the set accidentally
    relaxes the strict check."""
    from salmon_ibm.baltic_params import load_baltic_species_config
    p = tmp_path / "species.yaml"
    p.write_text("""
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    hatchery_overrides:
      unknown_field: 1.0
""")
    with pytest.raises(ValueError, match="hatchery_overrides supports only"):
        load_baltic_species_config(str(p))


def test_reproduction_skips_hatchery_at_p_skip_one():
    """Set p_skip=1.0 → ALL hatchery reproducers skip; wild always
    proceed. Deterministic test — no RNG dependence on outcome.
    Verifies the Bernoulli dispatch correctly reads parent origin."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.baltic_params import BalticBioParams, HatcheryDispatch
    from salmon_ibm.events_builtin import ReproductionEvent
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY

    pool = AgentPool(n=4, start_tri=0, rng_seed=42)
    pool.origin[:2] = ORIGIN_WILD       # 2 wild
    pool.origin[2:] = ORIGIN_HATCHERY   # 2 hatchery
    pop = Population(name="test", pool=pool)
    pop.group_id = np.array([0, 0, 0, 0], dtype=np.int32)  # all in same group

    hd = HatcheryDispatch(
        params=BalticBioParams(pre_spawn_skip_prob=1.0),
        activity_lut=np.ones(5),  # unused for reproduction test
    )
    landscape = {
        "rng": np.random.default_rng(0),
        "hatchery_dispatch": hd,
    }
    evt = ReproductionEvent(name="r", clutch_mean=10.0, min_group_size=1)
    n_before = pool.n
    evt.execute(pop, landscape, t=0, mask=np.ones(4, dtype=bool))
    n_offspring = pool.n - n_before
    # Only the 2 wild reproducers spawned (clutch_mean=10 each ≈ 20 expected).
    # No offspring from hatchery parents (skipped at p=1.0).
    # 2 reproducers × Poisson(10) → mean 20, 99% CI ~10-30
    # 4 reproducers × Poisson(10) → mean 40, 99% CI ~26-54
    # Window 25 is safely below the 4-reproducer lower bound.
    assert n_offspring < 30, (
        f"Expected wild-only Poisson(10) × 2 ≈ 20 offspring; got {n_offspring}. "
        f"Hatchery skip not applied?"
    )


def test_reproduction_no_skip_at_p_zero():
    """When pre_spawn_skip_prob=0.0, hatchery reproducers behave
    identically to wild. Lock-in for the explicit 0-value path
    (covers sensitivity-sweep null point and confirms the
    `pre_spawn_skip_prob > 0` guard short-circuits)."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.baltic_params import BalticBioParams, HatcheryDispatch
    from salmon_ibm.events_builtin import ReproductionEvent
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY

    pool = AgentPool(n=4, start_tri=0, rng_seed=42)
    pool.origin[:2] = ORIGIN_WILD
    pool.origin[2:] = ORIGIN_HATCHERY
    pop = Population(name="test", pool=pool)
    pop.group_id = np.array([0, 0, 0, 0], dtype=np.int32)

    # p=0.0 → no skipping; all 4 reproducers proceed
    hd = HatcheryDispatch(
        params=BalticBioParams(pre_spawn_skip_prob=0.0),
        activity_lut=np.ones(5),
    )
    landscape = {
        "rng": np.random.default_rng(0),
        "hatchery_dispatch": hd,
    }
    evt = ReproductionEvent(name="r", clutch_mean=10.0, min_group_size=1)
    n_before = pool.n
    evt.execute(pop, landscape, t=0, mask=np.ones(4, dtype=bool))
    n_offspring = pool.n - n_before
    # All 4 reproducers spawned → Poisson(10) × 4 ≈ 40, 99% CI ~26-54.
    # Verify count is in the 4-reproducer range, not the 2-reproducer range.
    assert n_offspring >= 25, (
        f"Expected all-reproducers Poisson(10) × 4 ≈ 40 offspring; got "
        f"{n_offspring}. Skip applied at p=0?"
    )


def test_reproduction_graceful_without_hatchery_dispatch():
    """When landscape has no 'hatchery_dispatch' key (pre-C3.1 scenarios
    or wild-only configs), ReproductionEvent.execute proceeds without
    ANY skip logic. Hatchery-tagged agents (if any) reproduce as
    normal. Locks the graceful-fallback semantics; no regression on
    pre-C2 reproductive scenarios."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.events_builtin import ReproductionEvent
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY

    pool = AgentPool(n=4, start_tri=0, rng_seed=42)
    pool.origin[:2] = ORIGIN_WILD
    pool.origin[2:] = ORIGIN_HATCHERY
    pop = Population(name="test", pool=pool)
    pop.group_id = np.array([0, 0, 0, 0], dtype=np.int32)

    # Landscape with NO hatchery_dispatch key — graceful fallback
    landscape = {"rng": np.random.default_rng(0)}
    evt = ReproductionEvent(name="r", clutch_mean=10.0, min_group_size=1)
    n_before = pool.n
    evt.execute(pop, landscape, t=0, mask=np.ones(4, dtype=bool))
    n_offspring = pool.n - n_before
    # No skip path executes; all 4 reproducers spawn → Poisson(10) × 4 ≈ 40.
    assert n_offspring >= 25, (
        f"Expected all-reproducers Poisson(10) × 4 ≈ 40 offspring; got "
        f"{n_offspring}. Skip applied without hatchery_dispatch key?"
    )
