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
