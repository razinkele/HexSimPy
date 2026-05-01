"""Tests for the C2 hatchery vs wild parameter divergence (Tier C2).

Spec: docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md
"""
import pytest


def test_activity_by_behavior_rejects_nonpositive_value():
    """BalticBioParams.__post_init__ rejects activity_by_behavior with
    non-positive values. Tests both negative AND zero (boundary case)
    so a future 'optimisation' that changes guard from `v <= 0` to
    `v < 0` would regress visibly."""
    from salmon_ibm.baltic_params import BalticBioParams
    with pytest.raises(ValueError, match="positive floats"):
        BalticBioParams(activity_by_behavior={0: 1.0, 1: -0.5, 2: 0.8, 3: 1.5, 4: 1.0})
    with pytest.raises(ValueError, match="positive floats"):
        BalticBioParams(activity_by_behavior={0: 1.0, 1: 0.0, 2: 0.8, 3: 1.5, 4: 1.0})


def _write_yaml(tmp_path, body: str) -> str:
    """Helper: write a YAML body to tmp_path/species.yaml and return the path."""
    p = tmp_path / "species.yaml"
    p.write_text(body)
    return str(p)


def test_hatchery_overrides_activity_by_behavior_loads(tmp_path):
    """Happy path: hatchery_overrides.activity_by_behavior is shallow-merged
    over the wild base, producing a BalticSpeciesConfig with both wild
    and hatchery objects. Identity check ensures no in-place mutation
    of the wild dict."""
    from salmon_ibm.baltic_params import (
        load_baltic_species_config, BalticBioParams, BalticSpeciesConfig,
    )
    yaml_body = """
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior:
      0: 1.0
      1: 1.2
      2: 0.8
      3: 1.5
      4: 1.0
    hatchery_overrides:
      activity_by_behavior:
        1: 1.5
        3: 1.875
"""
    path = _write_yaml(tmp_path, yaml_body)
    cfg = load_baltic_species_config(path)
    assert isinstance(cfg, BalticSpeciesConfig)
    assert isinstance(cfg.wild, BalticBioParams)
    assert isinstance(cfg.hatchery, BalticBioParams)
    # Wild dict unchanged
    assert cfg.wild.activity_by_behavior == {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    # Hatchery dict shallow-merged
    assert cfg.hatchery.activity_by_behavior == {0: 1.0, 1: 1.5, 2: 0.8, 3: 1.875, 4: 1.0}
    # Identity check: hatchery is a NEW instance, not wild mutated
    assert cfg.hatchery is not cfg.wild


def test_hatchery_overrides_unsupported_key_raises(tmp_path):
    """Strict loader: hatchery_overrides containing fields other than
    activity_by_behavior raises ValueError. C2 only supports
    activity_by_behavior; other dataclass fields would be biologically
    inert under C2's dispatch."""
    from salmon_ibm.baltic_params import load_baltic_species_config
    yaml_body = """
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    hatchery_overrides:
      T_OPT: 14.0
"""
    path = _write_yaml(tmp_path, yaml_body)
    with pytest.raises(ValueError, match="hatchery_overrides supports only"):
        load_baltic_species_config(path)


def test_hatchery_overrides_typo_raises(tmp_path):
    """Strict loader: typo in top-level override key (e.g.
    'activity_for_behavior' instead of 'activity_by_behavior') raises."""
    from salmon_ibm.baltic_params import load_baltic_species_config
    yaml_body = """
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    hatchery_overrides:
      activity_for_behavior:
        1: 1.5
"""
    path = _write_yaml(tmp_path, yaml_body)
    with pytest.raises(ValueError, match="hatchery_overrides supports only"):
        load_baltic_species_config(path)


def test_hatchery_overrides_invalid_behavior_key_raises(tmp_path):
    """Strict loader: hatchery_overrides.activity_by_behavior keys must
    be valid Behavior enum values (0-4). A typo like '999' raises;
    without this check, the LUT would silently grow to 1000 elements
    and the override would have zero behavioural effect."""
    from salmon_ibm.baltic_params import load_baltic_species_config
    yaml_body = """
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    hatchery_overrides:
      activity_by_behavior:
        999: 5.0
"""
    path = _write_yaml(tmp_path, yaml_body)
    with pytest.raises(ValueError, match="valid Behavior enum values"):
        load_baltic_species_config(path)


def test_hatchery_overrides_nonnumeric_behavior_key_raises(tmp_path):
    """Strict loader: hatchery_overrides.activity_by_behavior keys must
    be coercible to int. A YAML key like 'hold' (likely a user typo
    intending HOLD=0) raises with an actionable error message."""
    from salmon_ibm.baltic_params import load_baltic_species_config
    yaml_body = """
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    activity_by_behavior: {0: 1.0, 1: 1.2, 2: 0.8, 3: 1.5, 4: 1.0}
    hatchery_overrides:
      activity_by_behavior:
        hold: 1.5
"""
    path = _write_yaml(tmp_path, yaml_body)
    with pytest.raises(ValueError, match="keys must be integers"):
        load_baltic_species_config(path)


def test_origin_aware_activity_mult_dispatch():
    """Helper dispatches per-agent: WILD origin reads from lut_wild,
    HATCHERY from lut_hatch. Graceful path (lut_hatch is None) returns
    lut_wild for all agents (covers pre-C2 callers and test fixtures).
    Element-by-element equality check, not just shape (a stub returning
    wrong values must not pass)."""
    import numpy as np
    from salmon_ibm.bioenergetics import origin_aware_activity_mult
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY

    behavior = np.array([0, 1, 3, 1, 3], dtype=int)
    origin = np.array(
        [ORIGIN_WILD, ORIGIN_HATCHERY, ORIGIN_HATCHERY, ORIGIN_WILD, ORIGIN_WILD],
        dtype=np.int8,
    )
    lut_wild = np.array([1.0, 1.2, 0.8, 1.5, 1.0])
    lut_hatch = np.array([1.0, 1.5, 0.8, 1.875, 1.0])

    # Mixed dispatch
    out = origin_aware_activity_mult(behavior, origin, lut_wild, lut_hatch)
    expected = np.array([1.0, 1.5, 1.875, 1.2, 1.5])
    np.testing.assert_array_equal(out, expected)

    # Graceful: lut_hatch=None returns lut_wild[behavior] for all agents
    out_graceful = origin_aware_activity_mult(behavior, origin, lut_wild, None)
    np.testing.assert_array_equal(out_graceful, lut_wild[behavior])


def test_simulation_init_non_baltic_has_no_hatchery_dispatch():
    """Non-Baltic config (no species_config: key) routes through the
    legacy path. Verified at the loader level (not full Simulation
    construction, which requires mesh/env/etc.) since the loader's
    unified-return contract is the unit under test here."""
    from salmon_ibm.config import load_bio_params_from_config
    from salmon_ibm.baltic_params import BalticSpeciesConfig
    from salmon_ibm.bioenergetics import BioParams

    cfg_dict = {}  # No species_config: key — legacy path
    loaded = load_bio_params_from_config(cfg_dict)
    assert isinstance(loaded, BalticSpeciesConfig)
    assert loaded.hatchery is None
    # Legacy path returns plain BioParams as wild, NOT BalticBioParams
    assert isinstance(loaded.wild, BioParams)
