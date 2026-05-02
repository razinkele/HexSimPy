"""Tests for the C2 hatchery vs wild parameter divergence (Tier C2).

Spec: docs/superpowers/specs/2026-05-01-hatchery-c2-bioparams-design.md
"""
from pathlib import Path

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


_REPO_ROOT = Path(__file__).resolve().parent.parent
_BALTIC_CONFIG = _REPO_ROOT / "configs" / "config_curonian_baltic.yaml"
_MINIMAL_CONFIG = _REPO_ROOT / "config_curonian_minimal.yaml"
_DATA_DIR = str(_REPO_ROOT / "data")


def _baltic_sim_with_hatchery(tmp_path):
    """Construct a real Simulation from config_curonian_baltic.yaml,
    but with the species_config redirected to a tmp_path species YAML
    that includes hatchery_overrides:.

    Existing Simulation tests use:
        cfg = load_config("config_curonian_minimal.yaml")
        sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)

    We follow the same pattern but with the Baltic config and a
    patched species_config path so we don't have to ship the
    hatchery_overrides into a tracked config file just for tests.
    """
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation

    if not _BALTIC_CONFIG.exists():
        pytest.skip("config_curonian_baltic.yaml not found")

    species_yaml = tmp_path / "species_with_hatchery.yaml"
    species_yaml.write_text("""
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
""")
    cfg = load_config(str(_BALTIC_CONFIG))
    cfg["species_config"] = str(species_yaml)
    sim = Simulation(cfg, n_agents=10, data_dir=_DATA_DIR, rng_seed=42)
    return sim


def test_rebuild_luts_resilient_to_slider_replacement(tmp_path):
    """app.py:1493 sidebar replaces sim.bio_params with a plain
    BioParams from slider values. rebuild_luts() must NOT anchor the
    hatchery LUT to the plain-BioParams Chinook defaults — instead
    re-derive from the cached BalticSpeciesConfig. Locks in the
    cache-and-re-derive semantics."""
    from salmon_ibm.bioenergetics import BioParams
    sim = _baltic_sim_with_hatchery(tmp_path)
    # Mimic app.py:1493 sidebar: replace bio_params with plain BioParams
    sim.bio_params = BioParams(RA=0.005, RB=-0.2, RQ=0.07, ED_MORTAL=4.0,
                               T_OPT=15.0, T_MAX=24.0)
    sim.rebuild_luts()
    # Wild LUT reflects Baltic species-config (NOT plain BioParams Chinook)
    assert sim._activity_lut[3] == pytest.approx(1.5)  # UPSTREAM Baltic
    # Hatchery LUT reflects merged YAML overrides
    assert sim.hatchery_dispatch is not None
    assert sim.hatchery_dispatch.activity_lut[1] == pytest.approx(1.5)  # RANDOM hatchery
    assert sim.hatchery_dispatch.activity_lut[3] == pytest.approx(1.875)  # UPSTREAM hatchery


def test_step_injects_hatchery_dispatch_landscape_key(tmp_path, monkeypatch):
    """Simulation.step() injects hatchery_dispatch into the landscape
    dict. Without this test, a missing dict-key insertion in step()
    would silently degrade dispatch to wild-only without breaking any
    other test."""
    from salmon_ibm.baltic_params import HatcheryDispatch
    sim = _baltic_sim_with_hatchery(tmp_path)
    captured: dict = {}

    def spy(population, landscape, t):
        captured["landscape"] = dict(landscape)  # snapshot, not reference

    monkeypatch.setattr(sim._sequencer, "step", spy)
    sim.step()

    assert "hatchery_dispatch" in captured["landscape"]
    assert captured["landscape"]["hatchery_dispatch"] is not None
    assert isinstance(captured["landscape"]["hatchery_dispatch"], HatcheryDispatch)


def test_rebuild_luts_noop_on_non_baltic_sim():
    """Non-Baltic sim (no species_config in config) calls rebuild_luts()
    without exception; sim.hatchery_dispatch remains None. Without this,
    a future change could throw KeyError on missing _species_config.
    Uses config_curonian_minimal.yaml which has no species_config: key."""
    from salmon_ibm.config import load_config
    from salmon_ibm.simulation import Simulation
    if not _MINIMAL_CONFIG.exists():
        pytest.skip("config_curonian_minimal.yaml not found")
    cfg = load_config(str(_MINIMAL_CONFIG))
    assert "species_config" not in cfg, "fixture invariant: minimal config is non-Baltic"
    sim = Simulation(cfg, n_agents=10, data_dir=_DATA_DIR, rng_seed=42)
    assert sim.hatchery_dispatch is None
    sim.rebuild_luts()  # must not raise
    assert sim.hatchery_dispatch is None


def test_introduction_event_runtime_guard_no_hatchery_params():
    """IntroductionEvent.execute() raises when origin=HATCHERY but
    landscape.get('hatchery_dispatch') is None. Catches the case where
    a runtime IntroductionEvent at step N tries to add hatchery agents
    but no overrides are configured. Init-time guard alone wouldn't
    catch this because Simulation starts with all-WILD agents."""
    import numpy as np
    from salmon_ibm.events_builtin import IntroductionEvent
    from salmon_ibm.origin import ORIGIN_HATCHERY
    # Minimal setup: empty landscape (so hatchery_dispatch is None)
    evt = IntroductionEvent(
        name="bad_intro",
        n_agents=2,
        positions=[0],
        origin=ORIGIN_HATCHERY,
    )
    with pytest.raises(ValueError, match=r"HATCHERY.*hatchery_dispatch"):
        evt.execute(population=None, landscape={}, t=0, mask=None)


def test_patch_introduction_event_runtime_guard_no_hatchery_params():
    """PatchIntroductionEvent.execute() mirror of the IntroductionEvent
    guard. CRITICAL: must supply a non-empty spatial_data layer in the
    landscape, otherwise PatchIntroductionEvent.execute returns early
    at events_hexsim.py:414 when the layer lookup fails — and the test
    would PASS even without the runtime guard, defeating its purpose.
    Without this test, the hexsim-mode introduction guard could be
    silently omitted."""
    import numpy as np
    from salmon_ibm.events_hexsim import PatchIntroductionEvent
    from salmon_ibm.origin import ORIGIN_HATCHERY
    evt = PatchIntroductionEvent(
        name="bad_patch",
        patch_spatial_data="dummy_layer",
        origin=ORIGIN_HATCHERY,
    )
    # Provide a real spatial_data layer so execute() reaches the guard
    # rather than early-returning on missing layer.
    landscape = {
        "spatial_data": {"dummy_layer": np.array([0.0, 1.0, 1.0, 0.0])},
    }
    with pytest.raises(ValueError, match=r"HATCHERY.*hatchery_dispatch"):
        evt.execute(population=None, landscape=landscape, t=0, mask=None)
