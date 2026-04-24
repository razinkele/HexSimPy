"""Tests for Baltic Atlantic salmon parameter loader."""
import pytest


def test_baltic_bioparams_defaults_from_literature():
    """Default values must match the peer-reviewed values from the Baltic species config."""
    from salmon_ibm.baltic_params import BalticBioParams

    p = BalticBioParams()
    # Smith et al. 2009 CMax coefficients (marine-phase post-smolt S. salar)
    assert p.cmax_A == pytest.approx(0.303)
    assert p.cmax_B == pytest.approx(-0.275)
    # Jensen et al. 2001 thermal optimum
    assert p.T_OPT == pytest.approx(16.0)
    # Two-threshold thermal response (v2): avoidance vs acute mortality
    assert p.T_AVOID == pytest.approx(20.0)
    assert p.T_ACUTE_LETHAL == pytest.approx(24.0)
    # Length-weight (provenance under verification)
    assert p.LW_a == pytest.approx(0.0077)
    assert p.LW_b == pytest.approx(3.05)
    # Linear fecundity approximation
    assert p.fecundity_per_g == pytest.approx(2.0)
    # Backward-compat: T_MAX property still works (maps to T_AVOID)
    assert p.T_MAX == pytest.approx(20.0)


def test_baltic_bioparams_rejects_invalid_ranges():
    """Post-init validation must reject nonsense parameters."""
    from salmon_ibm.baltic_params import BalticBioParams

    with pytest.raises(ValueError, match="T_AVOID"):
        BalticBioParams(T_OPT=25.0, T_AVOID=20.0)
    with pytest.raises(ValueError, match="T_ACUTE_LETHAL"):
        BalticBioParams(T_AVOID=25.0, T_ACUTE_LETHAL=20.0)
    with pytest.raises(ValueError, match="cmax_A"):
        BalticBioParams(cmax_A=-1.0)


def test_baltic_species_config_loader_parses_yaml(tmp_path):
    """Loader must parse the canonical baltic_salmon_species.yaml and return BalticBioParams."""
    from salmon_ibm.baltic_params import load_baltic_species_config, BalticBioParams

    cfg_path = tmp_path / "species.yaml"
    cfg_path.write_text("""
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    cmax_B: -0.275
    T_OPT: 16.0
    T_AVOID: 20.0
    T_ACUTE_LETHAL: 24.0
    LW_a: 0.0077
    LW_b: 3.05
    fecundity_per_g: 2.0
    spawn_window_start_day: 288
    spawn_window_end_day: 334
""")
    params = load_baltic_species_config(cfg_path)
    assert isinstance(params, BalticBioParams)
    assert params.T_OPT == 16.0
    assert params.T_AVOID == 20.0
    assert params.T_ACUTE_LETHAL == 24.0
    assert params.spawn_window_start_day == 288  # Oct 15
    assert params.spawn_window_end_day == 334    # Nov 30


def test_baltic_species_config_rejects_missing_block(tmp_path):
    """Loader must raise if species.BalticAtlanticSalmon block is absent."""
    from salmon_ibm.baltic_params import load_baltic_species_config

    cfg_path = tmp_path / "species.yaml"
    cfg_path.write_text("species: {}\n")
    with pytest.raises(ValueError, match="BalticAtlanticSalmon"):
        load_baltic_species_config(cfg_path)


def test_baltic_species_config_filters_unknown_keys(tmp_path):
    """Unknown YAML keys must be silently filtered, not raise TypeError."""
    from salmon_ibm.baltic_params import load_baltic_species_config

    cfg_path = tmp_path / "species.yaml"
    cfg_path.write_text("""
species:
  BalticAtlanticSalmon:
    cmax_A: 0.303
    unknown_key_from_inSTREAM: 42
    another_unknown: "foo"
""")
    params = load_baltic_species_config(cfg_path)
    assert params.cmax_A == 0.303
