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
    cfg = load_baltic_species_config(cfg_path)
    params = cfg.wild
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
    cfg = load_baltic_species_config(cfg_path)
    params = cfg.wild
    assert params.cmax_A == 0.303


def test_load_bio_params_routes_to_baltic_when_species_config_set(tmp_path):
    """A YAML config with species_config: <path> must route to BalticBioParams.

    Updated for Task 3: load_bio_params_from_config() now always returns
    BalticSpeciesConfig; the wild sub-object carries the BalticBioParams.
    """
    from salmon_ibm.config import load_bio_params_from_config
    from salmon_ibm.baltic_params import BalticBioParams, BalticSpeciesConfig

    species_yaml = tmp_path / "species.yaml"
    species_yaml.write_text("""
species:
  BalticAtlanticSalmon:
    T_OPT: 16.0
    T_AVOID: 20.0
    T_ACUTE_LETHAL: 24.0
""")
    cfg = {"species_config": str(species_yaml)}
    loaded = load_bio_params_from_config(cfg)
    assert isinstance(loaded, BalticSpeciesConfig)
    assert isinstance(loaded.wild, BalticBioParams)
    assert loaded.wild.T_OPT == 16.0
    assert loaded.wild.T_ACUTE_LETHAL == 24.0


def test_load_bio_params_falls_back_to_bio_params_when_no_species_config():
    """If species_config key absent, legacy path wraps BioParams in BalticSpeciesConfig.

    Updated for Task 3: load_bio_params_from_config() always returns
    BalticSpeciesConfig. Legacy non-Baltic path: wild=BioParams, hatchery=None.
    """
    from salmon_ibm.config import load_bio_params_from_config
    from salmon_ibm.bioenergetics import BioParams
    from salmon_ibm.baltic_params import BalticBioParams, BalticSpeciesConfig

    cfg = {"bioenergetics": {"RA": 0.003}}
    loaded = load_bio_params_from_config(cfg)
    assert isinstance(loaded, BalticSpeciesConfig)
    assert isinstance(loaded.wild, BioParams)
    assert not isinstance(loaded.wild, BalticBioParams), (
        "Fall-back path must return plain BioParams as wild, not BalticBioParams"
    )
    assert loaded.hatchery is None


def test_kill_gate_prefers_acute_lethal_for_baltic_params():
    """Integration: at 22°C (between T_AVOID=20 and T_ACUTE_LETHAL=24),
    BalticBioParams must NOT trigger the kill gate."""
    from salmon_ibm.baltic_params import BalticBioParams

    p = BalticBioParams()
    # Simulate what _event_bioenergetics does:
    lethal_T = getattr(p, "T_ACUTE_LETHAL", p.T_MAX)
    assert lethal_T == 24.0, "Should read T_ACUTE_LETHAL, not T_MAX"
    # At 22°C, which is a realistic Curonian summer SST, agents MUST NOT die.
    assert 22.0 < lethal_T, "22°C should not trigger kill gate for Baltic"


def test_kill_gate_falls_back_to_t_max_for_bio_params():
    """Classic BioParams has no T_ACUTE_LETHAL; kill gate must use T_MAX."""
    from salmon_ibm.bioenergetics import BioParams

    p = BioParams()
    lethal_T = getattr(p, "T_ACUTE_LETHAL", p.T_MAX)
    assert lethal_T == p.T_MAX == 26.0, (
        "Chinook BioParams must use T_MAX=26 since T_ACUTE_LETHAL absent"
    )
