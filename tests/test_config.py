import pytest
from salmon_ibm.config import (
    load_config,
    bio_params_from_config,
    behavior_params_from_config,
    validate_config,
)
from salmon_ibm.bioenergetics import BioParams
from salmon_ibm.behavior import BehaviorParams


def test_load_config_returns_dict():
    cfg = load_config("config_curonian_minimal.yaml")
    assert isinstance(cfg, dict)


def test_config_has_grid_section():
    cfg = load_config("config_curonian_minimal.yaml")
    assert "grid" in cfg
    assert cfg["grid"]["file"] == "curonian_minimal_grid.nc"


def test_config_has_estuary_section():
    cfg = load_config("config_curonian_minimal.yaml")
    assert "estuary" in cfg
    assert cfg["estuary"]["salinity_cost"]["S_opt"] == 0.5
    assert cfg["estuary"]["do_avoidance"]["lethal"] == 2.0
    assert cfg["estuary"]["seiche_pause"]["dSSHdt_thresh_m_per_15min"] == 0.02


def test_config_has_forcings():
    cfg = load_config("config_curonian_minimal.yaml")
    assert "forcings" in cfg
    assert "physics_surface" in cfg["forcings"]
    assert cfg["forcings"]["physics_surface"]["temp_var"] == "tos"


# ------------------------------------------------------------------
# T4.7 — bio_params_from_config / behavior_params_from_config
# ------------------------------------------------------------------

def test_bio_params_from_config_defaults():
    bp = bio_params_from_config({})
    default = BioParams()
    assert bp.RA == default.RA
    assert bp.RB == default.RB
    assert bp.RQ == default.RQ
    assert bp.ED_MORTAL == default.ED_MORTAL


def test_bio_params_from_config_override():
    bp = bio_params_from_config({"bioenergetics": {"RA": 0.005}})
    assert bp.RA == 0.005
    # Other fields keep defaults
    assert bp.RB == BioParams.RB


def test_behavior_params_from_config_defaults():
    bp = behavior_params_from_config({})
    default = BehaviorParams.defaults()
    assert bp.temp_bins == default.temp_bins
    assert bp.max_cwr_hours == default.max_cwr_hours
    assert bp.p_table is not None


# ------------------------------------------------------------------
# T4.8 — validate_config
# ------------------------------------------------------------------

def test_validate_config_missing_grid():
    with pytest.raises(ValueError, match="grid"):
        validate_config({"forcings": {}})


def test_validate_config_invalid_bio():
    cfg = {"grid": {"file": "test.nc"}, "bioenergetics": {"RA": 0}}
    with pytest.raises(ValueError, match="RA"):
        validate_config(cfg)


from salmon_ibm.config import population_config_from_yaml, barrier_config_from_yaml, genetics_config_from_yaml


def test_population_config_default():
    cfg = {"grid": {"type": "netcdf", "file": "test.nc"}}
    assert population_config_from_yaml(cfg) == {}


def test_barrier_config_absent():
    cfg = {"grid": {"type": "hexsim"}}
    assert barrier_config_from_yaml(cfg) is None


def test_barrier_config_present():
    cfg = {"grid": {"type": "hexsim"}, "barriers": {"file": "test.hbf"}}
    result = barrier_config_from_yaml(cfg)
    assert result["file"] == "test.hbf"


def test_genetics_config_absent():
    cfg = {"grid": {"type": "hexsim"}}
    assert genetics_config_from_yaml(cfg) is None


def test_genetics_config_present():
    cfg = {"grid": {"type": "hexsim"}, "genetics": {"loci": [{"name": "color", "n_alleles": 4}]}}
    result = genetics_config_from_yaml(cfg)
    assert len(result["loci"]) == 1
