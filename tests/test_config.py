import pytest
from salmon_ibm.config import load_config


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
