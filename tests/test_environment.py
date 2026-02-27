import numpy as np
import pytest
from salmon_ibm.mesh import TriMesh
from salmon_ibm.environment import Environment
from salmon_ibm.config import load_config


@pytest.fixture
def env():
    cfg = load_config("config_curonian_minimal.yaml")
    mesh = TriMesh.from_netcdf("data/curonian_minimal_grid.nc")
    return Environment(cfg, mesh, data_dir="data")


def test_env_loads(env):
    assert env.n_timesteps > 0


def test_env_advance_loads_fields(env):
    env.advance(0)
    assert "temperature" in env.fields
    assert "salinity" in env.fields
    assert "u_current" in env.fields
    assert "ssh" in env.fields


def test_env_sample_returns_dict(env):
    env.advance(0)
    water_ids = np.where(env.mesh.water_mask)[0]
    tri = water_ids[0]
    s = env.sample(tri)
    assert "temperature" in s
    assert isinstance(s["temperature"], float)


def test_env_gradient_returns_tuple(env):
    env.advance(0)
    water_ids = np.where(env.mesh.water_mask)[0]
    tri = water_ids[0]
    grad = env.gradient("temperature", tri)
    assert len(grad) == 2


def test_env_dSSH_dt(env):
    env.advance(0)
    env.advance(1)
    water_ids = np.where(env.mesh.water_mask)[0]
    tri = water_ids[0]
    rate = env.dSSH_dt(tri)
    assert isinstance(rate, float)
