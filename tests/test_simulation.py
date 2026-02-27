import numpy as np
import pytest
from salmon_ibm.simulation import Simulation
from salmon_ibm.config import load_config


def test_simulation_initializes():
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    assert sim.pool.n == 10
    assert sim.env.n_timesteps > 0


def test_simulation_step():
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim.step()
    assert sim.current_t == 1
    assert sim.pool.steps[0] > 0


def test_simulation_run_multiple_steps():
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim.run(n_steps=5)
    assert sim.current_t == 5


def test_simulation_energy_decreases():
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    initial_ed = sim.pool.ed_kJ_g.copy()
    sim.run(n_steps=10)
    assert np.all(sim.pool.ed_kJ_g[sim.pool.alive] <= initial_ed[sim.pool.alive])
