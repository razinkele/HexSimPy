import pytest
from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


def test_full_simulation_24h():
    """End-to-end: 24 hours with 20 agents on stub data."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=20, data_dir="data", rng_seed=42)
    sim.run(n_steps=24)

    # At least some agents should survive 24h
    assert sim.pool.alive.sum() > 0
    # Energy should have decreased
    assert sim.pool.ed_kJ_g[sim.pool.alive].mean() < 6.5
    # History should have 24 entries
    assert len(sim.history) == 24

    sim.close()


def test_full_simulation_with_output(tmp_path):
    """End-to-end with track output."""
    cfg = load_config("config_curonian_minimal.yaml")
    out = str(tmp_path / "tracks.csv")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42, output_path=out)
    sim.run(n_steps=5)
    sim.close()

    import pandas as pd
    df = pd.read_csv(out)
    # OutputLogger.log_step writes one row per agent (all agents, not just alive)
    # per timestep, so total rows = n_agents * n_steps
    assert len(df) == 10 * 5  # 10 agents x 5 timesteps
    assert "ed_kJ_g" in df.columns
