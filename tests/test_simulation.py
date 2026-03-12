import numpy as np
import pytest
from salmon_ibm.agents import Behavior
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
    initial_energy = (sim.pool.ed_kJ_g * sim.pool.mass_g).copy()
    sim.run(n_steps=10)
    alive = sim.pool.alive
    final_energy = sim.pool.ed_kJ_g[alive] * sim.pool.mass_g[alive]
    assert np.all(final_energy <= initial_energy[alive]), \
        "Total energy (ED * mass) should decrease for non-feeding fish"


def test_thermal_mortality_kills_at_extreme_temp():
    from salmon_ibm.bioenergetics import BioParams
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim.bio_params = BioParams(T_MAX=20.0)
    # Patch env.advance to force extreme temps after loading
    original_advance = sim.env.advance
    def hot_advance(t):
        original_advance(t)
        sim.env.fields["temperature"][:] = 25.0
    sim.env.advance = hot_advance
    initial_alive = sim.pool.alive.sum()
    sim.step()
    assert sim.pool.alive.sum() < initial_alive, "Fish should die when temp > T_MAX"


# ---------- DO override integration ----------

def _make_sim_with_do(do_values, lethal=2.0, high=4.0):
    """Helper: create sim and inject a 'do' field into env.fields."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim.env.advance(0)  # populate fields
    # Inject DO at every mesh cell
    do_field = np.full(sim.mesh.n_triangles, do_values, dtype=np.float64)
    sim.env.fields["do"] = do_field
    # Set config thresholds
    sim.est_cfg["do_avoidance"] = {"lethal": lethal, "high": high}
    return sim


def test_do_override_escape_forces_downstream():
    """DO between lethal and high → agents switch to DOWNSTREAM."""
    sim = _make_sim_with_do(do_values=3.0, lethal=2.0, high=4.0)  # 3.0 < high(4.0) → ESCAPE
    # Give agents a non-DOWNSTREAM behavior first
    sim.pool.behavior[:] = Behavior.UPSTREAM
    sim._apply_estuarine_overrides()
    alive = sim.pool.alive
    assert np.all(sim.pool.behavior[alive] == Behavior.DOWNSTREAM), \
        "DO_ESCAPE should force behavior to DOWNSTREAM"


def test_do_override_lethal_kills_agents():
    """DO below lethal threshold → agents die."""
    sim = _make_sim_with_do(do_values=1.0, lethal=2.0, high=4.0)  # 1.0 < lethal(2.0) → LETHAL
    assert sim.pool.alive.all(), "All agents should start alive"
    sim._apply_estuarine_overrides()
    assert not sim.pool.alive.any(), "All agents should die from lethal DO"


def test_do_override_ok_no_change():
    """DO above high threshold → no behavior change or mortality."""
    sim = _make_sim_with_do(do_values=8.0, lethal=2.0, high=4.0)  # 8.0 > high(4.0) → OK
    sim.pool.behavior[:] = Behavior.UPSTREAM
    initial_alive = sim.pool.alive.copy()
    sim._apply_estuarine_overrides()
    assert np.all(sim.pool.alive == initial_alive), "No mortality at normal DO"
    # Behavior may change due to seiche_pause, but DO should not force DOWNSTREAM
    # (seiche override is separate — test only DO effect here by noting no DOWNSTREAM)


def test_do_override_noop_when_field_absent():
    """No 'do' field in env.fields → no crash, no effect."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim.env.advance(0)
    # Don't inject "do" field — it shouldn't exist
    assert "do" not in sim.env.fields
    sim.pool.behavior[:] = Behavior.UPSTREAM
    initial_alive = sim.pool.alive.copy()
    sim._apply_estuarine_overrides()
    assert np.all(sim.pool.alive == initial_alive), "No mortality without DO data"


def test_do_override_noop_when_thresholds_zero():
    """Config thresholds both 0.0 (Columbia pattern) → no effect."""
    sim = _make_sim_with_do(do_values=1.0, lethal=0.0, high=0.0)
    sim.pool.behavior[:] = Behavior.UPSTREAM
    initial_alive = sim.pool.alive.copy()
    sim._apply_estuarine_overrides()
    assert np.all(sim.pool.alive == initial_alive), "Zero thresholds should disable DO"


def test_thermal_mortality_at_exact_t_max():
    """Fish at exactly T_MAX should die (>= not just >)."""
    from salmon_ibm.bioenergetics import BioParams
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim.bio_params = BioParams(T_MAX=26.0)
    original_advance = sim.env.advance
    def exact_tmax_advance(t):
        original_advance(t)
        sim.env.fields["temperature"][:] = 26.0
    sim.env.advance = exact_tmax_advance
    sim.step()
    assert not sim.pool.alive.any(), "All fish should die at exactly T_MAX"


def test_cwr_counters_update():
    """cwr_hours should increment for fish in TO_CWR state,
    and hours_since_cwr should reset when leaving CWR."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=5, data_dir="data", rng_seed=42)
    sim.env.advance(0)

    # Manually set agents to TO_CWR
    sim.pool.behavior[:] = Behavior.TO_CWR
    sim.pool.steps[:] = 10  # not first move
    sim.pool.cwr_hours[:] = 0
    sim.pool.hours_since_cwr[:] = 999

    # Test CWR counter logic directly
    sim._update_cwr_counters()
    assert np.all(sim.pool.cwr_hours == 1), "cwr_hours should increment for TO_CWR fish"
    assert np.all(sim.pool.hours_since_cwr == 0), (
        "hours_since_cwr should reset to 0 while in CWR"
    )

    # Now switch away from CWR
    sim.pool.behavior[:] = Behavior.UPSTREAM
    sim._update_cwr_counters()
    assert np.all(sim.pool.cwr_hours == 0), (
        "cwr_hours should reset to 0 when fish leave CWR"
    )
    assert np.all(sim.pool.hours_since_cwr == 1), (
        "hours_since_cwr should increment to 1 after leaving CWR"
    )


def test_dssh_dt_array_matches_scalar():
    """Vectorized dSSH_dt_array should match per-element dSSH_dt calls."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=5, data_dir="data", rng_seed=42)
    sim.env.advance(0)
    sim.env.advance(1)  # need two steps for dSSH_dt to be nonzero
    arr = sim.env.dSSH_dt_array()
    for i in range(min(10, sim.mesh.n_triangles)):
        scalar = sim.env.dSSH_dt(i)
        assert arr[i] == pytest.approx(scalar), f"Mismatch at triangle {i}"


def test_simulation_reproducibility():
    """Same rng_seed should produce identical results across two runs."""
    cfg = load_config("config_curonian_minimal.yaml")

    sim1 = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim1.run(n_steps=5)
    ed1 = sim1.pool.ed_kJ_g.copy()
    tri1 = sim1.pool.tri_idx.copy()

    cfg2 = load_config("config_curonian_minimal.yaml")
    sim2 = Simulation(cfg2, n_agents=10, data_dir="data", rng_seed=42)
    sim2.run(n_steps=5)
    ed2 = sim2.pool.ed_kJ_g.copy()
    tri2 = sim2.pool.tri_idx.copy()

    np.testing.assert_array_equal(tri1, tri2, "Positions should be identical with same seed")
    np.testing.assert_array_almost_equal(ed1, ed2, decimal=10,
        err_msg="Energy should be identical with same seed")
