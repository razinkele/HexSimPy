"""HexSim compatibility tests: run the Python engine on real HexSim workspaces."""
import os

import numpy as np
import pytest

# Skip entire module if workspace data is not available
WS_PATH = "Columbia River Migration Model/Columbia [small]"
HAS_WORKSPACE = os.path.exists(WS_PATH)

pytestmark = pytest.mark.skipif(not HAS_WORKSPACE, reason="Columbia workspace not found")


class TestColumbiaWorkspaceCompatibility:
    """Test the full Python HexSim stack with the Columbia River workspace."""

    def test_load_workspace_and_create_hexmesh(self):
        """HexMesh loads from workspace with correct properties."""
        from salmon_ibm.hexsim import HexMesh
        mesh = HexMesh.from_hexsim(WS_PATH, species="chinook")
        assert mesh.n_cells > 50_000
        assert mesh.centroids.shape[1] == 2
        assert (mesh.areas > 0).all()
        assert hasattr(mesh, '_water_nbrs')
        assert hasattr(mesh, '_water_nbr_count')

    def test_load_barriers_from_workspace(self):
        """BarrierMap loads from workspace .hbf files."""
        import glob
        from salmon_ibm.barriers import BarrierMap
        from salmon_ibm.hexsim import HexMesh
        mesh = HexMesh.from_hexsim(WS_PATH, species="chinook")
        # glob.escape handles special characters like '[' in "Columbia [small]"
        ws_escaped = glob.escape(WS_PATH)
        hbf_files = glob.glob(f"{ws_escaped}/Spatial Data/barriers/**/*.hbf", recursive=True)
        if not hbf_files:
            pytest.skip("No .hbf files found")
        bmap = BarrierMap.from_hbf(hbf_files[0], mesh)
        assert bmap.n_edges > 0

    def test_full_simulation_with_config(self):
        """Full 24h simulation using config_columbia.yaml."""
        from salmon_ibm.config import load_config
        from salmon_ibm.simulation import Simulation
        cfg = load_config("config_columbia.yaml")
        sim = Simulation(cfg, n_agents=50, rng_seed=42)
        sim.run(n_steps=24)
        assert sim.pool.alive.sum() > 0
        assert len(sim.history) == 24
        # Energy density should remain in a biologically plausible range (kJ/g)
        # Note: mean_ed can rise when low-energy agents die (survivor bias),
        # so we only assert bounds rather than strict monotone decrease.
        assert sim.history[-1]["mean_ed"] > 4.0   # above starvation threshold
        assert sim.history[-1]["mean_ed"] < 9.0   # below physiological maximum

    def test_population_wrapper_with_hexsim(self):
        """Population class works with HexSim-loaded mesh."""
        from salmon_ibm.config import load_config
        from salmon_ibm.simulation import Simulation
        cfg = load_config("config_columbia.yaml")
        sim = Simulation(cfg, n_agents=20, rng_seed=42)
        # Simulation now uses Population internally
        assert hasattr(sim, 'population')
        assert sim.population.name == "salmon"
        sim.run(n_steps=5)
        assert sim.population.n_alive > 0

    def test_event_sequencer_with_hexsim(self):
        """EventSequencer runs correctly on HexSim landscape."""
        from salmon_ibm.config import load_config
        from salmon_ibm.simulation import Simulation
        cfg = load_config("config_columbia.yaml")
        sim = Simulation(cfg, n_agents=30, rng_seed=42)
        assert hasattr(sim, '_sequencer')
        assert len(sim._sequencer.events) > 0
        sim.step()
        assert sim.current_t == 1

    def test_reproducibility_on_hexsim(self):
        """Same seed produces identical results on HexSim landscape."""
        import salmon_ibm.movement as mov
        orig = mov.FORCE_NUMPY
        try:
            mov.FORCE_NUMPY = True
            from salmon_ibm.config import load_config
            from salmon_ibm.simulation import Simulation

            cfg1 = load_config("config_columbia.yaml")
            sim1 = Simulation(cfg1, n_agents=20, rng_seed=42)
            sim1.run(n_steps=5)

            cfg2 = load_config("config_columbia.yaml")
            sim2 = Simulation(cfg2, n_agents=20, rng_seed=42)
            sim2.run(n_steps=5)

            np.testing.assert_array_equal(sim1.pool.tri_idx, sim2.pool.tri_idx)
            np.testing.assert_array_almost_equal(sim1.pool.ed_kJ_g, sim2.pool.ed_kJ_g)
        finally:
            mov.FORCE_NUMPY = orig

    def test_census_event_on_hexsim(self):
        """CensusEvent works with HexSim simulation."""
        from salmon_ibm.config import load_config
        from salmon_ibm.events_builtin import CensusEvent
        from salmon_ibm.simulation import Simulation
        cfg = load_config("config_columbia.yaml")
        sim = Simulation(cfg, n_agents=20, rng_seed=42)

        # Build landscape dict matching what Simulation.step() uses
        landscape = {
            "mesh": sim.mesh,
            "fields": sim.env.fields,
            "rng": sim._rng,
            "activity_lut": sim._activity_lut,
            "est_cfg": sim.est_cfg,
            "barrier_arrays": getattr(sim, '_barrier_arrays', None),
        }
        sim.env.advance(0)
        census = CensusEvent(name="census")
        mask = sim.population.alive.copy()
        census.execute(sim.population, landscape, t=0, mask=mask)
        assert "census_records" in landscape
        assert landscape["census_records"][0]["n_alive"] == 20

    def test_xml_parser_on_workspace(self):
        """XML parser can handle real workspace structure."""
        import glob

        from salmon_ibm.xml_parser import load_scenario_xml
        # glob.escape handles special characters like '[' in "Columbia [small]"
        ws_escaped = glob.escape(WS_PATH)
        xml_files = glob.glob(f"{ws_escaped}/Scenarios/**/*.xml", recursive=True)
        if not xml_files:
            pytest.skip("No XML scenario files found")
        config = load_scenario_xml(xml_files[0])
        assert "events" in config
        assert "populations" in config
