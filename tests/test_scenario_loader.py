"""Tests for ScenarioLoader with real Columbia River workspace."""
import os
import numpy as np
import pytest
from salmon_ibm.scenario_loader import ScenarioLoader, HexSimSimulation

WS_PATH = "Columbia River Migration Model/Columbia [small]"
XML_PATH = f"{WS_PATH}/Scenarios/gr_Columbia2017B.xml"
HAS_WS = os.path.exists(WS_PATH) and os.path.exists(XML_PATH)
pytestmark = pytest.mark.skipif(not HAS_WS, reason="Columbia workspace not found")


class TestScenarioLoader:
    @pytest.fixture
    def sim(self):
        loader = ScenarioLoader()
        return loader.load(WS_PATH, XML_PATH, rng_seed=42)

    def test_returns_hexsim_simulation(self, sim):
        assert isinstance(sim, HexSimSimulation)

    def test_four_populations(self, sim):
        assert len(sim.populations.populations) == 4

    def test_population_names(self, sim):
        names = set(sim.populations.populations.keys())
        assert names == {"Chinook", "Iterator", "Refuges", "Steelhead"}

    def test_chinook_has_accumulators(self, sim):
        chinook = sim.populations.get("Chinook")
        assert chinook.accumulator_mgr is not None
        assert len(chinook.accumulator_mgr.definitions) >= 60

    def test_chinook_has_traits(self, sim):
        chinook = sim.populations.get("Chinook")
        assert chinook.trait_mgr is not None
        assert len(chinook.trait_mgr.definitions) >= 25

    def test_n_timesteps(self, sim):
        assert sim.n_timesteps == 2928

    def test_spatial_data_loaded(self, sim):
        assert len(sim.landscape.get("spatial_data", {})) >= 10

    def test_events_built(self, sim):
        assert len(sim.sequencer.events) == 9

    def test_runs_10_steps_without_crash(self, sim):
        sim.run(10)
        assert sim.current_t == 10
        assert len(sim.history) == 10

    def test_global_variables_in_landscape(self, sim):
        gv = sim.landscape.get("global_variables", {})
        assert gv.get("Hexagon Area") == 500.0
