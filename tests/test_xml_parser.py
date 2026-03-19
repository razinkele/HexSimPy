"""Tests for HexSim XML scenario parser against real Columbia River XML."""
import os
import numpy as np
import pytest
from salmon_ibm.xml_parser import load_scenario_xml

WS_PATH = "Columbia River Migration Model/Columbia [small]"
XML_PATH = f"{WS_PATH}/Scenarios/gr_Columbia2017B.xml"
HAS_XML = os.path.exists(XML_PATH)
pytestmark = pytest.mark.skipif(not HAS_XML, reason="Columbia XML not found")


# ---------------------------------------------------------------------------
# Helper: recursively collect all events from a tree
# ---------------------------------------------------------------------------

def _collect_all_events(event_dict):
    """Recursively collect all events from a tree."""
    result = [event_dict]
    for sub in event_dict.get("sub_events", []):
        result.extend(_collect_all_events(sub))
    return result


# ---------------------------------------------------------------------------
# Task 1: Simulation params, grid metadata, workspace
# ---------------------------------------------------------------------------

class TestSimulationParams:
    @pytest.fixture
    def config(self):
        return load_scenario_xml(XML_PATH)

    def test_n_timesteps(self, config):
        assert config["simulation"]["n_timesteps"] == 2928

    def test_start_log_step(self, config):
        assert config["simulation"]["start_log_step"] == 2929


class TestGridMetadata:
    @pytest.fixture
    def config(self):
        return load_scenario_xml(XML_PATH)

    def test_hex_count(self, config):
        assert config["grid"]["n_hexagons"] == 16046143

    def test_columns(self, config):
        assert config["grid"]["columns"] == 10195

    def test_rows(self, config):
        assert config["grid"]["rows"] == 1574

    def test_narrow(self, config):
        assert config["grid"]["narrow"] is True

    def test_cell_width(self, config):
        assert abs(config["grid"]["cell_width"] - 24.028) < 0.01

    def test_workspace_path(self, config):
        assert "workspace" in config
        assert "Columbia" in config["workspace"]


# ---------------------------------------------------------------------------
# Task 2: Global variables and spatial data series
# ---------------------------------------------------------------------------

class TestGlobalVariables:
    @pytest.fixture
    def config(self):
        return load_scenario_xml(XML_PATH)

    def test_count(self, config):
        assert len(config["global_variables"]) == 58

    def test_hexagon_area(self, config):
        assert config["global_variables"]["Hexagon Area"] == 500.0

    def test_negative_value(self, config):
        assert config["global_variables"]["Fish Respiration RB"] == -0.217

    def test_float_value(self, config):
        assert abs(config["global_variables"]["Fish Respiration alpha"] - 0.00264) < 1e-6


class TestSpatialDataSeries:
    @pytest.fixture
    def config(self):
        return load_scenario_xml(XML_PATH)

    def test_count(self, config):
        assert len(config["spatial_data_series"]) == 18

    def test_hexmap_type(self, config):
        assert config["spatial_data_series"]["River [ extent ]"]["datatype"] == "HexMap"

    def test_barrier_type(self, config):
        assert config["spatial_data_series"]["Fish Ladder Available"]["datatype"] == "Barrier"

    def test_time_series_flag(self, config):
        # All series in Columbia have timeSeries=1
        for name, series in config["spatial_data_series"].items():
            assert series["time_series"] is True

    def test_cycle_length(self, config):
        # All series in Columbia have cycleLength=0
        for name, series in config["spatial_data_series"].items():
            assert series["cycle_length"] == 0


# ---------------------------------------------------------------------------
# Task 3: Populations
# ---------------------------------------------------------------------------

class TestPopulations:
    @pytest.fixture
    def config(self):
        return load_scenario_xml(XML_PATH)

    def test_count(self, config):
        assert len(config["populations"]) == 4

    def test_names_in_xml_order(self, config):
        names = [p["name"] for p in config["populations"]]
        assert names == ["Chinook", "Iterator", "Refuges", "Steelhead"]

    def test_chinook_initial_size(self, config):
        chinook = config["populations"][0]
        assert chinook["initial_size"] == 0

    def test_steelhead_initial_size(self, config):
        steelhead = config["populations"][3]
        assert steelhead["initial_size"] == 0

    def test_iterator_initial_size(self, config):
        iterator = config["populations"][1]
        assert iterator["initial_size"] == 1

    def test_chinook_accumulator_count(self, config):
        chinook = config["populations"][0]
        assert len(chinook["accumulators"]) >= 60

    def test_accumulator_attributes(self, config):
        chinook = config["populations"][0]
        # Find "Fitness [ weight ]" accumulator
        acc = next(a for a in chinook["accumulators"] if a["name"] == "Fitness [ weight ]")
        assert "min_val" in acc
        assert "max_val" in acc
        assert "birth_lower" in acc
        assert "birth_upper" in acc
        assert "inherit" in acc
        assert acc["min_val"] == 0.0
        assert acc["max_val"] == 0.0

    def test_chinook_trait_count(self, config):
        chinook = config["populations"][0]
        assert len(chinook["traits"]) >= 25

    def test_probabilistic_trait(self, config):
        chinook = config["populations"][0]
        trait = next(t for t in chinook["traits"] if t["name"] == "Fish Status [ movement ]")
        assert trait["type"] == "probabilistic"
        assert len(trait["categories"]) == 5
        assert trait["categories"][0]["name"] == "Do Not Move"
        assert trait["categories"][0]["init"] == 100
        assert trait["categories"][1]["init"] == 0

    def test_accumulated_trait(self, config):
        chinook = config["populations"][0]
        trait = next(t for t in chinook["traits"] if t["name"] == "Fish Status [ thermal ]")
        assert trait["type"] == "accumulated"
        assert trait["accumulator"] == "Temperature [ mean ]"
        assert len(trait["categories"]) >= 10
        assert trait["categories"][0]["threshold"] == float("-inf")
        assert trait["categories"][1]["threshold"] == 16.0

    def test_exclusion_layer(self, config):
        chinook = config["populations"][0]
        assert chinook["exclusion_layer"] == "River [ extent ]"

    def test_affinities(self, config):
        chinook = config["populations"][0]
        assert len(chinook["affinities"]) >= 1
        aff = chinook["affinities"][0]
        assert aff["name"] == "Movement Goal"


# ---------------------------------------------------------------------------
# Task 4: Events — recursive tree with type identification
# ---------------------------------------------------------------------------

class TestEvents:
    @pytest.fixture
    def config(self):
        return load_scenario_xml(XML_PATH)

    def test_root_event_count(self, config):
        assert len(config["events"]) == 9

    def test_first_three_are_one_shot(self, config):
        for i in range(3):
            assert config["events"][i].get("timestep") == 1

    def test_remaining_are_every_step(self, config):
        for i in range(3, 9):
            assert config["events"][i].get("timestep") is None

    def test_first_event_is_event_group(self, config):
        e = config["events"][0]
        assert e["type"] == "event_group"
        assert e["name"] == "Initialize Refuge Population"

    def test_event_group_has_sub_events(self, config):
        e = config["events"][0]
        assert "sub_events" in e
        assert len(e["sub_events"]) >= 2

    def test_sub_event_types(self, config):
        e = config["events"][0]
        types = [se["type"] for se in e["sub_events"]]
        assert "patch_introduction" in types
        assert "accumulate" in types

    def test_sub_event_population_name(self, config):
        e = config["events"][0]
        for se in e["sub_events"]:
            assert se["population"] == "Refuges"

    def test_disabled_events(self, config):
        # Root event 3 (Initialize Fish Populations) contains disabled reanimation events
        init_fish = config["events"][2]
        all_events = _collect_all_events(init_fish)
        disabled = [e for e in all_events if not e.get("enabled", True)]
        assert len(disabled) >= 1  # at least one disabled reanimationEvent

    def test_accumulate_event_has_updater_functions(self, config):
        init_refuge = config["events"][0]
        acc_event = next(
            se for se in init_refuge["sub_events"] if se["type"] == "accumulate"
        )
        assert "updater_functions" in acc_event
        assert len(acc_event["updater_functions"]) >= 1

    def test_updater_function_structure(self, config):
        init_refuge = config["events"][0]
        acc_event = next(
            se for se in init_refuge["sub_events"] if se["type"] == "accumulate"
        )
        uf = acc_event["updater_functions"][0]
        assert "accumulator" in uf
        assert "function" in uf
        assert uf["function"] == "IndividualLocations"

    def test_deeply_nested_event_groups(self, config):
        # Root event 3 has event groups 3 levels deep
        init_fish = config["events"][2]
        assert init_fish["type"] == "event_group"
        # First child should also be an event group
        child = init_fish["sub_events"][0]
        assert child["type"] == "event_group"


# ---------------------------------------------------------------------------
# Task 5: Updater functions, trait filters, move/transition params
# ---------------------------------------------------------------------------

class TestUpdaterFunctions:
    @pytest.fixture
    def config(self):
        return load_scenario_xml(XML_PATH)

    def test_expression_updater(self, config):
        """Find an accumulateEvent with an ExpressionUpdaterFunction."""
        all_events = []
        for root_evt in config["events"]:
            all_events.extend(_collect_all_events(root_evt))
        acc_events = [e for e in all_events if e["type"] == "accumulate"
                      and "updater_functions" in e]
        # Find one with an Expression function
        expr_ufs = []
        for ae in acc_events:
            for uf in ae.get("updater_functions", []):
                if uf["function"] == "Expression":
                    expr_ufs.append(uf)
        assert len(expr_ufs) > 200  # ~252 in Columbia
        # Check structure: ExpressionUpdaterFunction always has 2 parameters
        uf = expr_ufs[0]
        assert "accumulator" in uf
        assert "parameters" in uf
        assert len(uf["parameters"]) == 2  # expression string + unused "0"
        assert uf["parameters"][0] != "0"  # first is expression, not the padding
        assert uf["parameters"][1] == "0"  # second is always "0"

    def test_clear_updater(self, config):
        all_events = []
        for root_evt in config["events"]:
            all_events.extend(_collect_all_events(root_evt))
        acc_events = [e for e in all_events if e["type"] == "accumulate"]
        clear_ufs = []
        for ae in acc_events:
            for uf in ae.get("updater_functions", []):
                if uf["function"] == "Clear":
                    clear_ufs.append(uf)
        assert len(clear_ufs) >= 30

    def test_qualified_name_stripped(self, config):
        """Function names should have HexSimDomain. prefix and UpdaterFunction suffix stripped."""
        all_events = []
        for root_evt in config["events"]:
            all_events.extend(_collect_all_events(root_evt))
        for ae in all_events:
            for uf in ae.get("updater_functions", []):
                assert not uf["function"].startswith("HexSimDomain.")
                assert not uf["function"].endswith("UpdaterFunction")

    def test_source_trait_parsed(self, config):
        """TraitId updater should have source_trait field."""
        all_events = []
        for root_evt in config["events"]:
            all_events.extend(_collect_all_events(root_evt))
        trait_id_ufs = []
        for ae in all_events:
            for uf in ae.get("updater_functions", []):
                if uf["function"] == "TraitId":
                    trait_id_ufs.append(uf)
        assert len(trait_id_ufs) >= 1
        assert trait_id_ufs[0].get("source_trait") is not None


class TestMoveEvents:
    @pytest.fixture
    def config(self):
        return load_scenario_xml(XML_PATH)

    def test_move_events_exist(self, config):
        all_events = []
        for root_evt in config["events"]:
            all_events.extend(_collect_all_events(root_evt))
        moves = [e for e in all_events if e["type"] == "move"]
        assert len(moves) >= 20

    def test_move_strategy(self, config):
        all_events = []
        for root_evt in config["events"]:
            all_events.extend(_collect_all_events(root_evt))
        moves = [e for e in all_events if e["type"] == "move"]
        strategies = {m["params"]["move_strategy"] for m in moves}
        assert "onlyDisperse" in strategies

    def test_dispersal_use_affinity(self, config):
        all_events = []
        for root_evt in config["events"]:
            all_events.extend(_collect_all_events(root_evt))
        moves = [e for e in all_events if e["type"] == "move"]
        affinity_moves = [m for m in moves if m["params"].get("dispersal_use_affinity")]
        assert len(affinity_moves) >= 5
        # Verify named form returns string, not just True
        named = [m for m in affinity_moves
                 if isinstance(m["params"]["dispersal_use_affinity"], str)]
        assert len(named) >= 1
        assert any(m["params"]["dispersal_use_affinity"] == "Movement Goal"
                   for m in named)


class TestTransitionEvents:
    @pytest.fixture
    def config(self):
        return load_scenario_xml(XML_PATH)

    def test_transition_events_exist(self, config):
        all_events = []
        for root_evt in config["events"]:
            all_events.extend(_collect_all_events(root_evt))
        transitions = [e for e in all_events if e["type"] == "transition"]
        assert len(transitions) >= 30

    def test_transition_has_matrix(self, config):
        all_events = []
        for root_evt in config["events"]:
            all_events.extend(_collect_all_events(root_evt))
        transitions = [e for e in all_events if e["type"] == "transition"]
        t = transitions[0]
        assert "transition_trait" in t["params"]
        assert "matrix_data" in t["params"]
        assert "rows" in t["params"]
        assert "columns" in t["params"]
