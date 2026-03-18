"""Unit tests for XML scenario parser."""
import numpy as np
import pytest
import tempfile
from pathlib import Path

from salmon_ibm.xml_parser import load_scenario_xml, build_events_from_xml


SAMPLE_XML = """\
<?xml version="1.0"?>
<scenario>
  <simulation>
    <timeSteps>500</timeSteps>
    <replicates>3</replicates>
  </simulation>
  <population name="salmon">
    <initialCount>1000</initialCount>
    <trait name="stage">
      <type>probabilistic</type>
      <category>juvenile</category>
      <category>adult</category>
    </trait>
    <accumulator name="energy">
      <min>0.0</min>
      <max>100.0</max>
    </accumulator>
  </population>
  <event type="movement" name="migrate">
    <n_micro_steps>3</n_micro_steps>
  </event>
  <event type="survival" name="mortality"/>
  <event type="census" name="count">
    <trigger>
      <type>periodic</type>
      <interval>10</interval>
    </trigger>
  </event>
</scenario>
"""


class TestLoadScenarioXml:
    @pytest.fixture
    def xml_file(self, tmp_path):
        path = tmp_path / "test_scenario.xml"
        path.write_text(SAMPLE_XML)
        return path

    def test_parses_simulation_params(self, xml_file):
        config = load_scenario_xml(xml_file)
        assert config["simulation"]["n_timesteps"] == 500
        assert config["simulation"]["n_replicates"] == 3

    def test_parses_population(self, xml_file):
        config = load_scenario_xml(xml_file)
        assert len(config["populations"]) == 1
        pop = config["populations"][0]
        assert pop["name"] == "salmon"
        assert pop["initial_count"] == 1000

    def test_parses_traits(self, xml_file):
        config = load_scenario_xml(xml_file)
        pop = config["populations"][0]
        assert len(pop["traits"]) == 1
        trait = pop["traits"][0]
        assert trait["name"] == "stage"
        assert trait["categories"] == ["juvenile", "adult"]

    def test_parses_accumulators(self, xml_file):
        config = load_scenario_xml(xml_file)
        pop = config["populations"][0]
        assert len(pop["accumulators"]) == 1
        acc = pop["accumulators"][0]
        assert acc["name"] == "energy"
        assert acc["min"] == 0.0
        assert acc["max"] == 100.0

    def test_parses_events(self, xml_file):
        config = load_scenario_xml(xml_file)
        assert len(config["events"]) == 3
        assert config["events"][0]["type"] == "movement"
        assert config["events"][0]["name"] == "migrate"
        assert config["events"][1]["type"] == "survival"
        assert config["events"][2]["type"] == "census"

    def test_parses_event_params(self, xml_file):
        config = load_scenario_xml(xml_file)
        move = config["events"][0]
        assert move["params"]["n_micro_steps"] == 3.0  # parsed as float

    def test_parses_trigger(self, xml_file):
        config = load_scenario_xml(xml_file)
        census = config["events"][2]
        assert "trigger" in census
        assert census["trigger"]["type"] == "periodic"
        assert census["trigger"]["interval"] == 10


class TestBuildEventsFromXml:
    @pytest.fixture
    def xml_file(self, tmp_path):
        path = tmp_path / "test_scenario.xml"
        path.write_text(SAMPLE_XML)
        return path

    def test_builds_event_objects(self, xml_file):
        config = load_scenario_xml(xml_file)
        events = build_events_from_xml(config)
        assert len(events) == 3
        assert events[0].name == "migrate"
        assert events[2].name == "count"

    def test_trigger_on_census(self, xml_file):
        config = load_scenario_xml(xml_file)
        events = build_events_from_xml(config)
        census = events[2]
        assert census.trigger.should_fire(0) is True
        assert census.trigger.should_fire(5) is False
        assert census.trigger.should_fire(10) is True
