"""Tests for typed event descriptors and XML event parsing."""
from dataclasses import asdict

import pytest

from salmon_ibm.event_descriptors import (
    EventDescriptor,
    MoveEventDescriptor,
    SurvivalEventDescriptor,
    AccumulateEventDescriptor,
    TransitionEventDescriptor,
    CensusEventDescriptor,
    IntroductionEventDescriptor,
    DataProbeEventDescriptor,
    DataLookupEventDescriptor,
    SetSpatialAffinityEventDescriptor,
    PatchIntroductionEventDescriptor,
    InteractionEventDescriptor,
    ReanimationEventDescriptor,
)


class TestEventDescriptors:
    """Event descriptors are typed dataclasses."""

    def test_base_descriptor_fields(self):
        d = EventDescriptor(
            name="test", event_type="move", timestep=1,
            population_name="pop1",
        )
        assert d.name == "test"
        assert d.event_type == "move"
        assert d.enabled is True

    def test_move_descriptor_defaults(self):
        d = MoveEventDescriptor(
            name="m1", event_type="move", timestep=1,
            population_name="pop1",
        )
        assert d.move_type == ""
        assert d.max_steps == 0

    def test_survival_descriptor_accumulator_refs(self):
        d = SurvivalEventDescriptor(
            name="s1", event_type="hexsim_survival", timestep=1,
            population_name="pop1",
            survival_expression="exp(-0.1 * age)",
            accumulator_refs=["age", "energy"],
        )
        assert d.accumulator_refs == ["age", "energy"]


import xml.etree.ElementTree as ET

from salmon_ibm.xml_parser import _parse_event_to_descriptor


class TestEventParameterExtraction:
    """Per-type XML parameter extraction produces correct descriptors."""

    def test_parse_move_event(self):
        xml_str = """
        <event timestep="5">
            <moveEvent>
                <eventName>Fish Migration</eventName>
                <population>chinook</population>
                <enabled>true</enabled>
            </moveEvent>
        </event>
        """
        elem = ET.fromstring(xml_str)
        desc = _parse_event_to_descriptor(elem)
        assert isinstance(desc, MoveEventDescriptor)
        assert desc.name == "Fish Migration"
        assert desc.event_type == "move"
        assert desc.timestep == 5
        assert desc.population_name == "chinook"


from salmon_ibm.event_descriptors import DataProbeEventDescriptor
from salmon_ibm.events_hexsim import DataProbeEvent


class TestRegistryDrivenLoading:
    """Events can be constructed from descriptors."""

    def test_data_probe_from_descriptor(self):
        desc = DataProbeEventDescriptor(
            name="test_probe", event_type="data_probe",
            timestep=1, population_name="pop1",
        )
        evt = DataProbeEvent.from_descriptor(desc)
        assert evt.name == "test_probe"
        assert evt.population_name == "pop1"
