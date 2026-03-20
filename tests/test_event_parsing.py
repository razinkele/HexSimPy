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
