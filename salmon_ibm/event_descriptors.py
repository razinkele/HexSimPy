"""Typed event descriptors for HexSim XML event parsing.

Each descriptor is a dataclass that carries validated, typed fields
extracted from an XML event element. These replace the raw dicts
previously passed between xml_parser.py and scenario_loader.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EventDescriptor:
    """Base typed representation of a parsed XML event."""
    name: str
    event_type: str
    timestep: int
    population_name: str
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class MoveEventDescriptor(EventDescriptor):
    move_type: str = ""
    max_steps: int = 0
    affinity_name: str = ""


@dataclass
class SurvivalEventDescriptor(EventDescriptor):
    survival_expression: str = ""
    accumulator_refs: list[str] = field(default_factory=list)


@dataclass
class AccumulateEventDescriptor(EventDescriptor):
    updater_functions: list[dict] = field(default_factory=list)


@dataclass
class TransitionEventDescriptor(EventDescriptor):
    trait_name: str = ""
    transition_matrix: list[list[float]] = field(default_factory=list)


@dataclass
class CensusEventDescriptor(EventDescriptor):
    pass


@dataclass
class IntroductionEventDescriptor(EventDescriptor):
    n_agents: int = 0
    initialization_spatial_data: str = ""


@dataclass
class PatchIntroductionEventDescriptor(EventDescriptor):
    n_agents: int = 0
    spatial_data: str = ""


@dataclass
class DataProbeEventDescriptor(EventDescriptor):
    pass


@dataclass
class DataLookupEventDescriptor(EventDescriptor):
    lookup_file: str = ""
    accumulator_name: str = ""


@dataclass
class SetSpatialAffinityEventDescriptor(EventDescriptor):
    affinity_name: str = ""
    spatial_data: str = ""


@dataclass
class InteractionEventDescriptor(EventDescriptor):
    interaction_type: str = ""


@dataclass
class ReanimationEventDescriptor(EventDescriptor):
    pass


DESCRIPTOR_REGISTRY: dict[str, type[EventDescriptor]] = {
    "move": MoveEventDescriptor,
    "hexsim_survival": SurvivalEventDescriptor,
    "accumulate": AccumulateEventDescriptor,
    "transition": TransitionEventDescriptor,
    "census": CensusEventDescriptor,
    "introduction": IntroductionEventDescriptor,
    "patch_introduction": PatchIntroductionEventDescriptor,
    "data_probe": DataProbeEventDescriptor,
    "data_lookup": DataLookupEventDescriptor,
    "set_spatial_affinity": SetSpatialAffinityEventDescriptor,
    "interaction": InteractionEventDescriptor,
    "reanimation": ReanimationEventDescriptor,
}
