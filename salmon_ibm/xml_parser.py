"""XML scenario parser: load HexSim .xml scenario files into Python events."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from salmon_ibm.events import (
    Event, EventSequencer, EveryStep, Once, Periodic, Window,
    load_events_from_config, EVENT_REGISTRY,
)


def load_scenario_xml(path: str | Path) -> dict:
    """Parse a HexSim scenario XML file into a config dict.

    Returns a dict with keys:
      - 'populations': list of population definitions
      - 'events': list of event definitions (compatible with load_events_from_config)
      - 'simulation': dict with n_timesteps, n_replicates
      - 'spatial_data': dict of hex-map references
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    config = {
        "populations": [],
        "events": [],
        "simulation": {},
        "spatial_data": {},
    }

    # Parse simulation parameters
    sim_elem = root.find(".//simulation") or root.find(".//Simulation")
    if sim_elem is not None:
        config["simulation"]["n_timesteps"] = _int_text(sim_elem, "timeSteps", 100)
        config["simulation"]["n_replicates"] = _int_text(sim_elem, "replicates", 1)

    # Parse populations
    for pop_elem in root.iter("population"):
        pop_def = _parse_population(pop_elem)
        if pop_def:
            config["populations"].append(pop_def)

    # Also check for Population tag (case variation)
    for pop_elem in root.iter("Population"):
        pop_def = _parse_population(pop_elem)
        if pop_def:
            config["populations"].append(pop_def)

    # Parse event sequence
    for event_elem in _find_events(root):
        event_def = _parse_event(event_elem)
        if event_def:
            config["events"].append(event_def)

    # Parse spatial data references
    for data_elem in root.iter("hexMap"):
        name = data_elem.get("name", data_elem.text or "")
        if name:
            config["spatial_data"][name] = {
                "name": name,
                "file": data_elem.get("file", ""),
                "type": data_elem.get("type", "static"),
            }

    return config


def _int_text(parent, tag, default=0):
    elem = parent.find(tag)
    if elem is not None and elem.text:
        return int(elem.text)
    return default


def _float_text(parent, tag, default=0.0):
    elem = parent.find(tag)
    if elem is not None and elem.text:
        return float(elem.text)
    return default


def _text(parent, tag, default=""):
    elem = parent.find(tag)
    if elem is not None and elem.text:
        return elem.text.strip()
    return default


def _parse_population(elem) -> dict | None:
    """Parse a population XML element into a definition dict."""
    name = elem.get("name") or _text(elem, "name")
    if not name:
        return None

    pop_def = {
        "name": name,
        "initial_count": _int_text(elem, "initialCount", 100),
        "traits": [],
        "accumulators": [],
    }

    # Parse traits
    for trait_elem in elem.iter("trait"):
        trait_def = {
            "name": _text(trait_elem, "name") or trait_elem.get("name", ""),
            "type": _text(trait_elem, "type") or trait_elem.get("type", "probabilistic"),
            "categories": [],
        }
        for cat in trait_elem.iter("category"):
            cat_name = cat.text.strip() if cat.text else cat.get("name", "")
            if cat_name:
                trait_def["categories"].append(cat_name)
        if trait_def["name"]:
            pop_def["traits"].append(trait_def)

    # Parse accumulators
    for acc_elem in elem.iter("accumulator"):
        acc_def = {
            "name": _text(acc_elem, "name") or acc_elem.get("name", ""),
            "min": _float_text(acc_elem, "min"),
            "max": _float_text(acc_elem, "max"),
        }
        if acc_def["name"]:
            pop_def["accumulators"].append(acc_def)

    return pop_def


def _find_events(root) -> list:
    """Find event elements in the XML tree."""
    events = []
    # Try common HexSim XML structures
    for tag in ["event", "Event", "eventSequence", "EventSequence"]:
        for elem in root.iter(tag):
            if tag.lower() == "eventsequence":
                # Container: extract child events
                for child in elem:
                    events.append(child)
            else:
                events.append(elem)
    return events


def _parse_event(elem) -> dict | None:
    """Parse an event XML element into a definition dict."""
    event_type = elem.get("type") or _text(elem, "type") or elem.tag.lower()
    name = elem.get("name") or _text(elem, "name") or event_type

    # Map HexSim event type names to registered Python event types
    type_map = {
        "movement": "movement",
        "move": "movement",
        "survival": "survival",
        "survive": "survival",
        "reproduction": "reproduction",
        "reproduce": "reproduction",
        "census": "census",
        "accumulate": "accumulate",
        "mutation": "mutation",
        "mutate": "mutation",
        "transition": "transition",
        "introduction": "introduction",
        "introduce": "introduction",
        "floatercreation": "floater_creation",
        "floater_creation": "floater_creation",
        "interaction": "interaction",
    }

    mapped_type = type_map.get(event_type.lower(), event_type.lower())

    event_def = {
        "type": mapped_type,
        "name": name,
        "params": {},
    }

    # Extract parameters from child elements
    for child in elem:
        if child.tag not in ("name", "type", "trigger"):
            if child.text and child.text.strip():
                try:
                    event_def["params"][child.tag] = float(child.text)
                except ValueError:
                    event_def["params"][child.tag] = child.text.strip()

    # Parse trigger if present
    trigger_elem = elem.find("trigger")
    if trigger_elem is not None:
        trigger_type = trigger_elem.get("type") or _text(trigger_elem, "type", "every_step")
        trigger_def = {"type": trigger_type}
        for child in trigger_elem:
            if child.tag != "type" and child.text:
                try:
                    trigger_def[child.tag] = int(child.text)
                except ValueError:
                    try:
                        trigger_def[child.tag] = float(child.text)
                    except ValueError:
                        trigger_def[child.tag] = child.text.strip()
        event_def["trigger"] = trigger_def

    return event_def


def build_events_from_xml(xml_config: dict, callback_registry: dict | None = None) -> list[Event]:
    """Convert parsed XML config into Event objects."""
    # Ensure all event modules are imported so EVENT_REGISTRY is populated
    import salmon_ibm.events_builtin  # noqa: F401
    try:
        import salmon_ibm.events_phase3  # noqa: F401
    except ImportError:
        pass

    return load_events_from_config(xml_config["events"], callback_registry)
