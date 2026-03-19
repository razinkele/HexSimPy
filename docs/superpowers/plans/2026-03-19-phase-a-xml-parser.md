# Phase A: XML Parser Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `salmon_ibm/xml_parser.py` so `load_scenario_xml()` correctly parses real HexSim scenario XML files, specifically `gr_Columbia2017B.xml` (9644 lines, 4 populations, 9 root events, 58 global variables, 18 spatial data series).

**Architecture:** Complete rewrite of `load_scenario_xml()` with new helper functions. The old synthetic-XML tests are replaced with tests against the real Columbia XML. The output dict structure changes to match the parity spec. `build_events_from_xml()` is preserved but updated to consume the new dict format.

**Tech Stack:** Python `xml.etree.ElementTree`, numpy, pytest.

**Spec reference:** `docs/superpowers/specs/2026-03-19-hexsim-parity-design.md` (Phase A section + items 42-55 from reviews)

---

## File Structure

- **Rewrite:** `salmon_ibm/xml_parser.py` — complete replacement of all functions
- **Rewrite:** `tests/test_xml_parser.py` — new tests against real Columbia XML
- **No changes** to any other `salmon_ibm/` files (parser is decoupled from event engine)

---

## Task 1: Parse Simulation Metadata (simulationParameters, hexagonGrid, workspace)

**Files:**
- Modify: `salmon_ibm/xml_parser.py`
- Modify: `tests/test_xml_parser.py`

These are the simplest sections at the END of the XML file (lines 9440-9457).

- [ ] **Step 1: Write failing tests**

Replace the entire `tests/test_xml_parser.py` with a new file. Keep imports and add:

```python
"""Tests for HexSim XML scenario parser against real Columbia River XML."""
import os
import numpy as np
import pytest
from salmon_ibm.xml_parser import load_scenario_xml

WS_PATH = "Columbia River Migration Model/Columbia [small]"
XML_PATH = f"{WS_PATH}/Scenarios/gr_Columbia2017B.xml"
HAS_XML = os.path.exists(XML_PATH)
pytestmark = pytest.mark.skipif(not HAS_XML, reason="Columbia XML not found")


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest tests/test_xml_parser.py -v --tb=short`
Expected: FAIL — current parser returns wrong values

- [ ] **Step 3: Implement simulation/grid/workspace parsing**

Replace the top of `salmon_ibm/xml_parser.py` with new implementation. Keep the helper functions `_int_text`, `_float_text`, `_text` and rewrite `load_scenario_xml`:

```python
"""HexSim XML scenario parser: load real HexSim .xml files into config dicts."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


def load_scenario_xml(path: str | Path) -> dict:
    """Parse a HexSim scenario XML file into a structured config dict."""
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    config: dict[str, Any] = {
        "simulation": _parse_simulation_params(root),
        "grid": _parse_grid_metadata(root),
        "workspace": _parse_workspace(root),
        "global_variables": _parse_global_variables(root),
        "spatial_data_series": _parse_spatial_data_series(root),
        "populations": _parse_populations(root),
        "events": _parse_root_events(root),
    }
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _bool_text(parent, tag, default=False):
    elem = parent.find(tag)
    if elem is not None and elem.text:
        t = elem.text.strip().lower()
        return t in ("true", "1", "yes")
    return default


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

def _parse_simulation_params(root) -> dict:
    elem = root.find("simulationParameters")
    if elem is None:
        return {"n_timesteps": 100, "n_replicates": 1, "start_log_step": 0}
    return {
        "n_timesteps": _int_text(elem, "timesteps", 100),
        "n_replicates": 1,  # not present in Columbia XML
        "start_log_step": _int_text(elem, "startLogStep", 0),
    }


def _parse_grid_metadata(root) -> dict:
    elem = root.find("hexagonGrid")
    if elem is None:
        return {}
    return {
        "n_hexagons": _int_text(elem, "hexCount", 0),
        "rows": _int_text(elem, "rows", 0),
        "columns": _int_text(elem, "columns", 0),
        "narrow": _bool_text(elem, "narrow", False),
        "cell_width": _float_text(elem, "hexagonWidth", 0.0),
    }


def _parse_workspace(root) -> str:
    elem = root.find("workspace")
    if elem is not None and elem.text:
        return elem.text.strip()
    return ""
```

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_xml_parser.py::TestSimulationParams tests/test_xml_parser.py::TestGridMetadata -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/xml_parser.py tests/test_xml_parser.py
git commit -m "feat(xml): rewrite parser - simulation params, grid metadata, workspace"
```

---

## Task 2: Parse Global Variables and Spatial Data Series

**Files:**
- Modify: `salmon_ibm/xml_parser.py`
- Modify: `tests/test_xml_parser.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_xml_parser.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement global variables and spatial data series parsing**

Add to `salmon_ibm/xml_parser.py`:

```python
def _parse_global_variables(root) -> dict[str, float]:
    """Parse <globalVariables> block. Note: attributes are capital-case Name/Value."""
    result = {}
    gv_elem = root.find("globalVariables")
    if gv_elem is None:
        return result
    for var_elem in gv_elem.iter("globalVariable"):
        name = var_elem.get("Name", "")  # capital N
        value_str = var_elem.get("Value", "0")  # capital V
        if name:
            result[name] = float(value_str)
    return result


def _parse_spatial_data_series(root) -> dict[str, dict]:
    """Parse all <spatialDataSeries> elements."""
    result = {}
    for elem in root.iter("spatialDataSeries"):
        name = _text(elem, "name")
        if not name:
            continue
        result[name] = {
            "datatype": _text(elem, "datatype", "HexMap"),
            "time_series": _int_text(elem, "timeSeries", 0) != 0,
            "cycle_length": _int_text(elem, "cycleLength", 0),
        }
    return result
```

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_xml_parser.py::TestGlobalVariables tests/test_xml_parser.py::TestSpatialDataSeries -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/xml_parser.py tests/test_xml_parser.py
git commit -m "feat(xml): parse global variables (58 entries) and spatial data series (18 entries)"
```

---

## Task 3: Parse Populations (accumulators, traits, affinities, range params)

**Files:**
- Modify: `salmon_ibm/xml_parser.py`
- Modify: `tests/test_xml_parser.py`

This is the most complex population parsing task. Key challenges:
- Accumulators use XML attributes (`lowerBound=`, `upperBound=`, etc.), not child elements
- Traits use `<probabilisticTrait>` and `<accumulatedTrait>` tags (not `<trait>`)
- `<accumulatedTrait>` has `accumulator=` attribute linking to accumulator name
- `-INF` threshold must be parsed as `float("-inf")`
- `<value>` elements carry `name`, `init`/`birth` (probabilistic) or `threshold` (accumulated)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_xml_parser.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement population parsing**

Add to `salmon_ibm/xml_parser.py`:

```python
def _parse_populations(root) -> list[dict]:
    """Parse all <population> elements in document order."""
    result = []
    for pop_elem in root.findall("population"):
        pop_def = _parse_population(pop_elem)
        if pop_def:
            result.append(pop_def)
    return result


def _parse_population(elem) -> dict | None:
    name = _text(elem, "name")
    if not name:
        return None

    pop = {
        "name": name,
        "type": _text(elem, "type", "terrestrial"),
        "initial_size": _int_text(elem, "initialSize", 0),
        "initialization_spatial_data": _text(elem, "initializationSpatialData"),
        "exclusion_layer": _text(elem, "exclusionLayer"),
        "exclude_if_zero": _bool_text(elem, "excludeIfZero", True),
        "accumulators": _parse_accumulators(elem),
        "traits": _parse_traits(elem),
        "affinities": _parse_affinities(elem),
        "range_parameters": _parse_range_parameters(elem),
    }
    return pop


def _parse_accumulators(pop_elem) -> list[dict]:
    """Parse <accumulator> elements. Bounds are XML ATTRIBUTES, not children."""
    result = []
    accs_container = pop_elem.find("accumulators")
    if accs_container is None:
        return result
    for acc_elem in accs_container.findall("accumulator"):
        name = acc_elem.get("name", "")
        if not name:
            continue
        result.append({
            "name": name,
            "min_val": float(acc_elem.get("lowerBound", "0")),
            "max_val": float(acc_elem.get("upperBound", "0")),
            "birth_lower": float(acc_elem.get("birthLower", "0")),
            "birth_upper": float(acc_elem.get("birthUpper", "0")),
            "holds_id": acc_elem.get("holdsId", "False").lower() == "true",
            "inherit": acc_elem.get("inherit", "False").lower() == "true",
        })
    return result


def _parse_traits(pop_elem) -> list[dict]:
    """Parse <probabilisticTrait> and <accumulatedTrait> elements."""
    result = []
    # Probabilistic traits
    for trait_elem in pop_elem.iter("probabilisticTrait"):
        name = trait_elem.get("name", "")
        if not name:
            continue
        categories = []
        for val_elem in trait_elem.iter("value"):
            categories.append({
                "name": val_elem.get("name", ""),
                "init": int(val_elem.get("init", "0")),
                "birth": int(val_elem.get("birth", "0")),
            })
        result.append({
            "name": name,
            "type": "probabilistic",
            "categories": categories,
        })

    # Accumulated traits
    for trait_elem in pop_elem.iter("accumulatedTrait"):
        name = trait_elem.get("name", "")
        if not name:
            continue
        accumulator = trait_elem.get("accumulator", "")
        categories = []
        for val_elem in trait_elem.iter("value"):
            thresh_str = val_elem.get("threshold", "0")
            if thresh_str.upper() in ("-INF", "-INFINITY"):
                thresh = float("-inf")
            else:
                thresh = float(thresh_str)
            categories.append({
                "name": val_elem.get("name", ""),
                "threshold": thresh,
            })
        result.append({
            "name": name,
            "type": "accumulated",
            "accumulator": accumulator,
            "categories": categories,
        })
    return result


def _parse_affinities(pop_elem) -> list[dict]:
    affinities_elem = pop_elem.find("affinities")
    if affinities_elem is None:
        return []
    result = []
    for aff_elem in affinities_elem.iter("affinity"):
        result.append({
            "name": _text(aff_elem, "name"),
            "sub_type": _text(aff_elem, "subType", "spatial"),
            "maximum_size": _int_text(aff_elem, "maximumSize", 1),
            "strategy": _text(aff_elem, "strategy", "random"),
            "threshold": _float_text(aff_elem, "threshold", 0.0),
            "group": _bool_text(aff_elem, "group", False),
        })
    return result


def _parse_range_parameters(pop_elem) -> dict:
    rp_elem = pop_elem.find("rangeParameters")
    if rp_elem is None:
        return {}
    return {
        "resources_target": _float_text(rp_elem, "resourcesTarget", 0),
        "range_threshold": _float_text(rp_elem, "rangeThreshold", 0),
        "max_indiv_in_group": _int_text(rp_elem, "maxIndivInGroup", 1),
        "max_range_distance": _float_text(rp_elem, "maxRangeDistance", 0),
        "max_range_hectares": _float_text(rp_elem, "maxRangeHectares", 0),
        "range_spatial_data": _text(rp_elem, "rangeSpatialData"),
        "min_range_resource": _float_text(rp_elem, "minRangeResource", 0),
        "is_competitive": _bool_text(rp_elem, "isCompetitive", False),
    }
```

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_xml_parser.py::TestPopulations -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/xml_parser.py tests/test_xml_parser.py
git commit -m "feat(xml): parse populations with correct trait/accumulator attributes"
```

---

## Task 4: Parse Events — Recursive Tree with Type Identification

**Files:**
- Modify: `salmon_ibm/xml_parser.py`
- Modify: `tests/test_xml_parser.py`

Key parsing rules from spec items 42-43:
- Root `<event>` elements are direct children of `<scenario>`
- `timestep="N"` attribute present → fires once at step N; absent → every step
- `eventOff="True"` → disabled
- Event TYPE is determined by the CHILD TAG NAME (e.g., `<accumulateEvent>`, `<moveEvent>`)
- Strip the `Event` suffix to get the type key (e.g., `accumulateEvent` → `accumulate`)
- `<eventGroupEvent>` recurses — its children are `<event>` wrappers containing typed events
- `<populationName>` inside the typed event gives the target population
- `<permanent>` is parsed but NOT used for scheduling (spec item 42)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_xml_parser.py`:

```python
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


def _collect_all_events(event_dict):
    """Recursively collect all events from a tree."""
    result = [event_dict]
    for sub in event_dict.get("sub_events", []):
        result.extend(_collect_all_events(sub))
    return result
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement recursive event parsing**

Add to `salmon_ibm/xml_parser.py`:

```python
# ---------------------------------------------------------------------------
# Event type mapping: strip "Event" suffix from child tag name
# ---------------------------------------------------------------------------

_EVENT_TAG_MAP = {
    "eventGroupEvent": "event_group",
    "accumulateEvent": "accumulate",
    "transitionEvent": "transition",
    "moveEvent": "move",
    "survivalEvent": "hexsim_survival",
    "introductionEvent": "introduction",
    "patchIntroductionEvent": "patch_introduction",
    "interactionEvent": "interaction",
    "dataLookupEvent": "data_lookup",
    "setSpatialAffinityEvent": "set_spatial_affinity",
    "censusEvent": "census",
    "dataProbeEvent": "data_probe",
    "reanimationEvent": "reanimation",
}


def _parse_root_events(root) -> list[dict]:
    """Parse direct <event> children of <scenario> in document order."""
    result = []
    for event_wrapper in root.findall("event"):
        event_def = _parse_event_wrapper(event_wrapper)
        if event_def is not None:
            result.append(event_def)
    return result


def _parse_event_wrapper(wrapper_elem) -> dict | None:
    """Parse an <event> wrapper element.

    The wrapper has optional attributes (timestep=, eventOff=).
    Its single child element's tag name determines the event type.
    """
    timestep = wrapper_elem.get("timestep")
    event_off = wrapper_elem.get("eventOff", "").lower() == "true"

    # Find the single typed child element
    typed_child = None
    for child in wrapper_elem:
        if child.tag in _EVENT_TAG_MAP:
            typed_child = child
            break

    if typed_child is None:
        return None

    event_def = _parse_typed_event(typed_child)
    if event_def is None:
        return None

    # Apply wrapper attributes
    if timestep is not None:
        event_def["timestep"] = int(timestep)
    if event_off:
        event_def["enabled"] = False

    return event_def


def _parse_typed_event(elem) -> dict | None:
    """Parse a typed event element (accumulateEvent, moveEvent, etc.)."""
    event_type = _EVENT_TAG_MAP.get(elem.tag, elem.tag)
    name = _text(elem, "name", elem.tag)
    population = _text(elem, "populationName")
    if not population:
        population = None

    event_def: dict[str, Any] = {
        "type": event_type,
        "name": name,
        "population": population,
        "enabled": True,
    }

    # Type-specific parsing
    if event_type == "event_group":
        event_def["iterations"] = _int_text(elem, "iterations", 1)
        event_def["sub_events"] = []
        # Group-level trait filter (sibling of <event> children)
        group_filter = _parse_trait_filter(elem)
        if group_filter:
            event_def["group_trait_filter"] = group_filter
        # Recurse into child <event> wrappers
        for child_wrapper in elem.findall("event"):
            child_def = _parse_event_wrapper(child_wrapper)
            if child_def is not None:
                event_def["sub_events"].append(child_def)

    elif event_type == "accumulate":
        event_def["updater_functions"] = _parse_updater_functions(elem)

    elif event_type == "move":
        event_def["params"] = _parse_move_params(elem)

    elif event_type == "transition":
        event_def["params"] = _parse_transition_params(elem)

    elif event_type == "hexsim_survival":
        event_def["params"] = {
            "use_accumulator": _bool_text(elem, "useAccumulator", False),
            "survival_accumulator": _text(elem, "survivalAccumulator"),
        }

    elif event_type == "introduction":
        event_def["params"] = {
            "initial_size": _int_text(elem, "initialSize", 0),
            "initialization_spatial_data": _text(elem, "initializationSpatialData"),
        }

    elif event_type == "patch_introduction":
        event_def["params"] = {
            "patch_spatial_data": _text(elem, "patchSpatialData"),
            "form_groups": _bool_text(elem, "formGroups", False),
        }

    elif event_type == "data_lookup":
        event_def["params"] = {
            "file_name": _text(elem, "fileName"),
            "row_accumulator": _text(elem, "rowAccumulator"),
            "column_accumulator": _text(elem, "columnAccumulator"),
            "target_accumulator": _text(elem, "targetAccumulator"),
            "has_column_header": _bool_text(elem, "hasColumnHeader", False),
            "has_row_header": _bool_text(elem, "hasRowHeader", False),
        }

    elif event_type == "set_spatial_affinity":
        event_def["params"] = {
            "affinity": _text(elem, "affinity"),
            "strategy": _text(elem, "strategy", "better"),
            "spatial_series": _text(elem, "spatialSeries"),
            "error_accumulator": _text(elem, "errorAccumulator"),
            "use_bounds": _bool_text(elem, "useBounds", False),
            "min_accumulator": _text(elem, "minAccumulator"),
            "max_accumulator": _text(elem, "maxAccumulator"),
            "min_value": _float_text(elem, "min", 0),
            "max_value": _float_text(elem, "max", 0),
        }

    elif event_type == "interaction":
        event_def["params"] = _parse_interaction_params(elem)

    elif event_type == "census":
        census_traits = []
        for t in elem.findall("trait"):
            if t.text:
                census_traits.append(t.text.strip())
        event_def["params"] = {"trait_names": census_traits}

    # data_probe and reanimation: store minimal info, no behavioral params needed

    # Parse trait filter for non-group leaf events only
    # (event_group handles its own filter as group_trait_filter above)
    if event_type != "event_group":
        trait_filter = _parse_trait_filter(elem)
        if trait_filter:
            event_def["trait_filter"] = trait_filter

    return event_def
```

- [ ] **Step 4: Run tests**

Run: `conda run -n shiny python -m pytest tests/test_xml_parser.py::TestEvents -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/xml_parser.py tests/test_xml_parser.py
git commit -m "feat(xml): recursive event tree parsing with type identification from child tag"
```

---

## Task 5: Parse Updater Functions, Trait Filters, Move/Transition/Interaction Params

**Files:**
- Modify: `salmon_ibm/xml_parser.py`
- Modify: `tests/test_xml_parser.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_xml_parser.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement remaining parser functions**

Add to `salmon_ibm/xml_parser.py`:

```python
def _parse_updater_functions(acc_elem) -> list[dict]:
    """Parse <updaterFunction> elements inside an accumulateEvent.

    Handles both global (direct children) and stratified
    (<stratifiedUpdaterFunctions> groups).
    """
    result = []

    # Global updater functions (direct children)
    for uf_elem in acc_elem.findall("updaterFunction"):
        uf = _parse_single_updater(uf_elem)
        if uf:
            result.append(uf)

    # Stratified updater function groups
    for strat_elem in acc_elem.iter("stratifiedUpdaterFunctions"):
        group_traits = [t.text.strip() for t in strat_elem.findall("trait") if t.text]
        combos_text = _text(strat_elem, "traitCombinations")
        for uf_elem in strat_elem.findall("updaterFunction"):
            uf = _parse_single_updater(uf_elem)
            if uf:
                uf["stratified_traits"] = group_traits
                uf["trait_combinations"] = combos_text
                result.append(uf)

    return result


def _parse_single_updater(uf_elem) -> dict | None:
    """Parse one <updaterFunction> element."""
    func_raw = _text(uf_elem, "function")
    if not func_raw:
        return None

    # Strip HexSimDomain. prefix and UpdaterFunction suffix
    func_name = func_raw
    if func_name.startswith("HexSimDomain."):
        func_name = func_name[len("HexSimDomain."):]
    if func_name.endswith("UpdaterFunction"):
        func_name = func_name[:-len("UpdaterFunction")]

    accumulator = _text(uf_elem, "accumulator")
    spatial_data = _text(uf_elem, "accumulateSpatialData")
    source_trait = _text(uf_elem, "sourceTrait")

    parameters = []
    for p_elem in uf_elem.findall("parameter"):
        if p_elem.text is not None:
            parameters.append(p_elem.text.strip())

    uf = {
        "function": func_name,
        "accumulator": accumulator,
        "parameters": parameters,
    }
    if spatial_data:
        uf["spatial_data"] = spatial_data
    if source_trait:
        uf["source_trait"] = source_trait

    return uf


def _parse_trait_filter(elem) -> dict | None:
    """Extract trait filter from event element.

    <trait> + <traitCombinations> can be direct children.
    """
    traits = [t.text.strip() for t in elem.findall("trait") if t.text]
    combos = _text(elem, "traitCombinations")
    if traits:
        return {"traits": traits, "combinations": combos}
    return None


def _parse_move_params(elem) -> dict:
    """Parse <moveEvent> parameters."""
    # dispersalUseAffinity: empty element = True (default slot), named = specific slot
    dua_elem = elem.find("dispersalUseAffinity")
    dispersal_use_affinity = None
    if dua_elem is not None:
        if dua_elem.text and dua_elem.text.strip():
            dispersal_use_affinity = dua_elem.text.strip()  # named affinity
        else:
            dispersal_use_affinity = True  # empty element = use default

    return {
        "move_strategy": _text(elem, "moveStrategy", "onlyDisperse"),
        "dispersal_spatial_data": _text(elem, "dispersalSpatialData"),
        "walk_up_gradient": _bool_text(elem, "walkUpGradient", False),
        "barrier_series": _text(elem, "barrierSeries"),
        "dispersal_accumulator": _text(elem, "dispersalAccumulator"),
        "distance_accumulator": _text(elem, "distanceAccumulator"),
        "dispersal_halt_minimum": _float_text(elem, "dispersalHaltMinimum", 0),
        "dispersal_halt_target": _float_text(elem, "dispersalHaltTarget", 0),
        "dispersal_halt_memory": _float_text(elem, "dispersalHaltMemory", 0),
        "dispersal_auto_correlation": _float_text(elem, "dispersalAutoCorrelation", 0),
        "dispersal_use_affinity": dispersal_use_affinity,
        "attraction_coefficients": _text(elem, "attractionCoefficients"),
        "attraction_multiplier": _float_text(elem, "attractionMultiplier", 0),
        "trend_period": _int_text(elem, "trendPeriod", 0),
        "resource_threshold": _float_text(elem, "resourceThreshold", 0),
        "avoid_explored_area": _bool_text(elem, "avoidExploredArea", False),
    }


def _parse_transition_params(elem) -> dict:
    """Parse <transitionEvent> parameters including matrix data."""
    transition_trait = _text(elem, "transitionTrait")
    conditioning_traits = [t.text.strip() for t in elem.findall("trait") if t.text]

    # Parse matrixSet
    ms_elem = elem.find("matrixSet")
    rows = 0
    columns = 0
    matrix_data = None
    if ms_elem is not None:
        rows = _int_text(ms_elem, "rows", 0)
        columns = _int_text(ms_elem, "columns", 0)
        matrices_elem = ms_elem.find("matrices")
        if matrices_elem is not None and matrices_elem.text:
            # Raw whitespace-separated floats
            matrix_data = [float(x) for x in matrices_elem.text.split()]

    rows_reversed = False
    if ms_elem is not None:
        rows_reversed = ms_elem.get("rowsReversed", "false").lower() == "true"

    return {
        "transition_trait": transition_trait,
        "conditioning_traits": conditioning_traits,
        "rows": rows,
        "columns": columns,
        "rows_reversed": rows_reversed,
        "matrix_data": matrix_data,
    }


def _parse_interaction_params(elem) -> dict:
    """Parse <interactionEvent> parameters."""
    presence = _text(elem, "presence", "explored")
    colocation_elem = elem.find("colocation")
    colocation_type = colocation_elem.get("type", "always") if colocation_elem is not None else "always"

    interactions = []
    for int_elem in elem.findall("interaction"):
        int_def = {
            "name": _text(int_elem, "name"),
            "population": _text(int_elem, "populationName"),
            "presence": _text(int_elem, "presence", "explored"),
            "encounter_probability": _float_text(int_elem, "encounterProbability", 1.0),
        }
        # Parse outcomes
        outcomes_elem = int_elem.find("outcomes")
        if outcomes_elem is not None:
            int_def["outcomes"] = _parse_outcomes(outcomes_elem)
        interactions.append(int_def)

    return {
        "presence": presence,
        "colocation_type": colocation_type,
        "interactions": interactions,
    }


def _parse_outcomes(outcomes_elem) -> list[dict]:
    """Parse <outcomes> block with p1/p2 trait filters and changes."""
    p1_traits = [t.text.strip() for t in outcomes_elem.findall("p1Traits/trait") if t.text]
    p2_traits = [t.text.strip() for t in outcomes_elem.findall("p2Traits/trait") if t.text]

    results = []
    for out_elem in outcomes_elem.findall("outcome"):
        outcome = {
            "p1_combo_index": _int_text(out_elem, "p1ComboIndex", 0),
            "p2_combo_index": _int_text(out_elem, "p2ComboIndex", 0),
            "probability": _float_text(out_elem, "outcomeProbability", 1.0),
            "p1_changes": _parse_changes(out_elem.find("p1Changes")),
            "p2_changes": _parse_changes(out_elem.find("p2Changes")),
        }
        results.append(outcome)

    return {"p1_traits": p1_traits, "p2_traits": p2_traits, "outcomes": results}


def _parse_changes(changes_elem) -> list[dict]:
    """Parse <p1Changes> or <p2Changes> block."""
    if changes_elem is None:
        return []
    result = []
    for ch_elem in changes_elem.findall("change"):
        result.append({
            "accumulator": ch_elem.get("accumulator", ""),
            "updater": ch_elem.get("updater", "assign"),
            "value": ch_elem.get("value", "0"),
        })
    return result


# ---------------------------------------------------------------------------
# Backward compat: build_events_from_xml (Phase C/D will use this)
# ---------------------------------------------------------------------------

def build_events_from_xml(xml_config: dict, callback_registry: dict | None = None):
    """Convert parsed XML config into Event objects.

    NOTE: This function is a stub for Phase C/D. The new dict format
    from load_scenario_xml() is not yet compatible with EVENT_REGISTRY.
    Phase C (MultiPopEventSequencer) and Phase D (new event types) will
    wire this up. For now, return the raw event dicts.
    """
    return xml_config.get("events", [])
```

- [ ] **Step 4: Run all tests**

Run: `conda run -n shiny python -m pytest tests/test_xml_parser.py -v --tb=short`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add salmon_ibm/xml_parser.py tests/test_xml_parser.py
git commit -m "feat(xml): parse updater functions, trait filters, move/transition/interaction params"
```

---

## Task 6: Integration Verification

- [ ] **Step 1: Run the full test suite**

Run: `conda run -n shiny python -m pytest tests/ -v --ignore=tests/test_playwright.py --tb=short`
Expected: 391+ tests PASS. The old `TestBuildEventsFromXml` tests will break because `build_events_from_xml` now returns raw dicts instead of Event objects — this is expected and correct (Phase C/D will wire it up).

- [ ] **Step 2: If old build_events tests fail, remove them**

The old `TestBuildEventsFromXml` class tested the YAML-style event building path which is incompatible with the new real-HexSim dict format. Remove those tests — Phase C/D will add new integration tests.

- [ ] **Step 3: Verify Columbia XML parses completely**

Run a quick check:
```bash
conda run -n shiny python -c "
from salmon_ibm.xml_parser import load_scenario_xml
c = load_scenario_xml('Columbia River Migration Model/Columbia [small]/Scenarios/gr_Columbia2017B.xml')
print(f'Populations: {len(c[\"populations\"])}')
print(f'Pop names: {[p[\"name\"] for p in c[\"populations\"]]}')
print(f'Events: {len(c[\"events\"])}')
print(f'Globals: {len(c[\"global_variables\"])}')
print(f'Spatial: {len(c[\"spatial_data_series\"])}')
print(f'Timesteps: {c[\"simulation\"][\"n_timesteps\"]}')
"
```
Expected output:
```
Populations: 4
Pop names: ['Chinook', 'Iterator', 'Refuges', 'Steelhead']
Events: 9
Globals: 58
Spatial: 18
Timesteps: 2928
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(xml): complete Phase A parser rewrite for real HexSim XML"
```
