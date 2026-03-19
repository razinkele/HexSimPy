"""HexSim XML scenario parser: load real HexSim .xml files into config dicts."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def load_scenario_xml(path: str | Path) -> dict:
    """Parse a HexSim scenario XML file into a structured config dict.

    Returns a dict with keys:
      - 'simulation': dict with n_timesteps, n_replicates, start_log_step
      - 'grid': dict with n_hexagons, rows, columns, narrow, cell_width
      - 'workspace': str path from <workspace> element
      - 'global_variables': dict[name -> float]
      - 'spatial_data_series': dict[name -> {datatype, time_series, cycle_length}]
      - 'populations': list of population defs
      - 'events': list of event defs (recursive tree)
    """
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


# ---------------------------------------------------------------------------
# Global variables and spatial data series
# ---------------------------------------------------------------------------

def _parse_global_variables(root) -> dict[str, float]:
    """Parse <globalVariables> block. Attributes are capital-case Name/Value."""
    result = {}
    gv_elem = root.find("globalVariables")
    if gv_elem is None:
        return result
    for var_elem in gv_elem.iter("globalVariable"):
        name = var_elem.get("Name", "")   # capital N
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


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

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
    """Parse <accumulator> elements. Bounds are XML ATTRIBUTES, not child elements."""
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
    traits_container = pop_elem.find("traits")
    if traits_container is None:
        return result

    for child in traits_container:
        tag = child.tag
        name = child.get("name", "")
        if not name:
            continue

        if tag == "probabilisticTrait":
            categories = []
            for val_elem in child.findall("value"):
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

        elif tag == "accumulatedTrait":
            accumulator = child.get("accumulator", "")
            categories = []
            for val_elem in child.findall("value"):
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
    for aff_elem in affinities_elem.findall("affinity"):
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
        # Group-level trait filter
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
    if event_type != "event_group":
        trait_filter = _parse_trait_filter(elem)
        if trait_filter:
            event_def["trait_filter"] = trait_filter

    return event_def


# ---------------------------------------------------------------------------
# Updater functions
# ---------------------------------------------------------------------------

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
        func_name = func_name[: -len("UpdaterFunction")]

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


# ---------------------------------------------------------------------------
# Trait filter
# ---------------------------------------------------------------------------

def _parse_trait_filter(elem) -> dict | None:
    """Extract trait filter from event element.

    <trait> + <traitCombinations> can be direct children.
    """
    traits = [t.text.strip() for t in elem.findall("trait") if t.text]
    combos = _text(elem, "traitCombinations")
    if traits:
        return {"traits": traits, "combinations": combos}
    return None


# ---------------------------------------------------------------------------
# Move event params
# ---------------------------------------------------------------------------

def _parse_move_params(elem) -> dict:
    """Parse <moveEvent> parameters."""
    # dispersalUseAffinity: empty element = True (default slot), named text = specific slot
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


# ---------------------------------------------------------------------------
# Transition event params
# ---------------------------------------------------------------------------

def _parse_transition_params(elem) -> dict:
    """Parse <transitionEvent> parameters including matrix data."""
    transition_trait = _text(elem, "transitionTrait")
    conditioning_traits = [t.text.strip() for t in elem.findall("trait") if t.text]

    # Parse matrixSet
    ms_elem = elem.find("matrixSet")
    rows = 0
    columns = 0
    matrix_data = None
    rows_reversed = False
    if ms_elem is not None:
        rows = _int_text(ms_elem, "rows", 0)
        columns = _int_text(ms_elem, "columns", 0)
        rows_reversed = ms_elem.get("rowsReversed", "false").lower() == "true"
        matrices_elem = ms_elem.find("matrices")
        if matrices_elem is not None and matrices_elem.text:
            matrix_data = [float(x) for x in matrices_elem.text.split()]

    return {
        "transition_trait": transition_trait,
        "conditioning_traits": conditioning_traits,
        "rows": rows,
        "columns": columns,
        "rows_reversed": rows_reversed,
        "matrix_data": matrix_data,
    }


# ---------------------------------------------------------------------------
# Interaction event params
# ---------------------------------------------------------------------------

def _parse_interaction_params(elem) -> dict:
    """Parse <interactionEvent> parameters."""
    presence = _text(elem, "presence", "explored")
    colocation_elem = elem.find("colocation")
    colocation_type = (
        colocation_elem.get("type", "always") if colocation_elem is not None else "always"
    )

    interactions = []
    for int_elem in elem.findall("interaction"):
        int_def = {
            "name": _text(int_elem, "name"),
            "population": _text(int_elem, "populationName"),
            "presence": _text(int_elem, "presence", "explored"),
            "encounter_probability": _float_text(int_elem, "encounterProbability", 1.0),
        }
        outcomes_elem = int_elem.find("outcomes")
        if outcomes_elem is not None:
            int_def["outcomes"] = _parse_outcomes(outcomes_elem)
        interactions.append(int_def)

    return {
        "presence": presence,
        "colocation_type": colocation_type,
        "interactions": interactions,
    }


def _parse_outcomes(outcomes_elem) -> dict:
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
