"""Parity test: run the same HexSim scenario on C++ and Python, compare outputs."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# The hardcoded path in original scenario XMLs
_ORIGINAL_PREFIX = r"F:\Marcia\Columbia [small]"


# ---------------------------------------------------------------------------
# TASK 1: XML patching and accumulator map
# ---------------------------------------------------------------------------


def patch_scenario_xml(
    src: str,
    dst: str,
    workspace: str,
    start_log_step: int = 1,
    n_steps: int = 2928,
) -> None:
    """Patch a HexSim scenario XML for local execution.

    - Replace hardcoded F:\\Marcia\\Columbia [small] paths with *workspace*
    - Override <timesteps> and <startLogStep> elements
    - Write result to *dst*
    """
    src_path = Path(src)
    text = src_path.read_text(encoding="utf-8")

    # Replace hardcoded path with local workspace (handle both slash styles)
    text = text.replace(_ORIGINAL_PREFIX, str(workspace))
    text = text.replace(_ORIGINAL_PREFIX.replace("\\", "/"), str(workspace))

    # Parse and override simulation parameters
    tree = ET.ElementTree(ET.fromstring(text))
    root = tree.getroot()

    ts_el = root.find(".//timesteps")
    if ts_el is not None:
        ts_el.text = str(n_steps)

    sls_el = root.find(".//startLogStep")
    if sls_el is not None:
        sls_el.text = str(start_log_step)

    tree.write(str(dst), encoding="unicode", xml_declaration=True)


def build_accumulator_map(scenario_xml: str) -> dict[int, dict[int, str]]:
    """Extract accumulator names per population in definition order.

    Returns {pop_index: {col_index: accumulator_name}}.
    """
    tree = ET.parse(scenario_xml)
    root = tree.getroot()

    result: dict[int, dict[int, str]] = {}
    populations = root.findall("population")
    for pop_idx, pop_el in enumerate(populations):
        accumulators = pop_el.findall(".//accumulator")
        col_map: dict[int, str] = {}
        for col_idx, acc_el in enumerate(accumulators):
            name = acc_el.get("name", f"unnamed_{col_idx}")
            col_map[col_idx] = name
        if col_map:
            result[pop_idx] = col_map

    return result
