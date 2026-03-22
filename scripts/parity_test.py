"""Parity test: run the same HexSim scenario on C++ and Python, compare outputs."""

from __future__ import annotations

import csv  # noqa: F401
import re  # noqa: F401
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


# ---------------------------------------------------------------------------
# TASK 2: HexSim census CSV parsing
# ---------------------------------------------------------------------------

# Pattern: scenario_name.POPID.csv
_CENSUS_RE = re.compile(r"^.+\.(\d+)\.csv$")


def parse_hexsim_census(result_dir: str) -> dict[int, dict[int, dict]]:
    """Parse HexSim census CSV files from a results directory.

    Population ID is extracted from filename suffix: scenario.N.csv -> pop_id=N.

    Returns {pop_id: {step: {size, traits, lambda}}}.
    """
    result_path = Path(result_dir)
    census: dict[int, dict[int, dict]] = {}

    for csv_file in sorted(result_path.glob("*.csv")):
        m = _CENSUS_RE.match(csv_file.name)
        if m is None:
            continue
        pop_id = int(m.group(1))
        pop_data: dict[int, dict] = {}

        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue

            # Clean whitespace from header fields
            header = [h.strip().strip('"') for h in header]

            # Find trait columns
            trait_cols: dict[int, int] = {}  # col_idx -> trait_index
            for ci, col_name in enumerate(header):
                tm = re.match(r"Trait Index\s+(\d+)", col_name)
                if tm:
                    trait_cols[ci] = int(tm.group(1))

            for row in reader:
                if not row:
                    continue
                step = int(row[1].strip())
                size = int(row[2].strip())
                lam = float(row[5].strip())

                traits: dict[int, int] = {}
                for ci, ti in trait_cols.items():
                    traits[ti] = int(row[ci].strip())

                pop_data[step] = {"size": size, "lambda": lam, "traits": traits}

        if pop_data:
            census[pop_id] = pop_data

    return census
