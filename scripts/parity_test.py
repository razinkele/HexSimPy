"""Parity test: run the same HexSim scenario on C++ and Python, compare outputs."""

from __future__ import annotations

import csv  # noqa: F401
import re  # noqa: F401
import sys  # noqa: F401
import time  # noqa: F401
import xml.etree.ElementTree as ET

import numpy as np  # noqa: F401
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


# ---------------------------------------------------------------------------
# TASK 3: Run HexSimPy
# ---------------------------------------------------------------------------


def run_hexsimpy(
    workspace: str,
    scenario_xml: str,
    seed: int = 42,
    n_steps: int = 100,
    probe_interval: int = 100,
) -> tuple[float, list[dict], list[dict]]:
    """Run HexSimPy on a scenario and collect per-step census + agent snapshots.

    Uses ScenarioLoader.load() then manual sim.step() loop.

    Returns (elapsed_seconds, census_list, snapshots_list).
    """
    # Add project root to path so salmon_ibm is importable
    proj = str(PROJECT)
    if proj not in sys.path:
        sys.path.insert(0, proj)

    from salmon_ibm.scenario_loader import ScenarioLoader

    loader = ScenarioLoader()
    sim = loader.load(workspace, scenario_xml, rng_seed=seed)

    census_list: list[dict] = []
    snapshots_list: list[dict] = []

    t0 = time.perf_counter()
    for step in range(n_steps):
        sim.step()

        # Collect per-population census
        for pop_name, pop in sim.populations.populations.items():
            entry = {
                "step": step,
                "pop_name": pop_name,
                "n_alive": pop.n_alive,
            }
            # Add trait distribution if trait_mgr available
            if pop.trait_mgr is not None and pop.trait_mgr.definitions:
                trait_counts = {}
                for tname in pop.trait_mgr.definitions:
                    arr = pop.trait_mgr.get(tname)
                    alive_mask = pop.alive
                    alive_vals = arr[alive_mask]
                    trait_counts[tname] = (
                        int(np.sum(alive_vals > 0)) if len(alive_vals) > 0 else 0
                    )
                entry["traits"] = trait_counts
            census_list.append(entry)

        # Collect agent snapshots at probe_interval
        if (step + 1) % probe_interval == 0:
            for pop_name, pop in sim.populations.populations.items():
                alive_mask = pop.alive
                snapshot = {
                    "step": step,
                    "pop_name": pop_name,
                    "n_alive": pop.n_alive,
                    "n_cells": sim.landscape["mesh"].n_cells,
                }
                # Snapshot agent arrays for alive agents
                if hasattr(pop.pool, "ed_kJ_g"):
                    snapshot["ed_kJ_g"] = pop.pool.ed_kJ_g[alive_mask].copy()
                if hasattr(pop.pool, "mass_g"):
                    snapshot["mass_g"] = pop.pool.mass_g[alive_mask].copy()
                if hasattr(pop.pool, "tri_idx"):
                    snapshot["tri_idx"] = pop.pool.tri_idx[alive_mask].copy()
                snapshots_list.append(snapshot)

    elapsed = time.perf_counter() - t0
    return (elapsed, census_list, snapshots_list)
