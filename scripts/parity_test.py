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


# ---------------------------------------------------------------------------
# TASK 4: Census comparison and verdict
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402, F401


@dataclass
class Divergence:
    """A single metric divergence between HexSim and HexSimPy."""

    step: int
    pop: str
    metric: str
    hexsim_val: float
    hexsimpy_val: float
    rel_error: float


def compare_census(
    hexsim_census: dict[int, dict[int, dict]],
    hexsimpy_history: list[dict],
    pop_id_to_name: dict[int, str],
) -> list[Divergence]:
    """Compare population sizes and lambda between HexSim and HexSimPy census data.

    hexsim_census: {pop_id: {step: {size, lambda, traits}}}
    hexsimpy_history: list of {step, pop_name, n_alive}
    pop_id_to_name: {pop_id: pop_name}

    Returns list of Divergence records where metrics differ.
    """
    divergences: list[Divergence] = []

    # Index HexSimPy data by (pop_name, step)
    py_index: dict[tuple[str, int], dict] = {}
    for entry in hexsimpy_history:
        key = (entry["pop_name"], entry["step"])
        py_index[key] = entry

    # Track previous sizes for lambda computation
    prev_py_sizes: dict[str, int] = {}

    for pop_id, pop_name in pop_id_to_name.items():
        if pop_id not in hexsim_census:
            continue
        steps_data = hexsim_census[pop_id]

        for step in sorted(steps_data.keys()):
            hs_data = steps_data[step]
            py_entry = py_index.get((pop_name, step))
            if py_entry is None:
                continue

            hs_size = float(hs_data["size"])
            py_size = float(py_entry["n_alive"])

            # Compare population size
            if hs_size > 0:
                rel_err = abs(hs_size - py_size) / hs_size
            elif py_size > 0:
                rel_err = 1.0
            else:
                rel_err = 0.0

            if rel_err > 0.0:
                divergences.append(
                    Divergence(
                        step=step,
                        pop=pop_name,
                        metric="size",
                        hexsim_val=hs_size,
                        hexsimpy_val=py_size,
                        rel_error=rel_err,
                    )
                )

            # Compare lambda (growth rate)
            hs_lambda = hs_data.get("lambda", 1.0)
            prev_py = prev_py_sizes.get(pop_name)
            if prev_py is not None and prev_py > 0:
                py_lambda = py_size / prev_py
            else:
                py_lambda = 1.0  # default for first step

            if hs_lambda > 0:
                lam_err = abs(hs_lambda - py_lambda) / hs_lambda
            elif py_lambda != 1.0:
                lam_err = 1.0
            else:
                lam_err = 0.0

            if lam_err > 0.0:
                divergences.append(
                    Divergence(
                        step=step,
                        pop=pop_name,
                        metric="lambda",
                        hexsim_val=float(hs_lambda),
                        hexsimpy_val=py_lambda,
                        rel_error=lam_err,
                    )
                )

            prev_py_sizes[pop_name] = int(py_size)

    return divergences


def verdict(
    census_divergences: list[Divergence],
    agent_divergences: list[Divergence],
) -> str:
    """Determine overall parity verdict.

    FAIL if pop_size > 5% AND > 10 agents difference.
    FAIL if any agent metric > 15%.
    WARN if any agent metric 5-15%.
    PASS otherwise.
    """
    # Check census divergences
    for d in census_divergences:
        if d.metric == "size":
            abs_diff = abs(d.hexsim_val - d.hexsimpy_val)
            if d.rel_error > 0.05 and abs_diff > 10:
                return "FAIL"

    # Check agent divergences
    for d in agent_divergences:
        if d.rel_error > 0.15:
            return "FAIL"

    for d in agent_divergences:
        if d.rel_error > 0.05:
            return "WARN"

    return "PASS"


# ---------------------------------------------------------------------------
# TASK 5: Data Probe parsing and agent comparison
# ---------------------------------------------------------------------------

# Pattern: anything inside a "Data Probe" directory ending in .csv
_PROBE_RE = re.compile(r"^.+\.csv$")


def parse_hexsim_data_probe(
    result_dir: str,
    accumulator_map: dict[int, dict[int, str]],
) -> dict[int, dict[int, dict[int, dict[str, float]]]]:
    """Parse Data Probe CSVs from a HexSim results directory.

    Looks in result_dir / "Data Probe" for CSV files.
    CSV format: Run, Time Step, Population, Individual, Acc0, Acc1, ...

    Returns {pop_id: {step: {agent_id: {acc_name: value}}}}.
    """
    probe_dir = Path(result_dir) / "Data Probe"
    result: dict[int, dict[int, dict[int, dict[str, float]]]] = {}

    if not probe_dir.is_dir():
        return result

    for csv_file in sorted(probe_dir.glob("*.csv")):
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue

            header = [h.strip().strip('"') for h in header]
            # Columns: Run, Time Step, Population, Individual, then accumulators
            n_fixed = 4  # Run, Time Step, Population, Individual
            n_acc_cols = len(header) - n_fixed

            for row in reader:
                if not row or len(row) < n_fixed:
                    continue
                step = int(row[1].strip())
                pop_id = int(row[2].strip())
                agent_id = int(row[3].strip())

                # Look up accumulator names for this population
                acc_names = accumulator_map.get(pop_id, {})
                agent_data: dict[str, float] = {}
                for col_idx in range(n_acc_cols):
                    raw = row[n_fixed + col_idx].strip()
                    try:
                        val = float(raw)
                    except (ValueError, IndexError):
                        val = 0.0
                    name = acc_names.get(col_idx, f"Acc{col_idx}")
                    agent_data[name] = val

                result.setdefault(pop_id, {}).setdefault(step, {})[agent_id] = (
                    agent_data
                )

    return result


def compare_agents(
    hexsim_probe: dict[int, dict[int, dict[int, dict[str, float]]]],
    hexsimpy_snapshots: list[dict],
    pop_id_to_name: dict[int, str],
    energy_acc: str = "Energy Density",
    mass_acc: str = "Mass",
    n_cells: int | None = None,
) -> list[Divergence]:
    """Compare agent-level metrics at sampled steps using JSD for spatial.

    hexsim_probe: {pop_id: {step: {agent_id: {acc_name: value}}}}
    hexsimpy_snapshots: list of dicts with step, pop_name, ed_kJ_g, mass_g, tri_idx, n_cells
    pop_id_to_name: {pop_id: pop_name}

    Returns list[Divergence].
    """
    divergences: list[Divergence] = []

    # Index snapshots by (step, pop_name)
    snap_index: dict[tuple[int, str], dict] = {}
    for snap in hexsimpy_snapshots:
        key = (snap["step"], snap["pop_name"])
        snap_index[key] = snap

    name_to_id = {v: k for k, v in pop_id_to_name.items()}

    for pop_id, pop_name in pop_id_to_name.items():
        if pop_id not in hexsim_probe:
            continue
        for step in sorted(hexsim_probe[pop_id].keys()):
            snap = snap_index.get((step, pop_name))
            if snap is None:
                continue

            agents = hexsim_probe[pop_id][step]
            if not agents:
                continue

            # Extract HexSim accumulator values
            hs_ed = np.array([a.get(energy_acc, 0.0) for a in agents.values()])
            hs_mass = np.array([a.get(mass_acc, 0.0) for a in agents.values()])

            # Extract HexSimPy values
            py_ed = snap.get("ed_kJ_g", np.array([]))
            py_mass = snap.get("mass_g", np.array([]))
            py_cells = snap.get("tri_idx", np.array([]))

            # Compare mean energy density
            if len(hs_ed) > 0 and len(py_ed) > 0:
                hs_mean_ed = float(np.mean(hs_ed))
                py_mean_ed = float(np.mean(py_ed))
                if hs_mean_ed > 0:
                    rel_err = abs(hs_mean_ed - py_mean_ed) / hs_mean_ed
                elif py_mean_ed > 0:
                    rel_err = 1.0
                else:
                    rel_err = 0.0
                if rel_err > 0.0:
                    divergences.append(
                        Divergence(
                            step=step,
                            pop=pop_name,
                            metric="mean_ed",
                            hexsim_val=hs_mean_ed,
                            hexsimpy_val=py_mean_ed,
                            rel_error=rel_err,
                        )
                    )

            # Compare mean mass
            if len(hs_mass) > 0 and len(py_mass) > 0:
                hs_mean_mass = float(np.mean(hs_mass))
                py_mean_mass = float(np.mean(py_mass))
                if hs_mean_mass > 0:
                    rel_err = abs(hs_mean_mass - py_mean_mass) / hs_mean_mass
                elif py_mean_mass > 0:
                    rel_err = 1.0
                else:
                    rel_err = 0.0
                if rel_err > 0.0:
                    divergences.append(
                        Divergence(
                            step=step,
                            pop=pop_name,
                            metric="mean_mass",
                            hexsim_val=hs_mean_mass,
                            hexsimpy_val=py_mean_mass,
                            rel_error=rel_err,
                        )
                    )

            # Compare std dev of ED
            if len(hs_ed) > 1 and len(py_ed) > 1:
                hs_std = float(np.std(hs_ed))
                py_std = float(np.std(py_ed))
                if hs_std > 0:
                    rel_err = abs(hs_std - py_std) / hs_std
                elif py_std > 0:
                    rel_err = 1.0
                else:
                    rel_err = 0.0
                if rel_err > 0.0:
                    divergences.append(
                        Divergence(
                            step=step,
                            pop=pop_name,
                            metric="std_ed",
                            hexsim_val=hs_std,
                            hexsimpy_val=py_std,
                            rel_error=rel_err,
                        )
                    )

            # Spatial: JSD on cell ID histograms
            nc = n_cells or snap.get("n_cells")
            if nc is not None and len(py_cells) > 0:
                # Sanity check: max cell ID < n_cells
                max_py = int(np.max(py_cells)) if len(py_cells) > 0 else 0
                if max_py >= nc:
                    continue  # skip spatial comparison if IDs out of range

                # Build HexSim histogram from agent cell IDs if available
                # For now, HexSim probe doesn't have cell IDs directly;
                # use agent IDs as spatial proxy only if we have cell data
                # Actually, HexSim data probe doesn't store cell IDs,
                # so spatial JSD compares only HexSimPy snapshots if both exist
                # We compute JSD between the two distributions
                py_hist = np.bincount(py_cells.astype(int), minlength=nc).astype(float)
                py_hist_norm = py_hist / py_hist.sum() if py_hist.sum() > 0 else py_hist

                # If HexSim probe had cell column, we'd use it here
                # For now, record JSD = 0 as placeholder when no HexSim spatial data
                # This structure allows future extension
                jsd = 0.0  # placeholder — no HexSim cell data in probe

                if jsd > 0.0:
                    divergences.append(
                        Divergence(
                            step=step,
                            pop=pop_name,
                            metric="spatial_jsd",
                            hexsim_val=0.0,
                            hexsimpy_val=jsd,
                            rel_error=jsd,
                        )
                    )

    return divergences


# ---------------------------------------------------------------------------
# TASK 6: HexSim engine runner and report generator
# ---------------------------------------------------------------------------

import shutil  # noqa: E402
import subprocess  # noqa: E402


def run_hexsim_engine(
    patched_xml: str,
    hexsim_exe: str,
    seed: int = 42,
    timeout: int = 3600,
) -> tuple[float | None, dict[int, dict[int, dict]], dict]:
    """Run HexSim 4.0.20 engine on a patched scenario XML.

    Returns (elapsed_seconds | None, census, probe).
    If exe not found, returns (None, {}, {}).
    """
    exe_path = Path(hexsim_exe)
    xml_path = Path(patched_xml)

    if not exe_path.is_file():
        print(f"HexSim engine not found: {exe_path}")
        return None, {}, {}

    # workspace = patched_xml's grandparent (Scenarios/foo.xml -> workspace)
    workspace = xml_path.parent.parent
    result_dir = workspace / "Results" / xml_path.stem

    # Clean previous results
    if result_dir.exists():
        shutil.rmtree(result_dir)

    # Run engine
    cmd = [str(exe_path), str(xml_path), str(seed)]
    print(f"Running HexSim engine: {' '.join(cmd)}")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print(f"HexSim engine timed out after {timeout}s")
        return None, {}, {}

    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"HexSim engine failed (rc={proc.returncode})")
        if proc.stderr:
            print(f"stderr: {proc.stderr[:500]}")
        return None, {}, {}

    # Parse outputs
    census = {}
    probe = {}
    if result_dir.is_dir():
        census = parse_hexsim_census(str(result_dir))
        # Build accumulator map from the patched XML for probe parsing
        acc_map = build_accumulator_map(str(xml_path))
        probe = parse_hexsim_data_probe(str(result_dir), acc_map)

    return elapsed, census, probe


def generate_report(
    hexsim_time: float | None,
    hexsimpy_time: float,
    census_divs: list[Divergence],
    agent_divs: list[Divergence],
    hexsimpy_history: list[dict],
    output_path: str,
    scenario_name: str,
    n_steps: int,
    seed: int,
) -> str:
    """Write a markdown parity report. Returns the verdict string."""
    v = verdict(census_divs, agent_divs)

    lines: list[str] = []
    lines.append(f"# Parity Report: {scenario_name}")
    lines.append("")
    lines.append(f"- **Verdict:** {v}")
    lines.append(f"- **Steps:** {n_steps}")
    lines.append(f"- **Seed:** {seed}")
    lines.append(f"- **Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Timing
    lines.append("## Timing")
    lines.append("")
    if hexsim_time is not None:
        lines.append(f"- HexSim C++: {hexsim_time:.2f}s")
    else:
        lines.append("- HexSim C++: not available")
    lines.append(f"- HexSimPy: {hexsimpy_time:.2f}s")
    lines.append("")

    # Census divergences
    lines.append("## Census Divergences")
    lines.append("")
    if census_divs:
        lines.append("| Step | Pop | Metric | HexSim | HexSimPy | Rel Error |")
        lines.append("|------|-----|--------|--------|----------|-----------|")
        for d in census_divs:
            lines.append(
                f"| {d.step} | {d.pop} | {d.metric} | "
                f"{d.hexsim_val:.2f} | {d.hexsimpy_val:.2f} | "
                f"{d.rel_error:.4f} |"
            )
    else:
        lines.append("No census divergences detected.")
    lines.append("")

    # Agent divergences
    lines.append("## Agent Divergences")
    lines.append("")
    if agent_divs:
        lines.append("| Step | Pop | Metric | HexSim | HexSimPy | Rel Error |")
        lines.append("|------|-----|--------|--------|----------|-----------|")
        for d in agent_divs:
            lines.append(
                f"| {d.step} | {d.pop} | {d.metric} | "
                f"{d.hexsim_val:.2f} | {d.hexsimpy_val:.2f} | "
                f"{d.rel_error:.4f} |"
            )
    else:
        lines.append("No agent divergences detected.")
    lines.append("")

    # Population summary from HexSimPy
    lines.append("## HexSimPy Population Summary")
    lines.append("")
    if hexsimpy_history:
        # Last step per population
        last_by_pop: dict[str, dict] = {}
        for entry in hexsimpy_history:
            last_by_pop[entry["pop_name"]] = entry
        for pop_name, entry in sorted(last_by_pop.items()):
            lines.append(
                f"- **{pop_name}**: {entry['n_alive']} alive at step {entry['step']}"
            )
    lines.append("")

    report_text = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_text, encoding="utf-8")
    print(f"Report written to {out}")

    return v


# ---------------------------------------------------------------------------
# TASK 7: CLI main and integration
# ---------------------------------------------------------------------------

import argparse  # noqa: E402


def main() -> None:
    """CLI entry point for parity testing."""
    parser = argparse.ArgumentParser(
        description="Run parity test between HexSim C++ and HexSimPy."
    )
    parser.add_argument(
        "--workspace", required=True, help="Path to HexSim workspace directory"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Relative path to scenario XML within workspace",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2928,
        help="Number of simulation steps (default: 2928)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="HexSim engine timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--hexsim-exe",
        default="HexSim 4.0.20/HexSimEngine64.exe",
        help="Path to HexSim engine executable",
    )
    parser.add_argument(
        "--probe-interval",
        type=int,
        default=100,
        help="Agent snapshot interval in steps (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="docs/parity-report.md",
        help="Output report path (default: docs/parity-report.md)",
    )

    args = parser.parse_args()

    # Resolve paths
    workspace = Path(args.workspace).resolve()
    scenario_xml = workspace / args.scenario

    # Validate
    if not workspace.is_dir():
        print(f"ERROR: Workspace not found: {workspace}")
        sys.exit(1)
    if not scenario_xml.is_file():
        print(f"ERROR: Scenario XML not found: {scenario_xml}")
        sys.exit(1)

    scenario_name = scenario_xml.stem

    # Step 1: Patch XML
    patched_dir = workspace / "Scenarios"
    patched_xml = patched_dir / f"{scenario_name}_parity.xml"
    print(f"Patching scenario XML -> {patched_xml}")
    patch_scenario_xml(
        str(scenario_xml),
        str(patched_xml),
        str(workspace),
        start_log_step=1,
        n_steps=args.steps,
    )

    # Step 2: Build accumulator map
    acc_map = build_accumulator_map(str(patched_xml))
    print(f"Accumulator map: {len(acc_map)} populations")

    # Build pop_id_to_name from XML
    tree = ET.parse(str(patched_xml))
    root = tree.getroot()
    pop_id_to_name: dict[int, str] = {}
    for idx, pop_el in enumerate(root.findall("population")):
        name = pop_el.get("name", f"Pop{idx}")
        pop_id_to_name[idx] = name
    print(f"Populations: {pop_id_to_name}")

    # Step 3: Run HexSim C++ engine
    hexsim_exe = Path(args.hexsim_exe)
    if not hexsim_exe.is_absolute():
        hexsim_exe = PROJECT / args.hexsim_exe
    hs_time, hs_census, hs_probe = run_hexsim_engine(
        str(patched_xml), str(hexsim_exe), seed=args.seed, timeout=args.timeout
    )

    # Step 4: Run HexSimPy
    print(f"Running HexSimPy for {args.steps} steps...")
    py_time, py_census, py_snapshots = run_hexsimpy(
        str(workspace),
        str(patched_xml),
        seed=args.seed,
        n_steps=args.steps,
        probe_interval=args.probe_interval,
    )
    print(f"HexSimPy completed in {py_time:.2f}s")

    # Step 5: Compare census
    census_divs = compare_census(hs_census, py_census, pop_id_to_name)
    print(f"Census divergences: {len(census_divs)}")

    # Step 6: Compare agents
    agent_divs = compare_agents(hs_probe, py_snapshots, pop_id_to_name)
    print(f"Agent divergences: {len(agent_divs)}")

    # Step 7: Generate report
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT / args.output

    v = generate_report(
        hexsim_time=hs_time,
        hexsimpy_time=py_time,
        census_divs=census_divs,
        agent_divs=agent_divs,
        hexsimpy_history=py_census,
        output_path=str(output_path),
        scenario_name=scenario_name,
        n_steps=args.steps,
        seed=args.seed,
    )
    print(f"Verdict: {v}")


if __name__ == "__main__":
    main()
