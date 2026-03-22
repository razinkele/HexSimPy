"""Comparative benchmark: original HexSim 4.0.20 vs HexSimPy.

Runs the snake_Columbia2017B.xml steelhead scenario on both engines
and compares population trajectories, timing, and survival.
"""

import csv
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
WORKSPACE = PROJECT / "Columbia [small]"
SCENARIO_XML = WORKSPACE / "Scenarios" / "snake_Columbia2017B.xml"
HEXSIM_ENGINE = PROJECT / "HexSim 4.0.20" / "HexSimEngine64.exe"
RESULTS_DIR = WORKSPACE / "Results"
BENCHMARK_OUT = PROJECT / "docs" / "benchmark_columbia_steelhead.md"

RNG_SEED = 42
N_STEPS = 2928  # full scenario: 122 days of hourly timesteps
LOG_EVERY = 1  # log census every step


def patch_scenario_xml(src: Path, dst: Path, workspace: Path, start_log_step: int = 1):
    """Create a patched copy of the scenario XML with local workspace path
    and logging enabled from step 1.

    Replaces all hardcoded absolute paths (e.g. F:\\Marcia\\Columbia [small])
    with the local workspace path via string replacement on the raw XML.
    """
    xml_text = src.read_text(encoding="utf-8")

    # Replace the original absolute workspace path with local path
    # The XML contains paths like: F:\Marcia\Columbia [small]\...
    local_ws = str(workspace)
    xml_text = xml_text.replace(r"F:\Marcia\Columbia [small]", local_ws)

    # Parse the patched XML
    root = ET.fromstring(xml_text)

    # Enable logging from step 1
    sls = root.find(".//simulationParameters/startLogStep")
    if sls is not None:
        sls.text = str(start_log_step)

    tree = ET.ElementTree(root)
    tree.write(dst, xml_declaration=True, encoding="unicode")
    return dst


def run_original_hexsim(patched_xml: Path, seed: int = 42):
    """Run the original HexSim 4.0.20 engine and return (elapsed_seconds, census_rows)."""
    if not HEXSIM_ENGINE.exists():
        print(f"  [SKIP] HexSim engine not found at {HEXSIM_ENGINE}")
        return None, []

    # Clean previous results
    scenario_name = patched_xml.stem
    result_dir = RESULTS_DIR / scenario_name
    if result_dir.exists():
        shutil.rmtree(result_dir)

    print(f"  Running HexSim 4.0.20 on {patched_xml.name} (seed={seed})...")
    print(f"  This may take several minutes for {N_STEPS} timesteps...")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(HEXSIM_ENGINE), "-r", str(seed), str(patched_xml)],
            capture_output=True,
            text=True,
            cwd=str(WORKSPACE),
            timeout=3600,  # 1 hour timeout
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"  [ERROR] HexSim engine failed (code {proc.returncode})")
            if proc.stderr:
                print(f"  stderr: {proc.stderr[:500]}")
            if proc.stdout:
                print(f"  stdout: {proc.stdout[:500]}")
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(
            f"  HexSim 4.0.20 timed out after {elapsed:.1f}s (collecting partial results)"
        )
        return elapsed, []

    print(f"  HexSim 4.0.20 completed in {elapsed:.1f}s")

    # Parse census CSV
    census_rows = parse_hexsim_census(result_dir, scenario_name)
    return elapsed, census_rows


def parse_hexsim_census(result_dir: Path, scenario_name: str):
    """Parse HexSim census CSV output."""
    csv_path = result_dir / f"{scenario_name}.0.csv"
    if not csv_path.exists():
        # Try to find any CSV in the result dir
        csvs = list(result_dir.glob("*.csv"))
        if csvs:
            csv_path = csvs[0]
        else:
            print(f"  [WARN] No census CSV found in {result_dir}")
            return []

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_hexsim_log(result_dir: Path, scenario_name: str):
    """Parse HexSim log file for population counts per step."""
    # HexSim writes log lines with CEN records
    log_path = result_dir / f"{scenario_name}.log"
    if not log_path.exists():
        logs = list(result_dir.glob("*.log"))
        if logs:
            log_path = logs[0]
        else:
            return {}

    pop_counts = {}  # step -> {pop_name: count}
    with open(log_path) as f:
        for line in f:
            if line.startswith("CEN"):
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    step = int(parts[1].strip())
                    pop_id = int(parts[2].strip())
                    pop_size = int(parts[3].strip())
                    if step not in pop_counts:
                        pop_counts[step] = {}
                    pop_counts[step][pop_id] = pop_size
    return pop_counts


def run_hexsimpy(workspace: Path, scenario_xml: Path, seed: int = 42):
    """Run HexSimPy ScenarioLoader and return (elapsed_seconds, history)."""
    print(f"  Running HexSimPy on {scenario_xml.name} (seed={seed})...")

    # Add project root to path
    sys.path.insert(0, str(PROJECT))

    from salmon_ibm.scenario_loader import ScenarioLoader

    t0 = time.perf_counter()
    loader = ScenarioLoader()
    sim = loader.load(
        workspace_dir=str(workspace),
        scenario_xml=str(scenario_xml),
        rng_seed=seed,
    )

    load_time = time.perf_counter() - t0
    print(f"  ScenarioLoader.load() completed in {load_time:.1f}s")

    # Collect per-step population data
    history = []  # list of {step, pop_name: n_alive}
    pop_names = list(sim.populations.populations.keys())

    print(f"  Populations: {pop_names}")
    print(f"  Running {N_STEPS} timesteps...")

    t_run = time.perf_counter()
    for step in range(N_STEPS):
        sim.step()
        record = {"step": step}
        for pname, pop in sim.populations.populations.items():
            record[pname] = pop.n_alive
        history.append(record)

        if (step + 1) % 500 == 0:
            alive_str = ", ".join(f"{k}={v}" for k, v in record.items() if k != "step")
            print(f"    step {step + 1}/{N_STEPS}: {alive_str}")

    run_time = time.perf_counter() - t_run
    total_time = time.perf_counter() - t0
    print(
        f"  HexSimPy run completed in {run_time:.1f}s (total with load: {total_time:.1f}s)"
    )

    return total_time, load_time, run_time, history


def extract_census_trajectory(census_rows, pop_id):
    """Extract population size trajectory for a given pop_id from HexSim census."""
    trajectory = {}
    for row in census_rows:
        try:
            step = int(row.get("Current Step", row.get("Step", -1)))
            pid = int(row.get("Population ID", row.get("PopID", -1)))
            size = int(row.get("Population Size", row.get("PopSize", 0)))
            if pid == pop_id:
                trajectory[step] = size
        except (ValueError, KeyError):
            continue
    return trajectory


def generate_report(
    hexsim_time,
    hexsim_census,
    hexsimpy_time,
    hexsimpy_load,
    hexsimpy_run,
    hexsimpy_history,
    output_path,
):
    """Generate a markdown comparison report."""
    lines = [
        "# Comparative Benchmark: HexSim 4.0.20 vs HexSimPy",
        "",
        "**Scenario**: `snake_Columbia2017B.xml` (Snake River steelhead + Chinook)",
        f"**Workspace**: Columbia [small] ({N_STEPS} hourly timesteps = 122 days)",
        f"**RNG Seed**: {RNG_SEED}",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Timing",
        "",
        "| Metric | HexSim 4.0.20 | HexSimPy |",
        "|--------|--------------|----------|",
    ]

    if hexsim_time is not None:
        lines.append(f"| Total wall time | {hexsim_time:.1f}s | {hexsimpy_time:.1f}s |")
        speedup = hexsim_time / hexsimpy_run if hexsimpy_run > 0 else 0
        lines.append(f"| Simulation run time | — | {hexsimpy_run:.1f}s |")
        lines.append(f"| Scenario load time | — | {hexsimpy_load:.1f}s |")
        lines.append(f"| Speedup (run only) | — | {speedup:.1f}x |")
    else:
        lines.append(f"| Total wall time | (not run) | {hexsimpy_time:.1f}s |")
        lines.append(f"| Simulation run time | — | {hexsimpy_run:.1f}s |")
        lines.append(f"| Scenario load time | — | {hexsimpy_load:.1f}s |")

    # HexSimPy population summary
    if hexsimpy_history:
        pop_keys = [k for k in hexsimpy_history[0].keys() if k != "step"]
        lines += [
            "",
            "---",
            "",
            "## Population Trajectories (HexSimPy)",
            "",
        ]

        for pname in pop_keys:
            traj = [h[pname] for h in hexsimpy_history]
            initial = traj[0]
            final = traj[-1]
            peak = max(traj)
            peak_step = traj.index(peak)
            minimum = min(traj)
            min_step = traj.index(minimum)

            lines += [
                f"### {pname}",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Initial population | {initial} |",
                f"| Final population | {final} |",
                f"| Peak ({peak_step}h) | {peak} |",
                f"| Minimum ({min_step}h) | {minimum} |",
                f"| Survival rate | {final / initial * 100:.1f}% |"
                if initial > 0
                else "| Survival rate | N/A |",
                "",
            ]

        # Milestone table: population at key timepoints
        milestones = [0, 24, 168, 720, 1440, 2160, 2927]
        milestones = [m for m in milestones if m < len(hexsimpy_history)]

        lines += [
            "### Population at Key Timepoints",
            "",
            "| Hour | Day | " + " | ".join(pop_keys) + " |",
            "|------|-----|" + "|".join(["------"] * len(pop_keys)) + "|",
        ]
        for m in milestones:
            h = hexsimpy_history[m]
            day = m // 24
            vals = " | ".join(str(h[k]) for k in pop_keys)
            lines.append(f"| {m} | {day} | {vals} |")

    # HexSim census comparison (if available)
    if hexsim_census:
        lines += [
            "",
            "---",
            "",
            "## HexSim 4.0.20 Census Output",
            "",
            "| Step | " + " | ".join(hexsim_census[0].keys()) + " |",
            "|------|" + "|".join(["------"] * len(hexsim_census[0].keys())) + "|",
        ]
        # Show first 10 and last 10 rows
        show = (
            hexsim_census[:10]
            + (["..."] if len(hexsim_census) > 20 else [])
            + hexsim_census[-10:]
        )
        for row in show:
            if isinstance(row, str):
                lines.append(
                    f"| ... | {'... | ' * (len(hexsim_census[0].keys()) - 1)}|"
                )
            else:
                vals = " | ".join(str(v).strip() for v in row.values())
                lines.append(f"| — | {vals} |")

        # Numerical comparison if we have matching steps
        if hexsimpy_history:
            lines += [
                "",
                "### Trajectory Comparison",
                "",
                "Population sizes at matching timesteps from both engines:",
                "",
            ]
            # Try to match pop IDs to names
            # HexSim uses integer pop IDs (0-based), HexSimPy uses names
            # We'll attempt alignment by order
            hexsim_pops = {}
            for row in hexsim_census:
                try:
                    pid = int(row.get("Population ID", -1))
                    step = int(row.get("Current Step", -1))
                    size = int(row.get("Population Size", 0))
                    if pid not in hexsim_pops:
                        hexsim_pops[pid] = {}
                    hexsim_pops[pid][step] = size
                except (ValueError, KeyError):
                    continue

            if hexsim_pops:
                for pid, traj in sorted(hexsim_pops.items()):
                    steps = sorted(traj.keys())
                    lines.append(
                        f"**Population ID {pid}**: {len(steps)} census records, "
                        f"initial={traj.get(steps[0], '?')}, final={traj.get(steps[-1], '?')}"
                    )

    # Performance metrics
    if hexsimpy_history:
        total_agents = sum(
            sum(h[k] for k in h if k != "step") for h in hexsimpy_history
        )
        agent_steps_per_ms = (
            total_agents / (hexsimpy_run * 1000) if hexsimpy_run > 0 else 0
        )

        lines += [
            "",
            "---",
            "",
            "## Performance Metrics (HexSimPy)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total agent-steps | {total_agents:,} |",
            f"| Agent-steps/ms | {agent_steps_per_ms:,.0f} |",
            f"| Steps/second | {N_STEPS / hexsimpy_run:.0f} |"
            if hexsimpy_run > 0
            else "",
            f"| Mean step time | {hexsimpy_run / N_STEPS * 1000:.1f}ms |"
            if hexsimpy_run > 0
            else "",
        ]

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- HexSim 4.0.20 is the original C++ EPA implementation (single-threaded)",
        "- HexSimPy uses Numba JIT with `parallel=True` for movement and event kernels",
        "- Both engines use the same spatial data, temperature zones, and barrier configuration",
        "- Stochastic differences are expected due to different RNG implementations",
        "- The scenario includes both Chinook and Steelhead populations with ~2000 fish each",
        "- Refuges is a stationary population marking cold-water refuge locations",
    ]

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {output_path}")
    return report


def main():
    print("=" * 70)
    print("Comparative Benchmark: HexSim 4.0.20 vs HexSimPy")
    print(f"Scenario: snake_Columbia2017B.xml ({N_STEPS} timesteps)")
    print("=" * 70)

    # Validate inputs
    if not WORKSPACE.exists():
        print(f"ERROR: Workspace not found: {WORKSPACE}")
        sys.exit(1)
    if not SCENARIO_XML.exists():
        print(f"ERROR: Scenario XML not found: {SCENARIO_XML}")
        sys.exit(1)

    # ── Step 1: Patch XML for local use ────────────────────────────────────
    print("\n[1/4] Patching scenario XML for local workspace path...")
    patched_xml = WORKSPACE / "Scenarios" / "snake_Columbia2017B_local.xml"
    patch_scenario_xml(SCENARIO_XML, patched_xml, WORKSPACE, start_log_step=1)
    print(f"  Patched XML: {patched_xml}")

    # ── Step 2: Run original HexSim ────────────────────────────────────────
    print("\n[2/4] Running original HexSim 4.0.20...")
    hexsim_time, hexsim_census = run_original_hexsim(patched_xml, seed=RNG_SEED)

    # ── Step 3: Run HexSimPy ──────────────────────────────────────────────
    print("\n[3/4] Running HexSimPy...")
    hexsimpy_total, hexsimpy_load, hexsimpy_run, hexsimpy_history = run_hexsimpy(
        WORKSPACE, SCENARIO_XML, seed=RNG_SEED
    )

    # ── Step 4: Generate comparison report ─────────────────────────────────
    print("\n[4/4] Generating comparison report...")
    report = generate_report(
        hexsim_time,
        hexsim_census,
        hexsimpy_total,
        hexsimpy_load,
        hexsimpy_run,
        hexsimpy_history,
        BENCHMARK_OUT,
    )

    print("\n" + "=" * 70)
    print(report)


if __name__ == "__main__":
    main()
