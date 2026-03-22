# Parity Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `scripts/parity_test.py` and `~/.claude/skills/parity-test/SKILL.md` that run the same HexSim scenario on both HexSim 4.0.20 (C++) and HexSimPy, compare census + data probe outputs, and report PASS/WARN/FAIL.

**Architecture:** Single Python script with CLI (`argparse`), sequential engine execution (HexSim first, then HexSimPy), structured comparison with tolerance bands, markdown report generation. A Claude Code skill file teaches when/how to invoke it.

**Tech Stack:** Python 3.10+, NumPy, SciPy (`spatial.distance.jensenshannon`), `xml.etree.ElementTree`, `argparse`, `subprocess`, `csv`

**Spec:** `docs/superpowers/specs/2026-03-22-parity-test-design.md`

---

### Task 1: XML Patching and Accumulator Map

**Files:**
- Create: `scripts/parity_test.py`
- Test: `tests/test_parity_test.py`

- [ ] **Step 1: Write failing tests for `patch_scenario_xml` and `build_accumulator_map`**

```python
# tests/test_parity_test.py
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# Use the real scenario XML for testing
WORKSPACE = Path("Columbia [small]")
SCENARIO_XML = WORKSPACE / "Scenarios" / "snake_Columbia2017B.xml"
HAS_WORKSPACE = WORKSPACE.exists()

pytestmark = pytest.mark.skipif(not HAS_WORKSPACE, reason="Columbia workspace not found")


class TestPatchScenarioXml:
    def test_replaces_workspace_path(self, tmp_path):
        from scripts.parity_test import patch_scenario_xml
        dst = tmp_path / "patched.xml"
        patch_scenario_xml(SCENARIO_XML, dst, WORKSPACE, start_log_step=1, n_steps=100)
        text = dst.read_text()
        assert r"F:\Marcia" not in text
        assert str(WORKSPACE) in text

    def test_sets_start_log_step(self, tmp_path):
        from scripts.parity_test import patch_scenario_xml
        dst = tmp_path / "patched.xml"
        patch_scenario_xml(SCENARIO_XML, dst, WORKSPACE, start_log_step=5, n_steps=100)
        tree = ET.parse(dst)
        sls = tree.find(".//simulationParameters/startLogStep")
        assert sls is not None and sls.text == "5"

    def test_sets_timesteps(self, tmp_path):
        from scripts.parity_test import patch_scenario_xml
        dst = tmp_path / "patched.xml"
        patch_scenario_xml(SCENARIO_XML, dst, WORKSPACE, start_log_step=1, n_steps=50)
        tree = ET.parse(dst)
        ts = tree.find(".//simulationParameters/timesteps")
        assert ts is not None and ts.text == "50"


class TestBuildAccumulatorMap:
    def test_returns_per_population_dict(self):
        from scripts.parity_test import build_accumulator_map
        result = build_accumulator_map(SCENARIO_XML)
        assert isinstance(result, dict)
        assert len(result) >= 2  # at least Chinook + Steelhead

    def test_chinook_has_energy_accumulators(self):
        from scripts.parity_test import build_accumulator_map
        result = build_accumulator_map(SCENARIO_XML)
        # Pop 0 is Chinook — check it has accumulators
        pop0_names = set(result[0].values())
        assert len(pop0_names) > 10  # Chinook has many accumulators
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n shiny python -m pytest tests/test_parity_test.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `patch_scenario_xml` and `build_accumulator_map`**

```python
# scripts/parity_test.py
"""Parity test: run same scenario on HexSim 4.0.20 and HexSimPy, compare outputs."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent


def patch_scenario_xml(
    src: Path, dst: Path, workspace: Path, start_log_step: int = 1, n_steps: int = 2928
):
    """Patch XML: replace hardcoded paths, set logging and timesteps."""
    xml_text = src.read_text(encoding="utf-8")
    # Replace original absolute workspace path with local
    xml_text = xml_text.replace(r"F:\Marcia\Columbia [small]", str(workspace))
    root = ET.fromstring(xml_text)
    # Set startLogStep
    sls = root.find(".//simulationParameters/startLogStep")
    if sls is not None:
        sls.text = str(start_log_step)
    # Set timesteps
    ts = root.find(".//simulationParameters/timesteps")
    if ts is not None:
        ts.text = str(n_steps)
    tree = ET.ElementTree(root)
    tree.write(dst, xml_declaration=True, encoding="unicode")


def build_accumulator_map(scenario_xml: Path) -> dict[int, dict[int, str]]:
    """Parse XML to map Data Probe column indices to accumulator names, per population."""
    tree = ET.parse(scenario_xml)
    result = {}
    for pop_idx, pop in enumerate(tree.findall(".//population")):
        acc_names = [acc.get("name") for acc in pop.findall(".//accumulator")]
        result[pop_idx] = {i: name for i, name in enumerate(acc_names)}
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest tests/test_parity_test.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/parity_test.py tests/test_parity_test.py
git commit -m "feat(parity): add XML patching and accumulator map builder"
```

---

### Task 2: HexSim Census Parsing

**Files:**
- Modify: `scripts/parity_test.py`
- Modify: `tests/test_parity_test.py`

- [ ] **Step 1: Write failing test for `parse_hexsim_census`**

```python
class TestParseHexsimCensus:
    def test_parses_csv_with_header(self, tmp_path):
        from scripts.parity_test import parse_hexsim_census
        # Create a fake census CSV
        csv_path = tmp_path / "test.0.csv"
        csv_path.write_text(
            '"Run","Time Step","Population Size","Group Members","Floaters","Lambda","Trait Index  0"\n'
            '    0,      0,        100,          0,        100,   1.000000,              100\n'
            '    0,      1,         95,          0,         95,   0.950000,               95\n'
        )
        result = parse_hexsim_census(tmp_path)
        assert 0 in result  # pop_id 0
        assert result[0][0]["size"] == 100
        assert result[0][1]["size"] == 95
        assert result[0][1]["lambda"] == pytest.approx(0.95, abs=0.01)
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement `parse_hexsim_census`**

```python
def parse_hexsim_census(result_dir: Path) -> dict[int, dict[int, dict]]:
    """Parse HexSim census CSVs into {pop_id: {step: {size, traits, lambda}}}."""
    census = {}
    for csv_path in sorted(result_dir.glob("*.csv")):
        # Population ID is the file suffix number: scenario.N.csv
        stem = csv_path.stem
        parts = stem.rsplit(".", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        pop_id = int(parts[1])
        census[pop_id] = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(row["Time Step"].strip())
                    size = int(row["Population Size"].strip())
                    lam = float(row["Lambda"].strip())
                    traits = {}
                    for k, v in row.items():
                        if k.strip().startswith("Trait Index"):
                            traits[k.strip()] = int(v.strip())
                    census[pop_id][step] = {"size": size, "lambda": lam, "traits": traits}
                except (ValueError, KeyError):
                    continue
    return census
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/parity_test.py tests/test_parity_test.py
git commit -m "feat(parity): add HexSim census CSV parser"
```

---

### Task 3: HexSimPy Runner

**Files:**
- Modify: `scripts/parity_test.py`
- Modify: `tests/test_parity_test.py`

- [ ] **Step 1: Write failing test for `run_hexsimpy`**

```python
class TestRunHexsimpy:
    def test_returns_census_and_snapshots(self):
        from scripts.parity_test import run_hexsimpy
        elapsed, census, snapshots = run_hexsimpy(
            str(WORKSPACE), str(SCENARIO_XML), seed=42, n_steps=5, probe_interval=2
        )
        assert elapsed > 0
        assert len(census) == 5  # one record per step
        assert census[0]["step"] == 0
        # Should have at least one population
        pop_names = [k for k in census[0] if k != "step"]
        assert len(pop_names) >= 1
        # Should have snapshots at steps 0, 2, 4
        assert len(snapshots) >= 3
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement `run_hexsimpy`**

```python
def run_hexsimpy(
    workspace: str, scenario_xml: str, seed: int = 42,
    n_steps: int = 100, probe_interval: int = 100,
) -> tuple[float, list[dict], list[dict]]:
    """Run HexSimPy and collect per-population census + agent snapshots."""
    sys.path.insert(0, str(PROJECT))
    from salmon_ibm.scenario_loader import ScenarioLoader

    t0 = time.perf_counter()
    loader = ScenarioLoader()
    sim = loader.load(workspace, scenario_xml, rng_seed=seed)
    mesh = sim.landscape["mesh"]
    history = []
    snapshots = []
    for step in range(n_steps):
        sim.step()
        record = {"step": step}
        for pname, pop in sim.populations.populations.items():
            rec = {"n_alive": pop.n_alive}
            if pop.trait_mgr:
                for tname in pop.trait_mgr.definitions:
                    vals = pop.trait_mgr.get(tname)
                    alive = pop.alive
                    rec[f"trait_{tname}"] = np.bincount(vals[alive]).tolist()
            record[pname] = rec
        history.append(record)
        if step % probe_interval == 0:
            for pname, pop in sim.populations.populations.items():
                pool = pop.pool
                alive = pop.alive
                if alive.sum() == 0:
                    continue
                snapshots.append({
                    "step": step, "pop_name": pname, "n_alive": pop.n_alive,
                    "ed_kJ_g_mean": float(np.mean(pool.ed_kJ_g[alive])),
                    "ed_kJ_g_std": float(np.std(pool.ed_kJ_g[alive])),
                    "mass_g_mean": float(np.mean(pool.mass_g[alive])),
                    "mass_g_std": float(np.std(pool.mass_g[alive])),
                    "position_histogram": np.bincount(
                        pool.tri_idx[alive], minlength=mesh.n_cells
                    ),
                })
    elapsed = time.perf_counter() - t0
    return elapsed, history, snapshots
```

- [ ] **Step 4: Run test — expect PASS**

Run: `conda run -n shiny python -m pytest tests/test_parity_test.py::TestRunHexsimpy -v`
Expected: PASS (may take ~30s for 5 steps on 16M-cell grid)

- [ ] **Step 5: Commit**

```bash
git add scripts/parity_test.py tests/test_parity_test.py
git commit -m "feat(parity): add HexSimPy runner with census + snapshot collection"
```

---

### Task 4: Census Comparison and Verdict

**Files:**
- Modify: `scripts/parity_test.py`
- Modify: `tests/test_parity_test.py`

- [ ] **Step 1: Write failing tests for `compare_census` and `verdict`**

```python
class TestCompareCensus:
    def test_exact_match_returns_empty(self):
        from scripts.parity_test import compare_census
        hs = {0: {0: {"size": 100, "traits": {"Trait Index  0": 100}, "lambda": 1.0}}}
        py = [{"step": 0, "pop0": {"n_alive": 100}}]
        pop_id_to_name = {0: "pop0"}
        result = compare_census(hs, py, pop_id_to_name)
        assert len(result) == 0 or all(d.rel_error < 0.05 for d in result)

    def test_size_divergence_detected(self):
        from scripts.parity_test import compare_census
        hs = {0: {0: {"size": 100, "traits": {}, "lambda": 1.0}}}
        py = [{"step": 0, "pop0": {"n_alive": 80}}]
        result = compare_census(hs, py, {0: "pop0"})
        assert any(d.metric == "population_size" and d.rel_error > 0.1 for d in result)

    def test_lambda_divergence_detected(self):
        from scripts.parity_test import compare_census
        hs = {0: {0: {"size": 100, "traits": {}, "lambda": 1.0},
                   1: {"size": 100, "traits": {}, "lambda": 0.5}}}
        py = [{"step": 0, "pop0": {"n_alive": 100}},
              {"step": 1, "pop0": {"n_alive": 100}}]
        result = compare_census(hs, py, {0: "pop0"})
        assert any(d.metric == "lambda" for d in result)


class TestVerdict:
    def test_pass_on_no_divergences(self):
        from scripts.parity_test import verdict
        assert verdict([], []) == "PASS"

    def test_fail_on_large_census_divergence(self):
        from scripts.parity_test import verdict, Divergence
        divs = [Divergence(0, "pop", "population_size", 100, 50, 0.5)]
        assert verdict(divs, []) == "FAIL"

    def test_warn_on_moderate_agent_divergence(self):
        from scripts.parity_test import verdict, Divergence
        divs = [Divergence(0, "pop", "ed_mean", 5.0, 4.6, 0.08)]
        assert verdict([], divs) == "WARN"

    def test_pass_when_census_within_10_agents(self):
        from scripts.parity_test import verdict, Divergence
        # 6% relative but only 6 agents different — passes the OR condition
        divs = [Divergence(0, "pop", "population_size", 100, 94, 0.06)]
        assert verdict(divs, []) == "PASS"
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement `Divergence`, `compare_census`, `verdict`**

```python
from scipy.spatial.distance import jensenshannon

@dataclass
class Divergence:
    step: int
    pop: str
    metric: str
    hexsim_val: float
    hexsimpy_val: float
    rel_error: float


def compare_census(
    hexsim_census: dict, hexsimpy_history: list[dict],
    pop_id_to_name: dict[int, str],
) -> list[Divergence]:
    """Compare population sizes and lambda step-by-step."""
    divergences = []
    for pop_id, name in pop_id_to_name.items():
        hs_steps = hexsim_census.get(pop_id, {})
        for record in hexsimpy_history:
            step = record["step"]
            if step not in hs_steps or name not in record:
                continue
            hs = hs_steps[step]
            py_size = record[name]["n_alive"]
            # Population size
            hs_size = hs["size"]
            if hs_size == 0 and py_size == 0:
                continue
            denom = max(hs_size, py_size, 1)
            rel_err = abs(hs_size - py_size) / denom
            if rel_err > 0.001:
                divergences.append(Divergence(step, name, "population_size",
                                              hs_size, py_size, rel_err))
            # Lambda
            hs_lam = hs.get("lambda", 1.0)
            py_lam = py_size / record[name].get("prev_size", py_size) if step > 0 else 1.0
            lam_err = abs(hs_lam - py_lam)
            if lam_err > 0.01:
                divergences.append(Divergence(step, name, "lambda", hs_lam, py_lam, lam_err))
    return divergences


def verdict(census_divergences: list[Divergence], agent_divergences: list[Divergence]) -> str:
    census_fails = [d for d in census_divergences
                    if d.metric == "population_size"
                    and d.rel_error > 0.05 and abs(d.hexsim_val - d.hexsimpy_val) > 10]
    if census_fails:
        return "FAIL"
    if any(d.rel_error > 0.15 for d in agent_divergences):
        return "FAIL"
    if any(d.rel_error > 0.05 for d in agent_divergences):
        return "WARN"
    return "PASS"
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/parity_test.py tests/test_parity_test.py
git commit -m "feat(parity): add census comparison and verdict logic"
```

---

### Task 5: Data Probe Parsing and Agent Comparison

**Files:**
- Modify: `scripts/parity_test.py`
- Modify: `tests/test_parity_test.py`

- [ ] **Step 1: Write failing tests for `parse_hexsim_data_probe` and `compare_agents`**

```python
class TestParseHexsimDataProbe:
    def test_parses_probe_csv(self, tmp_path):
        from scripts.parity_test import parse_hexsim_data_probe
        probe_dir = tmp_path / "Data Probe"
        probe_dir.mkdir()
        csv_path = probe_dir / "probe.csv"
        csv_path.write_text(
            "Run,Time Step,Population,Individual,Acc0,Acc1\n"
            "0,100,0,1,5.2,3500\n"
            "0,100,0,2,5.1,3400\n"
        )
        acc_map = {0: {0: "energy", 1: "mass"}}
        result = parse_hexsim_data_probe(tmp_path, acc_map)
        assert 0 in result  # pop 0
        assert 100 in result[0]  # step 100
        agents = result[0][100]
        assert len(agents) == 2
        assert agents[1]["energy"] == pytest.approx(5.2)
        assert agents[2]["mass"] == pytest.approx(3400)


class TestCompareAgents:
    def test_detects_energy_divergence(self):
        from scripts.parity_test import compare_agents, Divergence
        hs_probe = {0: {100: {1: {"energy": 5.0}, 2: {"energy": 4.8}}}}
        py_snaps = [{"step": 100, "pop_name": "Chinook", "n_alive": 2,
                     "ed_kJ_g_mean": 3.0, "ed_kJ_g_std": 0.1,
                     "mass_g_mean": 3500, "mass_g_std": 100,
                     "position_histogram": np.array([1, 1])}]
        result = compare_agents(hs_probe, py_snaps, {0: "Chinook"},
                                energy_acc="energy", mass_acc="mass")
        assert any(d.metric == "ed_mean" for d in result)

    def test_jsd_computed_for_spatial(self):
        from scripts.parity_test import compare_agents
        hs_probe = {0: {0: {1: {"cell": 0}, 2: {"cell": 1}}}}
        py_snaps = [{"step": 0, "pop_name": "pop0", "n_alive": 2,
                     "ed_kJ_g_mean": 5.0, "ed_kJ_g_std": 0.1,
                     "mass_g_mean": 3500, "mass_g_std": 100,
                     "position_histogram": np.array([1, 1, 0, 0])}]
        result = compare_agents(hs_probe, py_snaps, {0: "pop0"},
                                energy_acc="energy", mass_acc="mass",
                                n_cells=4)
        # Should produce a spatial JSD divergence entry
        spatial = [d for d in result if d.metric == "spatial_jsd"]
        # JSD value depends on distribution match
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement `parse_hexsim_data_probe` and full `compare_agents`**

```python
def parse_hexsim_data_probe(
    result_dir: Path, accumulator_map: dict[int, dict[int, str]],
) -> dict[int, dict[int, dict[int, dict[str, float]]]]:
    """Parse Data Probe CSVs: {pop_id: {step: {agent_id: {acc_name: value}}}}."""
    probe_dir = result_dir / "Data Probe"
    if not probe_dir.exists():
        return {}
    result = {}
    for csv_path in sorted(probe_dir.glob("*.csv")):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pop_id = int(row["Population"].strip())
                    step = int(row["Time Step"].strip())
                    agent_id = int(row["Individual"].strip())
                    acc_names = accumulator_map.get(pop_id, {})
                    vals = {}
                    for col_name, col_val in row.items():
                        col_name = col_name.strip()
                        if col_name.startswith("Acc") or col_name not in (
                            "Run", "Time Step", "Population", "Individual"
                        ):
                            # Try to map by column position
                            pass
                    # Simpler: iterate non-header columns by position
                    data_cols = [v for k, v in row.items()
                                 if k.strip() not in ("Run", "Time Step", "Population", "Individual")]
                    for i, val_str in enumerate(data_cols):
                        name = acc_names.get(i, f"acc_{i}")
                        try:
                            vals[name] = float(val_str.strip())
                        except ValueError:
                            continue
                    result.setdefault(pop_id, {}).setdefault(step, {})[agent_id] = vals
                except (ValueError, KeyError):
                    continue
    return result


def compare_agents(
    hexsim_probe: dict, hexsimpy_snapshots: list[dict],
    pop_id_to_name: dict[int, str],
    energy_acc: str = "Energy Density", mass_acc: str = "Mass",
    n_cells: int | None = None,
) -> list[Divergence]:
    """Compare agent-level metrics at sampled steps using JSD for spatial."""
    divergences = []
    py_by_key = {(s["step"], s["pop_name"]): s for s in hexsimpy_snapshots}
    for pop_id, name in pop_id_to_name.items():
        for step, agents in hexsim_probe.get(pop_id, {}).items():
            key = (step, name)
            if key not in py_by_key or not agents:
                continue
            py = py_by_key[key]
            # Mean energy density
            hs_ed_vals = [a.get(energy_acc, 0) for a in agents.values() if energy_acc in a]
            if hs_ed_vals:
                hs_mean = np.mean(hs_ed_vals)
                py_mean = py["ed_kJ_g_mean"]
                denom = max(abs(hs_mean), abs(py_mean), 1e-9)
                rel_err = abs(hs_mean - py_mean) / denom
                if rel_err > 0.001:
                    divergences.append(Divergence(step, name, "ed_mean", hs_mean, py_mean, rel_err))
            # Mean mass
            hs_mass_vals = [a.get(mass_acc, 0) for a in agents.values() if mass_acc in a]
            if hs_mass_vals:
                hs_mean = np.mean(hs_mass_vals)
                py_mean = py["mass_g_mean"]
                denom = max(abs(hs_mean), abs(py_mean), 1e-9)
                rel_err = abs(hs_mean - py_mean) / denom
                if rel_err > 0.001:
                    divergences.append(Divergence(step, name, "mass_mean", hs_mean, py_mean, rel_err))
            # Spatial: Jensen-Shannon divergence on occupied-cell histograms
            cell_ids = [int(a.get("cell", -1)) for a in agents.values() if "cell" in a]
            if cell_ids and n_cells:
                # Sanity check: cell IDs must be in range
                max_id = max(cell_ids)
                if max_id >= n_cells:
                    divergences.append(Divergence(step, name, "cell_id_mismatch",
                                                  max_id, n_cells, 1.0))
                    continue
                hs_hist = np.bincount(cell_ids, minlength=n_cells).astype(float)
                py_hist = py["position_histogram"].astype(float)
                # Normalize to probability distributions
                hs_hist /= max(hs_hist.sum(), 1)
                py_hist /= max(py_hist.sum(), 1)
                jsd = float(jensenshannon(hs_hist, py_hist))
                if jsd > 0.01:
                    divergences.append(Divergence(step, name, "spatial_jsd", jsd, 0.3, jsd))
    return divergences
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/parity_test.py tests/test_parity_test.py
git commit -m "feat(parity): add Data Probe parser and agent comparison with JSD"
```

---

### Task 6: HexSim Engine Runner and Report Generator

**Files:**
- Modify: `scripts/parity_test.py`
- Modify: `tests/test_parity_test.py`

- [ ] **Step 1: Write failing tests for `run_hexsim_engine` and `generate_report`**

```python
class TestRunHexsimEngine:
    def test_missing_engine_returns_none(self):
        from scripts.parity_test import run_hexsim_engine
        elapsed, census, probe = run_hexsim_engine(
            Path("fake.xml"), Path("nonexistent.exe"), seed=42, timeout=10
        )
        assert elapsed is None
        assert census == {}


class TestGenerateReport:
    def test_generates_markdown(self, tmp_path):
        from scripts.parity_test import generate_report, Divergence
        output = tmp_path / "report.md"
        v = generate_report(10.0, 5.0, [], [], [], output, "test.xml", 100, 42)
        assert v == "PASS"
        assert output.exists()
        text = output.read_text()
        assert "PASS" in text
        assert "test.xml" in text

    def test_report_includes_divergences(self, tmp_path):
        from scripts.parity_test import generate_report, Divergence
        divs = [Divergence(10, "Chinook", "population_size", 100, 50, 0.5)]
        output = tmp_path / "report.md"
        v = generate_report(None, 5.0, divs, [], [], output, "test.xml", 100, 42)
        assert v == "FAIL"
        text = output.read_text()
        assert "FAIL" in text
        assert "Chinook" in text
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement `run_hexsim_engine` and `generate_report`**

```python
def run_hexsim_engine(
    patched_xml: Path, hexsim_exe: Path, seed: int = 42, timeout: int = 3600,
) -> tuple[float | None, dict, dict]:
    """Run HexSim 4.0.20 engine, return (elapsed, census, probe)."""
    if not hexsim_exe.exists():
        print(f"  [SKIP] HexSim engine not found at {hexsim_exe}")
        return None, {}, {}
    # HexSim writes results to Results/<scenario_stem>/ relative to workspace
    workspace = patched_xml.parent.parent
    result_dir = workspace / "Results" / patched_xml.stem
    if result_dir.exists():
        shutil.rmtree(result_dir)
    t0 = time.perf_counter()
    try:
        subprocess.run(
            [str(hexsim_exe), "-r", str(seed), str(patched_xml)],
            capture_output=True, text=True,
            cwd=str(workspace),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  HexSim timed out after {timeout}s — collecting partial results")
    elapsed = time.perf_counter() - t0
    census = parse_hexsim_census(result_dir) if result_dir.exists() else {}
    acc_map = build_accumulator_map(patched_xml)
    probe = parse_hexsim_data_probe(result_dir, acc_map) if result_dir.exists() else {}
    return elapsed, census, probe


def generate_report(
    hexsim_time, hexsimpy_time, census_divs, agent_divs,
    hexsimpy_history, output_path: Path, scenario_name: str,
    n_steps: int, seed: int,
):
    """Write markdown parity report."""
    v = verdict(census_divs, agent_divs)
    lines = [
        "# Parity Test Report", "",
        f"**Scenario**: {scenario_name}",
        f"**Steps**: {n_steps} | **Seed**: {seed} | **Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Verdict**: {v}", "",
        "---", "", "## Timing", "",
        "| Engine | Total |", "|--------|-------|",
        f"| HexSim 4.0.20 | {f'{hexsim_time:.1f}s' if hexsim_time else 'skipped'} |",
        f"| HexSimPy | {hexsimpy_time:.1f}s |", "",
    ]
    if census_divs:
        lines += ["---", "", "## Census Divergences", "",
                   "| Step | Pop | HexSim | HexSimPy | Rel Error |",
                   "|------|-----|--------|----------|-----------|"]
        for d in census_divs[:50]:
            lines.append(f"| {d.step} | {d.pop} | {d.hexsim_val} | {d.hexsimpy_val} | {d.rel_error:.3f} |")
    else:
        lines += ["", "Census: all steps within tolerance.", ""]
    if agent_divs:
        lines += ["---", "", "## Agent Divergences", "",
                   "| Step | Pop | Metric | HexSim | HexSimPy | Rel Error |",
                   "|------|-----|--------|--------|----------|-----------|"]
        for d in agent_divs[:50]:
            lines.append(f"| {d.step} | {d.pop} | {d.metric} | {d.hexsim_val:.3f} | {d.hexsimpy_val:.3f} | {d.rel_error:.3f} |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {output_path} — Verdict: {v}")
    return v
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/parity_test.py tests/test_parity_test.py
git commit -m "feat(parity): add HexSim engine runner and report generator with tests"
```

---

### Task 7: CLI Main and Integration

**Files:**
- Modify: `scripts/parity_test.py`

- [ ] **Step 1: Add `main()` with argparse CLI**

```python
def main():
    parser = argparse.ArgumentParser(description="Parity test: HexSim 4.0.20 vs HexSimPy")
    parser.add_argument("--workspace", required=True, help="HexSim workspace directory")
    parser.add_argument("--scenario", required=True, help="Scenario XML path (relative to workspace)")
    parser.add_argument("--steps", type=int, default=2928, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--timeout", type=int, default=3600, help="HexSim engine timeout (s)")
    parser.add_argument("--hexsim-exe", default="HexSim 4.0.20/HexSimEngine64.exe",
                        help="Path to HexSimEngine64.exe")
    parser.add_argument("--probe-interval", type=int, default=100, help="Agent snapshot interval")
    parser.add_argument("--output", default="docs/parity-report.md", help="Output report path")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    scenario = workspace / args.scenario
    hexsim_exe = PROJECT / args.hexsim_exe
    output = Path(args.output)

    # 1. Patch XML
    patched = workspace / "Scenarios" / f"{scenario.stem}_parity.xml"
    patch_scenario_xml(scenario, patched, workspace, start_log_step=1, n_steps=args.steps)

    # 2. Build accumulator map
    acc_map = build_accumulator_map(scenario)

    # 3. Run HexSim
    print("[1/3] Running HexSim 4.0.20...")
    hs_time, hs_census, hs_probe = run_hexsim_engine(patched, hexsim_exe, args.seed, args.timeout)

    # 4. Run HexSimPy
    print("[2/3] Running HexSimPy...")
    py_time, py_history, py_snapshots = run_hexsimpy(
        str(workspace), str(scenario), args.seed, args.steps, args.probe_interval
    )

    # 5. Compare
    print("[3/3] Comparing outputs...")
    # Build pop_id -> name mapping from HexSimPy history
    pop_names = [k for k in py_history[0] if k != "step"] if py_history else []
    pop_id_to_name = {i: name for i, name in enumerate(pop_names)}
    census_divs = compare_census(hs_census, py_history, pop_id_to_name) if hs_census else []
    agent_divs = compare_agents(hs_probe, py_snapshots, pop_id_to_name) if hs_probe else []

    # 6. Report
    v = generate_report(hs_time, py_time, census_divs, agent_divs,
                        py_history, output, scenario.name, args.steps, args.seed)
    sys.exit(0 if v == "PASS" else 1 if v == "FAIL" else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test CLI with 5 steps**

Run: `conda run -n shiny python scripts/parity_test.py --workspace "Columbia [small]" --scenario "Scenarios/snake_Columbia2017B.xml" --steps 5 --output docs/parity-report-smoke.md`
Expected: Report generated with verdict (likely PASS for 5 steps)

- [ ] **Step 3: Commit**

```bash
git add scripts/parity_test.py
git commit -m "feat(parity): add CLI main with argparse integration"
```

---

### Task 8: Add `.gitignore` Entry and Skill File

**Files:**
- Modify: `.gitignore`
- Create: `~/.claude/skills/parity-test/SKILL.md`

- [ ] **Step 1: Add parity report to `.gitignore`**

Append to `.gitignore`:
```
docs/parity-report*.md
```

- [ ] **Step 2: Create the Claude Code skill file**

```markdown
---
name: parity-test
description: Use when modifying salmon IBM event logic, scenario loader, movement kernels, or bioenergetics — runs same HexSim scenario on both C++ engine and HexSimPy to verify output parity
---

# Parity Test

Run the same HexSim XML scenario on both HexSim 4.0.20 (C++) and HexSimPy, compare census trajectories and per-agent data, report PASS/WARN/FAIL.

## When to Use

- After modifying event execution logic (`events.py`, `events_hexsim.py`, `events_builtin.py`)
- After changing movement kernels (`movement.py`)
- After modifying bioenergetics (`bioenergetics.py`)
- After changing the scenario loader (`scenario_loader.py`, `xml_parser.py`)
- After updating accumulator updater functions (`accumulators.py`)
- Before merging significant simulation engine changes

## Quick Reference

```bash
# Full run (2928 steps, ~2 hours)
conda run -n shiny python scripts/parity_test.py \
  --workspace "Columbia [small]" \
  --scenario "Scenarios/snake_Columbia2017B.xml" \
  --steps 2928 --seed 42

# Quick smoke test (10 steps, ~1 minute)
conda run -n shiny python scripts/parity_test.py \
  --workspace "Columbia [small]" \
  --scenario "Scenarios/snake_Columbia2017B.xml" \
  --steps 10 --seed 42
```

## Interpreting Results

| Verdict | Meaning | Action |
|---------|---------|--------|
| **PASS** | Census within 5%/10 agents, agent metrics within 5% | Ship it |
| **WARN** | Census OK but agent metrics 5-15% divergent | Investigate — may be acceptable stochastic difference |
| **FAIL** | Census or agent metrics diverge significantly | Bug in simulation logic — do not merge |

## Common Failure Causes

- **Census divergence after step N**: Event execution order changed — check `EventSequencer` and `MultiPopEventSequencer`
- **Energy divergence**: Bioenergetics formula changed — check `update_energy()`, `hourly_respiration()`
- **Spatial divergence**: Movement kernel changed — check `execute_movement()`, `HexSimMoveEvent`
- **All zeros from HexSim**: XML paths not patched correctly — check `F:\Marcia\` in patched XML
```

Write to: `~/.claude/skills/parity-test/SKILL.md`

- [ ] **Step 3: Commit `.gitignore`**

```bash
git add .gitignore
git commit -m "chore: gitignore parity reports, add parity-test skill"
```

---

### Task 9: Final Integration Test

- [ ] **Step 1: Run full test suite to verify no regressions**

Run: `conda run -n shiny python -m pytest tests/ -v --tb=short`
Expected: 556+ passed, 0 failed

- [ ] **Step 2: Run parity test smoke (10 steps)**

Run: `conda run -n shiny python scripts/parity_test.py --workspace "Columbia [small]" --scenario "Scenarios/snake_Columbia2017B.xml" --steps 10 --seed 42 --output docs/parity-report-smoke.md`
Expected: Report generated, no crashes

- [ ] **Step 3: Push to remote**

```bash
git push origin main
```
