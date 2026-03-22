"""Tests for scripts/parity_test.py — parity test between HexSim C++ and HexSimPy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: F401
import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

PROJECT = Path(__file__).resolve().parent.parent
WORKSPACE = PROJECT / "Columbia [small]"
SCENARIO_XML = WORKSPACE / "Scenarios" / "snake_Columbia2017B.xml"

# ---------------------------------------------------------------------------
# TASK 1 tests: patch_scenario_xml, build_accumulator_map
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_xml(tmp_path):
    """Patch the real scenario XML into tmp_path and return path."""
    from parity_test import patch_scenario_xml

    dst = tmp_path / "patched.xml"
    patch_scenario_xml(
        str(SCENARIO_XML), str(dst), str(WORKSPACE), start_log_step=1, n_steps=10
    )
    return dst


@pytest.mark.skipif(not SCENARIO_XML.exists(), reason="Columbia workspace not present")
class TestPatchScenarioXml:
    def test_patched_file_exists(self, patched_xml):
        assert patched_xml.exists()

    def test_old_paths_removed(self, patched_xml):
        text = patched_xml.read_text(encoding="utf-8")
        assert r"F:\Marcia" not in text

    def test_local_paths_inserted(self, patched_xml):
        text = patched_xml.read_text(encoding="utf-8")
        assert str(WORKSPACE) in text or str(WORKSPACE).replace("\\", "/") in text

    def test_timesteps_overridden(self, patched_xml):
        import xml.etree.ElementTree as ET

        tree = ET.parse(str(patched_xml))
        root = tree.getroot()
        ts = root.find(".//timesteps")
        assert ts is not None and ts.text == "10"

    def test_start_log_step_overridden(self, patched_xml):
        import xml.etree.ElementTree as ET

        tree = ET.parse(str(patched_xml))
        root = tree.getroot()
        sls = root.find(".//startLogStep")
        assert sls is not None and sls.text == "1"


@pytest.mark.skipif(not SCENARIO_XML.exists(), reason="Columbia workspace not present")
class TestBuildAccumulatorMap:
    def test_returns_dict(self):
        from parity_test import build_accumulator_map

        result = build_accumulator_map(str(SCENARIO_XML))
        assert isinstance(result, dict)

    def test_has_population_indices(self):
        from parity_test import build_accumulator_map

        result = build_accumulator_map(str(SCENARIO_XML))
        # Scenario has 4 populations: Chinook(0), Iterator(1), Refuges(2), Steelhead(3)
        assert 0 in result

    def test_accumulator_names_in_order(self):
        from parity_test import build_accumulator_map

        result = build_accumulator_map(str(SCENARIO_XML))
        pop0 = result[0]
        # First accumulator of Chinook is "Affinity Bound [ max ]"
        assert pop0[0] == "Affinity Bound [ max ]"

    def test_col_indices_are_sequential(self):
        from parity_test import build_accumulator_map

        result = build_accumulator_map(str(SCENARIO_XML))
        for pop_idx, accum_map in result.items():
            indices = sorted(accum_map.keys())
            assert indices == list(range(len(indices)))


# ---------------------------------------------------------------------------
# TASK 2 tests: parse_hexsim_census
# ---------------------------------------------------------------------------


class TestParseHexsimCensus:
    def _write_csv(self, path, rows):
        import csv

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)

    def test_basic_parsing(self, tmp_path):
        from parity_test import parse_hexsim_census

        header = [
            "Run",
            "Time Step",
            "Population Size",
            "Group Members",
            "Floaters",
            "Lambda",
            "Trait Index  0",
        ]
        rows = [header, ["0", "0", "100", "0", "100", "1.000000", "100"]]
        self._write_csv(tmp_path / "scenario.0.csv", rows)
        result = parse_hexsim_census(str(tmp_path))
        assert 0 in result
        assert 0 in result[0]
        assert result[0][0]["size"] == 100
        assert result[0][0]["lambda"] == pytest.approx(1.0)

    def test_multiple_steps(self, tmp_path):
        from parity_test import parse_hexsim_census

        header = [
            "Run",
            "Time Step",
            "Population Size",
            "Group Members",
            "Floaters",
            "Lambda",
        ]
        rows = [
            header,
            ["0", "0", "100", "0", "100", "1.0"],
            ["0", "1", "110", "0", "110", "1.1"],
        ]
        self._write_csv(tmp_path / "test.0.csv", rows)
        result = parse_hexsim_census(str(tmp_path))
        assert result[0][1]["size"] == 110
        assert result[0][1]["lambda"] == pytest.approx(1.1)

    def test_multiple_populations(self, tmp_path):
        from parity_test import parse_hexsim_census

        header = [
            "Run",
            "Time Step",
            "Population Size",
            "Group Members",
            "Floaters",
            "Lambda",
        ]
        self._write_csv(
            tmp_path / "test.0.csv", [header, ["0", "0", "50", "0", "50", "1.0"]]
        )
        self._write_csv(
            tmp_path / "test.3.csv", [header, ["0", "0", "200", "0", "200", "1.0"]]
        )
        result = parse_hexsim_census(str(tmp_path))
        assert 0 in result and 3 in result
        assert result[0][0]["size"] == 50
        assert result[3][0]["size"] == 200

    def test_trait_columns_parsed(self, tmp_path):
        from parity_test import parse_hexsim_census

        header = [
            "Run",
            "Time Step",
            "Population Size",
            "Group Members",
            "Floaters",
            "Lambda",
            "Trait Index  0",
            "Trait Index  1",
        ]
        rows = [header, ["0", "5", "100", "0", "100", "1.0", "60", "40"]]
        self._write_csv(tmp_path / "s.2.csv", rows)
        result = parse_hexsim_census(str(tmp_path))
        assert result[2][5]["traits"] == {0: 60, 1: 40}

    def test_ignores_non_csv(self, tmp_path):
        from parity_test import parse_hexsim_census

        (tmp_path / "readme.txt").write_text("not a csv")
        result = parse_hexsim_census(str(tmp_path))
        assert result == {}


# ---------------------------------------------------------------------------
# TASK 3 tests: run_hexsimpy
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SCENARIO_XML.exists(), reason="Columbia workspace not present")
class TestRunHexsimpy:
    @pytest.fixture(scope="class")
    def patched_scenario(self, tmp_path_factory):
        """Patch scenario XML with local workspace paths."""
        from parity_test import patch_scenario_xml

        tmp = tmp_path_factory.mktemp("scenario")
        dst = tmp / "patched.xml"
        patch_scenario_xml(
            str(SCENARIO_XML), str(dst), str(WORKSPACE), start_log_step=1, n_steps=5
        )
        return dst

    @pytest.fixture(scope="class")
    def run_result(self, patched_scenario):
        """Run HexSimPy for 5 steps and cache result for the class."""
        from parity_test import run_hexsimpy

        return run_hexsimpy(
            str(WORKSPACE), str(patched_scenario), seed=42, n_steps=5, probe_interval=3
        )

    def test_returns_tuple_of_three(self, run_result):
        assert len(run_result) == 3

    def test_elapsed_positive(self, run_result):
        elapsed, _, _ = run_result
        assert elapsed > 0

    def test_census_has_entries(self, run_result):
        _, census, _ = run_result
        assert len(census) > 0

    def test_census_entry_has_required_keys(self, run_result):
        _, census, _ = run_result
        entry = census[0]
        assert "pop_name" in entry
        assert "n_alive" in entry
        assert "step" in entry

    def test_snapshots_collected(self, run_result):
        """Snapshots should be collected at probe_interval steps."""
        _, _, snapshots = run_result
        # With 5 steps and probe_interval=3, expect snapshots at step 3
        assert len(snapshots) >= 1


# ---------------------------------------------------------------------------
# TASK 4 tests: Divergence, compare_census, verdict
# ---------------------------------------------------------------------------


class TestDivergence:
    def test_dataclass_fields(self):
        from parity_test import Divergence

        d = Divergence(
            step=0,
            pop="Chinook",
            metric="size",
            hexsim_val=100.0,
            hexsimpy_val=105.0,
            rel_error=0.05,
        )
        assert d.step == 0
        assert d.pop == "Chinook"
        assert d.rel_error == pytest.approx(0.05)


class TestCompareCensus:
    def test_exact_match_returns_empty(self):
        from parity_test import compare_census

        # HexSim census: {pop_id: {step: {size, lambda, traits}}}
        hexsim = {
            0: {
                0: {"size": 100, "lambda": 1.0, "traits": {}},
                1: {"size": 110, "lambda": 1.1, "traits": {}},
            }
        }
        # HexSimPy census: list of dicts with step, pop_name, n_alive
        hexsimpy = [
            {"step": 0, "pop_name": "Chinook", "n_alive": 100},
            {"step": 1, "pop_name": "Chinook", "n_alive": 110},
        ]
        pop_id_to_name = {0: "Chinook"}
        result = compare_census(hexsim, hexsimpy, pop_id_to_name)
        assert result == []

    def test_divergence_detected(self):
        from parity_test import compare_census

        hexsim = {
            0: {
                0: {"size": 100, "lambda": 1.0, "traits": {}},
                1: {"size": 200, "lambda": 2.0, "traits": {}},
            }
        }
        hexsimpy = [
            {"step": 0, "pop_name": "Chinook", "n_alive": 100},
            {"step": 1, "pop_name": "Chinook", "n_alive": 150},
        ]
        pop_id_to_name = {0: "Chinook"}
        result = compare_census(hexsim, hexsimpy, pop_id_to_name)
        assert len(result) > 0
        # Should detect size divergence at step 1
        size_divs = [d for d in result if d.metric == "size" and d.step == 1]
        assert len(size_divs) == 1
        assert size_divs[0].hexsim_val == 200.0
        assert size_divs[0].hexsimpy_val == 150.0

    def test_lambda_divergence(self):
        from parity_test import compare_census

        hexsim = {
            0: {
                0: {"size": 100, "lambda": 1.0, "traits": {}},
                1: {"size": 100, "lambda": 1.0, "traits": {}},
                2: {"size": 100, "lambda": 1.0, "traits": {}},
            }
        }
        # HexSimPy: step 0 -> 100, step 1 -> 100 (lambda=1.0), step 2 -> 50 (lambda=0.5)
        hexsimpy = [
            {"step": 0, "pop_name": "Chinook", "n_alive": 100},
            {"step": 1, "pop_name": "Chinook", "n_alive": 100},
            {"step": 2, "pop_name": "Chinook", "n_alive": 50},
        ]
        pop_id_to_name = {0: "Chinook"}
        result = compare_census(hexsim, hexsimpy, pop_id_to_name)
        lambda_divs = [d for d in result if d.metric == "lambda"]
        assert len(lambda_divs) > 0


class TestVerdict:
    def test_pass_no_divergences(self):
        from parity_test import verdict

        assert verdict([], []) == "PASS"

    def test_fail_large_pop_divergence(self):
        from parity_test import Divergence, verdict

        divs = [
            Divergence(
                step=1,
                pop="Chinook",
                metric="size",
                hexsim_val=200.0,
                hexsimpy_val=150.0,
                rel_error=0.25,
            )
        ]
        assert verdict(divs, []) == "FAIL"

    def test_warn_agent_metric(self):
        from parity_test import Divergence, verdict

        agent_divs = [
            Divergence(
                step=1,
                pop="Chinook",
                metric="ed_kJ_g",
                hexsim_val=10.0,
                hexsimpy_val=9.0,
                rel_error=0.10,
            )
        ]
        assert verdict([], agent_divs) == "WARN"

    def test_fail_agent_metric_over_15pct(self):
        from parity_test import Divergence, verdict

        agent_divs = [
            Divergence(
                step=1,
                pop="Chinook",
                metric="ed_kJ_g",
                hexsim_val=10.0,
                hexsimpy_val=8.0,
                rel_error=0.20,
            )
        ]
        assert verdict([], agent_divs) == "FAIL"

    def test_pass_small_pop_divergence_under_threshold(self):
        """Pop size <5% relative error should pass even with >10 agent diff."""
        from parity_test import Divergence, verdict

        divs = [
            Divergence(
                step=1,
                pop="Chinook",
                metric="size",
                hexsim_val=1000.0,
                hexsimpy_val=980.0,
                rel_error=0.02,
            )
        ]
        assert verdict(divs, []) == "PASS"


# ---------------------------------------------------------------------------
# TASK 5 tests: parse_hexsim_data_probe, compare_agents
# ---------------------------------------------------------------------------


class TestParseHexsimDataProbe:
    def _write_probe_csv(self, probe_dir, filename, rows):
        import csv

        probe_dir.mkdir(parents=True, exist_ok=True)
        with open(probe_dir / filename, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)

    def test_basic_parsing(self, tmp_path):
        from parity_test import parse_hexsim_data_probe

        header = ["Run", "Time Step", "Population", "Individual", "Acc0", "Acc1"]
        rows = [
            header,
            ["0", "5", "0", "100", "3.5", "10.0"],
            ["0", "5", "0", "101", "4.0", "12.0"],
        ]
        probe_dir = tmp_path / "Data Probe"
        self._write_probe_csv(probe_dir, "probe.csv", rows)

        acc_map = {0: {0: "Energy Density", 1: "Mass"}}
        result = parse_hexsim_data_probe(str(tmp_path), acc_map)

        assert 0 in result
        assert 5 in result[0]
        assert 100 in result[0][5]
        assert result[0][5][100]["Energy Density"] == pytest.approx(3.5)
        assert result[0][5][101]["Mass"] == pytest.approx(12.0)

    def test_empty_dir_returns_empty(self, tmp_path):
        from parity_test import parse_hexsim_data_probe

        result = parse_hexsim_data_probe(str(tmp_path), {})
        assert result == {}

    def test_no_data_probe_dir_returns_empty(self, tmp_path):
        from parity_test import parse_hexsim_data_probe

        result = parse_hexsim_data_probe(str(tmp_path / "nonexistent"), {})
        assert result == {}

    def test_unnamed_accumulators(self, tmp_path):
        from parity_test import parse_hexsim_data_probe

        header = ["Run", "Time Step", "Population", "Individual", "Acc0"]
        rows = [header, ["0", "0", "1", "0", "99.0"]]
        probe_dir = tmp_path / "Data Probe"
        self._write_probe_csv(probe_dir, "probe.csv", rows)

        # No accumulator map for pop 1 — should fall back to Acc0
        result = parse_hexsim_data_probe(str(tmp_path), {})
        assert result[1][0][0]["Acc0"] == pytest.approx(99.0)

    def test_multiple_steps(self, tmp_path):
        from parity_test import parse_hexsim_data_probe

        header = ["Run", "Time Step", "Population", "Individual", "Acc0"]
        rows = [
            header,
            ["0", "0", "0", "1", "1.0"],
            ["0", "5", "0", "1", "2.0"],
        ]
        probe_dir = tmp_path / "Data Probe"
        self._write_probe_csv(probe_dir, "probe.csv", rows)

        result = parse_hexsim_data_probe(str(tmp_path), {0: {0: "Energy"}})
        assert 0 in result[0]
        assert 5 in result[0]
        assert result[0][0][1]["Energy"] == pytest.approx(1.0)
        assert result[0][5][1]["Energy"] == pytest.approx(2.0)


class TestCompareAgents:
    def test_detects_energy_divergence(self):
        from parity_test import compare_agents

        hexsim_probe = {
            0: {
                10: {
                    1: {"Energy Density": 10.0, "Mass": 50.0},
                    2: {"Energy Density": 12.0, "Mass": 55.0},
                }
            }
        }
        snapshots = [
            {
                "step": 10,
                "pop_name": "Chinook",
                "n_alive": 2,
                "ed_kJ_g": np.array([5.0, 6.0]),  # much lower than HexSim
                "mass_g": np.array([50.0, 55.0]),
                "tri_idx": np.array([0, 1]),
                "n_cells": 100,
            }
        ]
        pop_id_to_name = {0: "Chinook"}
        result = compare_agents(hexsim_probe, snapshots, pop_id_to_name)
        ed_divs = [d for d in result if d.metric == "mean_ed"]
        assert len(ed_divs) == 1
        assert ed_divs[0].rel_error > 0.4  # ~50% divergence

    def test_exact_match_no_divergence(self):
        from parity_test import compare_agents

        hexsim_probe = {
            0: {
                10: {
                    1: {"Energy Density": 10.0, "Mass": 50.0},
                }
            }
        }
        snapshots = [
            {
                "step": 10,
                "pop_name": "Chinook",
                "n_alive": 1,
                "ed_kJ_g": np.array([10.0]),
                "mass_g": np.array([50.0]),
                "tri_idx": np.array([0]),
                "n_cells": 100,
            }
        ]
        pop_id_to_name = {0: "Chinook"}
        result = compare_agents(hexsim_probe, snapshots, pop_id_to_name)
        assert result == []

    def test_mass_divergence(self):
        from parity_test import compare_agents

        hexsim_probe = {0: {5: {1: {"Energy Density": 10.0, "Mass": 100.0}}}}
        snapshots = [
            {
                "step": 5,
                "pop_name": "Chinook",
                "n_alive": 1,
                "ed_kJ_g": np.array([10.0]),
                "mass_g": np.array([50.0]),  # 50% off
                "tri_idx": np.array([0]),
                "n_cells": 10,
            }
        ]
        result = compare_agents(hexsim_probe, snapshots, {0: "Chinook"})
        mass_divs = [d for d in result if d.metric == "mean_mass"]
        assert len(mass_divs) == 1
        assert mass_divs[0].rel_error == pytest.approx(0.5)
