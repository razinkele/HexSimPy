"""Tests for scripts/parity_test.py — parity test between HexSim C++ and HexSimPy."""

from __future__ import annotations

import sys
from pathlib import Path

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
