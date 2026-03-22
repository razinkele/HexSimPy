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
