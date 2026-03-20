"""Tests for I/O validation error messages."""
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from heximpy.hxnparser import GridMeta, HexMap


class TestHxnReadValidation:
    """HexMap.from_file() should reject corrupt files."""

    def test_truncated_data_raises(self):
        """File with fewer values than header claims should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            with open(tmp, "wb") as f:
                f.write(struct.pack("<i", 3))   # version
                f.write(struct.pack("<i", 10))  # ncols (width)
                f.write(struct.pack("<i", 10))  # nrows (height)
                f.write(struct.pack("<d", 1.0)) # cell_size
                f.write(struct.pack("<d", 0.0)) # origin_x
                f.write(struct.pack("<d", 0.0)) # origin_y
                f.write(struct.pack("<i", 1))   # dtype_code (float32)
                f.write(struct.pack("<i", 0))   # nodata
                # Only write 50 values instead of 100
                f.write(np.zeros(50, dtype=np.float32).tobytes())

            with pytest.raises(ValueError, match="mismatch|truncated|expected"):
                HexMap.from_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_invalid_dtype_code_raises(self):
        """dtype_code other than 1 or 2 should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            with open(tmp, "wb") as f:
                f.write(struct.pack("<i", 3))
                f.write(struct.pack("<i", 5))
                f.write(struct.pack("<i", 5))
                f.write(struct.pack("<d", 1.0))
                f.write(struct.pack("<d", 0.0))
                f.write(struct.pack("<d", 0.0))
                f.write(struct.pack("<i", 99))  # invalid dtype_code
                f.write(struct.pack("<i", 0))
                f.write(np.zeros(25, dtype=np.float32).tobytes())

            with pytest.raises(ValueError, match="dtype"):
                HexMap.from_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)


class TestHxnWriteValidation:
    """HexMap.to_file() should reject mismatched data."""

    def test_write_wrong_length_plain_raises(self):
        """Writing plain HexMap with values length != n_hexagons should raise."""
        hm = HexMap(
            format="plain", version=3, height=10, width=10,
            flag=0, max_val=1.0, min_val=0.0, hexzero=0.0,
            values=np.zeros(50, dtype=np.float32),
        )
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="data length"):
                hm.to_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_write_wrong_length_patch_hexmap_raises(self):
        """Writing PATCH_HEXMAP with values length != n_hexagons should raise."""
        hm = HexMap(
            format="patch_hexmap", version=1, height=10, width=10,
            flag=0, max_val=1.0, min_val=0.0, hexzero=0.0,
            values=np.zeros(50, dtype=np.float32),
        )
        with tempfile.NamedTemporaryFile(suffix=".hxn", delete=False) as f:
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="data length"):
                hm.to_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)


class TestGridMetaValidation:
    """GridMeta.from_file() should reject implausible values."""

    def test_zero_extent_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".grid", delete=False) as f:
            tmp = Path(f.name)
        try:
            with open(tmp, "wb") as f:
                f.write(b"PATCH_GRID")
                f.write(struct.pack("<I", 1))    # version
                f.write(struct.pack("<I", 100))  # n_hexes
                f.write(struct.pack("<I", 10))   # ncols
                f.write(struct.pack("<I", 10))   # nrows
                f.write(struct.pack("<?", False))  # flag
                f.write(struct.pack("<d", 0.0))  # x_extent = 0 (invalid)
                f.write(struct.pack("<d", 0.0))
                f.write(struct.pack("<d", 1000.0))
                f.write(struct.pack("<d", 0.0))
                f.write(struct.pack("<d", 24.0)) # row_spacing

            with pytest.raises(ValueError, match="extent|plausib"):
                GridMeta.from_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)


class TestTemperatureCSVValidation:
    """HexSimEnvironment should validate temperature CSV shape."""

    def test_wrong_shape_raises(self):
        from salmon_ibm.hexsim_env import _validate_temp_table
        data = np.zeros((3, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="Temperature CSV shape"):
            _validate_temp_table(data, n_zones=5)


import warnings


class TestXmlParserValidation:
    """XML parser should check for required elements."""

    def test_missing_simulation_params_raises(self):
        from salmon_ibm.xml_parser import load_scenario_xml
        xml_str = "<scenario><hexagonGrid/></scenario>"
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_str)
            tmp = Path(f.name)
        try:
            with pytest.raises(ValueError, match="simulationParameters"):
                load_scenario_xml(tmp)
        finally:
            tmp.unlink(missing_ok=True)


class TestEventLoadingWarning:
    """Unregistered event types should emit a warning."""

    def test_unregistered_event_warns(self):
        from salmon_ibm.scenario_loader import ScenarioLoader

        loader = ScenarioLoader.__new__(ScenarioLoader)
        edef = {
            "type": "reanimation",
            "name": "Test Reanimation",
            "timestep": 1,
            "population": "pop1",
            "enabled": True,
            "params": {},
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evt = loader._build_single_event(edef)
            matching = [x for x in w if "not in EVENT_REGISTRY" in str(x.message)]
            assert len(matching) == 1, f"Expected 1 warning, got {len(matching)}"
            assert "reanimation" in str(matching[0].message)
