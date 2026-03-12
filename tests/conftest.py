"""Shared fixtures and marks for salmon_ibm tests."""
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GRID_FILE = DATA_DIR / "curonian_minimal_grid.nc"
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config_curonian_minimal.yaml"

has_data = pytest.mark.skipif(
    not GRID_FILE.exists(), reason="curonian_minimal_grid.nc not found"
)


@pytest.fixture
def mesh():
    """Load the Curonian minimal mesh (skips if data missing)."""
    if not GRID_FILE.exists():
        pytest.skip("curonian_minimal_grid.nc not found")
    from salmon_ibm.mesh import TriMesh
    return TriMesh.from_netcdf(str(GRID_FILE))


@pytest.fixture
def curonian_config():
    """Load the minimal Curonian config (skips if data missing)."""
    if not CONFIG_FILE.exists():
        pytest.skip("config_curonian_minimal.yaml not found")
    from salmon_ibm.config import load_config
    return load_config(str(CONFIG_FILE))
