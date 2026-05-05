"""Tests for hatchery vs wild C3.3 — homing precision divergence.

Spec: docs/superpowers/specs/2026-05-03-hatchery-c3.3-homing-design.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from salmon_ibm.baltic_params import (
    BalticBioParams,
    BalticSpeciesConfig,
    _apply_hatchery_overrides,
    load_baltic_species_config,
)


CONFIG_PATH = Path("configs/baltic_salmon_species.yaml")


def test_homing_precision_default_loads():
    """C3.3 test 1 (partial): default BalticBioParams has wild
    homing_precision = 0.95."""
    p = BalticBioParams()
    assert p.homing_precision == 0.95


def test_homing_precision_validation_rejects_out_of_range():
    """C3.3 test 2: __post_init__ raises ValueError on -0.1 and 1.5
    with a message naming the field."""
    with pytest.raises(ValueError, match=r"homing_precision"):
        BalticBioParams(homing_precision=-0.1)
    with pytest.raises(ValueError, match=r"homing_precision"):
        BalticBioParams(homing_precision=1.5)
    # Boundaries 0.0 and 1.0 are valid.
    BalticBioParams(homing_precision=0.0)
    BalticBioParams(homing_precision=1.0)


def test_homing_precision_in_scalar_override_fields():
    """C3.3 test 3: hatchery override flows through `dataclasses.replace`
    via the existing SCALAR_OVERRIDE_FIELDS mechanism."""
    wild = BalticBioParams()
    overrides = {"homing_precision": 0.65}
    hatchery = _apply_hatchery_overrides(wild, overrides)
    assert hatchery.homing_precision == 0.65
    # Wild unchanged:
    assert wild.homing_precision == 0.95


def test_homing_precision_loads_from_yaml():
    """C3.3 test 1 (full): deployed YAML has wild=0.95 + hatchery=0.65."""
    cfg = load_baltic_species_config(CONFIG_PATH)
    assert isinstance(cfg, BalticSpeciesConfig)
    assert cfg.wild.homing_precision == 0.95
    assert cfg.hatchery is not None
    assert cfg.hatchery.homing_precision == 0.65
    assert cfg.hatchery is not cfg.wild
