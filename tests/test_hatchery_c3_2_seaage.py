"""Tests for hatchery vs wild C3.2 — sea-age sampling.

Spec: docs/superpowers/specs/2026-05-03-hatchery-c3.2-seaage-design.md
"""

from __future__ import annotations

import numpy as np
import pytest

from salmon_ibm.agents import AgentPool
from salmon_ibm.sea_age import (
    SEA_AGE_1SW,
    SEA_AGE_2SW,
    SEA_AGE_3SW,
    SEA_AGE_NAMES,
    SEA_AGE_UNSET,
    SeaAge,
    VALID_SEA_AGES,
)


def test_sea_age_module_constants():
    """C3.2: smoke test for sea_age module — IntEnum + aliases + names tuple
    mirror C1 origin.py structure. Locks the SEA_AGE_NAMES indexing
    contract (NAMES[value+1] returns the human-readable name) which
    OutputLogger / YAML reflective serialization may rely on."""
    # IntEnum values + aliases consistent with literal int values.
    assert int(SeaAge.UNSET) == -1
    assert SEA_AGE_UNSET == -1
    assert SEA_AGE_1SW == 1
    assert SEA_AGE_2SW == 2
    assert SEA_AGE_3SW == 3
    # Reflective indexing: NAMES[value + 1] gives the name (offset by 1
    # since UNSET == -1).
    assert SEA_AGE_NAMES[int(SeaAge.UNSET) + 1] == "unset"
    assert SEA_AGE_NAMES[int(SeaAge.SW1) + 1] == "1SW"
    assert SEA_AGE_NAMES[int(SeaAge.SW2) + 1] == "2SW"
    assert SEA_AGE_NAMES[int(SeaAge.SW3) + 1] == "3SW"
    # Index 1 (SW0 slot) is structurally unused — there is no
    # sea_age == 0; salmon either return after >= 1 sea winter or
    # don't return at all.
    assert SEA_AGE_NAMES[1].startswith("_unused")
    # VALID_SEA_AGES round-trip through SeaAge enum.
    for a in VALID_SEA_AGES:
        assert SeaAge(a).value == a
    # Sentinel must not leak into the valid set.
    assert SEA_AGE_UNSET not in VALID_SEA_AGES


def test_agentpool_initializes_sea_age_to_unset():
    """C3.2: every newly-constructed AgentPool agent has sea_age == -1
    (offspring / non-Baltic / not-yet-drawn sentinel)."""
    pool = AgentPool(n=10, start_tri=0, rng_seed=42)
    assert "sea_age" in AgentPool.ARRAY_FIELDS
    assert pool.sea_age.dtype == np.int8
    assert pool.sea_age.shape == (10,)
    np.testing.assert_array_equal(pool.sea_age, np.full(10, SEA_AGE_UNSET))
