"""Tests for hatchery vs wild C3.2 — sea-age sampling.

Spec: docs/superpowers/specs/2026-05-03-hatchery-c3.2-seaage-design.md
"""

from __future__ import annotations

import numpy as np
import pytest

from salmon_ibm.agents import AgentPool
from salmon_ibm.population import Population
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


def _make_population(n: int = 5) -> Population:
    """Helper: minimal Population with n agents at cell 0.

    Population dataclass requires `name: str` as its first field. Mirror
    the existing test idiom from tests/test_hatchery_c3_spawn.py.
    """
    pool = AgentPool(n=n, start_tri=0, rng_seed=42)
    return Population(name="test", pool=pool)


def test_add_agents_sea_age_length_mismatch_raises():
    """C3.2 test 13: add_agents raises ValueError on shape mismatch
    between n and the sea_age array. Also covers n=0 no-op edge case."""
    pop = _make_population(n=5)
    with pytest.raises(ValueError, match=r"sea_age array shape"):
        pop.add_agents(
            n=5,
            positions=np.zeros(5, dtype=int),
            sea_age=np.array([1, 2], dtype=np.int8),
        )

    # n=0 with empty array: no-op, no raise.
    pop.add_agents(
        n=0,
        positions=np.zeros(0, dtype=int),
        sea_age=np.array([], dtype=np.int8),
    )


def test_population_sea_age_proxy():
    """C3.2: Population.sea_age @property returns pool.sea_age, AND
    continues to track pool.sea_age after add_agents rebinds the
    underlying array (load-bearing invariant — add_agents calls
    setattr(self.pool, attr, arr) for every ARRAY_FIELDS entry)."""
    pop = _make_population(n=3)
    assert pop.sea_age is pop.pool.sea_age
    pop.pool.sea_age[0] = SEA_AGE_1SW
    assert pop.sea_age[0] == 1
    # add_agents rebinds pool.sea_age. Proxy must follow the new array,
    # not stale-cache the old reference.
    pop.add_agents(n=2, positions=np.zeros(2, dtype=int))
    assert pop.sea_age is pop.pool.sea_age
    assert pop.sea_age.shape == (5,)


def test_adult_sea_age_mask_excludes_unset_and_garbage():
    """C3.2: adult_sea_age_mask uses np.isin(VALID_SEA_AGES) — NOT
    `>= 1` — so it excludes both the -1 sentinel AND any garbage int8
    write outside VALID_SEA_AGES. Locks the np.isin semantics against
    a "simplifying" refactor to `>= 1` that would silently admit
    bogus values (e.g. 4, 99) once VALID_SEA_AGES is extended for 4SW."""
    pop = _make_population(n=6)
    # Mix sentinel + valid values + bogus int8 garbage.
    pop.pool.sea_age[:] = np.array(
        [SEA_AGE_UNSET, SEA_AGE_1SW, SEA_AGE_2SW, SEA_AGE_3SW, 4, 99],
        dtype=np.int8,
    )
    mask = pop.adult_sea_age_mask()
    np.testing.assert_array_equal(
        mask, [False, True, True, True, False, False]
    )
