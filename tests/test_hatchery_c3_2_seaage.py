"""Tests for hatchery vs wild C3.2 — sea-age sampling.

Spec: docs/superpowers/specs/2026-05-03-hatchery-c3.2-seaage-design.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from salmon_ibm.agents import AgentPool
from salmon_ibm.baltic_params import (
    BalticBioParams,
    BalticSpeciesConfig,
    _apply_hatchery_overrides,
    load_baltic_species_config,
)
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


def test_sea_age_distribution_default_loads():
    """C3.2 test 1 (partial): default BalticBioParams has the wild
    sea-age trinomial; validates non-empty + sums-to-1."""
    p = BalticBioParams()
    assert p.sea_age_distribution == {1: 0.35, 2: 0.55, 3: 0.10}
    assert abs(sum(p.sea_age_distribution.values()) - 1.0) < 1e-9


def test_sea_age_distribution_rejects_invalid():
    """C3.2 test 2: __post_init__ raises on empty, non-summing, invalid
    keys, nonpositive values, bool-as-int, and numpy-int keys."""
    # Empty
    with pytest.raises(ValueError, match=r"non-empty"):
        BalticBioParams(sea_age_distribution={})

    # Non-summing
    with pytest.raises(ValueError, match=r"sum to 1.0"):
        BalticBioParams(sea_age_distribution={1: 0.5, 2: 0.5, 3: 0.5})

    # Invalid key (out of {1, 2, 3})
    with pytest.raises(ValueError, match=r"keys must be in \{1, 2, 3\}"):
        BalticBioParams(sea_age_distribution={0: 1.0})

    # Nonpositive value
    with pytest.raises(ValueError, match=r"positive floats"):
        BalticBioParams(sea_age_distribution={1: -0.1, 2: 0.5, 3: 0.6})

    # Bool key (type(True) is bool, NOT int — `type(k) is int` rejects it).
    # Error message format is `f"got {type(k).__module__}.{type(k).__name__}"`,
    # which renders as "got builtins.bool" — regex must allow the
    # "builtins." prefix.
    with pytest.raises(ValueError, match=r"got.*bool"):
        BalticBioParams(sea_age_distribution={True: 0.5, 2: 0.25, 3: 0.25})

    # Numpy int key — error message must include type name AND actionable
    # hint (`int(k)`). Locks both halves of the spec's mandate so a
    # regression that drops the actionable hint can't slip through.
    with pytest.raises(ValueError, match=r"numpy\.int64.*int\(k\)"):
        BalticBioParams(sea_age_distribution={
            np.int64(1): 0.5, np.int64(2): 0.25, np.int64(3): 0.25,
        })


CONFIG_PATH = Path("configs/baltic_salmon_species.yaml")


def test_sea_age_distribution_loads_from_yaml(tmp_path):
    """C3.2 test 1: deployed YAML loads both wild + hatchery sea_age
    distributions; cfg.hatchery is not cfg.wild; both sum to 1."""
    cfg = load_baltic_species_config(CONFIG_PATH)
    assert isinstance(cfg, BalticSpeciesConfig)
    assert cfg.wild.sea_age_distribution == {1: 0.35, 2: 0.55, 3: 0.10}
    assert cfg.hatchery is not None
    assert cfg.hatchery.sea_age_distribution == {1: 0.55, 2: 0.40, 3: 0.05}
    assert cfg.hatchery is not cfg.wild
    assert abs(sum(cfg.wild.sea_age_distribution.values()) - 1.0) < 1e-9
    assert abs(sum(cfg.hatchery.sea_age_distribution.values()) - 1.0) < 1e-9


def test_sea_age_full_replacement_override():
    """C3.2 test 3: hatchery override fully replaces the wild distribution
    (NOT shallow-merge). All three keys must be supplied."""
    wild = BalticBioParams()
    overrides = {"sea_age_distribution": {1: 0.55, 2: 0.40, 3: 0.05}}
    hatchery = _apply_hatchery_overrides(wild, overrides)
    assert hatchery.sea_age_distribution == {1: 0.55, 2: 0.40, 3: 0.05}
    # Wild unchanged (not aliased through the merge):
    assert wild.sea_age_distribution == {1: 0.35, 2: 0.55, 3: 0.10}


def test_sea_age_partial_override_rejected():
    """C3.2 test 4: partial sea_age_distribution override raises rather
    than silently merging into the wild base (closes silent-no-op
    failure mode where override happens to merge to wild)."""
    wild = BalticBioParams()
    overrides = {"sea_age_distribution": {1: 0.55}}  # missing 2 and 3
    with pytest.raises(ValueError, match=r"keys must be exactly \{1, 2, 3\}"):
        _apply_hatchery_overrides(wild, overrides)


def test_sea_age_no_override_inherits_wild():
    """C3.2 test 12: hatchery_overrides without sea_age_distribution key
    inherits the wild trinomial; locks the 'absent entirely' branch of
    full-replacement semantics."""
    wild = BalticBioParams()
    overrides = {"activity_by_behavior": {1: 1.5}}  # no sea_age_distribution
    hatchery = _apply_hatchery_overrides(wild, overrides)
    assert hatchery.sea_age_distribution == wild.sea_age_distribution


def test_sea_age_override_path_rejects_bool_keys():
    """C3.2 test 2 (override-path subset, bool variant): the
    hatchery-override coercion path applies the same `type(k) is int`
    rejection BEFORE int(k) coercion, so int(True) == 1 cannot silently
    pass."""
    wild = BalticBioParams()
    overrides = {"sea_age_distribution": {True: 0.55, 2: 0.40, 3: 0.05}}
    with pytest.raises(ValueError, match=r"got.*bool"):
        _apply_hatchery_overrides(wild, overrides)


def test_sea_age_override_path_rejects_numpy_int_keys():
    """C3.2 test 2 (override-path subset, numpy variant): mirrors the
    __post_init__ check so YAML loaders that emit np.int64 (some do)
    don't bypass the override-path safeguard via silent int(k)
    coercion."""
    wild = BalticBioParams()
    overrides = {"sea_age_distribution": {
        np.int64(1): 0.55, np.int64(2): 0.40, np.int64(3): 0.05,
    }}
    with pytest.raises(ValueError, match=r"numpy\.int64.*int\(k\)"):
        _apply_hatchery_overrides(wild, overrides)


def test_extended_overrides_still_reject_unknown_keys():
    """C3.2: ALLOWED_OVERRIDE_KEYS expansion still rejects unknown keys
    (regression for C2/C3.1)."""
    wild = BalticBioParams()
    overrides = {"unknown_key": 42}
    with pytest.raises(ValueError, match=r"unsupported keys"):
        _apply_hatchery_overrides(wild, overrides)


def test_simulation_step_injects_species_config_into_landscape():
    """C3.2: Simulation.step() injects _species_config into the
    landscape dict so events can access it. Load-bearing — the spec
    flagged the absence of this injection as a CRITICAL silent-failure
    mode in pass-1 review. Locks the wire-through via static-source
    inspection (avoids needing full Simulation construction, which
    requires mesh / environment / scenario fixtures the C3.2 plan does
    not own)."""
    import inspect
    from salmon_ibm.simulation import Landscape, Simulation

    # The Landscape TypedDict declares the field.
    assert "species_config" in Landscape.__annotations__

    # Simulation.step() body literally constructs the landscape dict
    # with `"species_config": ...getattr(self, "_species_config", None)...`.
    # A regression that drops this entry would crash CI here.
    src = inspect.getsource(Simulation.step)
    assert '"species_config"' in src and '"_species_config"' in src, (
        "Simulation.step() must inject species_config into landscape; "
        "C3.2 spec section 'simulation.py changes' mandates this entry."
    )
    # Residual false-positive: a refactor that nests species_config
    # into another dict (e.g. `est_cfg["species_config"]`) would keep
    # the literal in the source but break landscape.get("species_config").
    # Lock the top-level pairing.
    assert 'species_config": getattr(self, "_species_config"' in src, (
        "species_config must be a top-level entry of the landscape dict; "
        "nesting it (e.g. into est_cfg) would break "
        "landscape.get('species_config') in event handlers."
    )
