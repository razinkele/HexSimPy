"""Tests for the Origin enum and module constants (Tier C1)."""
import pytest


def test_origin_enum_values():
    """Origin enum values match the int8 column convention (WILD=0, HATCHERY=1)."""
    from salmon_ibm.origin import Origin
    assert Origin.WILD == 0
    assert Origin.HATCHERY == 1


def test_origin_names_roundtrip():
    """ORIGIN_NAMES index aligns with Origin enum values."""
    from salmon_ibm.origin import Origin, ORIGIN_NAMES
    assert ORIGIN_NAMES.index("wild") == 0
    assert ORIGIN_NAMES.index("hatchery") == 1
    assert ORIGIN_NAMES[Origin.WILD] == "wild"
    assert ORIGIN_NAMES[Origin.HATCHERY] == "hatchery"


def test_origin_names_invalid_raises():
    """Unknown origin string raises ValueError via list.index."""
    from salmon_ibm.origin import ORIGIN_NAMES
    with pytest.raises(ValueError):
        ORIGIN_NAMES.index("salmon")


def test_agent_pool_origin_default_wild():
    """A freshly-allocated AgentPool has all agents tagged as WILD (0)."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.origin import ORIGIN_WILD
    pool = AgentPool(n=10, start_tri=0, rng_seed=42)
    assert pool.origin.shape == (10,)
    assert pool.origin.dtype == np.int8
    assert (pool.origin == ORIGIN_WILD).all()


def test_population_add_agents_with_origin():
    """add_agents(origin=ORIGIN_HATCHERY) writes 1 to new agents only."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY
    pool = AgentPool(n=3, start_tri=0, rng_seed=42)
    # Population is a @dataclass with `name` as first required field.
    pop = Population(name="test", pool=pool)
    new_idx = pop.add_agents(
        n=2,
        positions=np.array([0, 0]),
        origin=ORIGIN_HATCHERY,
    )
    # Existing 3 agents stay WILD
    assert (pool.origin[:3] == ORIGIN_WILD).all()
    # New 2 agents are HATCHERY
    assert (pool.origin[new_idx] == ORIGIN_HATCHERY).all()


def test_introduction_event_propagates_origin():
    """IntroductionEvent(origin=ORIGIN_HATCHERY) tags the new agents
    as hatchery in the population's origin column."""
    import numpy as np
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.events_builtin import IntroductionEvent
    from salmon_ibm.origin import ORIGIN_WILD, ORIGIN_HATCHERY

    pool = AgentPool(n=2, start_tri=0, rng_seed=42)
    # Population is a @dataclass with `name` as first required field.
    pop = Population(name="test", pool=pool)
    landscape = {"rng": np.random.default_rng(0)}

    # Event base class requires `name`; trigger defaults to EveryStep.
    evt = IntroductionEvent(
        name="intro_test",
        n_agents=3,
        positions=[0],
        origin=ORIGIN_HATCHERY,
    )
    evt.execute(pop, landscape, t=0, mask=None)

    # Pre-existing 2 agents stay WILD
    assert (pool.origin[:2] == ORIGIN_WILD).all()
    # New 3 agents are HATCHERY
    assert (pool.origin[2:5] == ORIGIN_HATCHERY).all()
