"""Unit tests for the Population class."""

import numpy as np
import pytest

from salmon_ibm.agents import AgentPool
from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
from salmon_ibm.population import Population


@pytest.fixture
def basic_pool():
    return AgentPool(n=10, start_tri=0, rng_seed=42)


@pytest.fixture
def basic_pop(basic_pool):
    return Population(name="test", pool=basic_pool)


class TestPopulationInit:
    def test_name_and_size(self, basic_pop):
        assert basic_pop.name == "test"
        assert basic_pop.n == 10
        assert basic_pop.n_alive == 10

    def test_all_start_as_floaters(self, basic_pop):
        assert (basic_pop.group_id == -1).all()
        assert basic_pop.floaters.sum() == 10
        assert basic_pop.grouped.sum() == 0

    def test_agent_ids_sequential(self, basic_pop):
        np.testing.assert_array_equal(basic_pop.agent_ids, np.arange(10))

    def test_with_accumulators(self, basic_pool):
        acc_defs = [AccumulatorDef("energy", min_val=0.0)]
        acc_mgr = AccumulatorManager(10, acc_defs)
        pop = Population("test", basic_pool, accumulator_mgr=acc_mgr)
        assert pop.accumulator_mgr is not None

    def test_with_traits(self, basic_pool):
        trait_defs = [
            TraitDefinition("stage", TraitType.PROBABILISTIC, ["juvenile", "adult"])
        ]
        trait_mgr = TraitManager(10, trait_defs)
        pop = Population("test", basic_pool, trait_mgr=trait_mgr)
        assert pop.trait_mgr is not None

    def test_proxy_properties(self, basic_pop):
        """Population proxies should reflect AgentPool state."""
        assert len(basic_pop.behavior) == 10
        assert len(basic_pop.ed_kJ_g) == 10
        assert len(basic_pop.mass_g) == 10
        assert len(basic_pop.steps) == 10
        assert len(basic_pop.target_spawn_hour) == 10
        assert basic_pop.t3h_mean().shape == (10,)


class TestRemoveAgents:
    def test_remove_marks_dead(self, basic_pop):
        basic_pop.remove_agents(np.array([0, 3, 7]))
        assert not basic_pop.pool.alive[0]
        assert not basic_pop.pool.alive[3]
        assert not basic_pop.pool.alive[7]
        assert basic_pop.n_alive == 7

    def test_compact_shrinks_arrays(self, basic_pop):
        basic_pop.remove_agents(np.array([0, 3, 7]))
        basic_pop.compact()
        assert basic_pop.n == 7
        assert basic_pop.pool.alive.all()
        assert len(basic_pop.group_id) == 7
        assert len(basic_pop.agent_ids) == 7


class TestAddAgents:
    def test_add_extends_arrays(self, basic_pop):
        positions = np.array([5, 5, 5])
        new_idx = basic_pop.add_agents(3, positions)
        assert basic_pop.n == 13
        assert basic_pop.n_alive == 13
        np.testing.assert_array_equal(new_idx, [10, 11, 12])

    def test_new_agents_get_unique_ids(self, basic_pop):
        basic_pop.add_agents(3, np.array([0, 0, 0]))
        assert len(np.unique(basic_pop.agent_ids)) == 13

    def test_add_agents_with_accumulators(self, basic_pool):
        acc_defs = [AccumulatorDef("energy", min_val=0.0)]
        acc_mgr = AccumulatorManager(10, acc_defs)
        pop = Population("test", basic_pool, accumulator_mgr=acc_mgr)
        pop.add_agents(5, np.zeros(5, dtype=int))
        assert pop.accumulator_mgr.data.shape == (1, 15)

    def test_add_then_compact_roundtrip(self, basic_pop):
        basic_pop.add_agents(5, np.zeros(5, dtype=int))
        basic_pop.remove_agents(np.array([0, 1, 2]))
        basic_pop.compact()
        assert basic_pop.n == 12
        assert basic_pop.pool.alive.all()


def test_compact_with_genome_traits_accumulators():
    """Compact should preserve genome, trait, and accumulator data alignment."""
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
    from salmon_ibm.traits import TraitManager, TraitDefinition

    pool = AgentPool(n=5, start_tri=np.zeros(5, dtype=int))
    pop = Population(name="test", pool=pool)

    # Add accumulators
    pop.accumulator_mgr = AccumulatorManager(5, [AccumulatorDef("energy")])
    pop.accumulator_mgr.data[0, :] = [10.0, 20.0, 30.0, 40.0, 50.0]

    # Add traits
    pop.trait_mgr = TraitManager(
        5,
        [
            TraitDefinition(
                name="stage",
                trait_type=TraitType.PROBABILISTIC,
                categories=["juv", "adult"],
            )
        ],
    )
    pop.trait_mgr._data["stage"][:] = [0, 1, 0, 1, 0]

    # Kill agents 1 and 3
    pool.alive[1] = False
    pool.alive[3] = False

    pop.compact()

    # Should have 3 survivors (indices 0, 2, 4)
    assert pop.n == 3
    assert pop.accumulator_mgr.data.shape == (1, 3)
    np.testing.assert_array_equal(pop.accumulator_mgr.data[0, :], [10.0, 30.0, 50.0])
    np.testing.assert_array_equal(pop.trait_mgr._data["stage"], [0, 0, 0])


def test_add_agents_extends_genome():
    """add_agents should extend genome array with zeros for new agents."""
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    from salmon_ibm.genetics import GenomeManager, LocusDefinition

    pool = AgentPool(n=3, start_tri=np.zeros(3, dtype=int))
    pop = Population(name="test", pool=pool)
    loci = [
        LocusDefinition("a", n_alleles=4, position=0.0),
        LocusDefinition("b", n_alleles=4, position=1.0),
    ]
    pop.genome = GenomeManager(3, loci=loci)
    pop.genome.genotypes[:] = 1  # set existing agents to allele 1

    new_ids = pop.add_agents(2, np.array([0, 0]))
    assert pop.genome.n_agents == 5
    assert pop.genome.genotypes.shape == (5, 2, 2)
    # New agents should have zero genotypes
    np.testing.assert_array_equal(pop.genome.genotypes[3:], 0)
    # Old agents should keep their alleles
    np.testing.assert_array_equal(pop.genome.genotypes[:3], 1)


def test_add_agents_defaults_natal_and_exit_to_minus_one():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np
    pool = AgentPool(n=2, start_tri=0)
    pop = Population(name="test", pool=pool)
    pop.natal_reach_id[:] = np.array([5, 7], dtype=np.int8)
    pop.exit_branch_id[:] = np.array([3, 4], dtype=np.int8)
    new_idx = pop.add_agents(n=3, positions=np.array([0, 1, 2]))
    assert (pop.natal_reach_id[new_idx] == -1).all()
    assert (pop.exit_branch_id[new_idx] == -1).all()
    assert pop.natal_reach_id[0] == 5
    assert pop.natal_reach_id[1] == 7


def test_set_natal_reach_from_cells_writes_correct_reach_ids():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np

    class _FakeMesh:
        reach_names = ["Nemunas", "Atmata", "Skirvyte"]
        reach_id = np.array([0, 0, 1, 2, 1], dtype=np.int8)  # 5 cells

    pool = AgentPool(n=3, start_tri=np.array([1, 3, 4]))  # cells 1,3,4
    pop = Population(name="test", pool=pool)
    pop.set_natal_reach_from_cells(np.arange(3), _FakeMesh())
    expected = np.array([0, 2, 1], dtype=np.int8)  # Nemunas, Skirvyte, Atmata
    assert (pop.natal_reach_id == expected).all()


def test_set_natal_reach_from_cells_no_op_without_reach_names():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np

    class _NoMesh:
        pass

    pool = AgentPool(n=2, start_tri=0)
    pop = Population(name="test", pool=pool)
    pop.set_natal_reach_from_cells(np.arange(2), _NoMesh())
    assert (pop.natal_reach_id == -1).all()


def test_assert_natal_tagged_fires_on_untagged_alive_on_mesh():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np
    import pytest

    pool = AgentPool(n=3, start_tri=np.array([0, 1, 2]))
    pop = Population(name="test", pool=pool)
    pool.natal_reach_id[:] = np.array([5, -1, 7], dtype=np.int8)
    pool.alive[:] = True
    with pytest.raises(AssertionError, match="natal_reach_id tagging"):
        pop.assert_natal_tagged()


def test_assert_natal_tagged_silent_when_all_tagged():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np
    pool = AgentPool(n=3, start_tri=np.array([0, 1, 2]))
    pop = Population(name="test", pool=pool)
    pool.natal_reach_id[:] = np.array([5, 6, 7], dtype=np.int8)
    pool.alive[:] = True
    pop.assert_natal_tagged()


def test_assert_natal_tagged_ignores_dead_agents():
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    import numpy as np
    pool = AgentPool(n=3, start_tri=np.array([0, 1, 2]))
    pop = Population(name="test", pool=pool)
    pool.natal_reach_id[:] = np.array([5, -1, 7], dtype=np.int8)
    pool.alive[:] = np.array([True, False, True])
    pop.assert_natal_tagged()
