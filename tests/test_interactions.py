"""Unit tests for multi-species interactions."""

import numpy as np
from salmon_ibm.interactions import (
    MultiPopulationManager,
    InteractionEvent,
    InteractionOutcome,
)

from tests.helpers import MockPopulation


class TestMultiPopulationManager:
    def test_register_and_get(self):
        mgr = MultiPopulationManager()
        pop = MockPopulation(5, [0, 1, 2, 3, 4])
        mgr.register("prey", pop)
        assert mgr.get("prey") is pop

    def test_cell_index(self):
        mgr = MultiPopulationManager()
        pop = MockPopulation(4, [0, 0, 1, 1])
        mgr.register("fish", pop)
        mgr.build_cell_index("fish")
        agents_0 = mgr.agents_at_cell("fish", 0)
        agents_1 = mgr.agents_at_cell("fish", 1)
        np.testing.assert_array_equal(sorted(agents_0), [0, 1])
        np.testing.assert_array_equal(sorted(agents_1), [2, 3])

    def test_dead_agents_excluded(self):
        mgr = MultiPopulationManager()
        pop = MockPopulation(4, [0, 0, 1, 1], alive=[True, False, True, False])
        mgr.register("fish", pop)
        mgr.build_cell_index("fish")
        assert len(mgr.agents_at_cell("fish", 0)) == 1
        assert len(mgr.agents_at_cell("fish", 1)) == 1

    def test_co_located_pairs(self):
        mgr = MultiPopulationManager()
        prey = MockPopulation(3, [0, 1, 2])
        pred = MockPopulation(2, [0, 2])
        mgr.register("prey", prey)
        mgr.register("pred", pred)
        pairs = mgr.co_located_pairs("prey", "pred")
        # Cell 0 and cell 2 are shared
        assert len(pairs) == 2

    def test_no_overlap(self):
        mgr = MultiPopulationManager()
        pop_a = MockPopulation(2, [0, 1])
        pop_b = MockPopulation(2, [2, 3])
        mgr.register("a", pop_a)
        mgr.register("b", pop_b)
        pairs = mgr.co_located_pairs("a", "b")
        assert len(pairs) == 0


class TestInteractionEvent:
    def test_predation_kills_prey(self):
        mgr = MultiPopulationManager()
        prey = MockPopulation(5, [0, 0, 0, 1, 1])
        pred = MockPopulation(2, [0, 1])
        mgr.register("prey", prey)
        mgr.register("pred", pred)

        event = InteractionEvent(
            name="predation",
            pop_a_name="pred",
            pop_b_name="prey",
            encounter_probability=1.0,
            outcome=InteractionOutcome.PREDATION,
        )
        landscape = {"multi_pop_mgr": mgr, "rng": np.random.default_rng(42)}
        mask = pred.alive.copy()
        event.execute(pred, landscape, t=0, mask=mask)
        # Some prey should have been killed
        assert prey.alive.sum() < 5


class TestTransitionEvent:
    def test_seir_transition(self):
        from salmon_ibm.events_phase3 import TransitionEvent
        from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
        from salmon_ibm.population import Population
        from salmon_ibm.agents import AgentPool

        pool = AgentPool(n=100, start_tri=0, rng_seed=42)
        trait_defs = [
            TraitDefinition("disease", TraitType.PROBABILISTIC, ["S", "I", "R"])
        ]
        trait_mgr = TraitManager(100, trait_defs)
        # Start all as Susceptible (0)
        trait_mgr._data["disease"][:] = 0
        pop = Population("host", pool, trait_mgr=trait_mgr)

        # Transition: S->I with p=0.5, I->R with p=1.0, R stays
        T = np.array(
            [
                [0.5, 0.5, 0.0],  # S: 50% chance to become I
                [0.0, 0.0, 1.0],  # I: always becomes R
                [0.0, 0.0, 1.0],  # R: stays R
            ]
        )
        event = TransitionEvent(
            name="infect", trait_name="disease", transition_matrix=T
        )
        landscape = {"rng": np.random.default_rng(42)}
        mask = pop.alive.copy()
        event.execute(pop, landscape, t=0, mask=mask)

        vals = pop.trait_mgr.get("disease")
        # Some should have transitioned from S to I
        n_infected = (vals == 1).sum()
        assert n_infected > 0, "Some agents should be infected"
        assert (vals == 0).sum() > 0, "Some should remain susceptible"
