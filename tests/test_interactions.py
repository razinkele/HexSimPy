"""Unit tests for multi-species interactions."""

import numpy as np
import pytest

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


class TestInteractionEventVectorized:
    """Invariant tests for the Option 2 (first-A-wins) vectorized execute()."""

    def _setup(self, n_pred=3, n_prey=5, cell_pred=0, cell_prey=0):
        from salmon_ibm.agents import AgentPool
        from salmon_ibm.population import Population
        mgr = MultiPopulationManager()
        pred_pool = AgentPool(n=n_pred, start_tri=0, rng_seed=1)
        pred_pool.tri_idx[:] = cell_pred
        prey_pool = AgentPool(n=n_prey, start_tri=0, rng_seed=2)
        prey_pool.tri_idx[:] = cell_prey
        pred = Population(name="pred", pool=pred_pool)
        prey = Population(name="prey", pool=prey_pool)
        mgr.register("pred", pred)
        mgr.register("prey", prey)
        return mgr, pred, prey

    def test_p_eq_1_kills_every_live_prey_in_shared_cells(self):
        """With p=1.0, every B in a shared cell gets killed (first A wins each)."""
        mgr, pred, prey = self._setup(n_pred=3, n_prey=5)
        landscape = {"multi_pop_mgr": mgr, "rng": np.random.default_rng(0)}
        event = InteractionEvent(
            name="p", pop_a_name="pred", pop_b_name="prey",
            encounter_probability=1.0,
            outcome=InteractionOutcome.PREDATION,
        )
        event.execute(pred, landscape, t=0, mask=pred.pool.alive)
        assert int(prey.pool.alive.sum()) == 0, (
            f"All prey should be dead with p=1.0; alive={prey.pool.alive.sum()}"
        )
        stats = landscape["interaction_stats"][0]
        assert stats["kills"] == 5
        # Encounters are total hits in the |A|*|B| matrix (pre-dedup).
        # With p=1.0 and 3*5 pairs, encounters = 15.
        assert stats["encounters"] == 15

    def test_p_eq_0_yields_no_kills(self):
        """With p=0, no rolls succeed; no kills, no encounters."""
        mgr, pred, prey = self._setup(n_pred=3, n_prey=5)
        landscape = {"multi_pop_mgr": mgr, "rng": np.random.default_rng(0)}
        event = InteractionEvent(
            name="p", pop_a_name="pred", pop_b_name="prey",
            encounter_probability=0.0,
            outcome=InteractionOutcome.PREDATION,
        )
        event.execute(pred, landscape, t=0, mask=pred.pool.alive)
        assert int(prey.pool.alive.sum()) == 5
        stats = landscape["interaction_stats"][0]
        assert stats["kills"] == 0
        assert stats["encounters"] == 0

    def test_no_shared_cells_no_kills(self):
        """Predator cell 0, prey cell 1 — no overlap, no interactions."""
        mgr, pred, prey = self._setup(n_pred=3, n_prey=5, cell_pred=0, cell_prey=1)
        landscape = {"multi_pop_mgr": mgr, "rng": np.random.default_rng(0)}
        event = InteractionEvent(
            name="p", pop_a_name="pred", pop_b_name="prey",
            encounter_probability=1.0,
            outcome=InteractionOutcome.PREDATION,
        )
        event.execute(pred, landscape, t=0, mask=pred.pool.alive)
        assert int(prey.pool.alive.sum()) == 5
        assert landscape["interaction_stats"][0]["kills"] == 0

    def test_each_kill_credits_exactly_one_predator(self):
        """With p=1.0 and a resource accumulator, total resource gain
        equals kills * resource_gain_amount (conservation)."""
        from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
        mgr, pred, prey = self._setup(n_pred=3, n_prey=5)
        pred.accumulator_mgr = AccumulatorManager(
            3, [AccumulatorDef("food", min_val=0.0)]
        )
        landscape = {"multi_pop_mgr": mgr, "rng": np.random.default_rng(0)}
        event = InteractionEvent(
            name="p", pop_a_name="pred", pop_b_name="prey",
            encounter_probability=1.0,
            outcome=InteractionOutcome.PREDATION,
            resource_gain_acc="food",
            resource_gain_amount=2.0,
        )
        event.execute(pred, landscape, t=0, mask=pred.pool.alive)
        food_total = pred.accumulator_mgr.data[0, :].sum()
        kills = landscape["interaction_stats"][0]["kills"]
        assert food_total == pytest.approx(kills * 2.0), (
            f"Food total {food_total} should equal {kills} kills * 2.0"
        )

    def test_first_predator_wins_when_p_equals_1(self):
        """With p=1.0, every hit in row 0 — so predator index 0 in each
        cell's agents_a list claims ALL kills in that cell."""
        from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
        mgr, pred, prey = self._setup(n_pred=3, n_prey=5)
        pred.accumulator_mgr = AccumulatorManager(
            3, [AccumulatorDef("food", min_val=0.0)]
        )
        landscape = {"multi_pop_mgr": mgr, "rng": np.random.default_rng(0)}
        event = InteractionEvent(
            name="p", pop_a_name="pred", pop_b_name="prey",
            encounter_probability=1.0,
            outcome=InteractionOutcome.PREDATION,
            resource_gain_acc="food",
            resource_gain_amount=1.0,
        )
        event.execute(pred, landscape, t=0, mask=pred.pool.alive)
        # First predator (index 0) gets all 5 kills' resources.
        foods = pred.accumulator_mgr.data[0, :]
        assert foods[0] == pytest.approx(5.0)
        assert foods[1] == pytest.approx(0.0)
        assert foods[2] == pytest.approx(0.0)
