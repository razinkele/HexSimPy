"""Integration tests for Phase 2: multi-generation simulation."""
import numpy as np
import pytest

from salmon_ibm.agents import AgentPool
from salmon_ibm.population import Population
from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
from salmon_ibm.events import EventSequencer, EveryStep, Periodic
from salmon_ibm.events_builtin import (
    StageSpecificSurvivalEvent,
    IntroductionEvent,
    ReproductionEvent,
    FloaterCreationEvent,
)
from salmon_ibm.barriers import BarrierMap, BarrierOutcome


class TestMultiGenerationSimulation:
    @pytest.fixture
    def setup_simulation(self):
        pool = AgentPool(n=50, start_tri=0, rng_seed=42)
        trait_defs = [
            TraitDefinition("stage", TraitType.PROBABILISTIC,
                          ["egg", "juvenile", "adult"]),
        ]
        trait_mgr = TraitManager(50, trait_defs)
        trait_mgr._data["stage"][:] = 2  # all start as adult
        acc_defs = [AccumulatorDef("age", min_val=0.0)]
        acc_mgr = AccumulatorManager(50, acc_defs)
        pop = Population("salmon", pool,
                        accumulator_mgr=acc_mgr, trait_mgr=trait_mgr)
        pop.group_id[:25] = 0
        events = [
            StageSpecificSurvivalEvent(
                name="mortality",
                mortality_rates={"egg": 0.5, "juvenile": 0.1, "adult": 0.02},
            ),
            ReproductionEvent(
                name="reproduction",
                trigger=Periodic(interval=10),
                clutch_mean=3.0,
                offspring_trait_name="stage",
                offspring_trait_value="egg",
                offspring_mass_mean=10.0,
            ),
            FloaterCreationEvent(
                name="disperse",
                trigger=Periodic(interval=5),
                probability=0.2,
            ),
        ]
        sequencer = EventSequencer(events)
        landscape = {"rng": np.random.default_rng(42)}
        return pop, sequencer, landscape

    def test_population_survives_20_steps(self, setup_simulation):
        pop, sequencer, landscape = setup_simulation
        for t in range(20):
            sequencer.step(pop, landscape, t)
        assert pop.n_alive > 0

    def test_reproduction_adds_agents(self, setup_simulation):
        pop, sequencer, landscape = setup_simulation
        n_initial = pop.n
        for t in range(11):
            sequencer.step(pop, landscape, t)
        assert pop.n > n_initial

    def test_survival_reduces_population(self, setup_simulation):
        pop, sequencer, landscape = setup_simulation
        n_initial_alive = pop.n_alive
        survival = StageSpecificSurvivalEvent(
            name="kill_eggs",
            mortality_rates={"egg": 1.0, "juvenile": 0.0, "adult": 0.0},
        )
        pop.trait_mgr._data["stage"][:10] = 0
        mask = pop.alive.copy()
        survival.execute(pop, landscape, t=0, mask=mask)
        assert pop.n_alive == n_initial_alive - 10

    def test_compact_after_deaths(self, setup_simulation):
        pop, sequencer, landscape = setup_simulation
        for t in range(20):
            sequencer.step(pop, landscape, t)
        n_alive_before = pop.n_alive
        pop.compact()
        assert pop.n == n_alive_before
        assert pop.pool.alive.all()


class TestBarrierIntegration:
    def test_impassable_barrier_prevents_crossing(self):
        bmap = BarrierMap()
        bmap.add_edge(0, 1, BarrierOutcome.impassable())
        bmap.add_edge(1, 0, BarrierOutcome.impassable())
        assert bmap.has_barriers()
        assert bmap.n_edges == 2

    def test_partial_barrier_allows_some_crossing(self):
        bmap = BarrierMap()
        outcome = BarrierOutcome(0.0, 0.5, 0.5)
        bmap.add_edge(0, 1, outcome)
        result = bmap.check(0, 1)
        assert result.p_transmission == 0.5
        assert result.p_deflection == 0.5
