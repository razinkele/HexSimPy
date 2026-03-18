"""Unit tests for the event engine."""
import numpy as np
import pytest

from salmon_ibm.events import (
    EveryStep, Once, Periodic, Window, RandomTrigger,
    Event, EventSequencer, EventGroup,
)
from salmon_ibm.agents import AgentPool
from salmon_ibm.events_builtin import StageSpecificSurvivalEvent, IntroductionEvent, ReproductionEvent, FloaterCreationEvent, CensusEvent
from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
from salmon_ibm.population import Population


class TestEveryStep:
    def test_always_fires(self):
        trigger = EveryStep()
        for t in range(20):
            assert trigger.should_fire(t) is True


class TestOnce:
    def test_fires_only_at_target(self):
        trigger = Once(at=5)
        assert trigger.should_fire(4) is False
        assert trigger.should_fire(5) is True
        assert trigger.should_fire(6) is False


class TestPeriodic:
    def test_fires_every_n_steps(self):
        trigger = Periodic(interval=3, offset=0)
        results = [trigger.should_fire(t) for t in range(10)]
        assert results == [True, False, False, True, False, False, True, False, False, True]

    def test_offset(self):
        trigger = Periodic(interval=4, offset=2)
        assert trigger.should_fire(0) is False
        assert trigger.should_fire(1) is False
        assert trigger.should_fire(2) is True
        assert trigger.should_fire(6) is True


class TestWindow:
    def test_fires_within_range(self):
        trigger = Window(start=3, end=6)
        results = [trigger.should_fire(t) for t in range(8)]
        assert results == [False, False, False, True, True, True, False, False]


class TestRandomTrigger:
    def test_probability_zero_never_fires(self):
        trigger = RandomTrigger(p=0.0)
        assert all(not trigger.should_fire(t) for t in range(100))

    def test_probability_one_always_fires(self):
        trigger = RandomTrigger(p=1.0)
        assert all(trigger.should_fire(t) for t in range(100))

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        t1 = RandomTrigger(p=0.5, _rng=rng1)
        t2 = RandomTrigger(p=0.5, _rng=rng2)
        results1 = [t1.should_fire(t) for t in range(50)]
        results2 = [t2.should_fire(t) for t in range(50)]
        assert results1 == results2


# ---------------------------------------------------------------------------
# Helpers for sequencer/group tests
# ---------------------------------------------------------------------------

class StubEvent(Event):
    def __init__(self, name, trigger=None):
        super().__init__(name=name, trigger=trigger or EveryStep())
        self.calls = []

    def execute(self, population, landscape, t, mask):
        self.calls.append({"t": t, "mask_sum": int(mask.sum())})


class FakePopulation:
    def __init__(self, n=10):
        self.n = n
        self.alive = np.ones(n, dtype=bool)
        self.arrived = np.zeros(n, dtype=bool)


class TestEventSequencer:
    def test_runs_events_in_order(self):
        call_order = []
        e1 = StubEvent("first")
        e2 = StubEvent("second")
        orig1, orig2 = e1.execute, e2.execute
        def track1(*a, **kw):
            call_order.append("first")
            orig1(*a, **kw)
        def track2(*a, **kw):
            call_order.append("second")
            orig2(*a, **kw)
        e1.execute = track1
        e2.execute = track2
        seq = EventSequencer([e1, e2])
        seq.step(FakePopulation(), {}, t=0)
        assert call_order == ["first", "second"]

    def test_respects_trigger(self):
        e_every = StubEvent("every", EveryStep())
        e_once = StubEvent("once", Once(at=2))
        seq = EventSequencer([e_every, e_once])
        pop = FakePopulation()
        for t in range(5):
            seq.step(pop, {}, t)
        assert len(e_every.calls) == 5
        assert len(e_once.calls) == 1
        assert e_once.calls[0]["t"] == 2

    def test_mask_excludes_dead_and_arrived(self):
        e = StubEvent("check")
        pop = FakePopulation(n=10)
        pop.alive[0:3] = False
        pop.arrived[7:10] = True
        seq = EventSequencer([e])
        seq.step(pop, {}, t=0)
        assert e.calls[0]["mask_sum"] == 4


class TestEventGroup:
    def test_runs_sub_events(self):
        sub1 = StubEvent("sub1")
        sub2 = StubEvent("sub2")
        group = EventGroup(name="group", sub_events=[sub1, sub2])
        seq = EventSequencer([group])
        seq.step(FakePopulation(), {}, t=0)
        assert len(sub1.calls) == 1
        assert len(sub2.calls) == 1

    def test_iterations(self):
        sub = StubEvent("sub")
        group = EventGroup(name="group", sub_events=[sub], iterations=3)
        seq = EventSequencer([group])
        seq.step(FakePopulation(), {}, t=0)
        assert len(sub.calls) == 3

    def test_nested_trigger_respected(self):
        sub_every = StubEvent("every", EveryStep())
        sub_once = StubEvent("once", Once(at=5))
        group = EventGroup(name="group", sub_events=[sub_every, sub_once])
        seq = EventSequencer([group])
        seq.step(FakePopulation(), {}, t=0)
        assert len(sub_every.calls) == 1
        assert len(sub_once.calls) == 0

    def test_group_trigger_gates_sub_events(self):
        sub = StubEvent("sub")
        group = EventGroup(name="group", trigger=Once(at=3), sub_events=[sub])
        seq = EventSequencer([group])
        for t in range(5):
            seq.step(FakePopulation(), {}, t)
        assert len(sub.calls) == 1


# ---------------------------------------------------------------------------
# Built-in event types
# ---------------------------------------------------------------------------

from salmon_ibm.events_builtin import MovementEvent, SurvivalEvent, AccumulateEvent, CustomEvent
from salmon_ibm.bioenergetics import BioParams


class TestMovementEvent:
    def test_calls_execute_movement(self, mocker):
        mock_move = mocker.patch("salmon_ibm.events_builtin.execute_movement")
        rng = np.random.default_rng(42)
        pop = FakePopulation(n=5)
        landscape = {"mesh": object(), "fields": {}, "rng": rng}
        mask = np.ones(5, dtype=bool)
        event = MovementEvent(name="movement", n_micro_steps=3, cwr_threshold=16.0)
        event.execute(pop, landscape, t=0, mask=mask)
        mock_move.assert_called_once()


class TestSurvivalEvent:
    def _make_landscape(self, n_cells=20, temperature=15.0):
        fields = {"temperature": np.full(n_cells, temperature)}
        activity_lut = np.ones(5)
        return {"fields": fields, "activity_lut": activity_lut, "est_cfg": {}}

    def test_thermal_mortality(self):
        pop = FakePopulation(n=5)
        pop.tri_idx = np.zeros(5, dtype=int)
        pop.ed_kJ_g = np.full(5, 6.5)
        pop.mass_g = np.full(5, 3500.0)
        pop.behavior = np.zeros(5, dtype=int)
        bio = BioParams(T_MAX=20.0)
        landscape = self._make_landscape(temperature=25.0)
        mask = pop.alive & ~pop.arrived
        event = SurvivalEvent(name="survival", bio_params=bio)
        event.execute(pop, landscape, t=0, mask=mask)
        assert not pop.alive.any(), "All agents should die at temp > T_MAX"

    def test_no_mortality_at_safe_temp(self):
        pop = FakePopulation(n=5)
        pop.tri_idx = np.zeros(5, dtype=int)
        pop.ed_kJ_g = np.full(5, 6.5)
        pop.mass_g = np.full(5, 3500.0)
        pop.behavior = np.zeros(5, dtype=int)
        bio = BioParams(T_MAX=26.0)
        landscape = self._make_landscape(temperature=10.0)
        mask = pop.alive & ~pop.arrived
        event = SurvivalEvent(name="survival", bio_params=bio)
        event.execute(pop, landscape, t=0, mask=mask)
        assert pop.alive.all(), "All agents should survive at safe temperature"


class TestAccumulateEvent:
    def test_runs_updaters_in_order(self):
        call_log = []
        def updater_a(pop, land, t, mask): call_log.append("a")
        def updater_b(pop, land, t, mask): call_log.append("b")
        event = AccumulateEvent(name="acc", updaters=[updater_a, updater_b])
        pop = FakePopulation()
        mask = pop.alive & ~pop.arrived
        event.execute(pop, {}, t=0, mask=mask)
        assert call_log == ["a", "b"]

    def test_updater_modifies_population(self):
        def increment_steps(pop, land, t, mask):
            pop.steps = getattr(pop, "steps", 0) + 1
        event = AccumulateEvent(name="acc", updaters=[increment_steps])
        pop = FakePopulation()
        pop.steps = 0
        mask = pop.alive & ~pop.arrived
        event.execute(pop, {}, t=0, mask=mask)
        assert pop.steps == 1


class TestCustomEvent:
    def test_calls_callback(self):
        calls = []
        def my_callback(pop, land, t, mask): calls.append(t)
        event = CustomEvent(name="custom", callback=my_callback)
        pop = FakePopulation()
        mask = pop.alive & ~pop.arrived
        event.execute(pop, {}, t=7, mask=mask)
        assert calls == [7]

    def test_callback_receives_correct_mask(self):
        received = []
        def capture(pop, land, t, mask): received.append(mask.copy())
        pop = FakePopulation(n=10)
        pop.alive[0:3] = False
        mask = pop.alive & ~pop.arrived
        event = CustomEvent(name="custom", callback=capture)
        event.execute(pop, {}, t=0, mask=mask)
        assert received[0].sum() == 7


# ---------------------------------------------------------------------------
# YAML event loading
# ---------------------------------------------------------------------------

from salmon_ibm.events import load_events_from_config, EVENT_REGISTRY


class TestLoadEventsFromConfig:
    def test_loads_custom_event(self):
        calls = []
        def my_cb(pop, land, t, mask): calls.append(t)
        defs = [{"type": "custom", "name": "my_cb"}]
        events = load_events_from_config(defs, {"my_cb": my_cb})
        assert len(events) == 1
        assert events[0].name == "my_cb"

    def test_loads_movement_event(self):
        # Ensure events_builtin is imported so register_event runs
        import salmon_ibm.events_builtin  # noqa: F401
        defs = [{"type": "movement", "name": "move", "params": {"n_micro_steps": 5}}]
        events = load_events_from_config(defs)
        assert len(events) == 1
        assert events[0].n_micro_steps == 5

    def test_unknown_type_raises(self):
        defs = [{"type": "nonexistent", "name": "bad"}]
        with pytest.raises(ValueError, match="Unknown event type"):
            load_events_from_config(defs)

    def test_missing_custom_callback_raises(self):
        defs = [{"type": "custom", "name": "missing"}]
        with pytest.raises(ValueError, match="No callback registered"):
            load_events_from_config(defs, {})

    def test_trigger_parsing(self):
        defs = [{"type": "custom", "name": "x", "trigger": {"type": "periodic", "interval": 5, "offset": 2}}]
        events = load_events_from_config(defs, {"x": lambda *a: None})
        assert events[0].trigger.should_fire(2) is True
        assert events[0].trigger.should_fire(3) is False
        assert events[0].trigger.should_fire(7) is True


# ---------------------------------------------------------------------------
# Phase 2 lifecycle events
# ---------------------------------------------------------------------------

class TestStageSpecificSurvival:
    @pytest.fixture
    def pop_with_stages(self):
        pool = AgentPool(n=100, start_tri=0, rng_seed=42)
        trait_defs = [TraitDefinition("stage", TraitType.PROBABILISTIC, ["juvenile", "adult"])]
        trait_mgr = TraitManager(100, trait_defs)
        trait_mgr._data["stage"][:50] = 0
        trait_mgr._data["stage"][50:] = 1
        pop = Population("fish", pool, trait_mgr=trait_mgr)
        return pop

    def test_juvenile_high_mortality(self, pop_with_stages):
        event = StageSpecificSurvivalEvent(name="survival", mortality_rates={"juvenile": 1.0, "adult": 0.0})
        mask = pop_with_stages.alive.copy()
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(pop_with_stages, landscape, t=0, mask=mask)
        assert pop_with_stages.pool.alive[:50].sum() == 0
        assert pop_with_stages.pool.alive[50:].sum() == 50

    def test_zero_mortality_preserves_all(self, pop_with_stages):
        event = StageSpecificSurvivalEvent(name="survival", mortality_rates={"juvenile": 0.0, "adult": 0.0})
        mask = pop_with_stages.alive.copy()
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(pop_with_stages, landscape, t=0, mask=mask)
        assert pop_with_stages.n_alive == 100


class TestIntroductionEvent:
    def test_adds_agents(self):
        pool = AgentPool(n=10, start_tri=0, rng_seed=42)
        pop = Population("fish", pool)
        event = IntroductionEvent(name="introduce", n_agents=5, positions=[3, 7])
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(pop, landscape, t=0, mask=pop.alive.copy())
        assert pop.n == 15
        assert pop.n_alive == 15

    def test_positions_are_set(self):
        pool = AgentPool(n=5, start_tri=0, rng_seed=42)
        pop = Population("fish", pool)
        event = IntroductionEvent(name="introduce", n_agents=3, positions=[10])
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(pop, landscape, t=0, mask=pop.alive.copy())
        assert (pop.pool.tri_idx[5:] == 10).all()

    def test_sets_initial_traits(self):
        pool = AgentPool(n=5, start_tri=0, rng_seed=42)
        trait_defs = [TraitDefinition("stage", TraitType.PROBABILISTIC, ["egg", "juvenile", "adult"])]
        trait_mgr = TraitManager(5, trait_defs)
        pop = Population("fish", pool, trait_mgr=trait_mgr)
        event = IntroductionEvent(name="introduce", n_agents=3, positions=[0], initial_traits={"stage": "juvenile"})
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(pop, landscape, t=0, mask=pop.alive.copy())
        assert (pop.trait_mgr._data["stage"][5:] == 1).all()


class TestReproductionEvent:
    @pytest.fixture
    def grouped_pop(self):
        pool = AgentPool(n=10, start_tri=5, rng_seed=42)
        trait_defs = [TraitDefinition("stage", TraitType.PROBABILISTIC, ["juvenile", "adult"])]
        trait_mgr = TraitManager(10, trait_defs)
        trait_mgr._data["stage"][:] = 1
        pop = Population("fish", pool, trait_mgr=trait_mgr)
        pop.group_id[:5] = 0
        pop.group_id[5:] = -1
        return pop

    def test_only_grouped_reproduce(self, grouped_pop):
        event = ReproductionEvent(name="repro", clutch_mean=2.0)
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(grouped_pop, landscape, t=0, mask=grouped_pop.alive.copy())
        assert grouped_pop.n > 10

    def test_offspring_inherit_position(self, grouped_pop):
        event = ReproductionEvent(name="repro", clutch_mean=1.0)
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(grouped_pop, landscape, t=0, mask=grouped_pop.alive.copy())
        assert (grouped_pop.pool.tri_idx[10:] == 5).all()

    def test_offspring_get_trait(self, grouped_pop):
        event = ReproductionEvent(name="repro", clutch_mean=1.0, offspring_trait_name="stage", offspring_trait_value="juvenile")
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(grouped_pop, landscape, t=0, mask=grouped_pop.alive.copy())
        assert (grouped_pop.trait_mgr._data["stage"][10:] == 0).all()

    def test_floaters_excluded(self, grouped_pop):
        grouped_pop.group_id[:] = -1
        event = ReproductionEvent(name="repro", clutch_mean=5.0)
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(grouped_pop, landscape, t=0, mask=grouped_pop.alive.copy())
        assert grouped_pop.n == 10


class TestFloaterCreationEvent:
    @pytest.fixture
    def all_grouped_pop(self):
        pool = AgentPool(n=20, start_tri=0, rng_seed=42)
        pop = Population("fish", pool)
        pop.group_id[:10] = 0
        pop.group_id[10:] = 1
        return pop

    def test_releases_some_agents(self, all_grouped_pop):
        event = FloaterCreationEvent(name="release", probability=0.5)
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(all_grouped_pop, landscape, t=0, mask=all_grouped_pop.alive.copy())
        n_floaters = (all_grouped_pop.group_id == -1).sum()
        assert 0 < n_floaters < 20

    def test_probability_zero_releases_none(self, all_grouped_pop):
        event = FloaterCreationEvent(name="release", probability=0.0)
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(all_grouped_pop, landscape, t=0, mask=all_grouped_pop.alive.copy())
        assert (all_grouped_pop.group_id >= 0).all()

    def test_probability_one_releases_all(self, all_grouped_pop):
        event = FloaterCreationEvent(name="release", probability=1.0)
        landscape = {"rng": np.random.default_rng(42)}
        event.execute(all_grouped_pop, landscape, t=0, mask=all_grouped_pop.alive.copy())
        assert (all_grouped_pop.group_id == -1).all()


class TestCensusEvent:
    def test_records_basic_counts(self):
        from salmon_ibm.agents import AgentPool
        pool = AgentPool(n=20, start_tri=0, rng_seed=42)
        pop = Population("fish", pool)
        pop.pool.alive[15:] = False  # 15 alive, 5 dead

        landscape = {}
        event = CensusEvent(name="census")
        mask = pop.alive.copy()
        event.execute(pop, landscape, t=5, mask=mask)

        assert len(landscape["census_records"]) == 1
        rec = landscape["census_records"][0]
        assert rec["time"] == 5
        assert rec["n_alive"] == 15
        assert rec["n_dead"] == 5
        assert rec["n_total"] == 20

    def test_records_trait_counts(self):
        from salmon_ibm.agents import AgentPool
        pool = AgentPool(n=10, start_tri=0, rng_seed=42)
        trait_defs = [TraitDefinition("stage", TraitType.PROBABILISTIC, ["juv", "adult"])]
        trait_mgr = TraitManager(10, trait_defs)
        trait_mgr._data["stage"][:6] = 0  # 6 juvenile
        trait_mgr._data["stage"][6:] = 1  # 4 adult
        pop = Population("fish", pool, trait_mgr=trait_mgr)

        landscape = {}
        event = CensusEvent(name="census", trait_names=["stage"])
        mask = pop.alive.copy()
        event.execute(pop, landscape, t=0, mask=mask)

        rec = landscape["census_records"][0]
        assert rec["trait_stage"]["juv"] == 6
        assert rec["trait_stage"]["adult"] == 4

    def test_multiple_timesteps(self):
        from salmon_ibm.agents import AgentPool
        pool = AgentPool(n=10, start_tri=0, rng_seed=42)
        pop = Population("fish", pool)
        landscape = {}
        event = CensusEvent(name="census")
        for t in range(5):
            event.execute(pop, landscape, t=t, mask=pop.alive.copy())
        assert len(landscape["census_records"]) == 5
