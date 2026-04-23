"""Event engine: base classes, triggers, and sequencer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


class EventTrigger(ABC):
    @abstractmethod
    def should_fire(self, t: int) -> bool: ...


class EveryStep(EventTrigger):
    def should_fire(self, t: int) -> bool:
        return True


@dataclass
class Once(EventTrigger):
    at: int

    def should_fire(self, t: int) -> bool:
        return t == self.at


@dataclass
class Periodic(EventTrigger):
    interval: int
    offset: int = 0

    def __post_init__(self):
        if self.interval <= 0:
            raise ValueError(
                f"Periodic trigger interval must be > 0, got {self.interval}"
            )

    def should_fire(self, t: int) -> bool:
        return (t - self.offset) % self.interval == 0 and t >= self.offset


@dataclass
class Window(EventTrigger):
    start: int
    end: int

    def should_fire(self, t: int) -> bool:
        return self.start <= t < self.end


@dataclass
class RandomTrigger(EventTrigger):
    p: float
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(), repr=False
    )

    def should_fire(self, t: int) -> bool:
        return self._rng.random() < self.p


# ---------------------------------------------------------------------------
# Event base class
# ---------------------------------------------------------------------------


@dataclass
class Event(ABC):
    name: str
    trigger: EventTrigger = field(default_factory=EveryStep)
    trait_filter: dict | None = None
    population_name: str | None = None
    enabled: bool = True

    @abstractmethod
    def execute(self, population, landscape, t: int, mask: np.ndarray) -> None: ...


# ---------------------------------------------------------------------------
# Event Sequencer
# ---------------------------------------------------------------------------


class EventSequencer:
    """Executes a list of events in order each timestep."""

    def __init__(self, events: list[Event]):
        self.events = events

    def step(self, population, landscape, t: int) -> None:
        landscape["step_alive_mask"] = population.alive & ~population.arrived
        # Clear per-step caches (parity with MultiPopEventSequencer)
        from salmon_ibm.events_hexsim import clear_combo_mask_cache

        clear_combo_mask_cache()
        for event in self.events:
            if event.trigger.should_fire(t):
                mask = self._compute_mask(population, event.trait_filter)
                event.execute(population, landscape, t, mask)

    @staticmethod
    def _compute_mask(population, trait_filter: dict | None) -> np.ndarray:
        base = population.alive & ~population.arrived
        if (
            trait_filter is not None
            and hasattr(population, "trait_mgr")
            and population.trait_mgr is not None
        ):
            trait_mask = population.trait_mgr.filter_by_traits(**trait_filter)
            base = base & trait_mask
        elif (
            trait_filter is not None
            and hasattr(population, "traits")
            and population.traits is not None
        ):
            trait_mask = population.traits.filter_by_traits(**trait_filter)
            base = base & trait_mask
        return base


# ---------------------------------------------------------------------------
# Multi-Population Event Sequencer
# ---------------------------------------------------------------------------


class MultiPopEventSequencer:
    """Executes events, routing each to its target population."""

    def __init__(self, events, multi_pop_mgr):
        self.events = events
        self.multi_pop = multi_pop_mgr

    def step(self, landscape, t):
        landscape["multi_pop_mgr"] = self.multi_pop
        # Clear per-step caches
        from salmon_ibm.events_hexsim import clear_combo_mask_cache

        clear_combo_mask_cache()
        for event in self.events:
            if not getattr(event, "enabled", True):
                continue
            if not event.trigger.should_fire(t):
                continue
            pop_name = getattr(event, "population_name", None)
            if pop_name:
                population = self.multi_pop.get(pop_name)
                mask = population.alive & ~population.arrived
            else:
                population = None
                mask = np.ones(0, dtype=bool)
            event.execute(population, landscape, t, mask)


# ---------------------------------------------------------------------------
# Event Group
# ---------------------------------------------------------------------------


@dataclass
class EventGroup(Event):
    sub_events: list[Event] = field(default_factory=list)
    iterations: int = 1

    def execute(self, population, landscape, t, mask):
        multi_pop = landscape.get("multi_pop_mgr")

        # Pre-filter: resolve triggers, populations, trait_mgr, and filter
        # type ONCE before the iteration loop. This eliminates all getattr,
        # hasattr, isinstance, and import calls from the hot inner loop.
        from salmon_ibm.events_hexsim import _apply_trait_combo_mask

        prepared = []
        for event in self.sub_events:
            if not getattr(event, "enabled", True):
                continue
            if not event.trigger.should_fire(t):
                continue
            child_pop = population
            child_pop_name = getattr(event, "population_name", None)
            if child_pop_name and multi_pop:
                child_pop = multi_pop.get(child_pop_name)
            tf = getattr(event, "trait_filter", None)
            # Pre-resolve filter type: 0=none, 1=combo, 2=simple
            tf_type = 0
            trait_mgr = None
            if tf is not None and child_pop is not None:
                trait_mgr = getattr(child_pop, "trait_mgr", None)
                if trait_mgr is not None:
                    if isinstance(tf, dict) and "traits" in tf:
                        tf_type = 1  # combo
                    elif isinstance(tf, dict):
                        tf_type = 2  # simple
            prepared.append((event, child_pop, tf, tf_type, trait_mgr))

        if not prepared:
            return

        _empty_mask = np.ones(0, dtype=bool)

        for _iter in range(self.iterations):
            # Cache base alive mask per population per iteration
            _mask_cache: dict[int, np.ndarray] = {}
            for event, child_pop, tf, tf_type, trait_mgr in prepared:
                if child_pop is not None:
                    pop_id = id(child_pop)
                    if pop_id not in _mask_cache:
                        _mask_cache[pop_id] = child_pop.alive & ~child_pop.arrived
                    child_mask = _mask_cache[pop_id]
                    if tf_type == 1:
                        child_mask = _apply_trait_combo_mask(child_mask, tf, child_pop)
                    elif tf_type == 2:
                        child_mask = child_mask & trait_mgr.filter_by_traits(**tf)
                else:
                    child_mask = _empty_mask
                event.execute(child_pop, landscape, t, child_mask)
            _mask_cache.clear()  # allow GC


# ---------------------------------------------------------------------------
# Event Registry & YAML Loading
# ---------------------------------------------------------------------------


EVENT_REGISTRY: dict[str, type[Event]] = {}


def register_event(type_name: str):
    """Decorator to register an Event subclass under a type name."""

    def decorator(cls):
        EVENT_REGISTRY[type_name] = cls
        return cls

    return decorator


def load_events_from_config(event_defs, callback_registry=None):
    """Build event list from YAML definitions."""
    events = []
    callback_registry = callback_registry or {}
    for defn in event_defs:
        event_type = defn["type"]
        name = defn.get("name", event_type)
        params = defn.get("params", {})
        trigger = _parse_trigger(defn.get("trigger"))
        if event_type == "custom":
            cb = callback_registry.get(name)
            if cb is None:
                raise ValueError(
                    f"No callback registered for custom event '{name}'. "
                    f"Available: {list(callback_registry.keys())}"
                )
            from salmon_ibm.events_builtin import CustomEvent

            events.append(CustomEvent(name=name, trigger=trigger, callback=cb))
        elif event_type in EVENT_REGISTRY:
            cls = EVENT_REGISTRY[event_type]
            events.append(cls(name=name, trigger=trigger, **params))
        else:
            raise ValueError(
                f"Unknown event type '{event_type}'. "
                f"Registered types: {list(EVENT_REGISTRY.keys())}"
            )
    return events


def _parse_trigger(trigger_def):
    """Parse a trigger definition from YAML."""
    if trigger_def is None:
        return EveryStep()
    kind = trigger_def.get("type", "every_step")
    if kind == "every_step":
        return EveryStep()
    elif kind == "once":
        return Once(at=trigger_def["at"])
    elif kind == "periodic":
        return Periodic(
            interval=trigger_def["interval"], offset=trigger_def.get("offset", 0)
        )
    elif kind == "window":
        return Window(start=trigger_def["start"], end=trigger_def["end"])
    elif kind == "random":
        return RandomTrigger(p=trigger_def["p"])
    else:
        raise ValueError(f"Unknown trigger type: {kind}")
