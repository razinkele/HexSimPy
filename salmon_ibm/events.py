"""Event engine: base classes, triggers, and sequencer."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

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
        for event in self.events:
            if event.trigger.should_fire(t):
                mask = self._compute_mask(population, event.trait_filter)
                event.execute(population, landscape, t, mask)

    @staticmethod
    def _compute_mask(population, trait_filter: dict | None) -> np.ndarray:
        base = population.alive & ~population.arrived
        if trait_filter is not None:
            pass  # Future: trait-based filtering
        return base


# ---------------------------------------------------------------------------
# Event Group
# ---------------------------------------------------------------------------

@dataclass
class EventGroup(Event):
    sub_events: list[Event] = field(default_factory=list)
    iterations: int = 1

    def execute(self, population, landscape, t, mask):
        for _ in range(self.iterations):
            for event in self.sub_events:
                if event.trigger.should_fire(t):
                    sub_mask = self._compute_sub_mask(population, event.trait_filter, mask)
                    event.execute(population, landscape, t, sub_mask)

    @staticmethod
    def _compute_sub_mask(population, trait_filter, parent_mask):
        child_mask = population.alive & ~population.arrived
        if trait_filter is not None:
            pass
        return parent_mask & child_mask


# ---------------------------------------------------------------------------
# Event Registry & YAML Loading
# ---------------------------------------------------------------------------

from typing import Callable

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
        return Periodic(interval=trigger_def["interval"], offset=trigger_def.get("offset", 0))
    elif kind == "window":
        return Window(start=trigger_def["start"], end=trigger_def["end"])
    elif kind == "random":
        return RandomTrigger(p=trigger_def["p"])
    else:
        raise ValueError(f"Unknown trigger type: {kind}")
