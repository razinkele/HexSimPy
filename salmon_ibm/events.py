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
