"""1D branching stream network for aquatic species."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np

from salmon_ibm.events import Event, register_event


@dataclass
class SegmentDefinition:
    """A single stream segment."""
    id: int
    length: float  # meters
    upstream_ids: list[int] = field(default_factory=list)
    downstream_ids: list[int] = field(default_factory=list)
    order: int = 1  # Strahler order


class StreamNetwork:
    """1D branching directional network topology."""

    def __init__(self, segments: list[SegmentDefinition]):
        self.segments = {s.id: s for s in segments}
        self._ids = [s.id for s in segments]
        self.n_segments = len(segments)

    def segment_length(self, seg_id: int) -> float:
        return self.segments[seg_id].length

    def upstream(self, seg_id: int) -> list[int]:
        return self.segments[seg_id].upstream_ids

    def downstream(self, seg_id: int) -> list[int]:
        return self.segments[seg_id].downstream_ids

    def is_headwater(self, seg_id: int) -> bool:
        return len(self.segments[seg_id].upstream_ids) == 0

    def is_outlet(self, seg_id: int) -> bool:
        return len(self.segments[seg_id].downstream_ids) == 0

    def all_upstream(self, seg_id: int) -> list[int]:
        """All segments upstream of seg_id (BFS)."""
        visited = set()
        queue = deque(self.upstream(seg_id))
        while queue:
            s = queue.popleft()
            if s not in visited:
                visited.add(s)
                queue.extend(self.upstream(s))
        return sorted(visited)

    def all_downstream(self, seg_id: int) -> list[int]:
        visited = set()
        queue = deque(self.downstream(seg_id))
        while queue:
            s = queue.popleft()
            if s not in visited:
                visited.add(s)
                queue.extend(self.downstream(s))
        return sorted(visited)


@dataclass
class NetworkPosition:
    """Position on a stream network: segment + offset along segment."""
    segment_id: int
    offset: float  # distance from start of segment [0, segment_length]


class NetworkMovement:
    """Move agents along a stream network."""

    def __init__(self, network: StreamNetwork, rng_seed: int | None = None):
        self.network = network
        self.rng = np.random.default_rng(rng_seed)

    def move_upstream(self, positions: list[NetworkPosition],
                      step_lengths: np.ndarray) -> list[NetworkPosition]:
        """Move agents upstream by given step lengths."""
        new_positions = []
        for pos, step in zip(positions, step_lengths):
            seg = self.network.segments[pos.segment_id]
            new_offset = pos.offset - step  # upstream = decreasing offset

            if new_offset >= 0:
                new_positions.append(NetworkPosition(pos.segment_id, new_offset))
            else:
                # Crossed segment boundary -- move to upstream segment
                remainder = abs(new_offset)
                us = self.network.upstream(pos.segment_id)
                if not us:
                    new_positions.append(NetworkPosition(pos.segment_id, 0.0))
                else:
                    chosen = us[0] if len(us) == 1 else self.rng.choice(us)
                    chosen_len = self.network.segment_length(chosen)
                    new_off = max(chosen_len - remainder, 0.0)
                    new_positions.append(NetworkPosition(chosen, new_off))
        return new_positions

    def move_downstream(self, positions: list[NetworkPosition],
                        step_lengths: np.ndarray) -> list[NetworkPosition]:
        """Move agents downstream by given step lengths."""
        new_positions = []
        for pos, step in zip(positions, step_lengths):
            seg = self.network.segments[pos.segment_id]
            new_offset = pos.offset + step

            if new_offset <= seg.length:
                new_positions.append(NetworkPosition(pos.segment_id, new_offset))
            else:
                remainder = new_offset - seg.length
                ds = self.network.downstream(pos.segment_id)
                if not ds:
                    new_positions.append(NetworkPosition(pos.segment_id, seg.length))
                else:
                    chosen = ds[0] if len(ds) == 1 else self.rng.choice(ds)
                    new_off = min(remainder, self.network.segment_length(chosen))
                    new_positions.append(NetworkPosition(chosen, new_off))
        return new_positions


@dataclass
class NetworkRange:
    """Territory on a stream network spanning one or more segments."""
    segments: list[int]
    start_offset: float
    end_offset: float

    def total_length(self, network: StreamNetwork) -> float:
        if len(self.segments) == 1:
            return self.end_offset - self.start_offset
        total = network.segment_length(self.segments[0]) - self.start_offset
        for seg in self.segments[1:-1]:
            total += network.segment_length(seg)
        total += self.end_offset
        return total


class NetworkRangeManager:
    """Manage territory allocation on a stream network."""

    def __init__(self, network: StreamNetwork):
        self.network = network
        self._occupied: dict[int, int] = {}  # segment_id -> owner agent index

    def is_available(self, segment_id: int) -> bool:
        return segment_id not in self._occupied

    def allocate(self, agent_idx: int, segment_id: int) -> bool:
        if not self.is_available(segment_id):
            return False
        self._occupied[segment_id] = agent_idx
        return True

    def release(self, agent_idx: int) -> None:
        to_remove = [k for k, v in self._occupied.items() if v == agent_idx]
        for k in to_remove:
            del self._occupied[k]

    def owner_of(self, segment_id: int) -> int | None:
        return self._occupied.get(segment_id)


@register_event("switch_population")
@dataclass
class SwitchPopulationEvent(Event):
    """Transfer agents between grid and network populations."""
    source_pop: str = ""
    target_pop: str = ""
    transfer_probability: float = 0.1

    def execute(self, population, landscape, t, mask):
        rng = landscape.get("rng", np.random.default_rng())
        multi_pop = landscape.get("multi_pop_mgr")
        if multi_pop is None:
            return
        source = multi_pop.get(self.source_pop)
        target = multi_pop.get(self.target_pop)
        if source is None or target is None:
            return

        candidates = np.where(mask & source.alive)[0]
        if len(candidates) == 0:
            return
        rolls = rng.random(len(candidates))
        transfer = candidates[rolls < self.transfer_probability]
        if len(transfer) == 0:
            return

        positions = source.tri_idx[transfer]
        target.add_agents(len(transfer), positions)
        source.alive[transfer] = False
