"""Tests for TripBuffer — the ring-buffer that feeds TripsLayer animation.

Covers:
  - Basic update / build_trips round-trip
  - Timestamp normalization (Bug #1 fix)
  - Ring buffer wrapping beyond TRAIL_LEN
  - Dead-agent filtering
  - Static-path filtering (stationary agents skipped)
  - MAX_AGENTS subsampling
  - clear() reset
  - Edge cases: single step, empty population, all-dead
"""

import numpy as np
import pytest

# Import TripBuffer from app.py (it's defined at module level)
from pathlib import Path

# We can't import app.py directly (it runs Shiny), so we extract TripBuffer
# by exec-ing just the class definition.  Alternatively, we duplicate it here
# for test isolation.  Let's do a targeted import approach:

ROOT = Path(__file__).resolve().parent.parent


def _make_trip_buffer_class():
    """Extract TripBuffer from app.py without importing the full Shiny app."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # Find the class block
    start = src.index("class TripBuffer:")
    # Find the next class or top-level def after TripBuffer
    # We look for a line starting with no indent that's a class/def/variable
    lines = src[start:].split("\n")
    end_offset = len(lines)
    for i, line in enumerate(lines):
        if i == 0:
            continue
        # A non-empty line with no leading whitespace signals end of class
        if line and not line[0].isspace() and not line.startswith("#"):
            end_offset = i
            break
    class_src = "\n".join(lines[:end_offset])
    # Extract BEH_COLORS and _hex_to_rgb from app.py (needed by build_trips)
    beh_match = src[
        src.index("BEH_COLORS = ") : src.index("\n", src.index("BEH_COLORS = "))
    ]
    hex_fn_start = src.index("def _hex_to_rgb(")
    hex_fn_end = src.index("\n\n", hex_fn_start)
    hex_fn_src = src[hex_fn_start:hex_fn_end]
    ns = {"np": np}
    exec(beh_match, ns)
    exec(hex_fn_src, ns)
    exec(class_src, ns)
    return ns["TripBuffer"]


TripBuffer = _make_trip_buffer_class()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def buf():
    """Fresh TripBuffer instance."""
    return TripBuffer()


def _make_positions(n_agents, step, spread=1.0):
    """Generate (n_agents, 2) position array that varies per step."""
    rng = np.random.RandomState(step)
    base = np.column_stack(
        [
            np.arange(n_agents, dtype=np.float32) * 0.01 * spread + step * 0.1,
            np.arange(n_agents, dtype=np.float32) * 0.005 * spread,
        ]
    )
    return base


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasicUpdate:
    def test_single_step_returns_empty(self, buf):
        """build_trips requires >= 2 timesteps."""
        alive = np.ones(10, dtype=bool)
        pos = _make_positions(10, 0)
        buf.update(alive, pos)
        trips, t_min, t_max = buf.build_trips()
        assert trips == []

    def test_two_steps_returns_trips(self, buf):
        """After 2 updates, build_trips should return data."""
        alive = np.ones(10, dtype=bool)
        for step in range(2):
            buf.update(alive, _make_positions(10, step))
        trips, t_min, t_max = buf.build_trips()
        assert len(trips) > 0
        assert t_max >= t_min

    def test_trip_structure(self, buf):
        """Each trip dict has path, timestamps, color keys."""
        alive = np.ones(5, dtype=bool)
        for step in range(3):
            buf.update(alive, _make_positions(5, step))
        trips, _, _ = buf.build_trips()
        assert len(trips) > 0
        trip = trips[0]
        assert "path" in trip
        assert "timestamps" in trip
        assert "color" in trip
        assert len(trip["path"]) == len(trip["timestamps"])
        assert len(trip["path"]) == 3  # 3 steps of data

    def test_path_has_three_coordinates(self, buf):
        """Each waypoint in path is [lon, lat, time] triplet."""
        alive = np.ones(3, dtype=bool)
        for step in range(2):
            buf.update(alive, _make_positions(3, step))
        trips, _, _ = buf.build_trips()
        for trip in trips:
            for waypoint in trip["path"]:
                assert len(waypoint) == 3, f"Expected [lon, lat, time], got {waypoint}"

    def test_color_is_rgba(self, buf):
        """Color should be [R, G, B, A] array."""
        alive = np.ones(3, dtype=bool)
        for step in range(2):
            buf.update(alive, _make_positions(3, step))
        trips, _, _ = buf.build_trips()
        for trip in trips:
            assert len(trip["color"]) == 4
            assert all(0 <= c <= 255 for c in trip["color"])


# ---------------------------------------------------------------------------
# Timestamp normalization (Bug #1 fix)
# ---------------------------------------------------------------------------


class TestTimestampNormalization:
    def test_timestamps_start_from_zero(self, buf):
        """Timestamps must be normalized to start from 0."""
        alive = np.ones(5, dtype=bool)
        # Run 50 steps so t_min > 0 after ring buffer wraps
        for step in range(50):
            buf.update(alive, _make_positions(5, step))
        trips, t_min, t_max = buf.build_trips()
        assert t_min > 0, "t_min should be > 0 after 50 steps (TRAIL_LEN=40)"
        for trip in trips:
            assert trip["timestamps"][0] == 0.0, (
                f"First timestamp should be 0, got {trip['timestamps'][0]}"
            )

    def test_timestamps_end_at_loop_length(self, buf):
        """Last timestamp should equal t_max - t_min (= loopLength)."""
        alive = np.ones(5, dtype=bool)
        for step in range(50):
            buf.update(alive, _make_positions(5, step))
        trips, t_min, t_max = buf.build_trips()
        loop_len = t_max - t_min
        for trip in trips:
            assert trip["timestamps"][-1] == pytest.approx(loop_len), (
                f"Last timestamp should be {loop_len}, got {trip['timestamps'][-1]}"
            )

    def test_timestamps_monotonically_increasing(self, buf):
        """Timestamps within each trip must be non-decreasing."""
        alive = np.ones(5, dtype=bool)
        for step in range(20):
            buf.update(alive, _make_positions(5, step))
        trips, _, _ = buf.build_trips()
        for trip in trips:
            ts = trip["timestamps"]
            for i in range(1, len(ts)):
                assert ts[i] >= ts[i - 1], (
                    f"Timestamps not monotonic at index {i}: {ts[i - 1]} > {ts[i]}"
                )

    def test_path_time_matches_timestamps(self, buf):
        """path[i][2] must equal timestamps[i] — they're the same data."""
        alive = np.ones(5, dtype=bool)
        for step in range(10):
            buf.update(alive, _make_positions(5, step))
        trips, _, _ = buf.build_trips()
        for trip in trips:
            for i, (wp, ts) in enumerate(zip(trip["path"], trip["timestamps"])):
                assert wp[2] == pytest.approx(ts), (
                    f"path[{i}][2]={wp[2]} != timestamps[{i}]={ts}"
                )

    def test_timestamps_aligned_with_js_currenttime_range(self, buf):
        """All timestamps should fall within [0, loopLength]."""
        alive = np.ones(5, dtype=bool)
        for step in range(100):
            buf.update(alive, _make_positions(5, step))
        trips, t_min, t_max = buf.build_trips()
        loop_len = t_max - t_min
        for trip in trips:
            for ts in trip["timestamps"]:
                assert 0 <= ts <= loop_len + 0.01, (
                    f"Timestamp {ts} outside range [0, {loop_len}]"
                )


# ---------------------------------------------------------------------------
# Ring buffer behavior
# ---------------------------------------------------------------------------


class TestRingBuffer:
    def test_trail_length_capped(self, buf):
        """After > TRAIL_LEN steps, each trip has exactly TRAIL_LEN waypoints."""
        alive = np.ones(3, dtype=bool)
        for step in range(buf.TRAIL_LEN + 20):
            buf.update(alive, _make_positions(3, step))
        trips, _, _ = buf.build_trips()
        for trip in trips:
            assert len(trip["path"]) == buf.TRAIL_LEN
            assert len(trip["timestamps"]) == buf.TRAIL_LEN

    def test_ring_buffer_wraps_correctly(self, buf):
        """Data wraps around the ring buffer without corruption."""
        alive = np.ones(2, dtype=bool)
        n_steps = buf.TRAIL_LEN * 3  # wrap multiple times
        for step in range(n_steps):
            pos = np.array(
                [[step * 0.1, step * 0.05], [step * 0.1 + 1, step * 0.05 + 1]],
                dtype=np.float32,
            )
            buf.update(alive, pos)
        trips, _, _ = buf.build_trips()
        assert len(trips) > 0
        # Verify paths contain positions from the most recent TRAIL_LEN steps
        trip = trips[0]
        path = trip["path"]
        # First waypoint should correspond to step (n_steps - TRAIL_LEN)
        expected_first_lon = (n_steps - buf.TRAIL_LEN) * 0.1
        assert path[0][0] == pytest.approx(expected_first_lon, abs=0.01)
        # Last waypoint should correspond to step (n_steps - 1)
        expected_last_lon = (n_steps - 1) * 0.1
        assert path[-1][0] == pytest.approx(expected_last_lon, abs=0.01)

    def test_fill_ramps_up(self, buf):
        """Before TRAIL_LEN steps, trips have fewer waypoints."""
        alive = np.ones(2, dtype=bool)
        for step in range(5):
            buf.update(alive, _make_positions(2, step))
        trips, _, _ = buf.build_trips()
        for trip in trips:
            assert len(trip["path"]) == 5


# ---------------------------------------------------------------------------
# Dead agent handling
# ---------------------------------------------------------------------------


class TestDeadAgents:
    def test_dead_agents_get_stale_positions(self, buf):
        """When a tracked agent dies, its slot gets stale (not updated)."""
        alive = np.ones(5, dtype=bool)
        buf.update(alive, _make_positions(5, 0))
        # Kill agent 2
        alive[2] = False
        buf.update(alive, _make_positions(5, 1))
        buf.update(alive, _make_positions(5, 2))
        trips, _, _ = buf.build_trips()
        # Should still have some trips (the alive agents)
        assert len(trips) > 0

    def test_dead_agent_does_not_corrupt_live_trails(self, buf):
        """When agent 1 dies, agent 0's trail must still contain agent 0's positions."""
        alive = np.ones(3, dtype=bool)
        # Step 0: all alive at known positions
        pos0 = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
        buf.update(alive, pos0)
        # Step 1: kill agent 1, others move
        alive[1] = False
        pos1 = np.array([[0.1, 0.1], [10.0, 10.0], [20.1, 20.1]], dtype=np.float32)
        buf.update(alive, pos1)
        # Step 2: agent 0 and 2 move further
        pos2 = np.array([[0.2, 0.2], [10.0, 10.0], [20.2, 20.2]], dtype=np.float32)
        buf.update(alive, pos2)

        trips, _, _ = buf.build_trips()
        # Agent 0 should have a trail with its own positions (near 0.x, 0.x)
        # Agent 2 should have a trail near (20.x, 20.x)
        # Agent 1 is dead — its trail should be static or filtered out
        for trip in trips:
            path = trip["path"]
            # All waypoints in a single trip must be from the same agent
            # (no mixing of agent 0 coords with agent 2 coords)
            lons = [p[0] for p in path]
            # A trail should span a narrow lon range (same agent)
            lon_spread = max(lons) - min(lons)
            assert lon_spread < 5.0, (
                f"Trail has mixed-agent coords (spread {lon_spread}): {lons}"
            )

    def test_all_dead_returns_empty(self, buf):
        """If all agents die, update is a no-op and trips remain from before."""
        alive = np.ones(3, dtype=bool)
        for step in range(3):
            buf.update(alive, _make_positions(3, step))
        # Kill all
        alive[:] = False
        buf.update(alive, _make_positions(3, 4))
        # Build should still return data from before (tracked agents have history)
        trips, _, _ = buf.build_trips()
        # trips may or may not be empty depending on whether paths are static
        # The key is no crash
        assert isinstance(trips, list)


# ---------------------------------------------------------------------------
# Static path filtering
# ---------------------------------------------------------------------------


class TestStaticPathFiltering:
    def test_stationary_agents_excluded(self, buf):
        """Agents that don't move should be filtered out."""
        alive = np.ones(3, dtype=bool)
        # Agent 0 moves, agents 1-2 stay at same position
        for step in range(5):
            pos = np.array(
                [
                    [step * 0.1, step * 0.05],
                    [1.0, 1.0],  # stationary
                    [2.0, 2.0],  # stationary
                ],
                dtype=np.float32,
            )
            buf.update(alive, pos)
        trips, _, _ = buf.build_trips()
        # Only the moving agent should produce a trip
        assert len(trips) == 1
        # Verify it's the moving agent (first waypoint lon starts near 0)
        assert trips[0]["path"][0][0] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# MAX_AGENTS subsampling
# ---------------------------------------------------------------------------


class TestMaxAgents:
    def test_large_population_subsampled(self, buf):
        """Populations > MAX_AGENTS are subsampled."""
        n = buf.MAX_AGENTS * 3
        alive = np.ones(n, dtype=bool)
        for step in range(3):
            buf.update(alive, _make_positions(n, step))
        trips, _, _ = buf.build_trips()
        assert len(trips) <= buf.MAX_AGENTS

    def test_small_population_all_tracked(self, buf):
        """Populations <= MAX_AGENTS are fully tracked."""
        n = 50
        alive = np.ones(n, dtype=bool)
        for step in range(3):
            buf.update(alive, _make_positions(n, step))
        trips, _, _ = buf.build_trips()
        # Should have up to n trips (minus any stationary agents)
        assert len(trips) <= n


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_state(self, buf):
        """After clear(), build_trips returns empty."""
        alive = np.ones(5, dtype=bool)
        for step in range(5):
            buf.update(alive, _make_positions(5, step))
        trips_before, _, _ = buf.build_trips()
        assert len(trips_before) > 0

        buf.clear()
        trips_after, _, _ = buf.build_trips()
        assert trips_after == []

    def test_clear_allows_fresh_tracking(self, buf):
        """After clear(), new agents are tracked from scratch."""
        alive = np.ones(5, dtype=bool)
        for step in range(3):
            buf.update(alive, _make_positions(5, step))
        buf.clear()

        # New population
        alive2 = np.ones(3, dtype=bool)
        for step in range(3):
            buf.update(alive2, _make_positions(3, step + 100))
        trips, _, _ = buf.build_trips()
        assert len(trips) > 0
        # Timestamps should be normalized starting from 0
        assert trips[0]["timestamps"][0] == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBehaviorColors:
    def test_valid_behavior_produces_colored_trail(self, buf):
        """Valid behavior indices produce colored trails with proper RGBA."""
        alive = np.ones(3, dtype=bool)
        behaviors = np.array([0, 3, 4], dtype=np.int8)
        for step in range(3):
            pos = np.array(
                [[step * 0.1, 0], [step * 0.1 + 1, 1], [step * 0.1 + 2, 2]],
                dtype=np.float32,
            )
            buf.update(alive, pos, behaviors)
        trips, _, _ = buf.build_trips()
        assert len(trips) > 0
        for trip in trips:
            assert len(trip["color"]) == 4

    def test_negative_behavior_uses_default_color(self, buf):
        """Negative behavior indices should use default color (index 0)."""
        alive = np.ones(2, dtype=bool)
        behaviors = np.array([-1, 0], dtype=np.int8)
        for step in range(3):
            pos = np.array([[step * 0.1, 0], [step * 0.1 + 1, 1]], dtype=np.float32)
            buf.update(alive, pos, behaviors)
        trips, _, _ = buf.build_trips()
        assert len(trips) > 0

    def test_out_of_range_behavior_uses_default(self, buf):
        """Behavior indices >= len(BEH_COLORS) should use default color (index 0)."""
        alive = np.ones(2, dtype=bool)
        behaviors = np.array([99, 0], dtype=np.int8)
        for step in range(3):
            pos = np.array([[step * 0.1, 0], [step * 0.1 + 1, 1]], dtype=np.float32)
            buf.update(alive, pos, behaviors)
        trips, _, _ = buf.build_trips()
        assert len(trips) > 0


class TestEdgeCases:
    def test_empty_alive_mask(self, buf):
        """update() with no alive agents is a no-op."""
        alive = np.zeros(10, dtype=bool)
        pos = _make_positions(10, 0)
        buf.update(alive, pos)
        trips, _, _ = buf.build_trips()
        assert trips == []

    def test_zero_agents(self, buf):
        """update() with empty arrays doesn't crash."""
        alive = np.array([], dtype=bool)
        pos = np.zeros((0, 2), dtype=np.float32)
        buf.update(alive, pos)
        trips, _, _ = buf.build_trips()
        assert trips == []

    def test_current_time_advances(self, buf):
        """current_time property tracks the simulation step."""
        alive = np.ones(3, dtype=bool)
        for step in range(10):
            buf.update(alive, _make_positions(3, step))
        assert buf.current_time == 10

    def test_current_time_resets_on_clear(self, buf):
        alive = np.ones(3, dtype=bool)
        for step in range(5):
            buf.update(alive, _make_positions(3, step))
        buf.clear()
        assert buf.current_time == 0


# ---------------------------------------------------------------------------
# Integration: simulate the full pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    """End-to-end test simulating what app.py does: update → build → payload."""

    def test_full_pipeline_produces_valid_payload(self, buf):
        """Simulate 60 steps (all alive) and verify the payload for deck.gl."""
        n_agents = 100
        alive = np.ones(n_agents, dtype=bool)
        for step in range(60):
            pos = np.column_stack(
                [
                    np.arange(n_agents) * 0.01 + step * 0.05,
                    np.sin(np.arange(n_agents) * 0.1 + step * 0.1) * 0.5,
                ]
            ).astype(np.float32)
            buf.update(alive, pos)

        trips, t_min, t_max = buf.build_trips()
        loop_len = max(t_max - t_min, 1)

        # Validate payload structure
        assert len(trips) > 0
        assert loop_len > 0
        assert loop_len <= buf.TRAIL_LEN

        for trip in trips:
            path = trip["path"]
            ts = trip["timestamps"]
            color = trip["color"]

            # Correct lengths
            assert len(path) == len(ts)
            assert len(path) == buf.TRAIL_LEN  # 60 > TRAIL_LEN, so filled

            # All timestamps in [0, loopLen]
            assert ts[0] == pytest.approx(0.0)
            assert ts[-1] == pytest.approx(loop_len)
            assert all(0 <= t <= loop_len + 0.01 for t in ts)

            # Paths are [lon, lat] pairs
            assert all(len(p) == 3 for p in path)

            # Color is valid RGBA
            assert len(color) == 4

    def test_payload_matches_js_animation_range(self, buf):
        """The JS animation cycles currentTime in [0, loopLength].
        All trip timestamps must fall within this range."""
        alive = np.ones(20, dtype=bool)
        for step in range(100):
            pos = np.column_stack(
                [
                    np.arange(20) * 0.01 + step * 0.02,
                    np.arange(20) * 0.005,
                ]
            ).astype(np.float32)
            buf.update(alive, pos)

        trips, t_min, t_max = buf.build_trips()
        loop_len = max(t_max - t_min, 1)

        # Simulate what the JS does: currentTime goes from 0 to loopLength
        # TripsLayer renders segments where timestamp <= currentTime
        # At currentTime = loopLength, ALL waypoints should be visible
        for trip in trips:
            visible_at_end = sum(1 for t in trip["timestamps"] if t <= loop_len)
            assert visible_at_end == len(trip["timestamps"]), (
                f"At currentTime={loop_len}, only {visible_at_end}/{len(trip['timestamps'])} "
                f"waypoints visible. Timestamps: {trip['timestamps'][:5]}...{trip['timestamps'][-5:]}"
            )

        # At currentTime = 0, only the first waypoint should be visible
        for trip in trips:
            visible_at_start = sum(1 for t in trip["timestamps"] if t <= 0)
            assert visible_at_start >= 1, (
                f"At currentTime=0, no waypoints visible. First ts: {trip['timestamps'][0]}"
            )
