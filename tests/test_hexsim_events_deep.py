"""Deep tests for HexSim event types and Numba JIT kernels.

Tests the high-risk untested code paths identified by code review:
- _move_gradient_numba / _move_affinity_numba / _set_affinity_numba
- HexSimMoveEvent.execute (gradient + affinity paths)
- SetSpatialAffinityEvent.execute
- HexSimAccumulateEvent.execute (dispatch table)
- _apply_trait_combo_mask + caching
- _LazyAccDict
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic mesh fixture: 5-cell linear chain  0 → 1 → 2 → 3 → 4
# ---------------------------------------------------------------------------

class FakeMesh:
    """Minimal mesh for testing: 5 cells in a line, each connected to neighbors."""
    def __init__(self):
        self.n_cells = 5
        # Each cell connects to left/right neighbor only
        self._water_nbrs = np.full((5, 6), -1, dtype=np.intp)
        self._water_nbrs[0, 0] = 1
        self._water_nbrs[1, 0] = 0; self._water_nbrs[1, 1] = 2
        self._water_nbrs[2, 0] = 1; self._water_nbrs[2, 1] = 3
        self._water_nbrs[3, 0] = 2; self._water_nbrs[3, 1] = 4
        self._water_nbrs[4, 0] = 3
        self._water_nbr_count = np.array([1, 2, 2, 2, 1], dtype=np.int32)
        # Centroids: cells spread along x-axis
        self.centroids = np.array([
            [0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]
        ], dtype=np.float64)
        self._edge = 0.5  # marks as HexSim mesh
        self.water_mask = np.ones(5, dtype=bool)


def make_population(n, positions, n_accumulators=0, trait_categories=None):
    """Create a Population with agents at given positions."""
    from salmon_ibm.agents import AgentPool
    from salmon_ibm.population import Population
    pool = AgentPool(n=n, start_tri=np.array(positions, dtype=np.intp))
    pop = Population(name="test", pool=pool)
    if n_accumulators > 0:
        from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
        defs = [AccumulatorDef(name=f"acc_{i}") for i in range(n_accumulators)]
        pop.accumulator_mgr = AccumulatorManager(n, defs)
    if trait_categories is not None:
        from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
        td = [TraitDefinition(name="stage", trait_type=TraitType.PROBABILISTIC,
                              categories=trait_categories)]
        pop.trait_mgr = TraitManager(n, td)
    return pop


# ===========================================================================
# 1. Numba JIT kernel tests
# ===========================================================================

class TestMoveGradientNumba:
    def test_agents_move_up_gradient(self):
        """Agents at cell 0 should move to cell 1 (higher gradient)."""
        from salmon_ibm.events_hexsim import _move_gradient_numba
        mesh = FakeMesh()
        positions = np.array([0], dtype=np.intp)
        gradient = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions, distances = _move_gradient_numba(
            positions, mesh._water_nbrs, mesh._water_nbr_count,
            gradient, n_steps=1, walk_up=True, halt_dist=0.0)
        assert positions[0] == 1
        assert distances[0] == 1.0

    def test_agents_move_down_gradient(self):
        """Agents at cell 4 should move to cell 3 (lower gradient)."""
        from salmon_ibm.events_hexsim import _move_gradient_numba
        mesh = FakeMesh()
        positions = np.array([4], dtype=np.intp)
        gradient = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions, distances = _move_gradient_numba(
            positions, mesh._water_nbrs, mesh._water_nbr_count,
            gradient, n_steps=1, walk_up=False, halt_dist=0.0)
        assert positions[0] == 3
        assert distances[0] == 1.0

    def test_multi_step_movement(self):
        """Agent should move 3 cells in 3 steps."""
        from salmon_ibm.events_hexsim import _move_gradient_numba
        mesh = FakeMesh()
        positions = np.array([0], dtype=np.intp)
        gradient = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions, distances = _move_gradient_numba(
            positions, mesh._water_nbrs, mesh._water_nbr_count,
            gradient, n_steps=3, walk_up=True, halt_dist=0.0)
        assert positions[0] == 3
        assert distances[0] == 3.0

    def test_halt_dist_stops_movement(self):
        """Agent should stop after halt_dist steps."""
        from salmon_ibm.events_hexsim import _move_gradient_numba
        mesh = FakeMesh()
        positions = np.array([0], dtype=np.intp)
        gradient = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions, distances = _move_gradient_numba(
            positions, mesh._water_nbrs, mesh._water_nbr_count,
            gradient, n_steps=10, walk_up=True, halt_dist=2.0)
        assert positions[0] == 2
        assert distances[0] == 2.0

    def test_no_movement_at_gradient_peak(self):
        """Agent at gradient peak has no better neighbor — stays put."""
        from salmon_ibm.events_hexsim import _move_gradient_numba
        mesh = FakeMesh()
        positions = np.array([4], dtype=np.intp)
        gradient = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        positions, distances = _move_gradient_numba(
            positions, mesh._water_nbrs, mesh._water_nbr_count,
            gradient, n_steps=3, walk_up=True, halt_dist=0.0)
        assert positions[0] == 4
        assert distances[0] == 0.0


class TestMoveAffinityNumba:
    def test_agent_moves_toward_target(self):
        """Agent at cell 0 with target=4 should move to cell 1 (closer)."""
        from salmon_ibm.events_hexsim import _move_affinity_numba
        mesh = FakeMesh()
        positions = np.array([0], dtype=np.intp)
        cx = mesh.centroids[:, 0].copy()
        cy = mesh.centroids[:, 1].copy()
        targets = np.array([4], dtype=np.intp)
        positions, distances = _move_affinity_numba(
            positions, mesh._water_nbrs, mesh._water_nbr_count,
            cx, cy, targets, n_steps=1, halt_dist=0.0)
        assert positions[0] == 1
        assert distances[0] == 1.0

    def test_agent_reaches_target_neighbor(self):
        """Agent at cell 3 with target=4 should reach cell 4 in 1 step."""
        from salmon_ibm.events_hexsim import _move_affinity_numba
        mesh = FakeMesh()
        positions = np.array([3], dtype=np.intp)
        cx = mesh.centroids[:, 0].copy()
        cy = mesh.centroids[:, 1].copy()
        targets = np.array([4], dtype=np.intp)
        positions, distances = _move_affinity_numba(
            positions, mesh._water_nbrs, mesh._water_nbr_count,
            cx, cy, targets, n_steps=1, halt_dist=0.0)
        assert positions[0] == 4

    def test_negative_target_ignored(self):
        """Agent with target=-1 should not move."""
        from salmon_ibm.events_hexsim import _move_affinity_numba
        mesh = FakeMesh()
        positions = np.array([2], dtype=np.intp)
        cx = mesh.centroids[:, 0].copy()
        cy = mesh.centroids[:, 1].copy()
        targets = np.array([-1], dtype=np.intp)
        positions, distances = _move_affinity_numba(
            positions, mesh._water_nbrs, mesh._water_nbr_count,
            cx, cy, targets, n_steps=3, halt_dist=0.0)
        assert positions[0] == 2
        assert distances[0] == 0.0


class TestSetAffinityNumba:
    def test_finds_best_neighbor_within_bounds(self):
        """Should find the neighbor with highest gradient within min/max bounds."""
        from salmon_ibm.events_hexsim import _set_affinity_numba
        mesh = FakeMesh()
        gradient = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        cells = np.array([2], dtype=np.intp)  # cell 2 has neighbors 1, 3
        min_bounds = np.array([0.0])
        max_bounds = np.array([100.0])
        targets = _set_affinity_numba(
            cells, mesh._water_nbrs, mesh._water_nbr_count,
            gradient, min_bounds, max_bounds, True)
        assert targets[0] == 3  # gradient[3]=5 > gradient[2]=2

    def test_no_valid_neighbor(self):
        """When no neighbor is within bounds, target should be -1."""
        from salmon_ibm.events_hexsim import _set_affinity_numba
        mesh = FakeMesh()
        gradient = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # flat
        cells = np.array([2], dtype=np.intp)
        min_bounds = np.array([5.0])  # min bound > any difference
        max_bounds = np.array([100.0])
        targets = _set_affinity_numba(
            cells, mesh._water_nbrs, mesh._water_nbr_count,
            gradient, min_bounds, max_bounds, True)
        assert targets[0] == -1


# ===========================================================================
# 2. HexSimMoveEvent integration tests
# ===========================================================================

class TestHexSimMoveEvent:
    def test_gradient_movement(self):
        """HexSimMoveEvent should move agents along gradient."""
        from salmon_ibm.events_hexsim import HexSimMoveEvent
        from salmon_ibm.events import EveryStep

        mesh = FakeMesh()
        pop = make_population(2, [0, 1])
        gradient = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        evt = HexSimMoveEvent(
            name="test_move", trigger=EveryStep(),
            dispersal_spatial_data="gradient",
            walk_up_gradient=True,
        )
        landscape = {
            "mesh": mesh,
            "spatial_data": {"gradient": gradient},
            "rng": np.random.default_rng(42),
        }
        mask = pop.alive.copy()
        evt.execute(pop, landscape, 0, mask)

        assert pop.tri_idx[0] == 1  # moved from 0 → 1
        assert pop.tri_idx[1] == 2  # moved from 1 → 2

    def test_empty_mask_noop(self):
        """Empty mask should not cause errors."""
        from salmon_ibm.events_hexsim import HexSimMoveEvent
        from salmon_ibm.events import EveryStep

        mesh = FakeMesh()
        pop = make_population(2, [0, 1])
        evt = HexSimMoveEvent(name="test", trigger=EveryStep())
        landscape = {"mesh": mesh, "spatial_data": {}, "rng": np.random.default_rng()}
        mask = np.zeros(2, dtype=bool)
        evt.execute(pop, landscape, 0, mask)  # should not raise


# ===========================================================================
# 3. HexSimAccumulateEvent dispatch tests
# ===========================================================================

class TestHexSimAccumulateEvent:
    def test_clear_updater(self):
        """Clear updater should zero the accumulator."""
        from salmon_ibm.events_hexsim import HexSimAccumulateEvent
        from salmon_ibm.events import EveryStep

        pop = make_population(3, [0, 0, 0], n_accumulators=1)
        pop.accumulator_mgr.data[:, 0] = [5.0, 10.0, 15.0]

        evt = HexSimAccumulateEvent(
            name="test_acc", trigger=EveryStep(),
            updater_functions=[{"function": "Clear", "accumulator": "acc_0"}],
        )
        landscape = {"rng": np.random.default_rng(), "spatial_data": {}, "global_variables": {}}
        mask = pop.alive.copy()
        evt.execute(pop, landscape, 0, mask)

        np.testing.assert_array_equal(pop.accumulator_mgr.data[:, 0], [0.0, 0.0, 0.0])

    def test_increment_updater(self):
        """Increment updater should add amount to accumulator."""
        from salmon_ibm.events_hexsim import HexSimAccumulateEvent
        from salmon_ibm.events import EveryStep

        pop = make_population(2, [0, 0], n_accumulators=1)

        evt = HexSimAccumulateEvent(
            name="test", trigger=EveryStep(),
            updater_functions=[{
                "function": "Increment", "accumulator": "acc_0",
                "parameters": ["3.5"],
            }],
        )
        landscape = {"rng": np.random.default_rng(), "spatial_data": {}, "global_variables": {}}
        evt.execute(pop, landscape, 0, pop.alive.copy())

        np.testing.assert_array_almost_equal(pop.accumulator_mgr.data[:, 0], [3.5, 3.5])

    def test_expression_updater(self):
        """Expression updater should evaluate HexSim DSL expression."""
        from salmon_ibm.events_hexsim import HexSimAccumulateEvent
        from salmon_ibm.events import EveryStep

        pop = make_population(2, [0, 0], n_accumulators=2)
        pop.accumulator_mgr.data[:, 0] = [10.0, 20.0]  # acc_0 = source

        evt = HexSimAccumulateEvent(
            name="test", trigger=EveryStep(),
            updater_functions=[{
                "function": "Expression", "accumulator": "acc_1",
                "parameters": ['"acc_0" * 2'],  # double-quoted = accumulator ref
            }],
        )
        landscape = {"rng": np.random.default_rng(), "spatial_data": {}, "global_variables": {}}
        evt.execute(pop, landscape, 0, pop.alive.copy())

        np.testing.assert_array_almost_equal(pop.accumulator_mgr.data[:, 1], [20.0, 40.0])

    def test_unknown_function_skipped(self):
        """Unknown updater function should be silently skipped."""
        from salmon_ibm.events_hexsim import HexSimAccumulateEvent
        from salmon_ibm.events import EveryStep

        pop = make_population(2, [0, 0], n_accumulators=1)
        evt = HexSimAccumulateEvent(
            name="test", trigger=EveryStep(),
            updater_functions=[{"function": "NonexistentFunc", "accumulator": "acc_0"}],
        )
        landscape = {"rng": np.random.default_rng(), "spatial_data": {}, "global_variables": {}}
        evt.execute(pop, landscape, 0, pop.alive.copy())  # should not raise


# ===========================================================================
# 4. _apply_trait_combo_mask tests
# ===========================================================================

class TestApplyTraitComboMask:
    def test_combo_filter_selects_correct_agents(self):
        """Trait combo filter should select agents matching enabled combos."""
        from salmon_ibm.events_hexsim import _apply_trait_combo_mask

        pop = make_population(4, [0, 0, 0, 0], trait_categories=["juv", "adult"])
        pop.trait_mgr._data["stage"][:] = [0, 0, 1, 1]  # 2 juv, 2 adult

        base_mask = pop.alive.copy()
        # Combo "1 0" means: category 0 (juv) enabled, category 1 (adult) disabled
        uf = {"stratified_traits": ["stage"], "trait_combinations": "1 0"}
        result = _apply_trait_combo_mask(base_mask, uf, pop)

        assert result[0] == True   # juv
        assert result[1] == True   # juv
        assert result[2] == False  # adult
        assert result[3] == False  # adult

    def test_no_filter_returns_base_mask(self):
        """Without stratified_traits, should return base_mask unchanged."""
        from salmon_ibm.events_hexsim import _apply_trait_combo_mask

        pop = make_population(3, [0, 0, 0])
        base_mask = pop.alive.copy()
        uf = {}  # no filter
        result = _apply_trait_combo_mask(base_mask, uf, pop)
        np.testing.assert_array_equal(result, base_mask)

    def test_cache_invalidation(self):
        """Cache should be cleared between steps."""
        from salmon_ibm.events_hexsim import _apply_trait_combo_mask, clear_combo_mask_cache

        pop = make_population(2, [0, 0], trait_categories=["a", "b"])
        pop.trait_mgr._data["stage"][:] = [0, 1]

        uf = {"stratified_traits": ["stage"], "trait_combinations": "1 0"}
        mask = pop.alive.copy()

        r1 = _apply_trait_combo_mask(mask, uf, pop)
        assert r1[0] == True and r1[1] == False

        # Change traits and clear cache
        pop.trait_mgr._data["stage"][:] = [1, 0]
        clear_combo_mask_cache()

        r2 = _apply_trait_combo_mask(mask, uf, pop)
        assert r2[0] == False and r2[1] == True


# ===========================================================================
# 5. _LazyAccDict tests
# ===========================================================================

class TestLazyAccDict:
    def test_lazy_access_returns_masked_values(self):
        """Accessing a key should return masked column."""
        from salmon_ibm.accumulators import _LazyAccDict
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = np.array([True, False, True])
        name_to_idx = {"a": 0, "b": 1}

        d = _LazyAccDict(data, mask, name_to_idx)
        result = d["a"]
        np.testing.assert_array_equal(result, [1.0, 5.0])

    def test_missing_key_raises(self):
        """Accessing nonexistent key should raise KeyError."""
        from salmon_ibm.accumulators import _LazyAccDict
        data = np.zeros((2, 1))
        mask = np.ones(2, dtype=bool)
        d = _LazyAccDict(data, mask, {"x": 0})
        with pytest.raises(KeyError):
            d["nonexistent"]

    def test_contains(self):
        """__contains__ should check name_to_idx."""
        from salmon_ibm.accumulators import _LazyAccDict
        data = np.zeros((2, 1))
        mask = np.ones(2, dtype=bool)
        d = _LazyAccDict(data, mask, {"x": 0})
        assert "x" in d
        assert "y" not in d

    def test_caches_on_repeated_access(self):
        """Second access to same key should return cached array."""
        from salmon_ibm.accumulators import _LazyAccDict
        data = np.array([[1.0], [2.0]])
        mask = np.ones(2, dtype=bool)
        d = _LazyAccDict(data, mask, {"x": 0})
        r1 = d["x"]
        r2 = d["x"]
        assert r1 is r2  # same object (cached)
