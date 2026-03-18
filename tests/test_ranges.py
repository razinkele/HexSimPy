"""Unit tests for range allocation."""
import numpy as np
import pytest
from salmon_ibm.ranges import RangeAllocator, AgentRange


class FakeMesh:
    """Minimal mesh mock for range testing: 10 cells in a line."""
    def __init__(self, n=10):
        self.n_cells = n
        self.n_triangles = n
        # Linear connectivity: cell i connects to i-1 and i+1
        max_nbrs = 6
        self._water_nbrs = np.full((n, max_nbrs), -1, dtype=np.intp)
        self._water_nbr_count = np.zeros(n, dtype=np.intp)
        self.neighbors = self._water_nbrs
        for i in range(n):
            k = 0
            if i > 0:
                self._water_nbrs[i, k] = i - 1
                k += 1
            if i < n - 1:
                self._water_nbrs[i, k] = i + 1
                k += 1
            self._water_nbr_count[i] = k


class TestRangeAllocator:
    @pytest.fixture
    def mesh(self):
        return FakeMesh(10)

    @pytest.fixture
    def allocator(self, mesh):
        return RangeAllocator(mesh)

    def test_initial_state(self, allocator):
        assert allocator.n_occupied == 0
        assert allocator.is_available(0)
        assert allocator.owner_of(0) == -1

    def test_allocate_cell(self, allocator):
        assert allocator.allocate_cell(0, 3)
        assert not allocator.is_available(3)
        assert allocator.owner_of(3) == 0
        assert allocator.n_occupied == 1

    def test_cannot_double_allocate(self, allocator):
        allocator.allocate_cell(0, 3)
        assert not allocator.allocate_cell(1, 3)

    def test_release_cell(self, allocator):
        allocator.allocate_cell(0, 3)
        allocator.release_cell(0, 3)
        assert allocator.is_available(3)
        assert allocator.get_range(0) is None

    def test_release_all(self, allocator):
        allocator.allocate_cell(0, 3)
        allocator.allocate_cell(0, 4)
        allocator.allocate_cell(0, 5)
        allocator.release_all(0)
        assert allocator.n_occupied == 0

    def test_expand_range(self, allocator, mesh):
        allocator.allocate_cell(0, 5)  # start at center
        resources = np.ones(10) * 10.0
        added = allocator.expand_range(0, resources, resource_threshold=1.0, max_cells=3)
        assert added == 2  # should claim 4 and 6
        assert len(allocator.get_range(0).cells) == 3

    def test_expand_respects_max(self, allocator, mesh):
        allocator.allocate_cell(0, 5)
        resources = np.ones(10) * 10.0
        allocator.expand_range(0, resources, max_cells=2)
        assert len(allocator.get_range(0).cells) <= 2

    def test_expand_skips_low_resource(self, allocator, mesh):
        allocator.allocate_cell(0, 5)
        resources = np.zeros(10)  # no resources anywhere
        added = allocator.expand_range(0, resources, resource_threshold=1.0)
        assert added == 0

    def test_contract_range(self, allocator, mesh):
        allocator.allocate_cell(0, 3)
        allocator.allocate_cell(0, 4)
        allocator.allocate_cell(0, 5)
        resources = np.array([0, 0, 0, 0.5, 2.0, 0.5, 0, 0, 0, 0])
        released = allocator.contract_range(0, resources, resource_threshold=1.0)
        assert released == 2  # cells 3 and 5 below threshold
        assert len(allocator.get_range(0).cells) == 1
        assert 4 in allocator.get_range(0).cells

    def test_compute_resources(self, allocator):
        allocator.allocate_cell(0, 2)
        allocator.allocate_cell(0, 3)
        resources = np.arange(10, dtype=float)
        total = allocator.compute_resources(0, resources)
        assert total == pytest.approx(5.0)  # 2 + 3

    def test_non_overlapping(self, allocator):
        allocator.allocate_cell(0, 3)
        allocator.allocate_cell(0, 4)
        allocator.allocate_cell(1, 6)
        allocator.allocate_cell(1, 7)
        resources = np.ones(10) * 10.0
        allocator.expand_range(0, resources, max_cells=5)
        allocator.expand_range(1, resources, max_cells=5)
        # Ranges should not overlap
        cells_0 = allocator.get_range(0).cells
        cells_1 = allocator.get_range(1).cells
        assert cells_0.isdisjoint(cells_1)

    def test_summary(self, allocator):
        allocator.allocate_cell(0, 3)
        allocator.allocate_cell(0, 4)
        allocator.allocate_cell(1, 7)
        s = allocator.summary()
        assert s["n_agents_with_ranges"] == 2
        assert s["n_occupied_cells"] == 3
        assert s["mean_range_size"] == pytest.approx(1.5)
        assert s["max_range_size"] == 2
