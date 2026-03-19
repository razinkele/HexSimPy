"""Tests for post-review bugfixes."""
import numpy as np
import pytest
from pathlib import Path


class TestDataLookupLoading:
    def test_lookup_table_loaded_from_csv(self, tmp_path):
        """ScenarioLoader should load CSV into DataLookupEvent.lookup_table."""
        csv_file = tmp_path / "Analysis" / "Data Lookup" / "test_table.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(csv_file, np.array([[1.0, 2.0], [3.0, 4.0]]), delimiter=",")

        from salmon_ibm.scenario_loader import ScenarioLoader
        loader = ScenarioLoader()

        edef = {
            "type": "data_lookup",
            "name": "test_lookup",
            "params": {
                "file_name": "test_table.csv",
                "row_accumulator": "row_acc",
                "column_accumulator": "col_acc",
                "target_accumulator": "target",
            },
        }
        import salmon_ibm.events_builtin
        import salmon_ibm.events_phase3
        import salmon_ibm.events_hexsim

        evt = loader._build_single_event(edef, {})
        loader._load_lookup_tables([evt], str(tmp_path))
        assert evt.lookup_table is not None
        assert evt.lookup_table.shape == (2, 2)
        assert evt.lookup_table[1, 0] == 3.0


class TestTraitFilterFormat:
    def test_trait_filter_applied_in_event_group(self):
        """EventGroup should apply trait filter to restrict which agents an event sees."""
        from salmon_ibm.population import Population
        from salmon_ibm.agents import AgentPool
        from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
        from salmon_ibm.events import EventGroup, EveryStep, Event
        from salmon_ibm.interactions import MultiPopulationManager
        from dataclasses import dataclass

        pool = AgentPool(n=4, start_tri=np.zeros(4, dtype=int))
        pop = Population(name="test", pool=pool)
        trait_defs = [TraitDefinition(
            name="stage", trait_type=TraitType.PROBABILISTIC,
            categories=["juvenile", "adult"]
        )]
        pop.trait_mgr = TraitManager(4, trait_defs)
        pop.trait_mgr._data["stage"][:] = np.array([0, 0, 1, 1])

        received_masks = []

        @dataclass
        class SpyEvent(Event):
            def execute(self, population, landscape, t, mask):
                received_masks.append(mask.copy())

        child = SpyEvent(
            name="spy",
            trigger=EveryStep(),
            trait_filter={"stage": "adult"},
        )
        child.population_name = "test"

        group = EventGroup(
            name="group",
            trigger=EveryStep(),
            sub_events=[child],
        )

        mgr = MultiPopulationManager()
        mgr.register(pop)
        landscape = {"multi_pop_mgr": mgr}

        group.execute(pop, landscape, 0, pop.alive.copy())
        assert len(received_masks) == 1
        assert received_masks[0][0] == False
        assert received_masks[0][1] == False
        assert received_masks[0][2] == True
        assert received_masks[0][3] == True


class TestReproductionMateSelection:
    def test_offspring_have_two_different_parents(self):
        """Offspring should inherit alleles from two different parents."""
        from salmon_ibm.population import Population
        from salmon_ibm.agents import AgentPool
        from salmon_ibm.genetics import GenomeManager, LocusDefinition
        from salmon_ibm.events_builtin import ReproductionEvent
        from salmon_ibm.events import EveryStep

        pool = AgentPool(n=4, start_tri=np.zeros(4, dtype=int))
        pop = Population(name="test", pool=pool)
        pop.group_id[:] = np.array([0, 0, 1, 1], dtype=np.int32)

        loci = [LocusDefinition(name="loc1", n_alleles=2, position=0.0)]
        gm = GenomeManager(4, loci, rng_seed=42)
        gm.genotypes[0, 0, :] = [0, 0]  # group 0 parent A: homozygous 0
        gm.genotypes[1, 0, :] = [1, 1]  # group 0 parent B: homozygous 1
        gm.genotypes[2, 0, :] = [0, 0]  # group 1 parent A: homozygous 0
        gm.genotypes[3, 0, :] = [1, 1]  # group 1 parent B: homozygous 1
        pop.genome = gm

        evt = ReproductionEvent(
            name="repro", trigger=EveryStep(),
            clutch_mean=2.0, min_group_size=2,
            offspring_mass_mean=100.0, offspring_mass_std=10.0,
        )
        rng = np.random.default_rng(42)
        landscape = {"rng": rng}
        mask = pop.alive.copy()
        evt.execute(pop, landscape, 0, mask)

        n_offspring = pop.pool.n - 4
        if n_offspring > 0:
            offspring_geno = pop.genome.genotypes[4:, 0, :]
            has_heterozygote = np.any(offspring_geno[:, 0] != offspring_geno[:, 1])
            assert has_heterozygote, "With two homozygous parents, offspring should be heterozygous"


class TestAccumulatorBounds:
    def test_lower_bound_zero_with_nonzero_upper_preserves_zero_bound(self):
        """lowerBound=0 + upperBound=1 should produce min_val=0, not None."""
        from salmon_ibm.scenario_loader import ScenarioLoader

        loader = ScenarioLoader()
        pop_def = {
            "name": "test",
            "initial_size": 0,
            "accumulators": [
                {"name": "survival_prob", "min_val": 0, "max_val": 1},
                {"name": "unbounded", "min_val": 0, "max_val": 0},
            ],
            "traits": [],
        }

        class FakeMesh:
            water_mask = np.ones(10, dtype=bool)

        rng = np.random.default_rng(42)
        pop = loader._create_population(pop_def, FakeMesh(), rng)
        mgr = pop.accumulator_mgr
        assert mgr is not None

        sp_def = mgr.definitions[mgr._resolve_idx("survival_prob")]
        assert sp_def.min_val == 0, "lowerBound=0 with upperBound=1 should keep min_val=0"
        assert sp_def.max_val == 1, "upperBound=1 should be preserved"

        ub_def = mgr.definitions[mgr._resolve_idx("unbounded")]
        assert ub_def.min_val is None, "Both 0 should mean unbounded (None)"
        assert ub_def.max_val is None, "Both 0 should mean unbounded (None)"


class TestScenarioLoaderReproducibility:
    def _make_mock_mesh(self, n_cells=20):
        """Create a minimal mock HexMesh with a water_mask."""
        from unittest.mock import MagicMock
        mesh = MagicMock()
        mesh.water_mask = np.ones(n_cells, dtype=bool)
        return mesh

    def test_same_seed_produces_same_agent_positions(self):
        """_create_population with same seeded RNG must produce identical tri_idx."""
        from salmon_ibm.scenario_loader import ScenarioLoader
        loader = ScenarioLoader()
        mesh = self._make_mock_mesh(n_cells=20)
        pop_def = {"name": "test_pop", "initial_size": 10}

        rng1 = np.random.default_rng(np.random.default_rng(42).integers(2**63))
        rng2 = np.random.default_rng(np.random.default_rng(42).integers(2**63))

        pop1 = loader._create_population(pop_def, mesh, rng1)
        pop2 = loader._create_population(pop_def, mesh, rng2)

        np.testing.assert_array_equal(
            pop1.tri_idx, pop2.tri_idx,
            err_msg="Same seed should produce identical agent positions"
        )

    def test_different_seeds_produce_different_positions(self):
        """_create_population with different seeds should produce different tri_idx."""
        from salmon_ibm.scenario_loader import ScenarioLoader
        loader = ScenarioLoader()
        mesh = self._make_mock_mesh(n_cells=20)
        pop_def = {"name": "test_pop", "initial_size": 10}

        rng_a = np.random.default_rng(111)
        rng_b = np.random.default_rng(999)

        pop_a = loader._create_population(pop_def, mesh, rng_a)
        pop_b = loader._create_population(pop_def, mesh, rng_b)

        # With 10 agents in 20 cells and different seeds, positions should differ
        assert not np.array_equal(pop_a.tri_idx, pop_b.tri_idx), \
            "Different seeds should produce different agent positions"

    def test_same_seed_produces_same_populations(self):
        """Two loads with same seed should produce identical agent positions."""
        import os
        ws = "HexSimPLE"
        xml = "HexSimPLE/Scenarios/HexSimPLE.xml"
        if not os.path.exists(xml):
            pytest.skip("HexSimPLE not available")

        from salmon_ibm.scenario_loader import ScenarioLoader
        loader = ScenarioLoader()
        sim1 = loader.load(ws, xml, rng_seed=123)
        sim2 = loader.load(ws, xml, rng_seed=123)

        for pname in sim1.populations.populations:
            p1 = sim1.populations.populations[pname]
            p2 = sim2.populations.populations[pname]
            np.testing.assert_array_equal(p1.tri_idx, p2.tri_idx,
                err_msg=f"Population {pname} positions differ with same seed")


class TestInteractionDeadPrey:
    def test_dead_prey_not_re_encountered(self):
        """Once prey is killed, subsequent predators in same cell should not encounter it."""
        from salmon_ibm.population import Population
        from salmon_ibm.agents import AgentPool
        from salmon_ibm.interactions import (
            MultiPopulationManager, InteractionEvent, InteractionOutcome,
        )
        from salmon_ibm.events import EveryStep

        # 2 predators, 1 prey, all at cell 0
        pred_pool = AgentPool(n=2, start_tri=np.array([0, 0]))
        pred = Population(name="pred", pool=pred_pool)
        prey_pool = AgentPool(n=1, start_tri=np.array([0]))
        prey = Population(name="prey", pool=prey_pool)

        mgr = MultiPopulationManager()
        mgr.register(pred)
        mgr.register(prey)

        evt = InteractionEvent(
            name="hunt", trigger=EveryStep(),
            pop_a_name="pred", pop_b_name="prey",
            outcome=InteractionOutcome.PREDATION,
            encounter_probability=1.0,
        )

        rng = np.random.default_rng(42)
        landscape = {"multi_pop_mgr": mgr, "rng": rng}
        mask = pred.alive.copy()
        stats = evt.execute(pred, landscape, 0, mask)

        assert not prey.alive[0], "Prey should be dead"
        # With the bug, kills==2 (dead prey re-killed by second predator).
        # After the fix, kills==1 because the dead prey is skipped.
        assert stats["kills"] == 1, (
            f"Expected 1 kill (prey already dead for 2nd predator), got {stats['kills']}"
        )


class TestStreamNetworkBFS:
    def test_all_upstream_returns_correct_segments(self):
        from salmon_ibm.network import StreamNetwork, SegmentDefinition
        segs = [
            SegmentDefinition(id=0, length=100, upstream_ids=[1, 2]),
            SegmentDefinition(id=1, length=100, upstream_ids=[3]),
            SegmentDefinition(id=2, length=100, upstream_ids=[]),
            SegmentDefinition(id=3, length=100, upstream_ids=[]),
        ]
        net = StreamNetwork(segs)
        result = net.all_upstream(0)
        assert set(result) == {1, 2, 3}

    def test_all_downstream_returns_correct_segments(self):
        from salmon_ibm.network import StreamNetwork, SegmentDefinition
        segs = [
            SegmentDefinition(id=0, length=100, upstream_ids=[], downstream_ids=[1]),
            SegmentDefinition(id=1, length=100, upstream_ids=[0], downstream_ids=[2]),
            SegmentDefinition(id=2, length=100, upstream_ids=[1], downstream_ids=[]),
        ]
        net = StreamNetwork(segs)
        result = net.all_downstream(0)
        assert set(result) == {1, 2}
