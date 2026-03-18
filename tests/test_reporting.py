"""Unit tests for reporting framework."""
import numpy as np
import pytest
from pathlib import Path

from salmon_ibm.reporting import (
    Report, ProductivityReport, DemographicReport, DispersalReport,
    GeneticReport, OccupancyTally, DensityTally, DispersalFluxTally,
    ReportManager,
)


class TestReport:
    def test_record_and_csv(self, tmp_path):
        r = Report("test")
        r.record({"time": 0, "value": 1.0})
        r.record({"time": 1, "value": 2.0})
        path = tmp_path / "test.csv"
        r.to_csv(path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 records

    def test_clear(self):
        r = Report("test")
        r.record({"a": 1})
        r.clear()
        assert len(r.records) == 0


class TestProductivityReport:
    def test_lambda_calculation(self):
        r = ProductivityReport("productivity")
        r.update(0, n_alive=100, n_births=0, n_deaths=0)
        r.update(1, n_alive=110, n_births=15, n_deaths=5)
        assert r.records[1]["lambda"] == pytest.approx(1.1)
        assert r.records[1]["n_births"] == 15


class TestDemographicReport:
    def test_records_population_stats(self):
        class MockPop:
            def __init__(self):
                self.alive = np.array([True, True, False, True, False])
                self.mass_g = np.array([100.0, 200.0, 0.0, 300.0, 0.0])
                self.ed_kJ_g = np.array([6.0, 7.0, 0.0, 8.0, 0.0])
                self.n = 5
        r = DemographicReport("demo")
        r.update(0, MockPop())
        assert r.records[0]["n_alive"] == 3
        assert r.records[0]["n_dead"] == 2
        assert r.records[0]["mean_mass"] == pytest.approx(200.0)


class TestGeneticReport:
    def test_allele_frequencies(self):
        from salmon_ibm.genetics import LocusDefinition, GenomeManager
        loci = [LocusDefinition("A", n_alleles=3, position=0.0)]
        gm = GenomeManager(n_agents=4, loci=loci, rng_seed=42)
        # All agents: allele0=0, allele1=1
        gm.genotypes[:, 0, 0] = 0
        gm.genotypes[:, 0, 1] = 1
        r = GeneticReport("genetics")
        r.update(0, gm, "A")
        rec = r.records[0]
        assert rec["heterozygosity"] == 1.0  # all heterozygous
        assert rec["freq_allele_0"] == 0.5
        assert rec["freq_allele_1"] == 0.5
        assert rec["freq_allele_2"] == 0.0


class TestOccupancyTally:
    def test_counts_occupied_timesteps(self):
        t = OccupancyTally("occ", n_cells=5)
        positions = np.array([0, 0, 2, 3])
        alive = np.ones(4, dtype=bool)
        t.update(positions, alive)
        t.update(positions, alive)
        assert t.data[0] == 2.0  # occupied both times
        assert t.data[1] == 0.0  # never occupied
        assert t.data[2] == 2.0


class TestDensityTally:
    def test_cumulative_density(self):
        t = DensityTally("density", n_cells=5)
        positions = np.array([0, 0, 1])
        alive = np.ones(3, dtype=bool)
        t.update(positions, alive)
        assert t.data[0] == 2.0
        assert t.data[1] == 1.0
        t.update(positions, alive)
        assert t.data[0] == 4.0


class TestDispersalFluxTally:
    def test_net_flux(self):
        t = DispersalFluxTally("flux", n_cells=5)
        old = np.array([0, 0, 1])
        new = np.array([1, 0, 2])
        alive = np.ones(3, dtype=bool)
        t.update(old, new, alive)
        assert t.data[0] == -1.0  # one left
        assert t.data[1] == 0.0   # one arrived, one left
        assert t.data[2] == 1.0   # one arrived


class TestReportManager:
    def test_add_and_get(self):
        mgr = ReportManager()
        mgr.add_report(ProductivityReport("prod"))
        mgr.add_tally(OccupancyTally("occ", n_cells=10))
        assert mgr.get_report("prod") is not None
        assert mgr.get_tally("occ") is not None

    def test_save_all(self, tmp_path):
        mgr = ReportManager()
        r = ProductivityReport("prod")
        r.update(0, 100)
        mgr.add_report(r)
        t = OccupancyTally("occ", n_cells=5)
        mgr.add_tally(t)
        mgr.save_all(tmp_path)
        assert (tmp_path / "prod.csv").exists()
        assert (tmp_path / "occ.npy").exists()

    def test_summary(self):
        mgr = ReportManager()
        r = Report("test")
        r.record({"a": 1})
        r.record({"a": 2})
        mgr.add_report(r)
        s = mgr.summary()
        assert s["total_records"] == 2
