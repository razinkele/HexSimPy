"""Reporting framework: HexSim-compatible reports and tallies."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv

import numpy as np


# ---------------------------------------------------------------------------
# Report base
# ---------------------------------------------------------------------------

@dataclass
class Report:
    """Base class for simulation reports."""
    name: str
    records: list[dict] = field(default_factory=list)

    def record(self, data: dict) -> None:
        self.records.append(data)

    def to_csv(self, path: str | Path) -> None:
        if not self.records:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)

    def clear(self) -> None:
        self.records.clear()


# ---------------------------------------------------------------------------
# Built-in reports
# ---------------------------------------------------------------------------

class ProductivityReport(Report):
    """Track births, deaths, and population growth rate (lambda)."""

    def update(self, t: int, n_alive: int, n_births: int = 0, n_deaths: int = 0):
        prev_alive = self.records[-1]["n_alive"] if self.records else n_alive
        lam = n_alive / prev_alive if prev_alive > 0 else 0.0
        self.record({
            "time": t,
            "n_alive": n_alive,
            "n_births": n_births,
            "n_deaths": n_deaths,
            "lambda": round(lam, 4),
        })


class DemographicReport(Report):
    """Per-timestep demographic summary."""

    def update(self, t: int, population):
        alive = population.alive
        self.record({
            "time": t,
            "n_alive": int(alive.sum()),
            "n_dead": int((~alive).sum()),
            "n_total": population.n,
            "mean_mass": float(population.mass_g[alive].mean()) if alive.any() else 0.0,
            "mean_ed": float(population.ed_kJ_g[alive].mean()) if alive.any() else 0.0,
        })


class DispersalReport(Report):
    """Track agent displacement from origin."""

    def update(self, t: int, current_positions: np.ndarray,
               initial_positions: np.ndarray, centroids: np.ndarray):
        alive_mask = np.ones(len(current_positions), dtype=bool)
        curr_coords = centroids[current_positions]
        init_coords = centroids[initial_positions]
        displacements = np.sqrt(
            ((curr_coords - init_coords) ** 2).sum(axis=1)
        )
        self.record({
            "time": t,
            "mean_displacement": float(displacements.mean()),
            "max_displacement": float(displacements.max()),
            "median_displacement": float(np.median(displacements)),
        })


class GeneticReport(Report):
    """Track allele frequencies and heterozygosity."""

    def update(self, t: int, genome_manager, locus_name: str,
               mask: np.ndarray | None = None):
        if genome_manager is None:
            return
        alleles = genome_manager.get_locus(locus_name)
        if mask is not None:
            alleles = alleles[mask]
        n_agents = len(alleles)
        if n_agents == 0:
            return
        n_alleles = genome_manager.loci[genome_manager.locus_index(locus_name)].n_alleles
        # Allele frequencies
        all_alleles = alleles.ravel()
        freq = np.bincount(all_alleles, minlength=n_alleles) / len(all_alleles)
        # Heterozygosity
        het = (alleles[:, 0] != alleles[:, 1]).mean()
        self.record({
            "time": t,
            "locus": locus_name,
            "n_agents": n_agents,
            "heterozygosity": round(float(het), 4),
            **{f"freq_allele_{i}": round(float(f), 4) for i, f in enumerate(freq)},
        })


# ---------------------------------------------------------------------------
# Tallies (per-cell spatial statistics)
# ---------------------------------------------------------------------------

@dataclass
class Tally:
    """Base class for per-cell spatial tallies."""
    name: str
    n_cells: int
    data: np.ndarray = field(init=False)

    def __post_init__(self):
        self.data = np.zeros(self.n_cells, dtype=np.float64)

    def reset(self) -> None:
        self.data[:] = 0.0

    def to_array(self) -> np.ndarray:
        return self.data.copy()


class OccupancyTally(Tally):
    """Count how many timesteps each cell is occupied."""

    def update(self, positions: np.ndarray, alive: np.ndarray):
        occupied = np.unique(positions[alive])
        self.data[occupied] += 1.0


class DensityTally(Tally):
    """Cumulative agent count per cell."""

    def update(self, positions: np.ndarray, alive: np.ndarray):
        counts = np.bincount(positions[alive], minlength=self.n_cells)
        self.data += counts.astype(np.float64)


class DispersalFluxTally(Tally):
    """Count net agent movement into/out of each cell."""

    def update(self, old_positions: np.ndarray, new_positions: np.ndarray,
               alive: np.ndarray):
        moved = alive & (old_positions != new_positions)
        if not moved.any():
            return
        departures = np.bincount(old_positions[moved], minlength=self.n_cells)
        arrivals = np.bincount(new_positions[moved], minlength=self.n_cells)
        self.data += (arrivals - departures).astype(np.float64)


class BarrierTally(Tally):
    """Count barrier interactions per cell."""

    def update(self, cell_indices: np.ndarray, died: np.ndarray,
               deflected: np.ndarray):
        if died.any():
            mort_cells = cell_indices[died]
            self.data[mort_cells] += 1.0


# ---------------------------------------------------------------------------
# Report Manager
# ---------------------------------------------------------------------------

class ReportManager:
    """Manages all reports and tallies for a simulation."""

    def __init__(self):
        self.reports: dict[str, Report] = {}
        self.tallies: dict[str, Tally] = {}

    def add_report(self, report: Report) -> None:
        self.reports[report.name] = report

    def add_tally(self, tally: Tally) -> None:
        self.tallies[tally.name] = tally

    def get_report(self, name: str) -> Report | None:
        return self.reports.get(name)

    def get_tally(self, name: str) -> Tally | None:
        return self.tallies.get(name)

    def save_all(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, report in self.reports.items():
            report.to_csv(output_dir / f"{name}.csv")
        for name, tally in self.tallies.items():
            np.save(output_dir / f"{name}.npy", tally.to_array())

    def summary(self) -> dict:
        return {
            "reports": list(self.reports.keys()),
            "tallies": list(self.tallies.keys()),
            "total_records": sum(len(r.records) for r in self.reports.values()),
        }
