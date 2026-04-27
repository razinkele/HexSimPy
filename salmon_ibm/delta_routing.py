"""Nemunas delta branch identity, discharge fractions, and exit-tracking utility.

Three branches: Atmata, Skirvyte, Gilija. Pakalne is intentionally lumped
into Atmata (small distributary, absent from the inSTREAM source). Rusne
is the island between branches, not a flowing channel.

Fractions follow Ramsar Site 629 Information Sheet (Nemunas Delta), 2010
https://rsis.ramsar.org/RISapp/files/41231939/documents/LT629_lit161122.pdf
— interior-of-range midpoints summing to 1.0. They are geographic constants
of the delta, not scenario-tunable parameters.

Tagging contract: every code path that calls Population.add_agents must
follow up with population.set_natal_reach_from_cells(new_idx, mesh) — except
the inter-population transfer case in network.py, which preserves
natal_reach_id from the source agent. Population.assert_natal_tagged()
enforces this at runtime.
"""
from __future__ import annotations

import numpy as np

BRANCH_FRACTIONS: dict[str, float] = {
    "Skirvyte": 0.51,   # main flow to Kaliningrad / SW lagoon
    "Atmata":   0.27,   # NE distributary; Pakalne lumped in
    "Gilija":   0.22,   # easternmost; branches off upstream of Rusne island
}
assert abs(sum(BRANCH_FRACTIONS.values()) - 1.0) < 1e-9, (
    f"BRANCH_FRACTIONS must sum to 1.0, got {sum(BRANCH_FRACTIONS.values())}"
)

DELTA_BRANCH_REACHES: frozenset[str] = frozenset(BRANCH_FRACTIONS)


def split_discharge(q_total):
    """Apply BRANCH_FRACTIONS to a Nemunas climatology (scalar or (T,) array).

    Returns a dict mapping branch name to the per-branch series, preserving
    insertion order of BRANCH_FRACTIONS.
    """
    return {br: q_total * f for br, f in BRANCH_FRACTIONS.items()}
