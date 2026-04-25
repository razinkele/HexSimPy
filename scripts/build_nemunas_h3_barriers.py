"""Generate a synthetic mid-lagoon barrier CSV for Nemunas H3.

Phase 4.3 of ``docs/superpowers/plans/2026-04-24-h3mesh-backend.md``.

A horizontal line across the open Curonian Lagoon at lat 55.30°N —
**not** an actual physical structure, just a contrived barrier the
integration test can use to prove the H3 barrier loader + Numba
barrier-resolution kernel work end-to-end.  The probabilities are
illustrative, not calibrated.

Why mid-lagoon and not Klaipėda Strait?  At H3 res 9 (~200 m cells)
the strait is only 2-4 cells wide, so a line across it produces only
1-4 water-water edges after the EMODnet land-mask filter — not a
robust signal for a 3-day mortality-difference test.  The open lagoon
at lat 55.30 has ~60 contiguous water cells E-W, so a barrier line
yields ≥ 50 in-mesh edges.

Two outputs:

* ``data/nemunas_h3_barriers.csv`` — gentle 5/90/5 split, the default
  the YAML config points at.
* ``data/nemunas_h3_barriers_strong.csv`` — aggressive 30/60/10 split
  for the integration test (so a 3-day run produces a reliably
  non-zero mortality difference).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Make ``salmon_ibm`` importable when this script is run directly from
# anywhere — the other ``scripts/`` files use the same pattern.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import h3
import xarray as xr

from salmon_ibm.h3_barriers import line_barrier_to_h3_edges


# Open Curonian Lagoon at lat ≈ 55.30°N — a 25-km E-W line through
# the wide middle of the lagoon, where water is contiguous at H3 res 9.
# See module docstring for why this is *not* the actual Klaipėda Strait.
LINE = (55.300, 21.100, 55.300, 21.500)
RESOLUTION = 9

LANDSCAPE_NC = "data/nemunas_h3_landscape.nc"
GENTLE_CSV = "data/nemunas_h3_barriers.csv"
STRONG_CSV = "data/nemunas_h3_barriers_strong.csv"


def filter_water_edges(
    edges: list[tuple[str, str]],
    landscape_nc: Path,
) -> list[tuple[str, str]]:
    """Drop edges whose endpoints aren't both water in the landscape."""
    ds = xr.open_dataset(landscape_nc, engine="h5netcdf")
    h3_to_water = {
        int(hid): bool(wm)
        for hid, wm in zip(ds["h3_id"].values, ds["water_mask"].values)
    }

    def is_water(cell_str: str) -> bool:
        return h3_to_water.get(int(h3.str_to_int(cell_str)), False)

    kept = [(a, b) for a, b in edges if is_water(a) and is_water(b)]
    return kept


def write_csv(
    path: Path,
    edges: list[tuple[str, str]],
    *,
    mortality: float,
    deflection: float,
    transmission: float,
    note: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["from_h3", "to_h3", "mortality",
                         "deflection", "transmission", "note"],
        )
        w.writeheader()
        for a, b in edges:
            w.writerow({
                "from_h3": a, "to_h3": b,
                "mortality": f"{mortality:.4f}",
                "deflection": f"{deflection:.4f}",
                "transmission": f"{transmission:.4f}",
                "note": note,
            })


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    landscape_path = project_root / LANDSCAPE_NC
    if not landscape_path.exists():
        raise SystemExit(
            f"{landscape_path} missing — run "
            "scripts/build_nemunas_h3_landscape.py first"
        )

    print(f"[1/3] Tracing line {LINE} at res {RESOLUTION}…")
    raw_edges = line_barrier_to_h3_edges(*LINE, resolution=RESOLUTION)
    print(f"  {len(raw_edges)} bidirectional edges from the line")

    print(f"[2/3] Filtering land-touching edges via {LANDSCAPE_NC}…")
    edges = filter_water_edges(raw_edges, landscape_path)
    print(f"  {len(edges)} water-only edges (dropped {len(raw_edges) - len(edges)})")
    if not edges:
        raise SystemExit(
            "no water-only edges left — check the line coordinates "
            "or the landscape NetCDF's water_mask"
        )

    print(f"[3/3] Writing CSVs…")
    gentle_path = project_root / GENTLE_CSV
    write_csv(
        gentle_path, edges,
        mortality=0.05, deflection=0.90, transmission=0.05,
        note="Curonian lagoon synthetic E-W barrier (illustrative)",
    )
    print(f"  wrote {gentle_path}")

    strong_path = project_root / STRONG_CSV
    write_csv(
        strong_path, edges,
        mortality=0.30, deflection=0.60, transmission=0.10,
        note="Curonian lagoon synthetic E-W barrier — TEST ONLY",
    )
    print(f"  wrote {strong_path}")


if __name__ == "__main__":
    main()
