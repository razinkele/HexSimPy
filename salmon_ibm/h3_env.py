"""H3-native environment / forcing loader.

``H3Environment`` is the H3 sibling of :class:`salmon_ibm.environment.Environment`
(TriMesh) and :class:`salmon_ibm.hexsim_env.HexSimEnvironment` (HexMesh).
It loads forcing data from a NetCDF produced by
``scripts/build_nemunas_h3_landscape.py`` and binds it to a
:class:`salmon_ibm.h3mesh.H3Mesh` via H3 cell-ID lookup.

Field names are the canonical keys consumed by ``movement.py`` and the
event handlers — ``temperature``, ``salinity``, ``u_current``,
``v_current`` — matching what the TriMesh and HexMesh paths expose.

Contract: ``env.fields[name]`` is the **current-timestep snapshot**
``(n_cells,) float32``.  ``advance(step)`` mutates the snapshot arrays
in place so ``Simulation.step`` can pass ``landscape["fields"] =
self.env.fields`` and downstream events read per-cell values without
knowing anything about time.  Mirrors the TriMesh
:class:`Environment` contract.

Phase 2.2 of ``docs/superpowers/plans/2026-04-24-h3mesh-backend.md``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

# Map landscape-NetCDF variable names → canonical movement-event keys.
# Diverging from these names silently no-ops the advection event because
# `_apply_current_advection_vec` reads `fields["u_current"]`/`["v_current"]`.
_FIELD_RENAME = {
    "tos": "temperature",
    "sos": "salinity",
    "uo": "u_current",
    "vo": "v_current",
}


class H3Environment:
    """Per-step forcing fields bound to an :class:`H3Mesh` cell ordering.

    See module docstring for the contract.  The full ``(n_time,
    n_cells)`` time-series is held privately on :attr:`_full_fields`
    for the rare event handler that wants it.
    """

    def __init__(
        self,
        mesh,
        full_fields: dict[str, np.ndarray],
        time: np.ndarray,
    ) -> None:
        self.mesh = mesh
        self._full_fields = full_fields
        self.time = time
        self._time_idx: int = 0
        # Snapshot dict — same keys as full_fields, each value the
        # current-step (n_cells,) slice.  Mutated in advance() (in place
        # so callers that captured a reference still see the latest data).
        self.fields: dict[str, np.ndarray] = {
            name: arr[0].copy() for name, arr in full_fields.items()
        }

    @classmethod
    def from_netcdf(cls, nc_path: str | Path, mesh) -> "H3Environment":
        """Load forcing from the landscape NetCDF and bind to ``mesh``.

        The NetCDF stores fields in its own cell order keyed by ``h3_id``;
        we permute every field array so the i-th column corresponds to
        ``mesh.h3_ids[i]``.  Uses ``np.searchsorted`` on the sorted
        NetCDF h3_id array — O(N log N) once, no Python dict.
        """
        # h5netcdf engine: NetCDF4 lets us carry h3_id as uint64
        # (NetCDF3 has no unsigned 64-bit type — see builder script).
        ds = xr.open_dataset(str(nc_path), engine="h5netcdf")

        ds_ids = ds["h3_id"].values.astype(np.uint64)
        # NetCDF builder writes h3_id sorted ascending, but tolerate
        # unsorted input by sorting here too.
        order = np.argsort(ds_ids)
        ds_ids_sorted = ds_ids[order]

        mesh_ids = mesh.h3_ids.astype(np.uint64)
        matches = np.searchsorted(ds_ids_sorted, mesh_ids)

        # Guard: every mesh cell must exist in the NetCDF.  Out-of-bounds
        # match index OR a sorted-array mismatch both signal a missing cell.
        bad = (matches >= len(ds_ids_sorted))
        bad_safe_idx = np.where(bad, 0, matches)
        bad |= ds_ids_sorted[bad_safe_idx] != mesh_ids
        if bad.any():
            first_missing = int(mesh_ids[bad][0])
            raise ValueError(
                f"{int(bad.sum())} mesh cell(s) not in forcing NetCDF; "
                f"first missing H3 id (int): {first_missing} "
                f"(0x{first_missing:x})"
            )
        reorder = order[matches]   # (n_mesh_cells,) — permutation into ds

        def load_renamed(src: str) -> np.ndarray | None:
            if src not in ds:
                return None
            arr = ds[src].values  # (time, ds_cell)
            return arr[:, reorder].astype(np.float32)

        full_fields: dict[str, np.ndarray] = {}
        for src, dst in _FIELD_RENAME.items():
            arr = load_renamed(src)
            if arr is not None:
                full_fields[dst] = arr

        # SSH is required by movement.py's upstream/downstream branches
        # (`fields["ssh"]`).  CMEMS BALTICSEA reanalysis doesn't ship
        # `zos`, so the Curonian regridder zero-fills it (see
        # scripts/fetch_cmems_forcing.py: "zos is zero-filled; real SSH
        # requires separate product").  Mirror that behaviour here —
        # seiche detection is a no-op, gradient-following degenerates
        # to first-neighbour selection.  Same deferral as the Curonian
        # plan; documented in the Nemunas spec "Known limitations".
        n_time = full_fields[next(iter(full_fields))].shape[0] \
            if full_fields else len(ds["time"].values)
        full_fields["ssh"] = np.zeros(
            (n_time, len(mesh.h3_ids)), dtype=np.float32,
        )

        return cls(mesh=mesh, full_fields=full_fields, time=ds["time"].values)

    # --- duck-typed, mirrors Environment / HexSimEnvironment --------

    def advance(self, step: int) -> None:
        """Set the active timestep and refresh :attr:`fields` in place.

        ``step`` may exceed the available range — for daily CMEMS
        forcing on an hourly simulation, indices map ``step // 24`` to
        the day index and clamp at the last day.  Out-of-range values
        are clamped silently rather than raising; the simulation loop
        passes ``current_t`` directly which can outrun any finite
        forcing series.
        """
        n_time = len(self.time)
        # Heuristic mapping: if the simulation runs more steps than we
        # have timesteps, assume hour→day ratio.  If the data is already
        # at simulation cadence (n_time == n_steps) the divisor of 1
        # gives a 1:1 map.  Most ratios in practice are 24× (daily
        # CMEMS / hourly sim) or 1× (synthetic test fixtures).
        if step < n_time:
            new_idx = step
        else:
            ratio = max(1, (step + 1) // n_time)  # 24 for daily-CMEMS / hourly-sim
            new_idx = step // ratio
        self._time_idx = max(0, min(new_idx, n_time - 1))
        for name, arr in self._full_fields.items():
            # In-place copy so callers that captured `env.fields[name]`
            # see the new data without needing to re-fetch.
            np.copyto(self.fields[name], arr[self._time_idx])

    def current(self) -> dict[str, np.ndarray]:
        """Return the current-timestep snapshot (alias for :attr:`fields`).

        Provided for API symmetry with :class:`HexSimEnvironment`; the
        returned dict is the live one — mutations affect subsequent
        reads, same as ``env.fields`` itself.
        """
        return self.fields

    def sample(self, name: str) -> np.ndarray:
        """Return ``fields[name]`` — convenience for callers that have
        a name in hand."""
        return self.fields[name]

    def dSSH_dt_array(self) -> np.ndarray:
        """Per-cell rate of SSH change (m/hour).

        CMEMS BALTICSEA reanalysis doesn't ship SSH at hourly cadence,
        so we synthesise a zero SSH field at construction time (see
        ``__init__``).  The seiche-pause estuarine override therefore
        sees zero everywhere → no agents are paused, which matches the
        ecological expectation that mid-Baltic salmon don't experience
        the harbour-seiche dynamics that motivated the override.
        """
        return np.zeros(self.mesh.n_cells, dtype=np.float32)
