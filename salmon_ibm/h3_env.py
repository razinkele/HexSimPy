"""H3-native environment / forcing loader.

``H3Environment`` is the H3 sibling of :class:`salmon_ibm.environment.Environment`
(TriMesh) and :class:`salmon_ibm.hexsim_env.HexSimEnvironment` (HexMesh).
It loads forcing data from a NetCDF produced by
``scripts/build_nemunas_h3_landscape.py`` and binds it to a
:class:`salmon_ibm.h3mesh.H3Mesh` via H3 cell-ID lookup.

Field names are the canonical keys consumed by ``movement.py`` and the
event handlers — ``temperature``, ``salinity``, ``u_current``,
``v_current`` — matching what the TriMesh and HexMesh paths expose.

Phase 2.2 of ``docs/superpowers/plans/2026-04-24-h3mesh-backend.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class H3Environment:
    """Per-step forcing fields bound to an :class:`H3Mesh` cell ordering.

    Parameters
    ----------
    mesh
        The :class:`H3Mesh` whose cell ordering the field arrays follow.
    fields
        ``{name: (n_time, n_cells) float32}`` for each variable present
        in the landscape NetCDF.  Keys are the canonical event names
        (see :data:`_FIELD_RENAME`).
    time
        ``(n_time,)`` ``datetime64`` array of timestep stamps.
    """

    mesh: object  # H3Mesh, declared as object to avoid circular import
    fields: dict[str, np.ndarray]
    time: np.ndarray
    _time_idx: int = field(default=0, init=False)

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

        fields: dict[str, np.ndarray] = {}
        for src, dst in _FIELD_RENAME.items():
            arr = load_renamed(src)
            if arr is not None:
                fields[dst] = arr

        return cls(mesh=mesh, fields=fields, time=ds["time"].values)

    # --- duck-typed, mirrors Environment / HexSimEnvironment --------

    def advance(self, step: int) -> None:
        """Set the active timestep index (clamped to the available range)."""
        self._time_idx = max(0, min(step, len(self.time) - 1))

    def current(self) -> dict[str, np.ndarray]:
        """Return ``{name: (n_cells,) float32}`` for the active timestep.

        The dict is freshly built each call (cheap — n_cells slices) so
        callers may mutate without affecting subsequent reads.
        """
        return {name: arr[self._time_idx] for name, arr in self.fields.items()}

    def sample(self, name: str) -> np.ndarray:
        """Return the field at the active timestep — convenience for code
        that already has a name in hand."""
        return self.fields[name][self._time_idx]
