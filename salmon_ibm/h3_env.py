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

# C4: err-id constants for grep-able operational logging.
# Mirrors ERR_HOMING_HATCHERY_NO_DISPATCH in delta_routing.py:43.
ERR_DIST_FROM_SEA_MISSING = "dist-from-sea-missing"
ERR_DIST_FROM_SEA_SHAPE_MISMATCH = "dist-from-sea-shape-mismatch"
ERR_DIST_FROM_SEA_NAN_ON_WATER = "dist-from-sea-nan-on-water"
ERR_DIST_FROM_SEA_ALL_ZERO = "dist-from-sea-all-zero"
ERR_DIST_FROM_SEA_NO_SOURCES = "dist-from-sea-no-sources"

# C5: arrival event err-ids. ArrivalEvent lives in events_builtin.py
# but err-ids are centralised here next to the C4 dist_from_sea
# constants for grep-able operational logging.
ERR_C5_MISSING_ARRIVAL_EVENT = "c5-arrival-event-missing"
ERR_C5_ARRIVAL_EVENT_MISORDERED = "c5-arrival-event-misordered"

# C5.1: round-trip arrival err-ids. BeenToSeaEvent lives in
# events_builtin.py; err-ids centralised here next to the C5 block.
ERR_C5_1_BEEN_TO_SEA_MISSING = "c5.1-been-to-sea-missing"
ERR_C5_1_BEEN_TO_SEA_MISORDERED = "c5.1-been-to-sea-misordered"
ERR_C5_1_AT_SEA_REACHES_MISSING = "c5.1-at-sea-reaches-missing"

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

        env = cls(mesh=mesh, full_fields=full_fields, time=ds["time"].values)
        _load_dist_from_sea(env, ds, mesh)
        ds.close()
        return env

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
        sees zero everywhere -> no agents are paused, which matches the
        ecological expectation that mid-Baltic salmon don't experience
        the harbour-seiche dynamics that motivated the override.
        """
        return np.zeros(self.mesh.n_cells, dtype=np.float32)


def _load_dist_from_sea(env, ds, mesh) -> None:
    """C4: load ``dist_from_sea`` from NC into env.fields and mesh.

    Two cases per spec section "Where it lives -> Validation discipline ->
    Sim-init":

    Case A -- variable absent: warn + zero-fill + flag init.
    Case B -- variable present but structurally invalid: raise.

    On all-checks-pass: inject env.fields["dist_from_sea"] +
    mesh.dist_from_sea (same array reference) + flag init.
    """
    import logging

    logger = logging.getLogger("salmon_ibm.h3_env")
    n = mesh.N_cells if hasattr(mesh, "N_cells") else len(mesh.reach_id)

    if "dist_from_sea" not in ds.variables:
        # Case A: backward-compat for pre-C4 NCs.
        logger.warning(
            "%s: dist_from_sea missing from NC; movement gradient will "
            "be flat -- agents will not migrate. Rebuild landscape with "
            "build_h3_multires_landscape.py.",
            ERR_DIST_FROM_SEA_MISSING,
        )
        zero = np.zeros(n, dtype=np.float32)
        env.fields["dist_from_sea"] = zero
        mesh.dist_from_sea = zero
        env._dormant_gradient_check_done = False
        return

    # Case B: variable present — run 4 structural checks. Each
    # failure raises RuntimeError with the matching err-id; no
    # zero-fill, no graceful continuation.
    arr = ds["dist_from_sea"].values

    # (a) Shape match.
    if arr.shape != (n,):
        raise RuntimeError(
            f"{ERR_DIST_FROM_SEA_SHAPE_MISMATCH}: dist_from_sea shape "
            f"{arr.shape} != expected ({n},). Stale NC built against a "
            "different mesh? Rebuild with "
            "build_h3_multires_landscape.py."
        )

    # (b) No NaN/Inf on water cells.
    water_arr = arr[mesh.water_mask]
    if not np.all(np.isfinite(water_arr)):
        n_bad = int(np.sum(~np.isfinite(water_arr)))
        raise RuntimeError(
            f"{ERR_DIST_FROM_SEA_NAN_ON_WATER}: dist_from_sea has "
            f"{n_bad} NaN/Inf value(s) on water cells. NC build is "
            "corrupt; rebuild required."
        )

    # (c) max > 0 — catches all-zero from a build that crashed
    # mid-Dijkstra but still wrote the file.
    if float(arr.max()) <= 0.0:
        raise RuntimeError(
            f"{ERR_DIST_FROM_SEA_ALL_ZERO}: dist_from_sea has max "
            f"{float(arr.max())}, expected > 0. NC was written but "
            "Dijkstra didn't run; rebuild required."
        )

    # (d) Sources exist — at least one OpenBaltic water cell at
    # distance 0. Skip the entire check if "OpenBaltic" not in
    # reach_names (non-Baltic mesh — no validation possible).
    if "OpenBaltic" in mesh.reach_names:
        ob_id = mesh.reach_names.index("OpenBaltic")
        ob_mask = (mesh.reach_id == ob_id) & mesh.water_mask
        if not ob_mask.any():
            # Mesh declares OpenBaltic reach but has zero water cells
            # in it. Degenerate mesh — raise (not a silent skip, per
            # pass-1 review-loop finding).
            raise RuntimeError(
                f"{ERR_DIST_FROM_SEA_NO_SOURCES}: mesh has reach "
                "'OpenBaltic' but no water_mask=True cells in it. "
                "Cannot validate sources; rebuild mesh."
            )
        if not np.any(arr[ob_mask] == 0.0):
            raise RuntimeError(
                f"{ERR_DIST_FROM_SEA_NO_SOURCES}: no OpenBaltic water "
                "cell has dist_from_sea == 0; the source set is empty "
                "or the build used a different source definition."
            )

    # All checks pass: inject (single cast, shared reference).
    arr32 = arr.astype(np.float32)
    env.fields["dist_from_sea"] = arr32
    mesh.dist_from_sea = arr32
    env._dormant_gradient_check_done = False
