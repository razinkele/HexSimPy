"""Zone-based environment adapter for HexSim workspaces.

Serves the same ``fields`` dict interface as ``Environment`` but sources
temperature from a zone lookup table (Temperature Zones layer +
River Temperature.csv) instead of spatially-explicit NetCDF forcings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range

from heximpy.hxnparser import HexMap
from salmon_ibm.hexsim import HexMesh


@njit(cache=True, parallel=True)
def _gather_by_zone(col: np.ndarray, zone_ids: np.ndarray, out: np.ndarray) -> None:
    """Scatter a small per-zone vector to per-cell buffer.

    For each cell i, out[i] = col[zone_ids[i]]. zone_ids is int32 and
    col is float64, both contiguous; n_cells can be ~O(1M) on Columbia.
    Parallel prange over cells — independent writes, no data race.
    """
    n = out.shape[0]
    for i in prange(n):
        out[i] = col[zone_ids[i]]


def _validate_temp_table(data: np.ndarray, n_zones: int) -> None:
    """Validate temperature lookup table dimensions."""
    if data.ndim != 2:
        raise ValueError(f"Temperature CSV must be 2D, got shape {data.shape}")
    if data.shape[0] != n_zones:
        raise ValueError(
            f"Temperature CSV shape {data.shape} doesn't match "
            f"expected {n_zones} zones (rows). Got {data.shape[0]} rows."
        )


class HexSimEnvironment:
    """Environmental forcing from HexSim zone-based data.

    Parameters
    ----------
    workspace_dir : path to HexSim workspace directory.
    mesh : HexMesh instance (water-only compacted).
    temperature_csv : filename of the River Temperature CSV
        (relative to workspace Analysis/Data Lookup/).
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        mesh: HexMesh,
        temperature_csv: str = "River Temperature.csv",
    ):
        self.mesh = mesh
        ws = Path(workspace_dir)
        hex_dir = ws / "Spatial Data" / "Hexagons"

        # ── Temperature zones (optional — absent in non-fish workspaces) ──
        tz_path = hex_dir / "Temperature Zones" / "Temperature Zones.1.hxn"
        self._has_temperature = False
        if tz_path.exists():
            tz_hm = HexMap.from_file(tz_path)
            # Zone IDs for water cells (float → int, 0-based: subtract 1,
            # since zone values 1-45 map to CSV rows 0-44).
            # int32 suffices (max zone ~45) and halves gather indexing bandwidth
            # vs int64 for million-cell meshes.
            tz_water = tz_hm.values[mesh._water_full_idx]
            self._zone_ids = np.clip(tz_water.astype(np.int32) - 1, 0, None).astype(
                np.int32
            )

            # ── Temperature lookup table ─────────────────────────────────
            csv_path = ws / "Analysis" / "Data Lookup" / temperature_csv
            if csv_path.exists():
                # CSV has no header row; each row = a zone, each column = a timestep
                self._temp_table = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
                n_zones_found = int(self._zone_ids.max()) + 1
                _validate_temp_table(self._temp_table, n_zones_found)
                # Shape: (n_zones, n_timesteps)
                self.n_timesteps = self._temp_table.shape[1]
                self._has_temperature = True

        if not self._has_temperature:
            import warnings

            warnings.warn(
                "No temperature zone data found in workspace. "
                "Using constant 15°C for all cells. "
                "Temperature-dependent processes (respiration, behavior, thermal mortality) "
                "will not function correctly.",
                UserWarning,
                stacklevel=2,
            )
            self._zone_ids = np.zeros(mesh.n_cells, dtype=np.int32)
            self._temp_table = np.full((1, 1), 15.0, dtype=np.float32)
            self.n_timesteps = 1

        # Transpose the temp table to (n_timesteps, n_zones) in float64 once,
        # so advance() can slice one contiguous row per step and hand it to
        # _gather_by_zone without per-step conversion or gather-from-strided.
        # The table is tiny (e.g., 2975 * 45 * 8 = ~1 MB) — memory is cheap.
        self._temp_table_T = np.ascontiguousarray(
            self._temp_table.T.astype(np.float64)
        )

        # ── Upstream gradient (static SSH proxy) ─────────────────────────
        up_path = hex_dir / "Gradient [ upstream ]" / "Gradient [ upstream ].1.hxn"
        if up_path.exists():
            up_hm = HexMap.from_file(up_path)
            self._upstream = up_hm.values[mesh._water_full_idx].astype(np.float64)
        else:
            self._upstream = np.zeros(mesh.n_cells)

        # Negate so that lower values = upstream (matching SSH semantics
        # where movement uses SSH descent for upstream migration)
        max_up = self._upstream.max() if self._upstream.max() > 0 else 1.0
        self._ssh_static = -(self._upstream / max_up)

        # ── Fields dict (populated by advance()) ─────────────────────────
        n = mesh.n_cells
        self._temp_buf = np.zeros(n, dtype=np.float64)
        self._zero_buf = np.zeros(n, dtype=np.float64)
        self._zero_buf.flags.writeable = False
        self.fields: dict[str, np.ndarray] = {
            "temperature": self._temp_buf,
            "salinity": np.zeros(n, dtype=np.float64),
            "ssh": self._ssh_static,
            "u_current": np.zeros(n, dtype=np.float64),
            "v_current": np.zeros(n, dtype=np.float64),
        }
        self.current_t: int = -1

    def advance(self, t: int) -> None:
        """Update fields for timestep *t* (wraps around).

        Slice one contiguous row from the transposed table (shape n_zones,
        ~O(1 KB)) and scatter to cells via a Numba gather kernel. This was
        previously a 2D fancy index `self._temp_table[self._zone_ids, t_idx]`
        doing a strided gather across n_cells (~O(1M)) — ~9x slower.
        """
        self.current_t = t
        t_idx = t % self.n_timesteps
        col = self._temp_table_T[t_idx]  # (n_zones,), contiguous, float64
        _gather_by_zone(col, self._zone_ids, self._temp_buf)

    def sample(self, cell_idx: int) -> dict[str, float]:
        """All field values at a single cell."""
        return {name: float(arr[cell_idx]) for name, arr in self.fields.items()}

    def gradient(self, field_name: str, cell_idx: int) -> tuple[float, float]:
        """Compute field gradient via mesh.gradient()."""
        return self.mesh.gradient(self.fields[field_name], cell_idx)

    def dSSH_dt(self, cell_idx: int) -> float:
        """Rate of change of SSH. Always 0 for static river gradient."""
        return 0.0

    def dSSH_dt_array(self) -> np.ndarray:
        """Rate of SSH change (m/timestep) for all cells. Always 0 for static gradient."""
        return self._zero_buf

    def close(self) -> None:
        """No-op (no open file handles to release)."""
        pass
