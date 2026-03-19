"""Zone-based environment adapter for HexSim workspaces.

Serves the same ``fields`` dict interface as ``Environment`` but sources
temperature from a zone lookup table (Temperature Zones layer +
River Temperature.csv) instead of spatially-explicit NetCDF forcings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from heximpy.hxnparser import HexMap
from salmon_ibm.hexsim import HexMesh


class HexSimEnvironment:
    """Environmental forcing from HexSim zone-based data.

    Parameters
    ----------
    workspace_dir : path to HexSim workspace directory.
    mesh : HexMesh instance (water-only compacted).
    temperature_csv : filename of the River Temperature CSV
        (relative to workspace Analysis/Data Lookup/).
    """

    def __init__(self, workspace_dir: str | Path, mesh: HexMesh,
                 temperature_csv: str = "River Temperature.csv"):
        self.mesh = mesh
        ws = Path(workspace_dir)
        hex_dir = ws / "Spatial Data" / "Hexagons"

        # ── Temperature zones (optional — absent in non-fish workspaces) ──
        tz_path = hex_dir / "Temperature Zones" / "Temperature Zones.1.hxn"
        self._has_temperature = False
        if tz_path.exists():
            tz_hm = HexMap.from_file(tz_path)
            # Zone IDs for water cells (float → int, 0-based: subtract 1,
            # since zone values 1-45 map to CSV rows 0-44)
            tz_water = tz_hm.values[mesh._water_full_idx]
            self._zone_ids = np.clip(tz_water.astype(int) - 1, 0, None)

            # ── Temperature lookup table ─────────────────────────────────
            csv_path = ws / "Analysis" / "Data Lookup" / temperature_csv
            if csv_path.exists():
                # CSV has no header row; each row = a zone, each column = a timestep
                self._temp_table = np.loadtxt(csv_path, delimiter=",",
                                              dtype=np.float32)
                # Shape: (n_zones, n_timesteps)
                self.n_timesteps = self._temp_table.shape[1]
                self._has_temperature = True

        if not self._has_temperature:
            self._zone_ids = np.zeros(mesh.n_cells, dtype=int)
            self._temp_table = np.full((1, 1), 15.0, dtype=np.float32)
            self.n_timesteps = 1

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
        """Update fields for timestep *t* (wraps around)."""
        self.current_t = t
        t_idx = t % self.n_timesteps
        self._temp_buf[:] = self._temp_table[self._zone_ids, t_idx]

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
