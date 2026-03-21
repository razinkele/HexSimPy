"""Time-varying environmental fields interpolated onto the triangular mesh."""

from __future__ import annotations

import numpy as np
import xarray as xr

from salmon_ibm.mesh import TriMesh


class Environment:
    """Manages hourly environmental forcing on the mesh."""

    def __init__(self, config: dict, mesh: TriMesh, data_dir: str = "data"):
        self.mesh = mesh
        self.config = config
        self.data_dir = data_dir
        self.fields: dict[str, np.ndarray] = {}
        self._prev_ssh: np.ndarray | None = None
        self.current_t: int = -1

        phy_cfg = config["forcings"]["physics_surface"]
        phy_path = f"{data_dir}/{phy_cfg['file']}"
        self._phy = xr.open_dataset(phy_path, engine="scipy")
        self.n_timesteps = self._phy.sizes["time"]

        wind_cfg = config["forcings"]["winds"]
        wind_path = f"{data_dir}/{wind_cfg['file']}"
        self._wind = xr.open_dataset(wind_path, engine="scipy")

        riv_cfg = config["forcings"].get("river_discharge", {})
        riv_file = riv_cfg.get("file")
        if riv_file:
            self._riv = xr.open_dataset(f"{data_dir}/{riv_file}", engine="scipy")
        else:
            self._riv = None

        self._var = {
            "temperature": phy_cfg["temp_var"],
            "salinity": phy_cfg["salt_var"],
            "u_current": phy_cfg["u_var"],
            "v_current": phy_cfg["v_var"],
            "ssh": phy_cfg["ssh_var"],
        }

        # Pre-load and stack all fields for single-pass advance
        self._field_names = list(self._var.keys())
        arrays = []
        for field_name in self._field_names:
            var_name = self._var[field_name]
            data = self._phy[var_name].values  # (n_times, ...)
            arrays.append(data.reshape(self.n_timesteps, -1))  # flatten spatial dims
        self._stacked = np.stack(arrays, axis=-1)  # (n_times, n_nodes, 5)

        # Close datasets no longer needed after preloading
        self._phy.close()
        self._phy = None
        self._wind.close()
        self._wind = None
        if self._riv is not None:
            self._riv.close()
            self._riv = None

    def advance(self, t: int):
        self._prev_ssh = self.fields.get("ssh")
        self.current_t = t
        t_idx = t % self.n_timesteps

        # Single fancy-index + mean for all 5 fields at once
        all_raw = self._stacked[t_idx]  # (n_nodes, 5)
        tri_vals = all_raw[self.mesh.triangles].mean(axis=1)  # (n_tri, 5)
        for i, name in enumerate(self._field_names):
            self.fields[name] = tri_vals[:, i]

    def sample(self, tri_idx: int) -> dict[str, float]:
        return {name: float(arr[tri_idx]) for name, arr in self.fields.items()}

    def gradient(self, field_name: str, tri_idx: int) -> tuple[float, float]:
        return self.mesh.gradient(self.fields[field_name], tri_idx)

    def dSSH_dt(self, tri_idx: int) -> float:
        if self._prev_ssh is None:
            return 0.0
        return float(self.fields["ssh"][tri_idx] - self._prev_ssh[tri_idx])

    def dSSH_dt_array(self) -> np.ndarray:
        """Rate of SSH change (m/timestep) for all triangles."""
        if self._prev_ssh is None:
            return np.zeros(self.mesh.n_triangles)
        return self.fields["ssh"] - self._prev_ssh

    def close(self):
        pass  # Datasets closed after preloading in __init__
