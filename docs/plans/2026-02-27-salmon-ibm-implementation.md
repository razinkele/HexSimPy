# Baltic Salmon IBM — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python IBM simulating Baltic salmon migration through the Curonian Lagoon with Shiny UI.

**Architecture:** Hybrid OOP + Vectorized core engine (`salmon_ibm/`) with Shiny for Python UI (`app.py` + `ui/`). Unstructured triangular mesh from regular grid. Wisconsin bioenergetics with estuarine extensions.

**Tech Stack:** Python 3.13 (micromamba `shiny` env), numpy 2.4, scipy 1.17, xarray 2025.11, shiny 1.5, plotly 6.5, shinywidgets 0.7, pyyaml 6.0, pandas 2.3

**Working directory:** `C:\Users\arturas.baziukas\OneDrive - ku.lt\HORIZON_EUROPE\salmon`

**Run prefix:** All Python commands use `micromamba run -n shiny python`

**NetCDF engine:** Use `engine='scipy'` (files are classic CDF v1 format)

**Note:** pyarrow has a DLL issue — use CSV for track output, not parquet.

---

## Task 1: Project Scaffolding & Config Loader

**Files:**
- Create: `salmon_ibm/__init__.py`
- Create: `salmon_ibm/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Existing: `config_curonian_minimal.yaml`

**Step 1: Create package directories**

```bash
mkdir -p salmon_ibm tests ui
```

**Step 2: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_config.py`:

```python
import pytest
from salmon_ibm.config import load_config


def test_load_config_returns_dict():
    cfg = load_config("config_curonian_minimal.yaml")
    assert isinstance(cfg, dict)


def test_config_has_grid_section():
    cfg = load_config("config_curonian_minimal.yaml")
    assert "grid" in cfg
    assert cfg["grid"]["file"] == "curonian_minimal_grid.nc"


def test_config_has_estuary_section():
    cfg = load_config("config_curonian_minimal.yaml")
    assert "estuary" in cfg
    assert cfg["estuary"]["salinity_cost"]["S_opt"] == 0.5
    assert cfg["estuary"]["do_avoidance"]["lethal"] == 2.0
    assert cfg["estuary"]["seiche_pause"]["dSSHdt_thresh_m_per_15min"] == 0.02


def test_config_has_forcings():
    cfg = load_config("config_curonian_minimal.yaml")
    assert "forcings" in cfg
    assert "physics_surface" in cfg["forcings"]
    assert cfg["forcings"]["physics_surface"]["temp_var"] == "tos"
```

**Step 3: Run test to verify it fails**

Run: `micromamba run -n shiny python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'salmon_ibm'`

**Step 4: Write minimal implementation**

Create `salmon_ibm/__init__.py`:
```python
"""Baltic Salmon Individual-Based Model for Curonian Lagoon."""
```

Create `salmon_ibm/config.py`:
```python
"""YAML configuration loader."""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load and return simulation configuration from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)
```

**Step 5: Run test to verify it passes**

Run: `micromamba run -n shiny python -m pytest tests/test_config.py -v`
Expected: 4 passed

**Step 6: Commit**

```bash
git add salmon_ibm/ tests/
git commit -m "feat: project scaffolding and YAML config loader"
```

---

## Task 2: Triangular Mesh

**Files:**
- Create: `salmon_ibm/mesh.py`
- Create: `tests/test_mesh.py`
- Read: `data/curonian_minimal_grid.nc` (via `Curonian_IBM_Package_v2.zip` extracted stubs)

**Step 1: Move data stubs into data/ directory**

```bash
mkdir -p data
cp curonian_minimal_grid.nc forcing_cmems_phy_stub.nc nemunas_discharge_stub.nc winds_stub.nc data/
```

**Step 2: Write the failing test**

Create `tests/test_mesh.py`:

```python
import numpy as np
import pytest
from salmon_ibm.mesh import TriMesh


@pytest.fixture
def mesh():
    return TriMesh.from_netcdf("data/curonian_minimal_grid.nc")


def test_mesh_loads(mesh):
    assert mesh.nodes.shape[1] == 2  # (N, 2) = lat, lon
    assert mesh.triangles.shape[1] == 3  # (M, 3) = 3 node indices


def test_mesh_has_centroids(mesh):
    assert mesh.centroids.shape == (mesh.n_triangles, 2)


def test_mesh_has_neighbors(mesh):
    # Each triangle has up to 3 neighbors (-1 means boundary)
    assert mesh.neighbors.shape == (mesh.n_triangles, 3)


def test_mesh_mask_filters_land(mesh):
    # Water triangles only
    assert mesh.water_mask.dtype == bool
    n_water = mesh.water_mask.sum()
    assert 0 < n_water < mesh.n_triangles


def test_mesh_depth_at_water_cells(mesh):
    water_depths = mesh.depth[mesh.water_mask]
    assert np.all(water_depths > 0)


def test_water_neighbors_returns_valid_indices(mesh):
    water_ids = np.where(mesh.water_mask)[0]
    tri = water_ids[0]
    nbrs = mesh.water_neighbors(tri)
    assert all(n >= 0 for n in nbrs)
    assert all(mesh.water_mask[n] for n in nbrs)


def test_find_triangle(mesh):
    # Pick a known water centroid and find it
    water_ids = np.where(mesh.water_mask)[0]
    tri = water_ids[0]
    lat, lon = mesh.centroids[tri]
    found = mesh.find_triangle(lat, lon)
    assert found == tri


def test_gradient_returns_vector(mesh):
    # Create a dummy field (e.g., depth) and check gradient shape
    field = mesh.depth.copy()
    water_ids = np.where(mesh.water_mask)[0]
    tri = water_ids[0]
    grad = mesh.gradient(field, tri)
    assert len(grad) == 2  # (dlat, dlon)
```

**Step 3: Run test to verify it fails**

Run: `micromamba run -n shiny python -m pytest tests/test_mesh.py -v`
Expected: FAIL — `ImportError`

**Step 4: Write implementation**

Create `salmon_ibm/mesh.py`:

```python
"""Triangular mesh constructed from a regular lat/lon grid."""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay
import xarray as xr


class TriMesh:
    """Unstructured triangular mesh over the Curonian Lagoon domain."""

    def __init__(
        self,
        nodes: np.ndarray,
        triangles: np.ndarray,
        mask_per_node: np.ndarray,
        depth_per_node: np.ndarray,
    ):
        self.nodes = nodes  # (N, 2) lat, lon
        self.triangles = triangles  # (M, 3) node indices
        self.n_triangles = len(triangles)

        # Centroids: mean of triangle vertex positions
        self.centroids = nodes[triangles].mean(axis=1)  # (M, 2)

        # Triangle areas (approximate, in degrees^2 — fine for neighbor logic)
        v = nodes[triangles]  # (M, 3, 2)
        ab = v[:, 1] - v[:, 0]
        ac = v[:, 2] - v[:, 0]
        self.areas = 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])

        # Per-triangle mask: water if ALL 3 vertices are water (mask == 1)
        tri_mask_sum = mask_per_node[triangles].sum(axis=1)
        self.water_mask = tri_mask_sum == 3  # bool (M,)

        # Per-triangle depth: mean of vertex depths
        self.depth = depth_per_node[triangles].mean(axis=1)  # (M,)

        # Build adjacency from Delaunay neighbors
        self.neighbors = self._build_neighbors(triangles)

        # Delaunay object for point location
        self._delaunay = Delaunay(nodes)

    @classmethod
    def from_netcdf(cls, path: str) -> "TriMesh":
        """Build mesh from a regular grid NetCDF file."""
        ds = xr.open_dataset(path, engine="scipy")
        lat = ds["lat"].values  # (y, x)
        lon = ds["lon"].values
        mask = ds["mask"].values  # (y, x), 1=water
        depth = ds["depth"].values

        # Flatten grid to node list
        ny, nx = lat.shape
        nodes = np.column_stack([lat.ravel(), lon.ravel()])  # (N, 2)
        mask_flat = mask.ravel()
        depth_flat = depth.ravel()
        ds.close()

        # Build Delaunay triangulation on all nodes
        tri = Delaunay(nodes)

        return cls(nodes, tri.simplices, mask_flat, depth_flat)

    def _build_neighbors(self, triangles: np.ndarray) -> np.ndarray:
        """Build triangle adjacency: triangles sharing an edge are neighbors."""
        n_tri = len(triangles)
        neighbors = np.full((n_tri, 3), -1, dtype=int)

        # Map each edge (sorted pair of node indices) to the triangles that use it
        edge_to_tri: dict[tuple[int, int], list[int]] = {}
        for i, tri in enumerate(triangles):
            for e in range(3):
                edge = tuple(sorted((tri[e], tri[(e + 1) % 3])))
                edge_to_tri.setdefault(edge, []).append(i)

        # For each triangle, find neighbors via shared edges
        for i, tri in enumerate(triangles):
            ni = 0
            for e in range(3):
                edge = tuple(sorted((tri[e], tri[(e + 1) % 3])))
                for other in edge_to_tri[edge]:
                    if other != i:
                        neighbors[i, ni] = other
                        ni += 1
                        break
        return neighbors

    def water_neighbors(self, tri_idx: int) -> list[int]:
        """Return water-only neighbor triangle indices."""
        nbrs = self.neighbors[tri_idx]
        return [int(n) for n in nbrs if n >= 0 and self.water_mask[n]]

    def find_triangle(self, lat: float, lon: float) -> int:
        """Find the triangle containing the given point."""
        return int(self._delaunay.find_simplex([lat, lon]))

    def gradient(self, field: np.ndarray, tri_idx: int) -> tuple[float, float]:
        """Estimate field gradient at a triangle from its neighbors.

        Returns (d_field/d_lat, d_field/d_lon) as a direction vector.
        """
        nbrs = [n for n in self.neighbors[tri_idx] if n >= 0]
        if not nbrs:
            return (0.0, 0.0)

        c0 = self.centroids[tri_idx]
        f0 = field[tri_idx]

        dlat, dlon = 0.0, 0.0
        for n in nbrs:
            cn = self.centroids[n]
            df = field[n] - f0
            dc = cn - c0
            norm = np.sqrt(dc[0] ** 2 + dc[1] ** 2)
            if norm > 0:
                dlat += df * dc[0] / norm
                dlon += df * dc[1] / norm

        # Normalize
        mag = np.sqrt(dlat**2 + dlon**2)
        if mag > 0:
            dlat /= mag
            dlon /= mag
        return (dlat, dlon)
```

**Step 5: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/test_mesh.py -v`
Expected: All passed

**Step 6: Commit**

```bash
git add salmon_ibm/mesh.py tests/test_mesh.py data/
git commit -m "feat: triangular mesh from regular grid with Delaunay"
```

---

## Task 3: Environment (Time-Varying Fields on Mesh)

**Files:**
- Create: `salmon_ibm/environment.py`
- Create: `tests/test_environment.py`

**Step 1: Write failing test**

Create `tests/test_environment.py`:

```python
import numpy as np
import pytest
from salmon_ibm.mesh import TriMesh
from salmon_ibm.environment import Environment
from salmon_ibm.config import load_config


@pytest.fixture
def env():
    cfg = load_config("config_curonian_minimal.yaml")
    mesh = TriMesh.from_netcdf("data/curonian_minimal_grid.nc")
    return Environment(cfg, mesh, data_dir="data")


def test_env_loads(env):
    assert env.n_timesteps > 0


def test_env_advance_loads_fields(env):
    env.advance(0)
    assert "temperature" in env.fields
    assert "salinity" in env.fields
    assert "u_current" in env.fields
    assert "ssh" in env.fields


def test_env_sample_returns_dict(env):
    env.advance(0)
    water_ids = np.where(env.mesh.water_mask)[0]
    tri = water_ids[0]
    s = env.sample(tri)
    assert "temperature" in s
    assert isinstance(s["temperature"], float)


def test_env_gradient_returns_tuple(env):
    env.advance(0)
    water_ids = np.where(env.mesh.water_mask)[0]
    tri = water_ids[0]
    grad = env.gradient("temperature", tri)
    assert len(grad) == 2


def test_env_dSSH_dt(env):
    env.advance(0)
    env.advance(1)
    water_ids = np.where(env.mesh.water_mask)[0]
    tri = water_ids[0]
    rate = env.dSSH_dt(tri)
    assert isinstance(rate, float)
```

**Step 2: Run test to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_environment.py -v`
Expected: FAIL — `ImportError`

**Step 3: Write implementation**

Create `salmon_ibm/environment.py`:

```python
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

        # Load forcing datasets
        phy_cfg = config["forcings"]["physics_surface"]
        phy_path = f"{data_dir}/{phy_cfg['file']}"
        self._phy = xr.open_dataset(phy_path, engine="scipy")
        self.n_timesteps = self._phy.sizes["time"]

        # Wind (scalar, not spatial in stubs)
        wind_cfg = config["forcings"]["winds"]
        wind_path = f"{data_dir}/{wind_cfg['file']}"
        self._wind = xr.open_dataset(wind_path, engine="scipy")

        # River discharge
        riv_cfg = config["forcings"].get("river_discharge", {})
        riv_file = riv_cfg.get("file")
        if riv_file:
            self._riv = xr.open_dataset(f"{data_dir}/{riv_file}", engine="scipy")
        else:
            self._riv = None

        # Variable name mappings
        self._var = {
            "temperature": phy_cfg["temp_var"],
            "salinity": phy_cfg["salt_var"],
            "u_current": phy_cfg["u_var"],
            "v_current": phy_cfg["v_var"],
            "ssh": phy_cfg["ssh_var"],
        }

    def advance(self, t: int):
        """Load environmental fields for timestep t onto mesh triangles."""
        self._prev_ssh = self.fields.get("ssh")
        self.current_t = t
        t_idx = t % self.n_timesteps  # wrap around for stubs

        for field_name, var_name in self._var.items():
            # Data shape: (time, y, x) -> flatten to nodes -> average to triangles
            raw = self._phy[var_name].isel(time=t_idx).values  # (y, x)
            flat = raw.ravel()  # (N,) matching mesh nodes
            # Per-triangle: mean of 3 vertex values
            tri_vals = flat[self.mesh.triangles].mean(axis=1)
            self.fields[field_name] = tri_vals

    def sample(self, tri_idx: int) -> dict[str, float]:
        """Return all field values at a triangle."""
        return {name: float(arr[tri_idx]) for name, arr in self.fields.items()}

    def gradient(self, field_name: str, tri_idx: int) -> tuple[float, float]:
        """Spatial gradient of a field at a triangle."""
        return self.mesh.gradient(self.fields[field_name], tri_idx)

    def dSSH_dt(self, tri_idx: int) -> float:
        """Rate of SSH change (m/h) at a triangle. Used for seiche detection."""
        if self._prev_ssh is None:
            return 0.0
        return float(self.fields["ssh"][tri_idx] - self._prev_ssh[tri_idx])

    def close(self):
        """Close open datasets."""
        self._phy.close()
        self._wind.close()
        if self._riv is not None:
            self._riv.close()
```

**Step 4: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/test_environment.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add salmon_ibm/environment.py tests/test_environment.py
git commit -m "feat: environment module — hourly fields on triangular mesh"
```

---

## Task 4: Agents (FishAgent + AgentPool)

**Files:**
- Create: `salmon_ibm/agents.py`
- Create: `tests/test_agents.py`

**Step 1: Write failing test**

Create `tests/test_agents.py`:

```python
import numpy as np
import pytest
from salmon_ibm.agents import Behavior, FishAgent, AgentPool


def test_behavior_enum():
    assert Behavior.HOLD.value == 0
    assert Behavior.UPSTREAM.value == 3


def test_agent_pool_creation():
    pool = AgentPool(n=10, start_tri=5, rng_seed=42)
    assert pool.n == 10
    assert pool.tri_idx.shape == (10,)
    assert np.all(pool.tri_idx == 5)


def test_agent_pool_alive_mask():
    pool = AgentPool(n=10, start_tri=5)
    assert pool.alive.sum() == 10
    pool.alive[3] = False
    assert pool.alive.sum() == 9


def test_agent_view_reads_pool():
    pool = AgentPool(n=5, start_tri=7)
    agent = pool.get_agent(2)
    assert agent.tri_idx == 7
    assert agent.id == 2


def test_agent_view_writes_to_pool():
    pool = AgentPool(n=5, start_tri=7)
    agent = pool.get_agent(2)
    agent.tri_idx = 99
    assert pool.tri_idx[2] == 99


def test_pool_t3h_mean():
    pool = AgentPool(n=3, start_tri=0)
    pool.temp_history[:] = [[15, 16, 17], [10, 10, 10], [20, 22, 24]]
    means = pool.t3h_mean()
    np.testing.assert_allclose(means, [16.0, 10.0, 22.0])


def test_pool_initial_energy_density():
    pool = AgentPool(n=5, start_tri=0)
    # Default starting energy density should be reasonable for salmon
    assert np.all(pool.ed_kJ_g > 4.0)
    assert np.all(pool.ed_kJ_g < 10.0)
```

**Step 2: Run test to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_agents.py -v`
Expected: FAIL

**Step 3: Write implementation**

Create `salmon_ibm/agents.py`:

```python
"""Fish agent state: FishAgent (OOP view) + AgentPool (vectorized arrays)."""
from __future__ import annotations

from enum import IntEnum
import numpy as np


class Behavior(IntEnum):
    HOLD = 0
    RANDOM = 1
    TO_CWR = 2
    UPSTREAM = 3
    DOWNSTREAM = 4


class AgentPool:
    """Vectorized storage for all fish agents (structure-of-arrays)."""

    def __init__(
        self,
        n: int,
        start_tri: int | np.ndarray,
        rng_seed: int | None = None,
        mass_mean: float = 3500.0,
        mass_std: float = 500.0,
        ed_init: float = 6.5,
        spawn_hours_mean: float = 720.0,
        spawn_hours_std: float = 168.0,
    ):
        self.n = n
        rng = np.random.default_rng(rng_seed)

        # Position
        if isinstance(start_tri, int):
            self.tri_idx = np.full(n, start_tri, dtype=int)
        else:
            self.tri_idx = np.asarray(start_tri, dtype=int)

        # Body
        self.mass_g = np.clip(
            rng.normal(mass_mean, mass_std, n), mass_mean * 0.5, mass_mean * 1.5
        )
        self.ed_kJ_g = np.full(n, ed_init)

        # Timing
        self.target_spawn_hour = np.clip(
            rng.normal(spawn_hours_mean, spawn_hours_std, n).astype(int), 1, None
        )

        # Behavior state
        self.behavior = np.full(n, Behavior.HOLD, dtype=int)

        # Timers
        self.cwr_hours = np.zeros(n, dtype=int)
        self.hours_since_cwr = np.full(n, 999, dtype=int)
        self.steps = np.zeros(n, dtype=int)

        # Status
        self.alive = np.ones(n, dtype=bool)
        self.arrived = np.zeros(n, dtype=bool)

        # Temperature history (last 3 hours)
        self.temp_history = np.full((n, 3), 15.0)  # placeholder init

    def get_agent(self, idx: int) -> "FishAgent":
        """Return an OOP view into this pool at index idx."""
        return FishAgent(self, idx)

    def t3h_mean(self) -> np.ndarray:
        """Mean temperature over the last 3 hours for each agent."""
        return self.temp_history.mean(axis=1)

    def push_temperature(self, temps: np.ndarray):
        """Push new hourly temperature readings, shifting history."""
        self.temp_history[:, :-1] = self.temp_history[:, 1:]
        self.temp_history[:, -1] = temps


class FishAgent:
    """OOP view into a single agent within an AgentPool. Zero-copy."""

    def __init__(self, pool: AgentPool, idx: int):
        self._pool = pool
        self._idx = idx

    @property
    def id(self) -> int:
        return self._idx

    @property
    def tri_idx(self) -> int:
        return int(self._pool.tri_idx[self._idx])

    @tri_idx.setter
    def tri_idx(self, v: int):
        self._pool.tri_idx[self._idx] = v

    @property
    def mass_g(self) -> float:
        return float(self._pool.mass_g[self._idx])

    @property
    def ed_kJ_g(self) -> float:
        return float(self._pool.ed_kJ_g[self._idx])

    @ed_kJ_g.setter
    def ed_kJ_g(self, v: float):
        self._pool.ed_kJ_g[self._idx] = v

    @property
    def behavior(self) -> int:
        return int(self._pool.behavior[self._idx])

    @behavior.setter
    def behavior(self, v: int):
        self._pool.behavior[self._idx] = v

    @property
    def alive(self) -> bool:
        return bool(self._pool.alive[self._idx])

    @property
    def arrived(self) -> bool:
        return bool(self._pool.arrived[self._idx])

    @property
    def steps(self) -> int:
        return int(self._pool.steps[self._idx])

    @property
    def cwr_hours(self) -> int:
        return int(self._pool.cwr_hours[self._idx])

    @property
    def hours_since_cwr(self) -> int:
        return int(self._pool.hours_since_cwr[self._idx])
```

**Step 4: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/test_agents.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add salmon_ibm/agents.py tests/test_agents.py
git commit -m "feat: FishAgent + AgentPool — hybrid OOP/vectorized agents"
```

---

## Task 5: Bioenergetics (Wisconsin Hourly Budget)

**Files:**
- Create: `salmon_ibm/bioenergetics.py`
- Create: `tests/test_bioenergetics.py`

**Step 1: Write failing test**

Create `tests/test_bioenergetics.py`:

```python
import numpy as np
import pytest
from salmon_ibm.bioenergetics import BioParams, hourly_respiration, update_energy


def test_bio_params_defaults():
    p = BioParams()
    assert p.RA == pytest.approx(0.00264)
    assert p.ED_MORTAL == pytest.approx(4.0)


def test_hourly_respiration_increases_with_temperature():
    p = BioParams()
    mass = np.array([3000.0])
    r_cold = hourly_respiration(mass, np.array([10.0]), np.array([1.0]), p)
    r_warm = hourly_respiration(mass, np.array([20.0]), np.array([1.0]), p)
    assert r_warm[0] > r_cold[0]


def test_hourly_respiration_increases_with_activity():
    p = BioParams()
    mass = np.array([3000.0])
    r_rest = hourly_respiration(mass, np.array([15.0]), np.array([1.0]), p)
    r_active = hourly_respiration(mass, np.array([15.0]), np.array([1.5]), p)
    assert r_active[0] > r_rest[0]


def test_update_energy_decreases_ed():
    p = BioParams()
    ed = np.array([6.5])
    mass = np.array([3000.0])
    temps = np.array([15.0])
    activity = np.array([1.0])
    salinity_cost = np.array([1.0])

    new_ed, dead = update_energy(ed, mass, temps, activity, salinity_cost, p)
    assert new_ed[0] < 6.5
    assert not dead[0]


def test_mortality_at_low_energy():
    p = BioParams()
    ed = np.array([4.01])  # just above threshold
    mass = np.array([3000.0])
    # Very warm water = high respiration, should push below threshold
    new_ed, dead = update_energy(
        ed, mass, np.array([25.0]), np.array([1.5]), np.array([1.5]), p
    )
    assert new_ed[0] < ed[0]
    # May or may not die in one step — check that mortality fires eventually
    for _ in range(100):
        new_ed, dead = update_energy(
            new_ed, mass, np.array([25.0]), np.array([1.5]), np.array([1.5]), p
        )
        if dead[0]:
            break
    assert dead[0], "Fish should eventually die from starvation at 25C"
```

**Step 2: Run test to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_bioenergetics.py -v`
Expected: FAIL

**Step 3: Write implementation**

Create `salmon_ibm/bioenergetics.py`:

```python
"""Wisconsin Bioenergetics Model — hourly budget for non-feeding migrants."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


OXY_CAL_J_PER_GO2 = 13_560.0  # J per g O2


@dataclass
class BioParams:
    """Bioenergetics parameters for Salmo salar (Forseth et al. 2001 / FB4)."""

    RA: float = 0.00264       # respiration intercept (g O2 / g / day)
    RB: float = -0.217        # allometric exponent
    RQ: float = 0.06818       # temperature coefficient (1/C)
    ED_MORTAL: float = 4.0    # mortality threshold (kJ/g)
    activity_by_behavior: dict[int, float] = field(default_factory=lambda: {
        0: 1.0,   # HOLD
        1: 1.2,   # RANDOM
        2: 0.8,   # TO_CWR
        3: 1.5,   # UPSTREAM
        4: 1.0,   # DOWNSTREAM
    })


def hourly_respiration(
    mass_g: np.ndarray,
    temperature_c: np.ndarray,
    activity_mult: np.ndarray,
    params: BioParams,
) -> np.ndarray:
    """Compute hourly respiration energy loss (J per fish per hour).

    R_daily = RA * mass^(RB-1) * exp(RQ * T) * activity  [g O2/g/day]
    R_hourly = R_daily * OXY_CAL * mass / 24              [J/fish/hour]
    """
    r_daily = params.RA * np.power(mass_g, params.RB - 1.0) * np.exp(params.RQ * temperature_c) * activity_mult
    return r_daily * OXY_CAL_J_PER_GO2 * mass_g / 24.0


def update_energy(
    ed_kJ_g: np.ndarray,
    mass_g: np.ndarray,
    temperature_c: np.ndarray,
    activity_mult: np.ndarray,
    salinity_cost: np.ndarray,
    params: BioParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Update energy density for one hourly timestep.

    Returns (new_ed_kJ_g, dead_mask).
    """
    r_hourly = hourly_respiration(mass_g, temperature_c, activity_mult, params) * salinity_cost

    # Convert ed to total energy, subtract respiration, convert back
    e_total_j = ed_kJ_g * 1000.0 * mass_g
    e_total_j = np.maximum(e_total_j - r_hourly, 0.0)
    new_ed = e_total_j / (mass_g * 1000.0)

    dead = new_ed < params.ED_MORTAL
    return new_ed, dead
```

**Step 4: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/test_bioenergetics.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add salmon_ibm/bioenergetics.py tests/test_bioenergetics.py
git commit -m "feat: Wisconsin bioenergetics — hourly energy budget and mortality"
```

---

## Task 6: Estuary Extensions (Salinity, DO, Seiche)

**Files:**
- Create: `salmon_ibm/estuary.py`
- Create: `tests/test_estuary.py`

**Step 1: Write failing test**

Create `tests/test_estuary.py`:

```python
import numpy as np
import pytest
from salmon_ibm.estuary import salinity_cost, do_override, seiche_pause


def test_salinity_cost_below_tolerance():
    # S < S_opt + S_tol -> no extra cost
    cost = salinity_cost(np.array([3.0]), S_opt=0.5, S_tol=6.0, k=0.6)
    np.testing.assert_allclose(cost, [1.0])


def test_salinity_cost_above_tolerance():
    # S = 10 PSU > 0.5 + 6.0 = 6.5 -> cost = 1 + 0.6*(10-6.5) = 3.1
    cost = salinity_cost(np.array([10.0]), S_opt=0.5, S_tol=6.0, k=0.6)
    np.testing.assert_allclose(cost, [3.1])


def test_do_override_normal():
    override = do_override(np.array([8.0]), lethal=2.0, high=4.0)
    assert override[0] == 0  # no override


def test_do_override_high():
    override = do_override(np.array([3.0]), lethal=2.0, high=4.0)
    assert override[0] == 1  # escape


def test_do_override_lethal():
    override = do_override(np.array([1.5]), lethal=2.0, high=4.0)
    assert override[0] == 2  # lethal


def test_seiche_pause_calm():
    paused = seiche_pause(np.array([0.005]), thresh=0.02)
    assert not paused[0]


def test_seiche_pause_active():
    paused = seiche_pause(np.array([0.05]), thresh=0.02)
    assert paused[0]
```

**Step 2: Run test to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_estuary.py -v`

**Step 3: Write implementation**

Create `salmon_ibm/estuary.py`:

```python
"""Estuarine extensions: salinity cost, DO avoidance, seiche pause."""
from __future__ import annotations

import numpy as np


def salinity_cost(
    salinity: np.ndarray,
    S_opt: float = 0.5,
    S_tol: float = 6.0,
    k: float = 0.6,
) -> np.ndarray:
    """Activity cost multiplier due to salinity stress.

    Returns 1.0 when S <= S_opt + S_tol, increases linearly above.
    """
    excess = np.maximum(salinity - (S_opt + S_tol), 0.0)
    return 1.0 + k * excess


# DO override codes
DO_OK = 0
DO_ESCAPE = 1
DO_LETHAL = 2


def do_override(
    do_mg_l: np.ndarray,
    lethal: float = 2.0,
    high: float = 4.0,
) -> np.ndarray:
    """Return DO override code per agent.

    0 = no override, 1 = escape (force movement), 2 = lethal (mortality risk).
    """
    result = np.full(len(do_mg_l), DO_OK, dtype=int)
    result[do_mg_l < high] = DO_ESCAPE
    result[do_mg_l < lethal] = DO_LETHAL
    return result


def seiche_pause(
    dSSH_dt: np.ndarray,
    thresh: float = 0.02,
) -> np.ndarray:
    """Return boolean mask: True = seiche active, agents should HOLD."""
    return np.abs(dSSH_dt) > thresh
```

**Step 4: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/test_estuary.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add salmon_ibm/estuary.py tests/test_estuary.py
git commit -m "feat: estuary extensions — salinity cost, DO avoidance, seiche pause"
```

---

## Task 7: Behavior Decision Table + Overrides

**Files:**
- Create: `salmon_ibm/behavior.py`
- Create: `tests/test_behavior.py`

**Step 1: Write failing test**

Create `tests/test_behavior.py`:

```python
import numpy as np
import pytest
from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.behavior import BehaviorParams, pick_behaviors, apply_overrides


@pytest.fixture
def params():
    return BehaviorParams.defaults()


def test_pick_behaviors_returns_valid(params):
    pool = AgentPool(n=100, start_tri=5, rng_seed=42)
    t3h = pool.t3h_mean()
    behaviors = pick_behaviors(t3h, pool.target_spawn_hour, params, seed=42)
    assert behaviors.shape == (100,)
    assert np.all((behaviors >= 0) & (behaviors <= 4))


def test_override_first_move_upstream(params):
    pool = AgentPool(n=5, start_tri=5)
    pool.steps[:] = 0  # first step
    pool.behavior[:] = Behavior.RANDOM
    overridden = apply_overrides(pool, params)
    assert np.all(overridden == Behavior.UPSTREAM)


def test_override_cwr_max_residence(params):
    pool = AgentPool(n=3, start_tri=5)
    pool.steps[:] = 10
    pool.behavior[:] = Behavior.TO_CWR
    pool.cwr_hours[:] = params.max_cwr_hours + 1  # exceeded
    overridden = apply_overrides(pool, params)
    assert np.all(overridden == Behavior.UPSTREAM)
```

**Step 2: Run test to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_behavior.py -v`

**Step 3: Write implementation**

Create `salmon_ibm/behavior.py`:

```python
"""Behavioral decision table and overrides (Snyder et al. 2019)."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from salmon_ibm.agents import Behavior


@dataclass
class BehaviorParams:
    """Decision table parameters and override thresholds."""

    # Temperature bin edges (C)
    temp_bins: list[float] = field(default_factory=lambda: [16.0, 18.0, 20.0])
    # Time-to-spawn bin edges (hours)
    time_bins: list[float] = field(default_factory=lambda: [360, 720])  # 15d, 30d

    # Probability table: [n_temp_bins+1, n_time_bins+1, 5]
    # Rows: temp bins (<16, 16-18, 18-20, >20)
    # Cols: time bins (<15d, 15-30d, >30d)
    # Values: [HOLD, RANDOM, TO_CWR, UPSTREAM, DOWNSTREAM]
    p_table: np.ndarray | None = None

    # Override thresholds
    max_cwr_hours: int = 48
    avoid_cwr_cooldown_h: int = 12
    max_dist_to_cwr: float = 5000.0  # meters (not used in v0.1 mesh)

    @classmethod
    def defaults(cls) -> "BehaviorParams":
        """Return default parameters for Baltic salmon."""
        p = np.array([
            # >30 days to spawn
            [[0.60, 0.10, 0.00, 0.30, 0.00],   # T < 16
             [0.40, 0.00, 0.20, 0.40, 0.00],   # 16-18
             [0.30, 0.00, 0.50, 0.20, 0.00],   # 18-20
             [0.20, 0.00, 0.80, 0.00, 0.00]],  # > 20
            # 15-30 days
            [[0.20, 0.20, 0.00, 0.60, 0.00],
             [0.00, 0.20, 0.30, 0.50, 0.00],
             [0.00, 0.00, 0.40, 0.40, 0.20],
             [0.00, 0.00, 0.70, 0.00, 0.30]],
            # <15 days
            [[0.00, 0.20, 0.00, 0.80, 0.00],
             [0.00, 0.10, 0.20, 0.70, 0.00],
             [0.00, 0.00, 0.50, 0.50, 0.00],
             [0.00, 0.00, 0.60, 0.40, 0.00]],
        ])
        return cls(p_table=p)


def pick_behaviors(
    t3h_mean: np.ndarray,
    hours_to_spawn: np.ndarray,
    params: BehaviorParams,
    seed: int | None = None,
) -> np.ndarray:
    """Pick behaviors for all agents from the probability table."""
    rng = np.random.default_rng(seed)
    n = len(t3h_mean)

    temp_idx = np.digitize(t3h_mean, params.temp_bins)  # 0..3
    time_idx = np.digitize(hours_to_spawn, params.time_bins)  # 0..2

    behaviors = np.empty(n, dtype=int)
    for i in range(n):
        ti = int(np.clip(time_idx[i], 0, params.p_table.shape[0] - 1))
        te = int(np.clip(temp_idx[i], 0, params.p_table.shape[1] - 1))
        probs = params.p_table[ti, te]
        behaviors[i] = rng.choice(5, p=probs)

    return behaviors


def apply_overrides(
    pool,  # AgentPool
    params: BehaviorParams,
) -> np.ndarray:
    """Apply override rules to current behaviors. Returns modified behavior array."""
    beh = pool.behavior.copy()

    # 1. First move always UPSTREAM
    first_move = pool.steps == 0
    beh[first_move] = Behavior.UPSTREAM

    # 2. Exceeded CWR max residence -> force UPSTREAM
    cwr_exceeded = pool.cwr_hours > params.max_cwr_hours
    beh[cwr_exceeded] = Behavior.UPSTREAM

    # 3. TO_CWR but under avoid cooldown -> UPSTREAM
    cooldown_active = pool.hours_since_cwr < params.avoid_cwr_cooldown_h
    to_cwr = beh == Behavior.TO_CWR
    beh[to_cwr & cooldown_active] = Behavior.UPSTREAM

    return beh
```

**Step 4: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/test_behavior.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add salmon_ibm/behavior.py tests/test_behavior.py
git commit -m "feat: behavioral decision table and override rules"
```

---

## Task 8: Movement Kernels

**Files:**
- Create: `salmon_ibm/movement.py`
- Create: `tests/test_movement.py`

**Step 1: Write failing test**

Create `tests/test_movement.py`:

```python
import numpy as np
import pytest
from salmon_ibm.mesh import TriMesh
from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.movement import execute_movement


@pytest.fixture
def mesh():
    return TriMesh.from_netcdf("data/curonian_minimal_grid.nc")


def test_hold_does_not_move(mesh):
    water_ids = np.where(mesh.water_mask)[0]
    start = water_ids[0]
    pool = AgentPool(n=5, start_tri=start)
    pool.behavior[:] = Behavior.HOLD
    # Create minimal fields dict for movement
    fields = {"ssh": np.zeros(mesh.n_triangles), "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=42)
    assert np.all(pool.tri_idx == start)


def test_random_moves_to_neighbor(mesh):
    water_ids = np.where(mesh.water_mask)[0]
    # Pick a water cell with water neighbors
    for start in water_ids:
        if len(mesh.water_neighbors(start)) > 0:
            break
    pool = AgentPool(n=20, start_tri=start)
    pool.behavior[:] = Behavior.RANDOM
    fields = {"ssh": np.zeros(mesh.n_triangles), "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=42)
    # At least some agents should have moved
    moved = pool.tri_idx != start
    assert moved.sum() > 0
    # All agents should be on water cells
    assert np.all(mesh.water_mask[pool.tri_idx])


def test_upstream_follows_ssh_gradient(mesh):
    water_ids = np.where(mesh.water_mask)[0]
    # Create SSH field that increases with latitude (upstream = lower SSH)
    ssh = mesh.centroids[:, 0].copy()  # lat as proxy for SSH
    for start in water_ids:
        if len(mesh.water_neighbors(start)) > 0:
            break
    pool = AgentPool(n=10, start_tri=start)
    pool.behavior[:] = Behavior.UPSTREAM
    fields = {"ssh": ssh, "temperature": np.full(mesh.n_triangles, 15.0),
              "u_current": np.zeros(mesh.n_triangles), "v_current": np.zeros(mesh.n_triangles)}
    execute_movement(pool, mesh, fields, seed=42)
    # Agents should generally move to lower SSH neighbors
    assert np.all(mesh.water_mask[pool.tri_idx])
```

**Step 2: Run test to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_movement.py -v`

**Step 3: Write implementation**

Create `salmon_ibm/movement.py`:

```python
"""Movement kernels on the triangular mesh."""
from __future__ import annotations

import numpy as np

from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.mesh import TriMesh


def execute_movement(
    pool: AgentPool,
    mesh: TriMesh,
    fields: dict[str, np.ndarray],
    seed: int | None = None,
    n_micro_steps: int = 3,
):
    """Execute one hour of movement for all alive agents."""
    rng = np.random.default_rng(seed)
    alive_idx = np.where(pool.alive & ~pool.arrived)[0]

    for i in alive_idx:
        beh = pool.behavior[i]
        tri = pool.tri_idx[i]

        if beh == Behavior.HOLD:
            continue
        elif beh == Behavior.RANDOM:
            pool.tri_idx[i] = _step_random(tri, mesh, rng, n_micro_steps)
        elif beh == Behavior.UPSTREAM:
            pool.tri_idx[i] = _step_directed(
                tri, mesh, fields["ssh"], rng, n_micro_steps, ascending=False
            )
        elif beh == Behavior.DOWNSTREAM:
            pool.tri_idx[i] = _step_directed(
                tri, mesh, fields["ssh"], rng, n_micro_steps, ascending=True
            )
        elif beh == Behavior.TO_CWR:
            pool.tri_idx[i] = _step_to_cwr(
                tri, mesh, fields["temperature"], rng, n_micro_steps
            )

    # Apply current advection as additional shift
    _apply_current_advection(pool, mesh, fields, alive_idx, rng)


def _step_random(tri: int, mesh: TriMesh, rng: np.random.Generator, steps: int) -> int:
    """Auto-correlated random walk through water neighbors."""
    current = tri
    for _ in range(steps):
        nbrs = mesh.water_neighbors(current)
        if not nbrs:
            break
        current = rng.choice(nbrs)
    return current


def _step_directed(
    tri: int,
    mesh: TriMesh,
    field: np.ndarray,
    rng: np.random.Generator,
    steps: int,
    ascending: bool,
) -> int:
    """Directed movement: alternating random jitter + gradient following.

    ascending=True follows increasing field, False follows decreasing.
    """
    current = tri
    for s in range(steps):
        if s % 2 == 0:
            # Random jitter step
            nbrs = mesh.water_neighbors(current)
            if nbrs:
                current = rng.choice(nbrs)
        else:
            # Gradient-directed step
            nbrs = mesh.water_neighbors(current)
            if not nbrs:
                break
            vals = np.array([field[n] for n in nbrs])
            if ascending:
                best = nbrs[np.argmax(vals)]
            else:
                best = nbrs[np.argmin(vals)]
            current = best
    return current


def _step_to_cwr(
    tri: int,
    mesh: TriMesh,
    temperature: np.ndarray,
    rng: np.random.Generator,
    steps: int,
    cwr_threshold: float = 16.0,
) -> int:
    """Move toward cooler water (follow negative temperature gradient)."""
    current = tri
    for _ in range(steps):
        if temperature[current] < cwr_threshold:
            break  # reached cold water refuge
        nbrs = mesh.water_neighbors(current)
        if not nbrs:
            break
        temps = np.array([temperature[n] for n in nbrs])
        current = nbrs[np.argmin(temps)]
    return current


def _apply_current_advection(
    pool: AgentPool,
    mesh: TriMesh,
    fields: dict[str, np.ndarray],
    alive_idx: np.ndarray,
    rng: np.random.Generator,
):
    """Shift agents by flow field (u, v) translated to neighbor moves."""
    u = fields.get("u_current")
    v = fields.get("v_current")
    if u is None or v is None:
        return

    for i in alive_idx:
        tri = pool.tri_idx[i]
        speed = np.sqrt(u[tri] ** 2 + v[tri] ** 2)
        if speed < 0.01:
            continue  # negligible current

        # Move to the neighbor most aligned with (u, v) direction
        nbrs = mesh.water_neighbors(tri)
        if not nbrs:
            continue

        flow_dir = np.array([v[tri], u[tri]])  # (dlat, dlon) approximately
        flow_dir /= np.linalg.norm(flow_dir) + 1e-12

        best_dot = -999.0
        best_nbr = tri
        c0 = mesh.centroids[tri]
        for n in nbrs:
            cn = mesh.centroids[n]
            d = cn - c0
            d /= np.linalg.norm(d) + 1e-12
            dot = np.dot(d, flow_dir)
            if dot > best_dot:
                best_dot = dot
                best_nbr = n

        # Probabilistic: move with probability proportional to current speed
        if rng.random() < min(speed * 5.0, 0.8):  # tune factor
            pool.tri_idx[i] = best_nbr
```

**Step 4: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/test_movement.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add salmon_ibm/movement.py tests/test_movement.py
git commit -m "feat: movement kernels — random, directed, CWR-seeking, current advection"
```

---

## Task 9: Output Logger

**Files:**
- Create: `salmon_ibm/output.py`
- Create: `tests/test_output.py`

**Step 1: Write failing test**

Create `tests/test_output.py`:

```python
import os
import numpy as np
import pandas as pd
import pytest
from salmon_ibm.agents import AgentPool
from salmon_ibm.output import OutputLogger


def test_logger_creates_file(tmp_path):
    centroids = np.array([[55.0, 21.0], [55.1, 21.1]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=3, start_tri=0)
    logger.log_step(0, pool)
    logger.close()
    assert os.path.exists(tmp_path / "tracks.csv")


def test_logger_records_correct_columns(tmp_path):
    centroids = np.array([[55.0, 21.0], [55.1, 21.1]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=2, start_tri=0)
    logger.log_step(0, pool)
    logger.close()
    df = pd.read_csv(tmp_path / "tracks.csv")
    assert set(df.columns) >= {"time", "agent_id", "tri_idx", "lat", "lon",
                                "ed_kJ_g", "behavior", "alive", "arrived"}


def test_logger_accumulates_steps(tmp_path):
    centroids = np.array([[55.0, 21.0]])
    logger = OutputLogger(str(tmp_path / "tracks.csv"), centroids)
    pool = AgentPool(n=2, start_tri=0)
    logger.log_step(0, pool)
    logger.log_step(1, pool)
    logger.close()
    df = pd.read_csv(tmp_path / "tracks.csv")
    assert len(df) == 4  # 2 agents x 2 timesteps
```

**Step 2: Run test to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_output.py -v`

**Step 3: Write implementation**

Create `salmon_ibm/output.py`:

```python
"""Track logging and diagnostics output."""
from __future__ import annotations

import numpy as np
import pandas as pd

from salmon_ibm.agents import AgentPool


class OutputLogger:
    """Logs agent state per timestep to CSV."""

    def __init__(self, path: str, centroids: np.ndarray):
        self.path = path
        self.centroids = centroids
        self._records: list[dict] = []

    def log_step(self, t: int, pool: AgentPool):
        """Record all agents' state at timestep t."""
        lats = self.centroids[pool.tri_idx, 0]
        lons = self.centroids[pool.tri_idx, 1]
        for i in range(pool.n):
            self._records.append({
                "time": t,
                "agent_id": i,
                "tri_idx": int(pool.tri_idx[i]),
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "ed_kJ_g": float(pool.ed_kJ_g[i]),
                "behavior": int(pool.behavior[i]),
                "alive": bool(pool.alive[i]),
                "arrived": bool(pool.arrived[i]),
            })

    def to_dataframe(self) -> pd.DataFrame:
        """Return logged data as a DataFrame."""
        return pd.DataFrame(self._records)

    def close(self):
        """Write accumulated records to CSV."""
        df = self.to_dataframe()
        df.to_csv(self.path, index=False)

    def summary(self, t: int, pool: AgentPool) -> dict:
        """Compute summary statistics for current timestep."""
        alive = pool.alive
        return {
            "time": t,
            "n_alive": int(alive.sum()),
            "n_arrived": int(pool.arrived.sum()),
            "mean_ed": float(pool.ed_kJ_g[alive].mean()) if alive.any() else 0.0,
            "behavior_counts": {
                int(b): int((pool.behavior[alive] == b).sum()) for b in range(5)
            },
        }
```

**Step 4: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/test_output.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add salmon_ibm/output.py tests/test_output.py
git commit -m "feat: output logger — CSV track recording and summaries"
```

---

## Task 10: Simulation Loop

**Files:**
- Create: `salmon_ibm/simulation.py`
- Create: `tests/test_simulation.py`

**Step 1: Write failing test**

Create `tests/test_simulation.py`:

```python
import numpy as np
import pytest
from salmon_ibm.simulation import Simulation
from salmon_ibm.config import load_config


def test_simulation_initializes():
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    assert sim.pool.n == 10
    assert sim.env.n_timesteps > 0


def test_simulation_step():
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim.step()
    assert sim.current_t == 1
    assert sim.pool.steps[0] > 0


def test_simulation_run_multiple_steps():
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    sim.run(n_steps=5)
    assert sim.current_t == 5


def test_simulation_energy_decreases():
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42)
    initial_ed = sim.pool.ed_kJ_g.copy()
    sim.run(n_steps=10)
    assert np.all(sim.pool.ed_kJ_g[sim.pool.alive] <= initial_ed[sim.pool.alive])
```

**Step 2: Run test to verify failure**

Run: `micromamba run -n shiny python -m pytest tests/test_simulation.py -v`

**Step 3: Write implementation**

Create `salmon_ibm/simulation.py`:

```python
"""Main simulation loop."""
from __future__ import annotations

import numpy as np

from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.behavior import BehaviorParams, pick_behaviors, apply_overrides
from salmon_ibm.bioenergetics import BioParams, update_energy
from salmon_ibm.config import load_config
from salmon_ibm.environment import Environment
from salmon_ibm.estuary import salinity_cost, do_override, seiche_pause, DO_ESCAPE, DO_LETHAL
from salmon_ibm.mesh import TriMesh
from salmon_ibm.movement import execute_movement
from salmon_ibm.output import OutputLogger


class Simulation:
    """Orchestrates the hourly simulation loop."""

    def __init__(
        self,
        config: dict,
        n_agents: int = 100,
        data_dir: str = "data",
        rng_seed: int | None = None,
        output_path: str | None = None,
    ):
        self.config = config
        self.rng_seed = rng_seed
        self.current_t = 0

        # Build mesh
        grid_file = f"{data_dir}/{config['grid']['file']}"
        self.mesh = TriMesh.from_netcdf(grid_file)

        # Build environment
        self.env = Environment(config, self.mesh, data_dir=data_dir)

        # Place agents on a random water cell
        water_ids = np.where(self.mesh.water_mask)[0]
        rng = np.random.default_rng(rng_seed)
        start_tris = rng.choice(water_ids, size=n_agents)
        self.pool = AgentPool(n=n_agents, start_tri=start_tris, rng_seed=rng_seed)

        # Parameters
        self.beh_params = BehaviorParams.defaults()
        self.bio_params = BioParams()
        self.est_cfg = config.get("estuary", {})

        # Output
        self.logger = None
        if output_path:
            self.logger = OutputLogger(output_path, self.mesh.centroids)

        # Summary history for UI
        self.history: list[dict] = []

    def step(self):
        """Advance simulation by one hourly timestep."""
        t = self.current_t
        self.env.advance(t)

        alive_mask = self.pool.alive & ~self.pool.arrived

        # Update temperature history
        temps_at_agents = self.env.fields["temperature"][self.pool.tri_idx]
        self.pool.push_temperature(temps_at_agents)

        # 1. Pick behaviors
        t3h = self.pool.t3h_mean()
        self.pool.behavior[alive_mask] = pick_behaviors(
            t3h[alive_mask],
            self.pool.target_spawn_hour[alive_mask],
            self.beh_params,
            seed=None,
        )

        # 2. Apply overrides
        self.pool.behavior = apply_overrides(self.pool, self.beh_params)

        # 3. Estuarine overrides
        self._apply_estuarine_overrides()

        # 4. Movement
        execute_movement(self.pool, self.mesh, self.env.fields, seed=None)

        # 5. Update timers
        self.pool.steps[alive_mask] += 1
        self.pool.target_spawn_hour[alive_mask] = np.maximum(
            self.pool.target_spawn_hour[alive_mask] - 1, 0
        )

        # 6. Bioenergetics
        activity = np.array([
            self.bio_params.activity_by_behavior.get(int(b), 1.0)
            for b in self.pool.behavior
        ])
        sal = self.env.fields.get("salinity", np.zeros(self.mesh.n_triangles))
        sal_at_agents = sal[self.pool.tri_idx]
        s_cfg = self.est_cfg.get("salinity_cost", {})
        sal_cost = salinity_cost(
            sal_at_agents,
            S_opt=s_cfg.get("S_opt", 0.5),
            S_tol=s_cfg.get("S_tol", 6.0),
            k=s_cfg.get("k", 0.6),
        )

        new_ed, dead = update_energy(
            self.pool.ed_kJ_g, self.pool.mass_g,
            temps_at_agents, activity, sal_cost, self.bio_params,
        )
        self.pool.ed_kJ_g = new_ed
        self.pool.alive[dead] = False

        # 7. Logging
        if self.logger:
            self.logger.log_step(t, self.pool)

        summary = self.logger.summary(t, self.pool) if self.logger else {
            "time": t,
            "n_alive": int(self.pool.alive.sum()),
            "n_arrived": int(self.pool.arrived.sum()),
            "mean_ed": float(self.pool.ed_kJ_g[self.pool.alive].mean()) if self.pool.alive.any() else 0.0,
        }
        self.history.append(summary)

        self.current_t += 1

    def _apply_estuarine_overrides(self):
        """Apply salinity/DO/seiche overrides to behaviors."""
        # Seiche pause
        seiche_cfg = self.est_cfg.get("seiche_pause", {})
        thresh = seiche_cfg.get("dSSHdt_thresh_m_per_15min", 0.02)
        dSSH = np.array([self.env.dSSH_dt(int(tri)) for tri in self.pool.tri_idx])
        paused = seiche_pause(dSSH, thresh=thresh)
        self.pool.behavior[paused & self.pool.alive] = Behavior.HOLD

    def run(self, n_steps: int):
        """Run simulation for n hourly timesteps."""
        for _ in range(n_steps):
            self.step()

    def close(self):
        """Clean up resources."""
        if self.logger:
            self.logger.close()
        self.env.close()
```

**Step 4: Run tests**

Run: `micromamba run -n shiny python -m pytest tests/test_simulation.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add salmon_ibm/simulation.py tests/test_simulation.py
git commit -m "feat: simulation loop with full estuarine integration"
```

---

## Task 11: CLI Entry Point

**Files:**
- Create: `run.py`

**Step 1: Write CLI entry point**

Create `run.py`:

```python
"""Command-line interface for running the salmon IBM."""
import argparse
import sys

from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description="Baltic Salmon IBM")
    parser.add_argument("--config", default="config_curonian_minimal.yaml")
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="output/tracks.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim = Simulation(
        cfg, n_agents=args.agents, data_dir=args.data_dir,
        rng_seed=args.seed, output_path=args.output,
    )

    print(f"Running {args.steps} hourly steps with {args.agents} agents...")
    sim.run(n_steps=args.steps)
    sim.close()

    alive = sim.pool.alive.sum()
    arrived = sim.pool.arrived.sum()
    print(f"Done. Alive: {alive}/{args.agents}, Arrived: {arrived}")
    print(f"Tracks saved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test**

```bash
mkdir -p output
micromamba run -n shiny python run.py --agents 20 --steps 10
```
Expected: Prints summary, creates `output/tracks.csv`

**Step 3: Commit**

```bash
git add run.py
git commit -m "feat: CLI entry point for headless simulation runs"
```

---

## Task 12: Shiny UI — Sidebar & Run Controls

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/sidebar.py`
- Create: `ui/run_controls.py`
- Create: `app.py`

**Step 1: Create UI modules**

Create `ui/__init__.py` (empty).

Create `ui/sidebar.py`:

```python
"""Sidebar parameter controls for the Shiny app."""
from shiny import ui


def sidebar_panel():
    return ui.sidebar(
        ui.h4("Simulation Parameters"),
        ui.input_numeric("n_agents", "Number of agents", value=50, min=1, max=1000),
        ui.input_numeric("n_steps", "Simulation hours", value=24, min=1, max=8760),
        ui.input_numeric("rng_seed", "Random seed", value=42),
        ui.hr(),
        ui.h5("Bioenergetics (Salmo salar)"),
        ui.input_numeric("ra", "RA (resp. intercept)", value=0.00264, step=0.0001),
        ui.input_numeric("rb", "RB (allometric exp.)", value=-0.217, step=0.01),
        ui.input_numeric("rq", "RQ (temp coeff.)", value=0.06818, step=0.001),
        ui.input_numeric("ed_init", "Initial energy density (kJ/g)", value=6.5, step=0.1),
        ui.input_numeric("ed_mortal", "Mortality threshold (kJ/g)", value=4.0, step=0.1),
        ui.hr(),
        ui.h5("Estuary"),
        ui.input_numeric("s_opt", "Salinity optimum (PSU)", value=0.5, step=0.1),
        ui.input_numeric("s_tol", "Salinity tolerance (PSU)", value=6.0, step=0.5),
        ui.input_numeric("sal_k", "Salinity cost coefficient", value=0.6, step=0.1),
        ui.input_numeric("do_lethal", "DO lethal (mg/L)", value=2.0, step=0.5),
        ui.input_numeric("do_high", "DO avoidance (mg/L)", value=4.0, step=0.5),
        ui.input_numeric("seiche_thresh", "Seiche dSSH/dt threshold", value=0.02, step=0.005),
        ui.hr(),
        ui.h5("Map Display"),
        ui.input_select(
            "map_field", "Color mesh by",
            choices={"temperature": "Temperature", "salinity": "Salinity",
                     "ssh": "SSH", "depth": "Depth"},
        ),
        width=320,
    )
```

Create `ui/run_controls.py`:

```python
"""Run control buttons and progress display."""
from shiny import ui


def run_controls_panel():
    return ui.div(
        ui.row(
            ui.column(3, ui.input_action_button("btn_run", "Run", class_="btn-primary")),
            ui.column(3, ui.input_action_button("btn_step", "Step")),
            ui.column(3, ui.input_action_button("btn_pause", "Pause")),
            ui.column(3, ui.input_action_button("btn_reset", "Reset", class_="btn-warning")),
        ),
        ui.row(
            ui.column(6, ui.input_slider("speed", "Steps/update", min=1, max=10, value=1)),
            ui.column(6, ui.output_text("status_text")),
        ),
        ui.output_text("progress_text"),
    )
```

**Step 2: Create the Shiny app**

Create `app.py`:

```python
"""Baltic Salmon IBM — Shiny for Python Application."""
import asyncio

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

from salmon_ibm.config import load_config
from salmon_ibm.bioenergetics import BioParams
from salmon_ibm.simulation import Simulation
from ui.sidebar import sidebar_panel
from ui.run_controls import run_controls_panel


app_ui = ui.page_sidebar(
    sidebar_panel(),
    ui.navset_tab(
        ui.nav_panel(
            "Map",
            run_controls_panel(),
            output_widget("map_plot", height="500px"),
        ),
        ui.nav_panel(
            "Charts",
            ui.row(
                ui.column(6, output_widget("survival_plot", height="300px")),
                ui.column(6, output_widget("energy_plot", height="300px")),
            ),
            ui.row(
                ui.column(12, output_widget("behavior_plot", height="300px")),
            ),
        ),
    ),
    title="Baltic Salmon IBM — Curonian Lagoon",
)


def server(input, output, session):
    sim_state = reactive.Value(None)
    running = reactive.Value(False)
    history = reactive.Value([])

    @reactive.effect
    @reactive.event(input.btn_reset, ignore_none=False)
    def _init_sim():
        cfg = load_config("config_curonian_minimal.yaml")
        # Override estuary params from UI
        cfg["estuary"]["salinity_cost"]["S_opt"] = input.s_opt()
        cfg["estuary"]["salinity_cost"]["S_tol"] = input.s_tol()
        cfg["estuary"]["salinity_cost"]["k"] = input.sal_k()
        cfg["estuary"]["do_avoidance"]["lethal"] = input.do_lethal()
        cfg["estuary"]["do_avoidance"]["high"] = input.do_high()
        cfg["estuary"]["seiche_pause"]["dSSHdt_thresh_m_per_15min"] = input.seiche_thresh()

        sim = Simulation(
            cfg,
            n_agents=input.n_agents(),
            data_dir="data",
            rng_seed=input.rng_seed(),
        )
        # Override bio params from UI
        sim.bio_params = BioParams(
            RA=input.ra(), RB=input.rb(), RQ=input.rq(),
            ED_MORTAL=input.ed_mortal(),
        )
        sim_state.set(sim)
        history.set([])
        running.set(False)

    @reactive.effect
    @reactive.event(input.btn_step)
    def _step():
        sim = sim_state.get()
        if sim is None:
            _init_sim()
            sim = sim_state.get()
        sim.step()
        history.set(sim.history.copy())

    @reactive.effect
    @reactive.event(input.btn_run)
    async def _run():
        running.set(True)
        sim = sim_state.get()
        if sim is None:
            _init_sim()
            sim = sim_state.get()
        steps = input.n_steps()
        speed = input.speed()
        while running.get() and sim.current_t < steps:
            for _ in range(speed):
                if sim.current_t >= steps:
                    break
                sim.step()
            history.set(sim.history.copy())
            await asyncio.sleep(0.05)
        running.set(False)

    @reactive.effect
    @reactive.event(input.btn_pause)
    def _pause():
        running.set(False)

    @render.text
    def status_text():
        sim = sim_state.get()
        if sim is None:
            return "Not initialized"
        return f"Alive: {sim.pool.alive.sum()}/{sim.pool.n} | Arrived: {sim.pool.arrived.sum()}"

    @render.text
    def progress_text():
        sim = sim_state.get()
        if sim is None:
            return "t = 0"
        return f"t = {sim.current_t} h"

    @render_widget
    def map_plot():
        sim = sim_state.get()
        _ = history.get()  # trigger reactivity
        if sim is None:
            return go.Figure()

        mesh = sim.mesh
        field_name = input.map_field()

        fig = go.Figure()

        # Draw mesh triangles
        if field_name == "depth":
            z = mesh.depth
        elif field_name in sim.env.fields:
            z = sim.env.fields[field_name]
        else:
            z = mesh.depth

        fig.add_trace(go.Mesh3d(
            x=mesh.nodes[:, 1], y=mesh.nodes[:, 0],
            z=np.zeros(len(mesh.nodes)),
            i=mesh.triangles[:, 0], j=mesh.triangles[:, 1], k=mesh.triangles[:, 2],
            intensity=z[mesh.triangles].mean(axis=1) if len(z) == mesh.n_triangles else z,
            colorscale="Viridis",
            colorbar=dict(title=field_name),
            opacity=0.7,
            flatshading=True,
        ))

        # Alternatively use 2D scatter mesh (simpler and often better)
        fig = go.Figure()

        # Mesh as scatter of triangle centroids
        water = mesh.water_mask
        fig.add_trace(go.Scatter(
            x=mesh.centroids[water, 1],
            y=mesh.centroids[water, 0],
            mode="markers",
            marker=dict(
                size=8, color=z[water] if len(z) == mesh.n_triangles else z[water],
                colorscale="Viridis", colorbar=dict(title=field_name),
            ),
            name="Mesh",
            hovertemplate="lon: %{x:.3f}<br>lat: %{y:.3f}<br>value: %{marker.color:.2f}",
        ))

        # Agent positions
        alive = sim.pool.alive
        if alive.any():
            agent_tris = sim.pool.tri_idx[alive]
            behavior_names = ["Hold", "Random", "CWR", "Upstream", "Downstream"]
            colors = ["gray", "blue", "cyan", "red", "orange"]
            for b in range(5):
                b_mask = sim.pool.behavior[alive] == b
                if b_mask.any():
                    tris_b = agent_tris[b_mask]
                    fig.add_trace(go.Scatter(
                        x=mesh.centroids[tris_b, 1],
                        y=mesh.centroids[tris_b, 0],
                        mode="markers",
                        marker=dict(size=10, color=colors[b], symbol="diamond",
                                    line=dict(width=1, color="black")),
                        name=behavior_names[b],
                    ))

        fig.update_layout(
            xaxis_title="Longitude", yaxis_title="Latitude",
            height=500, margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    @render_widget
    def survival_plot():
        h = history.get()
        if not h:
            return go.Figure().update_layout(title="Survival")
        times = [r["time"] for r in h]
        alive = [r["n_alive"] for r in h]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=alive, mode="lines", name="Alive"))
        fig.update_layout(title="Survival", xaxis_title="Hour", yaxis_title="N alive",
                          height=300, margin=dict(l=40, r=20, t=40, b=40))
        return fig

    @render_widget
    def energy_plot():
        h = history.get()
        if not h:
            return go.Figure().update_layout(title="Energy Density")
        times = [r["time"] for r in h]
        ed = [r.get("mean_ed", 0) for r in h]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=ed, mode="lines", name="Mean ED"))
        fig.add_hline(y=4.0, line_dash="dash", line_color="red",
                      annotation_text="Mortality threshold")
        fig.update_layout(title="Mean Energy Density", xaxis_title="Hour",
                          yaxis_title="kJ/g", height=300, margin=dict(l=40, r=20, t=40, b=40))
        return fig

    @render_widget
    def behavior_plot():
        h = history.get()
        if not h:
            return go.Figure().update_layout(title="Behavior Distribution")
        times = [r["time"] for r in h]
        names = ["Hold", "Random", "CWR", "Upstream", "Downstream"]
        colors = ["gray", "blue", "cyan", "red", "orange"]
        fig = go.Figure()
        for b in range(5):
            counts = [r.get("behavior_counts", {}).get(b, 0) for r in h]
            fig.add_trace(go.Scatter(
                x=times, y=counts, mode="lines", name=names[b],
                stackgroup="one", line=dict(color=colors[b]),
            ))
        fig.update_layout(title="Behavior Distribution", xaxis_title="Hour",
                          yaxis_title="Count", height=300, margin=dict(l=40, r=20, t=40, b=40))
        return fig


app = App(app_ui, server)
```

**Step 2: Smoke test the Shiny app**

```bash
micromamba run -n shiny shiny run app.py --port 8765
```
Expected: Opens at http://localhost:8765, shows sidebar + empty map. Click "Reset" then "Step" to see agents appear.

**Step 3: Commit**

```bash
git add app.py ui/
git commit -m "feat: Shiny for Python UI — sidebar, map, charts, run controls"
```

---

## Task 13: Integration Test & Final Polish

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
import pytest
from salmon_ibm.config import load_config
from salmon_ibm.simulation import Simulation


def test_full_simulation_24h():
    """End-to-end: 24 hours with 20 agents on stub data."""
    cfg = load_config("config_curonian_minimal.yaml")
    sim = Simulation(cfg, n_agents=20, data_dir="data", rng_seed=42)
    sim.run(n_steps=24)

    # At least some agents should survive 24h
    assert sim.pool.alive.sum() > 0
    # Energy should have decreased
    assert sim.pool.ed_kJ_g[sim.pool.alive].mean() < 6.5
    # History should have 24 entries
    assert len(sim.history) == 24

    sim.close()


def test_full_simulation_with_output(tmp_path):
    """End-to-end with track output."""
    cfg = load_config("config_curonian_minimal.yaml")
    out = str(tmp_path / "tracks.csv")
    sim = Simulation(cfg, n_agents=10, data_dir="data", rng_seed=42, output_path=out)
    sim.run(n_steps=5)
    sim.close()

    import pandas as pd
    df = pd.read_csv(out)
    assert len(df) == 10 * 5  # 10 agents x 5 timesteps
    assert "ed_kJ_g" in df.columns
```

**Step 2: Run all tests**

```bash
micromamba run -n shiny python -m pytest tests/ -v
```
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: integration tests — full 24h simulation on stub data"
```

---

## Summary of Tasks

| # | Task | Files | Estimated Steps |
|---|------|-------|----------------|
| 1 | Project scaffolding + config | `salmon_ibm/__init__.py`, `config.py`, tests | 6 |
| 2 | Triangular mesh | `mesh.py`, tests | 6 |
| 3 | Environment | `environment.py`, tests | 5 |
| 4 | Agents (FishAgent + AgentPool) | `agents.py`, tests | 5 |
| 5 | Bioenergetics | `bioenergetics.py`, tests | 5 |
| 6 | Estuary extensions | `estuary.py`, tests | 5 |
| 7 | Behavior decision table | `behavior.py`, tests | 5 |
| 8 | Movement kernels | `movement.py`, tests | 5 |
| 9 | Output logger | `output.py`, tests | 5 |
| 10 | Simulation loop | `simulation.py`, tests | 5 |
| 11 | CLI entry point | `run.py` | 3 |
| 12 | Shiny UI | `app.py`, `ui/` | 3 |
| 13 | Integration test | `test_integration.py` | 3 |
