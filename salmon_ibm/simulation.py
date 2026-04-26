"""Main simulation loop."""

from __future__ import annotations

import numpy as np
from typing import TypedDict


class Landscape(TypedDict, total=False):
    """Typed dict passed to every event. All keys are optional (total=False)."""

    mesh: object  # TriMesh | HexMesh
    fields: dict[str, np.ndarray]
    rng: np.random.Generator
    activity_lut: np.ndarray
    est_cfg: dict
    barrier_arrays: tuple | None
    genome: object | None  # GenomeManager | None
    multi_pop_mgr: object | None  # MultiPopulationManager | None
    network: object | None  # StreamNetwork | None
    step_alive_mask: np.ndarray
    spatial_data: dict[str, np.ndarray]
    global_variables: dict[str, float]
    census_records: list
    summary_reports: list
    log_dir: str


from salmon_ibm.agents import AgentPool, Behavior
from salmon_ibm.behavior import pick_behaviors, apply_overrides
from salmon_ibm.population import Population
from salmon_ibm.bioenergetics import update_energy
from salmon_ibm.config import (
    bio_params_from_config,
    load_bio_params_from_config,
    behavior_params_from_config,
    genome_from_config,
    network_from_config,
)
from salmon_ibm.environment import Environment
from salmon_ibm.estuary import (
    salinity_cost,
    do_override,
    seiche_pause,
    DO_ESCAPE,
    DO_LETHAL,
)
from salmon_ibm.events import EventSequencer
from salmon_ibm.events_builtin import MovementEvent, CustomEvent
from salmon_ibm.mesh import TriMesh
from salmon_ibm.output import OutputLogger


class Simulation:
    def __init__(
        self, config, n_agents=100, data_dir="data", rng_seed=None, output_path=None
    ):
        self.config = config
        self.rng_seed = rng_seed
        self.current_t = 0

        # Three mesh backends: legacy ``grid.type=hexsim``, default
        # NetCDF TriMesh, and the new top-level ``mesh_backend=h3``.
        mesh_backend = config.get("mesh_backend", "")
        grid_type = config.get("grid", {}).get("type", "netcdf")

        if mesh_backend == "h3":
            from salmon_ibm.h3mesh import H3Mesh
            from salmon_ibm.h3_env import H3Environment
            import h3 as _h3
            import xarray as _xr

            landscape_path = config["h3_landscape_nc"]
            ds = _xr.open_dataset(landscape_path, engine="h5netcdf")
            cells = [_h3.int_to_str(int(i)) for i in ds["h3_id"].values]
            self.mesh = H3Mesh.from_h3_cells(
                cells,
                depth=ds["depth"].values,
                water_mask=ds["water_mask"].values.astype(bool),
            )
            # Carry per-cell reach IDs from the NC if present (only
            # landscapes built with the inSTREAM-polygon mask have
            # this; the loader leaves the defaults if missing).
            if "reach_id" in ds.variables:
                self.mesh.reach_id = ds["reach_id"].values.astype(np.int8)
                names_attr = ds.attrs.get("reach_names", "")
                if names_attr:
                    self.mesh.reach_names = names_attr.split(",")
            self.env = H3Environment.from_netcdf(landscape_path, self.mesh)
        elif mesh_backend == "h3_multires":
            from salmon_ibm.h3_multires import H3MultiResMesh
            from salmon_ibm.h3_env import H3Environment
            import h3 as _h3
            import xarray as _xr

            landscape_path = config["h3_multires_landscape_nc"]
            ds = _xr.open_dataset(landscape_path, engine="h5netcdf")

            # reach metadata MUST be loaded as a pair — without `reach_id`,
            # `reach_names` is meaningless (every cell would report "Land").
            # Loading them independently was rejected in plan review pass 6.
            if "reach_id" in ds:
                reach_id_arr = ds["reach_id"].values.astype(np.int8)
                names_attr = ds.attrs.get("reach_names", "")
                reach_names = names_attr.split(",") if names_attr else None
            else:
                reach_id_arr = None
                reach_names = None

            # Build areas from the H3 cells — cell_area is per-cell at res-aware
            # scale, so a multi-res mesh has multi-scale areas natively.  The
            # NC stores h3_id in compact-index order; the H3MultiResMesh
            # constructor receives these arrays in the same order, so areas
            # below align cell-for-cell with everything else.  We can't call
            # `from_h3_cells` (which would also recompute areas) because that
            # would discard the pre-built CSR neighbour table in the NC.
            cells_str = [_h3.int_to_str(int(i)) for i in ds["h3_id"].values]
            areas = np.array(
                [_h3.cell_area(c, unit="m^2") for c in cells_str],
                dtype=np.float32,
            )

            centroids = np.column_stack(
                [ds["lat"].values, ds["lon"].values],
            )

            self.mesh = H3MultiResMesh(
                h3_ids=ds["h3_id"].values.astype(np.uint64),
                resolutions=ds["resolution"].values.astype(np.int8),
                centroids=centroids,
                nbr_starts=ds["nbr_starts"].values.astype(np.int32),
                nbr_idx=ds["nbr_idx"].values.astype(np.int32),
                water_mask=ds["water_mask"].values.astype(bool),
                depth=ds["depth"].values.astype(np.float32),
                areas=areas,
                reach_id=reach_id_arr,
                reach_names=reach_names,
            )
            ds.close()
            self.env = H3Environment.from_netcdf(landscape_path, self.mesh)
        elif grid_type == "hexsim":
            from salmon_ibm.hexsim import HexMesh
            from salmon_ibm.hexsim_env import HexSimEnvironment

            hs = config["hexsim"]
            self.mesh = HexMesh.from_hexsim(
                hs["workspace"],
                species=hs.get("species", "chinook"),
                extent_layer=hs.get("extent_layer"),
                depth_layer=hs.get("depth_layer"),
            )
            self.env = HexSimEnvironment(
                hs["workspace"],
                self.mesh,
                temperature_csv=hs.get("temperature_csv", "River Temperature.csv"),
            )
        else:
            grid_file = f"{data_dir}/{config['grid']['file']}"
            self.mesh = TriMesh.from_netcdf(grid_file)
            self.env = Environment(config, self.mesh, data_dir=data_dir)

        # Agent placement.  The legacy default (every backend up to now)
        # is uniform-over-water-cells.  H3 landscapes can opt into the
        # same explicit ``uniform_random_water`` strategy via the YAML
        # so the contract is documented; in either case we record an
        # initial_cells snapshot for movement-sanity diagnostics.
        base_rng = np.random.default_rng(rng_seed)
        rng = np.random.default_rng(base_rng.integers(2**63))
        self._rng = np.random.default_rng(base_rng.integers(2**63))

        initial_state = config.get("initial_state", {}) or {}
        strategy = initial_state.get("initial_cell_strategy", "uniform_random_water")
        if strategy == "uniform_random_water":
            water_ids = np.where(self.mesh.water_mask)[0]
            if len(water_ids) == 0:
                raise RuntimeError(
                    "no water cells in mesh — cannot place agents"
                )
            start_tris = rng.choice(water_ids, size=n_agents)
        else:
            raise ValueError(
                f"unknown initial_cell_strategy: {strategy!r}"
            )

        self.pool = AgentPool(n=n_agents, start_tri=start_tris, rng_seed=rng_seed)
        # Snapshot for movement diagnostics — see test_at_least_one_agent_moved
        # in tests/test_nemunas_h3_integration.py.
        self.initial_cells = self.pool.tri_idx.copy()
        self.population = Population(name="salmon", pool=self.pool)

        # --- Optional Phase 2-3 components ---
        # Barriers — three loader paths depending on mesh backend:
        #   * grid.type == "hexsim": HexSim .hbf file
        #   * mesh_backend == "h3":  CSV via from_csv_h3
        #   * neither:               no barriers
        barrier_cfg = self.config.get("barriers")
        h3_barriers_csv = self.config.get("barriers_csv")
        self._barrier_arrays = None
        if barrier_cfg and grid_type == "hexsim":
            from salmon_ibm.barriers import BarrierMap

            hbf_path = barrier_cfg.get("file")
            if hbf_path:
                bmap = BarrierMap.from_hbf_hexsim(hbf_path, self.mesh)
                if bmap.has_barriers():
                    self._barrier_arrays = bmap.to_arrays(self.mesh)
        elif h3_barriers_csv and mesh_backend == "h3":
            from salmon_ibm.barriers import BarrierMap

            bmap = BarrierMap.from_csv_h3(h3_barriers_csv, self.mesh)
            if bmap.has_barriers():
                self._barrier_arrays = bmap.to_arrays(self.mesh)

        # Genome
        self._genome = genome_from_config(config, n_agents)
        if self._genome is not None:
            self.population.genome = self._genome

        # Network
        self._network = network_from_config(config)

        # Multi-population manager
        from salmon_ibm.interactions import MultiPopulationManager

        self._multi_pop_mgr = MultiPopulationManager()
        self._multi_pop_mgr.register(self.population)

        self.beh_params = behavior_params_from_config(config)
        # Routes to BalticBioParams if config has `species_config:` key,
        # otherwise returns classic BioParams.
        self.bio_params = load_bio_params_from_config(config)
        self._activity_lut = self._build_activity_lut()
        self.est_cfg = config.get("estuary", {})
        self._skip_estuarine_overrides = self._detect_estuarine_noop()

        # Scratch buffers for per-step operations (avoid alloc in hot loops).
        # _zero_salinity is used as the "salinity" field when the env doesn't
        # provide one; _event_bioenergetics previously allocated a fresh
        # np.zeros(n_triangles) every step via fields.get(..., default).
        self._zero_salinity = np.zeros(self.mesh.n_triangles, dtype=np.float64)

        # Per-cell hourly survival probability for the fish-predation event.
        # Config supplies DAILY rates per reach; we convert to hourly so
        # applying once per simulation step (= 1 h) reproduces the daily
        # target.  Backends without `cells_in_reach` (TriMesh, HexMesh)
        # silently fall back to the global default — they don't have
        # inSTREAM reach polygons.
        mort_cfg = config.get("mortality_per_reach") or {}
        if mort_cfg:
            default_daily = float(mort_cfg.get("default", 0.985))
            self._cell_survival_hourly = np.full(
                self.mesh.n_triangles,
                default_daily ** (1.0 / 24.0),
                dtype=np.float32,
            )
            if hasattr(self.mesh, "cells_in_reach"):
                for name, rate_daily in mort_cfg.items():
                    if name == "default":
                        continue
                    cells = self.mesh.cells_in_reach(name)
                    if len(cells):
                        self._cell_survival_hourly[cells] = (
                            float(rate_daily) ** (1.0 / 24.0)
                        )
        else:
            self._cell_survival_hourly = None  # event will no-op

        self.logger = None
        if output_path:
            self.logger = OutputLogger(output_path, self.mesh.centroids)

        self.history = []
        self._sequencer = EventSequencer(self._build_events())

    # ------------------------------------------------------------------
    # Event sequencer setup
    # ------------------------------------------------------------------

    def _build_events(self):
        """Build event sequence from config or use default salmon sequence."""
        event_defs = self.config.get("events")
        if event_defs is not None:
            from salmon_ibm.events import load_events_from_config

            return load_events_from_config(event_defs, self._build_callback_registry())

        # Default: hardcoded salmon migration sequence
        return [
            CustomEvent(name="push_temperature", callback=self._event_push_temperature),
            CustomEvent(
                name="behavior_selection", callback=self._event_behavior_selection
            ),
            CustomEvent(
                name="estuarine_overrides", callback=self._event_estuarine_overrides
            ),
            CustomEvent(
                name="update_cwr_counters", callback=self._event_update_cwr_counters
            ),
            MovementEvent(
                name="movement",
                n_micro_steps=3,
                cwr_threshold=self.beh_params.temp_bins[0],
            ),
            # Fish predation fires AFTER movement so agents are killed
            # at their final cell of this step (not their starting cell).
            # No-ops on backends without reach_id (TriMesh, HexMesh).
            CustomEvent(
                name="fish_predation", callback=self._event_fish_predation
            ),
            CustomEvent(name="update_timers", callback=self._event_update_timers),
            CustomEvent(name="bioenergetics", callback=self._event_bioenergetics),
            CustomEvent(name="logging", callback=self._event_logging),
        ]

    def _build_callback_registry(self):
        """Map custom event names to their callback methods."""
        return {
            "push_temperature": self._event_push_temperature,
            "behavior_selection": self._event_behavior_selection,
            "estuarine_overrides": self._event_estuarine_overrides,
            "update_cwr_counters": self._event_update_cwr_counters,
            "fish_predation": self._event_fish_predation,
            "update_timers": self._event_update_timers,
            "bioenergetics": self._event_bioenergetics,
            "logging": self._event_logging,
        }

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def _event_push_temperature(self, population, landscape, t, mask):
        temps = landscape["fields"]["temperature"][population.tri_idx]
        alive_mask = population.alive & ~population.arrived
        population.push_temperature(temps, alive_mask=alive_mask)

    def _event_behavior_selection(self, population, landscape, t, mask):
        step_mask = population.alive & ~population.arrived
        t3h = population.t3h_mean()
        population.behavior[step_mask] = pick_behaviors(
            t3h[step_mask],
            population.target_spawn_hour[step_mask],
            self.beh_params,
            seed=int(self._rng.integers(2**31)),
        )
        overridden = apply_overrides(population, self.beh_params)
        alive_mask = population.alive & ~population.arrived
        population.behavior[alive_mask] = overridden[alive_mask]

    def _event_estuarine_overrides(self, population, landscape, t, mask):
        if self._skip_estuarine_overrides:
            return
        self._apply_estuarine_overrides()

    def _event_update_cwr_counters(self, population, landscape, t, mask):
        self._update_cwr_counters()

    def _event_fish_predation(self, population, landscape, t, mask):
        """Per-cell Bernoulli mortality from fish predation.

        Reads the hourly survival rate at each agent's current cell
        (built from `mortality_per_reach` config at sim init).  Each
        alive-and-not-arrived agent draws a Uniform(0, 1) — if it
        exceeds the cell's survival probability, the agent dies.

        No-ops when `_cell_survival_hourly` is None (config didn't
        specify `mortality_per_reach`) or when no agents are alive.
        Backends without inSTREAM reach polygons (TriMesh, HexMesh)
        get the global `default` rate at every cell — no error.
        """
        if self._cell_survival_hourly is None:
            return
        active = population.alive & ~getattr(
            population, "arrived", np.zeros(population.n, dtype=bool)
        )
        if not active.any():
            return
        p_surv = self._cell_survival_hourly[population.tri_idx[active]]
        # Use a fresh draw per call so the predation RNG is decoupled
        # from movement / behaviour selection RNG streams.
        rng = np.random.default_rng(self._rng.integers(2**31))
        died = rng.random(len(p_surv)) > p_surv
        if died.any():
            active_indices = np.where(active)[0]
            population.alive[active_indices[died]] = False

    def _event_update_timers(self, population, landscape, t, mask):
        # Recompute alive mask — defensive against custom event orderings
        # where mortality events may run before this callback.
        step_mask = population.alive & ~population.arrived
        population.steps[step_mask] += 1
        population.target_spawn_hour[step_mask] = np.maximum(
            population.target_spawn_hour[step_mask] - 1, 0
        )

    def _event_bioenergetics(self, population, landscape, t, mask):
        fields = landscape["fields"]
        temps_at_agents = fields["temperature"][population.tri_idx]
        activity = np.take(self._activity_lut, population.behavior, mode="clip")
        sal = fields.get("salinity", self._zero_salinity)
        sal_at_agents = sal[population.tri_idx]
        s_cfg = self.est_cfg.get("salinity_cost", {})
        sal_cost = salinity_cost(
            sal_at_agents,
            S_opt=s_cfg.get("S_opt", 0.5),
            S_tol=s_cfg.get("S_tol", 6.0),
            k=s_cfg.get("k", 0.6),
        )
        alive = population.alive & ~population.arrived
        if alive.any():
            new_ed, dead, new_mass = update_energy(
                population.ed_kJ_g[alive],
                population.mass_g[alive],
                temps_at_agents[alive],
                activity[alive],
                sal_cost[alive],
                self.bio_params,
            )
            population.ed_kJ_g[alive] = new_ed
            population.mass_g[alive] = new_mass
            dead_indices = np.where(alive)[0][dead]
            population.alive[dead_indices] = False
        # Recompute alive mask after starvation kills to avoid double-counting.
        # Use T_ACUTE_LETHAL when available (BalticBioParams) — T_MAX on Baltic
        # returns T_AVOID=20°C which is a behavioral threshold, not acute lethal.
        # For Chinook BioParams, T_MAX=26°C is already the acute-lethal threshold.
        lethal_T = getattr(
            self.bio_params, "T_ACUTE_LETHAL", self.bio_params.T_MAX
        )
        thermal_kill = (population.alive & ~population.arrived) & (
            temps_at_agents >= lethal_T
        )
        population.alive[thermal_kill] = False

    def _event_logging(self, population, landscape, t, mask):
        if self.logger:
            self.logger.log_step(t, population)
        summary = {
            "time": t,
            "n_alive": int(population.alive.sum()),
            "n_arrived": int(population.arrived.sum()),
            "mean_ed": float(population.ed_kJ_g[population.alive].mean())
            if population.alive.any()
            else 0.0,
            "mean_mass": float(population.mass_g[population.alive].mean())
            if population.alive.any()
            else 0.0,
            "behavior_counts": {
                int(b): int(c)
                for b, c in enumerate(
                    np.bincount(population.behavior[population.alive], minlength=5)
                )
            },
        }
        self.history.append(summary)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self):
        t = self.current_t
        self.env.advance(t)
        landscape: Landscape = {
            "mesh": self.mesh,
            "fields": self.env.fields,
            "rng": self._rng,
            "activity_lut": self._activity_lut,
            "est_cfg": self.est_cfg,
            "barrier_arrays": self._barrier_arrays,
            "genome": self._genome,
            "multi_pop_mgr": self._multi_pop_mgr,
            "network": self._network,
        }
        self._sequencer.step(self.population, landscape, t)
        self.current_t += 1

    def _build_activity_lut(self):
        """Build vectorized activity multiplier lookup table."""
        max_beh = max(self.bio_params.activity_by_behavior.keys())
        lut = np.ones(max_beh + 1)
        for k, v in self.bio_params.activity_by_behavior.items():
            lut[k] = v
        return lut

    def _update_cwr_counters(self):
        """Update CWR tracking counters based on current behavior state."""
        in_cwr = self.pool.behavior == Behavior.TO_CWR
        not_in_cwr = ~in_cwr

        # Fish in CWR: increment cwr_hours, reset hours_since_cwr
        self.pool.cwr_hours[in_cwr] += 1
        self.pool.hours_since_cwr[in_cwr] = 0

        # Fish not in CWR: reset cwr_hours if they were in CWR, increment hours_since_cwr
        was_in_cwr = not_in_cwr & (self.pool.cwr_hours > 0)
        self.pool.cwr_hours[was_in_cwr] = 0
        self.pool.hours_since_cwr[not_in_cwr] += 1

    def _detect_estuarine_noop(self) -> bool:
        """Return True if estuarine overrides are effectively disabled."""
        est = self.est_cfg
        if not est:
            return True
        seiche = est.get("seiche_pause", {})
        thresh = seiche.get("dSSHdt_thresh_m_per_hour")
        if thresh is None:
            thresh = seiche.get("dSSHdt_thresh_m_per_15min", 0.02) * 4.0
        seiche_noop = thresh >= 100.0
        do_cfg = est.get("do_avoidance", {})
        do_noop = do_cfg.get("lethal", 0.0) <= 0 and do_cfg.get("high", 0.0) <= 0
        return seiche_noop and do_noop

    def _apply_estuarine_overrides(self):
        alive = self.pool.alive

        # Seiche pause
        seiche_cfg = self.est_cfg.get("seiche_pause", {})
        thresh = seiche_cfg.get("dSSHdt_thresh_m_per_hour")
        if thresh is None:
            # Convert per-15-min threshold to per-hour (dSSH_dt_array returns m/hour)
            thresh = seiche_cfg.get("dSSHdt_thresh_m_per_15min", 0.02) * 4.0
        dSSH_all = self.env.dSSH_dt_array()
        dSSH = (
            dSSH_all[self.pool.tri_idx] if len(dSSH_all) > 0 else np.zeros(self.pool.n)
        )
        paused = seiche_pause(dSSH, thresh=thresh)
        self.pool.behavior[paused & alive] = Behavior.HOLD

        # Dissolved oxygen avoidance
        do_cfg = self.est_cfg.get("do_avoidance", {})
        do_lethal = do_cfg.get("lethal", 0.0)
        do_high = do_cfg.get("high", 0.0)
        do_field = self.env.fields.get("do")
        if do_field is not None and (do_lethal > 0 or do_high > 0):
            do_at_agents = do_field[self.pool.tri_idx]
            do_state = do_override(do_at_agents, lethal=do_lethal, high=do_high)
            self.pool.behavior[(do_state == DO_ESCAPE) & alive] = Behavior.DOWNSTREAM
            self.pool.alive[(do_state == DO_LETHAL) & alive] = False

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()

    def close(self):
        if self.logger:
            self.logger.close()
        self.env.close()
