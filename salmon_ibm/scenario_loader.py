"""ScenarioLoader: load a HexSim workspace + XML scenario → runnable simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from salmon_ibm.xml_parser import load_scenario_xml
from salmon_ibm.hexsim import HexMesh
from salmon_ibm.hexsim_env import HexSimEnvironment
from salmon_ibm.agents import AgentPool
from salmon_ibm.population import Population
from salmon_ibm.accumulators import AccumulatorManager, AccumulatorDef
from salmon_ibm.traits import TraitManager, TraitDefinition, TraitType
from salmon_ibm.events import (
    Event,
    EventGroup,
    EveryStep,
    Once,
    MultiPopEventSequencer,
)
from salmon_ibm.interactions import MultiPopulationManager


class HexSimSimulation:
    """Multi-population simulation driven by event sequencer.

    Coexists with the existing Simulation class (salmon migration specific).
    This class is XML-driven and general-purpose.
    """

    def __init__(self, populations, sequencer, environment, landscape, n_timesteps):
        self.populations = populations  # MultiPopulationManager
        self.sequencer = sequencer  # MultiPopEventSequencer
        self.environment = environment  # HexSimEnvironment
        self.landscape = landscape  # dict with mesh, spatial_data, rng, etc.
        self.n_timesteps = n_timesteps
        self.current_t = 0
        self.history = []

    def step(self):
        t = self.current_t
        self.environment.advance(t)
        self.landscape["fields"] = self.environment.fields
        self.sequencer.step(self.landscape, t)
        # Record summary
        total_alive = sum(p.n_alive for p in self.populations.populations.values())
        self.history.append({"time": t, "n_alive": total_alive})
        self.current_t += 1

    def run(self, n_steps=None):
        if n_steps is None:
            n_steps = self.n_timesteps
        for _ in range(n_steps):
            self.step()


class ScenarioLoader:
    """Load a HexSim workspace + XML scenario → HexSimSimulation."""

    def load(
        self, workspace_dir: str, scenario_xml: str, rng_seed: int | None = None
    ) -> HexSimSimulation:
        ws_dir = Path(workspace_dir)
        config = load_scenario_xml(scenario_xml)

        # 1. Load mesh
        mesh = HexMesh.from_hexsim(str(ws_dir))

        # 2. Load environment
        env = HexSimEnvironment(str(ws_dir), mesh)

        # 3. Build spatial data registry from workspace hex-maps
        spatial_data = self._build_spatial_registry(mesh, config)

        # 4. Create populations (with derived RNG for reproducibility)
        base_rng = np.random.default_rng(rng_seed)
        multi_pop = MultiPopulationManager()
        for pop_def in config["populations"]:
            pop_rng = np.random.default_rng(base_rng.integers(2**63))
            pop = self._create_population(pop_def, mesh, pop_rng)
            multi_pop.register(pop)

        # 5. Build event tree
        events = self._build_events(
            config["events"], config.get("global_variables", {})
        )

        # 5b. Load lookup tables from workspace CSVs
        self._load_lookup_tables(events, str(ws_dir))

        # 6. Build sequencer
        sequencer = MultiPopEventSequencer(events, multi_pop)

        # 7. Build landscape dict (reuse base_rng so the full sequence is reproducible)
        landscape = {
            "mesh": mesh,
            "spatial_data": spatial_data,
            "global_variables": config.get("global_variables", {}),
            "rng": base_rng,
        }

        return HexSimSimulation(
            populations=multi_pop,
            sequencer=sequencer,
            environment=env,
            landscape=landscape,
            n_timesteps=config["simulation"].get("n_timesteps", 100),
        )

    def _build_spatial_registry(self, mesh, config) -> dict[str, np.ndarray]:
        """Load all spatial data layers from workspace into a name→array dict."""
        registry = {}
        ws = mesh._workspace
        if ws is None:
            return registry
        for name in config.get("spatial_data_series", {}):
            hm = ws.hexmaps.get(name)
            if hm is not None:
                # Compact to water-only cells
                registry[name] = hm.values[mesh._water_full_idx].astype(np.float64)
        return registry

    def _create_population(self, pop_def: dict, mesh: HexMesh, rng=None) -> Population:
        """Create a Population from parsed XML definition."""
        if rng is None:
            rng = np.random.default_rng()
        n = pop_def.get("initial_size", 0)
        # For populations initialized later by events (e.g., introductionEvent),
        # start with n=0 and a minimal pool
        if n == 0:
            n = 1  # need at least 1 for array allocation; compact later
            pool = AgentPool(n=1, start_tri=np.array([0]))
            pool.alive[0] = False  # placeholder, not a real agent
        else:
            water_ids = np.where(mesh.water_mask)[0]
            start_tris = (
                rng.choice(water_ids, size=n, replace=True)
                if len(water_ids) > 0
                else np.zeros(n, dtype=int)
            )
            pool = AgentPool(n=n, start_tri=start_tris)

        pop = Population(name=pop_def["name"], pool=pool)

        # Create accumulators
        acc_defs = []
        for acc in pop_def.get("accumulators", []):
            # HexSim convention: BOTH being 0 (or absent) means unbounded.
            # If either is non-zero, both are explicit bounds.
            min_val_raw = acc.get("min_val")
            max_val_raw = acc.get("max_val")
            both_zero = (min_val_raw in (0, None)) and (max_val_raw in (0, None))
            acc_defs.append(
                AccumulatorDef(
                    name=acc["name"],
                    min_val=None
                    if both_zero
                    else (min_val_raw if min_val_raw is not None else 0),
                    max_val=None
                    if both_zero
                    else (max_val_raw if max_val_raw is not None else 0),
                )
            )
        if acc_defs:
            pop.accumulator_mgr = AccumulatorManager(pop.pool.n, acc_defs)

        # Create traits
        trait_defs = []
        for trait in pop_def.get("traits", []):
            if trait["type"] == "probabilistic":
                categories = [c["name"] for c in trait["categories"]]
                trait_defs.append(
                    TraitDefinition(
                        name=trait["name"],
                        trait_type=TraitType.PROBABILISTIC,
                        categories=categories,
                    )
                )
            elif trait["type"] == "accumulated":
                categories = [c["name"] for c in trait["categories"]]
                # Strip leading -inf from thresholds (np.digitize handles it implicitly)
                thresholds = np.array(
                    [
                        c["threshold"]
                        for c in trait["categories"]
                        if c["threshold"] != float("-inf")
                    ]
                )
                trait_defs.append(
                    TraitDefinition(
                        name=trait["name"],
                        trait_type=TraitType.ACCUMULATED,
                        categories=categories,
                        accumulator_name=trait.get("accumulator"),
                        thresholds=thresholds,
                    )
                )
        if trait_defs:
            pop.trait_mgr = TraitManager(pop.pool.n, trait_defs)

        return pop

    def _build_events(
        self, event_defs: list[dict], global_variables: dict
    ) -> list[Event]:
        """Convert parsed XML event dicts into Event objects."""
        # Import all event modules to populate EVENT_REGISTRY
        import salmon_ibm.events_builtin  # noqa: F401
        import salmon_ibm.events_phase3  # noqa: F401
        import salmon_ibm.events_hexsim  # noqa: F401

        events = []
        for edef in event_defs:
            evt = self._build_single_event(edef, global_variables)
            if evt is not None:
                events.append(evt)
        return events

    def _build_single_event(
        self, edef: dict, global_variables: dict | None = None
    ) -> Event | None:
        """Recursively build an Event from a parsed dict."""
        etype = edef.get("type", "")
        name = edef.get("name", etype)
        enabled = edef.get("enabled", True)
        population_name = edef.get("population")

        # Determine trigger
        timestep = edef.get("timestep")
        if timestep is not None:
            trigger = Once(at=timestep - 1)  # HexSim is 1-indexed
        else:
            trigger = EveryStep()

        if etype == "event_group":
            sub_events = []
            for sub_def in edef.get("sub_events", []):
                sub_evt = self._build_single_event(sub_def, global_variables)
                if sub_evt is not None:
                    sub_events.append(sub_evt)
            group = EventGroup(
                name=name,
                trigger=trigger,
                population_name=population_name,
                enabled=enabled,
                sub_events=sub_events,
                iterations=edef.get("iterations", 1),
            )
            return group

        # Look up in EVENT_REGISTRY
        from salmon_ibm.events import EVENT_REGISTRY

        cls = EVENT_REGISTRY.get(etype)
        if cls is None:
            import warnings

            warnings.warn(
                f"Event type '{etype}' not in EVENT_REGISTRY — "
                f"replaced with no-op. Event: {name}",
                stacklevel=2,
            )
            from salmon_ibm.events_hexsim import DataProbeEvent

            return DataProbeEvent(
                name=f"[unimplemented:{etype}] {name}",
                trigger=trigger,
                population_name=population_name,
                enabled=enabled,
            )

        # Try from_descriptor if available (Phase 3 incremental migration)
        if hasattr(cls, "from_descriptor"):
            from salmon_ibm.event_descriptors import DESCRIPTOR_REGISTRY

            desc_cls = DESCRIPTOR_REGISTRY.get(etype)
            if desc_cls is not None:
                descriptor = desc_cls(
                    name=name,
                    event_type=etype,
                    timestep=edef.get("timestep", 0),
                    population_name=population_name,
                    enabled=enabled,
                )
                evt = cls.from_descriptor(descriptor)
                evt.trigger = trigger
                return evt

        # Build event with available params
        try:
            evt = cls(
                name=name,
                trigger=trigger,
                population_name=population_name,
                enabled=enabled,
            )
        except TypeError:
            evt = cls(name=name, trigger=trigger)
            evt.population_name = population_name
            evt.enabled = enabled

        # Apply params if present
        # Some XML param keys differ from event field names — remap them
        _PARAM_ALIASES = {
            "transition_trait": "trait_name",
            "survival_accumulator": "survival_accumulator",
            "initial_size": "n_agents",
            "initialization_spatial_data": "initialization_spatial_data",
        }
        params = edef.get("params", {})
        for key, val in params.items():
            field_name = _PARAM_ALIASES.get(key, key)
            if hasattr(evt, field_name):
                setattr(evt, field_name, val)
            elif hasattr(evt, key):
                setattr(evt, key, val)

        # Wire updater_functions for accumulate events
        if etype == "accumulate":
            updater_functions = edef.get("updater_functions", [])
            if updater_functions and hasattr(evt, "updater_functions"):
                evt.updater_functions = updater_functions

        return evt

    def _load_lookup_tables(self, events, workspace_dir: str) -> None:
        """Post-process events: load CSV files for DataLookupEvents."""
        from salmon_ibm.events_hexsim import DataLookupEvent

        ws = Path(workspace_dir)
        lookup_dir = ws / "Analysis" / "Data Lookup"

        for evt in events:
            if (
                isinstance(evt, DataLookupEvent)
                and evt.file_name
                and evt.lookup_table is None
            ):
                csv_path = lookup_dir / evt.file_name
                if csv_path.exists():
                    evt.lookup_table = np.loadtxt(
                        csv_path, delimiter=",", dtype=np.float64
                    )
                else:
                    import warnings

                    warnings.warn(
                        f"DataLookupEvent '{evt.name}': CSV file '{csv_path}' not found. "
                        f"Event will be a no-op.",
                        UserWarning,
                        stacklevel=2,
                    )
            # Recurse into EventGroups
            if hasattr(evt, "sub_events"):
                self._load_lookup_tables(evt.sub_events, workspace_dir)
