"""YAML configuration loader."""
from __future__ import annotations

from pathlib import Path

import yaml

from salmon_ibm.bioenergetics import BioParams
from salmon_ibm.behavior import BehaviorParams


def load_config(path: str | Path) -> dict:
    """Load and return simulation configuration from a YAML file.

    For HexSim configs (grid.type == "hexsim"), resolves the workspace path
    relative to the config file's parent directory.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve HexSim workspace path relative to the config file
    if cfg.get("grid", {}).get("type") == "hexsim":
        hs = cfg.get("hexsim", {})
        ws = hs.get("workspace", "")
        if ws and not Path(ws).is_absolute():
            hs["workspace"] = str(path.parent / ws)

    validate_config(cfg)
    return cfg


# ------------------------------------------------------------------
# T4.7  — build param dataclasses from optional YAML sections
# ------------------------------------------------------------------

def bio_params_from_config(cfg: dict) -> BioParams:
    """Create a BioParams instance from the optional ``bioenergetics`` section.

    Keys present in the YAML override the dataclass defaults; missing keys
    keep their defaults.
    """
    overrides = dict(cfg.get("bioenergetics", {}) or {})
    return BioParams(**overrides)


def load_bio_params_from_config(cfg: dict) -> "BalticSpeciesConfig":
    """Route to BalticSpeciesConfig from species_config if present,
    else wrap a plain BioParams in BalticSpeciesConfig(wild=BioParams, hatchery=None).

    Always returns BalticSpeciesConfig — the unified return type
    eliminates isinstance branching at the caller (simulation.py:294)
    and the AttributeError failure mode.
    """
    from salmon_ibm.baltic_params import BalticSpeciesConfig, load_baltic_species_config

    species_config = cfg.get("species_config")
    if species_config is not None:
        return load_baltic_species_config(species_config)
    # Legacy non-Baltic path: wrap plain BioParams
    plain = bio_params_from_config(cfg)
    return BalticSpeciesConfig(wild=plain, hatchery=None)


def behavior_params_from_config(cfg: dict) -> BehaviorParams:
    """Create a BehaviorParams instance from the optional ``behavior`` section.

    If the section is absent or empty the standard defaults (including the
    probability table) are returned via ``BehaviorParams.defaults()``.
    """
    overrides = dict(cfg.get("behavior", {}) or {})
    if not overrides:
        return BehaviorParams.defaults()
    # Start from defaults so p_table is populated, then apply overrides
    bp = BehaviorParams.defaults()
    for key, value in overrides.items():
        if hasattr(bp, key):
            setattr(bp, key, value)
    return bp


# ------------------------------------------------------------------
# T4.8  — basic config validation
# ------------------------------------------------------------------

def validate_config(cfg: dict) -> None:
    """Validate the configuration dictionary at load time.

    Raises ``ValueError`` with a descriptive message on failure.
    """
    # --- mesh backend section ---
    # H3 landscapes use a top-level ``mesh_backend: h3`` + ``h3_landscape_nc``
    # path instead of the legacy ``grid`` section.  TriMesh and HexSim configs
    # still go through the grid-section path.
    mesh_backend = cfg.get("mesh_backend", "trimesh")
    if mesh_backend == "h3":
        if "h3_landscape_nc" not in cfg:
            raise ValueError(
                "mesh_backend=h3 requires 'h3_landscape_nc' (path to landscape "
                "NetCDF produced by scripts/build_*_h3_landscape.py)"
            )
    elif mesh_backend == "h3_multires":
        if "h3_multires_landscape_nc" not in cfg:
            raise ValueError(
                "mesh_backend=h3_multires requires 'h3_multires_landscape_nc' "
                "(path to a multi-res landscape NetCDF built by "
                "scripts/build_h3_multires_landscape.py)"
            )
    else:
        grid = cfg.get("grid")
        if not grid:
            raise ValueError("Config must contain a 'grid' section")
        if "file" not in grid and "type" not in grid:
            raise ValueError("grid section must contain a 'file' or 'type' key")

    # --- optional bioenergetics section ---
    bio = cfg.get("bioenergetics")
    if bio:
        if "RA" in bio and bio["RA"] <= 0:
            raise ValueError(f"bioenergetics.RA must be > 0, got {bio['RA']}")
        if "RB" in bio and bio["RB"] >= 0:
            raise ValueError(f"bioenergetics.RB must be < 0, got {bio['RB']}")
        if "RQ" in bio and bio["RQ"] <= 0:
            raise ValueError(f"bioenergetics.RQ must be > 0, got {bio['RQ']}")
        if "ED_MORTAL" in bio and bio["ED_MORTAL"] <= 0:
            raise ValueError(
                f"bioenergetics.ED_MORTAL must be > 0, got {bio['ED_MORTAL']}"
            )
        t_max = bio.get("T_MAX", BioParams.T_MAX)
        t_opt = bio.get("T_OPT", BioParams.T_OPT)
        if "T_MAX" in bio or "T_OPT" in bio:
            if t_max <= t_opt:
                raise ValueError(
                    f"bioenergetics.T_MAX ({t_max}) must be > T_OPT ({t_opt})"
                )

    # --- optional genetics section ---
    gen = cfg.get("genetics")
    if gen:
        loci = gen.get("loci")
        if not loci or not isinstance(loci, list):
            raise ValueError("genetics.loci must be a non-empty list")
        for loc in loci:
            if "name" not in loc or "n_alleles" not in loc:
                raise ValueError("Each locus must have 'name' and 'n_alleles'")
            if loc["n_alleles"] < 2:
                raise ValueError(f"n_alleles must be >= 2, got {loc['n_alleles']}")

    # --- optional barriers section ---
    bar = cfg.get("barriers")
    if bar:
        if "file" not in bar:
            raise ValueError("barriers section must have a 'file' key")


def population_config_from_yaml(cfg: dict) -> dict:
    """Extract population configuration from YAML."""
    return cfg.get("population", {})


def barrier_config_from_yaml(cfg: dict) -> dict | None:
    """Extract barrier configuration. Returns None if no barriers configured."""
    return cfg.get("barriers")


def genetics_config_from_yaml(cfg: dict) -> dict | None:
    """Extract genetics configuration. Returns None if no genetics configured."""
    return cfg.get("genetics")


# ------------------------------------------------------------------
# Builder functions: config dict → domain objects
# ------------------------------------------------------------------

def genome_from_config(cfg: dict, n_agents: int):
    """Create GenomeManager from optional ``genetics`` YAML section.

    Example YAML::

        genetics:
          loci:
            - name: run_timing
              n_alleles: 4
              position: 0.0
            - name: growth_rate
              n_alleles: 3
              position: 50.0
          rng_seed: 42

    Returns None if no genetics section is present.
    """
    gen_cfg = cfg.get("genetics")
    if not gen_cfg:
        return None
    from salmon_ibm.genetics import GenomeManager, LocusDefinition
    loci = [LocusDefinition(**loc) for loc in gen_cfg["loci"]]
    seed = gen_cfg.get("rng_seed")
    gm = GenomeManager(n_agents, loci, rng_seed=seed)
    if gen_cfg.get("initialize_random", True):
        gm.initialize_random()
    return gm


def barrier_map_from_config(cfg: dict, mesh):
    """Create BarrierMap + arrays from optional ``barriers`` YAML section.

    Example YAML::

        barriers:
          file: barriers.hbf
          classes:
            dam:
              forward: {mortality: 0.1, deflection: 0.8, transmission: 0.1}
              reverse: {mortality: 0.0, deflection: 1.0, transmission: 0.0}

    Returns ``(BarrierMap, barrier_arrays)`` or None.
    """
    bar_cfg = cfg.get("barriers")
    if not bar_cfg:
        return None
    from salmon_ibm.barriers import BarrierMap, BarrierClass, BarrierOutcome
    class_config = {}
    for name, cls_def in bar_cfg.get("classes", {}).items():
        fwd = cls_def.get("forward", {})
        rev = cls_def.get("reverse", {})
        class_config[name] = BarrierClass(
            name=name,
            forward=BarrierOutcome(
                fwd.get("mortality", 0), fwd.get("deflection", 1), fwd.get("transmission", 0),
            ),
            reverse=BarrierOutcome(
                rev.get("mortality", 0), rev.get("deflection", 1), rev.get("transmission", 0),
            ),
        )
    hbf_path = bar_cfg["file"]
    bmap = BarrierMap.from_hbf_hexsim(hbf_path, mesh, class_config=class_config) if class_config else BarrierMap.from_hbf_hexsim(hbf_path, mesh)
    if not bmap.has_barriers():
        return None
    arrays = bmap.to_arrays(mesh)
    return bmap, arrays


def network_from_config(cfg: dict):
    """Create StreamNetwork from optional ``network`` YAML section.

    Example YAML::

        network:
          segments:
            - id: 0
              length: 1000.0
              upstream_ids: []
              downstream_ids: [1]
            - id: 1
              length: 2000.0
              upstream_ids: [0]
              downstream_ids: []

    Returns None if no network section is present.
    """
    net_cfg = cfg.get("network")
    if not net_cfg:
        return None
    from salmon_ibm.network import StreamNetwork, SegmentDefinition
    segs = [SegmentDefinition(**s) for s in net_cfg["segments"]]
    return StreamNetwork(segs)
