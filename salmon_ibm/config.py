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
    # --- grid section ---
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


def population_config_from_yaml(cfg: dict) -> dict:
    """Extract population configuration from YAML."""
    return cfg.get("population", {})


def barrier_config_from_yaml(cfg: dict) -> dict | None:
    """Extract barrier configuration. Returns None if no barriers configured."""
    return cfg.get("barriers")


def genetics_config_from_yaml(cfg: dict) -> dict | None:
    """Extract genetics configuration. Returns None if no genetics configured."""
    return cfg.get("genetics")
