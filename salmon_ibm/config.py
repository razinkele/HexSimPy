"""YAML configuration loader."""
from pathlib import Path

import yaml


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

    return cfg
