"""Loads the three YAML configs into plain dicts."""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path("configs")


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path} (run scripts from the repo root)")
    with open(path) as f:
        return yaml.safe_load(f)


def load_all() -> tuple[dict, dict, dict]:
    """Returns (data_cfg, features_cfg, model_cfg)."""
    return load_config("data"), load_config("features"), load_config("model")
