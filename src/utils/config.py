"""Configuration loading utilities."""

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    """Merge multiple config dicts; later keys override earlier ones."""
    merged: dict = {}
    for cfg in configs:
        merged.update(cfg)
    return merged
