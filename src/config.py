import copy
import yaml


def deep_update(base: dict, new: dict) -> dict:
    """Recursively update nested dictionary base with new values."""
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_cfg(*paths: str) -> dict:
    """Load and deep-merge YAML configuration files in sequence.
    Later files override earlier ones.
    """
    cfg = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            deep_update(cfg, copy.deepcopy(yaml.safe_load(f) or {}))
    return cfg

