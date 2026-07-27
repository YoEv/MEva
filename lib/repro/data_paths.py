import os
from typing import Dict

import pandas as pd
import yaml


def detect_project_root() -> str:
    """Locate the repo root.

    Resolution order (first hit wins):
      1. explicit env var ``PROJECT_ROOT``
      2. nearest ancestor directory that contains ``config/paths.yaml``
      3. nearest ancestor directory that contains ``.git``
      4. current working directory (last-resort fallback)
    """
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isfile(os.path.join(here, "config", "paths.yaml")):
            return here
        if os.path.isdir(os.path.join(here, ".git")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return os.getcwd()


def _load_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_root(cfg: dict) -> str:
    """Config-aware project root. YAML value wins, otherwise auto-detect."""
    value = cfg.get("project_root") if cfg else None
    if value:
        return os.path.abspath(value)
    return detect_project_root()


def get_exp11_split_paths(config_path: str, splits: str = "clean") -> Dict[str, str]:
    """Return absolute train/val/test split paths driven purely by ``config.data.splits``.

    Historical note: the name keeps ``exp11`` for backwards-compatible imports, but this
    function **does not** read from any exp11 directory. All resolution is self-contained
    within ```` via the release config (typically ``config/paths.yaml``).
    """
    cfg = _load_cfg(config_path)
    root = resolve_project_root(cfg)
    split_cfg = cfg.get("data", {}).get("splits", {})
    if splits not in split_cfg:
        raise ValueError(
            f"No split paths configured for '{splits}' in {config_path}. "
            f"Configure data.splits.{splits}.{{train,val,test}} in the release config."
        )
    chosen = split_cfg[splits]
    return {
        "train": os.path.join(root, chosen["train"]),
        "val": os.path.join(root, chosen["val"]),
        "test": os.path.join(root, chosen["test"]),
    }


def load_exp11_splits(config_path: str, splits: str = "clean") -> Dict[str, pd.DataFrame]:
    paths = get_exp11_split_paths(config_path, splits=splits)
    return {k: pd.read_csv(v) for k, v in paths.items()}
