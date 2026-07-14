"""Configuration loading and lightweight access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class Config:
    """Thin wrapper around the parsed YAML config.

    Access nested values with dotted keys, e.g. ``cfg.get("data.horizon")``.
    Paths declared in the config are resolved relative to the project root.
    """

    raw: dict[str, Any]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted_key: str) -> Path:
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"No path configured at '{dotted_key}'")
        return (PROJECT_ROOT / value).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Config(raw=raw)
