"""YAML loader for :class:`EvalSuite`."""

from __future__ import annotations

from pathlib import Path

import yaml

from .cases import EvalSuite


def load_suite(path: str | Path) -> EvalSuite:
    """Parse a YAML file into an :class:`EvalSuite`."""
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top-level YAML must be a mapping")
    if "name" not in data:
        data["name"] = p.stem
    return EvalSuite.model_validate(data)
