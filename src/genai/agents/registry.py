"""Simple in-memory agent registry.

Agents register themselves at import time::

    from genai.agents import register

    @register
    class CodegenAgent:
        name = "codegen"
        ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .base import Agent

_REGISTRY: dict[str, type[Agent] | Callable[..., Agent]] = {}

T = TypeVar("T", bound=type[Agent])


def register(cls: T) -> T:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls!r} must declare a class-level `name` attribute")
    if name in _REGISTRY:
        raise ValueError(f"agent {name!r} is already registered")
    _REGISTRY[name] = cls
    return cls


def get(name: str) -> type[Agent] | Callable[..., Agent]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown agent {name!r}; registered: {sorted(_REGISTRY)}"
        ) from exc


def all_agents() -> dict[str, type[Agent] | Callable[..., Agent]]:
    return dict(_REGISTRY)
