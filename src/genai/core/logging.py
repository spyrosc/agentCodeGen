"""Lightweight logging helper. Wraps :mod:`logging` with a sensible default."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("genai")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    if not name.startswith("genai"):
        name = f"genai.{name}"
    return logging.getLogger(name)
