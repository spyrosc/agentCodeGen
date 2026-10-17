"""Pydantic models for the eval suite YAML schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FunctionalCheck(BaseModel):
    """One executable check run inside the sandbox after the agent finishes."""

    name: str
    type: Literal["pytest", "python_import", "npm_build", "command"]
    # Working directory relative to the project workspace root.
    cwd: str = "."
    # Free-form command (used by `command`); other types build their own.
    command: str | None = None
    # For `python_import`: the module name(s) to import.
    module: str | None = None
    # Per-check timeout (seconds). Falls back to the case timeout.
    timeout_s: int | None = None
    # Required check failures fail the case; non-required ones are reported only.
    required: bool = True
    # Expected exit code (for `command`); other types default to 0.
    expect_exit_code: int = 0


class EvalCase(BaseModel):
    """One eval scenario."""

    id: str
    agent: str
    prompt: str
    tags: list[str] = Field(default_factory=list)
    timeout_s: int = 1800
    expect_files: list[str] = Field(default_factory=list)
    functional: list[FunctionalCheck] = Field(default_factory=list)
    skip: str | None = None  # non-empty reason → skipped
    # Optional per-case agent overrides forwarded to ``stream(...)``.
    max_iterations: int | None = None

    @field_validator("id")
    @classmethod
    def _safe_id(cls, v: str) -> str:
        import re

        if not re.fullmatch(r"[a-zA-Z0-9_-]+", v):
            raise ValueError(f"unsafe case id: {v!r}")
        return v


class EvalSuite(BaseModel):
    name: str
    description: str | None = None
    cases: list[EvalCase]

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, v: list[EvalCase]) -> list[EvalCase]:
        seen: set[str] = set()
        for case in v:
            if case.id in seen:
                raise ValueError(f"duplicate case id: {case.id!r}")
            seen.add(case.id)
        return v
