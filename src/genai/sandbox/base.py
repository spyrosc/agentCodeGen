"""Sandbox transport protocol.

Any object that can run a shell command in an isolated environment, against
a project directory on a shared volume, satisfies :class:`Sandbox`. Agents
depend only on this protocol — never on a concrete transport — so swapping
``DockerExecSandbox`` for, e.g., an HTTP shim is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    output: str  # combined stdout+stderr, already truncated for LLM consumption


class Sandbox(Protocol):
    """Run shell commands inside an isolated execution environment."""

    def exec(self, project_id: str, command: str, timeout_s: int) -> ExecResult:
        """Execute ``command`` with cwd = the project's directory in the sandbox."""
        ...
