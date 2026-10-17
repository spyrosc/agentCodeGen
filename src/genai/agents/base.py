"""Agent contracts.

An :class:`Agent` is a callable that streams :class:`AgentEvent`\\ s while
working on a task. The contract is intentionally minimal so agents of very
different shapes (ReAct, plan-and-execute, multi-agent) can coexist.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class AgentEvent:
    """One observable step in an agent's execution.

    ``kind`` is a short tag the UI uses to render an icon (e.g.
    ``"write_file"``, ``"message"``, ``"tool_result"``). Free-form data goes
    into ``data``.
    """

    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    submitted: bool
    summary: str | None = None
    project_id: str | None = None


class Agent(Protocol):
    """Streaming agent protocol."""

    name: str
    description: str

    def stream(self, task: str, **kwargs: Any) -> Iterator[AgentEvent]:
        """Yield events as the agent works on ``task``."""
        ...

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run the agent to completion and return the terminal result."""
        ...
