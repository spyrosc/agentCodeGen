"""Code-generation agent: a LangGraph ReAct agent over the codegen toolkit.

This is the user-facing entry point. The :class:`CodegenAgent` is registered
with :mod:`genai.agents.registry` so the UI can discover it without an
explicit import.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ...core.config import get_settings
from ...core.logging import get_logger
from ...llm import load_llm
from ...reflection.injection import prepend_lessons
from ...sandbox import DockerExecSandbox, Sandbox, Workspace, new_project_id
from .._common.runtime import EventBuffer, relay_react
from ..base import AgentEvent, AgentResult
from ..registry import register
from .prompts import SYSTEM_PROMPT
from .tools import build_tools

log = get_logger("agents.codegen")


@register
@dataclass
class CodegenAgent:
    """Turn natural-language functional requirements into a working Python project."""

    name: str = field(default="codegen", init=False)
    description: str = field(
        default=(
            "Plans, scaffolds, tests, and self-corrects a multi-file Python "
            "project from a free-text requirements brief."
        ),
        init=False,
    )

    llm: Any = None
    sandbox: Sandbox | None = None

    # ----------------------------------------------------------- factory
    def __post_init__(self) -> None:
        if self.llm is None:
            self.llm = load_llm()
        if self.sandbox is None:
            self.sandbox = DockerExecSandbox.from_settings()

    # ----------------------------------------------------------- helpers
    def _new_workspace(self, project_id: str | None = None) -> Workspace:
        return Workspace.create(project_id or new_project_id())

    def _build_inputs(self, task: str, workspace: Workspace) -> dict:
        system = prepend_lessons(SYSTEM_PROMPT, category="codegen", query=task)
        return {
            "messages": [
                SystemMessage(content=system),
                HumanMessage(
                    content=(
                        "FUNCTIONAL REQUIREMENTS:\n"
                        "----------------------------------------\n"
                        f"{task.strip()}\n"
                        "----------------------------------------\n"
                        f"Project workspace id: {workspace.project_id}\n"
                        "Begin."
                    )
                ),
            ]
        }

    # -------------------------------------------------------- public API
    def stream(
        self,
        task: str,
        *,
        workspace: Workspace | None = None,
        max_iterations: int | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> Iterator[AgentEvent]:
        """Yield :class:`AgentEvent`\\ s as the agent works.

        The first event is always ``"workspace_created"`` so the UI can show
        the project id immediately.
        """
        cfg = get_settings().agent
        workspace = workspace or self._new_workspace()
        recursion_limit = max_iterations or cfg.max_iterations
        events = EventBuffer(on_event)

        events.emit(
            "workspace_created",
            {"project_id": workspace.project_id, "root": str(workspace.root)},
        )
        yield from events.drain()

        tools, state = build_tools(workspace, self.sandbox, on_event=events.emit)
        graph = create_react_agent(self.llm, tools=tools)
        outcome = yield from relay_react(
            graph,
            self._build_inputs(task, workspace),
            recursion_limit=recursion_limit,
            state=state,
            events=events,
        )

        summary = state["summary"]
        if outcome == "budget_exhausted":
            summary = summary or f"step budget exhausted after {recursion_limit} steps"
            events.emit("budget_exhausted", {"limit": recursion_limit})
        events.emit(
            "done",
            {
                "submitted": state["submitted"],
                "summary": summary,
                "project_id": workspace.project_id,
            },
        )
        yield from events.drain()

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        result = AgentResult(submitted=False)
        for ev in self.stream(task, **kwargs):
            if ev.kind == "done":
                result = AgentResult(
                    submitted=bool(ev.data.get("submitted")),
                    summary=ev.data.get("summary"),
                    project_id=ev.data.get("project_id"),
                )
        return result
