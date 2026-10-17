"""``ReflectAgent`` \u2014 on-demand self-reflection over the platform's own source.

Single-loop ReAct agent (mirrors :class:`~genai.agents.codegen.agent.CodegenAgent`)
with a much narrower tool surface: read / write inside an isolated clone,
sandboxed pytest + ruff, eval-harness lookups, and a terminal
``propose_patch`` tool that emits a reviewable branch.

Failure modes are surfaced as ``error`` events; the agent never crashes the
caller. See [docs/self-reflection.md](../../../../docs/self-reflection.md).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ...core.config import get_settings
from ...core.logging import get_logger
from ...llm import load_llm
from ...reflection.injection import prepend_lessons
from ...reflection.store import LessonStore
from ...sandbox import DockerExecSandbox, Sandbox, Workspace, new_project_id
from .._common.runtime import EventBuffer, message_event, relay_react
from ..base import AgentEvent, AgentResult
from ..registry import register
from .prompts import render_prompt
from .repo import ReflectionUnavailable, prepare_clone
from .tools import OperationCap, build_reflect_tools

log = get_logger("agents.reflect")

# Tool results with these prefixes are surfaced as warnings in the UI.
_to_event = partial(message_event, warn_prefixes=("PATCH_REJECTED", "PERMISSION_DENIED"))


@register
@dataclass
class ReflectAgent:
    """Self-reflection agent that proposes patches to the platform's own code."""

    name: str = field(default="reflect", init=False)
    description: str = field(
        default=(
            "Diagnoses failures in past eval runs, proposes a small patch to "
            "the platform's source on an isolated branch, and validates the "
            "patch against pytest + ruff (+ optional eval suite)."
        ),
        init=False,
    )

    llm: Any = None
    sandbox: Sandbox | None = None
    lesson_store: LessonStore | None = None

    # ----------------------------------------------------------- factory
    def __post_init__(self) -> None:
        if self.llm is None:
            self.llm = load_llm()
        if self.sandbox is None:
            self.sandbox = DockerExecSandbox.from_settings()
        if self.lesson_store is None:
            self.lesson_store = LessonStore.from_settings()

    # ----------------------------------------------------------- helpers
    def _new_workspace(self, project_id: str | None = None) -> Workspace:
        return Workspace.create(project_id or new_project_id("reflect"))

    def _build_user_message(self, task: str, *, eval_run_id: str | None) -> str:
        body = task.strip() or "Review the platform source and propose an improvement."
        if not eval_run_id:
            return f"TASK:\n{body}\n\nBegin by calling `query_lessons` and `git_status`."
        return (
            f"TASK:\n{body}\n\n"
            f"EVAL RUN: {eval_run_id}\n"
            "Start by calling `read_eval_run` for that run id and "
            "`read_failed_case` for every failing case. Use the failure "
            "evidence to guide your edits.\n"
        )

    # -------------------------------------------------------- public API
    def stream(
        self,
        task: str,
        *,
        workspace: Workspace | None = None,
        max_iterations: int | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
        eval_run_id: str | None = None,
    ) -> Iterator[AgentEvent]:
        cfg = get_settings().agent
        reflect_cfg = get_settings().reflect
        workspace = workspace or self._new_workspace()
        recursion_limit = max_iterations or cfg.max_iterations
        events = EventBuffer(on_event)

        events.emit(
            "workspace_created",
            {"project_id": workspace.project_id, "root": str(workspace.root)},
        )

        # Phase 1: prepare the clone. Failure here is graceful.
        events.emit("clone_started", {"mount": reflect_cfg.repo_mount_path})
        yield from events.drain()
        try:
            clone = prepare_clone(
                workspace,
                repo_mount=reflect_cfg.repo_mount_path,
                base_branch=reflect_cfg.base_branch,
            )
        except ReflectionUnavailable as exc:
            events.emit("error", {"message": f"reflection unavailable: {exc}"})
            events.emit(
                "done",
                {"submitted": False, "summary": str(exc), "project_id": workspace.project_id},
            )
            yield from events.drain()
            return
        events.emit(
            "clone_ready",
            {
                "path": str(clone.clone_path),
                "is_git": clone.is_git,
                "base_sha": clone.base_sha,
                "base_branch": clone.base_branch,
            },
        )
        yield from events.drain()

        tools, state = build_reflect_tools(
            workspace,
            clone,
            self.sandbox,
            lesson_store=self.lesson_store,
            llm=self.llm,
            on_event=events.emit,
        )

        # System prompt: render template + force-inject lessons (the reflect
        # agent always benefits from them, regardless of the master switch).
        system = render_prompt(
            max_files_changed=reflect_cfg.max_files_changed,
            max_patch_lines=reflect_cfg.max_patch_lines,
        )
        system = prepend_lessons(
            system,
            category="reflect",
            store=self.lesson_store,
            query=task,
            tags=[eval_run_id] if eval_run_id else None,
            force=True,
        )
        user = self._build_user_message(task, eval_run_id=eval_run_id)

        graph = create_react_agent(self.llm, tools=tools)
        inputs = {"messages": [SystemMessage(content=system), HumanMessage(content=user)]}

        try:
            outcome = yield from relay_react(
                graph,
                inputs,
                recursion_limit=recursion_limit,
                state=state,
                events=events,
                to_event=_to_event,
            )
        except OperationCap as exc:
            events.emit("error", {"message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            log.exception("reflect graph crashed")
            events.emit("error", {"message": f"{type(exc).__name__}: {exc}"})
        else:
            if outcome == "budget_exhausted":
                events.emit("budget_exhausted", {"limit": recursion_limit})

        events.emit(
            "done",
            {
                "submitted": bool(state.get("submitted")),
                "summary": state.get("summary"),
                "project_id": workspace.project_id,
                "patch": state.get("patch"),
                "denied_attempts": state.get("denied_attempts", []),
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


# Keep the JSON / Path imports referenced when this module is sliced down to
# its public API in dev tooling.
_ = json, Path
