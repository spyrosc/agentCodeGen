"""Frontend specialist node.

Mirrors :mod:`backend` but scoped to the ``frontend/`` subtree and using a
``run_frontend_check`` tool that runs ``npm install``, ``tsc --noEmit`` and
``vite build``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ...core.config import get_settings
from ...core.logging import get_logger
from ...reflection.injection import prepend_lessons
from ...sandbox.base import Sandbox
from ...sandbox.workspace import Workspace
from .._common.fs_tools import build_fs_tools, emit, scope_to
from .._common.runtime import ai_text, drive_react
from ._prompts import feedback_section, load_prompt
from .spec import ProjectSpec

log = get_logger("agents.fullstack.frontend")


def run_frontend(
    llm: Any,
    workspace: Workspace,
    sandbox: Sandbox,
    spec: ProjectSpec,
    *,
    on_event: Callable[[str, dict], None] | None = None,
    max_iterations: int = 40,
    enable_build: bool = True,
    feedback: str | None = None,
) -> dict:
    """Run the frontend specialist; returns the same shape as ``run_backend``."""
    cfg = get_settings().sandbox
    scoped = scope_to(workspace, "frontend")
    state: dict = {"submitted": False, "summary": None, "checks_passed": False}

    def on_fs_event(kind: str, payload: dict) -> None:
        if kind in {"write_file", "delete_file"}:
            state["checks_passed"] = False
        emit(on_event, kind, **payload)

    fs_tools = build_fs_tools(scoped, on_event=on_fs_event, label="frontend")

    build_step = "npm run build" if enable_build else "echo '<vite build skipped by config>'"

    @tool
    def run_frontend_check() -> str:
        """Run ``npm install``, ``tsc --noEmit`` and ``vite build`` in the sandbox."""
        cmd = (
            "set -e; cd frontend; "
            "if [ ! -d node_modules ]; then npm install --no-fund --no-audit --loglevel=error; fi; "
            "npx tsc --noEmit; "
            f"{build_step}"
        )
        # Frontend builds are slow on first run — give them headroom.
        timeout = max(cfg.default_timeout_s, 300)
        emit(on_event, "run_frontend_check", scope="frontend", timeout=timeout)
        result = sandbox.exec(workspace.project_id, cmd, timeout)
        state["checks_passed"] = result.exit_code == 0
        emit(on_event, "run_frontend_check_done", scope="frontend", exit=result.exit_code)
        return f"exit={result.exit_code}\n{result.output}"

    @tool
    def submit_frontend(summary: str) -> str:
        """Mark the frontend complete (or hand off with a note)."""
        state["submitted"] = True
        state["summary"] = summary
        emit(on_event, "submit_frontend", summary=summary)
        return "FRONTEND_SUBMITTED"

    tools = [*fs_tools, run_frontend_check, submit_frontend]
    graph = create_react_agent(llm, tools=tools)

    system = load_prompt("frontend")
    system = prepend_lessons(
        system,
        category="fullstack-frontend",
        query=spec.meta.name + " " + " ".join(p.name for p in spec.pages),
        tags=["fullstack", "frontend"],
    )
    user = (
        "PROJECT SPEC (JSON):\n"
        f"{spec.to_json()}\n\n"
        "The skeleton already exists under `frontend/` (your scoped root).\n"
        "Implement entity pages, wire routes/links in `src/App.tsx`, and run "
        "`run_frontend_check` until it passes."
    ) + feedback_section(feedback, "run_frontend_check")
    inputs = {"messages": [SystemMessage(content=system), HumanMessage(content=user)]}

    def on_message(message: Any) -> None:
        if text := ai_text(message):
            emit(on_event, "specialist_message", scope="frontend", content=text[:500])

    outcome = drive_react(
        graph, inputs, recursion_limit=max_iterations, state=state, on_message=on_message
    )
    if outcome == "budget_exhausted":
        emit(on_event, "budget_exhausted", scope="frontend", limit=max_iterations)
    return {**state, "outcome": outcome}
