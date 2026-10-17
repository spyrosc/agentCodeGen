"""Backend specialist node.

Builds a small ReAct sub-agent over backend-scoped FS tools plus a
``run_python_tests`` tool. Iterates until pytest passes or the model gives up
by calling ``submit_backend``.
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

log = get_logger("agents.fullstack.backend")


def run_backend(
    llm: Any,
    workspace: Workspace,
    sandbox: Sandbox,
    spec: ProjectSpec,
    *,
    on_event: Callable[[str, dict], None] | None = None,
    max_iterations: int = 40,
    feedback: str | None = None,
) -> dict:
    """Run the backend specialist.

    ``feedback`` explains why a previous attempt failed verification (re-runs only).
    Returns ``{"submitted", "summary", "checks_passed", "outcome"}`` where
    ``checks_passed`` means the last pytest run passed and no file changed since.
    """
    cfg = get_settings().sandbox
    scoped = scope_to(workspace, "backend")
    state: dict = {"submitted": False, "summary": None, "checks_passed": False}

    def on_fs_event(kind: str, payload: dict) -> None:
        if kind in {"write_file", "delete_file"}:
            state["checks_passed"] = False
        emit(on_event, kind, **payload)

    fs_tools = build_fs_tools(scoped, on_event=on_fs_event, label="backend")

    @tool
    def run_python_tests() -> str:
        """Run the backend test suite (pytest) inside the sandbox.

        Uses an in-memory SQLite db; the FastAPI app is wired through the
        skeleton's dependency override. Returns ``"exit=<code>\\n<output>"``.
        """
        cmd = (
            "set -e; cd backend; "
            "pip install --quiet --disable-pip-version-check -e . >/tmp/pip.log 2>&1 || "
            "  (echo '<pip install -e . failed>'; tail -50 /tmp/pip.log; exit 1); "
            "DATABASE_URL=sqlite:///./test.db JWT_SECRET=test-secret "
            "  pytest -q --maxfail=5 --color=no"
        )
        emit(on_event, "run_python_tests", scope="backend")
        result = sandbox.exec(workspace.project_id, cmd, cfg.default_timeout_s)
        state["checks_passed"] = result.exit_code == 0
        emit(on_event, "run_python_tests_done", scope="backend", exit=result.exit_code)
        return f"exit={result.exit_code}\n{result.output}"

    @tool
    def submit_backend(summary: str) -> str:
        """Mark the backend complete (or hand off with a note)."""
        state["submitted"] = True
        state["summary"] = summary
        emit(on_event, "submit_backend", summary=summary)
        return "BACKEND_SUBMITTED"

    tools = [*fs_tools, run_python_tests, submit_backend]
    graph = create_react_agent(llm, tools=tools)

    system = load_prompt("backend")
    system = prepend_lessons(
        system,
        category="fullstack-backend",
        query=spec.meta.name + " " + " ".join(e.name for e in spec.entities),
        tags=["fullstack", "backend"],
    )
    user = (
        "PROJECT SPEC (JSON):\n"
        f"{spec.to_json()}\n\n"
        "The skeleton already exists under `backend/` (your scoped root).\n"
        "Implement entity models, schemas, routers, the initial Alembic migration, "
        "and entity tests, then run `run_python_tests` until it passes."
    ) + feedback_section(feedback, "run_python_tests")
    inputs = {"messages": [SystemMessage(content=system), HumanMessage(content=user)]}

    def on_message(message: Any) -> None:
        if text := ai_text(message):
            emit(on_event, "specialist_message", scope="backend", content=text[:500])

    outcome = drive_react(
        graph, inputs, recursion_limit=max_iterations, state=state, on_message=on_message
    )
    if outcome == "budget_exhausted":
        emit(on_event, "budget_exhausted", scope="backend", limit=max_iterations)
    return {**state, "outcome": outcome}
