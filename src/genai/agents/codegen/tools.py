"""Tools the LangGraph ReAct codegen agent can call.

Thin wrapper over :mod:`genai.agents._common.fs_tools`: same FS surface,
plus the two sandbox tools (`run_in_sandbox`, `run_tests`) and the
terminal `submit_project` tool.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import StructuredTool, tool

from ...core.config import get_settings
from ...sandbox.base import Sandbox
from ...sandbox.workspace import Workspace
from .._common.fs_tools import build_fs_tools, emit


def build_tools(
    workspace: Workspace,
    sandbox: Sandbox,
    *,
    on_event: Callable[[str, dict], None] | None = None,
) -> tuple[list[StructuredTool], dict]:
    """Return ``(tools, state)``.

    ``state["submitted"]`` flips to ``True`` once the agent calls
    ``submit_project``. ``state["summary"]`` holds its summary text.
    """
    cfg = get_settings().sandbox
    state: dict = {"submitted": False, "summary": None}

    fs_tools = build_fs_tools(workspace, on_event=on_event)

    @tool
    def run_in_sandbox(command: str, timeout: int = 0) -> str:
        """Run a shell ``command`` inside the sandbox at the project root.

        ``command`` is interpreted by ``sh -c`` so it may use pipes and
        redirection, but it cannot escape the sandbox container. Returns
        ``"exit=<code>\\n<combined stdout+stderr>"``.
        """
        t = timeout or cfg.default_timeout_s
        emit(on_event, "run_in_sandbox", command=command, timeout=t)
        result = sandbox.exec(workspace.project_id, command, t)
        emit(on_event, "run_in_sandbox_done", exit=result.exit_code)
        return f"exit={result.exit_code}\n{result.output}"

    @tool
    def run_tests() -> str:
        """Install ``requirements.txt`` (if present) and run pytest in the sandbox."""
        cmd = (
            "set -e; "
            "if [ -f requirements.txt ]; then "
            "  pip install --quiet --disable-pip-version-check -r requirements.txt; "
            "fi; "
            "pytest -q --maxfail=5 --color=no"
        )
        emit(on_event, "run_tests", command=cmd)
        result = sandbox.exec(workspace.project_id, cmd, cfg.default_timeout_s)
        emit(on_event, "run_tests_done", exit=result.exit_code)
        return f"exit={result.exit_code}\n{result.output}"

    @tool
    def submit_project(summary: str) -> str:
        """End the session and submit the project with a short summary."""
        state["submitted"] = True
        state["summary"] = summary
        emit(on_event, "submit_project", summary=summary)
        return "PROJECT_SUBMITTED"

    return ([*fs_tools, run_in_sandbox, run_tests, submit_project], state)
