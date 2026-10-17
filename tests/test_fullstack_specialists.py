"""Fullstack specialists: step-budget handling and the ``checks_passed`` signal.

``create_react_agent`` is replaced by a fake graph that invokes the
specialist's real tools in a scripted order, so no LLM is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from genai.agents.fullstack import backend as backend_mod
from genai.agents.fullstack import frontend as frontend_mod
from genai.agents.fullstack.spec import Entity, Page, ProjectMeta, ProjectSpec
from genai.core.config import reload_settings
from genai.sandbox.base import ExecResult
from genai.sandbox.workspace import Workspace, new_project_id


@pytest.fixture(autouse=True)
def _fresh_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REFLECT_LESSONS_ENABLED", raising=False)
    monkeypatch.setenv("REFLECT_LESSONS_PATH", str(tmp_path / "lessons.jsonl"))
    reload_settings()
    yield
    reload_settings()


class ExitCodes:
    """Sandbox whose successive ``exec`` calls return the given exit codes."""

    def __init__(self, *codes: int):
        self.codes = list(codes)

    def exec(self, _project_id, _command, _timeout_s):
        return ExecResult(exit_code=self.codes.pop(0), output="")


def _scripted_graph(monkeypatch, module, *steps):
    """Patch ``module.create_react_agent`` with a graph that runs ``steps``.

    Each step is ``(tool_name, args)`` (invoked on the real tool), an AI text
    string, or an exception to raise.
    """

    def factory(_llm, tools):
        by_name = {t.name: t for t in tools}

        class Graph:
            def stream(self, _inputs, **_kwargs):
                for step in steps:
                    if isinstance(step, BaseException):
                        raise step
                    if isinstance(step, str):
                        yield {"agent": {"messages": [AIMessage(content=step)]}}
                        continue
                    name, args = step
                    by_name[name].invoke(args)
                    yield {"tools": {"messages": []}}

        return Graph()

    monkeypatch.setattr(module, "create_react_agent", factory)


def _ws(tmp_path: Path) -> Workspace:
    return Workspace.create(new_project_id("t"), base=tmp_path)


def _backend_spec() -> ProjectSpec:
    return ProjectSpec(meta=ProjectMeta(name="x", title="X"), entities=[Entity(name="Note", fields=[])])


def _frontend_spec() -> ProjectSpec:
    return ProjectSpec(meta=ProjectMeta(name="x", title="X"), pages=[Page(name="NotesPage", route="/notes")])


def _run_backend(tmp_path, sandbox, events=None, max_iterations=10):
    on_event = (lambda k, p: events.append((k, p))) if events is not None else None
    return backend_mod.run_backend(
        object(), _ws(tmp_path), sandbox, _backend_spec(), on_event=on_event, max_iterations=max_iterations
    )


def test_backend_checks_pass_when_last_test_run_is_green(tmp_path, monkeypatch):
    _scripted_graph(
        monkeypatch, backend_mod,
        ("write_file", {"path": "app/x.py", "content": "x = 1\n"}),
        ("run_python_tests", {}),
        ("submit_backend", {"summary": "done"}),
    )
    res = _run_backend(tmp_path, ExitCodes(0))
    assert res["checks_passed"] is True
    assert res["submitted"] is True
    assert res["outcome"] == "submitted"


def test_backend_write_after_green_tests_invalidates_checks(tmp_path, monkeypatch):
    _scripted_graph(
        monkeypatch, backend_mod,
        ("run_python_tests", {}),
        ("write_file", {"path": "app/x.py", "content": "x = 2\n"}),
        ("submit_backend", {"summary": "all tests pass"}),
    )
    res = _run_backend(tmp_path, ExitCodes(0))
    assert res["checks_passed"] is False


def test_backend_failing_tests_are_not_ok_despite_upbeat_summary(tmp_path, monkeypatch):
    _scripted_graph(
        monkeypatch, backend_mod,
        ("run_python_tests", {}),
        ("submit_backend", {"summary": "looks great"}),
    )
    res = _run_backend(tmp_path, ExitCodes(1))
    assert res["checks_passed"] is False
    assert res["submitted"] is True


def test_backend_budget_exhaustion_returns_instead_of_raising(tmp_path, monkeypatch):
    _scripted_graph(
        monkeypatch, backend_mod,
        "working on it",
        ("run_python_tests", {}),
        GraphRecursionError("limit"),
    )
    events: list = []
    res = _run_backend(tmp_path, ExitCodes(0), events=events, max_iterations=7)
    assert res["outcome"] == "budget_exhausted"
    assert res["submitted"] is False
    assert res["checks_passed"] is True
    assert ("budget_exhausted", {"scope": "backend", "limit": 7}) in events
    assert ("specialist_message", {"scope": "backend", "content": "working on it"}) in events


def test_frontend_tracks_check_result_and_budget(tmp_path, monkeypatch):
    _scripted_graph(
        monkeypatch, frontend_mod,
        ("run_frontend_check", {}),
        GraphRecursionError("limit"),
    )
    events: list = []
    res = frontend_mod.run_frontend(
        object(), _ws(tmp_path), ExitCodes(2), _frontend_spec(),
        on_event=lambda k, p: events.append((k, p)), max_iterations=9,
    )
    assert res["checks_passed"] is False
    assert res["outcome"] == "budget_exhausted"
    assert ("budget_exhausted", {"scope": "frontend", "limit": 9}) in events


def _capture_user_message(monkeypatch, module) -> dict:
    captured: dict = {}

    class Graph:
        def stream(self, inputs, **_kwargs):
            captured["user"] = inputs["messages"][1].content
            return iter(())

    monkeypatch.setattr(module, "create_react_agent", lambda *_a, **_k: Graph())
    return captured


@pytest.mark.parametrize(
    ("module", "spec", "check_tool"),
    [(backend_mod, _backend_spec, "run_python_tests"),
     (frontend_mod, _frontend_spec, "run_frontend_check")],
)
def test_rerun_specialist_receives_verification_feedback(tmp_path, monkeypatch, module, spec, check_tool):
    captured = _capture_user_message(monkeypatch, module)
    run = module.run_backend if module is backend_mod else module.run_frontend

    run(object(), _ws(tmp_path), ExitCodes(), spec(),
        feedback="backend-1 | ModuleNotFoundError: No module named 'app'")

    assert "FAILED VERIFICATION" in captured["user"]
    assert "ModuleNotFoundError: No module named 'app'" in captured["user"]
    assert f"run `{check_tool}` again" in captured["user"]


def test_first_run_has_no_feedback_section(tmp_path, monkeypatch):
    captured = _capture_user_message(monkeypatch, backend_mod)

    backend_mod.run_backend(object(), _ws(tmp_path), ExitCodes(), _backend_spec())

    assert "FAILED VERIFICATION" not in captured["user"]
