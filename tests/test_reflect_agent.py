"""Smoke tests for :class:`ReflectAgent`.

The agent itself is mostly orchestration over already-tested primitives
(``prepare_clone``, ``build_reflect_tools``, ``LessonStore``). These tests
verify the wiring is sane: the agent registers, gracefully reports an
unavailable clone, and runs end-to-end against a fake LLM.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from genai.agents import get as get_agent
from genai.agents.base import AgentEvent  # noqa: F401
from genai.agents.reflect.agent import ReflectAgent
from genai.core.config import reload_settings
from genai.reflection.store import LessonStore
from genai.sandbox import Workspace


@pytest.fixture(autouse=True)
def _fresh_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REFLECT_LESSONS_PATH", str(tmp_path / "lessons.jsonl"))
    monkeypatch.delenv("REFLECT_REPO_MOUNT", raising=False)
    reload_settings()
    yield
    reload_settings()


class _StaticLLM:
    """Returns one final assistant message with no tool calls.

    Implemented as a Runnable so ``create_react_agent`` accepts it.
    """

    def __init__(self, text: str = "no further action") -> None:
        self.text = text
        self._runnable = RunnableLambda(lambda _msgs: AIMessage(content=self.text))

    def bind_tools(self, _tools: Iterable[Any]):
        return self._runnable

    def invoke(self, messages: Any, **_: Any) -> AIMessage:
        return self._runnable.invoke(messages)


class _FakeSandbox:
    def exec(self, *_a: Any, **_kw: Any):
        raise AssertionError("sandbox not used in smoke tests")


def test_reflect_agent_is_registered():
    agent = get_agent("reflect")
    assert agent is not None
    assert agent.name == "reflect"


def test_reflect_agent_handles_missing_repo_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No /repo mount available => emit error + done(submitted=False), no crash."""
    monkeypatch.setenv("REFLECT_REPO_MOUNT", str(tmp_path / "does-not-exist"))
    reload_settings()

    agent = ReflectAgent(
        llm=_StaticLLM(),
        sandbox=_FakeSandbox(),
        lesson_store=LessonStore.from_settings(),
    )
    workspace = Workspace.create("reflect-smoke")

    events = list(agent.stream("audit yourself", workspace=workspace))
    kinds = [e.kind for e in events]

    assert "workspace_created" in kinds
    assert "clone_started" in kinds
    assert "error" in kinds
    assert kinds[-1] == "done"
    assert events[-1].data["submitted"] is False


def test_reflect_agent_runs_with_fake_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end: fake source mount + static LLM => completes with done(False)."""
    source = tmp_path / "src-mount"
    (source / "src" / "genai").mkdir(parents=True)
    (source / "src" / "genai" / "__init__.py").write_text("__all__ = []\n")
    (source / "tests").mkdir()
    monkeypatch.setenv("REFLECT_REPO_MOUNT", str(source))
    reload_settings()

    agent = ReflectAgent(
        llm=_StaticLLM("done thinking"),
        sandbox=_FakeSandbox(),
        lesson_store=LessonStore.from_settings(),
    )
    workspace = Workspace.create("reflect-smoke-2")

    events = list(agent.stream("look around", workspace=workspace))
    kinds = [e.kind for e in events]

    assert "clone_ready" in kinds
    assert kinds[-1] == "done"
    assert events[-1].data["submitted"] is False
    # Clone directory exists.
    assert (workspace.root / "repo" / "src" / "genai" / "__init__.py").exists()


class _LoopingLLM:
    """Calls ``query_lessons`` forever, so only the step budget can stop it."""

    def __init__(self) -> None:
        self._ids = iter(range(10_000))

    def bind_tools(self, _tools: Iterable[Any]):
        return RunnableLambda(
            lambda _msgs: AIMessage(
                content="",
                tool_calls=[{"name": "query_lessons", "args": {"query": "x"}, "id": f"c{next(self._ids)}"}],
            )
        )


def test_reflect_agent_reports_budget_exhaustion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "src-mount"
    (source / "src").mkdir(parents=True)
    monkeypatch.setenv("REFLECT_REPO_MOUNT", str(source))
    reload_settings()

    seen: list = []
    agent = ReflectAgent(
        llm=_LoopingLLM(),
        sandbox=_FakeSandbox(),
        lesson_store=LessonStore.from_settings(),
    )
    events = list(
        agent.stream("loop", workspace=Workspace.create("reflect-budget"), max_iterations=5, on_event=seen.append)
    )

    kinds = [e.kind for e in events]
    assert kinds[-2:] == ["budget_exhausted", "done"]
    assert "error" not in kinds
    assert events[-1].data["submitted"] is False
    assert [e.kind for e in seen] == kinds
