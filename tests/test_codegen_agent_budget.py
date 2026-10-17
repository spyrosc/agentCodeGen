"""CodegenAgent driven through a real ReAct graph by a scripted fake LLM.

No provider and no sandbox: the script only calls FS tools and
``submit_project``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from genai.agents.codegen.agent import CodegenAgent
from genai.core.config import reload_settings
from genai.sandbox.workspace import Workspace, new_project_id


@pytest.fixture(autouse=True)
def _fresh_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REFLECT_LESSONS_ENABLED", raising=False)
    monkeypatch.setenv("REFLECT_LESSONS_PATH", str(tmp_path / "lessons.jsonl"))
    reload_settings()
    yield
    reload_settings()


class ScriptedLLM:
    """Plays one turn per call: ``(tool_name, args)`` is a tool call, a string is
    a final answer. The last turn repeats forever."""

    def __init__(self, *turns):
        self.turns = turns
        self.calls = 0
        self._ids = itertools.count()

    def bind_tools(self, _tools):
        return RunnableLambda(self._respond)

    def _respond(self, _messages):
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        if isinstance(turn, str):
            return AIMessage(content=turn)
        name, args = turn
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{next(self._ids)}"}],
        )


def _workspace(tmp_path: Path) -> Workspace:
    return Workspace.create(new_project_id("t"), base=tmp_path)


# langgraph ends even budgets with a "need more steps" message and raises
# GraphRecursionError on odd ones; both must end in a clean `done`.
@pytest.mark.parametrize("limit", [4, 5])
def test_budget_exhaustion_ends_with_done_instead_of_crashing(tmp_path, limit):
    agent = CodegenAgent(llm=ScriptedLLM(("list_files", {"path": "."})), sandbox=object())

    events = list(agent.stream("loop forever", workspace=_workspace(tmp_path), max_iterations=limit))

    kinds = [e.kind for e in events]
    assert kinds[0] == "workspace_created"
    assert kinds[-2:] == ["budget_exhausted", "done"]
    assert events[-2].data == {"limit": limit}
    assert events[-1].data["submitted"] is False
    assert f"budget exhausted after {limit} steps" in events[-1].data["summary"]


def test_submit_reports_summary_and_orders_tool_events(tmp_path):
    llm = ScriptedLLM(("list_files", {"path": "."}), ("submit_project", {"summary": "all good"}))
    agent = CodegenAgent(llm=llm, sandbox=object())
    ws = _workspace(tmp_path)

    events = list(agent.stream("do it", workspace=ws, max_iterations=20))

    kinds = [e.kind for e in events]
    assert kinds.index("list_files") < kinds.index("tool_result")
    assert "budget_exhausted" not in kinds
    assert kinds[-1] == "done"
    assert events[-1].data == {"submitted": True, "summary": "all good", "project_id": ws.project_id}
    assert llm.calls == 2


def test_on_event_sees_every_yielded_event(tmp_path):
    seen = []
    agent = CodegenAgent(
        llm=ScriptedLLM(("write_file", {"path": "a.py", "content": "x = 1\n"}),
                        ("submit_project", {"summary": "done"})),
        sandbox=object(),
    )

    events = list(agent.stream("t", workspace=_workspace(tmp_path), max_iterations=20, on_event=seen.append))

    assert [e.kind for e in seen] == [e.kind for e in events]
    assert "write_file" in [e.kind for e in events]


def test_run_returns_unsubmitted_result_when_budget_runs_out(tmp_path):
    agent = CodegenAgent(llm=ScriptedLLM(("list_files", {"path": "."})), sandbox=object())

    result = agent.run("loop", workspace=_workspace(tmp_path), max_iterations=5)

    assert result.submitted is False
    assert "budget exhausted" in result.summary
