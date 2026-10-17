"""End-to-end harness test using a fake agent and fake sandbox.

Honours the repo-wide invariant: tests must not require Docker or a live LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from genai.agents.base import AgentEvent, AgentResult
from genai.agents.registry import _REGISTRY
from genai.evals import run_suite
from genai.evals.cases import EvalCase, EvalSuite, FunctionalCheck
from genai.evals.report import render_markdown, render_table, write_markdown
from genai.sandbox import ExecResult

# ----------------------------------------------------------------- fakes


@dataclass
class FakeSandbox:
    log: list[str] = field(default_factory=list)

    def exec(self, project_id: str, command: str, timeout_s: int) -> ExecResult:  # noqa: ARG002
        self.log.append(command)
        # Make pytest "pass" and any other check pass cleanly.
        return ExecResult(exit_code=0, output="ok")


@dataclass
class FakeLLM:
    """Just needs to satisfy ``hasattr(llm, 'with_config')``."""

    def with_config(self, _cfg: dict[str, Any]) -> FakeLLM:
        return self


@dataclass
class FakeAgent:
    name: str = "fake"
    description: str = "fake agent for tests"
    llm: Any = None
    sandbox: Any = None

    def stream(
        self,
        task: str,  # noqa: ARG002
        *,
        workspace: Any = None,
        max_iterations: Any = None,  # noqa: ARG002
        on_event: Any = None,
    ):
        yield AgentEvent("workspace_created", {"project_id": workspace.project_id})
        # Drop a couple of files so structural scoring has something to look at.
        (workspace.root / "pyproject.toml").write_text("[project]\nname='x'\n")
        (workspace.root / "main.py").write_text("def main():\n    return 1\n")
        if on_event:
            on_event(AgentEvent("write_file", {"path": "main.py"}))
        yield AgentEvent("done", {"submitted": True, "summary": "ok",
                                  "project_id": workspace.project_id})

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        result = AgentResult(submitted=False)
        for ev in self.stream(task, **kwargs):
            if ev.kind == "done":
                result = AgentResult(
                    submitted=True, summary=ev.data["summary"],
                    project_id=ev.data["project_id"],
                )
        return result


@pytest.fixture
def register_fake_agent():
    prev = dict(_REGISTRY)
    _REGISTRY["fake"] = FakeAgent
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(prev)


# ---------------------------------------------------------------- tests


def test_run_suite_writes_artifacts(tmp_path: Path, register_fake_agent) -> None:
    suite = EvalSuite(
        name="t",
        cases=[
            EvalCase(
                id="case_a",
                agent="fake",
                prompt="hello",
                expect_files=["pyproject.toml", "main.py"],
                functional=[FunctionalCheck(name="t", type="pytest")],
            ),
            EvalCase(id="case_skipped", agent="fake", prompt="x", skip="not today"),
        ],
    )

    base = tmp_path / "_evals"
    result = run_suite(suite, base_dir=base, sandbox=FakeSandbox(), llm=FakeLLM())

    assert result.passed_count == 1
    assert result.skipped_count == 1
    assert result.failed_count == 0

    run_dir = base / result.run_id
    jsonl = run_dir / "run.jsonl"
    assert jsonl.exists()
    lines = [json.loads(line) for line in jsonl.read_text().splitlines() if line]
    types = [obj["type"] for obj in lines]
    assert types[0] == "suite_started"
    assert types[-1] == "suite_summary"
    assert "case_summary" in types
    assert any(obj["type"] == "event" and obj["kind"] == "write_file" for obj in lines)

    # Markdown rendering.
    md_path = write_markdown(run_dir, result)
    md = md_path.read_text()
    assert "case_a" in md
    assert "case_skipped" in md
    assert "Eval run" in md

    # Plain table rendering for stdout.
    table = render_table(result)
    assert "case_a" in table and "pass" in table


def test_run_suite_records_failure(tmp_path: Path, register_fake_agent) -> None:
    suite = EvalSuite(
        name="t",
        cases=[
            EvalCase(
                id="missing_file",
                agent="fake",
                prompt="hello",
                expect_files=["does_not_exist.py"],
            ),
        ],
    )
    result = run_suite(
        suite,
        base_dir=tmp_path / "_evals",
        sandbox=FakeSandbox(),
        llm=FakeLLM(),
        skip_functional=True,
    )
    assert result.passed_count == 0
    assert result.failed_count == 1
    case = result.cases[0]
    assert case.passed is False
    assert "does_not_exist.py" in case.structural["expected_files_missing"]

    md = render_markdown(result)
    assert "FAIL" in md


def test_run_suite_only_filter(tmp_path: Path, register_fake_agent) -> None:
    suite = EvalSuite(
        name="t",
        cases=[
            EvalCase(id="a", agent="fake", prompt="x", expect_files=["main.py"]),
            EvalCase(id="b", agent="fake", prompt="x", expect_files=["main.py"]),
        ],
    )
    result = run_suite(
        suite,
        base_dir=tmp_path / "_evals",
        sandbox=FakeSandbox(),
        llm=FakeLLM(),
        only=["b"],
        skip_functional=True,
    )
    assert [c.case_id for c in result.cases] == ["b"]
