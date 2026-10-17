"""Phase 6 tests: lessons injection into agents and autoseed from eval runs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from genai.core.config import reload_settings
from genai.reflection.store import Lesson, LessonStore


@pytest.fixture(autouse=True)
def _fresh_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("REFLECT_LESSONS_PATH", str(tmp_path / "lessons.jsonl"))
    monkeypatch.delenv("REFLECT_LESSONS_ENABLED", raising=False)
    monkeypatch.delenv("REFLECT_LESSONS_AUTOSEED", raising=False)
    reload_settings()
    yield
    reload_settings()


# ----------------------------------------------------------------- injection


def test_codegen_injection_noop_when_disabled(monkeypatch):
    """With lessons disabled (default), the codegen prompt is unchanged."""
    from genai.agents.codegen.agent import CodegenAgent
    from genai.agents.codegen.prompts import SYSTEM_PROMPT

    store = LessonStore.from_settings()
    store.append(Lesson(category="codegen", problem="x", fix="y"))

    class _LLM:
        def bind_tools(self, _):
            return self

    class _SB:
        pass

    agent = CodegenAgent(llm=_LLM(), sandbox=_SB())  # type: ignore[arg-type]
    from genai.sandbox.workspace import Workspace

    ws = Workspace.create("noop")
    inputs = agent._build_inputs("hello", ws)
    system = inputs["messages"][0].content
    assert system == SYSTEM_PROMPT


def test_codegen_injection_active_when_enabled(monkeypatch):
    monkeypatch.setenv("REFLECT_LESSONS_ENABLED", "true")
    reload_settings()

    from genai.agents.codegen.agent import CodegenAgent
    from genai.agents.codegen.prompts import SYSTEM_PROMPT

    store = LessonStore.from_settings()
    store.append(
        Lesson(
            category="codegen",
            problem="forgets requirements.txt",
            fix="always write requirements.txt before pip install",
            tags=["codegen"],
        )
    )

    class _LLM:
        def bind_tools(self, _):
            return self

    class _SB:
        pass

    agent = CodegenAgent(llm=_LLM(), sandbox=_SB())  # type: ignore[arg-type]
    from genai.sandbox.workspace import Workspace

    ws = Workspace.create("inject")
    inputs = agent._build_inputs("requirements stuff", ws)
    system = inputs["messages"][0].content
    assert system != SYSTEM_PROMPT
    assert "<lessons" in system
    assert "requirements.txt" in system


def test_planner_injection_active_when_enabled(monkeypatch):
    monkeypatch.setenv("REFLECT_LESSONS_ENABLED", "true")
    reload_settings()

    from genai.agents.fullstack import planner as planner_mod
    from genai.agents.fullstack.spec import ProjectSpec

    store = LessonStore.from_settings()
    store.append(
        Lesson(
            category="fullstack-plan",
            problem="forgets to add a database table",
            fix="always include a 'tables' field in the spec",
            tags=["fullstack-plan"],
        )
    )

    captured: dict = {}

    class _StopHere(Exception):
        pass

    class _LLM:
        def invoke(self, messages, **_):
            captured["messages"] = messages
            raise _StopHere

    with pytest.raises(_StopHere):
        planner_mod.plan(_LLM(), "build a thing with tables")
    assert "messages" in captured
    system = captured["messages"][0].content
    assert "<lessons" in system
    assert "tables" in system
    # ProjectSpec import is exercised to ensure module loads.
    assert ProjectSpec is not None


def test_backend_specialist_injects_lessons_when_enabled(monkeypatch):
    monkeypatch.setenv("REFLECT_LESSONS_ENABLED", "true")
    reload_settings()

    from genai.agents.fullstack import backend as backend_mod
    from genai.agents.fullstack.spec import Entity, ProjectMeta, ProjectSpec

    store = LessonStore.from_settings()
    store.append(
        Lesson(
            category="fullstack-backend",
            problem="forgets email-validator",
            fix="add pydantic[email] to backend pyproject.toml",
            tags=["fullstack", "backend", "pydantic"],
        )
    )

    captured: dict = {}

    class _ReactAgent:
        def stream(self, inputs, **_):
            captured["inputs"] = inputs
            return iter(())

    monkeypatch.setattr(backend_mod, "create_react_agent", lambda *_a, **_k: _ReactAgent())

    class _Sandbox:
        def exec(self, *_a, **_k):
            raise AssertionError("should not be called")

    class _Workspace:
        project_id = "test"
        root = Path("/tmp/test")

        def list_tree(self):
            return []

    spec = ProjectSpec(meta=ProjectMeta(name="x", title="X"), entities=[Entity(name="Note", fields=[])])
    backend_mod.run_backend(_LLM(), _Workspace(), _Sandbox(), spec, max_iterations=1)
    system = captured["inputs"]["messages"][0].content
    assert "<lessons" in system
    assert "email-validator" in system


def test_frontend_specialist_injects_lessons_when_enabled(monkeypatch):
    monkeypatch.setenv("REFLECT_LESSONS_ENABLED", "true")
    reload_settings()

    from genai.agents.fullstack import frontend as frontend_mod
    from genai.agents.fullstack.spec import Page, ProjectMeta, ProjectSpec

    store = LessonStore.from_settings()
    store.append(
        Lesson(
            category="fullstack-frontend",
            problem="uses react-query v4 isLoading on useMutation",
            fix="React Query v5 uses isPending on useMutation results",
            tags=["fullstack", "frontend", "react-query"],
        )
    )

    captured: dict = {}

    class _ReactAgent:
        def stream(self, inputs, **_):
            captured["inputs"] = inputs
            return iter(())

    monkeypatch.setattr(frontend_mod, "create_react_agent", lambda *_a, **_k: _ReactAgent())

    class _Sandbox:
        def exec(self, *_a, **_k):
            raise AssertionError("should not be called")

    class _Workspace:
        project_id = "test"
        root = Path("/tmp/test")

        def list_tree(self):
            return []

    spec = ProjectSpec(
        meta=ProjectMeta(name="x", title="X"),
        pages=[Page(name="NotesPage", route="/notes")],
    )
    frontend_mod.run_frontend(_LLM(), _Workspace(), _Sandbox(), spec, max_iterations=1)
    system = captured["inputs"]["messages"][0].content
    assert "<lessons" in system
    assert "isPending" in system


class _LLM:
    """Minimal LLM stub for tests that only inspect the prompt."""

    pass


# ----------------------------------------------------------------- autoseed


def test_autoseed_writes_lessons_after_run(monkeypatch, tmp_path):
    """When REFLECT_LESSONS_AUTOSEED=true, run_suite seeds lessons from failures."""
    monkeypatch.setenv("REFLECT_LESSONS_AUTOSEED", "true")
    reload_settings()

    from genai.evals import harness as harness_mod
    from genai.evals.cases import EvalCase, EvalSuite
    from genai.evals.harness import CaseResult, run_suite

    # Bypass the real run_case: have it record one passing and one failing case.
    def fake_run_case(case, *, run_dir, sandbox, llm, jsonl_writer, **_):
        passed = case.id == "ok"
        result = CaseResult(
            case_id=case.id,
            agent=case.agent,
            passed=passed,
            skipped=False,
            summary="ok" if passed else "boom",
            structural={"expected_files_present": [], "expected_files_missing": []},
            functional={
                "checks": (
                    []
                    if passed
                    else [
                        {
                            "name": "tests",
                            "kind": "shell",
                            "passed": False,
                            "exit_code": 1,
                            "output_tail": "AssertionError: nope",
                            "wall_seconds": 0.1,
                        }
                    ]
                )
            },
            telemetry={"tool_calls": 1},
            submitted=passed,
            error=None if passed else "boom",
            wall_seconds=0.5,
            project_id=case.id,
            skip_reason=None,
        )
        jsonl_writer.write({"type": "case_summary", **result.to_dict()})
        return result

    monkeypatch.setattr(harness_mod, "run_case", fake_run_case)
    monkeypatch.setattr(harness_mod, "DockerExecSandbox", _DummySandboxFactory)
    monkeypatch.setattr(harness_mod, "load_llm", lambda: object())

    suite = EvalSuite(
        name="autoseed-test",
        cases=[
            EvalCase(id="ok", agent="codegen", prompt="x", expect_files=[], functional=[]),
            EvalCase(id="fail", agent="codegen", prompt="y", expect_files=[], functional=[]),
        ],
    )
    base = tmp_path / "evals"
    run_suite(suite, base_dir=base)

    lessons_path = Path(harness_mod.get_settings().reflect.lessons_path)
    assert lessons_path.exists()
    lines = [ln for ln in lessons_path.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 1
    parsed = [json.loads(ln) for ln in lines]
    # The fail case should have produced at least one lesson tagged with its id.
    assert any("fail" in (p.get("tags") or []) for p in parsed)


class _DummySandboxFactory:
    @classmethod
    def from_settings(cls):
        return _DummySandboxFactory()

    def exec(self, *_a, **_kw):  # pragma: no cover - not invoked
        raise AssertionError("sandbox should not be invoked in autoseed test")


# Avoid a flaky run_id collision when tests execute back-to-back.
_ = time
