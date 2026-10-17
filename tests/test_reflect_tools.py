"""Tests for the reflect agent's tool surface.

No real LLM, no real Docker \u2014 we use a fake sandbox and patch in a fake
clone. Eval-gate tests use a synthetic ``run.jsonl`` baseline and the real
:func:`evals.run_suite` driven by the same fake-agent fixture as
``tests/test_evals_harness.py`` (registered via the registry).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from genai.agents.base import AgentEvent, AgentResult
from genai.agents.reflect.repo import ClonedRepo, prepare_clone
from genai.agents.reflect.tools import (
    OperationCap,
    _check_write_path,
    _diff_stat,
    _slugify,
    build_reflect_tools,
)
from genai.agents.registry import _REGISTRY
from genai.core.config import get_settings, reload_settings
from genai.reflection.store import LessonStore
from genai.sandbox import ExecResult
from genai.sandbox.workspace import Workspace

# ---------------------------------------------------------------- fakes


@dataclass
class FakeSandbox:
    """Records each command and returns the configured exit_code/output."""

    log: list[str] = field(default_factory=list)
    exit_code: int = 0
    output: str = "ok"
    handler: Any = None  # optional callable(cmd) -> ExecResult

    def exec(self, project_id: str, command: str, timeout_s: int) -> ExecResult:  # noqa: ARG002
        self.log.append(command)
        if self.handler:
            return self.handler(command)
        return ExecResult(exit_code=self.exit_code, output=self.output)


@dataclass
class FakeLLM:
    def with_config(self, _cfg: dict[str, Any]) -> FakeLLM:
        return self


def _have_git() -> bool:
    return shutil.which("git") is not None


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True)


# ---------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    monkeypatch.delenv("REFLECT_LESSONS_ENABLED", raising=False)
    monkeypatch.delenv("REFLECT_EVAL_GATE_REQUIRED", raising=False)
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace.create("reflect-tools-test", base=tmp_path / "ws")


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    src.mkdir()
    (src / "src").mkdir()
    (src / "src" / "module.py").write_text("VALUE = 1\n")
    (src / "tests").mkdir()
    (src / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    (src / "pyproject.toml").write_text("[project]\nname='p'\n")
    return src


@pytest.fixture
def git_source_repo(source_repo: Path) -> Path:
    if not _have_git():
        pytest.skip("git not available")
    _git(["init", "-q", "-b", "main"], source_repo)
    _git(["config", "user.email", "t@e.x"], source_repo)
    _git(["config", "user.name", "t"], source_repo)
    _git(["add", "-A"], source_repo)
    _git(["commit", "-q", "-m", "init"], source_repo)
    return source_repo


@pytest.fixture
def clone(workspace: Workspace, source_repo: Path) -> ClonedRepo:
    return prepare_clone(workspace, repo_mount=source_repo)


@pytest.fixture
def store(tmp_path: Path) -> LessonStore:
    return LessonStore(tmp_path / "lessons.jsonl")


def _build(workspace, clone, sandbox=None, store=None, events=None):
    sandbox = sandbox or FakeSandbox()
    store = store or LessonStore(workspace.root / "lessons.jsonl")
    events = events if events is not None else []
    tools, state = build_reflect_tools(
        workspace,
        clone,
        sandbox,
        lesson_store=store,
        llm=FakeLLM(),
        on_event=lambda k, p: events.append((k, p)),
    )
    return {t.name: t for t in tools}, state, sandbox, events


# ============================================================ unit helpers


def test_check_write_path_denylist():
    cfg = get_settings().reflect
    with pytest.raises(PermissionError):
        _check_write_path("pyproject.toml", cfg)
    with pytest.raises(PermissionError):
        _check_write_path("docker/agent.Dockerfile", cfg)
    with pytest.raises(PermissionError):
        _check_write_path(".github/workflows/ci.yml", cfg)


def test_check_write_path_allowlist_accepts_src_and_evals():
    cfg = get_settings().reflect
    _check_write_path("src/genai/core/logging.py", cfg)
    _check_write_path("evals/suites/new.yaml", cfg)
    _check_write_path("tests/test_new.py", cfg)


def test_check_write_path_rejects_outside_allowlist():
    cfg = get_settings().reflect
    with pytest.raises(PermissionError):
        _check_write_path("README.md", cfg)
    with pytest.raises(PermissionError):
        _check_write_path("notebooks/x.ipynb", cfg)


def test_check_write_path_rejects_traversal():
    cfg = get_settings().reflect
    with pytest.raises(PermissionError):
        _check_write_path("src/../etc/passwd", cfg)


def test_diff_stat_counts_correctly():
    diff = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/src/y.py b/src/y.py\n"
        "--- a/src/y.py\n"
        "+++ b/src/y.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+brand new\n"
    )
    files, adds, dels = _diff_stat(diff)
    assert (files, adds, dels) == (2, 2, 1)


def test_slugify():
    assert _slugify("Improve get_settings docstring") == "improve-get-settings-docstring"
    assert _slugify("!!!") == "patch"


# ============================================================ tools


def test_write_file_enforces_denylist(workspace, clone, store):
    tools, state, _sb, events = _build(workspace, clone, store=store)
    res = tools["write_file"].invoke({"path": "pyproject.toml", "content": "x"})
    assert res.startswith("PERMISSION_DENIED")
    assert state["writes"] == 0
    assert state["denied_attempts"] == ["pyproject.toml"]
    kinds = [k for k, _ in events]
    assert "write_denied" in kinds


def test_write_file_allows_src(workspace, clone, store):
    tools, state, _sb, _ev = _build(workspace, clone, store=store)
    res = tools["write_file"].invoke({"path": "src/genai/core/logging.py", "content": "X = 1\n"})
    assert "wrote" in res or res.startswith("PERMISSION_DENIED") is False
    assert state["writes"] == 1
    target = clone.clone_path / "src" / "genai" / "core" / "logging.py"
    # Note: source_repo only has src/module.py; ScopedWorkspace creates parents.
    assert target.exists()


def test_write_file_invalidates_cached_test_verdict(workspace, clone, store):
    tools, state, _sb, _ev = _build(workspace, clone, store=store)
    state["tests_pass"] = True
    state["lint_pass"] = True
    state["eval_gate"] = {"baseline_run_id": "x"}
    tools["write_file"].invoke({"path": "src/x.py", "content": "1"})
    assert state["tests_pass"] is None
    assert state["lint_pass"] is None
    assert state["eval_gate"] is None


def test_write_cap_enforced(workspace, clone, store, monkeypatch):
    monkeypatch.setenv("REFLECT_MAX_WRITES", "2")
    reload_settings()
    tools, _state, _sb, _ev = _build(workspace, clone, store=store)
    tools["write_file"].invoke({"path": "src/a.py", "content": "1"})
    tools["write_file"].invoke({"path": "src/b.py", "content": "2"})
    with pytest.raises(OperationCap):
        tools["write_file"].invoke({"path": "src/c.py", "content": "3"})


def test_write_file_traversal_blocked(workspace, clone, store):
    tools, state, _sb, _ev = _build(workspace, clone, store=store)
    res = tools["write_file"].invoke({"path": "../../etc/passwd", "content": "x"})
    assert res.startswith("PERMISSION_DENIED")
    assert state["writes"] == 0


def test_run_tests_repo_records_verdict(workspace, clone, store):
    sb = FakeSandbox(exit_code=0, output="passed")
    tools, state, _sb, _ev = _build(workspace, clone, sandbox=sb, store=store)
    out = tools["run_tests_repo"].invoke({})
    assert "exit=0" in out
    assert state["tests_pass"] is True
    assert "cd repo" in sb.log[0]
    assert "pytest" in sb.log[0]


def test_run_tests_repo_records_failure(workspace, clone, store):
    sb = FakeSandbox(exit_code=1, output="FAILED")
    tools, state, _sb, _ev = _build(workspace, clone, sandbox=sb, store=store)
    tools["run_tests_repo"].invoke({})
    assert state["tests_pass"] is False


def test_run_lint_repo(workspace, clone, store):
    sb = FakeSandbox(exit_code=0, output="All checks passed!")
    tools, state, _sb, _ev = _build(workspace, clone, sandbox=sb, store=store)
    tools["run_lint_repo"].invoke({})
    assert state["lint_pass"] is True
    assert "ruff check" in sb.log[0]


def test_query_and_record_lessons(workspace, clone, store):
    tools, _state, _sb, events = _build(workspace, clone, store=store)
    out = tools["record_lesson"].invoke(
        {"category": "codegen", "problem": "missing tests", "fix": "add pytest", "tags": "missing,tests"}
    )
    assert "recorded" in out
    assert any(k == "lesson_recorded" for k, _ in events)

    res = tools["query_lessons"].invoke({"query": "missing", "category": "codegen"})
    assert "missing tests" in res


# ============================================================ propose_patch


def test_propose_patch_refuses_without_tests(workspace, clone, store):
    tools, _state, _sb, _ev = _build(workspace, clone, store=store)
    res = tools["propose_patch"].invoke(
        {"summary": "x", "motivation": "y", "baseline_run_id": ""}
    )
    assert res.startswith("PATCH_REJECTED")
    assert "run_tests_repo" in res


def test_propose_patch_refuses_when_lint_fails(workspace, clone, store):
    sb = FakeSandbox(exit_code=0)
    tools, state, _sb, _ev = _build(workspace, clone, sandbox=sb, store=store)
    state["tests_pass"] = True
    state["lint_pass"] = False
    res = tools["propose_patch"].invoke({"summary": "x", "motivation": "y"})
    assert res.startswith("PATCH_REJECTED")
    assert "run_lint_repo" in res


def test_propose_patch_refuses_clean_tree(workspace: Workspace, git_source_repo: Path, store):
    """With a git clone and no writes, the diff is empty → reject."""
    if not _have_git():
        pytest.skip("git not available")
    clone = prepare_clone(workspace, repo_mount=git_source_repo)
    tools, state, _sb, _ev = _build(workspace, clone, store=store)
    state["tests_pass"] = True
    state["lint_pass"] = True
    res = tools["propose_patch"].invoke({"summary": "x", "motivation": "y"})
    assert res.startswith("PATCH_REJECTED")
    assert "clean" in res


def test_propose_patch_refuses_when_gate_required_and_missing(
    workspace, clone, store, monkeypatch
):
    monkeypatch.setenv("REFLECT_EVAL_GATE_REQUIRED", "true")
    reload_settings()
    tools, state, _sb, _ev = _build(workspace, clone, store=store)
    state["tests_pass"] = True
    state["lint_pass"] = True
    res = tools["propose_patch"].invoke({"summary": "x", "motivation": "y"})
    assert res.startswith("PATCH_REJECTED")
    assert "run_eval_gate" in res


def test_propose_patch_refuses_when_gate_has_regressions(
    workspace, clone, store, monkeypatch
):
    monkeypatch.setenv("REFLECT_EVAL_GATE_REQUIRED", "true")
    reload_settings()
    tools, state, _sb, _ev = _build(workspace, clone, store=store)
    state["tests_pass"] = True
    state["lint_pass"] = True
    state["eval_gate"] = {"regressions": ["case_a"], "suite": "x"}
    res = tools["propose_patch"].invoke({"summary": "x", "motivation": "y"})
    assert res.startswith("PATCH_REJECTED")
    assert "case_a" in res


def test_propose_patch_succeeds_on_git_clone(
    workspace: Workspace, git_source_repo: Path, store
):
    if not _have_git():
        pytest.skip("git not available")
    clone = prepare_clone(workspace, repo_mount=git_source_repo)
    tools, state, _sb, events = _build(workspace, clone, store=store)
    # Real allowlisted edit.
    tools["write_file"].invoke({"path": "src/module.py", "content": "VALUE = 2\n"})
    state["tests_pass"] = True
    state["lint_pass"] = True
    res = tools["propose_patch"].invoke(
        {"summary": "Bump VALUE", "motivation": "trivial change", "baseline_run_id": ""}
    )
    assert res == "PATCH_PROPOSED"
    assert state["submitted"] is True
    assert state["patch"] is not None
    assert state["patch"]["files_changed"] == 1

    # Branch was actually created.
    branches = subprocess.run(
        ["git", "branch", "--list"],
        cwd=str(clone.clone_path), capture_output=True, text=True, check=True,
    ).stdout
    assert "reflect/bump-value-" in branches

    # patch.diff was written to the workspace.
    patch_path = workspace.root / "patch.diff"
    assert patch_path.exists()
    assert "VALUE = 2" in patch_path.read_text()

    # Event was emitted.
    kinds = [k for k, _ in events]
    assert "patch_proposed" in kinds


def test_propose_patch_rejects_oversized_patch(
    workspace: Workspace, git_source_repo: Path, store, monkeypatch
):
    if not _have_git():
        pytest.skip("git not available")
    monkeypatch.setenv("REFLECT_MAX_PATCH_LINES", "5")
    reload_settings()
    clone = prepare_clone(workspace, repo_mount=git_source_repo)
    tools, state, _sb, _ev = _build(workspace, clone, store=store)
    big = "\n".join(f"line_{i} = {i}" for i in range(50)) + "\n"
    tools["write_file"].invoke({"path": "src/module.py", "content": big})
    state["tests_pass"] = True
    state["lint_pass"] = True
    res = tools["propose_patch"].invoke({"summary": "huge", "motivation": "n/a"})
    assert res.startswith("PATCH_REJECTED")
    assert "lines" in res


# ============================================================ eval-harness inputs


def _write_run_jsonl(run_dir: Path, *, run_id: str, suite_name: str, cases: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "type": "suite_summary",
        "run_id": run_id,
        "suite_name": suite_name,
        "model": "x",
        "passed": sum(1 for c in cases if c.get("passed")),
        "failed": sum(1 for c in cases if not c.get("passed") and not c.get("skipped")),
        "skipped": sum(1 for c in cases if c.get("skipped")),
        "cases": cases,
    }
    lines = [
        {"type": "suite_started", "run_id": run_id, "suite_name": suite_name, "model": "x", "started_at": "now"},
    ]
    for c in cases:
        lines.append({"type": "event", "case_id": c["case_id"], "kind": "write_file", "data": {"path": "main.py"}})
        lines.append({"type": "case_summary", **c})
    lines.append(summary)
    (run_dir / "run.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def test_list_eval_runs_and_read_failed_case(
    workspace, clone, store, tmp_path, monkeypatch
):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "ws_root"))
    reload_settings()
    evals_root = tmp_path / "ws_root" / "_evals"
    case_a = {"case_id": "a", "agent": "codegen", "passed": True, "skipped": False, "summary": "ok",
              "structural": {"expected_files_present": ["main.py"], "expected_files_missing": []},
              "functional": {"checks": []}, "telemetry": {"llm_calls": 1}, "submitted": True,
              "error": None, "wall_seconds": 1.0, "project_id": "a", "skip_reason": None}
    case_b = {"case_id": "b", "agent": "codegen", "passed": False, "skipped": False, "summary": "fail",
              "structural": {"expected_files_present": [], "expected_files_missing": ["x.py"]},
              "functional": {"checks": []}, "telemetry": {"llm_calls": 2}, "submitted": True,
              "error": None, "wall_seconds": 1.0, "project_id": "b", "skip_reason": None}
    _write_run_jsonl(evals_root / "20260501-aaa", run_id="20260501-aaa", suite_name="t",
                     cases=[case_a, case_b])
    _write_run_jsonl(evals_root / "20260501-bbb", run_id="20260501-bbb", suite_name="t",
                     cases=[case_a])

    tools, _state, _sb, _ev = _build(workspace, clone, store=store)
    listing = tools["list_eval_runs"].invoke({"limit": 10})
    assert "20260501-aaa" in listing and "20260501-bbb" in listing

    summary = tools["read_eval_run"].invoke({"run_id": "20260501-aaa"})
    parsed = json.loads(summary)
    assert parsed["passed"] == 1
    assert parsed["failed"] == 1
    assert set(parsed["case_ids"]) == {"a", "b"}

    failed = tools["read_failed_case"].invoke({"run_id": "20260501-aaa", "case_id": "b"})
    parsed = json.loads(failed)
    assert parsed["case_summary"]["case_id"] == "b"
    assert parsed["last_events"], "should include at least one event"


# ============================================================ eval gate (integration-lite)


@dataclass
class _GateAgent:
    """Minimal agent for the gate's run_suite to drive."""

    name: str = "gate-fake"
    description: str = "x"
    llm: Any = None
    sandbox: Any = None
    behaviour: dict = field(default_factory=dict)  # case_id -> {pass: bool}

    def stream(self, task, *, workspace, max_iterations=None, on_event=None):  # noqa: ARG002
        yield AgentEvent("workspace_created", {"project_id": workspace.project_id})
        beh = self.behaviour.get(workspace.project_id, {"pass": True})
        if beh.get("pass"):
            (workspace.root / "main.py").write_text("ok\n")
        yield AgentEvent("done", {"submitted": True, "summary": "ok",
                                  "project_id": workspace.project_id})

    def run(self, task, **kwargs):
        for ev in self.stream(task, **kwargs):
            if ev.kind == "done":
                return AgentResult(submitted=True, summary="ok",
                                   project_id=ev.data["project_id"])
        return AgentResult(submitted=False)


@pytest.fixture
def register_gate_agent():
    prev = dict(_REGISTRY)
    behaviour: dict[str, dict] = {}

    def factory(**kwargs):
        a = _GateAgent(**kwargs)
        a.behaviour = behaviour
        return a

    _REGISTRY["gate-fake"] = factory
    try:
        yield behaviour
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(prev)


def _write_gate_suite(clone_path: Path) -> Path:
    suite_dir = clone_path / "evals" / "suites"
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite = suite_dir / "gate.yaml"
    suite.write_text(
        "name: gate\n"
        "cases:\n"
        "  - id: case_a\n"
        "    agent: gate-fake\n"
        "    prompt: x\n"
        "    expect_files: ['main.py']\n"
        "  - id: case_b\n"
        "    agent: gate-fake\n"
        "    prompt: x\n"
        "    expect_files: ['main.py']\n",
        encoding="utf-8",
    )
    return suite


def test_run_eval_gate_detects_regression(
    workspace, clone, store, register_gate_agent, monkeypatch, tmp_path
):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "ws_root"))
    suite_path = _write_gate_suite(clone.clone_path)
    monkeypatch.setenv("REFLECT_EVAL_GATE_SUITE", "evals/suites/gate.yaml")
    reload_settings()

    behaviour = register_gate_agent
    behaviour["case_a"] = {"pass": True}
    behaviour["case_b"] = {"pass": True}

    sandbox_root = Path(get_settings().sandbox.workspace_root)
    baseline_dir = sandbox_root / "_evals" / "baseline-run"
    baseline_cases = [
        {"case_id": "case_a", "agent": "gate-fake", "passed": True, "skipped": False,
         "summary": "ok", "structural": {"expected_files_present": ["main.py"], "expected_files_missing": []},
         "functional": {"checks": []}, "telemetry": {}, "submitted": True, "error": None,
         "wall_seconds": 0.1, "project_id": "case_a", "skip_reason": None},
        {"case_id": "case_b", "agent": "gate-fake", "passed": True, "skipped": False,
         "summary": "ok", "structural": {"expected_files_present": ["main.py"], "expected_files_missing": []},
         "functional": {"checks": []}, "telemetry": {}, "submitted": True, "error": None,
         "wall_seconds": 0.1, "project_id": "case_b", "skip_reason": None},
    ]
    _write_run_jsonl(baseline_dir, run_id="baseline-run", suite_name="gate", cases=baseline_cases)

    behaviour["case_a"] = {"pass": False}

    tools, state, _sb, events = _build(workspace, clone, store=store)
    res = tools["run_eval_gate"].invoke({"case_ids": "", "baseline_run_id": "baseline-run"})
    parsed = json.loads(res)
    assert "case_a" in parsed["regressions"]
    assert state["eval_gate"] is not None
    assert state["eval_gate"]["regressions"] == ["case_a"]
    kinds = [k for k, _ in events]
    assert "eval_gate_started" in kinds
    assert "eval_gate_done" in kinds
    assert suite_path.exists()
