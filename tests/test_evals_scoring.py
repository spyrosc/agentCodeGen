"""Tests for structural and functional scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from genai.evals.cases import EvalCase, FunctionalCheck
from genai.evals.scoring import score_functional, score_structural
from genai.sandbox import ExecResult, Workspace


@dataclass
class FakeSandbox:
    """In-memory ``Sandbox`` stand-in. Maps command substring → ExecResult."""

    rules: dict[str, ExecResult]

    def exec(self, project_id: str, command: str, timeout_s: int) -> ExecResult:  # noqa: ARG002
        for needle, res in self.rules.items():
            if needle in command:
                return res
        return ExecResult(exit_code=127, output=f"no rule for: {command}")


def _ws(tmp_path: Path) -> Workspace:
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    return Workspace(project_id="p1", root=root)


def test_structural_files_present_and_missing(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws.root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (ws.root / "tests").mkdir()
    (ws.root / "tests" / "test_x.py").write_text("def test_a(): pass\n")

    case = EvalCase(
        id="c", agent="codegen", prompt="x",
        expect_files=["pyproject.toml", "tests/**/test_*.py", "missing.txt"],
    )
    sb = FakeSandbox({"compileall": ExecResult(exit_code=0, output="")})
    score = score_structural(case, ws, fullstack=False, sandbox=sb)

    assert "pyproject.toml" in score.expected_files_present
    assert "tests/**/test_*.py" in score.expected_files_present
    assert score.expected_files_missing == ["missing.txt"]
    assert score.syntax_ok is True
    assert score.passed is False  # missing file


def test_structural_codegen_syntax_failure(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws.root / "pyproject.toml").write_text("")
    case = EvalCase(id="c", agent="codegen", prompt="x", expect_files=["pyproject.toml"])
    sb = FakeSandbox({"compileall": ExecResult(exit_code=1, output="SyntaxError: bad")})

    score = score_structural(case, ws, fullstack=False, sandbox=sb)
    assert score.syntax_ok is False
    assert "SyntaxError" in (score.syntax_output or "")
    assert score.passed is False


def test_functional_pytest_pass_and_fail(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    case = EvalCase(
        id="c", agent="codegen", prompt="x",
        functional=[
            FunctionalCheck(name="ok", type="pytest"),
            FunctionalCheck(name="bad", type="pytest", cwd="sub", required=False),
        ],
    )
    sb = FakeSandbox({
        "cd sub && pytest -q": ExecResult(exit_code=1, output="1 failed"),
        "pytest -q": ExecResult(exit_code=0, output="2 passed"),
    })
    score = score_functional(case, ws, sb)
    by_name = {c.name: c for c in score.checks}
    assert by_name["ok"].passed is True
    assert by_name["bad"].passed is False
    # `bad` is non-required → suite still passes
    assert score.passed is True


def test_functional_npm_build_skipped_by_default(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    case = EvalCase(
        id="c", agent="fullstack", prompt="x",
        functional=[FunctionalCheck(name="fe", type="npm_build", cwd="frontend")],
    )
    sb = FakeSandbox({})
    score = score_functional(case, ws, sb, enable_frontend_build=False)
    assert score.checks[0].passed is True
    assert "skipped" in score.checks[0].output_tail


def test_functional_python_import_command_built(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    case = EvalCase(
        id="c", agent="fullstack", prompt="x",
        functional=[FunctionalCheck(name="imp", type="python_import",
                                    cwd="backend", module="app.main, app.db")],
    )
    sb = FakeSandbox({"import app.main; import app.db": ExecResult(exit_code=0, output="")})
    score = score_functional(case, ws, sb)
    assert score.checks[0].passed is True
