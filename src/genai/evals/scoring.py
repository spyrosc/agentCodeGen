"""Structural and functional scoring for eval cases."""

from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass, field
from typing import Any

from ..sandbox import Sandbox, Workspace
from .cases import EvalCase, FunctionalCheck


@dataclass
class StructuralScore:
    expected_files_present: list[str] = field(default_factory=list)
    expected_files_missing: list[str] = field(default_factory=list)
    integrator_ok: bool | None = None
    integrator_missing: list[str] = field(default_factory=list)
    integrator_issues: list[str] = field(default_factory=list)
    syntax_ok: bool | None = None
    syntax_output: str | None = None

    @property
    def passed(self) -> bool:
        if self.expected_files_missing:
            return False
        if self.integrator_ok is False:
            return False
        if self.syntax_ok is False:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionalCheckResult:
    name: str
    type: str
    required: bool
    exit_code: int | None
    passed: bool
    output_tail: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionalScore:
    checks: list[FunctionalCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.required)

    def to_dict(self) -> dict[str, Any]:
        return {"checks": [c.to_dict() for c in self.checks]}


# ---------------------------------------------------------------- structural


def score_structural(
    case: EvalCase,
    workspace: Workspace,
    *,
    fullstack: bool,
    sandbox: Sandbox | None = None,
) -> StructuralScore:
    """Check expected files, run integrator (fullstack), and compileall (codegen)."""
    score = StructuralScore()

    for pattern in case.expect_files:
        # Glob relative to workspace root; require ≥1 match.
        matches = list(workspace.root.glob(pattern))
        if matches:
            score.expected_files_present.append(pattern)
        else:
            score.expected_files_missing.append(pattern)

    if fullstack:
        from ..agents.fullstack.integrator import integrate

        report = integrate(workspace.root)
        score.integrator_ok = report.ok
        score.integrator_missing = list(report.missing)
        score.integrator_issues = list(report.issues)
    elif sandbox is not None:
        # Codegen: syntax check via compileall.
        result = sandbox.exec(
            workspace.project_id,
            "python -m compileall -q .",
            timeout_s=60,
        )
        score.syntax_ok = result.exit_code == 0
        score.syntax_output = _tail(result.output)

    return score


# ---------------------------------------------------------------- functional


def score_functional(
    case: EvalCase,
    workspace: Workspace,
    sandbox: Sandbox,
    *,
    enable_frontend_build: bool = False,
) -> FunctionalScore:
    """Run each :class:`FunctionalCheck` in the sandbox; collect results."""
    import time

    score = FunctionalScore()
    for check in case.functional:
        if check.type == "npm_build" and not enable_frontend_build:
            score.checks.append(
                FunctionalCheckResult(
                    name=check.name,
                    type=check.type,
                    required=False,
                    exit_code=None,
                    passed=True,
                    output_tail="(skipped: frontend build disabled)",
                    seconds=0.0,
                )
            )
            continue

        cmd = _build_command(check)
        timeout = check.timeout_s or max(60, case.timeout_s // 4)
        started = time.perf_counter()
        try:
            res = sandbox.exec(workspace.project_id, cmd, timeout_s=timeout)
            elapsed = time.perf_counter() - started
            passed = res.exit_code == check.expect_exit_code
            score.checks.append(
                FunctionalCheckResult(
                    name=check.name,
                    type=check.type,
                    required=check.required,
                    exit_code=res.exit_code,
                    passed=passed,
                    output_tail=_tail(res.output),
                    seconds=round(elapsed, 2),
                )
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            score.checks.append(
                FunctionalCheckResult(
                    name=check.name,
                    type=check.type,
                    required=check.required,
                    exit_code=None,
                    passed=False,
                    output_tail=f"sandbox error: {exc}",
                    seconds=round(elapsed, 2),
                )
            )
    return score


def _build_command(check: FunctionalCheck) -> str:
    cwd = check.cwd or "."
    cd_prefix = f"cd {shlex.quote(cwd)} && " if cwd != "." else ""
    if check.type == "pytest":
        return f"{cd_prefix}pytest -q"
    if check.type == "python_import":
        if not check.module:
            raise ValueError(f"check {check.name!r}: python_import requires `module`")
        modules = [m.strip() for m in check.module.split(",") if m.strip()]
        importer = "; ".join(f"import {m}" for m in modules)
        return f'{cd_prefix}python -c {shlex.quote(importer)}'
    if check.type == "npm_build":
        return f"{cd_prefix}npm ci --no-audit --no-fund && npm run build"
    if check.type == "command":
        if not check.command:
            raise ValueError(f"check {check.name!r}: command requires `command`")
        return f"{cd_prefix}{check.command}"
    raise ValueError(f"unknown check type: {check.type}")


def _tail(text: str, lines: int = 40) -> str:
    if not text:
        return ""
    parts = text.splitlines()
    if len(parts) <= lines:
        return text
    return "\n".join(parts[-lines:])
