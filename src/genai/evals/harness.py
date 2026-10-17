"""Eval harness driver.

Iterates an :class:`EvalSuite`, running each case against the configured
agent + sandbox, and emits JSONL artifacts as it goes so partial runs
remain analysable on crash.
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..agents.base import AgentEvent, AgentResult
from ..agents.registry import get as get_agent
from ..core.config import get_settings
from ..core.logging import get_logger
from ..llm import load_llm
from ..sandbox import DockerExecSandbox, Sandbox, Workspace
from .cases import EvalCase, EvalSuite
from .scoring import (
    FunctionalScore,
    StructuralScore,
    score_functional,
    score_structural,
)
from .telemetry import TelemetryHandler, attach

log = get_logger("evals.harness")

_FULLSTACK_AGENTS = {"fullstack"}


@dataclass
class CaseResult:
    case_id: str
    agent: str
    project_id: str | None
    skipped: bool
    skip_reason: str | None
    submitted: bool
    summary: str | None
    error: str | None
    wall_seconds: float
    structural: dict[str, Any]
    functional: dict[str, Any]
    telemetry: dict[str, Any]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteResult:
    run_id: str
    suite_name: str
    model: str
    started_at: str
    finished_at: str
    wall_seconds: float
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.cases if not c.passed and not c.skipped)

    @property
    def skipped_count(self) -> int:
        return sum(1 for c in self.cases if c.skipped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "suite_name": self.suite_name,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_seconds": round(self.wall_seconds, 2),
            "passed": self.passed_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "cases": [c.to_dict() for c in self.cases],
        }


# ----------------------------------------------------------------- run_case


def run_case(
    case: EvalCase,
    *,
    run_dir: Path,
    sandbox: Sandbox,
    llm: Any,
    enable_frontend_build: bool = False,
    skip_functional: bool = False,
    jsonl_writer: Any = None,
) -> CaseResult:
    """Execute a single eval case end-to-end."""
    started = time.perf_counter()

    if case.skip:
        return CaseResult(
            case_id=case.id,
            agent=case.agent,
            project_id=None,
            skipped=True,
            skip_reason=case.skip,
            submitted=False,
            summary=None,
            error=None,
            wall_seconds=0.0,
            structural={},
            functional={},
            telemetry={},
            passed=False,
        )

    handler = TelemetryHandler()
    instrumented_llm = attach(llm, handler)

    try:
        agent_cls = get_agent(case.agent)
    except KeyError as exc:
        return _error_result(case, started, f"unknown agent: {exc}")

    agent = agent_cls(llm=instrumented_llm, sandbox=sandbox)

    projects_dir = run_dir / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace.create(case.id, base=projects_dir)

    def on_event(ev: AgentEvent) -> None:
        handler.record_event(ev.kind)
        if jsonl_writer is not None:
            jsonl_writer.write(
                {"type": "event", "case_id": case.id, "kind": ev.kind, "data": _safe_data(ev.data)}
            )

    result_holder: dict[str, Any] = {}

    def _drive() -> None:
        try:
            res = agent.run(
                case.prompt,
                workspace=workspace,
                max_iterations=case.max_iterations,
                on_event=on_event,
            )
            result_holder["result"] = res
        except Exception:  # noqa: BLE001
            result_holder["error"] = traceback.format_exc()

    error: str | None = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_drive)
        try:
            future.result(timeout=case.timeout_s)
        except FutureTimeout:
            error = f"timed out after {case.timeout_s}s"
            log.warning("case %s: %s", case.id, error)

    if "error" in result_holder:
        error = result_holder["error"]

    agent_result: AgentResult = result_holder.get(
        "result", AgentResult(submitted=False, summary=None, project_id=workspace.project_id)
    )

    fullstack = case.agent in _FULLSTACK_AGENTS
    try:
        structural = score_structural(
            case, workspace, fullstack=fullstack, sandbox=sandbox
        )
    except Exception as exc:  # noqa: BLE001
        structural = StructuralScore()
        error = (error + " | " if error else "") + f"structural error: {exc}"

    if skip_functional or error:
        functional = FunctionalScore()
    else:
        try:
            functional = score_functional(
                case, workspace, sandbox, enable_frontend_build=enable_frontend_build
            )
        except Exception as exc:  # noqa: BLE001
            functional = FunctionalScore()
            error = (error + " | " if error else "") + f"functional error: {exc}"

    wall = time.perf_counter() - started
    passed = (
        error is None
        and structural.passed
        and functional.passed
        and bool(agent_result.submitted)
    )

    case_result = CaseResult(
        case_id=case.id,
        agent=case.agent,
        project_id=workspace.project_id,
        skipped=False,
        skip_reason=None,
        submitted=bool(agent_result.submitted),
        summary=agent_result.summary,
        error=error,
        wall_seconds=round(wall, 2),
        structural=structural.to_dict(),
        functional=functional.to_dict(),
        telemetry=handler.snapshot.to_dict(),
        passed=passed,
    )

    if jsonl_writer is not None:
        jsonl_writer.write({"type": "case_summary", **case_result.to_dict()})

    return case_result


def _error_result(case: EvalCase, started: float, msg: str) -> CaseResult:
    return CaseResult(
        case_id=case.id,
        agent=case.agent,
        project_id=None,
        skipped=False,
        skip_reason=None,
        submitted=False,
        summary=None,
        error=msg,
        wall_seconds=round(time.perf_counter() - started, 2),
        structural={},
        functional={},
        telemetry={},
        passed=False,
    )


def _safe_data(data: dict[str, Any]) -> dict[str, Any]:
    """Best-effort JSON-safe copy of an event payload."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out


# ---------------------------------------------------------------- run_suite


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def run_suite(
    suite: EvalSuite,
    *,
    base_dir: Path | str | None = None,
    sandbox: Sandbox | None = None,
    llm: Any = None,
    only: list[str] | None = None,
    enable_frontend_build: bool = False,
    skip_functional: bool = False,
) -> SuiteResult:
    """Run every case in ``suite`` (optionally filtered by ``only`` ids)."""
    cfg = get_settings()
    base = Path(base_dir or Path(cfg.sandbox.workspace_root) / "_evals")
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if sandbox is None:
        sandbox = DockerExecSandbox.from_settings()
    if llm is None:
        llm = load_llm()

    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    writer = _JsonlWriter(run_dir / "run.jsonl")
    writer.write(
        {
            "type": "suite_started",
            "run_id": run_id,
            "suite_name": suite.name,
            "model": cfg.llm.model,
            "started_at": started_at,
        }
    )

    case_results: list[CaseResult] = []
    try:
        for case in suite.cases:
            if only and case.id not in only:
                continue
            log.info("eval: case %s (agent=%s)", case.id, case.agent)
            result = run_case(
                case,
                run_dir=run_dir,
                sandbox=sandbox,
                llm=llm,
                enable_frontend_build=enable_frontend_build,
                skip_functional=skip_functional,
                jsonl_writer=writer,
            )
            case_results.append(result)
    finally:
        wall = time.perf_counter() - started
        suite_result = SuiteResult(
            run_id=run_id,
            suite_name=suite.name,
            model=cfg.llm.model,
            started_at=started_at,
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            wall_seconds=round(wall, 2),
            cases=case_results,
        )
        writer.write({"type": "suite_summary", **suite_result.to_dict()})
        writer.close()

    # Optional: seed the lessons store from any failed cases so the next
    # codegen / fullstack / reflect run benefits automatically.
    if cfg.reflect.lessons_autoseed:
        try:
            from ..reflection.seed_from_evals import seed_from_run
            from ..reflection.store import LessonStore

            seeded = seed_from_run(run_dir, store=LessonStore.from_settings(cfg.reflect))
            if seeded:
                log.info("autoseeded %d lessons from %s", seeded, run_id)
        except Exception:  # noqa: BLE001
            log.exception("lesson autoseed failed for run %s", run_id)

    return suite_result
