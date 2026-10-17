"""Tests for the docker-compose smoke node (sandbox-mocked)."""

from __future__ import annotations

from dataclasses import dataclass

from genai.agents.fullstack.compose import (
    _compose_project_name,
    run_compose_smoke,
    runtime_log_errors,
)
from genai.sandbox.base import ExecResult


@dataclass
class _FakeSandbox:
    """Records every exec call and returns scripted outputs."""

    script: list[ExecResult]
    calls: list[tuple[str, str, int]]

    def exec(self, project_id: str, command: str, timeout_s: int) -> ExecResult:
        self.calls.append((project_id, command, timeout_s))
        if self.script:
            return self.script.pop(0)
        return ExecResult(exit_code=0, output="")


def _sb(*results: ExecResult) -> _FakeSandbox:
    return _FakeSandbox(script=list(results), calls=[])


def test_compose_project_name_is_slugified():
    name = _compose_project_name("Proj-20260601-AbC.def")
    assert name == "proj-20260601-abc-def"
    assert "." not in name


def test_run_compose_smoke_happy_path():
    events: list[tuple[str, dict]] = []
    sb = _sb(
        ExecResult(0, ""),                        # pre-emptive down
        ExecResult(0, "Creating ..."),           # up -d --build
        ExecResult(0, "backend\trunning\tUp\nfrontend\trunning\tUp\ndb\trunning\tUp"),
        ExecResult(0, "200"),                    # health check
        ExecResult(0, ""),                        # teardown
    )
    report = run_compose_smoke(
        sb, "proj-x", backend_port=8000,
        on_event=lambda k, p: events.append((k, p)),
    )
    assert report.ok
    assert report.project == "proj-x"
    assert "backend" in report.services_up
    # We always emit start + ok + down.
    kinds = [k for k, _ in events]
    assert "compose_up_started" in kinds
    assert "compose_up_ok" in kinds
    assert "compose_down" in kinds


def test_run_compose_smoke_build_failure_captures_log():
    sb = _sb(
        ExecResult(0, ""),
        ExecResult(1, "ERROR: build failed: missing package"),
        ExecResult(0, ""),
    )
    report = run_compose_smoke(sb, "proj-y")
    assert not report.ok
    assert "build failed" in report.log
    # Teardown still ran (we always run it).
    teardown_cmds = [c[1] for c in sb.calls if "down -v" in c[1]]
    assert len(teardown_cmds) >= 2  # pre-emptive + final


def test_run_compose_smoke_health_failure_collects_logs():
    sb = _sb(
        ExecResult(0, ""),                        # pre-down
        ExecResult(0, "ok"),                      # up
        ExecResult(0, "backend\trunning\tUp"),   # ps
        ExecResult(1, "<health check timed out>"),
        ExecResult(0, "backend  | ImportError: no module 'foo'"),
        ExecResult(0, ""),                        # teardown
    )
    report = run_compose_smoke(sb, "proj-z", health_timeout_s=3)
    assert not report.ok
    assert "ImportError" in report.log


def test_run_compose_smoke_always_tears_down_even_on_success():
    sb = _sb(
        ExecResult(0, ""),
        ExecResult(0, ""),
        ExecResult(0, "backend\trunning\tUp"),
        ExecResult(0, "200"),
        ExecResult(0, ""),
    )
    run_compose_smoke(sb, "proj-w")
    down_calls = [c for c in sb.calls if "down -v" in c[1]]
    # one pre-emptive + one final
    assert len(down_calls) == 2


# Real log from a run where start.sh's create_all fallback hid a broken alembic setup.
_MASKED_ALEMBIC_CRASH = """\
db-1  | 2026-10-14 13:52:43.115 UTC [68] FATAL:  database "app" does not exist
backend-1  | [start.sh] running migrations...
backend-1  | Traceback (most recent call last):
backend-1  |   File "/usr/local/bin/alembic", line 8, in <module>
backend-1  |     sys.exit(main())
backend-1  |   File "/app/alembic/env.py", line 7, in <module>
backend-1  |     from app.db import Base  # noqa: F401
backend-1  | ModuleNotFoundError: No module named 'app'
backend-1  | [start.sh] alembic upgrade failed; falling back to metadata.create_all
backend-1  | [start.sh] ensured tables via Base.metadata.create_all
backend-1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
"""


def test_runtime_log_errors_finds_masked_alembic_crash():
    excerpt = runtime_log_errors(_MASKED_ALEMBIC_CRASH)
    assert "ModuleNotFoundError: No module named 'app'" in excerpt
    assert "alembic upgrade failed" in excerpt
    assert "[start.sh] running migrations..." in excerpt


def test_runtime_log_errors_ignores_clean_logs():
    clean = (
        "db-1  | LOG:  database system is ready to accept connections\n"
        "backend-1  | [start.sh] running migrations...\n"
        "backend-1  | INFO  [alembic.runtime.migration] Running upgrade  -> 0001\n"
        "backend-1  | INFO:     Uvicorn running on http://0.0.0.0:8000\n"
    )
    assert runtime_log_errors(clean) == ""


def test_run_compose_smoke_demotes_healthy_stack_with_crash_in_logs():
    events: list[tuple[str, dict]] = []
    sb = _sb(
        ExecResult(0, ""),                         # pre-emptive down
        ExecResult(0, "ok"),                       # up
        ExecResult(0, "backend\trunning\tUp"),    # ps
        ExecResult(0, "200"),                      # health passes thanks to the fallback
        ExecResult(0, _MASKED_ALEMBIC_CRASH),      # logs scan
        ExecResult(0, ""),                         # teardown
    )
    report = run_compose_smoke(sb, "proj-m", on_event=lambda k, p: events.append((k, p)))

    assert not report.ok
    assert "[runtime log errors]" in report.log
    assert "ModuleNotFoundError" in report.log
    kinds = [k for k, _ in events]
    assert kinds.index("compose_up_ok") < kinds.index("compose_log_errors")
    assert any("logs --no-color" in cmd for _pid, cmd, _t in sb.calls)
    assert "down -v" in sb.calls[-1][1]
