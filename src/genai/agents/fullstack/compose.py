"""Compose-up smoke node.

Brings the generated stack up via ``docker compose`` from inside the
sandbox container (docker-out-of-docker through the mounted host docker
socket), waits for the backend ``/health`` endpoint to respond, optionally
runs a real CRUD probe against every entity in the spec, captures logs on
failure, and tears everything down with ``-v``.

The generated project is identified by ``project_id`` so multiple runs do
not collide (``docker compose -p <slug> ...``).

Returns a :class:`ComposeReport` describing the outcome; the agent uses
``ok`` for routing and ``log`` for the reviewer prompt.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.logging import get_logger
from ...core.text import truncate_middle
from ...sandbox.base import Sandbox
from .._common.fs_tools import emit
from .smoke import SmokeReport, run_smoke_crud
from .spec import ProjectSpec

log = get_logger("agents.fullstack.compose")

# Hard cap so logs we hand to the LLM stay bounded.
_LOG_TAIL_LINES = 80
_LOG_MAX_CHARS = 6_000

# Log lines that mean a service is broken even though /health answers
# (start.sh falls back to create_all when migrations crash).
_LOG_ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "alembic upgrade failed",
    "metadata.create_all skipped",
)
_LOG_ERROR_CONTEXT_LINES = 40

# Compose project names must be a-z0-9 and start/end with [a-z0-9].
_NAME_SAFE = re.compile(r"[^a-z0-9_-]+")


@dataclass
class ComposeReport:
    ok: bool
    project: str
    health_status: str = ""
    log: str = ""
    services_up: list[str] = field(default_factory=list)
    smoke: SmokeReport | None = None



def _compose_project_name(project_id: str) -> str:
    slug = _NAME_SAFE.sub("-", project_id.lower()).strip("-_") or "genapp"
    return slug[:48]


def _truncate(text: str) -> str:
    return truncate_middle(text, _LOG_MAX_CHARS)


def _tail(text: str, n: int = _LOG_TAIL_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "\n".join(lines[-n:])


def runtime_log_errors(logs: str) -> str:
    """Excerpt around the first error marker in ``logs``, or ``""`` if clean."""
    lines = logs.splitlines()
    for i, line in enumerate(lines):
        if any(marker in line for marker in _LOG_ERROR_MARKERS):
            return "\n".join(lines[max(0, i - 3) : i + _LOG_ERROR_CONTEXT_LINES])
    return ""


def run_compose_smoke(
    sandbox: Sandbox,
    project_id: str,
    *,
    backend_port: int = 8000,
    on_event: Callable[[str, dict], None] | None = None,
    build_timeout_s: int = 600,
    health_timeout_s: int = 60,
    spec: ProjectSpec | None = None,
    workspace_root: Path | None = None,
    smoke_timeout_s: int = 180,
) -> ComposeReport:
    """Bring up the generated stack, validate, then tear down.

    If ``spec`` and ``workspace_root`` are supplied, additionally runs a
    real CRUD probe against every entity in the spec via
    :func:`smoke.run_smoke_crud` *while the stack is still up*. A failed
    probe is treated as a failed run (``report.ok = False``), and the
    captured runner output is appended to ``report.log`` so the reviewer
    sees what actually broke.

    Always issues a ``docker compose down -v --remove-orphans`` at the
    end — even on failure — so the host doesn't accumulate dangling
    networks/volumes/containers across runs.
    """
    project = _compose_project_name(project_id)
    report = ComposeReport(ok=False, project=project)

    emit(on_event, "compose_up_started", project=project)

    # Pre-emptively clear any stale stack from a prior failed run.
    sandbox.exec(
        project_id,
        f"docker compose -p {project} down -v --remove-orphans 2>&1 || true",
        60,
    )

    up_cmd = (
        f"docker compose -p {project} up -d --build --remove-orphans"
    )
    up = sandbox.exec(project_id, up_cmd, build_timeout_s)
    if up.exit_code != 0:
        report.log = _truncate(up.output)
        emit(
            on_event,
            "compose_up_failed",
            project=project,
            stage="build",
            exit=up.exit_code,
        )
        _teardown(sandbox, project_id, project, on_event=on_event)
        return report

    ps = sandbox.exec(
        project_id,
        f"docker compose -p {project} ps --format '{{{{.Service}}}}\t{{{{.State}}}}\t{{{{.Status}}}}'",
        20,
    )
    report.services_up = [
        line.split("\t", 1)[0]
        for line in (ps.output or "").splitlines()
        if line.strip() and "\t" in line
    ]

    # Poll the backend /health endpoint from inside the backend container,
    # so we don't depend on host port-forwarding being reachable.
    health_cmd = (
        f"for i in $(seq 1 {max(1, health_timeout_s // 3)}); do "
        f"  out=$(docker compose -p {project} exec -T backend "
        f"      sh -c 'python -c \"import urllib.request,sys;"
        f"r=urllib.request.urlopen(\\\"http://127.0.0.1:{backend_port}/health\\\",timeout=2);"
        f"print(r.status); sys.exit(0 if r.status==200 else 1)\"' 2>&1) && "
        f"    echo \"$out\" && exit 0; "
        f"  sleep 3; "
        f"done; "
        f"echo \"<health check timed out>\"; exit 1"
    )
    health = sandbox.exec(project_id, health_cmd, health_timeout_s + 30)
    report.health_status = (health.output or "").strip()[:200]

    if health.exit_code == 0:
        report.ok = True
        emit(on_event, "compose_up_ok", project=project, services=report.services_up)

        # Optional runtime CRUD probe. Only meaningful once /health is green.
        if spec is not None and workspace_root is not None and spec.entities:
            smoke = run_smoke_crud(
                sandbox,
                workspace_root,
                project_id,
                project,
                spec,
                backend_port=backend_port,
                timeout_s=smoke_timeout_s,
                on_event=on_event,
            )
            report.smoke = smoke
            if not smoke.ok and not smoke.skipped:
                # Demote the overall run: the stack came up but a real user
                # request would fail. Surface the runner transcript so the
                # reviewer can route back to the right specialist.
                report.ok = False
                report.log = _truncate(
                    (report.log + "\n\n" if report.log else "")
                    + "[crud smoke output]\n"
                    + smoke.log
                )

        logs = sandbox.exec(project_id, f"docker compose -p {project} logs --no-color 2>&1", 60)
        runtime_errors = runtime_log_errors(logs.output or "")
        if runtime_errors:
            report.ok = False
            report.log = _truncate(
                (report.log + "\n\n" if report.log else "")
                + "[runtime log errors]\n"
                + runtime_errors
            )
            emit(on_event, "compose_log_errors", project=project)
    else:
        logs = sandbox.exec(
            project_id,
            f"docker compose -p {project} logs --tail 200 --no-color 2>&1",
            60,
        )
        report.log = _truncate(_tail(logs.output or "", _LOG_TAIL_LINES))
        emit(
            on_event,
            "compose_up_failed",
            project=project,
            stage="health",
            status=report.health_status[:120],
        )

    _teardown(sandbox, project_id, project, on_event=on_event)
    return report


def _teardown(
    sandbox: Sandbox,
    project_id: str,
    project: str,
    *,
    on_event: Callable[[str, dict], None] | None,
) -> None:
    sandbox.exec(
        project_id,
        f"docker compose -p {project} down -v --remove-orphans 2>&1 || true",
        120,
    )
    emit(on_event, "compose_down", project=project)


# Re-exported for convenience.
__all__ = ["ComposeReport", "run_compose_smoke"]


def _coerce_any(_: Any) -> Any:  # pragma: no cover - placeholder for typing
    return _
