"""Integrator node: deterministic post-checks on the assembled project.

Validates that the docker-compose file references the directories the
specialists actually populated, that the YAML parses, and that essential
files exist. Returns a structured report; never modifies user code in the
v0.1 implementation (kept simple — the scaffolder + specialist prompts
already produce a wired-together project).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .._common.fs_tools import emit


@dataclass
class IntegrationReport:
    ok: bool
    missing: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


_REQUIRED = (
    "docker-compose.yml",
    "README.md",
    ".env.example",
    "backend/Dockerfile",
    "backend/start.sh",
    "backend/pyproject.toml",
    "backend/app/main.py",
    "backend/app/db.py",
    "backend/app/auth.py",
    "backend/app/models.py",
    "backend/app/routers/auth.py",
    "backend/alembic.ini",
    "backend/alembic/env.py",
    "frontend/Dockerfile",
    "frontend/package.json",
    "frontend/vite.config.ts",
    "frontend/src/main.tsx",
    "frontend/src/App.tsx",
    "frontend/src/lib/api.ts",
    "frontend/src/lib/auth.ts",
)


def integrate(
    root: Path,
    *,
    on_event: Callable[[str, dict], None] | None = None,
) -> IntegrationReport:
    """Validate the assembled project at ``root``."""
    report = IntegrationReport(ok=True)

    for rel in _REQUIRED:
        if not (root / rel).is_file():
            report.missing.append(rel)
            report.ok = False

    compose = root / "docker-compose.yml"
    if compose.is_file():
        try:
            import yaml  # PyYAML ships with Streamlit; available in the agent container.

            data = yaml.safe_load(compose.read_text())
            services = (data or {}).get("services") or {}
            for required_service in ("db", "backend", "frontend"):
                if required_service not in services:
                    report.issues.append(
                        f"docker-compose.yml is missing service `{required_service}`"
                    )
                    report.ok = False
        except Exception as exc:  # noqa: BLE001
            report.issues.append(f"docker-compose.yml does not parse: {exc}")
            report.ok = False

    # Spot-check that an alembic migration was generated.
    versions_dir = root / "backend" / "alembic" / "versions"
    if versions_dir.is_dir():
        migs = [p for p in versions_dir.iterdir() if p.suffix == ".py"]
        if not migs:
            report.issues.append(
                "no Alembic migration generated under backend/alembic/versions/"
            )
            # Not fatal — a developer can run `alembic revision --autogenerate`.

    emit(
        on_event,
        "compose_validated",
        ok=report.ok,
        missing=report.missing,
        issues=report.issues,
    )
    return report
