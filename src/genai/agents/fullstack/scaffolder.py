"""Deterministic project skeleton emitter.

Given a :class:`ProjectSpec` and a :class:`Workspace`, write the file set
that *every* generated app shares: docker-compose, env example, READMEs,
backend skeleton (FastAPI app boilerplate, auth, db, alembic config, test
conftest) and frontend skeleton (Vite + React + TS + Tailwind boilerplate).

The specialist nodes (backend / frontend) layer entity-specific code on
top of this skeleton — they don't need to re-emit the boilerplate.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ...sandbox.workspace import Workspace
from .._common.fs_tools import emit
from .spec import ProjectSpec
from .templates import BACKEND_FILES, FRONTEND_FILES, ROOT_FILES

# Docker image reference: [registry/]name[:tag][@digest]; keeps YAML injection out.
_IMAGE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*")


@dataclass
class ScaffoldResult:
    files_written: list[str]


def scaffold(
    workspace: Workspace,
    spec: ProjectSpec,
    *,
    backend_port: int = 8000,
    frontend_port: int = 5173,
    postgres_image: str = "postgres:16-alpine",
    on_event: Callable[[str, dict], None] | None = None,
) -> ScaffoldResult:
    """Write the immutable skeleton for ``spec`` into ``workspace``."""
    if not _IMAGE_REF.fullmatch(postgres_image):
        raise ValueError(f"invalid postgres image reference: {postgres_image!r}")
    ctx = {
        "project_name": spec.meta.name,
        "project_title": spec.meta.title,
        "project_description": spec.meta.description or spec.meta.title,
        "backend_port": backend_port,
        "frontend_port": frontend_port,
        "postgres_image": postgres_image,
    }

    written: list[str] = []

    def _write(rel: str, content: str) -> None:
        target = workspace.resolve(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel)

    for rel, template in ROOT_FILES.items():
        _write(rel, template.safe_substitute(**ctx))
    for rel, template in BACKEND_FILES.items():
        _write(f"backend/{rel}", template.safe_substitute(**ctx))
    for rel, template in FRONTEND_FILES.items():
        _write(f"frontend/{rel}", template.safe_substitute(**ctx))

    # Drop the spec into the workspace as a reference artifact.
    _write("project_spec.json", spec.to_json())

    emit(on_event, "scaffolded", count=len(written))
    return ScaffoldResult(files_written=written)


# Convenience: simple existence check used by the integrator + tests.
def list_skeleton_files(spec: ProjectSpec, root: Path) -> list[Path]:
    return [
        root / rel
        for rel in (
            list(ROOT_FILES.keys())
            + [f"backend/{r}" for r in BACKEND_FILES]
            + [f"frontend/{r}" for r in FRONTEND_FILES]
        )
    ]
