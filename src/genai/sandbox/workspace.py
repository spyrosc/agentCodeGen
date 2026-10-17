"""Per-project workspace directories on the shared volume.

The same host directory is bind-mounted into both the agent container and
the sandbox container at the same path (``/workspaces`` by default), so file
paths agree on both sides.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..core.config import get_settings

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]+$")


def new_project_id(prefix: str = "proj") -> str:
    """Filesystem-safe, time-ordered, collision-resistant project id."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}-{uuid.uuid4().hex[:6]}"


@dataclass
class Workspace:
    """A single project's directory.

    All file-system operations route through :meth:`resolve` which guarantees
    the resolved absolute path stays inside :attr:`root`.
    """

    project_id: str
    root: Path  # absolute

    # ------------------------------------------------------------ factory
    @classmethod
    def create(
        cls,
        project_id: str,
        base: str | os.PathLike[str] | None = None,
    ) -> Workspace:
        if not _SAFE_ID.match(project_id):
            raise ValueError(f"unsafe project id: {project_id!r}")
        base_path = Path(base or get_settings().sandbox.workspace_root)
        base_path.mkdir(parents=True, exist_ok=True)
        root = (base_path / project_id).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return cls(project_id=project_id, root=root)

    # -------------------------------------------------------------- paths
    def resolve(self, rel: str) -> Path:
        """Resolve ``rel`` under :attr:`root`. Refuses traversal outside it."""
        if rel is None:
            raise ValueError("path must not be None")
        cleaned = str(rel).lstrip("/\\").strip()
        if not cleaned or cleaned in (".", "./"):
            return self.root
        candidate = (self.root / cleaned).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"path {rel!r} escapes project workspace {self.root}"
            ) from exc
        return candidate

    def relative(self, abs_path: Path) -> str:
        return str(abs_path.relative_to(self.root))

    # --------------------------------------------------------------- misc
    def list_tree(self, max_entries: int = 500) -> list[str]:
        out: list[str] = []
        for p in sorted(self.root.rglob("*")):
            rel_parts = p.relative_to(self.root).parts
            if "__pycache__" in p.parts or any(
                part.startswith(".") for part in rel_parts
            ):
                continue
            rel = self.relative(p)
            out.append(rel + ("/" if p.is_dir() else ""))
            if len(out) >= max_entries:
                out.append(f"... (truncated at {max_entries} entries)")
                break
        return out

    def zip_bytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in self.root.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    zf.write(p, arcname=str(p.relative_to(self.root)))
        return buf.getvalue()

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
