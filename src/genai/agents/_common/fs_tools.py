"""Reusable LangChain `StructuredTool` builders for workspace file I/O.

The codegen agent and the fullstack agent's specialists both need the same
``list_files`` / ``read_file`` / ``write_file`` / ``delete_file`` shape. They
share this module so behaviour stays identical.

The :func:`scope_to` helper returns a "view" of a workspace rooted at a
sub-directory — used by the fullstack agent to give the backend specialist
tools that can only touch ``backend/`` and the frontend specialist tools
that can only touch ``frontend/``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool, tool

from ...core.config import get_settings
from ...sandbox.workspace import Workspace


def emit(on_event: Callable[[str, dict], None] | None, kind: str, **payload) -> None:
    """Best-effort UI event emit. Never raises into the agent loop."""
    if on_event is None:
        return
    try:
        on_event(kind, payload)
    except Exception:  # pragma: no cover - UI callback must not break the agent
        pass


# --------------------------------------------------------------- scoping
@dataclass
class ScopedWorkspace:
    """Restricts a :class:`Workspace` to a sub-directory.

    All paths handed to the LLM are *relative to the sub-directory* and any
    attempt to traverse outside it raises ``ValueError`` (same primitive as
    :meth:`Workspace.resolve`).
    """

    workspace: Workspace
    subdir: str  # e.g. "backend" — relative to project root, no leading "/"

    @property
    def project_id(self) -> str:
        return self.workspace.project_id

    @property
    def root(self) -> Path:
        sub = (self.workspace.root / self.subdir).resolve()
        sub.relative_to(self.workspace.root)  # safety: must stay inside
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    def resolve(self, rel: str) -> Path:
        if rel is None:
            raise ValueError("path must not be None")
        cleaned = str(rel).lstrip("/\\").strip()
        target = (self.root / (cleaned or ".")).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"path {rel!r} escapes scoped directory {self.subdir!r}"
            ) from exc
        return target

    def relative(self, abs_path: Path) -> str:
        return str(abs_path.relative_to(self.root))


def scope_to(workspace: Workspace, subdir: str) -> ScopedWorkspace:
    """Return a :class:`ScopedWorkspace` rooted at ``workspace.root / subdir``."""
    if not subdir or subdir.startswith("/") or ".." in Path(subdir).parts:
        raise ValueError(f"unsafe scope subdir: {subdir!r}")
    return ScopedWorkspace(workspace=workspace, subdir=subdir)


# -------------------------------------------------------------- builders
def build_fs_tools(
    ws: Workspace | ScopedWorkspace,
    *,
    on_event: Callable[[str, dict], None] | None = None,
    label: str | None = None,
) -> list[StructuredTool]:
    """Return ``[list_files, read_file, write_file, delete_file]`` over ``ws``.

    ``label`` (e.g. ``"backend"``) is included in emitted events so the UI
    can show which specialist made the call.
    """
    cfg = get_settings().sandbox

    def _evt(kind: str, **payload):
        if label:
            payload.setdefault("scope", label)
        emit(on_event, kind, **payload)

    @tool
    def list_files(path: str = ".") -> str:
        """List files and directories under a relative path. Use ``"."`` for the whole tree."""
        target = ws.resolve(path)
        if not target.exists():
            return f"<no such path: {path}>"
        if target.is_file():
            return ws.relative(target)
        entries: list[str] = []
        for p in sorted(target.rglob("*")):
            if "__pycache__" in p.parts or "node_modules" in p.parts:
                continue
            entries.append(ws.relative(p) + ("/" if p.is_dir() else ""))
            if len(entries) >= 500:
                entries.append("... (truncated)")
                break
        _evt("list_files", path=path, count=len(entries))
        return "\n".join(entries) if entries else "<empty>"

    @tool
    def read_file(path: str) -> str:
        """Return the UTF-8 contents of a file (truncated to the configured byte limit)."""
        target = ws.resolve(path)
        if not target.exists() or not target.is_file():
            return f"<no such file: {path}>"
        data = target.read_bytes()
        truncated = len(data) > cfg.max_file_bytes
        if truncated:
            data = data[: cfg.max_file_bytes]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary file, {len(data)} bytes>"
        _evt("read_file", path=path)
        return text + (
            f"\n<... truncated, file is larger than {cfg.max_file_bytes} bytes>"
            if truncated
            else ""
        )

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file. Parent directories are created automatically."""
        target = ws.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        _evt("write_file", path=path, bytes=len(content))
        return f"wrote {len(content)} bytes to {ws.relative(target)}"

    @tool
    def delete_file(path: str) -> str:
        """Delete a single file. Refuses to delete directories."""
        target = ws.resolve(path)
        if not target.exists():
            return f"<no such file: {path}>"
        if target.is_dir():
            return f"<refusing to delete directory: {path}>"
        target.unlink()
        _evt("delete_file", path=path)
        return f"deleted {path}"

    return [list_files, read_file, write_file, delete_file]
