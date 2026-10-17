"""Source-tree isolation primitive for the reflect agent.

The reflect agent never touches the live working copy. Instead, it operates
on a *clone* of the platform's source rooted at ``<workspace>/repo``.

Two paths are supported:

* **Git checkout** — preferred. We use ``git worktree add`` so we share the
  object database with the source repo (cheap, fast) but get an independent
  working tree the agent can mutate freely.
* **Plain directory** — fallback. We ``shutil.copytree`` with a small
  ignore filter (``__pycache__``, ``.venv``, ``node_modules``, ``.git`` of
  any nested repos, ``workspaces/``).

In either case, mutating the clone is guaranteed not to affect the source.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ...core.logging import get_logger
from ...sandbox.workspace import Workspace

log = get_logger("agents.reflect.repo")


class ReflectionUnavailable(RuntimeError):
    """Raised when the platform source tree cannot be reached.

    Typically means the agent container was started without the read-only
    ``/repo`` mount. The reflect agent surfaces this as an ``error`` event
    and exits cleanly rather than crashing.
    """


@dataclass
class ClonedRepo:
    """The result of :func:`prepare_clone`.

    ``base_sha`` is ``None`` for non-git checkouts. ``is_git`` distinguishes
    the two modes so the propose_patch tool can pick branch-vs-unified-diff.
    """

    clone_path: Path
    base_sha: str | None
    base_branch: str | None
    is_git: bool


# --------------------------------------------------------------------- helpers


def _ignore(_dir: str, names: list[str]) -> list[str]:
    skip = {"__pycache__", ".venv", "venv", "node_modules", "workspaces", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    return [n for n in names if n in skip]


def _git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def _is_git_checkout(path: Path) -> bool:
    return (path / ".git").exists()


# ------------------------------------------------------------------- public API


def prepare_clone(
    workspace: Workspace,
    *,
    repo_mount: str | Path = "/repo",
    base_branch: str = "main",
) -> ClonedRepo:
    """Materialise a writable copy of the source repo under ``workspace.root``.

    The returned :class:`ClonedRepo` has a ``clone_path`` always equal to
    ``workspace.root / "repo"``. Behaviour:

    * If ``repo_mount`` does not exist → :class:`ReflectionUnavailable`.
    * If it is a git checkout AND ``git`` is available → ``git worktree add``
      to a *detached* HEAD pointing at the source's current commit. The
      worktree is fully writable, but pushes/pulls are isolated.
    * Otherwise → ``shutil.copytree`` with the ignore filter above.

    Idempotency: if ``workspace.root / "repo"`` already exists, it is removed
    first so callers can re-prepare on retry.
    """
    src = Path(repo_mount).resolve()
    if not src.exists():
        raise ReflectionUnavailable(
            f"repo mount {src} does not exist; the agent container must be "
            "started with a read-only bind of the platform repo at /repo "
            "(see docker-compose.yml)"
        )
    if not src.is_dir():
        raise ReflectionUnavailable(f"repo mount {src} is not a directory")

    clone_path = (workspace.root / "repo").resolve()
    # Safety: must remain inside the workspace.
    clone_path.relative_to(workspace.root)

    if clone_path.exists():
        shutil.rmtree(clone_path)

    if _is_git_checkout(src) and shutil.which("git"):
        return _clone_via_worktree(src, clone_path, base_branch)
    return _clone_via_copytree(src, clone_path)


def _clone_via_worktree(src: Path, clone_path: Path, base_branch: str) -> ClonedRepo:
    """Materialise via ``git worktree add`` and return the base SHA."""
    head = _git(["rev-parse", "HEAD"], cwd=src)
    if head.returncode != 0:
        log.warning("git rev-parse HEAD failed in %s: %s; falling back to copytree", src, head.stderr.strip())
        return _clone_via_copytree(src, clone_path)
    base_sha = head.stdout.strip()

    # Detect the source repo's current branch (best-effort).
    branch_proc = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=src)
    actual_branch = branch_proc.stdout.strip() if branch_proc.returncode == 0 else base_branch
    if actual_branch == "HEAD":  # detached
        actual_branch = base_branch

    # Detached worktree at the source HEAD. We do *not* create a branch yet —
    # propose_patch creates the reflect/<slug> branch later from this SHA.
    add = _git(
        ["worktree", "add", "--detach", str(clone_path), base_sha],
        cwd=src,
    )
    if add.returncode != 0:
        log.warning(
            "git worktree add failed (%s); falling back to copytree", add.stderr.strip()
        )
        return _clone_via_copytree(src, clone_path)

    log.info("prepared git worktree at %s (base %s)", clone_path, base_sha[:8])
    return ClonedRepo(
        clone_path=clone_path,
        base_sha=base_sha,
        base_branch=actual_branch,
        is_git=True,
    )


def _clone_via_copytree(src: Path, clone_path: Path) -> ClonedRepo:
    shutil.copytree(src, clone_path, ignore=_ignore, symlinks=False)
    log.info("prepared copytree clone at %s", clone_path)
    return ClonedRepo(
        clone_path=clone_path,
        base_sha=None,
        base_branch=None,
        is_git=False,
    )


def cleanup_clone(clone: ClonedRepo, *, source_repo: str | Path = "/repo") -> None:
    """Remove a previously prepared clone.

    For git worktrees we call ``git worktree remove --force`` so the source
    repo's worktree list stays consistent.
    """
    if not clone.clone_path.exists():
        return
    if clone.is_git:
        src = Path(source_repo)
        if _is_git_checkout(src) and shutil.which("git"):
            _git(["worktree", "remove", "--force", str(clone.clone_path)], cwd=src)
            if not clone.clone_path.exists():
                return
    shutil.rmtree(clone.clone_path, ignore_errors=True)
