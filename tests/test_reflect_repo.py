"""Tests for the reflect agent's source-isolation primitive."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from genai.agents.reflect.repo import (
    ClonedRepo,
    ReflectionUnavailable,
    cleanup_clone,
    prepare_clone,
)
from genai.sandbox.workspace import Workspace


def _hash_tree(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _have_git() -> bool:
    return shutil.which("git") is not None


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True)


# -------------------------------------------------------------------- fixtures


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace.create("reflect-test", base=tmp_path / "ws")


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    src.mkdir()
    (src / "src").mkdir()
    (src / "src" / "hello.py").write_text("print('hi')\n")
    (src / "README.md").write_text("# repo\n")
    return src


@pytest.fixture
def git_source_repo(source_repo: Path) -> Path:
    if not _have_git():
        pytest.skip("git not available")
    _git(["init", "-q", "-b", "main"], source_repo)
    _git(["config", "user.email", "t@e.x"], source_repo)
    _git(["config", "user.name", "t"], source_repo)
    _git(["add", "-A"], source_repo)
    _git(["commit", "-q", "-m", "init"], source_repo)
    return source_repo


# ---------------------------------------------------------------------- tests


def test_missing_mount_raises(tmp_path: Path, workspace: Workspace) -> None:
    with pytest.raises(ReflectionUnavailable):
        prepare_clone(workspace, repo_mount=tmp_path / "does-not-exist")


def test_copytree_clone_is_isolated(workspace: Workspace, source_repo: Path) -> None:
    before = _hash_tree(source_repo)
    clone = prepare_clone(workspace, repo_mount=source_repo)

    assert clone.clone_path == (workspace.root / "repo").resolve()
    assert clone.is_git is False
    assert clone.base_sha is None
    assert (clone.clone_path / "src" / "hello.py").read_text() == "print('hi')\n"

    # Mutate the clone — source must stay bit-identical.
    (clone.clone_path / "src" / "hello.py").write_text("print('mutated')\n")
    (clone.clone_path / "new_file.txt").write_text("x")

    after = _hash_tree(source_repo)
    assert before == after, "writes to clone leaked into source"

    cleanup_clone(clone, source_repo=source_repo)
    assert not clone.clone_path.exists()


def test_copytree_ignores_workspaces_and_caches(workspace: Workspace, source_repo: Path) -> None:
    (source_repo / "workspaces").mkdir()
    (source_repo / "workspaces" / "stale.txt").write_text("ignore me")
    (source_repo / "__pycache__").mkdir()
    (source_repo / "__pycache__" / "x.pyc").write_text("ignore me")

    clone = prepare_clone(workspace, repo_mount=source_repo)

    assert not (clone.clone_path / "workspaces").exists()
    assert not (clone.clone_path / "__pycache__").exists()


def test_git_clone_uses_worktree(workspace: Workspace, git_source_repo: Path) -> None:
    if not _have_git():
        pytest.skip("git not available")
    before = _hash_tree(git_source_repo)
    clone = prepare_clone(workspace, repo_mount=git_source_repo)

    assert clone.is_git is True
    assert clone.base_sha is not None and len(clone.base_sha) == 40
    assert clone.base_branch == "main"

    # Mutate the worktree — source working copy must be untouched.
    (clone.clone_path / "src" / "hello.py").write_text("print('mutated')\n")
    after = _hash_tree(git_source_repo)
    assert before == after

    # Cleanup removes the worktree and de-registers it.
    cleanup_clone(clone, source_repo=git_source_repo)
    assert not clone.clone_path.exists()
    listing = _git(["worktree", "list"], git_source_repo).stdout
    assert str(clone.clone_path) not in listing


def test_prepare_clone_is_idempotent(workspace: Workspace, source_repo: Path) -> None:
    first = prepare_clone(workspace, repo_mount=source_repo)
    (first.clone_path / "src" / "hello.py").write_text("dirty\n")
    # Second call wipes the previous clone.
    second = prepare_clone(workspace, repo_mount=source_repo)
    assert isinstance(second, ClonedRepo)
    assert (second.clone_path / "src" / "hello.py").read_text() == "print('hi')\n"
