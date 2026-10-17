"""Tools the reflect agent can call.

Three concerns are layered here:

1. **Path enforcement.** Every write goes through :func:`_check_write_path`
   which enforces the allow / deny lists from
   :class:`~genai.core.config.ReflectSettings` and the per-run write
   counter. The wrapped tools also reuse
   :class:`~genai.agents._common.fs_tools.ScopedWorkspace` so directory
   traversal outside the clone is impossible (already proven by
   ``tests/test_fs_tools_scope.py``).

2. **Sandbox plumbing.** Test / lint / git commands run in the *sandbox*
   container, with cwd inside the clone (``cd repo && \u2026``). The reflect
   workspace's project_id is the bind-mount target.

3. **Eval-harness integration.** ``read_failed_case`` reads existing
   ``run.jsonl`` artefacts so the agent has a concrete failure to chase;
   ``run_eval_gate`` re-runs the suite against the patched clone for
   non-regression validation.

The terminal :func:`propose_patch` emits a ``patch_proposed`` event and
flips ``state["submitted"] = True`` \u2014 mirroring the codegen agent's
``submit_project``.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool, tool

from ...core.config import ReflectSettings, get_settings
from ...core.logging import get_logger
from ...core.text import truncate_middle
from ...reflection.store import Lesson, LessonStore
from ...sandbox.base import Sandbox
from ...sandbox.workspace import Workspace
from .._common.fs_tools import build_fs_tools, emit, scope_to
from .repo import ClonedRepo

log = get_logger("agents.reflect.tools")

# Output truncation for git / sandbox return values handed back to the LLM.
_TOOL_OUTPUT_LIMIT = 8_000


# --------------------------------------------------------------- exceptions


class OperationCap(RuntimeError):
    """Per-run write cap exceeded; caught by the reflect agent loop."""


# --------------------------------------------------------------- helpers


def _truncate(text: str) -> str:
    return truncate_middle(text, _TOOL_OUTPUT_LIMIT)


def _normalise_rel(path: str) -> str:
    """Drop leading ``/`` and ``./`` to compare against allow/deny prefixes."""
    p = (path or "").strip().lstrip("/\\")
    while p.startswith("./"):
        p = p[2:]
    return p


def _check_write_path(path: str, cfg: ReflectSettings) -> None:
    """Raise :class:`PermissionError` when ``path`` violates allow / deny rules."""
    rel = _normalise_rel(path)
    if not rel:
        raise PermissionError("write path is empty")
    if ".." in Path(rel).parts:
        raise PermissionError(f"path {path!r}: '..' segments are not allowed")
    for denied in cfg.denied_paths:
        if rel == denied.rstrip("/") or rel.startswith(denied):
            raise PermissionError(f"path {path!r} hits denylist entry {denied!r}")
    for allowed in cfg.allowed_paths:
        if rel == allowed.rstrip("/") or rel.startswith(allowed):
            return
    raise PermissionError(
        f"path {path!r} is outside the allowlist {cfg.allowed_paths}"
    )


def _slugify(text: str, *, max_len: int = 32) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return s[:max_len] or "patch"


def _git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False
    )


def _git_or_raise(args: list[str], *, cwd: Path) -> str:
    proc = _git(args, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _diff_stat(diff_text: str) -> tuple[int, int, int]:
    """Return ``(files_changed, additions, deletions)`` from a unified diff."""
    files = additions = deletions = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            files += 1
        elif line.startswith("+++ ") or line.startswith("--- "):
            continue
        elif line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return files, additions, deletions


def _changed_files(diff_text: str) -> list[str]:
    """Extract the list of changed paths from ``git diff`` output."""
    out: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            # ``diff --git a/<path> b/<path>``
            parts = line.split(" ")
            if len(parts) >= 4 and parts[3].startswith("b/"):
                out.append(parts[3][2:])
    return out


def _evals_root(workspace_root: str) -> Path:
    return Path(workspace_root) / "_evals"


# --------------------------------------------------------------- builder


def build_reflect_tools(
    workspace: Workspace,
    clone: ClonedRepo,
    sandbox: Sandbox,
    *,
    lesson_store: LessonStore,
    llm: Any,
    on_event: Callable[[str, dict], None] | None = None,
) -> tuple[list[StructuredTool], dict]:
    """Return ``(tools, state)`` for the reflect agent.

    ``state`` is the shared scratchpad the tools mutate so ``propose_patch``
    can verify preconditions (tests pass, lint pass, eval gate ran, write
    counter under cap, \u2026).
    """
    cfg = get_settings().reflect
    sandbox_cfg = get_settings().sandbox

    # Path-safe FS view rooted at the clone.
    scoped = scope_to(workspace, "repo")

    state: dict[str, Any] = {
        "submitted": False,
        "summary": None,
        "writes": 0,
        "tests_pass": None,  # None | True | False
        "lint_pass": None,
        "eval_gate": None,  # None | dict
        "patch": None,  # filled by propose_patch on success
        "denied_attempts": [],  # list of paths the agent tried to write to
    }

    # ---------- shared FS surface (read-only ones are reused as-is)

    base_fs = build_fs_tools(scoped, on_event=on_event, label="reflect")
    fs_by_name = {t.name: t for t in base_fs}
    list_files = fs_by_name["list_files"]
    read_file = fs_by_name["read_file"]
    inner_write = fs_by_name["write_file"]
    inner_delete = fs_by_name["delete_file"]

    # ---------- write path enforcement

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file inside the platform-source clone.

        Subject to the reflect allow/deny lists. Returns a structured
        permission-denied message the agent can react to.
        """
        if state["writes"] >= cfg.max_write_calls_per_run:
            emit(on_event, "write_denied", path=path, reason="write-cap")
            raise OperationCap(
                f"write cap of {cfg.max_write_calls_per_run} reached for this run"
            )
        try:
            _check_write_path(path, cfg)
        except PermissionError as exc:
            state["denied_attempts"].append(path)
            emit(on_event, "write_denied", path=path, reason=str(exc))
            return f"PERMISSION_DENIED: {exc}"
        state["writes"] += 1
        # Invalidate cached test/lint/gate verdicts \u2014 they no longer reflect
        # what's on disk.
        state["tests_pass"] = None
        state["lint_pass"] = None
        state["eval_gate"] = None
        return inner_write.invoke({"path": path, "content": content})

    @tool
    def delete_file(path: str) -> str:
        """Delete a single file inside the clone, subject to allow/deny lists."""
        try:
            _check_write_path(path, cfg)
        except PermissionError as exc:
            state["denied_attempts"].append(path)
            emit(on_event, "write_denied", path=path, reason=str(exc))
            return f"PERMISSION_DENIED: {exc}"
        state["tests_pass"] = None
        state["lint_pass"] = None
        state["eval_gate"] = None
        return inner_delete.invoke({"path": path})

    # ---------- sandbox commands (cwd = clone root)

    def _sandbox_cd(command: str, *, timeout: int) -> tuple[int, str]:
        full = f"cd repo && {command}"
        emit(on_event, "run_in_sandbox", command=full, timeout=timeout)
        res = sandbox.exec(workspace.project_id, full, timeout)
        emit(on_event, "run_in_sandbox_done", exit=res.exit_code)
        return res.exit_code, res.output

    @tool
    def run_tests_repo() -> str:
        """Run the platform's pytest suite against the patched clone."""
        emit(on_event, "run_tests", command="pytest")
        cmd = "PYTHONPATH=src pytest tests/ -q --maxfail=5 --color=no"
        exit_code, output = _sandbox_cd(cmd, timeout=sandbox_cfg.default_timeout_s)
        state["tests_pass"] = exit_code == 0
        emit(on_event, "run_tests_done", exit=exit_code)
        return f"exit={exit_code}\n{_truncate(output)}"

    @tool
    def run_lint_repo() -> str:
        """Run ``ruff check`` against the patched clone."""
        emit(on_event, "run_lint", command="ruff check src tests scripts")
        cmd = "ruff check src tests scripts --output-format=concise"
        exit_code, output = _sandbox_cd(cmd, timeout=120)
        state["lint_pass"] = exit_code == 0
        emit(on_event, "run_lint_done", exit=exit_code)
        return f"exit={exit_code}\n{_truncate(output)}"

    @tool
    def run_in_sandbox_repo(command: str, timeout: int = 0) -> str:
        """Run an arbitrary shell command in the sandbox at the clone root.

        Use sparingly \u2014 prefer ``run_tests_repo`` and ``run_lint_repo`` for
        the supported validation paths.
        """
        t = timeout or sandbox_cfg.default_timeout_s
        exit_code, output = _sandbox_cd(command, timeout=t)
        return f"exit={exit_code}\n{_truncate(output)}"

    # ---------- git status / diff (run on the host \u2014 fast)

    @tool
    def git_status() -> str:
        """Return ``git status --short`` for the clone (or 'not a git repo')."""
        if not clone.is_git:
            return "<clone is not a git checkout; status unavailable>"
        proc = _git(["status", "--short"], cwd=clone.clone_path)
        return _truncate(proc.stdout or "<clean>")

    @tool
    def git_diff(path: str = "") -> str:
        """Return ``git diff`` (optionally limited to ``path``) for the clone."""
        if not clone.is_git:
            return _diff_via_diff_command(clone)
        args = ["--no-pager", "diff", "--", path] if path else ["--no-pager", "diff"]
        proc = _git(args, cwd=clone.clone_path)
        return _truncate(proc.stdout or "<no diff>")

    # ---------- eval-harness inputs (read-only)

    @tool
    def list_eval_runs(limit: int = 10) -> str:
        """List recent eval runs from ``<workspace_root>/_evals/``."""
        root = _evals_root(sandbox_cfg.workspace_root)
        if not root.exists():
            return "<no eval runs>"
        runs = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)[:limit]
        out_lines: list[str] = []
        for r in runs:
            summary = _read_suite_summary(r)
            if summary:
                out_lines.append(
                    f"{r.name}\tpassed={summary.get('passed')}"
                    f" failed={summary.get('failed')}"
                    f" skipped={summary.get('skipped')}"
                    f" suite={summary.get('suite_name')}"
                )
            else:
                out_lines.append(f"{r.name}\t<no suite_summary record>")
        return "\n".join(out_lines) if out_lines else "<no eval runs>"

    @tool
    def read_eval_run(run_id: str) -> str:
        """Return the parsed ``suite_summary`` for an eval run."""
        run_dir = _evals_root(sandbox_cfg.workspace_root) / run_id
        if not run_dir.exists():
            return f"<no such eval run: {run_id}>"
        summary = _read_suite_summary(run_dir)
        if not summary:
            return f"<no suite_summary in {run_id}>"
        # Strip per-case "cases" detail \u2014 use read_failed_case for that.
        compact = {k: v for k, v in summary.items() if k != "cases"}
        compact["case_ids"] = [c.get("case_id") for c in summary.get("cases", [])]
        return json.dumps(compact, indent=2)

    @tool
    def read_failed_case(run_id: str, case_id: str) -> str:
        """Return the case_summary record + last events for a single case."""
        run_dir = _evals_root(sandbox_cfg.workspace_root) / run_id
        jsonl = run_dir / "run.jsonl"
        if not jsonl.exists():
            return f"<no such eval run: {run_id}>"
        case_summary: dict | None = None
        events: list[dict] = []
        for raw in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "case_summary" and obj.get("case_id") == case_id:
                case_summary = obj
            elif obj.get("type") == "event" and obj.get("case_id") == case_id:
                events.append({"kind": obj.get("kind"), "data": obj.get("data")})
        if not case_summary:
            return f"<no such case in run: {case_id}>"
        compact = {
            "case_summary": case_summary,
            "last_events": events[-20:],
        }
        return _truncate(json.dumps(compact, indent=2, default=str))

    # ---------- lessons

    @tool
    def query_lessons(query: str = "", category: str = "") -> str:
        """Search the lessons store for related past failures."""
        hits = lesson_store.search(query, category=category or None, k=cfg.lessons_top_k)
        if not hits:
            return "<no lessons matched>"
        return json.dumps(
            [
                {
                    "category": h.category,
                    "problem": h.problem,
                    "fix": h.fix,
                    "tags": h.tags,
                    "source": h.source,
                }
                for h in hits
            ],
            indent=2,
        )

    @tool
    def record_lesson(category: str, problem: str, fix: str, tags: str = "") -> str:
        """Append a new lesson to the persistent store.

        ``tags`` is a comma-separated string (LLM tools play poorly with
        list arguments across providers).
        """
        tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
        ls = Lesson(
            category=category or "uncategorised",
            problem=problem,
            fix=fix,
            tags=tag_list,
            source="reflect-agent",
        )
        appended = lesson_store.append(ls)
        emit(on_event, "lesson_recorded", id=ls.id, category=ls.category, deduped=not appended)
        return f"lesson {ls.id} {'recorded' if appended else 'already present'}"

    # ---------- eval gate

    @tool
    def run_eval_gate(case_ids: str = "", baseline_run_id: str = "") -> str:
        """Run the configured eval suite against the patched clone.

        ``case_ids`` is comma-separated. ``baseline_run_id`` is the
        previously-recorded run we compare against \u2014 omitted means "latest
        run for the same suite".

        Default mode (``eval_gate_skip_functional=True``) does structural
        scoring only \u2014 fast and no sandbox check execution \u2014 but still
        spins the agent through its real LLM. Slow either way; the agent is
        free to skip this entirely unless ``eval_gate_required`` is True.
        """
        from ...evals import load_suite, run_suite

        suite_path = clone.clone_path / cfg.eval_gate_suite
        if not suite_path.exists():
            return f"<no eval suite at {suite_path}>"
        suite = load_suite(suite_path)
        only = [c.strip() for c in case_ids.split(",") if c.strip()] or None

        emit(on_event, "eval_gate_started", suite=suite.name, only=only)
        gate_dir = workspace.root / "_evals_gate"
        gate_dir.mkdir(parents=True, exist_ok=True)
        result = run_suite(
            suite,
            base_dir=gate_dir,
            sandbox=sandbox,
            llm=llm,
            only=only,
            skip_functional=cfg.eval_gate_skip_functional,
        )

        baseline = _load_baseline(
            sandbox_cfg.workspace_root,
            suite_name=suite.name,
            run_id=baseline_run_id or None,
        )
        comparison = _compare_results(baseline, result.to_dict())

        gate_record = {
            "suite": suite.name,
            "run_id": result.run_id,
            "baseline_run_id": comparison["baseline_run_id"],
            "passed": result.passed_count,
            "failed": result.failed_count,
            "skipped": result.skipped_count,
            "regressions": comparison["regressions"],
            "fixed_cases": comparison["fixed_cases"],
            "after_per_case": comparison["after_per_case"],
        }
        state["eval_gate"] = gate_record
        emit(
            on_event,
            "eval_gate_done",
            run_id=result.run_id,
            regressions=comparison["regressions"],
            fixed_cases=comparison["fixed_cases"],
        )
        return _truncate(json.dumps(gate_record, indent=2))

    # ---------- terminal: propose_patch

    @tool
    def propose_patch(summary: str, motivation: str, baseline_run_id: str = "") -> str:
        """End the session and propose a patch.

        Refuses unless every gate has been satisfied. Returns either
        ``PATCH_PROPOSED`` (success) or ``PATCH_REJECTED: <reason>``.
        """
        rejection = _validate_patch_preconditions(
            state=state, cfg=cfg, clone=clone
        )
        if rejection:
            emit(on_event, "patch_rejected", reason=rejection)
            return f"PATCH_REJECTED: {rejection}"

        diff_text = _collect_diff(clone)
        if not diff_text.strip():
            emit(on_event, "patch_rejected", reason="working tree clean")
            return "PATCH_REJECTED: working tree clean \u2014 nothing to propose"

        files_changed, additions, deletions = _diff_stat(diff_text)
        line_count = additions + deletions
        if files_changed > cfg.max_files_changed:
            reason = f"patch touches {files_changed} files (cap {cfg.max_files_changed})"
            emit(on_event, "patch_rejected", reason=reason)
            return f"PATCH_REJECTED: {reason}"
        if line_count > cfg.max_patch_lines:
            reason = f"patch is {line_count} lines (cap {cfg.max_patch_lines})"
            emit(on_event, "patch_rejected", reason=reason)
            return f"PATCH_REJECTED: {reason}"

        # Final allowlist check on the actual changed paths (defence in depth).
        for path in _changed_files(diff_text):
            try:
                _check_write_path(path, cfg)
            except PermissionError as exc:
                emit(on_event, "patch_rejected", reason=str(exc))
                return f"PATCH_REJECTED: {exc}"

        slug = _slugify(summary)
        short = uuid.uuid4().hex[:6]
        branch = f"reflect/{slug}-{short}"
        cfg_llm = get_settings().llm

        commit_msg = _format_commit_message(
            summary=summary,
            motivation=motivation,
            base_sha=clone.base_sha,
            base_branch=clone.base_branch,
            model=cfg_llm.model,
            gate=state.get("eval_gate"),
            baseline_run_id_hint=baseline_run_id,
        )

        patch_dest = workspace.root / "patch.diff"
        patch_dest.write_text(diff_text, encoding="utf-8")

        if clone.is_git:
            try:
                _git_or_raise(["checkout", "-b", branch], cwd=clone.clone_path)
                _git_or_raise(["add", "-A"], cwd=clone.clone_path)
                _git_or_raise(
                    ["-c", "user.email=reflect@genai", "-c", "user.name=reflect-agent",
                     "commit", "-m", commit_msg],
                    cwd=clone.clone_path,
                )
            except RuntimeError as exc:
                emit(on_event, "patch_rejected", reason=str(exc))
                return f"PATCH_REJECTED: git error: {exc}"
        else:
            branch = f"<unified-diff: {patch_dest.name}>"

        patch_record = {
            "branch": branch,
            "base_sha": clone.base_sha,
            "base_branch": clone.base_branch,
            "summary": summary,
            "motivation": motivation,
            "files_changed": files_changed,
            "additions": additions,
            "deletions": deletions,
            "tests_pass": state["tests_pass"],
            "lint_pass": state["lint_pass"],
            "eval_gate": state.get("eval_gate"),
            "patch_path": str(patch_dest),
        }
        state["patch"] = patch_record
        state["summary"] = summary
        state["submitted"] = True
        emit(on_event, "patch_proposed", **patch_record)
        return "PATCH_PROPOSED"

    tools: list[StructuredTool] = [
        list_files,
        read_file,
        write_file,
        delete_file,
        run_tests_repo,
        run_lint_repo,
        run_in_sandbox_repo,
        git_status,
        git_diff,
        list_eval_runs,
        read_eval_run,
        read_failed_case,
        query_lessons,
        record_lesson,
        run_eval_gate,
        propose_patch,
    ]
    return tools, state


# --------------------------------------------------------------- internals


def _read_suite_summary(run_dir: Path) -> dict | None:
    jsonl = run_dir / "run.jsonl"
    if not jsonl.exists():
        return None
    last_summary: dict | None = None
    for raw in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "suite_summary":
            last_summary = obj
    return last_summary


def _load_baseline(
    workspace_root: str,
    *,
    suite_name: str,
    run_id: str | None,
) -> dict | None:
    root = _evals_root(workspace_root)
    if not root.exists():
        return None
    if run_id:
        return _read_suite_summary(root / run_id)
    candidates = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    for cand in candidates:
        summary = _read_suite_summary(cand)
        if summary and summary.get("suite_name") == suite_name:
            return summary
    return None


def _compare_results(baseline: dict | None, after: dict) -> dict:
    after_pass = {c["case_id"]: c.get("passed", False) for c in after.get("cases", [])}
    if not baseline:
        return {
            "baseline_run_id": None,
            "regressions": [],
            "fixed_cases": [],
            "after_per_case": after_pass,
        }
    baseline_pass = {c["case_id"]: c.get("passed", False) for c in baseline.get("cases", [])}
    regressions = sorted(
        cid for cid, passed in baseline_pass.items()
        if passed and not after_pass.get(cid, False)
    )
    fixed = sorted(
        cid for cid, passed in baseline_pass.items()
        if not passed and after_pass.get(cid, False)
    )
    return {
        "baseline_run_id": baseline.get("run_id"),
        "regressions": regressions,
        "fixed_cases": fixed,
        "after_per_case": after_pass,
    }


def _validate_patch_preconditions(
    *,
    state: dict,
    cfg: ReflectSettings,
    clone: ClonedRepo,
) -> str | None:
    """Return a rejection reason or ``None`` to proceed."""
    if state["tests_pass"] is not True:
        return "run run_tests_repo and ensure exit=0 before propose_patch"
    if state["lint_pass"] is not True:
        return "run run_lint_repo and ensure exit=0 before propose_patch"
    if cfg.eval_gate_required:
        gate = state.get("eval_gate")
        if not gate:
            return (
                "eval_gate_required=True but run_eval_gate has not been called"
            )
        if gate.get("regressions"):
            return f"eval gate detected regressions: {gate['regressions']}"
    if not clone.clone_path.exists():
        return "clone path missing"
    return None


def _collect_diff(clone: ClonedRepo) -> str:
    if clone.is_git:
        # ``git add -N`` so untracked files appear in diff output.
        _git(["add", "-N", "."], cwd=clone.clone_path)
        proc = _git(["--no-pager", "diff"], cwd=clone.clone_path)
        return proc.stdout
    return _diff_via_diff_command(clone)


def _diff_via_diff_command(clone: ClonedRepo) -> str:
    """Fallback for non-git clones: ``diff -ruN`` against the source mount."""
    src = Path(get_settings().reflect.repo_mount_path)
    if not src.exists() or not shutil.which("diff"):
        return "<unable to compute diff: no source mount or diff command>"
    proc = subprocess.run(
        [
            "diff", "-ruN",
            "--exclude=__pycache__",
            "--exclude=.venv",
            "--exclude=node_modules",
            "--exclude=workspaces",
            str(src),
            str(clone.clone_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # ``diff`` exits 1 when files differ \u2014 that's success here.
    return proc.stdout


def _format_commit_message(
    *,
    summary: str,
    motivation: str,
    base_sha: str | None,
    base_branch: str | None,
    model: str,
    gate: dict | None,
    baseline_run_id_hint: str,
) -> str:
    lines = [
        summary.strip().splitlines()[0][:72],
        "",
        motivation.strip(),
        "",
        "--- reflect-agent metadata ---",
        f"base_sha: {base_sha or 'n/a'}",
        f"base_branch: {base_branch or 'n/a'}",
        f"model: {model}",
    ]
    if gate:
        lines.append(
            f"eval_gate: suite={gate.get('suite')}"
            f" passed={gate.get('passed')}"
            f" failed={gate.get('failed')}"
            f" baseline={gate.get('baseline_run_id')}"
        )
        if gate.get("regressions"):
            lines.append(f"regressions: {', '.join(gate['regressions'])}")
        if gate.get("fixed_cases"):
            lines.append(f"fixed: {', '.join(gate['fixed_cases'])}")
    elif baseline_run_id_hint:
        lines.append(f"baseline_hint: {baseline_run_id_hint}")
    return "\n".join(lines) + "\n"


__all__ = [
    "OperationCap",
    "build_reflect_tools",
]


# Tools imported from langchain expect callable .invoke; quiet imports we keep
_ = shlex  # silence unused-import linter in case of future direct shelling
_ = time
