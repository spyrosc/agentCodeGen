"""Seed the :class:`~genai.reflection.store.LessonStore` from eval-run artefacts.

The eval harness writes ``workspaces/_evals/<run_id>/run.jsonl`` containing
``case_summary`` records (one per case) with structural / functional /
telemetry detail. Each *failed* case becomes one :class:`Lesson` with
``source="eval-failure"`` and a structured ``evidence`` dict.

Idempotent on ``(run_id, case_id)``: re-seeding the same run is a no-op.

Can be invoked programmatically (used by :mod:`genai.evals.harness` when
``REFLECT_LESSONS_AUTOSEED=true``) or as a CLI:

    python -m genai.reflection.seed_from_evals <run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from .store import Lesson, LessonStore

log = get_logger("reflection.seed_from_evals")


def _parse_run_jsonl(run_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(run_id, case_summaries)`` for a single eval run dir."""
    jsonl = run_dir / "run.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"no run.jsonl in {run_dir}")
    run_id = run_dir.name
    summaries: list[dict[str, Any]] = []
    with jsonl.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "suite_started" and obj.get("run_id"):
                run_id = obj["run_id"]
            if obj.get("type") == "case_summary":
                summaries.append(obj)
    return run_id, summaries


def _summarise_failure(case: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
    """Pull ``(problem, evidence, tags)`` out of a failing case_summary record."""
    parts: list[str] = []
    tags: list[str] = [case.get("agent", "unknown"), case.get("case_id", "unknown")]
    evidence: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "agent": case.get("agent"),
    }

    if case.get("error"):
        first_line = (case["error"].splitlines() or [""])[0][:200]
        parts.append(f"error: {first_line}")
        evidence["error"] = first_line

    structural = case.get("structural") or {}
    if structural.get("expected_files_missing"):
        missing = list(structural["expected_files_missing"])
        parts.append(f"missing files: {', '.join(missing[:5])}")
        evidence["missing_files"] = missing
        tags.append("structural")
    if structural.get("integrator_ok") is False:
        parts.append("integrator failed")
        evidence["integrator_issues"] = structural.get("integrator_issues", [])
        evidence["integrator_missing"] = structural.get("integrator_missing", [])
        tags.append("integrator")
    if structural.get("syntax_ok") is False:
        parts.append("syntax check failed")
        tags.append("syntax")

    functional = (case.get("functional") or {}).get("checks") or []
    failed_required = [c for c in functional if c.get("required") and not c.get("passed")]
    if failed_required:
        names = [c.get("name", "?") for c in failed_required]
        parts.append(f"failed checks: {', '.join(names)}")
        evidence["failed_checks"] = [
            {
                "name": c.get("name"),
                "type": c.get("type"),
                "exit_code": c.get("exit_code"),
                "tail": (c.get("output_tail") or "")[-500:],
            }
            for c in failed_required
        ]
        tags.append("functional")
        for c in failed_required:
            ctype = c.get("type")
            if ctype:
                tags.append(ctype)

    if not parts:
        parts.append("case did not pass; cause unknown from summary")

    problem = "; ".join(parts)
    # de-dup tags preserving order
    seen: set[str] = set()
    unique_tags = []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            unique_tags.append(t)
    return problem, evidence, unique_tags


def seed_from_run(run_dir: Path | str, store: LessonStore) -> int:
    """Append one :class:`Lesson` per failed case in ``run_dir``.

    Returns the number of lessons newly appended (deduped against the
    existing store via :meth:`LessonStore.append`).
    """
    run_dir = Path(run_dir)
    run_id, summaries = _parse_run_jsonl(run_dir)
    appended = 0
    for case in summaries:
        if case.get("skipped") or case.get("passed"):
            continue
        problem, evidence, tags = _summarise_failure(case)
        evidence["run_id"] = run_id
        category = case.get("agent", "unknown")
        fix = (
            "Investigate the failing case in the eval harness output, "
            "reproduce locally, and submit a patch via the reflect agent. "
            "See evidence.failed_checks for the failing command output."
        )
        lesson = Lesson(
            category=category,
            problem=problem,
            fix=fix,
            tags=tags,
            source="eval-failure",
            evidence=evidence,
        )
        if store.append(lesson):
            appended += 1
    log.info("seeded %d lessons from run %s", appended, run_id)
    return appended


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = Path(args[0])
    store = LessonStore.from_settings()
    n = seed_from_run(run_dir, store)
    print(f"appended {n} lessons from {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
