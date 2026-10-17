"""Tests for :func:`genai.reflection.seed_from_evals.seed_from_run`.

Builds a synthetic ``run.jsonl`` matching the shape the real eval harness
emits (``CaseResult.to_dict()``) so this test stays in lock-step with the
harness without spinning up the harness itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genai.reflection.seed_from_evals import seed_from_run
from genai.reflection.store import LessonStore


def _write_run_jsonl(run_dir: Path, run_id: str, cases: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines: list[dict] = [
        {"type": "suite_started", "run_id": run_id, "suite_name": "t", "model": "x", "started_at": "now"},
    ]
    lines.extend({"type": "case_summary", **c} for c in cases)
    lines.append({"type": "suite_summary", "run_id": run_id, "suite_name": "t", "model": "x"})
    (run_dir / "run.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def _passing_case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "agent": "codegen",
        "passed": True,
        "skipped": False,
        "submitted": True,
        "summary": "ok",
        "error": None,
        "wall_seconds": 1.0,
        "structural": {"expected_files_present": ["main.py"], "expected_files_missing": []},
        "functional": {"checks": []},
        "telemetry": {"llm_calls": 1, "total_tokens": 10},
        "project_id": case_id,
        "skip_reason": None,
    }


def _failing_structural(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "agent": "codegen",
        "passed": False,
        "skipped": False,
        "submitted": True,
        "summary": "ok",
        "error": None,
        "wall_seconds": 1.0,
        "structural": {
            "expected_files_present": [],
            "expected_files_missing": ["pyproject.toml", "tests/test_x.py"],
            "syntax_ok": False,
            "syntax_output": "SyntaxError: invalid syntax",
        },
        "functional": {"checks": []},
        "telemetry": {"llm_calls": 3, "total_tokens": 50},
        "project_id": case_id,
        "skip_reason": None,
    }


def _failing_functional(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "agent": "fullstack",
        "passed": False,
        "skipped": False,
        "submitted": True,
        "summary": "ok",
        "error": None,
        "wall_seconds": 30.0,
        "structural": {"expected_files_present": ["docker-compose.yml"], "expected_files_missing": [], "integrator_ok": True},
        "functional": {
            "checks": [
                {
                    "name": "backend_pytest",
                    "type": "pytest",
                    "required": True,
                    "exit_code": 1,
                    "passed": False,
                    "output_tail": "FAILED tests/test_auth.py::test_login - assert 401 == 200",
                    "seconds": 5.0,
                }
            ]
        },
        "telemetry": {"llm_calls": 12, "total_tokens": 4000},
        "project_id": case_id,
        "skip_reason": None,
    }


@pytest.fixture
def store(tmp_path: Path) -> LessonStore:
    return LessonStore(tmp_path / "lessons.jsonl")


def test_seeds_one_lesson_per_failure(tmp_path: Path, store: LessonStore) -> None:
    run_dir = tmp_path / "evals" / "20260501-120000-abcdef"
    _write_run_jsonl(
        run_dir,
        "20260501-120000-abcdef",
        [_passing_case("good"), _failing_structural("bad1"), _failing_functional("bad2")],
    )

    n = seed_from_run(run_dir, store)
    assert n == 2
    lessons = store.all()
    assert {ls.evidence["case_id"] for ls in lessons} == {"bad1", "bad2"}
    assert all(ls.source == "eval-failure" for ls in lessons)
    # Categories track the agent name.
    categories = {ls.category for ls in lessons}
    assert categories == {"codegen", "fullstack"}


def test_idempotent_on_second_call(tmp_path: Path, store: LessonStore) -> None:
    run_dir = tmp_path / "evals" / "run1"
    _write_run_jsonl(run_dir, "run1", [_failing_structural("a")])
    assert seed_from_run(run_dir, store) == 1
    assert seed_from_run(run_dir, store) == 0
    assert len(store.all()) == 1


def test_structural_vs_functional_get_distinct_tags(tmp_path: Path, store: LessonStore) -> None:
    run_dir = tmp_path / "evals" / "run2"
    _write_run_jsonl(run_dir, "run2", [_failing_structural("s"), _failing_functional("f")])
    seed_from_run(run_dir, store)
    by_case = {ls.evidence["case_id"]: ls for ls in store.all()}
    assert "structural" in by_case["s"].tags
    assert "functional" in by_case["f"].tags
    assert "pytest" in by_case["f"].tags


def test_skipped_cases_are_ignored(tmp_path: Path, store: LessonStore) -> None:
    run_dir = tmp_path / "evals" / "run3"
    skipped = _passing_case("x")
    skipped["skipped"] = True
    skipped["passed"] = False
    skipped["skip_reason"] = "not today"
    _write_run_jsonl(run_dir, "run3", [skipped])
    assert seed_from_run(run_dir, store) == 0


def test_missing_run_jsonl_raises(tmp_path: Path, store: LessonStore) -> None:
    with pytest.raises(FileNotFoundError):
        seed_from_run(tmp_path / "nope", store)
