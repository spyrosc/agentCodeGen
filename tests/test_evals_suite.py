"""Tests for the eval suite YAML loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from genai.evals import load_suite
from genai.evals.cases import EvalSuite


def test_load_suite_minimal(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(
        textwrap.dedent(
            """
            name: smoke
            cases:
              - id: c1
                agent: codegen
                prompt: do a thing
                expect_files:
                  - pyproject.toml
                functional:
                  - name: t
                    type: pytest
            """
        )
    )
    suite = load_suite(p)
    assert isinstance(suite, EvalSuite)
    assert suite.name == "smoke"
    assert len(suite.cases) == 1
    case = suite.cases[0]
    assert case.id == "c1"
    assert case.agent == "codegen"
    assert case.expect_files == ["pyproject.toml"]
    assert case.functional[0].type == "pytest"
    assert case.functional[0].required is True


def test_load_suite_rejects_duplicate_ids(tmp_path: Path) -> None:
    p = tmp_path / "dup.yaml"
    p.write_text(
        textwrap.dedent(
            """
            name: dup
            cases:
              - id: x
                agent: codegen
                prompt: a
              - id: x
                agent: codegen
                prompt: b
            """
        )
    )
    with pytest.raises(Exception) as exc:
        load_suite(p)
    assert "duplicate" in str(exc.value).lower()


def test_load_suite_rejects_unsafe_id(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        textwrap.dedent(
            """
            name: bad
            cases:
              - id: "../etc"
                agent: codegen
                prompt: x
            """
        )
    )
    with pytest.raises(Exception):  # noqa: B017
        load_suite(p)


def test_repo_suites_parse() -> None:
    """The shipped seed suites must always be parseable."""
    root = Path(__file__).resolve().parent.parent
    for name in ("codegen_smoke.yaml", "fullstack_smoke.yaml"):
        suite = load_suite(root / "evals" / "suites" / name)
        assert suite.cases, f"{name} has no cases"
        for case in suite.cases:
            assert case.prompt.strip()
