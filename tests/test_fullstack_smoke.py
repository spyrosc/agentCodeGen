"""Tests for the runtime CRUD smoke probe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from genai.agents.fullstack.compose import run_compose_smoke
from genai.agents.fullstack.smoke import (
    build_plan,
    run_smoke_crud,
)
from genai.agents.fullstack.spec import (
    AuthSpec,
    Entity,
    EntityField,
    ProjectMeta,
    ProjectSpec,
)
from genai.sandbox.base import ExecResult


@dataclass
class _FakeSandbox:
    script: list[ExecResult]
    calls: list[tuple[str, str, int]]

    def exec(self, project_id: str, command: str, timeout_s: int) -> ExecResult:
        self.calls.append((project_id, command, timeout_s))
        if self.script:
            return self.script.pop(0)
        return ExecResult(exit_code=0, output="")


def _sb(*results: ExecResult) -> _FakeSandbox:
    return _FakeSandbox(script=list(results), calls=[])


def _spec_one_entity() -> ProjectSpec:
    return ProjectSpec(
        meta=ProjectMeta(name="bookmarks", title="Bookmarks"),
        auth=AuthSpec(enabled=True),
        entities=[
            Entity(
                name="Bookmark",
                table_name="bookmarks",
                owned_by_user=True,
                fields=[
                    EntityField(name="url", type="str"),
                    EntityField(name="title", type="str"),
                    EntityField(name="tags", type="str"),
                    EntityField(name="created_at", type="datetime"),  # server-only
                ],
            )
        ],
    )


# --------------------------------------------------------------------- plan


def test_build_plan_includes_auth_and_candidate_paths():
    plan = build_plan(_spec_one_entity(), backend_port=8000)
    assert plan["base_url"] == "http://127.0.0.1:8000"
    assert plan["auth"] is not None
    assert plan["auth"]["login_form"]["username"] == plan["auth"]["credentials"]["email"]
    ent = plan["entities"][0]
    assert ent["name"] == "Bookmark"
    # table_name first, then PascalCase->snake variants
    assert ent["candidate_paths"][0] == "/bookmarks"
    assert "/bookmark" in ent["candidate_paths"]
    # server-only field stripped from payload
    assert "created_at" not in ent["create_payload"]
    assert ent["create_payload"]["url"] == "https://example.com"


def test_build_plan_no_auth_when_disabled():
    spec = _spec_one_entity()
    spec = spec.model_copy(update={"auth": AuthSpec(enabled=False)})
    plan = build_plan(spec, backend_port=8000)
    assert plan["auth"] is None


def test_build_plan_handles_camelcase_entity():
    spec = ProjectSpec(
        meta=ProjectMeta(name="x", title="X"),
        entities=[Entity(name="BlogPost", fields=[EntityField(name="title")])],
    )
    paths = build_plan(spec, backend_port=8000)["entities"][0]["candidate_paths"]
    assert "/blog_posts" in paths


# --------------------------------------------------------------------- runner


def test_run_smoke_crud_skipped_when_no_entities(tmp_path: Path):
    spec = ProjectSpec(meta=ProjectMeta(name="x", title="X"))
    rep = run_smoke_crud(
        _sb(), tmp_path, "proj", "proj", spec, backend_port=8000
    )
    assert rep.skipped and rep.ok


def test_run_smoke_crud_materializes_plan_and_runner(tmp_path: Path):
    sb = _sb(ExecResult(0, "PASSED: [\"Bookmark\"]\nFAILED: []\nCRUD SMOKE OK"))
    rep = run_smoke_crud(
        sb, tmp_path, "proj-1", "proj-1", _spec_one_entity(),
        backend_port=8000,
    )
    assert rep.ok
    plan_file = tmp_path / ".smoke" / "smoke_plan.json"
    runner_file = tmp_path / ".smoke" / "smoke_crud.py"
    assert plan_file.exists() and runner_file.exists()
    plan = json.loads(plan_file.read_text())
    assert plan["entities"][0]["name"] == "Bookmark"
    # One sandbox call, chained with `;`, that contains both cp + exec.
    assert len(sb.calls) == 1
    cmd = sb.calls[0][1]
    assert "docker compose -p proj-1 cp" in cmd
    assert "smoke_plan.json" in cmd and "smoke_crud.py" in cmd
    assert "exec -T" in cmd and "SMOKE_PLAN_FILE=/tmp/smoke_plan.json" in cmd


def test_run_smoke_crud_parses_failed_entities(tmp_path: Path):
    out = (
        "[entity] Bookmark\n"
        "  POST /bookmarks -> 500\n"
        "  FAIL create: all candidate paths failed\n"
        "PASSED: []\n"
        "FAILED: [\"Bookmark\"]\n"
        "CRUD SMOKE FAILED\n"
    )
    rep = run_smoke_crud(
        _sb(ExecResult(1, out)),
        tmp_path, "p", "p", _spec_one_entity(), backend_port=8000,
    )
    assert not rep.ok
    assert rep.entities_failed == ["Bookmark"]
    assert "Bookmark" in rep.log


# --------------------------------------------------------------------- wiring


def test_compose_runs_smoke_after_healthy(tmp_path: Path):
    """When health=200 + spec is supplied, compose runs the CRUD probe."""
    events: list[tuple[str, dict]] = []
    sb = _sb(
        ExecResult(0, ""),                       # pre-emptive down
        ExecResult(0, "ok"),                     # up -d --build
        ExecResult(0, "backend\trunning\tUp"),  # ps
        ExecResult(0, "200"),                    # health
        ExecResult(0, "PASSED: [\"Bookmark\"]\nFAILED: []\nCRUD SMOKE OK"),  # smoke runner
        ExecResult(0, ""),                       # teardown
    )
    report = run_compose_smoke(
        sb, "proj-s", backend_port=8000,
        on_event=lambda k, p: events.append((k, p)),
        spec=_spec_one_entity(),
        workspace_root=tmp_path,
    )
    assert report.ok
    assert report.smoke is not None and report.smoke.ok
    kinds = [k for k, _ in events]
    assert "smoke_crud_started" in kinds and "smoke_crud_ok" in kinds


def test_compose_demotes_ok_when_crud_fails(tmp_path: Path):
    events: list[tuple[str, dict]] = []
    sb = _sb(
        ExecResult(0, ""),
        ExecResult(0, "ok"),
        ExecResult(0, "backend\trunning\tUp"),
        ExecResult(0, "200"),
        ExecResult(1, "FAIL create: /bookmarks -> 500\nPASSED: []\nFAILED: [\"Bookmark\"]\nCRUD SMOKE FAILED"),
        ExecResult(0, ""),                       # teardown
    )
    report = run_compose_smoke(
        sb, "proj-f", backend_port=8000,
        on_event=lambda k, p: events.append((k, p)),
        spec=_spec_one_entity(),
        workspace_root=tmp_path,
    )
    assert not report.ok, "stack health was ok but CRUD failed, must demote"
    assert "[crud smoke output]" in report.log
    assert "Bookmark" in report.log
    kinds = [k for k, _ in events]
    assert "compose_up_ok" in kinds  # /health passed first
    assert "smoke_crud_failed" in kinds


def test_compose_skips_smoke_when_spec_missing(tmp_path: Path):
    """Old call-sites without spec/workspace still work and don't trigger smoke."""
    sb = _sb(
        ExecResult(0, ""),
        ExecResult(0, "ok"),
        ExecResult(0, "backend\trunning\tUp"),
        ExecResult(0, "200"),
        ExecResult(0, ""),
    )
    report = run_compose_smoke(sb, "proj-q", backend_port=8000)
    assert report.ok and report.smoke is None
    # down, up, ps, health, logs scan, teardown: no smoke step.
    assert len(sb.calls) == 6
    assert not any("smoke_crud" in cmd for _pid, cmd, _t in sb.calls)


@pytest.mark.parametrize(
    "field_type,expected_substr",
    [
        ("str", "smoke"),
        ("int", 1),
        ("bool", True),
        ("date", "2026-01-01"),
        ("datetime", "2026-01-01T00:00:00"),
    ],
)
def test_sample_value_per_type(field_type, expected_substr):
    spec = ProjectSpec(
        meta=ProjectMeta(name="x", title="X"),
        entities=[Entity(name="Item", fields=[EntityField(name="value", type=field_type)])],
    )
    payload = build_plan(spec, backend_port=8000)["entities"][0]["create_payload"]
    val = payload["value"]
    if isinstance(expected_substr, str):
        assert expected_substr in str(val)
    else:
        assert val == expected_substr
