"""Tests for the integrator (deterministic checks)."""

from genai.agents.fullstack.integrator import integrate
from genai.agents.fullstack.scaffolder import scaffold
from genai.agents.fullstack.spec import ProjectMeta, ProjectSpec
from genai.sandbox.workspace import Workspace, new_project_id


def test_integrate_fresh_skeleton_is_ok(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    scaffold(ws, ProjectSpec(meta=ProjectMeta(name="x", title="X")))
    report = integrate(ws.root)
    assert report.ok, f"missing={report.missing} issues={report.issues}"


def test_integrate_flags_missing_files(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    scaffold(ws, ProjectSpec(meta=ProjectMeta(name="x", title="X")))
    (ws.root / "frontend" / "package.json").unlink()
    report = integrate(ws.root)
    assert not report.ok
    assert "frontend/package.json" in report.missing


def test_integrate_flags_broken_compose(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    scaffold(ws, ProjectSpec(meta=ProjectMeta(name="x", title="X")))
    (ws.root / "docker-compose.yml").write_text("services:\n  db:\n  : invalid : yaml :\n")
    report = integrate(ws.root)
    assert not report.ok
    assert any("docker-compose.yml" in i for i in report.issues)
