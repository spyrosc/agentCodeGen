"""Tests for the deterministic scaffolder."""

import ast
import configparser
import json
import re

import pytest
import yaml

from genai.agents.fullstack.scaffolder import scaffold
from genai.agents.fullstack.spec import ProjectMeta, ProjectSpec
from genai.sandbox.workspace import Workspace, new_project_id


@pytest.fixture()
def workspace(tmp_path):
    return Workspace.create(new_project_id("t"), base=tmp_path)


@pytest.fixture()
def spec():
    return ProjectSpec(meta=ProjectMeta(name="todo_app", title="Todo App", description="tracks todos"))


def test_scaffold_writes_expected_files(workspace, spec):
    res = scaffold(workspace, spec)
    written = set(res.files_written)
    for required in (
        "docker-compose.yml",
        ".env.example",
        "README.md",
        "backend/Dockerfile",
        "backend/pyproject.toml",
        "backend/app/main.py",
        "backend/app/db.py",
        "backend/app/auth.py",
        "backend/app/models.py",
        "backend/app/routers/auth.py",
        "backend/alembic.ini",
        "backend/alembic/env.py",
        "backend/tests/conftest.py",
        "frontend/Dockerfile",
        "frontend/package.json",
        "frontend/vite.config.ts",
        "frontend/src/App.tsx",
        "frontend/src/lib/api.ts",
        "frontend/src/lib/auth.ts",
        "project_spec.json",
    ):
        assert required in written, required


def test_compose_yaml_is_valid(workspace, spec):
    scaffold(workspace, spec)
    data = yaml.safe_load((workspace.root / "docker-compose.yml").read_text())
    services = data["services"]
    assert {"db", "backend", "frontend"} <= set(services)
    assert services["db"]["image"] == "postgres:16-alpine"


def test_compose_uses_configured_postgres_image(workspace, spec):
    scaffold(workspace, spec, postgres_image="registry.local:5000/pg/postgres:17.2@sha256:abc123")
    data = yaml.safe_load((workspace.root / "docker-compose.yml").read_text())
    assert data["services"]["db"]["image"] == "registry.local:5000/pg/postgres:17.2@sha256:abc123"


@pytest.mark.parametrize("image", ["", "postgres:16\n    privileged: true", "postgres 16", "-postgres"])
def test_rejects_unsafe_postgres_image(workspace, spec, image):
    with pytest.raises(ValueError, match="invalid postgres image"):
        scaffold(workspace, spec, postgres_image=image)


def test_db_healthcheck_targets_the_app_database(workspace, spec):
    # Without -d, pg_isready probes a database named after the user and Postgres logs FATAL.
    scaffold(workspace, spec)
    data = yaml.safe_load((workspace.root / "docker-compose.yml").read_text())
    check = " ".join(data["services"]["db"]["healthcheck"]["test"])
    assert "-d ${POSTGRES_DB:-todo_app}" in check


def test_backend_app_package_is_importable_by_alembic(workspace, spec):
    # The `alembic` console script doesn't add the cwd to sys.path, but env.py imports `app`.
    scaffold(workspace, spec)
    ini = configparser.ConfigParser()
    ini.read(workspace.root / "backend/alembic.ini")
    assert ini["alembic"]["prepend_sys_path"] == "."
    dockerfile = (workspace.root / "backend/Dockerfile").read_text()
    assert "PYTHONPATH=/app" in dockerfile


def _settings_default(workspace, field: str):
    tree = ast.parse((workspace.root / "backend/app/config.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == field:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{field} not found in generated config.py")


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        ("http://localhost:5173", True),
        ("http://localhost:5174", True),  # port picked by VS Code forwarding
        ("http://127.0.0.1:5173", True),
        ("http://localhost", True),
        ("http://localhost.evil.com", False),
        ("http://evil.com/?http://localhost:5173", False),
        ("http://192.168.1.20:5173", False),
    ],
)
def test_backend_cors_accepts_any_local_port_only(workspace, spec, origin, allowed):
    # Starlette matches allow_origin_regex with fullmatch.
    scaffold(workspace, spec)
    regex = _settings_default(workspace, "cors_origin_regex")
    assert bool(re.fullmatch(regex, origin)) is allowed
    main = (workspace.root / "backend/app/main.py").read_text()
    assert "allow_origin_regex=settings.cors_origin_regex or None" in main


def test_package_json_uses_project_name(workspace, spec):
    scaffold(workspace, spec)
    pkg = json.loads((workspace.root / "frontend/package.json").read_text())
    assert pkg["name"] == "todo_app-frontend"


def test_template_literal_escape_in_api_ts(workspace, spec):
    scaffold(workspace, spec)
    content = (workspace.root / "frontend/src/lib/api.ts").read_text()
    # Must be a JS template literal, not a Python placeholder.
    assert "`Bearer ${token}`" in content


def test_alembic_mako_preserves_placeholders(workspace, spec):
    scaffold(workspace, spec)
    content = (workspace.root / "backend/alembic/script.py.mako").read_text()
    assert "${message}" in content
    assert "${up_revision}" in content


def test_main_has_router_marker(workspace, spec):
    scaffold(workspace, spec)
    content = (workspace.root / "backend/app/main.py").read_text()
    assert "# >>> AUTOGEN ROUTERS <<<" in content


def test_app_tsx_has_route_marker(workspace, spec):
    scaffold(workspace, spec)
    content = (workspace.root / "frontend/src/App.tsx").read_text()
    assert "{/* >>> AUTOGEN ROUTES <<< */}" in content


def test_spec_json_round_trips(workspace, spec):
    scaffold(workspace, spec)
    parsed = ProjectSpec.model_validate_json((workspace.root / "project_spec.json").read_text())
    assert parsed.meta.name == "todo_app"
