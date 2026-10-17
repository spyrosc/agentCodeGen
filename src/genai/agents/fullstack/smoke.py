"""Runtime CRUD smoke check for a generated fullstack project.

Once :mod:`compose` has brought the stack up and ``/health`` returns 200,
this module runs a small register → login → POST → GET → DELETE → GET
sequence against the *actual* backend, per ``owned_by_user`` entity in the
spec.

Why: a green ``/health`` only tells us the process started. It does not
catch the class of bugs we hit by hand — wrong dependency pins
(``passlib + bcrypt 4.x``), missing imports under ``from __future__ import
annotations`` that only blow up at response-validation time, ownership
filters silently returning empty lists, etc. Running real CRUD against
the same routes a user would call is the cheapest gate that surfaces
these immediately.

Strategy:

1. Build a JSON ``SmokePlan`` from the spec on the agent side.
2. Materialize plan + a small Python runner under
   ``<workspace>/.smoke/`` (visible to both agent and sandbox via the
   shared ``/workspaces`` bind-mount).
3. ``docker compose cp`` both files into the running ``backend``
   container, then ``exec`` the runner there. Talking to
   ``http://127.0.0.1:<port>`` from *inside* backend avoids any host
   port-forwarding ambiguity.
4. Parse stdout for ``CRUD SMOKE OK`` vs ``CRUD SMOKE FAILED``.

The runner script is intentionally dependency-free (only stdlib) so it
works against any generated backend.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.logging import get_logger
from ...core.text import truncate_middle
from ...sandbox.base import Sandbox
from .._common.fs_tools import emit
from .spec import Entity, EntityField, ProjectSpec

log = get_logger("agents.fullstack.smoke")

_SMOKE_DIR = ".smoke"
_PLAN_FILE = "smoke_plan.json"
_RUNNER_FILE = "smoke_crud.py"
_LOG_MAX_CHARS = 6_000


@dataclass
class SmokeReport:
    """Outcome of the runtime CRUD probe."""

    ok: bool
    skipped: bool = False
    skip_reason: str = ""
    log: str = ""
    entities_passed: list[str] = field(default_factory=list)
    entities_failed: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- runner

# Standalone script written into the backend container. Reads
# ``SMOKE_PLAN_FILE`` and walks the plan. Exits 0 iff every entity
# survives the create/list/delete/list round-trip.
_RUNNER_SOURCE = r'''"""Auto-generated CRUD smoke runner. Stdlib only."""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def _http(method, url, *, headers=None, data=None, form=False, timeout=10):
    headers = dict(headers or {})
    body = None
    if data is not None:
        if form:
            body = urllib.parse.urlencode(data).encode()
            headers.setdefault("content-type",
                               "application/x-www-form-urlencoded")
        else:
            body = json.dumps(data).encode()
            headers.setdefault("content-type", "application/json")
    req = urllib.request.Request(url, method=method,
                                 headers=headers, data=body)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            text = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status = e.code
        text = e.read().decode("utf-8", "replace")
    except Exception as e:  # pragma: no cover - defensive
        return 0, f"<transport error: {e!r}>"
    return status, text


def _try_paths(method, base, paths, *, headers, data, form, accept):
    last = (0, "<no paths tried>", "")
    for path in paths:
        st, text = _http(method, base + path,
                         headers=headers, data=data, form=form)
        print(f"  {method} {path} -> {st}")
        if st in accept:
            return path, st, text
        last = (st, text, path)
    return None, last[0], f"all candidate paths failed; last was "\
                           f"{last[2]} -> {last[0]}: {last[1][:200]}"


def main():
    plan_path = os.environ.get("SMOKE_PLAN_FILE", "/tmp/smoke_plan.json")
    with open(plan_path) as f:
        plan = json.load(f)

    base = plan["base_url"].rstrip("/")
    token = None

    auth = plan.get("auth")
    if auth:
        print("[auth] register")
        st, text = _http("POST", base + "/auth/register",
                         data=auth["credentials"])
        print(f"  POST /auth/register -> {st}")
        if st not in (200, 201, 400, 409, 422):
            print(f"FAIL: register returned {st}: {text[:300]}")
            return 2

        print("[auth] login")
        st, text = _http("POST", base + "/auth/login",
                         data=auth["login_form"], form=True)
        print(f"  POST /auth/login -> {st}")
        if st != 200:
            print(f"FAIL: login returned {st}: {text[:300]}")
            return 2
        try:
            token = json.loads(text).get("access_token")
        except Exception:
            token = None
        if not token:
            print(f"FAIL: login body missing access_token: {text[:300]}")
            return 2

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    failed = []
    passed = []
    for entity in plan["entities"]:
        name = entity["name"]
        print(f"[entity] {name}")
        # 1. POST — find a working route among candidates.
        chosen, st, text = _try_paths(
            "POST", base, entity["candidate_paths"],
            headers=headers, data=entity["create_payload"],
            form=False, accept=(200, 201),
        )
        if chosen is None:
            print(f"  FAIL create: {text}")
            failed.append(name)
            continue
        try:
            created = json.loads(text)
        except Exception:
            print(f"  FAIL create: response not JSON: {text[:200]}")
            failed.append(name)
            continue
        if not isinstance(created, dict) or "id" not in created:
            print(f"  FAIL create: no id in response: {str(created)[:200]}")
            failed.append(name)
            continue
        obj_id = created["id"]
        print(f"  created id={obj_id}")

        # 2. GET list — must include the created id.
        st, text = _http("GET", base + chosen, headers=headers)
        print(f"  GET {chosen} -> {st}")
        if st != 200:
            print(f"  FAIL list: {st}: {text[:200]}")
            failed.append(name)
            continue
        try:
            listing = json.loads(text)
        except Exception:
            print(f"  FAIL list: not JSON: {text[:200]}")
            failed.append(name)
            continue
        # Some routers wrap lists in {items: [...]}. Accept both.
        items = listing if isinstance(listing, list) \
            else listing.get("items") if isinstance(listing, dict) else None
        if not isinstance(items, list) or not any(
                isinstance(x, dict) and x.get("id") == obj_id for x in items):
            print(f"  FAIL list: id={obj_id} not present: {text[:300]}")
            failed.append(name)
            continue

        # 3. DELETE.
        st, text = _http("DELETE", f"{base}{chosen}/{obj_id}",
                         headers=headers)
        print(f"  DELETE {chosen}/{obj_id} -> {st}")
        if st not in (200, 202, 204):
            print(f"  FAIL delete: {st}: {text[:200]}")
            failed.append(name)
            continue

        # 4. GET list — id must be gone.
        st, text = _http("GET", base + chosen, headers=headers)
        print(f"  GET {chosen} -> {st} (after delete)")
        if st != 200:
            print(f"  FAIL list-after-delete: {st}: {text[:200]}")
            failed.append(name)
            continue
        try:
            listing2 = json.loads(text)
        except Exception:
            print(f"  FAIL list-after-delete: not JSON: {text[:200]}")
            failed.append(name)
            continue
        items2 = listing2 if isinstance(listing2, list) \
            else listing2.get("items") if isinstance(listing2, dict) else None
        if isinstance(items2, list) and any(
                isinstance(x, dict) and x.get("id") == obj_id for x in items2):
            print(f"  FAIL: id={obj_id} still present after DELETE")
            failed.append(name)
            continue

        print(f"  OK {name} via {chosen}")
        passed.append(name)

    print("")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    if failed:
        print("CRUD SMOKE FAILED")
        return 1
    print("CRUD SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


# --------------------------------------------------------------------- plan

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _snake(name: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def _candidate_paths(entity: Entity) -> list[str]:
    """Plausible REST base paths for an entity, ordered most-likely-first."""
    snake = _snake(entity.name)
    candidates: list[str] = []
    if entity.table_name:
        candidates.append("/" + entity.table_name)
    candidates += [
        "/" + snake + "s",
        "/" + snake,
    ]
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _sample_value(field: EntityField) -> Any:
    """Best-effort sample value for a create payload."""
    if field.default is not None:
        return field.default
    name = field.name.lower()
    t = field.type
    if t in ("str", "text"):
        if "email" in name:
            return "smoke-entity@example.com"
        if "url" in name:
            return "https://example.com"
        if "phone" in name:
            return "+10000000000"
        if "color" in name:
            return "#abcdef"
        return f"smoke-{field.name}"[:64]
    if t == "int":
        return 1
    if t == "float":
        return 1.0
    if t == "bool":
        return True
    if t == "datetime":
        return "2026-01-01T00:00:00"
    if t == "date":
        return "2026-01-01"
    if t == "uuid":
        return "00000000-0000-0000-0000-000000000001"
    if t == "json":
        return {}
    return "smoke"


# Fields the server fills in; never include in a create payload.
_SERVER_ONLY = {"id", "user_id", "owner_id", "created_at", "updated_at"}


def _create_payload(entity: Entity) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for f in entity.fields:
        if f.name in _SERVER_ONLY:
            continue
        if f.nullable and f.default is None:
            # Optional fields: still send a value so we exercise the column.
            payload[f.name] = _sample_value(f)
        else:
            payload[f.name] = _sample_value(f)
    return payload


def build_plan(spec: ProjectSpec, *, backend_port: int) -> dict[str, Any]:
    """Build the JSON plan consumed by the in-container runner."""
    auth_enabled = bool(spec.auth.enabled)
    creds = {"email": "smoke-user@example.com", "password": "smoke-password-1"}
    plan: dict[str, Any] = {
        "base_url": f"http://127.0.0.1:{backend_port}",
        "auth": (
            {
                "credentials": creds,
                "login_form": {
                    "username": creds["email"],
                    "password": creds["password"],
                },
            }
            if auth_enabled
            else None
        ),
        "entities": [
            {
                "name": e.name,
                "candidate_paths": _candidate_paths(e),
                "create_payload": _create_payload(e),
            }
            for e in spec.entities
        ],
    }
    return plan


# --------------------------------------------------------------------- runner


def _truncate(text: str) -> str:
    return truncate_middle(text, _LOG_MAX_CHARS)


def _parse_results(output: str) -> tuple[list[str], list[str]]:
    """Extract PASSED/FAILED entity lists from the runner stdout."""
    passed: list[str] = []
    failed: list[str] = []
    for line in output.splitlines():
        if line.startswith("PASSED:"):
            try:
                passed = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
            except Exception:
                pass
        elif line.startswith("FAILED:"):
            try:
                failed = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
            except Exception:
                pass
    return passed, failed


def run_smoke_crud(
    sandbox: Sandbox,
    workspace_root: Path,
    project_id: str,
    project: str,
    spec: ProjectSpec,
    *,
    backend_port: int = 8000,
    timeout_s: int = 180,
    on_event: Callable[[str, dict], None] | None = None,
) -> SmokeReport:
    """Run the CRUD probe against the live ``backend`` container.

    ``workspace_root`` must be the path *as seen by the agent process* —
    typically ``Workspace.root`` — because we write the plan + runner there
    using normal filesystem I/O. The sandbox sees the same files at
    ``/workspaces/<project_id>/.smoke/`` via the shared bind-mount.
    """
    if not spec.entities:
        emit(on_event, "smoke_crud_skipped", reason="no_entities")
        return SmokeReport(ok=True, skipped=True, skip_reason="no entities in spec")

    smoke_dir = workspace_root / _SMOKE_DIR
    smoke_dir.mkdir(parents=True, exist_ok=True)
    plan_host = smoke_dir / _PLAN_FILE
    runner_host = smoke_dir / _RUNNER_FILE

    plan = build_plan(spec, backend_port=backend_port)
    plan_host.write_text(json.dumps(plan, indent=2))
    runner_host.write_text(_RUNNER_SOURCE)

    # Paths inside the sandbox (and inside backend after cp).
    plan_sandbox = f"/workspaces/{project_id}/{_SMOKE_DIR}/{_PLAN_FILE}"
    runner_sandbox = f"/workspaces/{project_id}/{_SMOKE_DIR}/{_RUNNER_FILE}"

    emit(on_event, "smoke_crud_started", entities=[e.name for e in spec.entities])

    cmd = (
        "set -e; "
        f"docker compose -p {project} cp {plan_sandbox} backend:/tmp/{_PLAN_FILE}; "
        f"docker compose -p {project} cp {runner_sandbox} backend:/tmp/{_RUNNER_FILE}; "
        f"docker compose -p {project} exec -T "
        f"-e SMOKE_PLAN_FILE=/tmp/{_PLAN_FILE} "
        f"backend python /tmp/{_RUNNER_FILE}"
    )
    result = sandbox.exec(project_id, cmd, timeout_s)
    output = result.output or ""
    passed, failed = _parse_results(output)
    ok = result.exit_code == 0 and not failed

    report = SmokeReport(
        ok=ok,
        log=_truncate(output),
        entities_passed=passed,
        entities_failed=failed or ([e.name for e in spec.entities] if not ok and not passed else []),
    )

    if ok:
        emit(on_event, "smoke_crud_ok", passed=report.entities_passed)
    else:
        emit(
            on_event,
            "smoke_crud_failed",
            exit=result.exit_code,
            passed=report.entities_passed,
            failed=report.entities_failed,
        )
    return report


__all__ = ["SmokeReport", "build_plan", "run_smoke_crud"]
