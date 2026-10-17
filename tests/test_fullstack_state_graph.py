"""Tests for the FullStackAgent state graph routing.

We mock every node module so no LLM, sandbox, or filesystem work happens.
The point is to assert the wiring: the conditional edge after `review`
routes back to the right specialist and terminates after MAX_REVIEW_PASSES.
"""

import threading
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

from genai.agents.fullstack.agent import FullStackAgent
from genai.agents.fullstack.reviewer import ReviewResult
from genai.agents.fullstack.spec import ProjectMeta, ProjectSpec
from genai.core.config import reload_settings
from genai.sandbox.workspace import Workspace, new_project_id

_SPECIALIST_OK = {"submitted": True, "summary": "ok", "checks_passed": True, "outcome": "submitted"}
_INTEGRATION_OK = type("R", (), {"ok": True, "missing": [], "issues": []})()
_COMPOSE_OK = type("C", (), {"ok": True, "log": "", "services_up": []})()


def _build_agent(max_review_passes: int = 2):
    return FullStackAgent(llm=MagicMock(), sandbox=MagicMock(), max_review_passes=max_review_passes)


def _spec():
    return ProjectSpec(meta=ProjectMeta(name="x", title="X"))


def test_happy_path_runs_each_phase_once(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    agent = _build_agent()
    spec = _spec()
    with (
        patch("genai.agents.fullstack.agent.plan", return_value=spec) as m_plan,
        patch("genai.agents.fullstack.agent.scaffold") as m_scaffold,
        patch("genai.agents.fullstack.agent.backend_node.run_backend",
              return_value=_SPECIALIST_OK) as m_be,
        patch("genai.agents.fullstack.agent.frontend_node.run_frontend",
              return_value=_SPECIALIST_OK) as m_fe,
        patch("genai.agents.fullstack.agent.integrate",
              return_value=type("R", (), {"ok": True, "missing": [], "issues": []})()) as m_int,
        patch("genai.agents.fullstack.agent.run_compose_smoke", return_value=_COMPOSE_OK),
        patch("genai.agents.fullstack.agent.review",
              return_value=ReviewResult(status="ok", notes="lgtm")) as m_rev,
    ):
        events = list(agent.stream("brief", workspace=ws))

    assert m_plan.call_count == 1
    assert m_scaffold.call_count == 1
    assert m_be.call_count == 1
    assert m_fe.call_count == 1
    assert m_int.call_count == 1
    assert m_rev.call_count == 1
    kinds = [e.kind for e in events]
    assert kinds[0] == "workspace_created"
    assert kinds[-1] == "done"
    assert "submit_project" in kinds


def test_review_routes_back_to_backend_then_submits_at_budget(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    agent = _build_agent()
    spec = _spec()

    # Always say "needs_backend" — the cap (2 passes) must end the loop.
    with (
        patch("genai.agents.fullstack.agent.plan", return_value=spec),
        patch("genai.agents.fullstack.agent.scaffold"),
        patch("genai.agents.fullstack.agent.backend_node.run_backend",
              return_value=_SPECIALIST_OK) as m_be,
        patch("genai.agents.fullstack.agent.frontend_node.run_frontend",
              return_value=_SPECIALIST_OK) as m_fe,
        patch("genai.agents.fullstack.agent.integrate",
              return_value=type("R", (), {"ok": True, "missing": [], "issues": []})()) as m_int,
        patch("genai.agents.fullstack.agent.run_compose_smoke", return_value=_COMPOSE_OK),
        patch("genai.agents.fullstack.agent.review",
              return_value=ReviewResult(status="needs_backend", notes="add more")) as m_rev,
    ):
        events = list(agent.stream("brief", workspace=ws))

    # Pass 1: backend → frontend → integrate → review (needs_backend) →
    # backend (pass 2) → frontend → integrate → review (passes==2 → submit).
    assert m_be.call_count == 2
    assert m_fe.call_count == 2
    assert m_rev.call_count == 2
    assert m_int.call_count == 2
    assert events[-1].kind == "done"
    assert events[-1].data["submitted"] is True


def test_route_needs_frontend_skips_backend_rerun(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    agent = _build_agent()
    spec = _spec()
    statuses = iter([
        ReviewResult(status="needs_frontend", notes="fix UI"),
        ReviewResult(status="ok", notes="lgtm"),
    ])
    with (
        patch("genai.agents.fullstack.agent.plan", return_value=spec),
        patch("genai.agents.fullstack.agent.scaffold"),
        patch("genai.agents.fullstack.agent.backend_node.run_backend",
              return_value=_SPECIALIST_OK) as m_be,
        patch("genai.agents.fullstack.agent.frontend_node.run_frontend",
              return_value=_SPECIALIST_OK) as m_fe,
        patch("genai.agents.fullstack.agent.integrate",
              return_value=type("R", (), {"ok": True, "missing": [], "issues": []})()),
        patch("genai.agents.fullstack.agent.run_compose_smoke", return_value=_COMPOSE_OK),
        patch("genai.agents.fullstack.agent.review", side_effect=lambda *a, **kw: next(statuses)),
    ):
        list(agent.stream("brief", workspace=ws))

    assert m_be.call_count == 1   # backend NOT re-run
    assert m_fe.call_count == 2   # frontend re-run once


@contextmanager
def _nodes(**overrides):
    """Patch every node collaborator; ``overrides`` maps a target to patch kwargs."""
    targets = {
        "plan": {"return_value": _spec()},
        "scaffold": {},
        "backend_node.run_backend": {"return_value": _SPECIALIST_OK},
        "frontend_node.run_frontend": {"return_value": _SPECIALIST_OK},
        "integrate": {"return_value": _INTEGRATION_OK},
        "run_compose_smoke": {"return_value": _COMPOSE_OK},
        "review": {"return_value": ReviewResult(status="ok", notes="lgtm")},
        **overrides,
    }
    with ExitStack() as stack:
        yield {
            name: stack.enter_context(patch(f"genai.agents.fullstack.agent.{name}", **kwargs))
            for name, kwargs in targets.items()
        }


def test_done_reports_real_summary_and_check_based_status(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    # Status must come from the last check run, not from wording in the summary.
    backend = {**_SPECIALIST_OK, "summary": "0 failures, 12 passed"}
    frontend = {**_SPECIALIST_OK, "summary": "all good", "checks_passed": False}
    with _nodes(**{
        "backend_node.run_backend": {"return_value": backend},
        "frontend_node.run_frontend": {"return_value": frontend},
    }):
        events = list(_build_agent().stream("brief", workspace=ws))

    done = events[-1]
    assert done.kind == "done"
    assert done.data["submitted"] is True
    assert "Backend tests: pass" in done.data["summary"]
    assert "Frontend build: incomplete" in done.data["summary"]
    completed = {
        e.data["phase"]: e.data for e in events
        if e.kind == "phase_completed" and e.data["phase"] in {"backend", "frontend"}
    }
    assert completed["backend"]["ok"] is True
    assert completed["frontend"]["ok"] is False


def test_many_review_passes_fit_in_the_graph_recursion_limit(tmp_path):
    # 6 passes = 33 node runs, more than langgraph's default limit of 25.
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    with _nodes(review={"return_value": ReviewResult(status="needs_backend", notes="again")}) as m:
        events = list(_build_agent(max_review_passes=6).stream("brief", workspace=ws))

    assert "error" not in [e.kind for e in events]
    assert m["backend_node.run_backend"].call_count == 6
    assert events[-1].data["submitted"] is True


def test_node_events_stream_while_the_node_is_still_running(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    delivered = threading.Event()

    def slow_backend(*_args, on_event, **_kwargs):
        on_event("specialist_message", {"scope": "backend", "content": "halfway"})
        if not delivered.wait(timeout=5):
            raise AssertionError("event was not delivered while the node was running")
        return _SPECIALIST_OK

    events = []
    with _nodes(**{"backend_node.run_backend": {"side_effect": slow_backend}}):
        for ev in _build_agent().stream("brief", workspace=ws):
            events.append(ev)
            if ev.kind == "specialist_message":
                delivered.set()

    assert "error" not in [e.kind for e in events]
    assert events[-1].data["submitted"] is True


def test_on_event_receives_every_yielded_event(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    seen = []
    with _nodes():
        events = list(_build_agent().stream("brief", workspace=ws, on_event=seen.append))

    assert [e.kind for e in seen] == [e.kind for e in events]


def test_node_crash_ends_with_error_and_unsubmitted_done(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    with _nodes(plan={"side_effect": RuntimeError("planner exploded")}):
        events = list(_build_agent().stream("brief", workspace=ws))

    assert [e.kind for e in events][-2:] == ["error", "done"]
    assert "planner exploded" in events[-2].data["message"]
    assert events[-1].data["submitted"] is False


def test_scaffold_receives_configured_postgres_image(tmp_path, monkeypatch):
    monkeypatch.setenv("FULLSTACK_POSTGRES_IMAGE", "postgres:17-alpine")
    reload_settings()
    try:
        ws = Workspace.create(new_project_id("t"), base=tmp_path)
        with _nodes() as m:
            list(_build_agent().stream("brief", workspace=ws))
    finally:
        monkeypatch.delenv("FULLSTACK_POSTGRES_IMAGE")
        reload_settings()

    assert m["scaffold"].call_args.kwargs["postgres_image"] == "postgres:17-alpine"


def test_rerun_gets_reviewer_notes_and_compose_log_as_feedback(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    crashed = type("C", (), {
        "ok": False,
        "log": "[runtime log errors]\nbackend-1 | ModuleNotFoundError: No module named 'app'",
        "services_up": ["backend"],
    })()
    reviews = iter([
        ReviewResult(status="needs_backend", notes="alembic cannot import app"),
        ReviewResult(status="ok", notes="lgtm"),
    ])
    with _nodes(
        run_compose_smoke={"return_value": crashed},
        review={"side_effect": lambda *a, **kw: next(reviews)},
    ) as m:
        list(_build_agent().stream("brief", workspace=ws))

    first, second = (c.kwargs["feedback"] for c in m["backend_node.run_backend"].call_args_list)
    assert first is None
    assert "alembic cannot import app" in second
    assert "ModuleNotFoundError: No module named 'app'" in second
    # The review targeted the backend, so the frontend re-run gets no feedback.
    assert all(c.kwargs["feedback"] is None for c in m["frontend_node.run_frontend"].call_args_list)


def test_needs_frontend_route_sends_feedback_to_frontend_only(tmp_path):
    ws = Workspace.create(new_project_id("t"), base=tmp_path)
    reviews = iter([
        ReviewResult(status="needs_frontend", notes="tsc: Property 'isLoading' does not exist"),
        ReviewResult(status="ok", notes="lgtm"),
    ])
    with _nodes(review={"side_effect": lambda *a, **kw: next(reviews)}) as m:
        list(_build_agent().stream("brief", workspace=ws))

    fe_feedback = [c.kwargs["feedback"] for c in m["frontend_node.run_frontend"].call_args_list]
    assert fe_feedback[0] is None
    assert "isLoading" in fe_feedback[1]
    assert m["backend_node.run_backend"].call_args.kwargs["feedback"] is None
