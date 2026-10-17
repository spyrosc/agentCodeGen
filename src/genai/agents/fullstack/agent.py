"""``FullStackAgent`` — multi-agent orchestrator for full-stack app generation.

Uses a LangGraph :class:`StateGraph` with these nodes:

    plan -> scaffold -> backend -> frontend -> integrate -> review -> END
                              ^                                |
                              |________ (route back if needed) |

The agent registers itself with :mod:`genai.agents.registry` so the UI
discovers it automatically.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from ...core.config import get_settings
from ...core.logging import get_logger
from ...core.text import truncate_middle
from ...llm import load_llm
from ...sandbox import DockerExecSandbox, Sandbox, Workspace, new_project_id
from .._common.runtime import pump_events
from ..base import AgentEvent, AgentResult
from ..registry import register
from . import backend as backend_node
from . import frontend as frontend_node
from .compose import run_compose_smoke
from .integrator import integrate
from .planner import plan
from .reviewer import review
from .scaffolder import scaffold
from .spec import ProjectSpec

log = get_logger("agents.fullstack")


class FullStackState(TypedDict, total=False):
    requirements: str
    spec: ProjectSpec
    backend_summary: str | None
    backend_ok: bool
    frontend_summary: str | None
    frontend_ok: bool
    integration_ok: bool
    integration_issues: list[str]
    compose_ok: bool
    compose_log: str
    compose_services: list[str]
    review_status: Literal["ok", "needs_backend", "needs_frontend"]
    review_notes: str
    review_passes: int
    submitted: bool
    summary: str | None


# Default — UI / config can raise this for fullstack runs.
DEFAULT_FULLSTACK_RECURSION = 80

_FEEDBACK_LOG_CHARS = 3_000


def _review_feedback(s: FullStackState, target: str) -> str | None:
    """Why the last review routed back to ``target``; ``None`` on first runs or other routes."""
    if not s.get("review_passes") or s.get("review_status") != target:
        return None
    parts = [f"Reviewer: {s.get('review_notes') or '(no notes)'}"]
    if s.get("integration_issues"):
        parts.append("Integration issues:\n- " + "\n- ".join(s["integration_issues"]))
    if s.get("compose_log"):
        parts.append(
            "docker compose log excerpt:\n"
            + truncate_middle(s["compose_log"], _FEEDBACK_LOG_CHARS)
        )
    return "\n\n".join(parts)


@register
@dataclass
class FullStackAgent:
    """End-to-end full-stack project generator."""

    name: str = field(default="fullstack", init=False)
    description: str = field(
        default=(
            "Plans, scaffolds and self-validates a complete app: FastAPI + "
            "SQLAlchemy backend, React + Vite + Tailwind frontend, Postgres in "
            "a per-project docker-compose."
        ),
        init=False,
    )

    llm: Any = None
    sandbox: Sandbox | None = None
    max_review_passes: int = 2
    enable_frontend_build: bool = True

    # ---------------------------------------------------------- factory
    def __post_init__(self) -> None:
        if self.llm is None:
            self.llm = load_llm()
        if self.sandbox is None:
            self.sandbox = DockerExecSandbox.from_settings()
        fs_cfg = get_settings().fullstack
        # Allow caller-provided overrides (kept from dataclass init); else env-driven defaults.
        if self.max_review_passes == 2:
            object.__setattr__(self, "max_review_passes", fs_cfg.max_review_passes)
        if self.enable_frontend_build is True:
            object.__setattr__(self, "enable_frontend_build", fs_cfg.enable_frontend_build)

    # ----------------------------------------------------------- helpers
    def _new_workspace(self, project_id: str | None = None) -> Workspace:
        return Workspace.create(project_id or new_project_id("fs"))

    def _build_graph(
        self,
        workspace: Workspace,
        emit: Callable[[str, dict], None],
        specialist_iters: int,
    ):
        # Nodes ----------------------------------------------------------
        def n_plan(s: FullStackState) -> dict:
            emit("phase_started", {"phase": "plan"})
            spec = plan(self.llm, s["requirements"])
            emit("spec_drafted", {"entities": [e.name for e in spec.entities]})
            emit("phase_completed", {"phase": "plan"})
            return {"spec": spec}

        def n_scaffold(s: FullStackState) -> dict:
            emit("phase_started", {"phase": "scaffold"})
            fs_cfg = get_settings().fullstack
            scaffold(
                workspace,
                s["spec"],
                backend_port=fs_cfg.backend_port,
                frontend_port=fs_cfg.frontend_port,
                postgres_image=fs_cfg.postgres_image,
                on_event=lambda k, p: emit(k, p),
            )
            emit("phase_completed", {"phase": "scaffold"})
            return {}

        def n_backend(s: FullStackState) -> dict:
            emit("phase_started", {"phase": "backend"})
            res = backend_node.run_backend(
                self.llm, workspace, self.sandbox, s["spec"],
                on_event=lambda k, p: emit(k, p),
                max_iterations=specialist_iters,
                feedback=_review_feedback(s, "needs_backend"),
            )
            ok = bool(res.get("checks_passed"))
            emit("phase_completed", {"phase": "backend", "ok": ok, "outcome": res.get("outcome")})
            return {"backend_summary": res.get("summary"), "backend_ok": ok}

        def n_frontend(s: FullStackState) -> dict:
            emit("phase_started", {"phase": "frontend"})
            res = frontend_node.run_frontend(
                self.llm, workspace, self.sandbox, s["spec"],
                on_event=lambda k, p: emit(k, p),
                max_iterations=specialist_iters,
                enable_build=self.enable_frontend_build,
                feedback=_review_feedback(s, "needs_frontend"),
            )
            ok = bool(res.get("checks_passed"))
            emit("phase_completed", {"phase": "frontend", "ok": ok, "outcome": res.get("outcome")})
            return {"frontend_summary": res.get("summary"), "frontend_ok": ok}

        def n_integrate(s: FullStackState) -> dict:
            emit("phase_started", {"phase": "integrate"})
            report = integrate(workspace.root, on_event=lambda k, p: emit(k, p))
            emit("phase_completed", {"phase": "integrate", "ok": report.ok})
            return {
                "integration_ok": report.ok,
                "integration_issues": report.missing + report.issues,
            }

        def n_compose(s: FullStackState) -> dict:
            # Only attempt to bring the stack up if integration produced the
            # required compose file; otherwise we'd just spam errors at the LLM.
            if not s.get("integration_ok", False):
                emit("compose_skipped", {"reason": "integration_failed"})
                return {"compose_ok": False, "compose_log": "", "compose_services": []}
            emit("phase_started", {"phase": "compose"})
            fs_cfg = get_settings().fullstack
            report = run_compose_smoke(
                self.sandbox,
                workspace.project_id,
                backend_port=fs_cfg.backend_port,
                on_event=lambda k, p: emit(k, p),
                spec=s["spec"],
                workspace_root=workspace.root,
            )
            emit("phase_completed", {"phase": "compose", "ok": report.ok})
            return {
                "compose_ok": report.ok,
                "compose_log": report.log,
                "compose_services": report.services_up,
            }

        def n_review(s: FullStackState) -> dict:
            emit("phase_started", {"phase": "review"})
            file_tree = workspace.list_tree()
            result = review(
                self.llm, s["requirements"], s["spec"], file_tree,
                backend_ok=s.get("backend_ok", False),
                frontend_ok=s.get("frontend_ok", False),
                compose_ok=s.get("compose_ok", False),
                compose_log=s.get("compose_log", ""),
            )
            passes = int(s.get("review_passes", 0)) + 1
            emit("review", {"status": result.status, "notes": result.notes, "pass": passes})
            emit("phase_completed", {"phase": "review", "status": result.status})
            return {
                "review_status": result.status,
                "review_notes": result.notes,
                "review_passes": passes,
            }

        def n_submit(s: FullStackState) -> dict:
            spec = s["spec"]
            summary = (
                f"Generated {spec.meta.name}: {len(spec.entities)} entities, "
                f"{len(spec.pages)} pages. "
                f"Backend tests: {'pass' if s.get('backend_ok') else 'incomplete'}. "
                f"Frontend build: {'pass' if s.get('frontend_ok') else 'incomplete'}. "
                f"Integration: {'ok' if s.get('integration_ok') else 'with issues'}. "
                f"docker compose smoke: {'ok' if s.get('compose_ok') else 'failed/skipped'}. "
                f"Reviewer: {s.get('review_notes', '')}"
            )
            emit("submit_project", {"summary": summary})
            return {"submitted": True, "summary": summary}

        # Routing -------------------------------------------------------
        def route_review(s: FullStackState) -> str:
            if s.get("review_passes", 0) >= self.max_review_passes:
                return "submit"
            status = s.get("review_status", "ok")
            if status == "needs_backend":
                return "backend"
            if status == "needs_frontend":
                return "frontend"
            return "submit"

        # Graph ---------------------------------------------------------
        g = StateGraph(FullStackState)
        g.add_node("plan", n_plan)
        g.add_node("scaffold", n_scaffold)
        g.add_node("backend", n_backend)
        g.add_node("frontend", n_frontend)
        g.add_node("integrate", n_integrate)
        g.add_node("compose", n_compose)
        g.add_node("review", n_review)
        g.add_node("submit", n_submit)

        g.set_entry_point("plan")
        g.add_edge("plan", "scaffold")
        g.add_edge("scaffold", "backend")
        g.add_edge("backend", "frontend")
        g.add_edge("frontend", "integrate")
        g.add_edge("integrate", "compose")
        g.add_edge("compose", "review")
        g.add_conditional_edges(
            "review", route_review,
            {"backend": "backend", "frontend": "frontend", "submit": "submit"},
        )
        g.add_edge("submit", END)
        return g.compile()

    # ------------------------------------------------------- public API
    def stream(
        self,
        task: str,
        *,
        workspace: Workspace | None = None,
        max_iterations: int | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> Iterator[AgentEvent]:
        cfg = get_settings().agent
        workspace = workspace or self._new_workspace()
        # Per-phase iteration budget; the orchestrator graph itself is small.
        specialist_iters = max_iterations or max(cfg.max_iterations, DEFAULT_FULLSTACK_RECURSION)
        # A review pass re-runs at most 5 nodes; +10 covers plan/scaffold/submit.
        graph_recursion_limit = 5 * self.max_review_passes + 10

        def announce(event: AgentEvent) -> AgentEvent:
            if on_event:
                on_event(event)
            return event

        yield announce(
            AgentEvent(
                "workspace_created",
                {"project_id": workspace.project_id, "root": str(workspace.root)},
            )
        )

        def work(put: Callable[[AgentEvent], None]) -> dict:
            graph = self._build_graph(
                workspace, lambda kind, payload: put(AgentEvent(kind, payload)), specialist_iters
            )
            initial: FullStackState = {"requirements": task.strip(), "review_passes": 0}
            return graph.invoke(initial, config={"recursion_limit": graph_recursion_limit})

        try:
            # Nodes run for minutes, so events are pumped from a worker as they happen.
            final = yield from pump_events(work, on_event=on_event)
        except Exception as exc:  # noqa: BLE001
            log.exception("fullstack graph crashed")
            yield announce(AgentEvent("error", {"message": f"{type(exc).__name__}: {exc}"}))
            yield announce(
                AgentEvent(
                    "done",
                    {"submitted": False, "summary": str(exc), "project_id": workspace.project_id},
                )
            )
            return

        yield announce(
            AgentEvent(
                "done",
                {
                    "submitted": bool(final.get("submitted")),
                    "summary": final.get("summary"),
                    "project_id": workspace.project_id,
                },
            )
        )

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        result = AgentResult(submitted=False)
        for ev in self.stream(task, **kwargs):
            if ev.kind == "done":
                result = AgentResult(
                    submitted=bool(ev.data.get("submitted")),
                    summary=ev.data.get("summary"),
                    project_id=ev.data.get("project_id"),
                )
        return result
