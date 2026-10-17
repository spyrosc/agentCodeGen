"""Reviewer node: a single LLM judgement that may route the graph back."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.logging import get_logger
from ._prompts import load_prompt
from .spec import ProjectSpec

log = get_logger("agents.fullstack.reviewer")

ReviewStatus = Literal["ok", "needs_backend", "needs_frontend"]


@dataclass
class ReviewResult:
    status: ReviewStatus
    notes: str


def review(
    llm: Any,
    requirements: str,
    spec: ProjectSpec,
    file_tree: list[str],
    *,
    backend_ok: bool,
    frontend_ok: bool,
    compose_ok: bool = False,
    compose_log: str = "",
) -> ReviewResult:
    # Deterministic short-circuit: if the whole stack came up via
    # `docker compose` and the backend `/health` responded, there is nothing
    # for the LLM reviewer to second-guess.
    if compose_ok:
        return ReviewResult(
            status="ok",
            notes="docker compose stack came up and backend /health responded.",
        )
    system = load_prompt("reviewer")
    log_excerpt = (compose_log or "").strip()
    if len(log_excerpt) > 1500:
        log_excerpt = log_excerpt[:750] + "\n... [truncated] ...\n" + log_excerpt[-750:]
    user = (
        f"REQUIREMENTS:\n{requirements.strip()}\n\n"
        f"SPEC:\n{spec.to_json()}\n\n"
        f"FILE TREE ({len(file_tree)} entries):\n"
        + "\n".join(file_tree[:200])
        + "\n\n"
        f"VALIDATION: backend_ok={backend_ok} frontend_ok={frontend_ok} "
        f"compose_ok={compose_ok}"
    )
    if log_excerpt:
        user += f"\n\nDOCKER COMPOSE LOG (excerpt):\n{log_excerpt}"
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = response.content if isinstance(response.content, str) else str(response.content)

    text = raw.strip()
    # Strip code fence if any.
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        status = data.get("status", "ok")
        if status not in ("ok", "needs_backend", "needs_frontend"):
            status = "ok"
        notes = str(data.get("notes", ""))[:500]
    except Exception:  # noqa: BLE001
        log.warning("reviewer JSON parse failed, defaulting to ok: %r", raw[:200])
        status, notes = "ok", "(reviewer output unparseable; accepted by default)"
    return ReviewResult(status=status, notes=notes)
