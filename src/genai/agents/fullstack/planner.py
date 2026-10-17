"""Planner node: free-text brief -> validated :class:`ProjectSpec`."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.logging import get_logger
from ...reflection.injection import prepend_lessons
from ._prompts import load_prompt
from .spec import ProjectSpec

log = get_logger("agents.fullstack.planner")


class PlannerError(RuntimeError):
    pass


def plan(llm: Any, requirements: str, *, max_attempts: int = 3) -> ProjectSpec:
    """Call the LLM with the planner prompt; parse and validate the JSON.

    Retries up to ``max_attempts`` on JSON / schema errors, telling the model
    what was wrong each time.
    """
    system = load_prompt("planner")
    system = prepend_lessons(system, category="fullstack-plan", query=requirements)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"FUNCTIONAL REQUIREMENTS:\n{requirements.strip()}"),
    ]
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        if last_error:
            messages.append(
                HumanMessage(
                    content=(
                        f"Your previous response could not be parsed: {last_error}\n"
                        "Reply with a single valid JSON object matching the schema. "
                        "No prose. No markdown fences."
                    )
                )
            )
        response = llm.invoke(messages)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        log.debug("planner attempt %d raw=%r", attempt, raw[:200])
        try:
            return ProjectSpec.from_llm_json(raw)
        except ValueError as exc:
            last_error = str(exc)
            messages.append(response)  # keep the bad response in context
    raise PlannerError(f"planner failed after {max_attempts} attempts: {last_error}")
