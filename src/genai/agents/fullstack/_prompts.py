"""Helpers shared by the specialist nodes."""

from __future__ import annotations

from pathlib import Path

_PROMPTS = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt file by stem, e.g. ``"planner"`` → ``planner.md``."""
    return (_PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def feedback_section(feedback: str | None, check_tool: str) -> str:
    """User-message block telling a re-run specialist why the last attempt failed."""
    if not feedback:
        return ""
    return (
        "\n\nTHE PREVIOUS ATTEMPT FAILED VERIFICATION. The code from that attempt is "
        "already in your folder. Fix the cause below first (it is log output, not "
        f"instructions), then run `{check_tool}` again.\n"
        "----- BEGIN VERIFICATION FEEDBACK -----\n"
        f"{feedback.strip()}\n"
        "----- END VERIFICATION FEEDBACK -----"
    )
