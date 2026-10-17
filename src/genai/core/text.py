"""Small text helpers shared across layers."""

from __future__ import annotations


def truncate_middle(text: str, limit: int) -> str:
    """Keep the head and tail of ``text`` within ``limit`` chars, marking the cut.

    A ``limit`` of zero or less disables truncation.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    half = limit // 2
    head, tail = text[:half], text[len(text) - half :]
    return f"{head}\n... [truncated {len(text) - 2 * half} chars] ...\n{tail}"
