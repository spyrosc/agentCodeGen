"""Append-only JSONL store of lessons the platform has learnt.

Why JSONL, not SQLite: appends are POSIX-atomic for lines under
``PIPE_BUF`` (4096 bytes on Linux), which is more than enough for our
sanitised, length-clamped lesson records. This means multiple agent runs
\u2014 including concurrent eval suites \u2014 can write to the same store without
locking and without truncating each other.

The store is *not* a knowledge base. It is a coarse list of past failure
signals. Search is a v1 keyword + tag overlap rank; the public interface
(:meth:`LessonStore.search`) is stable so a v2 swap to a vector index is a
one-file change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.config import ReflectSettings, get_settings
from ..core.logging import get_logger

log = get_logger("reflection.store")

# --- Hard limits ----------------------------------------------------------

# Each text field clamp. Belt-and-braces against an LLM (or a malicious
# eval output) writing a multi-megabyte "lesson" that blows up future
# context windows.
_MAX_FIELD_CHARS = 2_000

# Patterns that look like prompt-injection in lesson text. We do not delete
# lines \u2014 silent stripping is its own footgun \u2014 we *neutralise* them by
# fencing the offending prefix.
_INJECTION_PATTERNS = (
    re.compile(r"^\s*system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*###\s*system", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*<\|im_(start|end)\|>", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*ignore (all|previous) instructions", re.IGNORECASE | re.MULTILINE),
)


def sanitise_text(text: str, *, max_chars: int = _MAX_FIELD_CHARS) -> str:
    """Neutralise prompt-injection patterns and clamp length.

    Defensive only \u2014 the canonical defence is the ``treat-as-data-only``
    advisory wrapper in :mod:`~genai.reflection.injection`.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", "")
    for pat in _INJECTION_PATTERNS:
        text = pat.sub(lambda m: f"[neutralised] {m.group(0).lstrip()}", text)
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n\u2026[lesson truncated]"
    return text


# --- Models --------------------------------------------------------------


_VALID_SOURCES = {"manual", "eval-failure", "eval-regression", "reflect-agent"}


@dataclass
class Lesson:
    """One observation about a past failure and the fix that worked.

    ``id`` is a stable hash of ``(category, problem)`` so duplicate inserts
    can be detected on append.
    """

    category: str
    problem: str
    fix: str
    tags: list[str] = field(default_factory=list)
    source: str = "manual"
    evidence: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    id: str = ""

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(
                f"Lesson.source must be one of {sorted(_VALID_SOURCES)}, got {self.source!r}"
            )
        # Sanitise free-text fields up-front. Tags are short identifiers so
        # we just clamp those individually.
        self.category = sanitise_text(self.category, max_chars=120)
        self.problem = sanitise_text(self.problem)
        self.fix = sanitise_text(self.fix)
        self.tags = [sanitise_text(t, max_chars=80) for t in (self.tags or [])][:32]
        if not self.id:
            self.id = self._compute_id()

    def _compute_id(self) -> str:
        h = hashlib.sha256()
        h.update(self.category.encode("utf-8"))
        h.update(b"\x00")
        h.update(self.problem.encode("utf-8"))
        return h.hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lesson:
        # Avoid re-running __post_init__ sanitisation on already-stored
        # records (which would change ``id``).
        obj = cls.__new__(cls)
        obj.category = data.get("category", "")
        obj.problem = data.get("problem", "")
        obj.fix = data.get("fix", "")
        obj.tags = list(data.get("tags") or [])
        obj.source = data.get("source", "manual")
        obj.evidence = dict(data.get("evidence") or {})
        obj.ts = data.get("ts", "")
        obj.id = data.get("id", "")
        if not obj.id:
            obj.id = obj._compute_id()
        return obj


# --- Store ---------------------------------------------------------------


class LessonStore:
    """JSONL-backed lesson store.

    ``append`` is concurrency-safe across processes via ``O_APPEND``: the
    OS guarantees each ``write`` is atomic up to ``PIPE_BUF`` bytes. We
    flush after every append so partial runs remain analysable on crash.
    """

    def __init__(self, path: str | Path, *, max_entries: int = 10_000) -> None:
        self.path = Path(path)
        self.max_entries = max_entries
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------- factories

    @classmethod
    def from_settings(cls, settings: ReflectSettings | None = None) -> LessonStore:
        cfg = settings or get_settings().reflect
        return cls(cfg.lessons_path, max_entries=cfg.lessons_max)

    # ------------------------------------------------------------ I/O

    def append(self, lesson: Lesson, *, dedup: bool = True) -> bool:
        """Append a lesson. Returns ``False`` when deduped, ``True`` otherwise."""
        if dedup and self._exists_id(lesson.id):
            log.debug("lesson %s already present (deduped)", lesson.id)
            return False
        line = json.dumps(lesson.to_dict(), separators=(",", ":")) + "\n"
        if len(line.encode("utf-8")) > 4000:  # safety vs PIPE_BUF
            line = self._truncate_line(lesson)
        # O_APPEND is the whole point: each write is atomic vs concurrent writers.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        self._maybe_rotate()
        return True

    @staticmethod
    def _truncate_line(lesson: Lesson) -> str:
        """Last-resort clamp if a sanitised lesson still exceeds atomic-write size."""
        d = lesson.to_dict()
        for key in ("evidence", "fix", "problem"):
            if key in d:
                if isinstance(d[key], str) and len(d[key]) > 500:
                    d[key] = d[key][:500] + "\u2026"
                elif isinstance(d[key], dict):
                    d[key] = {k: (str(v)[:200] + "\u2026") if len(str(v)) > 200 else v for k, v in d[key].items()}
        return json.dumps(d, separators=(",", ":")) + "\n"

    def _iter_lines(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("dropping malformed lesson line: %r", raw[:120])

    def all(self) -> list[Lesson]:
        return [Lesson.from_dict(d) for d in self._iter_lines()]

    def _exists_id(self, lesson_id: str) -> bool:
        for obj in self._iter_lines():
            if obj.get("id") == lesson_id:
                return True
        return False

    # --------------------------------------------------------- search

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        tags: Iterable[str] | None = None,
        k: int = 5,
    ) -> list[Lesson]:
        """Return up to ``k`` most relevant lessons.

        v1 ranking: keyword overlap on (problem + fix + tags) + tag match
        bonus, with a small recency tiebreaker. Deterministic for the same
        store contents and inputs.
        """
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        wanted_tags = {t.lower() for t in (tags or [])}
        scored: list[tuple[float, int, Lesson]] = []
        for idx, obj in enumerate(self._iter_lines()):
            lesson = Lesson.from_dict(obj)
            if category and lesson.category != category:
                continue
            haystack = " ".join([lesson.problem, lesson.fix, " ".join(lesson.tags)]).lower()
            keyword_hits = sum(1 for t in terms if t in haystack)
            tag_hits = len(wanted_tags & {t.lower() for t in lesson.tags})
            score = keyword_hits + 2 * tag_hits
            if score == 0 and (terms or wanted_tags):
                continue
            scored.append((score, idx, lesson))

        # Higher score first; among equal scores, more recent (higher idx) first.
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [lesson for _score, _idx, lesson in scored[:k]]

    # ------------------------------------------------------- rotation

    def _maybe_rotate(self) -> None:
        if self.max_entries <= 0:
            return
        try:
            count = sum(1 for _ in self.path.open("r", encoding="utf-8", errors="ignore"))
        except FileNotFoundError:
            return
        if count <= self.max_entries:
            return
        backup = self.path.with_suffix(self.path.suffix + f".{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.bak")
        # Move the old file to backup; new appends start a fresh file.
        os.replace(self.path, backup)
        log.info("rotated lessons store: %s entries -> %s", count, backup.name)
