"""LLM telemetry for eval runs.

Wraps an LLM with a LangChain callback that aggregates call counts, wall
time, and (when the provider exposes it) prompt + completion token usage.
For providers that do not return token counts (notably Ollama) we fall back
to a ``len(text) // 4`` heuristic and flag ``tokens_estimated=True``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class TelemetrySnapshot:
    llm_calls: int = 0
    llm_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    event_kinds: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "llm_seconds": round(self.llm_seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_estimated": self.tokens_estimated,
            "event_kinds": dict(self.event_kinds),
        }


class TelemetryHandler(BaseCallbackHandler):
    """Aggregate per-case LLM telemetry."""

    def __init__(self) -> None:
        self.snapshot = TelemetrySnapshot()
        self._starts: dict[str, float] = {}
        # Buffer prompts per run so we can estimate tokens for providers that
        # don't return real usage data.
        self._prompt_chars: dict[str, int] = {}

    # --------------------------------------------------------------- LLM
    def on_llm_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        prompts: list[str],
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        key = str(run_id)
        self._starts[key] = time.perf_counter()
        self._prompt_chars[key] = sum(len(p) for p in prompts)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        messages: list[list[Any]],
        *,
        run_id: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        key = str(run_id)
        self._starts[key] = time.perf_counter()
        chars = 0
        for batch in messages:
            for m in batch:
                content = getattr(m, "content", "")
                chars += len(content) if isinstance(content, str) else len(str(content))
        self._prompt_chars[key] = chars

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:  # noqa: ARG002
        key = str(run_id)
        start = self._starts.pop(key, None)
        prompt_chars = self._prompt_chars.pop(key, 0)
        if start is not None:
            self.snapshot.llm_seconds += time.perf_counter() - start
        self.snapshot.llm_calls += 1

        usage = self._extract_usage(response)
        if usage:
            self.snapshot.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.snapshot.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        else:
            # Heuristic fallback (~4 chars per token).
            self.snapshot.prompt_tokens += prompt_chars // 4
            completion_chars = 0
            for gens in getattr(response, "generations", []) or []:
                for gen in gens:
                    text = getattr(gen, "text", "") or ""
                    completion_chars += len(text)
            self.snapshot.completion_tokens += completion_chars // 4
            self.snapshot.tokens_estimated = True

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any] | None:
        out = getattr(response, "llm_output", None)
        if isinstance(out, dict):
            usage = out.get("token_usage") or out.get("usage")
            if isinstance(usage, dict) and usage:
                return usage
        # Bedrock/OpenAI new path: response.generations[0][0].message.usage_metadata
        try:
            gen = response.generations[0][0]
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if isinstance(meta, dict) and meta:
                return {
                    "prompt_tokens": meta.get("input_tokens"),
                    "completion_tokens": meta.get("output_tokens"),
                }
        except (AttributeError, IndexError, TypeError):
            pass
        return None

    # ------------------------------------------------------------ events
    def record_event(self, kind: str) -> None:
        self.snapshot.event_kinds[kind] = self.snapshot.event_kinds.get(kind, 0) + 1


def attach(llm: Any, handler: TelemetryHandler) -> Any:
    """Return ``llm`` with ``handler`` registered for callbacks.

    Uses the LCEL ``with_config`` API so we don't touch
    :func:`genai.llm.providers.load_llm`.
    """
    if hasattr(llm, "with_config"):
        return llm.with_config({"callbacks": [handler]})
    return llm
