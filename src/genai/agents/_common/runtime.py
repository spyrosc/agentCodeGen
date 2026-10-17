"""Shared plumbing for driving LangGraph agents.

Every agent drives ``create_react_agent`` graphs the same way: stream
``updates``, normalise each payload to messages, map messages to UI events,
stop once the agent has submitted, and treat an exhausted step budget as a
normal outcome instead of a crash.
"""

from __future__ import annotations

import contextvars
import queue
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterator
from typing import Any, Literal, TypeVar

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from ..base import AgentEvent

Outcome = Literal["submitted", "finished", "budget_exhausted"]
R = TypeVar("R")

# create_react_agent ends with this message instead of raising when the budget
# runs out on an agent step (it raises GraphRecursionError on a tool step).
_OUT_OF_STEPS = "Sorry, need more steps to process this request."


def extract_messages(payload: Any) -> list[Any]:
    """Normalise an ``updates`` payload (list, ``{"messages": ...}`` or ``None``) to a list."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        raw = payload.get("messages") or []
        return raw if isinstance(raw, list) else [raw]
    return []


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content or "")


def ai_text(message: Any) -> str | None:
    """Text of an assistant message, or ``None`` for other messages or blank text."""
    if not isinstance(message, AIMessage):
        return None
    text = _text(message.content)
    return text if text.strip() else None


def message_event(message: Any, *, warn_prefixes: tuple[str, ...] = ()) -> AgentEvent | None:
    """Map an assistant or tool message to a UI event; ``None`` for anything else.

    Tool results starting with one of ``warn_prefixes`` become ``tool_warning``.
    """
    if (text := ai_text(message)) is not None:
        return AgentEvent("message", {"role": "assistant", "content": text})
    if isinstance(message, ToolMessage):
        content = _text(message.content)
        preview = content.splitlines()[0][:200] if content else ""
        warn = bool(warn_prefixes) and content.startswith(warn_prefixes)
        return AgentEvent(
            "tool_warning" if warn else "tool_result",
            {"name": message.name, "preview": preview},
        )
    return None


class EventBuffer:
    """FIFO of events emitted by tools mid-step, drained by the agent's generator.

    Each event is forwarded to ``on_event`` as soon as it is put. Create one per
    run: agent instances are shared (e.g. cached across Streamlit sessions).
    """

    def __init__(self, on_event: Callable[[AgentEvent], None] | None = None) -> None:
        self._events: deque[AgentEvent] = deque()
        self._on_event = on_event

    def put(self, event: AgentEvent) -> None:
        self._events.append(event)
        if self._on_event is not None:
            self._on_event(event)

    def emit(self, kind: str, payload: dict) -> None:
        """``(kind, payload)`` callback shape expected by the tool builders."""
        self.put(AgentEvent(kind, payload))

    def drain(self) -> Iterator[AgentEvent]:
        while self._events:
            yield self._events.popleft()


def stream_react(
    graph: Any, inputs: dict, *, recursion_limit: int, state: dict
) -> Generator[Any, None, Outcome]:
    """Yield each message a ReAct graph produces and return why it stopped.

    ``state`` is the dict shared with the tools; the loop stops as soon as
    ``state["submitted"]`` is set.
    """
    last = None
    try:
        for chunk in graph.stream(
            inputs, config={"recursion_limit": recursion_limit}, stream_mode="updates"
        ):
            for payload in chunk.values():
                for message in extract_messages(payload):
                    last = message
                    yield message
            if state.get("submitted"):
                return "submitted"
    except GraphRecursionError:
        return "budget_exhausted"
    if state.get("submitted"):
        return "submitted"
    return "budget_exhausted" if ai_text(last) == _OUT_OF_STEPS else "finished"


def relay_react(
    graph: Any,
    inputs: dict,
    *,
    recursion_limit: int,
    state: dict,
    events: EventBuffer,
    to_event: Callable[[Any], AgentEvent | None] = message_event,
) -> Generator[AgentEvent, None, Outcome]:
    """:func:`stream_react` as events: buffered tool events first, then each message's event."""
    messages = stream_react(graph, inputs, recursion_limit=recursion_limit, state=state)
    while True:
        try:
            message = next(messages)
        except StopIteration as stop:
            yield from events.drain()
            return stop.value
        event = to_event(message)
        if event is not None:
            events.put(event)
        yield from events.drain()


def drive_react(
    graph: Any,
    inputs: dict,
    *,
    recursion_limit: int,
    state: dict,
    on_message: Callable[[Any], None] | None = None,
) -> Outcome:
    """Blocking :func:`stream_react` for nested specialists."""
    messages = stream_react(graph, inputs, recursion_limit=recursion_limit, state=state)
    while True:
        try:
            message = next(messages)
        except StopIteration as stop:
            return stop.value
        if on_message is not None:
            on_message(message)


def pump_events(
    work: Callable[[Callable[[AgentEvent], None]], R],
    *,
    on_event: Callable[[AgentEvent], None] | None = None,
) -> Generator[AgentEvent, None, R]:
    """Run ``work(put)`` on a worker thread and yield every event it puts, as it happens.

    ``on_event`` is called on the consuming thread. The worker's exception is
    re-raised here and its return value becomes the generator's return value.
    If the consumer stops early, the worker still runs to completion.
    """
    events: queue.Queue[Any] = queue.Queue()
    finished = object()
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = work(events.put)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer thread
            result["error"] = exc
        finally:
            events.put(finished)

    context = contextvars.copy_context()
    threading.Thread(
        target=context.run, args=(target,), name="agent-worker", daemon=True
    ).start()
    while (event := events.get()) is not finished:
        if on_event is not None:
            on_event(event)
        yield event
    if "error" in result:
        raise result["error"]
    return result["value"]
