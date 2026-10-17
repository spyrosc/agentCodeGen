"""Tests for the shared agent runtime (:mod:`genai.agents._common.runtime`)."""

from __future__ import annotations

import contextvars
import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from genai.agents._common.runtime import (
    EventBuffer,
    ai_text,
    drive_react,
    extract_messages,
    message_event,
    pump_events,
    relay_react,
    stream_react,
)
from genai.agents.base import AgentEvent


class FakeGraph:
    """Replays scripted ``updates`` chunks.

    A step may be a chunk dict, a callable returning one (to run side effects
    while the "node" executes), or an exception to raise.
    """

    def __init__(self, *steps):
        self.steps = steps
        self.config = None
        self.consumed = 0

    def stream(self, inputs, *, config, stream_mode):
        assert stream_mode == "updates"
        self.config = config
        for step in self.steps:
            self.consumed += 1
            if isinstance(step, BaseException):
                raise step
            yield step() if callable(step) else step


def _ai(text: str) -> dict:
    return {"agent": {"messages": [AIMessage(content=text)]}}


def _collect(gen):
    """Exhaust a generator; return ``(items, return_value)``."""
    items = []
    while True:
        try:
            items.append(next(gen))
        except StopIteration as stop:
            return items, stop.value


# ---------------------------------------------------------------- messages


def test_extract_messages_normalises_payload_shapes():
    m = AIMessage(content="x")
    assert extract_messages(None) == []
    assert extract_messages([m]) == [m]
    assert extract_messages({"messages": [m]}) == [m]
    assert extract_messages({"messages": m}) == [m]
    assert extract_messages({"other": 1}) == []
    assert extract_messages("junk") == []


def test_ai_text_only_for_non_blank_assistant_messages():
    assert ai_text(AIMessage(content="hi")) == "hi"
    assert ai_text(AIMessage(content="   ")) is None
    assert ai_text(AIMessage(content=[])) is None
    assert ai_text(HumanMessage(content="hi")) is None
    assert ai_text(None) is None


def test_message_event_maps_assistant_text():
    ev = message_event(AIMessage(content="hi"))
    assert ev.kind == "message"
    assert ev.data == {"role": "assistant", "content": "hi"}


def test_message_event_ignores_blank_assistant_and_other_messages():
    assert message_event(AIMessage(content=" \n")) is None
    assert message_event(HumanMessage(content="hi")) is None


def test_message_event_tool_preview_is_first_line():
    msg = ToolMessage(content="exit=0\nlots of output", tool_call_id="1", name="run_tests")
    ev = message_event(msg)
    assert ev.kind == "tool_result"
    assert ev.data == {"name": "run_tests", "preview": "exit=0"}


def test_message_event_handles_empty_tool_output():
    # Regression: "".splitlines()[0] raised IndexError and killed the run.
    ev = message_event(ToolMessage(content="", tool_call_id="1", name="write_file"))
    assert ev.data["preview"] == ""


def test_message_event_warn_prefixes_mark_tool_warnings():
    msg = ToolMessage(content="PATCH_REJECTED: too big", tool_call_id="1", name="propose_patch")
    assert message_event(msg).kind == "tool_result"
    assert message_event(msg, warn_prefixes=("PATCH_REJECTED",)).kind == "tool_warning"


# ------------------------------------------------------------ EventBuffer


def test_event_buffer_forwards_immediately_and_drains_in_order():
    seen: list[AgentEvent] = []
    buf = EventBuffer(seen.append)
    buf.emit("a", {"n": 1})
    buf.put(AgentEvent("b"))
    assert [e.kind for e in seen] == ["a", "b"]
    assert [e.kind for e in buf.drain()] == ["a", "b"]
    assert list(buf.drain()) == []


# ------------------------------------------------------------ stream_react


def test_stream_react_yields_messages_and_passes_recursion_limit():
    graph = FakeGraph(_ai("one"), _ai("two"))
    msgs, outcome = _collect(stream_react(graph, {}, recursion_limit=7, state={}))
    assert [m.content for m in msgs] == ["one", "two"]
    assert outcome == "finished"
    assert graph.config == {"recursion_limit": 7}


def test_stream_react_stops_right_after_submit():
    state = {"submitted": False}

    def submit():
        state["submitted"] = True
        return _ai("submitted")

    graph = FakeGraph(_ai("working"), submit, _ai("never"))
    msgs, outcome = _collect(stream_react(graph, {}, recursion_limit=10, state=state))
    assert outcome == "submitted"
    assert [m.content for m in msgs] == ["working", "submitted"]
    assert graph.consumed == 2


def test_stream_react_recursion_error_is_budget_exhausted():
    graph = FakeGraph(_ai("working"), GraphRecursionError("limit"))
    msgs, outcome = _collect(stream_react(graph, {}, recursion_limit=3, state={}))
    assert outcome == "budget_exhausted"
    assert len(msgs) == 1


def test_stream_react_out_of_steps_message_is_budget_exhausted():
    graph = FakeGraph(_ai("Sorry, need more steps to process this request."))
    _, outcome = _collect(stream_react(graph, {}, recursion_limit=4, state={}))
    assert outcome == "budget_exhausted"


def test_stream_react_propagates_other_errors():
    with pytest.raises(RuntimeError, match="boom"):
        list(stream_react(FakeGraph(RuntimeError("boom")), {}, recursion_limit=3, state={}))


# ------------------------------------------------------ relay / drive


def test_relay_react_yields_tool_events_before_their_tool_result():
    events = EventBuffer()

    def tool_step():
        events.emit("write_file", {"path": "a.py"})
        return {"tools": {"messages": [ToolMessage(content="ok", tool_call_id="1", name="write_file")]}}

    graph = FakeGraph(_ai("plan"), tool_step)
    out, outcome = _collect(relay_react(graph, {}, recursion_limit=5, state={}, events=events))
    assert [e.kind for e in out] == ["message", "write_file", "tool_result"]
    assert outcome == "finished"


def test_relay_react_forwards_message_events_to_on_event():
    seen: list[AgentEvent] = []
    events = EventBuffer(seen.append)
    _collect(relay_react(FakeGraph(_ai("hi")), {}, recursion_limit=5, state={}, events=events))
    assert [e.kind for e in seen] == ["message"]


def test_relay_react_uses_custom_mapper():
    events = EventBuffer()
    out, _ = _collect(
        relay_react(
            FakeGraph(_ai("hi")), {}, recursion_limit=5, state={}, events=events,
            to_event=lambda m: AgentEvent("custom", {"text": m.content}),
        )
    )
    assert [(e.kind, e.data) for e in out] == [("custom", {"text": "hi"})]


def test_drive_react_calls_on_message_and_returns_outcome():
    seen = []
    graph = FakeGraph(_ai("a"), _ai("b"), GraphRecursionError("limit"))
    outcome = drive_react(graph, {}, recursion_limit=3, state={}, on_message=seen.append)
    assert [m.content for m in seen] == ["a", "b"]
    assert outcome == "budget_exhausted"


# ------------------------------------------------------------- pump_events


def test_pump_events_yields_live_and_returns_result():
    released = threading.Event()

    def work(put):
        put(AgentEvent("first"))
        # Blocks until the consumer has seen "first", so batching would fail here.
        if not released.wait(timeout=5):
            raise AssertionError("consumer never received the first event")
        put(AgentEvent("second"))
        return 42

    gen = pump_events(work)
    assert next(gen).kind == "first"
    released.set()
    rest, result = _collect(gen)
    assert [e.kind for e in rest] == ["second"]
    assert result == 42


def test_pump_events_reraises_worker_errors_after_earlier_events():
    def work(put):
        put(AgentEvent("before"))
        raise ValueError("bad node")

    gen = pump_events(work)
    assert next(gen).kind == "before"
    with pytest.raises(ValueError, match="bad node"):
        next(gen)


def test_pump_events_calls_on_event_on_the_consumer_thread():
    worker, notified = [], []

    def work(put):
        worker.append(threading.get_ident())
        put(AgentEvent("x"))

    list(pump_events(work, on_event=lambda _e: notified.append(threading.get_ident())))
    assert notified == [threading.get_ident()]
    assert worker[0] != threading.get_ident()


_CALLER = contextvars.ContextVar("caller", default="unset")


def test_pump_events_runs_work_in_the_callers_context():
    token = _CALLER.set("caller")
    try:
        seen = []
        list(pump_events(lambda _put: seen.append(_CALLER.get())))
    finally:
        _CALLER.reset(token)
    assert seen == ["caller"]
