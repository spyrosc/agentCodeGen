"""Tests for the codegen tool builders. Sandbox is mocked."""

from unittest.mock import MagicMock

from genai.agents.codegen.tools import build_tools
from genai.sandbox.base import ExecResult
from genai.sandbox.workspace import Workspace, new_project_id


def _ws(tmp_path) -> Workspace:
    return Workspace.create(new_project_id("t"), base=tmp_path)


def test_write_then_read_then_list(tmp_path):
    ws = _ws(tmp_path)
    sandbox = MagicMock()
    tools, state = build_tools(ws, sandbox)
    by_name = {t.name: t for t in tools}

    out = by_name["write_file"].invoke({"path": "main.py", "content": "print('hi')\n"})
    assert "wrote" in out
    assert "print('hi')" in by_name["read_file"].invoke({"path": "main.py"})
    assert "main.py" in by_name["list_files"].invoke({"path": "."})


def test_write_blocks_traversal(tmp_path):
    ws = _ws(tmp_path)
    tools, _ = build_tools(ws, MagicMock())
    by_name = {t.name: t for t in tools}
    try:
        by_name["write_file"].invoke({"path": "../escape.txt", "content": "x"})
    except ValueError:
        pass
    else:
        raise AssertionError("traversal should have raised ValueError")


def test_run_in_sandbox_dispatches_to_transport(tmp_path):
    ws = _ws(tmp_path)
    sandbox = MagicMock()
    sandbox.exec.return_value = ExecResult(exit_code=0, output="ok")
    tools, _ = build_tools(ws, sandbox)
    by_name = {t.name: t for t in tools}
    out = by_name["run_in_sandbox"].invoke({"command": "echo hi", "timeout": 30})
    sandbox.exec.assert_called_once()
    assert out.startswith("exit=0")


def test_submit_project_flips_state(tmp_path):
    ws = _ws(tmp_path)
    tools, state = build_tools(ws, MagicMock())
    by_name = {t.name: t for t in tools}
    out = by_name["submit_project"].invoke({"summary": "all done"})
    assert out == "PROJECT_SUBMITTED"
    assert state["submitted"] is True
    assert state["summary"] == "all done"


def test_events_emitted(tmp_path):
    ws = _ws(tmp_path)
    events = []
    tools, _ = build_tools(ws, MagicMock(), on_event=lambda k, p: events.append((k, p)))
    by_name = {t.name: t for t in tools}
    by_name["write_file"].invoke({"path": "a.py", "content": "x = 1"})
    by_name["read_file"].invoke({"path": "a.py"})
    kinds = [k for k, _ in events]
    assert "write_file" in kinds and "read_file" in kinds
