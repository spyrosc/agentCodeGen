"""Tests for the scoped workspace + shared FS tools."""

import pytest

from genai.agents._common.fs_tools import build_fs_tools, scope_to
from genai.sandbox.workspace import Workspace, new_project_id


def _ws(tmp_path) -> Workspace:
    return Workspace.create(new_project_id("t"), base=tmp_path)


def test_scope_to_resolves_under_subdir(tmp_path):
    ws = _ws(tmp_path)
    scoped = scope_to(ws, "backend")
    assert scoped.root == ws.root / "backend"
    assert scoped.resolve("app/main.py") == ws.root / "backend" / "app" / "main.py"


def test_scope_to_blocks_traversal(tmp_path):
    ws = _ws(tmp_path)
    scoped = scope_to(ws, "backend")
    with pytest.raises(ValueError):
        scoped.resolve("../escape.txt")
    with pytest.raises(ValueError):
        scoped.resolve("../../etc/passwd")


def test_scope_to_rejects_unsafe_subdir(tmp_path):
    ws = _ws(tmp_path)
    with pytest.raises(ValueError):
        scope_to(ws, "../escape")
    with pytest.raises(ValueError):
        scope_to(ws, "/etc")


def test_scoped_write_then_read_then_list(tmp_path):
    ws = _ws(tmp_path)
    scoped = scope_to(ws, "frontend")
    tools = build_fs_tools(scoped, label="frontend")
    by_name = {t.name: t for t in tools}

    by_name["write_file"].invoke({"path": "src/App.tsx", "content": "// hi\n"})
    out = by_name["read_file"].invoke({"path": "src/App.tsx"})
    assert "// hi" in out
    listing = by_name["list_files"].invoke({"path": "."})
    assert "src/App.tsx" in listing
    # File lands under frontend/, never at workspace root.
    assert (ws.root / "frontend" / "src" / "App.tsx").exists()
    assert not (ws.root / "src" / "App.tsx").exists()


def test_scoped_write_blocks_traversal(tmp_path):
    ws = _ws(tmp_path)
    scoped = scope_to(ws, "backend")
    tools = build_fs_tools(scoped)
    by_name = {t.name: t for t in tools}
    with pytest.raises(ValueError):
        by_name["write_file"].invoke({"path": "../leak.txt", "content": "x"})


def test_event_label_is_attached(tmp_path):
    ws = _ws(tmp_path)
    scoped = scope_to(ws, "backend")
    events: list[tuple[str, dict]] = []
    tools = build_fs_tools(scoped, on_event=lambda k, p: events.append((k, p)), label="backend")
    by_name = {t.name: t for t in tools}
    by_name["write_file"].invoke({"path": "x.py", "content": "x = 1"})
    assert any(k == "write_file" and p.get("scope") == "backend" for k, p in events)
