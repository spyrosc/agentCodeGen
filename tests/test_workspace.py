
import pytest

from genai.sandbox.workspace import Workspace, new_project_id


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(new_project_id("test"), base=tmp_path)


def test_resolve_creates_paths_inside_root(ws):
    p = ws.resolve("a/b/c.py")
    assert str(p).startswith(str(ws.root))


def test_resolve_blocks_traversal(ws):
    with pytest.raises(ValueError):
        ws.resolve("../../etc/passwd")


def test_resolve_strips_leading_slash(ws):
    # absolute-looking input is reinterpreted relative to the project root
    p = ws.resolve("/a/b/c.py")
    assert str(p).startswith(str(ws.root))


def test_write_then_list(ws):
    target = ws.resolve("pkg/mod.py")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x = 1")
    tree = ws.list_tree()
    assert "pkg/" in tree
    assert "pkg/mod.py" in tree


def test_zip_and_cleanup(ws):
    (ws.resolve("a.txt")).write_text("hi")
    blob = ws.zip_bytes()
    assert blob.startswith(b"PK")
    ws.cleanup()
    assert not ws.root.exists()


def test_unsafe_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        Workspace.create("../escape", base=tmp_path)
