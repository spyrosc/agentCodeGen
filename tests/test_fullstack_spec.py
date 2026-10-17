"""Tests for the fullstack ProjectSpec parser."""

import pytest

from genai.agents.fullstack.spec import (
    Entity,
    EntityField,
    ProjectMeta,
    ProjectSpec,
)

_VALID = """
{
  "meta": {"name": "todo_app", "title": "Todo", "description": "tracks todos"},
  "auth": {"enabled": true, "allow_register": true},
  "entities": [
    {"name": "Task", "owned_by_user": true,
     "fields": [{"name": "title", "type": "str"}, {"name": "done", "type": "bool"}]}
  ],
  "endpoints": [
    {"method": "GET", "path": "/tasks", "entity": "Task", "summary": "list", "requires_auth": true}
  ],
  "pages": [{"name": "TasksPage", "route": "/tasks", "entities": ["Task"]}],
  "notes": ""
}
"""


def test_parse_clean_json():
    s = ProjectSpec.from_llm_json(_VALID)
    assert s.meta.name == "todo_app"
    assert [e.name for e in s.entities] == ["Task"]
    assert s.entities[0].owned_by_user is True


def test_parse_with_code_fence():
    s = ProjectSpec.from_llm_json(f"Here you go:\n```json\n{_VALID}\n```")
    assert s.meta.title == "Todo"


def test_parse_with_surrounding_prose():
    s = ProjectSpec.from_llm_json(f"Sure! {_VALID}\nDone.")
    assert s.entities[0].fields[1].name == "done"


def test_rejects_bad_entity_name():
    bad = _VALID.replace('"name": "Task"', '"name": "task_lowercase"')
    with pytest.raises(ValueError):
        ProjectSpec.from_llm_json(bad)


def test_rejects_bad_field_name():
    bad = _VALID.replace('"name": "title"', '"name": "Title"')
    with pytest.raises(ValueError):
        ProjectSpec.from_llm_json(bad)


def test_rejects_no_json():
    with pytest.raises(ValueError):
        ProjectSpec.from_llm_json("just words, no JSON at all")


def test_round_trip_to_json():
    s = ProjectSpec.from_llm_json(_VALID)
    s2 = ProjectSpec.model_validate_json(s.to_json())
    assert s2 == s


def test_constructed_minimal():
    s = ProjectSpec(meta=ProjectMeta(name="x", title="X"))
    s.entities.append(Entity(name="Foo", fields=[EntityField(name="bar")]))
    assert s.entity_by_name("Foo") is not None
    assert s.entity_by_name("Missing") is None
