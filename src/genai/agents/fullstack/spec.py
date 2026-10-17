"""Typed project specification produced by the planner node.

The planner LLM emits a JSON document; we parse it through these Pydantic
models so the downstream nodes (scaffolder, specialists) can rely on a
typed, validated shape rather than free-form text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

# --------------------------------------------------------------------- atoms

FieldType = Literal[
    "str", "int", "float", "bool", "datetime", "date", "uuid", "text", "json"
]
RelationKind = Literal["one_to_many", "many_to_one", "many_to_many", "one_to_one"]
HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")
_PASCAL = re.compile(r"^[A-Z][A-Za-z0-9]*$")


class EntityField(BaseModel):
    name: str
    type: FieldType = "str"
    nullable: bool = False
    unique: bool = False
    default: Any | None = None
    description: str | None = None

    @field_validator("name")
    @classmethod
    def _ident(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError(f"field name must be snake_case identifier: {v!r}")
        return v


class Relation(BaseModel):
    kind: RelationKind
    target: str  # entity name (PascalCase)
    backref: str | None = None

    @field_validator("target")
    @classmethod
    def _pascal(cls, v: str) -> str:
        if not _PASCAL.match(v):
            raise ValueError(f"relation target must be PascalCase: {v!r}")
        return v


class Entity(BaseModel):
    name: str  # PascalCase, e.g. "Task"
    table_name: str | None = None  # defaults to snake_case plural
    description: str | None = None
    owned_by_user: bool = False  # adds a user_id FK + ownership filter
    fields: list[EntityField] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _pascal(cls, v: str) -> str:
        if not _PASCAL.match(v):
            raise ValueError(f"entity name must be PascalCase: {v!r}")
        return v


class Endpoint(BaseModel):
    method: HttpMethod
    path: str  # e.g. "/tasks/{id}"
    entity: str | None = None  # PascalCase, optional for custom routes
    summary: str = ""
    requires_auth: bool = True


class Page(BaseModel):
    name: str  # PascalCase component name, e.g. "TasksPage"
    route: str  # e.g. "/tasks"
    description: str = ""
    requires_auth: bool = True
    entities: list[str] = Field(default_factory=list)  # entities used


class AuthSpec(BaseModel):
    enabled: bool = True
    allow_register: bool = True
    user_fields: list[EntityField] = Field(
        default_factory=lambda: [
            EntityField(name="email", type="str", unique=True),
            EntityField(name="hashed_password", type="str"),
        ]
    )


class ProjectMeta(BaseModel):
    name: str  # snake_case, used as directory and compose project name
    title: str  # human-readable
    description: str = ""

    @field_validator("name")
    @classmethod
    def _ident(cls, v: str) -> str:
        if not _IDENT.match(v):
            raise ValueError(f"project name must be snake_case: {v!r}")
        return v


class ProjectSpec(BaseModel):
    meta: ProjectMeta
    auth: AuthSpec = Field(default_factory=AuthSpec)
    entities: list[Entity] = Field(default_factory=list)
    endpoints: list[Endpoint] = Field(default_factory=list)
    pages: list[Page] = Field(default_factory=list)
    notes: str = ""

    # ------------------------------------------------------------ helpers
    def entity_by_name(self, name: str) -> Entity | None:
        return next((e for e in self.entities if e.name == name), None)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    # ---------------------------------------------------- LLM JSON parsing
    @classmethod
    def from_llm_json(cls, raw: str) -> ProjectSpec:
        """Parse a planner LLM response into a :class:`ProjectSpec`.

        Tolerates Markdown code fences and surrounding prose; raises
        :class:`ValueError` with a useful message on malformed input.
        """
        text = _extract_json_object(raw)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"planner output is not valid JSON: {exc.msg}") from exc
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"planner JSON does not match ProjectSpec: {exc}") from exc


# --------------------------------------------------------------------- utils
_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def _extract_json_object(raw: str) -> str:
    """Pull the first JSON object out of an LLM response.

    Handles three common shapes: pure JSON, a fenced ```json block, or a
    JSON object embedded in prose.
    """
    if raw is None:
        raise ValueError("planner returned no content")
    raw = raw.strip()
    if not raw:
        raise ValueError("planner returned empty content")
    m = _FENCE.search(raw)
    if m:
        return m.group(1).strip()
    # First '{' to its matching '}' (depth-counted, ignoring strings).
    start = raw.find("{")
    if start == -1:
        raise ValueError("planner output contains no JSON object")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    raise ValueError("planner output has unbalanced braces")
