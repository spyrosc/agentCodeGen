"""Centralised, env-driven configuration.

A single :class:`Settings` dataclass is the source of truth for runtime
configuration. Every module reads it via :func:`get_settings` rather than
calling ``os.getenv`` directly — this makes the configuration surface
explicit, testable, and trivially overridable from tests or scripts.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMSettings:
    """LLM provider configuration. Provider is inferred from ``model``."""

    model: str = "llama2"
    temperature: float = 0.0
    ollama_base_url: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    anthropic_api_key: str | None = None
    # Anthropic requires an explicit output cap; the SDK default (1024) is too small for codegen.
    anthropic_max_tokens: int = 8192
    aws_region: str | None = None
    # Per-request timeout (0 disables) and provider-level retries on transient errors.
    request_timeout_s: float = 600.0
    max_retries: int = 2


@dataclass(frozen=True)
class SandboxSettings:
    """Configuration for the code-execution sandbox."""

    # Where workspaces live INSIDE both the agent and sandbox containers.
    workspace_root: str = "/workspaces"
    # docker-exec transport
    container_name: str = "genai-sandbox"
    default_timeout_s: int = 180
    # Output truncation budgets exposed to the LLM
    max_file_bytes: int = 64_000
    max_output_chars: int = 8_000


@dataclass(frozen=True)
class AgentSettings:
    """Common knobs for any agent built on this platform."""

    max_iterations: int = 40
    enable_tracing: bool = False


@dataclass(frozen=True)
class FullStackSettings:
    """Knobs specific to the multi-agent ``fullstack`` agent."""

    backend_port: int = 8000
    frontend_port: int = 5173
    postgres_image: str = "postgres:16-alpine"
    max_review_passes: int = 2
    enable_frontend_build: bool = True


@dataclass(frozen=True)
class ReflectSettings:
    """Knobs specific to the self-reflection ``reflect`` agent.

    Defaults are intentionally conservative: lessons injection and the eval
    non-regression gate are *opt-in*, so existing tests and runs behave
    identically until an operator explicitly turns them on.
    """

    # Source-tree isolation
    repo_mount_path: str = "/repo"
    base_branch: str = "main"

    # Allow / deny — relative POSIX prefixes against the clone root.
    allowed_paths: tuple[str, ...] = ("src/", "tests/", "docs/", "scripts/", "evals/")
    denied_paths: tuple[str, ...] = (
        "docker-compose.yml",
        "docker/",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        ".env",
        ".github/",
        "Makefile",
    )

    # Per-run hard caps (defence-in-depth on top of the LLM's iteration limit).
    max_files_changed: int = 15
    max_patch_lines: int = 800
    max_write_calls_per_run: int = 30

    # LessonStore
    lessons_path: str = "/workspaces/.lessons/lessons.jsonl"
    lessons_enabled: bool = False  # opt-in injection into other agents' prompts
    lessons_top_k: int = 5
    lessons_max: int = 10_000  # rotate beyond this
    lessons_autoseed: bool = False  # populate from eval-run artefacts

    # Eval non-regression gate
    eval_gate_suite: str = "evals/suites/codegen_smoke.yaml"
    eval_gate_skip_functional: bool = True
    eval_gate_required: bool = False


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    fullstack: FullStackSettings = field(default_factory=FullStackSettings)
    reflect: ReflectSettings = field(default_factory=ReflectSettings)
    log_level: str = "INFO"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _from_env() -> Settings:
    return Settings(
        llm=LLMSettings(
            model=os.getenv("LLM", "llama2"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192")),
            aws_region=os.getenv("AWS_DEFAULT_REGION"),
            request_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "600")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        ),
        sandbox=SandboxSettings(
            workspace_root=os.getenv("WORKSPACE_ROOT", "/workspaces"),
            container_name=os.getenv("SANDBOX_CONTAINER", "genai-sandbox"),
            default_timeout_s=int(os.getenv("SANDBOX_TIMEOUT_S", "180")),
            max_file_bytes=int(os.getenv("SANDBOX_MAX_FILE_BYTES", "64000")),
            max_output_chars=int(os.getenv("SANDBOX_MAX_OUTPUT_CHARS", "8000")),
        ),
        agent=AgentSettings(
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "40")),
            enable_tracing=_bool(os.getenv("LANGCHAIN_TRACING_V2"), False),
        ),
        fullstack=FullStackSettings(
            backend_port=int(os.getenv("FULLSTACK_BACKEND_PORT", "8000")),
            frontend_port=int(os.getenv("FULLSTACK_FRONTEND_PORT", "5173")),
            postgres_image=os.getenv("FULLSTACK_POSTGRES_IMAGE", "postgres:16-alpine"),
            max_review_passes=int(os.getenv("FULLSTACK_MAX_REVIEW_PASSES", "2")),
            enable_frontend_build=_bool(
                os.getenv("FULLSTACK_ENABLE_FRONTEND_BUILD"), True
            ),
        ),
        reflect=ReflectSettings(
            repo_mount_path=os.getenv("REFLECT_REPO_MOUNT", "/repo"),
            base_branch=os.getenv("REFLECT_BASE_BRANCH", "main"),
            max_files_changed=int(os.getenv("REFLECT_MAX_FILES_CHANGED", "15")),
            max_patch_lines=int(os.getenv("REFLECT_MAX_PATCH_LINES", "800")),
            max_write_calls_per_run=int(os.getenv("REFLECT_MAX_WRITES", "30")),
            lessons_path=os.getenv(
                "REFLECT_LESSONS_PATH", "/workspaces/.lessons/lessons.jsonl"
            ),
            lessons_enabled=_bool(os.getenv("REFLECT_LESSONS_ENABLED"), False),
            lessons_top_k=int(os.getenv("REFLECT_LESSONS_TOP_K", "5")),
            lessons_max=int(os.getenv("REFLECT_LESSONS_MAX", "10000")),
            lessons_autoseed=_bool(os.getenv("REFLECT_LESSONS_AUTOSEED"), False),
            eval_gate_suite=os.getenv(
                "REFLECT_EVAL_GATE_SUITE", "evals/suites/codegen_smoke.yaml"
            ),
            eval_gate_skip_functional=_bool(
                os.getenv("REFLECT_EVAL_GATE_SKIP_FUNCTIONAL"), True
            ),
            eval_gate_required=_bool(os.getenv("REFLECT_EVAL_GATE_REQUIRED"), False),
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` (cached)."""
    return _from_env()


def reload_settings() -> Settings:
    """Force re-read from the environment. Useful in tests."""
    get_settings.cache_clear()
    return get_settings()
