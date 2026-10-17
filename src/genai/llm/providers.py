"""LLM provider factory.

Heuristics on the model name pick the right LangChain wrapper. Adding a
provider: add a new branch to :func:`load_llm`. Keep this module thin — no
prompt templates, no retrievers — so any agent can reuse it.
"""

from __future__ import annotations

from typing import Any

from ..core.config import LLMSettings, get_settings
from ..core.logging import get_logger

log = get_logger("llm")


_AWS_PREFIXES = (
    "ai21.",
    "amazon.",
    "anthropic.",
    "cohere.",
    "meta.",
    "mistral.",
)


def load_llm(settings: LLMSettings | None = None, **overrides: Any):
    """Return a streaming chat LLM matching the configured provider.

    Imports are inlined so missing optional dependencies only fail when their
    provider is actually selected. Every provider gets the configured request
    timeout; all but Ollama (local, no retry support) get provider-level retries.
    """
    cfg = settings or get_settings().llm
    model = overrides.get("model", cfg.model)
    temperature = float(overrides.get("temperature", cfg.temperature))
    timeout = cfg.request_timeout_s or None
    retries = max(0, cfg.max_retries)

    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        from langchain_openai import ChatOpenAI

        log.info("LLM: OpenAI %s", model)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True,
            timeout=timeout,
            max_retries=retries,
        )

    # Direct Anthropic API. Bare "claude-*" ids only; Bedrock ids use "anthropic." prefixes.
    if model.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic

        log.info("LLM: Anthropic %s", model)
        anthropic_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": cfg.anthropic_max_tokens,
            "streaming": True,
            "timeout": timeout,
            "max_retries": retries,
        }
        if cfg.anthropic_api_key:
            anthropic_kwargs["api_key"] = cfg.anthropic_api_key
        return ChatAnthropic(**anthropic_kwargs)

    if model.lower() == "claudev2" or any(model.startswith(p) for p in _AWS_PREFIXES):
        from botocore.config import Config
        from langchain_aws import ChatBedrock

        model_id = "anthropic.claude-v2" if model.lower() == "claudev2" else model
        log.info("LLM: AWS Bedrock %s", model_id)
        return ChatBedrock(
            model_id=model_id,
            model_kwargs={"temperature": temperature},
            streaming=True,
            region_name=cfg.aws_region,
            config=Config(
                read_timeout=timeout,
                retries={"max_attempts": retries, "mode": "standard"},
            ),
        )

    if model.startswith("gemini") or model.startswith("models/gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        log.info("LLM: Google Generative AI %s", model)
        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature, timeout=timeout, max_retries=retries
        )

    # Default: Ollama (local).
    from langchain_ollama import ChatOllama

    base_url = overrides.get("ollama_base_url", cfg.ollama_base_url)
    log.info("LLM: Ollama %s @ %s", model, base_url or "<default>")
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "client_kwargs": {"timeout": timeout},
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs)
