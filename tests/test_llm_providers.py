"""Tests for :func:`genai.llm.providers.load_llm` wiring (no network calls)."""

from __future__ import annotations

import pytest

from genai.core.config import LLMSettings
from genai.llm.providers import load_llm


class _Recorder:
    """Stands in for a LangChain chat class and records constructor kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture()
def recorder(monkeypatch):
    def install(module_name: str, class_name: str) -> None:
        module = pytest.importorskip(module_name)
        monkeypatch.setattr(module, class_name, _Recorder)

    return install


def test_openai_gets_timeout_and_retries(recorder):
    recorder("langchain_openai", "ChatOpenAI")
    llm = load_llm(LLMSettings(model="gpt-4o-mini", request_timeout_s=90, max_retries=4))
    assert llm.kwargs["timeout"] == 90
    assert llm.kwargs["max_retries"] == 4
    assert llm.kwargs["model"] == "gpt-4o-mini"


def test_gemini_gets_timeout_and_retries(recorder):
    recorder("langchain_google_genai", "ChatGoogleGenerativeAI")
    llm = load_llm(LLMSettings(model="gemini-1.5-pro", request_timeout_s=30, max_retries=1))
    assert llm.kwargs["timeout"] == 30
    assert llm.kwargs["max_retries"] == 1


def test_anthropic_gets_timeout_retries_and_max_tokens(recorder):
    recorder("langchain_anthropic", "ChatAnthropic")
    llm = load_llm(
        LLMSettings(
            model="claude-sonnet-4-5",
            anthropic_api_key="sk-ant-test",
            anthropic_max_tokens=4096,
            request_timeout_s=60,
            max_retries=5,
        )
    )
    assert llm.kwargs["model"] == "claude-sonnet-4-5"
    assert llm.kwargs["api_key"] == "sk-ant-test"
    assert llm.kwargs["max_tokens"] == 4096
    assert llm.kwargs["timeout"] == 60
    assert llm.kwargs["max_retries"] == 5
    assert llm.kwargs["streaming"] is True


def test_anthropic_without_key_defers_to_env(recorder):
    recorder("langchain_anthropic", "ChatAnthropic")
    llm = load_llm(LLMSettings(model="claude-3-5-haiku-latest"))
    assert "api_key" not in llm.kwargs
    assert llm.kwargs["max_tokens"] == 8192


@pytest.mark.parametrize(
    ("model", "model_id"),
    [("claudev2", "anthropic.claude-v2"), ("anthropic.claude-3-haiku", "anthropic.claude-3-haiku")],
)
def test_bedrock_gets_region_timeout_and_retries(recorder, model, model_id):
    recorder("langchain_aws", "ChatBedrock")
    llm = load_llm(
        LLMSettings(model=model, aws_region="eu-west-1", request_timeout_s=120, max_retries=3)
    )
    assert llm.kwargs["model_id"] == model_id
    assert llm.kwargs["region_name"] == "eu-west-1"
    config = llm.kwargs["config"]
    assert config.read_timeout == 120
    assert config.retries == {"max_attempts": 3, "mode": "standard"}


def test_ollama_gets_client_timeout_and_base_url(recorder):
    recorder("langchain_ollama", "ChatOllama")
    llm = load_llm(
        LLMSettings(model="llama3", ollama_base_url="http://ollama:11434", request_timeout_s=45)
    )
    assert llm.kwargs["client_kwargs"] == {"timeout": 45}
    assert llm.kwargs["base_url"] == "http://ollama:11434"


def test_zero_timeout_means_no_timeout_and_negative_retries_clamp(recorder):
    recorder("langchain_openai", "ChatOpenAI")
    llm = load_llm(LLMSettings(model="gpt-4o", request_timeout_s=0, max_retries=-1))
    assert llm.kwargs["timeout"] is None
    assert llm.kwargs["max_retries"] == 0
