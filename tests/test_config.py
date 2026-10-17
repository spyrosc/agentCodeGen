from genai.core.config import reload_settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LLM", "gpt-4o-mini")
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "12")
    monkeypatch.setenv("SANDBOX_CONTAINER", "my-sbx")
    s = reload_settings()
    assert s.llm.model == "gpt-4o-mini"
    assert s.agent.max_iterations == 12
    assert s.sandbox.container_name == "my-sbx"


def test_settings_defaults(monkeypatch):
    for var in ("LLM", "AGENT_MAX_ITERATIONS", "SANDBOX_CONTAINER"):
        monkeypatch.delenv(var, raising=False)
    s = reload_settings()
    assert s.llm.model == "llama2"
    assert s.agent.max_iterations == 40
    assert s.sandbox.container_name == "genai-sandbox"


def test_llm_timeout_and_retries_from_env(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_S", "42.5")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    s = reload_settings()
    assert s.llm.request_timeout_s == 42.5
    assert s.llm.max_retries == 5


def test_llm_timeout_and_retries_defaults(monkeypatch):
    monkeypatch.delenv("LLM_TIMEOUT_S", raising=False)
    monkeypatch.delenv("LLM_MAX_RETRIES", raising=False)
    s = reload_settings()
    assert s.llm.request_timeout_s == 600
    assert s.llm.max_retries == 2


def test_sandbox_output_budgets_from_env(monkeypatch):
    monkeypatch.setenv("SANDBOX_MAX_FILE_BYTES", "1234")
    monkeypatch.setenv("SANDBOX_MAX_OUTPUT_CHARS", "567")
    s = reload_settings()
    assert s.sandbox.max_file_bytes == 1234
    assert s.sandbox.max_output_chars == 567
    monkeypatch.delenv("SANDBOX_MAX_FILE_BYTES")
    monkeypatch.delenv("SANDBOX_MAX_OUTPUT_CHARS")
    s = reload_settings()
    assert s.sandbox.max_file_bytes == 64_000
    assert s.sandbox.max_output_chars == 8_000
