"""LLM provider selection.

A single :func:`load_llm` entry point returns a LangChain ``BaseChatModel``
instance for the configured provider. Adding a new provider means adding a
branch here and listing the wheel in :mod:`requirements.txt`.
"""

from .providers import load_llm  # noqa: F401
