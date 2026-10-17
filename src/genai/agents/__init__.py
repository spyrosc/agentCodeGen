"""Agent platform.

Agents implement the :class:`Agent` protocol and register themselves in the
:mod:`registry`. The Streamlit UI discovers agents through the registry, so
adding a new agent is a one-file change with no UI edits.

See [docs/adding-an-agent.md](../../../docs/adding-an-agent.md) for a
step-by-step guide.
"""

# Side-effect import: register built-in agents.
from . import (
    codegen,  # noqa: F401
    fullstack,  # noqa: F401
    reflect,  # noqa: F401
)
from .base import Agent, AgentEvent, AgentResult  # noqa: F401
from .registry import all_agents, get, register  # noqa: F401
