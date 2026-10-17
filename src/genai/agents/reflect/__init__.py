"""Self-reflection agent.

Operates on an *isolated clone* of the platform's own source tree, never on
the live working copy. Validates proposed changes against the eval harness
plus pytest + ruff, and emits a reviewable patch + branch. See
``docs/self-reflection.md`` for the security model.
"""

from .agent import ReflectAgent  # noqa: F401
