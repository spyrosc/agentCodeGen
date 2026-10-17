"""Sandboxed code execution.

Two responsibilities:

1. **Workspaces** ([workspace.py](workspace.py)) — per-project directories on a
   shared volume, with directory-traversal-safe path resolution.
2. **Sandbox transports** — a small :class:`Sandbox` protocol with a
   :class:`docker_exec.DockerExecSandbox` implementation. Future transports
   (e.g. an HTTP shim, Firecracker microVM, gVisor) plug in here without any
   agent-side changes.
"""

from .base import ExecResult, Sandbox  # noqa: F401
from .docker_exec import DockerExecSandbox  # noqa: F401
from .workspace import Workspace, new_project_id  # noqa: F401
