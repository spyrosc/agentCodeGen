"""Sandbox transport: ``docker exec`` against a long-running sidecar container.

The sidecar (``genai-sandbox`` by default) shares the workspace volume with
the agent container. We shell into it through the mounted Docker socket.

SECURITY: mounting ``/var/run/docker.sock`` grants the agent container
effective root on the host. Use this transport for local development only.
For shared/production deployments, implement an HTTP-shim transport that
satisfies the same :class:`~genai.sandbox.base.Sandbox` protocol.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from ..core.config import SandboxSettings, get_settings
from ..core.logging import get_logger
from ..core.text import truncate_middle
from .base import ExecResult

log = get_logger("sandbox.docker_exec")


@dataclass
class DockerExecSandbox:
    """``docker exec`` based :class:`~genai.sandbox.base.Sandbox` implementation."""

    container: str
    workspace_root: str
    max_output_chars: int = 8_000

    @classmethod
    def from_settings(cls, settings: SandboxSettings | None = None) -> DockerExecSandbox:
        cfg = settings or get_settings().sandbox
        return cls(
            container=cfg.container_name,
            workspace_root=cfg.workspace_root,
            max_output_chars=cfg.max_output_chars,
        )

    # ----------------------------------------------------------- protocol
    def exec(self, project_id: str, command: str, timeout_s: int) -> ExecResult:
        timeout_s = max(5, min(int(timeout_s), 600))
        project_dir = f"{self.workspace_root.rstrip('/')}/{project_id}"
        argv = [
            "docker",
            "exec",
            "-w",
            project_dir,
            self.container,
            "sh",
            "-c",
            command,
        ]
        log.debug("docker exec: %s", " ".join(argv))
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                exit_code=124,
                output=truncate_middle(
                    f"<command timed out after {timeout_s}s>\n"
                    f"{exc.stdout or ''}\n{exc.stderr or ''}",
                    self.max_output_chars,
                ),
            )
        except FileNotFoundError:
            return ExecResult(
                exit_code=127,
                output=(
                    "<docker CLI not available; rebuild the agent image and "
                    "ensure /var/run/docker.sock is mounted into the container>"
                ),
            )
        combined = (proc.stdout or "") + (proc.stderr or "")
        return ExecResult(
            exit_code=proc.returncode,
            output=truncate_middle(combined, self.max_output_chars),
        )
