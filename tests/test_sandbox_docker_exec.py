"""Tests for DockerExecSandbox using a stubbed subprocess.run."""

import subprocess
from unittest.mock import patch

from genai.sandbox.docker_exec import DockerExecSandbox


def _fake_completed(stdout="", stderr="", code=0):
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=stdout, stderr=stderr)


def test_exec_builds_correct_argv():
    sbx = DockerExecSandbox(container="genai-sandbox", workspace_root="/workspaces")
    with patch("subprocess.run", return_value=_fake_completed(stdout="hello\n")) as m:
        result = sbx.exec("proj-123", "echo hello", timeout_s=10)
    argv = m.call_args.args[0]
    assert argv[:2] == ["docker", "exec"]
    assert "-w" in argv and "/workspaces/proj-123" in argv
    assert argv[argv.index(sbx.container) + 1] == "sh"
    assert result.exit_code == 0
    assert "hello" in result.output


def test_exec_handles_timeout():
    sbx = DockerExecSandbox(container="genai-sandbox", workspace_root="/workspaces")
    err = subprocess.TimeoutExpired(cmd="x", timeout=1, output="partial", stderr="")
    with patch("subprocess.run", side_effect=err):
        result = sbx.exec("p", "sleep 9999", timeout_s=1)
    assert result.exit_code == 124
    assert "timed out" in result.output


def test_exec_missing_docker_returns_127():
    sbx = DockerExecSandbox(container="genai-sandbox", workspace_root="/workspaces")
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        result = sbx.exec("p", "ls", timeout_s=5)
    assert result.exit_code == 127
    assert "docker CLI" in result.output


def test_exec_truncates_huge_output():
    sbx = DockerExecSandbox(container="x", workspace_root="/workspaces", max_output_chars=200)
    big = "A" * 5000
    with patch("subprocess.run", return_value=_fake_completed(stdout=big)):
        result = sbx.exec("p", "x", timeout_s=5)
    assert "truncated" in result.output
    assert len(result.output) < 1000


def test_exec_clamps_timeout_to_safe_range():
    sbx = DockerExecSandbox(container="x", workspace_root="/workspaces")
    with patch("subprocess.run", return_value=_fake_completed()) as m:
        sbx.exec("p", "ls", timeout_s=99999)
    assert m.call_args.kwargs["timeout"] == 600
    with patch("subprocess.run", return_value=_fake_completed()) as m:
        sbx.exec("p", "ls", timeout_s=0)
    assert m.call_args.kwargs["timeout"] == 5
