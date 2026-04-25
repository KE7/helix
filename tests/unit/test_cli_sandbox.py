"""Unit tests for sandbox auth CLI commands."""

from __future__ import annotations

from subprocess import CompletedProcess

from click.testing import CliRunner

from helix.cli import cli


def test_sandbox_login_invokes_backend_auth_volume(mocker):
    mock_run = mocker.patch(
        "helix.cli.run_sandbox_auth_command",
        return_value=CompletedProcess([], 0, stdout="", stderr=""),
    )

    result = CliRunner().invoke(cli, ["sandbox", "login", "cursor"])

    assert result.exit_code == 0
    mock_run.assert_called_once_with(
        "cursor",
        action="login",
        image=None,
        network="bridge",
        add_host_gateway=False,
        interactive=True,
    )
    assert "helix-auth-cursor" in result.output


def test_sandbox_status_all_backends(mocker):
    mock_run = mocker.patch(
        "helix.cli.run_sandbox_auth_command",
        return_value=CompletedProcess([], 0, stdout="ok\n", stderr=""),
    )

    result = CliRunner().invoke(cli, ["sandbox", "status"])

    assert result.exit_code == 0
    assert mock_run.call_count == 5
    assert "claude" in result.output
    assert "opencode" in result.output
