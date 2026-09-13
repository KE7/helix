"""Unit tests for sandbox auth CLI commands."""

from __future__ import annotations

from subprocess import CompletedProcess, TimeoutExpired

from click.testing import CliRunner

from helix.cli import cli
from helix.sandbox import SANDBOX_AUTH_COMMAND_TIMEOUT_SECONDS


def test_sandbox_login_invokes_backend_auth_volume(mocker):
    mocker.patch("os.isatty", return_value=True)
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
        extra_hosts={},
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


def test_sandbox_status_is_bounded_and_names_its_container(mocker):
    """An unattended probe must not be able to hang the CLI forever, and the
    container has to be addressable so a timed-out client can stop it."""
    mock_run = mocker.patch(
        "helix.cli.run_sandbox_auth_command",
        return_value=CompletedProcess([], 0, stdout="ok\n", stderr=""),
    )

    assert CliRunner().invoke(cli, ["sandbox", "status", "codex"]).exit_code == 0

    kwargs = mock_run.call_args.kwargs
    assert kwargs["timeout"] == SANDBOX_AUTH_COMMAND_TIMEOUT_SECONDS
    assert kwargs["container_name"].startswith("helix-auth-status-codex-")


def test_sandbox_logout_is_bounded(mocker):
    mock_run = mocker.patch(
        "helix.cli.run_sandbox_auth_command",
        return_value=CompletedProcess([], 0, stdout="", stderr=""),
    )

    assert CliRunner().invoke(cli, ["sandbox", "logout", "codex"]).exit_code == 0

    kwargs = mock_run.call_args.kwargs
    assert kwargs["timeout"] == SANDBOX_AUTH_COMMAND_TIMEOUT_SECONDS
    assert kwargs["container_name"].startswith("helix-auth-logout-codex-")


def test_interactive_login_is_never_bounded(mocker):
    """``login`` waits on a human finishing a device flow in a browser; a
    timeout there would kill a working sign-in mid-flight."""
    mocker.patch("os.isatty", return_value=True)
    mock_run = mocker.patch(
        "helix.cli.run_sandbox_auth_command",
        return_value=CompletedProcess([], 0, stdout="", stderr=""),
    )

    assert CliRunner().invoke(cli, ["sandbox", "login", "codex"]).exit_code == 0

    assert "timeout" not in mock_run.call_args.kwargs


def test_sandbox_status_reports_a_timeout_instead_of_raising(mocker):
    mocker.patch(
        "helix.cli.run_sandbox_auth_command",
        side_effect=TimeoutExpired(cmd=["docker"], timeout=300),
    )

    result = CliRunner().invoke(cli, ["sandbox", "status", "codex"])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "did not finish within" in result.output


def test_parse_extra_hosts_ipv4_colon_form():
    from helix.cli import _parse_extra_hosts

    assert _parse_extra_hosts(("api.example.com:10.0.0.1",)) == {
        "api.example.com": "10.0.0.1",
    }


def test_parse_extra_hosts_ipv6_equals_form():
    from helix.cli import _parse_extra_hosts

    # ``=`` form is required for IPv6 addresses (which themselves contain ``:``).
    assert _parse_extra_hosts(("api.example.com=2001:db8::1",)) == {
        "api.example.com": "2001:db8::1",
    }


def test_parse_extra_hosts_ipv6_bracketed():
    from helix.cli import _parse_extra_hosts

    assert _parse_extra_hosts(("api.example.com=[2001:db8::1]",)) == {
        "api.example.com": "2001:db8::1",
    }


def test_parse_extra_hosts_rejects_malformed():
    import click
    import pytest

    from helix.cli import _parse_extra_hosts

    with pytest.raises(click.BadParameter):
        _parse_extra_hosts(("nohost",))
    with pytest.raises(click.BadParameter):
        _parse_extra_hosts((":1.2.3.4",))
