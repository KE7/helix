"""Unit tests for sandbox auth CLI commands."""

from __future__ import annotations

from subprocess import CompletedProcess

from click.testing import CliRunner

from helix.cli import cli


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
