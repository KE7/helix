"""Unit tests for sandbox auth CLI commands."""

from __future__ import annotations

from subprocess import CompletedProcess

from click.testing import CliRunner

from helix.cli import cli


PINNED = "ghcr.io/ke7/helix-evo-runner-claude@sha256:6be6fef"


def test_sandbox_login_invokes_backend_auth_volume(mocker):
    """Login uses the EXPLICIT image it was given, never a default tag.

    Catches: reintroducing the ``image or resolve_sandbox_image(...)`` default.
    Non-vacuity: asserts the image is the pinned string itself, not merely
    that some image was passed — a regression making everything resolve to
    ``:latest`` would otherwise pass.
    """
    mocker.patch("os.isatty", return_value=True)
    mocker.patch("helix.cli.docker_volume_exists", return_value=True)
    mocker.patch("helix.cli._stamp_auth_volume")
    mock_run = mocker.patch(
        "helix.cli.run_sandbox_auth_command",
        return_value=CompletedProcess([], 0, stdout="", stderr=""),
    )
    result = CliRunner().invoke(
        cli, ["sandbox", "login", "cursor", "--image", PINNED]
    )

    assert result.exit_code == 0
    mock_run.assert_called_once_with(
        "cursor",
        action="login",
        image=PINNED,
        network="bridge",
        add_host_gateway=False,
        extra_hosts={},
        interactive=True,
    )
    assert "helix-auth-cursor" in result.output


def test_login_announces_volume_creation(mocker):
    """R9: login MAY create a volume, but never as an unannounced side effect."""
    mocker.patch("os.isatty", return_value=True)
    mocker.patch("helix.cli.docker_volume_exists", return_value=False)
    mocker.patch("helix.cli._stamp_auth_volume")
    mocker.patch(
        "helix.cli.run_sandbox_auth_command",
        return_value=CompletedProcess([], 0, stdout="", stderr=""),
    )
    result = CliRunner().invoke(
        cli, ["sandbox", "login", "cursor", "--image", PINNED]
    )
    assert result.exit_code == 0
    assert "Creating auth volume" in result.output


def test_auth_command_without_project_config_is_a_hard_error(mocker):
    """T21: no discoverable helix.toml and no --image is a hard error.

    Catches: any silent fallback to ``DEFAULT_BACKEND_IMAGES`` (``:latest``),
    which on this host is an OLDER CLI than the pinned runner — so the default
    writes credentials with one CLI for a runner that executes another.
    Non-vacuity: asserts NO container was started. A test asserting only on the
    message would pass against a version that errors *after* running docker.
    """
    mocker.patch("os.isatty", return_value=True)
    mocker.patch("helix.cli.docker_volume_exists", return_value=False)
    mock_run = mocker.patch("helix.cli.run_sandbox_auth_command")

    with CliRunner().isolated_filesystem():  # no helix.toml here
        result = CliRunner().invoke(cli, ["sandbox", "login", "claude"])

    assert result.exit_code == 2
    assert mock_run.call_count == 0, "no container may start before the refusal"
    assert "--image" in result.output
    assert "sandbox.image" in result.output
    # The declined default is named explicitly so the operator can see what
    # HELIX refused to silently use.
    assert "helix-evo-runner-claude:latest" in result.output


def test_status_does_not_create_volume(mocker):
    """T17: status NEVER creates a volume, and never runs an auth container.

    Catches: the old implementation, which ran ``docker run -v`` — silently
    creating the volume it was asked to merely report on.
    Non-vacuity: status is invoked TWICE and absence is asserted after each.
    A test checking only the second call passes trivially once the first has
    created the volume.
    """
    exists = mocker.patch("helix.cli.docker_volume_exists", return_value=False)
    mock_run = mocker.patch("helix.cli.run_sandbox_auth_command")

    for _ in range(2):
        result = CliRunner().invoke(cli, ["sandbox", "status"])
        assert "not provisioned" in result.output
        assert mock_run.call_count == 0, "status must not start a container"

    # Existence was established by inspection only, for each of the 5 backends.
    assert exists.call_count == 10


def test_status_reports_unknown_provenance_without_promoting_it(mocker):
    """A missing HELIX stamp is 'unknown' and is never reported as valid."""
    mocker.patch("helix.cli.docker_volume_exists", return_value=True)
    mocker.patch("helix.cli.read_auth_manifest", return_value=None)
    mock_run = mocker.patch("helix.cli.run_sandbox_auth_command")

    result = CliRunner().invoke(
        cli, ["sandbox", "status", "claude", "--image", PINNED]
    )

    assert "provisioned" in result.output
    assert "provenance: unknown" in result.output
    # 'provisioned' must never be presented as 'valid'.
    assert "not 'valid'" in result.output
    assert mock_run.call_count == 0


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
