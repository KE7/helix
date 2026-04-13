"""Unit tests for `helix --version`."""

from __future__ import annotations

from click.testing import CliRunner

from helix.cli import cli


def test_version_flag() -> None:
    """helix --version should print version 0.1.0."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
