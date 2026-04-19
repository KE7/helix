"""Unit tests for `helix --version`."""

from __future__ import annotations

from click.testing import CliRunner

from helix import __version__
from helix.cli import cli


def test_version_flag() -> None:
    """helix --version should print the package version."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
