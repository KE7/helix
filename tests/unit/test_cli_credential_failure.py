"""``helix evolve`` / ``helix resume`` never let a credential failure escape
as a raw traceback.

Every in-loop path handles :class:`CredentialRefreshError` itself, so these
handlers are the last line: if a future path forgets, the operator still gets
the error panel with its suggestion and a resume hint, not a stack dump.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from helix.cli import cli
from helix.exceptions import CredentialRefreshError


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / "helix.toml").write_text(
        'objective = "test"\n\n[evaluator]\ncommand = "true"\n'
    )
    return tmp_path


@pytest.mark.parametrize("command", ["evolve", "resume"])
@pytest.mark.parametrize("state_saved", [True, False])
def test_escaped_credential_error_is_a_panel_not_a_traceback(
    mocker, tmp_path: Path, command: str, state_saved: bool
) -> None:
    project = _make_project(tmp_path)
    if state_saved:
        (project / ".helix").mkdir()
        (project / ".helix" / "state.json").write_text("{}")
    # Both commands import run_evolution lazily inside the function body.
    mocker.patch(
        "helix.evolution.run_evolution",
        side_effect=CredentialRefreshError(
            "Codex CLI could not use its stored credential",
            suggestion="Re-authenticate with `helix sandbox login codex`.",
        ),
    )
    if command == "resume":
        mocker.patch("helix.cli.load_state", return_value=None)

    result = CliRunner().invoke(cli, [command, "--dir", str(project)])

    assert result.exit_code == 2, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    out = " ".join(result.output.lower().split())
    assert "credential" in out
    assert "helix sandbox login" in out
    if state_saved:
        assert "state has been saved" in out
        assert "helix resume" in out
    else:
        # The only path that reaches this handler is seedless seed
        # generation, which fails before the first save; ``helix resume``
        # would start a fresh run and repeat the failure.
        assert "no evolution state was saved" in out
        assert "helix resume" not in out
        assert "helix evolve" in out


def test_transient_failure_does_not_demand_a_relogin(mocker, tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    (project / ".helix").mkdir()
    (project / ".helix" / "state.json").write_text("{}")
    exc = CredentialRefreshError("lost a refresh race", suggestion="resume")
    exc.transient = True
    mocker.patch("helix.evolution.run_evolution", side_effect=exc)

    result = CliRunner().invoke(cli, ["evolve", "--dir", str(project)])

    out = " ".join(result.output.lower().split())
    assert result.exit_code == 2
    assert "helix resume" in out
    assert "helix sandbox login" not in out
