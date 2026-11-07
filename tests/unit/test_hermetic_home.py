"""Unit tests never read the operator's real home directory.

Unsandboxed transcript collection reads the agent CLIs' stores under
``Path.home()``.  Run against the real home, a test copied the operator's
multi-gigabyte ``~/.local/share/opencode/opencode.db`` (with the opencode
account's tokens in it) into pytest's temp dir and filled the disk.
``tests/conftest.py::_hermetic_home`` redirects ``$HOME``; these tests fail
if that protection is lost.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helix.backends import BACKENDS
from helix.config import AgentConfig
from helix.mutator import BACKEND_RESULT_ARTIFACT_NAME, invoke_claude_code
from tests.conftest import REAL_HOME, reads_real_home_state


def test_home_is_not_the_real_home() -> None:
    assert Path.home().resolve() != REAL_HOME
    assert not Path.home().resolve().is_relative_to(REAL_HOME / ".local")


def test_copying_real_home_state_fails_the_test(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="real home"):
        shutil.copy2(REAL_HOME / ".local/share/opencode/opencode.db", tmp_path / "x")


_STDOUT = {
    "agy": '{"conversation_id":"conv_1","response":"ok"}',
    "claude": '{"type":"result","session_id":"sess_1"}',
    "codex": '{"type":"thread.started","thread_id":"thr_1"}\n',
    "cursor": '{"type":"system","session_id":"sess_1"}\n',
    "opencode": '{"type":"step_start","sessionID":"ses_1"}\n',
    "pi": '{"type":"session","id":"sess_1"}\n',
}


@pytest.mark.parametrize("backend", BACKENDS)
def test_unsandboxed_transcript_sources_avoid_the_real_home(
    backend: str, tmp_path: Path, mocker
) -> None:
    mocker.patch(
        "helix.mutator.subprocess.run",
        return_value=MagicMock(stdout=_STDOUT[backend], stderr="", returncode=0),
    )

    invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend=backend))

    payload = json.loads((tmp_path / BACKEND_RESULT_ARTIFACT_NAME).read_text())
    assert payload["transcript_artifacts"], backend
    for artifact in payload["transcript_artifacts"]:
        assert not reads_real_home_state(artifact["source"]), artifact
