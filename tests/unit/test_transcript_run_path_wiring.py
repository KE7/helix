"""F-12: the transcript controls must be CALLED, not merely correct.

Three mutations of the production wiring were each green against the whole
suite, because every existing transcript test calls the library directly and a
library test cannot detect non-wiring:

  W1  delete the ``capture_claude_transcript(...)`` call  -> the rewritten
      module becomes dead code and ``preserve_backend_transcripts`` (default
      TRUE) silently loses every transcript.
  W2  delete ``ensure_transcript_host_dir(...)``          -> Docker auto-creates
      the missing bind source as ``root:root``, which sandbox_home's own
      docstring names as reproducing "exactly the unwritable-HOME failure this
      module exists to prevent -- and, worse, ... silently for the transcript
      path only".
  W3  wrap the call in ``contextlib.suppress(Exception)`` -> the
      ``TranscriptCaptureError`` the F-1 fix made the function raise is
      swallowed one frame up. F-1 was closed at the LIBRARY boundary and open
      at the RUN PATH.

These drive ``run_sandboxed_commands`` -- the public boundary -- so deleting or
suppressing the call is what fails, not a change to the library.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from helix.config import SandboxConfig
from helix.envpolicy import EnvGrant
from helix.sandbox import run_sandboxed_commands
from helix.sandbox_home import transcript_host_dir
from helix.transcripts import TranscriptCaptureError


_SESSION = "sess-run-path"


def _grants() -> list[EnvGrant]:
    return [
        EnvGrant(
            name="ANTHROPIC_API_KEY",
            value="SYNTHETIC-NOT-REAL",
            origin="auth_env_allow",
            scopes=frozenset({"agent"}),
        )
    ]


def _sandbox() -> SandboxConfig:
    return SandboxConfig(
        enabled=True,
        image="helix-test:latest",
        network="none",
        auth="env",
        auth_env_allow=["ANTHROPIC_API_KEY"],
    )


def _run(
    cwd: Path,
    mocker,
    *,
    plant: str | None = None,
    argv_sink: list | None = None,
    sync_back: bool = False,
):
    """Drive the real public entry point with Docker mocked out."""

    def fake_run(args, **kwargs):
        if argv_sink is not None and args[:2] == ["docker", "run"]:
            argv_sink.append(list(args))
        if plant is not None and args[:2] == ["docker", "run"]:
            # Emulate the agent writing its transcript into the candidate bind
            # while the container runs.
            for index, token in enumerate(args):
                if token == "-v" and args[index + 1].endswith(
                    "/home/node/.claude/projects:rw"
                ):
                    host = Path(args[index + 1].split(":")[0]) / "-workspace"
                    host.mkdir(parents=True, exist_ok=True)
                    (host / f"{_SESSION}.jsonl").write_text(plant)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=f'{{"type":"result","session_id":"{_SESSION}"}}\n',
            stderr="",
        )

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)
    return run_sandboxed_commands(
        [["claude", "-p", "prompt"]],
        cwd=cwd,
        env={"ANTHROPIC_API_KEY": "SYNTHETIC-NOT-REAL"},
        sandbox=_sandbox(),
        scope="agent",
        sync_back=sync_back,
        image="helix-test:latest",
        agent_backend="claude",
        grants=_grants(),
    )


def test_W1_run_path_actually_captures_the_transcript(tmp_path: Path, mocker) -> None:
    """Deleting the ``capture_claude_transcript`` call must RED.

    Non-vacuity: asserts the ARTIFACT exists with the planted bytes, so a stub
    that returns an outcome without copying also fails.
    """
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("x\n")
    _run(source, mocker, plant='{"line":"PLANTED"}\n', sync_back=True)

    artifact = (
        source
        / ".helix_artifacts"
        / "backend_transcripts"
        / "claude"
        / f"{_SESSION}.jsonl"
    )
    assert artifact.is_file(), "run path did not capture the transcript"
    assert artifact.read_text() == '{"line":"PLANTED"}\n'


def test_W2_run_path_creates_the_bind_dir_before_any_container(
    tmp_path: Path, mocker
) -> None:
    """Deleting ``ensure_transcript_host_dir`` must RED.

    Docker auto-creates a missing bind source as ``root:root``, handing the
    agent a transcript directory its own uid cannot write -- silently, and only
    for transcripts.

    Asserts ORDERING, which is the property that matters: the directory must
    ALREADY EXIST at the moment the container argv names it. The bind is keyed
    off the per-call temp workspace, so the path is read OUT of the argv rather
    than recomputed -- recomputing it from ``cwd`` would test a different path
    and pass regardless.
    """
    source = tmp_path / "candidate"
    source.mkdir()
    observed: list[tuple[str, bool]] = []

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"]:
            for index, token in enumerate(args):
                if token == "-v" and args[index + 1].endswith(
                    "/home/node/.claude/projects:rw"
                ):
                    host = args[index + 1].split(":")[0]
                    observed.append((host, Path(host).is_dir()))
        return subprocess.CompletedProcess(
            args, 0, stdout=f'{{"session_id":"{_SESSION}"}}\n', stderr=""
        )

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)
    run_sandboxed_commands(
        [["claude", "-p", "p"]],
        cwd=source,
        env={"ANTHROPIC_API_KEY": "SYNTHETIC-NOT-REAL"},
        sandbox=_sandbox(),
        scope="agent",
        sync_back=False,
        image="helix-test:latest",
        agent_backend="claude",
        grants=_grants(),
    )

    assert observed, "no container argv named a transcript bind at all"
    for host, existed in observed:
        assert existed, (
            f"transcript bind source {host!r} did not exist when the container "
            f"argv named it; Docker would create it as root:root"
        )


def test_W3_capture_failure_propagates_out_of_the_run_path(
    tmp_path: Path, mocker
) -> None:
    """Wrapping the call in ``contextlib.suppress`` must RED.

    F-1 made the library raise on an unreadable transcript. That is worth
    nothing if the caller swallows it -- and the property "a failed capture is
    visible to the caller" was asserted nowhere.

    The planted transcript is made unreadable, so capture raises; the test
    requires the error to reach the caller of ``run_sandboxed_commands``.
    """
    source = tmp_path / "candidate"
    source.mkdir()

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"]:
            for index, token in enumerate(args):
                if token == "-v" and args[index + 1].endswith(
                    "/home/node/.claude/projects:rw"
                ):
                    host = Path(args[index + 1].split(":")[0]) / "-workspace"
                    host.mkdir(parents=True, exist_ok=True)
                    planted = host / f"{_SESSION}.jsonl"
                    planted.write_text("x")
                    planted.chmod(0o000)
        return subprocess.CompletedProcess(
            args, 0, stdout=f'{{"session_id":"{_SESSION}"}}\n', stderr=""
        )

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)

    with pytest.raises(TranscriptCaptureError):
        run_sandboxed_commands(
            [["claude", "-p", "p"]],
            cwd=source,
            env={"ANTHROPIC_API_KEY": "SYNTHETIC-NOT-REAL"},
            sandbox=_sandbox(),
            scope="agent",
            sync_back=False,
            image="helix-test:latest",
            agent_backend="claude",
            grants=_grants(),
        )
    # restore so tmp_path cleanup can proceed
    for path in transcript_host_dir(source).rglob("*.jsonl"):
        path.chmod(0o600)
