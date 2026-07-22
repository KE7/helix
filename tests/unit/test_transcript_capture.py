"""Transcript capture: typed outcomes, detectable failures, no silent drops.

The old capture path re-mounted the persistent auth volume ``:ro`` in a second
container and ran ``cp`` under ``check=False`` with no return value.  That made
``missing`` and ``failed`` indistinguishable -- both were silence -- and it is
why root-owned transcripts are being lost today with preservation defaulting on.

Each test below names the mutation it catches.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from helix.sandbox_home import transcript_host_dir
from helix.transcripts import (
    TranscriptCaptureError,
    capture_claude_transcript,
)


def _plant(workspace: Path, session_id: str, body: str = '{"t":"x"}\n') -> Path:
    """Write a transcript where the container's host bind would leave it."""
    root = transcript_host_dir(workspace) / "-workspace"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{session_id}.jsonl"
    path.write_text(body)
    return path


def _capture(workspace: Path, session_id: str | None, *, enabled: bool = True):
    return capture_claude_transcript(
        workspace=workspace,
        artifact_dir=".helix_artifacts/backend_transcripts",
        session_id=session_id,
        enabled=enabled,
        backend="claude",
    )


# --- (1) readable, node-owned transcript succeeds ---------------------------


def test_readable_transcript_is_captured(tmp_path: Path) -> None:
    """Catches: a capture path that never produces an artifact at all.

    Non-vacuity: asserts the artifact EXISTS and its bytes match the source,
    so a stub returning ``captured`` without copying fails.
    """
    ws = tmp_path / "cand-1"
    ws.mkdir()
    src = _plant(ws, "sess-a", body='{"line":1}\n')

    outcome = _capture(ws, "sess-a")

    assert outcome.status == "captured"
    assert outcome.artifact is not None and outcome.artifact.is_file()
    assert outcome.artifact.read_text() == src.read_text()
    assert outcome.ok


# --- (2) root-owned 0600 fails DETECTABLY, without reading or deleting ------


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_unreadable_transcript_raises_without_reading_or_deleting(
    tmp_path: Path,
) -> None:
    """Catches: the live silent-loss bug.

    Two root-owned ``0600`` transcripts exist in the real auth volume today;
    the old ``cp`` failed, docker exited nonzero, ``check=False`` swallowed it
    and no artifact appeared -- with no error and no log.

    The failure must be detected from METADATA. Those files are incident
    evidence: the policy forbids reading their contents and forbids deleting
    them, so a probe that discovers the problem by attempting a read (and
    possibly succeeding partially) is not acceptable even though it would also
    "detect" the failure.

    Non-vacuity: the file is asserted to still exist and still be unreadable
    afterwards, proving the code neither deleted nor chmod'd its way in.
    """
    ws = tmp_path / "cand-2"
    ws.mkdir()
    src = _plant(ws, "sess-b")
    src.chmod(0o000)
    try:
        with pytest.raises(TranscriptCaptureError) as exc:
            _capture(ws, "sess-b")

        message = str(exc.value)
        assert "not readable" in message
        assert "sess-b" in message
        assert "preserve_backend_transcripts" in message, "must name the opt-out"
        # evidence preserved, untouched
        assert src.exists()
        assert not os.access(src, os.R_OK)
    finally:
        src.chmod(0o600)


# --- (3) missing is DISTINGUISHED from failure ------------------------------


def test_missing_transcript_is_not_the_same_as_a_failure(tmp_path: Path) -> None:
    """Catches: collapsing 'nothing to copy' into 'copy failed', or vice versa.

    They have different remedies, and the old code made both silent nothing.

    BOTH directions are asserted here. An earlier version of this test claimed
    "or vice versa" in its docstring while the body only ever exercised the
    MISSING direction -- so a mutation that turned a copy FAILURE into a silent
    ``missing`` passed with a green suite, in exactly the direction the
    docstring promised was covered. A docstring is a declaration; it has to be
    checked against what the body actually asserts.
    """
    ws = tmp_path / "cand-3"
    ws.mkdir()

    # direction 1: nothing to copy -> missing, not an error
    outcome = _capture(ws, "sess-absent")
    assert outcome.status == "missing"
    assert outcome.artifact is None
    assert "sess-absent" in outcome.detail

    # and a missing session id is missing, not an error
    assert _capture(ws, None).status == "missing"

    # direction 2: the transcript EXISTS but the copy fails -> an ERROR, never
    # a silent ``missing``. This is the direction the mutation exploits.
    ws2 = tmp_path / "cand-3b"
    ws2.mkdir()
    _plant(ws2, "sess-present")
    artifact_parent = ws2 / ".helix_artifacts" / "backend_transcripts"
    artifact_parent.parent.mkdir(parents=True, exist_ok=True)
    # a REGULAR FILE where the artifact directory must go: mkdir then fails
    artifact_parent.write_text("not a directory")

    with pytest.raises(TranscriptCaptureError):
        _capture(ws2, "sess-present")


def test_copy_failure_raises_and_names_both_paths_without_leaking_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing copy MUST raise -- it must never degrade to ``missing``.

    Catches (auditor mutation F-1): replacing the ``except OSError`` around
    ``shutil.copy2`` with ``return TranscriptOutcome(status="missing", ...)``.
    That single edit reinstates the exact defect this module was written to
    remove -- ``missing`` and ``failed`` collapsing into one silent nothing --
    and the full suite previously stayed GREEN, because the only
    ``pytest.raises`` in this file covered the unreadable branch.

    Also asserts the diagnostic is useful AND redacted: it must name the source
    and the destination (an operator cannot act on "copy failed"), and it must
    contain none of the transcript's bytes.
    """
    ws = tmp_path / "cand-copyfail"
    ws.mkdir()
    secret = '{"transcript":"SENSITIVE-TRANSCRIPT-BODY"}\n'
    source = _plant(ws, "sess-copyfail", body=secret)

    real_copy2 = __import__("shutil").copy2

    def boom(*args: object, **kwargs: object):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("helix.transcripts.shutil.copy2", boom)

    with pytest.raises(TranscriptCaptureError) as exc:
        _capture(ws, "sess-copyfail")

    message = str(exc.value)
    assert str(source) in message, "must name the source"
    assert "sess-copyfail.jsonl" in message, "must name the destination"
    assert "No space left on device" in message, "must surface the real cause"
    assert "SENSITIVE-TRANSCRIPT-BODY" not in message, "must not leak content"

    # non-vacuity: with the real copy2 restored, the same input SUCCEEDS, so
    # the raise above is attributable to the failure and not to the fixture.
    monkeypatch.setattr("helix.transcripts.shutil.copy2", real_copy2)
    assert _capture(ws, "sess-copyfail").status == "captured"


# --- (4) candidate-keyed bind: survives outcomes, cannot collide ------------


@pytest.mark.parametrize("exit_shape", ["success", "nonzero", "timeout"])
def test_transcript_survives_regardless_of_command_outcome(
    tmp_path: Path, exit_shape: str
) -> None:
    """LIBRARY scope only: capture succeeds for a present transcript.

    SCOPE CORRECTION (F-16a). This claimed to catch "capture wired to the
    success path only" -- which it cannot: it calls ``_capture`` directly,
    never invokes ``run_sandboxed_commands``, and the ``exit_shape``
    parametrisation varies ONLY a naming string, so the three cases are
    byte-identical in behaviour. It is the test a reviewer would have cited to
    claim F-12's mutations were covered.

    The real property -- that capture runs regardless of how the command exits
    -- is carried by ``tests/unit/test_transcript_run_path_wiring.py``, which
    drives the public boundary.
    """
    ws = tmp_path / f"cand-{exit_shape}"
    ws.mkdir()
    _plant(ws, f"sess-{exit_shape}")
    outcome = _capture(ws, f"sess-{exit_shape}")
    assert outcome.status == "captured", exit_shape


def test_concurrent_candidates_cannot_collide_on_an_identical_session_id(
    tmp_path: Path,
) -> None:
    """FORCES the collision rather than asserting distinct paths.

    The candidate id never entered the old path: the source was built from a
    session id parsed out of backend stdout, so two candidates reporting the
    SAME id read and wrote one file (reproduced as a single interleaved
    transcript). Keying off the per-candidate workspace makes that impossible.

    Catches: any reversion to a shared or backend-derived transcript root.
    """
    shared_id = "IDENTICAL-SESSION-ID"
    a = tmp_path / "cand-a"
    b = tmp_path / "cand-b"
    a.mkdir()
    b.mkdir()
    _plant(a, shared_id, body="CANDIDATE-A\n")
    _plant(b, shared_id, body="CANDIDATE-B\n")

    out_a = _capture(a, shared_id)
    out_b = _capture(b, shared_id)

    assert out_a.artifact != out_b.artifact
    assert out_a.artifact is not None and out_b.artifact is not None
    assert out_a.artifact.read_text() == "CANDIDATE-A\n"
    assert out_b.artifact.read_text() == "CANDIDATE-B\n"


# --- (5) disabled is the ONLY silent case ----------------------------------


def test_disabled_is_the_only_intentionally_silent_outcome(tmp_path: Path) -> None:
    """Catches: silence creeping back in for any other condition."""
    ws = tmp_path / "cand-5"
    ws.mkdir()
    _plant(ws, "sess-e")

    disabled = _capture(ws, "sess-e", enabled=False)
    assert disabled.status == "disabled"
    assert disabled.ok

    # a non-claude backend is likewise a declared no-op, not a failure
    other = capture_claude_transcript(
        workspace=ws,
        artifact_dir="a",
        session_id="sess-e",
        enabled=True,
        backend="codex",
    )
    assert other.status == "disabled"

    # everything else speaks: enabled + present -> captured
    assert _capture(ws, "sess-e").status == "captured"


# --- structural: no fallback may re-mount the persistent auth volume -------


def test_capture_module_cannot_reach_the_auth_volume() -> None:
    """Make the bad path IMPOSSIBLE, not merely absent.

    The old design's source path lived inside the shared auth volume, so
    capture worked only BECAUSE home was shared. Reading from the candidate
    host bind means capture runs no container at all.

    Catches: a future 'fallback' that re-mounts ``helix-auth-*`` to fetch a
    transcript the bind does not have.
    """
    import helix.transcripts as mod

    # Asserted on the module's EXECUTABLE capability, not on word-matching its
    # prose: the docstring necessarily describes the Docker-based design it
    # replaced, so a substring check over the whole file would fail on its own
    # explanation. What matters is that the module cannot start a process or
    # name the auth volume.
    assert not hasattr(mod, "subprocess"), "capture must not import subprocess"
    assert not hasattr(mod, "_run_docker")

    import ast

    tree = ast.parse(Path(mod.__file__).read_text())
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    docstrings = {
        ast.get_docstring(node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef)
    }
    code_literals = literals - docstrings
    assert not any("helix-auth" in s for s in code_literals), (
        "no code literal may name the persistent auth volume"
    )


def test_old_auth_volume_copy_helper_is_gone() -> None:
    """The removed path must stay removed.

    Catches: reintroduction of ``_copy_claude_transcript_from_auth_volume``,
    which mounted ``helix-auth-<backend>:/home/node:ro`` in a second container.
    """
    import helix.sandbox as sandbox_mod

    assert not hasattr(sandbox_mod, "_copy_claude_transcript_from_auth_volume")
    source = Path(sandbox_mod.__file__).read_text()
    assert ":/home/node:ro" not in source
