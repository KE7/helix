"""Engine floor and auth-subpath bootstrap.

Both failures this guards present as OPAQUE DAEMON ERRORS -- an old engine and
a never-authenticated volume both surface as internal Docker noise rather than
as the fixable problems they are. Turning each into an actionable message is
the entire point, so the tests assert the MESSAGE CONTENT, not just the raise.
"""

from __future__ import annotations

import pytest

from helix.subpath_bootstrap import (
    DockerCapabilityError,
    assert_volume_subpath_supported,
    auth_subpath_bootstrap_command,
    missing_subpath_error,
    supports_volume_subpath,
)


@pytest.mark.parametrize(
    ("server", "api", "expected"),
    [
        ("29.6.1", "1.55", True),  # the host this was developed against
        ("25.0.0", "1.45", True),  # exactly the floor
        ("24.0.9", "1.43", False),  # below the floor
        ("20.10.24", "1.41", False),
    ],
)
def test_engine_floor_is_enforced_at_the_api_version(
    server: str, api: str, expected: bool
) -> None:
    """API 1.45 is where ``volume-subpath`` landed; it is authoritative."""
    assert supports_volume_subpath(server_version=server, api_version=api) is expected


def test_unparseable_versions_are_treated_as_unsupported() -> None:
    """Catches: assuming "probably new enough" when the probe fails.

    Guessing produces exactly the opaque daemon failure this module exists to
    prevent, and it would do so only on the hosts least able to diagnose it.
    """
    assert not supports_volume_subpath(server_version="", api_version="")
    assert not supports_volume_subpath(server_version="garbage", api_version="n/a")


def test_engine_major_is_a_fallback_when_api_is_unparseable() -> None:
    """Non-vacuity for the check above: it is not simply always-false."""
    assert supports_volume_subpath(server_version="26.1.0", api_version="")
    assert not supports_volume_subpath(server_version="24.0.9", api_version="")


def test_old_engine_error_names_the_floor_and_the_alternative() -> None:
    """Catches: a bare capability failure with no route forward.

    Must also state that HELIX will NOT fall back to a whole-HOME mount --
    otherwise the obvious "fix" for an old host is to reintroduce the defect.
    """
    with pytest.raises(DockerCapabilityError) as exc:
        assert_volume_subpath_supported(server_version="24.0.9", api_version="1.43")
    message = str(exc.value)
    assert "25.0+" in message or "Engine 25" in message
    assert "1.45" in message
    assert 'auth = "env"' in message, "must offer the mode with no daemon floor"
    assert "will not fall back" in message.lower()


def test_supported_engine_does_not_raise() -> None:
    """Non-vacuity control: the assertion is not unconditional."""
    assert_volume_subpath_supported(server_version="29.6.1", api_version="1.55")


# ---------------------------------------------------------------------------
# Bootstrap ordering
# ---------------------------------------------------------------------------


def test_bootstrap_creates_the_subpath_idempotently_and_touches_nothing_else() -> None:
    """``login`` may create the SUBPATH; it must not disturb the store.

    Catches: a bootstrap that writes, moves or clears anything. The credential
    is the one file in this tree that must never be touched by HELIX.
    """
    command = auth_subpath_bootstrap_command(".claude")
    assert "mkdir -p" in command
    assert "/helix-auth-root/.claude" in command
    for destructive in ("rm ", "rm -", "mv ", "cp ", "chown", "chmod", ">"):
        assert destructive not in command, f"{destructive!r} in bootstrap: {command}"


@pytest.mark.parametrize("bad", ["", "/", "../escape", "a/../../etc"])
def test_bootstrap_rejects_paths_that_escape_the_volume(bad: str) -> None:
    """Catches: a traversal reaching outside the mounted volume root."""
    with pytest.raises(ValueError):
        auth_subpath_bootstrap_command(bad)


def test_nested_subpath_is_allowed() -> None:
    """Non-vacuity: the traversal guard must not reject legitimate nesting.

    opencode's auth directory is ``.local/share/opencode``, so a guard that
    rejected any ``/`` would break a real backend.
    """
    command = auth_subpath_bootstrap_command(".local/share/opencode")
    assert "/helix-auth-root/.local/share/opencode" in command


def test_missing_subpath_message_explains_the_daemon_error_it_replaces() -> None:
    """Catches: reporting the raw daemon failure.

    A never-authenticated volume otherwise fails at ``docker run`` with
    ``cannot access path ...: no such file or directory`` -- an auth problem
    wearing the costume of an internal error.
    """
    message = missing_subpath_error(
        backend="claude", volume="helix-auth-claude", subpath=".claude"
    )
    assert "helix sandbox login claude" in message
    assert "cannot access path" in message, "must name the error it pre-empts"


# ---------------------------------------------------------------------------
# The production path must CALL these, not merely be able to
# ---------------------------------------------------------------------------


def test_preflight_calls_the_capability_check_before_touching_the_volume() -> None:
    """Tested-but-uncalled is indistinguishable from working.

    This is the CODEX_SQLITE_HOME defect class: a correct, well-tested helper
    that the production artifact never invokes. The tests above prove the
    function behaves; this proves preflight actually uses it, and does so
    BEFORE any volume operation -- an operator must learn their daemon is too
    old before a proposal is created or budget charged.

    Catches: deleting the call while every test in this module still passes.
    """
    import subprocess

    from helix.authpreflight import preflight_auth, reset_preflight_cache
    from helix.config import (
        AgentConfig,
        EvaluatorConfig,
        HelixConfig,
        SandboxConfig,
    )

    calls: list[list[str]] = []

    def recording_runner(args, **_kw):
        calls.append(list(args))
        # Report a daemon BELOW the floor: the capability check must reject it.
        if args[:2] == ["docker", "version"]:
            return subprocess.CompletedProcess(args, 0, stdout="24.0.9|1.43\n")
        return subprocess.CompletedProcess(args, 0, stdout="")

    config = HelixConfig(
        objective="o",
        evaluator=EvaluatorConfig(command="true", score_parser="helix_result"),
        agent=AgentConfig(backend="claude"),
        sandbox=SandboxConfig(enabled=True, image="i:latest", auth="volume"),
    )

    reset_preflight_cache()
    try:
        with pytest.raises(DockerCapabilityError):
            preflight_auth(config, runner=recording_runner)
    finally:
        reset_preflight_cache()

    assert calls, "preflight made no Docker call at all"
    assert calls[0][:2] == ["docker", "version"], (
        f"capability probe must come FIRST; got {calls[0]}"
    )
    # and nothing touched the auth volume before the refusal
    assert not any("helix-auth" in " ".join(call) for call in calls), calls


def test_preflight_capability_check_passes_on_a_supported_daemon() -> None:
    """Non-vacuity control: the wiring is not an unconditional refusal.

    Without this, deleting the version parsing and always raising would look
    identical to a correct implementation in the test above.
    """
    import subprocess

    from helix.authpreflight import preflight_auth, reset_preflight_cache
    from helix.config import (
        AgentConfig,
        EvaluatorConfig,
        HelixConfig,
        SandboxConfig,
    )

    def runner(args, **_kw):
        if args[:2] == ["docker", "version"]:
            return subprocess.CompletedProcess(args, 0, stdout="29.6.1|1.55\n")
        # volume does not exist -> preflight fails LATER, for a different and
        # correct reason, proving it got past the capability gate
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

    config = HelixConfig(
        objective="o",
        evaluator=EvaluatorConfig(command="true", score_parser="helix_result"),
        agent=AgentConfig(backend="claude"),
        sandbox=SandboxConfig(enabled=True, image="i:latest", auth="volume"),
    )

    reset_preflight_cache()
    try:
        with pytest.raises(Exception) as exc:
            preflight_auth(config, runner=runner)
        assert not isinstance(exc.value, DockerCapabilityError), (
            "a supported daemon must pass the capability gate"
        )
    finally:
        reset_preflight_cache()
