"""Env mode must never execute a volume path (call-site shaped assertions).

The argv tests in ``test_sandbox_env_mode_isolation`` prove env mode does not
MOUNT the persistent auth volume.  That is necessary but not sufficient: the
volume lifecycle also reaches Docker through preflight, provenance-manifest
reads and the refresh probe, none of which appear in the agent argv at all.

In env mode there is no volume, so ANY of those executing is a bug -- it would
mean a mode that advertises "no shared store" still touching the shared store.

Technique: pass a runner that RAISES on any invocation.  A mode that performs
zero Docker calls cannot be distinguished from one that performs the "right"
calls by inspecting a mock's recorded args, so the runner is made explosive
rather than recording.  Non-vacuity is established by showing the same runner
IS reached in volume mode.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from helix.authpreflight import preflight_auth, reset_preflight_cache
from helix.config import (
    AgentConfig,
    EvaluatorConfig,
    HelixConfig,
    SandboxConfig,
)


class DockerWasInvoked(AssertionError):
    """Raised by the exploding runner when any Docker call is attempted."""


def _exploding_runner(
    args: list[str], **_kwargs: Any
) -> subprocess.CompletedProcess[str]:
    raise DockerWasInvoked(" ".join(args))


def _config(auth: str) -> HelixConfig:
    return HelixConfig(
        objective="o",
        evaluator=EvaluatorConfig(command="true", score_parser="helix_result"),
        agent=AgentConfig(backend="claude"),
        sandbox=SandboxConfig(
            enabled=True,
            image="helix-test:latest",
            network="none",
            auth=auth,  # type: ignore[arg-type]
            auth_env_allow=["ANTHROPIC_API_KEY"] if auth == "env" else [],
        ),
    )


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    reset_preflight_cache()


def test_env_mode_preflight_makes_zero_docker_calls() -> None:
    """Preflight must short-circuit before ANY Docker invocation in env mode.

    Catches: a future preflight that checks volume existence, reads the
    provenance manifest, or runs the auth probe before consulting the mode --
    each of which would touch the shared credential volume for a run that has
    no business touching it, and the manifest/probe paths can ROTATE A TOKEN.
    """
    result = preflight_auth(_config("env"), runner=_exploding_runner)
    assert result.outcome == "skipped_env_mode"


def test_volume_mode_does_reach_docker() -> None:
    """Non-vacuity control for the test above.

    If preflight stopped calling Docker altogether -- or the runner parameter
    stopped being used -- the env-mode assertion would pass for the wrong
    reason.  This proves the exploding runner is genuinely wired in.
    """
    with pytest.raises(DockerWasInvoked):
        preflight_auth(_config("volume"), runner=_exploding_runner)


def test_env_mode_preflight_is_not_merely_cached() -> None:
    """The env-mode skip must be a mode decision, not a cache hit.

    Catches: an implementation that returns a memoised verdict from an earlier
    volume-mode run, which would make the zero-Docker result an artifact of
    test ordering rather than a property of env mode.
    """
    reset_preflight_cache()
    first = preflight_auth(_config("env"), runner=_exploding_runner)
    reset_preflight_cache()
    second = preflight_auth(_config("env"), runner=_exploding_runner)
    assert first.outcome == second.outcome == "skipped_env_mode"


def test_disabled_sandbox_also_makes_zero_docker_calls() -> None:
    """Guard: the non-sandboxed path must not reach the volume either."""
    config = HelixConfig(
        objective="o",
        evaluator=EvaluatorConfig(command="true", score_parser="helix_result"),
        agent=AgentConfig(backend="claude"),
        sandbox=SandboxConfig(enabled=False),
    )
    assert preflight_auth(config, runner=_exploding_runner).outcome == (
        "skipped_no_sandbox"
    )
