"""Non-vacuity proof for the unit-tier Docker safety guard.

A guard nobody has seen fire is a guard nobody knows works.  These tests make
the guard in ``tests/unit/conftest.py`` actually trigger, so its protection is
demonstrated rather than assumed.

The guard exists because of a real incident: wiring ``preflight_auth`` into
``run_evolution`` made a real authenticated probe against the shared
``helix-auth-claude`` volume reachable from ``pytest tests/unit/``.  The
credential record survived unchanged, but non-credential volume state was
mutated, and a *successful* refresh would have rotated the stored token and
invalidated the volume for every lane.
"""

from __future__ import annotations

import subprocess

import pytest


def test_guard_blocks_real_docker_invocation():
    """The general block fires on any real docker command.

    Non-vacuity: this test FAILS (does not raise) if the guard is removed,
    because ``docker version`` would then simply run.
    """
    with pytest.raises(Exception) as exc:
        subprocess.run(["docker", "version"], capture_output=True)
    assert "attempted to run Docker for real" in str(exc.value)


def test_guard_refuses_shared_auth_volume_by_name():
    """The belt-and-braces block fires on a helix-auth-* volume name.

    This is the same refusal rule the synthetic refresh harness adopts, and it
    holds even for a command that does not look like ``docker`` to the general
    check — so a helper or wrapper cannot route around it.
    """
    with pytest.raises(Exception) as exc:
        subprocess.run(
            ["some-wrapper", "run", "-v", "helix-auth-claude:/home/node:rw", "img"],
            capture_output=True,
        )
    message = str(exc.value)
    assert "SHARED auth volume" in message
    assert "helix-auth-claude" in message
    assert "DISPOSABLE" in message


def test_guard_allows_non_docker_subprocesses():
    """The guard must not break ordinary subprocess use.

    Without this, the guard could be 'proved' by a version that blocks
    everything, which would be useless and would mask real failures.
    """
    result = subprocess.run(["echo", "ok"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_preflight_is_unreachable_from_a_default_unit_run():
    """The auth preflight must be UNREACHABLE BY DEFAULT off the production path.

    ``preflight_auth`` performs a real authenticated operation against a real
    shared volume — dangerous by construction, and demonstrated empirically
    rather than hypothetically.  Any attempt to reach Stage 0 from a unit test
    must fail before a container starts.
    """
    from helix.config import AgentConfig, EvaluatorConfig, HelixConfig, SandboxConfig
    from helix.authpreflight import preflight_auth, reset_preflight_cache

    reset_preflight_cache()
    config = HelixConfig(
        objective="x",
        evaluator=EvaluatorConfig(command="true", score_parser="helix_result"),
        agent=AgentConfig(backend="claude"),
        sandbox=SandboxConfig(enabled=True, image="pinned@sha256:deadbeef"),
    )
    with pytest.raises(Exception) as exc:
        preflight_auth(config)
    # It must be stopped by the guard, NOT by reaching a real container.
    assert "helix-auth-claude" in str(exc.value)
    reset_preflight_cache()
