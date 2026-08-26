"""The per-generation credential warm: who gets one, and what it costs.

Candidates share one login volume read-write.  When a refresh comes due they
can each decide to refresh independently and spend the same single-use refresh
token; whoever loses that race can fail without saying anything.  The warm
closes the window by doing the refresh once, in one container, before a
generation dispatches anything.

Two properties are load-bearing and both are asserted here: the warm runs
through the existing single-writer auth path, and a backend is warmed only when
warming it is both useful and free.  A backend with no warm command must carry
a written reason -- skipping is a claim about correctness, not an omission.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from helix.backends import (
    BACKEND_AUTH_COMMANDS,
    BACKENDS,
    CREDENTIAL_WARM_SKIP_REASONS,
    backend_credential_warm_skip_reason,
)
from helix.config import AgentConfig, EvaluatorConfig, HelixConfig, SandboxConfig
from helix.evolution import _warm_generation_credential
from helix.sandbox import (
    CredentialWarmResult,
    sandbox_auth_docker_args,
    warm_backend_credential,
)


WARMED_BACKENDS = ("codex",)
SKIPPED_BACKENDS = ("claude", "cursor", "gemini", "opencode")


def _completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["docker"], returncode=returncode, stdout="", stderr=stderr
    )


# ---------------------------------------------------------------------------
# Registry: every backend is either warmed or explained
# ---------------------------------------------------------------------------


class TestWarmRegistry:
    def test_every_backend_is_warmed_or_has_a_reason(self) -> None:
        """No backend may fall through silently.

        A missing warm command with no recorded reason is indistinguishable
        from an oversight, which is exactly the state this change exists to
        leave behind.
        """
        for backend in BACKENDS:
            has_warm = "warm" in BACKEND_AUTH_COMMANDS[backend]
            has_reason = backend in CREDENTIAL_WARM_SKIP_REASONS
            assert has_warm != has_reason, backend

    @pytest.mark.parametrize("backend", WARMED_BACKENDS)
    def test_warmed_backend_has_no_skip_reason(self, backend: str) -> None:
        assert backend_credential_warm_skip_reason(backend) is None

    @pytest.mark.parametrize("backend", SKIPPED_BACKENDS)
    def test_skipped_backend_states_why(self, backend: str) -> None:
        reason = backend_credential_warm_skip_reason(backend)
        assert reason is not None
        # A reason a maintainer cannot act on is not a reason.
        assert len(reason) > 60

    def test_unknown_backend_is_not_warmed(self) -> None:
        assert backend_credential_warm_skip_reason("nope") is not None

    def test_codex_warm_is_not_login_status(self) -> None:
        """``codex login status`` never takes the refresh path.

        Measured against codex-cli 0.130.0 with a synthetic credential in a
        throwaway volume: it exits 0 and issues no request whether the stored
        ``last_refresh`` is minutes or 30 days old.  Warming with it would look
        like protection while providing none, so the registry must not drift
        back to it.
        """
        warm = BACKEND_AUTH_COMMANDS["codex"]["warm"]
        assert warm != BACKEND_AUTH_COMMANDS["codex"]["status"]
        assert "login" not in " ".join(warm)


# ---------------------------------------------------------------------------
# The warm reuses the single-writer sandboxed auth path
# ---------------------------------------------------------------------------


class TestWarmUsesSingleWriterPath:
    def test_warm_action_mounts_the_shared_login_volume_read_write(self) -> None:
        args = sandbox_auth_docker_args(
            "codex", image="img:latest", action="warm"
        )
        assert "-v" in args
        assert "helix-auth-codex:/home/node:rw" in args
        # One container, one command, nothing else attached to the volume.
        assert args[:3] == ["docker", "run", "--rm"]

    def test_warm_action_runs_the_registered_warm_command(self) -> None:
        args = sandbox_auth_docker_args(
            "codex", image="img:latest", action="warm"
        )
        assert args[-len(BACKEND_AUTH_COMMANDS["codex"]["warm"]) :] == (
            BACKEND_AUTH_COMMANDS["codex"]["warm"]
        )

    def test_warm_action_is_rejected_for_a_backend_without_one(self) -> None:
        with pytest.raises(ValueError):
            sandbox_auth_docker_args("cursor", image="img:latest", action="warm")

    def test_warm_forwards_sandbox_network_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def _fake(backend: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            seen.update(kwargs)
            seen["backend"] = backend
            return _completed(0)

        monkeypatch.setattr("helix.sandbox.run_sandbox_auth_command", _fake)
        sandbox = SandboxConfig(
            enabled=True,
            network="none",
            add_host_gateway=True,
            extra_hosts={"h": "1.2.3.4"},
            image="custom:tag",
        )
        result = warm_backend_credential("codex", sandbox=sandbox)

        assert result.warmed is True
        assert seen["backend"] == "codex"
        assert seen["action"] == "warm"
        assert seen["network"] == "none"
        assert seen["add_host_gateway"] is True
        assert seen["extra_hosts"] == {"h": "1.2.3.4"}
        # A configured image must win, or the warm would refresh the credential
        # with a different CLI build than the candidates use.
        assert seen["image"] == "custom:tag"


# ---------------------------------------------------------------------------
# Failure of the warm is reported, never fatal
# ---------------------------------------------------------------------------


class TestWarmFailureIsNotFatal:
    def test_skipped_backend_starts_no_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: Any, **_k: Any) -> None:
            raise AssertionError("a skipped backend must not run a container")

        monkeypatch.setattr("helix.sandbox.run_sandbox_auth_command", _boom)
        result = warm_backend_credential(
            "cursor", sandbox=SandboxConfig(enabled=True)
        )
        assert result.skipped is True
        assert result.warmed is False
        assert result.failed is False

    def test_non_zero_exit_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "helix.sandbox.run_sandbox_auth_command",
            lambda *a, **k: _completed(3, "warm blew up"),
        )
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True)
        )
        assert result.failed is True
        assert result.returncode == 3
        assert "warm blew up" in result.detail

    def test_docker_exception_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(*_a: Any, **_k: Any) -> None:
            raise OSError("no docker here")

        monkeypatch.setattr("helix.sandbox.run_sandbox_auth_command", _raise)
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True)
        )
        assert result.failed is True
        assert "no docker here" in result.detail

    def test_detail_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "helix.sandbox.run_sandbox_auth_command",
            lambda *a, **k: _completed(1, "x" * 5000),
        )
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True)
        )
        assert 0 < len(result.detail) <= 400


# ---------------------------------------------------------------------------
# Once per generation, and only where there is a volume to warm
# ---------------------------------------------------------------------------


def _config(backend: str, *, sandboxed: bool) -> HelixConfig:
    return HelixConfig(
        objective="Improve the code",
        evaluator=EvaluatorConfig(command="pytest -q"),
        agent=AgentConfig(backend=backend),  # type: ignore[arg-type]
        sandbox=SandboxConfig(enabled=sandboxed),
    )


class TestGenerationWarm:
    def test_unsandboxed_run_warms_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """There is no HELIX-managed login volume to arbitrate without the sandbox."""

        def _boom(*_a: Any, **_k: Any) -> None:
            raise AssertionError("no warm without a sandbox")

        monkeypatch.setattr("helix.evolution.warm_backend_credential", _boom)
        assert (
            _warm_generation_credential(
                _config("codex", sandboxed=False), gen=1, announce_skip=True
            )
            is None
        )

    def test_sandboxed_run_warms_the_configured_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def _fake(backend: str, **_k: Any) -> CredentialWarmResult:
            calls.append(backend)
            return CredentialWarmResult(backend=backend, warmed=True, returncode=0)

        monkeypatch.setattr("helix.evolution.warm_backend_credential", _fake)
        result = _warm_generation_credential(
            _config("codex", sandboxed=True), gen=4, announce_skip=False
        )
        assert calls == ["codex"]
        assert result is not None and result.warmed

    def test_failed_warm_returns_and_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            "helix.evolution.warm_backend_credential",
            lambda backend, **_k: CredentialWarmResult(
                backend=backend, warmed=False, returncode=1, detail="boom"
            ),
        )
        result = _warm_generation_credential(
            _config("codex", sandboxed=True), gen=2, announce_skip=False
        )
        assert result is not None and result.failed
        # The operator is told what protection was lost, not just that a
        # command exited non-zero.
        printed = capsys.readouterr().out
        assert "refresh" in printed.lower()
        assert "run continues" in printed.lower()
