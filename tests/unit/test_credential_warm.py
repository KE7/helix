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
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from helix.backends import (
    BACKEND_AUTH_COMMANDS,
    BACKENDS,
    CREDENTIAL_WARM_SKIP_REASONS,
    backend_credential_warm_skip_reason,
)
from helix.config import (
    AgentConfig,
    EvaluatorConfig,
    EvolutionConfig,
    HelixConfig,
    SandboxConfig,
)
from helix.evolution import _warm_generation_credential
from helix.sandbox import (
    CODEX_TOKEN_REFRESH_INTERVAL,
    CREDENTIAL_WARM_TIMEOUT_SECONDS,
    CredentialWarmResult,
    credential_warm_timeout,
    sandbox_auth_docker_args,
    verify_codex_credential_fresh,
    warm_backend_credential,
)


WARMED_BACKENDS = ("codex",)
SKIPPED_BACKENDS = ("agy", "claude", "cursor", "opencode")


def _completed(
    returncode: int, stderr: str = "", stdout: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["docker"], returncode=returncode, stdout=stdout, stderr=stderr
    )


NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ") + "\n"


class _FakeAuthCommands:
    """Stand-in for ``run_sandbox_auth_command`` that answers the
    ``last_refresh`` probes from a script and records the warm call.

    ``timeline`` is the sequence of values the probe returns (before, after);
    ``warm`` is the warm command's result.
    """

    def __init__(
        self,
        timeline: list[str | None],
        warm: subprocess.CompletedProcess[str] | BaseException | None = None,
    ) -> None:
        self.timeline = list(timeline)
        self.warm = warm if warm is not None else _completed(0)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, backend: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        record = dict(kwargs)
        record["backend"] = backend
        self.calls.append(record)
        if kwargs.get("command") is not None:
            value = self.timeline.pop(0)
            return _completed(0, stdout="" if value is None else value)
        if isinstance(self.warm, BaseException):
            raise self.warm
        return self.warm

    @property
    def warm_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c.get("command") is None]

    @property
    def probe_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c.get("command") is not None]


def _fresh_timeline() -> list[str | None]:
    """A credential refreshed an hour ago: nothing to do, verifiably fresh."""
    stamp = _stamp(NOW - timedelta(hours=1))
    return [stamp, stamp]


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeAuthCommands) -> None:
    monkeypatch.setattr("helix.sandbox.run_sandbox_auth_command", fake)
    monkeypatch.setattr("helix.sandbox.datetime", _FrozenDatetime)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> "datetime":  # type: ignore[override]
        return NOW if tz is not None else NOW.replace(tzinfo=None)


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
        fake = _FakeAuthCommands(_fresh_timeline())
        _install(monkeypatch, fake)
        sandbox = SandboxConfig(
            enabled=True,
            network="none",
            add_host_gateway=True,
            extra_hosts={"h": "1.2.3.4"},
            image="custom:tag",
        )
        result = warm_backend_credential("codex", sandbox=sandbox)

        assert result.warmed is True
        [seen] = fake.warm_calls
        assert seen["backend"] == "codex"
        assert seen["action"] == "warm"
        assert seen["network"] == "none"
        assert seen["add_host_gateway"] is True
        assert seen["extra_hosts"] == {"h": "1.2.3.4"}
        # A configured image must win, or the warm would refresh the credential
        # with a different CLI build than the candidates use.
        assert seen["image"] == "custom:tag"

    def test_operator_env_reaches_the_warm_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Proxy / CA settings the candidates get must reach the warm too, or
        it cannot reach the token endpoint the candidates can."""
        fake = _FakeAuthCommands(_fresh_timeline())
        _install(monkeypatch, fake)
        env = {"HTTPS_PROXY": "http://proxy:3128", "SSL_CERT_FILE": "/ca.pem"}
        warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True), env=env
        )
        [seen] = fake.warm_calls
        assert seen["env"] == env
        args = sandbox_auth_docker_args(
            "codex", image="img:latest", action="warm", env=env
        )
        joined = " ".join(args)
        assert "-e HTTPS_PROXY=http://proxy:3128" in joined
        assert "-e SSL_CERT_FILE=/ca.pem" in joined
        # HOME and PATH stay pinned to the container's own values.
        assert "-e HOME=/home/node" in joined
        assert sandbox_auth_docker_args(
            "codex", image="img", action="warm", env={"HOME": "/x", "PATH": "/y"}
        ).count("-e") == 2

    def test_no_auth_command_uses_a_login_shell(self) -> None:
        """``sh -l`` sources ``$HOME/.profile`` from the shared login volume
        that every candidate can write; PATH is pinned with ``-e`` instead."""
        for backend, actions in BACKEND_AUTH_COMMANDS.items():
            for action, argv in actions.items():
                if argv[0] != "sh":
                    continue
                assert argv[1] == "-c", (backend, action, argv)
                assert not any(
                    flag.startswith("-") and "l" in flag for flag in argv[1:-1]
                ), (backend, action, argv)

    def test_last_refresh_probe_runs_without_network_and_reads_one_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeAuthCommands(_fresh_timeline())
        _install(monkeypatch, fake)
        warm_backend_credential("codex", sandbox=SandboxConfig(enabled=True))
        before, after = fake.probe_calls
        for probe in (before, after):
            assert probe["network"] == "none"
            script = probe["command"][-1]
            assert "last_refresh" in script
            assert "access_token" not in script and "cat " not in script


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
        fake = _FakeAuthCommands(_fresh_timeline(), warm=_completed(3, "warm blew up"))
        _install(monkeypatch, fake)
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True)
        )
        assert result.failed is True
        assert result.returncode == 3
        assert "warm blew up" in result.detail

    def test_docker_exception_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeAuthCommands([None, None], warm=OSError("no docker here"))
        _install(monkeypatch, fake)
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True)
        )
        assert result.failed is True
        assert "no docker here" in result.detail

    def test_detail_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeAuthCommands(_fresh_timeline(), warm=_completed(1, "x" * 5000))
        _install(monkeypatch, fake)
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True)
        )
        assert 0 < len(result.detail) <= 400


# ---------------------------------------------------------------------------
# ``warmed`` means verified fresh, not "exited 0"
# ---------------------------------------------------------------------------


class TestWarmVerifiesFreshness:
    """``codex debug models`` swallows a rejected refresh and exits 0.

    The exit code therefore proves nothing.  ``last_refresh`` is read back
    from ``auth.json`` (one field, no network) and the warm is reported as
    such only when the credential is verifiably inside codex's refresh
    interval afterwards -- so no candidate will attempt a refresh.
    """

    def test_advanced_last_refresh_is_warmed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale = _stamp(NOW - CODEX_TOKEN_REFRESH_INTERVAL - timedelta(days=1))
        fake = _FakeAuthCommands([stale, _stamp(NOW - timedelta(seconds=2))])
        _install(monkeypatch, fake)
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True)
        )
        assert result.warmed is True
        assert result.stale is False
        assert "refresh performed" in result.detail

    def test_stale_and_unchanged_is_not_warmed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A rejected refresh: exit 0, ``last_refresh`` still past the interval."""
        stale = _stamp(NOW - CODEX_TOKEN_REFRESH_INTERVAL - timedelta(days=1))
        fake = _FakeAuthCommands([stale, stale])
        _install(monkeypatch, fake)
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True)
        )
        assert result.warmed is False
        assert result.stale is True
        assert result.failed is True
        assert result.returncode == 0
        assert "rejected" in result.detail
        assert "models_cache.json" in result.detail

        # The loop turns it into a WARNING that names the cause and the race.
        monkeypatch.setattr(
            "helix.evolution.warm_backend_credential", lambda backend, **_k: result
        )
        _warm_generation_credential(
            _config("codex", sandboxed=True), gen=2, announce_skip=False
        )
        out = " ".join(capsys.readouterr().out.lower().split())
        assert "not verifiably fresh" in out
        assert "rejected" in out
        assert "same single-use refresh token" in out

    def test_unchanged_but_inside_the_interval_is_fresh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No refresh was due (or a fresh catalog cache short-circuited one):
        either way no candidate will refresh, so there is nothing to race."""
        fake = _FakeAuthCommands(_fresh_timeline())
        _install(monkeypatch, fake)
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True)
        )
        assert result.warmed is True
        assert "inside codex's 8-day refresh interval" in result.detail

    def test_unreadable_last_refresh_is_unverified_not_warmed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeAuthCommands([None, None])
        _install(monkeypatch, fake)
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True)
        )
        assert result.warmed is False
        assert result.stale is True
        assert "unverified" in result.detail

    def test_verdict_function(self) -> None:
        old = NOW - CODEX_TOKEN_REFRESH_INTERVAL - timedelta(hours=1)
        recent = NOW - timedelta(minutes=5)
        assert verify_codex_credential_fresh(old, recent, now=NOW)[0] is True
        assert verify_codex_credential_fresh(old, old, now=NOW)[0] is False
        assert verify_codex_credential_fresh(recent, recent, now=NOW)[0] is True
        assert verify_codex_credential_fresh(None, recent, now=NOW)[0] is True
        assert verify_codex_credential_fresh(None, None, now=NOW)[0] is False
        assert verify_codex_credential_fresh(old, None, now=NOW)[0] is False


# ---------------------------------------------------------------------------
# A stalled warm cannot wedge the loop
# ---------------------------------------------------------------------------


class TestWarmIsBounded:
    """The warm runs on the main thread before any candidate is dispatched.

    Without a timeout, a token-refresh HTTP call that blackholes blocks the
    whole evolution loop indefinitely with nothing but a debug log line, and
    the unnamed container keeps running against the shared login volume
    after the operator gives up and hits Ctrl-C.
    """

    def test_timeout_always_applies(self) -> None:
        assert credential_warm_timeout(SandboxConfig(enabled=True)) == (
            CREDENTIAL_WARM_TIMEOUT_SECONDS
        )

    def test_sandbox_timeout_can_only_tighten_the_cap(self) -> None:
        assert credential_warm_timeout(
            SandboxConfig(enabled=True, timeout_seconds=30)
        ) == 30.0
        assert credential_warm_timeout(
            SandboxConfig(enabled=True, timeout_seconds=10_000)
        ) == CREDENTIAL_WARM_TIMEOUT_SECONDS

    def test_warm_runs_with_a_timeout_and_a_named_container(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def _fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if "last_refresh" in args[-1]:
                return _completed(0, stdout=_stamp(NOW - timedelta(hours=1)))
            seen["args"] = args
            seen.update(kwargs)
            return _completed(0)

        monkeypatch.setattr("helix.sandbox.subprocess.run", _fake_run)
        monkeypatch.setattr("helix.sandbox.datetime", _FrozenDatetime)
        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True, timeout_seconds=45)
        )

        assert result.warmed is True
        assert seen["timeout"] == 45.0
        name = seen["args"][seen["args"].index("--name") + 1]
        assert name.startswith("helix-warm-codex-")

    def test_timeout_is_reported_not_raised_and_the_container_is_stopped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        removed: list[list[str]] = []

        def _hang(args: list[str], **kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])

        def _fake_docker(args: list[str], **_k: Any) -> subprocess.CompletedProcess[str]:
            removed.append(args)
            return _completed(0)

        monkeypatch.setattr("helix.sandbox.subprocess.run", _hang)
        monkeypatch.setattr("helix.sandbox._run_docker", _fake_docker)

        result = warm_backend_credential(
            "codex", sandbox=SandboxConfig(enabled=True, timeout_seconds=7)
        )

        assert result.failed is True
        assert result.timed_out is True
        assert result.warmed is False
        assert result.skipped is False
        assert "7s" in result.detail
        # Killing the docker client does not stop the container; the named
        # container must be force-removed so it stops touching the volume.
        # (The probe before the warm hung too under this fake and was removed
        # the same way.)
        assert all(r[:3] == ["docker", "rm", "-f"] for r in removed)
        assert any(r[3].startswith("helix-warm-codex-") for r in removed)

    def test_timed_out_warm_is_a_warning_in_the_loop(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            "helix.evolution.warm_backend_credential",
            lambda backend, **_k: CredentialWarmResult(
                backend=backend, warmed=False, timed_out=True, detail="slow"
            ),
        )
        result = _warm_generation_credential(
            _config("codex", sandboxed=True), gen=3, announce_skip=False
        )
        assert result is not None and result.failed and result.timed_out
        printed = " ".join(capsys.readouterr().out.lower().split())
        assert "timed out" in printed
        assert "exit none" not in printed
        assert "run continues" in printed


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
