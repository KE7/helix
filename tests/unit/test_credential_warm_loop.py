"""The warm's placement in the loop, and how a credential failure ends up
somewhere an operator will actually read it.

Two things are asserted end-to-end against the real evolution loop:

* the warm fires once per generation -- not once per run, because a run
  outlives any refresh interval, and not once per candidate, because that is
  the race it exists to prevent; and
* when a mutation dies on the shared credential, the run says so in its own
  terms and keeps going, instead of filing the slot under "the agent wrote bad
  code" and finishing silently.
"""

from __future__ import annotations

from typing import Any

import pytest

from helix.config import SandboxConfig
from helix.evolution import run_evolution
from helix.exceptions import CredentialRefreshError
from helix.sandbox import CredentialWarmResult
from tests.unit.test_evolution import (  # type: ignore[import-untyped]
    all_mocks,  # noqa: F401, F811 — re-exported pytest fixture
    make_candidate,
    make_config,
    make_eval_result,
)


def _sandboxed(config: Any) -> Any:
    return config.model_copy(update={"sandbox": SandboxConfig(enabled=True)})


@pytest.fixture()
def warm_calls(mocker: Any) -> list[str]:
    calls: list[str] = []

    def _fake(backend: str, **_kwargs: Any) -> CredentialWarmResult:
        calls.append(backend)
        return CredentialWarmResult(backend=backend, warmed=True, returncode=0)

    mocker.patch("helix.evolution.warm_backend_credential", side_effect=_fake)
    return calls


class TestWarmRunsOncePerGeneration:
    def test_one_warm_per_generation(
        self, mocker, tmp_path, all_mocks, warm_calls  # noqa: F811
    ) -> None:
        """Three generations, three warms.

        A single warm at startup would leave a long run unprotected the moment
        the credential next goes stale mid-flight, so the count must track
        generations rather than runs.
        """
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None
        all_mocks["run_evaluator"].side_effect = (
            lambda candidate, *a, **k: make_eval_result(
                candidate.id, {"i1": 0.5, "i2": 0.5}
            )
        )

        config = _sandboxed(
            make_config(max_generations=3, perfect_score_threshold=None)
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert warm_calls == ["claude", "claude", "claude"]

    def test_warm_precedes_every_mutation(
        self, mocker, tmp_path, all_mocks  # noqa: F811
    ) -> None:
        """Ordering, not just counting: the credential is fresh before any
        candidate could start refreshing it for itself."""
        order: list[str] = []
        mocker.patch(
            "helix.evolution.warm_backend_credential",
            side_effect=lambda backend, **_k: (
                order.append("warm"),
                CredentialWarmResult(backend=backend, warmed=True, returncode=0),
            )[1],
        )

        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].side_effect = lambda *a, **k: (
            order.append("mutate"),
            None,
        )[1]
        all_mocks["run_evaluator"].side_effect = (
            lambda candidate, *a, **k: make_eval_result(
                candidate.id, {"i1": 0.5, "i2": 0.5}
            )
        )

        config = _sandboxed(
            make_config(max_generations=2, perfect_score_threshold=None)
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert order[0] == "warm"
        assert order.count("warm") == 2
        for index, event in enumerate(order):
            if event == "mutate":
                assert "warm" in order[:index]

    def test_unsandboxed_run_warms_nothing(
        self, mocker, tmp_path, all_mocks, warm_calls  # noqa: F811
    ) -> None:
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None
        all_mocks["run_evaluator"].side_effect = (
            lambda candidate, *a, **k: make_eval_result(
                candidate.id, {"i1": 0.5, "i2": 0.5}
            )
        )

        config = make_config(max_generations=2, perfect_score_threshold=None)
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert warm_calls == []


class TestCredentialFailureIsVisible:
    def test_failure_is_named_and_the_run_survives(
        self,
        mocker,  # noqa: F811
        tmp_path,
        all_mocks,  # noqa: F811
        warm_calls,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A dead credential must not read as a run that merely found nothing.

        The per-slot error names the cause, and the permanent end-of-run
        summary repeats it after the live display is gone -- an operator who
        looks only at the tail of the run still learns that the login, not the
        code, is what failed.
        """
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["run_evaluator"].side_effect = (
            lambda candidate, *a, **k: make_eval_result(
                candidate.id, {"i1": 0.5, "i2": 0.5}
            )
        )
        all_mocks["mutate"].side_effect = CredentialRefreshError(
            "Codex CLI could not use its stored credential "
            "(matched 'your access token could not be refreshed' in stderr)",
            suggestion="Re-authenticate with `helix sandbox login codex`.",
        )

        config = _sandboxed(
            make_config(max_generations=2, perfect_score_threshold=None)
        )
        # The run completes rather than crashing: candidates may still work.
        result = run_evolution(config, tmp_path, tmp_path / ".helix")
        assert result.best_candidate.id == "g0-s0"

        # Rich wraps the console output; compare on collapsed whitespace.
        out = " ".join(capsys.readouterr().out.lower().split())
        assert "credential" in out
        assert "not a failure of the candidate's code" in out
        assert "helix sandbox login" in out

    def test_clean_run_says_nothing_about_credentials(
        self,
        mocker,  # noqa: F811
        tmp_path,
        all_mocks,  # noqa: F811
        warm_calls,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None
        all_mocks["run_evaluator"].side_effect = (
            lambda candidate, *a, **k: make_eval_result(
                candidate.id, {"i1": 0.5, "i2": 0.5}
            )
        )

        config = _sandboxed(
            make_config(max_generations=2, perfect_score_threshold=None)
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert "credential" not in capsys.readouterr().out.lower()
