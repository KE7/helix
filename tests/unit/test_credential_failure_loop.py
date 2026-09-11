"""How a credential failure ends up somewhere an operator will actually read
it.

Asserted end-to-end against the real evolution loop: when a mutation or a
merge dies on the shared credential, the run says so in its own terms and
keeps going, instead of filing the slot under "the agent wrote bad code" and
finishing silently.  A lost refresh race that the one-shot retry recovered is
reported too -- it is not a failure, but it is the only sign the operator gets
that candidates are competing to refresh the shared login.
"""

from __future__ import annotations

from typing import Any

import pytest

from helix.config import AgentConfig, SandboxConfig
from helix.evolution import run_evolution
from helix.exceptions import CredentialRefreshError
from tests.unit.test_evolution import (  # type: ignore[import-untyped]
    all_mocks,  # noqa: F401, F811 — re-exported pytest fixture
    make_candidate,
    make_config,
    make_eval_result,
)


def _sandboxed(config: Any) -> Any:
    """Sandboxed, on a backend whose credential can actually be raced (codex)."""
    return config.model_copy(
        update={
            "sandbox": SandboxConfig(enabled=True),
            "agent": AgentConfig(backend="codex"),
        }
    )


class TestCredentialFailureIsVisible:
    def test_failure_is_named_and_the_run_survives(
        self,
        mocker,  # noqa: F811
        tmp_path,
        all_mocks,  # noqa: F811
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

    def test_merge_credential_failure_is_named_and_the_run_survives(
        self,
        mocker,  # noqa: F811
        tmp_path,
        all_mocks,  # noqa: F811
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The merge gate fires before any mutation in a generation, so a
        stale login can surface there first.  It must get the same treatment
        as a mutation: counted, named, and the run continues into mutation
        rather than dying with a traceback."""
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        all_mocks["merge"].side_effect = CredentialRefreshError(
            "Codex CLI could not use its stored credential "
            "(matched 'your access token could not be refreshed' in stderr)",
            suggestion="Re-authenticate with `helix sandbox login codex`.",
        )
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                return make_eval_result("g1-s1", {"i1": 0.9, "i2": 0.5})
            return make_eval_result(candidate.id, {"i1": 0.5, "i2": 0.8})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _sandboxed(
            make_config(
                max_generations=2,
                merge_enabled=True,
                max_merge_invocations=5,
                merge_val_overlap_floor=1,
                max_evaluations=10000,
            )
        )
        result = run_evolution(config, tmp_path, tmp_path / ".helix")

        all_mocks["merge"].assert_called_once()
        # The run went on to mutate after the merge died on the credential.
        assert all_mocks["mutate"].call_count >= 1
        assert result.best_candidate is not None

        out = " ".join(capsys.readouterr().out.lower().split())
        assert "merge" in out
        assert "credential" in out
        assert "not a failure of the merged code" in out
        # Exactly one diagnosis at the merge site: the generic "returned no
        # output" wording must not follow and contradict it.
        assert out.count("not a failure of the merged code") == 1
        assert "returned no output" not in out
        # The end-of-run summary labels the merge slot as a merge.
        assert "1 merge(s) failed on the shared" in out
        assert "mutation(s) failed" not in out
        assert "helix sandbox login" in out

    def test_merge_lost_refresh_race_is_worded_as_transient(
        self,
        mocker,  # noqa: F811
        tmp_path,
        all_mocks,  # noqa: F811
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        exc = CredentialRefreshError(
            "Codex CLI lost a refresh race on the shared credential "
            "(matched 'because your refresh token was already used' in "
            "stderr; failed again on retry)",
            suggestion="Run `helix resume`.",
        )
        exc.transient = True
        all_mocks["merge"].side_effect = exc
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                return make_eval_result("g1-s1", {"i1": 0.9, "i2": 0.5})
            return make_eval_result(candidate.id, {"i1": 0.5, "i2": 0.8})

        all_mocks["run_evaluator"].side_effect = run_eval
        config = _sandboxed(
            make_config(
                max_generations=2,
                merge_enabled=True,
                max_merge_invocations=5,
                merge_val_overlap_floor=1,
                max_evaluations=10000,
            )
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        out = " ".join(capsys.readouterr().out.lower().split())
        assert "refreshed by another candidate first" in out
        assert "could not be used or refreshed" not in out
        assert "helix sandbox login" not in out
        assert "helix resume" in out

    def test_lost_refresh_race_does_not_demand_a_relogin(
        self,
        mocker,  # noqa: F811
        tmp_path,
        all_mocks,  # noqa: F811
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When every failure was a lost refresh race, the winner has stored a
        working credential.  The summary must say resume, not re-login."""
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["run_evaluator"].side_effect = (
            lambda candidate, *a, **k: make_eval_result(
                candidate.id, {"i1": 0.5, "i2": 0.5}
            )
        )
        exc = CredentialRefreshError(
            "Codex CLI lost a refresh race on the shared credential "
            "(matched 'because your refresh token was already used' in "
            "stderr; failed again on retry)",
            suggestion="Run `helix resume`.",
        )
        exc.transient = True
        all_mocks["mutate"].side_effect = exc

        config = _sandboxed(
            make_config(max_generations=2, perfect_score_threshold=None)
        )
        result = run_evolution(config, tmp_path, tmp_path / ".helix")
        assert result.best_candidate.id == "g0-s0"

        out = " ".join(capsys.readouterr().out.lower().split())
        assert "credential" in out
        assert "refresh race" in out
        assert "helix resume" in out
        assert "helix sandbox login" not in out

    def test_clean_run_says_nothing_about_credentials(
        self,
        mocker,  # noqa: F811
        tmp_path,
        all_mocks,  # noqa: F811
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

    def test_recovered_refresh_race_reaches_the_summary(
        self,
        mocker,  # noqa: F811
        tmp_path,
        all_mocks,  # noqa: F811
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A race the retry recovered from is not a failure, but it is the
        only sign the operator gets that candidates are refreshing the shared
        login for themselves."""
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["run_evaluator"].side_effect = (
            lambda candidate, *a, **k: make_eval_result(
                candidate.id, {"i1": 0.5, "i2": 0.5}
            )
        )

        def _mutate(*args: Any, **kwargs: Any) -> None:
            kwargs["on_refresh_race_recovered"](
                f"{kwargs['new_id']} recovered after a lost refresh race on the "
                "shared Codex CLI credential"
            )
            return None

        all_mocks["mutate"].side_effect = _mutate
        config = _sandboxed(
            make_config(max_generations=2, perfect_score_threshold=None)
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert all_mocks["mutate"].call_count >= 1
        # ``all_mocks`` stubs ``print_warning``; the summary goes through it
        # because a recovered race is a warning, not a failure.
        warnings = " ".join(
            str(call.args[0]) for call in all_mocks["print_warning"].call_args_list
        ).lower()
        assert "recovered after a lost refresh race" in warnings
        assert "g1-s1" in warnings
        assert "failed on the shared" not in capsys.readouterr().out.lower()
