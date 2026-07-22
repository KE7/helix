"""Production-CALL assertions for controls whose bodies were already tested.

F-12, F-17, F-18 and F-20 were ONE defect repeated four times: a correct,
well-tested function whose production CALL was unasserted. A library test
cannot detect non-wiring, by construction -- each was found only by mutation.

Everything here asserts the CALL, at the public/run boundary, and each states
the mutation it catches. See the wiring table in
``docs/design/sandbox-home-isolation.md`` §8d.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from helix.config import (
    AgentConfig,
    EvaluatorConfig,
    EvolutionConfig,
    HelixConfig,
    SandboxConfig,
)


def _config(auth: str) -> HelixConfig:
    return HelixConfig(
        objective="o",
        evaluator=EvaluatorConfig(command="true", score_parser="helix_result"),
        agent=AgentConfig(backend="claude"),
        evolution=EvolutionConfig(max_generations=0, max_evaluations=1),
        sandbox=SandboxConfig(
            enabled=True,
            image="pinned@sha256:deadbeef",
            auth=auth,  # type: ignore[arg-type]
            auth_env_allow=["ANTHROPIC_API_KEY"] if auth == "env" else [],
        ),
    )


# ---------------------------------------------------------------------------
# F-17 -- the env-mode disclosure must be EMITTED, not merely correct
# ---------------------------------------------------------------------------


def test_F17_env_mode_disclosure_is_emitted_on_the_run_path(
    tmp_path: Path, capsys, mocker
) -> None:
    """Deleting ``print(env_mode_disclosure(config))`` must RED.

    EA mandated this disclosure as a CONDITION OF SHIPPING env mode, and env
    mode is the only supported path for all four demo lanes. The CONTENT was
    asserted (making the function return "" goes red); the EMISSION was not, so
    one deleted line stopped every demo run disclosing that the named host
    credential is present in the agent container and that OAuth refresh is
    suppressed.

    Note the granularity: deleting the whole ``if config.sandbox.enabled:``
    block also goes red, so a COARSE mutation would have reported this area as
    covered. Only the surgical single-line deletion exposes it -- which is why
    this asserts stdout rather than reachability.
    """
    from helix.evolution import run_evolution

    # Stop the run immediately after startup; the disclosure precedes this.
    # Stop immediately AFTER startup; the disclosure precedes this call.
    mocker.patch(
        "helix.evolution._run_evolution_impl", side_effect=RuntimeError("stop-here")
    )
    with pytest.raises(Exception):
        run_evolution(_config("env"), tmp_path, tmp_path / ".helix")

    out = capsys.readouterr().out
    assert "refresh" in out.lower(), "must disclose that OAuth refresh is suppressed"
    assert "container" in out.lower(), (
        "must disclose that the credential is present in the agent container"
    )
    # and it must not leak a value while doing so
    assert "sk-" not in out


def test_F17_non_vacuity_disclosure_is_absent_when_not_in_env_mode(
    tmp_path: Path, capsys, mocker
) -> None:
    """Control: the assertion above is not matching unrelated startup output.

    Without this, any run that printed the word "container" anywhere would
    satisfy the test and the emission could still be deleted.
    """
    from helix.evolution import run_evolution

    mocker.patch(
        "helix.evolution._run_evolution_impl", side_effect=RuntimeError("stop-here")
    )
    with pytest.raises(Exception):
        run_evolution(_config("volume"), tmp_path, tmp_path / ".helix")
    out = capsys.readouterr().out
    assert "OAuth refresh is suppressed" not in out


# ---------------------------------------------------------------------------
# preflight_auth -- purpose-built, previously HELD ONLY INCIDENTALLY
# ---------------------------------------------------------------------------


def test_preflight_auth_is_invoked_on_the_volume_mode_run_path(
    tmp_path: Path, mocker
) -> None:
    """A purpose-built assertion that ``preflight_auth`` runs.

    It was previously held ONLY by ``test_docker_guard.py``'s
    ``test_run_evolution_unit_path_cannot_reach_docker``, whose stated purpose
    is that the unit path must not reach Docker -- NOT that preflight runs. A
    refactor of the docker guard could have deleted the only assertion
    protecting this call without anyone realising that is what it was doing.

    Catches: deleting the ``preflight_auth(...)`` call.
    """
    from helix.evolution import run_evolution

    called: list[bool] = []

    def spy(config, **kwargs):
        called.append(True)
        raise RuntimeError("stop-after-preflight")

    mocker.patch("helix.evolution.preflight_auth", side_effect=spy)
    with pytest.raises(Exception):
        run_evolution(_config("volume"), tmp_path, tmp_path / ".helix")
    assert called, "preflight_auth was not invoked on the volume-mode run path"


# ---------------------------------------------------------------------------
# F-18 / B1 -- the subpath bootstrap CALL
# ---------------------------------------------------------------------------


def test_F18_login_invokes_the_subpath_bootstrap(mocker) -> None:
    """Deleting the bootstrap call from ``helix sandbox login`` must RED.

    Only ``auth_subpath_bootstrap_command`` was tested, by direct call. Without
    the bootstrap, login reports success over a store whose auth subpath does
    not exist, and the daemon error ``missing_subpath_error`` exists to
    translate surfaces later -- or never.
    """
    import helix.cli as cli

    called: list[str] = []
    mocker.patch.object(
        cli, "auth_subpath_bootstrap_command", side_effect=lambda p: called.append(p)
    )
    mocker.patch.object(
        cli,
        "production_docker_runner",
        return_value=lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""),
    )
    cli._ensure_auth_subpath("claude", "img:latest")
    assert called == [".claude"], "login did not build the bootstrap command"


def test_F18_nonzero_bootstrap_is_not_silent(mocker) -> None:
    """A failed bootstrap must RAISE, never be swallowed.

    This is the ``check=False`` with an unbound result pattern that
    ``helix/transcripts.py``'s own docstring names as the defect it replaced --
    reintroduced here once and now guarded.
    """
    import helix.cli as cli
    from helix.exceptions import AuthSubpathBootstrapError

    mocker.patch.object(
        cli,
        "production_docker_runner",
        return_value=lambda *a, **k: subprocess.CompletedProcess([], 1, "", "boom"),
    )
    with pytest.raises(AuthSubpathBootstrapError):
        cli._ensure_auth_subpath("claude", "img:latest")


# ---------------------------------------------------------------------------
# EA gate (2): no argv mutation AFTER validation -- by CONSTRUCTION
# ---------------------------------------------------------------------------


def test_mount_validation_is_the_last_statement_before_return() -> None:
    """Validate-then-append is the same hole as declare-then-not-apply.

    "No argv mutation occurs after validation" previously held by INSPECTION.
    That distinction -- checking the declaration instead of the artifact -- is
    the subject of this entire audit, so it is now enforced structurally: the
    guard call must be the LAST statement before ``return args``.

    Catches: appending to ``args`` after the guard has run, which would let a
    mount be added to the final argv without ever being validated.
    """
    import helix.sandbox as sandbox_mod

    tree = ast.parse(Path(sandbox_mod.__file__).read_text())
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_docker_args"
    )
    body = func.body
    assert isinstance(body[-1], ast.Return), "_docker_args must end with a return"

    guard_index = max(
        index
        for index, stmt in enumerate(body)
        if "_assert_no_shared_home_mount" in ast.dump(stmt)
    )
    assert guard_index == len(body) - 2, (
        "the mount validation must be the LAST statement before `return args`; "
        f"found {len(body) - 2 - guard_index} statement(s) after it, which "
        "could mutate the argv post-validation"
    )
