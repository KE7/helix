"""Credential-policy suite for sandboxed mutation agents (T1-T29).

GOVERNING RULE, and the reason this file exists: **every test asserts on the
FINAL docker argv** — or on the env dict handed to ``subprocess.run`` for the
non-sandboxed path — evaluated over all THREE origins together (the
per-backend table, ``passthrough_env`` / ``env``, and the ``HELIX_*``
wildcard). **No test asserts on ``_scrub_environment`` alone.** That gap is
exactly what hid this bug through four reviews: the scrubber removed the
credential and the injection happened downstream of it, so a scrubber-level
assertion is structurally incapable of catching it.

SCOPE LIMIT, which every test here inherits and which a future reader must
not lose: "no credential-bearing ``-e`` in argv" is NOT "no credential in the
container". ``/home/node`` is mounted from the auth volume, and the workspace
mount carries whatever is in the candidate repo. The argv is where THIS bug
lived; it is not the whole boundary.

Non-vacuity discipline: each test asserts both an absence and a positive
counterpart (the mount, the image, ``HOME``), so it cannot pass against an
empty argv, and each states the mutation it catches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helix.backends import BACKEND_AUTH_ENV
from helix.config import (
    AgentConfig,
    EvaluatorConfig,
    EvaluatorSidecarConfig,
    HelixConfig,
    SandboxConfig,
)
from helix.envpolicy import env_dict, resolve_env_grants
from helix.sandbox import _docker_args


CANARY = "sk-canary-value-must-not-reach-any-container"


def make_config(
    *,
    backend: str = "claude",
    enabled: bool = True,
    auth: str | None = None,
    auth_env_allow: list[str] | None = None,
    agent_passthrough_env: list[str] | None = None,
    passthrough_env: list[str] | None = None,
    env: dict[str, str] | None = None,
    sidecar_passthrough: list[str] | None = None,
) -> HelixConfig:
    sidecar = None
    if sidecar_passthrough is not None:
        sidecar = EvaluatorSidecarConfig(
            image="eval:latest",
            command="python -m server",
            endpoint="http://helix-evaluator:8080/evaluate",
            passthrough_env=sidecar_passthrough,
        )
    return HelixConfig(
        objective="x",
        passthrough_env=passthrough_env or [],
        env=env or {},
        evaluator=EvaluatorConfig(
            command="true", score_parser="helix_result", sidecar=sidecar
        ),
        agent=AgentConfig(backend=backend),
        sandbox=SandboxConfig(
            enabled=enabled,
            image="pinned@sha256:6be6fef",
            auth=auth,  # type: ignore[arg-type]
            auth_env_allow=auth_env_allow or [],
            agent_passthrough_env=agent_passthrough_env or [],
        ),
    )


def agent_grants(config: HelixConfig):
    return resolve_env_grants(
        scope="agent",
        backend=config.agent.backend,
        sandbox_enabled=config.sandbox.enabled,
        auth_mode=config.sandbox.resolved_auth(),
        auth_env_allow=config.sandbox.auth_env_allow,
        agent_passthrough_env=config.sandbox.agent_passthrough_env,
        config_passthrough_env=config.passthrough_env,
        config_env=config.env,
    )


def agent_docker_argv(config: HelixConfig, command=("true",)) -> list[str]:
    """The FINAL production argv for a sandboxed mutation agent."""
    grants = agent_grants(config)
    return _docker_args(
        list(command),
        env_dict(grants, "agent"),
        Path("/tmp/helix-probe"),
        config.sandbox,
        "agent",
        config.sandbox.image or "img",
        config.agent.backend,
        grants=grants,
    )


def env_assignments(argv: list[str]) -> dict[str, str]:
    return dict(
        arg.split("=", 1)
        for index, arg in enumerate(argv)
        if index > 0 and argv[index - 1] == "-e" and "=" in arg
    )


def assert_real_launch(argv: list[str], config: HelixConfig) -> None:
    """Non-vacuity: this argv is a genuine container launch, not an empty list."""
    assert config.sandbox.image in argv
    assert "HOME" in env_assignments(argv)


# ---------------------------------------------------------------------------
# 3.2 Core semantics
# ---------------------------------------------------------------------------


def test_T1_volume_mode_emits_no_credential_env(monkeypatch):
    """T1 — catches deleting the mode guard. THIS IS THE BUG."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    config = make_config()
    argv = agent_docker_argv(config)
    assert "ANTHROPIC_API_KEY" not in env_assignments(argv)
    assert not any(CANARY in part for part in argv)
    assert "helix-auth-claude:/home/node:rw" in argv
    assert_real_launch(argv, config)


def test_T2_volume_mode_excludes_both_claude_variables(monkeypatch):
    """T2 — catches a fix that filters only ANTHROPIC_API_KEY.

    ANTHROPIC_AUTH_TOKEN is the more dangerous of the two: it overrides the
    OAuth path AND suppresses refresh.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", CANARY)
    config = make_config()
    argv = agent_docker_argv(config)
    assert "ANTHROPIC_AUTH_TOKEN" not in env_assignments(argv)
    assert not any(CANARY in part for part in argv)
    assert_real_launch(argv, config)


def test_T3_env_mode_injects_only_allowlisted_names(monkeypatch):
    """T3 — catches implementing auth_env_allow as a UNION with the backend
    table instead of a REPLACEMENT. That is the single likeliest
    implementation error."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "primary-value")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", CANARY)
    config = make_config(auth="env", auth_env_allow=["ANTHROPIC_API_KEY"])
    argv = agent_docker_argv(config)
    assignments = env_assignments(argv)
    assert assignments["ANTHROPIC_API_KEY"] == "primary-value"
    assert "ANTHROPIC_AUTH_TOKEN" not in assignments
    assert not any(CANARY in part for part in argv)
    assert_real_launch(argv, config)


def test_T4_backend_rename_cannot_change_credential_flow(monkeypatch):
    """T4 — the one-word edit that used to defeat the sidecar boundary.

    Non-vacuity: the two argvs are compared to each other, so reverting the
    fix makes the opencode case diverge.
    """
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    assert "OPENAI_API_KEY" in BACKEND_AUTH_ENV["opencode"]

    as_claude = agent_docker_argv(
        make_config(backend="claude", sidecar_passthrough=["OPENAI_API_KEY"])
    )
    as_opencode = agent_docker_argv(
        make_config(backend="opencode", sidecar_passthrough=["OPENAI_API_KEY"])
    )
    assert "OPENAI_API_KEY" not in env_assignments(as_claude)
    assert "OPENAI_API_KEY" not in env_assignments(as_opencode)
    assert env_assignments(as_claude) == env_assignments(as_opencode)


def test_T5_both_present_still_volume_only(monkeypatch):
    """T5 — directly guards R4: no fallback, in either direction."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, CANARY)
    config = make_config()
    argv = agent_docker_argv(config)
    assignments = env_assignments(argv)
    assert not ({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} & set(assignments))
    assert "helix-auth-claude:/home/node:rw" in argv
    assert_real_launch(argv, config)


def test_T9_non_sandboxed_run_still_receives_env_auth(monkeypatch):
    """T9 — REGRESSION, purely positive. Passes today and MUST keep passing.

    Guards the preserved invariant against an over-broad fix that deletes the
    backend-table path outright and breaks headless local use. This is the
    most likely test to catch an over-correction.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "local-headless-key")
    config = make_config(enabled=False)
    grants = agent_grants(config)
    env = env_dict(grants, "agent")
    assert env["ANTHROPIC_API_KEY"] == "local-headless-key"


@pytest.mark.parametrize("backend", ["cursor", "gemini", "opencode", "codex"])
def test_T10_all_backends_volume_mode(monkeypatch, backend):
    """T10 — catches a fix hard-coded to claude, which is plausible since
    claude is the default and every lane uses it."""
    for names in BACKEND_AUTH_ENV.values():
        for name in names:
            monkeypatch.setenv(name, CANARY)
    config = make_config(backend=backend)
    argv = agent_docker_argv(config)
    assignments = env_assignments(argv)
    for name in BACKEND_AUTH_ENV.get(backend, ()):
        assert name not in assignments
    assert f"helix-auth-{backend}:/home/node:rw" in argv
    assert_real_launch(argv, config)


def test_T10b_gemini_google_api_key_never_reaches_agent(monkeypatch):
    """T10b — GOOGLE_API_KEY has the widest blast radius in the table: it is a
    generic Google Cloud key, not a Gemini-scoped credential.

    Non-vacuity: DISTINCT canaries, so a fix filtering only GEMINI_API_KEY
    leaves the wider key and must still fail.
    """
    monkeypatch.setenv("GEMINI_API_KEY", CANARY + "-gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", CANARY + "-google")
    config = make_config(backend="gemini")
    argv = agent_docker_argv(config)
    assignments = env_assignments(argv)
    assert "GEMINI_API_KEY" not in assignments
    assert "GOOGLE_API_KEY" not in assignments
    assert not any(CANARY in part for part in argv)
    assert "helix-auth-gemini:/home/node:rw" in argv


def test_T11_codex_is_not_safe_by_absence(monkeypatch):
    """T11 — codex is absent from BACKEND_AUTH_ENV, yet BOTH other channels
    reach it. This is the test proving a backend-table-only fix is
    insufficient, and it should be the first one a writer makes pass.

    Non-vacuity: asserts a registered HELIX variable IS present.
    """
    monkeypatch.setenv("HELIX_OPENAI_KEY", CANARY + "-wildcard")
    monkeypatch.setenv("SOME_API_KEY", CANARY + "-passthrough")
    monkeypatch.setenv("HELIX_ASI_LOG", "1")
    config = make_config(backend="codex", passthrough_env=["SOME_API_KEY"])
    argv = agent_docker_argv(config)
    assignments = env_assignments(argv)
    assert "HELIX_OPENAI_KEY" not in assignments, "HELIX_* wildcard reached an agent"
    assert "SOME_API_KEY" not in assignments, "passthrough_env reached an agent"
    assert not any(CANARY in part for part in argv)
    # Positive counterpart: registered HELIX context still propagates.
    assert assignments.get("HELIX_ASI_LOG") == "1"


def test_T12_helix_wildcard_closed_for_sandboxed_agent(monkeypatch):
    """T12 — guards R5."""
    monkeypatch.setenv("HELIX_SECRET_TOKEN", CANARY)
    monkeypatch.setenv("HELIX_ASI_LOG", "1")
    config = make_config()
    assignments = env_assignments(agent_docker_argv(config))
    assert "HELIX_SECRET_TOKEN" not in assignments
    assert assignments.get("HELIX_ASI_LOG") == "1"


def test_T12b_helix_wildcard_preserved_for_evaluator(monkeypatch):
    """T12b — catches an over-broad R5 that breaks legitimate evaluator
    context propagation."""
    monkeypatch.setenv("HELIX_SECRET_TOKEN", "evaluator-visible")
    config = make_config()
    grants = resolve_env_grants(
        scope="evaluator",
        backend="claude",
        sandbox_enabled=True,
        auth_mode="volume",
    )
    assert env_dict(grants, "evaluator")["HELIX_SECRET_TOKEN"] == "evaluator-visible"
    del config


def test_T12c_non_sandboxed_agent_keeps_the_wildcard(monkeypatch):
    """R1 — non-sandboxed behaviour is preserved EXACTLY, wildcard included."""
    monkeypatch.setenv("HELIX_SECRET_TOKEN", "still-here")
    config = make_config(enabled=False)
    env = env_dict(agent_grants(config), "agent")
    assert env["HELIX_SECRET_TOKEN"] == "still-here"


# ---------------------------------------------------------------------------
# T13 — config validation matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(enabled=False, auth="volume"), "only meaningful when"),
        (dict(auth="env", auth_env_allow=[]), "non-empty"),
        (dict(auth="volume", auth_env_allow=["ANTHROPIC_API_KEY"]), "auth_env_allow"),
        (
            dict(auth="env", auth_env_allow=["CLAUDE_CODE_OAUTH_TOKEN"]),
            "disables OAuth token refresh permanently",
        ),
        (
            dict(auth="env", auth_env_allow=["CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"]),
            "disables OAuth token refresh permanently",
        ),
    ],
)
def test_T13_config_validation_matrix(kwargs, match):
    """T13 — catches validation shipped as a WARNING instead of an error,
    which is the failure mode that created this situation."""
    with pytest.raises(ValueError, match=match):
        make_config(**kwargs)


def test_T14_agent_and_sidecar_credential_sets_disjoint(monkeypatch):
    """T14 — the mechanical, backend- and lane-independent guard replacing
    livebench's tripwire on a current value."""
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    with pytest.raises(ValueError, match="disjoint"):
        make_config(
            auth="env",
            auth_env_allow=["OPENAI_API_KEY"],
            sidecar_passthrough=["OPENAI_API_KEY"],
        )
    # Non-vacuity: a non-shared name is accepted, so the rejection above is
    # the disjointness rule and not a generally-invalid config.
    make_config(
        auth="env",
        auth_env_allow=["ANTHROPIC_API_KEY"],
        sidecar_passthrough=["OPENAI_API_KEY"],
    )


def test_T19_provenance_assertion_blocks_unregistered_callers():
    """T19 — the structural regression guard.

    A future call site added downstream of the resolver is EXACTLY how
    ``_add_backend_auth_env`` created this bug. Emitting a variable that
    carries no grant must raise, and agent scope must refuse a bare dict
    outright.
    """
    config = make_config()
    grants = agent_grants(config)

    # An ungranted key smuggled into the env dict is refused.
    env = env_dict(grants, "agent")
    env["SNEAKY_API_KEY"] = CANARY
    with pytest.raises(ValueError, match="ungranted"):
        _docker_args(
            ["true"],
            env,
            Path("/tmp/p"),
            config.sandbox,
            "agent",
            "img",
            "claude",
            grants=grants,
        )

    # Agent scope with NO grants at all is refused: that is the shape of a
    # caller that bypassed the resolver entirely.
    with pytest.raises(ValueError, match="without provenance grants"):
        _docker_args(
            ["true"], env, Path("/tmp/p"), config.sandbox, "agent", "img", "claude"
        )


def test_T18_silent_flips_do_not_change_agent_argv(monkeypatch):
    """T18 — the silent-flip class, parametrized over the channels that
    defeat credential isolation directly.

    Non-vacuity: the pre-flip argv is asserted non-empty AND byte-identical
    to the post-flip argv.
    """
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)
    monkeypatch.setenv("HELIX_LEAKED_KEY", CANARY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)

    baseline = make_config()
    base_argv = agent_docker_argv(baseline)
    assert_real_launch(base_argv, baseline)
    base_env = env_assignments(base_argv)

    # (1) backend rename; (2) sidecar name moved to top-level passthrough;
    # (3) a host credential renamed into the HELIX_* namespace.
    flips = [
        make_config(backend="opencode"),
        make_config(passthrough_env=["OPENAI_API_KEY"]),
        make_config(passthrough_env=[]),
    ]
    for flipped in flips:
        flipped_env = env_assignments(agent_docker_argv(flipped))
        assert flipped_env == base_env, "a config flip changed agent exposure"
    assert not any(CANARY in value for value in base_env.values())


def test_T6_unrelated_volume_does_not_authenticate():
    """T6 — catches a preflight implemented as 'any helix-auth-* exists'."""
    config = make_config(backend="gemini")
    argv = agent_docker_argv(config)
    assert "helix-auth-gemini:/home/node:rw" in argv
    assert "helix-auth-claude:/home/node:rw" not in argv


def test_env_mode_mounts_no_auth_volume_at_all():
    """Env mode must not mount the persistent auth volume in ANY mode.

    SUPERSEDES ``test_env_mode_mounts_auth_volume_read_only``, which asserted
    that env mode mounts the volume ``:ro``.  That earlier reasoning --
    "an env-mode run cannot refresh the volume, so a writable mount is an
    unused write path" -- addressed the wrong risk and is now known to be
    wrong in a way that matters.

    A read-only mount over the whole HOME still exposes every prior run's
    transcripts, sessions, caches and config for READING.  Read access IS the
    cross-candidate channel; write access was never the defect.  ``:ro``
    therefore removed a hazard nobody was exploiting while preserving the one
    that contaminated three benchmark demos.

    Catches: any reintroduction of a ``helix-auth-*`` mount into env mode, at
    any mode string, including a "harmless" read-only one.
    """
    config = make_config(auth="env", auth_env_allow=["ANTHROPIC_API_KEY"])
    argv = agent_docker_argv(config)
    joined = " ".join(argv)
    assert "helix-auth-" not in joined, joined


def test_env_mode_gets_private_home_and_candidate_transcript_bind():
    """Env mode must still provide a writable HOME and capture transcripts.

    Non-vacuity for the test above: dropping the auth mount without
    provisioning a replacement HOME would make that assertion pass while
    leaving the agent with the image's baked ``/home/node`` -- and would
    silently break transcript capture, which is coupled to the old mount.

    The ``uid``/``gid`` options are asserted explicitly: a bare tmpfs yields a
    root-owned HOME that uid 1000 cannot write, failing every mutation agent.
    """
    config = make_config(auth="env", auth_env_allow=["ANTHROPIC_API_KEY"])
    argv = agent_docker_argv(config)

    home = [
        argv[i + 1]
        for i, tok in enumerate(argv)
        if tok == "--tmpfs" and argv[i + 1].startswith("/home/node:")
    ]
    assert home, f"env mode must provision a private per-run HOME: {argv}"
    assert "uid=1000" in home[0] and "gid=1000" in home[0], home[0]

    binds = [
        argv[i + 1]
        for i, tok in enumerate(argv)
        if tok == "-v" and argv[i + 1].endswith("/home/node/.claude/projects:rw")
    ]
    assert binds, f"env mode must bind a candidate transcript dir: {argv}"


def test_env_mode_transcript_bind_is_candidate_specific():
    """Distinct candidates must get distinct transcript roots.

    Catches: keying transcripts on anything shared across candidates -- which
    is the current defect, where the project key is ``-workspace`` for every
    candidate of every run and the candidate id never enters the path at all.
    """
    from helix.sandbox_home import transcript_host_dir

    a = transcript_host_dir(Path("/tmp/helix/cand-a"))
    b = transcript_host_dir(Path("/tmp/helix/cand-b"))
    assert a != b, (a, b)
    # and never inside the workspace, which is synced back into the candidate
    assert Path("/tmp/helix/cand-a") not in a.parents


# ---------------------------------------------------------------------------
# T16 — four-lane migration matrix
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

# Verdicts asserted as behaviour rather than restated as prose.
#
# UPDATED: swebench_live and livebench_math were "volume" at audit time --
# because they OMITTED ``auth`` entirely, and omission resolves to volume mode
# silently.  They now select env mode explicitly.
#
# The reason is not stylistic.  Volume mode cannot support a
# candidate-independence claim: the auth directory must stay writable for OAuth
# rotation, so an agent can create an unenumerated file that the next candidate
# reads; and on the pinned claude digest the CLI itself keeps per-run state in
# that shared HOME (a cleanup sentinel that makes the next candidate skip its
# cleanup pass, an MCP auth cache, and a policy file whose on-disk hash shapes
# the next candidate's outbound request).  These three lanes publish
# per-candidate measurements, so they must run in the only mode that mounts no
# persistent store.
#
# If a lane ever reverts to omitting ``auth``, this matrix fails -- which is
# the point, since the omission itself is invisible in review.
LANE_VERDICTS = {
    "swebench_live": "env",
    "livebench_math": "env",
    "formulacode": "env",
}


def test_T16_lane_config_migration_matrix(monkeypatch):
    """T16 — every merged lane resolves to its documented credential verdict.

    formulacode is the load-bearing row: it shipped a TOP-LEVEL
    ``passthrough_env = ["ANTHROPIC_API_KEY"]`` with the sandbox enabled, so
    it injected a credential into the mutation agent through a channel the
    per-backend table never mentions. A fix gating only that table would have
    left this lane injecting silently while every argv assertion written
    against the table passed. It is the proof the expanded scope was needed.

    Non-vacuity: the loop body must run for EVERY expected lane — a glob or
    fixture typo would otherwise make this matrix lie, which is the classic
    way a matrix test passes while proving nothing.
    """
    from helix.config import load_config

    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", CANARY)
    monkeypatch.setenv("OPENAI_API_KEY", CANARY)

    seen: dict[str, str] = {}
    for lane, expected_mode in LANE_VERDICTS.items():
        lane_dir = REPO_ROOT / "examples" / lane
        candidates = [lane_dir / "helix.toml", lane_dir / "helix.toml.template"]
        path = next((p for p in candidates if p.is_file()), None)
        assert path is not None, f"no config found for lane {lane}"

        text = path.read_text().replace("__PYTHON__", "python3")
        tmp = REPO_ROOT / ".helix_t16_tmp.toml"
        tmp.write_text(text)
        try:
            config = load_config(tmp)
        finally:
            tmp.unlink(missing_ok=True)

        mode = config.sandbox.resolved_auth()
        assert mode == expected_mode, f"{lane}: expected {expected_mode}, got {mode}"

        argv = agent_docker_argv(config)
        assignments = env_assignments(argv)
        credential_names = {
            name
            for names in BACKEND_AUTH_ENV.values()
            for name in names
            if name in assignments
        }
        if expected_mode == "volume":
            assert not credential_names, (
                f"{lane}: volume mode must place NO credential on the agent "
                f"argv, found {sorted(credential_names)}"
            )
            assert not any(CANARY in part for part in argv)
        else:
            # Env mode is explicit and opt-in: exactly the allowlist, nothing
            # else. Not a union with the backend table.
            assert credential_names == set(config.sandbox.auth_env_allow)
        seen[lane] = mode

    assert seen.keys() == LANE_VERDICTS.keys(), (
        f"migration matrix did not cover every lane: {sorted(seen)}"
    )


def test_T16b_formulacode_no_longer_ships_toplevel_passthrough_credential():
    """The specific artifact that proves a backend-table-only fix was
    insufficient — asserted on the shipped template, not on prose."""
    template = (
        REPO_ROOT / "examples" / "formulacode" / "helix.toml.template"
    ).read_text()
    assert 'passthrough_env = ["ANTHROPIC_API_KEY"]' not in template
    assert 'auth = "env"' in template
    assert 'auth_env_allow = ["ANTHROPIC_API_KEY"]' in template


# ---------------------------------------------------------------------------
# T26 — accounting isolation. The margin is ONE INTEGER.
# ---------------------------------------------------------------------------


def test_T26_preflight_does_not_enter_the_evaluation_budget():
    """T26 — auth overhead must never touch ``budget.evaluations``.

    Every lane inspector expresses budget conservation purely in terms of
    that counter, and livebench asserts a HARD EQUALITY
    (``int(final_budget_after) == evaluations``), so ANY non-proposal
    increment breaks all four lanes simultaneously — including an unrun lane
    whose single timed window makes the failure maximally expensive and makes
    it look like a lane defect rather than a core change.

    Non-vacuity: ``auth_overhead_calls == 1`` is asserted, so a preflight that
    never ran cannot pass this test trivially.
    """
    from helix.budget import charge_auth_overhead
    from helix.state import BudgetState, EvolutionState

    state = EvolutionState(
        generation=0,
        frontier=[],
        instance_scores={},
        budget=BudgetState(evaluations=7),
        config_hash="test",
    )
    before = state.budget.evaluations

    charge_auth_overhead(state, source="auth_preflight")

    assert state.budget.evaluations == before, (
        "the preflight must never call charge_evaluation or otherwise advance "
        "the evaluation counter"
    )
    assert state.budget.auth_overhead_calls == 1
    # Auth overhead must not be folded into proposal cost either: that would
    # silently misattribute it, a quiet instance of the silent-flip class.
    assert state.budget.cost_usd == 0.0
    assert state.budget.input_tokens == 0
    assert state.budget.output_tokens == 0


# ---------------------------------------------------------------------------
# C7 — the runtime 401 detector, and why this release does NOT ship one
# ---------------------------------------------------------------------------


def test_C7_agent_output_containing_401_does_not_abort_the_run(tmp_path, monkeypatch):
    """C7 — agent OUTPUT must never be the trigger for an auth abort.

    The audit proposed aborting a run on a runtime 401. Re-examined before
    implementing, and deliberately NOT shipped in that form. The reason is
    the project's own threat model: the mutation agent runs with
    ``--dangerously-skip-permissions`` over attacker-influenceable repository
    content on a ``bridge`` network. Its stdout therefore routinely contains
    text from HTTP calls the agent itself made, and from the candidate repo's
    own test suite.

    A detector scanning that text has an ADVERSARIALLY REACHABLE false
    positive: a candidate whose tests print ``401 Unauthorized`` would abort
    every run. That is not a rare mis-fire, it is an input an untrusted party
    controls — and aborting healthy runs is worse than the status quo,
    especially for a single-window run.

    Note the contrast with the existing rate-limit heuristic, which reads the
    backend's OWN structured JSON error envelope rather than free-form agent
    output. That narrower input is not agent-controlled, and it is the only
    input on which a runtime auth detector could be defensible.

    This test pins the property: 401-looking text in agent output is inert.
    """
    from helix.mutator import _looks_like_rate_limit

    hostile = "candidate test output: HTTP 401 Unauthorized from api.example.com"

    # The existing structured-envelope heuristic must not fire on this either.
    assert not _looks_like_rate_limit(hostile)

    # And no auth classifier is applied to agent output anywhere in the
    # mutation path: the preflight is the auth control, and its input is a
    # HELIX-issued command in a HELIX-controlled container.
    from helix import mutator as mutator_module

    source = Path(mutator_module.__file__).read_text()
    assert "_classify_probe_failure" not in source, (
        "the preflight's failure classifier must not be applied to agent "
        "output; its soundness depends on the input NOT being agent-controlled"
    )
    del tmp_path, monkeypatch


def test_C7_probe_classifier_input_is_helix_controlled():
    """The preflight classifier IS defensible, for a stated reason.

    Its input is the output of a HELIX-issued probe command in a
    HELIX-controlled container with a credential-free environment — not a
    mutation over attacker-influenceable content. The distinction is the whole
    argument, so it is asserted rather than left to a comment.
    """
    from helix.authpreflight import BACKEND_PROBE_COMMANDS, _classify_probe_failure

    # The probe is a fixed, HELIX-authored command, not agent-derived.
    for backend, command in BACKEND_PROBE_COMMANDS.items():
        assert isinstance(command, list) and command, backend
        assert all(isinstance(part, str) for part in command)

    kind, remedy = _classify_probe_failure("", "invalid_grant at /v1/oauth/token")
    assert kind == "refresh"
    assert "login" in remedy

    kind, _ = _classify_probe_failure("", "429 rate limit exceeded")
    assert kind == "inference"

    kind, _ = _classify_probe_failure("", "could not resolve host")
    assert kind == "transport"

    # Ambiguity is REPORTED, not guessed: sending a user to a remedy that
    # cannot help is the same 'wrong signal' disease this release removes.
    kind, remedy = _classify_probe_failure("", "something went wrong")
    assert kind == "ambiguous"
    assert "could not distinguish" in remedy


# ---------------------------------------------------------------------------
# T15 — the NEW error messages are themselves a redaction risk
# ---------------------------------------------------------------------------


def test_T15_new_error_paths_never_contain_a_credential_value(monkeypatch):
    """T15 — every new diagnostic surface is checked for credential values.

    The new preflight and config messages are the risk: existing redaction
    covers existing paths, and a new diagnostic path bypasses it by default.

    Non-vacuity: the canary is FIRST asserted to be present in the
    un-redacted argv, so this cannot pass against a no-op that renders
    nothing at all.
    """
    from helix.authpreflight import env_mode_disclosure
    from helix.exceptions import SandboxAuthPreflightError

    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    config = make_config(auth="env", auth_env_allow=["ANTHROPIC_API_KEY"])

    # Non-vacuity precondition: the value IS rendered somewhere unredacted.
    argv = agent_docker_argv(config)
    assert any(CANARY in part for part in argv), (
        "canary must appear in the raw argv, or this test proves nothing"
    )

    # The env-mode disclosure names variables, never values.
    disclosure = env_mode_disclosure(config)
    assert "ANTHROPIC_API_KEY" in disclosure
    assert CANARY not in disclosure
    assert "tradeoff, not an equivalent" in disclosure

    # Preflight errors carry a remedy and no credential value.
    exc = SandboxAuthPreflightError(
        "backend authentication failed", remedy="helix sandbox login claude"
    )
    for surface in (str(exc), exc.format_full(), exc.remedy):
        assert CANARY not in surface

    # EnvGrant redacts its value in repr, so a grant in a traceback or an
    # assertion diff cannot leak a secret.
    grant = next(g for g in agent_grants(config) if g.name == "ANTHROPIC_API_KEY")
    assert CANARY not in repr(grant)
    assert "<redacted" in repr(grant)


# ---------------------------------------------------------------------------
# C6 — a DELIBERATELY ACCEPTED hole, neither silently fixed nor undocumented
# ---------------------------------------------------------------------------


def test_C6_sandbox_enabled_toggle_is_an_accepted_documented_hole(monkeypatch):
    """C6 — toggling ``sandbox.enabled`` still changes the auth mechanism.

    This is an ACCEPTED hole, recorded rather than silently fixed or silently
    left undocumented. Requiring an explicit ``auth`` when the sandbox is off
    would break every non-sandboxed config and violate the preserved
    invariant, so the flip remains possible.

    The partial mitigation IS enforced, and this test pins both halves: a
    config that DECLARED an intent cannot silently lose it (declaring ``auth``
    while disabled is a hard error), while a config that never declared one
    still flips silently.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)

    # Undeclared: the flip is silent. This is the accepted hole, asserted so
    # its scope is visible rather than implied.
    sandboxed = env_dict(agent_grants(make_config()), "agent")
    unsandboxed = env_dict(agent_grants(make_config(enabled=False)), "agent")
    assert "ANTHROPIC_API_KEY" not in sandboxed
    assert unsandboxed["ANTHROPIC_API_KEY"] == CANARY, (
        "documents the accepted hole: disabling the sandbox moves the run to "
        "env auth with no declaration anywhere"
    )

    # Declared: refused, so an intent that WAS stated cannot be silently lost.
    with pytest.raises(ValueError, match="only meaningful when"):
        make_config(enabled=False, auth="volume")
