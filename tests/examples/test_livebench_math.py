from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, NoReturn

import pytest

from helix.config import load_config
from helix.executor import _scrub_environment
from helix.mutator import build_mutation_prompt
from helix.population import EvalResult
from helix.sandbox import _copy_tree_contents


ROOT = Path(__file__).parents[2] / "examples" / "livebench_math"
RUNNER_IMAGE = "ghcr.io/ke7/helix-evo-runner-claude@sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba"
sys.path.insert(0, str(ROOT))

import constants  # noqa: E402
import dataset  # noqa: E402
import evaluate  # noqa: E402
import inspect_run  # noqa: E402
import scoring  # noqa: E402


RELEASE_TIP = "c9371f4"
RELEASE_ANCESTORS = ("4622413", "94f9751", "402dcc8", "e5c260f", RELEASE_TIP)
REPO_ROOT = Path(__file__).parents[2]


def _digest_of(image: str) -> str:
    return image.split("@", 1)[1]


def test_runner_pin_is_registry_resolvable_not_a_local_image_id() -> None:
    """The runner pin must resolve in the REGISTRY, not merely parse as a digest.

    This lane previously pinned
    ``ghcr.io/ke7/helix-evo-runner-codex@sha256:18cba771...``, which is
    correctly *shaped* but was never published. The hash was the local image
    config ID (``RepoDigests`` was empty), a different hash space from a
    registry manifest digest. It ran green here purely because the image was
    cached locally, and was unreproducible anywhere else.

    String-equality assertions cannot catch that, so this test performs the
    network resolution. It skips (never silently passes) when docker or the
    network is unavailable.
    """
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        pytest.skip("docker unavailable")

    assert "@sha256:" in RUNNER_IMAGE, "runner must be pinned by digest, not a tag"

    probe = subprocess.run(
        ["docker", "manifest", "inspect", RUNNER_IMAGE],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 and "manifest unknown" not in probe.stderr.lower():
        pytest.skip(f"registry unreachable: {probe.stderr.strip()[:120]}")
    assert probe.returncode == 0, (
        f"{RUNNER_IMAGE} does not resolve in the registry "
        f"({probe.stderr.strip()[:120]}). A digest that only exists locally is "
        "not a content-addressed pin."
    )

    # A registry manifest digest must never equal a local image config ID.
    local = subprocess.run(
        ["docker", "image", "inspect", RUNNER_IMAGE, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
    )
    if local.returncode == 0:
        assert local.stdout.strip() != _digest_of(RUNNER_IMAGE), (
            "pin equals the local image config ID, so it is a local build "
            "wearing a registry-digest costume"
        )


def test_lane_runtime_is_the_exact_current_release() -> None:
    """Criterion 2: the imported helix is THIS worktree, at 0.3.0, and the
    branch descends from the canonical 0.3.0 release line.

    This lane previously forked at 84c7bcd and re-implemented the core fixes
    locally, which left it on 0.2.1 with none of these commits as ancestors.
    """
    import subprocess

    import helix

    assert Path(helix.__file__).resolve().is_relative_to(REPO_ROOT.resolve())
    assert helix.__version__ == "0.3.0"

    for commit in RELEASE_ANCESTORS:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        assert result.returncode == 0, f"{commit} is not an ancestor of HEAD"


def test_lane_adds_no_core_source_changes_over_release_tip() -> None:
    """The lane must be purely additive: demo files only, no core src/ edits.

    A core edit here would mean a duplicated release commit was re-applied
    instead of inherited.
    """
    import subprocess

    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{RELEASE_TIP}..HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert changed, "expected the lane to add demo files"
    for path in changed:
        assert path.startswith(
            ("examples/livebench_math/", "tests/examples/test_livebench_math.py")
        ), f"lane unexpectedly modifies {path}"


def test_manifest_pins_exact_revisions_and_representative_smoke_ids() -> None:
    assert constants.GEPA_BLOG_COMMIT == "121084499247e7ddfa05ec453a53e0d644838b7a"
    assert constants.TERRARIUM_COMMIT == "e2c8b59079ed26de2d38e8aaf4ac2b4437703fe9"
    assert constants.LIVEBENCH_CODE_COMMIT == "1de6a43e82a137beeeaf2b92d683eedb67f0cf97"
    assert (
        constants.LIVEBENCH_DATA_REVISION == "bb66571c8ccf32d3df9e6f48b920d3770ff4aacb"
    )
    assert {name: len(ids) for name, ids in constants.SMOKE_IDS.items()} == {
        "train": 4,
        "val": 4,
    }
    assert len(set(constants.SMOKE_IDS["train"] + constants.SMOKE_IDS["val"])) == 8
    assert constants.PUBLICATION_PROPOSER_MODEL == "gpt-5-mini"
    assert constants.SMOKE_PROPOSER_MODEL == "haiku"
    assert constants.SMOKE_PROPOSER_BACKEND == "claude"


def test_official_dispatch_routes_all_math_families() -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def record(name: str, value: float) -> Callable[..., float]:
        def inner(*args: Any, **kwargs: Any) -> float:
            calls.append((name, args, kwargs))
            return value

        return inner

    fakes = scoring.OfficialScorers(
        mathcontest=record("contest", 1),
        aime=record("aime", 1),
        olympiad=record("olympiad", 0.5),
        amps_hard=record("amps", 1),
    )
    base = {"ground_truth": "A", "turns": ["problem"]}
    assert (
        scoring.score_livebench_math({**base, "subtask": "amc_12a_2023"}, "A", fakes)
        == 1
    )
    assert (
        scoring.score_livebench_math(
            {**base, "subtask": "updated_amc_12a_2023"}, "A", fakes
        )
        == 1
    )
    assert (
        scoring.score_livebench_math({**base, "subtask": "aime_i_2024"}, "A", fakes)
        == 1
    )
    assert scoring.score_livebench_math({**base, "subtask": "imo"}, "A", fakes) == 0.5
    assert (
        scoring.score_livebench_math({**base, "subtask": "amps_hard_gcd"}, "A", fakes)
        == 1
    )
    olympiad = next(call for call in calls if call[0] == "olympiad")
    assert olympiad[2] == {"edit_distance": True, "debug": False}


def test_official_dispatch_rejects_unknown_subtask() -> None:
    fake = scoring.OfficialScorers(*(lambda *args, **kwargs: 0 for _ in range(4)))
    with pytest.raises(ValueError, match="unroutable"):
        scoring.score_livebench_math(
            {"subtask": "unknown", "ground_truth": "x", "turns": ["q"]}, "x", fake
        )


def test_largest_remainder_is_deterministic_and_exact() -> None:
    assert dataset.largest_remainder(4, {"a": 5, "b": 3, "c": 2}) == {
        "a": 2,
        "b": 1,
        "c": 1,
    }
    with pytest.raises(ValueError):
        dataset.largest_remainder(1, {})


def test_smoke_selection_is_pinned_and_deterministic() -> None:
    rows = {
        name: [
            {
                "question_id": question_id,
                "subtask": "x",
                "turns": ["q"],
                "ground_truth": "a",
            }
            for question_id in reversed(ids)
        ]
        for name, ids in constants.SMOKE_IDS.items()
    }
    selected = dataset.select_smoke_rows(rows)
    assert {
        name: tuple(row["question_id"] for row in split_rows)
        for name, split_rows in selected.items()
    } == constants.SMOKE_IDS


def test_evaluate_adapter_preserves_repeated_and_padded_positions(
    tmp_path: Path,
) -> None:
    (tmp_path / "prompt.txt").write_text("candidate")
    (tmp_path / "helix_batch.json").write_text('["0", "0", "3"]')
    assert evaluate.build_request(tmp_path, "train") == {
        "prompt": "candidate",
        "split": "train",
        "ids": ["0", "0", "3"],
    }


def test_evaluate_adapter_rejects_malformed_ids_and_split(tmp_path: Path) -> None:
    (tmp_path / "prompt.txt").write_text("candidate")
    (tmp_path / "helix_batch.json").write_text("[0]")
    with pytest.raises(ValueError, match="string IDs"):
        evaluate.build_request(tmp_path, "train")
    (tmp_path / "helix_batch.json").write_text('["0"]')
    with pytest.raises(ValueError, match="train or val"):
        evaluate.build_request(tmp_path, "test")


def test_client_timeout_is_bounded_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, float] = {}

    def timeout(request: Any, timeout: float) -> NoReturn:
        observed["timeout"] = timeout
        raise TimeoutError("bounded evaluator timeout")

    monkeypatch.setattr(urllib.request, "urlopen", timeout)
    with pytest.raises(TimeoutError, match="bounded evaluator timeout"):
        evaluate.run_client(
            "http://evaluator.invalid/evaluate",
            {"prompt": "p", "split": "val", "ids": ["0"]},
            timeout=3.5,
        )
    assert observed == {"timeout": 3.5}


def test_client_rejects_wrong_result_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return b'{"results":[]}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="cardinality"):
        evaluate.run_client(
            "http://evaluator.invalid/evaluate",
            {"prompt": "p", "split": "val", "ids": ["0"]},
        )


def test_parallel_and_baseline_configs_pin_expected_shapes() -> None:
    import tomllib

    p2n2 = tomllib.loads((ROOT / "helix.toml").read_text())
    one = tomllib.loads((ROOT / "helix.1x1.toml").read_text())
    assert (
        p2n2["evolution"]["num_parallel_proposals"],
        p2n2["evolution"]["mutations_per_parent"],
    ) == (2, 2)
    assert (
        one["evolution"]["num_parallel_proposals"],
        one["evolution"]["mutations_per_parent"],
    ) == (1, 1)
    assert p2n2["agent"]["model"] == constants.SMOKE_PROPOSER_MODEL
    assert one["agent"]["model"] == constants.SMOKE_PROPOSER_MODEL
    assert p2n2["sandbox"]["enabled"] and p2n2["sandbox"]["evaluator"]
    assert p2n2["sandbox"]["image"] == RUNNER_IMAGE
    assert one["sandbox"]["image"] == RUNNER_IMAGE
    assert (ROOT / "Dockerfile").read_text().splitlines()[0] == (f"FROM {RUNNER_IMAGE}")
    assert p2n2["evaluator"]["sidecar"]["image"].startswith("helix-livebench-math:")
    assert p2n2["evaluator"]["sidecar"]["runner_image"].startswith(
        "helix-livebench-math:"
    )
    assert p2n2["sandbox"]["image"] != p2n2["evaluator"]["sidecar"]["image"]
    assert "passthrough_env" not in p2n2
    assert p2n2["evaluator"]["sidecar"]["passthrough_env"] == ["OPENAI_API_KEY"]
    assert "passthrough_env" not in one
    assert one["evaluator"]["sidecar"]["passthrough_env"] == ["OPENAI_API_KEY"]
    assert ".gitignore" not in p2n2["evaluator"]["protected_files"]
    assert ".gitignore" not in one["evaluator"]["protected_files"]


def test_agent_command_environment_excludes_sidecar_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-agent-must-not-see-this")
    config = load_config(ROOT / "helix.toml")
    agent_env = _scrub_environment(
        passthrough_env=config.passthrough_env,
        fixed_env=config.env,
    )
    assert config.passthrough_env == []
    assert config.evaluator.sidecar is not None
    assert config.evaluator.sidecar.passthrough_env == ["OPENAI_API_KEY"]
    assert "OPENAI_API_KEY" not in agent_env


def _agent_env_names_in_docker_argv(config: Any) -> list[str]:
    """Env var NAMES on the real agent docker argv, via the production path."""
    from helix.mutator import _add_backend_auth_env
    from helix.sandbox import _docker_args

    env = _scrub_environment(
        passthrough_env=config.passthrough_env, fixed_env=config.env
    )
    # The production sequence re-adds backend auth AFTER scrubbing, so a test
    # that stops at the scrubber stops one step too early.
    _add_backend_auth_env(env, config.agent.backend)
    argv = _docker_args(
        ["true"],
        env,
        Path("/tmp"),
        config.sandbox,
        "agent",
        config.sandbox.image,
        config.agent.backend,
        None,
        container_name="test-probe",
    )
    return [
        arg.split("=", 1)[0]
        for index, arg in enumerate(argv)
        if index > 0 and argv[index - 1] == "-e"
    ]


def test_sidecar_key_never_reaches_agent_docker_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The solver credential must be absent from the ACTUAL docker argv.

    Asserting only on ``_scrub_environment`` is insufficient: HELIX re-adds
    backend auth env via ``_add_backend_auth_env`` after scrubbing, so a key
    can be absent at the scrubber and still be handed to the container. This
    test therefore asserts on the final argv.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-agent-must-not-see-this")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-backend-auth")
    config = load_config(ROOT / "helix.toml")
    names = _agent_env_names_in_docker_argv(config)

    # The whole point of core fix 94f9751: the solver key is sidecar-only.
    assert "OPENAI_API_KEY" not in names
    assert config.evaluator.sidecar is not None
    assert config.evaluator.sidecar.passthrough_env == ["OPENAI_API_KEY"]


def test_backend_choice_is_what_keeps_the_solver_key_out_of_the_agent() -> None:
    """Guard the assumption the sidecar boundary silently depends on.

    ``BACKEND_AUTH_ENV`` re-adds each backend's auth vars after scrubbing.
    ``opencode`` lists OPENAI_API_KEY, so switching this demo to that backend
    would hand the solver credential straight to the mutation agent and
    silently defeat the sidecar isolation, with no config looking wrong.
    """
    from helix.backends import BACKEND_AUTH_ENV

    config = load_config(ROOT / "helix.toml")
    assert config.agent.backend == "claude"
    assert "OPENAI_API_KEY" not in BACKEND_AUTH_ENV.get(config.agent.backend, ())
    # Documents the hazard rather than asserting the whole table is safe.
    assert "OPENAI_API_KEY" in BACKEND_AUTH_ENV["opencode"], (
        "if this changes, revisit whether the backend allowlist still governs "
        "sidecar credential isolation"
    )


def test_agent_image_and_evaluator_image_are_separate_trust_domains() -> None:
    config = load_config(ROOT / "helix.toml")
    assert config.evaluator.sidecar is not None
    assert config.sandbox.image == RUNNER_IMAGE
    assert config.evaluator.sidecar.image == "helix-livebench-math:e2c8b590-smoke"
    assert config.evaluator.sidecar.resolved_runner_image == (
        "helix-livebench-math:e2c8b590-smoke"
    )
    assert config.sandbox.image != config.evaluator.sidecar.image


def test_agent_snapshot_omits_every_non_prompt_demo_file(tmp_path: Path) -> None:
    config = load_config(ROOT / "helix.toml")
    omitted = {Path(item) for item in config.sandbox.omit_from_agent}
    # HELIX-managed runtime state (.git from the dirty-seed snapshot repo,
    # .helix*, helix_batch.json) is created by a run and stripped by
    # _ignore_for_copy in core, not by omit_from_agent. Excluding it here keeps
    # the assertion about DEMO files; the snapshot assertion below is what
    # actually proves nothing extra reaches the agent, and it is unconditional.
    runtime_state = {".git", ".helix", "helix_batch.json"}
    source_entries = {
        path.name
        for path in ROOT.iterdir()
        if path.name not in runtime_state and not path.name.startswith(".helix")
    }
    assert source_entries - {"prompt.txt"} <= {path.as_posix() for path in omitted}
    assert Path("helix.toml") in omitted
    assert Path("constants.py") in omitted
    assert Path("server.py") in omitted
    assert Path("prompt.txt") not in omitted

    snapshot = tmp_path / "agent-snapshot"
    _copy_tree_contents(ROOT, snapshot, omit_paths=omitted)
    assert {path.name for path in snapshot.iterdir()} == {"prompt.txt"}
    assert "helix-evaluator:8080" not in "\n".join(
        path.read_text() for path in snapshot.iterdir() if path.is_file()
    )


def test_sidecar_errors_are_zero_scored_and_secrets_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_text

    def fake_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if str(path) == "/opt/livebench-math/data.json":
            return json.dumps(
                {
                    "dataset_revision": constants.LIVEBENCH_DATA_REVISION,
                    "smoke_splits": {"train": [], "val": []},
                }
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-12345678")
    spec = importlib.util.spec_from_file_location(
        "livebench_math_server_errors", ROOT / "server.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {
        "question_id": "q",
        "subtask": "aime_i_2024",
        "ground_truth": "1",
        "turns": ["q"],
    }

    def fail(prompt: str, example: dict[str, Any]) -> NoReturn:
        raise TimeoutError("request exposed sk-test-secret-12345678")

    result = module.evaluate_one("prompt", row, fail)
    rendered = json.dumps(result)
    assert result[0] == 0.0
    assert "sk-test-secret" not in rendered
    assert "<redacted>" in rendered
    assert "OPENAI_API_KEY" not in __import__("os").environ


def test_ground_truth_never_enters_side_info_or_mutation_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "SENTINEL_GROUND_TRUTH_MUST_NOT_LEAK"
    original = Path.read_text

    def fake_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if str(path) == "/opt/livebench-math/data.json":
            return json.dumps(
                {
                    "dataset_revision": constants.LIVEBENCH_DATA_REVISION,
                    "smoke_splits": {"train": [], "val": []},
                }
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read)
    spec = importlib.util.spec_from_file_location(
        "livebench_math_server_no_leak", ROOT / "server.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "score_livebench_math", lambda row, answer: 0.0)
    row = {
        "question_id": "q",
        "subtask": "aime_i_2024",
        "ground_truth": sentinel,
        "turns": ["question"],
    }
    result = module.evaluate_one("prompt", row, lambda prompt, example: ("wrong", {}))
    serialized = json.dumps(result)
    assert sentinel not in serialized
    assert "expected" not in serialized.lower()

    evaluation = EvalResult(
        candidate_id="candidate",
        scores={"success": 0.0},
        asi={},
        instance_scores={"0": 0.0},
        per_example_side_info=[result[1]],
    )
    reflection_prompt = build_mutation_prompt("Improve prompt.txt", evaluation)
    assert sentinel not in reflection_prompt
    assert "expected" not in reflection_prompt.lower()
    assert "Official score 0.000" in reflection_prompt


def test_sidecar_preserves_duplicate_positions_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_text

    def fake_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if str(path) == "/opt/livebench-math/data.json":
            return json.dumps(
                {
                    "dataset_revision": constants.LIVEBENCH_DATA_REVISION,
                    "smoke_splits": {"train": [], "val": []},
                }
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read)
    spec = importlib.util.spec_from_file_location(
        "livebench_math_server_order", ROOT / "server.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {
            "question_id": f"q{index}",
            "subtask": "aime_i_2024",
            "ground_truth": str(index),
            "turns": [f"question {index}"],
        }
        for index in range(4)
    ]
    data = {
        "dataset_revision": constants.LIVEBENCH_DATA_REVISION,
        "smoke_splits": {"train": rows, "val": rows},
    }
    monkeypatch.setattr(module, "score_livebench_math", lambda row, answer: 1.0)
    results = module.evaluate_request(
        {"prompt": "p", "split": "train", "ids": ["0", "0", "3"]},
        data=data,
        solver=lambda prompt, row: (row["ground_truth"], {}),
    )
    assert [result[1]["question_id"] for result in results] == ["q0", "q0", "q3"]


def _completed_p2n2_state() -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for index in range(4):
        group, mutation = divmod(index, 2)
        tasks.append(
            {
                "batch_id": "g1-b1",
                "p": 2,
                "n": 2,
                "task_index": index,
                "parent_group": group,
                "mutation_index": mutation,
                "parent_id": f"parent-{group}",
                "child_id": f"child-{index}",
                "status": "applied" if index == 0 else "rejected",
                "selection": "selected" if index == 0 else "not_selected",
                "cleanup": "not_required" if index == 0 else "removed",
                "budget_accounted": True,
                "budget_charge": {"evaluations": 2},
            }
        )
    return {
        "schema_version": 4,
        "generation": 1,
        "frontier": ["seed", "child-0"],
        "active_frontier": {"0": ["child-0"]},
        "instance_scores": {"seed": {"0": 0.0}, "child-0": {"0": 1.0}},
        "budget": {"evaluations": 12, "input_tokens": 10, "output_tokens": 4},
        "proposal_batches": [
            {
                "batch_id": "g1-b1",
                "p": 2,
                "n": 2,
                "phase": "complete",
                "budget_before_dispatch": 4,
                "budget_after_apply": 12,
                "tasks": tasks,
            }
        ],
        "scheduler_state": {"phase": "complete"},
    }


def test_state_inspection_is_resume_stable_and_conserves_accounting() -> None:
    state = _completed_p2n2_state()
    before = inspect_run.audit_state(state, require_terminal=True)
    after = inspect_run.audit_state(
        json.loads(json.dumps(state)), require_terminal=True
    )
    assert before == after
    assert before["candidate_ids"] == ["child-0", "child-1", "child-2", "child-3"]
    assert before["ledger_evaluations"] == 8
    assert before["nonproposal_evaluations"] == 4
    assert before["budget_conserved"] is True


def test_state_inspection_rejects_duplicate_candidate_ids() -> None:
    state = _completed_p2n2_state()
    state["proposal_batches"][0]["tasks"][1]["child_id"] = "child-0"
    with pytest.raises(ValueError, match="globally distinct"):
        inspect_run.audit_state(state, require_terminal=True)


def test_release_inspection_rejects_1x1_shape() -> None:
    state = _completed_p2n2_state()
    batch = state["proposal_batches"][0]
    batch["p"] = batch["n"] = 1
    batch["tasks"] = batch["tasks"][:1]
    batch["tasks"][0]["p"] = batch["tasks"][0]["n"] = 1
    batch["budget_after_apply"] = 6
    state["budget"]["evaluations"] = 6
    with pytest.raises(ValueError, match="P=2,N=2"):
        inspect_run.audit_state(state, require_terminal=True)
    summary = inspect_run.audit_state(state)
    assert summary["batches"][0]["p"] == 1
    assert summary["batches"][0]["n"] == 1


def test_release_inspection_rejects_interrupted_batch() -> None:
    state = _completed_p2n2_state()
    state["proposal_batches"][0]["phase"] = "interrupted"
    with pytest.raises(ValueError, match="phase exactly complete"):
        inspect_run.audit_state(state, require_terminal=True)


def test_release_inspection_rejects_pending_selection() -> None:
    state = _completed_p2n2_state()
    state["proposal_batches"][0]["tasks"][2]["selection"] = "pending"
    with pytest.raises(ValueError, match="terminal and fully accounted"):
        inspect_run.audit_state(state, require_terminal=True)


def test_release_inspection_rejects_failed_cleanup() -> None:
    state = _completed_p2n2_state()
    state["proposal_batches"][0]["tasks"][2]["cleanup"] = "failed"
    with pytest.raises(ValueError, match="terminal and fully accounted"):
        inspect_run.audit_state(state, require_terminal=True)


def test_release_inspection_rejects_accounting_mismatch() -> None:
    state = _completed_p2n2_state()
    state["proposal_batches"][0]["tasks"][2]["budget_charge"]["evaluations"] = 3
    with pytest.raises(ValueError, match="does not conserve"):
        inspect_run.audit_state(state, require_terminal=True)


def test_release_inspection_rejects_unexplained_post_batch_spend() -> None:
    state = _completed_p2n2_state()
    state["budget"]["evaluations"] = 13
    with pytest.raises(ValueError, match="unexplained post-batch spend"):
        inspect_run.audit_state(state, require_terminal=True)
