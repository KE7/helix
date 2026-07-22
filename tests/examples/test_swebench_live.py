"""Focused contract tests for the SWE-bench-Live example."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "swebench_live"
sys.path.insert(0, str(EXAMPLE))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upstream_pins_are_immutable_and_digest_qualified() -> None:
    pins = _load("swebench_live_pins", "pins.py")
    assert pins.HARNESS_COMMIT == "70ec57e852e3f2d195790fe71f553e272c691833"
    assert pins.DATASET_REVISION == "608f7ae9ab8ea1f9f0d030fe04562cf6bd1a0c8b"
    assert pins.TASK_ID == "capstone-engine__capstone-2743"
    assert pins.OFFICIAL_IMAGE_PLATFORM == "linux/amd64"
    assert pins.OFFICIAL_IMAGE.endswith(
        "@sha256:c3d6222106db9afce1eaf6036f67d540011e46ea8e59419097c32d0555032ed9"
    )


def test_parallel_config_loads_with_supported_schema_and_measured_limits() -> None:
    from helix.config import load_config

    config = load_config(EXAMPLE / "helix.toml")
    assert config.evolution.num_parallel_proposals == 2
    assert config.evolution.mutations_per_parent == 2
    assert config.evolution.max_workers == 2
    raw = tomllib.loads((EXAMPLE / "helix.toml").read_text())
    assert "allowed_tools" not in raw["agent"]
    assert config.sandbox.enabled is True
    assert config.sandbox.evaluator is False
    # The runner must be pinned by a REGISTRY digest, not a bare local image
    # ID.  A bare "sha256:..." resolves only on the host that built it, so the
    # demo would be unreproducible elsewhere and would die at dispatch (HELIX
    # never runs `docker pull`) on any machine lacking that local layer.
    assert config.sandbox.image == (
        "ghcr.io/ke7/helix-evo-runner-claude"
        "@sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba"
    )
    assert not config.sandbox.image.startswith("sha256:")
    repository, _, digest = config.sandbox.image.partition("@")
    assert "/" in repository, "runner pin must be registry-qualified"
    assert digest.startswith("sha256:") and len(digest) == 71
    assert config.sandbox.cpus == 2.0
    assert config.sandbox.memory == "2g"
    assert config.sandbox.pids_limit == 256
    assert config.sandbox.timeout_seconds == 900
    evaluator_source = (EXAMPLE / "evaluate.py").read_text()
    assert '"--memory",\n        "4g"' in evaluator_source


def test_mutation_sandbox_exposes_only_agent_and_public_contract(
    tmp_path: Path,
) -> None:
    from helix.config import load_config
    from helix.sandbox import _copy_tree_contents, _docker_args

    config = load_config(EXAMPLE / "helix.toml")
    omitted = {Path(item) for item in config.sandbox.omit_from_agent}
    assert Path("coding_agent.py") not in omitted
    assert Path("TASK.md") not in omitted
    for sensitive in (
        "prepare.py",
        "evaluate.py",
        "official_runner.py",
        "pins.py",
        "helix.toml",
        "cleanup.py",
        "SOURCE.md",
    ):
        assert Path(sensitive) in omitted

    workspace = tmp_path / "workspace"
    _copy_tree_contents(
        EXAMPLE,
        workspace,
        skip_special_files=config.sandbox.skip_special_files,
        omit_paths=omitted,
    )
    visible = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    assert visible == {"TASK.md", "coding_agent.py"}

    args = _docker_args(
        ["claude", "--version"],
        {},
        workspace,
        config.sandbox,
        "agent",
        config.sandbox.image or "",
        "claude",
        # Agent scope requires provenance grants; an empty environment has an
        # empty (but present) grant list.
        grants=[],
    )
    mounts = [args[index + 1] for index, item in enumerate(args[:-1]) if item == "-v"]
    assert any(mount.endswith(":/workspace:rw") for mount in mounts)
    assert "helix-auth-claude:/home/node:rw" in mounts
    assert all("docker.sock" not in mount for mount in mounts)
    assert all("swebench-live-capstone-2743-private" not in mount for mount in mounts)
    assert args[args.index("--cpus") + 1] == "2.0"
    assert args[args.index("--memory") + 1] == "2g"
    assert args[args.index("--pids-limit") + 1] == "256"


def test_task_row_validation_rejects_drift_without_echoing_private_values() -> None:
    pins = _load("swebench_live_pins_validation", "pins.py")
    row = {
        "instance_id": pins.TASK_ID,
        "repo": pins.TASK_REPOSITORY,
        "base_commit": pins.TASK_BASE_COMMIT,
        "docker_image": pins.OFFICIAL_IMAGE_REPOSITORY.removeprefix("docker.io/"),
        "FAIL_TO_PASS": list(pins.EXPECTED_FAIL_TO_PASS),
        "PASS_TO_PASS": list(pins.EXPECTED_PASS_TO_PASS),
        "rebuild_cmds": list(pins.EXPECTED_REBUILD_COMMANDS),
        "test_cmds": list(pins.EXPECTED_TEST_COMMANDS),
        "print_cmds": list(pins.EXPECTED_PRINT_COMMANDS),
        "patch": "secret-gold-patch",
        "test_patch": "secret-test-patch",
        "problem_statement": "public issue",
        "log_parser": "def parser(log): return {}",
    }
    pins.validate_task_row(row)
    row["base_commit"] = "wrong-secret-value"
    with pytest.raises(pins.PinMismatch) as exc_info:
        pins.validate_task_row(row)
    assert str(exc_info.value) == "task row differs at: base_commit"
    assert "wrong-secret-value" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("status", "resolved"),
    [
        ({"IssueTests": "pass"}, True),
        ({}, False),
        ({"IssueTests": "fail"}, False),
        ({"IssueTests": "pass", "unit_utils": "fail"}, False),
        ({"IssueTests": "pass", "unscored_extra": "fail"}, True),
    ],
)
def test_official_resolution_semantics_are_not_replaced_by_auxiliary_scores(
    status: dict[str, str], resolved: bool
) -> None:
    pins = _load("swebench_live_pins_resolution", "pins.py")
    runner = _load("swebench_live_runner_resolution", "official_runner.py")
    task = {
        "instance_id": pins.TASK_ID,
        "FAIL_TO_PASS": list(pins.EXPECTED_FAIL_TO_PASS),
        "PASS_TO_PASS": list(pins.EXPECTED_PASS_TO_PASS),
    }
    report = runner.official_resolution(status, task)
    assert report["resolved"] is resolved


def test_official_ctest_parser_handles_repeated_and_padded_log_lines() -> None:
    runner = _load("swebench_live_runner_parser", "official_runner.py")
    source = """
def parser(log):
    import re
    results = {}
    for line in log.splitlines():
        m = re.match(r"\\s*\\d+/\\d+\\s+Test\\s+#?\\d+:\\s+(\\S+)\\s+\\.+\\s+(Passed|Failed|Skipped)", line, re.I)
        if m:
            name, state = m.groups()
            results[name] = state.lower()
    return results
"""
    status = runner._parse_status(
        source,
        "noise\n 1/13 Test #1: IssueTests ........ Passed\n"
        "padded noise\n 1/13 Test #1: IssueTests ........ Passed\n",
    )
    assert status == {"IssueTests": "passed"}


def test_diagnostics_redact_secrets_and_bearer_tokens() -> None:
    evaluator = _load("swebench_live_evaluator_redaction", "evaluate.py")
    text = evaluator.redact_diagnostic(
        "token=top-secret password=hunter2 Authorization: Bearer abc.def"
    )
    assert "top-secret" not in text
    assert "hunter2" not in text
    assert "abc.def" not in text
    assert text.count("[REDACTED]") == 3


def test_container_names_are_distinct_and_cleanup_runs_on_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _load("swebench_live_evaluator_isolation", "evaluate.py")
    (tmp_path / "official_runner.py").write_text("# runner\n")
    (tmp_path / "coding_agent.py").write_text("# agent\n")
    created_names: list[str] = []
    removed_names: list[str] = []

    def fake_docker(*args: str, timeout: int = 60):
        del timeout
        if args[0] == "create":
            name = args[args.index("--name") + 1]
            created_names.append(name)
            return SimpleNamespace(stdout="container-id\n")
        if args[0] == "start":
            return SimpleNamespace(stdout="not-json\n")
        if args[0] == "inspect":
            return SimpleNamespace(stdout='[{"State":{"ExitCode":0}}]')
        return SimpleNamespace(stdout="[]")

    def fake_run(args, **kwargs):
        del kwargs
        if args[:3] == ["docker", "rm", "--force"]:
            removed_names.append(args[3])
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(evaluator, "_docker", fake_docker)
    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
    for _ in range(2):
        with pytest.raises(json.JSONDecodeError):
            evaluator.run_container(tmp_path)
    assert len(created_names) == len(set(created_names)) == 2
    assert removed_names == created_names


def test_timeout_is_zero_scored_and_secret_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = _load("swebench_live_evaluator_timeout", "evaluate.py")
    monkeypatch.setattr(
        evaluator,
        "run_container",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("token=never-print-me", 3)
        ),
    )
    monkeypatch.setattr(evaluator, "original_project_root", lambda _cwd: tmp_path)
    result = evaluator.evaluation_result(tmp_path)
    assert result["accuracy"] == 0.0
    rendered = json.dumps(result)
    assert "never-print-me" not in rendered
    artifacts = list((tmp_path / "artifacts" / "evaluations").glob("*.json"))
    assert len(artifacts) == 1


def test_artifact_root_falls_back_before_helix_initializes_git(tmp_path: Path) -> None:
    evaluator = _load("swebench_live_evaluator_no_git", "evaluate.py")
    assert not (tmp_path / ".git").exists()
    assert evaluator.original_project_root(tmp_path) == tmp_path


def _current_state() -> dict:
    tasks = [
        {
            "batch_id": "g1-b0",
            "p": 2,
            "n": 2,
            "task_index": index,
            "parent_group": index // 2,
            "mutation_index": index % 2,
            "parent_id": "g0-s0",
            "child_id": f"g1-s{index + 1}",
            "status": "applied" if index == 0 else "rejected",
            "score_delta": 1.0 if index == 0 else 0.0,
            "selection": "selected" if index == 0 else "not_selected",
            "cleanup": "not_required" if index == 0 else "removed",
            "budget_charge": {
                "evaluations": 1,
                "input_tokens": 100,
                "output_tokens": 10,
                "cached_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.001,
            },
            "budget_accounted": True,
            "applied": index == 0,
            "detail": None,
        }
        for index in range(4)
    ]
    return {
        "schema_version": 4,
        "generation": 1,
        "frontier": ["g0-s0", "g1-s1"],
        "active_frontier": {"instance": ["g1-s1"]},
        "budget": {
            "evaluations": 5,
            "input_tokens": 400,
            "output_tokens": 40,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.004,
        },
        "mutation_counter": 4,
        "proposal_batches": [
            {
                "batch_id": "g1-b0",
                "generation": 1,
                "p": 2,
                "n": 2,
                "phase": "complete",
                "budget_before_dispatch": 1,
                "budget_state_before_dispatch": {"evaluations": 1},
                "max_evaluations": 9,
                "max_in_flight_evaluations": 4,
                "maximum_overshoot": 3,
                "budget_after_apply": 5,
                "tasks": tasks,
            }
        ],
    }


def _write_state(project: Path, state: dict) -> None:
    helix_dir = project / ".helix"
    helix_dir.mkdir(exist_ok=True)
    (helix_dir / "state.json").write_text(json.dumps(state))


def test_inspection_reports_p_by_n_ids_and_durable_budget(tmp_path: Path) -> None:
    inspect_run = _load("swebench_live_inspect", "inspect_run.py")
    state = _current_state()
    _write_state(tmp_path, state)
    summary = inspect_run.summarize(tmp_path)
    assert summary["candidate_ids"] == ["g1-s1", "g1-s2", "g1-s3", "g1-s4"]
    assert summary["candidate_ids_distinct"] is True
    assert summary["candidate_ids_parent_major"] is True
    assert summary["budget_conserved"] is True
    assert summary["accounting"] == {
        "global_evaluations": 5,
        "proposal_evaluations": 4,
        "nonproposal_evaluations": 1,
    }


def test_inspection_rejects_obsolete_task_keys(tmp_path: Path) -> None:
    inspect_run = _load("swebench_live_inspect_old", "inspect_run.py")
    state = _current_state()
    task = state["proposal_batches"][0]["tasks"][0]
    task["reserved_child_id"] = task.pop("child_id")
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="obsolete key"):
        inspect_run.summarize(tmp_path)


def test_inspection_rejects_missing_current_key(tmp_path: Path) -> None:
    inspect_run = _load("swebench_live_inspect_missing", "inspect_run.py")
    state = _current_state()
    del state["proposal_batches"][0]["tasks"][0]["budget_charge"]
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="missing required key"):
        inspect_run.summarize(tmp_path)


def test_inspection_rejects_duplicate_or_non_parent_major_ids(tmp_path: Path) -> None:
    inspect_run = _load("swebench_live_inspect_duplicate", "inspect_run.py")
    state = _current_state()
    state["proposal_batches"][0]["tasks"][1]["child_id"] = "g1-s1"
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="duplicate child_id"):
        inspect_run.summarize(tmp_path)

    state = _current_state()
    state["proposal_batches"][0]["tasks"][2]["parent_group"] = 0
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="not parent-major"):
        inspect_run.summarize(tmp_path)


def test_inspection_rejects_nonterminal_task_or_batch(tmp_path: Path) -> None:
    inspect_run = _load("swebench_live_inspect_terminal", "inspect_run.py")
    state = _current_state()
    state["proposal_batches"][0]["tasks"][3]["status"] = "running"
    state["proposal_batches"][0]["tasks"][3]["cleanup"] = "pending"
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="status is not terminal"):
        inspect_run.summarize(tmp_path)

    state = _current_state()
    state["proposal_batches"][0]["phase"] = "applying"
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="phase must equal 'complete'"):
        inspect_run.summarize(tmp_path)


def test_inspection_rejects_pending_selection_or_failed_cleanup(tmp_path: Path) -> None:
    inspect_run = _load("swebench_live_inspect_completion", "inspect_run.py")
    state = _current_state()
    state["proposal_batches"][0]["tasks"][0]["selection"] = "pending"
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="selection is pending"):
        inspect_run.summarize(tmp_path)

    state = _current_state()
    state["proposal_batches"][0]["tasks"][1]["cleanup"] = "failed"
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="cleanup failed"):
        inspect_run.summarize(tmp_path)


def test_inspection_rejects_wrong_generation_or_task_count(tmp_path: Path) -> None:
    inspect_run = _load("swebench_live_inspect_generation", "inspect_run.py")
    state = _current_state()
    state["generation"] = 2
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="generation must equal 1"):
        inspect_run.summarize(tmp_path)

    state = _current_state()
    state["proposal_batches"][0]["tasks"].pop()
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="exactly 4 entries"):
        inspect_run.summarize(tmp_path)


def test_inspection_rejects_unaccounted_or_mismatched_budget(tmp_path: Path) -> None:
    inspect_run = _load("swebench_live_inspect_accounting", "inspect_run.py")
    state = _current_state()
    state["proposal_batches"][0]["tasks"][1]["budget_accounted"] = False
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="budget_accounted"):
        inspect_run.summarize(tmp_path)

    state = _current_state()
    state["proposal_batches"][0]["tasks"][1]["budget_charge"]["evaluations"] = 0
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="does not equal budget delta"):
        inspect_run.summarize(tmp_path)

    state = _current_state()
    state["budget"]["evaluations"] = 3
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="exceed the global"):
        inspect_run.summarize(tmp_path)

    state = _current_state()
    state["budget"]["evaluations"] = 6
    _write_state(tmp_path, state)
    with pytest.raises(inspect_run.InspectionError, match="does not equal the global"):
        inspect_run.summarize(tmp_path)


def test_cleanup_removes_digest_tag_and_exact_image_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup = _load("swebench_live_cleanup", "cleanup.py")
    baseline_id = "sha256:exact-task-image-id"
    image_ids = iter([baseline_id, None])
    monkeypatch.setattr(cleanup, "image_id", lambda: next(image_ids))
    inspect_counts: dict[str, int] = {}

    def fake_inspect(reference: str) -> str | None:
        count = inspect_counts.get(reference, 0)
        inspect_counts[reference] = count + 1
        latest = f"{cleanup.OFFICIAL_IMAGE_REPOSITORY}:latest"
        if reference == latest and count == 0:
            return baseline_id
        return None

    monkeypatch.setattr(cleanup, "inspect_image", fake_inspect)
    output_calls: list[list[str]] = []
    ps_calls = 0

    def fake_output(args: list[str]) -> str:
        nonlocal ps_calls
        output_calls.append(args)
        if args[:3] == ["docker", "ps", "-aq"]:
            ps_calls += 1
            return "container-exact-id" if ps_calls == 1 else ""
        return ""

    monkeypatch.setattr(cleanup, "_output", fake_output)
    volume_inspections = iter(
        [
            subprocess.CompletedProcess(
                ["docker"],
                0,
                json.dumps(
                    [
                        {
                            "Name": cleanup.PRIVATE_VOLUME,
                            "Labels": {
                                "com.helix.demo": "swebench-live-capstone-2743"
                            },
                        }
                    ]
                ),
                "",
            ),
            subprocess.CompletedProcess(["docker"], 1, "", "missing"),
        ]
    )
    monkeypatch.setattr(cleanup.subprocess, "run", lambda *_args, **_kwargs: next(volume_inspections))
    report = cleanup.cleanup(remove_image=True)
    assert report["image_id_before"] == baseline_id
    assert report["image_id_after"] is None
    assert report["latest_tag_id_after"] is None
    assert report["private_volume_name_before"] == cleanup.PRIVATE_VOLUME
    assert report["private_volume_exists"] is False
    assert report["remaining_labeled_containers"] == []
    assert ["docker", "rm", "--force", "container-exact-id"] in output_calls
    assert ["docker", "image", "rm", cleanup.OFFICIAL_IMAGE] in output_calls
    assert [
        "docker",
        "image",
        "rm",
        f"{cleanup.OFFICIAL_IMAGE_REPOSITORY}:latest",
    ] in output_calls


def test_patch_staging_does_not_follow_candidate_planted_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The candidate shares /tmp with the root runner and may plant a symlink.

    Staging the solution patch at a fixed, candidate-writable path would let an
    unprivileged candidate redirect a root-owned write to an arbitrary file.
    """
    runner = _load("swebench_live_runner_staging", "official_runner.py")

    victim = tmp_path / "victim.txt"
    victim.write_text("untouched\n", encoding="utf-8")
    planted = Path("/tmp/helix-solution.patch")
    if planted.exists() or planted.is_symlink():
        planted.unlink()
    planted.symlink_to(victim)

    staged: list[Path] = []

    def _fake_git(*args: str, timeout: int = 60):
        staged.append(Path(args[-1]))
        return subprocess.CompletedProcess(["git", *args], 0, "", "")

    monkeypatch.setattr(runner, "_git", _fake_git)
    try:
        assert runner._apply_patch("SOLUTION-PATCH-BODY") is True
    finally:
        if planted.is_symlink() or planted.exists():
            planted.unlink()

    # The root write must not have followed the planted symlink.
    assert victim.read_text(encoding="utf-8") == "untouched\n"
    assert len(staged) == 1
    assert staged[0] != planted
    # Staging is per-call and cleaned up, leaving no candidate-reachable residue.
    assert not staged[0].exists()
    assert not staged[0].parent.exists()


def test_candidate_clone_cannot_reach_commits_past_base(tmp_path: Path) -> None:
    """The official task image's /testbed carries upstream history past
    base_commit, including the commit that fixes the task instance.  Cloning it
    with full refs hands the candidate the gold patch, so the runner must clone
    shallowly and must fail closed if anything beyond base_commit is reachable.
    """
    runner_source = (EXAMPLE / "official_runner.py").read_text(encoding="utf-8")
    assert "--depth" in runner_source and "--no-tags" in runner_source
    assert "--no-hardlinks" not in runner_source
    assert "exposes commits beyond base_commit" in runner_source

    def _git(*args: str, cwd: Path) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
        ).stdout.strip()

    testbed = tmp_path / "testbed"
    testbed.mkdir()
    _git("init", "-q", ".", cwd=testbed)
    _git("config", "user.email", "t@t", cwd=testbed)
    _git("config", "user.name", "t", cwd=testbed)
    (testbed / "src.c").write_text("buggy\n", encoding="utf-8")
    _git("add", ".", cwd=testbed)
    _git("commit", "-qm", "base", cwd=testbed)
    base = _git("rev-parse", "HEAD", cwd=testbed)
    (testbed / "src.c").write_text("GOLD ANSWER\n", encoding="utf-8")
    _git("commit", "-qam", "Handle zero case (#2743)", cwd=testbed)
    _git("branch", "upstream-fix", cwd=testbed)
    _git("checkout", "-q", base, cwd=testbed)

    full = tmp_path / "full"
    _git("clone", "--quiet", "--no-hardlinks", str(testbed), str(full), cwd=tmp_path)
    leaked = _git("log", "--all", "--oneline", "--not", "HEAD", cwd=full)
    assert "#2743" in leaked, "fixture must reproduce the leak the runner had"

    shallow = tmp_path / "shallow"
    _git(
        "clone", "--quiet", "--depth", "1", "--no-tags",
        f"file://{testbed}", str(shallow), cwd=tmp_path,
    )
    assert _git("rev-parse", "HEAD", cwd=shallow) == base
    assert _git("log", "--all", "--oneline", "--not", "HEAD", cwd=shallow) == ""
    assert "GOLD ANSWER" not in _git("log", "--all", "-p", cwd=shallow)
    (shallow / "src.c").write_text("candidate fix\n", encoding="utf-8")
    assert "candidate fix" in _git("diff", "--binary", "--text", "HEAD", cwd=shallow)


def test_no_credentialed_sidecar_so_passthrough_env_must_be_empty() -> None:
    """(a)+(b) Scope guard, asserted for THIS lane's reason.

    This lane's evaluator is host-side with no credentialed sidecar, so no
    credential should be forwarded anywhere.  This is NOT a universal rule:
    a lane with a protected evaluator sidecar legitimately sets
    ``passthrough_env`` so the sidecar can receive a key while the mutation
    agent cannot.  Assert the effective PARSED values, not raw TOML keys --
    ``passthrough_env`` is a top-level HelixConfig field (and separately an
    EvaluatorSidecarConfig field); it is not a SandboxConfig field, so a
    ``[sandbox]`` key of that name is a pydantic error and asserting its
    absence there would guard an impossible condition.
    """
    from helix.config import HelixConfig, SandboxConfig, load_config

    assert "passthrough_env" in HelixConfig.model_fields
    assert "passthrough_env" not in SandboxConfig.model_fields

    config = load_config(EXAMPLE / "helix.toml")
    assert config.sandbox.evaluator is False, "this lane evaluates on the host"
    assert config.evaluator.sidecar is None, "no credentialed sidecar on this lane"
    assert config.passthrough_env == []
    assert config.env == {}


def test_agent_container_argv_carries_no_credential_when_host_env_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) Behavioral guard on the REAL production launch sequence.

    INVERTED for 0.3.0.  This test previously ASSERTED THE INJECTION — that a
    dirty host renders ``-e ANTHROPIC_API_KEY=<value>`` into the mutation
    agent's argv — on the reasoning that recording the exposure was better
    than hiding it.  That was the right instinct and the wrong assertion: a
    test that encodes current behaviour without asserting the intended
    PROPERTY will faithfully defend the defect. Under ``sandbox.auth =
    "volume"`` (the default) the correct assertion is that NO credential-
    bearing ``-e`` appears, on a clean host or a dirty one.

    Scope, stated exactly: this proves the agent container's docker argv
    carries no credential. It does NOT prove that no secret can ever reach the
    container — ``/home/node`` is mounted from the auth volume and the
    workspace mount carries whatever is in the candidate repo. The argv is
    where THIS bug lived.

    A scrubber-only assertion is insufficient and always was: the injection
    happened downstream of the scrubber, so such a test passes while the key
    still reaches the container. This asserts on the FINAL docker argv,
    across all three origins together.
    """
    from pathlib import Path as _Path

    from helix.config import load_config
    from helix.envpolicy import env_dict, resolve_env_grants
    from helix.sandbox import _docker_args

    config = load_config(EXAMPLE / "helix.toml")

    def agent_argv() -> list[str]:
        grants = resolve_env_grants(
            scope="agent",
            backend=config.agent.backend,
            sandbox_enabled=config.sandbox.enabled,
            auth_mode=config.sandbox.resolved_auth(),
            auth_env_allow=config.sandbox.auth_env_allow,
            agent_passthrough_env=config.sandbox.agent_passthrough_env,
            config_passthrough_env=config.passthrough_env,
            config_env=config.env,
        )
        return _docker_args(
            ["claude", "--version"],
            env_dict(grants, "agent"),
            _Path("/tmp/helix-guard-workspace"),
            config.sandbox,
            "agent",
            config.sandbox.image or "",
            config.agent.backend,
            grants=grants,
        )

    canary = "sk-ant-canary-value-must-not-reach-the-container"

    # This lane needs NO config edit under 0.3.0: it has no passthrough_env,
    # no sidecar, and auth defaults to "volume".
    assert config.sandbox.resolved_auth() == "volume"

    # Clean host: no credential may appear in the launched container's argv.
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    argv = agent_argv()
    assert not any(part.startswith("ANTHROPIC_API_KEY=") for part in argv)
    assert not any(part.startswith("ANTHROPIC_AUTH_TOKEN=") for part in argv)
    assert not any(canary in part for part in argv)
    # Non-vacuity: the argv must be a real launch, not an empty list.
    assert "helix-auth-claude:/home/node:rw" in argv
    assert config.sandbox.image in argv

    # DIRTY HOST — the inversion. This is the assertion that used to require
    # the credential to be present. Both claude variables are set, and
    # ANTHROPIC_AUTH_TOKEN is the more dangerous of the two (it overrides the
    # OAuth path and suppresses refresh), so a fix that filtered only
    # ANTHROPIC_API_KEY must still fail here.
    monkeypatch.setenv("ANTHROPIC_API_KEY", canary)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", canary + "-auth")
    dirty = agent_argv()
    assert not any(part.startswith("ANTHROPIC_API_KEY=") for part in dirty)
    assert not any(part.startswith("ANTHROPIC_AUTH_TOKEN=") for part in dirty)
    assert not any(canary in part for part in dirty), (
        "no credential value may reach the mutation agent's argv under "
        'sandbox.auth = "volume"'
    )
    # The volume — not an env var — is what authenticates the agent.
    assert "helix-auth-claude:/home/node:rw" in dirty
