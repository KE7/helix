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
    evaluator_source = (EXAMPLE / "evaluate.py").read_text()
    assert '"--memory",\n        "4g"' in evaluator_source


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


def test_inspection_reports_p_by_n_ids_and_durable_budget(tmp_path: Path) -> None:
    inspect_run = _load("swebench_live_inspect", "inspect_run.py")
    tasks = [
        {
            "task_index": index,
            "parent_slot": index // 2,
            "mutation_slot": index % 2,
            "reserved_child_id": f"g1-s{index + 1}",
            "status": "completed",
            "selection": "selected" if index == 0 else "not_selected",
            "cleanup": "not_required" if index == 0 else "removed",
            "applied": index == 0,
            "evaluation_delta": 1,
        }
        for index in range(4)
    ]
    state = {
        "schema_version": 4,
        "generation": 1,
        "frontier": ["g0-s0", "g1-s1"],
        "active_frontier": {"instance": ["g1-s1"]},
        "budget": {"evaluations": 5, "mutations": 4},
        "mutation_counter": 4,
        "proposal_batches": [
            {
                "batch_id": "g1-b0",
                "p": 2,
                "n": 2,
                "phase": "completed",
                "budget_before_dispatch": 1,
                "budget_after_apply": 5,
                "tasks": tasks,
            }
        ],
    }
    helix_dir = tmp_path / ".helix"
    helix_dir.mkdir()
    (helix_dir / "state.json").write_text(json.dumps(state))
    summary = inspect_run.summarize(tmp_path)
    assert summary["candidate_ids"] == ["g1-s1", "g1-s2", "g1-s3", "g1-s4"]
    assert summary["candidate_ids_distinct"] is True
    assert summary["candidate_ids_parent_major"] is True
    assert summary["budget"] == {"evaluations": 5, "mutations": 4}


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
