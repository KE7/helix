from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from helix.config import load_config
from helix.executor import _scrub_environment
from helix.mutator import build_mutation_prompt
from helix.population import EvalResult
from helix.sandbox import _copy_tree_contents


ROOT = Path(__file__).parents[2] / "examples" / "livebench_math"
sys.path.insert(0, str(ROOT))

import constants  # noqa: E402
import dataset  # noqa: E402
import evaluate  # noqa: E402
import inspect_run  # noqa: E402
import scoring  # noqa: E402


def test_manifest_pins_exact_revisions_and_representative_smoke_ids() -> None:
    assert constants.GEPA_BLOG_COMMIT == "121084499247e7ddfa05ec453a53e0d644838b7a"
    assert constants.TERRARIUM_COMMIT == "e2c8b59079ed26de2d38e8aaf4ac2b4437703fe9"
    assert constants.LIVEBENCH_CODE_COMMIT == "1de6a43e82a137beeeaf2b92d683eedb67f0cf97"
    assert constants.LIVEBENCH_DATA_REVISION == "bb66571c8ccf32d3df9e6f48b920d3770ff4aacb"
    assert {name: len(ids) for name, ids in constants.SMOKE_IDS.items()} == {
        "train": 4,
        "val": 4,
    }
    assert len(set(constants.SMOKE_IDS["train"] + constants.SMOKE_IDS["val"])) == 8
    assert constants.PUBLICATION_PROPOSER_MODEL == "gpt-5-mini"
    assert constants.SMOKE_PROPOSER_MODEL == "gpt-5.4"


def test_official_dispatch_routes_all_math_families() -> None:
    calls = []

    def record(name, value):
        def inner(*args, **kwargs):
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
    assert scoring.score_livebench_math({**base, "subtask": "amc_12a_2023"}, "A", fakes) == 1
    assert scoring.score_livebench_math({**base, "subtask": "updated_amc_12a_2023"}, "A", fakes) == 1
    assert scoring.score_livebench_math({**base, "subtask": "aime_i_2024"}, "A", fakes) == 1
    assert scoring.score_livebench_math({**base, "subtask": "imo"}, "A", fakes) == 0.5
    assert scoring.score_livebench_math({**base, "subtask": "amps_hard_gcd"}, "A", fakes) == 1
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
            {"question_id": question_id, "subtask": "x", "turns": ["q"], "ground_truth": "a"}
            for question_id in reversed(ids)
        ]
        for name, ids in constants.SMOKE_IDS.items()
    }
    selected = dataset.select_smoke_rows(rows)
    assert {
        name: tuple(row["question_id"] for row in split_rows)
        for name, split_rows in selected.items()
    } == constants.SMOKE_IDS


def test_evaluate_adapter_preserves_repeated_and_padded_positions(tmp_path: Path) -> None:
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


def test_client_timeout_is_bounded_and_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def timeout(request, timeout):
        observed["timeout"] = timeout
        raise TimeoutError("bounded evaluator timeout")

    monkeypatch.setattr(evaluate.urllib.request, "urlopen", timeout)
    with pytest.raises(TimeoutError, match="bounded evaluator timeout"):
        evaluate.run_client(
            "http://evaluator.invalid/evaluate",
            {"prompt": "p", "split": "val", "ids": ["0"]},
            timeout=3.5,
        )
    assert observed == {"timeout": 3.5}


def test_client_rejects_wrong_result_cardinality(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"results":[]}'

    monkeypatch.setattr(evaluate.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="cardinality"):
        evaluate.run_client(
            "http://evaluator.invalid/evaluate",
            {"prompt": "p", "split": "val", "ids": ["0"]},
        )


def test_parallel_and_baseline_configs_pin_expected_shapes() -> None:
    import tomllib

    p2n2 = tomllib.loads((ROOT / "helix.toml").read_text())
    one = tomllib.loads((ROOT / "helix.1x1.toml").read_text())
    assert (p2n2["evolution"]["num_parallel_proposals"], p2n2["evolution"]["mutations_per_parent"]) == (2, 2)
    assert (one["evolution"]["num_parallel_proposals"], one["evolution"]["mutations_per_parent"]) == (1, 1)
    assert p2n2["agent"]["model"] == constants.SMOKE_PROPOSER_MODEL
    assert one["agent"]["model"] == constants.SMOKE_PROPOSER_MODEL
    assert p2n2["sandbox"]["enabled"] and p2n2["sandbox"]["evaluator"]
    assert p2n2["sandbox"]["image"] == "ghcr.io/ke7/helix-evo-runner-codex:0.2.0"
    assert p2n2["evaluator"]["sidecar"]["image"].startswith(
        "helix-livebench-math:"
    )
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


def test_agent_image_and_evaluator_image_are_separate_trust_domains() -> None:
    config = load_config(ROOT / "helix.toml")
    assert config.evaluator.sidecar is not None
    assert config.sandbox.image == "ghcr.io/ke7/helix-evo-runner-codex:0.2.0"
    assert config.evaluator.sidecar.image == "helix-livebench-math:e2c8b590-smoke"
    assert config.evaluator.sidecar.resolved_runner_image == (
        "helix-livebench-math:e2c8b590-smoke"
    )
    assert config.sandbox.image != config.evaluator.sidecar.image


def test_agent_snapshot_omits_every_non_prompt_demo_file(tmp_path: Path) -> None:
    config = load_config(ROOT / "helix.toml")
    omitted = {Path(item) for item in config.sandbox.omit_from_agent}
    source_entries = {path.name for path in ROOT.iterdir()}
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


def test_sidecar_errors_are_zero_scored_and_secrets_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text

    def fake_read(path: Path, *args, **kwargs):
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
    spec = importlib.util.spec_from_file_location("livebench_math_server_errors", ROOT / "server.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {"question_id": "q", "subtask": "aime_i_2024", "ground_truth": "1", "turns": ["q"]}

    def fail(prompt, example):
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

    def fake_read(path: Path, *args, **kwargs):
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
    result = module.evaluate_one(
        "prompt", row, lambda prompt, example: ("wrong", {})
    )
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


def test_sidecar_preserves_duplicate_positions_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text

    def fake_read(path: Path, *args, **kwargs):
        if str(path) == "/opt/livebench-math/data.json":
            return json.dumps(
                {
                    "dataset_revision": constants.LIVEBENCH_DATA_REVISION,
                    "smoke_splits": {"train": [], "val": []},
                }
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read)
    spec = importlib.util.spec_from_file_location("livebench_math_server_order", ROOT / "server.py")
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


def _completed_p2n2_state() -> dict:
    tasks = []
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
    after = inspect_run.audit_state(json.loads(json.dumps(state)), require_terminal=True)
    assert before == after
    assert before["candidate_ids"] == ["child-0", "child-1", "child-2", "child-3"]
    assert before["ledger_evaluations"] == 8
    assert before["nonproposal_evaluations"] == 4


def test_state_inspection_rejects_duplicate_candidate_ids() -> None:
    state = _completed_p2n2_state()
    state["proposal_batches"][0]["tasks"][1]["child_id"] = "child-0"
    with pytest.raises(ValueError, match="globally distinct"):
        inspect_run.audit_state(state, require_terminal=True)
