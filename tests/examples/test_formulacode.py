from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO_ROOT / "examples" / "formulacode"
sys.path.insert(0, str(DEMO_ROOT))

evaluator = importlib.import_module("evaluator")
manage = importlib.import_module("manage")
official_score = importlib.import_module("official_score")
workloads = importlib.import_module("workloads")


def test_official_speedup_and_advantage_semantics() -> None:
    metrics = official_score.score_measurements(
        ordered_workloads=["components.connected.case"],
        nop_samples={"components.connected.case": [10.0, 10.0, 10.0]},
        oracle_samples={"components.connected.case": [5.0, 5.0, 5.0]},
        agent_samples={"components.connected.case": [4.0, 4.0, 4.0]},
        correctness_passed=True,
    )
    assert metrics["task_speedup"] == pytest.approx(2.5)
    assert metrics["oracle_task_speedup"] == pytest.approx(2.0)
    assert metrics["agent_advantage"] == pytest.approx(0.5)
    assert metrics["agent_advantage_level4"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("benchmark", "expected"),
    [
        ("pkg.sub.module.func", ("pkg.sub.module", None, "func")),
        ("pkg.sub.module.func(10, 'x')", ("pkg.sub.module", None, "func")),
        ("pkg.sub.Bench.time_case(10)", ("pkg.sub", "Bench", "time_case")),
        ("func(1)", (None, None, "func")),
        ("", (None, None, None)),
    ],
)
def test_fc_eval_benchmark_deconstruction_is_exact(
    benchmark: str, expected: tuple[str | None, str | None, str | None]
) -> None:
    assert official_score._deconstruct_benchmark(benchmark) == expected


def test_lowercase_and_class_groups_match_fc_eval_levels() -> None:
    workloads_by_name = [
        "pkg.module.func_a(1)",
        "pkg.module.func_b(2)",
        "pkg.Bench.time_case(3)",
    ]
    nop = {name: [12.0] for name in workloads_by_name}
    oracle = {name: [6.0] for name in workloads_by_name}
    agent = {
        workloads_by_name[0]: [4.0],
        workloads_by_name[1]: [3.0],
        workloads_by_name[2]: [2.0],
    }
    metrics = official_score.score_measurements(
        ordered_workloads=workloads_by_name,
        nop_samples=nop,
        oracle_samples=oracle,
        agent_samples=agent,
        correctness_passed=True,
    )
    # Level 1 groups lower-case functions together under pkg.module and the
    # class benchmark under pkg; level 2 keeps the same two groups here.
    grouped = (((3.0 * 4.0) ** 0.5 - 2.0) + (6.0 - 2.0)) / 2
    assert metrics["agent_advantage_level1"] == pytest.approx(grouped)
    assert metrics["agent_advantage_level2"] == pytest.approx(grouped)
    assert metrics["agent_advantage_level4"] == pytest.approx(
        (3.0 * 4.0 * 6.0) ** (1 / 3) - 2.0
    )


def test_correctness_failure_reverts_to_baseline() -> None:
    metrics = official_score.score_measurements(
        ordered_workloads=["components.connected.case"],
        nop_samples={"components.connected.case": [10.0]},
        oracle_samples={"components.connected.case": [5.0]},
        agent_samples={},
        correctness_passed=False,
    )
    assert metrics["task_speedup"] == 1.0
    assert metrics["agent_advantage"] == pytest.approx(-1.0)
    assert metrics["fallback_to_baseline"] is True


def test_repeated_padded_ids_preserve_positional_multiplicity() -> None:
    names = ["components.connected.case", "components.weak.case"]
    metrics = official_score.score_measurements(
        ordered_workloads=[names[0], names[0], names[1]],
        nop_samples={names[0]: [8.0], names[1]: [27.0]},
        oracle_samples={names[0]: [4.0], names[1]: [9.0]},
        agent_samples={names[0]: [2.0], names[1]: [9.0]},
        correctness_passed=True,
    )
    expected_speedup = (4.0 * 4.0 * 3.0) ** (1 / 3)
    expected_oracle = (2.0 * 2.0 * 3.0) ** (1 / 3)
    assert metrics["task_speedup"] == pytest.approx(expected_speedup)
    assert metrics["agent_advantage"] == pytest.approx(
        expected_speedup - expected_oracle
    )
    pairs = official_score.helix_result_pairs(
        metrics,
        ["0", "0", "1"],
        {"0": names[0], "1": names[1]},
    )
    assert len(pairs) == 3
    assert [pair[1]["instance_id"] for pair in pairs] == ["0", "0", "1"]


@pytest.mark.parametrize(
    "payload",
    [{"0": "not-a-list"}, [], [0], ["../../secret"]],
)
def test_batch_parser_rejects_malformed_or_unknown_inputs(
    tmp_path: Path, payload: object
) -> None:
    (tmp_path / "helix_batch.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        evaluator.load_requested_ids(tmp_path, "train")


def test_correctness_timeout_becomes_failure_without_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args: object, **kwargs: object) -> object:
        raise evaluator.subprocess.TimeoutExpired(cmd=["pytest"], timeout=1)

    monkeypatch.setattr(evaluator.subprocess, "run", timeout)
    passed, kind = evaluator.run_correctness(
        {"correctness_tests": ["test.py"], "correctness_timeout_seconds": 1}
    )
    assert passed is False
    assert kind == "correctness_timeout"


def test_contract_error_redacts_exception_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-secret-must-not-appear"

    def fail() -> list[list[object]]:
        raise ValueError(f"bad credential {secret}")

    monkeypatch.setattr(evaluator, "evaluate", fail)
    assert evaluator.main() == 2
    captured = capsys.readouterr()
    assert "ValueError" in captured.err
    assert secret not in captured.err
    assert secret not in captured.out


def test_pins_and_parallel_isolation_configuration_are_exact() -> None:
    pins = json.loads((DEMO_ROOT / "pins.json").read_text())
    assert pins["helix"]["base_commit"] == ("84c7bcd2b82a56c8dd5c18b7fe5828101b6a7023")
    assert pins["helix"]["target_release"] == "0.3.0"
    assert pins["helix"]["relative_path_fix_commit"] == (
        "402dcc8cfb2c461144de8f019e6ec49811dc2da9"
    )
    assert pins["fc_eval"]["commit"] == "c08f665e7bf3b4de225b72dc02ce9b15b7aaba2b"
    assert pins["dataset"]["artifact_sha256"] == (
        "d872c4f3025e2331c012ce311e4330c73a72b87034c287fc9ce5f4d1b23e81d7"
    )
    assert pins["task"]["task_id"] == "networkx_networkx_7971"
    config = (DEMO_ROOT / "helix.toml.template").read_text()
    assert "num_parallel_proposals = 2" in config
    assert "mutations_per_parent = 2" in config
    assert "rng_seed = 7971" in config
    assert 'proposal_selection = "best_improvement"' in config
    assert 'omit_from_agent = [".formulacode", "helix.toml"]' in config
    assert 'protected_files = [".formulacode", "helix.toml"]' in config


def test_state_summary_preserves_parallel_accounting_and_order() -> None:
    tasks = [
        {
            "task_index": index,
            "parent_group": index // 2,
            "mutation_index": index % 2,
            "parent_id": f"parent-{index // 2}",
            "child_id": f"g1-s{index}",
            "status": "applied" if index in (1, 3) else "rejected",
            "selection": "selected" if index in (1, 3) else "not_selected",
            "cleanup": "not_required" if index in (1, 3) else "removed",
            "budget_charge": {"evaluations": 2},
            "budget_accounted": True,
            "applied": index in (1, 3),
        }
        for index in range(4)
    ]
    state = {
        "generation": 1,
        "budget": {"evaluations": 10, "agent_steps": 8},
        "proposal_batches": [
            {
                "batch_id": "g1-proposals",
                "phase": "complete",
                "p": 2,
                "n": 2,
                "budget_before_dispatch": 2,
                "budget_after_apply": 10,
                "tasks": tasks,
            }
        ],
    }
    summary = manage._state_summary(state)
    assert summary["distinct_child_ids"] is True
    assert summary["parent_major_order"] is True
    assert summary["budget"] == state["budget"]
    assert len(summary["tasks"]) == 4
    assert all(task["budget_accounted"] for task in summary["tasks"])


def test_state_summary_detects_duplicate_ids_and_resets_order_per_batch() -> None:
    def batch(batch_id: str, child_ids: list[str]) -> dict[str, object]:
        return {
            "batch_id": batch_id,
            "phase": "complete",
            "p": 1,
            "n": 2,
            "tasks": [
                {
                    "task_index": index,
                    "parent_group": 0,
                    "mutation_index": index,
                    "child_id": child_id,
                }
                for index, child_id in enumerate(child_ids)
            ],
        }

    state = {
        "proposal_batches": [
            batch("g1-proposals", ["g1-s0", "duplicate"]),
            batch("g2-proposals", ["g2-s0", "duplicate"]),
        ]
    }
    summary = manage._state_summary(state)
    assert summary["distinct_child_ids"] is False
    assert summary["parent_major_order"] is True


def test_geometric_mean_rejects_invalid_inputs() -> None:
    with pytest.raises(official_score.ScoreInputError):
        official_score.geometric_mean([])
    with pytest.raises(official_score.ScoreInputError):
        official_score.geometric_mean([math.inf])
    with pytest.raises(official_score.ScoreInputError):
        official_score.geometric_mean([0.0])


def test_official_adversarial_graph_inserts_singleton_first() -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.nodes: list[int] = []
            self.adj: dict[int, set[int]] = {}

        def add_node(self, node: int) -> None:
            if node not in self.adj:
                self.nodes.append(node)
                self.adj[node] = set()

        def update(self, other: "FakeGraph") -> None:
            for node in other.nodes:
                self.add_node(node)
                self.adj[node].update(other.adj[node])

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self.nodes)

        def __getitem__(self, node: int) -> set[int]:
            return self.adj[node]

    class FakeNetworkX:
        Graph = FakeGraph
        DiGraph = FakeGraph

        @staticmethod
        def complete_graph(nodes, create_using=None):  # type: ignore[no-untyped-def]
            graph = create_using() if isinstance(create_using, type) else FakeGraph()
            node_list = list(nodes)
            for node in node_list:
                graph.add_node(node)
            for node in node_list:
                graph.adj[node].update(other for other in node_list if other != node)
            return graph

    for split in workloads.WORKLOADS.values():
        for workload in split.values():
            graph = workloads._build_graph(FakeNetworkX, workload)
            assert next(iter(graph)) == 0
            assert len(graph[0]) == 0
