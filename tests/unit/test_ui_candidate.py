"""`GET /api/candidate/{id}`: lineage, scores, read-only git diff and weakest instances."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from helix.config import EvaluatorConfig, HelixConfig
from helix.population import EvalResult
from helix.state import BudgetState, EvolutionState, save_state
from helix.ui import candidate as candidate_module
from helix.ui.candidate import candidate_detail, compact_side_info, instance_detail
from helix.ui.redact import MASK


class FakeReader:
    """The four RunReader methods candidate_detail uses."""

    def __init__(self, snapshot: dict[str, Any] | None, records: list[dict[str, Any]] | None = None,
                 rows: list[dict[str, Any]] | None = None) -> None:
        self._snapshot = snapshot
        self._records = records or []
        self._rows = rows or []

    def proposals(self) -> dict[str, Any]:
        return {"rows": self._rows, "counts": {}}

    def snapshot(self, now: float | None = None) -> dict[str, Any] | None:
        return self._snapshot

    def timing_records(self) -> list[dict[str, Any]]:
        return self._records

    def candidate_ids(self) -> set[str]:
        return {row["id"] for row in (self._snapshot or {}).get("candidates", [])}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _config() -> HelixConfig:
    return HelixConfig(objective="grasp", evaluator=EvaluatorConfig(command="python eval.py"))


def _row(cid: str, parent: str | None, mean: float, n: int, front: bool) -> dict[str, Any]:
    return {
        "id": cid, "generation": int(cid[1]), "parent": parent, "parents": [parent] if parent else [],
        "operation": "mutate" if parent else "seed", "mean": mean, "n": n, "front": front,
        "keys": 2 if front else 0, "sole": 1 if front else 0, "zero": 0, "objective_means": None,
        "note": None, "agent_seconds": None, "validation_seconds": None,
    }


def _project(tmp_path: Path, *, diff_lines: int = 3) -> Path:
    """A repo with seed g0-s0 (detached worktree), accepted child g1-s1 and rejected g1-s2."""
    root = tmp_path / "proj"
    root.mkdir()
    _git(root, "init", "-q")
    (root / ".gitignore").write_text(".helix/\n")
    (root / "policy.py").write_text("x = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "seed")
    helix = root / ".helix"
    _git(root, "worktree", "add", "-q", "--detach", str(helix / "worktrees" / "g0-s0"), "HEAD")
    child = helix / "worktrees" / "g1-s1"
    _git(root, "worktree", "add", "-q", "-b", "helix/g1-s1", str(child), "HEAD")
    (child / "policy.py").write_text("".join(f"x = {i}\n" for i in range(diff_lines)))
    (child / "notes.md").write_text("OPENAI_API_KEY=hunter2\n")
    _git(child, "add", ".")
    _git(child, "commit", "-qm", "helix: mutate g1-s1")

    (helix / "lineage.json").write_text(json.dumps([
        {"id": "g0-s0", "parent": None, "parents": [], "operation": "seed", "generation": 0, "files_changed": []},
        {"id": "g1-s1", "parent": "g0-s0", "parents": ["g0-s0"], "operation": "mutate", "generation": 1,
         "files_changed": ["policy.py"]},
        {"id": "g1-s2", "parent": "g0-s0", "parents": ["g0-s0"], "operation": "mutate", "generation": 1,
         "files_changed": []},
    ]))
    (helix / "evaluations").mkdir()
    for cid, scores in {"g0-s0": [0.0, 1.0, 0.0], "g1-s1": [1.0, 1.0, 0.0]}.items():
        result = EvalResult(
            candidate_id=cid, scores={}, asi={},
            instance_scores={"lift__1": scores[0], "lift__2": scores[1], "stack__1": scores[2]},
            per_example_side_info=[
                {"feedback": json.dumps({"reason": "slip", "_internal": 1, "t": 3.2}), "_hidden": "x"},
                {"feedback": "ok"},
                {"feedback": "token sk-abcdefghijklmnop1234 leaked", "hardware": {"LUT": 10}},
            ],
        )
        (helix / "evaluations" / f"{cid}.json").write_text(json.dumps(result.to_dict()))
    (helix / "attempts").mkdir()
    rejected = EvalResult(candidate_id="g1-s2", scores={}, asi={}, instance_scores={"lift__1": 0.0, "lift__2": 1.0})
    (helix / "attempts" / "g1-s2.json").write_text(json.dumps({
        **rejected.to_dict(),
        "attempt": {"status": "rejected", "reason": "minibatch_gate", "parent_id": "g0-s0", "generation": 1,
                    "stage": "train_minibatch", "example_ids": ["lift__1", "lift__2"], "retained": False},
    }))
    save_state(
        EvolutionState(
            generation=1, frontier=["g0-s0", "g1-s1"], instance_scores={}, budget=BudgetState(),
            config_hash="x",
            sibling_history={"g0-s0": [
                {"child_id": "g1-s1", "score": 1.0, "accepted": True, "summary": "Tightened the grasp."},
                {"child_id": "g1-s2", "score": 0.5, "accepted": False, "summary": "Tried a wider gripper."},
            ]},
        ),
        root,
    )
    return root


def _snapshot() -> dict[str, Any]:
    return {
        "binary_scores": True,
        "candidates": [_row("g0-s0", None, 1 / 3, 3, False), _row("g1-s1", "g0-s0", 2 / 3, 3, True)],
    }


_RECORDS = [
    {"event": "end", "span": "agent", "candidate": "g1-s1", "duration_seconds": 100.0, "attrs": {}},
    {"event": "end", "span": "agent", "candidate": "g1-s1", "duration_seconds": 5.0, "attrs": {}},
    # The parent's minibatch evaluation inside the child's proposal is not the child's.
    {"event": "end", "span": "evaluate", "candidate": "g1-s1", "duration_seconds": 50.0,
     "attrs": {"candidate_id": "g0-s0", "split": "train"}},
    {"event": "end", "span": "evaluate", "candidate": "g1-s1", "duration_seconds": 20.0,
     "attrs": {"candidate_id": "g1-s1", "split": "train"}},
    {"event": "end", "span": "evaluate", "candidate": "g1-s1", "duration_seconds": 30.0,
     "attrs": {"candidate_id": "g1-s1", "split": "val"}},
    {"event": "start", "span": "agent", "candidate": "g1-s1", "attrs": {}},
    # The child's own evaluation as a parent, inside a grandchild's proposal.
    {"event": "end", "span": "evaluate", "candidate": "g2-s3", "duration_seconds": 40.0,
     "attrs": {"candidate_id": "g1-s1", "split": "train"}},
]


def test_accepted_candidate_has_diff_files_commit_and_lowest(tmp_path: Path) -> None:
    root = _project(tmp_path)
    detail = candidate_detail(root, _config(), FakeReader(_snapshot(), _RECORDS), "g1-s1")
    assert detail is not None
    head = _git(root / ".helix" / "worktrees" / "g1-s1", "rev-parse", "HEAD")
    assert detail["commit"] == head[:7]
    assert detail["accepted"] is True and detail["front"] is True and detail["rank"] == 1
    assert detail["parent"] == "g0-s0" and detail["operation"] == "mutate" and detail["generation"] == 1
    assert detail["successes"] == 2 and detail["keys"] == 2 and detail["sole"] == 1
    assert detail["seed"] is None
    assert detail["worktree"] == ".helix/worktrees/g1-s1" and detail["worktree_exists"] is True
    assert {f["path"]: (f["added"], f["removed"]) for f in detail["files"]} == {
        "notes.md": (1, 0), "policy.py": (2, 0),
    }
    assert "+x = 2" in detail["diff"] and "hunter2" not in detail["diff"] and MASK in detail["diff"]
    assert detail["diff_truncated"] is False
    assert detail["note"] == "Tightened the grasp."
    assert detail["attempt"] is None
    assert detail["timing"] == {"agent_seconds": 105.0, "gate_seconds": 20.0, "validation_seconds": 30.0}

    lowest = detail["lowest"]
    assert [x["instance"] for x in lowest] == ["stack__1", "lift__1", "lift__2"]
    assert lowest[0]["side_info"] == f'feedback: token {MASK} leaked · hardware.LUT: 10'
    assert lowest[1]["side_info"] == "feedback.reason: slip · feedback.t: 3.2"


def test_git_runs_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    calls: list[tuple[list[str], dict[str, str]]] = []
    real = subprocess.run

    def spy(args: list[str], **kwargs: Any) -> Any:
        calls.append((args, kwargs["env"]))
        return real(args, **kwargs)

    monkeypatch.setattr(candidate_module.subprocess, "run", spy)
    candidate_detail(root, _config(), FakeReader(_snapshot()), "g1-s1")
    assert calls
    for args, env in calls:
        assert args[0] == "git" and args[3] in ("rev-parse", "diff")
        assert env["GIT_OPTIONAL_LOCKS"] == "0"
        if args[3] == "diff":
            assert "--no-ext-diff" in args


def test_diff_is_capped_at_400_lines(tmp_path: Path) -> None:
    root = _project(tmp_path, diff_lines=500)
    detail = candidate_detail(root, _config(), FakeReader(_snapshot()), "g1-s1")
    assert detail is not None
    assert detail["diff_truncated"] is True
    assert len(detail["diff"].splitlines()) == 400


def test_missing_worktree_gives_nulls(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _git(root, "worktree", "remove", "--force", str(root / ".helix" / "worktrees" / "g1-s1"))
    detail = candidate_detail(root, _config(), FakeReader(_snapshot()), "g1-s1")
    assert detail is not None
    assert detail["worktree_exists"] is False
    assert detail["commit"] is None and detail["files"] is None and detail["diff"] is None
    assert detail["mean"] == pytest.approx(2 / 3)


def test_seed_has_no_diff(tmp_path: Path) -> None:
    root = _project(tmp_path)
    detail = candidate_detail(root, _config(), FakeReader(_snapshot()), "g0-s0")
    assert detail is not None
    assert detail["seed"] == "repository" and detail["parent"] is None
    assert detail["commit"] is not None and detail["diff"] is None and detail["rank"] == 2


def test_attempt_only_candidate(tmp_path: Path) -> None:
    root = _project(tmp_path)
    detail = candidate_detail(root, _config(), FakeReader(_snapshot()), "g1-s2")
    assert detail is not None
    assert detail["accepted"] is False and detail["front"] is False and detail["rank"] is None
    assert detail["mean"] is None and detail["successes"] is None
    assert detail["attempt"] == {"reason": "minibatch_gate", "stage": "train_minibatch", "score": 0.5, "n": 2}
    assert detail["parent"] == "g0-s0" and detail["generation"] == 1
    assert detail["worktree_exists"] is False
    assert detail["note"] == "Tried a wider gripper."
    assert [x["instance"] for x in detail["lowest"]] == ["lift__1", "lift__2"]


@pytest.mark.parametrize("cid", ["../g1-s1", "g1-s1/../../x", "g1-x1", "", "g1-s1\n", "seed"])
def test_bad_ids_are_rejected(tmp_path: Path, cid: str) -> None:
    root = _project(tmp_path)
    assert candidate_detail(root, _config(), FakeReader(_snapshot()), cid) is None


def test_unknown_id_is_none(tmp_path: Path) -> None:
    root = _project(tmp_path)
    assert candidate_detail(root, _config(), FakeReader(_snapshot()), "g9-s99") is None


def test_change_summary_file_is_the_fallback_note(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / ".helix" / "worktrees" / "g1-s1" / ".agent_change_summary.md").write_text("Raised the arm.\n")
    state = json.loads((root / ".helix" / "state.json").read_text())
    state["sibling_history"] = {}
    (root / ".helix" / "state.json").write_text(json.dumps(state))
    detail = candidate_detail(root, _config(), FakeReader(_snapshot()), "g1-s1")
    assert detail is not None and detail["note"] == "Raised the arm.\n"


def test_compact_side_info_truncates_to_one_line() -> None:
    text = compact_side_info({"log": "line one\nline two " + "x" * 600})
    assert text is not None and "\n" not in text
    assert len(text) == 400 and text.endswith("…")
    assert compact_side_info(None) is None


def test_timing_skips_an_agent_interrupted_before_resume() -> None:
    """Ctrl-C ends the agent span with KeyboardInterrupt and ``helix resume`` reuses the id."""
    interrupted = {"event": "end", "span": "agent", "candidate": "g1-s1", "duration_seconds": 60.0,
                   "attrs": {}, "outcome": "error", "error_type": "KeyboardInterrupt"}
    assert candidate_module._timing([*_RECORDS, interrupted], "g1-s1") == {
        "agent_seconds": 105.0, "gate_seconds": 20.0, "validation_seconds": 30.0}


def _grandchild(root: Path) -> Path:
    """g2-s3, a child of g1-s1 that changes policy.py once more."""
    helix = root / ".helix"
    wt = helix / "worktrees" / "g2-s3"
    _git(root, "worktree", "add", "-q", "-b", "helix/g2-s3", str(wt), "helix/g1-s1")
    (wt / "policy.py").write_text("x = 0\nx = 1\nx = 2\ny = 9\n")
    _git(wt, "add", ".")
    _git(wt, "commit", "-qm", "helix: mutate g2-s3")
    return wt


def test_diff_against_the_seed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _grandchild(root)
    snapshot = _snapshot()
    snapshot["candidates"].append(_row("g2-s3", "g1-s1", 2 / 3, 3, True))
    reader = FakeReader(snapshot)

    vs_parent = candidate_detail(root, _config(), reader, "g2-s3")
    assert vs_parent is not None and vs_parent["diff_base"] == "parent"
    assert [f["path"] for f in vs_parent["files"]] == ["policy.py"]
    assert "+y = 9" in vs_parent["diff"] and "+x = 2" not in vs_parent["diff"]

    vs_seed = candidate_detail(root, _config(), reader, "g2-s3", vs="seed")
    assert vs_seed is not None and vs_seed["diff_base"] == "seed"
    assert {f["path"] for f in vs_seed["files"]} == {"notes.md", "policy.py"}
    assert "+x = 2" in vs_seed["diff"] and "+y = 9" in vs_seed["diff"]

    seed = candidate_detail(root, _config(), reader, "g0-s0", vs="seed")
    assert seed is not None and seed["diff"] is None and seed["files"] is None
    assert candidate_detail(root, _config(), reader, "g2-s3", vs="g0-s0") is None


def test_task_prompt_is_redacted_capped_and_own_worktree_only(tmp_path: Path) -> None:
    root = _project(tmp_path)
    wt = root / ".helix" / "worktrees" / "g1-s1"
    reader = FakeReader(_snapshot())
    detail = candidate_detail(root, _config(), reader, "g1-s1")
    assert detail is not None and detail["prompt"] is None and detail["prompt_truncated"] is False

    (wt / ".agent_internal").mkdir()
    (wt / ".agent_internal" / "task_prompt.md").write_text("Fallback prompt.\n")
    detail = candidate_detail(root, _config(), reader, "g1-s1")
    assert detail is not None and detail["prompt"] == "Fallback prompt.\n"

    (wt / ".agent_task_prompt.md").write_text("Improve the grasp. ANTHROPIC_API_KEY=sk-ant-abc123\n")
    detail = candidate_detail(root, _config(), reader, "g1-s1")
    assert detail is not None
    assert detail["prompt"].startswith("Improve the grasp.") and "sk-ant-abc123" not in detail["prompt"]

    (wt / ".agent_task_prompt.md").write_text("p" * 20500)
    detail = candidate_detail(root, _config(), reader, "g1-s1")
    assert detail is not None and len(detail["prompt"]) == 20000 and detail["prompt_truncated"] is True

    # A symlink out of the worktree is never followed; the parent's prompt is never used.
    (wt / ".agent_task_prompt.md").unlink()
    (wt / ".agent_internal" / "task_prompt.md").unlink()
    secret = tmp_path / "outside.md"
    secret.write_text("outside")
    (wt / ".agent_task_prompt.md").symlink_to(secret)
    (root / ".helix" / "worktrees" / "g0-s0" / ".agent_task_prompt.md").write_text("seed prompt")
    detail = candidate_detail(root, _config(), reader, "g1-s1")
    assert detail is not None and detail["prompt"] is None


def test_vs_parent_counts_solved_and_regressed_instances(tmp_path: Path) -> None:
    root = _project(tmp_path)
    child = EvalResult(candidate_id="g1-s1", scores={}, asi={},
                       instance_scores={"lift__1": 1.0, "lift__2": 0.0, "stack__1": 0.0})
    (root / ".helix" / "evaluations" / "g1-s1.json").write_text(json.dumps(child.to_dict()))
    reader = FakeReader(_snapshot())
    detail = candidate_detail(root, _config(), reader, "g1-s1")
    assert detail is not None
    # parent g0-s0 scored [0, 1, 0]
    assert detail["vs_parent"] == {"improved": 1, "regressed": 1,
                                   "improved_ids": ["lift__1"], "regressed_ids": ["lift__2"]}
    seed = candidate_detail(root, _config(), reader, "g0-s0")
    rejected = candidate_detail(root, _config(), reader, "g1-s2")
    assert seed is not None and seed["vs_parent"] is None
    assert rejected is not None and rejected["vs_parent"] is None


def test_children_come_from_the_proposal_rows(tmp_path: Path) -> None:
    root = _project(tmp_path)
    rows = [
        {"id": "g1-s1", "parents": ["g0-s0"], "outcome": "frontier", "reason": None, "score": 2 / 3,
         "basis": "val"},
        {"id": "g1-s2", "parents": ["g0-s0"], "outcome": "rejected", "reason": "minibatch_gate",
         "score": 0.5, "basis": "train_minibatch"},
        {"id": "skip:g2:g0-s0", "parents": ["g0-s0"], "outcome": "skipped", "reason": "perfect_subsample",
         "score": 1.0, "basis": "train_minibatch"},
        {"id": "g2-m3", "parents": ["g1-s1", "g0-s0"], "outcome": "running", "reason": None,
         "score": None, "basis": None},
        {"id": "g2-s4", "parents": ["g1-s1"], "outcome": "failed", "reason": "failed mutation",
         "score": None, "basis": None},
    ]
    detail = candidate_detail(root, _config(), FakeReader(_snapshot(), rows=rows), "g0-s0")
    assert detail is not None
    assert [c["id"] for c in detail["children"]] == ["g1-s1", "g1-s2", "g2-m3"]
    assert detail["children"][1]["basis"] == "train_minibatch"
    detail = candidate_detail(root, _config(), FakeReader(_snapshot(), rows=rows), "g1-s1")
    assert detail is not None and [c["id"] for c in detail["children"]] == ["g2-m3", "g2-s4"]


def test_instances_and_families_compare_with_the_parent(tmp_path: Path) -> None:
    root = _project(tmp_path)
    reader = FakeReader(_snapshot())
    detail = candidate_detail(root, _config(), reader, "g1-s1")
    assert detail is not None
    # parent g0-s0 scored [0, 1, 0]; every instance, in evaluation order, no side_info
    assert (detail["compare_to"], detail["instances_basis"]) == ("g0-s0", "val")
    assert detail["instances"] == [{"id": "lift__1", "score": 1.0, "parent_score": 0.0},
                                   {"id": "lift__2", "score": 1.0, "parent_score": 1.0},
                                   {"id": "stack__1", "score": 0.0, "parent_score": 0.0}]
    assert detail["families"] == [
        {"name": "lift", "n": 2, "mean": 1.0, "successes": 2, "parent_mean": 0.5, "parent_successes": 1,
         "delta": 0.5},
        {"name": "stack", "n": 1, "mean": 0.0, "successes": 0, "parent_mean": 0.0, "parent_successes": 0,
         "delta": 0.0},
    ]

    seed = candidate_detail(root, _config(), reader, "g0-s0")
    assert seed is not None and seed["compare_to"] is None
    assert {r["parent_score"] for r in seed["instances"]} == {None}
    assert seed["families"] is not None and seed["families"][0]["parent_mean"] is None
    assert seed["families"][0]["delta"] is None

    # A rejected candidate's scores are gate scores: no parent comparison, basis = its stage.
    rejected = candidate_detail(root, _config(), reader, "g1-s2")
    assert rejected is not None
    assert (rejected["compare_to"], rejected["instances_basis"]) == (None, "train_minibatch")
    assert [r["parent_score"] for r in rejected["instances"]] == [None, None]
    assert rejected["families"] is None  # one family only


def test_instance_detail_has_side_info_and_the_parents(tmp_path: Path) -> None:
    root = _project(tmp_path)
    reader = FakeReader(_snapshot())
    one = instance_detail(root, reader, "g1-s1", "stack__1")
    assert one == {
        "instance": "stack__1", "basis": "val", "score": 0.0,
        "side_info": f"feedback: token {MASK} leaked · hardware.LUT: 10",
        "parent": {"id": "g0-s0", "score": 0.0, "side_info": f"feedback: token {MASK} leaked · hardware.LUT: 10"},
    }
    gate = instance_detail(root, reader, "g1-s2", "lift__2")
    assert gate == {"instance": "lift__2", "basis": "train_minibatch", "score": 1.0, "side_info": None,
                    "parent": None}
    assert instance_detail(root, reader, "g0-s0", "lift__1")["parent"] is None  # type: ignore[index]
    for cid, iid in (("g1-s1", "nope"), ("g1-s1", "../state.json"), ("g1-s2", "stack__1"),
                     ("g9-s9", "lift__1"), ("../x", "lift__1")):
        assert instance_detail(root, reader, cid, iid) is None


def test_instance_side_info_allows_a_longer_line() -> None:
    info = {"log": "x" * 5000}
    assert len(compact_side_info(info) or "") == 400
    long = compact_side_info(info, 4000) or ""
    assert len(long) == 4000 and long.endswith("…")


def _usage_record(cid: str, cost: float, tokens: tuple[int, int, int]) -> dict[str, Any]:
    return {"event": "usage", "candidate": cid, "source": "mutation", "input_tokens": tokens[0],
            "output_tokens": tokens[1], "cached_input_tokens": tokens[2], "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0, "reasoning_tokens": 0, "cost_usd": cost, "parent_span_id": None,
            "run_id": "r1", "generation": 1, "wall_time": 1.0, "thread_id": 1}


def _backend_result(worktree: Path, name: str, cost: float, tokens: tuple[int, int, int]) -> None:
    usage = {"input_tokens": tokens[0], "output_tokens": tokens[1], "cached_input_tokens": tokens[2],
             "cost_usd": cost, "num_turns": 3, "tool_names": []}
    (worktree / name).write_text(json.dumps({"backend": "claude", "usage": usage, "parsed": {}}))


def test_usage_comes_from_the_candidates_usage_records(tmp_path: Path) -> None:
    root = _project(tmp_path)
    worktree = root / ".helix" / "worktrees" / "g1-s1"
    _backend_result(worktree, ".helix_backend_result.json", 9.0, (9, 9, 9))  # records win
    records = [*_RECORDS, _usage_record("g1-s1", 0.5, (100, 10, 40)), _usage_record("g1-s1", 0.25, (50, 5, 0)),
               _usage_record("g0-s0", 7.0, (1, 1, 1))]
    detail = candidate_detail(root, _config(), FakeReader(_snapshot(), records), "g1-s1")
    assert detail is not None
    assert detail["usage_source"] == "timing"
    assert detail["usage"] == {"cost_usd": 0.75, "input_tokens": 150, "output_tokens": 15,
                               "cached_input_tokens": 40}
    # Usage records leave the timing line unchanged.
    assert detail["timing"] == {"agent_seconds": 105.0, "gate_seconds": 20.0, "validation_seconds": 30.0}


def test_usage_falls_back_to_the_worktrees_backend_results(tmp_path: Path) -> None:
    """A run older than usage records: the mutation's and change summary's result files."""
    root = _project(tmp_path)
    worktree = root / ".helix" / "worktrees" / "g1-s1"
    _backend_result(worktree, ".helix_backend_result.json", 0.4, (1000, 200, 600))
    _backend_result(worktree, ".helix_backend_summary_result.json", 0.1, (100, 20, 0))
    detail = candidate_detail(root, _config(), FakeReader(_snapshot(), _RECORDS), "g1-s1")
    assert detail is not None
    assert detail["usage_source"] == "worktree"
    assert detail["usage"] == {"cost_usd": pytest.approx(0.5), "input_tokens": 1100, "output_tokens": 220,
                               "cached_input_tokens": 600}

    # A result file that links out of the worktree is not read.
    outside = tmp_path / "outside.json"
    _backend_result(tmp_path, "outside.json", 5.0, (5, 5, 5))
    (worktree / ".helix_backend_summary_result.json").unlink()
    (worktree / ".helix_backend_summary_result.json").symlink_to(outside)
    detail = candidate_detail(root, _config(), FakeReader(_snapshot(), _RECORDS), "g1-s1")
    assert detail is not None and detail["usage"]["cost_usd"] == pytest.approx(0.4)

    # Malformed usage, then no result file at all: nothing to show.
    (worktree / ".helix_backend_result.json").write_text(json.dumps({"usage": {"cost_usd": "x"}}))
    detail = candidate_detail(root, _config(), FakeReader(_snapshot(), _RECORDS), "g1-s1")
    assert detail is not None and (detail["usage"], detail["usage_source"]) == (None, None)


def test_no_usage_without_records_or_worktree(tmp_path: Path) -> None:
    root = _project(tmp_path)
    detail = candidate_detail(root, _config(), FakeReader(_snapshot()), "g1-s2")  # rejected, worktree removed
    assert detail is not None and (detail["usage"], detail["usage_source"]) == (None, None)
    detail = candidate_detail(root, _config(), FakeReader(_snapshot(), [_usage_record("g1-s2", 0.0, (80, 8, 0))]),
                              "g1-s2")
    assert detail is not None and detail["usage_source"] == "timing"
    assert detail["usage"] == {"cost_usd": 0.0, "input_tokens": 80, "output_tokens": 8, "cached_input_tokens": 0}
