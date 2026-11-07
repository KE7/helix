"""`helix log`: one row per candidate, joined from the run's existing records."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from helix.cli import cli
from helix.population import EvalResult
from helix.state import BudgetState, EvolutionState, save_state


def _run(tmp_path: Path, *, summaries: bool = True) -> Path:
    """A finished run: seed g0-s0, kept child g1-s1, rejected child g2-s2."""
    helix = tmp_path / ".helix"
    helix.mkdir()
    (helix / "lineage.json").write_text(
        json.dumps(
            [
                {"id": "g0-s0", "parent": None, "parents": [], "operation": "seed",
                 "generation": 0, "files_changed": []},
                {"id": "g1-s1", "parent": "g0-s0", "parents": ["g0-s0"],
                 "operation": "mutate", "generation": 1, "files_changed": ["solve.py"]},
                {"id": "g2-s2", "parent": "g1-s1", "parents": ["g1-s1"],
                 "operation": "mutate", "generation": 2, "files_changed": ["notes.md"]},
            ]
        )
    )
    (helix / "evaluations").mkdir()
    for cid, scores in {"g0-s0": [0.0, 0.5], "g1-s1": [1.0, 1.0]}.items():
        result = EvalResult(
            candidate_id=cid,
            scores={"success": sum(scores) / 2},
            asi={},
            instance_scores={"a": scores[0], "b": scores[1]},
        )
        (helix / "evaluations" / f"{cid}.json").write_text(json.dumps(result.to_dict()))
    (helix / "attempts").mkdir()
    (helix / "attempts" / "g2-s2.json").write_text(
        json.dumps(
            {
                "candidate_id": "g2-s2",
                "instance_scores": {"a": 0.5},
                "attempt": {"status": "rejected", "reason": "minibatch_gate",
                            "parent_id": "g1-s1", "generation": 2,
                            "stage": "train_minibatch", "example_ids": ["a"]},
            }
        )
    )
    history = {
        "g0-s0": [{"child_id": "g1-s1", "score": 1.0, "accepted": True,
                   "summary": "# Change Summary\n\nFixed the off-by-one in f. It now starts at 1.\n\nMore detail."}],
    }
    save_state(
        EvolutionState(
            generation=2,
            frontier=["g0-s0", "g1-s1"],
            instance_scores={},
            budget=BudgetState(),
            config_hash="x",
            active_frontier={"a": ["g1-s1"], "b": ["g1-s1"]},
            sibling_history=history if summaries else {},
        ),
        tmp_path,
    )
    return tmp_path


def _log(project: Path) -> str:
    result = CliRunner().invoke(cli, ["log", "--dir", str(project)], env={"COLUMNS": "240"})
    assert result.exit_code == 0, result.output
    return result.output


def test_log_joins_outcome_score_files_and_summary(tmp_path: Path) -> None:
    lines = _log(_run(tmp_path)).splitlines()
    seed = next(ln for ln in lines if "g0-s0" in ln and "seed" in ln)
    kept = next(ln for ln in lines if "g1-s1" in ln and "frontier" in ln)
    rejected = next(ln for ln in lines if "g2-s2" in ln)

    assert "seed, dominated" in seed
    assert "frontier, best on 2 ids" in kept
    assert "1.000 (+0.750)" in kept
    assert "solve.py" in kept
    # The heading is skipped and only the first sentence is shown.
    assert "Fixed the off-by-one in f." in kept
    assert "starts at 1" not in kept
    assert "More detail" not in kept
    assert "rejected (minibatch_gate)" in rejected
    assert "0.500 on train_minibatch" in rejected
    # Rejected children keep their files: lineage records them before removal.
    assert "notes.md" in rejected


def test_log_says_how_to_get_summaries_when_none_were_recorded(tmp_path: Path) -> None:
    output = _log(_run(tmp_path, summaries=False))
    assert "evolution.sibling_history" in output


def test_log_with_no_run_warns(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["log", "--dir", str(tmp_path)])
    assert "no candidates found" in result.output.lower()


def test_log_caps_the_files_cell(tmp_path: Path) -> None:
    project = _run(tmp_path)
    lineage_path = project / ".helix" / "lineage.json"
    lineage = json.loads(lineage_path.read_text())
    lineage[1]["files_changed"] = [f"f{i}.py" for i in range(8)]
    lineage_path.write_text(json.dumps(lineage))
    output = _log(project)
    assert "f4.py" in output
    assert "f5.py" not in output
    assert "... 3 more" in output


def test_log_prints_brackets_in_summaries_literally(tmp_path: Path) -> None:
    project = _run(tmp_path)
    from helix.state import load_state

    state = load_state(project)
    assert state is not None
    state.sibling_history["g0-s0"][0]["summary"] = "Removed the [/INST] delimiter from a[i]."
    save_state(state, project)
    output = _log(project)
    assert "Removed the [/INST] delimiter from a[i]." in output


def test_log_without_a_saved_front_says_accepted_not_dominated(tmp_path: Path) -> None:
    """Runs saved before active_frontier existed have no per-instance front."""
    project = _run(tmp_path)
    from helix.state import load_state

    state = load_state(project)
    assert state is not None
    state.active_frontier = {}
    save_state(state, project)
    output = _log(project)
    assert "dominated" not in output
    assert "accepted" in output
