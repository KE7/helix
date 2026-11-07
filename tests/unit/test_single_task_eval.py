"""Single-task (no-dataset) evaluation end to end.

GEPA parity: ``optimize_anything`` runs ``dataset=None, valset=None`` on a
one-element sentinel dataset (``src/gepa/optimize_anything.py``,
``effective_dataset = [_SINGLE_INSTANCE_SENTINEL]``).  HELIX's ``_cached_eval``
writes the matching one-id ``helix_batch.json`` so a stub evaluator that
prints one ``[score, side_info]`` pair parses to a score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from helix.config import EvaluatorConfig, HelixConfig
from helix.evolution import _cached_eval
from helix.population import Candidate


def test_single_task_eval_scores_one_pair(tmp_path: Path) -> None:
    (tmp_path / "evaluate.py").write_text(
        "import json\n"
        'print("HELIX_RESULT=" + json.dumps([[0.5, {"feedback": "x"}]]))\n'
    )
    candidate = Candidate(
        id="g0-s0",
        worktree_path=str(tmp_path),
        branch_name="helix/g0-s0",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )
    config = HelixConfig(
        objective="test objective",
        evaluator=EvaluatorConfig(command=f"{sys.executable} evaluate.py"),
    )

    result, was_cached = _cached_eval(candidate, config, "train", None)

    assert was_cached is False
    assert json.loads((tmp_path / "helix_batch.json").read_text()) == ["0"]
    assert result.scores["success"] == 0.5
    assert result.instance_scores == {"0": 0.5}
    assert result.per_example_side_info == [{"feedback": "x"}]
