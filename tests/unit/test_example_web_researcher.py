"""examples/web_researcher/evaluate.py reports a solve() exception instead of hiding it."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EVALUATOR = Path(__file__).resolve().parents[2] / "examples" / "web_researcher" / "evaluate.py"

QUESTION = {
    "id": "q_static",
    "question": "In which Python version was @asyncio.coroutine removed?",
    "ground_truth": "3.11",
    "match_mode": "exact",
    "category": "language_change",
}


def _evaluate(tmp_path: Path, agent_source: str) -> list[list[object]]:
    shutil.copy(EVALUATOR, tmp_path / "evaluate.py")
    (tmp_path / "val").mkdir()
    (tmp_path / "val" / "q_static.json").write_text(json.dumps(QUESTION))
    (tmp_path / "agent.py").write_text(agent_source)
    out = subprocess.run(
        [sys.executable, "evaluate.py"], cwd=tmp_path, env={"HELIX_SPLIT": "val", "PATH": ""},
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout
    line = next(x for x in out.splitlines() if x.startswith("HELIX_RESULT="))
    result: list[list[object]] = json.loads(line.removeprefix("HELIX_RESULT="))
    return result


def test_a_solve_exception_is_reported_in_side_info(tmp_path: Path) -> None:
    agent = (
        "def solve(question):\n"
        "    raise RuntimeError('Your credit balance is too low to access the Anthropic API.')\n"
    )
    [[score, info]] = _evaluate(tmp_path, agent)
    assert score == 0.0
    assert isinstance(info, dict)
    assert info["got"] == ""
    assert info["error"] == "solve() raised RuntimeError: Your credit balance is too low to access the Anthropic API."


@pytest.mark.parametrize(("answer", "score"), [("3.11", 1.0), ("3.10", 0.0)])
def test_an_answer_has_no_error(tmp_path: Path, answer: str, score: float) -> None:
    [[got_score, info]] = _evaluate(tmp_path, f"def solve(question):\n    return {answer!r}\n")
    assert got_score == score
    assert isinstance(info, dict)
    assert "error" not in info
