"""Thin dispatcher to the pinned official LiveBench math scorers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class OfficialScorers:
    mathcontest: Callable[..., Any]
    aime: Callable[..., Any]
    olympiad: Callable[..., Any]
    amps_hard: Callable[..., Any]


def load_official_scorers() -> OfficialScorers:
    from livebench.process_results.math.AMPS_Hard.utils import (
        amps_hard_process_results,
    )
    from livebench.process_results.math.math_competitions.utils import (
        aime_process_results,
        mathcontest_process_results,
    )
    from livebench.process_results.math.olympiad.utils import (
        proof_rearrangement_process_results,
    )

    return OfficialScorers(
        mathcontest=mathcontest_process_results,
        aime=aime_process_results,
        olympiad=proof_rearrangement_process_results,
        amps_hard=amps_hard_process_results,
    )


def score_livebench_math(
    row: dict[str, Any], answer: str, scorers: OfficialScorers | None = None
) -> float:
    """Mirror LiveBench's math branch at the pinned code commit."""
    scorers = scorers or load_official_scorers()
    subtask = str(row["subtask"])
    ground_truth = str(row["ground_truth"])
    turns = row["turns"]
    question = str(turns[0] if isinstance(turns, (list, tuple)) else turns)
    parts = subtask.split("_")
    if parts[0] in {"amc", "smc"} or (len(parts) > 1 and parts[1] == "amc"):
        return float(scorers.mathcontest(ground_truth, answer, question, False))
    if parts[0] == "aime":
        return float(scorers.aime(ground_truth, answer, False))
    if parts[0] in {"imo", "usamo"}:
        return float(scorers.olympiad(ground_truth, answer, edit_distance=True, debug=False))
    if "amps_hard" in subtask:
        # server.py removes OPENAI_API_KEY from os.environ at startup, making
        # LiveBench's optional o3 fallback unreachable and scoring deterministic.
        return float(scorers.amps_hard(ground_truth, answer, False))
    raise ValueError(f"unroutable pinned LiveBench math subtask: {subtask!r}")

