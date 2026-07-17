"""Regression tests for proposal scoring on padded minibatches.

The epoch sampler deliberately pads a short epoch by repeating example IDs.
Evaluation results and caches are keyed by ID, so proposal scoring must recover
the original multiplicity from ``ProposalTask.minibatch_ids``.
"""

from __future__ import annotations

import threading
from typing import cast

import pytest

from helix.eval_cache import EvaluationCache
from helix.eval_policy import StrictImprovementAcceptance
from helix.executor import EvalBatchItem, run_evaluator_batch
from helix.population import Candidate, EvalResult
from helix.proposals import (
    AllImprovements,
    EvaluatedProposal,
    ProposalAcceptanceCriterion,
    ProposalTask,
)


PADDED_IDS = ("rare", "other", "rare")


def _candidate(candidate_id: str) -> Candidate:
    return Candidate(
        id=candidate_id,
        worktree_path=f"/tmp/{candidate_id}",
        branch_name=f"helix/{candidate_id}",
        generation=1,
        parent_id="parent" if candidate_id != "parent" else None,
        parent_ids=[] if candidate_id == "parent" else ["parent"],
        operation="mutation",
    )


def _result(candidate_id: str, *, rare: float, other: float) -> EvalResult:
    return EvalResult(
        candidate_id=candidate_id,
        scores={},
        asi={},
        instance_scores={"rare": rare, "other": other},
    )


def _proposal(
    index: int,
    child: Candidate,
    parent_result: EvalResult,
    child_result: EvalResult,
) -> EvaluatedProposal:
    return EvaluatedProposal(
        task=ProposalTask(
            batch_index=index,
            parent_group_index=0,
            mutation_index=index,
            parent_candidate=_candidate("parent"),
            minibatch_ids=PADDED_IDS,
            reserved_child_id=child.id,
        ),
        parent_eval_result=parent_result,
        child_candidate=child,
        child_eval_result=child_result,
    )


def _strict_criterion() -> ProposalAcceptanceCriterion:
    return cast(ProposalAcceptanceCriterion, StrictImprovementAcceptance())


def test_padded_minibatch_multiplicity_can_flip_acceptance() -> None:
    """The repeated padding slot must carry the repeated example's score."""

    proposal = _proposal(
        0,
        _candidate("child"),
        _result("parent", rare=0.9, other=0.0),
        _result("child", rare=0.5, other=0.6),
    )

    assert proposal.subsample_scores_before == [0.9, 0.0, 0.9]
    assert proposal.subsample_scores_after == [0.5, 0.6, 0.5]
    assert proposal.improvement == pytest.approx(-0.2)
    assert not _strict_criterion().should_accept(proposal)


def _cached_result(
    candidate_id: str,
    *,
    cached_scores: dict[str, float],
    fresh_scores: dict[str, float],
) -> tuple[EvalResult, int, list[tuple[str, ...]]]:
    cache: EvaluationCache[object, str] = EvaluationCache()
    candidate_key = {"content_key": candidate_id, "split": "train"}
    for example_id, score in cached_scores.items():
        cache.put(candidate_key, example_id, None, score)

    fetched: list[tuple[str, ...]] = []

    def fetcher(example_ids: list[str]) -> list[str]:
        fetched.append(tuple(example_ids))
        return example_ids

    def evaluator(
        batch: list[str], _candidate_key: dict[str, str]
    ) -> tuple[list[object], list[float], None, None]:
        return [None] * len(batch), [fresh_scores[eid] for eid in batch], None, None

    _, scores_by_id, _, _, n_uncached = cache.evaluate_with_cache_full(
        candidate_key,
        list(PADDED_IDS),
        fetcher,
        evaluator,
    )
    return (
        EvalResult(
            candidate_id=candidate_id,
            scores={},
            asi={},
            instance_scores=scores_by_id,
        ),
        n_uncached,
        fetched,
    )


def test_partial_cache_collapse_does_not_remove_padding_weight() -> None:
    """A cache returns one dict entry per ID; the task restores slot weight."""

    parent_result, parent_uncached, parent_fetches = _cached_result(
        "parent",
        cached_scores={"rare": 0.8, "other": 0.0},
        fresh_scores={},
    )
    child_result, child_uncached, child_fetches = _cached_result(
        "child",
        cached_scores={"rare": 0.55},
        fresh_scores={"other": 0.6},
    )
    proposal = _proposal(
        0,
        _candidate("child"),
        parent_result,
        child_result,
    )

    assert parent_result.instance_scores == {"rare": 0.8, "other": 0.0}
    assert child_result.instance_scores == {"rare": 0.55, "other": 0.6}
    assert (parent_uncached, parent_fetches) == (0, [])
    assert (child_uncached, child_fetches) == (1, [("other",)])
    assert proposal.subsample_scores_before == [0.8, 0.0, 0.8]
    assert proposal.subsample_scores_after == [0.55, 0.6, 0.55]
    assert proposal.improvement == pytest.approx(0.1)


def test_reverse_completion_and_dedup_preserve_padded_selection() -> None:
    """Batch completion and follower reuse cannot erase repeated slot scores."""

    children = [_candidate(f"child-{index}") for index in range(3)]
    items = [
        EvalBatchItem(children[0], "shared", "train", PADDED_IDS),
        EvalBatchItem(children[1], "unique", "train", PADDED_IDS),
        EvalBatchItem(children[2], "shared", "train", PADDED_IDS),
    ]
    both_leaders_started = threading.Event()
    unique_finished = threading.Event()
    lock = threading.Lock()
    started: set[str] = set()
    completion_order: list[str] = []

    def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
        with lock:
            started.add(item.candidate.id)
            if len(started) == 2:
                both_leaders_started.set()
        assert both_leaders_started.wait(timeout=5)

        if item.candidate.id == "child-0":
            assert unique_finished.wait(timeout=5)
            result = _result(item.candidate.id, rare=0.55, other=0.6)
        else:
            result = _result(item.candidate.id, rare=0.65, other=0.3)

        completion_order.append(item.candidate.id)
        if item.candidate.id == "child-1":
            unique_finished.set()
        return result, len(PADDED_IDS)

    batch_results = run_evaluator_batch(items, runner, max_workers=2)

    assert completion_order == ["child-1", "child-0"]
    assert [result.deduplicated_from for result in batch_results] == [None, None, 0]
    assert [result.num_actual_evaluations for result in batch_results] == [3, 3, 0]
    assert all(result.result is not None for result in batch_results)

    parent_result = _result("parent", rare=0.8, other=0.0)
    proposals = [
        _proposal(index, child, parent_result, batch_result.result)
        for index, (child, batch_result) in enumerate(zip(children, batch_results))
        if batch_result.result is not None
    ]
    selected = AllImprovements().select(proposals, _strict_criterion())

    assert [item.proposal.child_candidate.id for item in selected] == [
        "child-0",
        "child-2",
    ]
    assert [item.improvement for item in selected] == pytest.approx([0.1, 0.1])
