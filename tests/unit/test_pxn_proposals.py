"""Tests for the P×N proposal batch: sampling, selection, and dedupe.

``num_parallel_proposals`` (P) picks how many parents an iteration proposes
from; ``mutations_per_parent`` (N) picks how many children each of those
parents gets.  ``proposal_selection`` then decides how many of the children
that clear the acceptance gate are actually promoted to full validation.

The first test in this file is a *golden trace*: it pins the exact ordered
sequence of side effects the default configuration produces, so the apply
phase restructure that P×N required cannot silently reorder budget charges,
snapshots or evaluations for runs that never opt in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from helix.config import (
    DatasetConfig,
    EvaluatorConfig,
    EvolutionConfig,
    HelixConfig,
    SeedlessConfig,
    WorktreeConfig,
)
from helix.evolution import run_evolution
from helix.population import Candidate, EvalResult, ParetoFrontier
from helix.proposals import (
    AcceptanceJudgement,
    AcceptanceMemo,
    EvaluatedProposal,
    GatedProposal,
    ScoreVectors,
    select_all_improvements,
    select_best_improvement,
    select_proposals,
    select_top_k,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/unit/test_evolution_minibatch.py)
# ---------------------------------------------------------------------------


def _make_candidate(cid: str = "g0-s0") -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=f"/tmp/helix/{cid}",
        branch_name=f"helix/{cid}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def _make_result(cid: str, scores: dict[str, float]) -> EvalResult:
    return EvalResult(
        candidate_id=cid,
        scores={},
        asi={},
        instance_scores=dict(scores),
    )


def _write_train_jsonl(path: Path, n: int = 6) -> Path:
    p = path / "train.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            f.write(json.dumps({"idx": i, "x": i}) + "\n")
    return p


def _make_config(
    train_path: Path,
    *,
    minibatch_size: int = 2,
    max_generations: int = 1,
    max_evaluations: int = 10_000,
    **evo: Any,
) -> HelixConfig:
    evo_kwargs: dict[str, Any] = dict(
        max_generations=max_generations,
        max_evaluations=max_evaluations,
        perfect_score_threshold=None,
        minibatch_size=minibatch_size,
        cache_evaluation=True,
        frontier_type="instance",
    )
    evo_kwargs.update(evo)
    return HelixConfig(
        objective="P×N test",
        evaluator=EvaluatorConfig(command="pytest -q"),
        dataset=DatasetConfig(val_size=None),
        seedless=SeedlessConfig(train_path=train_path),
        evolution=EvolutionConfig(**evo_kwargs),
        worktree=WorktreeConfig(),
    )


@pytest.fixture
def all_mocks(mocker: Any) -> dict[str, Any]:
    return {
        "create_seed_worktree": mocker.patch("helix.evolution.create_seed_worktree"),
        "run_evaluator": mocker.patch("helix.evolution.run_evaluator"),
        "mutate": mocker.patch("helix.evolution.mutate"),
        "merge": mocker.patch("helix.evolution.merge", return_value=None),
        "remove_worktree": mocker.patch("helix.evolution.remove_worktree"),
        "load_state": mocker.patch("helix.evolution.load_state", return_value=None),
        "save_state": mocker.patch("helix.evolution.save_state"),
        "init_base_dir": mocker.patch("helix.evolution.init_base_dir"),
        "_save_evaluation": mocker.patch("helix.evolution._save_evaluation"),
        "_load_evaluation": mocker.patch(
            "helix.evolution._load_evaluation", return_value=None
        ),
        "record_entry": mocker.patch("helix.evolution.record_entry"),
        "load_lineage": mocker.patch("helix.evolution.load_lineage", return_value={}),
        "find_merge_triplet": mocker.patch(
            "helix.evolution.find_merge_triplet", return_value=None
        ),
        "snapshot_candidate": mocker.patch("helix.evolution.snapshot_candidate"),
        "set_phase": mocker.patch("helix.evolution.set_phase"),
        "print_info": mocker.patch("helix.evolution.print_info"),
        "print_success": mocker.patch("helix.evolution.print_success"),
        "print_warning": mocker.patch("helix.evolution.print_warning"),
        "print_error": mocker.patch("helix.evolution.print_error"),
        "render_budget": mocker.patch("helix.evolution.render_budget"),
        "render_generation": mocker.patch("helix.evolution.render_generation"),
        "_check_evaluator_script_exists": mocker.patch(
            "helix.evolution._check_evaluator_script_exists"
        ),
    }


def _install_trace(all_mocks: dict[str, Any], seed_id: str) -> list[str]:
    """Record an ordered, human-readable log of the run's side effects.

    Only the effects the apply phase is responsible for sequencing are
    recorded: evaluations (the budget-charged unit), lineage writes, state
    saves, snapshots and worktree removals.
    """
    trace: list[str] = []

    def run_eval(
        candidate: Candidate,
        config: HelixConfig,
        split: str = "val",
        instance_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> EvalResult:
        ids = "-" if instance_ids is None else ",".join(instance_ids)
        trace.append(f"eval {candidate.id} {split} [{ids}]")
        if instance_ids is None:
            return _make_result(candidate.id, {"v1": 0.5})
        if split == "train":
            # Parent scores low, every child improves → all clear the gate.
            score = 0.3 if candidate.id == seed_id else 0.9
            return _make_result(candidate.id, {i: score for i in instance_ids})
        return _make_result(candidate.id, {i: 0.7 for i in instance_ids})

    all_mocks["run_evaluator"].side_effect = run_eval
    all_mocks["save_state"].side_effect = lambda *a, **k: trace.append("save_state")
    all_mocks["snapshot_candidate"].side_effect = lambda c, *a, **k: trace.append(
        f"snapshot {c.id}"
    )
    all_mocks["record_entry"].side_effect = lambda _p, entry, *a, **k: trace.append(
        f"lineage {entry.id}<-{entry.parent}"
    )
    all_mocks["remove_worktree"].side_effect = lambda c, *a, **k: trace.append(
        f"remove {c.id if isinstance(c, Candidate) else c}"
    )
    all_mocks["_save_evaluation"].side_effect = lambda _d, r, *a, **k: trace.append(
        f"frontier {r.candidate_id}"
    )
    return trace


def _apply_phase(trace: list[str]) -> list[str]:
    """Drop the train-minibatch evals, which run concurrently in the workers.

    Their completion order is thread-timing dependent and deliberately not a
    guarantee.  Everything left is sequenced by the apply phase and *is* a
    guarantee — that is what the golden trace pins.
    """
    return [e for e in trace if not e.startswith("eval ") or " val " in e]


def _worker_evals(trace: list[str]) -> list[str]:
    return sorted(e for e in trace if e.startswith("eval ") and " train " in e)


# ---------------------------------------------------------------------------
# 1. The default path must not move
# ---------------------------------------------------------------------------


class TestDefaultPathUnchanged:
    def test_all_improvements_n1_golden_trace(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Default config (P=2, N=1, all_improvements) keeps its exact ordering.

        Written and pinned against the pre-P×N implementation.  ``P=2`` is
        used rather than ``P=1`` because it is the smallest configuration in
        which gate and apply *interleave* across proposals — the exact
        property the ``best_improvement`` / ``top_k`` strategies had to break
        and that ``all_improvements`` must keep.  A two-phase
        gate-all-then-apply-all rewrite of the shared path would reorder this
        list; that is the regression this test exists to catch.
        """
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(kw["new_id"])

        trace = _install_trace(all_mocks, seed.id)

        config = _make_config(train_path, num_parallel_proposals=2)
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert _apply_phase(trace) == [
            "eval g0-s0 val [0,1,2,3,4,5]",
            "frontier g0-s0",
            "save_state",
            "lineage g0-s0<-None",
            # Proposal 0 is gated *and* fully applied before proposal 1 is
            # touched.  A gate-all-then-apply-all rewrite would hoist both
            # lineage writes above both val evals.
            "lineage g1-s1<-g0-s0",
            "save_state",
            "snapshot g1-s1",
            "eval g1-s1 val [0,1,2,3,4,5]",
            "frontier g1-s1",
            "lineage g1-s2<-g0-s0",
            "save_state",
            "snapshot g1-s2",
            "eval g1-s2 val [0,1,2,3,4,5]",
            "frontier g1-s2",
            "save_state",
        ]
        # Two parents sampled with replacement (both the seed here), two
        # disjoint minibatch draws, one child evaluated on each.
        assert _worker_evals(trace) == [
            "eval g0-s0 train [4,1]",
            "eval g0-s0 train [5,2]",
            "eval g1-s1 train [4,1]",
            "eval g1-s2 train [5,2]",
        ]


# ---------------------------------------------------------------------------
# 2. Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_top_k_requires_proposal_top_k(self) -> None:
        with pytest.raises(ValueError, match="proposal_top_k is required"):
            EvolutionConfig(proposal_selection="top_k")

    @pytest.mark.parametrize("k", [0, -1, 5])
    def test_top_k_must_be_within_batch_size(self, k: int) -> None:
        # P*N = 2*2 = 4, so 5 is out of range and 0 / -1 are degenerate.
        with pytest.raises(ValueError, match="proposal_top_k must be between 1 and"):
            EvolutionConfig(
                proposal_selection="top_k",
                proposal_top_k=k,
                num_parallel_proposals=2,
                mutations_per_parent=2,
            )

    def test_top_k_at_batch_size_is_accepted(self) -> None:
        cfg = EvolutionConfig(
            proposal_selection="top_k",
            proposal_top_k=4,
            num_parallel_proposals=2,
            mutations_per_parent=2,
        )
        assert cfg.proposal_top_k == 4

    @pytest.mark.parametrize(
        "strategy", ["all_improvements", "best_improvement"]
    )
    def test_top_k_rejected_for_other_strategies(self, strategy: str) -> None:
        """A bound that would silently do nothing is a config error, not a default."""
        with pytest.raises(ValueError, match="only valid when"):
            EvolutionConfig(proposal_selection=strategy, proposal_top_k=1)

    @pytest.mark.parametrize("n", [0, -1])
    def test_non_positive_mutations_per_parent_rejected(self, n: int) -> None:
        with pytest.raises(ValueError, match="mutations_per_parent must be >= 1"):
            EvolutionConfig(mutations_per_parent=n)

    @pytest.mark.parametrize("p", [0, -1])
    def test_non_positive_num_parallel_proposals_rejected(self, p: int) -> None:
        """Previously accepted: P<=0 planned an empty batch every iteration."""
        with pytest.raises(ValueError, match="num_parallel_proposals must be >= 1"):
            EvolutionConfig(num_parallel_proposals=p)

    @pytest.mark.parametrize("w", [0, -1])
    def test_non_positive_max_workers_rejected(self, w: int) -> None:
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            EvolutionConfig(max_workers=w)

    def test_max_workers_checked_before_num_parallel_proposals(self) -> None:
        """A zero worker count is rejected regardless of a valid P."""
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            EvolutionConfig(num_parallel_proposals=5, max_workers=0)

    def test_defaults_are_the_historical_single_proposal_shape(self) -> None:
        cfg = EvolutionConfig()
        assert cfg.mutations_per_parent == 1
        assert cfg.proposal_selection == "all_improvements"
        assert cfg.proposal_top_k is None


# ---------------------------------------------------------------------------
# 3. Selection functions in isolation
# ---------------------------------------------------------------------------


def _gated(order: int, improvement: float, child_id: str = "") -> GatedProposal:
    """A GatedProposal carrying just enough to be selected over."""
    child = _make_candidate(child_id or f"c{order}")
    ctx = (_make_candidate("p"), None, None, child.id)
    return GatedProposal(
        order=order,
        proposal=EvaluatedProposal(presample_ctx=ctx, parent_eval_result=None, child=child),  # type: ignore[arg-type]
        judgement=AcceptanceJudgement(
            accepted=True, before=[0.0], after=[improvement]
        ),
        gating_result=_make_result(child.id, {}),
    )


class TestSelectionFunctions:
    def test_all_improvements_keeps_everything_in_sampled_order(self) -> None:
        batch = [_gated(0, 0.5), _gated(1, 0.9), _gated(2, 0.1)]
        assert [g.order for g in select_all_improvements(batch)] == [0, 1, 2]

    def test_best_improvement_picks_the_largest(self) -> None:
        batch = [_gated(0, 0.5), _gated(1, 0.9), _gated(2, 0.1)]
        assert [g.order for g in select_best_improvement(batch)] == [1]

    def test_best_improvement_breaks_ties_toward_earliest_sampled(self) -> None:
        """Completion order must never decide a tie — sampled order does."""
        batch = [_gated(0, 0.4), _gated(1, 0.9), _gated(2, 0.9)]
        assert [g.order for g in select_best_improvement(batch)] == [1]

    def test_best_improvement_on_empty_batch(self) -> None:
        assert select_best_improvement([]) == []

    def test_top_k_picks_the_k_largest(self) -> None:
        batch = [_gated(0, 0.5), _gated(1, 0.9), _gated(2, 0.1), _gated(3, 0.7)]
        assert [g.order for g in select_top_k(batch, 2)] == [1, 3]

    def test_top_k_applies_best_first(self) -> None:
        """Survivors stay ranked, so a truncated batch keeps the best work."""
        batch = [_gated(0, 0.5), _gated(1, 0.9)]
        assert [g.order for g in select_top_k(batch, 2)] == [1, 0]

    def test_top_k_breaks_ties_toward_earliest_sampled(self) -> None:
        batch = [_gated(0, 0.9), _gated(1, 0.9), _gated(2, 0.9)]
        assert [g.order for g in select_top_k(batch, 2)] == [0, 1]

    def test_top_k_larger_than_batch_returns_everything(self) -> None:
        batch = [_gated(0, 0.5), _gated(1, 0.9)]
        assert sorted(g.order for g in select_top_k(batch, 10)) == [0, 1]

    def test_negative_improvements_still_rank(self) -> None:
        """A gated proposal always improved on *its own* parent; ranking is relative."""
        batch = [_gated(0, -0.5), _gated(1, -0.1)]
        assert [g.order for g in select_best_improvement(batch)] == [1]

    @pytest.mark.parametrize(
        "strategy,expected",
        [("all_improvements", [0, 1, 2]), ("best_improvement", [1])],
    )
    def test_dispatcher_routes_to_the_named_strategy(
        self, strategy: Any, expected: list[int]
    ) -> None:
        batch = [_gated(0, 0.5), _gated(1, 0.9), _gated(2, 0.1)]
        picked = select_proposals(batch, strategy=strategy, top_k=None)
        assert [g.order for g in picked] == expected

    def test_dispatcher_routes_top_k(self) -> None:
        batch = [_gated(0, 0.5), _gated(1, 0.9), _gated(2, 0.1)]
        picked = select_proposals(batch, strategy="top_k", top_k=1)
        assert [g.order for g in picked] == [1]


# ---------------------------------------------------------------------------
# 4. Acceptance memo
# ---------------------------------------------------------------------------


class _CountingCriterion:
    def __init__(self) -> None:
        self.calls = 0

    def should_accept(self, proposal: ScoreVectors) -> bool:
        self.calls += 1
        return sum(proposal.subsample_scores_after or []) > sum(
            proposal.subsample_scores_before or []
        )


class TestAcceptanceMemo:
    def test_repeated_consultation_runs_the_criterion_once(self) -> None:
        criterion = _CountingCriterion()
        memo = AcceptanceMemo(criterion)
        first = memo.judge(0, [0.1], [0.9])
        again = memo.judge(0, [0.1], [0.9])
        assert first is again
        assert criterion.calls == 1
        assert memo.criterion_calls == 1

    def test_distinct_slots_are_judged_independently(self) -> None:
        criterion = _CountingCriterion()
        memo = AcceptanceMemo(criterion)
        assert memo.judge(0, [0.1], [0.9]).accepted is True
        assert memo.judge(1, [0.9], [0.1]).accepted is False
        assert criterion.calls == 2

    def test_improvement_is_the_gate_margin(self) -> None:
        memo = AcceptanceMemo(_CountingCriterion())
        judgement = memo.judge(0, [0.1, 0.2], [0.5, 0.4])
        assert judgement.improvement == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 5. P×N planning and selection, end to end
# ---------------------------------------------------------------------------


def _scored_children(
    all_mocks: dict[str, Any], seed_id: str, child_scores: dict[str, float]
) -> list[str]:
    """Wire N children with per-child minibatch scores; return the frontier log."""
    all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(kw["new_id"])

    def run_eval(
        candidate: Candidate,
        config: HelixConfig,
        split: str = "val",
        instance_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> EvalResult:
        if instance_ids is None:
            return _make_result(candidate.id, {"v1": 0.5})
        if split == "train":
            score = 0.3 if candidate.id == seed_id else child_scores[candidate.id]
            return _make_result(candidate.id, {i: score for i in instance_ids})
        return _make_result(candidate.id, {i: 0.7 for i in instance_ids})

    all_mocks["run_evaluator"].side_effect = run_eval

    promoted: list[str] = []
    all_mocks["_save_evaluation"].side_effect = lambda _d, r, *a, **k: promoted.append(
        r.candidate_id
    )
    return promoted


class TestPxNPlanning:
    def test_n_children_per_parent_each_on_its_own_minibatch(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """P=1, N=3 proposes three children from one parent, on three minibatches."""
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(kw["new_id"])
        trace = _install_trace(all_mocks, seed.id)

        config = _make_config(
            train_path, num_parallel_proposals=1, mutations_per_parent=3
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        child_evals = [
            e for e in _worker_evals(trace) if not e.startswith("eval g0-s0 ")
        ]
        assert len(child_evals) == 3, (
            f"P=1 N=3 must propose 3 children; got {child_evals}"
        )
        # Each slot draws its own minibatch, so the three children are gated
        # on three different example sets.
        batches = {e.split("[")[1] for e in child_evals}
        assert len(batches) == 3, (
            f"Each of the N slots must draw its own minibatch; got {batches}"
        )
        # All three share the one sampled parent (parent-major planning).
        assert all("<-g0-s0" in e for e in trace if e.startswith("lineage g1-"))

    def test_p_times_n_is_the_batch_size(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """P=2, N=2 proposes four children."""
        train_path = _write_train_jsonl(tmp_path, n=12)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(kw["new_id"])
        trace = _install_trace(all_mocks, seed.id)

        config = _make_config(
            train_path, num_parallel_proposals=2, mutations_per_parent=2
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert sorted(e for e in trace if e.startswith("lineage g1-")) == [
            "lineage g1-s1<-g0-s0",
            "lineage g1-s2<-g0-s0",
            "lineage g1-s3<-g0-s0",
            "lineage g1-s4<-g0-s0",
        ]


class TestSelectionEndToEnd:
    def test_all_improvements_promotes_every_passer(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        promoted = _scored_children(
            all_mocks, seed.id, {"g1-s1": 0.4, "g1-s2": 0.9, "g1-s3": 0.6}
        )

        config = _make_config(
            train_path, num_parallel_proposals=1, mutations_per_parent=3
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert sorted(p for p in promoted if p != seed.id) == [
            "g1-s1",
            "g1-s2",
            "g1-s3",
        ]

    def test_best_improvement_promotes_only_the_largest(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Only the biggest gate margin reaches full validation and the frontier."""
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        promoted = _scored_children(
            all_mocks, seed.id, {"g1-s1": 0.4, "g1-s2": 0.9, "g1-s3": 0.6}
        )

        config = _make_config(
            train_path,
            num_parallel_proposals=1,
            mutations_per_parent=3,
            proposal_selection="best_improvement",
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert [p for p in promoted if p != seed.id] == ["g1-s2"]

    def test_top_k_promotes_exactly_k(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        promoted = _scored_children(
            all_mocks, seed.id, {"g1-s1": 0.4, "g1-s2": 0.9, "g1-s3": 0.6}
        )

        config = _make_config(
            train_path,
            num_parallel_proposals=1,
            mutations_per_parent=3,
            proposal_selection="top_k",
            proposal_top_k=2,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # The two largest margins, applied in sampled order.
        assert [p for p in promoted if p != seed.id] == ["g1-s2", "g1-s3"]

    def test_unselected_passers_are_removed_not_left_dangling(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """A child that cleared the gate but lost selection must be cleaned up."""
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _scored_children(
            all_mocks, seed.id, {"g1-s1": 0.4, "g1-s2": 0.9, "g1-s3": 0.6}
        )

        removed: list[str] = []
        all_mocks["remove_worktree"].side_effect = lambda c, *a, **k: removed.append(
            c.id if isinstance(c, Candidate) else str(c)
        )

        config = _make_config(
            train_path,
            num_parallel_proposals=1,
            mutations_per_parent=3,
            proposal_selection="best_improvement",
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert sorted(removed) == ["g1-s1", "g1-s3"]

    def test_selection_consults_one_judgement_per_proposal(
        self, tmp_path: Path, all_mocks: dict[str, Any], mocker: Any
    ) -> None:
        """The criterion runs once per proposal, not once per consultation.

        ``best_improvement`` has to know every proposal's gate outcome before
        it can promote any of them, and then works with the winner. Routing
        that through the memo keeps the number of underlying judgements equal
        to the batch size.
        """
        from helix.eval_policy import StrictImprovementAcceptance

        spy = mocker.spy(StrictImprovementAcceptance, "should_accept")

        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _scored_children(
            all_mocks, seed.id, {"g1-s1": 0.4, "g1-s2": 0.9, "g1-s3": 0.6}
        )

        config = _make_config(
            train_path,
            num_parallel_proposals=1,
            mutations_per_parent=3,
            proposal_selection="best_improvement",
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert spy.call_count == 3, (
            f"Expected one acceptance judgement per proposal (3); got "
            f"{spy.call_count}"
        )


class TestChildDedupe:
    def test_byte_identical_siblings_collapse_to_one_frontier_entry(
        self, tmp_path: Path, all_mocks: dict[str, Any], mocker: Any
    ) -> None:
        """Two siblings on the same tree must not both enter the frontier."""
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        promoted = _scored_children(
            all_mocks, seed.id, {"g1-s1": 0.9, "g1-s2": 0.9, "g1-s3": 0.6}
        )
        # s1 and s2 committed the same tree; s3 is its own.
        mocker.patch(
            "helix.evolution._candidate_content_key",
            side_effect=lambda c: {"g1-s1": "tree-a", "g1-s2": "tree-a"}.get(
                c.id, f"tree-{c.id}"
            ),
        )

        config = _make_config(
            train_path, num_parallel_proposals=1, mutations_per_parent=3
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert [p for p in promoted if p != seed.id] == ["g1-s1", "g1-s3"], (
            "The duplicate sibling must be dropped before frontier insertion"
        )

    def test_duplicate_survivor_is_the_earlier_sampled_slot(
        self, tmp_path: Path, all_mocks: dict[str, Any], mocker: Any
    ) -> None:
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _scored_children(all_mocks, seed.id, {"g1-s1": 0.9, "g1-s2": 0.9})
        mocker.patch(
            "helix.evolution._candidate_content_key", side_effect=lambda c: "same"
        )

        removed: list[str] = []
        all_mocks["remove_worktree"].side_effect = lambda c, *a, **k: removed.append(
            c.id if isinstance(c, Candidate) else str(c)
        )

        config = _make_config(
            train_path, num_parallel_proposals=1, mutations_per_parent=2
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert removed == ["g1-s2"]

    def test_single_slot_batches_skip_the_content_key_entirely(
        self, tmp_path: Path, all_mocks: dict[str, Any], mocker: Any
    ) -> None:
        """P=N=1 has no sibling to collide with, so it pays nothing for dedupe."""
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _scored_children(all_mocks, seed.id, {"g1-s1": 0.9})
        key_spy = mocker.patch(
            "helix.evolution._candidate_content_key", side_effect=lambda c: c.id
        )

        config = _make_config(
            train_path,
            num_parallel_proposals=1,
            mutations_per_parent=1,
            cache_evaluation=False,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert key_spy.call_count == 0


class TestMinibatchRepeatWarning:
    def test_warns_when_batch_exceeds_available_minibatches(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """P*N=4 over 6 examples at minibatch_size=2 leaves only 3 disjoint draws."""
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _scored_children(
            all_mocks,
            seed.id,
            {"g1-s1": 0.9, "g1-s2": 0.9, "g1-s3": 0.9, "g1-s4": 0.9},
        )

        config = _make_config(
            train_path,
            minibatch_size=2,
            num_parallel_proposals=2,
            mutations_per_parent=2,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        warnings = [str(c.args[0]) for c in all_mocks["print_warning"].call_args_list]
        assert any("minibatches will repeat within an iteration" in w for w in warnings), (
            f"Expected a minibatch-repeat warning; got {warnings}"
        )

    def test_no_warning_when_the_train_set_is_large_enough(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        train_path = _write_train_jsonl(tmp_path, n=20)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _scored_children(all_mocks, seed.id, {"g1-s1": 0.9, "g1-s2": 0.9})

        config = _make_config(
            train_path,
            minibatch_size=2,
            num_parallel_proposals=1,
            mutations_per_parent=2,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        warnings = [str(c.args[0]) for c in all_mocks["print_warning"].call_args_list]
        assert not any("minibatches will repeat" in w for w in warnings)


# ---------------------------------------------------------------------------
# 6. Dropped proposals stay accountable
# ---------------------------------------------------------------------------


class TestDroppedProposalsAreRecorded:
    """A dropped child already has a lineage entry; it must not point at nothing.

    A proposal that clears the gate gets its lineage entry written before
    selection runs.  If selection or dedupe then drops it, the drop has to
    land in ``attempts/`` like a gate rejection does — otherwise reading a
    run back shows a mutation that was attempted, charged, and then simply
    vanished.
    """

    def test_selection_drop_lands_in_the_attempts_ledger(
        self, tmp_path: Path, all_mocks: dict[str, Any], mocker: Any
    ) -> None:
        save_attempt = mocker.patch("helix.evolution._save_attempt_result")

        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _scored_children(
            all_mocks, seed.id, {"g1-s1": 0.4, "g1-s2": 0.9, "g1-s3": 0.6}
        )

        config = _make_config(
            train_path,
            num_parallel_proposals=1,
            mutations_per_parent=3,
            proposal_selection="best_improvement",
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        dropped = [
            c
            for c in save_attempt.call_args_list
            if c.kwargs.get("reason") == "proposal_selection"
        ]
        assert len(dropped) == 2, (
            "Both gate-passing losers must be recorded, not silently removed"
        )
        assert {c.args[1].candidate_id for c in dropped} == {"g1-s1", "g1-s3"}
        assert all(c.kwargs["status"] == "rejected" for c in dropped)

    def test_duplicate_drop_lands_in_the_attempts_ledger(
        self, tmp_path: Path, all_mocks: dict[str, Any], mocker: Any
    ) -> None:
        save_attempt = mocker.patch("helix.evolution._save_attempt_result")
        mocker.patch(
            "helix.evolution._candidate_content_key", side_effect=lambda c: "same"
        )

        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _scored_children(all_mocks, seed.id, {"g1-s1": 0.9, "g1-s2": 0.9})

        config = _make_config(
            train_path, num_parallel_proposals=1, mutations_per_parent=2
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        dropped = [
            c
            for c in save_attempt.call_args_list
            if c.kwargs.get("reason") == "duplicate_child"
        ]
        assert [c.args[1].candidate_id for c in dropped] == ["g1-s2"]

    def test_drop_reasons_are_distinguishable_from_gate_rejections(
        self, tmp_path: Path, all_mocks: dict[str, Any], mocker: Any
    ) -> None:
        """s1 loses on the criterion; s3 loses on selection. Different reasons."""
        save_attempt = mocker.patch("helix.evolution._save_attempt_result")

        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        # s1 scores below the parent's 0.3 → criterion rejection.
        _scored_children(
            all_mocks, seed.id, {"g1-s1": 0.1, "g1-s2": 0.9, "g1-s3": 0.6}
        )

        config = _make_config(
            train_path,
            num_parallel_proposals=1,
            mutations_per_parent=3,
            proposal_selection="best_improvement",
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        by_reason = {
            c.args[1].candidate_id: c.kwargs.get("reason")
            for c in save_attempt.call_args_list
        }
        assert by_reason["g1-s1"] == "minibatch_gate"
        assert by_reason["g1-s3"] == "proposal_selection"


# ---------------------------------------------------------------------------
# 7. Cost model and budget accounting under P×N
# ---------------------------------------------------------------------------


def _eval_log(all_mocks: dict[str, Any], seed_id: str, child_scores: dict[str, float]) -> list[tuple[str, str, int]]:
    """Wire scored children and return a log of (candidate_id, split, n_examples)."""
    calls: list[tuple[str, str, int]] = []
    all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(kw["new_id"])

    def run_eval(
        candidate: Candidate,
        config: HelixConfig,
        split: str = "val",
        instance_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> EvalResult:
        calls.append((candidate.id, split, 0 if instance_ids is None else len(instance_ids)))
        if instance_ids is None:
            return _make_result(candidate.id, {"v1": 0.5})
        if split == "train":
            score = 0.3 if candidate.id == seed_id else child_scores[candidate.id]
            return _make_result(candidate.id, {i: score for i in instance_ids})
        return _make_result(candidate.id, {i: 0.7 for i in instance_ids})

    all_mocks["run_evaluator"].side_effect = run_eval
    return calls


class TestCostModel:
    def test_selection_saves_validation_not_gate_work(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Every proposal is still evaluated on its minibatch.

        ``best_improvement`` is a validation-cost knob, not a gate-cost knob:
        all P*N children are still mutated and gated, and only the winner is
        promoted to full validation.  A reader sizing a run needs this to be
        explicit — picking ``best_improvement`` does not make N cheap.
        """
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        calls = _eval_log(
            all_mocks, seed.id, {"g1-s1": 0.4, "g1-s2": 0.9, "g1-s3": 0.6}
        )

        config = _make_config(
            train_path,
            num_parallel_proposals=1,
            mutations_per_parent=3,
            proposal_selection="best_improvement",
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        child_gate = [c for c in calls if c[0] != seed.id and c[1] == "train"]
        child_val = [c for c in calls if c[0] != seed.id and c[1] == "val"]
        assert len(child_gate) == 3, "all N children must still be gated"
        assert len(child_val) == 1, "only the selected child reaches full val"
        assert all_mocks["mutate"].call_count == 3, "all N mutations still run"

    def test_parent_draws_equal_p_not_p_times_n(
        self, tmp_path: Path, all_mocks: dict[str, Any], mocker: Any
    ) -> None:
        """P=3, N=2 draws 3 parents, not 6 — N is 'more tries at this parent'."""
        train_path = _write_train_jsonl(tmp_path, n=20)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _eval_log(all_mocks, seed.id, {f"g1-s{i}": 0.9 for i in range(1, 7)})

        spy = mocker.spy(ParetoFrontier, "select_parent")

        config = _make_config(
            train_path, num_parallel_proposals=3, mutations_per_parent=2
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert spy.call_count == 3, (
            f"Expected P=3 parent draws for a 3x2 batch; got {spy.call_count}"
        )

    def test_budget_ledger_matches_examples_actually_evaluated(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """The ledger equals independently counted evaluator work (caching off)."""
        train_path = _write_train_jsonl(tmp_path, n=20)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        calls = _eval_log(all_mocks, seed.id, {f"g1-s{i}": 0.9 for i in range(1, 5)})

        config = _make_config(
            train_path,
            num_parallel_proposals=2,
            mutations_per_parent=2,
            cache_evaluation=False,
            max_evaluations=100_000,
        )
        result = run_evolution(config, tmp_path, tmp_path / ".helix")

        assert result.budget.evaluations == sum(n for _, _, n in calls), (
            f"budget ledger {result.budget.evaluations} != "
            f"{sum(n for _, _, n in calls)} examples actually evaluated"
        )

    def test_budget_is_monotonic_across_a_pxn_batch(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Charges are sequential, so the ledger never decreases mid-batch."""
        train_path = _write_train_jsonl(tmp_path, n=20)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _eval_log(all_mocks, seed.id, {f"g1-s{i}": 0.9 for i in range(1, 5)})

        snapshots: list[int] = []
        all_mocks["save_state"].side_effect = lambda state, _p: snapshots.append(
            state.budget.evaluations
        )

        config = _make_config(
            train_path, num_parallel_proposals=2, mutations_per_parent=2
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert snapshots
        assert snapshots == sorted(snapshots), f"ledger went backwards: {snapshots}"

    def test_budget_overshoot_is_bounded_by_the_batch_size(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """The documented overshoot bound, enforced.

        ``max_evaluations`` is checked between slots, so a batch already
        dispatched runs to completion.  The engine documents the resulting
        overshoot as at most P*N parent-minibatch evals plus P*N child evals;
        this pins that bound rather than leaving it as prose.
        """
        train_path = _write_train_jsonl(tmp_path, n=20)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        _eval_log(all_mocks, seed.id, {f"g1-s{i}": 0.9 for i in range(1, 30)})

        p, n, mb = 2, 2, 2
        cap = 30
        config = _make_config(
            train_path,
            minibatch_size=mb,
            num_parallel_proposals=p,
            mutations_per_parent=n,
            max_generations=20,
            max_evaluations=cap,
            cache_evaluation=False,
        )
        result = run_evolution(config, tmp_path, tmp_path / ".helix")

        bound = cap + 2 * p * n * mb
        assert result.budget.evaluations <= bound, (
            f"overshoot {result.budget.evaluations - cap} exceeds the documented "
            f"bound of 2*P*N*minibatch_size = {2 * p * n * mb}"
        )


class TestDeterminism:
    def test_identical_config_produces_an_identical_run(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        """Same seed, same batch — completion timing must not leak in."""

        def _one_run(run_idx: int) -> list[str]:
            mocks = {
                k: mocker.patch(f"helix.evolution.{k}")
                for k in (
                    "create_seed_worktree", "run_evaluator", "mutate", "remove_worktree",
                    "save_state", "init_base_dir", "_save_evaluation", "record_entry",
                    "snapshot_candidate", "set_phase", "print_info", "print_success",
                    "print_warning", "print_error", "render_budget", "render_generation",
                    "_check_evaluator_script_exists",
                )
            }
            mocks["merge"] = mocker.patch("helix.evolution.merge", return_value=None)
            mocks["load_state"] = mocker.patch(
                "helix.evolution.load_state", return_value=None
            )
            mocks["_load_evaluation"] = mocker.patch(
                "helix.evolution._load_evaluation", return_value=None
            )
            mocks["load_lineage"] = mocker.patch(
                "helix.evolution.load_lineage", return_value={}
            )
            mocks["find_merge_triplet"] = mocker.patch(
                "helix.evolution.find_merge_triplet", return_value=None
            )
            run_dir = tmp_path / f"run{run_idx}"
            run_dir.mkdir()
            train_path = _write_train_jsonl(run_dir, n=12)
            seed = _make_candidate("g0-s0")
            mocks["create_seed_worktree"].return_value = seed
            trace = _install_trace(mocks, seed.id)
            mocks["mutate"].side_effect = lambda **kw: _make_candidate(kw["new_id"])

            config = _make_config(
                train_path,
                num_parallel_proposals=2,
                mutations_per_parent=2,
                proposal_selection="top_k",
                proposal_top_k=2,
            )
            run_evolution(config, run_dir, run_dir / ".helix")
            return _apply_phase(trace) + _worker_evals(trace)

        assert _one_run(1) == _one_run(2)
