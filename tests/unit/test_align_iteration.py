"""Regression tests for GEPA parity fixes on branch ``topic/align-iteration``.

Covers three audit findings:

- **M1** (audit-init-engine.md B1/B2, audit-mutation.md C1) — perfect-score
  hitting the threshold skips ONE proposal (``continue``) instead of
  terminating the run (``break``), and uses the GEPA per-example
  ``all(s >= threshold)`` criterion instead of mean ``aggregate_score()``.
  GEPA reference: ``reflective_mutation.py:308-327``.
- **M2** (audit-init-engine.md B3) — merge-branch fail-fast paths fall
  through to reflective mutation instead of consuming the iteration.
  Only an actually-evaluated merge (accepted or rejected) consumes the
  iteration.  GEPA reference: ``engine.py:664-741``.
- **MODERATE D** (audit-mutation.md C3) — legacy (no-minibatch) mutation
  gating uses sum-score acceptance only (GEPA ``engine.py:287-303`` /
  ``reflective_mutation.py:420``); the old ``degrades()`` pre-check is
  removed so the legacy path matches the minibatch path.

All tests run with ``val_stage_size=None`` (default) to exercise the
GEPA-parity configuration specified in the task directive.
"""
from __future__ import annotations

from helix.evolution import run_evolution
from helix.trace import EventType, TRACE

from tests.unit.test_evolution import (  # type: ignore[import-untyped]
    all_mocks,  # noqa: F401 — re-exported pytest fixture
    make_candidate,
    make_config,
    make_eval_result,
)


# ---------------------------------------------------------------------------
# M1 — perfect-score termination → per-proposal skip
# ---------------------------------------------------------------------------


class TestPerfectScoreContinues:
    """GEPA parity (M1) — audit-init-engine.md B1.

    Pre-fix: perfect-score set ``_budget_break=True`` + broke the inner loop,
    and the outer ``if _budget_break and not proposal_contexts: break`` exited
    the generation loop entirely after gen 1.  Post-fix: ``continue`` skips
    the single proposal; the outer generation loop runs to ``max_generations``.
    """

    def test_perfect_score_does_not_terminate_run(
        self, mocker, tmp_path, all_mocks
    ):
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            return make_eval_result(candidate.id, {"i1": 1.0, "i2": 1.0})

        all_mocks["run_evaluator"].side_effect = run_eval

        # val_stage_size stays at its default (None) — hard-constraint
        # GEPA-parity mode from the align-iteration task directive.
        config = make_config(
            max_generations=5,
            perfect_score_threshold=1.0,
            max_evaluations=100_000,
        )

        with TRACE.record() as events:
            run_evolution(config, tmp_path, tmp_path / ".helix")

        iter_starts = [e for e in events if e.type == EventType.ITER_START]
        assert len(iter_starts) == 5, (
            f"Expected 5 ITER_START events (one per generation after M1 "
            f"fix); got {len(iter_starts)}.  Pre-fix behaviour emitted only "
            f"1 because perfect-score set _budget_break and exited the "
            f"outer loop after gen 1 (evolution.py previously did "
            f"`_budget_break = True; break` at the perfect-score check)."
        )
        # Every parent (the seed, reused across gens) reports perfect
        # scores, so mutation is skipped in every generation — per
        # GEPA reflective_mutation.py:308-327 the proposal is dropped
        # but the outer iteration loop continues.
        all_mocks["mutate"].assert_not_called()

    def test_perfect_score_uses_per_example_all_not_mean(
        self, mocker, tmp_path, all_mocks
    ):
        """GEPA parity (M1) — audit-init-engine.md B2.

        Pre-fix: criterion was ``aggregate_score() >= threshold`` (mean),
        which fires on ``{0.5, 1.0, 1.0}`` at ``threshold=0.8`` because
        the mean is 0.83.  GEPA reflective_mutation.py:311 uses
        ``all(s >= perfect_score)`` — the 0.5 per-example score would
        prevent the skip, so mutation must proceed.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                # Child is irrelevant to the M1 criterion; we just need it
                # to pass gating so the run completes cleanly.
                return make_eval_result(
                    "g1-s1", {"i1": 0.6, "i2": 1.0, "i3": 1.0}
                )
            # Parent seed: mean = 0.833 >= 0.8 (old check fires) but
            # min = 0.5 < 0.8 (GEPA all() check does NOT fire).
            return make_eval_result(
                candidate.id, {"i1": 0.5, "i2": 1.0, "i3": 1.0}
            )

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            perfect_score_threshold=0.8,
            max_evaluations=100_000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Mean-based pre-fix check would have skipped; GEPA all() check
        # keeps mutation live (0.5 < 0.8 per-example).
        all_mocks["mutate"].assert_called_once()


# ---------------------------------------------------------------------------
# M2 — merge-branch fallthrough
# ---------------------------------------------------------------------------


class TestMergeFallthroughToMutation:
    """GEPA parity (M2) — audit-init-engine.md B3.

    When the merge gate is entered but no merge is actually evaluated
    (no triplet, pair already attempted, overlap floor fails, merge op
    failed, tamper-reject pre-eval), GEPA falls through to reflective
    mutation in the SAME iteration (engine.py:741-742).  Pre-fix HELIX
    ``continue``d on every merge-gate entry regardless of whether an
    attempt happened.
    """

    def test_merge_op_failure_falls_through_to_mutation(
        self, mocker, tmp_path, all_mocks
    ):
        """``merge()`` returns None (Claude Code / subprocess failure) →
        no merged candidate instantiated → no eval → GEPA falls through
        to reflective mutation (audit-init-engine.md B3).

        Pre-fix HELIX: ``print_error`` fell through to the end-of-merge
        ``save_state/update/continue`` at evolution.py ~1343 (INSIDE the
        ``if triplet is not None`` nesting), consuming the iteration.
        """
        seed = make_candidate("g0-s0")
        child1 = make_candidate("g1-s1", generation=1)
        child2 = make_candidate("g2-s1", generation=2)
        all_mocks["create_seed_worktree"].return_value = seed
        children = iter([child1, child2])
        all_mocks["mutate"].side_effect = lambda *a, **kw: next(
            children, None
        )

        # A valid triplet is returned, so the merge attempt proceeds
        # to calling ``merge(...)``.  But merge(...) returns None, so
        # no merged candidate is created and no eval runs.
        all_mocks["find_merge_triplet"].return_value = (
            "g0-s0", "g1-s1", "g0-s0",
        )
        all_mocks["merge"].return_value = None

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g0-s0":
                return make_eval_result(candidate.id, {"i1": 0.5, "i2": 0.8})
            # Children + seed complementary on instances → both remain
            # non-dominated → merge candidate pool size >= 2.  Child sum
            # 1.4 > seed sum 1.3 → acceptance passes in gen 1.
            return make_eval_result(candidate.id, {"i1": 0.9, "i2": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=2,
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            max_evaluations=100_000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Gen 1: mutation accepted → merges_due=1, last_iter_found=True.
        # Gen 2: merge gate entered, merge(...) returns None (no eval),
        #        post-fix falls through to reflective mutation →
        #        2 mutate calls.  Pre-fix: the end-of-merge ``continue``
        #        fired → only 1 mutate call (gen 1 only).
        assert all_mocks["mutate"].call_count == 2, (
            f"Expected 2 mutate calls (gen 1 + gen 2 after merge-op "
            f"failure falls through); got {all_mocks['mutate'].call_count}. "
            f"Pre-fix: 1 (merge branch consumed gen 2 via the "
            f"end-of-merge `continue` even though ``merged is None`` "
            f"meant no eval happened — audit-init-engine.md B3)."
        )
        # Sanity: merge was attempted (find_merge_triplet returned a
        # triplet, merge(...) was called), but returned None.
        all_mocks["merge"].assert_called_once()

    def test_merge_actually_attempted_still_consumes_iteration(
        self, mocker, tmp_path, all_mocks
    ):
        """Guard rail: when a merge IS attempted (eval runs), the iteration
        is still consumed (GEPA engine.py:719 on accept, 737 on reject).

        Failing this test would mean the M2 fix over-corrected and let
        mutation also run on the same iteration as an actual merge.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        merged = make_candidate("g2-m1", generation=2)
        children = iter([child])
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].side_effect = lambda *a, **kw: next(
            children, None
        )
        all_mocks["merge"].return_value = merged
        all_mocks["find_merge_triplet"].return_value = (
            "g0-s0", "g1-s1", "g0-s0",
        )

        def run_eval(candidate, config, split=None, instance_ids=None, **kwargs):
            # Instance ids round-trip through helix_batch.json as opaque
            # strings; these happen to be "1"/"2" for legibility.
            if candidate.id == "g0-s0":
                scores = {"1": 0.5, "2": 0.8}
            elif candidate.id == "g1-s1":
                scores = {"1": 0.9, "2": 0.5}
            else:  # merged
                scores = {"1": 1.0, "2": 1.0}
            if instance_ids is not None:
                scores = {k: scores.get(k, 0.0) for k in instance_ids}
            return make_eval_result(candidate.id, scores)

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=2,
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            max_evaluations=100_000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Gen 1: 1 mutate call (seed → child).  Gen 2: merge attempted
        # AND accepted, so mutation does NOT run in gen 2.  Total: 1.
        assert all_mocks["mutate"].call_count == 1, (
            f"When a merge is actually attempted in gen 2, mutation must "
            f"NOT also run (GEPA engine.py:719 `continue`).  Got "
            f"{all_mocks['mutate'].call_count} mutate calls."
        )
        all_mocks["merge"].assert_called_once()


# ---------------------------------------------------------------------------
# MODERATE D — legacy (no-minibatch) gating uses sum-score acceptance
# ---------------------------------------------------------------------------


class TestLegacyGatingUsesSumOnly:
    """GEPA parity (MODERATE D) — audit-mutation.md C3.

    GEPA has a single acceptance path: ``should_accept(proposal, state)``
    on sum-score (engine.py:287-303, reflective_mutation.py:420).  HELIX's
    legacy (no-train-loader) path previously ran ``degrades()`` as a
    pre-check, applying a tolerance that GEPA does not have.  After the
    fix, ``degrades()`` is not consulted on the legacy path; the same
    ``acceptance`` criterion used by the minibatch path gates everything.
    """

    def test_degrades_is_not_called_in_legacy_path(
        self, mocker, tmp_path, all_mocks, monkeypatch
    ):
        from helix import evolution as _evo

        called: list[int] = []

        def _boom(
            *args: object, **kwargs: object
        ) -> bool:  # pragma: no cover — failure path
            called.append(1)
            raise AssertionError(
                "degrades() must NOT be called in legacy gating "
                "(GEPA parity MODERATE D — audit-mutation.md C3)."
            )

        monkeypatch.setattr(_evo, "degrades", _boom)

        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                return make_eval_result("g1-s1", {"i1": 0.9})
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            max_evaluations=100_000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert called == [], (
            "Legacy gating path called degrades() — MODERATE D fix failed."
        )

    def test_legacy_accepts_strict_improvement(
        self, mocker, tmp_path, all_mocks
    ):
        """GEPA sum-score acceptance requires strict improvement only.
        A child with parent_sum=0.5 child_sum=0.55 must be accepted.

        Historical note: the pre-fix legacy path called ``degrades()``
        with a ``gating_threshold`` config knob (now removed); if that
        threshold was negative, the pre-fix path would have rejected
        this same child.  Post-fix the sum-score criterion is uniform
        and gating_threshold is gone.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                return make_eval_result("g1-s1", {"i1": 0.55})
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            max_evaluations=100_000,
        )
        best = run_evolution(config, tmp_path, tmp_path / ".helix")

        # Strict sum-score acceptance: 0.55 > 0.5 → child becomes best.
        assert best.id == "g1-s1", (
            f"Expected child to be accepted under GEPA strict-sum "
            f"acceptance; got best={best.id}."
        )
