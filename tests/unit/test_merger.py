"""Unit tests for helix.merger."""

from __future__ import annotations

import random
from pathlib import Path

from helix.population import Candidate
from helix.config import HelixConfig, EvaluatorConfig
from helix.mutator import MutationError
from helix.merger import (
    build_merge_prompt,
    merge,
    select_eval_subsample_for_merged_program,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candidate(cid: str = "g0-s0", worktree_path: str = "/tmp/wt") -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=worktree_path,
        branch_name=f"helix/{cid}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def make_config(objective: str = "Pass all tests") -> HelixConfig:
    return HelixConfig(
        objective=objective,
        evaluator=EvaluatorConfig(command="pytest -q"),
    )


# ---------------------------------------------------------------------------
# Tests: build_merge_prompt
# ---------------------------------------------------------------------------


class TestBuildMergePrompt:
    def test_contains_objective(self):
        prompt = build_merge_prompt("Improve sort speed", "")
        assert "Improve sort speed" in prompt

    def test_contains_diff(self):
        diff = "+added_line = True"
        prompt = build_merge_prompt("goal", diff)
        assert "+added_line = True" in prompt

    def test_contains_background(self):
        prompt = build_merge_prompt("goal", "", background="special bg")
        assert "special bg" in prompt

    def test_background_section_omitted_when_none(self):
        """GEPA parity: empty optional inputs skip the section entirely
        instead of emitting a placeholder.  Mirrors GEPA's
        ``_build_reflection_prompt_template`` in ``gepa_launcher.py``,
        which only appends a section when its content is non-empty.
        """
        prompt = build_merge_prompt("goal", "")
        assert "## Background / Context" not in prompt
        assert "no additional background" not in prompt

    def test_contains_execution_instructions(self):
        prompt = build_merge_prompt("goal", "")
        assert "Task instructions:" in prompt

    def test_no_score_sections(self):
        """The parents' results are validation results, so the prompt
        carries no scores at all (see
        ``test_merge_prompt_carries_no_validation_ids_or_scores``)."""
        prompt = build_merge_prompt("goal", "some diff")
        assert "Strengths" not in prompt
        assert "score" not in prompt.lower().split("## your task")[0]

    def test_diff_section_omitted_when_empty(self):
        """An empty diff → ``## Diff (B relative to A)`` section is
        skipped entirely (no ``"(no diff — candidates are identical)"``
        placeholder).  Upstream callers should not invoke merge on
        identical candidates; this test pins the prompt-level fallback.
        """
        prompt = build_merge_prompt("goal", "")
        assert "## Diff" not in prompt
        assert "identical" not in prompt
        assert "no diff" not in prompt


class TestBuildMergePromptTwoDiff:
    """GEPA-parity two-diff form: when ``ancestor_id`` and both
    ancestor-relative diffs are supplied, the prompt emits two labelled
    sections instead of the single A↔B diff.  Mirrors GEPA's
    three-way attribution reasoning
    (``gepa/proposer/merge.py::sample_and_attempt_merge_programs_by_common_predictors``)
    at the file-hunk level — agent reads off A's contribution and B's
    contribution directly instead of inferring three-way info from a
    two-way comparison.
    """

    def test_emits_two_ancestor_relative_sections(self):
        prompt = build_merge_prompt(
            "goal", "",
            ancestor_id="g0-s0",
            diff_a_from_ancestor="+self.bn = nn.BatchNorm1d(...)\n",
            diff_b_from_ancestor="+if epoch < 10: lr = 0.001\n",
        )
        assert "## Diff: Candidate A relative to common ancestor g0-s0" in prompt
        assert "## Diff: Candidate B relative to common ancestor g0-s0" in prompt
        assert "self.bn = nn.BatchNorm1d" in prompt
        assert "if epoch < 10" in prompt
        # Fallback section MUST NOT also render — the agent should see
        # exactly the two ancestor-relative diffs, not three sections.
        assert "## Diff (B relative to A)" not in prompt

    def test_two_diff_form_uses_two_diff_task_block(self):
        """The two-diff form pairs the ancestor-relative diffs with a
        task block that explicitly tells the agent A's contribution is
        already in the working tree (so it doesn't re-apply it) and B's
        contribution is what needs to be brought in.  Pinning this
        substring prevents a future refactor from accidentally swapping
        back to the single-diff task block, which would confuse the
        agent about whether A's diff needs to be re-applied.
        """
        prompt = build_merge_prompt(
            "goal", "",
            ancestor_id="g0-s0",
            diff_a_from_ancestor="+a contribution\n",
            diff_b_from_ancestor="+b contribution\n",
        )
        assert "A's contribution (the hunks shown in" in prompt
        # The single-diff task block's "Apply the changes from Candidate B"
        # phrasing must NOT appear in the two-diff form.
        assert "Apply the changes from\nCandidate B that are beneficial" not in prompt

    def test_single_diff_form_uses_single_diff_task_block(self):
        """Companion to the two-diff task-block test: the single-diff
        fallback must keep the legacy task framing intact.
        """
        prompt = build_merge_prompt("goal", "+some diff")
        assert "Apply the changes from\nCandidate B" in prompt
        # The two-diff task block's phrasing must NOT appear in the
        # single-diff form.
        assert "A's contribution (the hunks shown in" not in prompt

    def test_single_diff_fallback_when_ancestor_missing(self):
        """No ancestor + no ancestor diffs → single A↔B diff form
        (backward-compatible default).
        """
        prompt = build_merge_prompt("goal", "+some diff content")
        assert "## Diff (B relative to A)" in prompt
        assert "+some diff content" in prompt
        # Two-diff section headers must NOT appear in the fallback path.
        assert "relative to common ancestor" not in prompt

    def test_single_diff_fallback_when_ancestor_id_only(self):
        """Defensive: an ``ancestor_id`` without both diffs falls back
        to the single A↔B path.  Prevents a half-configured caller from
        emitting an "ancestor header pointing to nothing" prompt.
        """
        prompt = build_merge_prompt(
            "goal", "+legacy diff",
            ancestor_id="g0-s0",
            diff_a_from_ancestor=None,
            diff_b_from_ancestor=None,
        )
        assert "## Diff (B relative to A)" in prompt
        assert "+legacy diff" in prompt
        assert "relative to common ancestor" not in prompt

    def test_two_diff_form_omits_empty_side(self):
        """If one of the ancestor-relative diffs is empty (one parent
        didn't change anything relative to the ancestor), only the
        non-empty side renders.  Edge case: a parent might "improve" via
        metadata changes that ``git diff`` doesn't see.
        """
        prompt = build_merge_prompt(
            "goal", "",
            ancestor_id="g0-s0",
            diff_a_from_ancestor="+self.bn = nn.BatchNorm1d(...)\n",
            diff_b_from_ancestor="",
        )
        assert "## Diff: Candidate A relative to common ancestor g0-s0" in prompt
        assert "## Diff: Candidate B relative to common ancestor g0-s0" not in prompt


# ---------------------------------------------------------------------------
# Tests: merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_returns_candidate_on_success(self, mocker):
        ca = make_candidate("g0-s0")
        cb = make_candidate("g0-s1")
        config = make_config()

        child = make_candidate("g1-m0", "/tmp/g1-m0")
        mocker.patch("helix.merger.clone_candidate", return_value=child)
        mocker.patch("helix.merger.get_diff", return_value="+x = 1")
        mocker.patch("helix.merger.invoke_claude_code", return_value=({"result": "ok"}, {}))
        mocker.patch("helix.merger.snapshot_candidate", return_value="abc123")
        mocker.patch("helix.merger.remove_worktree")

        result = merge(ca, cb, "g1-m0", config, Path("/tmp"), project_root=Path("/tmp"))

        assert result is child

    def test_sets_operation_to_merge(self, mocker):
        ca = make_candidate("g0-s0")
        cb = make_candidate("g0-s1")
        config = make_config()

        child = make_candidate("g1-m0")
        mocker.patch("helix.merger.clone_candidate", return_value=child)
        mocker.patch("helix.merger.get_diff", return_value="")
        mocker.patch("helix.merger.invoke_claude_code", return_value=({}, {}))
        mocker.patch("helix.merger.snapshot_candidate", return_value="sha")
        mocker.patch("helix.merger.remove_worktree")

        result = merge(ca, cb, "g1-m0", config, Path("/tmp"), project_root=Path("/tmp"))

        assert result.operation == "merge"

    def test_sets_parent_ids(self, mocker):
        ca = make_candidate("g0-s0")
        cb = make_candidate("g0-s1")
        config = make_config()

        child = make_candidate("g1-m0")
        mocker.patch("helix.merger.clone_candidate", return_value=child)
        mocker.patch("helix.merger.get_diff", return_value="")
        mocker.patch("helix.merger.invoke_claude_code", return_value=({}, {}))
        mocker.patch("helix.merger.snapshot_candidate", return_value="sha")
        mocker.patch("helix.merger.remove_worktree")

        result = merge(ca, cb, "g1-m0", config, Path("/tmp"), project_root=Path("/tmp"))

        assert "g0-s0" in result.parent_ids
        assert "g0-s1" in result.parent_ids

    def test_returns_none_on_mutation_error(self, mocker):
        ca = make_candidate("g0-s0")
        cb = make_candidate("g0-s1")
        config = make_config()

        child = make_candidate("g1-m0")
        mocker.patch("helix.merger.clone_candidate", return_value=child)
        mocker.patch("helix.merger.get_diff", return_value="some diff")
        mocker.patch(
            "helix.merger.invoke_claude_code",
            side_effect=MutationError("timeout"),
        )
        mock_remove = mocker.patch("helix.merger.remove_worktree")
        mocker.patch("helix.merger.snapshot_candidate")

        result = merge(ca, cb, "g1-m0", config, Path("/tmp"), project_root=Path("/tmp"))

        assert result is None
        mock_remove.assert_called_once_with(child)

    def test_removes_worktree_on_failure(self, mocker):
        ca = make_candidate("g0-s0")
        cb = make_candidate("g0-s1")
        config = make_config()

        child = make_candidate("g1-m0")
        mocker.patch("helix.merger.clone_candidate", return_value=child)
        mocker.patch("helix.merger.get_diff", return_value="")
        mocker.patch(
            "helix.merger.invoke_claude_code",
            side_effect=MutationError("bad json"),
        )
        mock_remove = mocker.patch("helix.merger.remove_worktree")
        mocker.patch("helix.merger.snapshot_candidate")

        merge(ca, cb, "g1-m0", config, Path("/tmp"), project_root=Path("/tmp"))

        mock_remove.assert_called_once_with(child)

    def test_snapshot_not_called_by_merge_on_success(self, mocker):
        """merge() must NOT call snapshot_candidate — the caller owns that step.

        Callers (evolution.py) must call save_state() BEFORE snapshot_candidate()
        so that state is persisted even if the commit step crashes.
        """
        ca = make_candidate("g0-s0")
        cb = make_candidate("g0-s1")
        config = make_config()

        child = make_candidate("g1-m0")
        mocker.patch("helix.merger.clone_candidate", return_value=child)
        mocker.patch("helix.merger.get_diff", return_value="")
        mocker.patch("helix.merger.invoke_claude_code", return_value=({}, {}))
        mock_snapshot = mocker.patch("helix.merger.snapshot_candidate", return_value="sha")
        mocker.patch("helix.merger.remove_worktree")

        result = merge(ca, cb, "g1-m0", config, Path("/tmp"), project_root=Path("/tmp"))

        # merge() returns the child but does NOT snapshot internally
        assert result is child
        mock_snapshot.assert_not_called()

    def test_snapshot_not_called_on_failure(self, mocker):
        ca = make_candidate("g0-s0")
        cb = make_candidate("g0-s1")
        config = make_config()

        child = make_candidate("g1-m0")
        mocker.patch("helix.merger.clone_candidate", return_value=child)
        mocker.patch("helix.merger.get_diff", return_value="")
        mocker.patch(
            "helix.merger.invoke_claude_code",
            side_effect=MutationError("fail"),
        )
        mock_snapshot = mocker.patch("helix.merger.snapshot_candidate")
        mocker.patch("helix.merger.remove_worktree")

        merge(ca, cb, "g1-m0", config, Path("/tmp"), project_root=Path("/tmp"))

        mock_snapshot.assert_not_called()

    def test_passes_background_to_prompt(self, mocker):
        ca = make_candidate("g0-s0")
        cb = make_candidate("g0-s1")
        config = make_config()

        child = make_candidate("g1-m0")
        mocker.patch("helix.merger.clone_candidate", return_value=child)
        mocker.patch("helix.merger.get_diff", return_value="")
        mock_invoke = mocker.patch("helix.merger.invoke_claude_code", return_value=({}, {}))
        mocker.patch("helix.merger.snapshot_candidate", return_value="sha")
        mocker.patch("helix.merger.remove_worktree")

        merge(ca, cb, "g1-m0", config, Path("/tmp"), project_root=Path("/tmp"), background="unique_context_xyz")

        prompt_arg = mock_invoke.call_args[0][1]
        assert "unique_context_xyz" in prompt_arg

    def test_ancestor_triggers_two_diff_form(self, mocker):
        """When ``merge()`` receives an ``ancestor`` argument, it must
        call ``get_diff`` twice (ancestor→A and ancestor→B) and the
        rendered prompt must contain both ancestor-relative diff
        sections instead of the single A↔B diff.
        """
        ancestor = make_candidate("g0-s0")
        ca = make_candidate("g1-s0")
        cb = make_candidate("g1-s1")
        config = make_config()

        child = make_candidate("g2-m0")
        mocker.patch("helix.merger.clone_candidate", return_value=child)

        # get_diff is called per-side; sequence captures which two diffs
        # were requested so we can assert the right pair of arguments
        # was passed (ancestor→A and ancestor→B, NOT A↔B).
        diff_calls: list[tuple[str, str]] = []

        def fake_get_diff(x, y, exclude=()):
            diff_calls.append((x.id, y.id))
            return f"+contribution from {y.id}"

        mocker.patch("helix.merger.get_diff", side_effect=fake_get_diff)
        mock_invoke = mocker.patch(
            "helix.merger.invoke_claude_code", return_value=({}, {})
        )
        mocker.patch("helix.merger.snapshot_candidate", return_value="sha")
        mocker.patch("helix.merger.remove_worktree")

        merge(ca, cb, "g2-m0", config, Path("/tmp"), project_root=Path("/tmp"), ancestor=ancestor)

        # Two get_diff calls, both anchored on the ancestor.
        assert diff_calls == [("g0-s0", "g1-s0"), ("g0-s0", "g1-s1")], (
            f"expected two ancestor-relative diff calls; got {diff_calls}"
        )

        # Prompt must contain both ancestor-relative section headers
        # and neither the legacy A↔B header nor any leftover placeholder.
        prompt_arg = mock_invoke.call_args[0][1]
        assert "## Diff: Candidate A relative to common ancestor g0-s0" in prompt_arg
        assert "## Diff: Candidate B relative to common ancestor g0-s0" in prompt_arg
        assert "## Diff (B relative to A)" not in prompt_arg
        assert "contribution from g1-s0" in prompt_arg
        assert "contribution from g1-s1" in prompt_arg

    def test_no_ancestor_uses_single_diff_form(self, mocker):
        """Backward-compat: ``merge()`` without an ``ancestor`` argument
        still computes a single A↔B diff and renders the legacy section.
        """
        ca = make_candidate("g1-s0")
        cb = make_candidate("g1-s1")
        config = make_config()

        child = make_candidate("g2-m0")
        mocker.patch("helix.merger.clone_candidate", return_value=child)

        diff_calls: list[tuple[str, str]] = []

        def fake_get_diff(x, y, exclude=()):
            diff_calls.append((x.id, y.id))
            return "+A-to-B contribution"

        mocker.patch("helix.merger.get_diff", side_effect=fake_get_diff)
        mock_invoke = mocker.patch(
            "helix.merger.invoke_claude_code", return_value=({}, {})
        )
        mocker.patch("helix.merger.snapshot_candidate", return_value="sha")
        mocker.patch("helix.merger.remove_worktree")

        merge(ca, cb, "g2-m0", config, Path("/tmp"), project_root=Path("/tmp"))  # no ancestor

        # Single get_diff call with A↔B arguments.
        assert diff_calls == [("g1-s0", "g1-s1")], (
            f"expected single A↔B diff call; got {diff_calls}"
        )

        prompt_arg = mock_invoke.call_args[0][1]
        assert "## Diff (B relative to A)" in prompt_arg
        assert "## Diff: Candidate A relative to common ancestor" not in prompt_arg

    def test_imports_mutation_error_from_mutator(self):
        """merger.py must reuse MutationError from mutator to avoid duplication."""
        from helix.merger import MutationError as MergerME
        from helix.mutator import MutationError as MutatorME
        assert MergerME is MutatorME

    def test_imports_invoke_claude_code_from_mutator(self):
        """merger.py must reuse invoke_claude_code from mutator."""
        import helix.merger as merger_mod
        import helix.mutator as mutator_mod
        assert merger_mod.invoke_claude_code is mutator_mod.invoke_claude_code


# ---------------------------------------------------------------------------
# Tests: select_eval_subsample_for_merged_program (GEPA parity)
# ---------------------------------------------------------------------------


class TestSelectEvalSubsample:
    """Pure-function tests for the GEPA-parity subsample helper.

    Mirrors gepa/src/gepa/proposer/merge.py::MergeProposer.select_eval_subsample_for_merged_program
    — stratifies across
    p1_wins / p2_wins / ties buckets, tops up from unused common ids, and
    falls back to rng.choices (with replacement) when common ids are
    exhausted.
    """

    def test_subsample_stratifies_buckets(self):
        # 6 common ids, 2 per bucket: p1={a,b}, p2={c,d}, ties={e,f}.
        scores1 = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0, "e": 0.5, "f": 0.5}
        scores2 = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0, "e": 0.5, "f": 0.5}
        rng = random.Random(0)
        result = select_eval_subsample_for_merged_program(
            scores1, scores2, rng, num_subsample_ids=3
        )
        assert len(result) == 3
        p1_bucket = {"a", "b"}
        p2_bucket = {"c", "d"}
        tie_bucket = {"e", "f"}
        assert any(r in p1_bucket for r in result), f"no p1 pick in {result}"
        assert any(r in p2_bucket for r in result), f"no p2 pick in {result}"
        assert any(r in tie_bucket for r in result), f"no tie pick in {result}"

    def test_subsample_respects_size_cap(self):
        # 20 common ids split across buckets, size cap 5 → distinct set of 5.
        scores1 = {str(i): float(i) for i in range(20)}
        scores2 = {str(i): float(20 - i) for i in range(20)}
        rng = random.Random(0)
        result = select_eval_subsample_for_merged_program(
            scores1, scores2, rng, num_subsample_ids=5
        )
        assert len(result) == 5
        assert len(set(result)) == 5, f"ids must be distinct, got {result}"
        assert all(r in scores1 for r in result)

    def test_subsample_with_replacement_fallback(self):
        # 2 common ids, size 5 → must fall back to rng.choices (duplicates ok).
        scores1 = {"a": 1.0, "b": 1.0}
        scores2 = {"a": 0.0, "b": 0.0}
        rng = random.Random(0)
        result = select_eval_subsample_for_merged_program(
            scores1, scores2, rng, num_subsample_ids=5
        )
        assert len(result) == 5
        assert set(result).issubset({"a", "b"})

    def test_subsample_fewer_than_size(self):
        # 2 common ids, all in p1 bucket, size 5 → returns 5 entries via the
        # rng.choices top-up
        # (GEPA merge.py::MergeProposer.select_eval_subsample_for_merged_program)
        # drawn from the 2 ids.
        scores1 = {"a": 1.0, "b": 1.0}
        scores2 = {"a": 0.0, "b": 0.0}
        rng = random.Random(0)
        result = select_eval_subsample_for_merged_program(
            scores1, scores2, rng, num_subsample_ids=5
        )
        assert len(result) == 5
        assert set(result).issubset({"a", "b"})

    def test_subsample_empty_common_returns_empty(self):
        # No common ids → selected stays empty, fallback elif common_ids
        # branch is skipped, returns [].
        rng = random.Random(0)
        result = select_eval_subsample_for_merged_program(
            {"a": 1.0}, {"b": 1.0}, rng, num_subsample_ids=5
        )
        assert result == []


# ---------------------------------------------------------------------------
# Tests: merge diffs against real git worktrees
# ---------------------------------------------------------------------------


def test_merge_prompt_diffs_a_seed_ancestor_without_protected_files(
    tmp_path: Path, mocker, monkeypatch
):
    """Two children of the seed: the seed's worktree is detached (no
    ``helix/g0-s0`` branch), and A's history holds a refreshed protected
    evaluator.  Both contributions must reach the prompt; the evaluator's
    contents must not."""
    import subprocess

    from helix.worktree import clone_candidate, create_seed_worktree, snapshot_candidate

    for key in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{key}_NAME", "HELIX Test")
        monkeypatch.setenv(f"GIT_{key}_EMAIL", "helix@test.local")
    project = tmp_path / "project"
    project.mkdir()
    (project / "solve.py").write_text("ANSWER = 0\n")
    (project / "evaluate.py").write_text("print('score')\n")
    (project / "data").mkdir()
    (project / "data" / "hidden.txt").write_text("old\n")
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "add", "-A"], cwd=project, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=project, check=True)
    worktrees = tmp_path / "worktrees"
    seed = create_seed_worktree(project, worktrees)
    a = clone_candidate(seed, "g1-s1", worktrees)
    (Path(a.worktree_path) / "solve.py").write_text("ANSWER = 41\n")
    (Path(a.worktree_path) / "evaluate.py").write_text("SECRET_SCORER = 1\n")
    (Path(a.worktree_path) / "data" / "hidden.txt").write_text("SECRET_ANSWER\n")
    snapshot_candidate(a, "A")
    b = clone_candidate(seed, "g1-s2", worktrees)
    (Path(b.worktree_path) / "helper.py").write_text("HELPER = 2\n")
    snapshot_candidate(b, "B")

    config = HelixConfig(
        objective="goal",
        evaluator=EvaluatorConfig(command="python evaluate.py"),
    )
    config.sandbox.omit_from_agent = ["data/hidden.txt"]
    invoke = mocker.patch(
        "helix.merger.invoke_claude_code", return_value=({}, {})
    )

    merged = merge(
        a, b, "g2-m1", config, worktrees, ancestor=seed, project_root=project
    )

    assert merged is not None
    prompt = invoke.call_args[0][1]
    assert "+ANSWER = 41" in prompt
    assert "+HELPER = 2" in prompt
    assert "SECRET" not in prompt
