"""Unit tests for the per-parent sibling record.

The module has three jobs, tested separately here:

* **capture** — read a self-report the mutating agent left in its worktree.
  The artifact is attacker-controlled, so most of these tests are about what
  ``capture_summary`` *refuses*.
* **accumulate** — append one entry per evaluated child, keeping all of them.
* **render** — turn a parent's entries into a prompt section.

Note what is deliberately absent: there is no cap on what is stored; the only
limit is the optional display window ``render_history`` applies. The on/off
decision lives in the caller (``evolution.sibling_history``), which passes
``None`` to ``mutate`` when the feature is off, so nothing here has an "off"
mode to test.
"""

from __future__ import annotations

import os
from pathlib import Path

from helix.sibling_history import (
    MAX_SUMMARY_WORDS,
    SUMMARY_ARTIFACT_NAME,
    append_child,
    capture_summary,
    history_instruction,
    normalize_history,
    render_history,
    summary_prompt,
)


def _write_summary(worktree: Path, text: str) -> Path:
    target = worktree / SUMMARY_ARTIFACT_NAME
    target.write_text(text)
    return target


# ---------------------------------------------------------------------------
# summary_prompt — the whole prompt for the dedicated summary call
# ---------------------------------------------------------------------------


class TestSummaryPrompt:
    def test_names_the_artifact_the_capture_side_reads(self) -> None:
        """The call must be told the exact filename ``capture_summary`` opens.

        These are the two halves of one contract; a drift between them turns
        the feature into a silent no-op where every child reports nothing.
        """
        assert SUMMARY_ARTIFACT_NAME in summary_prompt(diff="- a\n+ b")

    def test_quotes_the_same_word_budget_that_is_enforced(self) -> None:
        assert str(MAX_SUMMARY_WORDS) in summary_prompt(diff="- a\n+ b")

    def test_custom_budget_is_quoted(self) -> None:
        assert "42" in summary_prompt(diff="- a\n+ b", max_words=42)

    def test_carries_the_diff(self) -> None:
        """The call has no memory of the mutation, so the diff is the only
        evidence of what happened; it has to be in the prompt."""
        prompt = summary_prompt(diff="--- a/solve.py\n+++ b/solve.py\n+import scipy")
        assert "+import scipy" in prompt
        assert "```diff" in prompt

    def test_carries_the_objective_when_given(self) -> None:
        assert "Maximise packing" in summary_prompt(
            objective="Maximise packing", diff="- a\n+ b"
        )

    def test_objective_section_omitted_when_absent(self) -> None:
        assert "## Objective being optimized" not in summary_prompt(diff="- a\n+ b")

    def test_carries_new_files_in_their_own_section(self) -> None:
        prompt = summary_prompt(diff="- a\n+ b", new_files="### features.py\n```\nX = 1\n```")
        assert "## Changes to existing files" in prompt
        assert "## New files added by this attempt" in prompt
        assert "X = 1" in prompt
        assert prompt.index("## Changes to existing files") < prompt.index("## New files")

    def test_new_files_alone_are_a_change(self) -> None:
        """A mutation that only added files has no diff, but it is not a
        mutation with nothing to describe."""
        prompt = summary_prompt(new_files="### features.py\n```\nX = 1\n```")
        assert "## New files added by this attempt" in prompt
        assert "## Changes to existing files" not in prompt
        assert "No diff was available" not in prompt

    def test_absent_diff_is_declared_rather_than_invented(self) -> None:
        """The summary is fed into later prompts, so a call with nothing to
        look at must be told to say so instead of guessing an approach."""
        prompt = summary_prompt(diff="")
        assert "No diff was available" in prompt
        assert "do not guess" in prompt

    def test_tells_the_call_not_to_edit_code(self) -> None:
        """It runs in a live candidate worktree; an edit here would silently
        become part of the candidate."""
        assert "Do not change any code" in summary_prompt(diff="- a\n+ b")

    def test_states_the_turn_bound_in_words(self) -> None:
        """Only claude enforces ``--max-turns``; on the other four backends
        this sentence is the only thing stopping the call from wandering."""
        prompt = summary_prompt(diff="- a\n+ b")
        assert "one or two tool calls" in prompt
        assert "do not explore the repository" in prompt.lower()


# ---------------------------------------------------------------------------
# history_instruction — how the agent is told to use the record
# ---------------------------------------------------------------------------


class TestHistoryInstruction:
    def test_tells_the_agent_to_diverge_from_what_was_tried(self) -> None:
        text = history_instruction()
        assert "already been tried" in text
        assert "differs from" in text

    def test_is_a_bullet_so_it_joins_the_task_instruction_list(self) -> None:
        """It is appended to the shared task-instruction block, which is a
        list of ``- `` bullets; anything else would render as a stray line."""
        assert history_instruction().startswith("- ")

    def test_speaks_in_the_indicative(self) -> None:
        """It is emitted only alongside a non-empty section, so it can state
        the attempts are there rather than hedging about whether they are."""
        assert history_instruction().startswith("- Previous attempts")


# ---------------------------------------------------------------------------
# capture_summary — reading untrusted agent output
# ---------------------------------------------------------------------------


class TestCaptureSummary:
    def test_reads_the_report(self, tmp_path: Path) -> None:
        _write_summary(tmp_path, "Replaced the bubble sort with a heapsort.")
        assert capture_summary(tmp_path) == "Replaced the bubble sort with a heapsort."

    def test_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        """The agent forgetting to write the file is the expected case, not a
        failure: most agent loops skip a closing instruction some of the time."""
        assert capture_summary(tmp_path) is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        _write_summary(tmp_path, "")
        assert capture_summary(tmp_path) is None

    def test_whitespace_only_returns_none(self, tmp_path: Path) -> None:
        _write_summary(tmp_path, "   \n\n\t  ")
        assert capture_summary(tmp_path) is None

    def test_multiline_structure_survives(self, tmp_path: Path) -> None:
        _write_summary(tmp_path, "Did two things:\n\n- one\n- two")
        assert capture_summary(tmp_path) == "Did two things:\n\n- one\n- two"

    def test_truncates_past_the_word_budget_and_says_so(self, tmp_path: Path) -> None:
        _write_summary(tmp_path, " ".join(f"w{i}" for i in range(50)))
        captured = capture_summary(tmp_path, max_words=5)
        assert captured is not None
        assert captured.startswith("w0 w1 w2 w3 w4")
        assert "truncated" in captured
        assert "w5" not in captured

    def test_exactly_at_the_budget_is_not_marked_truncated(
        self, tmp_path: Path
    ) -> None:
        _write_summary(tmp_path, "one two three")
        assert capture_summary(tmp_path, max_words=3) == "one two three"

    def test_control_characters_are_stripped(self, tmp_path: Path) -> None:
        """A prompt is assembled by concatenation; stray control bytes in
        agent-authored text can corrupt everything rendered after it."""
        _write_summary(tmp_path, "before\x00\x07after")
        assert capture_summary(tmp_path) == "beforeafter"

    def test_tabs_and_newlines_are_kept(self, tmp_path: Path) -> None:
        _write_summary(tmp_path, "line one\n\tindented")
        assert capture_summary(tmp_path) == "line one\n\tindented"

    def test_invalid_utf8_does_not_raise(self, tmp_path: Path) -> None:
        (tmp_path / SUMMARY_ARTIFACT_NAME).write_bytes(b"valid \xff\xfe tail")
        captured = capture_summary(tmp_path)
        assert captured is not None
        assert "valid" in captured

    def test_symlink_is_refused(self, tmp_path: Path) -> None:
        """The decisive security case.

        The agent chooses this file's contents *and* could choose to make it a
        symlink. Following one would copy an arbitrary readable file into
        persisted state and into a later prompt.
        """
        secret = tmp_path / "secret.txt"
        secret.write_text("PRIVATE KEY MATERIAL")
        worktree = tmp_path / "wt"
        worktree.mkdir()
        os.symlink(secret, worktree / SUMMARY_ARTIFACT_NAME)

        assert capture_summary(worktree) is None

    def test_symlink_to_a_file_inside_the_worktree_is_also_refused(
        self, tmp_path: Path
    ) -> None:
        """Refusal is on the link itself, not on where it points — no need to
        reason about which targets happen to be harmless."""
        worktree = tmp_path / "wt"
        worktree.mkdir()
        (worktree / "notes.md").write_text("ordinary file")
        os.symlink(worktree / "notes.md", worktree / SUMMARY_ARTIFACT_NAME)

        assert capture_summary(worktree) is None

    def test_directory_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / SUMMARY_ARTIFACT_NAME).mkdir()
        assert capture_summary(tmp_path) is None

    def test_enormous_file_is_bounded(self, tmp_path: Path) -> None:
        """The per-summary word budget is the one cap that remains: it bounds
        what a single agent can inject, independently of how many siblings a
        parent accumulates."""
        _write_summary(tmp_path, "word " * 500_000)
        captured = capture_summary(tmp_path)
        assert captured is not None
        assert len(captured.split()) <= MAX_SUMMARY_WORDS + 2


# ---------------------------------------------------------------------------
# append_child — accumulating the record
# ---------------------------------------------------------------------------


class TestAppendChild:
    def test_appends_under_the_parent(self) -> None:
        history: dict[str, list[dict[str, object]]] = {}
        append_child(
            history,
            "g0-s0",
            child_id="g1-s1",
            score=0.5,
            accepted=True,
            summary="did x",
        )
        assert history["g0-s0"] == [
            {"child_id": "g1-s1", "score": 0.5, "accepted": True, "summary": "did x"}
        ]

    def test_separate_parents_do_not_mix(self) -> None:
        history: dict[str, list[dict[str, object]]] = {}
        append_child(
            history, "p1", child_id="c1", score=0.1, accepted=False, summary=None
        )
        append_child(
            history, "p2", child_id="c2", score=0.2, accepted=True, summary=None
        )
        assert list(history) == ["p1", "p2"]
        assert len(history["p1"]) == len(history["p2"]) == 1

    def test_rejected_children_are_recorded_too(self) -> None:
        """The rejected half is the point: it is what stops the next attempt
        re-proposing an approach already known to lose."""
        history: dict[str, list[dict[str, object]]] = {}
        append_child(
            history, "p", child_id="c", score=0.0, accepted=False, summary="failed idea"
        )
        assert history["p"][0]["accepted"] is False

    def test_none_score_is_preserved(self) -> None:
        history: dict[str, list[dict[str, object]]] = {}
        append_child(
            history, "p", child_id="c", score=None, accepted=False, summary=None
        )
        assert history["p"][0]["score"] is None

    def test_score_is_coerced_to_float(self) -> None:
        history: dict[str, list[dict[str, object]]] = {}
        append_child(history, "p", child_id="c", score=1, accepted=True, summary=None)
        assert isinstance(history["p"][0]["score"], float)

    def test_every_child_is_retained(self) -> None:
        """No cap, and nothing evicted.

        A heavily re-sampled parent is exactly the case the feature exists for,
        so its oldest attempts are not the ones to discard.
        """
        history: dict[str, list[dict[str, object]]] = {}
        for i in range(200):
            append_child(
                history,
                "p",
                child_id=f"c{i}",
                score=float(i),
                accepted=True,
                summary=None,
            )
        entries = history["p"]
        assert len(entries) == 200
        assert entries[0]["child_id"] == "c0"
        assert entries[-1]["child_id"] == "c199"

    def test_insertion_order_is_preserved(self) -> None:
        history: dict[str, list[dict[str, object]]] = {}
        for cid in ("first", "second", "third"):
            append_child(
                history, "p", child_id=cid, score=0.1, accepted=False, summary=None
            )
        assert [e["child_id"] for e in history["p"]] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# render_history — what the agent actually sees
# ---------------------------------------------------------------------------


def _history(n: int = 3) -> dict[str, list[dict[str, object]]]:
    history: dict[str, list[dict[str, object]]] = {}
    for i in range(n):
        append_child(
            history,
            "p",
            child_id=f"c{i}",
            score=float(i) / 10,
            accepted=(i % 2 == 0),
            summary=f"approach number {i}",
        )
    return history


class TestRenderHistory:
    def test_unknown_parent_renders_nothing(self) -> None:
        """A parent sampled for the first time has no section, rather than an
        empty heading."""
        assert render_history(_history(), "never-sampled") == ""

    def test_parent_with_no_entries_renders_nothing(self) -> None:
        assert render_history({"p": []}, "p") == ""

    def test_renders_verdicts_scores_and_summaries(self) -> None:
        rendered = render_history(_history(), "p")
        assert "## Previous Attempts From This Parent" in rendered
        assert "KEPT" in rendered
        assert "REJECTED" in rendered
        assert "approach number 0" in rendered
        assert "+0.2000" in rendered

    def test_section_is_data_not_instructions(self) -> None:
        """What to do with the record is stated once, in the task
        instructions; repeating it here would give the same directive two
        homes that could drift apart."""
        rendered = render_history(_history(), "p")
        assert "already been tried" not in rendered
        assert "differs from" not in rendered

    def test_every_child_is_rendered(self) -> None:
        """No display limit: whatever was recorded is what the agent sees."""
        rendered = render_history(_history(25), "p")
        for i in range(25):
            assert f"approach number {i}" in rendered

    def test_oldest_first(self) -> None:
        rendered = render_history(_history(3), "p")
        assert rendered.index("approach number 0") < rendered.index("approach number 2")

    def test_missing_summary_is_called_out(self) -> None:
        """An attempt with no note still has to appear: its score and verdict
        are informative, and silently dropping it would misrepresent how many
        times this parent has been tried."""
        history: dict[str, list[dict[str, object]]] = {}
        append_child(
            history, "p", child_id="c", score=0.5, accepted=False, summary=None
        )
        rendered = render_history(history, "p")
        assert "left no summary" in rendered
        assert "REJECTED" in rendered

    def test_none_score_renders_as_na(self) -> None:
        history: dict[str, list[dict[str, object]]] = {}
        append_child(
            history, "p", child_id="c", score=None, accepted=False, summary="x"
        )
        assert "n/a" in render_history(history, "p")

    def test_multiline_summary_stays_inside_its_bullet(self) -> None:
        history: dict[str, list[dict[str, object]]] = {}
        append_child(
            history,
            "p",
            child_id="c",
            score=0.5,
            accepted=True,
            summary="first\nsecond",
        )
        rendered = render_history(history, "p")
        assert "    first" in rendered
        assert "    second" in rendered

    def test_rendering_does_not_mutate_the_store(self) -> None:
        history = _history(4)
        before = [dict(e) for e in history["p"]]
        render_history(history, "p")
        assert history["p"] == before


class TestRenderHistoryWindow:
    """``max_entries``: a FIFO window over what reaches the prompt."""

    def test_keeps_only_the_most_recent(self) -> None:
        rendered = render_history(_history(10), "p", max_entries=3)
        for i in range(7):
            assert f"approach number {i}" not in rendered
        for i in range(7, 10):
            assert f"approach number {i}" in rendered

    def test_window_stays_oldest_first(self) -> None:
        rendered = render_history(_history(10), "p", max_entries=3)
        assert rendered.index("approach number 7") < rendered.index("approach number 9")

    def test_heading_discloses_the_window(self) -> None:
        """The agent must know earlier attempts exist, or it will read the
        window as the parent's whole history."""
        rendered = render_history(_history(10), "p", max_entries=3)
        assert "3 most recent of the 10" in rendered

    def test_under_the_limit_renders_as_unbounded(self) -> None:
        """A window that cuts nothing changes nothing, heading included."""
        history = _history(3)
        assert render_history(history, "p", max_entries=3) == render_history(history, "p")
        assert render_history(history, "p", max_entries=50) == render_history(history, "p")

    def test_none_shows_everything(self) -> None:
        rendered = render_history(_history(25), "p", max_entries=None)
        for i in range(25):
            assert f"approach number {i}" in rendered

    def test_window_does_not_trim_the_store(self) -> None:
        """The cap is display-only: state keeps every child."""
        history = _history(10)
        render_history(history, "p", max_entries=2)
        assert len(history["p"]) == 10


# ---------------------------------------------------------------------------
# normalize_history — state.json is on-disk data, not a trusted structure
# ---------------------------------------------------------------------------


class TestNormalizeHistory:
    def test_missing_field_becomes_empty(self) -> None:
        assert normalize_history(None) == {}

    def test_wrong_top_level_type_becomes_empty(self) -> None:
        assert normalize_history(["not", "a", "dict"]) == {}

    def test_wellformed_payload_survives(self) -> None:
        payload = {
            "p": [{"child_id": "c", "score": 0.5, "accepted": True, "summary": "s"}]
        }
        assert normalize_history(payload) == payload

    def test_non_list_values_are_dropped(self) -> None:
        assert normalize_history({"p": "not a list"}) == {}

    def test_non_dict_entries_are_dropped(self) -> None:
        assert normalize_history({"p": ["bad", 3, None]}) == {}

    def test_partially_valid_parent_keeps_the_valid_entries(self) -> None:
        result = normalize_history({"p": ["bad", {"child_id": "c"}]})
        assert result == {"p": [{"child_id": "c"}]}

    def test_long_history_is_kept_whole(self) -> None:
        """Loading must not quietly shrink a long-running parent's record."""
        payload = {"p": [{"child_id": f"c{i}"} for i in range(300)]}
        result = normalize_history(payload)
        assert len(result["p"]) == 300
        assert result["p"][0]["child_id"] == "c0"
        assert result["p"][-1]["child_id"] == "c299"
