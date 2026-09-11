"""Line splitting for newline-delimited machine formats.

Regression cover for the U+0085 fragmentation that cost one production run a
whole generation's token accounting: Codex emitted a JSONL record whose
``command_execution`` output carried a raw NEL byte, ``str.splitlines()``
broke the record into two invalid fragments, and the strict JSONL parse
raised on the first fragment.

Every separator below is written as an escape on purpose — these characters
are invisible in an editor, and a test that depends on one has to say so.
"""

from __future__ import annotations

import json
from pathlib import Path

from helix.lines import split_lf_lines
from helix.mutator import (
    _count_codex_stdout_tool_events,
    _parse_jsonl_output,
    _strip_machine_protocol_from_evaluator_stream,
)
from helix.parsers.helix_result import _extract_helix_result_line


# Every boundary ``str.splitlines()`` treats as a line break but a
# newline-delimited format does not.
NON_LF_BREAKS = [
    "\v",  # U+000B LINE TABULATION
    "\f",  # U+000C FORM FEED
    "\x1c",  # U+001C FILE SEPARATOR
    "\x1d",  # U+001D GROUP SEPARATOR
    "\x1e",  # U+001E RECORD SEPARATOR
    "\x85",  # U+0085 NEXT LINE (NEL)
    "\u2028",  # U+2028 LINE SEPARATOR
    "\u2029",  # U+2029 PARAGRAPH SEPARATOR
]

# The subset a conforming JSON serializer leaves unescaped inside a string
# literal, and therefore the subset that actually reaches a JSONL parser.
JSON_UNESCAPED_BREAKS = ["\x85", "\u2028", "\u2029"]

LINE_SEPARATOR = "\u2028"


def _nonblank(pieces: list[str]) -> list[str]:
    """The pieces a JSONL parser would actually try to decode."""
    return [piece for piece in pieces if piece.strip()]


class TestSplitLfLines:
    def test_splits_on_line_feed_only(self) -> None:
        assert split_lf_lines("a\nb\nc") == ["a", "b", "c"]

    def test_trailing_line_feed_yields_trailing_empty_piece(self) -> None:
        assert split_lf_lines("a\n") == ["a", ""]

    def test_no_break_on_any_non_lf_unicode_boundary(self) -> None:
        for ch in NON_LF_BREAKS:
            text = f"left{ch}right"
            assert len(text.splitlines()) == 2, f"precondition failed for {ch!r}"
            assert split_lf_lines(text) == [text]

    def test_carriage_return_is_left_for_the_caller_to_strip(self) -> None:
        assert split_lf_lines("a\r\nb") == ["a\r", "b"]
        assert [piece.strip() for piece in split_lf_lines("a\r\nb")] == ["a", "b"]


def _codex_stdout(ch: str) -> str:
    """A two-record Codex stream whose first record embeds *ch* verbatim.

    Shaped like the real thing: a ``command_execution`` record carrying
    captured output, then the ``turn.completed`` record that reports usage.
    """
    return (
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "aggregated_output": f"before{ch}after",
                },
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 2500000, "output_tokens": 8000},
            }
        )
        + "\n"
    )


class TestJsonlParseSurvivesEmbeddedUnicodeBreaks:
    """``_parse_jsonl_output`` must key on the line feed, not on Unicode."""

    def test_strict_parse_keeps_every_record(self) -> None:
        for ch in JSON_UNESCAPED_BREAKS:
            stdout = _codex_stdout(ch)
            # Precondition: a splitlines()-based scan really does fragment this
            # stream into more pieces than there are records.
            assert len(_nonblank(stdout.splitlines())) == 3, repr(ch)
            assert len(_nonblank(split_lf_lines(stdout))) == 2, repr(ch)

            parsed = _parse_jsonl_output(
                stdout,
                backend="codex",
                cmd_str="codex exec --json",
                worktree_path="/wt",
                stderr="",
                exit_code=0,
                strict=True,
            )

            assert parsed["unparsable_lines"] == []
            assert len(parsed["events"]) == 2
            assert (
                parsed["events"][0]["item"]["aggregated_output"] == f"before{ch}after"
            )
            assert parsed["events"][1]["usage"]["input_tokens"] == 2500000

    def test_tool_event_counter_reads_one_record_not_two_fragments(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "stdout.jsonl"
        path.write_text(_codex_stdout("\x85"), encoding="utf-8")

        count, names = _count_codex_stdout_tool_events(path)

        assert count == 1
        assert names == ["exec_command"]


class TestHelixResultLineSurvivesEmbeddedUnicodeBreaks:
    """A ``HELIX_RESULT=`` payload may legitimately carry U+2028 in side info."""

    def test_result_line_is_not_truncated_at_a_unicode_break(self) -> None:
        payload = json.dumps([[1.0, {"note": "a\u2028b"}]], ensure_ascii=False)
        stdout = f"noise\nHELIX_RESULT={payload}\ntrailing\n"

        line = _extract_helix_result_line(stdout)

        assert line is not None
        assert json.loads(line.removeprefix("HELIX_RESULT=")) == [
            [1.0, {"note": "a\u2028b"}]
        ]

    def test_machine_protocol_fragment_does_not_leak_into_the_mutation_prompt(
        self,
    ) -> None:
        payload = json.dumps([[1.0, {"note": "a\u2028b"}]], ensure_ascii=False)
        stdout = f"real evaluator output\nHELIX_RESULT={payload}\n"

        kept = _strip_machine_protocol_from_evaluator_stream(stdout)

        # A splitlines() scan dropped only the fragment up to U+2028; the tail
        # no longer started with the prefix, so it leaked into the prompt.
        assert kept == "real evaluator output"
        assert "HELIX_RESULT" not in kept
        leaked_tail = payload.split(LINE_SEPARATOR)[1]
        assert leaked_tail not in kept
