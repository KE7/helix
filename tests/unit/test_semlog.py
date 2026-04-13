"""Unit tests for semantic mutation logging (semlog) functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helix.mutator import parse_mutation_summary


# ---------------------------------------------------------------------------
# Tests: parse_mutation_summary
# ---------------------------------------------------------------------------


class TestParseMutationSummaryComplete:
    """test_parse_mutation_summary_complete: full block parsed correctly."""

    def test_all_mutation_fields_parsed(self):
        output = """\
Some Claude output here...
[MUTATION COMPLETE]
[SUMMARY]
files_changed: src/model.py, src/utils.py
root_cause: The model was not normalising input features before training.
changes_made: Added a StandardScaler preprocessing step. Updated the training pipeline.
reasoning: Normalisation is the most direct fix for scale-sensitive algorithms.
expected_impact: Expect accuracy to improve by 5-10% based on typical gains from this fix.
[END SUMMARY]
"""
        result = parse_mutation_summary(output)

        assert result["files_changed"] == "src/model.py, src/utils.py"
        assert result["root_cause"] == "The model was not normalising input features before training."
        assert "StandardScaler" in result["changes_made"]
        assert result["reasoning"] == "Normalisation is the most direct fix for scale-sensitive algorithms."
        assert "5-10%" in result["expected_impact"]

    def test_returns_dict_type(self):
        output = "[SUMMARY]\nfiles_changed: foo.py\n[END SUMMARY]"
        result = parse_mutation_summary(output)
        assert isinstance(result, dict)

    def test_keys_are_stripped(self):
        output = "[SUMMARY]\n  files_changed : foo.py\n[END SUMMARY]"
        result = parse_mutation_summary(output)
        assert "files_changed" in result

    def test_values_are_stripped(self):
        output = "[SUMMARY]\nfiles_changed:   bar.py  \n[END SUMMARY]"
        result = parse_mutation_summary(output)
        assert result["files_changed"] == "bar.py"

    def test_value_with_colon_preserved(self):
        """Values containing ':' should keep everything after the first ': '."""
        output = "[SUMMARY]\nroot_cause: Issue in module: core/parser.py\n[END SUMMARY]"
        result = parse_mutation_summary(output)
        assert result["root_cause"] == "Issue in module: core/parser.py"

    def test_stops_at_end_summary(self):
        """Lines after [END SUMMARY] should not be parsed."""
        output = (
            "[SUMMARY]\nfiles_changed: a.py\n[END SUMMARY]\nextra_key: should_not_appear\n"
        )
        result = parse_mutation_summary(output)
        assert "extra_key" not in result


class TestParseMutationSummaryMissingBlock:
    """test_parse_mutation_summary_missing_block: returns empty dict gracefully (never crash)."""

    def test_returns_empty_dict_when_no_block(self):
        output = "Claude Code output with no summary block at all."
        result = parse_mutation_summary(output)
        assert result == {}

    def test_returns_empty_dict_on_empty_string(self):
        result = parse_mutation_summary("")
        assert result == {}

    def test_returns_empty_dict_on_partial_block_no_end(self):
        """[SUMMARY] without [END SUMMARY] — reads to EOF, may have partial data."""
        output = "[SUMMARY]\nfiles_changed: a.py\n"
        result = parse_mutation_summary(output)
        # Should not crash; may return partial dict
        assert isinstance(result, dict)

    def test_returns_empty_dict_on_only_end_marker(self):
        output = "[END SUMMARY]\nfiles_changed: a.py\n"
        result = parse_mutation_summary(output)
        assert result == {}

    def test_no_exception_on_none_like_empty_string(self):
        """Ensure no crash even on edge-case whitespace-only input."""
        result = parse_mutation_summary("   \n\n\n   ")
        assert result == {}

    def test_no_exception_on_malformed_lines(self):
        output = "[SUMMARY]\nthis line has no colon separator\nanother bad line\n[END SUMMARY]"
        result = parse_mutation_summary(output)
        assert isinstance(result, dict)
        # None of the malformed lines should produce keys
        assert len(result) == 0


class TestParseMergeSummary:
    """test_parse_merge_summary: merge-specific fields parsed."""

    def test_all_merge_fields_parsed(self):
        output = """\
Merging candidates...
[MERGE COMPLETE]
[SUMMARY]
files_changed: src/solver.py, tests/test_solver.py
candidate_a_kept: The base algorithm from A because it had better convergence.
candidate_b_applied: The caching layer from B because it reduced redundant computation.
conflicts_resolved: Both modified the solve() function; kept A's logic with B's cache wrapper.
merge_strategy: Apply B's performance improvements on top of A's correctness fixes.
expected_impact: Combined improvements should yield 15% speedup and maintain accuracy.
[END SUMMARY]
"""
        result = parse_mutation_summary(output)

        assert result["files_changed"] == "src/solver.py, tests/test_solver.py"
        assert "base algorithm" in result["candidate_a_kept"]
        assert "caching layer" in result["candidate_b_applied"]
        assert "solve()" in result["conflicts_resolved"]
        assert result["merge_strategy"] == "Apply B's performance improvements on top of A's correctness fixes."
        assert "15%" in result["expected_impact"]

    def test_merge_fields_present_as_keys(self):
        output = (
            "[SUMMARY]\n"
            "files_changed: x.py\n"
            "candidate_a_kept: kept A's approach\n"
            "candidate_b_applied: applied B's fix\n"
            "conflicts_resolved: none\n"
            "merge_strategy: selective apply\n"
            "expected_impact: minor improvement\n"
            "[END SUMMARY]"
        )
        result = parse_mutation_summary(output)
        expected_keys = {
            "files_changed",
            "candidate_a_kept",
            "candidate_b_applied",
            "conflicts_resolved",
            "merge_strategy",
            "expected_impact",
        }
        assert expected_keys.issubset(result.keys())

    def test_merge_summary_no_block_returns_empty(self):
        output = "Merge done. No structured summary provided."
        result = parse_mutation_summary(output)
        assert result == {}
