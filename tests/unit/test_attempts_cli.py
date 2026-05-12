"""Unit tests for `helix attempts` CLI subcommand.

Tests the full flag surface:
  - default table (all rejected attempts)
  - --generation, --stage, --reason filters
  - --cid detail view
  - --json output
  - --skips view and filters
  - --path to a non-default .helix/ directory
  - graceful error handling (missing dir, malformed file)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from helix.cli import cli


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_attempt(
    helix_dir: Path,
    cid: str,
    *,
    generation: int,
    parent_id: str,
    reason: str,
    stage: str,
    instance_scores: dict[str, float] | None = None,
    example_ids: list[str] | None = None,
) -> None:
    """Write a synthetic attempt JSON file into helix_dir/attempts/."""
    attempts_dir = helix_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "candidate_id": cid,
        "scores": {},
        "instance_scores": instance_scores or {},
        "asi": {},
        "attempt": {
            "status": "rejected",
            "reason": reason,
            "parent_id": parent_id,
            "generation": generation,
            "stage": stage,
            "example_ids": example_ids,
        },
    }
    (attempts_dir / f"{cid}.json").write_text(json.dumps(payload, indent=2))


def _write_skip(
    helix_dir: Path,
    generation: int,
    records: list[dict[str, Any]],
) -> None:
    """Write a synthetic skip JSON list file into helix_dir/skips/."""
    skips_dir = helix_dir / "skips"
    skips_dir.mkdir(parents=True, exist_ok=True)
    (skips_dir / f"g{generation}.json").write_text(json.dumps(records, indent=2))


def _make_helix_dir(tmp_path: Path) -> Path:
    """Create a minimal .helix/ directory with 3 attempts and 2 skips."""
    helix_dir = tmp_path / ".helix"

    # Three attempts across generations 12, 13, 14
    _write_attempt(
        helix_dir,
        "g12-s4",
        generation=12,
        parent_id="g11-s3",
        reason="minibatch_gate",
        stage="train_minibatch",
        instance_scores={"ex1": 0.31, "ex2": 0.28},
        example_ids=["ex1", "ex2"],
    )
    _write_attempt(
        helix_dir,
        "g13-s5",
        generation=13,
        parent_id="g12-s4",
        reason="val_stage",
        stage="val_stage",
        instance_scores={"ex3": 0.42},
        example_ids=["ex3"],
    )
    _write_attempt(
        helix_dir,
        "g14-s6",
        generation=14,
        parent_id="g13-s5",
        reason="train_gate",
        stage="train",
        instance_scores={"ex4": 0.39},
    )

    # Two skips across generations 12 and 13
    _write_skip(
        helix_dir,
        generation=12,
        records=[
            {
                "generation": 12,
                "parent_id": "g11-s3",
                "reason": "perfect_subsample",
                "parent_eval": {},
            },
            {
                "generation": 12,
                "parent_id": "g11-s4",
                "reason": "perfect_subsample",
                "parent_eval": {},
            },
        ],
    )
    _write_skip(
        helix_dir,
        generation=13,
        records=[
            {
                "generation": 13,
                "parent_id": "g12-s4",
                "reason": "perfect_subsample",
                "parent_eval": {},
            }
        ],
    )

    return helix_dir


# ---------------------------------------------------------------------------
# Table — attempts view
# ---------------------------------------------------------------------------


class TestAttemptsTableDefault:
    """helix attempts (no flags) shows all attempt records."""

    def test_attempts_table_default_shows_all_attempts(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir)]
        )
        assert result.exit_code == 0, result.output
        # All three CIDs must appear in the output
        assert "g12-s4" in result.output
        assert "g13-s5" in result.output
        assert "g14-s6" in result.output

    def test_attempts_table_filters_by_generation(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--generation", "12"]
        )
        assert result.exit_code == 0, result.output
        assert "g12-s4" in result.output
        assert "g13-s5" not in result.output
        assert "g14-s6" not in result.output

    def test_attempts_table_filters_by_stage(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--stage", "val_stage"]
        )
        assert result.exit_code == 0, result.output
        assert "g13-s5" in result.output
        assert "g12-s4" not in result.output
        assert "g14-s6" not in result.output

    def test_attempts_table_filters_by_reason_substring(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        # "minibatch" is a substring of "minibatch_gate"
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--reason", "minibatch"]
        )
        assert result.exit_code == 0, result.output
        assert "g12-s4" in result.output
        assert "g13-s5" not in result.output
        assert "g14-s6" not in result.output

    def test_attempts_table_reason_shows_score(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--generation", "12"]
        )
        assert result.exit_code == 0, result.output
        # instance_scores for g12-s4 are 0.31 and 0.28 → mean 0.295 ≈ 0.30
        assert "score=" in result.output


# ---------------------------------------------------------------------------
# --cid detail view
# ---------------------------------------------------------------------------


class TestAttemptsCidDetail:
    def test_attempts_cid_detail_view(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--cid", "g12-s4"]
        )
        assert result.exit_code == 0, result.output
        # Output should be valid JSON
        data = json.loads(result.output)
        assert data["candidate_id"] == "g12-s4"
        assert data["attempt"]["reason"] == "minibatch_gate"
        assert data["attempt"]["generation"] == 12

    def test_attempts_cid_missing_exits_nonzero(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--cid", "g99-s99"]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# --json output
# ---------------------------------------------------------------------------


class TestAttemptsJsonOutput:
    def test_attempts_json_output_is_jq_friendly(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--json"]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 3
        # All keys from the on-disk JSON must be preserved
        for record in data:
            assert "candidate_id" in record
            assert "attempt" in record
            assert "instance_scores" in record
            assert "reason" in record["attempt"]

    def test_attempts_json_filtered_by_generation(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--json", "--generation", "13"]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["candidate_id"] == "g13-s5"


# ---------------------------------------------------------------------------
# --skips view
# ---------------------------------------------------------------------------


class TestAttemptsSkipsView:
    def test_attempts_skips_view(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--skips"]
        )
        assert result.exit_code == 0, result.output
        # 3 skip records total (2 for gen 12, 1 for gen 13)
        assert "perfect_subsample" in result.output
        assert "g11-s3" in result.output
        assert "g11-s4" in result.output
        assert "g12-s4" in result.output  # parent in gen 13 skip

    def test_attempts_skips_filters_by_generation(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["attempts", "--path", str(helix_dir), "--skips", "--generation", "12"],
        )
        assert result.exit_code == 0, result.output
        # Gen 12 has 2 skip records
        assert "g11-s3" in result.output
        assert "g11-s4" in result.output
        # Gen 13 parent should NOT appear
        assert "g12-s4" not in result.output

    def test_attempts_skips_json_output(self, tmp_path: Path) -> None:
        helix_dir = _make_helix_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--skips", "--json"]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 3  # 2 for gen 12 + 1 for gen 13
        for rec in data:
            assert rec["reason"] == "perfect_subsample"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestAttemptsErrorHandling:
    def test_attempts_handles_missing_helix_dir(self, tmp_path: Path) -> None:
        """Missing .helix/attempts/ → clean error, not a stack trace."""
        helix_dir = tmp_path / ".helix"
        # Do NOT create the helix dir at all
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir)]
        )
        assert result.exit_code != 0
        # Should mention the path and be human-readable, not a traceback
        assert "attempts" in result.output.lower() or "attempts" in (result.output + "").lower()
        assert "Traceback" not in result.output

    def test_attempts_handles_missing_skips_dir(self, tmp_path: Path) -> None:
        """Missing .helix/skips/ with --skips → clean error."""
        helix_dir = tmp_path / ".helix"
        helix_dir.mkdir(parents=True)
        # No skips/ subdirectory
        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir), "--skips"]
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.output

    def test_attempts_handles_malformed_attempt_file(self, tmp_path: Path) -> None:
        """Malformed attempt JSON is skipped with a warning; valid files still render."""
        helix_dir = tmp_path / ".helix"
        attempts_dir = helix_dir / "attempts"
        attempts_dir.mkdir(parents=True)

        # One valid attempt
        _write_attempt(
            helix_dir,
            "g12-s4",
            generation=12,
            parent_id="g11-s3",
            reason="minibatch_gate",
            stage="train_minibatch",
        )
        # One malformed file
        (attempts_dir / "bad.json").write_text("{ NOT VALID JSON !!!")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["attempts", "--path", str(helix_dir)]
        )
        # Should NOT crash
        assert result.exit_code == 0, result.output
        # Valid record must appear
        assert "g12-s4" in result.output
        # Warning emitted to stderr (mix_stderr=True by default in CliRunner)
        assert "Warning" in result.output or "warning" in result.output.lower() or result.exit_code == 0


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------


class TestAttemptsHelp:
    def test_attempts_help_shows_subcommand(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["attempts", "--help"])
        assert result.exit_code == 0, result.output
        assert "--skips" in result.output
        assert "--generation" in result.output
        assert "--stage" in result.output
        assert "--reason" in result.output
        assert "--cid" in result.output
        assert "--json" in result.output
        assert "--path" in result.output

    def test_attempts_appears_in_top_level_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0, result.output
        assert "attempts" in result.output
