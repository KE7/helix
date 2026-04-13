"""Unit tests for rate-limit error handling and diagnostics fixes.

Fix 1 — RateLimitError inherits from HelixError (exceptions.py).
Fix 2 — RateLimitError carries full diagnostics (mirrors MutationError).
Fix 3 — Top-level RateLimitError catch in cli.py evolve command.
Fix 4 — Structured file logging via setup_file_logging().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from helix.cli import cli
from helix.config import ClaudeConfig
from helix.exceptions import HelixError, MutationError, RateLimitError, print_helix_error
from helix.logging_config import setup_file_logging
from helix.mutator import invoke_claude_code


# ---------------------------------------------------------------------------
# Fixture: clean up any FileHandlers added to the 'helix' logger during a test
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_helix_logger() -> Generator[None, None, None]:
    """Remove FileHandlers added to the helix logger during the test."""
    helix_logger = logging.getLogger("helix")
    before = list(helix_logger.handlers)
    yield
    for handler in list(helix_logger.handlers):
        if handler not in before:
            handler.close()
            helix_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Fix 1: RateLimitError is now a HelixError
# ---------------------------------------------------------------------------


class TestFix1RateLimitErrorInheritance:
    def test_rate_limit_error_is_helix_error(self) -> None:
        """isinstance(RateLimitError(...), HelixError) must be True after Fix 1."""
        err = RateLimitError("api rate limit hit")
        assert isinstance(err, HelixError)

    def test_rate_limit_error_is_not_mutation_error(self) -> None:
        """RateLimitError must NOT be a MutationError (it's a sibling, not a subclass).

        This ensures the ``except MutationError`` guard in mutate() / merge()
        does NOT swallow RateLimitError.
        """
        err = RateLimitError("rate limit")
        assert not isinstance(err, MutationError)

    def test_rate_limit_error_carries_message(self) -> None:
        msg = "Claude Code hit a rate/usage limit (exit code 529)"
        err = RateLimitError(msg)
        assert str(err) == msg

    def test_rate_limit_error_has_helix_error_attributes(self) -> None:
        """RateLimitError inherits the structured context attributes from HelixError."""
        err = RateLimitError(
            "rate limit",
            operation="mutate g1-s1",
            phase="subprocess exit",
            exit_code=1,
        )
        assert err.operation == "mutate g1-s1"
        assert err.phase == "subprocess exit"
        assert err.exit_code == 1


# ---------------------------------------------------------------------------
# Fix 2: RateLimitError carries full diagnostics (mirrors MutationError)
# ---------------------------------------------------------------------------


class TestFix2RateLimitDiagnostics:
    def test_rate_limit_error_accepts_all_diagnostic_fields(self) -> None:
        """RateLimitError can be constructed with the same fields as MutationError."""
        err = RateLimitError(
            "Claude Code hit a rate/usage limit (exit code 529)",
            operation="Claude Code invocation",
            phase="subprocess exit",
            command="claude --print -p ...",
            cwd="/tmp/worktree",
            stdout="",
            stderr="Error: 529 overloaded",
            exit_code=529,
            suggestion="Claude Code reported a rate limit. Retry after backoff.",
        )
        assert err.operation == "Claude Code invocation"
        assert err.phase == "subprocess exit"
        assert err.command == "claude --print -p ..."
        assert err.cwd == "/tmp/worktree"
        assert err.stdout == ""
        assert err.stderr == "Error: 529 overloaded"
        assert err.exit_code == 529
        assert "rate limit" in err.suggestion.lower()

    def test_invoke_claude_code_raises_rate_limit_with_diagnostics_on_stderr(
        self, mocker: MagicMock
    ) -> None:
        """invoke_claude_code raises RateLimitError with full diagnostics when stderr
        contains a rate-limit keyword and exit code is non-zero."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: 529 overloaded please retry"
        mocker.patch("helix.mutator.subprocess.run", return_value=mock_result)

        config = ClaudeConfig()
        with pytest.raises(RateLimitError) as exc_info:
            invoke_claude_code("/tmp/wt", "test prompt", config)

        err = exc_info.value
        assert err.exit_code == 1
        assert err.stderr == "Error: 529 overloaded please retry"
        assert err.stdout == ""
        assert err.cwd == "/tmp/wt"
        assert err.command != ""
        assert err.suggestion != ""

    def test_invoke_claude_code_raises_rate_limit_with_diagnostics_on_json(
        self, mocker: MagicMock
    ) -> None:
        """invoke_claude_code raises RateLimitError with full diagnostics when the
        returned JSON ``error`` field contains a rate-limit keyword."""
        import json

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"error": "usage limit exceeded"})
        mock_result.stderr = ""
        mocker.patch("helix.mutator.subprocess.run", return_value=mock_result)

        config = ClaudeConfig()
        with pytest.raises(RateLimitError) as exc_info:
            invoke_claude_code("/tmp/wt", "test prompt", config)

        err = exc_info.value
        assert err.exit_code == 0
        assert err.cwd == "/tmp/wt"
        assert err.command != ""
        assert err.suggestion != ""

    def test_rate_limit_error_format_full_includes_all_fields(self) -> None:
        """format_full() on RateLimitError should include all diagnostic fields."""
        err = RateLimitError(
            "rate limit hit",
            operation="invoke",
            phase="exit",
            command="claude -p ...",
            cwd="/tmp",
            stdout="out",
            stderr="rate limit exceeded",
            exit_code=1,
            suggestion="retry later",
        )
        full = err.format_full()
        assert "rate limit hit" in full
        assert "invoke" in full
        assert "exit" in full
        assert "claude -p ..." in full
        assert "/tmp" in full
        assert "out" in full
        assert "rate limit exceeded" in full
        assert "retry later" in full


# ---------------------------------------------------------------------------
# Fix 3: Top-level RateLimitError catch in cli.py evolve
# ---------------------------------------------------------------------------


class TestFix3CliEvolveRateLimitCatch:
    def _make_minimal_toml(self, tmp_path: Path) -> Path:
        """Write the minimum helix.toml for `helix evolve` to parse."""
        toml_path = tmp_path / "helix.toml"
        toml_path.write_text(
            'objective = "test"\n\n[evaluator]\ncommand = "true"\n'
        )
        return toml_path

    def test_evolve_exits_nonzero_on_rate_limit(self, mocker: MagicMock, tmp_path: Path) -> None:
        """evolve must exit with a non-zero code when run_evolution raises RateLimitError."""
        self._make_minimal_toml(tmp_path)

        mocker.patch(
            "helix.evolution.run_evolution",
            side_effect=RateLimitError("api rate limit"),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["evolve", "--dir", str(tmp_path), "--config", str(tmp_path / "helix.toml")],
        )

        assert result.exit_code != 0, (
            f"evolve must exit non-zero on RateLimitError; got {result.exit_code}"
        )

    def test_evolve_prints_friendly_message_on_rate_limit(
        self, mocker: MagicMock, tmp_path: Path
    ) -> None:
        """evolve must print a message suggesting helix resume."""
        self._make_minimal_toml(tmp_path)

        mocker.patch(
            "helix.evolution.run_evolution",
            side_effect=RateLimitError("api rate limit"),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["evolve", "--dir", str(tmp_path), "--config", str(tmp_path / "helix.toml")],
        )

        combined_output = result.output or ""
        assert "resume" in combined_output.lower() or "rate limit" in combined_output.lower(), (
            f"Expected 'resume' or 'rate limit' in output; got:\n{combined_output}"
        )

    def test_evolve_exit_code_2_on_rate_limit(self, mocker: MagicMock, tmp_path: Path) -> None:
        """evolve must specifically exit with code 2 on RateLimitError (distinct from other errors)."""
        self._make_minimal_toml(tmp_path)

        mocker.patch(
            "helix.evolution.run_evolution",
            side_effect=RateLimitError("api rate limit"),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["evolve", "--dir", str(tmp_path), "--config", str(tmp_path / "helix.toml")],
        )

        assert result.exit_code == 2, (
            f"Expected exit code 2 for rate limit; got {result.exit_code}"
        )


# ---------------------------------------------------------------------------
# Fix 4: Structured file logging via setup_file_logging
# ---------------------------------------------------------------------------


class TestFix4FileLogging:
    def test_setup_file_logging_creates_helix_dir(
        self, tmp_path: Path, clean_helix_logger: None
    ) -> None:
        """setup_file_logging creates the helix_dir if it does not exist."""
        helix_dir = tmp_path / ".helix"
        assert not helix_dir.exists()
        setup_file_logging(helix_dir)
        assert helix_dir.exists()

    def test_setup_file_logging_creates_log_file(
        self, tmp_path: Path, clean_helix_logger: None
    ) -> None:
        """setup_file_logging creates helix.log inside helix_dir."""
        helix_dir = tmp_path / ".helix"
        setup_file_logging(helix_dir)
        # The file itself is created lazily on first write; just verify the
        # handler is attached and the path is correct.
        helix_logger = logging.getLogger("helix")
        file_handlers = [
            h for h in helix_logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert file_handlers, "Expected at least one FileHandler on 'helix' logger"
        expected = str((helix_dir / "helix.log").resolve())
        assert any(h.baseFilename == expected for h in file_handlers)

    def test_setup_file_logging_no_duplicate_handlers(
        self, tmp_path: Path, clean_helix_logger: None
    ) -> None:
        """Calling setup_file_logging twice with the same path adds only one handler."""
        helix_dir = tmp_path / ".helix"
        setup_file_logging(helix_dir)
        setup_file_logging(helix_dir)

        helix_logger = logging.getLogger("helix")
        expected = str((helix_dir / "helix.log").resolve())
        count = sum(
            1
            for h in helix_logger.handlers
            if isinstance(h, logging.FileHandler) and h.baseFilename == expected
        )
        assert count == 1, f"Expected 1 FileHandler for path, got {count}"

    def test_print_helix_error_writes_to_log_file(
        self, tmp_path: Path, clean_helix_logger: None
    ) -> None:
        """After setup_file_logging, print_helix_error writes to helix.log."""
        helix_dir = tmp_path / ".helix"
        setup_file_logging(helix_dir)

        err = MutationError(
            "subprocess died unexpectedly",
            operation="mutate g1-s0",
            phase="subprocess exit",
            command="claude --print -p ...",
            cwd="/tmp/wt",
            stdout="some stdout text",
            stderr="some stderr text",
            exit_code=1,
            suggestion="check permissions",
        )
        print_helix_error(err)

        # Flush all handlers to ensure the write landed
        for handler in logging.getLogger("helix").handlers:
            handler.flush()

        log_path = helix_dir / "helix.log"
        assert log_path.exists(), "helix.log was not created"
        content = log_path.read_text()
        assert "subprocess died unexpectedly" in content
        assert "mutate g1-s0" in content
        assert "subprocess exit" in content
        assert "check permissions" in content

    def test_log_file_captures_rate_limit_error(
        self, tmp_path: Path, clean_helix_logger: None
    ) -> None:
        """RateLimitError diagnostics appear in helix.log via print_helix_error."""
        helix_dir = tmp_path / ".helix"
        setup_file_logging(helix_dir)

        err = RateLimitError(
            "Claude Code hit a rate/usage limit (exit code 529)",
            operation="Claude Code invocation",
            phase="subprocess exit",
            command="claude --print -p test",
            cwd="/tmp/wt",
            stdout="",
            stderr="Error: 529 overloaded",
            exit_code=529,
            suggestion="Retry after backoff or check your API quota.",
        )
        print_helix_error(err)

        for handler in logging.getLogger("helix").handlers:
            handler.flush()

        log_path = helix_dir / "helix.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "rate/usage limit" in content
        assert "Claude Code invocation" in content
        assert "Error: 529 overloaded" in content
