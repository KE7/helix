"""Unit tests for structured file logging via setup_file_logging()."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import pytest

from helix.exceptions import MutationError, print_helix_error
from helix.logging_config import setup_file_logging


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


class TestFileLogging:
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
