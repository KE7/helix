"""HELIX shared exception types and error formatting utilities.

All HELIX modules use these for consistent, never-truncated error diagnostics.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.panel import Panel


# ---------------------------------------------------------------------------
# Module-level Rich console for error output
# ---------------------------------------------------------------------------

_error_console = Console(stderr=True, highlight=False)
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------


class HelixError(Exception):
    """Base error for all HELIX operations.

    Carries structured context (operation, phase, command, cwd, stdout,
    stderr, exit_code, suggestion) so callers can emit rich diagnostics
    without parsing exception messages.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str = "",
        phase: str = "",
        command: str = "",
        cwd: str = "",
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        suggestion: str = "",
    ) -> None:
        self.operation = operation
        self.phase = phase
        self.command = command
        self.cwd = cwd
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.suggestion = suggestion
        super().__init__(message)

    def format_full(self) -> str:
        """Return the complete, never-truncated error report."""
        lines = [f"[HELIX ERROR] {self}"]
        if self.operation:
            lines.append(f"[HELIX ERROR] Operation: {self.operation}")
        if self.phase:
            lines.append(f"[HELIX ERROR] Phase: {self.phase}")
        if self.command:
            lines.append(f"[HELIX ERROR] Command: {self.command}")
        if self.exit_code is not None:
            lines.append(f"[HELIX ERROR] Exit code: {self.exit_code}")
        if self.cwd:
            lines.append(f"[HELIX ERROR] Working dir: {self.cwd}")
        if self.stdout:
            lines.append(f"[HELIX ERROR] Stdout:\n{self.stdout}")
        if self.stderr:
            lines.append(f"[HELIX ERROR] Stderr:\n{self.stderr}")
        if self.suggestion:
            lines.append(f"[HELIX ERROR] Suggestion: {self.suggestion}")
        return "\n".join(lines)


class GitError(HelixError):
    """Raised when a git subprocess fails."""


class MutationError(HelixError):
    """Raised when a mutation attempt fails (timeout, non-zero exit, bad JSON)."""


class PromptArtifactCollisionError(HelixError):
    """Raised when HELIX cannot reserve its prompt artifact path safely."""


class EvaluatorError(HelixError):
    """Raised when an evaluator subprocess fails."""


class RateLimitError(HelixError):
    """Raised when Claude Code hits a rate / usage limit.

    Inherits from :class:`HelixError` so that the parallel futures handler in
    ``evolution.py`` (which catches ``HelixError``) can route it properly —
    logging the failure and returning ``None`` for that proposal slot rather
    than crashing the entire run.

    Detected heuristically from non-zero exit codes or JSON error fields that
    contain keywords such as "rate limit", "overloaded", "529", "usage limit",
    or "extra usage".  After retries are exhausted the error bubbles up to the
    evolution loop, which logs it and continues with a smaller proposal set.
    """


# ---------------------------------------------------------------------------
# Formatted error printing
# ---------------------------------------------------------------------------


def print_helix_error(exc: HelixError) -> None:
    """Print a HelixError to stderr using Rich, with full context.

    Never truncates any field — stdout, stderr, and all diagnostics are
    printed in their entirety.
    """
    lines: list[str] = []
    lines.append(f"[bold red]{exc}[/bold red]")
    if exc.operation:
        lines.append(f"[red]Operation:[/red] {exc.operation}")
    if exc.phase:
        lines.append(f"[red]Phase:[/red] {exc.phase}")
    if exc.command:
        lines.append(f"[red]Command:[/red] {exc.command}")
    if exc.exit_code is not None:
        lines.append(f"[red]Exit code:[/red] {exc.exit_code}")
    if exc.cwd:
        lines.append(f"[red]Working dir:[/red] {exc.cwd}")
    if exc.stdout:
        lines.append(f"[red]Stdout (full):[/red]\n{exc.stdout}")
    if exc.stderr:
        lines.append(f"[red]Stderr (full):[/red]\n{exc.stderr}")
    if exc.suggestion:
        lines.append(f"[yellow]Suggestion:[/yellow] {exc.suggestion}")

    panel = Panel(
        "\n".join(lines),
        title="[bold red]HELIX ERROR[/bold red]",
        border_style="red",
        expand=False,
    )
    _error_console.print(panel)

    # Also emit to the structured log file (captured by setup_file_logging).
    _logger.error(exc.format_full())


def format_error_context(
    *,
    operation: str = "",
    phase: str = "",
    command: str = "",
    cwd: str = "",
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = None,
    suggestion: str = "",
) -> str:
    """Build a formatted, never-truncated error context string.

    Useful for log messages or exception messages where Rich is not available.
    """
    lines: list[str] = []
    if operation:
        lines.append(f"[HELIX ERROR] Operation: {operation}")
    if phase:
        lines.append(f"[HELIX ERROR] Phase: {phase}")
    if command:
        lines.append(f"[HELIX ERROR] Command: {command}")
    if exit_code is not None:
        lines.append(f"[HELIX ERROR] Exit code: {exit_code}")
    if cwd:
        lines.append(f"[HELIX ERROR] Working dir: {cwd}")
    if stdout:
        lines.append(f"[HELIX ERROR] Stdout (full):\n{stdout}")
    if stderr:
        lines.append(f"[HELIX ERROR] Stderr (full):\n{stderr}")
    if suggestion:
        lines.append(f"[HELIX ERROR] Suggestion: {suggestion}")
    return "\n".join(lines)
