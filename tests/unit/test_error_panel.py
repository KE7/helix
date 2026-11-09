"""print_helix_error prints every field literally, never as Rich markup."""

from __future__ import annotations

import io

from rich.console import Console

from helix import exceptions
from helix.exceptions import MutationError


def _render(exc: Exception, monkeypatch) -> str:
    buffer = io.StringIO()
    monkeypatch.setattr(
        exceptions, "_error_console", Console(file=buffer, width=200, force_terminal=False)
    )
    exceptions.print_helix_error(exc)
    return buffer.getvalue()


def test_bracketed_text_is_printed_literally(monkeypatch) -> None:
    exc = MutationError(
        "Codex CLI timed out after 45s (sandbox.timeout_seconds)",
        suggestion="Raise [sandbox].timeout_seconds, or lower agent.max_turns.",
        stderr="model said [/INST] and a[i]",
    )
    out = _render(exc, monkeypatch)
    assert "Raise [sandbox].timeout_seconds" in out
    assert "[/INST] and a[i]" in out
