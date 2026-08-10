"""Value-aware redaction for diagnostics rendered outside a trust boundary.

The redactor protects a configured value only in its literal spelling or the
JSON-string escaped spelling of that value.  It intentionally does not attempt
to discover transformed, partial, or line-wrapped forms (for example base64,
URL encoding, hexadecimal, or a token prefix): decoding arbitrary transforms
is both incomplete and prone to false positives.  A known encoded environment
value is protected when that exact encoded value is supplied to
:meth:`DiagnosticRedactor.from_values`.

Values shorter than eight characters and empty values are deliberately ignored
to keep ordinary diagnostic text useful; this means sub-eight-character
secrets are outside this utility's protection guarantee.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

REDACTED_VALUE: Final = "<redacted>"
_MIN_REDACTABLE_VALUE_LENGTH: Final = 8


@dataclass(frozen=True)
class DiagnosticRedactor:
    """Replace configured secret values in strings and nested diagnostics."""

    replacements: tuple[str, ...] = ()

    @classmethod
    def from_values(cls, values: Iterable[object]) -> DiagnosticRedactor:
        """Build a redactor from configured values safe to match literally."""
        literals = {
            value
            for value in values
            if isinstance(value, str) and len(value) >= _MIN_REDACTABLE_VALUE_LENGTH
        }
        forms: set[str] = set(literals)
        for value in literals:
            forms.add(json.dumps(value, ensure_ascii=False)[1:-1])
            forms.add(json.dumps(value, ensure_ascii=True)[1:-1])
        return cls(tuple(sorted(forms, key=len, reverse=True)))

    def redact(self, value: Any) -> Any:
        """Return a recursively redacted copy, preserving keys and primitives."""
        if isinstance(value, str):
            redacted = value
            for replacement in self.replacements:
                redacted = redacted.replace(replacement, REDACTED_VALUE)
            return redacted
        if isinstance(value, dict):
            return {key: self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact(item) for item in value)
        return value


def redact_diagnostics(value: Any, secret_values: Iterable[object]) -> Any:
    """Convenience wrapper for one-off diagnostic rendering boundaries."""
    return DiagnosticRedactor.from_values(secret_values).redact(value)
