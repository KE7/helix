"""Value-aware redaction for diagnostics crossing into agent prompts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

REDACTED_VALUE: Final = "<redacted>"
_MIN_REDACTABLE_VALUE_LENGTH: Final = 8


@dataclass(frozen=True)
class DiagnosticRedactor:
    """Replace configured secret values in nested diagnostics, preserving keys."""

    replacements: tuple[str, ...] = ()

    @classmethod
    def from_values(cls, values: Iterable[object]) -> DiagnosticRedactor:
        literals = {
            value for value in values if isinstance(value, str) and len(value) >= 8
        }
        forms: set[str] = set(literals)
        for value in literals:
            forms.add(json.dumps(value, ensure_ascii=False)[1:-1])
            forms.add(json.dumps(value, ensure_ascii=True)[1:-1])
        return cls(tuple(sorted(forms, key=len, reverse=True)))

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for replacement in self.replacements:
                value = value.replace(replacement, REDACTED_VALUE)
            return value
        if isinstance(value, dict):
            return {key: self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact(item) for item in value)
        return value


def redact_diagnostics(value: Any, secret_values: Iterable[object]) -> Any:
    """Return a copy with each of ``secret_values`` replaced wherever it appears."""
    return DiagnosticRedactor.from_values(secret_values).redact(value)
