"""Parser registry for HELIX score parsers."""

from __future__ import annotations

from typing import Any, Callable

from helix.parsers import (
    exitcode,
    pytest as pytest_parser,
    json_accuracy,
    json_score,
    helix_result,
)


_PARSERS: dict[str, Callable[..., Any]] = {
    "pytest": pytest_parser.parse,
    "exitcode": exitcode.parse,
    "json_accuracy": json_accuracy.parse,
    "json_score": json_score.parse,
    "helix_result": helix_result.parse,
}


def get_parser(name: str) -> Callable[..., Any]:
    """Return the parse function for the given parser name.

    Args:
        name: Parser name, one of "pytest", "exitcode", "json_accuracy",
            "json_score", or "helix_result".

    Returns:
        A callable parse function.

    Raises:
        KeyError: If the parser name is not recognized.
    """
    if name not in _PARSERS:
        raise KeyError(f"Unknown parser: {name!r}. Available: {list(_PARSERS)}")
    return _PARSERS[name]
