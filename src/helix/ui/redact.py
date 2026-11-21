"""Mask secrets in any text ``helix ui`` shows: transcripts, side_info, diffs, commands.

One helper, :func:`redact`, so every view masks the same way.  It masks known
token shapes, ``KEY=value`` assignments whose name looks secret, and the
values of credential environment variables present in this process: the
``helix.backends.BACKEND_AUTH_ENV`` names plus any names a page registers with
:func:`add_env_names` (the project's ``passthrough_env`` and ``env`` keys).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from helix.backends import BACKEND_AUTH_ENV

MASK = "••••"

_TOKEN_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
)
_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/-]+=*")
# ``NAME=value`` where NAME contains KEY, TOKEN, SECRET, PASSWORD, PASS or AUTH;
# a quoted value is masked whole, an unquoted one up to the next whitespace.
_ASSIGNMENT = re.compile(
    r"\b([A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASS|AUTH)[A-Za-z0-9_]*)="
    r"(\"[^\"]*\"|'[^']*'|[^\s\"']+)"
)
# Shorter values (``MUJOCO_GL=egl``, ``CUDA_VISIBLE_DEVICES=0``) are not
# secrets, and masking them everywhere would wreck the text.
_MIN_ENV_VALUE = 8

_env_names: set[str] = {name for names in BACKEND_AUTH_ENV.values() for name in names}


def add_env_names(names: Iterable[str]) -> None:
    """Also mask the values these environment variables have in this process."""
    _env_names.update(names)


def redact(text: str, *, extra_values: Iterable[str] = ()) -> str:
    """Return *text* with secrets and any supplied environment values masked."""
    values = set(extra_values)
    values.update(os.environ.get(name, "") for name in _env_names)
    for value in sorted((v for v in values if len(v) >= _MIN_ENV_VALUE), key=len, reverse=True):
        text = text.replace(value, MASK)
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(MASK, text)
    text = _BEARER.sub(lambda m: m.group(1) + MASK, text)
    return _ASSIGNMENT.sub(lambda m: f"{m.group(1)}={MASK}", text)
