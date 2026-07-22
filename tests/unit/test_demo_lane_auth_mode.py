"""Demo lanes must select env mode EXPLICITLY.

Volume mode cannot support a candidate-independence claim, and omission
resolves to volume mode silently (``SandboxConfig.resolved_auth`` returns
``self.auth or "volume"``).  So a lane config that simply says nothing gets the
mode whose results must not be read as per-candidate measurements -- with
nothing anywhere saying so.

This suite makes that failure loud at test time instead of at publication time.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

# Lanes whose results are published as per-candidate measurements.
#
# NOTE: the algotune lane has no config under ``examples/`` in this tree, so it
# cannot be asserted here.  That is recorded rather than silently omitted --
# see ``test_declared_lane_configs_all_exist``.
INDEPENDENCE_LANES = (
    "examples/formulacode/helix.toml.template",
    "examples/livebench_math/helix.toml",
    "examples/swebench_live/helix.toml",
)


def _sandbox(rel: str) -> dict[str, object]:
    data = tomllib.loads((REPO_ROOT / rel).read_text())
    sandbox = data.get("sandbox")
    assert isinstance(sandbox, dict), f"{rel}: no [sandbox] section"
    return sandbox


def test_declared_lane_configs_all_exist() -> None:
    """Non-vacuity: a renamed or moved config must not silently drop coverage."""
    for rel in INDEPENDENCE_LANES:
        assert (REPO_ROOT / rel).is_file(), rel


@pytest.mark.parametrize("rel", INDEPENDENCE_LANES)
def test_lane_selects_env_auth_explicitly(rel: str) -> None:
    """Catches: a lane omitting ``auth`` and silently resolving to volume mode.

    Omission is the dangerous case precisely because it looks like nothing.
    """
    sandbox = _sandbox(rel)
    assert sandbox.get("auth") == "env", (
        f'{rel}: sandbox.auth must be explicitly "env". Omission resolves to '
        f'"volume", which cannot support a candidate-independence claim.'
    )


@pytest.mark.parametrize("rel", INDEPENDENCE_LANES)
def test_lane_declares_a_usable_credential_allowlist(rel: str) -> None:
    """Env mode with an empty allowlist injects nothing.

    ``config.py`` rejects that combination at load time; asserting it here
    means a broken migration fails in CI rather than at lane start.
    """
    sandbox = _sandbox(rel)
    allow = sandbox.get("auth_env_allow")
    assert isinstance(allow, list) and allow, f"{rel}: auth_env_allow must be non-empty"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in allow, (
        f"{rel}: CLAUDE_CODE_OAUTH_TOKEN is forbidden -- it corrupts the "
        f"credential record so refresh becomes impossible"
    )


@pytest.mark.parametrize("rel", INDEPENDENCE_LANES)
def test_lane_documents_the_env_mode_tradeoff(rel: str) -> None:
    """The tradeoff must be stated where the choice is made.

    Env mode is not a free win: the named host credential is present inside the
    agent container and OAuth refresh is suppressed.  A config that selects it
    without saying so reads as an equivalent alternative, which is exactly the
    softening the disclosure rule forbids.
    """
    text = (REPO_ROOT / rel).read_text().lower()
    assert "refresh" in text, f"{rel}: must disclose that OAuth refresh is suppressed"
    assert "agent container" in text, (
        f"{rel}: must disclose that the credential is present in the container"
    )
