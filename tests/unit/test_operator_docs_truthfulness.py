"""Operator-facing docs must match what the code does.

No test guarded ``docs/sandbox-auth.md`` or ``CHANGELOG.md`` at all -- the only
wording guard asserted against ``backend_layout.py``'s OWN docstring, i.e. a
declaration guarding itself. Meanwhile the LANE CONFIGS were guarded by
``test_lane_documents_the_env_mode_tradeoff``, so the pattern existed; it was
simply never pointed at the guide.

That gap let the guide keep stating a design the code explicitly repudiates,
and the retirement made it worse: the guide's documented DEFAULT became an
unconditional error.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE = REPO_ROOT / "docs" / "sandbox-auth.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"


def test_operator_docs_exist() -> None:
    """Non-vacuity: a renamed file must not silently drop all coverage below."""
    assert GUIDE.is_file() and CHANGELOG.is_file()


@pytest.mark.parametrize("path", [GUIDE, CHANGELOG])
def test_docs_state_volume_mode_is_unsupported_for_agents(path: Path) -> None:
    """Catches: docs describing volume mode as usable, or as "the default".

    Omitting ``sandbox.auth`` still resolves to ``"volume"``, which now raises
    for every backend -- so a guide calling it "the default" without saying it
    always fails documents a hard failure as normal operation. That is the most
    user-visible defect this change could ship.
    """
    text = path.read_text()
    assert "UNSUPPORTED" in text or "unsupported" in text, (
        f"{path.name} must state that volume mode is unsupported for agents"
    )
    assert '"volume"` (the default)' not in text, (
        f"{path.name} still calls volume mode the default without qualification"
    )


def test_guide_does_not_carry_the_repudiated_ro_reasoning() -> None:
    """Catches: the ``:ro`` justification surviving as current design.

    ``sandbox.py`` explicitly repudiates it in a code comment -- "that
    reasoning addressed the wrong risk ... read access is the defect, not write
    access". The guide carried the argument verbatim as live design, so the
    repudiated reasoning was the one an operator would read.

    The phrase may appear only in text that marks it as FORMER reasoning.
    """
    text = GUIDE.read_text()
    if "`:ro`" in text:
        assert "earlier version of this document" in text, (
            "the :ro reasoning may only appear as explicitly retired history"
        )
    assert "mounts NO auth volume" in text or "mounts no auth volume" in text


def test_guide_scopes_the_mount_claim_to_agent_containers() -> None:
    """Catches: the FALSE absolute "HELIX never mounts the auth volume".

    ``helix sandbox login`` genuinely mounts it at HOME -- that is its purpose
    -- so an unscoped claim would be untrue, and would be F-14 repeating itself
    in the opposite direction.
    """
    text = GUIDE.read_text()
    assert "HELIX never mounts the auth volume" not in text
    assert "AGENT container" in text or "agent container" in text
