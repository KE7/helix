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

import re
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


def test_every_mount_claim_in_the_guide_carries_a_scope_qualifier() -> None:
    """POSITIVE property, replacing a single-literal denylist.

    The previous version asserted that one exact string --
    ``HELIX never mounts the auth volume`` -- was absent. That is a
    single-literal denylist standing in for a WORDING CLASS, in a document
    nobody writes from a template: the exact literal was caught, while the
    paraphrase "The auth volume is never mounted over HOME by HELIX." sailed
    through. F-14 recurring is exactly what this guard exists to prevent, and
    it was one paraphrase away.

    So the property is asserted directly: any sentence that talks about
    MOUNTING the AUTH VOLUME must also name WHO -- agent, login, status, or
    logout. Unscoped, such a sentence is false in one direction or the other,
    because agent containers no longer mount it and login still does.
    """
    text = GUIDE.read_text()
    # crude sentence split is fine here: we want over- rather than
    # under-inclusion, and a fragment that mentions both terms still must scope
    sentences = re.split(r"(?<=[.!?])\s+|\n\n", text)
    qualifiers = ("agent", "login", "status", "logout", "`helix sandbox")

    offenders = [
        sentence.strip()
        for sentence in sentences
        if "mount" in sentence.lower()
        and "auth volume" in sentence.lower()
        and not any(word in sentence.lower() for word in qualifiers)
    ]
    assert not offenders, (
        "unscoped claims about mounting the auth volume -- each must say WHO, "
        "since agent containers no longer mount it and `helix sandbox login` "
        f"still does: {offenders}"
    )


def test_the_scope_guard_actually_fires_on_an_unscoped_paraphrase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity, in the direction the old guard failed.

    The old literal check passed on a paraphrase. This proves the replacement
    catches one, rather than merely catching the original literal.
    """
    import tests.unit.test_operator_docs_truthfulness as module

    fake = tmp_path / "sandbox-auth.md"
    fake.write_text(
        GUIDE.read_text() + "\n\nThe auth volume is never mounted over HOME by HELIX.\n"
    )
    monkeypatch.setattr(module, "GUIDE", fake)
    with pytest.raises(AssertionError):
        module.test_every_mount_claim_in_the_guide_carries_a_scope_qualifier()
