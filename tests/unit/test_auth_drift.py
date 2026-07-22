"""Drift detector: fires on agent-invented entries, and is honest about limits.

The most important tests here are the two that assert the detector FAILS to
detect something.  A control that quietly overstates its coverage is worse than
no control, because it converts "we do not know" into "we checked".
"""

from __future__ import annotations

import pytest

from pathlib import Path

from helix.auth_drift import (
    AuthStoreDriftError,
    assert_no_drift,
    detect_drift,
    expected_entries,
)
from helix.backend_layout import BACKEND_LAYOUTS


CODEX = BACKEND_LAYOUTS["codex"]


def test_known_layout_entries_are_not_drift() -> None:
    """Catches: a detector that fires on the CLI's own files.

    A control that cries wolf on every run gets disabled, which is a slower
    path to the same silence.
    """
    observed = [
        CODEX.credential_file,
        *CODEX.ephemeral_subdirs,
        *CODEX.stable_files,
    ]
    assert detect_drift(CODEX, observed).clean


def test_agent_invented_entry_is_detected() -> None:
    """The channel this exists for: an unenumerated file in the shared dir.

    Proven reachable by canary through the full isolation layout -- candidate A
    wrote it, candidate B read it verbatim.
    """
    drift = detect_drift(CODEX, [CODEX.credential_file, "notes-for-next-candidate.txt"])
    assert not drift.clean
    assert drift.unexpected == ("notes-for-next-candidate.txt",)


def test_missing_entries_are_not_drift() -> None:
    """Catches: treating a fresh volume as compromised.

    Absence is normal -- a newly provisioned store has almost none of these --
    and firing on it would make the detector useless on first run.
    """
    assert detect_drift(CODEX, [CODEX.credential_file]).clean


def test_drift_fails_loudly_and_says_it_never_cleans() -> None:
    """Catches: a detector that logs, or worse, one that tidies up.

    The shared volume holds root-owned incident evidence; an automatic cleaner
    would destroy what an investigation needs. Deletion is prohibited by
    POLICY, not by permissions -- a --user node process can unlink those files
    -- so the message has to say so explicitly.
    """
    drift = detect_drift(CODEX, ["exfil.txt"])
    with pytest.raises(AuthStoreDriftError) as exc:
        assert_no_drift(drift)
    message = str(exc.value)
    assert "exfil.txt" in message
    assert "has NOT removed anything" in message
    assert 'auth = "env"' in message, "must point at the mode without the hole"


def test_detector_module_cannot_delete_anything() -> None:
    """The "never cleans" half, held in place STRUCTURALLY.

    The test above asserts the MESSAGE says nothing was removed. That is not
    the same property: a detector that deleted files while printing "has NOT
    removed anything" would pass it. A docstring claiming to catch "one that
    tidies up" has to be backed by something that actually would.

    So this asserts the module has no capability to mutate the filesystem at
    all -- no unlink, no rmtree, no rename, no chmod, and no subprocess to do
    it indirectly. The shared volume holds root-owned incident evidence, and a
    ``--user node`` process CAN unlink it (write+execute on the parent), so the
    prohibition is policy and needs a structural guard rather than trust.
    """
    import ast

    import helix.auth_drift as mod

    source = Path(mod.__file__).read_text()
    tree = ast.parse(source)

    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    for dangerous in ("os", "shutil", "subprocess", "pathlib"):
        assert dangerous not in imported, (
            f"auth_drift imports {dangerous!r}; it must have no means of "
            f"modifying or removing anything in the auth store"
        )

    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for verb in ("unlink", "remove", "rmtree", "rename", "chmod", "replace"):
        assert verb not in called, f"auth_drift calls {verb}()"


def test_report_carries_names_only_never_contents() -> None:
    """Catches: a diagnostic that opens the offending file.

    The detector must work on files it cannot read, and must never risk
    putting credential or transcript CONTENT into an error message.
    """
    drift = detect_drift(CODEX, ["secret-looking-name.json"])
    assert drift.unexpected == ("secret-looking-name.json",)
    # the dataclass has no field capable of holding content
    assert set(vars(drift)) == {"backend", "auth_dir", "unexpected"}


# ---------------------------------------------------------------------------
# The detector's OWN limits, asserted so they cannot be quietly forgotten
# ---------------------------------------------------------------------------


def test_write_read_delete_is_INVISIBLE_to_this_detector() -> None:
    """The sharper limitation, asserted rather than only documented.

    Candidate A writes, candidate B reads, candidate B deletes. At the end of
    the run the directory matches the expected set exactly, so the detector
    reports CLEAN -- while a full cross-candidate channel carried information
    and left no trace.

    NO end-of-run comparison can see this, including a better implementation
    of this one. This test exists so that a future reader cannot mistake a
    clean report for proof that no channel existed.
    """
    # what the directory looks like AFTER B deleted the file it read
    after_delete = [CODEX.credential_file, *CODEX.ephemeral_subdirs]
    drift = detect_drift(CODEX, after_delete)
    assert drift.clean, (
        "a clean report here is CORRECT and is precisely the limitation: the "
        "channel existed and completed, and end-of-run comparison cannot see it"
    )


def test_a_clean_report_is_not_evidence_of_isolation() -> None:
    """The docstring must keep saying so, in the module that would be cited.

    Catches: someone later trimming the caveats and leaving a control that
    reads as a guarantee.
    """
    import helix.auth_drift as mod

    doc = (mod.__doc__ or "").lower()
    assert "detection, not prevention" in doc
    assert "write-read-delete" in doc
    assert "race" in doc
    assert "never" in doc and "clean" in doc


def test_expected_set_is_derived_from_the_registry_not_duplicated() -> None:
    """Catches: a second, drifting copy of the layout inside the detector.

    Two hand-maintained lists of the same thing diverge; when they do, the
    detector fires on legitimate files and gets switched off.
    """
    expected = expected_entries(CODEX)
    assert CODEX.credential_file in expected
    for name in CODEX.ephemeral_subdirs:
        assert name in expected
    for name in CODEX.stable_files:
        assert name in expected
    # and it is genuinely a subset relationship, not an accept-everything set
    assert "definitely-not-a-real-entry" not in expected
