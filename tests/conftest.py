"""Root pytest conftest for HELIX.

Responsibilities:
- Ensure the HELIX TraceBus is disabled between tests (a test that forgets
  to exit ``TRACE.record()`` must not leak state into the next test).
- Prepend the GEPA differential-testing fixture root to ``sys.path`` so the
  new ``tests/unit/gepa_diff/`` package imports cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the diff-harness fixture package importable as a top-level module tree.
_DIFF_ROOT = Path(__file__).parent / "unit" / "gepa_diff"
if str(_DIFF_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(_DIFF_ROOT.parent.parent))


#: The operator's real home, captured before any test redirects ``$HOME``.
REAL_HOME = Path.home().resolve()


def reads_real_home_state(path: object) -> bool:
    """True for a path inside a dot-directory of the real home (``~/.local``...)."""
    resolved = Path(str(path)).resolve()
    if not resolved.is_relative_to(REAL_HOME) or resolved == REAL_HOME:
        return False
    return resolved.relative_to(REAL_HOME).parts[0].startswith(".")


@pytest.fixture(autouse=True)
def _hermetic_home(tmp_path_factory, monkeypatch):
    """Point ``$HOME`` at an empty directory so no test reads the real one.

    Backend transcript collection globs the agent CLIs' stores under
    ``Path.home()`` when unsandboxed; with the real home a test copied the
    operator's ``~/.local/share/opencode/opencode.db`` (gigabytes, with the
    opencode account's tokens in it) into pytest's temp dir and filled the
    disk.  ``shutil.copy2`` additionally refuses any source in a dot-directory
    of the real home (where every agent CLI keeps its state), so a code path
    that resolves the home some other way fails loudly; the checkout itself
    may live under the real home and stays copyable.  Git identity comes from
    the environment, since ``~/.gitconfig`` is gone.
    """
    import shutil

    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    for name in ("XDG_DATA_HOME", "XDG_STATE_HOME", "HELIX_CLAUDE_TRANSCRIPT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    for name in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(name, "HELIX Tests")
    for name in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(name, "tests@helix.invalid")

    real_copy2 = shutil.copy2

    def guarded_copy2(src, dst, *args, **kwargs):
        if reads_real_home_state(src):
            raise AssertionError(f"test copied {src} from the real home directory")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", guarded_copy2)
    yield home


@pytest.fixture(autouse=True)
def _trace_bus_clean_between_tests():
    """Guarantee ``TRACE.enabled`` is False and ``TRACE.events`` empty per test.

    Tests that use the diff harness opt into ``TRACE.record()`` explicitly;
    any leak (e.g. a raised exception inside ``record()``) would otherwise
    keep the bus enabled and contaminate subsequent tests.
    """
    try:
        from helix.trace import TRACE
    except Exception:
        # HELIX not importable in this environment — nothing to guard.
        yield
        return
    _prev_enabled = TRACE.enabled
    _prev_events = list(TRACE.events)
    try:
        yield
    finally:
        TRACE.enabled = False
        TRACE.events = []
        # Restore whatever the test harness set up pre-test (should be idle).
        TRACE.enabled = _prev_enabled if not TRACE.enabled else False
        if not TRACE.events:
            TRACE.events = _prev_events if _prev_events else []
