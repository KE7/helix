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
