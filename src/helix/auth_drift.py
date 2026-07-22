"""Drift detection for the shared auth directory under ``auth = "volume"``.

*** THIS IS DETECTION, NOT PREVENTION. ***

It cannot make volume mode candidate-independent and must never be described as
doing so.  It converts a SILENT cross-candidate channel into a NOISY one.  That
is worth having, and it is all it is.

Why prevention is impossible here
---------------------------------
The auth directory is mounted writable, because OAuth rotation rewrites the
credential by atomic rename inside the credential's own directory.  The agent
runs as ``node``, so it can create any file there.  Isolation mounts mask only
the paths enumerated in :mod:`helix.backend_layout`; a path invented at runtime
is by construction not one of them.

Two limitations, stated in order of severity
--------------------------------------------

1. **WRITE-READ-DELETE defeats it completely, and this is the sharper one.**
   Candidate A writes a file, candidate B reads it, candidate B deletes it.
   At the end of the run the directory matches the expected set exactly and the
   detector never fires.  The channel carried information and left no trace.
   **No end-of-run comparison can ever see this** -- not this implementation,
   and not a better one.  Closing it would require continuous observation of
   the directory, which is a different mechanism entirely and is not what this
   is.

2. **It races against concurrent candidates.** With
   ``num_parallel_proposals > 1`` several agents share the directory
   simultaneously.  A file created and removed between two observations is
   missed, and a file observed as unexpected cannot be attributed to a
   particular candidate.  A drift report identifies that SOMETHING wrote an
   unexpected entry, never who.

Never cleans
------------
On detection this reports and fails.  It does not delete, move, or modify
anything.  The shared volume contains root-owned files that are incident
evidence, and an automatic cleaner would destroy exactly the artifacts an
investigation needs.  Deletion is prohibited by policy, not by permissions --
a ``--user node`` process can in fact unlink them, which is why the prohibition
has to be explicit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from helix.backend_layout import BackendHomeLayout
from helix.exceptions import HelixError


class AuthStoreDriftError(HelixError):
    """An unexpected entry appeared in the shared auth directory."""


def expected_entries(layout: BackendHomeLayout) -> frozenset[str]:
    """Entry names the pinned runtime is known to create in the auth dir.

    Derived from the layout registry so the two cannot disagree: an entry that
    the registry classifies is, by definition, an entry we expect to see.
    """
    return frozenset(
        {layout.credential_file}
        | set(layout.ephemeral_subdirs)
        | set(layout.ephemeral_files)
        | set(layout.stable_files)
    )


@dataclass(frozen=True)
class AuthStoreDrift:
    """Names only -- never contents.

    The detector reads directory ENTRY NAMES. It does not open files, so it
    cannot leak credential or transcript content into a diagnostic, and it
    works on files it has no permission to read.
    """

    backend: str
    auth_dir: str
    unexpected: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.unexpected


def detect_drift(layout: BackendHomeLayout, observed: Iterable[str]) -> AuthStoreDrift:
    """Compare observed entry names against the expected set.

    Entries the registry knows about are ignored.  MISSING entries are NOT
    drift: a fresh volume legitimately lacks most of them, and treating absence
    as an error would make the detector fire on every first run.
    """
    expected = expected_entries(layout)
    unexpected = tuple(
        sorted(name for name in observed if name and name not in expected)
    )
    return AuthStoreDrift(
        backend=layout.backend,
        auth_dir=layout.auth_dir,
        unexpected=unexpected,
    )


def assert_no_drift(drift: AuthStoreDrift) -> None:
    """Fail loudly on drift.  Never cleans."""
    if drift.clean:
        return
    raise AuthStoreDriftError(
        f"unexpected entries appeared in the shared auth directory for "
        f"{drift.backend!r}:\n"
        f"  {drift.auth_dir}\n"
        f"  unexpected: {', '.join(drift.unexpected)}\n"
        f"\n"
        f"  These are not written by the pinned CLI, so something else created "
        f"them -- most likely a mutation agent. The auth directory is shared "
        f"ACROSS RUNS, so a later candidate can read them.\n"
        f"\n"
        f"  This is DETECTION, not prevention: an entry written and deleted "
        f"within the run is invisible here, so a clean report is NOT evidence "
        f"that no cross-candidate channel existed.\n"
        f"\n"
        f"  HELIX has NOT removed anything. Do not delete these files before "
        f"they are inspected.\n"
        f'  Remedy: use sandbox.auth = "env", which mounts no persistent '
        f"store at all."
    )
