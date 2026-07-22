"""Agent transcript capture with a typed, non-swallowing outcome.

What this replaces
------------------
Transcript capture used to run a SECOND container that re-mounted the
persistent auth volume ``:ro`` at ``/home/node`` and ``cp``'d the transcript
out.  That design had two independent defects:

1. **It depended on the leak.** The source path lived inside the shared auth
   volume, so capture only worked *because* HOME was shared across runs.  Any
   isolation fix breaks it silently -- the feature is coupled to the defect.

2. **It failed silently.** The shell was ``[ -f "$src" ] || exit 0`` and the
   call was ``_run_docker(args, check=False)`` with no return value and no
   logging.  A nonzero exit -- including a ``cp`` that cannot read a
   root-owned ``0600`` transcript, which is live in the real volume today --
   produced no error, no log line and no artifact.  ``missing`` and ``failed``
   were the same silent nothing, despite having different remedies.

The replacement reads from the **candidate-keyed host bind** established in
:mod:`helix.sandbox_home`.  The transcript is already on the host when the
container exits, so capture is a plain filesystem copy: **no container, no
Docker, and structurally no way to re-mount the auth volume.**

Everything here returns a typed outcome or raises.  Only an explicitly
disabled capture is silent.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from helix.exceptions import HelixError
from helix.sandbox_home import transcript_host_dir


TranscriptStatus = Literal["captured", "missing", "disabled"]


class TranscriptCaptureError(HelixError):
    """Transcript capture was expected to succeed and did not.

    Raised rather than logged: a run that was asked to preserve transcripts and
    could not is a result the caller must see.  Messages name the path and the
    remedy and never include transcript CONTENT.
    """


@dataclass(frozen=True)
class TranscriptOutcome:
    """The result of a capture attempt.

    ``missing`` and ``captured`` are both normal.  ``missing`` means the backend
    produced no transcript for this session (nothing to do); it is deliberately
    NOT collapsed with a copy failure, which is an error and raises.
    """

    status: TranscriptStatus
    session_id: str | None = None
    source: Path | None = None
    artifact: Path | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("captured", "disabled")


def _find_transcript(root: Path, session_id: str) -> Path | None:
    """Locate ``<session_id>.jsonl`` beneath the candidate's bind directory.

    Searched rather than joined to a fixed project key: the CLI derives that
    key from the working directory and it is not part of HELIX's contract.
    Scoping the search to the per-candidate bind keeps it unambiguous -- there
    is exactly one candidate's output in this tree.
    """
    if not root.is_dir():
        return None
    target = f"{session_id}.jsonl"
    for path in root.rglob(target):
        if path.is_file():
            return path
    return None


def capture_claude_transcript(
    *,
    workspace: Path,
    artifact_dir: str,
    session_id: str | None,
    enabled: bool,
    backend: str,
) -> TranscriptOutcome:
    """Copy this candidate's transcript out of its host bind.

    Never runs a container and never touches the persistent auth volume.
    """
    if backend != "claude" or not enabled:
        # The ONLY intentionally silent case.
        return TranscriptOutcome(status="disabled", detail="preservation disabled")

    if not session_id:
        return TranscriptOutcome(
            status="missing",
            detail="backend output contained no session id",
        )

    root = transcript_host_dir(workspace)
    source = _find_transcript(root, session_id)
    if source is None:
        return TranscriptOutcome(
            status="missing",
            session_id=session_id,
            detail=f"no {session_id}.jsonl beneath {root}",
        )

    # Detect an unreadable transcript by PERMISSION, not by attempting a read.
    #
    # This is deliberate and is a policy constraint, not an optimisation. The
    # real auth volume contains root-owned 0600 transcripts that are incident
    # evidence: they must be neither read nor deleted. ``os.access`` answers
    # "could this process read it" from metadata alone, so the failure is
    # detected without ever opening the file -- and without a partial read
    # that would both violate the policy and produce a truncated artifact.
    if not os.access(source, os.R_OK):
        raise TranscriptCaptureError(
            f"agent transcript is not readable: {source}\n"
            f"  session:  {session_id}\n"
            f"  The file exists but this process cannot read it, which usually "
            f"means it is owned by another uid (root-owned 0600 transcripts "
            f"exist in the legacy shared auth volume).\n"
            f"  HELIX will not silently drop a transcript it was asked to "
            f"preserve. It also will NOT delete or modify the file.\n"
            f"  Remedy: set sandbox.preserve_backend_transcripts = false to "
            f"opt out explicitly, or resolve the ownership of the path above."
        )

    destination = Path(workspace) / artifact_dir / "claude" / f"{session_id}.jsonl"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    except OSError as exc:
        # Redacted: path and errno only, never transcript content.
        raise TranscriptCaptureError(
            f"failed to copy agent transcript for session {session_id}: "
            f"{type(exc).__name__}: {exc.strerror or 'copy failed'}\n"
            f"  source:      {source}\n"
            f"  destination: {destination}"
        ) from exc

    return TranscriptOutcome(
        status="captured",
        session_id=session_id,
        source=source,
        artifact=destination,
    )
