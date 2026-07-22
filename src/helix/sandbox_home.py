"""Per-run HOME isolation primitives for sandboxed mutation agents.

Background
----------
Historically every sandboxed mutation agent mounted the persistent
``helix-auth-<backend>`` volume over the *entire* container HOME
(``/home/node``).  Because ``HOME`` is the mount point of a volume shared
across every run of a backend, one candidate's agent transcripts, sessions,
caches and config were readable -- and writable -- by every later *and
concurrent* candidate.  In a benchmark whose premise is that candidates are
independently generated and independently scored, that is a
benchmark-validity defect, not merely residue.

See ``docs/design/sandbox-home-isolation.md`` for the full architecture, the
five-backend layout table and the rejected alternatives (D1/D2/D3).

Scope of this module
--------------------
This module owns the *mechanical* pieces that both auth modes need:

- a uid-correct private per-run HOME, and
- a candidate-specific transcript location.

It deliberately does **not** know about the persistent auth volume.  Volume
mode's class-2/class-3 layout registry is separate; keeping the private-HOME
mechanics here means env mode does not have to depend on any of it.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Container identity
# ---------------------------------------------------------------------------

# All five runner images define ``node`` as uid/gid 1000, measured from the
# images themselves (see the layout table in the design doc).  The *default*
# user of every runner image is root, which is why this must be stated
# explicitly rather than inherited.
#
# This is a constant, not runtime discovery, because it is uniform today --
# but it is ENFORCED by a pinned-runtime guard rather than assumed.  A base
# image bump that moves ``node`` off 1000 must fail loudly: a tmpfs created
# with the wrong uid yields a HOME the agent cannot write, which breaks every
# mutation agent rather than degrading quietly.
NODE_UID = 1000
NODE_GID = 1000

CONTAINER_HOME = "/home/node"


def private_home_tmpfs_arg() -> list[str]:
    """Docker args for a writable, private, per-run ``/home/node``.

    The ``uid``/``gid`` options are load-bearing and are the single most
    common way to get this wrong.  A bare ``--tmpfs /home/node`` produces a
    directory owned by ``root:root`` mode 0755; the container runs as
    ``--user node``, so uid 1000 gets ``Permission denied`` writing its own
    HOME and *every* mutation agent fails.

    The alternative that also "works" -- ``tmpfs-mode=1777`` -- is
    deliberately not used: it yields a sticky world-writable HOME owned by
    root, which is weaker than simply owning the directory.
    """
    return [
        "--tmpfs",
        f"{CONTAINER_HOME}:rw,uid={NODE_UID},gid={NODE_GID},mode=0755",
    ]


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------

# Claude writes agent transcripts to ``~/.claude/projects/<project-key>/``.
# The project key is derived from the working directory, which is ``/workspace``
# for EVERY candidate of EVERY run -- so the historical default
# ``/home/node/.claude/projects/-workspace`` was a single directory shared by
# all candidates.  That shared key is the concrete cross-candidate channel and
# is eliminated by binding the ``projects`` directory itself to a
# candidate-specific host path.
CONTAINER_TRANSCRIPT_PARENT = f"{CONTAINER_HOME}/.claude/projects"


def transcript_host_dir(workspace: Path) -> Path:
    """Return the candidate-specific host directory for agent transcripts.

    Keyed off the candidate's workspace, which is created fresh per candidate
    per run, so distinct candidates get distinct transcript roots **by
    construction** -- including concurrently.

    This matters because the property is currently NOT guaranteed: the
    existing copy-out builds its source path from a session id extracted from
    backend stdout, so the candidate identity never enters the path at all.
    Two candidates that report the same session id write to, and read from,
    one file.  Deriving the location from the workspace makes the guarantee a
    property of HELIX rather than of backend behaviour.

    Placed as a SIBLING of the workspace, never inside it: anything under the
    workspace is subject to sync-back into the candidate's repository, which
    would turn agent transcripts into part of the candidate's diff.
    """
    workspace = Path(workspace)
    return workspace.parent / f"{workspace.name}.helix-transcripts"


def transcript_bind_arg(host_dir: Path) -> list[str]:
    """Docker args binding the candidate transcript directory into the agent.

    The caller MUST create ``host_dir`` first.  Docker auto-creates a missing
    bind source as ``root:root``, which reproduces exactly the unwritable-HOME
    failure this module exists to prevent -- and, worse, would do so silently
    for the transcript path only.
    """
    return ["-v", f"{host_dir}:{CONTAINER_TRANSCRIPT_PARENT}:rw"]


def ensure_transcript_host_dir(host_dir: Path) -> Path:
    """Create the transcript directory so Docker never auto-creates it as root."""
    host_dir = Path(host_dir)
    host_dir.mkdir(parents=True, exist_ok=True)
    return host_dir
