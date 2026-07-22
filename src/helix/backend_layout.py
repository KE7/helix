"""Per-backend HOME layout registry for sandboxed mutation agents.

WHAT THIS DOES AND DOES NOT CLAIM
=================================

This registry is an **incidental-state control**.  It closes cross-run state
that the backend CLI writes on its own -- transcripts, sessions, caches, logs,
history, agent memory databases -- which is exactly the contamination that
affected completed benchmark runs.

*** IT IS NOT A CANDIDATE-INDEPENDENCE GUARANTEE, AND MUST NEVER BE DESCRIBED
AS ONE. ***

The reason is structural and was proven by canary through the completed
design.  The auth directory must be mounted **writable**, because OAuth
rotation rewrites the credential by atomic rename inside the credential's own
directory (a per-file bind cannot be renamed over or unlinked -- it returns
EBUSY).  The agent runs as ``node`` and can therefore create *any* file in that
directory.  A denylist masks only the paths enumerated here; a path invented at
runtime is, by construction, not one of them:

    candidate A:  echo … > ~/.codex/notes-for-next-candidate.txt
    candidate B:  cat  …  -> read verbatim, through the full layout

Adding entries to this registry cannot fix that.  **The registry closes paths
the CLI writes; it cannot close paths an agent writes.**

Only ``sandbox.auth = "env"`` can support an independence claim, because it
mounts no persistent store at all -- there is no directory in which to invent a
file.  See :mod:`helix.sandbox_home` and
``docs/design/sandbox-home-isolation.md`` §8b.

THE FOUR CLASSES
================

Every path in a backend's HOME falls into exactly one class:

1. ``auth_dir``          -- shared, writable, mounted via ``volume-subpath``.
2. ``ephemeral_subdirs`` -- directories INSIDE ``auth_dir`` carrying per-run
   state; isolated by per-run tmpfs overlays.
3. ``ephemeral_files``   -- regular files INSIDE ``auth_dir`` carrying per-run
   state.  These CANNOT be isolated by any mount -- an overlay works on
   directories only, and a per-file bind is EBUSY on rename.  They must be
   relocated by a backend-native env knob, or the backend fails closed.
4. everything outside ``auth_dir`` -- private automatically, because HOME
   itself is a per-run tmpfs.

Class 3 is the class the original design missed.  It is not hypothetical:
codex keeps its agent ``memories`` and ``goals`` SQLite databases as regular
files beside ``auth.json``, and cursor has no subdirectories at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from helix.backends import BackendName


class UnsupportedBackendLayoutError(RuntimeError):
    """A backend's per-run state cannot be isolated on this runtime.

    Raised instead of silently running with a known cross-candidate channel.
    Fail-closed is deliberate: a per-file "accepted residual" would let a
    candidate read the previous candidate's state while the run reported
    success, which is a false certification rather than a documented gap.
    """


@dataclass(frozen=True)
class BackendHomeLayout:
    """How one backend's HOME is partitioned across the four classes."""

    backend: BackendName

    #: Container path of the directory holding the credential (class 1).
    auth_dir: str

    #: Path of that directory *within the auth volume*, for ``volume-subpath``.
    volume_subpath: str

    #: The credential file, relative to ``auth_dir``.  Documentation and
    #: preflight only -- never mounted individually (see EBUSY above).
    credential_file: str

    #: Class 2 -- directories inside ``auth_dir`` that carry per-run state.
    ephemeral_subdirs: tuple[str, ...] = ()

    #: Class 3 -- regular files inside ``auth_dir`` that carry per-run state,
    #: mapped to the env var that relocates them.  A file appearing here with
    #: no knob is a fail-closed condition.
    ephemeral_files: dict[str, str] = field(default_factory=dict)

    #: Env knobs set to per-run private paths, applied together with the
    #: mounts.  Values are container paths under the private HOME.
    env_redirects: dict[str, str] = field(default_factory=dict)

    #: Files inside ``auth_dir`` that are stable identity/configuration rather
    #: than per-run state.  Listed EXPLICITLY so that "not isolated" is a
    #: recorded decision rather than an oversight, and so the detection
    #: backstop can distinguish them from an agent-invented file.
    stable_files: tuple[str, ...] = ()

    #: Pinned CLI version this layout was measured against.
    pinned_cli_version: str = ""

    #: EXPLICIT statement that this backend is unsupported under volume mode,
    #: with the reason.  ``None`` means supported.
    #:
    #: This exists because emptiness is ambiguous.  Refusal used to be INFERRED
    #: from "a class-3 file whose knob is missing from ``env_redirects``" --
    #: which cannot distinguish "declares nothing unrelocatable" from "declares
    #: nothing at all".  An empty ``ephemeral_files`` therefore read as
    #: SUPPORTED, and that is exactly how opencode was certified safe while its
    #: session database sat beside the credential.
    #:
    #: A check keyed on an explicit field cannot be defeated by an omission.
    unsupported_reason: str | None = None

    #: Deliberate assertion that this backend's auth dir was MEASURED and
    #: genuinely holds no per-run state.  Required when every ephemeral class
    #: is empty.
    #:
    #: This closes the empty-set trap at the level of the CHECK'S SHAPE rather
    #: than of one entry's data.  Correcting opencode fixed the instance; a
    #: SIXTH backend added with empty sets would still have been certified
    #: isolatable exactly as opencode was.  Emptiness must now be something a
    #: human wrote on purpose, not something that happens when nobody looked.
    measured_empty: bool = False

    #: Deliberate assertion that ``ephemeral_files`` was MEASURED and is
    #: genuinely empty.
    #:
    #: PER-CLASS, and that is the whole point. The previous marker was
    #: whole-layout, so a NON-EMPTY ``ephemeral_subdirs`` made an EMPTY,
    #: UNMEASURED ``ephemeral_files`` look fine -- which is opencode's ACTUAL
    #: bug shape, not the degenerate all-empty case. A backend can have
    #: directories worth isolating and still have unmeasured sibling FILES
    #: beside its credential; that is precisely what happened.
    measured_empty_files: bool = False


# Private per-run scratch root, inside the tmpfs HOME.  Anything redirected
# here dies with the container.
HELIX_RUN_ROOT = "/home/node/.helix-run"


BACKEND_LAYOUTS: dict[BackendName, BackendHomeLayout] = {
    # ---------------------------------------------------------------
    # claude -- helix-auth-claude; Claude Code 2.1.138 (the PINNED DIGEST)
    # ---------------------------------------------------------------
    "claude": BackendHomeLayout(
        backend="claude",
        auth_dir="/home/node/.claude",
        volume_subpath=".claude",
        credential_file=".credentials.json",
        ephemeral_subdirs=(
            "projects",  # agent transcripts -- the primary channel
            "sessions",
            "backups",
            "shell-snapshots",
            "session-env",
            "cache",
        ),
        # ``~/.claude.json`` is NOT here: it lives at the HOME root, outside
        # auth_dir, so it is class 4 and private automatically.  Verified that
        # it is auto-created (231 B) and needs no seeding.
        #
        # FAILS CLOSED under volume mode.  An independent classification of
        # these three files on the pinned runtime found ALL THREE CARRYING --
        # no unknowns -- so each is per-run state with no relocation knob:
        #
        #   .last-cleanup             writer ``startBackgroundHousekeeping``
        #                             stamps an ISO timestamp; reader
        #                             ``isLastCleanupSentinelFresh`` returns
        #                             early when mtime is under 24h, so a
        #                             prior candidate makes the NEXT candidate
        #                             SKIP its entire cleanup pass.
        #   mcp-needs-auth-cache.json keys are MCP server names; a cache hit
        #                             makes the connect loop skip connection
        #                             entirely (TTL 15min http/sse, 4h proxy).
        #   policy-limits.json        the sha256 of the ON-DISK file is sent as
        #                             the conditional-request token, so a prior
        #                             candidate's file changes the NEXT
        #                             candidate's outbound request; on fetch
        #                             failure or timeout the effective policy
        #                             for this candidate IS the previous
        #                             candidate's file.
        #
        # ``--strict-mcp-config`` (see mutator) closes the MCP channel only.
        # It is NOT a volume-mode rescue: the other two are written and read by
        # the CLI itself inside the shared HOME, with no flag that stops them.
        ephemeral_files={
            ".last-cleanup": "",
            "mcp-needs-auth-cache.json": "",
            "policy-limits.json": "",
        },
        env_redirects={},
        unsupported_reason=(
            "the CLI itself keeps per-run state in the shared auth directory "
            "with no relocation knob: .last-cleanup (a prior candidate makes "
            "the next one skip its entire cleanup pass), "
            "mcp-needs-auth-cache.json (a cache hit skips connection), and "
            "policy-limits.json (its on-disk sha256 shapes the next "
            "candidate's outbound request). All three classified CARRYING on "
            "the pinned digest, no unknowns. --strict-mcp-config closes only "
            "the MCP channel and is not a rescue."
        ),
        # NOTE: pinned to the DIGEST the demos actually run
        # (@sha256:6be6fef…), which is 2.1.138 -- NOT the ``:latest`` tag,
        # which is 2.1.120 and is only the login/status path.  This distinction
        # is load-bearing: ``.last-cleanup`` is ABSENT from the 2.1.120 bundle
        # and PRESENT in 2.1.138, so a layout measured against ``:latest``
        # misses a live production channel.  Measure against the digest.
        pinned_cli_version="2.1.138",
    ),
    # ---------------------------------------------------------------
    # codex -- measured against helix-auth-codex and codex-cli 0.125.0
    # ---------------------------------------------------------------
    "codex": BackendHomeLayout(
        backend="codex",
        auth_dir="/home/node/.codex",
        volume_subpath=".codex",
        credential_file="auth.json",
        ephemeral_subdirs=(
            "sessions",
            "log",
            "shell_snapshots",
            "memories",
            "tmp",
            ".tmp",
            "cache",
            "skills",
        ),
        # The SQLite family sits as REGULAR FILES beside auth.json -- state,
        # logs, agent memories and goals.  No overlay can mask a sibling file,
        # so they are relocated wholesale by CODEX_SQLITE_HOME.
        #
        # PROVEN behaviourally against a positive control: without the knob, 4
        # sqlite files appear in ~/.codex; with it, 0 there and 6 in the
        # redirect directory.
        # The FULL SQLite family, from ``ls`` of the real volume. The -shm and
        # -wal siblings were missing from an earlier version of this entry,
        # which made the module's "every path falls into exactly one class"
        # claim FALSE against the artifact and would have made the drift
        # detector false-positive on six real files the moment it was wired.
        #
        # The evidence was already in hand and misread: the CODEX_SQLITE_HOME
        # control reported "0 in ~/.codex and SIX in the redirect dir" -- six,
        # not the four then declared. The count was already saying the family
        # was larger.
        #
        # No leak: SQLite creates -shm/-wal beside the main database, so the
        # knob relocates them together, which is what the control observed.
        ephemeral_files={
            "state_5.sqlite": "CODEX_SQLITE_HOME",
            "state_5.sqlite-shm": "CODEX_SQLITE_HOME",
            "state_5.sqlite-wal": "CODEX_SQLITE_HOME",
            "logs_2.sqlite": "CODEX_SQLITE_HOME",
            "logs_2.sqlite-shm": "CODEX_SQLITE_HOME",
            "logs_2.sqlite-wal": "CODEX_SQLITE_HOME",
            "memories_1.sqlite": "CODEX_SQLITE_HOME",
            "memories_1.sqlite-shm": "CODEX_SQLITE_HOME",
            "memories_1.sqlite-wal": "CODEX_SQLITE_HOME",
            "goals_1.sqlite": "CODEX_SQLITE_HOME",
            "goals_1.sqlite-shm": "CODEX_SQLITE_HOME",
            "goals_1.sqlite-wal": "CODEX_SQLITE_HOME",
        },
        env_redirects={
            "CODEX_SQLITE_HOME": f"{HELIX_RUN_ROOT}/codex-sqlite",
        },
        stable_files=("config.toml", "installation_id", "models_cache.json"),
        pinned_cli_version="0.125.0",
    ),
    # ---------------------------------------------------------------
    # cursor -- measured against helix-auth-cursor / 2026.04.17-787b533
    # ---------------------------------------------------------------
    # NOTE: cursor has NO subdirectories in its auth dir at all, so a
    # subdirectory-overlay design does literally nothing for it.  Its per-run
    # state is entirely class 3.
    "cursor": BackendHomeLayout(
        backend="cursor",
        auth_dir="/home/node/.cursor",
        volume_subpath=".cursor",
        credential_file="cli-config.json",
        # FAILS CLOSED under volume mode: the split is PLAUSIBLE but UNPROVEN.
        #
        # Source shows two independent knobs, which is promising -- the
        # credential is read from the CONFIG dir
        # (``join(configDir(), "cli-config.json")``, where
        # ``CURSOR_CONFIG_DIR || XDG_CONFIG_HOME/cursor || ~/.cursor``) while
        # ``CURSOR_DATA_DIR || ~/.cursor`` governs the DATA dir. Because both
        # default to ``~/.cursor`` the files are commingled today, so pointing
        # DATA elsewhere might genuinely split them.
        #
        # "Might" is not good enough to run a benchmark on. It has NOT been
        # verified which files follow which knob, and an earlier version of
        # this entry declared cursor isolatable on exactly that unverified
        # assumption -- the same declared-but-unproven error this registry
        # exists to prevent. Isolation stays refused until a both-halves
        # behavioural proof exists (ephemeral state moves; credential stays
        # readable and rotatable in place).
        ephemeral_files={
            "agent-cli-state.json": "CURSOR_DATA_DIR",
            "statsig-cache.json": "CURSOR_DATA_DIR",
        },
        env_redirects={},
        unsupported_reason=(
            "a CONFIG/DATA split is plausible (the credential is read from "
            "CURSOR_CONFIG_DIR while CURSOR_DATA_DIR governs the data dir) "
            "but it is UNVERIFIED which files follow which knob, and both "
            "default to ~/.cursor. Plausible is not proven; refused pending a "
            "both-halves behavioural proof"
        ),
        pinned_cli_version="2026.04.17-787b533",
    ),
    # ---------------------------------------------------------------
    # gemini -- measured against helix-auth-gemini / gemini-cli 0.39.1
    # ---------------------------------------------------------------
    "gemini": BackendHomeLayout(
        backend="gemini",
        auth_dir="/home/node/.gemini",
        volume_subpath=".gemini",
        # NOT PRESENT in the real helix-auth-gemini volume, which holds no
        # OAuth credential at all (that volume was authenticated by API key).
        # The path is taken from the CLI bundle, not from an artifact, so it
        # is an UNVERIFIED declaration. Inert today -- gemini refuses volume
        # mode -- but it must not be read as measured.
        credential_file="oauth_creds.json",
        ephemeral_subdirs=("tmp", "history"),
        # FAILS CLOSED under volume mode, deliberately.
        #
        # ``state.json`` is a regular file beside the credential, so no overlay
        # can mask it.  The only relocation knob gemini exposes is
        # GEMINI_CLI_HOME, which moves the ENTIRE gemini home -- including
        # ``oauth_creds.json`` -- and therefore cannot express a class-3 split.
        # It is the same shape as CLAUDE_CONFIG_DIR: it proves RELOCATION, not
        # a split.
        #
        # Listing the file with a knob that is absent from ``env_redirects``
        # is what makes ``assert_layout_is_isolatable`` refuse this backend.
        # That is the intended outcome: HELIX declines to run gemini under
        # volume mode rather than report an isolated run that is not isolated.
        # ``auth = "env"`` runs gemini with no shared store at all.
        #
        # Do NOT "fix" this by moving state.json into ``stable_files``; that
        # would assert it carries no per-run state, which has not been
        # measured.
        ephemeral_files={"state.json": "GEMINI_CLI_HOME"},
        env_redirects={},
        unsupported_reason=(
            "state.json is a regular file beside the credential, and the only "
            "knob (GEMINI_CLI_HOME) relocates the whole home INCLUDING "
            "oauth_creds.json -- relocation, not a class-3 split"
        ),
        stable_files=("installation_id", "projects.json"),
        pinned_cli_version="0.39.1",
    ),
    # ---------------------------------------------------------------
    # opencode -- measured against helix-auth-opencode / 1.14.24
    # ---------------------------------------------------------------
    # Credential lives under XDG data, not a dotdir in HOME.  The existing
    # XDG_DATA_HOME workaround in the mutator relocated only the session DB;
    # the config and cache roots are redirected here too.
    "opencode": BackendHomeLayout(
        backend="opencode",
        auth_dir="/home/node/.local/share/opencode",
        volume_subpath=".local/share/opencode",
        credential_file="auth.json",
        ephemeral_subdirs=("log", "snapshot", "storage"),
        # FAILS CLOSED under volume mode.
        #
        # Measured contents of the real auth dir: ``opencode.db``,
        # ``opencode.db-shm`` and ``opencode.db-wal`` sit as REGULAR FILES
        # beside the credential -- that is the session database, the very state
        # ``mutator.py``'s XDG_DATA_HOME workaround was introduced to isolate.
        # No overlay can mask a sibling file.
        #
        # The only knob that relocates them is XDG_DATA_HOME, which moves
        # ``~/.local/share/opencode`` WHOLESALE -- i.e. the credential too. That
        # is the same shape as CLAUDE_CONFIG_DIR and GEMINI_CLI_HOME: it proves
        # RELOCATION, not a class-3 SPLIT.
        #
        # An earlier version of this entry declared opencode isolatable with an
        # EMPTY ephemeral set, which made it vacuously pass the fail-closed
        # check while its session DB crossed runs. The measured contents are
        # now recorded so the refusal is grounded in evidence.
        ephemeral_files={
            "opencode.db": "XDG_DATA_HOME",
            "opencode.db-shm": "XDG_DATA_HOME",
            "opencode.db-wal": "XDG_DATA_HOME",
        },
        env_redirects={},
        unsupported_reason=(
            "opencode.db{,-shm,-wal} -- the session database -- are regular "
            "files beside the credential, and the only knob that relocates "
            "them (XDG_DATA_HOME) moves ~/.local/share/opencode wholesale "
            "including the credential -- relocation, not a class-3 split"
        ),
        pinned_cli_version="1.14.24",
    ),
}


def layout_for(backend: str) -> BackendHomeLayout:
    """Return the layout for ``backend``, or fail closed."""
    try:
        return BACKEND_LAYOUTS[backend]  # type: ignore[index]
    except KeyError:
        raise UnsupportedBackendLayoutError(
            f"no HOME-isolation layout is registered for backend {backend!r}.\n"
            f'  Running it under sandbox.auth = "volume" would mount a '
            f"persistent store over state HELIX has never measured, so "
            f"per-run isolation cannot be asserted.\n"
            f'  Remedy: use sandbox.auth = "env", or add a measured layout '
            f"to helix/backend_layout.py."
        ) from None


def unisolatable_files(layout: BackendHomeLayout) -> tuple[str, ...]:
    """Class-3 files with no relocation knob -- the fail-closed condition.

    A file listed in ``ephemeral_files`` whose knob is absent from
    ``env_redirects`` would be per-run state left sitting in the shared
    directory with no mechanism to move it.
    """
    return tuple(
        name
        for name, knob in sorted(layout.ephemeral_files.items())
        if knob not in layout.env_redirects
    )


def assert_layout_is_declared_completely(layout: BackendHomeLayout) -> None:
    """Reject a layout whose emptiness could be an omission rather than a fact.

    A layout that classifies NOTHING passes every derived check trivially. That
    is indistinguishable from "measured, and there is genuinely nothing" unless
    somebody says which they mean -- so this requires either a non-empty
    ephemeral class or an explicit ``measured_empty``.
    """
    # Per-class: a non-empty subdir set must NOT excuse an unmeasured file set.
    if not layout.ephemeral_files and not (
        layout.measured_empty_files or layout.measured_empty
    ):
        raise UnsupportedBackendLayoutError(
            f"layout for {layout.backend!r} declares NO ephemeral FILES.\n"
            f"  An empty file set passes every derived check trivially, and is "
            f"indistinguishable from one that was never measured -- which is "
            f"exactly how opencode was certified isolated while its session "
            f"database sat beside the credential. A non-empty subdir set does "
            f"not make an unmeasured file set safe.\n"
            f"  If the auth directory was measured and genuinely holds no "
            f"per-run sibling FILES, set measured_empty_files=True."
        )
    if layout.ephemeral_subdirs or layout.ephemeral_files or layout.measured_empty:
        return
    raise UnsupportedBackendLayoutError(
        f"layout for {layout.backend!r} classifies NO per-run state at all.\n"
        f"  An empty layout passes every isolation check trivially, so it is "
        f"indistinguishable from one that was never measured -- which is "
        f"exactly how a backend gets certified isolated while its session "
        f"state crosses runs.\n"
        f"  If the auth directory was measured and genuinely holds no per-run "
        f"state, set measured_empty=True deliberately. Otherwise measure it."
    )


def assert_layout_is_isolatable(layout: BackendHomeLayout) -> None:
    """Fail closed if this backend is unsupported, or if anything is unrelocatable.

    Two independent gates, deliberately:

    1. an EXPLICIT ``unsupported_reason`` -- which an omission cannot produce;
    2. the derived check, which catches a NEW class-3 file added without a knob.

    Gate 1 exists because gate 2 alone is defeated by emptiness: a layout that
    declares nothing passes it trivially.
    """
    assert_layout_is_declared_completely(layout)
    if layout.unsupported_reason is not None:
        raise UnsupportedBackendLayoutError(
            f"backend {layout.backend!r} (CLI {layout.pinned_cli_version}) is "
            f'NOT supported under sandbox.auth = "volume".\n'
            f"  {layout.unsupported_reason}\n"
            f"  HELIX refuses to run it there rather than report an isolated "
            f"run that is not isolated.\n"
            f'  Remedy: use sandbox.auth = "env", which mounts no '
            f"persistent store at all."
        )
    orphans = unisolatable_files(layout)
    if not orphans:
        return
    raise UnsupportedBackendLayoutError(
        f"backend {layout.backend!r} (CLI {layout.pinned_cli_version}) writes "
        f"per-run state to files inside its shared auth directory that HELIX "
        f"cannot relocate: {', '.join(orphans)}.\n"
        f"  These sit beside the credential in {layout.auth_dir}, which must "
        f"stay writable for OAuth rotation, and a per-file mount cannot be "
        f"renamed over (EBUSY). A later candidate would read the previous "
        f"candidate's state.\n"
        f'  HELIX refuses to run this backend under sandbox.auth = "volume" '
        f"rather than report an isolated run that is not isolated.\n"
        f'  Remedy: use sandbox.auth = "env" (mounts no persistent store), '
        f"or supply a relocation knob in helix/backend_layout.py."
    )
