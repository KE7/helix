# Per-candidate agent state, and why claude is not isolated

HELIX mounts one login volume per backend (`helix-auth-<backend>`) at
`/home/node`, read-write, in every candidate container. That mount is shared on
purpose and is not negotiable: it is what lets each CLI refresh its token and
take its cross-process refresh lock across concurrent candidates.

The problem this document is about is everything *else* the CLIs write into
that volume. `helix.agent_state` relocates what it safely can to a
per-candidate directory mounted at `/helix-state`; the README's "Per-candidate
agent state" section covers the three backends that worked. This note records
the reasoning for the one that did not, so it does not get re-litigated from
scratch.

All observations below are from the images HELIX ships
(`ghcr.io/ke7/helix-evo-runner-*:latest`), against synthetic credentials in
throwaway volumes.

## Why `CLAUDE_CONFIG_DIR` is not usable

It is all-or-nothing. It relocates `.credentials.json` together with the
transcripts, and it additionally pulls `.claude.json` into whatever it points
at. Pointing it at a per-candidate directory would give each candidate a clean
state tree and no credential, which defeats the entire purpose of the shared
login volume. There is no second knob: of the `CLAUDE_*` variables the CLI
understands, none relocates state alone.

## Why masking was evaluated and rejected

The alternative is *masking*: leave the config directory shared and mount empty
per-candidate volumes over its state subdirectories. This was tried against
Claude Code 2.1.138. It does relocate state — `projects/`, `sessions/`,
`telemetry/`, `backups/` and the contents of `.claude.json` all landed in the
per-candidate directory. It was still rejected, for four reasons.

**1. The mask list has to be maintained against the CLI, and is already
stale.** The subdirectories a reasonable person would name — `projects/`,
`sessions/`, `todos/` — are not the ones this version writes. There is no
`todos/` at all, and there are two that the obvious list misses: `telemetry/`
and `backups/`, both of which carry per-session identifiers. A mask list that
is already wrong for the currently shipped CLI is the clearest possible
evidence that it will drift again, and each drift is silent: a newly added
state directory simply starts leaking between candidates with nothing to
signal it.

**2. `.claude.json` is a file outside the config directory.** It lives at
`$HOME/.claude.json`, not under `.claude/`, so masking it needs a *file*-level
bind mount rather than a directory one. The CLI also rewrites it through a
backup-and-replace cycle — a `.claude/backups/.claude.json.backup.<timestamp>`
appears on every run. File bind mounts do not survive an atomic
rename-into-place, so this is a mechanism that works until the day the CLI
changes how it saves that file, and then fails in a way that is hard to
attribute.

**3. Masking writes to the shared volume, which the isolation work is not
allowed to do.** A bind mount needs its mountpoint to exist, and Docker creates
it inside the volume. Masking the four directories plus `.claude.json` added
five new entries to the shared login volume, including turning `.claude.json`
into a 0-byte file there. The knob-based approach used for codex, cursor and
opencode leaves the shared volume byte-for-byte unchanged; masking cannot.

**4. It silently breaks transcript preservation.**
`helix.sandbox._copy_claude_transcript_from_auth_volume` recovers the session
transcript by starting a *separate* container that mounts only the auth volume
read-only and copies from `sandbox.claude_transcript_root`. That container does
not carry the agent container's masks, so once `projects/` is masked the
transcript it is looking for is no longer in the volume. The helper's
`[ -f "$src" ] || exit 0` guard means this fails silently:
`preserve_backend_transcripts` would keep reporting success while saving
nothing.

## The verification gap

Independently of the above, requirement (c) of this work — *demonstrate the CLI
still reports itself authenticated* — cannot be met for claude without a real
grant. Claude Code validates the credential's shape before reporting status, so
a synthetic credential yields `Not logged in · Please run /login`. That is not
caused by masking (an unmasked container with the same synthetic credential
reports exactly the same thing), but it does mean the only way to prove a
claude change is safe is to run it against a live login. Proving isolation by
risking the credential it is supposed to protect is a bad trade.

## Conclusion

Claude is left exactly as it is. Its cross-candidate residue is recorded in
`helix.agent_state.UNRELOCATED_AGENT_STATE` under the `claude` key so that it
is discoverable rather than forgotten. Three of four backends are isolated;
this one is documented instead.

Anyone revisiting this should start by re-running the footprint check against
the current CLI, because the specific directories named above are version
facts, not stable API.
