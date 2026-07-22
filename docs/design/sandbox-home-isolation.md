# Sandbox HOME isolation — approved architecture (HELIX 0.3.0 release blocker)

- **Branch:** `fix/sandbox-home-isolation`, based on `9f2bcaa` (the credential fix).
- **Defect:** proven in `docs/audits/2026-07-22-auth-volume-cross-run-state.md`
  (branch `audit/auth-volume-state`, commit `b07db88`). Not re-litigated here.
- **Contains no credential values.** Real `helix-auth-*` volumes were read
  `:ro`, metadata and hashes only; no session, history or credential *contents*
  were read. Every behavioural test below ran on disposable volumes under this
  branch's own `helixiso-*` prefix.

This document records the architecture that supersedes design **D4** from the
audit memo, the evidence that forced the change, and the proof obligations that
remain before the mount rewrite can land.

---

## 1. Why D4 as written is not sufficient

D4 is: mount the persistent auth volume at the backend's auth directory via
`volume-subpath`, then `--tmpfs`-overlay the leaky **subdirectories** inside it.

That silently assumes every piece of per-run state lives in a *subdirectory* of
the auth directory. Metadata reads of the five real volumes show it does not.

### Behavioural proof — overlays isolate directories, not sibling files

Disposable volume `helixiso-codexshape`, codex runner image, exact D4 layout,
volume removed afterwards:

| Object written by run A | Seen by later run B | Meaning |
|---|---|---|
| `~/.codex/sessions/a.txt` | 0 files | **isolated** — the overlay works |
| `~/.codex/state_5.sqlite` | `RUN-A-CANARY` | **LEAK** — sibling file crosses runs |

A regular file sitting beside `auth.json` cannot be masked by a directory
overlay, and a per-file bind is design **D2**, already rejected: `rename`-over
and `unlink` both return `EBUSY`, which breaks OAuth rotation.

### Why this is release-critical, not a nitpick

`helix-auth-codex` holds these as regular files **directly beside `auth.json`**:

```
memories_1.sqlite (+ -wal/-shm)   goals_1.sqlite (+ -wal/-shm)
state_5.sqlite (188 KB)           logs_2.sqlite
models_cache.json                 installation_id
```

`memories` and `goals` are cross-run **agent memory**. `helix-auth-cursor` is
the same shape and has *no subdirectories at all* — `agent-cli-state.json`,
`cli-config.json`, `statsig-cache.json` are all siblings, so D4 does literally
nothing for cursor.

Shipping D4 verbatim would close claude's transcript channel and leave codex's
memory channel open, while all three completed demos were re-run and certified
clean. **A false certification is worse than a known gap.**

---

## 2. Approved architecture — four classes + backend-native env redirection

Per-backend registry. Every path in a backend's HOME falls into exactly one
class:

| Class | Contents | Mechanism |
|---|---|---|
| **1 — AUTH_DIR** | the directory holding the credential | shared: `--mount type=volume,volume-subpath=<subpath>,dst=<auth_dir>`, **writable** — preserves atomic rename/unlink, refresh rotation, and cross-container `flock` |
| **2 — ephemeral subdirs** *inside* AUTH_DIR | sessions, logs, transcripts, caches | per-run `--tmpfs` overlay (or per-candidate **host bind** for transcripts) |
| **3 — ephemeral sibling files** *inside* AUTH_DIR | DBs, state/config/cache files | **not isolatable by any mount** — must be relocated by the backend's own env knob |
| **4 — everything outside AUTH_DIR** | `~/.npm`, `~/.cache`, `~/.claude.json`, … | private per-run `--tmpfs /home/node:uid=1000,gid=1000` |

### Fail-closed rule (EA-approved, supersedes "accepted residual")

A class-3 file with **no proven redirection knob** is **not** an acceptable
declared residual. If a supported backend cannot be made isolated, HELIX
**refuses to run that backend** with an actionable "unsupported layout/runtime"
error naming what could not be isolated and why.

There is **no whole-HOME fallback**, under any circumstance. A per-file
"accepted residual" would reproduce the D4 failure at finer granularity: a
candidate could still read the previous candidate's memory DB while the release
claimed independence.

This applies to `claude` exactly as to the others — the backend we care most
about is not special-cased.

---

## 3. Mount mechanics — established facts

These are settled; do not re-derive.

- **`volume-subpath` requires Docker Engine 25.0+ / API 1.45+** (verified on
  29.6.1 / API 1.55). Any host below Engine 25 **cannot run this design at
  all.** This is a hard requirement bump; it must be preflight-checked with a
  clear message and documented for operators, never surfaced as a mystery
  failure.
- **A naive per-run tmpfs HOME is unwritable.** `--user node --tmpfs /home/node`
  yields `root:root 0755`; uid 1000 gets `Permission denied` and *every*
  mutation agent fails. Use `--tmpfs /home/node:rw,uid=1000,gid=1000,mode=0755`.
- **`volume-subpath` must already exist**, or the container never starts
  ("cannot access path …: no such file or directory"). `helix sandbox login`
  must **create the subpath** before any agent run; the login path itself
  cannot use the subpath mount. Creating the *subpath* on login is legitimate;
  creating the *volume* on `status` remains prohibited.
- **Nested mount ordering is not argv-order-dependent** — Docker orders mounts
  by destination depth.
- **Subpath symlink escape is blocked by the daemon** ("path concatenation
  escapes the base directory"). A genuine containment property.

### Pinned runtimes and uid (guard targets)

Measured from the images themselves, independently by two parties:

| Backend | CLI version | `node` uid:gid | default USER |
|---|---|---|---|
| claude | 2.1.120 | 1000:1000 | root |
| codex | codex-cli 0.125.0 | 1000:1000 | root |
| cursor | 2026.04.17-787b533 | 1000:1000 | root |
| gemini | gemini-cli 0.39.1 | 1000:1000 | root |
| opencode | 1.14.24 | 1000:1000 | root |

uid 1000 is uniform *today*, so the constant is used — but it is **enforced by a
pinned-runtime guard**, not assumed. A base-image bump that changes it must
fail loudly rather than silently produce an unwritable HOME.

---

## 4. Five-backend layout (source-grounded)

Credential locations and knobs, from the pinned bundles/binaries. Class-3
columns are the ones D4 missed.

| Backend | AUTH_DIR (class 1) | Credential | Class-2 subdirs | Class-3 siblings | Redirection knob (from source) |
|---|---|---|---|---|---|
| claude | `~/.claude` | `.credentials.json` | `projects`, `sessions`, `backups`, `shell-snapshots`, `session-env`, `cache` | `.last-cleanup`, `mcp-needs-auth-cache.json`, `policy-limits.json` | `CLAUDE_CONFIG_DIR` present (31 refs in 2.1.120) — **but it moves the whole config dir including credentials, so it is not usable for a class-3 split** |
| codex | `~/.codex` | `auth.json` | `sessions`, `log`, `shell_snapshots`, `memories`, `tmp`, `.tmp`, `cache`, `skills` | `*.sqlite{,-wal,-shm}`, `models_cache.json`, `installation_id`, `config.toml` | `CODEX_HOME`, `CODEX_SQLITE_HOME`, `CODEX_ROLLOUT_TRACE_ROOT`, `CODEX_JS_TMP_DIR` |
| cursor | `~/.cursor` | `cli-config.json` | *(none today)* | `agent-cli-state.json`, `statsig-cache.json` | `CURSOR_CONFIG_DIR`, `CURSOR_DATA_DIR` |
| gemini | `~/.gemini` | `oauth_creds.json`, `google_accounts.json` | `tmp`, `history` | `state.json`, `projects.json`, `installation_id` | `GEMINI_CLI_HOME` |
| opencode | `~/.local/share/opencode` | `auth.json` | — | — | `XDG_DATA_HOME` (already used), `XDG_CONFIG_HOME`, `XDG_CACHE_HOME` |

Two notes carried forward:

- `~/.claude.json` lives at the **HOME root**, outside AUTH_DIR, so it is class 4
  and becomes private automatically. See §5.
- gemini also uses `~/.config/google-gemini` (see `BACKEND_AUTH_COMMANDS`), which
  is outside `~/.gemini` and must be classified before gemini can be certified.

---

## 5. `~/.claude.json` on the exact pinned runtime — RESOLVED

The audit could not establish what Claude Code requires from `~/.claude.json`
once it becomes per-run. **This is now resolved on 2.1.120, without ever
touching the real token**, using a fresh tmpfs HOME and a synthetic credential.

Findings:

1. **It is auto-created on demand** when absent (231 bytes). No seeding needed.
2. Its contents are only:
   `firstStartTime`, `migrationVersion`, `opusProMigrationComplete`, `userID`.
3. The onboarding/trust keys that motivated the concern —
   `hasCompletedOnboarding`, `hasTrustDialogAccepted`,
   `bypassPermissionsModeAccepted`, `hasCompletedProjectOnboarding`,
   `projectOnboardingSeenCount` — exist in the binary but are **not written to
   the fresh file and are not required** to reach the inference path.
4. **No interactive prompt gates `-p` mode.** With no credential, `claude -p`
   exits immediately with `Not logged in · Please run /login` rather than
   hanging on an onboarding dialog. With a credential present it proceeds past
   config straight to the network path.

**Conclusion: making `~/.claude.json` per-run is safe and requires no seeding.**

Two declared consequences, neither blocking:

- `userID` is regenerated per run, so per-run telemetry identity changes. For a
  benchmark asserting candidate independence this is arguably correct.
- The real volume's `.claude.json` is 29 KB of accumulated project/trust/history
  state. Per-run isolation discards it. The fresh file works, so this is
  acceptable — and it is the point of the fix.

**Residual limitation, stated rather than waived:** a *successful authenticated*
run with a per-run `.claude.json` has not been observed, because that requires
the real token. The gating question — "does a fresh `.claude.json` cause an
interactive hang in a non-interactive container?" — is answered **no** on the
exact pinned runtime.

---

## 6. Root-owned files — mask, never clean

`helix-auth-claude` contains **four uid-0 entries**; the other four volumes have
none.

```
.claude/backups/.claude.json.backup.1784706605944   (07:50)
.claude/backups/.claude.json.backup.1784706667263   (07:51)
.claude/projects/-workspace/dc84f1e2-….jsonl        (07:50)
.claude/projects/-workspace/834a242e-….jsonl        (07:51)
```

### The mechanics, stated precisely so this guard is not "corrected" away

A `--user node` container **can** unlink and rename-over these files: POSIX
`unlink` requires write+execute on the **parent directory**, not ownership of
the file, and both parents are `1000:1000 drwxr-xr-x` with no sticky bit
(proven on a synthetic replica: read → DENIED, `cp` → DENIED, unlink →
DELETE_OK).

So an `rm -rf` over this tree would **succeed and destroy the evidence**. Any
comment claiming "node cannot delete these" is false, and a reader who tests
the premise would find it false and remove the safeguard.

- **Deletion is prohibited by POLICY** — these are incident evidence — not by
  permissions.
- **The operative technical hazard is READ**: the files are `0600 root`, so
  `--user node` cannot read them (verified by exit code only; a node-owned
  transcript in the same directory reads fine as a control).
- Class-2 isolation **masks** them, so no candidate inherits them.
- New transcript binds must be **node-owned**, so the hazard does not recur.

### Live bug at 9f2bcaa, independent of this fix

`_copy_claude_transcript_from_auth_volume` (`sandbox.py:207`) runs `--user node`
and does `cp "$src" "$dst"`. When a session's transcript is root-owned `0600` —
**two are, today** — the `cp` fails, docker exits non-zero, `check=False`
swallows it, and there is no return value and no logging.

**This is silent transcript loss, already happening**, with
`preserve_backend_transcripts` defaulting to `True`. It compounds the coupling
in §7: the same function also breaks under any tmpfs-only fix.

The transcript re-plumb must therefore fix **both** failure modes — it needs a
**detectable failure path**, not merely a new location.

### Attribution (settled as far as it can be)

Every `--user` in `sandbox.py` passes `node` (lines 239, 1379, 1462, 1537,
1749) except one — `_run_workspace_helper` (line 522) — which passes `root` but
bind-mounts only `{workspace}:/workspace:rw`, never the auth volume, with
`extra_args` appended after the image so they cannot inject docker flags.

**No production code path in `sandbox.py` can create a root-owned file inside an
auth volume.** The four files came from an ad-hoc root container outside
production code; the writer was not identified and is not guessed at. Nothing
here needs to defend against a production path — the requirement is to **mask
what exists**.

---

## 7. Transcripts

`preserve_backend_transcripts` is **coupled to the defect**: the default
`claude_transcript_root` (`config.py:626`) is
`/home/node/.claude/projects/-workspace`, inside the very directory being
masked, and `-workspace` is a single key for every candidate of every run
because every workspace mounts at `/workspace`.

A tmpfs-only fix **silently breaks the feature**. The transcript path must
therefore be a **per-candidate host bind**, with `sandbox.py`'s copy-out reading
from that bind instead of re-mounting the auth volume `:ro`. Distinct candidate
IDs must yield distinct transcript roots **even concurrently**, and the shared
`-workspace` key must be eliminated.

### Acceptance gate — transcript preservation

1. **The post-hoc auth-volume copy path is removed**, or made structurally
   unreachable and *proven* so. It is not retained as a fallback: a
   persistent-volume copy path is the defect.
2. **Typed outcome, no swallowing.** Preservation returns a typed
   success / missing / disabled outcome, or raises a redacted actionable
   failure. Nonzero docker or `cp` exits may **not** be swallowed. Today's
   shape — `[ -f "$src" ] || exit 0`, `_run_docker(args, check=False)`, no
   return value, no logging — is precisely what made this silent.
3. **Structural assertion on the final Docker argv:** no fallback ever remounts
   the persistent auth volume to copy transcripts. Make the bad path
   *impossible*, not merely absent, so a future edit cannot reintroduce it.

Five non-vacuous synthetic tests, each naming the mutation it catches:

| # | Case | Required outcome |
|---|---|---|
| 1 | readable node-owned transcript | succeeds |
| 2 | root-owned `0600` transcript | **fails detectably** — and must do so **without reading or deleting** the file. Detect via metadata (`stat` uid / `test -r`), never by attempting a read that partially succeeds. Live in the real volume today. |
| 3 | missing transcript vs. copy failure | **distinguished** — different remedies; today both are the same silent nothing |
| 4 | candidate-specific host bind | survives success, nonzero exit, **and** timeout; concurrent distinct candidates cannot collide — **force the collision attempt**, do not merely assert distinct paths |
| 5 | `preserve_backend_transcripts=false` | the **only** intentionally silent case; everything else speaks |

---

## 8. Reconciling the b07db88 regression suite

The 16 strict-xfail assertions flip only when the fix genuinely makes them pass.
One requires **re-expression rather than a flip**, and it is flagged here rather
than quietly rewritten:

`test_shared_auth_state_is_narrower_than_home` asserts the auth mount must not
target `~/.claude` — but the approved architecture mounts **exactly there**. Its
current form is also vacuous for the spec shape the fix produces:

- the assertion is `not spec.endswith("/home/node/.claude")`, on an f-string
  with no placeholder, and the loop variable `leaky` is unused;
- for `--mount type=volume,src=…,dst=/home/node/.claude,volume-subpath=.claude`
  the destination is not the end of the spec string, so it never fires.

The real invariant it was reaching for is:

> the destination **may** be the auth directory, but every class-2 subdir must
> be overlaid **and** every class-3 sibling file must be redirected or the
> backend must fail closed.

The re-expressed form must be genuinely falsifiable — it will be
mutation-tested, not merely read — and the registry itself needs an anti-vacuity
test proving that **mutating or ignoring any registry class turns the canary
suite red**.

---

## 8b. The denylist escape — volume mode cannot guarantee independence

`CODEX_SQLITE_HOME` is **proven to work**, against a properly established
positive control (production flags, git workspace, synthetic API key):

| Configuration | `~/.codex/*.sqlite*` | redirect dir |
|---|---|---|
| no knob (positive control) | **4** | — |
| `CODEX_SQLITE_HOME` set | **0** | **6** |

So codex's headline concern — `memories`/`goals`/`state` SQLite as siblings of
`auth.json` — is closable, and `memories/` `sessions/` `shell_snapshots/` are
directories that class-2 overlays handle. After redirection the only class-3
siblings left are `config.toml` and `installation_id`.

**But testing those revealed the real limit.** With the **full approved
layout** — subpath auth mount, every class-2 overlay, and the SQLite redirect —
a candidate wrote an *unenumerated* file into the shared auth directory and the
next candidate read it verbatim:

```
candidate A: echo … > ~/.codex/notes-for-next-candidate.txt
candidate B: cat  …  -> AGENT-INVENTED-CHANNEL
```

This is not a gap in the path list; it is **structural**. The auth directory
must be writable, because OAuth rotation requires atomic rename in the
credential's own directory. The agent runs as `node` and therefore can create
**any** file there. A denylist can only mask paths that are enumerated in
advance; a path invented at runtime is not one of them.

**Consequence, stated plainly:**

> Volume mode closes **incidental** cross-run state — the enumerated layout,
> which is what contaminated the three completed demos. It **cannot** prevent a
> deliberate or merely curious agent from opening a channel to the next
> candidate. For a benchmark whose premise is candidate independence, volume
> mode's isolation claim must be scoped to incidental state and must not be
> stated as independence.

Env mode has no such hole: there is no shared store, so there is no directory to
invent a file in. That is why env mode is a first-class mode and not a fallback.

Options for volume mode, none of which restore the guarantee:

1. **Scope the claim** — ship volume mode with the limitation documented, and
   never describe it as candidate independence.
2. **Add a detection control** — after each run, compare the auth directory
   against the pinned expected entry set and fail loudly on anything
   unexpected.
3. **Use env mode** for any lane that must assert independence.

Approved: (1) + (2) for volume mode, and (3) for all demo lanes.

### The drift detector — DESIGNED, IMPLEMENTED, THEN DELETED

A detection backstop was built and is now **removed from the tree**.

It was never wired to a run path, and once volume mode was retired for agent
execution there was no live path for it to guard. Its module docstring still
opened by describing itself as a shipped control for `auth = "volume"` — a
mode that no longer exists — which is the "declared but never wired" defect in
its purest form. Shipping it with a caveat would have preserved exactly the
false assurance this work exists to remove, so it was deleted.

Its limitations are recorded here because they explain why it would not have
rescued volume mode even if wired:

1. **Write-read-delete defeats any end-of-run comparison.** Candidate A writes,
   B reads, B deletes; the directory then matches expectations exactly. No
   end-of-run check can see this — not that implementation, not a better one.
2. **It races concurrent candidates**, and cannot attribute an unexpected entry
   to a particular one.

### Pre-existing class-3 files, and what remains in the shared directory

A class-3 file that was **already in the volume before this fix** stays
readable — an overlay cannot mask a file, and deletion is prohibited by policy.

That residual is a **constant, not a channel**: no post-fix run writes
candidate-derived content there, so every post-fix candidate reads the same
value and none can differentiate a later one. Under the carrying definition,
CARRYING requires a prior candidate's actions to alter a later candidate
*differentially*; a value read identically by all cannot do that.

**Constant means non-differential, not harmless.** A stale pre-fix
`policy-limits.json` can still gate features for every post-fix candidate
*uniformly* — that affects **results** without affecting **independence**. The
two must not be collapsed.

**Scope:** this residual is moot for the published demos, which run env mode and
mount no auth volume at all. It applies only to volume mode, which cannot claim
independence regardless.

The load-bearing half of the premise is proven: for codex, without
`CODEX_SQLITE_HOME` 4 SQLite files land in `~/.codex`; with it, 0 there and 6 in
the redirect directory — the candidate-derived state (memories, goals, session
DB) moves out. `test_argv_applies_every_class3_env_redirect` enforces the knob
and its mount target structurally.

What **remains** in codex's shared auth directory after redirection:

| File | Can it carry candidate-derived content? |
|---|---|
| `installation_id` | **No.** A per-installation UUID (`installation_id.rs`), generated once and stable across runs. |
| `config.toml` | **Could not establish — and the evidence leans toward yes.** It holds MCP server configuration (`MCP server \`…\` is not configured in config.toml`) plus plugin and skill migration state. MCP server naming is precisely the shape that was found CARRYING for claude. |

The `config.toml` answer is deliberately *"could not establish"* rather than a
reassuring one. That is an acceptable answer for a compatibility mode that
cannot claim independence anyway, and far better than a blanket "nothing is
written" premise that would have to be taken on trust.

---

## 8c. What is ARTIFACT-PROVEN, per backend, per mode

This table states what has been **proven against an artifact** — a Docker
canary, a final-argv assertion, or a filesystem measurement — as distinct from
what is *declared supported*. Anything not named here is not proven.

### `auth = "volume"`

| Backend | Status | Proof |
|---|---|---|
| codex | **supported, artifact-proven** | filesystem measurement (no knob → 4 SQLite in `~/.codex`; knob → 0 there, 6 redirected); Docker canary for isolation (prior/run-A sessions and sqlite = 0) **and** for rotation (atomic rename-over persists; next run reads the rotated value); final-argv assertions for every class-2 overlay and class-3 knob + target |
| claude | **refuses** | explicit `unsupported_reason`; refusal asserted at final-argv construction |
| gemini | **refuses** | as above |
| cursor | **refuses** | as above — the split is *plausible but unverified*, not disproven |
| opencode | **refuses** | as above, grounded in measured auth-dir contents |

### `auth = "env"`

| Property | Status | Proof |
|---|---|---|
| no `helix-auth-*` mount | **proven, all five backends** | final-argv assertion, parametrized over `BACKENDS` |
| private uid-correct HOME | **proven, all five backends** | final-argv assertion |
| candidate-keyed transcript bind | **proven, all five backends** | final-argv assertion |
| sequential cross-run isolation | **canary-proven on ALL FIVE runner images** | Docker canary A→B: `home_leak:0 transcript_leak:0 uid:1000 writable:yes` on claude, codex, cursor, gemini, opencode |
| concurrent isolation | **canary-proven — claude image** | C1/C2 with C1 verified `Running` during C2 |
| transcript capture | **canary-proven on ALL FIVE** | run A's transcript captured on the host bind in every image |
| forced transcript collision | **canary-proven — claude image** | identical session ids → two separate host files |
| fail-before | **proven** | the 9f2bcaa `:ro` argv reads a prior run's transcript verbatim; this branch reads nothing |

**Updated after the four remaining canaries ran.** Sequential isolation and
transcript capture are now canary-proven on **all five** runner images. The
*concurrent* and *forced-collision* canaries remain claude-only — stated
explicitly rather than generalised, since the earlier version of this note
overstated coverage in the opposite direction and had to be corrected.

### cursor: refused, and the decision recorded

cursor stays **refused** under volume mode, which is now moot in any case since
volume mode is retired for agent execution. The decision is recorded so a
future reader does not re-derive it:

- **Knob evidence:** the credential is read from the config dir
  (`CURSOR_CONFIG_DIR || XDG_CONFIG_HOME/cursor || ~/.cursor`) while
  `CURSOR_DATA_DIR` governs the data dir. Both default to `~/.cursor`, so the
  files are commingled today and pointing DATA elsewhere *might* split them.
- **What was never established:** which files follow which knob. Plausible is
  not proven, and a benchmark cannot rest on plausible.
- **What a future rescue would need:** a both-halves behavioural proof — the
  ephemeral files move to the redirect target, AND the credential stays
  readable and rotatable in place (atomic rename-over surviving, visible to
  the next run).
- **Why it was not pursued:** no demo uses volume mode, so a rescued backend
  adds audit surface for zero release capability. "We refused it because we did
  not verify it" is a better sentence than "we supported it because it looked
  plausible".

---

---

## 8d. WIRING TABLE — every control, its production call, and its wiring assertion

**This table is a required artifact of the change, not documentation.**

Four separate P1 findings in this branch were the *same* defect: a correct,
well-tested function whose **production call** was unasserted. A library test
cannot detect non-wiring, by construction — so each was found only by mutation,
four times, at four different transition moments (wiring something new, or
closing a different finding). The table exists so the next person inherits the
checklist rather than rediscovering the lesson.

**The rule:** every control needs a test that fails when its *call* is deleted,
not only when its *body* is broken. Two constructions in this branch are the
templates; do not invent a third:

- `test_preflight_calls_the_capability_check_before_touching_the_volume` — a
  recording runner asserting the call happens, and in the right order.
- `test_env_mode_no_volume_paths.py` — an **exploding** runner (zero calls
  cannot be distinguished from the *right* calls by inspecting a recording
  mock), plus a non-vacuity control proving the runner is genuinely wired in.

### The three defect classes this change kept producing

Named because they are distinct, and because a reviewer who knows only the
first will not see the other two.

**1. The check cannot fail (vacuity).** Twelve instances. A guard that passes
because it was given nothing to check, parses nothing, or asserts a string that
is never reached. The sharpest was `opencode`: a fail-closed check cannot
distinguish *"declares nothing unrelocatable"* from *"declares nothing"*,
because an empty set trivially satisfies "no unrelocatable member" — so the
registry certified as safe the backend whose session database sat beside the
credential. Every allow/deny registry anywhere has this trap.

**2. The check does not run (unasserted call).** Four instances —
`capture_claude_transcript`, `ensure_transcript_host_dir`, the transcript raise
propagation, the disclosure emission — plus two modules with no call site at
all. A correct, well-tested function whose *production call* nothing asserts.
**A library test cannot detect non-wiring, by construction**, so each was found
only by mutation, and each appeared at a moment of transition: wiring something
new, or closing a different finding. The wiring table below exists so this class
is *visible* rather than rediscovered.

**3. The check measures a value it constructed (computed expectation).** The
newest, and the one with no prior name. A test that **computes what it expects
instead of observing what shipped** is the same declaration-vs-artifact error in
test form — the check runs, it *can* fail, and it is asserting against a value
the test invented rather than one the system produced.

The instance: the first version of `test_W2_...` recomputed the transcript bind
path from `cwd` and **passed while the directory was absent**, because
production keys the bind off the per-call *temp* workspace. It called the right
production function with the wrong input, which is subtler than reimplementing
the logic and is not caught by "don't duplicate production code in tests". The
correction is to read the path **out of the emitted argv**.

It generalises: *any assertion whose expected value is computed by the test
rather than extracted from the artifact is suspect.* Note where it was nearly
written — into the fix for class 2.

**A second instance, worth recording because of who produced it.** The first
"no argv mutation after validation" test asserted that the guard was the last
statement in `_docker_args`'s **top-level** body. The guard call is nested
inside `if scope == "agent":`, so the statement it actually located was the
enclosing `if` — not the call. An append *inside* that block, after the guard,
was **not caught**; the same append one indent out **was**. It ran, it could
fail, and it measured a node *adjacent to* the property.

That instance was designed by one person and implemented by another, both of
whom had already named this class and were actively hunting it. The remedy is
to measure the property directly — walk every `args` mutation and require none
to follow the guard call in source order, at any nesting depth — rather than a
positional proxy for it.

> **No line numbers, deliberately.** An earlier version of this table cited
> `file:line` for every control. Every `sandbox.py` row was **wrong in the
> commit that created them** — off by exactly 40 lines, because the table was
> written against a pre-commit state of the file and never re-derived. One row
> pointed a reader at a comment inside the *retired, unreachable* volume-mode
> block.
>
> The table had been predicted to rot "at control number eleven". **It rotted
> at commit zero**, before any new control existed. A hand-typed table of code
> locations is a declaration about the artifact, which is what this entire
> finding list has been about — so the table became an instance of the class it
> was created to prevent.
>
> Locations are now stated as *function names*, which move with the code, and
> the exact call sites are **located by AST** in
> `tests/unit/test_wiring_table_is_current.py`, which asserts each control has
> **exactly one** production call site. Generated, never typed.

| Control | Production call site | Wiring assertion |
|---|---|---|
| `_assert_env_is_granted` | `sandbox.py` | `test_T19_provenance_assertion_blocks_unregistered_callers` |
| ~~`assert_layout_is_isolatable`~~ | **NO CALL SITE — retired** | Its only caller was the volume-mode branch, deleted with the retirement. Kept in `backend_layout.py` as the evidence record for why each backend was refused; it is no longer a live control. |
| `_assert_no_shared_home_mount` | `sandbox.py` — `_docker_args`, final statement | `test_docker_args_actually_invokes_the_guard` **(F-20 — added after mutation showed the whole guard was deletable green)** |
| `_assert_agent_argv_uses_only_known_flags` | `sandbox.py` — `_docker_args`, before the mount parser | `test_docker_args_actually_invokes_the_argv_allowlist` **(F-21 — this row was itself added only because the derivation test flagged it missing)** |
| `private_home_tmpfs_arg` | `sandbox.py` (`_docker_args`, agent scope) | `test_env_mode_home_is_private_and_uid_correct` — asserts the emitted argv, incl. `uid=1000` |
| `transcript_bind_arg` | `sandbox.py` (`_docker_args`, agent scope) | `test_env_mode_binds_a_candidate_keyed_transcript_dir` |
| `transcript_host_dir` | `sandbox.py` (`_docker_args` + `run_sandboxed_commands`) | `test_distinct_candidates_get_distinct_transcript_roots`; ordering by `test_W2_run_path_creates_the_bind_dir_before_any_container` |
| `layout_for` | `sandbox.py` — mount guard; `cli.py` — `_ensure_auth_subpath` | `test_docker_args_actually_invokes_the_guard`; `test_F18_login_invokes_the_subpath_bootstrap` |
| `auth_subpath_bootstrap_command` | `cli.py` — `_ensure_auth_subpath` (`_ensure_auth_subpath`) | `test_F18_login_invokes_the_subpath_bootstrap` |
| `assert_volume_subpath_supported` | `authpreflight.py` — `preflight_auth` | `test_preflight_calls_the_capability_check_before_touching_the_volume` |
| `missing_subpath_error` | `authpreflight.py` — `preflight_auth` (stage 0b) | `test_missing_subpath_message_explains_the_daemon_error_it_replaces`; reached via the same preflight call asserted above |
| `preflight_auth` | `evolution.py` — `run_evolution` | **HELD INCIDENTALLY ONLY** — `test_docker_guard.py::test_run_evolution_unit_path_cannot_reach_docker`, whose stated purpose is something else. Needs a purpose-built assertion. |
| `ensure_transcript_host_dir` | `sandbox.py` | **MISSING (F-12/W2)** |
| `capture_claude_transcript` | `sandbox.py` | **MISSING (F-12/W1)** |
| transcript raise propagation | `sandbox.py` | **MISSING (F-12/W3)** — `TranscriptCaptureError` can be swallowed one frame up with a green suite, so F-1 is closed at the library boundary and open at the run path |
| `env_mode_disclosure` emission | `evolution.py` — `run_evolution` | **MISSING (F-17)** — content asserted, emission not |
| subpath bootstrap call | `cli.py` — `_ensure_auth_subpath` | **MISSING (F-18/B1)** — result now bound and checked, but the call itself is unasserted |
| ~~`detect_drift` / `assert_no_drift`~~ | **DELETED** | n/a — retired with volume mode. It had zero call sites and described itself as a shipped control for a mode that no longer exists; an uncalled module that reads as a shipped control *is* the defect, so it was removed rather than left with a caveat. |

Filling the remaining cells is required before the branch is final. An empty
third column is not a documentation gap — it is an unprotected control.

---

## 8e. KNOWN-FAILING TESTS SHIPPED RED, DELIBERATELY

Four tests in `tests/integration/test_oauth_refresh_suppression.py` (the
T22–T25 synthetic refresh family) **fail at this head and are being shipped
that way on purpose.**

**What they test.** That each of `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`
and `CLAUDE_CODE_OAUTH_TOKEN` suppresses container-side OAuth refresh.

**How they fail.** Not on an assertion — on their **own non-vacuity control**:

> `CONTROL FAILED: no refresh was attempted even with no auth env. T23–T25 are
> VACUOUS until this passes — they would be measuring a broken harness rather
> than env-var suppression.`

**Why — CAUSE NOT ESTABLISHED. Two hypotheses have been FALSIFIED by
measurement, and the earlier claim here was wrong.**

An earlier version of this section attributed the failures to *"they need real
network egress to the OAuth token endpoint"*. **That is false and is
withdrawn** — a wrong declaration about the artifact, in the document written
to explain a failure. What has been measured:

| Hypothesis | Verdict |
|---|---|
| Missing network egress | **FALSIFIED.** DNS resolves and `https://platform.claude.com/v1/oauth/token` answers (HTTP 400 to POST, 405 to GET) from the pinned runtime, with no credential material. |
| CLI version difference (`:latest` 2.1.120 vs pinned digest 2.1.138) | **FALSIFIED.** Both give zero token-endpoint contacts on the same fixture. |
| Fixture misses a refresh-gate precondition | **FALSIFIED.** All four hold, checked as the probe runs (`--user node`, `:rw`): `refreshToken` present; `expiresAt: 1` trips any expiry margin; scopes include `user:inference` *and* `subscriptionType: max`; and the credential dir is writable — the lockfile is creatable, dir `1000:1000`, process uid 1000. That last one is the condition that fails *silently*, so it was tested rather than read. |

**What is observed:** the CLI goes straight to `POST /v1/messages`, receives 401,
and never contacts the token endpoint at all.

**Remaining candidates, untested:** that the pinned CLI performs no
*proactive-on-expiry* refresh in headless `-p` mode — in which case T22's
premise, not its fixture, is wrong — and the `linux/amd64`-under-emulation half
of the original diagnosis, which no probe has touched.

**Attribution, verified rather than assumed.** The same four fail identically
at `9f2bcaa`, the base commit. `git diff 9f2bcaa..HEAD` over that file is
empty, and it references none of the symbols this branch changed. They are
**pre-existing and unrelated**. They are also in **no CI gate** — the
`docker_integration` job is `workflow_dispatch`-gated and scoped to
`test_parallel_sandbox.py`.

> **Do not read the suppression property as verified.** These tests have never
> executed successfully in this environment, and the credential-fix author said
> so at the time. The property is asserted by the suite but not demonstrated
> by it.

**Why they are NOT marked skipped.** A skip would silence the one control in
the repository that is loudly announcing its own vacuity — the exact defect
class this branch exists to eliminate. **A red test that says "I am vacuous" is
more valuable than a green one that is.**

**Follow-up (deliberately NOT implemented here — out of scope).** Apply the
same remedy used for `_run_in_image` (F-11): decide availability **before** the
probe by positively identifying that the OAuth endpoint is unreachable, and
skip with that specific reason; if egress **is** available and the control
still fails, **fail**. As written they fail unconditionally, which conflates
*"the environment cannot support this test"* with *"the property is broken"* —
the same missing-vs-failed conflation removed from `transcripts.py` and from
`_run_in_image`.

---

## 9. Remaining proof obligations before the mount rewrite lands

For **each** pinned backend, both halves are required. A proof that shows only
isolation, without showing auth still persists, is half a proof.

1. **Source**: cite bundle/binary evidence for the knob. *(done — §4)*
2. **Behavioural, isolation half**: sequential A→B and live-concurrent C1/C2
   canaries show class-2 and class-3 state is per-run.
3. **Behavioural, persistence half**: on a **synthetic** volume, rotation via
   atomic rename-over reaches the persistent store and is visible to the
   **next** run; `flock` is mutually exclusive across concurrent containers.

Backends that cannot satisfy both **fail closed**.

Also outstanding: the Engine-25 preflight, the login-time ensure-subpath step,
the transcript re-plumb, the evaluator no-mount control, rewriting
`tests/integration/test_parallel_sandbox.py` against a disposable volume (it
currently runs `scope="agent"` against shared `helix-auth-opencode` `:rw` and
would **create** it), the three demo README rewordings, and the full gate
matrix.

### Retired phrasings

`shared volume untouched` and unconditional `zero residue` are retired. Cleanup
claims must distinguish **task-resource cleanup** from **persistent auth-store
state**.
