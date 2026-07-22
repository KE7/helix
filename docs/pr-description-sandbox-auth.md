# PR description — credential isolation for sandboxed mutation agents

## One-paragraph version

Credential isolation in HELIX previously depended on a table in
`helix/backends.py` that no configuration reviewer would think to check, and
the injection it performed ran *downstream* of the environment scrubber that
tests asserted on — so a credential could be absent at the scrubber and still
be handed to the container. Two further channels (a `HELIX_*` wildcard
passthrough and the `passthrough_env` field) bypassed that table entirely; one
of them is live in an existing lane configuration today. This change moves the
allowlist into each run's own configuration, makes the safe mode the default,
gives every variable an explicit origin and scope, removes two sufficiency
signals that were observed reporting success while authentication was broken,
and moves verification to a real authenticated operation performed before the
first mutation is dispatched.

## What is and is not established

Use these three categories verbatim; they are the reconciled form.

- **PROVEN:** `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` injection disables
  OAuth mode and prevents **container-side** proactive and 401-triggered
  refresh.
- **PROVEN:** with no auth env present, an expired synthetic OAuth record
  **does** reach the OAuth token POST (empirically, on a disposable volume).
- **NOT PROVEN:** *why* the real refresh token is rejected by the server.
  Rotation / invalidation remains a **hypothesis**.

Do not write, and do not let it survive into any artifact: that injection
caused *server-side* invalidation, or that it is the *sole* reason the shared
record is dead. Neither follows. Suppressed container-side refresh explains
why the container never renewed the token; it does not explain a server
rejecting a stored refresh token, and other causes remain live — notably the
host CLI refreshing against the same account and rotating it.

**The headline is exactly two statements:** fixing the injection is what makes
volume auth **self-sustaining in containers**, and the shared record is
currently **dead for a reason we have not established**. Both must travel
together, or the change will be oversold and then appear to fail.

## Why a backend-table-only fix would not have worked

`examples/formulacode` ships `passthrough_env = ["ANTHROPIC_API_KEY"]` at top
level with the sandbox enabled. That reaches the mutation agent through a
channel the per-backend table never mentions. Under a fix gating only the
table, this lane keeps injecting, and every argv assertion written against the
table still passes. It is the concrete proof that the expanded scope was
necessary, and the lane's migration is semantically correct rather than a
workaround — it genuinely wants environment auth, and now says so where a
reviewer reads it.

## Review notes

- The enforcement boundary is the **final container argv**, evaluated over all
  three origins together. No test asserts on the scrubber alone; that gap is
  what hid this through four reviews.
- `_docker_args` performs a **second, independent** check that every variable
  it emits carries a grant for the scope it is launching, and refuses agent
  scope without grants. The original bug was a new call site added downstream
  of the control everyone was asserting on, so the re-check lives at the point
  of emission, where there is nothing downstream left to hide.
- The preflight probe is built by the **same** `_docker_args(scope="agent")`
  call as the production mutation. This is deliberate: probing a *copy* of the
  auth volume, or mounting it `:ro`, produces a silently wrong result, and
  both look like improvements to a reviewer who has not read the analysis.
  Because there is no probe-specific mount, isolating the probe now requires
  changing the production agent mount at the same time — which breaks the run
  loudly instead of greening the probe silently.
- The preflight must **never** call `charge_evaluation`. Lane inspectors
  assert budget conservation as a hard equality on `budget["evaluations"]`, so
  any non-proposal increment breaks all of them at once. Auth cost is recorded
  in separate `auth_overhead_*` counters and reported distinctly.

## Known limits, carried deliberately

- **Toggling `sandbox.enabled`** still moves a run between volume and env auth
  without an explicit declaration. This is an accepted hole, not an oversight:
  requiring an explicit `auth` when the sandbox is disabled would break every
  non-sandboxed config. Declaring `auth` while disabled is a hard error, so a
  config that *stated* an intent cannot silently lose it; one that never
  stated one still flips silently.
- **A real-operation preflight can false-refuse** on a network blip, which is
  expensive for a single-window run. Transport failures are distinguished from
  auth failures in the message, but the classifier is a heuristic.
- **The concurrency lock cannot cover a non-HELIX CLI** on the same host
  touching the same account or volume. The preflight's verdict is scoped to
  "valid at the moment we checked".
- **Name-based disjointness compares names, not values.** Two differently
  named variables holding the same secret are not detected.
- **The provenance record is unauthenticated data inside the volume.** It
  detects accident (skew, wrong backend), never malice, and must not be
  described as a security control. It does not exist on any pre-existing
  volume, so for one release the skew check warns and catches nothing on the
  volumes that matter today.

## C7 — the runtime 401 detector: re-examined, and deliberately NOT shipped

The audit asked for this to be re-examined before implementing, flagging it as
the softest load-bearing part of the proposal. Conclusion: **the runtime 401
abort is not shipped in the proposed form, and should not be.**

The proposal was to abort a run when a mutation returns a 401, rather than
scoring a failed proposal. The detector's input would be the mutation agent's
output. But by this project's own threat model, the mutation agent runs with
`--dangerously-skip-permissions` over attacker-influenceable repository
content, on a `bridge` network with egress. Its output therefore routinely
contains text from HTTP calls the agent itself made and from the candidate
repo's own test suite.

That makes the false positive **adversarially reachable**, not merely
occasional: a candidate whose tests print `401 Unauthorized` would abort every
run. Aborting healthy runs is worse than the status quo, and worst precisely
where the audit says a clean refusal matters most — a single timed window.

Two supporting observations:

1. **The codebase already has the defensible version of this heuristic.**
   `_looks_like_rate_limit` is applied to the backend's *own structured JSON
   error field*, not to free-form agent output. That input is not
   agent-controlled. Any future runtime auth detector should be restricted to
   the same envelope; scanning stdout is a different and much weaker thing
   wearing the same name.
2. **The preflight already covers the dominant failure mode soundly.** A dead
   volume at run start — the actual observed failure — is caught before
   dispatch, with zero side effects, on input HELIX fully controls.

What remains uncovered is a token that dies *mid-run*. That gap is real and is
stated rather than papered over: the preflight's verdict is scoped to "valid
at the moment we checked". The auditor's safer variant (abort only when every
proposal in a generation fails the same way) is the right shape for closing
it, and it needs the structured-envelope input above rather than stdout
scraping. It is not in this release.

Both properties are pinned by tests: agent output containing 401 text is
inert, and the preflight classifier's input is asserted to be a HELIX-authored
probe command rather than anything agent-derived.

## C14 — copy-isolation made structurally impossible

The audit asked whether copy-isolation could be made structurally impossible
rather than merely tested against, and said that would beat its own proposal.
**It can, and it is.**

The proposal's defence was a source comment plus a test (T27) asserting the
mount string — which the auditor correctly called weak against a future
refactor by someone who has not read the analysis, because the violation
*looks like an improvement*.

Instead, the probe does not construct a mount at all. Its argv is produced by
the **same `_docker_args(scope="agent")` call** that launches the production
mutation container, with the same `SandboxConfig`. There is no probe-specific
volume string, no probe-specific mount mode, and no parameter that could point
it at a copy.

The consequence is what makes it structural: a reviewer who "isolates the
probe" must change the production agent mount at the same time. That does not
silently green the probe — it changes how every mutation container is
launched, which fails loudly and visibly. The dangerous edit and the obvious
breakage are now the same edit.
