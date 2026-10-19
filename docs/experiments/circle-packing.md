# 🔵 Circle Packing Experiment

> **Objective:** Pack 26 non-overlapping circles in a unit square, maximizing the sum of radii.

---

## Overview

Circle packing is a classic computational geometry problem and a common benchmark for LLM-driven code evolution: the search space is continuous, and evaluation is fast and deterministic.

The example runs a single task: `helix.toml` sets `[dataset]` sizes of 1 and 1, so HELIX passes the evaluator one example id, `"0"`, on both the train and val splits.

| | Value |
|---|---|
| **Circles** | 26 |
| **Container** | Unit square [0, 1] × [0, 1] |
| **Score** | Sum of radii of a valid packing (certified, see [Evaluation](#evaluation)) |
| **Constraints** | No overlaps, all circles within bounds, absolute tolerance 1e-6 |
| **Mode** | Example-id handoff with one example (`train_size = 1`, `val_size = 1`) |

Two setups exist:

- [`examples/circle_packing/`](../../examples/circle_packing/) is the quick local example. It runs the evaluator in the candidate tree, without isolation, and is not suitable for reporting a result.
- [`examples/circle_packing/clean_run/`](../../examples/circle_packing/clean_run/) is the [clean-run protocol](#clean-run-protocol): a sandboxed, leak-free run whose evaluation budget matches optimize_anything's. Results for this page come only from that protocol.

A two-generation smoke run produces a valid packing with radius sum
**2.63598308491755** under the clean-run protocol. The full run capped at
200 scoring executions remains incomplete. See [Current run state](#current-run-state).

---

## Configuration

The quick local example is [`examples/circle_packing/helix.toml`](../../examples/circle_packing/helix.toml):

```toml
[evaluator]
command = "uv run --no-project --python 3.12 --with numpy --with scipy python3 evaluate.py"
protected_files = ["evaluate.py"]

[dataset]
train_size = 1
val_size = 1

[evolution]
max_generations = 30
merge_enabled = false
max_evaluations = 500

[agent]
backend = "claude"
model = "haiku"
effort = "low"
max_turns = 20
```

`[agent].background` states the task, the constraints, the `pack_circles(n)` interface, and the evaluator's Python version and packages.

---

## Seed Solution

[`examples/circle_packing/solve.py`](../../examples/circle_packing/solve.py) places one circle at the center, a ring of 8 around it, and the remaining 17 on an outer ring clipped to the square. It then shrinks radii greedily until no circles overlap and every circle fits inside the square.

**Seed score: 0.979764.** The command in the example README reproduces it. It uses the same placement and radius computation as the seed in appendix B of GEPA's blog post, which the post says it adopted from ShinkaEvolve. The post's trajectory figure shows that seed at a score of 0.98.

---

## Evaluation

Both evaluators (`examples/circle_packing/evaluate.py` locally, and `clean_run/evaluator/packing_rules.py` in the clean run) apply the same rules.

**Validity.** A packing is valid when it has exactly 26 rows `(x, y, r)` of finite real numbers with `r ≥ 0` and, with absolute tolerance `tol = 1e-6`:

1. **Containment:** `r - x ≤ tol`, `r - y ≤ tol`, `x + r - 1 ≤ tol`, `y + r - 1 ≤ tol` for every circle.
2. **No overlap:** `r_i + r_j - hypot(x_i - x_j, y_i - y_j) ≤ tol` for every pair `i < j`.

This is optimize_anything's `validate_packing` rule and tolerance (`atol=1e-6`). An invalid packing, an exception or a timeout scores **0**, as in optimize_anything.

**Score.** Let `s = max(0, worst containment excess, worst pairwise overlap / 2)`. The score is `Σ max(0, r_i - s)`: every radius is reduced by the largest residual violation before summing. Shrinking every radius by `s` removes every violation, so the score never exceeds the radius sum of a packing that satisfies the constraints exactly. A packing with no residual violation scores its plain radius sum. optimize_anything scores a valid packing by its raw radius sum, so a packing that overlaps by up to 1e-6 can score up to about 1.3e-5 more there than any exactly valid packing; the certified score removes that gain. The score is never rounded. There is no normalization or clipping to a published result, and
`perfect_score_threshold` is left unset. Any valid improvement receives its
full certified radius sum, including one above a previously reported result.
Only the execution budget, generation guard or an operational failure ends a
run; matching a reference value does not.

The previous local evaluator on `main` summed radii without rejecting invalid
geometry, and its prompt supplied published targets and optimization recipes.
Results from that setup cannot establish valid packings or independent search.
The isolated protocol below replaces that setup for reported experiments; it
starts from the geometric seed, keeps reference coordinates and scores out of
the agent workspace, and validates the candidate's returned geometry in a
separate process. The existing smoke remains a historical protocol check;
a fresh completed run is still needed for a benchmark claim.

Output format (one pair per id in `helix_batch.json`), shown for the seed from the local evaluator:

```text
HELIX_RESULT=[[0.979764..., {"scores": {"sum_radii": 0.979764...}, "valid": true, "certification_shrink": 1.4e-17, "feedback": "All 26 circles are valid: inside the square, no overlaps. ...", "bounds_violations": {"count": 0, "worst": []}, "overlap_violations": {"count": 0, "total_overlap_depth": 0.0, "worst": []}, ...}]]
```

---

## Reference Values

| Source | Sum of radii (n=26) | Notes |
|---|:---:|---|
| Best known packing | 2.63598+ | [Erich Friedman's Packing Center](https://erich-friedman.github.io/packing/cirRsqu/) |
| GEPA optimize_anything | 2.63598+ | [Introducing optimize_anything](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/) |
| AlphaEvolve | 2.635 | [AlphaEvolve results notebook](https://github.com/google-deepmind/alphaevolve_results/blob/main/mathematical_results.ipynb), section B.12, which lists the previous state of the art as 2.634. The radii of its Construction 1 sum to 2.635863 |

These are best known values, not proven optima.

---

## Reference Packing

[`circle-packing/reference_known_best.py`](circle-packing/reference_known_best.py) is a reference packing for checking the evaluator setup. It returns supplied coordinates (`KNOWN_BEST_CONFIG`) whose radii sum to 2.635982. It performs no optimization or search. Scoring this fixture checks the evaluator installation; it is not evidence of discovery and is never used as the experiment seed.

```bash
tmp_dir=$(mktemp -d) && cp "$(git rev-parse --show-toplevel)/examples/circle_packing/evaluate.py" "$tmp_dir/" \
  && cp "$(git rev-parse --show-toplevel)/docs/experiments/circle-packing/reference_known_best.py" "$tmp_dir/solve.py" \
  && printf '["0"]\n' > "$tmp_dir/helix_batch.json" \
  && cd "$tmp_dir" \
  && uv run --no-project --python 3.12 --with numpy --with scipy python3 evaluate.py
# HELIX_RESULT=[[2.63598..., {"scores": {"sum_radii": 2.63598...}, "valid": true, ...}]]
```

---

## Clean-Run Protocol

The protocol uses a standalone seed repository, a private scoring service, an
isolated Claude Code mutation container, and a gateway that permits model
inference without remote-content tools. The implementation is in
[`examples/circle_packing/clean_run/`](../../examples/circle_packing/clean_run/).
The smoke run checks the protocol and candidate reproduction; it does not
establish a benchmark result.

### Current run state

The 2026-09-17 smoke uses a two-generation cap and a three-execution cap,
with `claude-sonnet-5`, `xhigh` effort and at most 40 turns per mutation.
Its best candidate, `g1-s1`, produces a radius sum of
**2.63598308491755** at scoring execution 2. The independent rational-arithmetic
check finds exactly 26 circles, no boundary violations and no overlapping
pairs at zero tolerance. A fresh networkless container reproduces the same
packing and score. The saved state reaches generation 2; the ledger records
two scoring executions and six answered requests, four served from cache.
The three-execution budget is not exhausted.

This is the output of a capped smoke run under the stated protocol, not a
benchmark result. The smoke summary records 42 Bash calls, including 29
running Python, and about 4,764 sampled agent CPU-seconds across two
containers. The final candidate contains all 78 coordinate/radius values as
literals. Network isolation does not rule out pretrained knowledge, and the
scoring-execution cap does not bound local search inside a mutation session.

The full budget-matched run has a cap of **200 scoring executions** and is
**incomplete**. Its own watchdog stops it with reason `credential`; the run
segment lasts **335 seconds**. The stop record's evidence contains three
HTTP **200** responses with no error type or error message. The gateway log
contains older subscription-login failures from an earlier preflight, followed
by successful inference in this run. The watchdog's historical-error check
causes the stop despite those successful sampled requests.

The watchdog now considers gateway records added after it starts, clears
historical login failures when later inference succeeds, and requires three
consecutive fatal attempts before recording the actual failures that trigger
a credential stop. Regression tests cover
successful requests after a login failure, startup with stale failures, and
consecutive new login failures after recovery. This code fix does not complete
or rerun the full experiment. Its saved state remains at generation 1 with only the
seed on the frontier; the ledger contains two scoring executions and four
answered requests, while HELIX's saved request counter is three. The full
run's summary directory is empty.

The evidence lives in separate run directories: the smoke is
`helix-circle-packing-stageC-smoke-20260917`, and the full run is
`helix-circle-packing-clean-run`. In each directory, `.helix/state.json`
records search state and `.helix-run/ledger/` records executions and requests.
The smoke's `.helix-run/summary/summary.json` records the exact check and
fresh-container reproduction; the full run's
`.helix-run/logs/watchdog_stop.json`, `run_times.jsonl`, and
`gateway/gateway_requests.jsonl` record the stop and request outcomes.

### Reference experiment and budget

The bundled optimize_anything circle-packing state is byte-identical to
[`gepa` commit `133d185a`](https://github.com/gepa-ai/gepa/tree/133d185a/examples/circle_packing).
That source configures `max_metric_calls = 200`, an `openai/gpt-5` proposer
and a GPT-5.1 refiner. Its fitness function caches code executions and its
refiner caches `(code, refiner_prompt)` pairs. Unchanged parents do not execute
again. The bundled StateTracker records 133 executions, first reaches the
reported 2.63598 threshold at execution 63, and reaches its final best at
execution 114. GEPA's internal counter is 89; it is a different quantity.
The state SHA-256 is
`ca0ec9aef9e77e02b6947e54134054b198b4c8a2228be8ab439c4977e020b06d`.

The [artifact's fresh-run source](https://github.com/gepa-ai/optimize-anything-artifact/tree/main/acm_cais_artifact_evaluation/domains/circle_packing)
sets 150 and GPT-5.1. This is the provenance of the 150 setting in the build
configuration; its 152 sidecar backstop allows two additional executions.
Neither number is this protocol's budget. The protocol caps the scoring
service at **exactly 200 executions**, including failed and interrupted
executions. There is no overshoot allowance.

The reference task has 26 circles, absolute tolerance `1e-6`, a 600-second
candidate limit, and a warm-start input containing the best packing so far.
Its score is the raw radius sum of a valid packing and zero on failure.
The [paper](https://arxiv.org/abs/2605.19633) reports 2.63598+ and a controlled
comparison reaching 2.63598 in 63 evaluations for approximately $3.18;
it also lists $6 for a GPT-5.1 circle-packing run. It gives no wall-clock time.
Those paper tables and the bundled trace are distinct evidence sources.

The private sidecar is the authority for this run's execution count:

- `execution_starts.jsonl` durably allocates a number before execution;
  `evaluations.jsonl` stores the result, packing, candidate id and archive.
  Crashes leave an identifiable unfinished start; restarts never reuse it.
- A deterministic archive digest caches identical trees across parent,
  child and validation requests. `requests.jsonl` records every answer,
  including cache hits and its originating execution. Cache hits consume
  zero executions. Every new scoring execution consumes one, including
  evaluations triggered by agent-generated proposals; failures and interrupted
  executions still count.
- At the cap, the service refuses new executions and new proposal rounds.
  Cached validation of the last candidate remains possible. The watchdog
  stops HELIX on the budget refusal and checks evaluator failures,
  deadline aborts and unexplained request-counter divergence.
- HELIX counts answered requests, including sidecar cache hits. Its own
  `max_evaluations = 0` disables that separate request cap. The external
  `MAX_EVALUATIONS=200` renders `CP_MAX_EXECUTIONS=200`. The 1,000-generation
  guard can stop earlier when proposals fail or repeat; summary marks an
  incomplete budget explicitly. `minibatch_size=1`, one proposal per
  generation, and disabled merges keep attribution unambiguous.
- The summary records the entire best-so-far curve, checkpoints at 63, 133
  and 200, and a curve indexed by first execution of distinct candidate
  trees. Unreached checkpoints are null, never extrapolated.

**The execution budget does not bound agent-side search.** The mutation
agent can compute radius sums and run local solvers. It cannot contact the
private evaluator, so it cannot trigger an unledgered execution of that
service. This is an evaluation-budget comparison with different proposer
compute, not a compute-matched comparison with optimize_anything.

### Mutation and subscription authentication

The smoke and stopped full run use Claude Code 2.1.274 with
`--model claude-sonnet-5 --effort xhigh --max-turns 40
--disallowedTools=WebSearch,WebFetch`. New run directories default to
`claude-haiku-4-5` with no effort override; `MODEL` and `EFFORT` select overrides
when running `setup_run_dir.sh`. The agent container has
2 CPUs and a 5,400-second session limit. The monitor records sampled
container CPU-seconds, memory, timing and CPU limits; transcripts supply
Bash/Python counts and tool-result latency. CPU totals are lower bounds:
short sessions and the unsampled tail can be missed. Tool latency can
include scheduling and does not measure background subprocess lifetime.
The best candidate's source is checked for coordinate literals; this is a
heuristic, not proof of how its packing is produced.

Authentication uses the subscription login created by
`helix sandbox login claude` in Docker volume `helix-auth-claude`.
The gateway mounts that volume at `/auth`; its credential file is
`/auth/.claude/.credentials.json` (uid 1000, mode 0600). It reads the OAuth
access token, renews it through Claude Code's fixed refresh endpoint, and
writes refreshed credentials only in that same volume. It never logs
credentials or mounts this volume into the mutation agent. A failed refresh
requires a renewed login and prevents successful inference.

The agent mounts a separate home volume, `helix-circle-clean-claude-home`,
containing no login credentials. Its `CLAUDE_CODE_OAUTH_TOKEN` is a literal
non-secret placeholder, which selects Claude Code's subscription request
format. The gateway replaces the placeholder with the real subscription
Bearer token and preserves the subscription beta header. No API key is used.
The launcher scrubs inherited credentials before starting HELIX and probes.
`stop.sh --purge` removes only the run's home volume, never the subscription
login volume. See [Claude Code authentication](https://code.claude.com/docs/en/authentication).

### Isolation and evaluator

The candidate repository contains only `solve.py`, `check_packing.py` and
`.gitignore`. The objective/background describe the task, rules, interface
and available libraries without targets, sources or strategy suggestions.
Config, state, ledger, scripts and logs live under excluded `.helix*` paths.
Agent containers mount a temporary candidate copy and their separate home,
with no Docker socket, host directories, evaluator, ledger or HELIX state.
Sync-back drops new links that escape the worktree, skips transcript links,
and replaces a symlinked `.gitignore` without reading or writing its target.

The agent network is Docker `--internal`. Only the gateway connects it to
an egress network. Direct HTTP, HTTPS and DNS requests to example.com,
Packomania, GitHub, PyPI and api.anthropic.com fail. The gateway forwards only
Messages and count_tokens to a fixed Anthropic host and allowed models.
It refuses WebSearch/WebFetch, hosted tools, MCP, containers, remote URL/file
sources (including nested document and tool-result content), unknown body
parameters and remote-content beta features. Logs contain policy metadata,
content/source types, status, usage and timing; request/response bodies and
authorization headers are omitted.

The evaluator occupies another internal network, unreachable from agents.
Candidate code runs as uid 2000 with a 600-second limit, resource limits,
single-threaded BLAS and deterministic Python hashing. Root-only parent
paths protect the evaluator code and ledger even on Docker Desktop mounts.
After execution the service kills candidate processes, removes candidate
files from all writable directories, and deletes owned System V IPC state.
Output files must be regular files and open nonblocking without following
links. A service-wide deadline records an aborted execution and exits if
postprocessing stalls. Host-stored stdout/stderr tails are capped at 256 KiB
each; full streams exist only temporarily inside the container.

The validity rule matches the reference tolerance. Invalid packings score
zero. Valid packings use the certified radius sum described in
[Evaluation](#evaluation), so numerical slack cannot increase the score.
The seed uses no embedded reference packing. Python 3.11, numpy 2.3.3 and
scipy 1.16.2 are shared between agent and evaluator images.

`isolation_probe.py` exercises real containers through HELIX's launch paths:
network/host/evaluator probes, web-tool and remote-source denials,
agent-planted symlinks, FIFO output, persistence, invalid geometry, timeout,
ledger and execution cap. It renders the first mutation prompt from a
separate seed evaluation using the production 600-second setting. Capture
checks observe the first Claude Code Messages request only; later turns
rely on gateway enforcement. The optional auth probe verifies a small real
subscription request through the gateway.

### Records and summaries

`.helix/` holds HELIX state, evaluations, lineage and attempts.
`.helix-run/ledger/` holds execution starts, results, requests, service events
and scored candidate archives. `.helix-run/logs/` holds gateway metadata,
container compute, watchdog decisions, launch provenance and wall times.
`.helix-run/transcripts/` holds native sessions, including rejected proposals.
Candidate ids in the ledger preserve generation attribution after worktree
removal. The build records the source commit and verifies image file hashes.

The summary independently checks the best packing using exact arithmetic
and, by default, reruns its archive in a fresh networkless evaluator
container with no host mounts. Failed checks produce an INVALID or NOT
REPRODUCED headline and nonzero exit. It reports incomplete budgets,
unfinished/gapped ledger entries, counter mismatches and unreached
checkpoints. Fresh-container reproduction is an audit execution outside the
search budget and is identified separately. `--no-rerun` skips it explicitly.

Usage is reported as subscription token counts and any provider-supplied
subscription limit headers. Dollar estimates from Claude Code are identified
as API-equivalent estimates, not subscription charges; token totals alone
do not establish subscription allowance consumption.

### Running it

```bash
cd /path/to/helix-checkout
uv run helix sandbox login claude
examples/circle_packing/clean_run/scripts/build_images.sh
examples/circle_packing/clean_run/scripts/setup_run_dir.sh /path/to/circle-run
cd /path/to/circle-run
.helix-run/bin/launch.sh --preflight-only --api-auth-check
.helix-run/bin/stop.sh --purge
.helix-run/bin/launch.sh
.helix-run/bin/summarize.sh
.helix-run/bin/stop.sh
```

The checkout must be clean and images must match its committed source.
`launch.sh` starts evolution in the background after preflight. A separate
capped smoke uses a temporary configuration and separate Docker objects;
it leaves the committed full-run configuration untouched.

### Comparison limits

The recorded smoke and stopped full run use a tool-enabled Sonnet mutator
and an entire repository;
the reference uses GPT-5 reflection plus a GPT-5.1 refiner and code strings.
HELIX supplies no warm-start argument to `pack_circles(26)` and removes
candidate state between executions. Parent feedback still carries the
parent's circles. Identical trees reuse their first result, so nondeterministic
solvers are not resampled; fresh-container reproduction can fail.
The scoring certification and available libraries also differ.

Pretraining knowledge of published coordinates cannot be removed.
Agent-side search and files retained in its home can carry knowledge across
sessions, even though HELIX resets native session state. Container kernel
escapes are outside this experiment's isolation model. Gateway policy
constrains API shapes; it cannot inspect Anthropic's internal behavior.
Transcript archiving and CPU sampling are best effort for very short sessions.

---

## Reproducing the quick local example

Copy the example to a standalone directory outside any git repository, then run HELIX there. The copy keeps the run's git repository, candidate worktrees, and `.helix/` state separate from the HELIX clone:

```bash
work_dir="$(mktemp -d)/circle_packing"
cp -R "$(git rev-parse --show-toplevel)/examples/circle_packing" "$work_dir"
cd "$work_dir"
helix evolve
```

`helix evolve` initializes the copy as a git repository and commits it as the seed. After the run, `helix frontier`, `helix best`, and `helix log` show the frontier, the best candidate, and the mutation log. This setup has no isolation: the agent can read the evaluator, run it outside the budget, and use the web. Report results only from the [clean-run protocol](#clean-run-protocol).

---

<div align="center">
<sub><a href="../../README.md">← Back to HELIX</a></sub>
</div>
