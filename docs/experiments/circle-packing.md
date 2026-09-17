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

No HELIX result is reported here yet.

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
perfect_score_threshold = 100.0
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

**Score.** Let `s = max(0, worst containment excess, worst pairwise overlap / 2)`. The score is `Σ max(0, r_i - s)`: every radius is reduced by the largest residual violation before summing. Shrinking every radius by `s` removes every violation, so the score never exceeds the radius sum of a packing that satisfies the constraints exactly. A packing with no residual violation scores its plain radius sum. optimize_anything scores a valid packing by its raw radius sum, so a packing that overlaps by up to 1e-6 can score up to about 1.3e-5 more there than any exactly valid packing; the certified score removes that gain. The score is never rounded.

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

[`circle-packing/reference_known_best.py`](circle-packing/reference_known_best.py) is a reference packing for checking the evaluator setup. It embeds a packing (`KNOWN_BEST_CONFIG`) whose radii sum to 2.635982, runs an SLSQP-based search for up to 50 seconds, and returns the embedded packing when that search does not find a packing scoring above 2.6.

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

The protocol turns the example into a test of what HELIX finds on its own, under the evaluation budget optimize_anything used. It removes three problems that made earlier runs of the example unsuitable for reporting:

- **Leaks.** Earlier backgrounds gave the target score and winning strategies. Without the sandbox the mutation agent can read any file on the host, including the reference packing above, and HELIX passes the backend its web tools, so an agent can look up published packings and solvers.
- **Budget accounting.** Without the sandbox the agent can run the evaluator itself during a mutation, and those runs do not count toward `max_evaluations`. HELIX's default `minibatch_size = 3` also charges 3 units for one evaluator run when there is a single task.
- **Scoring.** The earlier evaluator scored invalid packings by their radius sum.

The protocol lives in [`examples/circle_packing/clean_run/`](../../examples/circle_packing/clean_run/): the evaluator service, egress gateway, Dockerfiles, `helix.toml.template` and the scripts named below.

### optimize_anything's circle-packing setup

Verified on 2026-09-16 from the arXiv source of [2605.19633](https://arxiv.org/abs/2605.19633) (the CAIS'26 research paper), the launch blog post (appendix B), `gepa-ai/gepa` `examples/circle_packing/`, and `gepa-ai/optimize-anything-artifact` `acm_cais_artifact_evaluation/domains/circle_packing/` (code, `logs/state_tracker_logs.json`, `logs/gepa_state.bin`).

| Item | optimize_anything |
|---|---|
| n | 26 |
| Validity | `validate_packing(n, circles, atol=1e-6)`: shape (26, 3), no NaN, no negative radius, `x - r ≥ -atol`, `x + r ≤ 1 + atol` (same for y), `dist < r_i + r_j - atol` counts as overlap |
| Score | `validation_details["sum_radii"]` when execution and validation succeed, otherwise `0.0` (artifact `main.py`). The current `gepa` example's refactored `utils.py` (at `15ee314`) instead returns the radius sum of a packing that fails the containment or overlap check |
| Candidate | Python code defining `main(timeout, current_best_solution)`; the evaluator passes the best circles found so far (warm start) and a 600 s timeout |
| Proposer | `openai/gpt-5.1` in the artifact `main.py` and the paper's controlled comparison; `openai/gpt-5` in the blog appendix, the `gepa` example and the paper's summary table. `RefinerConfig()` is on: after each evaluation a refiner LLM call (same model) proposes a refined candidate that is evaluated too |
| Engine | `max_metric_calls=150`, `cache_evaluation=True`, `frontier_type="objective"`; single-task mode, `reflection_minibatch_size` defaults to 1 |
| Budget unit | metric calls: evaluator (fitness function) invocations. Seed evaluation, parent and child minibatch evaluations, full-validation evaluations and refiner re-evaluations all run the evaluator |
| Reported score | 2.63598+ (§5.6); controlled comparison vs OpenEvolve: "2.63598 in just 63 evaluations (costing ~$3.18)", table "Evaluations 63 / 100" |
| Bundled run log | The example's `StateTracker` counts every evaluator call: best first ≥ 2.63598 at call 63, 133 calls in total, best 2.635983362593453. GEPA's own `total_num_evals` in the same state file is 89, and 43 at the first candidate ≥ 2.63598, so it did not count every evaluator call in that run; the paper's 63 is the `StateTracker` count |
| Cost | $3.18 at 63 evaluations (controlled comparison); $6.00 for the circle-packing run (proposer-sensitivity and cost tables); GPT-5-nano run $0.50. No wall-clock figure is given |

### Budget

The budget unit is optimize_anything's: **one execution of the scoring function**. The budget is `max_evaluations = 150`, optimize_anything's configured `max_metric_calls`. Every execution counts: the seed evaluation, each parent re-evaluation, each child's gate evaluation and each full validation.

- **The agent cannot trigger an execution.** The scoring function runs only in the private evaluator sidecar, on a Docker network the agent containers are not attached to. Agents check validity with `check_packing.py`, which reports rule violations and the residual violation `s` but computes no score. This matches optimize_anything, whose proposer cannot call the evaluator, and it means no execution can happen outside the count.
- **HELIX's counter equals the number of executions.** The template sets `minibatch_size = 1` (optimize_anything's single-task default), so one evaluator run is one unit. HELIX also charges a child's gate evaluation and the mutation's tokens when a parent's re-evaluation exhausts the budget; before that fix the last child evaluation of a run went uncounted (`tests/unit/test_budget.py`).
- **An independent ledger counts too.** The sidecar appends one record per execution to its private ledger (`evaluations.jsonl`, with the packing and the scored candidate tree), and refuses any request beyond `max_evaluations + 2` (HELIX may finish the proposal it admitted just below its cap).
- **Headline metric.** The best valid score among the first 150 ledger executions. `summarize.sh` also reports the best score after 25, 50, 63, 100 and 133 executions, the best overall, and HELIX's best accepted candidate.

With one proposal per generation, each generation runs the scoring function two or three times (parent, child, and the child's full validation if it passes the gate). optimize_anything's engine does the same, plus one refiner evaluation per evaluation when the refiner produces a candidate.

### Mutation agent

HELIX's `claude` backend runs Claude Code 2.1.274 in the agent container with `--model claude-sonnet-5 --effort xhigh --max-turns 40 --disallowedTools=WebSearch,WebFetch`. HELIX forwards `[agent].effort` to `claude --effort` unchanged. Claude Code 2.1.274 accepts `low`, `medium`, `high`, `xhigh` and `max`, and Anthropic's API documentation lists all five for Claude Sonnet 5. HELIX's config validation now lists `xhigh` and `max` for `claude`; before, it warned on `xhigh` but still passed it through. The isolation checks capture the request Claude Code sends under HELIX: `model: claude-sonnet-5`, `output_config.effort: xhigh`, `thinking: adaptive`, with no WebSearch or WebFetch tool.

`max_turns = 40` is this protocol's choice; optimize_anything's proposer is a single LLM call per proposal.

### Isolation

| Layer | Mechanism |
|---|---|
| Run directory | A fresh git repository outside any HELIX clone whose only commit holds `solve.py`, `check_packing.py` and `.gitignore`. Config, ledger, logs and scripts live in `.helix-run/`, excluded from the candidate tree and never copied into agent workspaces |
| Agent-visible text | The objective and background give the task, constraints, scoring rule, interface and available libraries, with no target value, reference, source or strategy |
| Filesystem | Docker sandbox (`[sandbox].enabled`); an agent container mounts only a temporary copy of its candidate worktree and a dedicated home volume (`[sandbox].auth_volume`), with no credential in either |
| Web | `[agent].disallowed_tools = ["WebSearch", "WebFetch"]`, plus an `--internal` agent network (`[sandbox].network`) whose only other member is an egress gateway. The gateway forwards only `POST /v1/messages` and `/v1/messages/count_tokens` for allowed models to `api.anthropic.com`, refuses server tools (web search, web fetch, code execution), MCP servers and containers, and injects the API key, which never enters an agent container |
| Evaluator | `[sandbox].evaluator = true` with `[evaluator.sidecar]`: the scoring code, the ledger (`[evaluator.sidecar].volumes`) and the budget counter live only in the sidecar. Candidate code runs there as an unprivileged user that cannot read the scoring code or the ledger, reach the web, or request evaluations |

`isolation_probe.py` verifies all of this through HELIX's own code paths before every launch, at no API cost, and writes a JSON report to `.helix-run/isolation/`. Among its checks:

- the agent sees only the seed files;
- no host mounts or Docker socket, and no evaluator code, ledger or HELIX state on the agent filesystem;
- no credential in the agent's environment or home volume;
- `example.com`, Packomania, GitHub, PyPI, `api.anthropic.com` and `1.1.1.1` are unreachable, and external names do not resolve;
- the gateway refuses to proxy, refuses other routes, web tools, MCP servers and other models;
- the evaluator cannot be reached from an agent container;
- seed, valid, within-tolerance, invalid, NaN, huge/negative-radius, crashing and timed-out candidates score as the rules say;
- candidate code cannot read or write the ledger or read the scoring code;
- no candidate process survives an evaluation, and the execution cap refuses the next request;
- the fully rendered first mutation prompt carries no target or source;
- Claude Code's captured request has the configured model and effort and no web tools.

A real launch adds one `max_tokens=16` request through the gateway to prove the credential works.

### Record keeping

| Record | Location |
|---|---|
| HELIX state, lineage, evaluations, rejected attempts, `helix.log` | `<run dir>/.helix/` |
| Candidate worktrees (with each accepted mutation's prompt and transcript) | `<run dir>/.helix/worktrees/` |
| Every scoring execution, with packing, score, validity details, timing and the scored tree | `.helix-run/ledger/evaluations.jsonl`, `.helix-run/ledger/candidate_archives/` |
| Refused evaluator requests, sidecar starts | `.helix-run/ledger/rejected_requests.jsonl`, `service_events.jsonl` |
| Every model API request: decision, model, effort, tools, status, token usage | `.helix-run/logs/gateway/gateway_requests.jsonl` |
| Every Claude Code session transcript (including rejected mutations) | `.helix-run/transcripts/` |
| HELIX console output, wall-clock start/end, provenance (commit, image ids, versions) | `.helix-run/logs/` |
| Isolation reports, summary | `.helix-run/isolation/`, `.helix-run/summary/` |

### Running it

```bash
cd "$(git rev-parse --show-toplevel)/examples/circle_packing/clean_run"
scripts/build_images.sh                  # helix-circle-clean/{base,agent-claude,evaluator,gateway}:1
scripts/setup_run_dir.sh /path/to/helix-circle-packing-clean-run   # MAX_EVALUATIONS, MODEL, EFFORT, MAX_TURNS

cd /path/to/helix-circle-packing-clean-run
export ANTHROPIC_API_KEY=...             # handed only to the gateway container
.helix-run/bin/launch.sh                 # checks, isolation probe, auth check, then HELIX in the background
tail -f .helix-run/logs/helix_stdout.log
.helix-run/bin/summarize.sh --rerun-best # best score, generation, budget, cost, time, independent re-check
.helix-run/bin/stop.sh [--purge]         # stop (resumable with launch.sh --resume); --purge removes the run's Docker objects
```

`launch.sh --preflight-only` stops after the isolation checks. `launch.sh --dry-run` runs the whole pipeline against a capture-only gateway, so every mutation fails and no API budget is spent; use it on a throwaway run directory with a small `MAX_EVALUATIONS`. A watchdog stops HELIX (resumably) after three consecutive authentication or billing failures from the API, so a dead credential cannot burn the evaluation budget. `summarize.sh` checks the best packing with `independent_check.py`, which shares no code with the evaluator: it verifies validity in exact rational arithmetic and that the score does not exceed the certified exactly-valid radius sum. `--rerun-best` re-executes that candidate once in a network-less container, outside the budget, to check that it reproduces its packing.

The HELIX checkout must be clean at launch; the commit is recorded with the run.

### Differences from optimize_anything's setup

The budget unit, budget size, n, validity rule and invalid-packing score match. What differs is inherent to HELIX or chosen for this protocol:

- **Proposer.** HELIX's proposer is a multi-turn Claude Code session (Claude Sonnet 5, effort `xhigh`, up to 40 turns) that edits a repository and can run code, including its own packing searches, in its container. optimize_anything's proposer and refiner are single GPT-5.1 calls that cannot run code. Agent-side compute is bounded by the turn cap and the container, not by the evaluation budget.
- **No refiner, no warm start.** optimize_anything evaluates a refiner-proposed variant after each evaluation and passes the best circles found so far into every candidate. HELIX's candidates call `pack_circles(26)` with no prior solution.
- **Score.** Certified (see [Evaluation](#evaluation)), so tolerance slack earns nothing.
- **Environment.** Python 3.11, numpy 2.3.3 and scipy 1.16.2 in both agent and evaluator containers, and a 600 s limit for `pack_circles(26)`. optimize_anything's requirements list many more packages (optuna, cvxpy, numba and others).
- **Cost accounting.** Token cost is measured at the gateway at Anthropic list prices and reported next to Claude Code's own cost estimate; optimize_anything reports OpenAI proposer costs.

### Residual risks

- **Pretraining knowledge.** Claude Sonnet 5 may have seen published n=26 packings, their coordinates, or solver code from before its training cutoff. No isolation can remove what the model already knows.
- **The background states the scoring rule.** That is part of the task definition, not a hint, but it does tell the agent that the score is a radius sum.
- **Agent-side search.** The agent can compute radius sums of its own packings; only calls to the scoring function are budgeted, as in optimize_anything.
- **Home volume persists within the run.** HELIX resets Claude Code's session state every generation, but other files an agent writes under its home directory persist to later generations of the same run.
- **Candidate code shares the sidecar.** It runs as an unprivileged user in the evaluator container; kernel-level escapes are out of scope. Its stdout and stderr tails reach the agent as feedback.
- **Transcript archiving is best effort for very short sessions.** The archiver polls every 0.5 s; a session that fails within milliseconds can be wiped by the next generation's reset before it is copied.
- **Gateway trust.** The gateway enforces the request shape; Anthropic-side behavior of an allowed request is not observable beyond the logged usage.
- **Nondeterministic candidates.** The evaluator is deterministic for a given packing, but a candidate that searches until a time limit can return different packings on different runs; parents are re-scored every generation.

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
