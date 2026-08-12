<div align="center">

# 🧬 HELIX

### Hierarchical Evolution via LLM-Informed eXploration

*DNA evolves. So does your codebase.*

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![PyPI](https://img.shields.io/pypi/v/helix-evo.svg)](https://pypi.org/project/helix-evo/) [![CI](https://github.com/KE7/helix/actions/workflows/ci.yml/badge.svg)](https://github.com/KE7/helix/actions/workflows/ci.yml) [![mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/)

<br>

**Evolutionary optimization for full codebases using agentic coding tools as the mutation engine and git worktrees as the population pool.**

HELIX brings reflective Pareto evolution out of the single-artifact setting and into real software projects: entire repositories, multi-turn agentic mutation, tool use, web research, and verification loops, all inside a single evolutionary stage. Supported mutation backends include Claude Code, Codex CLI, Cursor Agent, Gemini CLI, and OpenCode.

<br>

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Configuration](#-configuration) · [CLI Reference](#-cli-reference) · [Results](#-results)

</div>

---

> **Safety:** HELIX never modifies your working branch, HEAD, staging area, or remote. All mutations live in detached worktrees under `.helix/worktrees/` and branches named `helix/*`. If your checkout is dirty, HELIX snapshots the current tracked and untracked changes into the seed worktree while leaving your original checkout untouched. Run `helix clean` to remove saved state and worktrees when you are done.

---

## Why HELIX?

HELIX is built for a setting that today's evolution systems still do not really handle, including systems like KISS and OpenEvolve: improving **real, multi-file codebases** where useful mutations require exploration, iteration, and tooling, not just a single blind rewrite.

Instead of treating one file or one patch as the candidate, HELIX treats the **entire repository** as the evolving organism. Each mutation is a full agentic coding session running inside an isolated git worktree, so a candidate can:

1. **Read across the codebase** to understand architecture and dependencies.
2. **Edit multiple files coherently** in one mutation.
3. **Use tools mid-mutation** like tests, linters, shell commands, and web search.
4. **Take multiple turns to diagnose and self-correct** before the mutation is scored.
5. **Stay inside one evolutionary stage** rather than requiring an outer orchestration loop to get tool use or iteration.

The result is a new kind of evolutionary optimizer: one that preserves the reflective Pareto-evolutionary core while making it practical for whole repositories and realistic software engineering tasks.

### Coding agent as the mutation engine

The difference between HELIX and `/chat/completions`-style evolvers (GEPA, DSPy-Refine, ShinkaEvolve) is that HELIX's mutation is driven by a **coding agent**, not a single LLM call. A GEPA-style mutation is one prompt → one completion → apply the diff. HELIX's mutation is a full agentic session bounded only by `max_turns`:

| | GEPA / chat-completion evolvers | HELIX |
|---|---|---|
| Mutation shape | Single request/response | Multi-step agentic session |
| Working surface | A single prompt / predictor string | The entire repository in a git worktree |
| Mid-mutation introspection | None | Read any file, grep, glob, `find`, follow imports |
| Mid-mutation verification | None | Run the test suite, type-checker, linter; read failures and react |
| External information | None | Fetch the web, hit GitHub API, query package indexes live |
| Self-correction | None per proposal (retries are separate generations) | Inside one mutation: diagnose a test failure, edit another file, re-run, commit only if green |
| Cost accounting | 1 LLM call = 1 proposal | 1 proposal = N turns, gated by `max_turns` + whatever the agent decides is enough |

This is why a full solver module or a shrinkwrap of an ML kernel behave qualitatively differently than a GEPA run on the same task: HELIX's candidate is the program a team of N humans could edit over an afternoon, not a single text blob produced in one shot.

---

## ✨ Key Features

| | Feature | Description |
|---|---|---|
| 🧬 | **Whole-codebase evolution** | The candidate is your repository, not a single file, prompt, or patch |
| 📂 | **Multi-file editing** | Mutate entire directory trees — edit `auth.py:42` and `routes.py:18` in one coherent session |
| 🔁 | **Multi-turn mutations** | A single mutation can inspect, edit, test, revise, and continue before being evaluated |
| 🔧 | **Tool access during mutation** | The configured backend can read, grep, run tests, inspect the codebase, and use the web mid-mutation |
| ✅ | **Self-verification** | Mutations verify themselves by running commands before committing |
| 📊 | **Pareto frontier** | Instance-level Pareto selection across test cases — no single metric bottleneck |
| ⚡ | **Parallel evaluation** | Worktrees are isolated → parallel proposals via `ThreadPoolExecutor` (proposal concurrency bounded by `evolution.max_workers`) |
| 🔀 | **Merge / crossover** | Combine two frontier candidates that excel on different instances |
| 💾 | **State persistence & resume** | Crash-safe generation-granular resume with `helix resume` |
| 🚦 | **Gated mutations** | Train-set gating rejects regressions before Pareto evaluation |
| 📋 | **Semantic mutation log** | Full trajectory with root-cause analysis, changes made, and parent lineage |

---

## 🔄 How It Works

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '14px'}}}%%
flowchart TD
    classDef host fill:#dbeafe,stroke:#0072B2,color:#1a365d
    classDef eval fill:#d1fae5,stroke:#009E73,color:#064e3b
    classDef agent fill:#fef3c7,stroke:#E69F00,color:#78350f
    classDef artifact fill:#fce7f3,stroke:#CC79A7,color:#500724
    classDef dec fill:#ffffff,stroke:#6b7280,color:#374151
    classDef ok fill:#f0fdf4,stroke:#009E73,color:#064e3b

    %% ── Phase 1: Startup ─────────────────────────────────────────────────────────
    CFG(["📄 helix.toml"]):::host
    LOAD["Load & validate config\n⚠ ValueError if max_gen ≤ 0\nAND max_eval ≤ 0"]:::host
    RES{"resume?"}:::dec
    REC["Resume from last completed generation:\ncleanup in-flight attempts\n& orphaned worktrees"]:::host
    SEED["Evaluate seed candidate"]:::eval
    FRONT["🏆 Pareto Frontier"]:::host

    CFG --> LOAD --> RES
    RES -->|yes| REC --> SEED
    RES -->|no| SEED
    SEED --> FRONT

    %% ── Phase 2: Generation loop — select parent ─────────────────────────────────
    SEL["Select parent\n(weighted by instance wins)"]:::host
    FRONT --> SEL

    %% ── Phase 3: Docker Sandbox — isolated execution ─────────────────────────────
    subgraph DOCKER["🐳  Docker Sandbox  —  isolated execution environment"]
        direction LR
        subgraph ANET["  Agent Network  (normal egress)  "]
            AGENT["Mutation / Merge Container\n──────────────────────────\nclaude · codex · cursor\ngemini · opencode\nAuth: private candidate store"]:::agent
        end
        subgraph ENET["  Private Evaluator Network  (helix-eval-*)  "]
            ERUN["Eval Runners\n(short-lived,\ncopied workspace)"]:::eval
            ESVC["Evaluator Sidecar\n(benchmark data\nno host ports)"]:::eval
            ERUN -->|"HTTP/RPC"| ESVC
        end
    end
    style DOCKER fill:#f8fafc,stroke:#64748b,color:#0f172a

    SEL -->|"spawn mutation/merge"| AGENT
    AGENT -->|"edits synced back"| GATE

    %% ── Phase 4: Evaluate & update ───────────────────────────────────────────────
    GATE["Train-set gate\n(minibatch eval)"]:::eval
    AREC["Rejected attempt\n.helix/attempts/"]:::artifact
    VEVAL["Full val eval"]:::eval
    PERF{"all instances\nat perfect score?"}:::dec
    SREC["Perfect-skip event\n.helix/skips/"]:::artifact
    PUPD["Pareto update"]:::host
    ADV["Advance gen counter\n(unconditionally)"]:::host
    STOPDEC{"stop?"}:::dec
    DONE(["✅  Evolution complete"]):::ok

    GATE -->|"reject"| AREC
    GATE -->|"pass"| VEVAL --> PERF
    PERF -->|"perfect skip"| SREC
    PERF -->|"no"| PUPD

    AREC --> ADV
    SREC --> ADV
    PUPD --> ADV
    ADV --> STOPDEC
    STOPDEC -->|"next gen"| SEL
    STOPDEC -->|"done"| DONE

    %% ── .helix/ Artifact Store ───────────────────────────────────────────────────
    ARTS[["💾 .helix/ Artifact Store\n─────────────────────────────────────\nstate.json · lineage.json · evaluations/\nworktrees/ · attempts/ · skips/\nbackend_transcripts/‹backend›/‹session›.jsonl"]]:::artifact
    style ARTS fill:#fdf2f8,stroke:#CC79A7,color:#500724

    PUPD -.->|"persist"| ARTS
    VEVAL -.->|"persist"| ARTS
    AREC -.-> ARTS
    SREC -.-> ARTS
```

**The loop in detail:**

1. **Seed** — Your starting code is copied into a git worktree and evaluated
2. **Evaluate** — Start the private evaluator sidecar once, then run short-lived evaluator-runner containers that call it and print `HELIX_RESULT`
3. **Select** — Pick a parent from the Pareto frontier (weighted by instance wins)
4. **Mutate** — Spawn the configured agent backend in an isolated Docker workspace. It can edit candidate files and use its backend auth, but it does not join the evaluator network
5. **Gate** — Re-evaluate on the train set. Reject if the mutation caused regressions
6. **Pareto Update** — Evaluate on the val set and update the Pareto frontier
7. **Merge** — Periodically combine two complementary frontier candidates via the configured backend
8. **Cleanup** — Remove dominated worktrees; persist state; repeat

---

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/KE7/helix.git
cd helix
pip install -e .

# Verify
helix --help
```

### Initialize a Project

```bash
cd your-project/
helix init
```

This creates a `helix.toml` config file and a `.helix/` directory. Edit `helix.toml` to set your objective and evaluator.

### Recommended: Enable Docker Sandboxing

For a first HELIX run, use Docker sandboxing. It keeps mutation agents in
copied workspaces and requires a private evaluator sidecar, so agents never see
the evaluator source, benchmark data, or evaluator endpoint.

Install and start Docker, then log in to your selected backend inside its
persistent sandbox auth volume:

```bash
helix sandbox login claude      # or codex, cursor, gemini, opencode
helix sandbox status claude
```

Then enable the sandbox and configure the evaluator sidecar in `helix.toml`:

```toml
[evaluator]
command = "python /runner/evaluate_client.py"

[evaluator.sidecar]
image = "my-private-evaluator:latest"
runner_image = "my-evaluator-runner:latest"
command = "python -m benchmark_server"
endpoint = "http://helix-evaluator:8080/evaluate"
startup_timeout_seconds = 120
# API keys needed only by the scoring service; they never reach mutation agents.
passthrough_env = ["OPENAI_API_KEY"]

[sandbox]
enabled = true
network = "bridge"
skip_special_files = true
```

HELIX keeps this setting opt-in so existing local workflows and machines without
Docker continue to work, but sandboxing is the recommended mode for new projects.

### Connect The HELIX Agent Skill

This repository ships an agent skill at `skills/helix/` with detailed guidance
for writing `helix.toml`, running and debugging HELIX, and migrating GEPA
`optimize_anything.py` workflows.

For Codex, install it into your local Codex skills directory:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -sfn "$(pwd)/skills/helix" "${CODEX_HOME:-$HOME/.codex}/skills/helix"
```

Then ask Codex:

```text
Use $helix to set up this project.
```

For Claude Code, this repo includes `.claude/commands/helix.md`, so inside this
checkout you can run:

```text
/helix set up Docker-sandboxed HELIX for this benchmark
```

To use the command from another project, copy or symlink both
`skills/helix/` and `.claude/commands/helix.md` into that project, preserving
the same relative paths.

### Whole-repo-as-candidate Model

HELIX treats your **entire working tree** as the candidate. There is no `target_file` — the configured backend may read, edit, create, or delete any file in the project tree during each mutation. A minimal project layout looks like:

```
my-project/
├── helix.toml       # HELIX config (run `helix init` to generate)
├── evaluate.py      # Your evaluator script (must print a HELIX_RESULT= line)
├── solve.py         # File(s) you want to evolve (the backend will find them)
└── ...              # Any other files; HELIX will consider them too
```

To restrict what the backend touches, set `agent.background` in `helix.toml`:

```toml
[agent]
backend = "claude"
# model = "sonnet"  # optional backend-specific model
background = "Only modify files under src/. Do not edit tests/ or config/."
```

### Run Evolution

```bash
helix evolve
```

### View Results

```bash
# Show the Pareto frontier
helix frontier

# Show the best candidate
helix best

# Export best candidate to a directory
helix best --export ./best-solution

# View full mutation log
helix log
```

---

## ⚙️ Configuration

HELIX is configured via `helix.toml` in your project root.

### Minimal Example

```toml
objective = "Maximize test pass rate and code coverage"

[evaluator]
# evaluate.py must print one HELIX_RESULT=... line for HELIX to score it.
command = "uv run python evaluate.py"
```

When your evaluator needs project dependencies, make `evaluator.command` use the
same environment those dependencies are installed in. Good patterns are
`uv run python evaluate.py` or a wrapper like `bash run_eval.sh`. Avoid bare
`python3 evaluate.py` unless that interpreter already has everything your
evaluator imports.

### Full Example

```toml
# What you want the code to do better
objective = "Maximize sum of radii of 26 non-overlapping circles packed in a unit square"

# Starting directory (default: current directory)
seed = "."

# RNG seed for deterministic parent selection (default: 0)
rng_seed = 0

[env]
# Fixed non-secret env values injected into evaluator and agent subprocesses
# after passthrough_env. Useful for repeatable run-local service endpoints.
# ANTHROPIC_BASE_URL = "https://model-service.example.invalid/v1"
# ANTHROPIC_API_KEY = "dummy"

[evaluator]
command = "uv run python evaluate.py"
# The only supported parser is "helix_result".
# "helix_result" takes a per-example list matching GEPA optimize_anything's
# `tuple[float, SideInfo] | float` union — each entry is either a bare
# score or a [score, side_info] pair, mixed allowed:
#   HELIX_RESULT=[s_0, s_1, ...]                        # all bare
#   HELIX_RESULT=[[s_0, si_0], [s_1, si_1], ...]        # all rich
#   HELIX_RESULT=[s_0, [s_1, si_1], s_2, ...]           # mixed
# Positional to `helix_batch.json`. HELIX zips it into id-keyed
# `instance_scores` and stores the side_info list for the reflection
# prompt. `HELIX_RESULT` is a machine protocol; use `from helix import log`
# for human-readable evaluator notes.
include_stdout = true
include_stderr = true
extra_commands = []               # additional commands to run for context
protected_files = ["evaluate.py"] # optional extra files HELIX must keep immutable

[dataset]
# Cardinality of the train / val splits.  Used by HELIX's minibatch
# sampler to generate example ids (stringified indices by default, or
# opaque "group__N" ids when evolution.batch_sampler = "stratified")
# that the evaluator (running in the worktree) filters against its own
# dataset via helix_batch.json — written as an opaque JSON list[str].
# Leave both unset for GEPA O.A. Single-Task Search / HELIX single-task-no-example mode (dataset=None, valset=None).
# train_size = 200
# val_size  = 200

[seedless]
# Seedless mode: generate initial candidate from objective via LLM
enabled = false
# Optional prompt-grounding training dataset (used only in seedless
# seed generation).  Accepts a JSON array file, a JSONL file, or a
# directory of JSON files.  When provided, the first 3 examples are
# included in the seed-generation prompt for representative grounding.
# train_path = "puzzles/train"
# val_path   = "puzzles/val"

[evolution]
max_generations = 20
perfect_score_threshold = 1.0    # skip proposals whose instance_scores all reach this
max_evaluations = -1             # evaluation budget cap (-1 = no cap)
                                 # NOTE: at least one stopping condition is required —
                                 # max_generations must be > 0 or max_evaluations must be > 0;
                                 # setting both to ≤ 0 raises ValueError at evolution start.
merge_enabled = false            # enable merge/crossover operations
max_merge_invocations = 5        # total merge cap across entire run
merge_val_overlap_floor = 5      # minimum val-set overlap for merge candidates
merge_subsample_size = 5         # stratified val subsample size for merge acceptance (GEPA parity)
max_workers = 8                  # thread-pool cap for parent-eval + mutation pools
                                 # (default: os.cpu_count(), or 32 if that returns None)
num_parallel_proposals = 1       # parallel mutations per generation
minibatch_size = 3               # train-set minibatch size for reflective mutation
cache_evaluation = true          # reuse per-instance evaluator results
acceptance_criterion = "strict_improvement"
val_stage_size = 0               # optional first-N val gate before full val
frontier_type = "instance"       # Pareto dimensionality (GEPA FrontierType parity):
                                 # "instance" | "objective" | "hybrid" | "cartesian".
                                 # The four modes mirror GEPA optimize_anything;
                                 # "instance" is HELIX's default.
                                 # Non-instance axes use per-example
                                 # side_info["scores"] dicts. Without them,
                                 # hybrid warns/no-ops its objective axis and
                                 # continues on instances; objective/cartesian
                                 # raise MissingObjectiveScoresError at selection.

[agent]
backend = "claude"               # "claude" | "codex" | "cursor" | "gemini" | "opencode"
# model = "sonnet"               # optional backend-specific model name
effort = "medium"                # optional: "low" | "medium" | "high" | "xhigh" | "max"
max_turns = 20
# background = "Only modify files under src/. Do not touch tests/ or config/."

[sandbox]
enabled = false                  # true = run agent/evaluator subprocesses in Docker
# image = "ghcr.io/ke7/helix-evo-runner-claude:latest"  # optional; defaults from agent.backend
network = "bridge"               # "bridge" | "none" | "host"
skip_special_files = true        # skip FIFOs/sockets/devices during workspace sync
auth = "volume"                # "volume" (default) or explicit-key "env"
# Volume credentials are seeded into a fresh candidate volume; agents never
# mount the persistent helix-auth-<backend> login volume.

[worktree]
base_dir = ".helix/worktrees"
```

### Resume and parallel-batch boundaries

Resume is generation-granular. If a run is interrupted, HELIX preserves the
last completed generation and reconciles incomplete children: it avoids
corruption, duplicate budget charges, duplicate frontier entries, and orphaned
worktrees, but discards the interrupted generation's in-flight mutations.
It does not resume a partially completed proposal batch slot-by-slot. The random
number generator is seeded again for each invocation, so a resumed search need
not take the same path as an uninterrupted run.

This behaviour was exercised in a multi-hour interrupted run: budget remained
`336`, cost remained `$0.77971`, the five-member frontier and its lineage were
unchanged, three full-validation results remained available, and the resumed
generation completed an accepted gate → validate → promote cycle. This is one
interruption point only; it does not cover the apply phase or a partially
written state file.

Evaluation caps are dispatch boundaries rather than cancellation points. For
an admitted evaluation phase, `U` is the bound on uncached evaluation units
that the phase may still consume (`max_in_flight_evaluations`); HELIX records
and checks the in-flight bound, and the maximum permitted overshoot is
`max(0, U - 1)`.

For this bound, `P` = `evolution.num_parallel_proposals` is the number of parents
sampled per iteration from HELIX's frequency-weighted Pareto list; HELIX samples
them with replacement, so the same parent can occupy multiple `P` slots in one
iteration. Upstream GEPA implements this same parent-major, with-replacement
design directly as `PxNSampling(p, n)`: it samples `p` parents with replacement,
then loops `n` times per parent to build that parent's mutation tasks, drawing a
fresh minibatch for each task (upstream also ships `SameParentSampling`,
`IndependentSampling`, and a single-parent/single-mutation default strategy).
HELIX's `P*N` layout is parity with upstream's `PxNSampling`, not an extension
of it: each selected parent is reused across its `N` consecutive slots, where
`N` = `evolution.mutations_per_parent` is the number of reflective mutations
proposed per selected parent. Each slot draws its own minibatch, so siblings are gated on
different examples rather than selecting whichever sibling got an easy batch.
Thus `P*N` is the number of proposal slots (logical proposals) per
iteration. For an admitted proposal batch, let `C` be the number
of proposal contexts
actually built (`C <= P*N`). Context construction checks the budget before
each slot in the nested `P`-by-`N` loop, so budget exhaustion can stop it early
and make `C < P*N`. As a consequence of that replacement policy, the frontier
does not cap `C` by its number of distinct parents. For context `i`, let `m_i` be
that slot's sampled minibatch of training example ids; in no-example mode,
there is no minibatch, so use `|m_i| = 1` by convention and the slot
contributes `2`. Let `s` be the
selected capacity: at most `1` (`min(1, C)`) for
`proposal_selection = "best_improvement"`, `min(proposal_top_k, C)` for
`"top_k"`, and `C` for `"all_improvements"`. Finally, let
`V_stage = len(stage_val_example_ids)`
and let `V_full = len(full_val_example_ids)`, or `1` when there is no
full-validation set. The exact conservative bound computed by HELIX is:

```
U = 2 * sum(|m_i| for i in 1..C) + s * (V_stage + V_full)
```

`U` is not bounded by `P*N`: they have different units. `P*N` counts proposal
slots, while `U` counts example-evaluation units. For the sharper comparison,
the tempting closed form `2*(P*N)*m + s*(V_stage + V_full)`, where `m` is a common
minibatch size, is looser than summing admitted slots: budget exhaustion can make
`C < P*N`, and a slot's minibatch can be short. Since
`maximum_overshoot = max(0, evaluations_before + U - max_evaluations)`, inflating
`U` directly inflates the overshoot tolerated by the guard. `enforce_batch_budget_guard`
raises when actual in-flight evaluations exceed `max_in_flight_evaluations` and
again when actual overshoot exceeds `maximum_overshoot`, so substituting `P*N`
would break runs rather than merely record a rough estimate. Computing `U` from
admitted work keeps both assertions strict. In short, `P` and `N` are configured
widths; `U` is what was actually admitted at this dispatch boundary. Cache hits,
failed mutations, short final minibatches, and `best_improvement` narrowing `s`
to `1` all pull real consumption below a `P*N`-derived estimate, while the guard
exists to catch accounting regressions.

For example, with `P=2`, `N=2` (`k=4`), minibatch size `10`, `V_stage=20`,
`V_full=100`, and the default `all_improvements` selection (`s=C=4`):

```
U = 2*(4*10) + 4*(20+100) = 80 + 480 = 560   versus   P*N = 4
```

The ratio is not fixed: minibatch size, validation-set sizes, and selection mode
all change it, and none appears in `P*N`. A bound of `4` against `560` real units
would therefore fire the in-flight guard immediately.

When every example-bearing context has the same minibatch size `m`, where `m` is
the common value of `|m_i|`, the first term is `2*C*m`. The factor `2` covers
the worker's parent and child training
minibatch evaluations; in no-example mode each context contributes two
single-evaluation units. These are example-evaluation units, not proposals:
`evaluation_budget_units` charges `0` for a cached result, one unit per uncached
example in a minibatch, and `1` for a no-example evaluation. This is why
`U` depends on the admitted contexts, sampled example counts, and validation
sizes rather than only on `P` and `N`. Cache hits and failed mutations reduce
observed use below the bound.

HELIX and upstream GEPA get their proposal-stage and validation-stage
concurrency from different places. HELIX threads its `P*N` proposal slots
through a single pool bounded by `evolution.max_workers`, because each HELIX
proposal worker synchronously runs the evaluator subprocess — that pool is
HELIX's only proposal-stage concurrency mechanism. Upstream GEPA has no
equivalent engine-managed proposal-worker pool: its reflective-mutation
throughput comes from one batched call at the reflection edge (`reflect_many`,
backed by `LM.batch_complete(..., max_workers=10)`) rather than from engine
threads. For the candidates that clear acceptance, upstream batches all of them
into a single `_evaluate_programs_on_valset` call and, for adapters that
implement `batch_evaluate` (the standard `OptimizeAnythingAdapter` does), fans
every `(candidate, example)` pair across one `ThreadPoolExecutor` — so
upstream's standard full-validation path is itself parallel. (An adapter
without `batch_evaluate` falls back to sequential evaluation, and
`write_agent_state=True` forces serial evaluation.) Only the acceptance
decision and pool mutation that follow (`_add_evaluated_program`) run
sequentially in upstream, by design. HELIX, by contrast, validates each
accepted candidate sequentially, one at a time in sampled order — there is no
batched or parallel full-validation call in HELIX today. The `max_workers`
knob therefore names different layers in the two systems: HELIX's
`evolution.max_workers` bounds the proposal pool described above, while
upstream's `EngineConfig.max_workers` bounds the adapter's evaluation pool
that backs full validation. When `P*N > evolution.max_workers`, HELIX queues
excess proposal work. The logical width `P*N` therefore does not guarantee that
every evaluation runs at once; worker-pool capacity and the sequential apply
phase still bound HELIX's wall-clock parallelism.

### Choosing P versus N

The [GEPA parallel proposals analysis](https://gepa-ai.github.io/gepa/blog/2026/07/30/parallel-proposals/)
provides the underlying scaling model. In HELIX, the practical knobs are
`evolution.num_parallel_proposals` (`P`), `evolution.mutations_per_parent`
(`N`), `evolution.max_workers` (the proposal-pool cap), and
`evolution.max_evaluations` (the evaluation-budget cap). Use their code-defined
interactions to reason about a workload. In the GEPA comparison below, `V` is
the validation-set size and `W` is the worker count:

- `P*N` proposal slots are submitted to one bounded pool. The pool uses
  `max_workers = min(len(contexts), evolution.max_workers)`, so when `P*N` is
  larger than `evolution.max_workers`, excess proposal work queues instead of
  adding proposal-stage parallelism.
- HELIX's proposal executor has no explicit round barrier: it submits all
  admitted contexts and workers take queued tasks as they finish. For
  similar-duration proposal workers, a full HELIX batch therefore behaves
  roughly like `ceil(k / max_workers)` proposal waves, followed by `j`
  sequential acceptance/full-validation passes, where `j` is the number of
  candidates that clear acceptance. HELIX validates each accepted candidate
  one at a time, so raising `evolution.max_workers` speeds proposal
  generation but not the `j`-candidate validation cost. Upstream's standard
  path differs here: it batches its `j` accepted candidates into one
  `adapter.batch_evaluate` call, so its validation cost is bounded by
  `EngineConfig.max_workers` rather than by `j` sequential passes.
- The batch bound charges `2 * sum(|m_i|)` for the parent and child training
  minibatches. With `C` admitted contexts sharing minibatch size `m`, that is
  approximately `2*C*m`; increasing `P*N` therefore increases `U` and the
  permitted `max(0, U - 1)` overshoot roughly linearly against the fixed
  `evolution.max_evaluations` cap.
- The blog's analytical `k*V <= W` model assumes a parallel full-validation
  stage: up to `k` accepted candidates are each evaluated on all `V` validation
  examples in parallel across `W` workers. That describes upstream's standard
  path: the engine batches accepted candidates into one `adapter.batch_evaluate`
  call, and the standard `OptimizeAnythingAdapter` fans every
  `(candidate, example)` pair across a `ThreadPoolExecutor` bounded by
  `max_workers`. It does not describe HELIX: HELIX validates accepted
  candidates sequentially, one at a time in sampled order, with no batched or
  parallel full-validation call. The blog's speedup figures are therefore a
  genuine upper bound for HELIX specifically, not a property shared by both
  systems. For HELIX, actual proposal-stage concurrency is capped by
  `evolution.max_workers`; upstream instead bounds its adapter-side
  full-validation concurrency with its `EngineConfig.max_workers` setting.

### Docker Sandboxing

When `[sandbox].enabled = true`, HELIX starts `[evaluator.sidecar]` once per
`helix evolve` on a private internal Docker network. Agent containers run in
copied workspaces on the normal agent network and cannot reach that sidecar.
Evaluator-runner containers run only during evaluation, join the private
network, call the sidecar, print evaluator output, and exit.
The long-lived sidecar does not mount the candidate workspace; the runner must
stream the needed candidate files/data over RPC, or execute candidate code
itself and call the sidecar only for private judging/simulation.
`[evaluator.sidecar].image` is the private service image; `runner_image` is the
short-lived client image used for `[evaluator].command`. Keep private benchmark
data in the service image, not in the runner image.

Agent changes are synced back to the real candidate worktree after the backend
exits; evaluator-runner file changes are discarded. HELIX never mounts the host
project root, parent directories, or home directory by default.
`helix.toml`, `.env`, `.env.*`, `.git`, and HELIX runtime artifacts are also
excluded from sandbox workspace copies/sync-back so agents cannot read sidecar
configuration or mutate run settings.
During copy and sync, HELIX skips unsupported special files such as FIFOs,
sockets, and device nodes by default. Set `skip_special_files = false` only if
you want unsupported workspace file types to raise instead of being ignored.

`helix sandbox login <backend>` writes to a persistent operator-facing volume.
For `sandbox.auth = "volume"` (the default), HELIX seeds only the credential
files into a fresh candidate-owned volume and mounts that volume below a private
tmpfs home; no agent container mounts the login volume. Evaluator sidecar and
runner containers never receive it. Run the login flow once per backend:

```bash
helix sandbox login claude
helix sandbox status claude
```

For Claude, HELIX uses `claude setup-token` for sandbox login because that is
the flow that works cleanly in browserless Docker/SSH-style environments; it
prints a URL and then accepts the code from the browser in the terminal.
Codex similarly uses `codex login --device-auth` so the callback does not depend
on a localhost server inside the container. Gemini starts its normal
interactive CLI with `--skip-trust` so its authentication picker is not blocked
by the temporary sandbox workspace trust prompt. OpenCode starts the normal TUI
so you can choose the provider/model and complete provider login in one setup
session.

The volume names are `helix-auth-claude`, `helix-auth-codex`,
`helix-auth-cursor`, `helix-auth-gemini`, and `helix-auth-opencode`.
This avoids copying host credential stores into Docker. On macOS, Claude/Cursor
browser-login tokens may live in Keychain; on Linux they may live in
Secret Service/libsecret, GNOME Keyring, KWallet, or another desktop keyring.
Those stores are session- and OS-specific, so copying their databases into a
Linux Docker image is not a reliable authentication mechanism. If your
evaluator uses a local proxy, keep that endpoint in your evaluator code as
usual. Docker Desktop supports `host.docker.internal`; Linux users can set
`add_host_gateway = true`.

By default HELIX chooses a published backend-specific mutator image from
`agent.backend`: `ghcr.io/ke7/helix-evo-runner-claude`,
`ghcr.io/ke7/helix-evo-runner-codex`,
`ghcr.io/ke7/helix-evo-runner-cursor`,
`ghcr.io/ke7/helix-evo-runner-gemini`, or
`ghcr.io/ke7/helix-evo-runner-opencode`. To build locally instead, build the
shared base first with
`docker build -t helix-runner-base:latest -f docker/base.Dockerfile .`, then
build the backend image you need, for example
`docker build -t helix-runner-codex:latest -f docker/codex.Dockerfile .`, and
set `[sandbox].image` to that local tag.

Evaluator sidecar images are benchmark-specific and are not published by HELIX.
Use `ghcr.io/ke7/helix-evo-runner-base` or `docker/base.Dockerfile` as a base
for your own `runner_image`; build/publish the private sidecar service image
from your evaluator repository.

### Dataset Modes

HELIX splits dataset concerns across two TOML sections:

| Section | Purpose |
|---|---|
| `[dataset]` | Cardinality only — `train_size` / `val_size` — drives the minibatch sampler when the evaluator owns the dataset and HELIX hands off example ids via `helix_batch.json` (Architecture A). |
| `[seedless]` | Seedless-mode toggle + optional prompt-grounding paths (`train_path` / `val_path`) — used only during seed generation to show the LLM representative inputs. |

| Mode | Config | Description |
|---|---|---|
| **Single-task / no-example** | neither set | GEPA O.A. Single-Task Search (`dataset=None`, `valset=None`): evaluator runs without example-id handoff; uncached eval calls count as 1 metric call. |
| **Example-id handoff** | `dataset.train_size` / `dataset.val_size` set | HELIX samples example ids — stringified indices into `range(train_size)` by default, or opaque task-prefixed ids like `"group_alpha__case_3"` under `evolution.batch_sampler = "stratified"`; the evaluator reads them from `helix_batch.json` (a JSON `list[str]`) in cwd and filters its own dataset. |
| **Seedless multi-task** | `seedless.enabled = true`, `seedless.train_path` set | Seed generation prompt includes the first 3 training examples for grounding. |

HELIX does not own separate dataset files for train/val; your evaluator remains
the source of truth. During evolution HELIX sets `HELIX_SPLIT` (`train` or `val`)
so evaluator-owned datasets can switch behavior by phase, mirroring GEPA's
`trainset` / `valset` duality.

When `evolution.val_stage_size` is set to a positive value and `dataset.val_size` is also set, accepted mutation proposals run a deterministic first-N validation stage before the full validation sweep. Stage-only results are never added to the frontier; HELIX still persists only full-val results for Pareto ranking and resume stability.

### Evaluator Integrity

For non-sandboxed local prototypes, HELIX can lock evaluator-critical files so
mutations and merges cannot game the score by editing the benchmark itself.
Sandboxed runs should use `[evaluator.sidecar]` instead of repo-local evaluator
files.

```toml
[evaluator]
command = "uv run python evaluate.py"
protected_files = [
  "evaluate.py",
  "goldens.json",
  "helpers/evaluator_utils.py",
]
```

At run start, HELIX hashes the evaluator command target plus any
`evaluator.protected_files` entries and writes the manifest to
`.helix/evaluator_manifest.json`. Candidates that modify any protected file are
rejected before evaluation.

### Per-example Parallelism Inside the Evaluator

HELIX parallelises across proposals (`num_parallel_proposals`) and across
worktrees, but each evaluator invocation sees one candidate and a batch of
instance ids as a single subprocess. Per-example parallelism — evaluating
multiple ids of one candidate concurrently — lives **inside the evaluator**,
not inside HELIX's engine.

This is a deliberate architectural split: GEPA's reference adapter fans out
per-example in-process, which is essentially free; HELIX's subprocess model
would pay full subprocess-startup cost for each example. If you want N-way
parallelism per batch, your `evaluate.py` should do it directly:

```python
from concurrent.futures import ThreadPoolExecutor

instance_ids = load_batch_from_helix()   # or argv / HELIX_SPLIT path
with ThreadPoolExecutor(max_workers=4) as pool:
    results = dict(zip(instance_ids, pool.map(evaluate_one, instance_ids)))
payload = [[score, {"scores": {"accuracy": score}}] for score in results.values()]
print("HELIX_RESULT=" + json.dumps(payload))
```

Pick the worker count however you like (constant, CLI arg, derived from
`os.cpu_count()`). HELIX remains agnostic — it just consumes the per-instance
scores the evaluator returns.

### Score Parser

HELIX has one score parser: `helix_result`. Evaluators read the positional ids
from `helix_batch.json` and emit one `[score, side_info]` pair per id:

```python
print("HELIX_RESULT=" + json.dumps([
    [2.63, {"scores": {"sum_radii": 2.63}}],
]))
```

For a batch, the payload must have the same length and order as
`helix_batch.json`. `side_info` is retained for reflection; its optional
`scores` dictionary supplies named objective axes.

For GEPA-style reflective feedback, prefer structured per-example
`side_info` with the built-in result parser. For additional
human-readable notes that are not tied to one example, evaluators may call
`from helix import log` and then `log("what happened", key=value)`. HELIX
captures these notes through `HELIX_ASI_LOG`, not stdout, and renders them as
Evaluator Notes in mutation prompts.

Raw `HELIX_RESULT=...` JSON is only consumed by HELIX as a machine protocol and
is not shown to mutators. Evaluator stdout/stderr remain stored for debugging,
but successful mutation prompts omit them when structured diagnostics or
`helix.log()` notes are available; failed evaluations still include stdout and
stderr as fallback debug context.

---

## 📖 CLI Reference

| Command | Description |
|---|---|
| `helix init` | Initialize HELIX in the current directory — creates `helix.toml` and `.helix/` |
| `helix sandbox login BACKEND` | Log into an agent backend inside its persistent Docker auth volume |
| `helix sandbox status [BACKEND]` | Show sandbox login status for one backend or all supported backends |
| `helix sandbox logout BACKEND` | Log out a backend from its persistent Docker auth volume |
| `helix evolve` | Run the evolutionary loop |
| `helix frontier` | Display the current Pareto frontier as a table |
| `helix best` | Show the best candidate; `--export PATH` to copy it out |
| `helix history` | Show the candidate lineage as a tree |
| `helix resume` | Resume a previously interrupted evolution run |
| `helix clean` | Remove all worktrees and `.helix/` state (with confirmation) |
| `helix log` | Show semantic mutation log — full trajectory with parent lineage |
| `helix attempts` | Surface rejected attempt records and perfect-skip events from `.helix/attempts/` and `.helix/skips/` |

### `helix evolve` Options

```
--dir PATH          Project directory containing helix.toml (default: .)
--config PATH       Path to config file (default: helix.toml)
--objective TEXT     Override the objective string
--evaluator TEXT     Override the evaluator command
--generations INT   Override max_generations
--no-merge          Disable merge operations
--backend BACKEND   Override the mutation backend [claude|codex|cursor|gemini|opencode]
--model TEXT        Override the backend model (backend-specific naming)
--effort LEVEL      Reasoning effort: low | medium | high | xhigh | max
```

---

## 🧪 Results

### 🔵 Circle Packing

Pack 26 non-overlapping circles in a unit square, maximizing sum of radii.

| | Score | Config |
|---|:---:|---|
| Seed (naive concentric grid) | 0.9798 | — |
| **HELIX best (gen 14 of 30)** | **2.6360** | `haiku` · `low` effort · `max_turns=20` |
| GEPA optimize_anything ([blog](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)) | 2.63598+ | gemini-3-flash |

> **Note:** HELIX **matched the best published result** (2.635982 vs 2.63598+) using Claude Haiku with low reasoning effort and a 20-turn per mutation budget, and exceeded AlphaEvolve's 2.6358. See [`examples/circle_packing/`](examples/circle_packing/) for the full fixture including `solve_optimized.py` (the best evolved solution).

---

## 🏗️ Architecture

```
.helix/
├── config.toml          # Snapshot of helix.toml at run start
├── evaluator_manifest.json # Protected evaluator file hashes
├── state.json           # Generation, frontier, budget
├── lineage.json         # Full ancestry graph
├── log/                 # Semantic mutation logs
│   ├── g1-m0.json
│   └── g2-x0.json
├── worktrees/
│   ├── g0-s0/           # Seed
│   ├── g1-m1/           # Gen 1 Mutation 1
│   └── g2-x1/           # Gen 2 Merge 1
├── evaluations/
│   └── g0-s0.json       # EvalResult per candidate
├── attempts/            # Per-rejected-candidate attempt records (JSON)
└── skips/               # Per-generation perfect-skip event lists (JSON)
```

### Module Overview

| Module | Role |
|---|---|
| `cli.py` | Click CLI — init, evolve, frontier, best, history, resume, clean, log, attempts |
| `config.py` | TOML config parsing via Pydantic v2 |
| `evolution.py` | Main generation loop with gating, merge, and termination on `max_generations` / `max_evaluations` |
| `population.py` | `Candidate`, `EvalResult`, `ParetoFrontier` |
| `worktree.py` | Git worktree lifecycle (create, clone, snapshot, remove) |
| `executor.py` | Run evaluator commands |
| `evaluator_manifest.py` | SHA-256 manifest for protected evaluator files; refresh helpers for mutation and merge candidates |
| `mutator.py` | Backend mutation invocation with autonomous system prompt and HELIX usage artifacts |
| `merger.py` | Backend merge/crossover between complementary candidates |
| `lineage.py` | Ancestry graph tracking |
| `state.py` | Atomic state persistence and resume |
| `display.py` | Rich terminal UI with phase tracking |

---

## 📚 Citation

```bibtex
@software{helix2026,
  title={HELIX: Hierarchical Evolution via LLM-Informed eXploration},
  author={Elmaaroufi, Karim and OMAR},
  year={2026},
  url={https://github.com/KE7/helix}
}
```

---

## 📄 License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

HELIX's core evolutionary algorithm is based on **GEPA optimize_anything** by Agrawal, Lee, Ma, Elmaaroufi, Tan, Seshia, Sen, Klein, Stoica, Gonzalez, Khattab, Dimakis, and Zaharia. Their work on applying reflective Pareto evolution to any text made HELIX possible — we extended their algorithm to full codebases and agentic mutation but the foundation is theirs.

- **[GEPA optimize_anything](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)** — The algorithmic foundation: minibatch-gated Pareto evolution with reflective LLM mutation
- **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** — Supported HELIX mutation backend
- **[Codex CLI](https://developers.openai.com/codex/cli)** — Supported HELIX mutation backend
- **[Cursor CLI](https://cursor.com/cli/)** — Supported HELIX mutation backend
- **[Gemini CLI](https://github.com/google-gemini/gemini-cli)** — Supported HELIX mutation backend
- **[OpenCode](https://opencode.ai/docs/cli/)** — Supported HELIX mutation backend
- **[OMAR](http://omar.tech/)** — The multi-agent orchestration system used to build HELIX

```bibtex
@article{gepa_optimize_anything2026,
  title={Introducing optimize\_anything},
  author={Agrawal, Lakshya A and Lee, Donghyun and Ma, Wenjie and Elmaaroufi, Karim and Tan, Shangyin and Seshia, Sanjit A. and Sen, Koushik and Klein, Dan and Stoica, Ion and Gonzalez, Joseph E. and Khattab, Omar and Dimakis, Alexandros G. and Zaharia, Matei},
  year={2026},
  url={https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/}
}
```

---

<div align="center">
<sub>Built with 🧬 by evolution, for evolution.</sub>
</div>
