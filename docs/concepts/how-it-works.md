# How HELIX works

## Terms

| Term | Meaning |
|---|---|
| **Candidate** | One version of your repository. HELIX scores and evolves candidates. |
| **Seed** | The starting candidate: your repo as it is when you run `helix evolve`. |
| **Worktree** | The git worktree under `.helix/worktrees/` that holds one candidate. Your own checkout is never modified. |
| **Instance** (or example) | One case your evaluator scores, such as a test, a benchmark problem or a simulation trial, named by an id in `helix_batch.json`. |
| **Generation** | One round of the loop: pick a parent, mutate it, gate the child, validate it. |
| **Mutation** | One coding-agent session that edits a copy of the parent into a child candidate. |
| **Minibatch** | A few training instances the child and parent are both scored on. |
| **Gate** | The check that keeps a child only if it beats its parent on the minibatch. |
| **Pareto frontier** | The candidates that are the best on at least one validation instance. Parents are drawn from it. |
| **side_info** | The per-instance feedback your evaluator returns with each score. The next mutation reads it. |
| **Merge** | An optional step (off by default) where an agent combines two frontier candidates. |

## Why HELIX?

A coding agent on its own makes one attempt and stops when it thinks it is done. HELIX runs many attempts against your evaluator, keeps only children that measurably beat their parents, and keeps every candidate that is the best on some instance, so a fix for one case is not lost to a regression on another. You get a search with a scorer you control, and the result is plain code you can read.

HELIX is built for improving **real, multi-file codebases**, where useful mutations require exploration, iteration, and tooling.

HELIX treats the **entire repository** as the evolving organism, rather than one file or one patch. Each mutation is an agentic coding session running inside an isolated git worktree, so a mutation can:

1. **Read across the codebase** to understand architecture and dependencies.
2. **Edit multiple files coherently** in one mutation.
3. **Use tools mid-mutation** like tests, linters, shell commands, and, where the backend provides it, web search.
4. **Take multiple turns to diagnose and self-correct** before the mutation is scored.

### Design

HELIX's search loop (minibatch-gated acceptance, instance-level Pareto parent selection, and merge) is ported from GEPA `optimize_anything` ([Agrawal et al., 2026](https://doi.org/10.1145/3786335.3813167); see [Acknowledgments](https://github.com/KE7/helix#acknowledgments)). In HELIX:

- The candidate is the whole working tree. There is no `target_file`, and a mutation may create, edit, or delete files.
- Each mutation is a multi-turn session of a coding-agent CLI (`agy`, `claude`, `codex`, `cursor`, `opencode`, or `pi`) that can read, edit, and run code in the candidate's worktree before HELIX scores it.
- Each candidate lives in its own git worktree.
- An optional Docker sandbox (`[sandbox].enabled = true`) runs mutation and merge agents in copied workspaces. An optional evaluator sidecar (`[sandbox].enabled = true` and `[sandbox].evaluator = true`, plus `[evaluator.sidecar]`) runs evaluation on a private Docker network that agent containers cannot reach.

Meta-Harness ([Lee et al., 2026](https://arxiv.org/abs/2603.28052)) is concurrent work that uses a coding-agent proposer to optimize single-file harness programs.

## Key Features

| | Feature | Description |
|---|---|---|
| 🧬 | **Whole-codebase evolution** | The candidate is your repository, not a single file, prompt, or patch |
| 📂 | **Multi-file editing** | Mutate entire directory trees — edit `auth.py:42` and `routes.py:18` in one coherent session |
| 🔁 | **Multi-turn mutations** | A single mutation can inspect, edit, test, revise, and continue before being evaluated |
| 🔧 | **Tool access during mutation** | The configured backend can read, grep, run tests, inspect the codebase, and, where the backend provides it, use the web mid-mutation |
| ✅ | **Self-verification** | A mutation can run tests and other commands to check its changes before HELIX scores it |
| 📊 | **Pareto frontier** | Instance-level Pareto selection across test cases — no single metric bottleneck |
| ⚡ | **Parallel evaluation** | Worktrees are isolated → parallel proposals via `ThreadPoolExecutor` (proposal concurrency bounded by `evolution.max_workers`) |
| 🔀 | **Merge / crossover** | Combine two frontier candidates that excel on different instances |
| 💾 | **State persistence & resume** | Crash-safe generation-granular resume with `helix resume` |
| 🚦 | **Gated mutations** | Train-set gating rejects regressions before Pareto evaluation |
| 📋 | **Semantic mutation log** | Full trajectory with root-cause analysis, changes made, and parent lineage |

## How It Works

![The HELIX search loop: pick a parent from the instance-level Pareto frontier, let a coding agent edit it over many turns in its own worktree, gate the child against its parent on a training minibatch, score it on the full validation set, and add it to the frontier. Both scoring steps use a protected evaluator.](../assets/helix-loop-light.svg#only-light)
![The HELIX search loop](../assets/helix-loop-dark.svg#only-dark)

**The loop in detail:**

1. **Seed** — Your starting code is copied into a git worktree and evaluated
2. **Evaluate** — Run the evaluator command in the candidate worktree and parse its `HELIX_RESULT` line. With sandboxed evaluation (`[sandbox].enabled = true`, `[sandbox].evaluator = true`, and `[evaluator.sidecar]`), HELIX starts the private evaluator sidecar once per run and runs short-lived evaluator-runner containers that call it
3. **Select** — Pick a parent from the Pareto frontier (weighted by instance wins)
4. **Mutate** — Spawn the configured agent backend in the candidate's worktree. With `[sandbox].enabled = true`, the backend runs in an isolated Docker workspace, where it can edit candidate files and use its backend auth but does not join the evaluator network
5. **Gate** — Re-evaluate on the train set. Reject if the mutation caused regressions
6. **Pareto Update** — Evaluate on the val set and update the Pareto frontier
7. **Merge** — Periodically combine two complementary frontier candidates via the configured backend
8. **Cleanup** — Remove dominated worktrees; persist state; repeat
