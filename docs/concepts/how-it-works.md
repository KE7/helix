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

A coding agent on its own makes one attempt and stops when it thinks it is done. HELIX runs many attempts against
your evaluator, keeps only children that measurably beat their parents, and keeps every candidate that is the best on
some instance, so a fix for one case is not lost to a regression on another. You get a search with a scorer you
control, and the result is plain code you can read.

Each candidate is the whole repository, not one file or one prompt, and each mutation is a full coding-agent session.
The agent can read the code, change several files, run your tests and its own scripts, and fix its mistakes before
HELIX scores the result. That suits changes that need exploring a real codebase, not a one-line edit.

## How HELIX differs from GEPA's optimize_anything

HELIX's search loop (minibatch-gated acceptance, instance-level Pareto parent selection, and merge) is ported from
GEPA `optimize_anything` ([Agrawal et al., 2026](https://doi.org/10.1145/3786335.3813167); see
[Acknowledgments](https://github.com/KE7/helix#acknowledgments)). In HELIX:

- The candidate is the whole working tree. There is no `target_file`, and a mutation may create, edit, or delete files.
- Each mutation is a multi-turn session of a coding-agent CLI (`agy`, `claude`, `codex`, `cursor`, `opencode`, or
  `pi`) that can read, edit, and run code in the candidate's worktree before HELIX scores it.
- Each candidate lives in its own git worktree, so proposals run in parallel (`evolution.max_workers`).
- An optional Docker sandbox (`[sandbox].enabled = true`) runs the agents in copies of the workspace. An optional
  evaluator sidecar (`[sandbox].evaluator = true` with `[evaluator.sidecar]`) runs evaluation on a private Docker
  network that agent containers cannot reach.

## One generation, step by step

![The HELIX search loop: pick a parent from the instance-level Pareto frontier, let a coding agent edit it over many turns in its own worktree, gate the child against its parent on a training minibatch, score it on the full validation set, and add it to the frontier. Both scoring steps use a protected evaluator.](../assets/helix-loop-light.svg#only-light)
![The HELIX search loop](../assets/helix-loop-dark.svg#only-dark)

Before the first generation, HELIX copies your repository into a worktree as the seed and scores it on the
validation set. Then each generation:

1. **Select.** HELIX picks a parent from the Pareto frontier, favouring candidates that are the best on more
   instances.
2. **Mutate.** The agent edits a copy of the parent in its own worktree. Its task prompt holds your objective, the
   parent's scores and the evaluator's feedback on a training minibatch.
3. **Gate.** HELIX scores the child on the same minibatch. A child that does not beat its parent there is rejected.
4. **Validate.** A child that passes is scored on the validation set and joins the frontier if it is the best on at
   least one instance.
5. **Merge** (when `evolution.merge_enabled` is on). An agent combines two frontier candidates that do well on
   different instances. The merge agent sees each parent's diff, without protected files, scores or validation ids. A
   merge that changes nothing, or fails a check on a sample of validation instances, is rejected. One that passes is
   scored on the full validation set even if that goes past `max_evaluations`, as in GEPA.

HELIX saves its state after each accepted candidate, so `helix resume` continues an interrupted run. Rejected
candidates' worktrees are removed (`evolution.retain_rejected_worktrees` keeps them), and `helix log` lists every
proposal with its parent, outcome, score and what the agent changed.
