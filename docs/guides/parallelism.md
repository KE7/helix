# Parallelism and resume

## Running several proposals per generation

Two settings decide how many proposals a generation makes:

- `evolution.num_parallel_proposals` (`P`): how many parents HELIX picks from the frontier. It picks them with
  replacement, so a strong parent can be picked more than once.
- `evolution.mutations_per_parent` (`N`): how many children each picked parent gets.

A generation makes `P × N` proposals, as in GEPA's `PxNSampling`. Each proposal is gated on its own training
minibatch, so siblings are not all judged on the same easy or hard batch.

`evolution.max_workers` caps how many proposals run at once (default: the number of CPUs). Proposals beyond it wait
for a free worker. A proposal's worker runs the agent and both minibatch evaluations, so this is also how many
evaluator processes can run at once.

`evolution.proposal_selection` decides which of the proposals that pass the gate go on to validation:
`all_improvements` (the default) sends all of them, `best_improvement` the one that improved most, and `top_k` the
`proposal_top_k` that improved most. Ties go to the earlier proposal, so the outcome does not depend on which worker
finished first.

HELIX validates the chosen candidates one at a time. Raising `max_workers` makes the proposal stage faster, but not
validation. The [GEPA analysis of parallel proposals](https://gepa-ai.github.io/gepa/blog/2026/07/30/parallel-proposals/)
assumes validation runs in parallel too, so its speedups are an upper bound for HELIX.

## How far a run can go past `max_evaluations`

HELIX checks `evolution.max_evaluations` before it starts a generation's proposals, and does not cancel work already
started. A generation that starts just under the cap can finish over it, by at most the evaluations that generation
can use:

- two per training instance in each proposal's minibatch (the parent's score and the child's), and
- for each candidate that goes on to validation, its validation set, plus the `evolution.val_stage_size` first stage if
  you set one.

For example, with `P = 2`, `N = 2`, a minibatch of 10, `proposal_selection = "all_improvements"` and 100 validation
instances, a generation can use up to 2 × 4 × 10 + 4 × 100 = 480 evaluations. Cached results, failed mutations and
proposals that fail the gate all use fewer.

A merge that passes its check on a sample of validation instances is scored on the full validation set even if that
goes past the cap, as in GEPA.

## Resuming an interrupted run

HELIX saves `.helix/state.json` as soon as a candidate joins the frontier, so an accepted candidate survives an
interrupt even if its generation does not finish. `helix resume` keeps:

- candidates on the saved frontier, and
- rejected candidates recorded under `.helix/attempts/`, including worktrees kept by
  `evolution.retain_rejected_worktrees`, which are never scored again.

Every other proposal from the interrupted generation is removed, with its worktree, lineage entry and evaluation, and
the generation runs again. HELIX reseeds its random number generator on each invocation, so a resumed run can take a
different path than an uninterrupted one would have.

## Evaluating instances in parallel

HELIX runs your evaluator once per batch of instance ids. To score the instances of one batch in parallel, do it
inside your evaluator. Starting a process per instance from HELIX would pay the startup cost every time.

```python
import json
from concurrent.futures import ThreadPoolExecutor

ids = json.load(open("helix_batch.json"))
with ThreadPoolExecutor(max_workers=4) as pool:
    scores = list(pool.map(evaluate_one, ids))   # evaluate_one is your code
print("HELIX_RESULT=" + json.dumps([[s, {"scores": {"accuracy": s}}] for s in scores]))
```

`pool.map` returns results in the order of `ids`, which is the order HELIX expects.
