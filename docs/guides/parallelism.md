# Parallelism and resume

## Resume and parallel-batch boundaries

Resume is generation-granular. If a run is interrupted, HELIX preserves the
last completed generation and reconciles incomplete children: it avoids
corruption, duplicate budget charges, duplicate frontier entries, and orphaned
worktrees, but discards the interrupted generation's in-flight mutations.
It does not resume a partially completed proposal batch slot-by-slot. The random
number generator is seeded again for each invocation, so a resumed search need
not take the same path as an uninterrupted run.

`tests/unit/test_resume.py` covers saving state when a run is interrupted,
resuming when candidate worktrees are missing, and the configuration checks a
resume performs.

Evaluation caps are dispatch boundaries rather than cancellation points. For
an admitted evaluation phase, `U` is the bound on uncached evaluation units
that the phase may still consume (`max_in_flight_evaluations`); HELIX records
and checks the in-flight bound, and the maximum permitted overshoot is
`max(0, U - 1)`. A merge is not cancelled either: one that passes its
validation-subsample gate also gets its full validation and joins the
frontier, as in GEPA, even when those evaluations cross the cap, which is
checked again at the start of the next iteration.

For this bound, `P` = `evolution.num_parallel_proposals` is the number of parents
sampled per iteration from HELIX's frequency-weighted Pareto list; HELIX samples
them with replacement, so the same parent can occupy multiple `P` slots in one
iteration. This parent-major, with-replacement `P*N` layout follows GEPA's
`PxNSampling(p, n)`: each selected parent is reused across its `N` consecutive
slots, where `N` = `evolution.mutations_per_parent` is the number of reflective
mutations proposed per selected parent. Each slot draws its own minibatch, so siblings are gated on
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

HELIX threads its `P*N` proposal slots through a single pool bounded by
`evolution.max_workers`, because each HELIX proposal worker synchronously runs
the evaluator subprocess. HELIX validates each accepted candidate sequentially,
one at a time in sampled order. When `P*N > evolution.max_workers`, HELIX queues
excess proposal work. The logical width `P*N` therefore does not guarantee that
every evaluation runs at once; worker-pool capacity and the sequential apply
phase still bound HELIX's wall-clock parallelism.

## Choosing P versus N

The [GEPA parallel proposals analysis](https://gepa-ai.github.io/gepa/blog/2026/07/30/parallel-proposals/)
provides the underlying scaling model. In HELIX, the practical knobs are
`evolution.num_parallel_proposals` (`P`), `evolution.mutations_per_parent`
(`N`), `evolution.max_workers` (the proposal-pool cap), and
`evolution.max_evaluations` (the evaluation-budget cap). Use their code-defined
interactions to reason about a workload. Below, `V` is the validation-set size
and `W` is the worker count:

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
  generation but not the `j`-candidate validation cost.
- The batch bound charges `2 * sum(|m_i|)` for the parent and child training
  minibatches. With `C` admitted contexts sharing minibatch size `m`, that is
  approximately `2*C*m`; increasing `P*N` therefore increases `U` and the
  permitted `max(0, U - 1)` overshoot roughly linearly against the fixed
  `evolution.max_evaluations` cap.
- The blog's analytical `k*V <= W` model assumes a parallel full-validation
  stage: up to `k` accepted candidates are each evaluated on all `V` validation
  examples in parallel across `W` workers. HELIX validates accepted candidates
  sequentially, one at a time in sampled order, so for HELIX the model's
  speedup figures are an upper bound. HELIX's proposal-stage concurrency is
  capped by `evolution.max_workers`.

## Per-example Parallelism Inside the Evaluator

HELIX parallelises across proposals (`num_parallel_proposals`) and across
worktrees, but each evaluator invocation sees one candidate and a batch of
instance ids as a single subprocess. Per-example parallelism — evaluating
multiple ids of one candidate concurrently — lives **inside the evaluator**,
not inside HELIX's engine.

HELIX runs one evaluator subprocess per batch, so fanning out per example in
HELIX would pay subprocess-startup cost for each example. If you want N-way
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
