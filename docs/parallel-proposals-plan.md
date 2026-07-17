# HELIX Parallel Proposals Work Plan

## Objective

Add GEPA-style P-by-N proposal batches to HELIX while preserving HELIX's
worktree, coding-agent, evaluator, Docker sandbox, budget, cache, lineage, and
resume guarantees.

- `P` is the number of parent-selection attempts from the frontier per
  optimization step.
- `N` is the number of independently sampled reflective mutations for each
  selected parent.
- A full step plans at most `P * N` child proposals.
- Parent selection is with replacement. GEPA PR #330 does not guarantee that
  the `P` selected candidate IDs are distinct.
- Reflection failures, missing diagnostics, perfect-parent skips, and security
  rejection can reduce the number of evaluated children below `P * N`.

## Ground Truth

### GEPA PR #314

- PR: <https://github.com/gepa-ai/gepa/pull/314>
- Merge-base: `85a2d428e549aec0460f427cfd0d160178f2dbbf`
- Head: `0c4213f9c5f227a717f4943e7a62f54b7513166b`
- It added `num_parallel_proposals` and a three-phase lifecycle:
  sequential prepare, threaded execute, sequential apply.
- It represented only a flat proposal width. It did not model parent groups or
  N sibling mutations for one selected parent.
- Its final aggregate diff did not add tests under `tests/`.

### GEPA PR #330

- PR: <https://github.com/gepa-ai/gepa/pull/330>
- Merge-base: `234970ac898b1c22a9012a15c72948f498994ebd`
- Head: `b86c64a26a1e0f618324e2c9f433f83e7291c848`
- Commit `71f7b4bcbf5f7d48d2f7d421645b6424d3fd4a31` explicitly reverts
  PR #314's proposal-level `ThreadPoolExecutor` design.
- It adds `ProposalTask`, `SingleMutationSampling`, `SameParentSampling`,
  `IndependentSampling`, `PxNSampling`, `AllImprovements`,
  `BestImprovement`, `TopKImprovements`, and optional adapter
  `batch_evaluate` support.
- The final PR path samples tasks, deduplicates and batch-evaluates parents,
  reflects per task, batch-evaluates surviving children, selects improvements,
  then performs full validation and state insertion sequentially.
- The final PR has 11 focused tests, not the 14 stated in its description.
  Its full branch validation was 487 passed, 3 skipped, with Pyright clean.
- Its default batch evaluator is sequential. True cross-candidate overlap is
  delegated to an adapter or custom batch evaluator.
- Reflection requests in the final PR are sequential, despite the later blog's
  description of one model batch.
- Budget checks remain at the iteration boundary, so a batch can overshoot the
  metric-call limit.

### GEPA Blog Contract

- Blog: <https://jialin-blog.gepa-docs-staging.pages.dev/blog/2026/06/30/parallel-proposals/>
- Source commit: `0b82286307a1a0c0555b82cd912d6a600f61c99e`
- The article defines `P` parents and `N` mutations per parent, for `P * N`
  planned children per step.
- The article's throughput results and model/evaluator batching claims describe
  the later v0.1.4 design and are not all implemented or tested by PR #330.

### Current HELIX

- Branch `feat/parallel-proposals` starts at
  `394a1b7dbbfbcb8362449c01661584cdec626f87`, the same commit as `main`
  and `origin/main` when the branch was created.
- `EvolutionConfig.num_parallel_proposals` is a flat width. The evolution loop
  calls `frontier.select_parent()` once per slot and creates one mutation for
  that selected parent. Its observable shape is `P=K, N=1`, not general P*N.
- Current atomic workers already overlap parent evaluation, agent mutation,
  tamper detection, and child minibatch evaluation with a bounded
  `ThreadPoolExecutor`. Results are restored to sampled order before sequential
  state mutation.
- Each mutation has a candidate-specific Git worktree. Sandboxed coding-agent
  and evaluator commands use candidate-specific `/workspace` mounts and Docker
  `--rm`, so bounded concurrent containers are the natural HELIX realization
  of proposal batching.
- OpenCode already gets a candidate-specific state directory, preventing SQLite
  WAL contention between concurrent proposal workers.
- No GEPA package is a HELIX dependency. Parity must be implemented against
  HELIX's own contracts rather than by importing GEPA.
- Current validation baseline: 867 unit tests pass and strict mypy passes for
  29 source files. `ruff check src/` currently reports seven pre-existing F401
  findings in `src/helix/evolution.py`; CI does not run Ruff.
- The current Pydantic model accepts invalid zero/negative values for
  `max_workers` and `num_parallel_proposals`; the new design must close that
  gap.

## Required Semantics

1. One step selects `P` parent slots in deterministic frontier RNG order.
2. Each selected slot samples `N` minibatches and reserves `N` stable candidate
   IDs before any concurrent work begins.
3. Task order is parent-major: `(p0,n0)`, `(p0,n1)`, ..., `(p1,n0)`, ... .
4. Completion order never changes IDs, lineage order, selection tie-breaking,
   logs, or persisted state.
5. Parent and child evaluations use one cross-candidate batch abstraction. The
   Docker implementation may realize that batch as multiple bounded container
   calls; it must preserve positional result alignment.
6. Proposal generation uses one isolated worktree/container per child and is
   bounded by the same global worker limit.
7. Selection runs only after all available child minibatch evaluations return.
8. Full-validation work for selected proposals may run concurrently, but
   frontier, lineage, budget, trace, cache, and state updates are applied in
   deterministic selected order.
9. `P=1, N=1` preserves existing behavior.
10. A failure in one proposal slot does not discard successful siblings unless
    the error is an existing run-fatal security/configuration error.

## Configuration Design

Use one P*N scheduler and extend the existing flat `[evolution]` contract:

```toml
[evolution]
num_parallel_proposals = 2       # P, parent-selection attempts
mutations_per_parent = 2         # N, mutations for each selected parent
proposal_selection = "all_improvements"
# proposal_top_k = 2             # required only for selection = "top_k"
```

The existing field becomes P because that is what the current loop already
does: it selects one parent for each of K slots. N defaults to one. Therefore
every existing `num_parallel_proposals=K` configuration is exactly the K-by-1
case of the new scheduler; there is no legacy runtime branch, adapter, or
deprecation path.

Rules:

- Defaults are `num_parallel_proposals=1`, `mutations_per_parent=1`, and
  `proposal_selection="all_improvements"`.
- Existing `num_parallel_proposals="auto"` resolution produces P and still
  lands on P-by-1 when N is omitted.
- Supported selection values are `all_improvements`, `best_improvement`, and
  `top_k`.
- `proposal_top_k` must be within `1..P*N`; it is rejected for other selection
  modes.
- `max_workers`, integer P, N, and K must all be at least one.
- `P*N` may exceed `max_workers`; tasks queue behind the worker bound.

Internally define strategy protocols and built-ins, but keep TOML declarative.
P*N sampling covers the three useful shapes without extra user-facing modes:
`1x1` single, `1xN` same-parent, and `Px1` independent.

## Baby-Step Implementation

### Step 1: Lock the configuration contract

Files:

- `src/helix/config.py`
- `tests/unit/test_config_new_fields.py`
- `skills/helix/references/toml.md`

Work:

- Add N and selection fields to `EvolutionConfig` and validate the effective
  P*N configuration.
- Treat the existing `num_parallel_proposals` value as P in the unified
  scheduler; N defaults to one.
- Reject invalid zero/negative P, N, `top_k`, and `max_workers` values.
- Document that old K-slot runs are the K-by-1 case.

Gate: configuration tests and strict mypy pass before runtime changes.

### Step 2: Introduce typed proposal tasks and strategies

Files:

- new `src/helix/proposals.py`
- new `tests/unit/test_proposals.py`

Work:

- Add immutable `ProposalTask` with batch index, parent group index, mutation
  index, parent candidate, minibatch IDs, and reserved child ID.
- Add typed proposal outcomes for skipped, failed, tampered, evaluated, and
  selected states.
- Add P*N sampling and three selection implementations.
- Preserve stable input order and stable first-on-tie behavior.

Gate: all 11 GEPA PR #330 strategy/fallback/default tests are represented in
HELIX, adapted to HELIX candidate and score types.

### Step 3: Extract an ordered cross-candidate evaluation batch

Files:

- `src/helix/executor.py`
- `src/helix/eval_cache.py`
- `tests/unit/test_executor.py`
- `tests/unit/test_eval_cache.py`

Work:

- Add a typed batch item and `run_evaluator_batch` API.
- Bound execution by `max_workers`, store results by input index, and validate
  exact result cardinality.
- Deduplicate only identical candidate-content plus identical split/minibatch
  keys. Do not deduplicate by candidate ID alone.
- Return per-item failures without losing sibling results; preserve existing
  fatal exception classes.
- Keep candidate-specific `helix_batch.json` writes behind the existing
  worktree lock.

Gate: reverse completion order still yields input-order results and exact
budget/cache accounting.

### Step 4: Replace flat context preparation with P*N planning

Files:

- `src/helix/evolution.py`
- `src/helix/budget.py`
- `src/helix/batch_sampler.py`
- `tests/unit/test_evolution_minibatch.py`

Work:

- Select P parent slots sequentially.
- For each slot, sample N minibatches and reserve N IDs sequentially.
- Snapshot all read-only task context before workers start.
- Use the new parent evaluation batch and preserve the perfect-parent skip.
- Keep the current default and K-by-1 behavior byte-for-byte where
  persistence formats permit.

Gate: `P=2,N=3` produces six tasks, exactly two parent selections, parent
mapping `[A,A,A,B,B,B]`, six stable IDs, and three minibatches per group.

### Step 5: Run isolated mutation workers as one bounded batch

Files:

- `src/helix/evolution.py`
- `src/helix/mutator.py`
- `src/helix/worktree.py`
- `src/helix/sandbox.py`
- `tests/unit/test_mutator.py`
- `tests/unit/test_sandbox.py`

Work:

- Reuse HELIX's candidate-specific worktree and Docker container isolation.
- Submit surviving mutation tasks to a bounded pool and restore sampled order.
- Retain per-candidate OpenCode state isolation.
- Preserve per-slot handling for rate limits, agent failures, tamper rejection,
  worktree cleanup, and usage capture.
- Do not claim literal model-API batching: current HELIX backends are coding
  CLI subprocesses. The supported batch implementation is bounded concurrent
  isolated agent containers.

Gate: two mutation tasks are simultaneously active, use different worktrees
and state directories, and leave no test containers behind.

### Step 6: Batch child scoring, select, then apply deterministically

Files:

- `src/helix/evolution.py`
- `src/helix/population.py`
- `src/helix/eval_policy.py`
- `tests/unit/test_evolution_minibatch.py`

Work:

- Batch child minibatch evaluation.
- Build ordered proposal records with before/after scores and parent metadata.
- Run the configured selection strategy once. Avoid GEPA PR #330's double call
  to potentially stateful acceptance criteria.
- Batch full-validation evaluation for selected children when safe.
- Apply successful results sequentially in selected order.
- Clean every unselected, rejected, failed, and tampered child worktree.

Gate: completion-order permutations produce identical frontier, lineage,
candidate IDs, budget, and selected results.

### Step 7: Make persistence and observability batch-aware

Files:

- `src/helix/state.py`
- `src/helix/evolution.py`
- `src/helix/trace.py`
- `src/helix/display.py`
- `tests/unit/test_resume.py`
- `tests/unit/test_state_persistence.py`
- `tests/unit/test_display.py`

Work:

- Persist per-task records rather than GEPA PR #330's first-task-only summary.
- Record batch ID, P, N, task index, parent group, mutation index, child ID,
  status, score delta, selection result, and cleanup result.
- Save before task execution and after deterministic application.
- Extend interrupted-batch reconciliation to all planned worktrees and IDs.
- Define batch budget semantics explicitly: check before dispatch, account all
  completed in-flight work, and document/test the maximum permitted overshoot.

Gate: interruption after a subset of workers completes can resume without ID
collision, double charge, orphan worktree, or duplicate frontier insertion.

### Step 8: Documentation, migration, and release validation

Files:

- `README.md`
- `skills/helix/references/toml.md`
- `skills/helix/references/gepa-migration.md`
- `CHANGELOG.md`
- `.github/workflows/ci.yml` or a new Docker integration workflow

Work:

- Document P and N with `1x1`, `1xN`, `Px1`, and `PxN` examples.
- Explain that parents are selected with replacement.
- Explain worker caps, Docker resource usage, budget acceleration/overshoot,
  selection, and how existing K-by-1 configurations extend to P*N.
- Add an opt-in Docker-daemon integration job with a small local fixture image.
- Keep the default at single mutation.

Gate: clean install, unit suite, strict typing, and daemon-backed Docker smoke
all pass on Python 3.11 and 3.12.

## Test Matrix

### GEPA PR #330 parity tests

1. Single mutation creates one task.
2. Same-parent sampling creates N tasks for one selected parent with N sampled
   minibatches.
3. Independent sampling calls parent selection P times.
4. P*N sampling with `P=2,N=3` creates six tasks and two parent selections.
5. All-improvements keeps every proposal passing acceptance.
6. Best-improvement keeps the largest score delta.
7. Best-improvement returns empty when none pass.
8. Top-K returns the K largest passing deltas in stable order.
9. Default evaluation-batch fallback preserves count, order, and call count.
10. Default sampling is `1x1`.
11. Default selection is all-improvements.

### HELIX additions

- Validation: zero/negative P, N, K, and `max_workers`; `"auto"` resolution;
  invalid `top_k` combinations.
- Shape: compare `1x4` and `4x1`; equal child count but different selector calls
  and parent grouping.
- Parent replacement: duplicate parent selections remain two groups and do not
  collapse N siblings incorrectly.
- True overlap: use `threading.Barrier`/`Event`, not sleeps alone, to prove
  simultaneous mutation and distinct-worktree child evaluation.
- Worker cap: peak active tasks is both at least two when capacity permits and
  never above `max_workers`.
- Ordering: force reverse completion and assert stable IDs, lineage, selection
  ties, persisted task order, and logs.
- Deduplication: identical parent-content/minibatch work executes once; different
  minibatches or content do not deduplicate.
- Cardinality: malformed batch result lengths fail clearly before state apply.
- Partial failure: parent eval, mutation, child eval, rate limit, and container
  failure in one slot do not discard successful siblings.
- Security: tampered children are never snapshotted/selected and their worktrees
  are removed.
- Selection cleanup: unselected and acceptance-rejected children are persisted
  as attempts and cleaned.
- Budget: exact parent/child/full-val charges, cache hits, in-flight overshoot
  bound, and early truncation when the batch cannot start.
- Cache: concurrent reads/writes and deduplicated results remain deterministic.
- State: all-improvements can insert multiple children without frontier races.
- Determinism: repeated runs with the same seed match despite different worker
  completion schedules.
- Resume: interruption before dispatch, mid-worker, after evaluation, and
  mid-apply produces no double charge, duplicate lineage, ID collision, or
  orphan worktree.
- Trace/display: every P*N slot has a terminal status; no first-task-only
  observability.
- Docker unit seam: candidate-specific mounts, sidecar visibility to workers,
  `--rm`, and per-proposal OpenCode state.
- Docker integration: two real sandbox containers overlap, produce isolated
  outputs, and leave no containers behind.
- Compatibility: omitted N and explicit `N=1` produce the same K-by-1 trace,
  lineage, budget, frontier, and persisted state for every existing K value.

## Validation Commands

```bash
uv run pytest tests/unit/test_proposals.py -q
uv run pytest tests/unit/test_config_new_fields.py \
  tests/unit/test_evolution_minibatch.py \
  tests/unit/test_executor.py tests/unit/test_eval_cache.py \
  tests/unit/test_mutator.py tests/unit/test_sandbox.py \
  tests/unit/test_resume.py tests/unit/test_state_persistence.py -q
uv run pytest tests/unit/ -q
uv run mypy --strict src/helix/
uv run ruff check src/ tests/unit/test_proposals.py
```

The Ruff gate should distinguish new findings from the seven pre-existing F401
findings in `src/helix/evolution.py`, or those findings should be fixed in a
separate preparatory commit.

For Docker integration, add a marked test and run it only when `docker info`
succeeds:

```bash
uv run pytest -m docker_integration tests/integration/test_parallel_sandbox.py -q
```

## Definition of Done

- Users can express P and N independently in TOML.
- `P=2,N=3` demonstrably plans six stable child slots from two parent-selection
  attempts.
- Concurrent coding-agent/evaluator containers are globally bounded and
  isolated.
- Sampling, selection, evaluation, persistence, and cleanup are deterministic
  regardless of worker completion order.
- Per-slot failure does not corrupt sibling results or global state.
- Budget and resume semantics are documented and tested.
- All 11 GEPA PR #330 parity tests and the HELIX-specific matrix pass.
- The existing 867-test baseline does not regress.
- Strict mypy and daemon-backed Docker smoke pass.
