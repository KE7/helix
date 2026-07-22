# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security — credential isolation for sandboxed mutation agents

HELIX changes how sandboxed mutation agents authenticate. Previously, a
sandboxed agent container received both a mounted backend auth volume **and**
any backend credentials present in the host environment. The environment
injection happened *after* environment scrubbing, so tests asserting on the
scrubber reported a clean environment while the credential still reached the
container. Three separate mechanisms could place a credential in a
mutation-agent container: a per-backend table in HELIX core, a wildcard
passthrough of `HELIX_`-prefixed variables, and the `passthrough_env` / `env`
configuration fields. Only the first was widely understood.

**What changed.** Sandboxed agent authentication is now explicit.
`sandbox.auth = "volume"` (the default) authenticates solely from the backend
auth volume and places no credentials in the container. `sandbox.auth = "env"`
is an explicit opt-in that injects only the variables named in
`sandbox.auth_env_allow`, and prints a non-suppressible disclosure of the
variable names and the container's network exposure. **There is no automatic
fallback between the two modes in either direction.** Non-sandboxed runs are
unchanged and continue to authenticate from the environment.

Every variable reaching a container now carries a recorded origin and an
explicit scope authorization. As a result, credentials scoped to a private
evaluator sidecar can no longer reach a mutation agent through a configuration
change elsewhere — including a change to the agent backend name, which
previously altered which credentials were injected with no other visible
signal.

**Environment credentials suppress token refresh — what is and is not
established.**

- **Established:** setting `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` turns
  OAuth mode off in the backend CLI, which prevents **container-side** token
  refresh — both the proactive-on-expiry path and the 401-triggered path.
  `CLAUDE_CODE_OAUTH_TOKEN` has the same net effect by a different mechanism.
- **Established:** with no such variable present, an expired credential record
  does reach the token-refresh request, headless and without a terminal.
  Refresh is not interactive-gated.
- **Not established:** why the previously provisioned shared credential record
  is rejected by the server. Token rotation or invalidation remains a
  hypothesis, and other causes — including a separate CLI on the host
  refreshing the same account — remain live.

Because HELIX injected these variables into every sandboxed agent container,
container-side refresh could not occur there. **Removing the injection is
therefore not only hardening: it is what makes volume-based authentication
self-sustaining inside containers**, since a container that authenticates from
the volume can now renew it. This does not by itself explain or repair the
currently-rejected record, and HELIX makes no claim that it does.

Consequently `sandbox.auth = "env"` is documented as a **tradeoff, not an
equivalent alternative**: it disables OAuth refresh in the container and will
let a mounted auth volume's token go stale. The mode announces this at
startup.

**Runtime identity.** `helix sandbox login`, `status` and `logout` previously
resolved their runner image to a default `:latest` tag rather than the image
the project pins, and could not consult the project configuration at all — so
credentials were written by a different CLI build than runs consume. These
commands now use the configured runtime identity, and refuse with an
actionable error rather than silently falling back to a default when no image
can be determined.

**Unsound signals removed.** HELIX no longer treats the presence of a
credential file, or a backend's own status text, as evidence of working
authentication. Both were observed reporting success against credentials that
a real request rejected. Authentication is now verified once per run, before
any mutation is dispatched, by a real authenticated operation using the exact
runner image and backend the run will use. Failures abort before dispatch with
a redacted, actionable message and **no proposal, budget, or run-state side
effects**, and distinguish a failed token refresh from a failed request after
a successful refresh, because those have different remedies.

**Volume lifecycle.** `docker run -v name:/path` silently creates a missing
named volume, so a mount could never fail and an unauthenticated host produced
a successful-looking container whose failure appeared mid-run. Volume
existence is now established with a side-effect-free inspection, and `helix
sandbox status` is idempotent — it no longer creates the volume it is asked to
report on. Auth volumes now record the backend and CLI version that
provisioned them, and a mismatch against the configured runner image is
reported; a missing record reports as **unknown**, which is never treated as
valid.

**Preflight side effects.** The preflight starts a real container with the
auth volume mounted read-write. The backend CLI writes non-credential state
there on startup (session files, caches, logs, a config backup), and a
successful refresh rotates the stored refresh token — which is the intended
repair path, and the reason the volume must never be probed through a copy.
The preflight makes one billable inference call, recorded separately as *auth
overhead*; it does **not** enter the evaluation budget, so budget-conservation
checks are unaffected. See `docs/sandbox-auth.md` for the safeguards.

**Silent failure in credential paths.** A backend's token-refresh path can
fail silently — logging internally, returning a negative result, and surfacing
no user-visible signal. The preflight performs a real authenticated operation
against a writable auth volume specifically so such a failure is observed
before a run begins rather than inferred from a run that degraded.

**A note on tests that defend defects.** Four tests in this codebase were
found asserting the behaviour they should have prevented: two encoded the
credential injection as expected output, one required `helix sandbox login` to
resolve a default image tag, and one asserted `status` starts a container. All
passed continuously while the properties they appeared to protect were false.
A test that encodes current behaviour without asserting the intended
*property* will faithfully defend the defect. The replacement suite asserts on
the final container argv across all three injection origins together — never
on the scrubber alone, which is structurally incapable of catching this — and
each test states the mutation it catches and is demonstrated failing when that
mutation is reintroduced.

#### Migration

- Configurations with `sandbox.enabled = false` are **unaffected**.
- Sandboxed configurations that relied on host environment credentials
  reaching the agent must now declare that explicitly: set
  `sandbox.auth = "env"` and list the variables in `sandbox.auth_env_allow`.
- Top-level `passthrough_env` no longer grants agent scope under a sandbox.
  Use `sandbox.agent_passthrough_env` for non-credential agent variables, and
  `sandbox.auth_env_allow` for credentials.
- `CLAUDE_CODE_OAUTH_TOKEN` is rejected in `sandbox.auth_env_allow`: it
  permanently disables OAuth token refresh, which would trade one problem for
  another.
- `HELIX_`-prefixed variables no longer reach a **sandboxed agent** by
  wildcard. A registry of HELIX's own names still propagates, and
  `sandbox.agent_passthrough_env` covers the rest. Evaluator and sidecar scope
  are unchanged.
- Existing auth volumes have no provenance record and report **unknown** until
  re-provisioned with `helix sandbox login <backend>`.
- `helix sandbox status` no longer creates the volume it is asked to report
  on. Scripts that relied on that must call `helix sandbox login`.
- `examples/formulacode` requires a config edit (its template is updated): it
  used top-level `passthrough_env = ["ANTHROPIC_API_KEY"]` and now declares
  `auth = "env"` with `auth_env_allow`. Other bundled examples need no edit.

## [0.3.0] - 2026-07-20

### Added
- P×N proposal batches: `num_parallel_proposals` selects P parent slots and
  the new `mutations_per_parent` creates N independently sampled children per
  slot. Parent selection is with replacement and task application remains
  deterministic regardless of worker completion order.
- Proposal selection modes `all_improvements`, `best_improvement`, and `top_k`
  (with validated `proposal_top_k`) after ordered child minibatch evaluation.
- Batch-aware state, trace, cleanup, and resume metadata for every planned
  proposal slot, including interrupted-batch reconciliation.
- Opt-in, manually dispatched Docker integration CI on Python 3.11 and 3.12 for
  the daemon-backed parallel sandbox smoke test.
- Sidecar-only `evaluator.sidecar.passthrough_env` injection for credentials
  that private evaluators need but mutation agents must never receive.

### Changed
- Existing `num_parallel_proposals = K` configurations now use the unified K×1
  scheduler. Omitting `mutations_per_parent` remains behaviorally equivalent to
  N=1; the default remains 1×1.
- Candidate evaluation and multi-turn coding-agent work use bounded concurrent
  batches rather than literal model-API batching. One global `max_workers` cap
  bounds active candidate work and Docker containers within each phase.
- Evaluation caps are checked before batch dispatch. All completed in-flight
  uncached work is charged, so an admitted phase with U uncached metric units
  can overshoot by at most `max(0, U - 1)` units.

### Fixed
- Relative `--dir` evolution paths are normalized before Git worktree
  creation, so dirty seed snapshots and later candidate subprocesses resolve
  the same worktree location.
- `helix clean` now removes a nested Git repository initialized by HELIX after
  validating paired cleanup markers, while preserving every pre-existing
  repository.

## [0.2.2] - 2026-05-13

### Added
- New CLI subcommand `helix attempts` — surfaces rejected attempt records and
  perfect-skip events; supports filtering by `--generation`, `--stage`,
  `--reason`, `--cid`; `--skips` flag to show only perfect-skip entries;
  `--json` for machine-readable output. (PR #30)
- New artifact directories `.helix/attempts/` and `.helix/skips/` — per-rejected-candidate
  JSON files and per-generation perfect-skip event lists persisted alongside
  `.helix/evaluations/` and `.helix/worktrees/`. (PR #30)
- 5-backend `tool_event_count` / `tool_names` tracking in `UsageStats` —
  normalised tool-use metrics for claude, codex, cursor, gemini, and opencode
  backends extracted from transcript artifacts. (PR #32)
- `session_id`, `to_dict`, `from_dict` helpers on `UsageStats`. (PR #32)
- New module `src/helix/evaluator_manifest.py` — SHA-256 manifest for
  protected evaluator files; refresh helpers for mutation and merge candidates.
  (PR #30 refactor)

### Changed
- **GEPA structural alignment**: generation counter advances unconditionally per
  GEPA spec — a perfect-subsample no longer rewinds the counter. Cache is also
  bypassed when re-evaluating the parent's minibatch, ensuring consistent
  parent-comparison semantics across resume scenarios. (PR #32)
- `exitcode` score parser now broadcasts the success score to *all* instance
  IDs in the batch, fixing a bug where only the first instance was populated
  when using `exitcode` with multi-example runs. (PR #32)

### Fixed
- Mandatory stop-condition enforcement: `run_evolution` now raises `ValueError`
  at startup if both `max_generations ≤ 0` and `max_evaluations ≤ 0`. Previously
  a config with both set to negative values would loop indefinitely. (PR #32)
- Resume reconciliation: incomplete attempts and orphaned worktrees from a
  crashed run are cleaned up at resume time before the first new generation
  starts. (PR #30)

## [0.2.1] - 2026-05-10

> Note: 0.2.0 was published from a broken build and is superseded by this
> release. Do not install 0.2.0; use 0.2.1 or later.

### Removed
- Orphan `helix.toml` template at the repository root.  The runtime
  template that `helix init` writes lives inline in
  `src/helix/cli.py::_HELIX_TOML_TEMPLATE`; the repo-root file was a
  stale stub no docs or code referenced.

### Packaging
- Tightened the sdist via `[tool.hatch.build.targets.sdist]`.  The
  source distribution now ships only `src/helix`, `pyproject.toml`,
  `uv.lock`, `README.md`, and `LICENSE` (mirrors GEPA's lean PyPI
  layout, minus `tests/`).  Wheel contents unchanged.

### Added
- Multi-axis Pareto frontier (GEPA `FrontierType` parity,
  `src/gepa/core/state.py:22-23`).  New
  `evolution.frontier_type: Literal["instance", "objective", "hybrid",
  "cartesian"]` with default `"hybrid"` — matches GEPA's own
  `optimize_anything` default (`src/gepa/optimize_anything.py:476`).
  `ParetoFrontier` now tracks per-objective-name and per-`(val_id,
  objective_name)` best sets alongside the existing per-example-id
  tracking, and `get_non_dominated()` / `select_parent()` dispatch on
  `frontier_type`.  The acceptance gate stays positional on
  `scores_list` unchanged.
- `EvalResult.per_example_side_info: list[dict[str, Any]] | None` —
  per-example diagnostic dicts from the new `helix_result` contract,
  positional to `instance_scores` by `helix_batch.json` id order.
  GEPA analogue: `EvaluationBatch.trajectories`
  (`src/gepa/core/adapter.py:25`).
- `EvalResult.objective_scores: list[dict[str, float]] | None` —
  per-example objective-axis harvest from `side_info["scores"]`.  GEPA
  analogue: `EvaluationBatch.objective_scores`
  (`src/gepa/core/adapter.py:26`).  Feeds `frontier_type ∈ {"objective",
  "hybrid", "cartesian"}`; harmless on the `"instance"` path.
- `DatasetConfig.train_size` / `val_size` — cardinality-only fields that drive
  the minibatch sampler when the evaluator owns the dataset (Architecture A
  example-id handoff).  HELIX writes sampled example ids to
  `{worktree}/helix_batch.json`; the evaluator filters its own dataset.
- `SeedlessConfig` — new section carrying `enabled` plus the optional
  prompt-grounding `train_path` / `val_path` used during seed generation.
- Evaluator-owned dataset integration wired into Architecture A with
  cardinality-only `train_size` / `val_size`.
- `load_config` now emits a dedicated hint for pydantic `extra_forbidden`
  validation errors, pointing users at likely typos or misplaced keys.

### Changed
- **BREAKING**: `score_parser = "helix_result"` now rejects malformed
  `side_info["scores"]` payloads instead of silently dropping fields.
  Non-dict `"scores"` values, non-string objective names, and non-finite
  / non-numeric objective values now raise `EvaluatorError` at parse
  time.  Previously these were dropped (with a logged warning at most),
  which let evaluators stuff arbitrary diagnostics under `"scores"` —
  including pairwise / Bradley-Terry payloads — and then misread the
  result as scalar objective semantics.  Pairwise / BT payloads are
  not implemented in HELIX yet; emit them under a different
  `side_info` key.  Migration: ensure `side_info["scores"]` is a
  `dict[str, float|int|bool]` with finite numeric values, or omit the
  key entirely.
- **BREAKING**: Non-`"instance"` `evolution.frontier_type` values now
  fail loudly when an `EvalResult` lacks per-example `objective_scores`,
  has length-mismatched `objective_scores`, or has all-empty objective
  slots.  `ParetoFrontier.add()` raises the new
  `MissingObjectiveScoresError` instead of silently degenerating to
  scalar / instance-axis behaviour.  `select_parent()` likewise raises
  rather than falling back to all-candidate sampling on objective-bearing
  modes.  The `"instance"` path keeps its existing fallback semantics.
- **BREAKING**: `evolution.cache_evaluation` now defaults to `False`
  (previously `True`).  Matches GEPA Optimize Anything's conservative
  cache_evaluation default
  (`src/gepa/optimize_anything.py:476`).  When the cache *is* enabled,
  entries are now keyed by candidate **content** (the worktree's
  `HEAD^{tree}` SHA, with a clean-state guard) rather than HELIX's
  lineage `candidate.id`, so equivalent candidates can reuse results
  across re-derivations.  Configs that previously relied on cache hits
  (e.g. resume scenarios that re-evaluate the seed) should set
  `cache_evaluation = true` explicitly.
- **BREAKING**: `score_parser = "helix_result"` now takes a **per-example**
  `HELIX_RESULT=[[score_0, side_info_0], [score_1, side_info_1], ...]`
  payload — one `[score, side_info]` pair per id in `helix_batch.json`.
  HELIX zips it into `instance_scores` and stores `side_info_i` on
  `EvalResult.per_example_side_info` for the reflection prompt.  GEPA
  `optimize_anything` parity (`src/gepa/optimize_anything.py:387-438`).
  The previous scalar-plus-id-keyed-dict contract is removed — it
  silently failed the minibatch gate whenever the evaluator keyed its
  dict by aggregate metric names (`task__metric`) instead of per-example
  ids (`task__trialN`).  Migration: replace
  `HELIX_RESULT=[mean, {"scores": {id_i: v_i, ...}, ...}]` with
  `HELIX_RESULT=[[v_0, {"info": "..."}], [v_1, {...}], ...]`.
- `helix.executor.run_evaluator` now emits a `WARNING` log line when the
  post-filter zero-fills any requested id (naming the count and a sample
  of up to five).  Non-breaking — behaviour is unchanged, only
  observability improves.  Catches parser / id-keying mismatches on
  parsers other than `helix_result` (e.g. `exitcode` plus `instance_ids`).
- **BREAKING**: `seedless` is now a section (`[seedless]` with `enabled = …`),
  not a top-level boolean.
- **BREAKING**: `dataset.train_path` / `dataset.val_path` have moved to
  `seedless.train_path` / `seedless.val_path`.  `[dataset]` is now
  cardinality-only (`train_size` / `val_size`).
- **BREAKING**: `helix_batch.json` payload shape is now `list[str]` instead
  of `list[int]`.  Example ids flow through the Architecture A evaluator
  handoff as opaque strings — the default `_RangeDataLoader` emits `"0"`,
  `"1"`, …, and `StratifiedBatchSampler` emits task-prefixed ids like
  `"group_alpha__case_3"`.  Evaluators that previously read the handoff as
  `list[int]` must cast on their side
  (`[int(s) for s in json.loads(Path("helix_batch.json").read_text())]`)
  or switch to string-keyed lookup.  Unblocks the stratified sampler on
  Architecture A, which previously died with
  `ValueError: invalid literal for int()` at the serialization boundary.
- **BREAKING**: All pydantic sub-models in `src/helix/config.py`
  (`EvaluatorConfig`, `DatasetConfig`, `SeedlessConfig`, `EvolutionConfig`,
  `ClaudeConfig`, `WorktreeConfig`, `HelixConfig`) now use
  `model_config = ConfigDict(extra="forbid")`.  Unknown / misplaced /
  mistyped TOML keys raise a validation error at load time instead of
  being silently dropped.  Previously, placing `batch_sampler` under
  `[evaluator]` (the key lives on `[evolution]`) silently left users on
  the default sampler with no warning.

## [0.1.0] - 2026-04-10

### Added
- GEPA (Gradual Enhancement with Progressive Adaptation) parity support for evolution strategies
- Seedless evolution mode allowing evolution without explicit random seeds
- Automatic retry logic for API rate-limit handling
- Rich progress bar for evolution tracking and visualization
- Multi-file evolution support for complex codebases

### Changed
- Refactored identity system for improved modularity and maintainability
- Enhanced configuration system with better validation

### Fixed
- Rate-limit handling in API interactions
- Progress tracking accuracy during long-running evolutions
