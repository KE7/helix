# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**: Removed the `evaluator.score_parser` configuration field. The built-in
  `helix_result` parser is now implicit; configurations that still provide the
  removed field are rejected by Pydantic's `extra="forbid"` validation.
  Persisted resume records created before 0.3.0 retain the old field in their
  semantic snapshot and must be restarted rather than resumed.
- **BREAKING**: Removed the `pytest`, `exitcode`, `json_accuracy`, and
  `json_score` score parsers. `helix_result` is now the only supported parser
  and the default. Migration: emit one `HELIX_RESULT=` line containing a
  positional `[score, side_info]` pair for every id in `helix_batch.json`.
- **BREAKING**: `evolution.frontier_type` now defaults to `"instance"`
  instead of `"hybrid"`.  The multi-axis frontiers require per-example
  objective scores, which only the built-in result parser can supply
  and only when the evaluator emits a `"scores"` sub-dict per example.
  Legacy scalar parser configurations left `objective_scores` empty, so the old default
  aborted the run with `MissingObjectiveScoresError` during the *seed*
  evaluation — before generation 1.  Migration: projects that were
  relying on the implicit `"hybrid"` default **and** do emit objective
  scores must now set `frontier_type = "hybrid"` explicitly, otherwise
  they silently get the single-axis instance frontier. Runs recorded under
  the old `"hybrid"` default also cannot be resumed after upgrading:
  resume validation rejects a saved `frontier_type` that differs from the
  new `"instance"` default before it compares parser-key snapshots.
  Explicitly requesting `"objective"` or `"cartesian"` without objective
  scores warns during frontier updates and raises
  `MissingObjectiveScoresError` when selection needs that axis; explicitly
  requesting `"hybrid"` warns and no-ops its objective axis, continuing on
  the instance axis.

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
- Scalar evaluator results now populate every requested instance ID in multi-example runs. (PR #32)

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
- Multi-axis Pareto frontier (GEPA's `FrontierType` type alias in
  `core/state.py`).  New
  `evolution.frontier_type: Literal["instance", "objective", "hybrid",
  "cartesian"]` with default `"hybrid"` — matches GEPA's own
  Optimize Anything engine default (`EngineConfig.frontier_type` in
  `gepa_launcher.py`).
  `ParetoFrontier` now tracks per-objective-name and per-`(val_id,
  objective_name)` best sets alongside the existing per-example-id
  tracking, and `get_non_dominated()` / `select_parent()` dispatch on
  `frontier_type`.  The acceptance gate stays positional on
  `scores_list` unchanged.
- `EvalResult.per_example_side_info: list[dict[str, Any]] | None` —
  per-example diagnostic dicts from the new `helix_result` contract,
  positional to `instance_scores` by `helix_batch.json` id order.
  GEPA analogue: `EvaluationBatch.trajectories` in `core/adapter.py`.
- `EvalResult.objective_scores: list[dict[str, float]] | None` —
  per-example objective-axis harvest from `side_info["scores"]`.  GEPA
  analogue: `EvaluationBatch.objective_scores` in `core/adapter.py`.
  Feeds `frontier_type ∈ {"objective",
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
- **BREAKING**: The built-in result parser now rejects malformed
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
  `cache_evaluation` default on `EngineConfig` in `gepa_launcher.py`.
  When the cache *is* enabled,
  entries are now keyed by candidate **content** (the worktree's
  `HEAD^{tree}` SHA, with a clean-state guard) rather than HELIX's
  lineage `candidate.id`, so equivalent candidates can reuse results
  across re-derivations.  Configs that previously relied on cache hits
  (e.g. resume scenarios that re-evaluate the seed) should set
  `cache_evaluation = true` explicitly.
- **BREAKING**: The built-in result parser now takes a **per-example**
  `HELIX_RESULT=[[score_0, side_info_0], [score_1, side_info_1], ...]`
  payload — one `[score, side_info]` pair per id in `helix_batch.json`.
  HELIX zips it into `instance_scores` and stores `side_info_i` on
  `EvalResult.per_example_side_info` for the reflection prompt.  GEPA
  Optimize Anything parity — evaluators return per-example
  `(score, side_info)` results, as `OptimizeAnythingAdapter` expects.
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
  evaluator/id mismatches.
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
- GEPA (Genetic-Pareto) parity support for evolution strategies
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
