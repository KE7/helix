# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-06

### Added
- `pi` backend (`agent.backend = "pi"`, verified with pi 1.5.0), on the host
  and in the Docker sandbox (`helix sandbox login pi`). pi has no turn cap, so
  `max_turns` is only a hint; for a local engine, see the configuration guide.
- `examples/maniskill_pick_place/` (#77, by Mahanth Komuravelli) evolves a
  multi-file solver for a Franka Panda on ManiSkill 3 push, pick and stack. In
  the recorded run one mutation took validation from 0.064 to 1.0.
- A docs site, https://ke7.github.io/helix/, and a HELIX logo.
- Every run writes a timing trace to `.helix/timing.jsonl`. `helix timing`
  shows where time went, including model and tool time inside agent calls
  (claude, codex, opencode, pi), and `--perfetto FILE` exports a timeline.
- `from helix import timed` times your own steps in the evaluator, its sidecar
  or scripts the agent runs; `helix timing` totals them.
- `helix attempts` lists rejected candidates and perfect-score skips, with
  filters and `--json`.
- `evolution.mutations_per_parent` (N) proposes N children per sampled parent,
  P × N per iteration; `evolution.proposal_selection` (`all_improvements`,
  `best_improvement`, `top_k` with `proposal_top_k`) picks which passing ones
  get full validation.
- `evolution.candidate_selection_strategy`: `pareto` (default), `current_best`,
  `epsilon_greedy` (`candidate_selection_epsilon`) or `top_k_pareto`
  (`candidate_selection_top_k`), GEPA's parent-selection strategies.
- `evolution.batch_sampler = "stratified"` takes `num_sampled_groups` and
  `num_examples_per_group`.
- `evolution.sibling_history` (default off) shows the mutation agent the
  children already proposed from the same parent, with their training scores
  and a summary of each change. Each summary is one more, charged backend call;
  `sibling_summary_max_words` and `sibling_history_max_entries` bound them.
- `evolution.stop_at_score` / `--stop-at-score` ends the run once an accepted
  candidate reaches the target score.
- `evolution.retain_rejected_worktrees` keeps the worktrees of children
  rejected at validation; `helix clean` removes them.
- `[agent].web_access = false` turns off the agent's built-in web tools. The
  shell can still reach the network; for a strict no-web run, use the sandbox.
- `[sandbox].network` accepts an existing Docker network, and
  `[sandbox].auth_volume` gives a run its own credentials volume.
- Sandboxed agents start each generation from a clean session; credentials and
  config are kept. Give concurrent runs separate `auth_volume`s.
- `xhigh` and `max` are accepted `effort` values for claude.
- Runner images are rebuilt weekly, and a PR check builds and probes them.

### Changed
- **BREAKING**: HELIX requires Python 3.12 or newer; 3.12 to 3.15 are tested.
- **BREAKING**: the `gemini` backend is replaced by `agy` (Google's
  Antigravity CLI). Run `helix sandbox login agy`.
- **BREAKING**: `evaluator.score_parser` and the `pytest`, `exitcode`,
  `json_accuracy` and `json_score` parsers are removed. Print one
  `HELIX_RESULT=` line with a `[score, side_info]` pair per id in
  `helix_batch.json`. Runs saved by 0.2.x cannot be resumed.
- **BREAKING**: `evolution.frontier_type` defaults to `"instance"` (was
  `"hybrid"`). Set `"hybrid"` yourself if your evaluator emits per-example
  objective scores.
- **BREAKING**: `agent.allowed_tools` and
  `sandbox.preserve_backend_transcripts` are removed (transcripts are always
  kept); configs that set them are rejected.
- **BREAKING**: `evolution.num_parallel_proposals` no longer accepts `"auto"`.
  It and `max_workers` must be at least 1, and `max_workers` now limits only
  concurrent proposal slots.
- **BREAKING**: configuration errors exit with status 2 (was 1).
- **BREAKING** (`examples/circle_packing`): the example is now
  optimize_anything's circle-packing experiment, with GEPA's fitness function;
  the old one leaked its answer to the agent.
- **BREAKING** (Python API): `helix.mutator.parse_mutation_summary` is
  removed, and prompts no longer ask for `[MUTATION COMPLETE]` markers.
- **BEHAVIOUR**: parent selection ranks candidates by mean instance score
  instead of the sum, as GEPA does.
- A rate-limited backend call is an ordinary failed mutation or merge, as in
  GEPA, instead of ending the run.
- The generation counter advances every iteration, as in GEPA.
- `helix evolve` flags are validated together with `helix.toml`.
- `evolution.val_stage_size`: a passing validation stage is reused, so final
  validation runs only the remaining ids. The evaluator must score each id the
  same way on any subset.
- The mutation prompt shows only the current minibatch's feedback.
- Identical children proposed in one batch become one frontier entry.
- A scalar evaluator result applies to every requested id.
- Runner images are published for amd64 and arm64.
- `examples/web_researcher` asks Claude Haiku directly and runs in the
  sandbox with its evaluator files protected.
- The README is a short overview; the reference material moved to the docs.
- `CONTRIBUTING.md` sets the rules for every PR, human or agent. The HELIX
  agent skill is a Claude Code project skill (`.claude/skills/helix`).

### Fixed
- Unsandboxed Claude Code on a Mac subscription works: the agent's
  environment keeps `USER`, so Claude finds its login.
- Merges work: the merge agent gets its task, a merge that changes nothing
  is rejected before evaluation, and the merge prompt no longer shows
  validation scores or protected files.
- Seedless mode works: the seed agent gets its task, the seed is committed so
  its children build on it, and the seed prompt no longer shows the evaluator
  command.
- Projects without a `[dataset]` (single task) can be scored.
- Sibling history shows each child's training score, never a validation score.
- `helix resume` keeps a candidate validated just before an interrupt, cleans
  up interrupted attempts, and no longer fails on a leftover branch.
- `helix log` shows the run, one row per candidate, and `lineage.json`
  records each child's `files_changed`.
- Token and cost accounting: Claude cache reads are counted once, opencode
  counts every model call, and failed calls, sandbox timeouts and every
  proposal slot are charged.
- Every backend's transcript is kept (it was claude only), and the live
  "Tools" count and `helix timing` read it, including unsandboxed Claude
  (`[sandbox].claude_transcript_root` now applies to sandboxed runs only).
- An agent call that hits `[sandbox].timeout_seconds` is a failed mutation.
- `max_evaluations` is checked between iterations, as in GEPA, so validations
  already paid for are not thrown away.
- A child whose evaluation raises is rejected instead of ending the run.
- `max_generations` must be at least 1; 0 used to run nothing or crash.
- `--dir` accepts relative paths, and `helix attempts` takes `--dir`.
- `helix best --export` copies only the candidate's files, not HELIX's logs.
- Change summaries no longer drop a successful mutation, cost nothing when
  the child changed nothing, and no longer describe HELIX's `.gitignore` edit.
- opencode: unsandboxed runs keep your stored logins and run parallel
  proposals without database errors; sandboxed runs work again.
- Sandbox sync-back drops symlinks that point outside the candidate.
- Backend output with U+2028 or U+2029 inside JSON strings parses.
- Parent selection is reproducible for a given random seed.
- Cached evaluations keep per-example feedback, and an evaluator that returns
  the wrong number of results no longer corrupts the cache.
- Concurrent worktree operations no longer drop proposal slots.
- Docker errors no longer print `passthrough_env` values.
- Error panels and `helix init` print bracketed text and TOML table names
  as written.
- `examples/web_researcher` installs HELIX in its evaluator's environment.
- Messages name the configured backend instead of always saying Claude Code.

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
