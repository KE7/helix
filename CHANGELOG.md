# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `DatasetConfig.train_size` / `val_size` — cardinality-only fields that drive
  the minibatch sampler when the evaluator owns the dataset (Architecture A
  positional-index handoff).  HELIX writes sampled integer indices to
  `{worktree}/helix_batch.json`; the evaluator filters its own dataset.
- `SeedlessConfig` — new section carrying `enabled` plus the optional
  prompt-grounding `train_path` / `val_path` used during seed generation.
- Evaluator-owned dataset integration wired into Architecture A with
  cardinality-only `train_size` / `val_size`.

### Changed
- **BREAKING**: `seedless` is now a section (`[seedless]` with `enabled = …`),
  not a top-level boolean.
- **BREAKING**: `dataset.train_path` / `dataset.val_path` have moved to
  `seedless.train_path` / `seedless.val_path`.  `[dataset]` is now
  cardinality-only (`train_size` / `val_size`).

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
