"""HELIX configuration models using Pydantic v2."""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class EvaluatorConfig(BaseModel):
    """Configuration for candidate evaluation.

    Defines how candidates are evaluated via shell commands and how
    their results are parsed into scores.

    ``score_parser`` selects how HELIX turns evaluator output into
    ``(scores, instance_scores)``.  For the minibatch-gate at
    :func:`helix.evolution._minibatch_gate_accept` the **per-id keys**
    in ``instance_scores`` matter: the gate looks up
    ``instance_scores[eid]`` where ``eid`` is whatever HELIX wrote to
    ``helix_batch.json`` pre-invocation.  Pick ``"helix_result"`` to
    hand HELIX a list of per-example ``[score, side_info]`` pairs
    (GEPA ``optimize_anything`` evaluator parity) — HELIX owns the
    id-keying so the evaluator never types a HELIX-internal id.  The
    other parsers (``"pytest"``, ``"exitcode"``, ``"json_accuracy"``,
    ``"json_score"``) aggregate to a single score and do NOT produce
    id-keyed per-instance scores; combining them with ``instance_ids``
    triggers the zero-fill warning in :mod:`helix.executor`.
    """
    model_config = ConfigDict(extra="forbid")

    command: str
    score_parser: Literal[
        "pytest", "exitcode", "json_accuracy", "json_score", "helix_result"
    ] = "pytest"
    include_stdout: bool = True
    include_stderr: bool = True
    extra_commands: list[str] = Field(default_factory=list)
    protected_files: list[str] = Field(
        default_factory=list,
        description=(
            "Additional repo-relative files that must remain immutable across "
            "mutations/merges (e.g. evaluator helpers, goldens)."
        ),
    )


class DatasetConfig(BaseModel):
    """Dataset cardinality for HELIX's minibatch sampler.

    HELIX evaluates candidates via shell commands (evaluator.command),
    not per-example function calls like GEPA.  This section therefore
    only carries the *cardinality* of the train and val splits; the
    evaluator owns the actual dataset.  Architecture A (example-id
    handoff): HELIX samples example ids — stringified indices into
    ``range(train_size)`` by default, or opaque structured ids like
    ``"cube_stack__3"`` when ``evolution.batch_sampler = "stratified"``
    — and writes them to ``{worktree}/helix_batch.json``; the evaluator
    reads that file (from its cwd) and filters its own loaded dataset
    by those ids.  Ids are opaque at the HELIX/evaluator boundary:
    evaluators are responsible for any interpretation (e.g. casting
    ``"7"`` back to ``int`` for positional indexing).

    Legacy prompt-grounding paths (``train_path`` / ``val_path``) now
    live on :class:`SeedlessConfig` — they only affect the seed-
    generation prompt and are unrelated to runtime minibatch sampling.
    """

    model_config = ConfigDict(extra="forbid")

    train_size: int | None = Field(
        default=None,
        description=(
            "If set, HELIX will use an EpochShuffledBatchSampler over "
            "range(train_size) for the minibatch acceptance test. "
            "Use this when the evaluator loads its own dataset and HELIX "
            "only needs to know the cardinality."
        ),
    )
    val_size: int | None = Field(
        default=None,
        description=(
            "Cardinality of the validation split.  If set, the full "
            "valset evaluation writes range(val_size) to "
            "{worktree}/helix_batch.json."
        ),
    )

    def model_post_init(self, __context: object) -> None:  # noqa: D401
        if self.train_size is not None and self.train_size < 0:
            raise ValueError(
                f"dataset.train_size must be >= 0 (got {self.train_size})"
            )
        if self.val_size is not None and self.val_size < 0:
            raise ValueError(
                f"dataset.val_size must be >= 0 (got {self.val_size})"
            )


class SeedlessConfig(BaseModel):
    """Seedless-mode configuration for HELIX.

    Seedless mode generates the initial candidate from the objective via
    a single LLM call instead of starting from the current working tree.

    ``train_path`` / ``val_path`` here are the GEPA-parity prompt-grounding
    paths used *only* during seed generation (and, historically, the
    legacy payload-based minibatch path):

    - **single-task mode** (default): ``train_path`` absent / ``None`` — no
      ``## Sample Inputs`` section is added to the seed prompt.  Matches GEPA
      ``optimize_anything(dataset=None)``.
    - **multi-task / generalization mode**: ``train_path`` points to a training
      dataset file (JSON array or JSONL) or a directory of JSON files.  The
      first 3 items are read and serialised as string representations, then
      included in the ``## Sample Inputs`` section of the seed prompt — exactly
      mirroring GEPA's ``_build_seed_generation_prompt(dataset=dataset[:3])``.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "When True, bootstrap an empty worktree and generate the "
            "initial candidate via a single LLM invocation."
        ),
    )
    train_path: Path | None = Field(
        default=None,
        description=(
            "Optional prompt-grounding training dataset for seedless seed "
            "generation.  Accepts a JSON array file, a JSONL file, or a "
            "directory of JSON files.  None → single-task mode (no Sample "
            "Inputs section)."
        ),
    )
    val_path: Path | None = Field(
        default=None,
        description=(
            "Optional validation dataset path — retained for GEPA-parity "
            "payload-based minibatch paths.  Falls back to ``train_path`` "
            "via :attr:`effective_val_path` when None."
        ),
    )

    @property
    def effective_val_path(self) -> Path | None:
        return self.val_path if self.val_path is not None else self.train_path


def load_dataset_examples(train_path: Path, max_examples: int = 3) -> list[str]:
    """Load up to *max_examples* string representations from a training dataset.

    Mirrors GEPA's ``dataset[:3]`` slice used in
    ``_build_seed_generation_prompt``.  Supports three layouts:

    - **JSON array file** (``*.json``): parsed as a list; first *max_examples*
      items serialised with :func:`json.dumps`.
    - **JSONL file** (``*.jsonl`` or any extension): each non-blank line parsed
      as a JSON object; first *max_examples* lines used.
    - **Directory**: each ``*.json`` file is one instance (sorted by name);
      first *max_examples* files parsed and serialised.

    Parameters
    ----------
    train_path:
        Path to a JSON array file, JSONL file, or directory of JSON files.
    max_examples:
        Maximum number of examples to return (default 3, matching GEPA).

    Returns
    -------
    list[str]
        String representations of the loaded examples, ready for inclusion in
        the seed-generation prompt.

    Raises
    ------
    ValueError
        If *train_path* does not exist or cannot be parsed.
    """
    if not train_path.exists():
        raise ValueError(f"train_path does not exist: {train_path}")

    items: list[Any] = []

    if train_path.is_dir():
        # Directory of JSON files — one instance per file, sorted by name.
        json_files = sorted(train_path.glob("*.json"))
        for p in json_files[:max_examples]:
            items.append(json.loads(p.read_text()))
    else:
        raw = train_path.read_text().strip()
        # Try JSON array first.
        if raw.startswith("["):
            data = json.loads(raw)
            if isinstance(data, list):
                items = data[:max_examples]
            else:
                raise ValueError(
                    f"train_path JSON file does not contain a top-level array: {train_path}"
                )
        else:
            # Treat as JSONL — one JSON object per non-blank line.
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    items.append(json.loads(line))
                if len(items) >= max_examples:
                    break

    return [json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item
            for item in items]


class EvolutionConfig(BaseModel):
    """Configuration for the evolution process.

    Controls generation count, frontier management, termination caps,
    and parallel proposal settings.
    """
    model_config = ConfigDict(extra="forbid")

    max_generations: int = 10
    perfect_score_threshold: float | None = None
    # Evaluation-budget cap. `-1` (default) disables — HELIX runs until
    # `max_generations` alone. Set to a positive int to match GEPA's
    # budget-exhaustion termination (see GEPA core/engine.py, which uses
    # the same evaluator-call budget the same way).
    max_evaluations: int = -1
    # Merge is OFF by default (GEPA parity: merge = None in GEPAConfig).
    merge_enabled: bool = False
    # Total cap on merge invocations across the entire run (not per-gen).
    max_merge_invocations: int = 5
    # Minimum val-set overlap floor for merge candidates. Must be > 0
    # (GEPA parity: merge.py:243-244 rejects val_overlap_floor <= 0).
    merge_val_overlap_floor: int = 5
    # Number of val ids sampled for merge acceptance. Default 5 matches
    # GEPA (merge.py:262 num_subsample_ids=5). Must be >= 1.
    merge_subsample_size: int = 5
    # GEPA parity: number of parallel proposals per generation. When > 1,
    # sample N parents, run N mutations in parallel via ThreadPoolExecutor,
    # then accept sequentially. See GEPA core/engine.py
    # _run_parallel_reflective_batch.
    num_parallel_proposals: int | Literal["auto"] = Field(
        default=1,
        description=(
            "Number of concurrent mutation proposals per iteration. "
            "GEPA parity: EngineConfig.num_parallel_proposals. "
            "Set to 'auto' to derive from max_workers // minibatch_size, "
            "matching GEPA's optimize_anything._resolve_num_parallel_proposals."
        ),
    )
    minibatch_size: int = Field(
        default=3,
        description=(
            "Number of training examples per minibatch acceptance test. "
            "GEPA parity: ReflectionConfig.reflection_minibatch_size default."
        ),
    )
    max_workers: int = Field(
        default_factory=lambda: os.cpu_count() or 32,
        description=(
            "Max parallel eval workers — bounds both the parent-eval and "
            "mutation ThreadPools in the num_parallel_proposals pipeline. "
            "GEPA parity: EngineConfig.max_workers "
            "(/tmp/gepa-official/src/gepa/optimize_anything.py:485, "
            "default os.cpu_count() or 32)."
        ),
    )
    cache_evaluation: bool = Field(
        default=True,
        description=(
            "Enable (candidate_hash, example_id) evaluation cache. "
            "GEPA parity: EngineConfig.cache_evaluation."
        ),
    )
    acceptance_criterion: Literal["strict_improvement", "improvement_or_equal"] = Field(
        default="strict_improvement",
        description=(
            "Acceptance criterion for minibatch gate. "
            "GEPA parity: EngineConfig.acceptance_criterion."
        ),
    )
    val_stage_size: int | None = Field(
        default=None,
        description=(
            "Optional deterministic first-N validation stage that runs after "
            "the train minibatch gate and before full validation. Disabled "
            "when unset or 0."
        ),
    )
    batch_sampler: Literal["epoch_shuffled", "stratified"] = Field(
        default="epoch_shuffled",
        description=(
            "Minibatch sampling strategy. "
            "'epoch_shuffled' (default): GEPA-parity EpochShuffledBatchSampler. "
            "'stratified': StratifiedBatchSampler guarantees each minibatch of "
            "size K touches K distinct groups, where the group key is derived "
            "from each instance id via 'evolution.group_key_separator'. Falls "
            "back to epoch_shuffled behaviour when fewer groups than "
            "minibatch_size are available."
        ),
    )
    group_key_separator: str = Field(
        default="__",
        description=(
            "Separator used to derive a group key from each instance id for "
            "the stratified batch sampler: the id is split on this separator "
            "and the first part is taken as the group key (e.g. "
            "'cube_stack__s3' -> 'cube_stack' when separator='__')."
        ),
    )
    frontier_type: Literal["instance", "objective", "hybrid", "cartesian"] = Field(
        default="hybrid",
        description=(
            "Pareto frontier dimensionality.  Mirrors GEPA's "
            "``FrontierType`` at ``src/gepa/core/state.py:22-23``.  Default "
            "is ``\"hybrid\"`` to match GEPA ``optimize_anything``'s own "
            "default at ``src/gepa/optimize_anything.py:476`` — O.A. is the "
            "right parent for HELIX, not the base ``api.py`` path whose "
            "default is ``\"instance\"``.\n\n"
            "- ``\"instance\"``: one frontier key per example-id.  Matches "
            "HELIX's historical behaviour and GEPA's ``frontier_type="
            "\"instance\"``.\n"
            "- ``\"objective\"``: one frontier key per objective-name, "
            "score = mean of that objective across the valset.  Harvested "
            "from ``side_info[\"scores\"]`` via ``helix_result``.\n"
            "- ``\"hybrid\"``: both instance and objective frontiers "
            "maintained; a candidate is retained if it survives on either.\n"
            "- ``\"cartesian\"``: one frontier key per (val_id, "
            "objective_name) pair.  Mirrors GEPA "
            "``_update_pareto_front_for_cartesian``.\n\n"
            "The acceptance gate stays positional on ``scores_list`` "
            "regardless of ``frontier_type`` (GEPA ``acceptance.py:39-48``); "
            "only the Pareto retention / parent-selection decision is "
            "multi-axis.  Non-``instance`` paths require ``helix_result`` "
            "to emit per-example ``side_info[\"scores\"]`` dicts — without "
            "them the objective / cartesian frontiers stay empty and "
            "behaviour degenerates to the instance path."
        ),
    )

    def model_post_init(self, __context: object) -> None:
        # GEPA parity: resolve ``num_parallel_proposals="auto"`` to
        # ``max(1, max_workers // minibatch_size)`` once at construction
        # time so every downstream consumer sees a plain int.  Mirrors
        # /tmp/gepa-official/src/gepa/optimize_anything.py:1108-1116.
        if self.num_parallel_proposals == "auto":
            self.num_parallel_proposals = max(
                1, self.max_workers // max(1, self.minibatch_size)
            )
        if self.val_stage_size is not None and self.val_stage_size < 0:
            raise ValueError(
                f"evolution.val_stage_size must be >= 0 (got {self.val_stage_size})"
            )
        # GEPA parity (merge.py:243-244): reject non-positive overlap floors.
        if self.merge_val_overlap_floor <= 0:
            raise ValueError(
                "evolution.merge_val_overlap_floor must be > 0 "
                f"(got {self.merge_val_overlap_floor})"
            )
        if self.merge_subsample_size < 1:
            raise ValueError(
                "evolution.merge_subsample_size must be >= 1 "
                f"(got {self.merge_subsample_size})"
            )
        # group_key_separator is only consumed by the stratified sampler;
        # validate it only on that path so default ('__') configs that use
        # the epoch_shuffled sampler aren't restricted unnecessarily.
        if self.batch_sampler == "stratified" and not self.group_key_separator:
            raise ValueError(
                "evolution.group_key_separator must be a non-empty string "
                "when evolution.batch_sampler='stratified' "
                f"(got {self.group_key_separator!r})"
            )


class ClaudeConfig(BaseModel):
    """Configuration for Claude Code integration.

    Specifies which Claude model to use, allowed tools, effort level,
    and optional background context for mutation sessions.
    """
    model_config = ConfigDict(extra="forbid")

    model: str = "sonnet"
    effort: str | None = None
    max_turns: int | None = None
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["Read", "Edit", "Write", "Bash", "Glob", "Grep"]
    )
    background: str | None = None


class WorktreeConfig(BaseModel):
    """Configuration for git worktree management.

    Defines where candidate worktrees are created during evolution.
    """
    model_config = ConfigDict(extra="forbid")

    base_dir: str = ".helix/worktrees"
    # Deprecated: GEPA uses append-only population — dominated candidates are
    # filtered at selection time, never pruned from storage.  Kept for TOML
    # back-compat; the value is ignored at runtime.
    cleanup_dominated: bool = False


class HelixConfig(BaseModel):
    """Top-level HELIX configuration.

    Combines all configuration sections (objective, evaluator, dataset,
    evolution, claude, worktree) and validates compatibility constraints.
    """
    model_config = ConfigDict(extra="forbid")

    objective: str
    seed: str = "."
    rng_seed: int = 0  # GEPA parity: deterministic RNG for selection
    passthrough_env: list[str] = Field(
        default_factory=list,
        description=(
            "Environment variable names to pass through the env scrub into "
            "evaluator and Claude Code subprocesses (e.g. "
            '["CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "HF_HOME"]).'
        ),
    )
    evaluator: EvaluatorConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    seedless: SeedlessConfig = Field(default_factory=SeedlessConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    worktree: WorktreeConfig = Field(default_factory=WorktreeConfig)

    def model_post_init(self, __context: object) -> None:
        if self.seedless.enabled and not self.objective.strip():
            raise ValueError(
                "'objective' must be non-empty when seedless.enabled=True. "
                "The LLM needs the objective to generate an initial candidate."
            )


def load_config(path: Path) -> HelixConfig:
    """Load a HelixConfig from a TOML file.

    Supports both flat format (``objective = "..."`` at root) and the
    ``[project]`` section format (fields are merged into the root).
    """
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(
            f"❌ Error parsing TOML file: {path}\n"
            f"   {e}\n"
            f"   Please check your TOML syntax.",
            file=sys.stderr,
        )
        sys.exit(1)

    # If a [project] section is present, promote its keys to the root level
    # (values already at root take precedence over those inside [project]).
    if "project" in data and isinstance(data["project"], dict):
        merged = {**data["project"], **{k: v for k, v in data.items() if k != "project"}}
        data = merged

    try:
        return HelixConfig.model_validate(data)
    except ValidationError as e:
        print(
            f"❌ Configuration validation error in {path}:\n",
            file=sys.stderr,
        )
        for error in e.errors():
            field_path = " → ".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            print(f"   Field: {field_path}", file=sys.stderr)
            print(f"   Error: {msg}", file=sys.stderr)
            if error["type"] == "missing":
                print(f"   Hint: Add '{error['loc'][-1]}' to your helix.toml", file=sys.stderr)
            elif error["type"] == "extra_forbidden":
                print(
                    f"   Hint: '{error['loc'][-1]}' is not a recognised key on this "
                    "section — check for typos or a misplaced sub-section "
                    "(e.g. a key that belongs under [evolution] placed under [evaluator]).",
                    file=sys.stderr,
                )
            elif "type" in str(error["type"]):
                print(f"   Hint: Check that the value is the correct type", file=sys.stderr)
            print(file=sys.stderr)
        sys.exit(1)
