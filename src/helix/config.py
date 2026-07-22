"""HELIX configuration models using Pydantic v2."""

from __future__ import annotations

import json
import os
import sys
import tomllib
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from helix.backends import (
    BackendName,
    EFFORT_AWARE_BACKENDS,
    EFFORT_VALID_VALUES,
    backend_display_name,
)


def _load_dotenv_file(path: Path) -> None:
    """Load simple KEY=VALUE dotenv entries into os.environ if unset."""
    if not path.exists() or not path.is_file():
        return
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            # Quoted value: keep contents verbatim (preserve embedded ``#``,
            # whitespace, ``=`` etc.) and only strip the surrounding quotes.
            value = value[1:-1]
        else:
            # Unquoted value: strip a trailing inline comment introduced by
            # ``#`` preceded by whitespace (matches POSIX-shell semantics for
            # ``.env`` files used by foreman, dotenv, docker-compose, etc.).
            for i, ch in enumerate(value):
                if ch == "#" and (i == 0 or value[i - 1].isspace()):
                    value = value[:i].rstrip()
                    break
        os.environ[key] = value


class EvaluatorSidecarConfig(BaseModel):
    """Warm private evaluator service for Docker-sandboxed runs."""

    model_config = ConfigDict(extra="forbid")

    image: str
    command: str
    endpoint: str
    runner_image: str | None = None
    healthcheck_command: str | None = None
    startup_timeout_seconds: int = 60
    internal_network: bool = True
    passthrough_env: list[str] = Field(
        default_factory=list,
        description=(
            "Environment variable names passed only to the private evaluator "
            "sidecar, never to mutation-agent or evaluator-runner processes."
        ),
    )

    @property
    def resolved_runner_image(self) -> str:
        return self.runner_image or self.image


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
    id-keying so the evaluator never types a HELIX-internal id.

    ``helix_result`` is the only parser that populates the new
    :attr:`helix.population.EvalResult.per_example_side_info` and
    :attr:`~helix.population.EvalResult.objective_scores` fields: the
    former from ``side_info_i`` verbatim (reflection), the latter from
    the reserved ``side_info_i["scores"]`` sub-dict (multi-axis Pareto
    frontier — see :attr:`EvolutionConfig.frontier_type`).

    The other parsers (``"pytest"``, ``"exitcode"``, ``"json_accuracy"``,
    ``"json_score"``) aggregate to a single score and do NOT produce
    id-keyed per-instance scores; combining them with ``instance_ids``
    triggers the zero-fill warning in :mod:`helix.executor`.  They also
    leave ``objective_scores`` empty, so ``frontier_type`` values other
    than ``"instance"`` fail loudly when selected with these parsers.
    """

    model_config = ConfigDict(extra="forbid")

    command: str
    score_parser: Literal[
        "pytest", "exitcode", "json_accuracy", "json_score", "helix_result"
    ] = "pytest"
    include_stdout: bool = True
    include_stderr: bool = True
    extra_commands: list[str] = Field(default_factory=list)
    sidecar: EvaluatorSidecarConfig | None = None
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
    ``"group_alpha__case_3"`` when ``evolution.batch_sampler = "stratified"``
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
            raise ValueError(f"dataset.train_size must be >= 0 (got {self.train_size})")
        if self.val_size is not None and self.val_size < 0:
            raise ValueError(f"dataset.val_size must be >= 0 (got {self.val_size})")


class SeedlessConfig(BaseModel):
    """Seedless-mode configuration for HELIX.

    Seedless mode generates the initial candidate from the objective via
    a single LLM call instead of starting from the current working tree.

    ``train_path`` / ``val_path`` here are the GEPA-parity prompt-grounding
    paths used *only* during seed generation (and, historically, the
    legacy payload-based minibatch path):

    - **single-task / no-example mode** (default): ``train_path`` absent /
      ``None`` — no ``## Sample Inputs`` section is added to the seed prompt.
      Matches GEPA O.A. Single-Task Search
      ``optimize_anything(dataset=None, valset=None)``.
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
            "directory of JSON files.  None → single-task/no-example mode (no Sample "
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

    return [
        json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item
        for item in items
    ]


class EvolutionConfig(BaseModel):
    """Configuration for the evolution process.

    Controls generation count, frontier management, termination caps,
    and parallel proposal settings.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    max_generations: int = 10
    perfect_score_threshold: float | None = None
    # Evaluation budget cap. `-1` (default) disables the cap, so HELIX runs until
    # `max_generations` alone. Dataset/minibatch evaluations consume one unit
    # per uncached example; single-task/no-example evaluator calls consume 0/1
    # metric calls (cached=0, uncached=1 because no per-example ids exist).
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
            "P — number of parent-selection attempts per iteration in the "
            "unified P*N proposal scheduler. This is the same value the "
            "current loop already uses: it selects one parent per slot, so "
            "an existing num_parallel_proposals=K run is exactly the K-by-1 "
            "case (P=K, N=1); there is no legacy runtime branch. "
            "GEPA parity: EngineConfig.num_parallel_proposals. "
            "Set to 'auto' to derive from max_workers // minibatch_size, "
            "matching GEPA's optimize_anything._resolve_num_parallel_proposals. "
            "Integer values must be >= 1."
        ),
    )
    mutations_per_parent: int = Field(
        default=1,
        description=(
            "N — number of independently sampled reflective mutations for "
            "each selected parent in the unified P*N proposal scheduler. A "
            "full step plans at most num_parallel_proposals * "
            "mutations_per_parent (P*N) children. Defaults to 1, so every "
            "existing num_parallel_proposals=K configuration lands on the "
            "K-by-1 case with the existing behavior preserved. Must be >= 1."
        ),
    )
    proposal_selection: Literal["all_improvements", "best_improvement", "top_k"] = (
        Field(
            default="all_improvements",
            description=(
                "Selection strategy applied to the evaluated child proposals of "
                "one step. 'all_improvements' (default) keeps every proposal that "
                "passes the acceptance gate; 'best_improvement' keeps the single "
                "largest score delta; 'top_k' keeps the proposal_top_k largest "
                "passing deltas in stable order."
            ),
        )
    )
    proposal_top_k: int | None = Field(
        default=None,
        description=(
            "Number of proposals kept when proposal_selection='top_k'. "
            "Required only for that mode and must be within 1..P*N "
            "(num_parallel_proposals * mutations_per_parent). Rejected for "
            "the other selection modes."
        ),
    )
    minibatch_size: int = Field(
        default=3,
        description=(
            "Number of training examples per reflective mutation. "
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
        default=False,
        description=(
            "Enable content-addressed evaluation caches. Defaults off, matching "
            "GEPA Optimize Anything's conservative cache_evaluation behavior."
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
    num_sampled_groups: int | None = Field(
        default=None,
        description=(
            "Optional stratified sampler group count. When set with "
            "num_examples_per_group, each minibatch contains num_sampled_groups "
            "groups and num_examples_per_group examples from each selected group. "
            "Unset preserves the legacy stratified behavior of "
            "minibatch_size distinct groups with one example each."
        ),
    )
    num_examples_per_group: int | None = Field(
        default=None,
        description=(
            "Optional stratified sampler example count per sampled group. Must "
            "be set together with num_sampled_groups. The effective minibatch "
            "cardinality becomes num_sampled_groups * num_examples_per_group."
        ),
    )
    group_key_separator: str = Field(
        default="__",
        description=(
            "Separator used to derive a group key from each instance id for "
            "the stratified batch sampler: the id is split on this separator "
            "and the first part is taken as the group key (e.g. "
            "'group_alpha__case_3' -> 'group_alpha' when separator='__')."
        ),
    )
    frontier_type: Literal["instance", "objective", "hybrid", "cartesian"] = Field(
        default="hybrid",
        description=(
            "Pareto frontier dimensionality.  Mirrors GEPA's "
            "``FrontierType`` at ``src/gepa/core/state.py:22-23``.  Default "
            'is ``"hybrid"`` to match GEPA ``optimize_anything``\'s own '
            "default at ``src/gepa/optimize_anything.py:476`` — O.A. is the "
            "right parent for HELIX, not the base ``api.py`` path whose "
            'default is ``"instance"``.\n\n'
            '- ``"instance"``: one frontier key per example-id.  Matches '
            "HELIX's historical behaviour and GEPA's ``frontier_type="
            '"instance"``.\n'
            '- ``"objective"``: one frontier key per objective-name, '
            "score = mean of that objective across the valset.  Harvested "
            'from ``side_info["scores"]`` via ``helix_result``.\n'
            '- ``"hybrid"``: both instance and objective frontiers '
            "maintained; a candidate is retained if it survives on either.\n"
            '- ``"cartesian"``: one frontier key per (val_id, '
            "objective_name) pair.  Mirrors GEPA "
            "``_update_pareto_front_for_cartesian``.\n\n"
            "The acceptance gate stays positional on ``scores_list`` "
            "regardless of ``frontier_type`` (GEPA ``acceptance.py:39-48``); "
            "only the Pareto retention / parent-selection decision is "
            "multi-axis.  Non-``instance`` paths require ``helix_result`` "
            'to emit per-example ``side_info["scores"]`` dicts — without '
            "them the objective / cartesian frontiers raise instead of "
            "falling back to scalar semantics."
        ),
    )

    def model_post_init(self, __context: object) -> None:
        # ``max_workers`` bounds every eval/mutation pool and feeds the
        # ``num_parallel_proposals="auto"`` derivation, so validate it first.
        # The prior Pydantic model accepted invalid zero/negative values; the
        # unified P*N scheduler closes that gap.
        if self.max_workers < 1:
            raise ValueError(
                f"evolution.max_workers must be >= 1 (got {self.max_workers})"
            )
        # Resolve P (``num_parallel_proposals``) to a plain int.  The existing
        # value *is* P in the unified P*N scheduler: the current loop already
        # selects one parent per slot, so ``num_parallel_proposals=K`` is the
        # K-by-1 case.  GEPA parity: resolve ``"auto"`` to
        # ``max(1, max_workers // minibatch_size)`` once at construction time
        # so every downstream consumer sees a plain int.  Mirrors
        # /tmp/gepa-official/src/gepa/optimize_anything.py:1108-1116.
        raw_p = self.num_parallel_proposals
        if isinstance(raw_p, int):
            if raw_p < 1:
                raise ValueError(
                    f"evolution.num_parallel_proposals must be >= 1 (got {raw_p})"
                )
            effective_p = raw_p
        else:  # "auto"
            effective_p = max(1, self.max_workers // max(1, self.minibatch_size))
            self.num_parallel_proposals = effective_p
        # ``mutations_per_parent`` is N; N defaults to 1 so omitting it keeps
        # the K-by-1 behavior byte-for-byte.
        if self.mutations_per_parent < 1:
            raise ValueError(
                "evolution.mutations_per_parent must be >= 1 "
                f"(got {self.mutations_per_parent})"
            )
        # ``proposal_top_k`` is required and bounded to 1..P*N only for the
        # ``top_k`` selection strategy; it is rejected for the other modes.
        effective_pn = effective_p * self.mutations_per_parent
        if self.proposal_selection == "top_k":
            if self.proposal_top_k is None:
                raise ValueError(
                    "evolution.proposal_top_k is required when "
                    "evolution.proposal_selection='top_k'"
                )
            if not 1 <= self.proposal_top_k <= effective_pn:
                raise ValueError(
                    "evolution.proposal_top_k must be within "
                    f"1..{effective_pn} (P*N = num_parallel_proposals * "
                    "mutations_per_parent) when "
                    "evolution.proposal_selection='top_k' "
                    f"(got {self.proposal_top_k})"
                )
        elif self.proposal_top_k is not None:
            raise ValueError(
                "evolution.proposal_top_k is only valid when "
                "evolution.proposal_selection='top_k' (got "
                f"proposal_selection={self.proposal_selection!r})"
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
        if (self.num_sampled_groups is None) != (self.num_examples_per_group is None):
            raise ValueError(
                "evolution.num_sampled_groups and evolution.num_examples_per_group "
                "must be set together"
            )
        if self.num_sampled_groups is not None and self.batch_sampler != "stratified":
            raise ValueError(
                "evolution.num_sampled_groups and evolution.num_examples_per_group "
                "require evolution.batch_sampler='stratified'"
            )
        if self.num_sampled_groups is not None and self.num_sampled_groups < 1:
            raise ValueError(
                "evolution.num_sampled_groups must be >= 1 "
                f"(got {self.num_sampled_groups})"
            )
        if self.num_examples_per_group is not None and self.num_examples_per_group < 1:
            raise ValueError(
                "evolution.num_examples_per_group must be >= 1 "
                f"(got {self.num_examples_per_group})"
            )


class AgentConfig(BaseModel):
    """Configuration for the mutation backend integration."""

    model_config = ConfigDict(extra="forbid")

    backend: BackendName = "claude"
    model: str | None = None
    effort: str | None = None
    max_turns: int | None = None
    background: str | None = None


class SandboxConfig(BaseModel):
    """Configuration for OS-level subprocess isolation.

    When enabled, HELIX runs untrusted agent/evaluator commands in a Docker
    container whose only project mount is a temporary copy of the candidate
    worktree.  Agent-side filesystem changes are synced back explicitly after
    the backend exits; evaluator-side changes are discarded.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    evaluator: bool = False
    backend: Literal["docker"] = "docker"
    image: str | None = None
    network: Literal["bridge", "none", "host"] = "bridge"
    cpus: float | None = None
    memory: str | None = None
    timeout_seconds: int | None = None
    pids_limit: int | None = 512
    add_host_gateway: bool = False
    extra_hosts: dict[str, str] = Field(default_factory=dict)
    skip_special_files: bool = True
    omit_from_agent: list[str] = Field(default_factory=list)
    preserve_backend_transcripts: bool = True
    transcript_artifact_dir: str = ".helix_artifacts/backend_transcripts"
    claude_transcript_root: str = "/home/node/.claude/projects/-workspace"

    # --- Credential policy (0.3.0) -------------------------------------
    # ``auth`` is the explicit, no-fallback choice of credential path for a
    # sandboxed mutation agent.  It is deliberately NOT auto-detected: every
    # signal available at mount time was measured reporting success against
    # credentials a real request rejected, so auto-detection trades a
    # credential leak for an outage whenever it guesses wrong.
    #
    # ``None`` means "unset".  It is only valid while ``enabled`` is False,
    # where the auth mode is inert and today's environment-variable behaviour
    # is preserved exactly.
    auth: Literal["volume", "env"] | None = None
    auth_env_allow: list[str] = Field(
        default_factory=list,
        description=(
            "Credential variable names injected into the sandboxed mutation "
            'agent. Meaningful only when auth = "env". This REPLACES the '
            "per-backend table in helix/backends.py rather than unioning with "
            "it, so the allowlist lives in the file a reviewer reads."
        ),
    )
    agent_passthrough_env: list[str] = Field(
        default_factory=list,
        description=(
            "Non-credential variable names passed through to a sandboxed "
            "mutation agent. Top-level passthrough_env no longer grants agent "
            "scope under a sandbox; use this for deliberate agent-scope "
            "passthrough."
        ),
    )
    require_cli_match: bool = False

    def resolved_auth(self) -> str | None:
        """Return the effective auth mode, or None when the sandbox is off.

        The default is ``"volume"`` — the safe mode — so every existing
        sandboxed lane becomes strictly safer without editing anything.
        """
        if not self.enabled:
            return None
        return self.auth or "volume"


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
    evolution, agent, worktree) and validates compatibility constraints.
    """

    model_config = ConfigDict(extra="forbid")

    objective: str
    seed: str = "."
    rng_seed: int = 0  # GEPA parity: deterministic RNG for selection
    passthrough_env: list[str] = Field(
        default_factory=list,
        description=(
            "Environment variable names to pass through the env scrub into "
            "evaluator and agent subprocesses (e.g. "
            '["CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "HF_HOME"]).'
        ),
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Fixed environment variable values to inject into evaluator and "
            "agent subprocesses after passthrough_env. Use this for repeatable "
            "run-local endpoints and non-secret defaults."
        ),
    )
    evaluator: EvaluatorConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    seedless: SeedlessConfig = Field(default_factory=SeedlessConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    worktree: WorktreeConfig = Field(default_factory=WorktreeConfig)

    def model_post_init(self, __context: object) -> None:
        if self.seedless.enabled and not self.objective.strip():
            raise ValueError(
                "'objective' must be non-empty when seedless.enabled=True. "
                "The LLM needs the objective to generate an initial candidate."
            )
        if self.sandbox.enabled and self.sandbox.evaluator:
            if self.evaluator.sidecar is None:
                raise ValueError(
                    "Docker sandboxing requires [evaluator.sidecar] with image, "
                    "command, and endpoint."
                )
        _validate_agent_effort(self.agent)
        _validate_sandbox_auth(self)


def _validate_sandbox_auth(config: HelixConfig) -> None:
    """Enforce the sandbox credential-policy matrix at config load.

    Every check here fires before Docker is touched.  These are hard errors,
    not warnings: shipping a policy check as a warning is the failure mode
    that created this situation in the first place.
    """
    from helix.envpolicy import FORBIDDEN_AUTH_ENV_NAMES

    sandbox = config.sandbox

    # R1: a non-sandboxed run always uses environment auth. Setting ``auth``
    # there means the author believes something false about their own run.
    if not sandbox.enabled and sandbox.auth is not None:
        raise ValueError(
            "sandbox.auth is only meaningful when sandbox.enabled = true. "
            "Non-sandboxed runs always use environment-variable auth. "
            "Remove sandbox.auth."
        )

    mode = sandbox.resolved_auth()

    # R3: env mode with an empty allowlist injects nothing, so the author
    # asked for a credential path that cannot carry a credential.
    if mode == "env" and not sandbox.auth_env_allow:
        raise ValueError(
            'sandbox.auth = "env" requires a non-empty sandbox.auth_env_allow.'
        )

    if mode == "volume" and sandbox.auth_env_allow:
        raise ValueError(
            "sandbox.auth_env_allow is set but sandbox.auth = \"volume\", so no "
            "environment credentials will be injected. Set auth = \"env\" or "
            "remove auth_env_allow."
        )

    # R6: CLAUDE_CODE_OAUTH_TOKEN is never an env-mode mechanism. It does not
    # merely bypass refresh — it makes the credential accessor return a record
    # with a null refresh token, disabling refresh permanently. Trading a leak
    # for an unrefreshable auth path is not a trade HELIX offers.
    for name in sandbox.auth_env_allow:
        if name in FORBIDDEN_AUTH_ENV_NAMES:
            raise ValueError(
                f"{name} cannot be used for sandbox env auth: it disables "
                "OAuth token refresh permanently. Use ANTHROPIC_API_KEY, or "
                'use sandbox.auth = "volume".'
            )

    # T14: the mechanical, backend-independent, lane-independent replacement
    # for livebench's backend tripwire. The sidecar credential boundary
    # (fix 94f9751) requires the agent and sidecar credential sets to be
    # disjoint, and this enforces it for every lane regardless of backend.
    sidecar = config.evaluator.sidecar
    if sidecar is not None:
        agent_names = set(sandbox.auth_env_allow) | set(sandbox.agent_passthrough_env)
        shared = sorted(agent_names & set(sidecar.passthrough_env))
        if shared:
            names = ", ".join(repr(n) for n in shared)
            raise ValueError(
                f"{names} appears in both sandbox.auth_env_allow/"
                "agent_passthrough_env and evaluator.sidecar.passthrough_env. "
                "The sidecar credential boundary (fix 94f9751) requires these "
                "sets to be disjoint."
            )

    # A name that is absent from the environment is a warning, never an error:
    # CI dry-runs must still be able to load the config.
    import os as _os

    for name in sandbox.auth_env_allow:
        if name not in _os.environ:
            warnings.warn(
                f"sandbox.auth_env_allow names {name!r}, which is not set in "
                "the environment; the mutation agent will receive no value "
                "for it.",
                UserWarning,
                stacklevel=2,
            )


def _validate_agent_effort(agent: AgentConfig) -> None:
    """Warn on ``agent.effort`` settings that the active backend ignores or rejects.

    Two cases are surfaced:

    1. **Backend ignores the field** (``cursor`` / ``gemini``):
       ``effort`` is silently dropped by ``helix.mutator`` because those CLIs
       don't expose an equivalent flag.  Without a warning, users assume the
       knob is taking effect.
    2. **Effort-aware backend, but unknown value** (e.g. ``claude`` with
       ``effort = "extreme"``): the subprocess will fail with an opaque error
       buried in stderr.  Surfacing the typo at config load gives a far
       clearer signal.

    Both cases are warnings (not errors) so users can override / try forward-
    compatible values; the underlying CLI remains the source of truth for
    what's actually accepted.
    """
    if agent.effort is None:
        return

    backend = agent.backend
    display = backend_display_name(backend)

    if backend not in EFFORT_AWARE_BACKENDS:
        aware = ", ".join(sorted(EFFORT_AWARE_BACKENDS))
        warnings.warn(
            (
                f"agent.effort={agent.effort!r} is set, but the {display!r} "
                f"backend does not propagate an effort/reasoning level; "
                f"the value will be silently ignored. Either remove the "
                f"setting or switch to an effort-aware backend ({aware})."
            ),
            UserWarning,
            stacklevel=2,
        )
        return

    valid = EFFORT_VALID_VALUES.get(backend)
    if valid is not None and agent.effort not in valid:
        known = ", ".join(sorted(valid))
        warnings.warn(
            (
                f"agent.effort={agent.effort!r} is not a recognized value "
                f"for the {display!r} backend (known values: {known}). The "
                f"value will still be passed through to the CLI, but you "
                f"may see an opaque subprocess error if it's a typo."
            ),
            UserWarning,
            stacklevel=2,
        )


def load_config(path: Path) -> HelixConfig:
    """Load a HelixConfig from a TOML file.

    Supports both flat format (``objective = "..."`` at root) and the
    ``[project]`` section format (fields are merged into the root).
    """
    _load_dotenv_file(path.parent / ".env")
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
        merged = {
            **data["project"],
            **{k: v for k, v in data.items() if k != "project"},
        }
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
                print(
                    f"   Hint: Add '{error['loc'][-1]}' to your helix.toml",
                    file=sys.stderr,
                )
            elif error["type"] == "extra_forbidden":
                print(
                    f"   Hint: '{error['loc'][-1]}' is not a recognised key on this "
                    "section — check for typos or a misplaced sub-section "
                    "(e.g. a key that belongs under [evolution] placed under [evaluator]).",
                    file=sys.stderr,
                )
            elif "type" in str(error["type"]):
                print(
                    "   Hint: Check that the value is the correct type", file=sys.stderr
                )
            print(file=sys.stderr)
        sys.exit(1)
