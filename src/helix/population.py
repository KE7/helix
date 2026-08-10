"""HELIX population management: candidates, eval results, and Pareto frontier.

Implements GEPA-style unbounded frontier with coverage-based dominance.
Algorithm translated line-for-line from gepa.gepa_utils (gepa-ai/gepa).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from helix.display import UsageStats

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from helix.state import BudgetState


FrontierType = Literal["instance", "objective", "hybrid", "cartesian"]


class MissingObjectiveScoresError(ValueError):
    """Raised by ``ParetoFrontier`` when an objective-bearing frontier mode
    cannot consume the supplied ``EvalResult.objective_scores``.

    Covers malformed positional data and the case where an objective-only
    frontier has no active objective entries at parent-selection time.
    Subclassing :class:`ValueError` keeps callers that catch ``ValueError``
    working unchanged. Empty objective slots are tolerated, matching
    upstream GEPA's update no-op; objective-only modes still fail clearly
    when selection cannot proceed without an objective axis.
    """


@dataclass
class Candidate:
    """Represents an evolved code candidate.

    Tracks identity, git worktree location, generation number,
    lineage, and the operation that created it (mutation or merge).
    """
    id: str
    worktree_path: str
    branch_name: str
    generation: int
    parent_id: str | None
    parent_ids: list[str]
    operation: str
    usage: UsageStats = field(default_factory=UsageStats)
    # A validated, untracked self-report captured after the mutation backend exits.
    # It is deliberately not part of the candidate's git snapshot.
    change_summary: dict[str, str] | None = None


@dataclass
class EvalResult:
    """Evaluation results for a candidate.

    Mirrors GEPA's :class:`gepa.core.adapter.EvaluationBatch` where it
    can.  Key invariants:

    - ``instance_scores`` is keyed **only** by HELIX's internal example
      ids — the same strings HELIX writes to
      ``{worktree}/helix_batch.json`` pre-invocation.  Never aggregate
      metric names.  Previously this was the footgun: an evaluator
      keying by ``task__metric`` instead of ``task__trialN`` silently
      zero-filled every requested id in the minibatch gate.  The
      per-example ``helix_result`` parser owns the id-keying now so
      evaluators never type a HELIX-internal id.
    - ``per_example_side_info`` is positional to ``instance_scores``
      by id order (same order as ``helix_batch.json``).  GEPA analogue:
      ``EvaluationBatch.trajectories``
      (``src/gepa/core/adapter.py::EvaluationBatch.trajectories``).
      Carries freeform per-example diagnostics; rendered into the
      mutation prompt's Diagnostics section by
      :func:`helix.mutator._render_per_example_diagnostics` (GEPA
      ``format_samples`` parity at
      ``src/gepa/strategies/instruction_proposal.py::format_samples``).
    """
    candidate_id: str
    scores: dict[str, float]          # aggregate/summary scores
    # Arbitrary string info (metadata) collected from the evaluator.
    # Mirrors GEPA O.A. ``EvaluationBatch``'s side-info bag — every
    # HELIX-specific signal that downstream tooling might want to
    # inspect lives here in the same string-keyed dict rather than as
    # a typed field on this dataclass.  That keeps the cross-language
    # GEPA contract intact.
    #
    # Reserved-key convention
    # -----------------------
    # Inside this otherwise GEPA-shaped result, keys starting with an
    # underscore (``_<name>``) are HELIX-internal sentinels: written by
    # ``executor._collect_asi``, consumed elsewhere in HELIX (e.g. the
    # mutation prompt builder), and *not* part of the evaluator-facing
    # surface.  They are:
    #
    # * stringified (``asi[k]`` is always ``str`` — callers must parse);
    # * filtered out of any catch-all rendering surface (the mutation
    #   prompt's "Extra Evaluator Info" section in
    #   :func:`helix.mutator.build_mutation_prompt` is the canonical
    #   example);
    # * HELIX-internal, not evaluator-facing — evaluator code (the
    #   user's ``evaluate.py`` or any GEPA-style downstream consumer)
    #   must never set or read these keys.  HELIX owns them and may
    #   rename them or change their string format without notice.
    #
    # Currently reserved:
    #
    # * ``_returncode`` — evaluator subprocess exit code (str), used by
    #   the mutator to gate stdout/stderr fallback rendering.
    #
    # Add new sentinels here (and to the mutator's filter list at
    # ``mutator.build_mutation_prompt``) when introducing more.
    #
    # Non-underscore keys (``stdout``, ``stderr``, ``log``, ``extra_N``,
    # arbitrary evaluator-emitted notes) ARE the evaluator-facing
    # surface — they round-trip unchanged to GEPA-style consumers and
    # are stable for evaluator authors to depend on.
    asi: dict[str, str]
    instance_scores: dict[str, float] # per-instance scores, keyed by HELIX example-id
    # Legacy batch-level diagnostics dict.  No longer populated by the
    # ``helix_result`` parser (which uses ``per_example_side_info``);
    # kept for back-compat with any other path that still sets it and
    # for the mutation prompt's existing Diagnostics section.
    side_info: dict[str, Any] | None = None
    # Per-example freeform side_info (GEPA O.A. evaluator contract:
    # ``(score, side_info)`` per example).  Positional to
    # ``instance_scores`` by helix_batch.json id order.  Rendered into
    # the mutation prompt's Diagnostics section by
    # :func:`helix.mutator._render_per_example_diagnostics`.
    per_example_side_info: list[dict[str, Any]] | None = None
    # Per-example objective-axis harvest: each slot is
    # ``side_info_i.get("scores", {})`` filtered down to
    # ``{str: float}`` entries.  GEPA analogue:
    # :attr:`gepa.core.adapter.EvaluationBatch.objective_scores`.
    # Feeds the multi-axis Pareto
    # frontier when ``evolution.frontier_type`` ∈ {"objective",
    # "hybrid", "cartesian"}; harmless on the default "instance" path.
    objective_scores: list[dict[str, float]] | None = None

    def aggregate_score(self) -> float:
        """Return mean of instance scores, or 0.0 if none."""
        if not self.instance_scores:
            return 0.0
        return sum(self.instance_scores.values()) / len(self.instance_scores)

    def sum_score(self) -> float:
        """Return sum of instance scores (GEPA parity: acceptance uses sum)."""
        return sum(self.instance_scores.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Optional fields are omitted when ``None`` so on-disk evaluations
        stay byte-identical when they don't carry the newer data.
        """
        d: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "scores": self.scores,
            "instance_scores": self.instance_scores,
            "asi": self.asi,
        }
        if self.side_info is not None:
            d["side_info"] = self.side_info
        if self.per_example_side_info is not None:
            d["per_example_side_info"] = self.per_example_side_info
        if self.objective_scores is not None:
            d["objective_scores"] = self.objective_scores
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        """Deserialize from a dict (as stored in evaluations/*.json)."""
        return cls(
            candidate_id=data["candidate_id"],
            scores=data.get("scores", {}),
            instance_scores=data.get("instance_scores", {}),
            asi=data.get("asi", {}),
            side_info=data.get("side_info"),
            per_example_side_info=data.get("per_example_side_info"),
            objective_scores=data.get("objective_scores"),
        )


@dataclass
class CandidateSummary:
    """Programmatic summary of one evolved candidate in a HELIX result."""

    candidate: Candidate
    aggregate_score: float
    sum_score: float
    scores: dict[str, float]
    instance_scores: dict[str, float]
    objective_scores: list[dict[str, float]] | None
    parents: list[str]
    operation: str
    generation: int
    discovered_at_evaluation: int | None = None
    is_non_dominated: bool = False

    @property
    def id(self) -> str:
        """Candidate id convenience alias."""
        return self.candidate.id

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": self.candidate.id,
            "worktree_path": self.candidate.worktree_path,
            "branch_name": self.candidate.branch_name,
            "generation": self.generation,
            "parents": self.parents,
            "operation": self.operation,
            "usage": self.candidate.usage.to_dict(),
            "aggregate_score": self.aggregate_score,
            "sum_score": self.sum_score,
            "scores": self.scores,
            "instance_scores": self.instance_scores,
            "objective_scores": self.objective_scores,
            "discovered_at_evaluation": self.discovered_at_evaluation,
            "is_non_dominated": self.is_non_dominated,
        }


@dataclass(frozen=True)
class HelixResult:
    """Immutable snapshot returned by a HELIX evolution run.

    Mirrors GEPA-style programmatic access without changing the on-disk
    worktree/state model: callers can inspect the best candidate, every
    accepted candidate, lineage, scores, budgets, frontier axes, and
    run identity without scraping CLI output or JSON files.

    GEPA parity: matches the immutability shape of ``GEPAResult``
    (``src/gepa/core/result.py``) — top-level ``frozen=True`` with plain
    list/dict fields and ``@property`` views recomputed on each access.
    The ``frozen=True`` dataclass blocks attribute reassignment; the
    construction site (``_build_helix_result``) takes shallow defensive
    copies of every collection so no field aliases the live
    :class:`EvolutionState`.

    Canonical per-candidate data lives on :attr:`candidate_summaries`
    (sorted by ``(generation, id)``).  ``parents``, ``aggregate_scores``,
    ``sum_scores``, ``instance_scores``, and ``objective_scores`` are
    plain ``@property`` views (not cached) — they recompute from
    ``candidate_summaries`` on every access, so they cannot drift out of
    sync with the canonical list.
    """

    best_candidate: Candidate
    best_result: EvalResult | None
    candidates: list[Candidate]
    candidate_summaries: list[CandidateSummary]
    # ``frontier_ids`` preserves insertion order from
    # ``EvolutionState.frontier``; ``non_dominated_ids`` is sorted for
    # stable diffing across runs.  Both intentional.
    frontier_ids: list[str]
    non_dominated_ids: list[str]
    frontier_type: FrontierType
    budget: "BudgetState"
    discovery_counts: dict[str, int]
    run_dir: str
    seed: int | None
    config_hash: str

    @property
    def id(self) -> str:
        """Compatibility alias for callers that previously received Candidate."""
        return self.best_candidate.id

    @property
    def parents(self) -> dict[str, list[str]]:
        """Lineage parent ids per candidate, derived from ``candidate_summaries``."""
        return {s.id: list(s.parents) for s in self.candidate_summaries}

    @property
    def aggregate_scores(self) -> dict[str, float]:
        """Per-candidate mean instance score (GEPA aggregate semantics)."""
        return {s.id: s.aggregate_score for s in self.candidate_summaries}

    @property
    def sum_scores(self) -> dict[str, float]:
        """Per-candidate sum of instance scores (GEPA acceptance semantics)."""
        return {s.id: s.sum_score for s in self.candidate_summaries}

    @property
    def instance_scores(self) -> dict[str, dict[str, float]]:
        """Per-candidate per-instance score map."""
        return {s.id: dict(s.instance_scores) for s in self.candidate_summaries}

    @property
    def objective_scores(self) -> dict[str, list[dict[str, float]] | None]:
        """Per-candidate per-example objective slots (None when not harvested)."""
        return {
            s.id: (list(s.objective_scores) if s.objective_scores is not None else None)
            for s in self.candidate_summaries
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        ``budget`` is rendered via :func:`dataclasses.asdict` so any new
        :class:`helix.state.BudgetState` field shows up automatically.
        """
        return {
            "best_candidate_id": self.best_candidate.id,
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "candidates": [summary.to_dict() for summary in self.candidate_summaries],
            "parents": self.parents,
            "aggregate_scores": self.aggregate_scores,
            "sum_scores": self.sum_scores,
            "instance_scores": self.instance_scores,
            "objective_scores": self.objective_scores,
            "frontier_ids": list(self.frontier_ids),
            "non_dominated_ids": list(self.non_dominated_ids),
            "frontier_type": self.frontier_type,
            "budget": asdict(self.budget),
            "discovery_counts": dict(self.discovery_counts),
            "run_dir": self.run_dir,
            "seed": self.seed,
            "config_hash": self.config_hash,
        }


class ParetoFrontier:
    """GEPA-style unbounded frontier with coverage-based dominance.

    Algorithm translated line-for-line from ``gepa.gepa_utils``:

    1. **Unbounded** — no frontier cap.  Every evaluated candidate is
       stored (``program_candidates`` in GEPA is append-only).
    2. **Per-key best tracking** — ``_per_key_best[val_id]`` is the set of
       candidate IDs that achieved the best score for that validation key.
       Mirrors ``program_at_pareto_front_valset`` in ``GEPAState``.
    3. **Coverage-based dominance** — ``remove_dominated_programs()`` uses
       iterative fixed-point elimination: sort candidates by score ascending,
       repeatedly scan for any candidate whose frontier-key coverage is fully
       covered by the remaining non-dominated set, mark it dominated, restart.
    4. **Parent selection** — ``select_program_candidate_from_pareto_front()``
       calls ``remove_dominated_programs()``, counts per-key frequency of
       surviving programs, builds a flat sampling list, picks uniformly.

    Multi-axis frontier (GEPA ``FrontierType`` parity)
    ---------------------------------------------------
    The ``frontier_type`` constructor arg mirrors GEPA's ``FrontierType``
    literal in ``core/state.py``:

    - ``"instance"`` (default): per-example-id keyspace, built from
      ``EvalResult.instance_scores``.  Matches HELIX's historical
      behaviour and GEPA's ``frontier_type="instance"`` path.
    - ``"objective"``: per-objective-name keyspace, values = mean of
      that objective across the valset, built from
      ``EvalResult.objective_scores``.  GEPA's
      ``GEPAState._update_objective_pareto_front``.
    - ``"hybrid"``: union of the instance and objective keyspaces.
      A candidate survives if it's non-dominated on the combined
      keyset.
    - ``"cartesian"``: per ``(val_id, objective_name)`` keyspace.
      GEPA's ``GEPAState._update_pareto_front_for_cartesian``.

    The **acceptance gate stays positional** on ``scores_list``
    regardless of ``frontier_type`` (GEPA's default
    ``StrictImprovementAcceptance`` in ``strategies/acceptance.py``
    sums subsample scores positionally); only the Pareto retention /
    parent-selection decision is multi-axis.

    Implementation: each axis has its own ``_..._best`` /
    ``_..._best_score`` dict, populated on ``add()``.
    :meth:`_active_frontier` returns the merged keyspace for the
    currently selected ``frontier_type``; dominance / selection run
    against that merged dict via the existing
    :func:`_remove_dominated_programs` helper.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
        frontier_type: FrontierType = "instance",
    ) -> None:
        # GEPA parity: seeded RNG for deterministic parent selection.
        self._rng = rng if rng is not None else random.Random(0)
        self._frontier_type: FrontierType = frontier_type
        # Append-only storage (never pruned) — mirrors GEPA's program_candidates list
        self._candidates: dict[str, Candidate] = {}
        self._results: dict[str, EvalResult] = {}
        # ``"instance"`` axis — mirrors GEPAState.program_at_pareto_front_valset
        # (val_id → {candidate IDs with best score}) and pareto_front_valset
        # (val_id → best score).
        self._per_key_best: dict[str, set[str]] = {}
        self._per_key_best_score: dict[str, float] = {}
        # ``"objective"`` axis — mirrors GEPAState.program_at_pareto_front_objectives
        # (objective_name → {candidates tied for best mean-across-valset})
        # and ``objective_pareto_front`` (objective_name → best mean).
        self._objective_best: dict[str, set[str]] = {}
        self._objective_best_score: dict[str, float] = {}
        # ``"cartesian"`` axis — mirrors
        # GEPAState.program_at_pareto_front_cartesian / pareto_front_cartesian
        # ((val_id, objective_name) → best).  Keys are encoded as
        # ``f"{val_id}::{objective_name}"`` so the existing
        # ``_remove_dominated_programs`` helper (which takes
        # ``dict[str, set[str]]``) works unmodified.
        self._cartesian_best: dict[str, set[str]] = {}
        self._cartesian_best_score: dict[str, float] = {}

    @property
    def frontier_type(self) -> FrontierType:
        """Selected Pareto dimensionality (GEPA parity)."""
        return self._frontier_type

    # ------------------------------------------------------------------
    # Public read-only accessors
    # ------------------------------------------------------------------
    # ``_candidates`` and ``_results`` are append-only internal storage;
    # external callers (e.g. ``_build_helix_result``, ``render_frontier_table``,
    # diagnostics replays) use these read-only views so the storage
    # layout can change without breaking them.

    @property
    def candidates(self) -> Mapping[str, Candidate]:
        """Read-only view of all stored candidates keyed by id."""
        return MappingProxyType(self._candidates)

    @property
    def results(self) -> Mapping[str, EvalResult]:
        """Read-only view of stored evaluation results keyed by candidate id."""
        return MappingProxyType(self._results)

    def get_result(self, candidate_id: str) -> EvalResult | None:
        """Return the stored ``EvalResult`` for *candidate_id* or ``None``."""
        return self._results.get(candidate_id)

    def iter_results(self) -> Iterator[tuple[str, EvalResult]]:
        """Yield ``(candidate_id, EvalResult)`` pairs for all known candidates.

        Iteration order matches the underlying dict's insertion order, which
        in turn mirrors the order in which candidates were ``add``-ed — useful
        for diagnostics that want a stable replay of population history.
        """
        return iter(self._results.items())

    def has_results(self) -> bool:
        """True iff at least one candidate has been added with a result."""
        return bool(self._results)

    # ------------------------------------------------------------------
    # Mutation helpers — mirrors GEPAState._update_pareto_front_for_val_id
    # ------------------------------------------------------------------

    def add(self, candidate: Candidate, result: EvalResult) -> None:
        """Add a candidate and its evaluation result.  Population is append-only."""
        self._validate_objective_scores(candidate.id, result)
        self._candidates[candidate.id] = candidate
        self._results[candidate.id] = result
        self._update_per_key(candidate.id, result)
        # Multi-axis updates are no-ops when the frontier_type doesn't
        # use them, but always populating keeps diagnostics useful and
        # lets a caller inspect any axis post-hoc.
        self._update_objective(candidate.id, result)
        self._update_cartesian(candidate.id, result)

    def _validate_objective_scores(self, cid: str, result: EvalResult) -> None:
        """Validate objective-score shape without rejecting empty axes.

        Upstream GEPA raises a plain ``ValueError`` at both the seed-time
        check in ``GEPAState.__init__`` and the per-update check in
        ``GEPAState.update_state_with_new_program`` when
        ``valset_evaluation.objective_scores_by_val_id`` is missing/empty
        for an objective-bearing ``frontier_type``.  HELIX diverges here:
        rather than raising immediately at add-time, it emits a warning
        and no-ops, deferring the hard failure to selection time, where
        objective-only selection raises a typed, actionable
        :class:`MissingObjectiveScoresError` if there is no objective
        frontier to sample.

        The positional length check remains a structural invariant for
        HELIX's list-shaped objective field.
        Empty objective mappings therefore emit a warning and no-op here;
        hybrid selection continues on its instance axis, while objective
        and cartesian selection report the missing axis later.
        """
        if self._frontier_type == "instance":
            return
        objective_scores = result.objective_scores
        if objective_scores is None or not objective_scores or not any(objective_scores):
            logger.warning(
                "frontier_type=%r candidate %r has no objective axes; "
                'objective_scores are empty or absent. Emit side_info["scores"] '
                'or use frontier_type="instance".',
                self._frontier_type,
                cid,
            )
            return
        if len(objective_scores) != len(result.instance_scores):
            raise MissingObjectiveScoresError(
                f"frontier_type={self._frontier_type!r} requires objective_scores "
                f"length to match instance_scores for candidate {cid!r} "
                f"({len(objective_scores)} != {len(result.instance_scores)})."
            )

    def _update_per_key(self, cid: str, result: EvalResult) -> None:
        """Incrementally update per-key best tracking for a single candidate.

        Mirrors ``GEPAState._update_pareto_front_for_val_id``.
        """
        for key, score in result.instance_scores.items():
            prev_score = self._per_key_best_score.get(key, float("-inf"))
            if score > prev_score:
                self._per_key_best[key] = {cid}
                self._per_key_best_score[key] = score
            elif score == prev_score:
                front = self._per_key_best.setdefault(key, set())
                front.add(cid)

    def _update_objective(self, cid: str, result: EvalResult) -> None:
        """Update the per-objective frontier using mean-across-valset.

        HELIX computes the per-candidate mean of each objective across
        its own ``objective_scores`` list, then tracks the best mean
        (and any candidates tied for it) per objective name — the same
        two-stage shape as upstream GEPA, which aggregates per-val_id
        objective scores into a per-program mean via
        ``GEPAState._aggregate_objective_scores`` before handing that
        pre-aggregated dict to ``GEPAState._update_objective_pareto_front``
        for the best-tracking step.
        """
        # Empty or absent objective scores are an intentional no-op, matching
        # GEPA's ``if not objective_scores: return`` update behavior.
        if result.objective_scores is None or not result.objective_scores:
            return
        aggregated: dict[str, list[float]] = {}
        for slot in result.objective_scores:
            for name, score in slot.items():
                aggregated.setdefault(name, []).append(float(score))
        for name, vals in aggregated.items():
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            prev = self._objective_best_score.get(name, float("-inf"))
            if mean > prev:
                self._objective_best[name] = {cid}
                self._objective_best_score[name] = mean
            elif mean == prev:
                self._objective_best.setdefault(name, set()).add(cid)

    def _update_cartesian(self, cid: str, result: EvalResult) -> None:
        """Update the (val_id, objective_name) frontier.

        Mirrors GEPA's ``GEPAState._update_pareto_front_for_cartesian``:
        each per-example ``objective_scores[i]``
        slot is combined with the corresponding ``val_id`` (from
        ``instance_scores``' ordered keys) to form a tuple key.  We
        encode the tuple as ``f"{val_id}::{objective_name}"`` so the
        existing ``dict[str, set[str]]`` dominance helper works.
        """
        if result.objective_scores is None or not result.objective_scores:
            return
        # ``instance_scores`` preserves helix_batch.json id order by
        # construction in ``helix_result.parse``; ``objective_scores``
        # is positional to the same id list.  Zip together.
        val_ids = list(result.instance_scores.keys())
        if len(val_ids) != len(result.objective_scores):
            # Defense-in-depth: ``add()`` and ``update_scores()`` validate
            # this mismatch up front, but retain the invariant if a caller
            # swaps in a malformed ``EvalResult`` during a rebuild.
            raise MissingObjectiveScoresError(
                f"cartesian frontier requires objective_scores length to match "
                f"instance_scores for candidate {cid!r} "
                f"({len(result.objective_scores)} != {len(val_ids)})."
            )
        for val_id, slot in zip(val_ids, result.objective_scores):
            for obj_name, score in slot.items():
                key = f"{val_id}::{obj_name}"
                prev = self._cartesian_best_score.get(key, float("-inf"))
                if float(score) > prev:
                    self._cartesian_best[key] = {cid}
                    self._cartesian_best_score[key] = float(score)
                elif float(score) == prev:
                    self._cartesian_best.setdefault(key, set()).add(cid)

    def update_scores(self, result: EvalResult) -> None:
        """Update the evaluation result for an existing candidate and rebuild tracking."""
        self._validate_objective_scores(result.candidate_id, result)
        self._results[result.candidate_id] = result
        self._rebuild_axes()

    def candidate_ids(self) -> list[str]:
        """Return all evaluated candidate ids in insertion order.

        Stable, JSON-serializable view used by callers (e.g.
        ``evolution._sync_frontier_state``) that need to mirror the
        append-only candidate set without reaching into ``_candidates``.
        """
        return list(self._candidates.keys())

    def _rebuild_axes(self) -> None:
        """Rebuild all per-axis best tracking from scratch.

        Called after a score update rewrites one candidate's
        ``EvalResult``.  Rebuilds every axis (instance, objective,
        cartesian) so a single consistent snapshot is preserved —
        hence the name; the original ``_rebuild_per_key`` pre-dated
        the multi-axis frontier and only touched ``_per_key_best``.
        """
        self._per_key_best.clear()
        self._per_key_best_score.clear()
        self._objective_best.clear()
        self._objective_best_score.clear()
        self._cartesian_best.clear()
        self._cartesian_best_score.clear()
        for cid, result in self._results.items():
            self._update_per_key(cid, result)
            self._update_objective(cid, result)
            self._update_cartesian(cid, result)

    # ------------------------------------------------------------------
    # Active frontier — merged keyspace per ``frontier_type``
    # ------------------------------------------------------------------

    def _active_frontier(self) -> dict[str, set[str]]:
        """Return the merged per-key-best dict for the active axis.

        Key collisions are avoided by prefixing per-axis keys when the
        hybrid path unions multiple keyspaces.  Consumers treat the
        returned dict opaquely via :func:`_remove_dominated_programs`,
        so the prefix is internal implementation detail.
        """
        if self._frontier_type == "instance":
            # Back-compat: return the raw instance dict so existing
            # tests asserting on ``_per_key_best`` keys (``"i1"``,
            # ``"i2"``, etc.) keep working through the same object.
            return self._per_key_best
        if self._frontier_type == "objective":
            return self._objective_best
        if self._frontier_type == "cartesian":
            return self._cartesian_best
        # "hybrid": union of instance ∪ objective keyspaces — GEPA's
        # default (``EngineConfig.frontier_type`` in ``gepa_launcher.py``).
        merged: dict[str, set[str]] = {}
        for k, v in self._per_key_best.items():
            merged[f"inst::{k}"] = v
        for k, v in self._objective_best.items():
            merged[f"obj::{k}"] = v
        return merged

    def active_frontier_snapshot(self) -> dict[str, list[str]]:
        """Return a JSON-ready copy of the active fronts by frontier key."""
        return {
            key: sorted(front)
            for key, front in sorted(self._active_frontier().items())
        }

    # ------------------------------------------------------------------
    # GEPA coverage-based dominance
    # Translated line-for-line from gepa.gepa_utils.{is_dominated,
    # remove_dominated_programs}
    # ------------------------------------------------------------------

    @staticmethod
    def _is_dominated(
        y: str,
        programs: set[str],
        program_at_pareto_front_valset: dict[str, set[str]],
    ) -> bool:
        """GEPA ``is_dominated(y, programs, program_at_pareto_front_valset)``.

        Returns True when *programs* already covers every frontier key that
        *y* participates in.
        """
        y_fronts = [front for front in program_at_pareto_front_valset.values() if y in front]
        for front in y_fronts:
            found_dominator_in_front = False
            for other_prog in front:
                if other_prog in programs:
                    found_dominator_in_front = True
                    break
            if not found_dominator_in_front:
                return False
        return True

    @staticmethod
    def _remove_dominated_programs(
        program_at_pareto_front_valset: dict[str, set[str]],
        scores: dict[str, float] | None = None,
    ) -> tuple[set[str], dict[str, set[str]]]:
        """GEPA ``remove_dominated_programs(program_at_pareto_front_valset, scores)``.

        Iterative fixed-point elimination:
        1. Collect all programs appearing in any frontier key.
        2. Sort by ``(score, candidate_id)`` ascending (worst first — lower-scoring
           programs are checked first and more likely to be eliminated).  The id
           is part of the sort key, not decoration: the programs are collected by
           walking ``set`` fronts, so equal-scoring candidates would otherwise be
           ordered by the per-process string-hash salt, and step 3 stops at the
           *first* dominated candidate it finds.
        3. Repeatedly scan: for each non-eliminated candidate y, check if y is
           dominated by ``set(programs) - {y} - dominated``.  If dominated,
           mark and restart scan.  Repeat until a full pass finds nothing.
        4. Return (dominator set, cleaned per-key-best dict with dominated
           programs removed from every front).
        """
        freq: dict[str, int] = {}
        for front in program_at_pareto_front_valset.values():
            for p in front:
                freq[p] = freq.get(p, 0) + 1

        dominated: set[str] = set()
        programs = list(freq.keys())

        if scores is None:
            scores = dict.fromkeys(programs, 1.0)

        programs = sorted(programs, key=lambda x: (scores.get(x, 0.0), x), reverse=False)

        found_to_remove = True
        while found_to_remove:
            found_to_remove = False
            for y in programs:
                if y in dominated:
                    continue
                if ParetoFrontier._is_dominated(
                    y,
                    set(programs).difference({y}).difference(dominated),
                    program_at_pareto_front_valset,
                ):
                    dominated.add(y)
                    found_to_remove = True
                    break

        dominators = set(p for p in programs if p not in dominated)

        new_program_at_pareto_front_valset = {
            val_id: {prog for prog in front if prog in dominators}
            for val_id, front in program_at_pareto_front_valset.items()
        }

        return dominators, new_program_at_pareto_front_valset

    def get_non_dominated(self) -> set[str]:
        """Return the non-dominated set via GEPA's iterative fixed-point elimination.

        Dominance is computed against :meth:`_active_frontier`, which
        dispatches on ``frontier_type``.  Membership in the per-key fronts
        is what decides dominance (see :meth:`_is_dominated`); the
        aggregate score passed here only *orders* the elimination scan,
        so it selects which of several mutually-eliminable candidates is
        dropped first, not what "dominated" means.

        GEPA parity: rank by ``aggregate_score()`` (the mean), not by
        ``sum_score()``.  Upstream feeds
        ``gepa.strategies.candidate_selector.ParetoCandidateSelector``
        from ``gepa.core.state.GEPAState.per_program_tracked_scores``,
        which returns ``get_program_average_val_subset(program_idx)[0]``
        — an average, not a sum.  Sum and mean rank pools identically
        only when every candidate carries the same instance count.
        """
        scores = {cid: r.aggregate_score() for cid, r in self._results.items()}
        dominators, _ = self._remove_dominated_programs(
            self._active_frontier(), scores,
        )
        return dominators

    def is_dominated(self, candidate_id: str) -> bool:
        """Return True if *candidate_id* is dominated (not in the non-dominated set)."""
        if candidate_id not in self._results:
            return False
        non_dominated = self.get_non_dominated()
        return candidate_id not in non_dominated

    def get_dominated(self) -> list[str]:
        """Return ids of all dominated candidates."""
        non_dominated = self.get_non_dominated()
        return [cid for cid in self._candidates if cid not in non_dominated]

    # ------------------------------------------------------------------
    # Selection — mirrors gepa.gepa_utils.select_program_candidate_from_pareto_front
    # ------------------------------------------------------------------

    def _instance_wins(self) -> dict[str, set[str]]:
        """Transpose of the active frontier: candidate → set of keys
        where it appears in the best set.

        Used by :meth:`select_complementary_pair` to find pairs with
        minimal overlap in best-key coverage.  Name retained for
        back-compat; now reflects the active frontier type.
        """
        active = self._active_frontier()
        wins: dict[str, set[str]] = {cid: set() for cid in self._candidates}
        for key, front in active.items():
            for cid in front:
                if cid in wins:
                    wins[cid].add(key)
        return wins

    def select_parent(self) -> Candidate:
        """GEPA ``select_program_candidate_from_pareto_front()``.

        1. Run ``remove_dominated_programs()`` over :meth:`_active_frontier`
           with mean aggregate scores, matching the ``scores`` argument
           upstream's ``ParetoCandidateSelector`` passes to
           ``gepa.gepa_utils.select_program_candidate_from_pareto_front``.
        2. Count per-key frequency of surviving programs in the *cleaned*
           frontier (dominated programs stripped from every front).
        3. Build a flat sampling list where each program appears *freq* times.
        4. Pick uniformly at random (``rng.choice(sampling_list)`` in GEPA).

        The sampling list is built in sorted candidate-id order so a
        seeded ``rng`` reproduces the same parent across processes; the
        cleaned fronts are ``set`` objects and HELIX keys them by ``str``
        ids, whose hashes are salted per interpreter.  (Upstream keys
        programs by ``int`` index, so upstream never had this exposure.)
        """
        if not self._candidates:
            raise ValueError("Frontier is empty — cannot select parent.")

        # GEPA parity: rank by mean, as GEPAState.per_program_tracked_scores does.
        scores = {cid: r.aggregate_score() for cid, r in self._results.items()}
        _, cleaned_per_key_best = self._remove_dominated_programs(
            self._active_frontier(), scores,
        )

        # Count frequency in cleaned frontier (GEPA: program_frequency_in_validation_pareto_front)
        program_frequency: dict[str, int] = {}
        for testcase_front in cleaned_per_key_best.values():
            for cid in testcase_front:
                if cid not in program_frequency:
                    program_frequency[cid] = 0
                program_frequency[cid] += 1

        # Build flat sampling list (GEPA: sampling_list).  Iterate in
        # sorted id order: ``program_frequency`` is populated by walking
        # ``set`` fronts, so its insertion order varies with the
        # per-process string-hash salt and would otherwise reach
        # ``rng.choice`` below.
        sampling_list = [
            cid
            for cid in sorted(program_frequency)
            for _ in range(program_frequency[cid])
        ]

        # Instance mode keeps HELIX's score-only fallback. Hybrid mode can
        # still select from its instance axis when objective updates no-op.
        # Objective/cartesian modes have no coherent fallback, so retain a
        # clear typed error rather than importing GEPA's bare assertion or
        # silently changing their semantics.
        if not sampling_list:
            if self._frontier_type != "instance":
                raise MissingObjectiveScoresError(
                    f"frontier_type={self._frontier_type!r} has no objective "
                    "axis and cannot select a parent; emit per-example "
                    'side_info["scores"] or use frontier_type="instance".'
                )
            sampling_list = list(self._candidates.keys())
        return self._candidates[self._rng.choice(sampling_list)]

    def select_complementary_pair(self) -> tuple[Candidate, Candidate]:
        """Select two candidates with minimal overlap in best-instance sets."""
        if len(self._candidates) < 2:
            raise ValueError("Need at least 2 candidates to select a pair.")

        wins = self._instance_wins()
        candidate_ids = list(self._candidates.keys())

        best_pair: tuple[str, str] | None = None
        best_complement = -1

        for i in range(len(candidate_ids)):
            for j in range(i + 1, len(candidate_ids)):
                a, b = candidate_ids[i], candidate_ids[j]
                overlap = len(wins.get(a, set()) & wins.get(b, set()))
                union = len(wins.get(a, set()) | wins.get(b, set()))
                complement = union - overlap
                if complement > best_complement:
                    best_complement = complement
                    best_pair = (a, b)

        assert best_pair is not None
        return self._candidates[best_pair[0]], self._candidates[best_pair[1]]

    def best(self) -> Candidate:
        """Return the candidate with the highest aggregate score."""
        if not self._candidates:
            raise ValueError("Frontier is empty — cannot find best.")
        return max(
            self._candidates.values(),
            key=lambda c: self._results[c.id].aggregate_score() if c.id in self._results else -1.0,
        )

    def signature(self) -> str:
        """Hash of the frontier state for convergence detection."""
        state = {
            cid: sorted(r.instance_scores.items())
            for cid, r in self._results.items()
        }
        serialised = json.dumps(state, sort_keys=True)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self._candidates)

    def __contains__(self, candidate_id: str) -> bool:
        return candidate_id in self._candidates


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def gen_from_id(candidate_id: str) -> int:
    """Parse the generation number from a candidate id like 'g3-s1'.

    Returns 0 if the id format is invalid or cannot be parsed.
    """
    try:
        return int(candidate_id.split("-")[0].lstrip("g"))
    except (IndexError, ValueError):
        return 0


def format_instance_scores_table(instance_scores: dict[str, float]) -> str:
    """Format instance scores as a readable markdown table.

    Sorts instances by score (worst first) for quick identification of
    problem areas. Returns a formatted string with table headers and rows.
    """
    if not instance_scores:
        return "  (no per-instance scores available)"
    sorted_scores = sorted(instance_scores.items(), key=lambda x: x[1])
    lines = ["  | Instance | Score |", "  |----------|-------|"]
    for inst, score in sorted_scores:
        lines.append(f"  | {inst} | {score:.4f} |")
    return "\n".join(lines)
