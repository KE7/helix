"""HELIX population management: candidates, eval results, and Pareto frontier.

Implements GEPA-style unbounded frontier with coverage-based dominance.
Algorithm translated line-for-line from gepa.gepa_utils (gepa-ai/gepa).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Literal


logger = logging.getLogger(__name__)


FrontierType = Literal["instance", "objective", "hybrid", "cartesian"]


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
    usage: dict[str, Any] = field(default_factory=dict)


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
      ``EvaluationBatch.trajectories`` (``src/gepa/core/adapter.py:25``).
      Carries freeform per-example diagnostics; rendered into the
      mutation prompt's Diagnostics section by
      :func:`helix.mutator._render_per_example_diagnostics` (GEPA
      ``format_samples`` parity at
      ``src/gepa/strategies/instruction_proposal.py:54-95``).
    """
    candidate_id: str
    scores: dict[str, float]          # aggregate/summary scores
    asi: dict[str, str]               # arbitrary string info (metadata)
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
    # :attr:`gepa.core.adapter.EvaluationBatch.objective_scores`
    # (``src/gepa/core/adapter.py:26``).  Feeds the multi-axis Pareto
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
    The ``frontier_type`` constructor arg mirrors GEPA's literal at
    ``src/gepa/core/state.py:22-23``:

    - ``"instance"`` (default): per-example-id keyspace, built from
      ``EvalResult.instance_scores``.  Matches HELIX's historical
      behaviour and GEPA's ``frontier_type="instance"`` path.
    - ``"objective"``: per-objective-name keyspace, values = mean of
      that objective across the valset, built from
      ``EvalResult.objective_scores``.  GEPA
      ``_update_objective_pareto_front`` (``state.py:474-484``).
    - ``"hybrid"``: union of the instance and objective keyspaces.
      A candidate survives if it's non-dominated on the combined
      keyset.  This is GEPA ``optimize_anything``'s default
      (``optimize_anything.py:476``) and HELIX's default
      (``evolution.frontier_type``).
    - ``"cartesian"``: per ``(val_id, objective_name)`` keyspace.
      GEPA ``_update_pareto_front_for_cartesian``
      (``state.py:512-525``).

    The **acceptance gate stays positional** on ``scores_list``
    regardless of ``frontier_type`` (GEPA ``acceptance.py:39-48``);
    only the Pareto retention / parent-selection decision is
    multi-axis.

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
        # ``"objective"`` axis — mirrors GEPAState.program_at_objective_pareto_front
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
    # Mutation helpers — mirrors GEPAState._update_pareto_front_for_val_id
    # ------------------------------------------------------------------

    def add(self, candidate: Candidate, result: EvalResult) -> None:
        """Add a candidate and its evaluation result.  Population is append-only."""
        self._candidates[candidate.id] = candidate
        self._results[candidate.id] = result
        self._update_per_key(candidate.id, result)
        # Multi-axis updates are no-ops when the frontier_type doesn't
        # use them, but always populating keeps diagnostics useful and
        # lets a caller inspect any axis post-hoc.
        self._update_objective(candidate.id, result)
        self._update_cartesian(candidate.id, result)

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

        Mirrors GEPA's ``_update_objective_pareto_front``
        (``state.py:474-484``) with its ``_per_prog_mean_objective_scores``
        helper (``state.py:462-472``): for each objective name present
        in any per-example slot, take the mean of that objective's
        scores across the entire ``objective_scores`` list.
        """
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

        Mirrors GEPA's ``_update_pareto_front_for_cartesian``
        (``state.py:512-525``): each per-example ``objective_scores[i]``
        slot is combined with the corresponding ``val_id`` (from
        ``instance_scores``' ordered keys) to form a tuple key.  We
        encode the tuple as ``f"{val_id}::{objective_name}"`` so the
        existing ``dict[str, set[str]]`` dominance helper works.
        """
        if result.objective_scores is None:
            return
        # ``instance_scores`` preserves helix_batch.json id order by
        # construction in ``helix_result.parse``; ``objective_scores``
        # is positional to the same id list.  Zip together.
        val_ids = list(result.instance_scores.keys())
        if len(val_ids) != len(result.objective_scores):
            # Defensive: skip cartesian updates when lengths disagree
            # (should not happen on the helix_result path but could on
            # hand-constructed EvalResults in tests).  Log at DEBUG so
            # the skip is traceable without polluting the default log.
            logger.debug(
                "ParetoFrontier._update_cartesian: skipping cid=%r — "
                "objective_scores length %d does not match "
                "instance_scores length %d",
                cid, len(result.objective_scores), len(val_ids),
            )
            return
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
        self._results[result.candidate_id] = result
        self._rebuild_axes()

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
        # "hybrid": union of instance ∪ objective keyspaces (GEPA O.A.
        # default — see src/gepa/optimize_anything.py:476).
        merged: dict[str, set[str]] = {}
        for k, v in self._per_key_best.items():
            merged[f"inst::{k}"] = v
        for k, v in self._objective_best.items():
            merged[f"obj::{k}"] = v
        return merged

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
        2. Sort by score ascending (worst first — lower-scoring programs are
           checked first and more likely to be eliminated).
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

        programs = sorted(programs, key=lambda x: scores.get(x, 0.0), reverse=False)

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
        dispatches on ``frontier_type``.  Uses each candidate's
        ``sum_score()`` as the tiebreaker (lower-scoring candidates are
        eliminated first, matching GEPA's
        ``train_val_weighted_agg_scores_for_all_programs``).

        GEPA parity (W1): use sum_score() to match GEPA semantics, which
        diverges from aggregate_score() when candidates have different instance counts.
        """
        scores = {cid: r.sum_score() for cid, r in self._results.items()}
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
           with sum scores (GEPA parity W1).
        2. Count per-key frequency of surviving programs in the *cleaned*
           frontier (dominated programs stripped from every front).
        3. Build a flat sampling list where each program appears *freq* times.
        4. Pick uniformly at random (``rng.choice(sampling_list)`` in GEPA).
        """
        if not self._candidates:
            raise ValueError("Frontier is empty — cannot select parent.")

        # GEPA parity (W1): use sum_score() to match GEPA semantics.
        scores = {cid: r.sum_score() for cid, r in self._results.items()}
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

        # Build flat sampling list (GEPA: sampling_list)
        sampling_list = [
            cid for cid, freq in program_frequency.items() for _ in range(freq)
        ]

        # GEPA-parity fix for score-only evaluators: if no instance scores
        # were recorded (e.g. evaluator uses aggregate scores only), the cleaned
        # per-key frontier is empty and sampling_list would be empty.  Fall back
        # to selecting uniformly from all candidates so evolution can continue.
        if not sampling_list:
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
