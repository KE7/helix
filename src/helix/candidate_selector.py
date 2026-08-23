"""Candidate-selection strategies: choosing which candidate to mutate next.

Ports the three strategies GEPA implements as ``CurrentBestCandidateSelector``,
``EpsilonGreedyCandidateSelector``, and ``TopKParetoCandidateSelector`` in
``gepa.strategies.candidate_selector`` (gepa-ai/gepa). HELIX's own ``"pareto"``
strategy is unchanged: :func:`select_candidate` delegates it to
:meth:`helix.population.ParetoFrontier.select_parent`, which ranks by
``sum_score()``. Which strategy runs is set by
``config.evolution.candidate_selection_strategy``.

Ranking
-------
The three ported strategies rank by ``EvalResult.aggregate_score()`` — the mean
of a candidate's instance scores — never ``sum_score()``. Both per-program score
arrays upstream ranks on are means, so the mean is the faithful analogue
everywhere here. The two quantities disagree whenever candidates carry unequal
numbers of scored instances, which HELIX permits, so this is not a free choice.

A candidate with no instance scores floors to ``-inf``
(:func:`_aggregate_score_or_floor`) and so ranks below every scored candidate,
including one with a negative mean.

Determinism
-----------
Selection draws from the caller's seeded ``random.Random``, so a run repeats
within a process. It does not repeat across processes: the ``pareto`` and
``top_k_pareto`` paths iterate ``set[str]`` fronts and CPython salts ``str``
hashing per process, so ties can break differently under a different
``PYTHONHASHSEED``. ``current_best`` draws no randomness and is unaffected.

Limits
------
No distinctness guarantee: with more than one proposal per iteration, every
proposal may draw the same parent.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Literal

from helix.population import Candidate, EvalResult, ParetoFrontier

CandidateSelectionStrategy = Literal[
    "pareto", "current_best", "epsilon_greedy", "top_k_pareto"
]


def _aggregate_score_or_floor(result: EvalResult | None) -> float:
    """Mean instance score, floored to ``-inf`` when there is nothing to average.

    Matches upstream's ``get_program_average_val_subset``, which yields
    ``float("-inf")`` for a program with no recorded subscores. Both an absent
    result and a present-but-scoreless one floor. ``aggregate_score()`` returns
    ``0.0`` for the scoreless case and other callers want that, so the floor
    belongs here at the selection site, not in ``aggregate_score``.
    """
    if result is None or not result.instance_scores:
        return float("-inf")
    return result.aggregate_score()


def _first_argmax(
    frontier: ParetoFrontier,
    score_of: Callable[[EvalResult | None], float],
) -> Candidate:
    """Highest-scoring candidate; ties go to the earliest discovered.

    ``ParetoFrontier`` only ever inserts into ``_candidates``, so iterating it
    walks discovery order. Scanning that order and displacing the leader only
    on a strict improvement reproduces GEPA's ``idxmax``
    (``lst.index(max(lst))`` over its append-only ``program_candidates``).

    The first candidate is taken unconditionally rather than on a strict
    improvement, because ``-inf`` is a reachable score: an all-unscored pool
    ties every candidate at ``-inf``, and ``-inf > -inf`` is False, so a pure
    strict-improvement scan would leave ``best_id`` unset.
    """
    best_id: str | None = None
    best_score = float("-inf")
    for cid in frontier.candidates:
        score = score_of(frontier.get_result(cid))
        if best_id is None or score > best_score:
            best_score = score
            best_id = cid
    assert best_id is not None  # caller guarantees a non-empty frontier
    return frontier.candidates[best_id]


def select_current_best(frontier: ParetoFrontier) -> Candidate:
    """Deterministic argmax over aggregate validation score.

    GEPA parity: ``CurrentBestCandidateSelector.select_candidate_idx``.
    Consumes no randomness.
    """
    if len(frontier) == 0:
        raise ValueError(
            "Frontier is empty — cannot select a current-best candidate."
        )
    return _first_argmax(frontier, _aggregate_score_or_floor)


def select_epsilon_greedy(
    frontier: ParetoFrontier, rng: random.Random, epsilon: float
) -> Candidate:
    """With probability ``epsilon``, pick uniformly from the whole pool.

    GEPA parity: ``EpsilonGreedyCandidateSelector.select_candidate_idx``. The
    exploration draw covers every evaluated candidate, not just the Pareto
    frontier, so a candidate dominated on every key can still be picked. The
    greedy branch delegates to :func:`select_current_best` and draws no
    randomness beyond the coin flip.
    """
    if len(frontier) == 0:
        raise ValueError(
            "Frontier is empty — cannot select an epsilon-greedy candidate."
        )
    if rng.random() < epsilon:
        ids = list(frontier.candidates.keys())
        return frontier.candidates[ids[rng.randint(0, len(ids) - 1)]]
    return select_current_best(frontier)


def select_top_k_pareto(
    frontier: ParetoFrontier, rng: random.Random, k: int
) -> Candidate:
    """Pareto draw restricted to the top-``k`` candidates by mean score.

    GEPA parity: ``TopKParetoCandidateSelector.select_candidate_idx``.

    1. Rank every evaluated candidate by ``aggregate_score()`` descending;
       ties keep discovery order, via a stable sort over a discovery-ordered
       id list.
    2. Intersect the top-k ids into each per-key front from
       :meth:`ParetoFrontier.active_frontier_snapshot`, dropping keys whose
       intersection comes out empty.
    3. If every key empties, fall back to an argmax over the same mean scores
       used for the ranking.
    4. Otherwise strip dominated candidates with
       :meth:`ParetoFrontier._remove_dominated_programs` and draw from the
       survivors weighted by how many fronts each appears on.

    ``k >= len(frontier)`` makes step 2 a no-op on membership, but the full
    path still runs. :meth:`ParetoFrontier.select_parent` is the tempting
    shortcut and is not equivalent: it feeds ``sum_score()`` into dominance
    removal, which selects a different candidate whenever candidates carry
    unequal numbers of scored instances.
    """
    if len(frontier) == 0:
        raise ValueError(
            "Frontier is empty — cannot select a top-k-pareto candidate."
        )

    ordered_ids = list(frontier.candidates.keys())
    scores = {
        cid: _aggregate_score_or_floor(frontier.get_result(cid)) for cid in ordered_ids
    }
    ranked = sorted(ordered_ids, key=lambda cid: scores[cid], reverse=True)
    top_k_ids = set(ranked[:k])

    filtered_mapping: dict[str, set[str]] = {}
    for key, ids in frontier.active_frontier_snapshot().items():
        filtered = set(ids) & top_k_ids
        if filtered:
            filtered_mapping[key] = filtered

    if not filtered_mapping:
        return _first_argmax(frontier, _aggregate_score_or_floor)

    _, cleaned = ParetoFrontier._remove_dominated_programs(filtered_mapping, scores)
    program_frequency: dict[str, int] = {}
    for front in cleaned.values():
        for cid in front:
            program_frequency[cid] = program_frequency.get(cid, 0) + 1
    sampling_list = [
        cid for cid, freq in program_frequency.items() for _ in range(freq)
    ]
    if not sampling_list:
        # Unreachable while filtered_mapping is non-empty. Guarded so that a
        # future change to dominance removal degrades to an argmax instead of
        # an opaque IndexError out of rng.choice([]).
        return _first_argmax(frontier, _aggregate_score_or_floor)
    return frontier.candidates[rng.choice(sampling_list)]


def select_candidate(
    strategy: CandidateSelectionStrategy,
    frontier: ParetoFrontier,
    rng: random.Random,
    *,
    epsilon: float | None = None,
    top_k: int | None = None,
) -> Candidate:
    """Dispatch to the configured candidate-selection strategy.

    ``"pareto"`` delegates to :meth:`ParetoFrontier.select_parent` unchanged.
    ``epsilon`` and ``top_k`` are read only by the strategy that owns them;
    ``EvolutionConfig.model_post_init`` already rejects a config missing the
    one it needs, so the asserts narrow types rather than validate input.
    """
    if strategy == "pareto":
        return frontier.select_parent()
    if strategy == "current_best":
        return select_current_best(frontier)
    if strategy == "epsilon_greedy":
        assert epsilon is not None
        return select_epsilon_greedy(frontier, rng, epsilon)
    if strategy == "top_k_pareto":
        assert top_k is not None
        return select_top_k_pareto(frontier, rng, top_k)
    raise ValueError(f"Unknown candidate_selection_strategy: {strategy!r}")
