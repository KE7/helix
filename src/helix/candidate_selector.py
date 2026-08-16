"""Pluggable candidate-selection strategies (GEPA parity).

Upstream attribution: ``gepa.strategies.candidate_selector``
(``ParetoCandidateSelector``, ``CurrentBestCandidateSelector``,
``EpsilonGreedyCandidateSelector``, ``TopKParetoCandidateSelector`` —
gepa-ai/gepa on GitHub).  Naming credit only; see this repo's PR hygiene
rules against upstream ``file:line`` citations.

HELIX already had the ``"pareto"`` strategy as
:meth:`helix.population.ParetoFrontier.select_parent`.  This module adds
the other three GEPA-parity strategies plus a single :func:`select_candidate`
dispatcher that ``evolution.py`` calls instead of ``frontier.select_parent()``
directly, keyed off ``config.evolution.candidate_selection_strategy``.

Score-quantity mapping
-----------------------
Upstream's four selectors read one of two differently-named per-program
score arrays off ``GEPAState``:

* ``program_full_scores_val_set`` — read by ``CurrentBestCandidateSelector``
  and ``EpsilonGreedyCandidateSelector`` via ``idxmax(...)``.
* ``per_program_tracked_scores`` — read by ``TopKParetoCandidateSelector``
  for its top-k ranking *and* its empty-mapping fallback, and is the same
  array ``ParetoCandidateSelector`` passes into
  ``select_program_candidate_from_pareto_front``.

Both properties are computed by the *identical* expression — a list
comprehension over ``get_program_average_val_subset(program_idx)[0]``,
which returns ``sum(scores.values()) / num_samples``. So both upstream
arrays are **means**, and the correct HELIX analogue for both is
``EvalResult.aggregate_score()`` (HELIX's own docstring: "mean of
instance scores"), never ``EvalResult.sum_score()``.

Both properties carry ``TODO`` comments about routing through the
``val_evaluation_policy`` instead. Those TODOs are about *where the
number should come from*, not a signal that the two arrays are expected
to diverge into different aggregation functions — and in any case a
speculative future divergence is not a licence to substitute a sum for
a mean today. If upstream ever does split them, this mapping should be
re-derived from the new definitions rather than guessed.

Accordingly every strategy in this module reads the mean:

* ``current_best`` and ``epsilon_greedy``'s greedy branch use
  ``aggregate_score()`` for ``program_full_scores_val_set``.
* ``top_k_pareto`` uses ``aggregate_score()`` for its top-k ranking, its
  filtered-front dominance-removal sort key, and its empty-mapping
  fallback — the three places upstream's
  ``TopKParetoCandidateSelector.select_candidate_idx`` reuses its single
  local ``scores = state.per_program_tracked_scores`` variable.

Sum-vs-mean only diverges when candidates have unequal numbers of scored
instances, which HELIX permits; ``tests/unit/test_candidate_selector.py``
— ``TestTopKPareto.test_ranking_uses_mean_not_sum_when_cardinality_differs``
pins the disagreeing case.

Pre-existing divergence (NOT addressed here): ``ParetoFrontier.select_parent``
and ``get_non_dominated`` in ``population.py`` pass ``sum_score()`` into
``_remove_dominated_programs`` under a "GEPA parity (W1)" comment. Upstream's
``ParetoCandidateSelector`` passes the mean array, so that comment does not
hold; it is deliberately left untouched by this module because the default
``"pareto"`` path must stay behaviourally identical. Tracked separately.

Determinism
-----------
Every strategy takes the SAME ``random.Random`` instance the caller already
threads through ``ParetoFrontier`` (see ``evolution.py``'s ``rng =
random.Random(config.rng_seed)``), so within a single process a seeded run
draws the same sequence from that ``rng`` on repeated invocations.
This is NOT a claim of cross-process/end-to-end reproducibility: both
``pareto`` (via ``ParetoFrontier.select_parent``) and ``top_k_pareto``'s
dominance-removal path iterate ``set[str]`` candidate-id fronts, and
CPython salts ``str`` hashing per process, so tie-broken outcomes can
differ across ``PYTHONHASHSEED`` values even with an identical ``rng``
seed. ``current_best`` never touches ``rng`` — GEPA parity, ``idxmax``
is a pure argmax. Its tie-break is defined explicitly
(see :func:`_first_argmax`) rather than left to dict/set iteration order,
and does not read any ``set[str]`` front, so it is unaffected by the
``PYTHONHASHSEED`` caveat above.

Anti-collapse / distinctness
-----------------------------
This module deliberately does NOT implement any minimum-distinct-parents
or anti-collapse guarantee (e.g. forcing P>1 proposals to sample >=2
distinct parents). That is a separate, still-open design question tracked
elsewhere. :func:`select_candidate` is a single seam a future distinctness
layer could wrap without touching the four strategies themselves.
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
    """HELIX analog of an entry in either upstream per-program score array.

    Upstream's ``program_full_scores_val_set`` and
    ``per_program_tracked_scores`` are the same mean-valued expression (see
    the module docstring), so one helper serves both. The ``-inf`` floor for
    an unevaluated candidate matches upstream's
    ``get_program_average_val_subset``, which returns ``float("-inf")`` when a
    program has no recorded subscores.
    """
    return result.aggregate_score() if result is not None else float("-inf")


def _first_argmax(
    frontier: ParetoFrontier,
    score_of: Callable[[EvalResult | None], float],
) -> Candidate:
    """GEPA ``idxmax`` parity: first occurrence of the max score wins ties.

    ``idxmax`` is ``lst.index(max(lst))`` over a list indexed by upstream's
    append-only ``program_candidates`` (discovery order). HELIX's
    ``ParetoFrontier._candidates`` dict is populated the same way — every
    ``add()`` call only ever inserts, never reorders — so Python's
    insertion-order-preserving dict iteration reproduces "first-discovered
    candidate among the tied leaders" exactly: scan in discovery order,
    keep only STRICT improvements, so an earlier candidate is never
    displaced by a later one with an equal score.
    """
    best_id: str | None = None
    best_score = float("-inf")
    for cid in frontier.candidates:
        score = score_of(frontier.get_result(cid))
        if score > best_score:
            best_score = score
            best_id = cid
    assert best_id is not None  # caller guarantees a non-empty frontier
    return frontier.candidates[best_id]


def select_current_best(frontier: ParetoFrontier) -> Candidate:
    """Deterministic argmax over aggregate validation score.

    GEPA parity: ``CurrentBestCandidateSelector.select_candidate_idx`` ->
    ``idxmax(state.program_full_scores_val_set)``. No randomness consumed.
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

    GEPA parity: ``EpsilonGreedyCandidateSelector.select_candidate_idx``.
    The random draw is over the FULL evaluated pool
    (``rng.randint(0, len(program_candidates) - 1)``), not the Pareto
    frontier — deliberately not routed through
    :func:`select_current_best`'s frontier-only machinery. The greedy
    branch always delegates to :func:`select_current_best` and consumes no
    extra randomness, matching upstream's ``else: return idxmax(...)``.
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
    """Pareto draw restricted to the top-``k`` candidates by ``aggregate_score()``.

    GEPA parity: ``TopKParetoCandidateSelector.select_candidate_idx``:

    1. Rank all evaluated candidates by score (HELIX: ``aggregate_score()``,
       the mean — see module docstring) descending; ties keep discovery
       order (a stable sort over an already discovery-ordered id list,
       matching upstream's stable ``sorted(range(len(scores)), ...)`` over
       program indices).
    2. Intersect the top-k id set into every per-key frontier front
       (:meth:`ParetoFrontier.active_frontier_snapshot`); drop any key
       whose intersection is empty.
    3. If EVERY key's intersection is empty, fall back to a direct argmax
       over the SAME mean array used for ranking — upstream's fallback
       reuses its local ``scores`` variable verbatim. Because both upstream
       score arrays are the same mean expression, this is also the same
       quantity ``current_best`` uses.
    4. Otherwise, run the identical dominance-removal + frequency-weighted
       draw :meth:`ParetoFrontier.select_parent` uses, over the filtered
       mapping (reusing :meth:`ParetoFrontier._remove_dominated_programs`,
       the same canonical implementation, rather than re-deriving it).

    When ``k`` covers the whole pool, the top-k intersection can never
    remove anything from any front — it is mathematically a no-op on the
    *membership* side — but there is no shortcut back to
    :meth:`ParetoFrontier.select_parent`: that method feeds ``sum_score()``
    into dominance removal, whereas every other read in this function uses
    the mean (``aggregate_score()``, see module docstring). Those two
    quantities only coincide when every candidate has the same number of
    scored instances, which HELIX does not guarantee. So ``k >= len(frontier)``
    is handled by simply letting the normal path run to completion with the
    full candidate set as its top-k slice: same mean mapping feeds the
    ranking, :meth:`ParetoFrontier._remove_dominated_programs`, and the
    empty-filter fallback, all three. The result is "Pareto frontier over
    the full mapping, ranked and dominance-checked by MEAN" — deliberately
    NOT identical to the legacy ``"pareto"`` strategy, which uses sums (see
    the module docstring's "Pre-existing divergence" section).
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
        # Unreachable given a non-empty filtered_mapping (the same
        # invariant ``ParetoFrontier.select_parent`` relies on for its own
        # non-"instance" path) — guarded so an algorithm change fails
        # loudly here instead of surfacing as an opaque IndexError from
        # ``rng.choice([])``.
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

    ``epsilon``/``top_k`` are only read for their matching strategy; HELIX's
    config layer (``EvolutionConfig.model_post_init``) already guarantees
    they're set when required, so the asserts here are pure type narrowing.
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
