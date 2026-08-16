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

At the upstream commit this was implemented against, both properties
happen to be computed with the identical formula (mean of each program's
validation subscores) — but a ``TODO`` on ``program_full_scores_val_set``
notes it "should be using the val_evaluation_policy instead", so the two
are documented as expected to diverge and must not be treated as
interchangeable by name alone.

HELIX maps each upstream array to the existing HELIX quantity that already
plays its role, rather than assuming they're the same array:

* ``per_program_tracked_scores`` feeds *both* the Pareto dominance-removal
  sort key and the top-k ranking upstream. HELIX's own ``ParetoFrontier``
  already uses ``EvalResult.sum_score()`` for exactly that role (see the
  "GEPA parity (W1)" comments on ``ParetoFrontier.get_non_dominated`` and
  ``select_parent`` in ``population.py``). So ``top_k_pareto`` here uses
  ``sum_score()`` for both its ranking *and* its empty-mapping fallback —
  matching upstream's reuse of the same local ``scores`` variable for both
  roles inside ``TopKParetoCandidateSelector.select_candidate_idx``.
* ``program_full_scores_val_set`` is upstream's "how good is this program
  overall" figure. The closest existing HELIX quantity, both by name and
  by mean-based semantics, is ``EvalResult.aggregate_score()`` (HELIX's own
  docstring: "mean of instance scores"). ``current_best`` and
  ``epsilon_greedy``'s greedy branch use ``aggregate_score()``.

This split is deliberately exercised by
``tests/unit/test_candidate_selector.py::TestScoreQuantityMapping``, which
constructs a pool where ``sum_score()`` and ``aggregate_score()`` argmax
disagree and asserts each strategy resolves to the quantity above.

Determinism
-----------
Every strategy takes the SAME ``random.Random`` instance the caller already
threads through ``ParetoFrontier`` (see ``evolution.py``'s ``rng =
random.Random(config.rng_seed)``), so a seeded run stays reproducible
end-to-end. ``current_best`` never touches ``rng`` — GEPA parity, ``idxmax``
is a pure argmax. Its tie-break is defined explicitly (see
:func:`_first_argmax`) rather than left to dict/set iteration order.

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
    """HELIX analog of upstream's ``program_full_scores_val_set`` entry."""
    return result.aggregate_score() if result is not None else float("-inf")


def _sum_score_or_floor(result: EvalResult | None) -> float:
    """HELIX analog of upstream's ``per_program_tracked_scores`` entry."""
    return result.sum_score() if result is not None else float("-inf")


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
    """Pareto draw restricted to the top-``k`` candidates by ``sum_score()``.

    GEPA parity: ``TopKParetoCandidateSelector.select_candidate_idx``:

    1. Rank all evaluated candidates by score (HELIX: ``sum_score()``,
       see module docstring) descending; ties keep discovery order (a
       stable sort over an already discovery-ordered id list, matching
       upstream's stable ``sorted(range(len(scores)), ...)`` over program
       indices).
    2. Intersect the top-k id set into every per-key frontier front
       (:meth:`ParetoFrontier.active_frontier_snapshot`); drop any key
       whose intersection is empty.
    3. If EVERY key's intersection is empty, fall back to a direct argmax
       over the SAME ``sum_score()`` array used for ranking — upstream's
       fallback reuses its local ``scores`` variable verbatim, not the
       ``program_full_scores_val_set``/``aggregate_score()`` quantity
       ``current_best`` uses. That fallback is load-bearing: replicate it
       exactly, including which score array it reads.
    4. Otherwise, run the identical dominance-removal + frequency-weighted
       draw :meth:`ParetoFrontier.select_parent` uses, over the filtered
       mapping (reusing :meth:`ParetoFrontier._remove_dominated_programs`,
       the same canonical implementation, rather than re-deriving it).

    When ``k`` covers the whole pool, the top-k intersection can never
    remove anything — it is mathematically a no-op — so this degrades
    directly to :meth:`ParetoFrontier.select_parent` (byte-for-byte
    ``"pareto"`` behaviour) instead of re-deriving an equivalent result
    through a differently-ordered mapping.
    """
    if len(frontier) == 0:
        raise ValueError(
            "Frontier is empty — cannot select a top-k-pareto candidate."
        )
    if k >= len(frontier):
        return frontier.select_parent()

    ordered_ids = list(frontier.candidates.keys())
    scores = {
        cid: _sum_score_or_floor(frontier.get_result(cid)) for cid in ordered_ids
    }
    ranked = sorted(ordered_ids, key=lambda cid: scores[cid], reverse=True)
    top_k_ids = set(ranked[:k])

    filtered_mapping: dict[str, set[str]] = {}
    for key, ids in frontier.active_frontier_snapshot().items():
        filtered = set(ids) & top_k_ids
        if filtered:
            filtered_mapping[key] = filtered

    if not filtered_mapping:
        return _first_argmax(frontier, _sum_score_or_floor)

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
        return _first_argmax(frontier, _sum_score_or_floor)
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


def select_parents(
    strategy: CandidateSelectionStrategy,
    frontier: ParetoFrontier,
    rng: random.Random,
    p: int,
    *,
    epsilon: float | None = None,
    top_k: int | None = None,
) -> list[Candidate]:
    """Batch seam: draw ``p`` parents, one independent :func:`select_candidate`
    call each.

    GEPA parity check: upstream's own ``CandidateSelector`` Protocol is
    scalar (``select_candidate_idx(self, state) -> int``,
    ``gepa.proposer.reflective_mutation.base``), and every upstream sampling
    strategy — including ``PxNSampling(p, n)``, the direct analog of
    HELIX's ``num_parallel_proposals * mutations_per_parent`` — draws its
    P parents with a plain ``for _ in range(self.p):
    candidate_selector.select_candidate_idx(state)`` loop
    (``gepa.strategies.proposal_sampling``). So this default — looping the
    existing scalar draw ``p`` times — isn't new behaviour; it's the same
    loop GEPA already runs today, exposed as a batch-shaped HELIX API.

    This exists purely as a seam: a future JOINT allocation strategy (e.g.
    systematic/low-variance resampling over the frontier's per-key
    frequency weights, so P parallel draws stop collapsing onto one
    candidate) can replace this default without HELIX callers needing to
    change how they call candidate selection. Deliberately NOT implemented
    here — every strategy above still draws independently with
    replacement, exactly as upstream GEPA does; see this module's
    docstring on distinctness. ``evolution.py``'s proposal loop is not
    migrated to call this: it already performs the equivalent "p
    sequential scalar draws" inline, interleaved with per-slot budget and
    minibatch-sampling logic that a batch call can't express without a
    larger, out-of-scope restructure.
    """
    return [
        select_candidate(strategy, frontier, rng, epsilon=epsilon, top_k=top_k)
        for _ in range(p)
    ]
