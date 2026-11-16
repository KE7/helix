# Eterna100 RNA design

HELIX evolved an RNA sequence designer on the Eterna100-V2 benchmark. Starting
from a designer that does no search, Codex agents raised it from 5 to 21 of 30
validation puzzles over 12 generations. On 50 held-out puzzles no agent ever
saw, the evolved designer solves 32 or 33, against 6 for the seed.

The example is [`examples/eterna100`](https://github.com/KE7/helix/tree/main/examples/eterna100).

## The task

An RNA molecule folds into a secondary structure: some bases pair into stems,
the rest form loops. The design problem runs the other way: given a target
structure, find a sequence that folds into exactly that structure.
[Eterna100](https://github.com/eternagame/eterna100-benchmarking) is the
standard benchmark for it: 100 targets from the Eterna citizen-science game,
12 to 400 bases long. A design solves its puzzle when the sequence's
minimum-free-energy structure under ViennaRNA 2 is exactly the target. This is
Eterna100-V2, which adapted 19 targets so that 99 of the 100 are solvable
under ViennaRNA 2.

The candidate is `designer.py`, with `design(target, budget_s, seed)`. The
evaluator gives each puzzle 8 seconds, kills the designer at 12, and folds the
result with ViennaRNA 2.8.0. The seed pairs every stem G-C, leaves every loop
A, and does no search.

The 100 puzzles are split 20 train, 30 validation and 50 held out, stratified
by how many of the 15 published algorithm variants solve each one, so the
three splits span the same difficulty. The agent sees the training puzzles and
their feedback only.

## Setup

| | |
|---|---|
| HELIX | the `helix ui` branch; the search loop is `main`'s |
| Agent | Codex CLI 0.167.0, model `gpt-6-luna`, in the Docker sandbox, signed in with a ChatGPT subscription; web tools off |
| Search | 12 generations, 6 proposals per generation, merges on, minibatch of 3 training puzzles |
| Evaluator | on the host, ViennaRNA 2.8.0, 4 puzzles at a time; the version before two review fixes (worker processes killed on timeout, only top-level splits skipped by the stored-sequence check), neither of which changes a score in this run |
| Machine | Apple M4 Pro, 14 cores |

## Results

Best validation score (of 30) by generation:

| Generation | 0 | 1 | 2 | 3–10 | 11 | 12 |
|---|---|---|---|---|---|---|
| Best so far | 5 | 11 | 19 | 19 | 21 | 21 |

- 67 proposals: 13 accepted (12 mutations and one merge), 54 rejected at the
  minibatch gate. 821 evaluations in 49 minutes.
- Tokens, as Codex reports them: 1.64M uncached input, 14.27M cached input and
  216k output (72k of it reasoning).
- The best candidate, `g11-s62`, solves 21 of 30 validation puzzles, 13 of 20
  training puzzles and **32 of 50 held-out puzzles**, 33 on a second scoring
  (the seed: 5, 5 and 6). The two held-out scorings differ by one borderline
  puzzle, which the 8-second wall-clock budget makes load-dependent.
  Its 7 held-out failures with an error are all timeouts on targets of 108 to
  400 bases.

On the same 50 held-out puzzles, the published Vienna 2 results in the
benchmark's `results.tsv`:

| Method | Held-out puzzles solved |
|---|---|
| DesiRNA | 45 |
| NEMO | 44 |
| LEARNA (pretrained) | 34 |
| SentRNA | 33 |
| **HELIX-evolved designer, 8 s per puzzle** | **32–33** |
| EternaBrain (pretrained) | 25 |
| RNAinverse | 8 |
| Seed designer | 6 |

These methods ran under their own time limits, not 8 seconds: LEARNA's median
time per puzzle in those results is 28 s and SentRNA's over an hour.

## What the agents built

`g11-s62` is a 131-line `designer.py`, four accepted mutations from the seed:

1. It uses ViennaRNA's own `RNA.inverse_fold` as a restarted proposal
   generator, from three kinds of starting sequence, and returns as soon as a
   proposal folds to the target.
2. With what is left of the budget it runs simulated annealing from the best
   proposal: a move changes one base pair to another legal pair, or one
   unpaired base, and a candidate is scored by base-pair distance to the
   target, with the energy gap between the target and the folded structure as
   a tie-breaker.
3. It stops `inverse_fold` restarts at 88% of the budget so the refinement
   always gets some time.

The generation-12 merge (`g12-m1`) combined it with another frontier
candidate and skips `inverse_fold` on targets of 250 bases or more; it scored
19 on validation and did not replace `g11-s62`. A single `inverse_fold` call on a long target cannot
be interrupted, which is where the held-out timeouts come from.

## Limits

- The 8-second budget is wall-clock time, and up to six evaluations run at
  once, so a borderline puzzle can pass under light load and fail under heavy
  load.
- The agent's sandbox had an open network. Its instructions forbid fetching
  published solutions, and the evaluator scores 0 for any candidate file that
  stores an RNA sequence of 40 or more letters, but nothing stops a designer
  from fetching at run time. `g11-s62` makes no network call.
- One run, one seed.
