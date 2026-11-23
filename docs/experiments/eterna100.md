# Eterna100 RNA design

In a 49-minute run, HELIX with `gpt-6-luna`, OpenAI's lowest-priced GPT-6
model, evolved an RNA sequence designer from 6 to 32–33 of 50 held-out
Eterna100-V2 puzzles, with 8 seconds per puzzle. That is on par with two
published learned designers, SentRNA (33) and LEARNA (34). The agents built it
on ViennaRNA's own `RNAinverse`, which solves 8 of the 50 alone, by restarting
it and refining its output with simulated annealing. The run's API cost was
$0.42.

The 50 held-out puzzles were never shown to the agents. The strongest
published designers, DesiRNA and NEMO, solve 45 and 44 of them.

The example is [`examples/eterna100`](https://github.com/KE7/helix/tree/main/examples/eterna100).

## The task

An RNA molecule folds into a secondary structure: some bases pair into stems,
the rest form loops. Inverse folding runs the other way: given a target
structure, find a sequence that folds into exactly that structure. It is
NP-hard even in the simplest energy model
([Bonnet et al.](https://arxiv.org/abs/1710.11513)), so practical designers are heuristic
searches. Designed RNAs such as riboswitches, aptamers and mRNA therapeutics
start from this step.

[Eterna100](https://github.com/eternagame/eterna100-benchmarking) is the
standard benchmark for it: 100 targets from the Eterna citizen-science game,
12 to 400 bases long. A design solves its puzzle when the sequence's
minimum-free-energy structure under ViennaRNA 2 is exactly the target. This is
Eterna100-V2, which adapted 19 targets so that 99 of the 100 are solvable under
ViennaRNA 2 ([Koodli et al.](https://www.biorxiv.org/content/10.1101/2021.08.26.457839)).

The candidate is `designer.py`, with `design(target, budget_s, seed)`. The
evaluator gives each puzzle 8 seconds, kills the designer at 12, and folds the
result with ViennaRNA 2.8.0. The seed pairs every stem G-C, leaves every loop A,
and does no search.

The 100 puzzles are split 20 train, 30 validation and 50 held out, stratified
by how many of the 15 published algorithm variants solve each one, so the
three splits span the same difficulty. The agent sees the training puzzles and
their feedback only.

## Setup

| | |
|---|---|
| Agent | Codex CLI 0.167.0, model `gpt-6-luna`, in HELIX's Docker sandbox with web tools off |
| Search | 12 generations: 6 mutations in each of generations 1–11 and one merge in generation 12; a minibatch of 3 training puzzles |
| Evaluator | on the host, ViennaRNA 2.8.0, 4 puzzles at a time |
| Machine | Apple M4 Pro, 14 cores |
| HELIX | 0.3.0 plus the `helix ui` dashboard; the search loop is unchanged |

The run used the evaluator from before two later fixes: worker processes are
now killed on timeout, and the stored-sequence check now skips only the
top-level split folders. Neither changes a score in this run.

## Results

The seed solves 6 of the 50 held-out puzzles. The best candidate, `g11-s62`,
solves 32 in two of three scorings and 33 in the third. They differ by one
borderline puzzle, which the 8-second wall-clock budget makes load-dependent.

| | Train (20) | Validation (30) | Held out (50) |
|---|---|---|---|
| Seed designer | 5 | 5 | 6 |
| `g11-s62` | 13 | 21 | 32–33 |

Best validation score by generation:

| Generation | 0 | 1 | 2 | 3–10 | 11 | 12 |
|---|---|---|---|---|---|---|
| Best so far | 5 | 11 | 19 | 19 | 21 | 21 |

The run made 67 proposals. 13 were accepted (12 mutations and the merge) and
54 were rejected at the minibatch gate, for 821 puzzle evaluations in all.

The benchmark's `results.tsv` holds the published Eterna100-V2 runs. On the
same 50 held-out puzzles:

| Method | Held-out puzzles solved | Median time per puzzle |
|---|---|---|
| DesiRNA | 45 | 2 min |
| NEMO | 44 | 2 s |
| LEARNA (pretrained) | 34 | 30 s |
| SentRNA (pretrained) | 33 | 73 min |
| **HELIX-evolved designer** | **32–33** | **8 s limit** |
| EternaBrain (pretrained) | 25 | 2.7 min |
| RNAinverse | 8 | 3 s |
| Seed designer | 6 | |

The published methods ran with their own time limits, up to 24 hours per
puzzle, on other hardware. Their medians count failed runs too. NEMO is fast as
well as strong: 32 of its 44 solves took 8 seconds or less.

### Cost

At OpenAI's [API prices](https://developers.openai.com/api/docs/pricing) for
`gpt-6-luna` (checked 2026-11-16), the run cost $0.42.

| | Tokens | Per 1M tokens | Cost |
|---|---|---|---|
| Input, uncached | 1,642,142 | $0.10 | $0.164 |
| Input, cached | 14,273,280 | $0.01 | $0.143 |
| Output, including 71,573 reasoning | 216,421 | $0.50 | $0.108 |
| **Total** | | | **$0.415** |

Codex counts cached input inside its input total; the table splits it out.
Long-context prices start at 272K input tokens per request. The largest request
in the 13 transcripts HELIX kept, one per accepted candidate, was 24K input
tokens.

## What the agents built

`g11-s62` is a 131-line `designer.py`, four accepted mutations from the seed
(`g1-s4`, `g2-s12`, `g6-s36`, `g11-s62`). It works in three stages:

1. It runs ViennaRNA's `RNA.inverse_fold` (the function behind `RNAinverse`)
   as a restarted proposal generator, from three kinds of starting sequence,
   and returns as soon as a proposal folds to the target.
2. With what is left of the budget it runs simulated annealing from the best
   proposal. A move changes one base pair to another legal pair, or one
   unpaired base. A candidate is scored by base-pair distance to the target,
   with the energy gap between the target and the folded structure as a
   tie-breaker.
3. It stops `inverse_fold` restarts at 88% of the budget so the refinement
   always gets some time.

The generation-12 merge, `g12-m1`, combined it with another frontier candidate
and skips `inverse_fold` on targets of 250 bases or more. It scored 19 on
validation and did not replace `g11-s62`.

A single `inverse_fold` call on a long target cannot be interrupted. All 7 of
`g11-s62`'s held-out failures with an error are timeouts on targets of 108 to
400 bases.

## Limits

- The 8-second budget is wall-clock time, and up to three evaluations, each
  running 4 puzzles at once, ran at the same time. A borderline puzzle can
  pass under light load and fail under heavy load.
- The agent's sandbox had an open network. Its instructions forbid fetching
  published solutions, and the evaluator scores 0 for any candidate file that
  stores an RNA sequence of 40 or more letters, but nothing stops a designer
  from fetching at run time. `g11-s62` makes no network call.
- One run, one seed.

## Next

- **A longer per-puzzle limit** (`BUDGET_S` in `evaluate.py`), during the
  search as well as on the held-out puzzles, to see how far more time takes
  the designer the agents evolve.
- **A stronger model**, such as `gpt-6.1-sol` (`agent.model` in
  `helix.toml`). Its API prices are 20 times `gpt-6-luna`'s for input and
  output and 10 times for cached input: this run would have cost about $6.88
  with it.
