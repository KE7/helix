# eterna100

HELIX evolves an RNA sequence designer on the Eterna100-V2 benchmark. In one
49-minute run with `gpt-6-luna`, OpenAI's lowest-priced GPT-6 model, the
designer went from 6 to 32–33 of 50 held-out puzzles at 8 seconds per puzzle,
on par with the published SentRNA (33) and LEARNA (34). See
[Results](#results).

## The problem

An RNA molecule folds into a secondary structure: some bases pair up into
stems, the rest form loops. Inverse folding asks the reverse question. Given a
target structure, find a sequence that folds into exactly that structure.
Designed RNAs such as riboswitches, aptamers and mRNA therapeutics start from
this step.

[Eterna100](https://github.com/eternagame/eterna100-benchmarking) is the
standard benchmark for it: 100 target structures from the
[Eterna](https://eternagame.org) citizen-science game, 12 to 400 bases long,
from simple hairpins to structures with long single strands, short stems and
large multiloops that few algorithms can design. A target is in dot-bracket
notation, one character per base: `(` pairs with its matching `)`, and `.` is
unpaired.

```text
target  ((((((.((((....))))))).)))..........   puzzle 3, "Prion Pseudoknot - Difficulty Level 0"
design  CGAGUUGUCGUAGGAACGAAACGUCGAAAAAAAAGA   folds to the target
```

A design solves its puzzle when the sequence's minimum-free-energy structure,
computed by ViennaRNA 2, is exactly the target. This is Eterna100-V2, which
adapted 19 targets so that 99 of the 100 are solvable under ViennaRNA 2
([Koodli et al.](https://www.biorxiv.org/content/10.1101/2021.08.26.457839)).
Published designers solve between 20 (ViennaRNA's own `RNAinverse`) and 92
(DesiRNA) of the 100 puzzles under the benchmark's own runs.

## The candidate

`designer.py` defines `design(target, budget_s, seed) -> str`. The seed
designer does no search: it pairs every stem as G-C and leaves every loop A.
It solves 5 of the 20 training puzzles and 5 of the 30 validation puzzles.
Everything else in the directory is the candidate's to change: new modules,
caches computed at runtime, a different algorithm.

## Scoring

`evaluate.py` runs `design(target, 8.0, 0)` for each requested puzzle in its
own Python process, kills it after 12 seconds, and folds the result with
ViennaRNA 2.8.0 (37 C, Turner 2004 parameters, `dangles=2`, the Eterna100-V2
protocol). A puzzle scores 1 when the folded structure equals the target, else
0. Its feedback carries the target, the design, the folded structure, the base
pair distance and a mismatch line marking where the two structures differ.

Published solutions are on the web, so the evaluator checks that designs are
computed: a candidate whose files store an RNA sequence of 40 or more letters
scores 0 on every puzzle.

ViennaRNA 2.8.0 folds all 99 published Eterna100-V2 sample solutions to their
targets, as 2.1.9 and 2.6.4 do (puzzle 100 has none).

## Splits

`make_splits.py` builds the splits from the benchmark's own files. Each
puzzle's difficulty is how many of the 15 published algorithm variants solve
it under ViennaRNA 2. Sorted by difficulty, each block of ten puzzles gives
five to `test/`, three to `val/` and two to `train/`, so all three span the
same range, from puzzles every method solves to ones none does.

| Split | Puzzles | Seen by |
|---|---|---|
| `train/` | 20 | the agent, and the minibatch gate |
| `val/` | 30 | the evaluator only; it ranks candidates |
| `test/` | 50 | nobody during a run; score the best candidate on it afterwards |

`val/`, `test/` and `evaluate.py` are protected and left out of the agent's
sandbox copy.

## How to run

The agent is Codex (`gpt-6-luna`) in HELIX's Docker sandbox. The evaluator
runs on the host with `uv`.

```bash
helix sandbox login codex
work_dir="$(mktemp -d)/eterna100"
cp -R examples/eterna100 "$work_dir"
cd "$work_dir"
git init -q && git add -A && git commit -qm seed
helix evolve
```

Score the best candidate on the held-out puzzles:

```bash
helix best --export ./best
cd ./best
HELIX_SPLIT=test uv run --no-project --python 3.12 --with ViennaRNA==2.8.0 --with 'helix-evo>=0.3.0' python evaluate.py
```

It prints one `HELIX_RESULT=` line with a `[score, feedback]` pair per held-out
puzzle; the scores add up to the number solved.

## Results

One run with Codex and `gpt-6-luna` on a ChatGPT subscription: 12 generations,
6 mutations in each of the first 11 and a merge in the last
([write-up](https://ke7.github.io/helix/experiments/eterna100.html)).

| | Train (20) | Validation (30) | Held out (50) |
|---|---|---|---|
| Seed designer | 5 | 5 | 6 |
| Best candidate, generation 11 | 13 | 21 | 32–33 (three scorings) |

The best validation score went 5, 11, 19 in the first two generations and
reached 21 in generation 11. The evolved designer restarts ViennaRNA's
`RNAinverse` (8 of the 50 held-out puzzles on its own) and refines its output
with simulated annealing. On the held-out puzzles it solves about as many as
SentRNA (33) and LEARNA (34) in the benchmark's published runs, where the
median time per puzzle is 73 minutes for SentRNA and 30 seconds for LEARNA;
DesiRNA solves 45 and NEMO 44. The run took 49 minutes, and its tokens would
cost about $0.42 at OpenAI's API list prices.

The 8-second limit is `BUDGET_S` in `evaluate.py`. Every held-out failure with
an error was a timeout on a long target, so a longer limit is the obvious next
experiment.

## Files

- `designer.py`: the seed candidate.
- `evaluate.py`: the evaluator.
- `train/`, `val/`, `test/`: the puzzles, one row each (`id`, `name`,
  `solved_by`, `structure`).
- `make_splits.py`: rebuilds the splits from the benchmark's
  `data/eterna100_puzzles.tsv` and `data/results.tsv`.
- `LICENSE-eterna100`: the benchmark's MIT license.
- `helix.toml`: run configuration.

## Sources

- The Eterna100 benchmark: Anderson-Lee et al., *Journal of Molecular Biology*, 2016.
- Eterna100-V2, the ViennaRNA 2 version, and the published algorithms' results:
  [Koodli et al.](https://www.biorxiv.org/content/10.1101/2021.08.26.457839) and
  [eternagame/eterna100-benchmarking](https://github.com/eternagame/eterna100-benchmarking)
  (MIT license, see `LICENSE-eterna100`).
