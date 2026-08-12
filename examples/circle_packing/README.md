# circle_packing

## Problem

Pack 26 non-overlapping circles inside the unit square `[0, 1] × [0, 1]` so as to **maximize the sum of their radii**. This is a classic geometric optimization problem and one of the benchmarks used by [GEPA](https://gepa-ai.github.io/gepa/). The objective is the sum `Σ rᵢ`; circles may touch but must not overlap, and every circle must lie fully inside the unit square.

## How to run

```bash
cd examples/circle_packing
helix evolve
```

This will evolve `solve.py` against `evaluate.py` using the configuration in `helix.toml`.

## Expected result

Starting from a trivial seed solver scoring **0.9798**, HELIX evolves a solution that reaches **2.635982** — matching the best published result of **2.63598+**, and above AlphaEvolve's **2.6358**.

Score progression along the winning lineage:

| Stage      | Score   |
|------------|---------|
| Seed       | 0.9798  |
| Gen ~3     | 2.5413  |
| Gen ~6     | 2.5561  |
| Gen ~10    | 2.6088  |
| **Gen 14** | **2.6360** |

The final score of **2.6360** was reached at **generation 14 of a 30-generation budget** — less than half the budget consumed.

> ### 💡 The kicker: this was the *cheapest* Claude setup available
>
> *Achieved with **haiku + low reasoning effort + max_turns=20**, arguably the cheapest Claude setup available. Demonstrates HELIX can extract strong results from tiny budgets.*

The exact `[agent]` block from `helix.toml` that produced the result:

```toml
[agent]
backend = "claude"
model = "haiku"
effort = "low"
max_turns = 20
```

No Sonnet, no Opus, no extended thinking — just Haiku with low effort and a hard 20-turn cap per mutation, and HELIX still matches GEPA.

## Files

- `solve.py` — the evolving solver (this is what HELIX mutates).
- `evaluate.py` — scorer that checks validity and emits one
  `HELIX_RESULT=[[score, side_info], ...]` pair per id in `helix_batch.json`.
- `helix.toml` — project configuration.
- `solve_optimized.py` — a hand-tuned reference implementation that scores **2.635982**. It is not used during evolution; it is provided as a sanity check / target for comparison.

## Requirements

The solver uses `numpy` and `scipy` and requires Python 3.10+ for PEP 604
(`X | None`) annotations. The evaluator command in `helix.toml` pins an
interpreter and installs those two packages itself via `uv`, so no manual
setup is needed and a fresh clone reproduces the reference score.

Verify the baseline before trusting a run:

```bash
tmp_dir=$(mktemp -d) && cp "$(git rev-parse --show-toplevel)/examples/circle_packing/evaluate.py" "$tmp_dir/" \
  && cp "$(git rev-parse --show-toplevel)/examples/circle_packing/solve_optimized.py" "$tmp_dir/solve.py" \
  && printf '["reference"]\n' > "$tmp_dir/helix_batch.json" \
  && cd "$tmp_dir" \
  && uv run --no-project --python 3.12 --with numpy --with scipy python3 evaluate.py
# HELIX_RESULT=[[2.635982, {"scores": {"sum_radii": 2.635982}, "violations": 0, "arrangement": "26 circles in 6x5 grid, avg_r=0.1014, violations=0"}]]
```

A score of `0.0` with `"ERROR: Could not import solve.py"` means the evaluator
reached the wrong interpreter, not that the candidate is bad.
