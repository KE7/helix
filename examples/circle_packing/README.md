# circle_packing

## Problem

Pack 26 non-overlapping circles inside the unit square `[0, 1] × [0, 1]` so as to **maximize the sum of their radii**. This is a classic geometric optimization problem and one of the benchmarks used by [GEPA](https://github.com/google-deepmind/gepa). The objective is the sum `Σ rᵢ`; circles may touch but must not overlap, and every circle must lie fully inside the unit square.

## How to run

```bash
cd examples/circle_packing
helix evolve
```

This will evolve `solve.py` against `evaluate.py` using the configuration in `helix.toml`.

## Expected result

Starting from a trivial seed solver scoring **0.9798**, HELIX evolves a solution that reaches **2.6360**, beating the published GEPA benchmark score of **2.635**.

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

The exact `[claude]` block from `helix.toml` that produced the result:

```toml
[claude]
model = "haiku"
effort = "low"
max_turns = 20
```

No Sonnet, no Opus, no extended thinking — just Haiku with low effort and a hard 20-turn cap per mutation, and HELIX still beats GEPA.

## Files

- `solve.py` — the evolving solver (this is what HELIX mutates).
- `evaluate.py` — scorer that checks validity and returns `Σ rᵢ`.
- `helix.toml` — project configuration.
- `solve_optimized.py` — a hand-tuned reference implementation that scores **2.635982**. It is not used during evolution; it is provided as a sanity check / target for comparison.
