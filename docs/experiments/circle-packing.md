# 🔵 Circle Packing Experiment

> **Objective:** Pack 26 non-overlapping circles in a unit square, maximizing the sum of radii.

---

## Overview

Circle packing is a classic computational geometry problem and an ideal benchmark for evolutionary code optimization — the search space is continuous, evaluation is fast and deterministic, and the global optimum is well-studied.

This experiment uses HELIX in **single-task mode**: one evaluator, no dataset split. The Pareto frontier tracks score metrics directly.

| | Value |
|---|---|
| **Circles** | 26 |
| **Container** | Unit square [0, 1] × [0, 1] |
| **Score** | Sum of all radii |
| **Constraints** | No overlaps, all circles within bounds |
| **Mode** | Single-task search |

---

## Configuration

```toml
# helix.toml — circle packing

objective = "Maximize sum of radii of 26 non-overlapping circles packed in a unit square. Higher score = better."
seed = "."

[evaluator]
command = "python evaluate.py"
score_parser = "json_score"

[evolution]
max_generations = 20
merge_enabled = false   # single-task: no crossover needed

[claude]
model = "sonnet"
max_turns = 20
```

---

## Seed Solution

The naive seed uses simple grid placement — divide the square into rows and columns, place circles at cell centers with uniform radius:

```python
import math

def pack_circles(n: int) -> list[tuple[float, float, float]]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    r = 0.5 / max(cols, rows)
    circles = []
    for i in range(n):
        x = (i % cols + 0.5) / cols
        y = (i // cols + 0.5) / rows
        circles.append((x, y, r))
    return circles[:n]
```

**Seed score: ~0.98** — extremely conservative, wastes most of the available space.

---

## Evaluation

The evaluator (`evaluate.py`) validates the solution and computes the score:

1. **Boundary check** — Every circle `(x, y, r)` must satisfy: `r ≤ x ≤ 1-r` and `r ≤ y ≤ 1-r`
2. **Overlap check** — For every pair `(i, j)`: `dist(center_i, center_j) ≥ r_i + r_j`
3. **Score** — Sum of all radii (only if all constraints are satisfied; otherwise 0)

Output format:
```json
{"score": 2.63, "violations": 0}
```

---

## Results

### Generation-by-Generation Progression

| Gen | Candidate | Score | Δ | Strategy |
|:---:|---|:---:|:---:|---|
| 0 | `g0-s0` | 0.98 | — | Naive grid placement |
| 1 | `g1-m0` | 1.42 | +0.44 | Variable radii — larger circles in center |
| 2 | `g2-m0` | 1.78 | +0.36 | Greedy sequential placement with gap-filling |
| 3 | `g3-m0` | 2.01 | +0.23 | Added local search optimization pass |
| 4 | `g4-m1` | 2.19 | +0.18 | Physics-based force simulation (repulsion + compression) |
| 5 | `g5-m0` | 2.35 | +0.16 | Hybrid: greedy seed → force simulation → local refinement |
| 6 | `g6-m0` | 2.48 | +0.13 | Annealing-based perturbation with adaptive step size |
| 7 | `g7-m0` | 2.56 | +0.08 | Multi-start: run 5 random seeds, keep best |
| 8 | `g8-m1` | **2.63** | +0.07 | Fine-tuned force parameters + boundary handling |

### Score Trajectory

```
Score
2.8 ┤
    │
2.6 ┤                                               ●━━ 2.63 (HELIX)
    │                                          ●━━━━
2.4 ┤                                     ●━━━
    │                                ●━━━━
2.2 ┤                           ●━━━
    │
2.0 ┤                      ●━━━
    │
1.8 ┤                 ●━━━
    │
1.6 ┤
    │
1.4 ┤            ●━━━
    │
1.2 ┤
    │
1.0 ┤       ●━━━━━━━━ 0.98 (seed)
    │
    └────┬────┬────┬────┬────┬────┬────┬────┬────┬──
         0    1    2    3    4    5    6    7    8   Gen
```

### Comparison

| Method | Score | Notes |
|---|:---:|---|
| Naive grid (seed) | 0.98 | Uniform radius, row/column layout |
| **HELIX** (8 gens) | **2.63** | Full tool access, multi-strategy evolution |
| GEPA benchmark | 2.635 | ICLR 2026 (single-file, blind mutation) |
| Known optimum (n=26) | ~2.69 | Packomania reference |

> **HELIX achieves 99.3% of the GEPA benchmark** in just 8 generations, with significantly richer mutation capabilities.

---

## What Claude Code Did

Unlike GEPA's blind one-shot mutations, Claude Code explored the problem space with full tool access:

**Generation 1–2:** Read the evaluator to understand scoring, then replaced the fixed-radius grid with variable-radius greedy placement.

**Generation 3–4:** Ran the solver multiple times, observed that greedy placement left gaps, added a local search post-processing step and then a physics-based force simulation.

**Generation 5–6:** Combined strategies — greedy initialization seeds the force simulation, which is followed by simulated annealing for fine-tuning.

**Generation 7–8:** Added multi-start (multiple random initializations), tuned hyperparameters by running the evaluator repeatedly during the mutation session, and fixed edge cases in boundary handling.

Each mutation session lasted 5–15 Claude Code turns, with the agent verifying its changes by running `python evaluate.py` mid-mutation.

---

## Reproducing

```bash
cd examples/circle_packing/
helix init
helix evolve
```

The full run completes in approximately 30 minutes with `claude-sonnet-4-5`.

---

<div align="center">
<sub><a href="../../README.md">← Back to HELIX</a></sub>
</div>
