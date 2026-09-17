# 🔵 Circle Packing Experiment

> **Objective:** Pack 26 non-overlapping circles in a unit square, maximizing the sum of radii.

---

## Overview

Circle packing is a classic computational geometry problem and a common benchmark for LLM-driven code evolution: the search space is continuous, and evaluation is fast and deterministic.

The example runs a single task: `helix.toml` sets `[dataset]` sizes of 1 and 1, so HELIX passes the evaluator one example id, `"0"`, on both the train and val splits.

| | Value |
|---|---|
| **Circles** | 26 |
| **Container** | Unit square [0, 1] × [0, 1] |
| **Score** | Sum of all radii |
| **Constraints** | No overlaps, all circles within bounds |
| **Mode** | Example-id handoff with one example (`train_size = 1`, `val_size = 1`) |

---

## Configuration

The configuration is [`examples/circle_packing/helix.toml`](../../examples/circle_packing/helix.toml):

```toml
[evaluator]
command = "uv run --no-project --python 3.12 --with numpy --with scipy python3 evaluate.py"
protected_files = ["evaluate.py"]

[dataset]
train_size = 1
val_size = 1

[evolution]
max_generations = 30
merge_enabled = false
perfect_score_threshold = 100.0
max_evaluations = 500

[agent]
backend = "claude"
model = "haiku"
effort = "low"
max_turns = 20
```

`[agent].background` states the task, the constraints, the `pack_circles(n)` interface, and the evaluator's Python version and packages.

---

## Seed Solution

[`examples/circle_packing/solve.py`](../../examples/circle_packing/solve.py) places one circle at the center, a ring of 8 around it, and the remaining 17 on an outer ring clipped to the square. It then shrinks radii greedily until no circles overlap and every circle fits inside the square.

**Seed score: 0.979764.** The command in the example README reproduces it. It uses the same placement and radius computation as the seed in appendix B of GEPA's blog post, which the post says it adopted from ShinkaEvolve. The post's trajectory figure shows that seed at a score of 0.98.

---

## Evaluation

The evaluator (`evaluate.py`) validates the solution and computes the score:

1. **Boundary check:** every circle `(x, y, r)` must satisfy `r ≤ x ≤ 1-r` and `r ≤ y ≤ 1-r`.
2. **Overlap check:** for every pair `(i, j)`, `dist(center_i, center_j) ≥ r_i + r_j`.
3. **Score:** the sum of all radii. Violations are reported in `side_info` and do not change the score, the same rule as the evaluator in GEPA's [`examples/circle_packing`](https://github.com/gepa-ai/gepa/tree/main/examples/circle_packing). A valid packing has both violation counts at 0.

Output format (one pair per id in `helix_batch.json`), shown for the seed:

```text
HELIX_RESULT=[[0.979764, {"scores": {"sum_radii": 0.979764}, "feedback": "All 26 circles are valid: inside the square, no overlaps. ...", "bounds_violations": {"count": 0, "worst": []}, "overlap_violations": {"count": 0, "total_overlap_depth": 0.0, "worst": []}, "radius_stats": {"n_circles": 26, "min": 0.01, "max": 0.130332, "mean": 0.037683}}]]
```

---

## Reference Values

| Source | Sum of radii (n=26) | Notes |
|---|:---:|---|
| Best known packing | 2.63598+ | [Erich Friedman's Packing Center](https://erich-friedman.github.io/packing/cirRsqu/) |
| GEPA optimize_anything | 2.63598+ | [Introducing optimize_anything](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/) |
| AlphaEvolve | 2.635 | [AlphaEvolve results notebook](https://github.com/google-deepmind/alphaevolve_results/blob/main/mathematical_results.ipynb), section B.12, which lists the previous state of the art as 2.634. The radii of its Construction 1 sum to 2.635863 |

These are best known values, not proven optima.

---

## Reference Packing

[`circle-packing/reference_known_best.py`](circle-packing/reference_known_best.py) is a reference packing for checking the evaluator setup. It embeds a packing (`KNOWN_BEST_CONFIG`) whose radii sum to 2.635982, runs an SLSQP-based search for up to 50 seconds, and returns the embedded packing when that search does not find a packing scoring above 2.6.

```bash
tmp_dir=$(mktemp -d) && cp "$(git rev-parse --show-toplevel)/examples/circle_packing/evaluate.py" "$tmp_dir/" \
  && cp "$(git rev-parse --show-toplevel)/docs/experiments/circle-packing/reference_known_best.py" "$tmp_dir/solve.py" \
  && printf '["0"]\n' > "$tmp_dir/helix_batch.json" \
  && cd "$tmp_dir" \
  && uv run --no-project --python 3.12 --with numpy --with scipy python3 evaluate.py
# HELIX_RESULT=[[2.635982, {"scores": {"sum_radii": 2.635982}, "feedback": "All 26 circles are valid: inside the square, no overlaps. ...", ...}]]
```

---

## Reproducing

Copy the example to a standalone directory outside any git repository, then run HELIX there. The copy keeps the run's git repository, candidate worktrees, and `.helix/` state separate from the HELIX clone:

```bash
work_dir="$(mktemp -d)/circle_packing"
cp -R "$(git rev-parse --show-toplevel)/examples/circle_packing" "$work_dir"
cd "$work_dir"
helix evolve
```

`helix evolve` initializes the copy as a git repository and commits it as the seed. After the run, `helix frontier`, `helix best`, and `helix log` show the frontier, the best candidate, and the mutation log.

A reported result should include:

- the `helix.toml` used, including the backend, model, effort, and `[sandbox]` settings;
- the `.helix/` state and mutation logs from the run;
- the best candidate's `solve.py` and its `HELIX_RESULT` line, with both violation counts at 0.

---

<div align="center">
<sub><a href="../../README.md">← Back to HELIX</a></sub>
</div>
