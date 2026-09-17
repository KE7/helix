# circle_packing

## Problem

Pack 26 non-overlapping circles inside the unit square `[0, 1] × [0, 1]` so as to **maximize the sum of their radii**. This is a classic geometric optimization problem. The objective is the sum `Σ rᵢ`; circles may touch but must not overlap, and every circle must lie fully inside the unit square.

## How to run

Copy the example to a standalone directory, then run HELIX there:

```bash
work_dir="$(mktemp -d)/circle_packing"
cp -R "$(git rev-parse --show-toplevel)/examples/circle_packing" "$work_dir"
cd "$work_dir"
helix evolve
```

`helix evolve` initializes the copy as a git repository and evolves `solve.py`
against `evaluate.py` using the configuration in `helix.toml`.

## Results

Run the example to produce results; see
[`docs/experiments/circle-packing.md`](../../docs/experiments/circle-packing.md)
for the evaluation protocol and reference values. The seed solver (`solve.py`)
scores **0.979764**; the command under [Requirements](#requirements)
reproduces that number.

## Configuration

The agent settings in `helix.toml` are:

```toml
[agent]
backend = "claude"
model = "haiku"
effort = "low"
max_turns = 20
```

`[agent].background` describes the task, the `pack_circles(n)` interface, and
the evaluator's Python version and packages.

## Files

- `solve.py` — the evolving solver (this is what HELIX mutates).
- `evaluate.py` — scorer that checks validity and emits one
  `HELIX_RESULT=[[score, side_info], ...]` pair per id in `helix_batch.json`.
  `side_info` carries actionable feedback (which circles overlap or escape
  the square and by how much, radius spread, which circle has the most room
  to grow and what stops it, where the largest remaining empty gap is) so
  the next mutation has something to act on beyond a bare number. The
  evaluator never sees or reports a target to match — every measure is
  computed from the candidate's own geometry, so it keeps producing signal
  no matter how good the packing gets.
- `helix.toml` — project configuration. It lists `evaluate.py` in
  `[evaluator].protected_files`, so a candidate that modifies the evaluator is
  rejected.

## Requirements

The seed solver uses `numpy`. The evaluator command in `helix.toml` pins
Python 3.12 and installs `numpy` and `scipy` itself via `uv`, so evaluation
does not depend on the host `python3` and a fresh clone reproduces the seed
score.

Verify the baseline before trusting a run:

```bash
tmp_dir=$(mktemp -d) && cp "$(git rev-parse --show-toplevel)/examples/circle_packing/evaluate.py" "$tmp_dir/" \
  && cp "$(git rev-parse --show-toplevel)/examples/circle_packing/solve.py" "$tmp_dir/" \
  && printf '["0"]\n' > "$tmp_dir/helix_batch.json" \
  && cd "$tmp_dir" \
  && uv run --no-project --python 3.12 --with numpy --with scipy python3 evaluate.py
# HELIX_RESULT=[[0.979764, {"scores": {"sum_radii": 0.979764}, "feedback": "All 26 circles are valid: inside the square, no overlaps. Radii range 0.0100-0.1303 (mean 0.0377, spread ratio 13.03) across 26 circles. Circle 0 could grow 0.1113 before circle 8 (total headroom 0.3665). Largest empty gap: r0.1211 near (0.88, 0.88).", "bounds_violations": {"count": 0, "worst": []}, "overlap_violations": {"count": 0, "total_overlap_depth": 0.0, "worst": []}, "radius_stats": {"n_circles": 26, "min": 0.01, "max": 0.130332, "mean": 0.037683}}]]
```

A score of `0.0` with `"ERROR: Could not import solve.py"` means the evaluator
reached the wrong interpreter, not that the candidate is bad.
