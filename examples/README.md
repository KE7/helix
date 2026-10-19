# HELIX Examples

A small gallery of end-to-end HELIX projects you can run, modify, and learn from. Each subdirectory is a self-contained `helix.toml` project.

## Index

- **[circle_packing/](circle_packing/)** — Pack 26 circles in the unit square to maximize the sum of radii, a geometric optimization benchmark also used by GEPA. HELIX evolves a Python solver starting from a naive seed (sum of radii 0.979764). [`docs/experiments/circle-packing.md`](../docs/experiments/circle-packing.md) describes the evaluation protocol and reference values.
- **[web_researcher/](web_researcher/)** — A small QA agent that answers software-engineering questions requiring up-to-date knowledge (current package versions, API behaviors). The seed asks Claude Haiku directly, with no web access. [`docs/experiments/web-researcher/reference_agent.py`](../docs/experiments/web-researcher/reference_agent.py) is a reference agent that answers version questions from live sources (PyPI, endoflife.date) and returns fixed answers for the stable questions.

## Running an example

Copy the example to a standalone directory and run HELIX there:

```bash
work_dir="$(mktemp -d)/<name>"
cp -R "$(git rev-parse --show-toplevel)/examples/<name>" "$work_dir"
cd "$work_dir"
helix evolve
```

`helix evolve` initializes the copy as a git repository and commits it as the seed. See each example's `README.md` for requirements and notes.
