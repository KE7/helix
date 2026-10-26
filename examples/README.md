# HELIX Examples

A small gallery of end-to-end HELIX projects you can run, modify, and learn from. Each subdirectory is a self-contained `helix.toml` project.

## Index

- **[circle_packing/](circle_packing/)** — optimize_anything's circle-packing experiment: pack 26 circles in the unit square to maximize the sum of radii, with GEPA's own fitness function (GPT-5.1 refiner, warm start, StateTracker) and budget. HELIX evolves GEPA's seed code and refiner prompt (seed sum of radii 0.979764). The mutation agent runs in Docker with no web access, on the network that [`circle_packing_sandbox/`](circle_packing_sandbox/) provides; [`docs/experiments/circle-packing.md`](../docs/experiments/circle-packing.md) describes the experiment and the reference values.
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
