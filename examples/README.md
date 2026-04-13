# HELIX Examples

A small gallery of end-to-end HELIX projects you can run, modify, and learn from. Each subdirectory is a self-contained `helix.toml` project.

## Index

- **[circle_packing/](circle_packing/)** — Pack 26 circles in the unit square to maximize the sum of radii. A geometric optimization benchmark used by GEPA. HELIX evolves a Python solver from a trivial seed (sum 0.9798) past the published GEPA score (2.635).
- **[web_researcher/](web_researcher/)** — A small QA agent that answers software-engineering questions requiring up-to-date knowledge (current package versions, API behaviors). HELIX evolves the agent from stale training-data answers toward live web lookups.

## Running an example

```bash
cd examples/<name>
helix evolve
```

See each example's `README.md` for the seed score, target score, and any notes.
