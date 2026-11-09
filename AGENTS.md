# Agent instructions

For agents working on the HELIX repository itself. To use HELIX in another project, see the agent skill in [`skills/helix/`](skills/helix/SKILL.md) instead.

Follow [CONTRIBUTING.md](CONTRIBUTING.md). Its rules apply to humans and agents alike; the ones agents most often miss:

- **Show the change working** (rule 9). A PR description has **Evidence** from the real system, a real `helix evolve` run with a real backend and model for search, evaluator or backend changes, not mocks; **Try it by hand** steps from a clean checkout; and what you could not run, and why.
- **Before opening a PR, run CI locally and review your own diff** (rule 10): the rule 14 commands, then `/code-review`, `/review` or the review prompt in rule 10. Fix only what matters in real use (below), with the smallest change, and say in the PR what the review found.
- **Fix only review findings that matter in real use** (rule 10): a bug a typical run hits, a leak, or wasted money in a typical setup. Drop edge cases nobody runs into.
- **Keep the evaluator protected.** Nothing may leak scoring code, held-out ids or answers to the mutation agent.
- **Say where a change follows GEPA and where it departs**, citing the GEPA symbol or the reason.
- **Update the docs page and `CHANGELOG.md` `[Unreleased]`** with any user-facing change.

Working here:

- Use `uv`: `uv run --all-extras pytest tests/unit/ -q`, `uv run --all-extras mypy --strict src/helix/`, `uv run helix ...`.
- Source is `src/helix/`; tests are `tests/unit/`; user docs are `docs/` (built by `mkdocs.yml`); examples are `examples/`, whose evaluators and splits are protected files.
- Never push to `main`, and never commit secrets or `.helix/` run state.
