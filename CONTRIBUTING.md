# Contributing to HELIX

Rules for every contributor, human or agent. The core rule: **a PR shows that its change works**, on the real system, in a way a reviewer can check in minutes (rule 9).

## Design principles

When a design question comes up, check here first; when a review settles one, add it here.

- **Say where HELIX follows GEPA and where it doesn't.** HELIX's search loop started from GEPA optimize_anything and has since diverged: whole-repository candidates, coding-agent mutations, P x N proposals, a protected evaluator. A change to selection, acceptance, minibatching, budgets or stopping says which it is. If it keeps GEPA's behaviour, cite the GEPA symbol (by name, not line number); if it departs, say why in the code and the PR.
- **The smallest design that covers every case.** Hook the single place where the behaviour lives instead of adding special cases around it, and prefer a default that is right to a new opt-in flag.
- **The evaluator is protected.** Nothing may leak scoring code, held-out ids or answers to the mutation agent, through files, prompts, environment variables or logs. A change near the evaluator, the sandbox or the prompt says in its PR how it keeps that true.

## Repository

1. **Docs live on the docs site.** User-facing docs are Markdown under `docs/`, built by `mkdocs.yml` and published to [ke7.github.io/helix](https://ke7.github.io/helix/). A change that adds or changes a config key, CLI flag or behaviour updates the matching page in the same PR. Keep the README a short overview; reference material belongs in `docs/`.
2. **Every user-facing change has a CHANGELOG entry** under `[Unreleased]`, written for users: what changed and what they must do. Mark breaking changes **BREAKING**.
3. **Binary files only under `docs/assets/`**, and only when the README or docs show them. PR screenshots and videos do not go on `main` (rule 9).
4. **Never commit secrets**: no API keys, tokens or credentials in files, fixtures, recorded runs or screenshots. Redact them from evidence.
5. **Justify each new dependency** in the PR description. Pin the versions an example's results depend on.

## Branches and PRs

6. Never push to `main`. Branch off `main` for each PR; don't pile unrelated changes onto one branch.
7. Stacked PR: target the parent PR's branch and start the description with "Stacked on #N".
8. When you push new commits to an open PR, update its description to match, evidence included.
9. **Show the change working.** Every PR description has both sections, no exceptions:
   - **Evidence**: proof from the real system at the PR's latest commit, not from mocks or test fakes. Choose what fits the change:
     - **Search, evaluator or backend changes**: a real `helix evolve` run with a real backend CLI and a real model. Give the HELIX commit, the backend, the model and its version, the command, and the output that shows the change (scores, `helix frontier`, `helix log`, `helix timing` or `.helix/` records). A local model counts as real; a scripted mock does not.
     - **A new backend**: its CLI's own `--help` against the flags HELIX passes, a real non-interactive call whose usage HELIX parses, `helix sandbox login/status/logout`, and a `helix evolve` run on the host and in the Docker sandbox.
     - **A new or changed example**: a recorded run (as in `docs/experiments/`), with any held-out check, and a video or image if the example is visual.
     - **Docs**: the strict docs build and a screenshot of the changed page.
     - **No visible change** (refactors, tests only): the relevant test's output, failing before and passing after for a bug fix.
   - **Try it by hand**: copy-pasteable steps from a clean checkout. Give the install command, the complete `helix.toml` and evaluator (not "an evaluator that scores X"), each command with its expected output, and one failure case where it matters (for example, a config the change now rejects).

   Also list **what you could not run, and why**, for example a sign-in that needs the maintainer's account or a paid model you have no credit for. A gap you name is fine; a gap you hide is not. Put screenshots and videos on an orphan `pr-assets/<branch>` branch and link them with `raw.githubusercontent.com`. Keep videos under two minutes; speed up waits and label them "(sped up)".
10. **CI passes locally before you open the PR** (rule 14).
11. A PR that fixes an issue links it (`Fixes #N`), so merging closes it.
12. Keep writing short and concrete: PR descriptions, docs, comments. Say what changed, why, and how you know it works.

## Tests and CI

13. Every feature and every bug fix comes with a test, and every test runs in CI.
14. Before you push, run what CI runs:

    ```bash
    for v in 3.11 3.12 3.13 3.14 3.15; do uv run --all-extras --python $v pytest tests/unit/ -q; done
    uv run --all-extras mypy --strict src/helix/
    uv run --all-extras ruff check src/
    uv run --only-group docs mkdocs build --strict
    ```

    A change under `docker/` or to `src/helix/backends.py` also builds the affected runner image and probes its CLI as the unprivileged `node` user, as `runner-images.yml` does:

    ```bash
    docker build -f docker/base.Dockerfile -t helix-runner-base:ci .
    docker build -f docker/<backend>.Dockerfile --build-arg BASE_IMAGE=helix-runner-base:ci -t helix-runner-<backend>:ci .
    docker run --rm --user node --network none helix-runner-<backend>:ci sh -lc '<backend> --version'
    ```
15. When CI fails, find out why and iterate until it passes.
16. Flaky tests: if a re-run passes, say so in the PR and open an issue for the flake. Don't just re-run.

## Development setup

```bash
git clone https://github.com/KE7/helix.git
cd helix
uv sync --all-extras
uv run helix --help
```

Code style: type hints and docstrings on public functions and classes; `mypy --strict` and `ruff check` must pass.

## Questions

Open an issue on GitHub.
