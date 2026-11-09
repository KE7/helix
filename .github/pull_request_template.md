<!-- Rules: CONTRIBUTING.md. Rule 9: every PR shows its change working on the real system. -->

## What and why

<!-- What changed, why, and anything a reviewer should know. "Fixes #N" if it closes an issue. -->

## Evidence

<!--
Proof from the real system at this PR's latest commit, not mocks or test fakes:
- search, evaluator or backend: a real `helix evolve` run with a real backend and model.
  Give the HELIX commit, backend, model and version, command, and the output that shows the change.
- new backend: its `--help` against HELIX's flags, a real call with parsed usage,
  `helix sandbox login/status/logout`, and `helix evolve` on the host and in the sandbox.
- example: a recorded run with any held-out check, plus a video or image if it is visual.
- docs: the strict build and a screenshot of the changed page.
- no visible change: the test output, failing before and passing after for a bug fix.
Screenshots and videos go on an orphan `pr-assets/<branch>` branch, linked via raw.githubusercontent.com.
-->

## Try it by hand

<!-- Copy-pasteable steps from a clean checkout: install, the full helix.toml and evaluator,
each command with its expected output, and one failure case where it matters. -->

## Checks run locally

- [ ] `uv run --all-extras --python $v pytest tests/unit/ -q` for each `v` in 3.12 to 3.15
- [ ] `uv run --all-extras mypy --strict src/helix/`
- [ ] `uv run --all-extras ruff check src/`
- [ ] `uv run --only-group docs mkdocs build --strict`
- [ ] Runner image built and probed (only for `docker/` or `src/helix/backends.py` changes)
- [ ] Docs page and `CHANGELOG.md` `[Unreleased]` updated (user-facing changes)
- [ ] Coding-agent review of the diff and of what it should have touched but didn't (rule 10); what it found and what I did:

## Not run, and why

<!-- Anything you could not check, for example a sign-in that needs the maintainer's account
or a paid model with no credit. "Nothing" is a valid answer. -->
