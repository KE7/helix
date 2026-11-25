# Live dashboard

`helix ui` shows a run in your browser. Start it in the project directory, during a run or after it:

```console
$ helix ui
HELIX UI for my-project: http://127.0.0.1:8790/#token=Ux3kQ9vN2pLm7RtA5yWc8dEf1HgJ4sZb
Read-only. Press Ctrl+C to stop.
```

It reads `helix.toml` and `.helix/`, plus the agents' session files for live transcripts, and never writes to any of
them. Before the first run it shows the configured loop. Its flags are in the
[CLI reference](../reference/cli.md#helix-ui-options).

![The helix ui dashboard during a run: header numbers, the helix progress chart, the loop with live numbers, run health, and the current generation's agents with a live transcript](../assets/ui/helix-ui-light.png#only-light)
![The helix ui dashboard during a run: header numbers, the helix progress chart, the loop with live numbers, run health, and the current generation's agents with a live transcript](../assets/ui/helix-ui-dark.png#only-dark)

## Remote runs

Start the dashboard where the run is, then forward the port it prints (8790 unless that was taken):

```bash
ssh my-server 'cd ~/my-project && helix ui --no-browser'
ssh -N -L 8790:127.0.0.1:8790 my-server
```

Then open the printed address on your machine. Leave `--host` at `127.0.0.1`. Any other address lets anyone who can
reach the port and has the token read your validation scores and evaluator feedback.

## Views

The main page is the **Loop**. The **Views** menu at the top right opens the Frontier, Lineage, Path and Proposals,
and clicking any candidate opens its page. Every view has its own URL, such as `#lineage/g12-s3` or
`#candidate/g12-s3`, so you can link to it.

### Run summary

The header shows the best validation score, the seed's score, generations, evaluations, an estimate of the time left
before the first limit is hit, and spend. The chart below plots the best score so far at each generation. With several
objectives (`frontier_type` other than `instance`), the header shows the best of each.

Run health shows generations since the last new best, the share of proposals that pass the minibatch gate, time per
phase, and failures (agent timeouts, agent errors, evaluator errors). Spend is broken down per generation, with the
share spent on proposals that were rejected or failed. Codex reports tokens but no cost, so a Codex run shows tokens.

### Current generation

The current generation lists each proposal with its phase and elapsed time. Select one to follow its agent's
transcript as it runs (see [Live transcripts](#live-transcripts)).

### Sandbox and protected files

The Loop draws the Docker sandbox when the agent runs in one, and the evaluator on the host or in its sidecar. A card
under mutation lists the files the agent may change, the protected files it must not change (a candidate that changes
one is rejected before it is scored), and how many files the sandbox hides from it. Without the sandbox, the agent can
read every file in its worktree. Settings that look like mistakes are marked in amber, such as web tools turned off
while the agent's shell can still reach the network.

## Frontier, Lineage, Path and Proposals

- **Frontier**: the candidates kept as parents. Each holds the best score on at least one frontier key (by default, a
  validation instance), and none is dominated by the others, as in GEPA. A strip shows the best score among them on
  each instance.
- **Lineage**: accepted and rejected proposals, each joined to its parent, with the path to the best candidate
  highlighted. Rejected proposals are small open dots. Click a node for its score, outcome and change summary;
  double-click it to open its page.
- **Path**: the best candidate's ancestors from the seed, one step per accepted change, with the score, the files
  changed and the cost of each step.
- **Proposals**: a table of every proposal, filterable by outcome (frontier, dominated, rejected, skipped, failed,
  running) and sortable by score, change from the parent, or cost.

A candidate's page has its diff against its parent (or the seed), the agent's change summary, the task prompt the
agent was given, its cost, and its children. **Instances** lists its score on every instance it was scored on; the
**Changed** tab shows where it differs from its parent. Click an instance for the evaluator's feedback on it. A
rejected candidate's page says where it was rejected and why. Its diff and transcript go with its worktree, unless
`evolution.retain_rejected_worktrees` keeps them.

![The Lineage view of an eterna100 run in generation 12: the path to the best candidate, g6-s25 at 21 of 30, runs straight across from the seed through g1-s5 and g4-s16; frontier members are ringed, rejected proposals are small open circles, and groups of three or four rejected children of one parent show as circles with their count](../assets/ui/lineage-light.png#only-light)
![The Lineage view of an eterna100 run in generation 12: the path to the best candidate, g6-s25 at 21 of 30, runs straight across from the seed through g1-s5 and g4-s16; frontier members are ringed, rejected proposals are small open circles, and groups of three or four rejected children of one parent show as circles with their count](../assets/ui/lineage-dark.png#only-dark)

*The Lineage view. Frontier members are ringed. Three or more rejected children of one parent in one generation share
a circle with their count.*

![The Path view of the same run: from the seed at 5 of 30 to g1-s5 at 20 (3 of 6 proposals from the seed accepted), g4-s16 at 19 (one fewer than its parent) and g6-s25 at 21, with files changed, agent time and tokens for each step and the generations between steps](../assets/ui/path-light.png#only-light)
![The Path view of the same run: from the seed at 5 of 30 to g1-s5 at 20 (3 of 6 proposals from the seed accepted), g4-s16 at 19 (one fewer than its parent) and g6-s25 at 21, with files changed, agent time and tokens for each step and the generations between steps](../assets/ui/path-dark.png#only-dark)

*The Path view. HELIX kept g4-s16 although it solves 19 of 30 validation puzzles, one fewer than its parent.*

<small>Screenshots from November 2026; `scripts/docs_screenshots.py` regenerates them from a running dashboard.</small>

## Scores and costs

- A score labelled `val` is a validation mean. One labelled `gate` comes from a gate: for a rejected proposal, the
  score it was rejected on; for a perfect-score skip, the parent's. When every score is 0 or 1, scores are shown as
  successes out of n.
- Candidates with the same score on every frontier key share one place on the frontier, as in GEPA, so the candidate
  with the best mean can be listed as dominated.
- Without a training set (no `dataset.train_size` and no `seedless.train_path`), the gate scores the single task once
  per proposal.
- Codex counts cached input inside its input tokens. The dashboard subtracts it, so uncached input, cached input and
  output add up to the run's total.
- Spend is projected from the median cost of a finished generation. Codex runs, which report no cost, get no
  projection. Runs from before 0.4.0 have no per-generation cost and project from the total.

## Keyboard shortcuts

Escape closes the open panel or returns to the Loop. In the Lineage, with a node selected, Left goes to its parent,
Right to its best child, and Up and Down to its neighbours in the same generation. In Proposals, Up and Down move the
selection and Enter opens it.

## Live transcripts

Transcripts are shown for the `codex` and `claude` backends, whichever one the agent ran with. They come from the
session file the CLI writes as it runs. While the agent runs, you see that file. When it finishes, you see the copy
HELIX keeps in the candidate's worktree.

Without the sandbox, the dashboard finds the session file by the agent's working directory and start time. With the
Docker sandbox, all agents share one login volume, so HELIX records each agent call's session id in
`.helix/timing.jsonl` (a `session` record) and the dashboard opens the file with that id, even when several agents run
in parallel. It reads the volume with a short-lived container that mounts it read-only, has no network, and runs at
most once every 4 seconds per transcript on screen.

Sandboxed runs from before 0.4.0 recorded no session ids. Their transcripts are matched by start time, and two agents
that started in the same second can show each other's transcript; the panel then says "session matched to this agent
by start time".

With `evolution.sibling_history` on, the change-summary call's transcript replaces the mutation's while it runs. A
single transcript record larger than 1 MiB is skipped.

## Security

- The dashboard never writes to the project, `.helix/`, worktrees, git refs or Docker volumes. Like every `helix`
  command it loads the project's `.env`; it reads no other credential files, such as the agents' logins.
- It masks known token formats and secret-like assignments in evaluator commands, transcripts, diffs and feedback, and
  shows environment variables by name only.
- The page itself loads without the token, but all run data needs the token in the printed address. The page moves the
  token out of the URL into the tab's session storage and sends it as an `Authorization: Bearer` header.

## Limits

- A run killed with `SIGKILL` writes no end marker, so it looks like it is still running. The dashboard marks a run
  that has written nothing for a long time as "Running?".
- Tokens and cost are added when an agent call ends, so spend lags the live transcript.
- The parent's score on the minibatch gate is not saved, so a rejected proposal shows only the child's score.
- Settings come from `helix.toml`. `helix evolve --generations` is not recorded, so a run that goes past the file's
  `max_generations` shows no generation total and no time left from it.
