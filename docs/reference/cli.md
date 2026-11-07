# CLI reference


| Command | Description |
|---|---|
| `helix init` | Initialize HELIX in the current directory — creates `helix.toml` and `.helix/` |
| `helix sandbox login BACKEND` | Log into an agent backend inside its persistent Docker auth volume |
| `helix sandbox status [BACKEND]` | Show sandbox login status for one backend or all supported backends |
| `helix sandbox logout BACKEND` | Log out a backend from its persistent Docker auth volume |
| `helix evolve` | Run the evolutionary loop |
| `helix frontier` | Display the current Pareto frontier as a table |
| `helix best` | Show the best candidate; `--export PATH` to copy it out |
| `helix history` | Show the candidate lineage as a tree |
| `helix resume` | Resume a previously interrupted evolution run |
| `helix clean` | Remove all worktrees and `.helix/` state (with confirmation) |
| `helix log` | Show every candidate in order: parent, outcome (frontier, dominated or rejected and why), score change against its parent, files changed, and the agent's change summary when `evolution.sibling_history` is on |
| `helix attempts` | Surface rejected attempt records and perfect-skip events from `.helix/attempts/` and `.helix/skips/` |
| `helix timing` | Show where the run's time went (work by kind and candidate fate, wall time per generation); `--perfetto FILE` exports a timeline |

## `helix evolve` Options

```
--dir PATH          Project directory containing helix.toml (default: .)
--config PATH       Path to config file (default: helix.toml)
--objective TEXT     Override the objective string
--evaluator TEXT     Override the evaluator command
--generations INT   Override max_generations
--stop-at-score X   Override stop_at_score: end the run once the best val score reaches X
--no-merge          Disable merge operations
--backend BACKEND   Override the mutation backend [agy|claude|codex|cursor|opencode|pi]
--model TEXT        Override the backend model (backend-specific naming)
--effort LEVEL      Reasoning effort: low | medium | high | xhigh | max
```

Every command takes `--help` for its full options, for example `helix best --help`.
