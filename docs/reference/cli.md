# CLI reference


| Command | Description |
|---|---|
| `helix init` | Initialize HELIX in the current directory — creates `helix.toml` and `.helix/` |
| `helix sandbox login BACKEND` | Log into an agent backend inside its persistent Docker auth volume |
| `helix sandbox status [BACKEND]` | Show sandbox login status for one backend or all supported backends |
| `helix sandbox logout BACKEND` | Log out a backend from its persistent Docker auth volume |
| `helix evolve` | Run the evolutionary loop |
| `helix frontier` | Display the current Pareto frontier as a table |
| `helix best` | Show the best candidate; `--export PATH` copies its files (what HELIX committed for it, without HELIX's prompt, logs and batch files) |
| `helix history` | Show the candidate lineage as a tree |
| `helix resume` | Resume a previously interrupted evolution run |
| `helix clean` | Remove all worktrees and `.helix/` state (with confirmation) |
| `helix log` | Show semantic mutation log — full trajectory with parent lineage |
| `helix attempts` | Surface rejected attempt records and perfect-skip events from `.helix/attempts/` and `.helix/skips/` |
| `helix timing` | Show where the run's time went (work by kind and candidate fate, wall time per generation); `--perfetto FILE` exports a timeline |

## `helix evolve` Options

```
--dir PATH          Project directory containing helix.toml (default: .)
--config PATH       Path to config file (default: helix.toml)
--objective TEXT     Override the objective string
--evaluator TEXT     Override the evaluator command
--generations INT   Override max_generations (0 = no cap; then max_evaluations must be set)
--stop-at-score X   Override stop_at_score: end the run once the best val score reaches X
--no-merge          Disable merge operations
--backend BACKEND   Override the mutation backend [agy|claude|codex|cursor|opencode|pi]
--model TEXT        Override the backend model (backend-specific naming)
--effort LEVEL      Reasoning effort: low | medium | high | xhigh | max
```

## `helix attempts` Options

```
--dir PATH          Project directory (default: .)
--generation INT    Only this generation
--stage STAGE       Only attempts whose Stage column is STAGE: train_minibatch | train | val_stage
--reason TEXT       Only attempts whose reason contains TEXT, e.g. minibatch_gate, duplicate_child
--cid ID            Full JSON for one attempt
--json              Machine-readable output
--skips             Show perfect-subsample skips from .helix/skips/ instead
```

Every command that reads or writes a run (`evolve`, `resume`, `frontier`,
`best`, `history`, `log`, `attempts`, `timing`, `clean`) takes `--dir` for
the project directory; a relative path is resolved against the current
directory. Every command takes `--help` for its full options, for example
`helix best --help`.
