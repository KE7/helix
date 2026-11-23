# Quickstart

## Prerequisites

- Python 3.12 or newer, and git.
- A coding-agent CLI, installed and logged in, such as Claude Code (the default) or Codex. HELIX runs it as the mutation agent; [`[agent].backend`](../guides/configuration.md) lists the supported ones. Each mutation is one session of it, billed to your account.
- Docker, optional but recommended, to run the agent in a sandbox.

## Installation

```bash
pip install helix-evo
helix --help
```

To work on HELIX itself, clone the repo and run `pip install -e .` there instead.

## Initialize a project

```bash
cd your-project/
helix init
```

This writes `helix.toml`, a `.helix/` state directory, and a `.gitignore` that ignores `.helix/` if you have none. Edit `helix.toml` to set your `objective` (what to improve) and `[evaluator].command` (how to score a candidate). The objective can also go under `[project]`, as the examples do.

Then write the evaluator: see [Writing an evaluator](evaluator.md). Run it by hand once before `helix evolve`.

## What HELIX copies

Each candidate is a git worktree of your repo at `HEAD`, plus your uncommitted and untracked changes. Three things
trip up a first run:

- Files your `.gitignore` excludes are **not** copied. If the evaluator needs an ignored virtualenv, downloaded data
  or a `.env` file, give it an absolute path or set the value under `[env]` in `helix.toml`.
- The evaluator and the agent run with a scrubbed environment. List the variables they need, such as an API key, in
  `passthrough_env`.
- `[evaluator].command` runs without a shell, so pipes and `&&` don't work. Put them in a script.

## Recommended: sandbox the agent

The simplest safe setup runs the mutation agent in a Docker copy of the candidate that leaves out your evaluator, while the evaluator itself runs on your machine:

```toml
[evaluator]
command = "uv run python evaluate.py"
protected_files = ["evaluate.py"]   # hashed; a candidate that edits it is rejected

[sandbox]
enabled = true
omit_from_agent = ["evaluate.py"]   # the agent never sees it
```

Log in to your backend once inside its sandbox volume:

```bash
helix sandbox login claude      # or agy, codex, cursor, opencode, pi
helix sandbox status claude
```

To keep benchmark data and scoring code off your machine's candidate copies entirely, run the evaluator in a private sidecar container as well: see [Docker sandboxing](../guides/sandboxing.md).

## Connect the HELIX agent skill

This repository ships an agent skill at [`skills/helix/`](https://github.com/KE7/helix/tree/main/skills/helix) (if you installed from PyPI, clone the repo to get it) with detailed guidance
for writing `helix.toml`, running and debugging HELIX, and migrating GEPA
`optimize_anything.py` workflows.

For Codex, install it into your local Codex skills directory:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -sfn "$(pwd)/skills/helix" "${CODEX_HOME:-$HOME/.codex}/skills/helix"
```

Then ask Codex:

```text
Use $helix to set up this project.
```

For Claude Code, this checkout registers the same skill at `.claude/skills/helix`
(a symlink to `skills/helix/`), so Claude Code loads it when a task matches, or
on demand:

```text
/helix set up Docker-sandboxed HELIX for this benchmark
```

To use it in every project, link it into your personal Claude Code skills:

```bash
mkdir -p ~/.claude/skills
ln -sfn "$(pwd)/skills/helix" ~/.claude/skills/helix
```

## Which files the agent can change

The candidate is your whole working tree. There is no `target_file`: the agent may read, edit, create or delete any
file in the project during a mutation. A minimal project looks like:

```
my-project/
├── helix.toml       # HELIX config (run `helix init` to generate)
├── evaluate.py      # Your evaluator script (must print a HELIX_RESULT= line)
├── solve.py         # File(s) you want to evolve (the backend will find them)
└── ...              # Any other files; the agent can read and edit them too
```

To steer what the agent changes, say so in `agent.background`. That is an instruction, not a guard: to reject any
candidate that changes a file, list it in `evaluator.protected_files`.

```toml
[agent]
backend = "claude"
# model = "sonnet"  # optional backend-specific model
background = "Only modify files under src/. Do not edit tests/ or config/."
```

## Run evolution

Start with a short run to check the setup end to end. Each generation runs one agent session plus your evaluator, so cost and time depend on your backend and benchmark:

```bash
helix evolve --generations 2
```

Then run the full search (10 generations by default, or `[evolution].max_generations`):

```bash
helix evolve
```

## Look at the results

```bash
# Show the Pareto frontier
helix frontier

# Show the best candidate
helix best

# Export best candidate to a directory
helix best --export ./best-solution

# Every candidate in order: parent, outcome, score, files and what changed
helix log

# A live dashboard of the run in the browser, during or after it
helix ui
```
