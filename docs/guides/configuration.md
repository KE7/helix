# Configuration

HELIX is configured via `helix.toml` in your project root. A minimal config is on [Writing an evaluator](../getting-started/evaluator.md).

## Full Example

Every key below is optional except `objective` and `[evaluator].command`. The values are examples, not necessarily the defaults.

```toml
# What you want the code to do better
objective = "Maximize sum of radii of 26 non-overlapping circles packed in a unit square"

# Starting directory (default: current directory)
seed = "."

# RNG seed for deterministic parent selection (default: 0)
rng_seed = 0

# Environment variables let through the env scrub into evaluator and agent
# subprocesses, such as API keys the evaluator needs (default: none)
# passthrough_env = ["ANTHROPIC_API_KEY"]

[env]
# Fixed non-secret env values injected into evaluator and agent subprocesses
# after passthrough_env. Useful for repeatable run-local service endpoints.
# ANTHROPIC_BASE_URL = "https://model-service.example.invalid/v1"
# ANTHROPIC_API_KEY = "dummy"

[evaluator]
command = "uv run python evaluate.py"
# The only supported parser is "helix_result".
# "helix_result" takes a per-example list matching GEPA optimize_anything's
# `tuple[float, SideInfo] | float` union — each entry is either a bare
# score or a [score, side_info] pair, mixed allowed:
#   HELIX_RESULT=[s_0, s_1, ...]                        # all bare
#   HELIX_RESULT=[[s_0, si_0], [s_1, si_1], ...]        # all rich
#   HELIX_RESULT=[s_0, [s_1, si_1], s_2, ...]           # mixed
# Positional to `helix_batch.json`. HELIX zips it into id-keyed
# `instance_scores` and stores the side_info list for the reflection
# prompt. `HELIX_RESULT` is a machine protocol; use `from helix import log`
# for human-readable evaluator notes.
include_stdout = true
include_stderr = true
extra_commands = []               # additional commands to run for context
protected_files = ["evaluate.py"] # optional extra files HELIX must keep immutable

[dataset]
# Cardinality of the train / val splits.  Used by HELIX's minibatch
# sampler to generate example ids (stringified indices by default, or
# opaque "group__N" ids when evolution.batch_sampler = "stratified")
# that the evaluator (running in the worktree) filters against its own
# dataset via helix_batch.json — written as an opaque JSON list[str].
# Leave both unset for GEPA O.A. Single-Task Search / HELIX single-task-no-example mode (dataset=None, valset=None).
# train_size = 200
# val_size  = 200

[seedless]
# Seedless mode: generate initial candidate from objective via LLM.
# The seed prompt does not show evaluator.command (as in GEPA), so the
# objective or agent.background must name the file and interface the
# evaluator expects.
enabled = false
# Optional prompt-grounding training dataset (used only in seedless
# seed generation).  Accepts a JSON array file, a JSONL file, or a
# directory of JSON files.  When provided, the first 3 examples are
# included in the seed-generation prompt for representative grounding.
# train_path = "puzzles/train"
# val_path   = "puzzles/val"

[evolution]
max_generations = 20             # generation cap, at least 1 (default: 10; 0 is a config error)
perfect_score_threshold = 1.0    # skip proposals whose instance_scores all reach this (default: unset)
max_evaluations = -1             # evaluation budget cap (-1 = no cap)
# stop_at_score = 0.95           # end the run once the best val score is >= this (unset = off);
                                 # unlike perfect_score_threshold, this stops the run
merge_enabled = false            # enable merge/crossover operations
max_merge_invocations = 5        # total merge cap across entire run
merge_val_overlap_floor = 5      # minimum val-set overlap for merge candidates
merge_subsample_size = 5         # stratified val subsample size for merge acceptance (GEPA parity)
max_workers = 8                  # thread-pool cap for parent-eval + mutation pools
                                 # (default: os.cpu_count(), or 32 if that returns None)
num_parallel_proposals = 1       # parallel mutations per generation
minibatch_size = 3               # train-set minibatch size for reflective mutation
cache_evaluation = true          # reuse per-instance evaluator results
acceptance_criterion = "strict_improvement"
val_stage_size = 0               # optional first-N val gate; passing results carry into the tail
sibling_history = false          # show the mutator every child already proposed from the same
                                 # parent: each one's training-gate score, whether it was kept, and a note on
                                 # what it changed. Every child is kept in state.
                                 # Advisory only — HELIX never filters a proposal on it.
sibling_summary_max_words = 150  # budget for each of those notes; quoted to the summarising
                                 # call and enforced when its answer is read back
# sibling_history_max_entries = 5  # show only the N most recent of those children in the
                                 # prompt (FIFO); unset shows all. State keeps every one.
frontier_type = "instance"       # Pareto dimensionality (GEPA FrontierType parity):
                                 # "instance" | "objective" | "hybrid" | "cartesian".
                                 # The four modes mirror GEPA optimize_anything;
                                 # "instance" is HELIX's default.
                                 # Non-instance axes use per-example
                                 # side_info["scores"] dicts. Without them,
                                 # hybrid warns/no-ops its objective axis and
                                 # continues on instances; objective/cartesian
                                 # raise MissingObjectiveScoresError at selection.

[agent]
backend = "claude"               # "agy" | "claude" | "codex" | "cursor" | "opencode" | "pi"
# model = "sonnet"               # optional backend-specific model name
effort = "medium"                # optional: "low" | "medium" | "high" | "xhigh" | "max"
max_turns = 20                   # hard cap for claude; an in-prompt hint for other backends
# background = "Only modify files under src/. Do not touch tests/ or config/."
# web_access = true              # false = no built-in web tools (claude/codex/opencode/pi)

[sandbox]
enabled = false                  # true = run mutation and merge agents in Docker
evaluator = false                # true (with enabled = true) = run evaluation in Docker
                                 # through [evaluator.sidecar], which is then required
# image = "ghcr.io/ke7/helix-evo-runner-claude:latest"  # optional; defaults from agent.backend
network = "bridge"               # "bridge" | "none" | "host" | an existing Docker network
                                 # (for example an --internal network with an egress proxy)
skip_special_files = true        # skip FIFOs/sockets/devices during workspace sync
# omit_from_agent = ["evaluate.py", "data/val"]  # paths left out of the agent's copy
# Agent containers mount a persistent Docker auth volume named
# helix-auth-<backend>. Run `helix sandbox login <backend>` once per backend.
# auth_volume = "my-run-home"    # optional: mount this volume at /home/node instead;
                                 # `helix sandbox login` never writes it, so use it
                                 # only when the credential reaches the agent another
                                 # way (e.g. examples/circle_packing_sandbox's gateway)

[worktree]
base_dir = ".helix/worktrees"
```

## Backend notes

**pi** (`backend = "pi"`):

- `model` takes pi's `provider/id` form, for example `"anthropic/claude-sonnet-4-5"` or a custom provider from `~/.pi/agent/models.json` such as `"jitengine/flint-int4"`. `pi --list-models` lists them.
- `effort` becomes `--thinking` (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`). pi runs a model that has no thinking support at `off`, without a warning.
- pi has no turn cap, so `max_turns` is only a request in the prompt.
- A local engine that answers one request at a time, such as JITEngine, queues the requests of parallel proposals. A request that waits longer than pi's `httpIdleTimeoutMs` (300000 ms by default) is cut ("terminated"), and after three retries the mutation fails. With `num_parallel_proposals` x `mutations_per_parent` above 1 on such an engine, set `"httpIdleTimeoutMs": "disabled"` in `~/.pi/agent/settings.json` (in the sandbox, the `helix-auth-pi` volume's copy).
