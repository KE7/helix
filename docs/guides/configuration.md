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
# Cardinality of the train / val splits.  HELIX samples the ids "0".."n-1"
# from them and writes each batch to helix_batch.json (a JSON list[str]);
# the evaluator (running in the worktree) filters its own dataset by those
# ids.  For grouped ids such as "pushcube__0" (batch_sampler = "stratified"),
# use [seedless].train_path / val_path instead; see the datasets guide.
# Leave both unset, with no [seedless] paths, for GEPA O.A. Single-Task Search /
# HELIX single-task-no-example mode (dataset=None, valset=None).
# train_size = 200
# val_size  = 200

[seedless]
# Seedless mode: generate initial candidate from objective via LLM.
# The seed prompt does not show evaluator.command (as in GEPA), so the
# objective or agent.background must name the file and interface the
# evaluator expects.
enabled = false
# Optional train / val datasets: a JSON array file, a JSONL file, or a
# directory of *.json files.  HELIX reads its train and val instance ids
# from them even with enabled = false: the sorted file stems of a
# directory ("pushcube__0", ...), or "0".."n-1" for a JSON array or
# JSONL file.  With val_path unset, validation uses train_path's ids.
# With enabled = true, the first 3 train examples also go into the
# seed-generation prompt.
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
num_parallel_proposals = 1       # P: parents sampled (with replacement) per iteration
mutations_per_parent = 1         # N: children proposed per sampled parent (>= 1);
                                 # one iteration proposes P*N, each on its own minibatch
proposal_selection = "all_improvements"  # which children that pass the minibatch gate go
                                 # on to full validation: "all_improvements" (default) |
                                 # "best_improvement" (only the largest) | "top_k"
# proposal_top_k = 1             # required with "top_k", 1..P*N; rejected otherwise
candidate_selection_strategy = "pareto"  # how each parent is picked (GEPA candidate_selector):
                                 # "pareto" (default) | "current_best" |
                                 # "epsilon_greedy" | "top_k_pareto"
# candidate_selection_epsilon = 0.1  # required with "epsilon_greedy", 0.0..1.0;
                                 # rejected otherwise; no default (GEPA uses 0.1)
# candidate_selection_top_k = 5  # required with "top_k_pareto", >= 1; rejected
                                 # otherwise; no default (GEPA uses 5)
minibatch_size = 3               # train-set minibatch size for reflective mutation
batch_sampler = "epoch_shuffled" # "epoch_shuffled" (default, GEPA) | "stratified"
group_key_separator = "__"       # stratified: an id's group is its text before the first
                                 # separator ("pushcube__0" -> "pushcube"); must be
                                 # non-empty with "stratified"
# num_sampled_groups = 3         # stratified only; set both or neither, each >= 1:
# num_examples_per_group = 2     # each minibatch takes this many groups x this many
                                 # examples per group. Unset: minibatch_size groups,
                                 # one example each
cache_evaluation = true          # reuse per-instance evaluator results
acceptance_criterion = "strict_improvement"
val_stage_size = 0               # optional first-N val gate; passing results carry into the tail
retain_rejected_worktrees = false  # keep the worktree of a child the val stage rejects, for
                                 # inspection (default: false; `helix clean` removes them)
sibling_history = false          # show the mutator every child already proposed from the same
                                 # parent: each one's training-gate score, whether it was kept, and a note on
                                 # what it changed. Every child is kept in state, and
                                 # `helix log` shows each note's first sentence.
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
# Limits for agent and evaluator-runner containers (default: unset, no limit)
# cpus = 4.0                     # docker --cpus
# memory = "16g"                 # docker --memory
# timeout_seconds = 3600         # wall-clock limit per sandboxed command (one agent call,
                                 # or one evaluator-runner command); that call then fails
pids_limit = 512                 # docker --pids-limit (default: 512)
add_host_gateway = false         # true = map host.docker.internal to the host in agent and
                                 # evaluator-runner containers (Linux; Docker Desktop has it)
# extra_hosts = { "proxy.internal" = "10.0.0.5" }  # more --add-host NAME:IP entries, for
                                 # agent, evaluator-runner and sidecar containers
# omit_from_agent = ["evaluate.py", "data/val"]  # paths left out of the agent's copy
# Agent containers mount a persistent Docker auth volume named
# helix-auth-<backend>. Run `helix sandbox login <backend>` once per backend.
# auth_volume = "my-run-home"    # optional: mount this volume at /home/node instead;
                                 # `helix sandbox login` never writes it, so use it
                                 # only when the credential reaches the agent another
                                 # way (e.g. examples/circle_packing_sandbox's gateway).
                                 # Give each concurrent run its own; see the
                                 # Docker sandboxing guide

[worktree]
base_dir = ".helix/worktrees"
# cleanup_dominated = false      # deprecated and ignored; still accepted so old configs load
```

## Backend notes

**opencode** (`backend = "opencode"`), without the Docker sandbox:

- Each candidate runs opencode with its own data directory, `<worktree>/.helix_opencode_state/.local/share/opencode` (via `XDG_DATA_HOME`), so parallel proposals never share one `opencode.db`.
- Your stored logins still apply: HELIX links your `auth.json` (`$XDG_DATA_HOME/opencode/auth.json`, by default `~/.local/share/opencode/auth.json`, where `opencode providers login` writes) into that directory. It is a link, so a token opencode refreshes is saved to your own file. API-key variables (`OPENCODE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) work as before.
- HELIX never reads or copies your own `opencode.db`; the run's transcript comes from the candidate's database.

**pi** (`backend = "pi"`):

- `model` takes pi's `provider/id` form, for example `"anthropic/claude-sonnet-4-5"` or a custom provider from `~/.pi/agent/models.json` such as `"jitengine/flint-int4"`. `pi --list-models` lists them.
- `effort` becomes `--thinking` (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`). pi runs a model that has no thinking support at `off`, without a warning.
- pi has no turn cap, so `max_turns` is only a request in the prompt.
- A local engine that answers one request at a time, such as JITEngine, queues the requests of parallel proposals. A request that waits longer than pi's `httpIdleTimeoutMs` (300000 ms by default) is cut ("terminated"), and after three retries the mutation fails. With `num_parallel_proposals` x `mutations_per_parent` above 1 on such an engine, set `"httpIdleTimeoutMs": "disabled"` in `~/.pi/agent/settings.json` (in the sandbox, the `helix-auth-pi` volume's copy).
