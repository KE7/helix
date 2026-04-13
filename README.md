<div align="center">

# 🧬 HELIX

### Hierarchical Evolution via LLM-Informed eXploration

*DNA evolves. So does your codebase.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![PyPI](https://img.shields.io/pypi/v/helix-evo.svg)](https://pypi.org/project/helix-evo/) [![CI](https://github.com/KE7/helix/actions/workflows/ci.yml/badge.svg)](https://github.com/KE7/helix/actions/workflows/ci.yml) [![mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/)

<br>

**Evolutionary optimization for full codebases using Claude Code as the mutation engine and git worktrees as the population pool.**

HELIX brings reflective Pareto evolution out of the single-artifact setting and into real software projects: entire repositories, multi-turn agentic mutation, tool use, web research, and verification loops, all inside a single evolutionary stage.

<br>

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Configuration](#-configuration) · [CLI Reference](#-cli-reference) · [Results](#-results)

</div>

---

> **Safety:** HELIX never modifies your working branch, HEAD, staging area, or remote. All mutations live in detached worktrees under `.helix/worktrees/` and branches named `helix/*`. If your checkout is dirty, HELIX snapshots the current tracked and untracked changes into the seed worktree while leaving your original checkout untouched. Run `helix clean` to remove saved state and worktrees when you are done.

---

## Why HELIX?

HELIX is built for a setting that today's evolution systems still do not really handle, including systems like KISS and OpenEvolve: improving **real, multi-file codebases** where useful mutations require exploration, iteration, and tooling, not just a single blind rewrite.

Instead of treating one file or one patch as the candidate, HELIX treats the **entire repository** as the evolving organism. Each mutation is a full agentic coding session running inside an isolated git worktree, so a candidate can:

1. **Read across the codebase** to understand architecture and dependencies.
2. **Edit multiple files coherently** in one mutation.
3. **Use tools mid-mutation** like tests, linters, shell commands, and web search.
4. **Take multiple turns to diagnose and self-correct** before the mutation is scored.
5. **Stay inside one evolutionary stage** rather than requiring an outer orchestration loop to get tool use or iteration.

The result is a new kind of evolutionary optimizer: one that preserves the reflective Pareto-evolutionary core while making it practical for whole repositories and realistic software engineering tasks.

---

## ✨ Key Features

| | Feature | Description |
|---|---|---|
| 🧬 | **Whole-codebase evolution** | The candidate is your repository, not a single file, prompt, or patch |
| 📂 | **Multi-file editing** | Mutate entire directory trees — edit `auth.py:42` and `routes.py:18` in one coherent session |
| 🔁 | **Multi-turn mutations** | A single mutation can inspect, edit, test, revise, and continue before being evaluated |
| 🔧 | **Tool access during mutation** | Claude Code can read, grep, run tests, inspect the codebase, and use the web mid-mutation |
| ✅ | **Self-verification** | Mutations verify themselves by running commands before committing |
| 📊 | **Pareto frontier** | Instance-level Pareto selection across test cases — no single metric bottleneck |
| ⚡ | **Parallel evaluation** | Worktrees are isolated → embarrassingly parallel via `ProcessPoolExecutor` |
| 🔀 | **Merge / crossover** | Combine two frontier candidates that excel on different instances |
| 🎯 | **Convergence detection** | Auto-stop when the frontier stagnates for N generations |
| 💾 | **State persistence & resume** | Crash-safe — resume from any generation with `helix resume` |
| 🚦 | **Gated mutations** | Dev-set gating rejects regressions before Pareto evaluation |
| 📋 | **Semantic mutation log** | Full trajectory with root-cause analysis, changes made, and parent lineage |

---

## 🔄 How It Works

```
                    ┌──────────────┐
                    │   Seed Code  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Evaluate   │◄──────────────────────┐
                    │  (parallel)  │                       │
                    └──────┬───────┘                       │
                           │                               │
                           ▼                               │
                    ┌──────────────┐                       │
                    │ Select Parent│                       │
                    │  (Pareto)    │                       │
                    └──────┬───────┘                       │
                           │                               │
                           ▼                               │
               ┌───────────────────────┐                   │
               │  Mutate via Claude    │                   │
               │  Code in Worktree     │                   │
               │                       │                   │
               │  • Read files         │                   │
               │  • Edit multi-file    │                   │
               │  • Run tests          │                   │
               │  • Self-correct       │                   │
               └───────────┬───────────┘                   │
                           │                               │
                           ▼                               │
                    ┌──────────────┐     ┌──────────┐      │
                    │  Gate on Dev │────▶│  Reject  │      │
                    │  (regress?)  │ yes └──────────┘      │
                    └──────┬───────┘                       │
                       no  │                               │
                           ▼                               │
                    ┌──────────────┐                       │
                    │Pareto Update │                       │
                    │ (val scores) │                       │
                    └──────┬───────┘                       │
                           │                               │
                      ┌────┴────┐                          │
                      │ Merge?  │ every N gens             │
                      └────┬────┘                          │
                           │                               │
                           ▼                               │
                    ┌──────────────┐                       │
                    │   Cleanup    │                       │
                    │  dominated   │───────────────────────┘
                    └──────────────┘
```

**The loop in detail:**

1. **Seed** — Your starting code is copied into a git worktree and evaluated
2. **Evaluate** — Run your evaluator command; parse scores per test/instance
3. **Select** — Pick a parent from the Pareto frontier (weighted by instance wins)
4. **Mutate** — Spawn Claude Code in an isolated worktree with full tool access. It reads files, diagnoses failures, makes surgical multi-file edits, and runs commands to verify
5. **Gate** — Re-evaluate on the dev set. Reject if the mutation caused regressions
6. **Pareto Update** — Evaluate on the val set and update the Pareto frontier
7. **Merge** — Periodically combine two complementary frontier candidates via Claude Code
8. **Cleanup** — Remove dominated worktrees; persist state; repeat

---

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/KE7/helix.git
cd helix
pip install -e .

# Verify
helix --help
```

### Initialize a Project

```bash
cd your-project/
helix init
```

This creates a `helix.toml` config file and a `.helix/` directory. Edit `helix.toml` to set your objective and evaluator.

### Whole-repo-as-candidate Model

HELIX treats your **entire working tree** as the candidate. There is no `target_file` — Claude Code may read, edit, create, or delete any file in the project tree during each mutation. A minimal project layout looks like:

```
my-project/
├── helix.toml       # HELIX config (run `helix init` to generate)
├── evaluate.py      # Your evaluator script (must output JSON with a "score" key)
├── solve.py         # File(s) you want to evolve (Claude Code will find them)
└── ...              # Any other files; HELIX will consider them too
```

To restrict what Claude Code touches, set `claude.background` in `helix.toml`:

```toml
[claude]
background = "Only modify files under src/. Do not edit tests/ or config/."
```

### Run Evolution

```bash
helix evolve
```

### View Results

```bash
# Show the Pareto frontier
helix frontier

# Show the best candidate
helix best

# Export best candidate to a directory
helix best --export ./best-solution

# View full mutation log
helix log
```

---

## ⚙️ Configuration

HELIX is configured via `helix.toml` in your project root.

### Minimal Example

```toml
objective = "Maximize test pass rate and code coverage"

[evaluator]
command = "pytest --tb=short -q"
```

When your evaluator needs project dependencies, make `evaluator.command` use the
same environment those dependencies are installed in. Good patterns are
`uv run python evaluate.py` or a wrapper like `bash run_eval.sh`. Avoid bare
`python3 evaluate.py` unless that interpreter already has everything your
evaluator imports.

### Full Example

```toml
# What you want the code to do better
objective = "Maximize sum of radii of 26 non-overlapping circles packed in a unit square"

# Starting directory (default: current directory)
seed = "."

# RNG seed for deterministic parent selection (default: 0)
rng_seed = 0

[evaluator]
command = "uv run python evaluate.py"
# Available parsers: "pytest" | "exitcode" | "json_accuracy" | "json_score"
score_parser = "json_score"
include_stdout = true
include_stderr = true
extra_commands = []               # additional commands to run for context
protected_files = ["evaluate.py"] # optional extra files HELIX must keep immutable

[dataset]
# Cardinality of the train / val splits.  Used by HELIX's minibatch
# sampler to generate integer indices that the evaluator (running in
# the worktree) filters against its own dataset via helix_batch.json.
# Leave both unset for single-task mode (legacy full-batch path).
# train_size = 200
# val_size  = 200

[seedless]
# Seedless mode: generate initial candidate from objective via LLM
enabled = false
# Optional prompt-grounding training dataset (used only in seedless
# seed generation).  Accepts a JSON array file, a JSONL file, or a
# directory of JSON files.  When provided, the first 3 examples are
# included in the seed-generation prompt for representative grounding.
# train_path = "puzzles/train"
# val_path   = "puzzles/val"

[evolution]
max_generations = 20
gating_threshold = 0.0           # minimum improvement to accept mutation
perfect_score_threshold = 1.0    # stop early if score reaches this
convergence_patience = 5         # stop if no improvement for N generations
max_metric_calls = 200
merge_enabled = false            # enable merge/crossover operations
max_merge_invocations = 5        # total merge cap across entire run
merge_val_overlap_floor = 5      # minimum val-set overlap for merge candidates
mutation_rate = 1.0              # probability of mutation (1.0 = always)
max_workers = 1                  # evaluator worker pool size
num_parallel_proposals = 1       # parallel mutations per generation
minibatch_size = 3               # train-set minibatch gate size
cache_evaluation = true          # reuse per-instance evaluator results
acceptance_criterion = "strict_improvement"
val_stage_size = 0               # optional first-N val gate before full val

[claude]
model = "sonnet"                 # or "opus", "haiku", full model name
effort = "medium"                # optional: "low" | "medium" | "high" | "max"
max_turns = 20
allowed_tools = ["Read", "Edit", "Write", "Bash", "Glob", "Grep"]
# background = "Only modify files under src/. Do not touch tests/ or config/."

[worktree]
base_dir = ".helix/worktrees"
```

### Dataset Modes

HELIX splits dataset concerns across two TOML sections:

| Section | Purpose |
|---|---|
| `[dataset]` | Cardinality only — `train_size` / `val_size` — drives the minibatch sampler when the evaluator owns the dataset and HELIX hands off positional indices via `helix_batch.json` (Architecture A). |
| `[seedless]` | Seedless-mode toggle + optional prompt-grounding paths (`train_path` / `val_path`) — used only during seed generation to show the LLM representative inputs. |

| Mode | Config | Description |
|---|---|---|
| **Single-task** | neither set | Optimize for a single task. Legacy full-batch evaluator path. |
| **Positional-index handoff** | `dataset.train_size` / `dataset.val_size` set | HELIX samples integer indices into `range(train_size)`; the evaluator reads `helix_batch.json` from cwd and filters its own dataset. |
| **Seedless multi-task** | `seedless.enabled = true`, `seedless.train_path` set | Seed generation prompt includes the first 3 training examples for grounding. |

HELIX does not own separate dataset files for train/dev/val; your evaluator remains
the source of truth. During evolution HELIX sets `HELIX_SPLIT` (`train`, `dev`, or
`val`) so evaluator-owned datasets can switch behavior by phase.

When `evolution.val_stage_size` is set to a positive value and `dataset.val_size` is also set, accepted mutation proposals run a deterministic first-N validation stage before the full validation sweep. Stage-only results are never added to the frontier; HELIX still persists only full-val results for Pareto ranking and resume stability.

### Evaluator Integrity

HELIX can lock evaluator-critical files so mutations and merges cannot game the
score by editing the benchmark itself.

```toml
[evaluator]
command = "uv run python evaluate.py"
score_parser = "json_accuracy"
protected_files = [
  "evaluate.py",
  "goldens.json",
  "helpers/evaluator_utils.py",
]
```

At run start, HELIX hashes the evaluator command target plus any
`evaluator.protected_files` entries and writes the manifest to
`.helix/evaluator_manifest.json`. Candidates that modify any protected file are
rejected before evaluation.

### Score Parsers

HELIX includes 4 built-in score parsers to extract metrics from evaluator output:

| Parser | Input | Output | Use Case |
|---|---|---|---|
| **pytest** | Parses `pytest -q` stdout | `scores`: `pass_rate`, `duration`<br>`instance_scores`: per-test pass/fail | Unit test suites |
| **exitcode** | Exit code only | `scores`: `success` (1.0 or 0.0) | Simple pass/fail evaluators |
| **json_accuracy** | JSON stdout with `accuracy` field | `scores`: `accuracy`<br>`instance_scores`: per-instance scores | Classification and benchmark tasks |
| **json_score** | JSON stdout with `score` field | `scores`: `score`<br>`instance_scores`: `score` | Optimization tasks (e.g., circle packing) |

**Example evaluator outputs:**

```python
# json_score parser expects:
print(json.dumps({"score": 2.63}))

# json_accuracy parser expects:
print(json.dumps({
    "accuracy": 0.85,
    "instance_scores": {"puzzle_001": 1.0, "puzzle_002": 0.0}
}))
```

---

## 📖 CLI Reference

| Command | Description |
|---|---|
| `helix init` | Initialize HELIX in the current directory — creates `helix.toml` and `.helix/` |
| `helix evolve` | Run the evolutionary loop |
| `helix frontier` | Display the current Pareto frontier as a table |
| `helix best` | Show the best candidate; `--export PATH` to copy it out |
| `helix history` | Show the candidate lineage as a tree |
| `helix resume` | Resume a previously interrupted evolution run |
| `helix clean` | Remove all worktrees and `.helix/` state (with confirmation) |
| `helix log` | Show semantic mutation log — full trajectory with parent lineage |

### `helix evolve` Options

```
--dir PATH          Project directory containing helix.toml (default: .)
--config PATH       Path to config file (default: helix.toml)
--objective TEXT     Override the objective string
--evaluator TEXT     Override the evaluator command
--generations INT   Override max_generations
--no-merge          Disable merge operations
--model TEXT        Claude model (e.g. sonnet, opus, claude-sonnet-4-5)
--effort LEVEL      Reasoning effort: low | medium | high | max
```

---

## 🧪 Results

### 🔵 Circle Packing

Pack 26 non-overlapping circles in a unit square, maximizing sum of radii.

| | Score | Config |
|---|:---:|---|
| Seed (naive concentric grid) | 0.9798 | — |
| **HELIX best (gen 14 of 30)** | **2.6360** | `haiku` · `low` effort · `max_turns=20` |
| GEPA optimize_anything ([blog](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)) | 2.635 | gemini-3-flash |

> **Note:** HELIX **beat the GEPA blog benchmark** (2.6360 vs 2.635) using Claude Haiku with low reasoning effort and a 20-turn per mutation budget. See [`examples/circle_packing/`](examples/circle_packing/) for the full fixture including `solve_optimized.py` (the best evolved solution).

---

## 🏗️ Architecture

```
.helix/
├── config.toml          # Snapshot of helix.toml at run start
├── evaluator_manifest.json # Protected evaluator file hashes
├── state.json           # Generation, frontier, budget
├── lineage.json         # Full ancestry graph
├── log/                 # Semantic mutation logs
│   ├── g1-m0.json
│   └── g2-x0.json
├── worktrees/
│   ├── g0-s0/           # Seed
│   ├── g1-m1/           # Gen 1 Mutation 1
│   └── g2-x1/           # Gen 2 Merge 1
└── evaluations/
    └── g0-s0.json       # EvalResult per candidate
```

### Module Overview

| Module | Role |
|---|---|
| `cli.py` | Click CLI — init, evolve, frontier, best, history, resume, clean, log |
| `config.py` | TOML config parsing via Pydantic v2 |
| `evolution.py` | Main generation loop with gating, merge, convergence |
| `population.py` | `Candidate`, `EvalResult`, `ParetoFrontier` |
| `worktree.py` | Git worktree lifecycle (create, clone, snapshot, remove) |
| `executor.py` | Run evaluator commands, parallel evaluation |
| `mutator.py` | Claude Code mutation invocation with autonomous system prompt |
| `merger.py` | Claude Code merge/crossover between complementary candidates |
| `lineage.py` | Ancestry graph tracking |
| `state.py` | Atomic state persistence and resume |
| `display.py` | Rich terminal UI with phase tracking |

---

## 📚 Citation

```bibtex
@software{helix2026,
  title={HELIX: Hierarchical Evolution via LLM-Informed eXploration},
  author={Elmaaroufi, Karim and OMAR},
  year={2026},
  url={https://github.com/KE7/helix}
}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

HELIX's core evolutionary algorithm is based on **GEPA optimize_anything** by Agrawal, Lee, Ma, Elmaaroufi, Tan, Seshia, Sen, Klein, Stoica, Gonzalez, Khattab, Dimakis, and Zaharia. Their work on applying reflective Pareto evolution to any text made HELIX possible — we extended their algorithm to full codebases and agentic mutation but the foundation is theirs.

- **[GEPA optimize_anything](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)** — The algorithmic foundation: minibatch-gated Pareto evolution with reflective LLM mutation
- **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** — The agentic coding tool powering HELIX's mutation and merge engine
- **[OMAR](http://omar.tech/)** — The multi-agent orchestration system used to build HELIX

```bibtex
@article{gepa_optimize_anything2026,
  title={Introducing optimize\_anything},
  author={Agrawal, Lakshya A and Lee, Donghyun and Ma, Wenjie and Elmaaroufi, Karim and Tan, Shangyin and Seshia, Sanjit A. and Sen, Koushik and Klein, Dan and Stoica, Ion and Gonzalez, Joseph E. and Khattab, Omar and Dimakis, Alexandros G. and Zaharia, Matei},
  year={2026},
  url={https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/}
}
```

---

<div align="center">
<sub>Built with 🧬 by evolution, for evolution.</sub>
</div>
