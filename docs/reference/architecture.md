# Architecture


```
.helix/
├── config.toml          # Snapshot of helix.toml at run start
├── helix.log            # Run log: errors and warnings land here
├── evaluator_manifest.json # Protected evaluator file hashes
├── state.json           # Generation, frontier, budget
├── timing.jsonl         # Timing spans, appended by every run
├── lineage.json         # Ancestry and files changed per candidate
├── worktrees/
│   ├── g0-s0/           # Seed
│   ├── g1-m1/           # Gen 1 Mutation 1
│   └── g2-x1/           # Gen 2 Merge 1
├── evaluations/
│   └── g0-s0.json       # EvalResult per candidate
├── attempts/            # Per-rejected-candidate attempt records (JSON)
└── skips/               # Per-generation perfect-skip event lists (JSON)
```

## Module Overview

| Module | Role |
|---|---|
| `cli.py` | Click CLI — init, evolve, frontier, best, history, resume, clean, log, attempts, timing |
| `config.py` | TOML config parsing via Pydantic v2 |
| `evolution.py` | Main generation loop with gating, merge, and termination on `max_generations` / `max_evaluations` |
| `population.py` | `Candidate`, `EvalResult`, `ParetoFrontier` |
| `worktree.py` | Git worktree lifecycle (create, clone, snapshot, remove) |
| `executor.py` | Run evaluator commands |
| `evaluator_manifest.py` | SHA-256 manifest for protected evaluator files; refresh helpers for mutation and merge candidates |
| `mutator.py` | Backend mutation invocation with autonomous system prompt and HELIX usage artifacts |
| `merger.py` | Backend merge/crossover between complementary candidates |
| `lineage.py` | Ancestry graph tracking |
| `state.py` | Atomic state persistence and resume |
| `display.py` | Rich terminal UI with phase tracking |
| `timing.py` | Always-on JSONL timing spans (`.helix/timing.jsonl`) |
