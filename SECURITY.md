# Security Policy

## Reporting a Vulnerability

Please open a private security advisory on GitHub: https://github.com/KE7/helix/security/advisories/new

## Trust Boundaries

HELIX executes user-provided evaluators and LLM-generated mutations. The following behaviors are by design and represent trust boundaries users should be aware of:

### 1. Evaluator command allow-list

Evaluator commands in `helix.toml` are parsed with `shlex.split()` and executed with `shell=False`, which eliminates shell metacharacter injection. HELIX additionally enforces an allow-list of accepted first tokens (python, python3, pytest, make, bash, sh, node, uv, poetry, cat) OR any executable whose path starts with `./`, `/usr/bin/`, `/home/`, or `/opt/`.

**This means:** any executable at a whitelisted path prefix can be invoked. If you trust the contents of your `helix.toml` and your evaluator script, this is safe. Do NOT run HELIX on a `helix.toml` from an untrusted source.

### 2. No evaluator timeout

HELIX does not impose a timeout on evaluator subprocess execution. A runaway evaluator (infinite loop, stuck network call) will hang the evolution loop indefinitely. This matches GEPAs behavior and is intentional — users are expected to build their own watchdogs if needed.

### 3. Claude Code runs unsandboxed

When HELIX invokes Claude Code to mutate a candidate, it uses `--dangerously-skip-permissions`. This means Claude Code can read, edit, create, or delete ANY file inside its working directory (the HELIX worktree at `.helix/worktrees/gN-sN/`). Worktrees are isolated via `git worktree`, but Claude Code is free to do anything within that worktree.

**This means:** HELIX should only be run on code you trust the author of, and in environments where a compromised Claude Code session is an acceptable failure mode. For production use, consider running HELIX inside a container or VM.

## Threat Model

- HELIX is NOT designed to defend against a malicious `helix.toml` file.
- HELIX is NOT designed to defend against a malicious evaluator script.
- HELIX IS designed to isolate each candidate mutation in a detached git worktree so mutations cannot corrupt the users source tree.
- HELIX IS designed to never modify the users existing git `.git/config` or remote state.

## Dependencies

HELIX depends on Claude Code CLI and on whatever LLM evaluators the user configures. Security and availability of those upstream components are out of HELIXs control.
