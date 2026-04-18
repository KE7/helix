# Security Policy

## Reporting a Vulnerability

Please open a private security advisory on GitHub: https://github.com/KE7/helix/security/advisories/new

## Trust Boundaries

HELIX executes user-provided evaluators and LLM-generated mutations. The following behaviors are by design and represent trust boundaries users should be aware of:

### 1. Evaluator command execution

Evaluator commands in `helix.toml` are parsed with `shlex.split()` and executed with `shell=False`. This eliminates shell metacharacter injection — pipes, redirects, and command substitution in the command string are treated as literal arguments.

**This means:** a `helix.toml` author can run any executable they choose (and can trivially run arbitrary code via e.g. `python -c "..."`). HELIX does not gate commands with an allow-list because the `helix.toml` author is already trusted. Do NOT run HELIX on a `helix.toml` from an untrusted source.

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
