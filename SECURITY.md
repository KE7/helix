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

### 3. Mutation backends can run locally or in Docker

By default, HELIX runs mutation backends locally in their candidate worktree. For
example, Claude Code is invoked with `--dangerously-skip-permissions`. This
means the backend can read, edit, create, or delete any file inside its working
directory, which is the HELIX candidate worktree at `.helix/worktrees/gN-sN/`.
Worktrees are isolated via `git worktree`, but the backend is free to do
anything within that worktree.

For new projects, the recommended setup is Docker sandboxing:

```toml
[sandbox]
enabled = true
```

In sandboxed mode, HELIX copies each candidate worktree into a temporary Docker
workspace. Agent-side changes are synced back after the backend exits;
evaluator-side changes are discarded. HELIX does not mount the host project
root, parent directories, or home directory into agent/evaluator containers by
default. Agent containers receive only their backend-specific persistent auth
volume, created by `helix sandbox login <backend>`.

**This means:** sandboxing reduces the blast radius of a compromised mutation
backend from "anything the local CLI can access" to the copied candidate
workspace plus the backend auth volume and configured network access. It is not
a defense against a malicious `helix.toml` or evaluator script.

## Threat Model

- HELIX is NOT designed to defend against a malicious `helix.toml` file.
- HELIX is NOT designed to defend against a malicious evaluator script.
- HELIX IS designed to isolate each candidate mutation in a detached git worktree so mutations cannot corrupt the users source tree.
- With `[sandbox].enabled = true`, HELIX IS designed to avoid direct host project/home mounts for agent and evaluator subprocesses.
- HELIX IS designed to never modify the users existing git `.git/config` or remote state.

## Dependencies

HELIX depends on Claude Code CLI and on whatever LLM evaluators the user configures. Security and availability of those upstream components are out of HELIXs control.
