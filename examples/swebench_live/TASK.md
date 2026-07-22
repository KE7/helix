# Pinned coding-agent task

The evolved artifact is `coding_agent.py`, not a precomputed patch. During each
official evaluation it runs as an unprivileged user against a fresh copy of the
pinned Capstone repository at commit
`56db8c2b690eb6372c91f8d76621f43a33c4dbe4`.

Public issue: Capstone 6 incorrectly disassembles the SystemZ/S390x instruction
`c60000000005` as `exrl 0, 0xa`, then exposes a null register operand and hits
the register-number assertion. Capstone 5 produced `%r0, 0xa`. Upstream issue
discussion establishes that LLVM deliberately represents this encoding's zero
register field as `NoRegister`; for Capstone's public detail API, the safe
representation is an immediate zero rather than a null register operand.

The agent must inspect the repository, implement the smallest production fix,
and leave the working tree diff as its answer. It receives no gold patch, hidden
test patch, expected test lists, credentials, Docker socket, or network access.
The private runner subsequently resets the repository, applies the official
test patch and the agent diff, runs the row's exact rebuild/test/print commands,
executes the official row parser, and applies the upstream PASS_TO_PASS and
FAIL_TO_PASS resolution rule. Auxiliary diagnostics never affect the score.
