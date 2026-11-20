"""Reserved invocation paths in a candidate's worktree root."""

from helix.sibling_history import SUMMARY_ARTIFACT_NAME

MUTATION_PROMPT_ARTIFACT_NAME = ".agent_task_prompt.md"
MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME = ".agent_internal/task_prompt.md"
SUMMARY_PROMPT_ARTIFACT_NAME = ".agent_summary_prompt.md"
SUMMARY_PROMPT_ARTIFACT_FALLBACK_NAME = ".agent_internal/summary_prompt.md"

# The summary call keeps its own records alongside the mutation's.
SUMMARY_BACKEND_ARTIFACT_INFIX = "_summary"
BACKEND_TRANSCRIPT_ARTIFACT_DIR = ".helix_artifacts/backend_transcripts"


def backend_artifact_names(infix: str = "") -> tuple[str, str, str]:
    """Return the result, stdout and stderr filenames for one backend call."""
    return (
        f".helix_backend{infix}_result.json",
        f".helix_backend{infix}_stdout.txt",
        f".helix_backend{infix}_stderr.txt",
    )


(
    BACKEND_RESULT_ARTIFACT_NAME,
    BACKEND_STDOUT_ARTIFACT_NAME,
    BACKEND_STDERR_ARTIFACT_NAME,
) = backend_artifact_names()

_ARTIFACT_PATTERNS = (
    MUTATION_PROMPT_ARTIFACT_NAME,
    ".agent_internal/",
    *backend_artifact_names(),
    *backend_artifact_names(SUMMARY_BACKEND_ARTIFACT_INFIX),
    ".helix_artifacts/",
    "helix_batch.json",
    ".helix_timing_*.jsonl",
    SUMMARY_ARTIFACT_NAME,
    SUMMARY_PROMPT_ARTIFACT_NAME,
    ".helix_opencode_state/",
)

# Git ignore syntax and command pathspecs differ. Pathspecs match files within
# reserved directories even with no trailing slash, and top keeps them scoped
# to the worktree root.
ARTIFACT_IGNORE_RULES = tuple(f"/{pattern}" for pattern in _ARTIFACT_PATTERNS)
ARTIFACT_PATHSPECS = tuple(
    f":(top,glob){pattern.rstrip('/')}" for pattern in _ARTIFACT_PATTERNS
)
ARTIFACT_EXCLUDE_PATHSPECS = tuple(
    f":(top,glob,exclude){pattern.rstrip('/')}" for pattern in _ARTIFACT_PATTERNS
)
