"""Real Git regression coverage for protected .gitignore and inherited artifacts."""

from pathlib import Path
import subprocess

import pytest

from helix.config import EvaluatorConfig, HelixConfig
from helix.evaluator_manifest import (
    _build_evaluator_integrity_manifest,
    _detect_evaluator_tamper,
    _refresh_protected_evaluator_files,
)
from helix.mutator import (
    _backend_artifact_names,
    _ignore_helix_artifacts,
    _write_mutation_prompt_artifact,
    MUTATION_PROMPT_ARTIFACT_NAME,
    MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME,
    SUMMARY_PROMPT_ARTIFACT_NAME,
    SUMMARY_PROMPT_ARTIFACT_FALLBACK_NAME,
    SUMMARY_BACKEND_ARTIFACT_INFIX,
)
from helix.population import Candidate
from helix.sibling_history import SUMMARY_ARTIFACT_NAME
from helix.worktree import _ensure_git_repo, clone_candidate, snapshot_candidate


ARTIFACTS = (
    MUTATION_PROMPT_ARTIFACT_NAME,
    MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME,
    SUMMARY_PROMPT_ARTIFACT_NAME,
    SUMMARY_PROMPT_ARTIFACT_FALLBACK_NAME,
    SUMMARY_ARTIFACT_NAME,
    *_backend_artifact_names(),
    *_backend_artifact_names(SUMMARY_BACKEND_ARTIFACT_INFIX),
    ".helix_artifacts/backend_transcripts/old.jsonl",
    "helix_batch.json",
    ".helix_timing_old.jsonl",
    ".helix_opencode_state/old.db",
)


def git(path, *args):
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    ).stdout


def test_protected_gitignore_with_inherited_artifacts_two_generations(
    tmp_path, monkeypatch
):
    """Children of an affected parent recover without collisions or tamper rejects."""
    monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@example.invalid")
    project = tmp_path / "project"
    project.mkdir()
    (project / ".gitignore").write_text("__pycache__/\n")
    for name in ARTIFACTS:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale " + name)
    _ensure_git_repo(project)
    parent = Candidate(
        id="g0-s0",
        worktree_path=str(project),
        branch_name="main",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )
    config = HelixConfig(
        objective="Improve solver",
        evaluator=EvaluatorConfig(command="true", protected_files=[".gitignore"]),
    )
    manifest = _build_evaluator_integrity_manifest(config, project, project)
    assert set(ARTIFACTS) <= set(git(project, "ls-files").splitlines())

    for generation in (1, 2):
        child = clone_candidate(parent, f"g{generation}-s0", tmp_path / "worktrees")
        wt = Path(child.worktree_path)
        _refresh_protected_evaluator_files(child, config, project)
        assert (
            _write_mutation_prompt_artifact(str(wt), f"task {generation}")
            == MUTATION_PROMPT_ARTIFACT_NAME
        )
        assert not (wt / MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME).exists()
        assert not (wt / SUMMARY_PROMPT_ARTIFACT_FALLBACK_NAME).exists()
        _write_mutation_prompt_artifact(
            str(wt),
            f"summary {generation}",
            primary_name=SUMMARY_PROMPT_ARTIFACT_NAME,
            fallback_name=SUMMARY_PROMPT_ARTIFACT_FALLBACK_NAME,
        )
        for name in ARTIFACTS:
            path = wt / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"new artifact {generation}")
        (wt / "solver.txt").write_text(str(generation))
        # Check while the backend's artifacts exist, before any refresh can
        # hide a spurious .gitignore edit from the tamper detector.
        assert _detect_evaluator_tamper(child, manifest, config, project) == []
        _refresh_protected_evaluator_files(child, config, project)
        snapshot_candidate(child, "test child")
        assert not set(ARTIFACTS) & set(
            git(wt, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
        )
        assert (wt / ".gitignore").read_bytes() == (project / ".gitignore").read_bytes()
        assert _detect_evaluator_tamper(child, manifest, config, project) == []
        parent = child


@pytest.mark.parametrize("gitignore", [None, "*.pyc"])
def test_shared_exclude_preserves_user_files_and_is_idempotent(tmp_path, gitignore):
    project = tmp_path / "project"
    project.mkdir()
    git(project, "init", "-q")
    if gitignore is not None:
        (project / ".gitignore").write_text(gitignore)
    exclude = project / ".git/info/exclude"
    # An exact line must be appended even if a comment mentions its name.
    exclude.write_text("# keep .agent_task_prompt.md out\nuser-pattern")
    _ignore_helix_artifacts(project)
    once = exclude.read_text()
    _ignore_helix_artifacts(project)
    assert exclude.read_text() == once
    assert once.startswith("# keep .agent_task_prompt.md out\nuser-pattern\n")
    assert MUTATION_PROMPT_ARTIFACT_NAME in once.splitlines()
    if gitignore is None:
        assert not (project / ".gitignore").exists()
    else:
        assert (project / ".gitignore").read_text() == gitignore


def test_evolution_recovers_an_affected_history(tmp_path, monkeypatch):
    """Exercise real proposals, snapshots and tamper checks for two generations."""
    import json
    import shlex
    import sys

    from helix.config import AgentConfig, EvolutionConfig, SandboxConfig
    from helix.evolution import run_evolution

    project = tmp_path / "project"
    project.mkdir()
    (project / ".gitignore").write_text(".helix/\n")
    (project / "value.txt").write_text("0")
    for name in ARTIFACTS:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale " + name)
    _ensure_git_repo(project)
    # Evaluator stays outside the candidate and never reaches the fake backend.
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text(
        "from pathlib import Path\n"
        'print("HELIX_RESULT=[[" + Path("value.txt").read_text() + ", {}]]")\n'
    )
    config = HelixConfig(
        objective="Increment value.txt",
        evaluator=EvaluatorConfig(
            command=f"{shlex.quote(sys.executable)} {shlex.quote(str(evaluator))}",
            protected_files=[".gitignore"],
        ),
        evolution=EvolutionConfig(
            max_generations=2,
            max_evaluations=10,
            max_workers=1,
            candidate_selection_strategy="current_best",
        ),
        agent=AgentConfig(backend="codex"),
        sandbox=SandboxConfig(enabled=False),
    )
    real_run = subprocess.run
    calls = []

    def backend_only(args, **kwargs):
        if args[0] != "codex":
            return real_run(args, **kwargs)
        wt = Path(kwargs["cwd"])
        assert (wt / MUTATION_PROMPT_ARTIFACT_NAME).read_text()
        calls.append(wt)
        value = wt / "value.txt"
        value.write_text(str(int(value.read_text()) + 1))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps({"type": "turn.completed", "usage": {}}),
            stderr="",
        )

    monkeypatch.setattr("helix.mutator.subprocess.run", backend_only)
    run_evolution(config, project, project / ".helix")
    assert len(calls) == 2  # A collision stops before either backend invocation.
    assert (calls[-1] / "value.txt").read_text() == "2"
    assert git(calls[-1], "show", "HEAD:value.txt") == "2"
    for wt in calls:
        assert not set(ARTIFACTS) & set(
            git(wt, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
        )
        assert (wt / ".gitignore").read_bytes() == (project / ".gitignore").read_bytes()
