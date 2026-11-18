"""Real Git regression coverage for protected .gitignore and inherited artifacts."""

from pathlib import Path
import subprocess

import pytest

from helix.config import EvaluatorConfig, HelixConfig
from helix.display import UsageStats
from helix.evolution import _candidate_content_key
from helix.evaluator_manifest import (
    _build_evaluator_integrity_manifest,
    _detect_evaluator_tamper,
    _refresh_protected_evaluator_files,
)
from helix.merger import merge
from helix.mutator import (
    _backend_artifact_names,
    _ignore_helix_artifacts,
    _write_mutation_prompt_artifact,
    capture_base_commit,
    generate_change_summary,
    MUTATION_PROMPT_ARTIFACT_NAME,
    MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME,
    SUMMARY_PROMPT_ARTIFACT_NAME,
    SUMMARY_PROMPT_ARTIFACT_FALLBACK_NAME,
    SUMMARY_BACKEND_ARTIFACT_INFIX,
)
from helix.population import Candidate
from helix.sibling_history import SUMMARY_ARTIFACT_NAME
from helix.worktree import _ensure_git_repo, changed_files, clone_candidate, snapshot_candidate


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


def affected_project(project: Path) -> Candidate:
    project.mkdir()
    (project / ".gitignore").write_text(
        "*\n!*/\n!*.py\n!*.json\n!*.md\n!.gitignore\n.helix/\n"
    )
    (project / "solver.py").write_text("value = 1\n")
    _ensure_git_repo(project)
    (project / "helix_batch.json").write_text('["old-synthetic-batch-id"]\n')
    git(project, "add", "-f", "helix_batch.json")
    git(
        project, "-c", "user.name=HELIX", "-c", "user.email=helix@noreply",
        "commit", "-qm", "affected parent",
    )
    return Candidate(
        id="g0-s0",
        worktree_path=str(project),
        branch_name="main",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def test_snapshot_and_cache_exclude_artifacts_under_a_gitignore_allowlist(tmp_path):
    project = tmp_path / "project"
    parent = affected_project(project)
    child = clone_candidate(parent, "g1-s0", tmp_path / "worktrees")
    wt = Path(child.worktree_path)
    _write_mutation_prompt_artifact(str(wt), "new mutation prompt")
    (wt / "helix_batch.json").write_text('["current-synthetic-batch-id"]\n')
    (wt / ".helix_backend_result.json").write_text("{}\n")
    (wt / "solver.py").write_text("value = 2\n")
    (wt / "fixtures").mkdir()
    (wt / "fixtures/helix_batch.json").write_text('["solver-fixture"]\n')
    git(wt, "add", "-f", MUTATION_PROMPT_ARTIFACT_NAME)

    content_key = _candidate_content_key(child)
    (wt / ".helix_backend_result.json").write_text('{"changed": true}\n')
    assert _candidate_content_key(child) == content_key
    assert changed_files(child, parent) == ["fixtures/helix_batch.json", "solver.py"]
    snapshot_candidate(child, "test snapshot")
    assert git(wt, "rev-parse", "HEAD^{tree}").strip() == content_key
    assert set(git(wt, "ls-tree", "-r", "--name-only", "HEAD").splitlines()) == {
        ".gitignore", "fixtures/helix_batch.json", "solver.py",
    }
    assert git(wt, "show", "HEAD:solver.py") == "value = 2\n"
    assert (wt / MUTATION_PROMPT_ARTIFACT_NAME).read_text() == "new mutation prompt"
    assert (wt / "helix_batch.json").is_file()
    assert (wt / ".helix_backend_result.json").is_file()


@pytest.mark.parametrize("edit_solver", [False, True])
def test_summary_ignores_artifact_retirement_and_allowlisted_artifacts(
    tmp_path, mocker, edit_solver
):
    parent = affected_project(tmp_path / "project")
    child = clone_candidate(parent, "g1-s0", tmp_path / "worktrees")
    wt = Path(child.worktree_path)
    base_sha = capture_base_commit(str(wt))
    _write_mutation_prompt_artifact(str(wt), "new mutation prompt")
    (wt / ".helix_backend_result.json").write_text('{"private": "invocation-only"}\n')
    if edit_solver:
        (wt / "solver.py").write_text("value = 2\n")

    def summarize(*args, **kwargs):
        (wt / SUMMARY_ARTIFACT_NAME).write_text("Changed the solver value.")
        return {}, UsageStats()

    backend = mocker.patch("helix.mutator.invoke_claude_code", side_effect=summarize)
    config = HelixConfig(objective="Improve solver", evaluator=EvaluatorConfig(command="true"))
    summary, _ = generate_change_summary(str(wt), config, base_sha)
    if edit_solver:
        backend.assert_called_once()
        prompt = backend.call_args.args[1]
        assert "+value = 2" in prompt
        assert "old-synthetic-batch-id" not in prompt
        assert "invocation-only" not in prompt
        assert "new mutation prompt" not in prompt
        assert summary == "Changed the solver value."
    else:
        backend.assert_not_called()


@pytest.mark.parametrize("with_ancestor", [False, True])
def test_merge_prompt_excludes_artifacts_from_affected_history(
    tmp_path, mocker, with_ancestor
):
    project = tmp_path / "project"
    parent = affected_project(project)
    worktrees = tmp_path / "worktrees"
    child = clone_candidate(parent, "g1-s0", worktrees)
    wt = Path(child.worktree_path)
    _write_mutation_prompt_artifact(str(wt), "new mutation prompt")
    (wt / "solver.py").write_text("value = 2\n")
    snapshot_candidate(child, "test snapshot")
    backend = mocker.patch("helix.merger.invoke_claude_code", return_value=({}, UsageStats()))
    config = HelixConfig(objective="Improve solver", evaluator=EvaluatorConfig(command="true"))

    assert merge(
        parent, child, "g2-m0", config, worktrees,
        project_root=project,
        ancestor=parent if with_ancestor else None,
    ) is not None
    prompt = backend.call_args.args[1]
    assert "+value = 2" in prompt
    assert "helix_batch.json" not in prompt
    assert "old-synthetic-batch-id" not in prompt


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
    assert f"/{MUTATION_PROMPT_ARTIFACT_NAME}" in once.splitlines()
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
