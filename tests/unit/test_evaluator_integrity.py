"""Unit tests for evaluator integrity hardening in helix.evolution."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from helix.config import EvaluatorConfig, HelixConfig
from helix.evolution import (
    _build_evaluator_integrity_manifest,
    _collect_protected_evaluator_paths,
    _detect_evaluator_tamper,
    _load_evaluator_integrity_manifest,
    _refresh_and_snapshot_protected_evaluator_files,
    _refresh_protected_evaluator_files,
    _write_evaluator_integrity_manifest,
)
from helix.exceptions import HelixError
from helix.population import Candidate


def _candidate(path: str) -> Candidate:
    return Candidate(
        id="g1-s1",
        worktree_path=path,
        branch_name="helix/g1-s1",
        generation=1,
        parent_id="g0-s0",
        parent_ids=["g0-s0"],
        operation="mutate",
    )


def test_collects_auto_and_explicit_protected_paths(tmp_path):
    config = HelixConfig(
        objective="test",
        evaluator=EvaluatorConfig(
            command="python evaluate.py",
            extra_commands=["bash scripts/report.sh"],
            protected_files=["fixtures/goldens.json"],
        ),
    )
    protected = _collect_protected_evaluator_paths(config, tmp_path)
    assert protected == [
        "evaluate.py",
        "fixtures/goldens.json",
        "scripts/report.sh",
    ]


def test_rejects_protected_path_outside_project(tmp_path):
    config = HelixConfig(
        objective="test",
        evaluator=EvaluatorConfig(
            command="python evaluate.py",
            protected_files=["/etc/passwd"],
        ),
    )
    with pytest.raises(HelixError):
        _collect_protected_evaluator_paths(config, tmp_path)


def test_detects_modified_and_deleted_evaluator_files(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "evaluate.py").write_text("print('ok')\n")
    (baseline / "fixtures").mkdir()
    (baseline / "fixtures" / "goldens.json").write_text('{"score": 1}\n')

    config = HelixConfig(
        objective="test",
        evaluator=EvaluatorConfig(
            command="python evaluate.py",
            protected_files=["fixtures/goldens.json"],
        ),
    )

    manifest = _build_evaluator_integrity_manifest(config, baseline, baseline)
    assert sorted(manifest.keys()) == ["evaluate.py", "fixtures/goldens.json"]

    candidate_dir = tmp_path / "candidate"
    shutil.copytree(baseline, candidate_dir)
    (candidate_dir / "evaluate.py").write_text("print('tampered')\n")
    (candidate_dir / "fixtures" / "goldens.json").unlink()

    violations = _detect_evaluator_tamper(_candidate(str(candidate_dir)), manifest)
    assert violations == ["evaluate.py", "fixtures/goldens.json"]


def test_manifest_roundtrip(tmp_path):
    manifest = {"evaluate.py": "abc123", "fixtures/goldens.json": "def456"}
    _write_evaluator_integrity_manifest(tmp_path, manifest)
    loaded = _load_evaluator_integrity_manifest(tmp_path)
    assert loaded == manifest


def test_refreshes_protected_files_and_directories_from_current_root(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "evaluate.py").write_text("fixed evaluator\n")
    (project / "splits" / "instance_ids").mkdir(parents=True)
    (project / "splits" / "train.yaml").write_text("seeds: [0, 1]\n")
    (project / "splits" / "instance_ids" / "cube_lifting__0.json").write_text("{}\n")
    (project / "splits" / "instance_ids" / "cube_lifting__1.json").write_text("{}\n")

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "evaluate.py").write_text("stale evaluator\n")
    (worktree / "splits" / "instance_ids").mkdir(parents=True)
    (worktree / "splits" / "train.yaml").write_text("trials_per_task: 1\n")
    (worktree / "splits" / "instance_ids" / "cube_lifting__0.json").write_text("{}\n")

    config = HelixConfig(
        objective="test",
        evaluator=EvaluatorConfig(
            command="python evaluate.py",
            protected_files=["splits"],
        ),
    )

    _refresh_protected_evaluator_files(_candidate(str(worktree)), config, project)

    assert (worktree / "evaluate.py").read_text() == "fixed evaluator\n"
    assert (worktree / "splits" / "train.yaml").read_text() == "seeds: [0, 1]\n"
    instance_ids = sorted((worktree / "splits" / "instance_ids").glob("*.json"))
    assert [path.name for path in instance_ids] == [
        "cube_lifting__0.json",
        "cube_lifting__1.json",
    ]


def test_refresh_snapshot_normalizes_protected_file_baseline(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "evaluate.py").write_text("current evaluator\n")

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    subprocess.run(["git", "init"], cwd=worktree, check=True, capture_output=True)
    (worktree / "evaluate.py").write_text("stale evaluator\n")
    subprocess.run(["git", "add", "evaluate.py"], cwd=worktree, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "seed",
        ],
        cwd=worktree,
        check=True,
        capture_output=True,
    )

    config = HelixConfig(
        objective="test",
        evaluator=EvaluatorConfig(command="python evaluate.py"),
    )
    candidate = _candidate(str(worktree))
    manifest = _build_evaluator_integrity_manifest(config, project, project)

    _refresh_and_snapshot_protected_evaluator_files(candidate, config, project)

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    assert _detect_evaluator_tamper(candidate, manifest) == []

    (worktree / "evaluate.py").write_text("backend tampered\n")
    assert _detect_evaluator_tamper(candidate, manifest) == ["evaluate.py"]
