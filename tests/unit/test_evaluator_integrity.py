"""Unit tests for evaluator integrity hardening in helix.evolution."""

from __future__ import annotations

import shutil

import pytest

from helix.config import EvaluatorConfig, HelixConfig
from helix.evolution import (
    _build_evaluator_integrity_manifest,
    _collect_protected_evaluator_paths,
    _detect_evaluator_tamper,
    _load_evaluator_integrity_manifest,
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

