"""examples/eterna100/evaluate.py: the stored-sequence guard and the splits."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "eterna100"
SEQUENCE = "GC" * 25


def _evaluator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eterna100_evaluate", EXAMPLE / "evaluate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_designer_file_with_a_stored_sequence_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "designer.py").write_text("def design(t, b, s):\n    return 'A' * len(t)\n")
    (tmp_path / "lookup.py").write_text(f"TABLE = ['{SEQUENCE}']\n")
    monkeypatch.chdir(tmp_path)
    assert _evaluator()._stored_sequence() == "lookup.py:1"


def test_helix_files_and_the_puzzles_are_not_designer_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # HELIX writes the agent's prompt, which quotes training designs, into the worktree.
    (tmp_path / "designer.py").write_text("def design(t, b, s):\n    return 'A' * len(t)\n")
    (tmp_path / ".agent_task_prompt.md").write_text(f"design {SEQUENCE}\n")
    (tmp_path / ".helix_artifacts").mkdir()
    (tmp_path / ".helix_artifacts" / "t.jsonl").write_text(SEQUENCE)
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "puzzles.tsv").write_text(SEQUENCE)
    monkeypatch.chdir(tmp_path)
    assert _evaluator()._stored_sequence() is None


def test_a_nested_directory_named_like_a_split_is_still_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "designer.py").write_text("def design(t, b, s):\n    return 'A' * len(t)\n")
    (tmp_path / "lib" / "train").mkdir(parents=True)
    (tmp_path / "lib" / "train" / "table.py").write_text(f"T = '{SEQUENCE}'\n")
    monkeypatch.chdir(tmp_path)
    assert _evaluator()._stored_sequence() == "lib/train/table.py:1"


def test_the_splits_cover_the_benchmark_once() -> None:
    ids: list[str] = []
    for split, size in (("train", 20), ("val", 30), ("test", 50)):
        rows = list(csv.DictReader(open(EXAMPLE / split / "puzzles.tsv", newline=""), delimiter="\t"))
        assert len(rows) == size
        assert all(set(r["structure"]) <= set("().") for r in rows)
        ids += [r["id"] for r in rows]
    assert sorted(ids) == [f"p{i:03d}" for i in range(1, 101)]
