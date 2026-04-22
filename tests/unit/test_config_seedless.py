"""Unit tests for seedless-mode additions to helix.config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

import json

from helix.config import (
    EvaluatorConfig,
    HelixConfig,
    SeedlessConfig,
    load_config,
    load_dataset_examples,
)
from helix.mutator import build_seed_generation_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "helix.toml"
    p.write_text(textwrap.dedent(content))
    return p


def make_config(**kwargs) -> HelixConfig:
    defaults = dict(
        objective="Maximise test coverage",
        evaluator=EvaluatorConfig(command="pytest -q"),
    )
    defaults.update(kwargs)
    return HelixConfig(**defaults)


# ---------------------------------------------------------------------------
# SeedlessConfig.enabled default
# ---------------------------------------------------------------------------


class TestSeedlessDefault:
    def test_seedless_default_false(self):
        """HelixConfig.seedless.enabled must default to False."""
        config = make_config()
        assert config.seedless.enabled is False


# ---------------------------------------------------------------------------
# SeedlessConfig.enabled=True
# ---------------------------------------------------------------------------


class TestSeedlessCanBeSetTrue:
    def test_seedless_can_be_set_true(self):
        """seedless.enabled=True should validate with non-empty objective."""
        config = make_config(
            seedless=SeedlessConfig(enabled=True),
            objective="Generate an optimised solver",
        )
        assert config.seedless.enabled is True

    def test_seedless_false_explicit(self):
        config = make_config(seedless=SeedlessConfig(enabled=False))
        assert config.seedless.enabled is False


# ---------------------------------------------------------------------------
# Seedless requires non-empty objective
# ---------------------------------------------------------------------------


class TestSeedlessRequiresObjective:
    def test_seedless_requires_nonempty_objective(self):
        with pytest.raises((ValueError, ValidationError)):
            HelixConfig(
                objective="",
                seedless=SeedlessConfig(enabled=True),
                evaluator=EvaluatorConfig(command="pytest -q"),
            )

    def test_seedless_requires_nonempty_objective_whitespace(self):
        with pytest.raises((ValueError, ValidationError)):
            HelixConfig(
                objective="   ",
                seedless=SeedlessConfig(enabled=True),
                evaluator=EvaluatorConfig(command="pytest -q"),
            )

    def test_non_seedless_empty_objective_allowed(self):
        config = HelixConfig(
            objective="",
            evaluator=EvaluatorConfig(command="pytest -q"),
        )
        assert config.seedless.enabled is False


# ---------------------------------------------------------------------------
# Seedless TOML round-trip
# ---------------------------------------------------------------------------


class TestSeedlessTomlRoundtrip:
    def test_seedless_true_from_toml(self, tmp_path):
        toml = write_toml(
            tmp_path,
            """
            objective = "Optimise the packing algorithm"

            [evaluator]
            command = "python evaluate.py"

            [seedless]
            enabled = true
            """,
        )
        cfg = load_config(toml)
        assert cfg.seedless.enabled is True

    def test_seedless_false_from_toml(self, tmp_path):
        toml = write_toml(
            tmp_path,
            """
            objective = "Optimise the packing algorithm"

            [evaluator]
            command = "python evaluate.py"

            [seedless]
            enabled = false
            """,
        )
        cfg = load_config(toml)
        assert cfg.seedless.enabled is False

    def test_seedless_absent_from_toml(self, tmp_path):
        toml = write_toml(
            tmp_path,
            """
            objective = "Optimise the packing algorithm"

            [evaluator]
            command = "python evaluate.py"
            """,
        )
        cfg = load_config(toml)
        assert cfg.seedless.enabled is False


# ---------------------------------------------------------------------------
# SeedlessConfig.train_path — GEPA mode parity tests
# ---------------------------------------------------------------------------


class TestSeedlessConfigTrainPath:
    def test_train_path_defaults_to_none(self):
        sc = SeedlessConfig()
        assert sc.train_path is None

    def test_train_path_accepts_path(self, tmp_path: Path):
        p = tmp_path / "train.json"
        p.write_text("[]")
        sc = SeedlessConfig(train_path=p)
        assert sc.train_path == p

    def test_helixconfig_seedless_train_path_defaults_none(self):
        config = HelixConfig(
            objective="Solve the problem",
            evaluator=EvaluatorConfig(command="pytest -q"),
        )
        assert config.seedless.train_path is None

    def test_helixconfig_seedless_train_path_roundtrip(self, tmp_path: Path):
        p = tmp_path / "train.jsonl"
        p.write_text('{"q": "hello"}\n')
        config = HelixConfig(
            objective="Solve the problem",
            evaluator=EvaluatorConfig(command="pytest -q"),
            seedless=SeedlessConfig(enabled=True, train_path=p),
        )
        assert config.seedless.train_path == p

    def test_train_path_from_toml(self, tmp_path: Path):
        train_file = tmp_path / "train.json"
        train_file.write_text('[{"x": 1}]')
        toml = write_toml(
            tmp_path,
            f"""
            objective = "Solve the problem"

            [evaluator]
            command = "python evaluate.py"

            [seedless]
            enabled = true
            train_path = {json.dumps(str(train_file))}
            """,
        )
        cfg = load_config(toml)
        assert cfg.seedless.train_path == train_file

    def test_effective_val_path_falls_back(self, tmp_path: Path):
        p = tmp_path / "train.json"
        p.write_text("[]")
        sc = SeedlessConfig(train_path=p)
        assert sc.effective_val_path == p

    def test_effective_val_path_prefers_val_path(self, tmp_path: Path):
        tp = tmp_path / "train.json"
        vp = tmp_path / "val.json"
        tp.write_text("[]")
        vp.write_text("[]")
        sc = SeedlessConfig(train_path=tp, val_path=vp)
        assert sc.effective_val_path == vp


# ---------------------------------------------------------------------------
# load_dataset_examples — file loading tests
# ---------------------------------------------------------------------------


class TestLoadDatasetExamples:
    def test_json_array_file(self, tmp_path: Path):
        data = [{"id": i, "question": f"q{i}"} for i in range(5)]
        p = tmp_path / "train.json"
        p.write_text(json.dumps(data))
        result = load_dataset_examples(p)
        assert len(result) == 3
        for i, s in enumerate(result):
            assert str(i) in s

    def test_jsonl_file(self, tmp_path: Path):
        lines = [json.dumps({"id": i, "val": f"item{i}"}) for i in range(5)]
        p = tmp_path / "train.jsonl"
        p.write_text("\n".join(lines))
        result = load_dataset_examples(p)
        assert len(result) == 3
        assert "item0" in result[0]
        assert "item1" in result[1]
        assert "item2" in result[2]

    def test_directory_of_json_files(self, tmp_path: Path):
        d = tmp_path / "train"
        d.mkdir()
        for i in range(5):
            (d / f"q0{i}.json").write_text(json.dumps({"id": i, "q": f"question{i}"}))
        result = load_dataset_examples(d)
        assert len(result) == 3
        assert "question0" in result[0]

    def test_max_examples_respected(self, tmp_path: Path):
        data = [{"id": i} for i in range(10)]
        p = tmp_path / "train.json"
        p.write_text(json.dumps(data))
        result = load_dataset_examples(p, max_examples=3)
        assert len(result) == 3

    def test_fewer_than_max_is_ok(self, tmp_path: Path):
        data = [{"id": 0}, {"id": 1}]
        p = tmp_path / "train.json"
        p.write_text(json.dumps(data))
        result = load_dataset_examples(p, max_examples=3)
        assert len(result) == 2

    def test_raises_on_missing_path(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            load_dataset_examples(tmp_path / "nonexistent.json")

    def test_returns_list_of_strings(self, tmp_path: Path):
        data = [{"x": i} for i in range(3)]
        p = tmp_path / "train.json"
        p.write_text(json.dumps(data))
        result = load_dataset_examples(p)
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# build_seed_generation_prompt — GEPA mode parity via dataset_examples
# ---------------------------------------------------------------------------


class TestBuildSeedGenerationPromptDatasetExamples:
    def test_no_sample_inputs_section_when_none(self):
        prompt = build_seed_generation_prompt(
            objective="Solve X",
            dataset_examples=None,
        )
        assert "Sample Inputs" not in prompt

    def test_sample_inputs_section_present_when_examples_given(self):
        examples = ["input_1", "input_2"]
        prompt = build_seed_generation_prompt(
            objective="Solve X",
            dataset_examples=examples,
        )
        assert "Sample Inputs" in prompt
        assert "input_1" in prompt
        assert "input_2" in prompt

    def test_at_most_3_examples_included(self):
        examples = ["ex1", "ex2", "ex3", "ex4", "ex5"]
        prompt = build_seed_generation_prompt(
            objective="Solve X",
            dataset_examples=examples,
        )
        assert "ex1" in prompt
        assert "ex2" in prompt
        assert "ex3" in prompt
        assert "ex4" not in prompt
        assert "ex5" not in prompt

    def test_evolution_train_path_populates_examples(self, tmp_path: Path):
        """Simulate evolution.py: seedless.train_path set → examples loaded."""
        train_data = [
            {"q": "What is 2+2?"},
            {"q": "What is 3+3?"},
            {"q": "What is 4+4?"},
        ]
        train_file = tmp_path / "train.json"
        train_file.write_text(json.dumps(train_data))

        config = HelixConfig(
            objective="Solve X",
            evaluator=EvaluatorConfig(command="pytest -q"),
            seedless=SeedlessConfig(enabled=True, train_path=train_file),
        )
        _dataset_examples: list[str] | None = None
        if config.seedless.train_path is not None:
            _dataset_examples = load_dataset_examples(config.seedless.train_path)
        prompt = build_seed_generation_prompt(
            objective=config.objective,
            background=config.agent.background,
            evaluator_cmd=config.evaluator.command,
            dataset_examples=_dataset_examples,
        )
        assert "Sample Inputs" in prompt
        assert "What is 2+2?" in prompt

    def test_evolution_no_train_path_skips_section(self):
        config = HelixConfig(
            objective="Solve X",
            evaluator=EvaluatorConfig(command="pytest -q"),
            seedless=SeedlessConfig(enabled=True),
        )
        assert config.seedless.train_path is None
        _dataset_examples: list[str] | None = None
        if config.seedless.train_path is not None:
            _dataset_examples = load_dataset_examples(config.seedless.train_path)
        prompt = build_seed_generation_prompt(
            objective=config.objective,
            background=config.agent.background,
            evaluator_cmd=config.evaluator.command,
            dataset_examples=_dataset_examples,
        )
        assert "Sample Inputs" not in prompt
