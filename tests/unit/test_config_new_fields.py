"""Unit tests for new config fields added in Phase 3 (minibatch/GEPA parity)."""

from __future__ import annotations

import textwrap
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from helix.config import (
    DatasetConfig,
    EvaluatorSidecarConfig,
    EvolutionConfig,
    HelixConfig,
    SandboxConfig,
    SeedlessConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# SeedlessConfig.effective_val_path
# ---------------------------------------------------------------------------

class TestSeedlessConfigValPath:
    def test_val_path_none_falls_back_to_train_path(self):
        cfg = SeedlessConfig(train_path=Path("/tmp/train.jsonl"))
        assert cfg.val_path is None
        assert cfg.effective_val_path == Path("/tmp/train.jsonl")

    def test_val_path_set_returns_val_path(self):
        cfg = SeedlessConfig(
            train_path=Path("/tmp/train.jsonl"),
            val_path=Path("/tmp/val.jsonl"),
        )
        assert cfg.effective_val_path == Path("/tmp/val.jsonl")

    def test_both_none_returns_none(self):
        cfg = SeedlessConfig()
        assert cfg.effective_val_path is None


# ---------------------------------------------------------------------------
# DatasetConfig.train_size / val_size (Architecture A positional-index handoff)
# ---------------------------------------------------------------------------

class TestDatasetConfigSizes:
    def test_defaults_none(self):
        cfg = DatasetConfig()
        assert cfg.train_size is None
        assert cfg.val_size is None

    def test_set_sizes(self):
        cfg = DatasetConfig(train_size=200, val_size=200)
        assert cfg.train_size == 200
        assert cfg.val_size == 200

    def test_zero_sizes_allowed(self):
        # train_size=0 is a single-task/no-example marker (no sampled ids).
        cfg = DatasetConfig(train_size=0, val_size=0)
        assert cfg.train_size == 0
        assert cfg.val_size == 0

    def test_negative_train_size_rejected(self):
        with pytest.raises(ValidationError):
            DatasetConfig(train_size=-1)

    def test_negative_val_size_rejected(self):
        with pytest.raises(ValidationError):
            DatasetConfig(val_size=-5)

    def test_toml_loads_sizes(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(textwrap.dedent("""
            objective = "Evaluator-owned dataset"

            [evaluator]
            command = "bash run_eval.sh"

            [dataset]
            train_size = 200
            val_size = 200
        """))
        cfg = load_config(toml)
        assert cfg.dataset.train_size == 200
        assert cfg.dataset.val_size == 200
        # train_path / val_path now live on SeedlessConfig; evaluator-owned
        # datasets use cardinality-only dataset settings instead.
        assert cfg.seedless.train_path is None
        assert cfg.seedless.val_path is None


# ---------------------------------------------------------------------------
# EvolutionConfig defaults for new fields
# ---------------------------------------------------------------------------

class TestEvolutionConfigNewFields:
    def test_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.minibatch_size == 3
        # GEPA parity: max_workers defaults to os.cpu_count() or 32
        # (optimize_anything.py:485).
        import os
        assert cfg.max_workers == (os.cpu_count() or 32)
        assert cfg.num_parallel_proposals == 1
        assert cfg.cache_evaluation is True
        assert cfg.acceptance_criterion == "strict_improvement"
        assert cfg.val_stage_size is None

    def test_acceptance_criterion_accepts_improvement_or_equal(self):
        cfg = EvolutionConfig(acceptance_criterion="improvement_or_equal")
        assert cfg.acceptance_criterion == "improvement_or_equal"

    def test_acceptance_criterion_rejects_invalid(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(acceptance_criterion="greedy")

    def test_val_stage_size_rejects_negative(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(val_stage_size=-1)

    def test_override_all_new_fields(self):
        cfg = EvolutionConfig(
            minibatch_size=5,
            max_workers=4,
            num_parallel_proposals=2,
            cache_evaluation=True,
            acceptance_criterion="improvement_or_equal",
            val_stage_size=50,
        )
        assert cfg.minibatch_size == 5
        assert cfg.max_workers == 4
        assert cfg.num_parallel_proposals == 2
        assert cfg.cache_evaluation is True
        assert cfg.acceptance_criterion == "improvement_or_equal"
        assert cfg.val_stage_size == 50

    def test_num_parallel_proposals_auto_resolves(self):
        """GEPA parity: ``num_parallel_proposals="auto"`` resolves to
        ``max(1, max_workers // minibatch_size)`` in model_post_init.

        Mirrors GEPA ``_resolve_num_parallel_proposals``
        (/tmp/gepa-official/src/gepa/optimize_anything.py:1108-1116).
        """
        cfg = EvolutionConfig(
            num_parallel_proposals="auto",
            max_workers=10,
            minibatch_size=3,
        )
        assert cfg.num_parallel_proposals == 3  # 10 // 3

    def test_num_parallel_proposals_auto_clamps_to_one(self):
        """When ``max_workers < minibatch_size``, ``"auto"`` clamps to 1
        (GEPA: ``max(1, max_workers // minibatch_size)``)."""
        cfg = EvolutionConfig(
            num_parallel_proposals="auto",
            max_workers=2,
            minibatch_size=5,
        )
        assert cfg.num_parallel_proposals == 1


# ---------------------------------------------------------------------------
# evolution.frontier_type — GEPA multi-axis Pareto dimensionality
# ---------------------------------------------------------------------------


class TestEvolutionFrontierType:
    """``evolution.frontier_type`` mirrors GEPA's ``FrontierType``
    literal (``src/gepa/core/state.py:22-23``).  HELIX's default is
    ``"hybrid"`` because O.A. is the right parent for HELIX — GEPA's
    O.A. defaults to ``"hybrid"`` at
    ``src/gepa/optimize_anything.py:476``.  The base ``api.py`` default
    is ``"instance"`` but that's not the right baseline for HELIX.
    """

    def test_default_is_hybrid(self):
        cfg = EvolutionConfig()
        assert cfg.frontier_type == "hybrid"

    @pytest.mark.parametrize(
        "variant", ["instance", "objective", "hybrid", "cartesian"],
    )
    def test_all_literal_variants_accepted(self, variant):
        cfg = EvolutionConfig(frontier_type=variant)
        assert cfg.frontier_type == variant

    def test_invalid_variant_rejected(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(frontier_type="instance_plus_one")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "variant", ["instance", "objective", "hybrid", "cartesian"],
    )
    def test_toml_round_trip_variant(self, tmp_path, variant):
        toml = tmp_path / "helix.toml"
        toml.write_text(textwrap.dedent(f"""
            objective = "Test"

            [evaluator]
            command = "pytest"

            [evolution]
            frontier_type = "{variant}"
        """))
        cfg = load_config(toml)
        assert cfg.evolution.frontier_type == variant

    def test_toml_default_when_omitted(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(textwrap.dedent("""
            objective = "Test"

            [evaluator]
            command = "pytest"
        """))
        cfg = load_config(toml)
        assert cfg.evolution.frontier_type == "hybrid"

    def test_toml_invalid_literal_rejected_at_load(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(textwrap.dedent("""
            objective = "Test"

            [evaluator]
            command = "pytest"

            [evolution]
            frontier_type = "not-a-real-type"
        """))
        with pytest.raises(SystemExit):
            # load_config prints + sys.exit(1) on validation errors.
            load_config(toml)


class TestSandboxConfig:
    def test_defaults_disabled_for_backwards_compatibility(self):
        cfg = SandboxConfig()
        assert cfg.enabled is False
        assert cfg.backend == "docker"
        assert cfg.image is None
        assert cfg.network == "bridge"
        assert cfg.skip_special_files is True

    def test_sandboxed_evaluator_requires_sidecar(self):
        with pytest.raises(ValueError, match=r"\[evaluator.sidecar\]"):
            HelixConfig(
                objective="Test",
                evaluator={"command": "pytest"},
                sandbox=SandboxConfig(enabled=True, evaluator=True),
            )

    def test_sidecar_requires_sandbox(self):
        with pytest.raises(ValueError, match="requires sandbox.enabled"):
            HelixConfig(
                objective="Test",
                evaluator={
                    "command": "python /runner/evaluate.py",
                    "sidecar": {
                        "image": "eval:latest",
                        "command": "python -m server",
                        "endpoint": "http://helix-evaluator:8080/evaluate",
                    },
                },
            )

    def test_sandbox_sidecar_config_is_valid(self):
        cfg = HelixConfig(
            objective="Test",
            evaluator={
                "command": "python /runner/evaluate.py",
                "score_parser": "helix_result",
                "sidecar": EvaluatorSidecarConfig(
                    image="eval:latest",
                    runner_image="eval-runner:latest",
                    command="python -m server",
                    endpoint="http://helix-evaluator:8080/evaluate",
                    healthcheck_command="python /runner/healthcheck.py",
                ),
            },
            sandbox=SandboxConfig(enabled=True, evaluator=True),
        )
        assert cfg.evaluator.sidecar is not None
        assert cfg.evaluator.sidecar.image == "eval:latest"
        assert cfg.evaluator.sidecar.runner_image == "eval-runner:latest"
        assert cfg.evaluator.sidecar.resolved_runner_image == "eval-runner:latest"
        assert cfg.evaluator.sidecar.healthcheck_command == "python /runner/healthcheck.py"

    def test_toml_loads_sandbox_config(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(textwrap.dedent("""
            objective = "Test"

            [evaluator]
            command = "python /runner/evaluate.py"
            score_parser = "helix_result"

            [evaluator.sidecar]
            image = "eval:latest"
            runner_image = "eval-runner:latest"
            command = "python -m server"
            endpoint = "http://helix-evaluator:8080/evaluate"
            healthcheck_command = "python /runner/healthcheck.py"

            [sandbox]
            enabled = true
            evaluator = true
            image = "custom-helix:latest"
            network = "none"
            cpus = 2.0
            memory = "4g"
            timeout_seconds = 300
            add_host_gateway = true
            skip_special_files = false
        """))
        cfg = load_config(toml)
        assert cfg.sandbox.enabled is True
        assert cfg.evaluator.sidecar is not None
        assert cfg.evaluator.sidecar.endpoint == "http://helix-evaluator:8080/evaluate"
        assert cfg.evaluator.sidecar.runner_image == "eval-runner:latest"
        assert cfg.evaluator.sidecar.healthcheck_command == "python /runner/healthcheck.py"
        assert cfg.sandbox.image == "custom-helix:latest"
        assert cfg.sandbox.network == "none"
        assert cfg.sandbox.cpus == 2.0
        assert cfg.sandbox.memory == "4g"
        assert cfg.sandbox.timeout_seconds == 300
        assert cfg.sandbox.add_host_gateway is True
        assert cfg.sandbox.skip_special_files is False

    def test_load_config_loads_adjacent_dotenv_without_overriding(self, tmp_path, monkeypatch):
        toml = tmp_path / "helix.toml"
        toml.write_text(textwrap.dedent("""
            objective = "Test"

            [evaluator]
            command = "pytest"
        """))
        (tmp_path / ".env").write_text(
            "ANTHROPIC_API_KEY=dotenv-key\n"
            "CURSOR_API_KEY='cursor dotenv key'\n"
            "EXISTING=from-dotenv\n"
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setenv("EXISTING", "from-shell")

        load_config(toml)

        assert os.environ["ANTHROPIC_API_KEY"] == "dotenv-key"
        assert os.environ["CURSOR_API_KEY"] == "cursor dotenv key"
        assert os.environ["EXISTING"] == "from-shell"


# ---------------------------------------------------------------------------
# Round-trip TOML loading
# ---------------------------------------------------------------------------

class TestTomlRoundTrip:
    def test_toml_loads_new_fields(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(textwrap.dedent("""
            objective = "Maximise coverage"

            [evaluator]
            command = "pytest"

            [seedless]
            train_path = "/tmp/train.jsonl"
            val_path = "/tmp/val.jsonl"

            [evolution]
            minibatch_size = 7
            max_workers = 3
            num_parallel_proposals = 2
            cache_evaluation = true
            acceptance_criterion = "improvement_or_equal"
            val_stage_size = 50
        """))
        cfg = load_config(toml)
        assert cfg.seedless.train_path == Path("/tmp/train.jsonl")
        assert cfg.seedless.val_path == Path("/tmp/val.jsonl")
        assert cfg.seedless.effective_val_path == Path("/tmp/val.jsonl")
        assert cfg.evolution.minibatch_size == 7
        assert cfg.evolution.max_workers == 3
        assert cfg.evolution.num_parallel_proposals == 2
        assert cfg.evolution.cache_evaluation is True
        assert cfg.evolution.acceptance_criterion == "improvement_or_equal"
        assert cfg.evolution.val_stage_size == 50

    def test_toml_val_path_omitted_falls_back(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(textwrap.dedent("""
            objective = "Maximise coverage"

            [evaluator]
            command = "pytest"

            [seedless]
            train_path = "/tmp/train.jsonl"
        """))
        cfg = load_config(toml)
        assert cfg.seedless.val_path is None
        assert cfg.seedless.effective_val_path == Path("/tmp/train.jsonl")

    def test_model_dump_roundtrip(self):
        cfg = HelixConfig(
            objective="Test",
            evaluator={"command": "pytest"},
            seedless={"train_path": "/tmp/train.jsonl", "val_path": "/tmp/val.jsonl"},
            evolution={
                "minibatch_size": 4,
                "max_workers": 2,
                "cache_evaluation": True,
                "acceptance_criterion": "improvement_or_equal",
                "val_stage_size": 25,
            },
        )
        dumped = cfg.model_dump()
        restored = HelixConfig.model_validate(dumped)
        assert restored.seedless.val_path == Path("/tmp/val.jsonl")
        assert restored.seedless.effective_val_path == Path("/tmp/val.jsonl")
        assert restored.evolution.minibatch_size == 4
        assert restored.evolution.max_workers == 2
        assert restored.evolution.cache_evaluation is True
        assert restored.evolution.acceptance_criterion == "improvement_or_equal"
        assert restored.evolution.val_stage_size == 25
