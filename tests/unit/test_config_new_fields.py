"""Unit tests for new config fields added in Phase 3 (minibatch/GEPA parity).

Evaluator command examples below refer to an ``evaluate.py`` script that emits
the required ``HELIX_RESULT=`` line; these tests exercise config handling only.
"""

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
        toml.write_text(
            textwrap.dedent("""
            objective = "Evaluator-owned dataset"

            [evaluator]
            command = "bash run_eval.sh"

            [dataset]
            train_size = 200
            val_size = 200
        """)
        )
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
        # (GEPA's ``EngineConfig.max_workers`` in ``gepa_launcher.py``).
        import os

        assert cfg.max_workers == (os.cpu_count() or 32)
        assert cfg.num_parallel_proposals == 1
        assert cfg.cache_evaluation is False
        assert cfg.acceptance_criterion == "strict_improvement"
        assert cfg.val_stage_size is None
        assert cfg.num_sampled_groups is None
        assert cfg.num_examples_per_group is None

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
            batch_sampler="stratified",
            num_sampled_groups=1,
            num_examples_per_group=4,
        )
        assert cfg.minibatch_size == 5
        assert cfg.max_workers == 4
        assert cfg.num_parallel_proposals == 2
        assert cfg.cache_evaluation is True
        assert cfg.acceptance_criterion == "improvement_or_equal"
        assert cfg.val_stage_size == 50
        assert cfg.num_sampled_groups == 1
        assert cfg.num_examples_per_group == 4

    def test_group_example_sampling_fields_must_be_paired(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(num_sampled_groups=1)
        with pytest.raises(ValidationError):
            EvolutionConfig(num_examples_per_group=2)

    def test_group_example_sampling_fields_reject_non_positive(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(
                batch_sampler="stratified",
                num_sampled_groups=0,
                num_examples_per_group=2,
            )
        with pytest.raises(ValidationError):
            EvolutionConfig(
                batch_sampler="stratified",
                num_sampled_groups=1,
                num_examples_per_group=0,
            )

    def test_group_example_sampling_requires_stratified_sampler(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(num_sampled_groups=1, num_examples_per_group=2)

    def test_num_parallel_proposals_auto_rejected(self):
        """``num_parallel_proposals`` is a plain int; the old ``"auto"``
        sentinel is no longer accepted and must fail validation."""
        with pytest.raises(ValidationError):
            EvolutionConfig(num_parallel_proposals="auto")


# ---------------------------------------------------------------------------
# evolution.frontier_type — GEPA FrontierType multi-axis dimensionality
# ---------------------------------------------------------------------------


class TestEvolutionFrontierType:
    """``evolution.frontier_type`` mirrors GEPA's ``FrontierType`` concept
    and selects the Pareto keyspace.  The HELIX default is ``"instance"``:
    it is the only value that every score
    parser can satisfy, since the multi-axis modes need per-example
    objective scores that only ``helix_result`` evaluators opt into.
    """

    def test_default_is_instance(self):
        cfg = EvolutionConfig()
        assert cfg.frontier_type == "instance"

    @pytest.mark.parametrize(
        "variant",
        ["instance", "objective", "hybrid", "cartesian"],
    )
    def test_all_literal_variants_accepted(self, variant):
        cfg = EvolutionConfig(frontier_type=variant)
        assert cfg.frontier_type == variant

    def test_invalid_variant_rejected(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(frontier_type="instance_plus_one")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "variant",
        ["instance", "objective", "hybrid", "cartesian"],
    )
    def test_toml_round_trip_variant(self, tmp_path, variant):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent(f"""
            objective = "Test"

            [evaluator]
            command = "python evaluate.py"

            [evolution]
            frontier_type = "{variant}"
        """)
        )
        cfg = load_config(toml)
        assert cfg.evolution.frontier_type == variant

    def test_toml_default_when_omitted(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Test"

            [evaluator]
            command = "python evaluate.py"
        """)
        )
        cfg = load_config(toml)
        assert cfg.evolution.frontier_type == "instance"

    def test_toml_invalid_literal_rejected_at_load(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Test"

            [evaluator]
            command = "python evaluate.py"

            [evolution]
            frontier_type = "not-a-real-type"
        """)
        )
        with pytest.raises(SystemExit):
            # load_config prints + sys.exit(1) on validation errors.
            load_config(toml)


class TestSandboxConfig:
    def test_toml_loads_fixed_env_config(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Test"

            [env]
            ANTHROPIC_BASE_URL = "https://model-service.example.invalid/v1"
            ANTHROPIC_API_KEY = "dummy"

            [evaluator]
            command = "python evaluate.py"
        """)
        )

        cfg = load_config(toml)

        assert cfg.env == {
            "ANTHROPIC_BASE_URL": "https://model-service.example.invalid/v1",
            "ANTHROPIC_API_KEY": "dummy",
        }

    def test_defaults_disabled_for_backwards_compatibility(self):
        cfg = SandboxConfig()
        assert cfg.enabled is False
        assert cfg.backend == "docker"
        assert cfg.image is None
        assert cfg.network == "bridge"
        assert cfg.extra_hosts == {}
        assert cfg.skip_special_files is True
        assert cfg.preserve_backend_transcripts is True
        assert cfg.transcript_artifact_dir == ".helix_artifacts/backend_transcripts"
        assert cfg.claude_transcript_root == "/home/node/.claude/projects/-workspace"

    def test_sandboxed_evaluator_requires_sidecar(self):
        with pytest.raises(ValueError, match=r"\[evaluator.sidecar\]"):
            HelixConfig(
                objective="Test",
                evaluator={"command": "python evaluate.py"},
                sandbox=SandboxConfig(enabled=True, evaluator=True),
            )

    def test_sandbox_sidecar_config_is_valid(self):
        cfg = HelixConfig(
            objective="Test",
            evaluator={
                "command": "python /runner/evaluate.py",
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
        assert (
            cfg.evaluator.sidecar.healthcheck_command == "python /runner/healthcheck.py"
        )

    def test_toml_loads_sandbox_config(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Test"

            [evaluator]
            command = "python /runner/evaluate.py"

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
            preserve_backend_transcripts = false
            transcript_artifact_dir = ".helix/custom-transcripts"
            claude_transcript_root = "/custom/claude/projects"

            [sandbox.extra_hosts]
            "env-endpoint" = "host-gateway"
            "local-service" = "127.0.0.1"
        """)
        )
        cfg = load_config(toml)
        assert cfg.sandbox.enabled is True
        assert cfg.evaluator.sidecar is not None
        assert cfg.evaluator.sidecar.endpoint == "http://helix-evaluator:8080/evaluate"
        assert cfg.evaluator.sidecar.runner_image == "eval-runner:latest"
        assert (
            cfg.evaluator.sidecar.healthcheck_command == "python /runner/healthcheck.py"
        )
        assert cfg.sandbox.image == "custom-helix:latest"
        assert cfg.sandbox.network == "none"
        assert cfg.sandbox.cpus == 2.0
        assert cfg.sandbox.memory == "4g"
        assert cfg.sandbox.timeout_seconds == 300
        assert cfg.sandbox.add_host_gateway is True
        assert cfg.sandbox.skip_special_files is False
        assert cfg.sandbox.preserve_backend_transcripts is False
        assert cfg.sandbox.transcript_artifact_dir == ".helix/custom-transcripts"
        assert cfg.sandbox.claude_transcript_root == "/custom/claude/projects"
        assert cfg.sandbox.extra_hosts == {
            "env-endpoint": "host-gateway",
            "local-service": "127.0.0.1",
        }

    def test_load_config_loads_adjacent_dotenv_without_overriding(
        self, tmp_path, monkeypatch
    ):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Test"

            [evaluator]
            command = "python evaluate.py"
        """)
        )
        (tmp_path / ".env").write_text(
            "ANTHROPIC_API_KEY=dotenv-key\n"
            "CURSOR_API_KEY='cursor dotenv key'\n"
            "EXISTING=from-dotenv\n"
            # Inline comment after unquoted value (POSIX/foreman convention).
            "OPENAI_API_KEY=plain-key   # this is a trailing comment\n"
            # Hash inside a quoted value must be preserved verbatim.
            'GEMINI_API_KEY="gem#key#with#hashes"\n'
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("EXISTING", "from-shell")

        load_config(toml)

        assert os.environ["ANTHROPIC_API_KEY"] == "dotenv-key"
        assert os.environ["CURSOR_API_KEY"] == "cursor dotenv key"
        assert os.environ["EXISTING"] == "from-shell"
        # Trailing inline comment must be stripped from unquoted values.
        assert os.environ["OPENAI_API_KEY"] == "plain-key"
        # Embedded ``#`` inside a quoted value must be preserved.
        assert os.environ["GEMINI_API_KEY"] == "gem#key#with#hashes"


# ---------------------------------------------------------------------------
# Round-trip TOML loading
# ---------------------------------------------------------------------------


class TestTomlRoundTrip:
    def test_toml_loads_new_fields(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Maximise coverage"

            [evaluator]
            command = "python evaluate.py"

            [seedless]
            train_path = "/tmp/train.jsonl"
            val_path = "/tmp/val.jsonl"

            [evolution]
            minibatch_size = 7
            batch_sampler = "stratified"
            max_workers = 3
            num_parallel_proposals = 2
            cache_evaluation = true
            acceptance_criterion = "improvement_or_equal"
            val_stage_size = 50
            num_sampled_groups = 1
            num_examples_per_group = 4
        """)
        )
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
        assert cfg.evolution.num_sampled_groups == 1
        assert cfg.evolution.num_examples_per_group == 4

    def test_toml_val_path_omitted_falls_back(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Maximise coverage"

            [evaluator]
            command = "python evaluate.py"

            [seedless]
            train_path = "/tmp/train.jsonl"
        """)
        )
        cfg = load_config(toml)
        assert cfg.seedless.val_path is None
        assert cfg.seedless.effective_val_path == Path("/tmp/train.jsonl")

    def test_model_dump_roundtrip(self):
        cfg = HelixConfig(
            objective="Test",
            evaluator={"command": "python evaluate.py"},
            seedless={"train_path": "/tmp/train.jsonl", "val_path": "/tmp/val.jsonl"},
            evolution={
                "minibatch_size": 4,
                "batch_sampler": "stratified",
                "max_workers": 2,
                "cache_evaluation": True,
                "acceptance_criterion": "improvement_or_equal",
                "val_stage_size": 25,
                "num_sampled_groups": 1,
                "num_examples_per_group": 2,
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
        assert restored.evolution.num_sampled_groups == 1
        assert restored.evolution.num_examples_per_group == 2


# ---------------------------------------------------------------------------
# EvolutionConfig.candidate_selection_strategy + its sub-knobs
# ---------------------------------------------------------------------------


class TestCandidateSelectionConfig:
    def test_default_strategy_is_pareto_with_no_sub_knobs(self):
        cfg = EvolutionConfig()
        assert cfg.candidate_selection_strategy == "pareto"
        assert cfg.candidate_selection_epsilon is None
        assert cfg.candidate_selection_top_k is None

    def test_current_best_accepts_no_sub_knobs(self):
        cfg = EvolutionConfig(candidate_selection_strategy="current_best")
        assert cfg.candidate_selection_epsilon is None
        assert cfg.candidate_selection_top_k is None

    def test_epsilon_greedy_requires_epsilon(self):
        with pytest.raises(ValidationError, match="candidate_selection_epsilon is required"):
            EvolutionConfig(candidate_selection_strategy="epsilon_greedy")

    def test_epsilon_greedy_accepts_boundary_values(self):
        cfg = EvolutionConfig(
            candidate_selection_strategy="epsilon_greedy",
            candidate_selection_epsilon=0.0,
        )
        assert cfg.candidate_selection_epsilon == 0.0
        cfg = EvolutionConfig(
            candidate_selection_strategy="epsilon_greedy",
            candidate_selection_epsilon=1.0,
        )
        assert cfg.candidate_selection_epsilon == 1.0

    def test_epsilon_greedy_rejects_out_of_range_epsilon(self):
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            EvolutionConfig(
                candidate_selection_strategy="epsilon_greedy",
                candidate_selection_epsilon=1.5,
            )
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            EvolutionConfig(
                candidate_selection_strategy="epsilon_greedy",
                candidate_selection_epsilon=-0.1,
            )

    def test_epsilon_rejected_for_other_strategies(self):
        for strategy in ("pareto", "current_best", "top_k_pareto"):
            kwargs: dict[str, object] = {"candidate_selection_strategy": strategy}
            if strategy == "top_k_pareto":
                kwargs["candidate_selection_top_k"] = 3
            with pytest.raises(
                ValidationError, match="candidate_selection_epsilon is only valid"
            ):
                EvolutionConfig(candidate_selection_epsilon=0.1, **kwargs)

    def test_top_k_pareto_requires_top_k(self):
        with pytest.raises(ValidationError, match="candidate_selection_top_k is required"):
            EvolutionConfig(candidate_selection_strategy="top_k_pareto")

    def test_top_k_pareto_rejects_non_positive_top_k(self):
        with pytest.raises(ValidationError, match="must be >= 1"):
            EvolutionConfig(
                candidate_selection_strategy="top_k_pareto",
                candidate_selection_top_k=0,
            )

    def test_top_k_rejected_for_other_strategies(self):
        for strategy in ("pareto", "current_best", "epsilon_greedy"):
            kwargs: dict[str, object] = {"candidate_selection_strategy": strategy}
            if strategy == "epsilon_greedy":
                kwargs["candidate_selection_epsilon"] = 0.1
            with pytest.raises(
                ValidationError, match="candidate_selection_top_k is only valid"
            ):
                EvolutionConfig(candidate_selection_top_k=5, **kwargs)

    def test_top_k_pareto_accepts_valid_top_k(self):
        cfg = EvolutionConfig(
            candidate_selection_strategy="top_k_pareto",
            candidate_selection_top_k=5,
        )
        assert cfg.candidate_selection_top_k == 5

    @pytest.mark.parametrize(
        "strategy,epsilon,top_k,error_match",
        [
            # pareto / current_best own neither knob.
            ("pareto", None, None, None),
            ("pareto", None, 3, "candidate_selection_top_k is only valid"),
            ("pareto", 0.1, None, "candidate_selection_epsilon is only valid"),
            ("pareto", 0.1, 3, "candidate_selection_epsilon is only valid"),
            ("current_best", None, None, None),
            ("current_best", None, 3, "candidate_selection_top_k is only valid"),
            ("current_best", 0.1, None, "candidate_selection_epsilon is only valid"),
            ("current_best", 0.1, 3, "candidate_selection_epsilon is only valid"),
            # epsilon_greedy owns epsilon only.
            ("epsilon_greedy", None, None, "candidate_selection_epsilon is required"),
            ("epsilon_greedy", None, 3, "candidate_selection_epsilon is required"),
            ("epsilon_greedy", 0.1, None, None),
            ("epsilon_greedy", 0.1, 3, "candidate_selection_top_k is only valid"),
            # top_k_pareto owns top_k only.
            ("top_k_pareto", None, None, "candidate_selection_top_k is required"),
            ("top_k_pareto", None, 3, None),
            ("top_k_pareto", 0.1, None, "candidate_selection_epsilon is only valid"),
            ("top_k_pareto", 0.1, 3, "candidate_selection_epsilon is only valid"),
        ],
    )
    def test_strategy_epsilon_top_k_matrix(self, strategy, epsilon, top_k, error_match):
        """Full strategy x epsilon-presence x top_k-presence matrix,
        including the cases where both knobs are set."""
        kwargs: dict[str, object] = {"candidate_selection_strategy": strategy}
        if epsilon is not None:
            kwargs["candidate_selection_epsilon"] = epsilon
        if top_k is not None:
            kwargs["candidate_selection_top_k"] = top_k
        if error_match is None:
            cfg = EvolutionConfig(**kwargs)
            assert cfg.candidate_selection_strategy == strategy
        else:
            with pytest.raises(ValidationError, match=error_match):
                EvolutionConfig(**kwargs)
