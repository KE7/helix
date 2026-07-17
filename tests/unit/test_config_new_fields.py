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
        # (optimize_anything.py:485).
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
# Unified P*N proposal scheduler: mutations_per_parent (N), proposal_selection,
# proposal_top_k, and the effective P*N validation (Step 1 of the parallel
# proposals plan).
# ---------------------------------------------------------------------------


class TestProposalSchedulerFields:
    def test_defaults_are_k_by_1(self):
        """Omitting the new fields is exactly the K-by-1 case: P defaults to 1,
        N defaults to 1, and selection defaults to all_improvements."""
        cfg = EvolutionConfig()
        assert cfg.num_parallel_proposals == 1  # P
        assert cfg.mutations_per_parent == 1  # N
        assert cfg.proposal_selection == "all_improvements"
        assert cfg.proposal_top_k is None

    def test_omitted_n_matches_explicit_n_one(self):
        """An existing K-slot run (N omitted) is identical to an explicit
        N=1 K-by-1 configuration for every existing K value."""
        for k in (1, 2, 4, 8):
            omitted = EvolutionConfig(num_parallel_proposals=k)
            explicit = EvolutionConfig(
                num_parallel_proposals=k, mutations_per_parent=1
            )
            assert omitted.num_parallel_proposals == k
            assert omitted.mutations_per_parent == 1
            assert (
                omitted.model_dump() == explicit.model_dump()
            ), f"omitted vs explicit N=1 diverged for K={k}"

    def test_override_pn_fields(self):
        cfg = EvolutionConfig(
            num_parallel_proposals=2,
            mutations_per_parent=3,
            proposal_selection="top_k",
            proposal_top_k=4,
        )
        assert cfg.num_parallel_proposals == 2
        assert cfg.mutations_per_parent == 3
        assert cfg.proposal_selection == "top_k"
        assert cfg.proposal_top_k == 4

    # -- mutations_per_parent (N) bounds ----------------------------------

    def test_mutations_per_parent_default_is_one(self):
        assert EvolutionConfig().mutations_per_parent == 1

    @pytest.mark.parametrize("n", [0, -1, -5])
    def test_mutations_per_parent_rejects_non_positive(self, n):
        with pytest.raises(ValidationError):
            EvolutionConfig(mutations_per_parent=n)

    def test_mutations_per_parent_accepts_large(self):
        assert EvolutionConfig(mutations_per_parent=16).mutations_per_parent == 16

    # -- num_parallel_proposals (P) bounds --------------------------------

    @pytest.mark.parametrize("p", [0, -1, -3])
    def test_num_parallel_proposals_rejects_non_positive(self, p):
        with pytest.raises(ValidationError):
            EvolutionConfig(num_parallel_proposals=p)

    def test_num_parallel_proposals_accepts_one(self):
        assert EvolutionConfig(num_parallel_proposals=1).num_parallel_proposals == 1

    # -- max_workers bounds ------------------------------------------------

    @pytest.mark.parametrize("w", [0, -1, -32])
    def test_max_workers_rejects_non_positive(self, w):
        with pytest.raises(ValidationError):
            EvolutionConfig(max_workers=w)

    def test_max_workers_accepts_one(self):
        assert EvolutionConfig(max_workers=1).max_workers == 1

    # -- proposal_selection ------------------------------------------------

    def test_proposal_selection_default(self):
        assert EvolutionConfig().proposal_selection == "all_improvements"

    @pytest.mark.parametrize(
        "strategy", ["all_improvements", "best_improvement", "top_k"]
    )
    def test_proposal_selection_accepts_supported(self, strategy):
        kwargs = {"proposal_selection": strategy}
        if strategy == "top_k":
            kwargs["proposal_top_k"] = 1
        cfg = EvolutionConfig(**kwargs)
        assert cfg.proposal_selection == strategy

    def test_proposal_selection_rejects_unknown(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(proposal_selection="pareto")  # type: ignore[arg-type]

    # -- proposal_top_k: required only / valid only for top_k -------------

    def test_top_k_required_for_top_k_selection(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(proposal_selection="top_k")

    def test_top_k_rejected_for_all_improvements(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(
                proposal_selection="all_improvements", proposal_top_k=1
            )

    def test_top_k_rejected_for_best_improvement(self):
        with pytest.raises(ValidationError):
            EvolutionConfig(
                proposal_selection="best_improvement", proposal_top_k=1
            )

    def test_top_k_none_ok_for_non_top_k_modes(self):
        assert (
            EvolutionConfig(proposal_selection="best_improvement").proposal_top_k
            is None
        )

    # -- proposal_top_k: within 1..P*N ------------------------------------

    def test_top_k_within_effective_pn(self):
        # P=2, N=3 => P*N = 6; top_k in 1..6 accepted.
        for k in range(1, 7):
            cfg = EvolutionConfig(
                num_parallel_proposals=2,
                mutations_per_parent=3,
                proposal_selection="top_k",
                proposal_top_k=k,
            )
            assert cfg.proposal_top_k == k

    def test_top_k_rejects_zero_and_negative(self):
        for k in (0, -1):
            with pytest.raises(ValidationError):
                EvolutionConfig(
                    num_parallel_proposals=2,
                    mutations_per_parent=3,
                    proposal_selection="top_k",
                    proposal_top_k=k,
                )

    def test_top_k_rejects_above_pn(self):
        # P=2, N=3 => P*N = 6; top_k=7 exceeds the effective width.
        with pytest.raises(ValidationError):
            EvolutionConfig(
                num_parallel_proposals=2,
                mutations_per_parent=3,
                proposal_selection="top_k",
                proposal_top_k=7,
            )

    def test_top_k_equals_pn_upper_bound_ok(self):
        cfg = EvolutionConfig(
            num_parallel_proposals=4,
            mutations_per_parent=1,
            proposal_selection="top_k",
            proposal_top_k=4,
        )
        assert cfg.proposal_top_k == 4

    def test_top_k_uses_resolved_auto_p_for_bound(self):
        # "auto" with max_workers=9, minibatch_size=3 => P=3; N=2 => P*N=6.
        cfg = EvolutionConfig(
            num_parallel_proposals="auto",
            max_workers=9,
            minibatch_size=3,
            mutations_per_parent=2,
            proposal_selection="top_k",
            proposal_top_k=6,
        )
        assert cfg.num_parallel_proposals == 3
        assert cfg.proposal_top_k == 6
        # 7 would exceed the resolved P*N of 6.
        with pytest.raises(ValidationError):
            EvolutionConfig(
                num_parallel_proposals="auto",
                max_workers=9,
                minibatch_size=3,
                mutations_per_parent=2,
                proposal_selection="top_k",
                proposal_top_k=7,
            )

    # -- TOML round trips --------------------------------------------------

    def test_toml_loads_pn_fields(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "P by N"

            [evaluator]
            command = "pytest"

            [evolution]
            num_parallel_proposals = 2
            mutations_per_parent = 2
            proposal_selection = "top_k"
            proposal_top_k = 3
        """)
        )
        cfg = load_config(toml)
        assert cfg.evolution.num_parallel_proposals == 2
        assert cfg.evolution.mutations_per_parent == 2
        assert cfg.evolution.proposal_selection == "top_k"
        assert cfg.evolution.proposal_top_k == 3

    def test_toml_defaults_when_omitted(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Defaults"

            [evaluator]
            command = "pytest"
        """)
        )
        cfg = load_config(toml)
        assert cfg.evolution.num_parallel_proposals == 1
        assert cfg.evolution.mutations_per_parent == 1
        assert cfg.evolution.proposal_selection == "all_improvements"
        assert cfg.evolution.proposal_top_k is None

    def test_toml_invalid_top_k_combo_rejected_at_load(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Bad top_k"

            [evaluator]
            command = "pytest"

            [evolution]
            num_parallel_proposals = 1
            mutations_per_parent = 1
            proposal_selection = "top_k"
            proposal_top_k = 5
        """)
        )
        with pytest.raises(SystemExit):
            load_config(toml)

    def test_model_dump_roundtrip_pn_fields(self):
        cfg = EvolutionConfig(
            num_parallel_proposals=2,
            mutations_per_parent=3,
            proposal_selection="top_k",
            proposal_top_k=5,
        )
        restored = EvolutionConfig.model_validate(cfg.model_dump())
        assert restored.num_parallel_proposals == 2
        assert restored.mutations_per_parent == 3
        assert restored.proposal_selection == "top_k"
        assert restored.proposal_top_k == 5


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
            command = "pytest"

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
            command = "pytest"
        """)
        )
        cfg = load_config(toml)
        assert cfg.evolution.frontier_type == "hybrid"

    def test_toml_invalid_literal_rejected_at_load(self, tmp_path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Test"

            [evaluator]
            command = "pytest"

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
            command = "pytest -q"
            score_parser = "exitcode"
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
                evaluator={"command": "pytest"},
                sandbox=SandboxConfig(enabled=True, evaluator=True),
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
            command = "pytest"
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
            command = "pytest"

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
            command = "pytest"

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
            evaluator={"command": "pytest"},
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
