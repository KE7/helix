"""Unit tests for helix.config — loading, validation, and defaults.

Evaluator command examples below refer to an ``evaluate.py`` script that emits
the required ``HELIX_RESULT=`` line; these tests exercise config handling only.
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from helix.config import (
    AgentConfig,
    DatasetConfig,
    EvaluatorConfig,
    EvolutionConfig,
    HelixConfig,
    WorktreeConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "helix.toml"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# Valid TOML loads correctly
# ---------------------------------------------------------------------------

class TestLoadConfig:
    @pytest.mark.parametrize(
        "config_path",
        sorted((Path(__file__).parents[2] / "examples").glob("*/helix.toml")),
        ids=lambda path: path.parent.name,
    )
    def test_shipped_example_config_loads(self, config_path: Path) -> None:
        """Every shipped example must remain loadable as its schema evolves."""
        load_config(config_path)

    def test_minimal_valid_toml(self, tmp_path):
        """A minimal config with only required fields should load with defaults."""
        toml = write_toml(tmp_path, """
            objective = "Maximise test coverage"
            seed = "."

            [evaluator]
            command = "python evaluate.py"
        """)

        cfg = load_config(toml)

        assert cfg.objective == "Maximise test coverage"
        assert cfg.seed == "."
        assert cfg.evaluator.command == "python evaluate.py"
        assert cfg.evaluator.include_stdout is True
        assert cfg.evaluator.include_stderr is True
        assert cfg.evaluator.extra_commands == []

    def test_full_config_toml(self, tmp_path):
        """All fields specified in TOML should be parsed correctly."""
        toml = write_toml(tmp_path, """
            objective = "Pass all benchmarks"
            seed = "/repo"

            [evaluator]
            command = "python evaluate.py"
            include_stdout = false
            include_stderr = false
            extra_commands = ["make lint", "make typecheck"]

            [evolution]
            max_generations = 20
            perfect_score_threshold = 0.95
            max_evaluations = 500
            merge_enabled = false
            max_merge_invocations = 2

            [agent]
            model = "claude-opus-4-5-20250514"
            background = "You are a coding expert."

            [worktree]
            base_dir = "/tmp/worktrees"
            cleanup_dominated = false
        """)

        cfg = load_config(toml)

        assert cfg.objective == "Pass all benchmarks"
        assert cfg.seed == "/repo"

        assert cfg.evaluator.command == "python evaluate.py"
        assert cfg.evaluator.include_stdout is False
        assert cfg.evaluator.include_stderr is False
        assert cfg.evaluator.extra_commands == ["make lint", "make typecheck"]

        assert cfg.evolution.max_generations == 20
        assert cfg.evolution.merge_enabled is False
        assert cfg.evolution.max_merge_invocations == 2
        assert cfg.agent.model == "claude-opus-4-5-20250514"
        assert cfg.agent.background == "You are a coding expert."

        assert cfg.worktree.base_dir == "/tmp/worktrees"
        assert cfg.worktree.cleanup_dominated is False

    def test_agent_section_loads(self, tmp_path):
        toml = write_toml(tmp_path, """
            objective = "Use a different backend"

            [evaluator]
            command = "python evaluate.py"

            [agent]
            backend = "codex"
            model = "gpt-5"
        """)

        cfg = load_config(toml)

        assert cfg.agent.backend == "codex"
        assert cfg.agent.model == "gpt-5"

    def test_defaults_applied_for_nested_sections(self, tmp_path):
        """Omitted nested sections should use their default values."""
        toml = write_toml(tmp_path, """
            objective = "Improve score"

            [evaluator]
            command = "python evaluate.py"
        """)

        cfg = load_config(toml)

        # DatasetConfig defaults (empty after Fix 8 — dead fields removed)
        assert cfg.dataset is not None

        # EvolutionConfig defaults
        assert cfg.evolution.max_generations == 10
        assert cfg.evolution.perfect_score_threshold is None
        assert cfg.evolution.max_evaluations == -1
        assert cfg.evolution.merge_enabled is False  # GEPA parity: off by default
        assert cfg.evolution.max_merge_invocations == 5
        assert cfg.evolution.retain_rejected_worktrees is False

        # AgentConfig defaults
        assert cfg.agent.backend == "claude"
        assert cfg.agent.model is None
        assert cfg.agent.background is None

        # WorktreeConfig defaults
        assert cfg.worktree.base_dir == ".helix/worktrees"
        assert cfg.worktree.cleanup_dominated is False  # deprecated: GEPA append-only


# ---------------------------------------------------------------------------
# Missing required fields raise clear errors
# ---------------------------------------------------------------------------

class TestMissingRequiredFields:
    def test_missing_objective_raises(self, tmp_path):
        """objective is required — omitting it should exit with friendly error."""
        toml = write_toml(tmp_path, """
            [evaluator]
            command = "python evaluate.py"
        """)
        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)
        assert exc_info.value.code == 1

    def test_missing_evaluator_command_raises(self, tmp_path):
        """evaluator.command is required."""
        toml = write_toml(tmp_path, """
            objective = "do something"

            [evaluator]
        """)
        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)
        assert exc_info.value.code == 1

    def test_missing_evaluator_section_raises(self, tmp_path):
        """evaluator section itself is required."""
        toml = write_toml(tmp_path, """
            objective = "do something"
        """)
        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)
        assert exc_info.value.code == 1

    def test_removed_parser_key_raises(self, tmp_path):
        """The removed parser setting is rejected as an unknown key."""
        removed_key = "score_" + "parser"
        toml = write_toml(tmp_path, """
            objective = "X"

            [evaluator]
            command = "python evaluate.py"
            command = "python evaluate.py"
        """)
        toml.write_text(toml.read_text() + f'{removed_key} = "helix_result"\n')
        with pytest.raises(SystemExit):
            load_config(toml)


# ---------------------------------------------------------------------------
# Direct model construction
# ---------------------------------------------------------------------------

class TestDirectModelConstruction:
    def test_helix_config_requires_objective_and_evaluator(self):
        with pytest.raises(ValidationError):
            HelixConfig()  # missing objective and evaluator

    def test_evaluator_config_requires_command(self):
        with pytest.raises(ValidationError):
            EvaluatorConfig()  # missing command

    def test_dataset_config_empty(self):
        """DatasetConfig can be constructed with no arguments (all fields removed)."""
        cfg = DatasetConfig()
        assert cfg is not None

    def test_evolution_config_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.max_generations == 10

    def test_merge_subsample_size_default_is_5(self) -> None:
        """Pin default to 5 per GEPA
        merge.py::MergeProposer.select_eval_subsample_for_merged_program.

        Changing this default without intent should be a conscious act — the
        constant is algorithmically load-bearing (stratification math uses
        `ceil(5/3) = 2` per bucket across 3 buckets).  An ablation study
        would vary the config field, not the default.
        """
        cfg = EvolutionConfig()
        assert cfg.merge_subsample_size == 5, (
            "Default must match GEPA's num_subsample_ids=5 constant "
            "(gepa/src/gepa/proposer/merge.py::MergeProposer."
            "select_eval_subsample_for_merged_program).  If you are intentionally "
            "changing this default, update this test AND the comment in "
            "config.py that cites the GEPA line."
        )

    def test_worktree_config_defaults(self):
        cfg = WorktreeConfig()
        assert cfg.base_dir == ".helix/worktrees"
        assert cfg.cleanup_dominated is False  # deprecated: GEPA append-only


# ---------------------------------------------------------------------------
# agent.effort validation (per-backend allowlist + ignored-backend warning)
# ---------------------------------------------------------------------------


class TestAgentEffortValidation:
    """Surface obvious mismatches between ``agent.effort`` and ``agent.backend``.

    HELIX forwards ``agent.effort`` to a backend-native CLI flag/config in
    ``helix.mutator`` (``claude --effort``, ``codex -c
    model_reasoning_effort=...``, ``opencode --variant``).  The other
    backends silently ignore the field — without a warning the setting looks
    like it's taking effect when it isn't.
    """

    def _base_kwargs(self):
        return {
            "objective": "test",
            "evaluator": EvaluatorConfig(command="echo 1"),
        }

    def test_effort_unset_is_silent(self, recwarn):
        HelixConfig(
            **self._base_kwargs(),
            agent=AgentConfig(backend="claude"),
        )
        assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []

    def test_effort_accepts_valid_value_on_codex(self, recwarn):
        for value in ("minimal", "low", "medium", "high", "xhigh"):
            HelixConfig(
                **self._base_kwargs(),
                agent=AgentConfig(backend="codex", effort=value),
            )
        assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []

    def test_effort_warns_on_ignoring_backend_gemini(self, recwarn):
        HelixConfig(
            **self._base_kwargs(),
            agent=AgentConfig(backend="gemini", effort="high"),
        )
        warnings = [str(w.message) for w in recwarn.list if w.category is UserWarning]
        assert any("does not propagate" in w for w in warnings), warnings

    def test_effort_warns_on_ignoring_backend_cursor(self, recwarn):
        HelixConfig(
            **self._base_kwargs(),
            agent=AgentConfig(backend="cursor", effort="medium"),
        )
        warnings = [str(w.message) for w in recwarn.list if w.category is UserWarning]
        assert any("does not propagate" in w for w in warnings), warnings

    def test_effort_accepts_valid_value_on_claude(self, recwarn):
        for value in ("low", "medium", "high"):
            HelixConfig(
                **self._base_kwargs(),
                agent=AgentConfig(backend="claude", effort=value),
            )
        assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []

    def test_effort_warns_on_unknown_value_for_claude(self, recwarn):
        HelixConfig(
            **self._base_kwargs(),
            agent=AgentConfig(backend="claude", effort="extreme"),
        )
        warnings = [str(w.message) for w in recwarn.list if w.category is UserWarning]
        assert any("not a recognized value" in w and "extreme" in w for w in warnings), warnings

    def test_effort_warns_on_unknown_value_for_codex(self, recwarn):
        HelixConfig(
            **self._base_kwargs(),
            agent=AgentConfig(backend="codex", effort="extreme"),
        )
        warnings = [str(w.message) for w in recwarn.list if w.category is UserWarning]
        assert any("not a recognized value" in w and "extreme" in w for w in warnings), warnings

    def test_effort_does_not_warn_on_unrestricted_backend(self, recwarn):
        """opencode accepts arbitrary --variant strings; HELIX must not warn."""
        HelixConfig(
            **self._base_kwargs(),
            agent=AgentConfig(backend="opencode", effort="custom-coder-plus"),
        )
        assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []

    def test_effort_warning_is_warning_not_error(self):
        """The validation must emit a warning (not raise) on unsupported combos."""
        # ``pytest.warns`` makes both invariants explicit at once: HelixConfig
        # must construct successfully *and* a UserWarning must be emitted.
        with pytest.warns(UserWarning, match="does not propagate"):
            HelixConfig(
                **self._base_kwargs(),
                agent=AgentConfig(backend="gemini", effort="high"),
            )

    def test_every_backend_has_effort_metadata(self):
        """Every entry in ``BACKENDS`` must be classified by the validator.

        Guards against drift: when a new backend is added to ``BACKENDS``
        without anyone updating ``EFFORT_AWARE_BACKENDS`` /
        ``EFFORT_VALID_VALUES``, this test surfaces it immediately rather
        than letting the validator silently emit the wrong message (or no
        message) for the new backend.
        """
        from helix.backends import (
            BACKENDS,
            EFFORT_AWARE_BACKENDS,
            EFFORT_VALID_VALUES,
        )

        # Every effort-aware backend must appear in EFFORT_VALID_VALUES so
        # the "is this a known value?" branch can fire.
        for backend in EFFORT_AWARE_BACKENDS:
            assert backend in EFFORT_VALID_VALUES, (
                f"{backend!r} is in EFFORT_AWARE_BACKENDS but missing from "
                "EFFORT_VALID_VALUES; the validator can't decide whether "
                "values are typos."
            )

        # Every BACKENDS entry must produce a predictable signal under a
        # sentinel ``effort`` value:
        #   - ignoring backends            -> "does not propagate" warning
        #   - aware + restricted allowlist -> "not a recognized value" warning
        #   - aware + unrestricted (None)  -> silent (any string is allowed)
        # Using a sentinel guaranteed not to be in any restricted allowlist
        # forces aware-restricted backends to warn too.
        sentinel = "__helix_effort_sentinel__"
        for backend in BACKENDS:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                HelixConfig(
                    **self._base_kwargs(),
                    agent=AgentConfig(backend=backend, effort=sentinel),
                )
            messages = [
                str(w.message) for w in captured if w.category is UserWarning
            ]
            if backend not in EFFORT_AWARE_BACKENDS:
                assert any(
                    "does not propagate" in m for m in messages
                ), f"{backend!r} produced no 'does not propagate' warning: {messages}"
            elif EFFORT_VALID_VALUES.get(backend) is not None:
                assert any(
                    "not a recognized value" in m for m in messages
                ), f"{backend!r} produced no 'not a recognized value' warning: {messages}"
            else:
                # Unrestricted aware backend (e.g. opencode): silent is correct.
                assert messages == [], (
                    f"{backend!r} is unrestricted but emitted warnings: {messages}"
                )
