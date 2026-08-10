"""Mode-aware ``evolution.minibatch_size`` default.

The default is 1 in single-task / no-example mode (no training split
configured) and 3 in multi-task / generalization mode (a training split is
configured via ``seedless.train_path`` or ``dataset.train_size``).

The one user-visible consequence in single-task mode runs through
``num_parallel_proposals``.  In single-task mode no train loader exists, so
``run_evolution`` never builds a batch sampler and ``minibatch_size`` is never
read for sampling — it does *not* change the number of metric calls there.  It
is read as the divisor when ``num_parallel_proposals="auto"`` resolves to
``max(1, max_workers // minibatch_size)``, and that resolved ``P`` drives the
proposal-slot loop unconditionally.  So the whole observable effect of this
default flows through that one derivation, which is why it is covered both at
the config layer (``TestAutoWidthUsesCorrectedMinibatch``) and end-to-end
against the real evolution loop (``TestAutoWidthReachesTheProposalLoop``).

That also makes *ordering* load-bearing.  ``EvolutionConfig.model_post_init``
resolves ``"auto"`` at ``EvolutionConfig`` construction time, which happens
before ``HelixConfig.model_post_init`` runs.  A mode-aware default applied
after ``HelixConfig`` is built would leave an already-resolved ``"auto"``
derived from the stale divisor — silently correct-looking ``minibatch_size``
with a wrong ``P``.  The auto-width tests fail against any such after-the-fact
implementation.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from helix.config import (
    MULTI_TASK_MINIBATCH_SIZE,
    SINGLE_TASK_MINIBATCH_SIZE,
    EvolutionConfig,
    HelixConfig,
    load_config,
)
from helix.evolution import run_evolution
from tests.unit.test_evolution import (  # type: ignore[import-untyped]
    all_mocks,  # noqa: F401, F811 — re-exported pytest fixture
    make_candidate,
    make_eval_result,
)

BASE: dict[str, object] = {
    "objective": "Maximise the score",
    "evaluator": {"command": "pytest"},
}


def _config(**overrides: object) -> HelixConfig:
    return HelixConfig.model_validate({**BASE, **overrides})


class TestModeAwareDefault:
    def test_single_task_omitted_resolves_to_one(self):
        """No training split + no explicit value -> 1."""
        cfg = _config()
        assert cfg.evolution.minibatch_size == SINGLE_TASK_MINIBATCH_SIZE == 1

    def test_single_task_with_other_evolution_keys_still_resolves_to_one(self):
        """Injection must merge into an existing [evolution] table, not
        replace it."""
        cfg = _config(evolution={"max_generations": 7, "max_workers": 4})
        assert cfg.evolution.minibatch_size == 1
        assert cfg.evolution.max_generations == 7
        assert cfg.evolution.max_workers == 4

    def test_multi_task_via_train_path_omitted_stays_three(self):
        cfg = _config(seedless={"train_path": "/tmp/train.jsonl"})
        assert cfg.evolution.minibatch_size == MULTI_TASK_MINIBATCH_SIZE == 3

    def test_multi_task_via_train_size_omitted_stays_three(self):
        """``dataset.train_size`` is the Architecture A training split: it
        synthesises a range train loader, so it is multi-task mode too."""
        cfg = _config(dataset={"train_size": 10})
        assert cfg.evolution.minibatch_size == 3

    def test_zero_train_size_is_single_task(self):
        """A zero cardinality builds no train loader, so it is not a split."""
        cfg = _config(dataset={"train_size": 0})
        assert cfg.evolution.minibatch_size == 1

    def test_explicit_three_in_single_task_mode_is_preserved(self):
        """A deliberately written value is never silently rewritten."""
        cfg = _config(evolution={"minibatch_size": 3})
        assert cfg.evolution.minibatch_size == 3

    def test_explicit_one_in_multi_task_mode_is_preserved(self):
        cfg = _config(
            dataset={"train_size": 10},
            evolution={"minibatch_size": 1},
        )
        assert cfg.evolution.minibatch_size == 1

    def test_prebuilt_evolution_config_is_left_untouched(self):
        """An ``EvolutionConfig`` instance was built deliberately by the
        caller and already resolved its own ``"auto"``; do not rewrite it."""
        cfg = _config(evolution=EvolutionConfig())
        assert cfg.evolution.minibatch_size == 3

    def test_standalone_evolution_config_default_is_the_multi_task_value(self):
        """``EvolutionConfig`` alone cannot see the mode, so its field default
        stays at the multi-task value; only ``HelixConfig`` knows better."""
        assert EvolutionConfig().minibatch_size == 3


class TestAutoWidthUsesCorrectedMinibatch:
    """The ordering test.

    ``"auto"`` must be derived from the *corrected* minibatch size.  An
    implementation that mutates ``minibatch_size`` on an already-constructed
    ``HelixConfig`` leaves ``num_parallel_proposals`` at ``12 // 3 == 4`` here
    and fails.
    """

    def test_single_task_auto_derives_from_one(self):
        cfg = _config(evolution={"num_parallel_proposals": "auto", "max_workers": 12})
        assert cfg.evolution.minibatch_size == 1
        assert cfg.evolution.num_parallel_proposals == 12  # 12 // 1, not 12 // 3

    def test_multi_task_auto_derives_from_three(self):
        cfg = _config(
            seedless={"train_path": "/tmp/train.jsonl"},
            evolution={"num_parallel_proposals": "auto", "max_workers": 12},
        )
        assert cfg.evolution.minibatch_size == 3
        assert cfg.evolution.num_parallel_proposals == 4  # 12 // 3

    def test_explicit_minibatch_in_single_task_still_drives_auto(self):
        cfg = _config(
            evolution={
                "num_parallel_proposals": "auto",
                "max_workers": 12,
                "minibatch_size": 3,
            }
        )
        assert cfg.evolution.num_parallel_proposals == 4  # 12 // 3


class TestLoadedFromToml:
    def test_single_task_toml_gets_one(self, tmp_path: Path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Pack circles"

            [evaluator]
            command = "python3 evaluate.py"

            [evolution]
            max_generations = 30
            num_parallel_proposals = "auto"
            max_workers = 6
        """)
        )
        cfg = load_config(toml)
        assert cfg.evolution.minibatch_size == 1
        assert cfg.evolution.num_parallel_proposals == 6

    def test_multi_task_toml_keeps_three(self, tmp_path: Path):
        toml = tmp_path / "helix.toml"
        toml.write_text(
            textwrap.dedent("""
            objective = "Generalise"

            [evaluator]
            command = "pytest"

            [dataset]
            train_size = 8

            [evolution]
            num_parallel_proposals = "auto"
            max_workers = 6
        """)
        )
        cfg = load_config(toml)
        assert cfg.evolution.minibatch_size == 3
        assert cfg.evolution.num_parallel_proposals == 2

    def test_roundtrip_through_model_dump_is_stable(self):
        """``model_dump`` emits an explicit ``minibatch_size``, so re-validating
        a dumped single-task config must not drift."""
        cfg = _config()
        restored = HelixConfig.model_validate(cfg.model_dump())
        assert restored.evolution.minibatch_size == 1


class TestAutoWidthReachesTheProposalLoop:
    """End-to-end: the corrected divisor really does widen the proposal loop.

    This is the test that demonstrates the change has an observable effect at
    all. It drives the real ``run_evolution`` loop with I/O mocked and counts
    proposal slots via ``mutate`` calls, rather than asserting on the resolved
    config value a second time.
    """

    @staticmethod
    def _run(tmp_path, all_mocks, evolution: dict[str, object]) -> int:  # noqa: F811
        all_mocks["create_seed_worktree"].return_value = make_candidate("g0-s0")
        # Every slot's mutation fails fast: we are counting slots, not
        # acceptances, so the loop stays cheap and deterministic.
        all_mocks["mutate"].return_value = None
        all_mocks["run_evaluator"].side_effect = (
            lambda candidate, config, split=None, instances=None, **kw: (
                make_eval_result(candidate.id, {"i1": 0.5})
            )
        )
        cfg = HelixConfig.model_validate(
            {
                **BASE,
                "evolution": {
                    "max_generations": 1,
                    "max_evaluations": -1,
                    "perfect_score_threshold": None,
                    "num_parallel_proposals": "auto",
                    "max_workers": 12,
                    **evolution,
                },
            }
        )
        run_evolution(cfg, tmp_path, tmp_path / ".helix")
        return int(all_mocks["mutate"].call_count)

    def test_single_task_auto_opens_max_workers_slots(self, tmp_path, all_mocks):  # noqa: F811
        """max_workers=12 // corrected minibatch 1 -> 12 proposal slots."""
        assert self._run(tmp_path, all_mocks, {}) == 12

    def test_explicit_minibatch_three_still_opens_a_third_of_them(
        self, tmp_path, all_mocks
    ):  # noqa: F811
        """The pre-fix behaviour, still reachable by writing the value
        explicitly: 12 // 3 -> 4 proposal slots."""
        assert self._run(tmp_path, all_mocks, {"minibatch_size": 3}) == 4
