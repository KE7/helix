"""Tests for examples/circle_packing, optimize_anything's circle-packing experiment.

The fitness function in ``gepa_fitness.py`` is a port of gepa-ai/gepa@133d185a
``examples/circle_packing``; these tests pin the seed files to GEPA's bytes
and exercise the executor, the StateTracker and the fitness function with a
fake refiner (no network).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

np = pytest.importorskip("numpy")

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "circle_packing"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# One module object for every test file: the StateTracker is pickled by name.
gf = sys.modules.get("gepa_fitness") or _load("gepa_fitness", EXAMPLE / "gepa_fitness.py")

SEED = (EXAMPLE / "solve.py").read_text()
SEED_PROMPT = (EXAMPLE / "refiner_prompt.txt").read_text()
SEED_SCORE = 0.9797642169962063

R = 1.0 / 16.0
GRID_ROWS = [[R + (i % 8) * 2 * R, R + (i // 8) * 2 * R, R] for i in range(26)]


def returning(body: str) -> str:
    """Candidate code whose main() runs ``body``."""
    indented = "\n".join("    " + line for line in body.splitlines())
    return f"import numpy as np\n\ndef main(timeout, current_best_solution):\n{indented}\n"


GRID = returning(f"c = np.array({GRID_ROWS})\nreturn {{'circles': c, 'all_scores': [float(c[:, 2].sum())]}}")


def run(code: str, best: Any = None, timeout: int = 60) -> dict:
    return gf.execute_code(code, timeout, best)


# ---------------------------------------------------------------------------
# The seed is GEPA's
# ---------------------------------------------------------------------------


def test_seed_files_are_gepas_bytes() -> None:
    # SEED_CODE and SEED_REFINEMENT_PROMPT of gepa@133d185a examples/circle_packing.
    assert hashlib.sha256(SEED.encode()).hexdigest() == (
        "f8c1a8826cbc5bbbf40a5881c6cbaa71ba2db34fe5f4de8c7123d94ac60ee7de"
    )
    assert hashlib.sha256(SEED_PROMPT.encode()).hexdigest() == (
        "bd6b7fbb3fa0ee0e37090f9db61ca64ebb6ab8b61b1ab62656a5e43f31f797d4"
    )


def test_seed_scores_its_radius_sum_and_warm_starts_from_the_best() -> None:
    result = run(SEED)
    assert result["success"]
    assert result["result"]["validation_details"]["sum_radii"] == pytest.approx(SEED_SCORE)
    warm = run(SEED, best=np.array(GRID_ROWS))
    assert warm["result"]["validation_details"]["sum_radii"] == pytest.approx(26 * R)


def test_config_is_gepas_run() -> None:
    config = tomllib.loads((EXAMPLE / "helix.toml").read_text())
    evolution = config["evolution"]
    assert evolution["max_evaluations"] == 200
    assert evolution["cache_evaluation"] is True
    assert evolution["frontier_type"] == "objective"
    assert evolution["minibatch_size"] == 1
    assert evolution["merge_enabled"] is False
    assert "perfect_score_threshold" not in evolution
    assert config["project"]["objective"] == (
        "Optimize circle packing code and refiner prompt to maximize sum of "
        "circle radii within a unit square for N=26 circles."
    )
    assert config["agent"]["background"].startswith(
        "Make BREAKTHROUGH improvements by trying fundamentally different approaches."
    )


def test_agent_is_sandboxed_without_web_access() -> None:
    from helix.config import load_config

    config = load_config(EXAMPLE / "helix.toml")
    assert config.agent.backend == "claude" and config.agent.model == "claude-haiku-4-5"
    assert config.agent.web_access is False
    # Agents run on up.sh's --internal network; the evaluator runs on the host.
    assert config.sandbox.enabled and not config.sandbox.evaluator
    assert config.sandbox.network == "helix-circle-agent"
    assert config.sandbox.image == "helix-circle/agent:1"
    assert config.sandbox.auth_volume == "helix-circle-agent-home"
    # The agent talks to the gateway with a placeholder; the gateway holds the login.
    assert config.env["ANTHROPIC_BASE_URL"] == "http://anthropic-gateway:8080"
    assert config.env["CLAUDE_CODE_OAUTH_TOKEN"].startswith("placeholder")
    up = (EXAMPLE.parent / "circle_packing_sandbox" / "up.sh").read_text()
    for name in ("helix-circle-agent", "helix-circle/agent:1", "helix-circle-agent-home",
                 "anthropic-gateway", "claude-haiku-4-5", "--internal"):
        assert name in up, name


@pytest.mark.parametrize("name", ["README.md", "helix.toml", "solve.py", "refiner_prompt.txt"])
def test_candidate_tree_states_no_result(name: str) -> None:
    text = (EXAMPLE / name).read_text().lower()
    for token in ("2.63", "2.635", "alphaevolve", "packomania", "friedman", "best known"):
        assert token not in text, f"{name} mentions {token!r}"


def test_reference_is_a_static_fixture_not_an_optimizer() -> None:
    reference = EXAMPLE.parents[1] / "docs/experiments/circle-packing/reference_known_best.py"
    result = run(reference.read_text())
    assert result["success"] and result["result"]["validation_details"]["sum_radii"] > 2.6
    module = _load("cp_reference_fixture", reference)
    assert not hasattr(module, "TIME_BUDGET") and not hasattr(module, "minimize")


# ---------------------------------------------------------------------------
# validate_packing and execute_code, as in GEPA's utils.py
# ---------------------------------------------------------------------------


class TestValidatePacking:
    def test_valid_grid(self) -> None:
        valid, details = gf.validate_packing(26, np.array(GRID_ROWS))
        assert valid and details["sum_radii"] == pytest.approx(26 * R)

    def test_overlap_within_tolerance_scores_the_raw_sum(self) -> None:
        rows = [list(r) for r in GRID_ROWS]
        rows[9][2] += 5e-7
        valid, details = gf.validate_packing(26, np.array(rows))
        assert valid and details["sum_radii"] == pytest.approx(26 * R + 5e-7)

    @pytest.mark.parametrize(
        "change, problem",
        [
            (lambda rows: rows[9].__setitem__(2, R + 5e-6), "overlaps"),
            (lambda rows: rows[0].__setitem__(0, -0.01), "boundary_violations"),
            (lambda rows: rows[3].__setitem__(2, -0.1), "negative_radii"),
            (lambda rows: rows[3].__setitem__(0, float("nan")), "nan_detected"),
        ],
    )
    def test_violations(self, change: Any, problem: str) -> None:
        rows = [list(r) for r in GRID_ROWS]
        change(rows)
        valid, details = gf.validate_packing(26, np.array(rows))
        assert not valid and details[problem]

    def test_wrong_shape(self) -> None:
        valid, details = gf.validate_packing(26, np.array(GRID_ROWS[:25]))
        assert not valid and details["shape_errors"]


class TestExecuteCode:
    def test_valid_packing(self) -> None:
        result = run(GRID)
        assert result["success"]
        assert result["result"]["all_scores"] == [pytest.approx(26 * R)]
        assert isinstance(result["result"]["circles"], np.ndarray)

    def test_invalid_packing_fails_validation(self) -> None:
        result = run(returning("return {'circles': np.full((26, 3), 0.4), 'all_scores': [0.0]}"))
        assert not result["success"]
        assert result["error"].startswith("Validation failed: ")
        assert "overlaps" in result["error"]

    def test_exception_is_reported_with_its_traceback(self) -> None:
        result = run(returning("raise RuntimeError('boom')"))
        assert not result["success"]
        assert result["error"] == "boom"
        assert "RuntimeError" in result["traceback"]

    def test_stdout_is_captured(self) -> None:
        result = run(returning(f"print('hello')\nc = np.array({GRID_ROWS})\nreturn {{'circles': c, 'all_scores': [1.0]}}"))
        assert result["stdout"] == "hello\n"

    @pytest.mark.parametrize(
        "body, error",
        [
            ("return [1, 2]", "main() must return a dict, got list"),
            ("return {'all_scores': [1.0]}", "main() return dict must contain 'circles' key"),
            ("return {'circles': []}", "main() return dict must contain 'all_scores' key"),
        ],
    )
    def test_malformed_return_values(self, body: str, error: str) -> None:
        result = run(returning(body))
        assert not result["success"] and result["error"] == error

    def test_missing_main(self) -> None:
        result = run("x = 1\n")
        assert result["error"] == "main() must return a dict, got NoneType"

    def test_process_exit_leaves_no_result(self) -> None:
        result = run(returning("import os\nos._exit(0)"))
        assert result["error"] == "Script crashed before completing"

    def test_timeout(self) -> None:
        result = run(returning("import time\ntime.sleep(60)"), timeout=1)
        assert result["error"] == "Timeout: execution exceeded 1 seconds"

    def test_timeout_is_passed_to_main(self) -> None:
        code = returning(
            f"assert timeout == 7\nc = np.array({GRID_ROWS})\nreturn {{'circles': c, 'all_scores': [1.0]}}"
        )
        assert run(code, timeout=7)["success"]

    def test_non_numeric_scores_fail_like_gepas_metrics(self) -> None:
        result = run(returning(f"return {{'circles': np.array({GRID_ROWS}), 'all_scores': ['x']}}"))
        assert result["success"]
        with pytest.raises(TypeError):
            gf.compute_multiple_metrics(0.0, result["result"]["all_scores"])


# ---------------------------------------------------------------------------
# StateTracker and the fitness function, as in GEPA's main.py
# ---------------------------------------------------------------------------


class CountingExecute:
    def __init__(self) -> None:
        self.codes: list[str] = []
        self.bests: list[Any] = []

    def __call__(self, code: str, timeout: int, best: Any) -> dict:
        self.codes.append(code)
        self.bests.append(best)
        return gf.execute_code(code, timeout, best)


class FakeRefiner:
    def __init__(self, refined: str) -> None:
        self.refined = refined
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **inputs: Any) -> str:
        self.calls.append(inputs)
        return "```python\n" + self.refined + "\n```"


def fenced(code: str) -> str:
    """What GEPA's refine_code makes of FakeRefiner's fenced answer."""
    return ("```python\n" + code + "\n```").strip().replace("```python", "").replace("```", "")


def fitness(tmp_path: Path, refined: str, execute: CountingExecute | None = None):
    tracker = gf.StateTracker(log_dir=str(tmp_path))
    refiner = FakeRefiner(refined)
    execute = execute or CountingExecute()
    fn = gf.create_fitness_function(state_tracker=tracker, refiner=refiner, execute=execute)
    return fn, tracker, refiner, execute


def test_candidate_scores_the_better_of_code_and_refined_code(tmp_path: Path) -> None:
    fn, tracker, refiner, execute = fitness(tmp_path, refined=GRID)
    score, output, side_info = fn({"code": SEED, "refiner_prompt": SEED_PROMPT})
    assert score == pytest.approx(26 * R)
    assert side_info["scores"] == {
        "best_score_from_code_and_refiner": pytest.approx(26 * R),
        "initial_code": pytest.approx(SEED_SCORE),
        "refiner_prompt": pytest.approx(26 * R),
    }
    assert output["best_code"] == fenced(GRID) == "\n" + GRID + "\n"
    assert side_info["Input"] == {"Timeout (s)": 600}
    # The refiner sees the prompt, the code and the code's results.
    assert refiner.calls[0]["refiner_prompt"] == SEED_PROMPT
    assert refiner.calls[0]["code_to_improve"] == SEED
    assert refiner.calls[0]["code_results"]["Code"] == SEED
    # Two executions (code, refined code), the run's best kept.
    assert tracker.metric_calls == 2
    assert tracker.best_score == pytest.approx(26 * R)
    # The refined code ran warm-started from the seed's packing.
    assert execute.bests[0] is None
    np.testing.assert_allclose(execute.bests[1], run(SEED)["result"]["circles"])


def test_repeated_candidates_run_nothing_again(tmp_path: Path) -> None:
    fn, tracker, refiner, execute = fitness(tmp_path, refined=GRID)
    first = fn({"code": SEED, "refiner_prompt": SEED_PROMPT})
    second = fn({"code": SEED, "refiner_prompt": SEED_PROMPT})
    assert first[0] == second[0]
    assert len(execute.codes) == 2 and len(refiner.calls) == 1 and tracker.metric_calls == 2
    # A new refiner prompt refines the cached code again; the code does not rerun.
    fn({"code": SEED, "refiner_prompt": "another prompt"})
    assert execute.codes.count(SEED) == 1 and len(refiner.calls) == 2


def test_later_candidates_warm_start_from_the_runs_best(tmp_path: Path) -> None:
    fn, tracker, refiner, execute = fitness(tmp_path, refined=SEED)
    fn({"code": GRID, "refiner_prompt": SEED_PROMPT})
    fn({"code": SEED, "refiner_prompt": SEED_PROMPT})
    np.testing.assert_allclose(execute.bests[-1], np.array(GRID_ROWS))


def test_failing_code_scores_zero_and_shows_the_error(tmp_path: Path) -> None:
    broken = returning("raise ValueError('bad')")
    fn, tracker, refiner, execute = fitness(tmp_path, refined=broken)
    score, _, side_info = fn({"code": broken, "refiner_prompt": SEED_PROMPT})
    assert score == 0.0
    assert side_info["code_specific_info"]["Error"] == "bad"
    assert side_info["refiner_prompt_specific_info"]["Error"] == "bad"
    assert tracker.best_solution is None


def test_refiner_failure_raises_like_gepa(tmp_path: Path) -> None:
    tracker = gf.StateTracker(log_dir=str(tmp_path))

    def refiner(**inputs: Any) -> str:
        raise ConnectionError("no network")

    fn = gf.create_fitness_function(state_tracker=tracker, refiner=refiner)
    with pytest.raises(Exception, match="Refinement failed: no network"):
        fn({"code": SEED, "refiner_prompt": SEED_PROMPT})
    # The code execution before the refiner is still recorded.
    assert tracker.metric_calls == 1


def test_tracker_survives_between_evaluator_processes(tmp_path: Path) -> None:
    refiner = FakeRefiner(GRID)
    with gf.StateTracker.open(tmp_path) as tracker:
        fn = gf.create_fitness_function(state_tracker=tracker, refiner=refiner)
        fn({"code": SEED, "refiner_prompt": SEED_PROMPT})
    logs = json.loads((tmp_path / "state_tracker_logs.json").read_text())
    assert [entry["metric_calls"] for entry in logs] == [1, 2]
    assert logs[-1]["best_score"] == pytest.approx(26 * R)
    assert logs[-1]["best_artifact_refined_code"] == fenced(GRID)
    with gf.StateTracker.open(tmp_path) as reloaded:
        assert reloaded.metric_calls == 2
        np.testing.assert_allclose(reloaded.best_solution, np.array(GRID_ROWS))
        fn = gf.create_fitness_function(state_tracker=reloaded, refiner=refiner)
        fn({"code": SEED, "refiner_prompt": SEED_PROMPT})
        assert reloaded.metric_calls == 2 and len(refiner.calls) == 1


def test_state_dir_sits_beside_helix_worktrees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree = tmp_path / ".helix" / "worktrees" / "g1-s1"
    worktree.mkdir(parents=True)
    monkeypatch.chdir(worktree)
    monkeypatch.delenv("HELIX_CP_STATE_DIR", raising=False)
    assert gf.state_dir() == tmp_path / ".helix" / "gepa_state_tracker"
    monkeypatch.setenv("HELIX_CP_STATE_DIR", str(tmp_path / "elsewhere"))
    assert gf.state_dir() == tmp_path / "elsewhere"


def test_evaluate_prints_a_helix_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    from helix.parsers import helix_result

    for name in ("solve.py", "refiner_prompt.txt"):
        (tmp_path / name).write_text((EXAMPLE / name).read_text())
    (tmp_path / "helix_batch.json").write_text('["0"]')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    monkeypatch.setenv("HELIX_CP_STATE_DIR", str(tmp_path / "tracker"))
    monkeypatch.setattr(gf, "make_refiner", lambda **kwargs: FakeRefiner(GRID))
    evaluate = _load("cp_evaluate", EXAMPLE / "evaluate.py")
    evaluate.main()
    stdout = capsys.readouterr().out
    _scores, instance_scores, _side_info, objective_scores = helix_result.parse(0, stdout, "", tmp_path)
    assert instance_scores["0"] == pytest.approx(26 * R)
    assert objective_scores[0]["initial_code"] == pytest.approx(SEED_SCORE)
