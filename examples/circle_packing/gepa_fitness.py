"""optimize_anything's circle-packing fitness function, ported to HELIX.

Source: gepa-ai/gepa at commit 133d185a, ``examples/circle_packing/`` (the code
of optimize_anything's published circle-packing run) and
``src/gepa/utils/code_execution.py`` (its subprocess executor).  Everything
that decides a score is GEPA's and is kept as GEPA wrote it:

* the candidate has two components, ``code`` and ``refiner_prompt``;
* ``code`` defines ``main(timeout, current_best_solution)``, runs in a fresh
  subprocess with a 600-second limit and receives the best packing found so
  far in the run;
* ``validate_packing`` checks it with ``atol=1e-6``, and a valid packing
  scores its raw radius sum, anything else 0;
* a GPT-5.1 refiner, driven by ``refiner_prompt``, rewrites the code, the
  rewrite runs the same way, and the candidate scores the better of the two;
* the ``StateTracker`` caches code and ``(code, refiner_prompt)`` results and
  keeps the run's best packing.

Only the plumbing differs.  HELIX starts the evaluator once per request, so
the ``StateTracker`` is saved to disk after every change instead of living in
one process.  The candidate subprocess hands back a JSON description of its
return value instead of a pickle, so a sandboxed evaluator never unpickles
candidate output (``run_main.py`` builds it).
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

NUM_CIRCLES = 26
LLM_MODEL = "openai/gpt-5.1"
TIMEOUT = 600  # 10 minutes, same as OpenEvolve

RUN_MAIN = Path(__file__).resolve().with_name("run_main.py")


# =============================================================================
# STATE TRACKER (main.py), saved to disk between evaluator processes
# =============================================================================


class StateTracker:
    """
    Manager for the state of the optimization process.
    Tracks the best solution found so far, the number of metric calls, and the cache.
    """

    def __init__(self, log_dir: str = None, max_metric_calls: int = 200):
        self.max_metric_calls = max_metric_calls
        self.metric_calls = 0
        self.cache = {}
        self.best_solution = None
        self.best_score = 0
        self.best_artifact = None
        self.logs = []
        self.log_dir = log_dir

    @classmethod
    @contextmanager
    def open(cls, log_dir: Path) -> Iterator["StateTracker"]:
        """Load the run's tracker under an exclusive lock and save it on every change."""
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "state_tracker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = log_dir / "state_tracker.pkl"
            if path.exists():
                with path.open("rb") as f:
                    tracker = pickle.load(f)
            else:
                tracker = cls(log_dir=str(log_dir))
            yield tracker

    def _persist(self) -> None:
        path = Path(self.log_dir) / "state_tracker.pkl"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self, f)
        os.replace(tmp, path)

    def _key_to_str(self, key_items: tuple) -> str:
        # Convert numpy arrays to lists, leave other items as-is
        serialized = [
            item.tolist() if isinstance(item, np.ndarray) else item
            for item in key_items
        ]
        return json.dumps(serialized)

    def get(self, key: tuple) -> tuple[float, Any, dict] | None:
        key_str = self._key_to_str(key)
        if key_str in self.cache:
            # cache hit
            return self.cache[key_str]
        return None

    def set(
        self,
        key: tuple,
        value: tuple[float, Any, dict],
        score: Optional[float] = None,
        solution: Optional[Any] = None,
        artifact: Optional[Any] = None,
    ) -> None:
        """
        Key is a dictionary of the inputs to the evaluation function.
            For example, for a code, the key is (code, best_solution)
            For a refinement prompt, the key is (prompt, code, best_solution)
        Value is a tuple of (score, output, side_info), same as the output of the evaluation function for the key.
        Score is the score of the candidate. Provide this if you want to track the best score found so far.
        Solution is the solution found by the candidate. Provide this if you want to track the best solution found so far.
        """
        key_str = self._key_to_str(key)
        self.cache[key_str] = value
        self.metric_calls += 1

        if score is not None and solution is not None:
            self._update_best_solution(solution, score, artifact)

        self.log_state()

        if self.metric_calls >= self.max_metric_calls:
            print("Max metric calls reached!", file=sys.stderr)

    def get_best_solution(self) -> tuple[float, Any]:
        return self.best_score, self.best_solution

    def _update_best_solution(
        self,
        solution,
        score,
        artifact,
    ):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_solution = solution
            self.best_artifact = artifact
            print(f"New best solution found: {score:.4f}", file=sys.stderr)

    def log_state(self) -> None:
        log = {
            "metric_calls": self.metric_calls,
            "best_score": self.best_score,
            "best_solution": json.dumps(self.best_solution.tolist()) if self.best_solution is not None else None,
        }
        if self.best_artifact is not None:
            for key, value in self.best_artifact.items():
                if isinstance(value, np.ndarray):
                    log[f"best_artifact_{key}"] = json.dumps(value.tolist())
                else:
                    log[f"best_artifact_{key}"] = value
        self.logs.append(log)
        self.save_logs()

    def save_logs(self) -> None:
        if self.log_dir is None:
            return
        self._persist()
        with open(os.path.join(self.log_dir, "state_tracker_logs.json"), "w") as f:
            json.dump(self.logs, f, indent=2)


def state_dir() -> Path:
    """Where the run's tracker lives: beside HELIX's worktrees, outside every candidate."""
    configured = os.environ.get("HELIX_CP_STATE_DIR")
    if configured:
        return Path(configured)
    cwd = Path.cwd()
    if cwd.parent.name == "worktrees":
        return cwd.parent.parent / "gepa_state_tracker"
    return cwd / ".gepa_state_tracker"


# =============================================================================
# CODE EXECUTION (utils.py, with code_execution.py's subprocess mode)
# =============================================================================


@dataclass
class RunResult:
    """What ``gepa.utils.code_execution.execute_code`` reports for one subprocess run."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    traceback: str = ""
    returned: Any = None  # run_main.py's JSON description of main()'s return value


Runner = Callable[[str, int, Any], RunResult]


def run_subprocess(code: str, timeout: int, current_best_solution: Any) -> RunResult:
    """Run ``main(timeout, current_best_solution)`` in a fresh Python subprocess."""
    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        (work_dir / "candidate.py").write_text(code)
        best = None if current_best_solution is None else np.asarray(current_best_solution).tolist()
        (work_dir / "best.json").write_text(json.dumps(best))
        out_path = work_dir / "result.json"
        process = subprocess.Popen(
            [sys.executable, str(RUN_MAIN), "candidate.py", "best.json", "result.json", str(timeout)],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # GEPA kills the process and its children on a timeout.
            os.killpg(process.pid, 9)
            process.wait()
            return RunResult(success=False, error=f"Timeout: execution exceeded {timeout} seconds")
        return read_run_output(
            out_path,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )


def read_run_output(out_path: Path, stdout: str, stderr: str) -> RunResult:
    """Turn ``run_main.py``'s output file into a ``RunResult``, as GEPA reads its results pickle."""
    if not out_path.exists():
        return RunResult(
            success=False, stdout=stdout, stderr=stderr,
            error="Script crashed before completing", traceback=stderr,
        )
    try:
        output = json.loads(out_path.read_text())
    except (OSError, ValueError, UnicodeDecodeError) as e:
        return RunResult(
            success=False, stdout=stdout, stderr=stderr,
            error=f"Failed to load results: {e}", traceback=stderr,
        )
    return RunResult(
        success=bool(output.get("success", False)),
        stdout=stdout,
        stderr=stderr,
        error=output.get("error", ""),
        traceback=output.get("traceback", ""),
        returned=output.get("return"),
    )


def execute_code(
    code: str,
    timeout: int,
    current_best_solution: Any = None,
    num_circles: int = 26,
    run: Runner = run_subprocess,
) -> dict:
    """
    Execute standalone code candidate in isolated subprocess with validation.

    The code must define a main(timeout, current_best_solution) function that returns
    a dict with 'circles' and 'all_scores' keys.

    Returns:
        dict with:
            - success: bool (True only if execution AND validation pass)
            - result: dict with 'circles', 'all_scores', 'validation_details' if successful
            - error: str if failed
            - traceback: str if failed
            - execution_time: float
            - stdout: str (prints from code)
            - stderr: str (warnings/logs)
    """
    start_time = time.time()

    result = run(code, timeout, current_best_solution)

    execution_time = time.time() - start_time

    # Handle execution failure
    if not result.success:
        return {
            "success": False,
            "error": result.error or "Execution failed",
            "traceback": result.traceback,
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    # Get main() return value
    main_result = result.returned or {"type": "NoneType"}

    # Validate result has required keys
    def _error(msg: str) -> dict:
        return {"success": False, "error": msg, "traceback": "",
                "execution_time": execution_time, "stdout": result.stdout, "stderr": result.stderr}

    if main_result.get("type") != "dict":
        return _error(f"main() must return a dict, got {main_result.get('type')}")
    if not main_result.get("has_circles"):
        return _error("main() return dict must contain 'circles' key")
    if not main_result.get("has_all_scores"):
        return _error("main() return dict must contain 'all_scores' key")
    main_result = {
        "circles": main_result.get("circles"),
        # GEPA's compute_multiple_metrics fails on scores that are not numbers;
        # None reproduces that failure.
        "all_scores": main_result.get("all_scores"),
    }

    # Validate the packing
    try:
        if result.returned.get("circles_error") is not None:
            raise ValueError(result.returned["circles_error"])
        circles = np.array(main_result["circles"])
        main_result["circles"] = circles
        is_valid, validation_details = validate_packing(num_circles, circles)
        main_result["validation_details"] = validation_details
    except Exception as e:
        import traceback as tb
        return {
            "success": False,
            "error": str(e),
            "traceback": tb.format_exc(),
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    if is_valid:
        return {
            "success": True,
            "result": main_result,
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    else:
        # Build error message from validation details
        d = validation_details
        errors = [msg for (cond, msg) in [
            (d["shape_errors"], f"Shape: {d['shape_errors']}"),
            (d["boundary_violations"], f"{len(d['boundary_violations'])} boundary violations"),
            (d["overlaps"], f"{len(d['overlaps'])} overlaps"),
            (d["negative_radii"], f"{len(d['negative_radii'])} negative radii"),
        ] if cond]

        return {
            "success": False,
            "error": "Validation failed: " + "; ".join(errors),
            "result": main_result,  # Still include result for debugging
            "validation_details": validation_details,
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


# =============================================================================
# VALIDATION (utils.py)
# =============================================================================


def validate_packing(
    n: int, circles: np.ndarray, atol: float = 1e-6
) -> Tuple[bool, dict[str, Any]]:
    """
    Validate that circles don't overlap and are inside the unit square.

    Args:
        n: Expected number of circles
        circles: np.array of shape (n, 3) with (x, y, r) for each circle
        atol: Absolute tolerance for numerical comparisons

    Returns:
        Tuple of (is_valid: bool, validation_details: dict)
    """
    validation_details = {
        "expected_circles": n,
        "actual_circles": circles.shape[0],
        "boundary_violations": [],
        "overlaps": [],
        "nan_detected": False,
        "negative_radii": [],
        "shape_errors": [],
    }

    # Check shape
    if circles.shape != (n, 3):
        validation_details["shape_errors"].append(
            f"Circles shape incorrect. Expected ({n}, 3), got {circles.shape}"
        )
        return False, validation_details

    # Check for NaN values
    if np.isnan(circles).any():
        validation_details["nan_detected"] = True
        validation_details["shape_errors"].append("NaN values detected in circles")
        return False, validation_details

    centers = circles[:, :2]
    radii = circles[:, 2]

    # Check for negative radii
    for i in range(n):
        if radii[i] < 0:
            validation_details["negative_radii"].append(
                f"Circle {i} has negative radius {radii[i]:.6f}"
            )

    if validation_details["negative_radii"]:
        return False, validation_details

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol:
            validation_details["boundary_violations"].append(
                f"Circle {i} at ({x:.6f}, {y:.6f}) with r={r:.6f} outside unit square"
            )

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                validation_details["overlaps"].append(
                    f"Circles {i} and {j} overlap: dist={dist:.6f}, r_sum={radii[i] + radii[j]:.6f}"
                )

    # Compute statistics
    validation_details["min_radius"] = float(np.min(radii))
    validation_details["max_radius"] = float(np.max(radii))
    validation_details["avg_radius"] = float(np.mean(radii))
    validation_details["sum_radii"] = float(np.sum(radii))

    # Valid if no violations
    is_valid = (
        len(validation_details["boundary_violations"]) == 0
        and len(validation_details["overlaps"]) == 0
        and len(validation_details["shape_errors"]) == 0
        and len(validation_details["negative_radii"]) == 0
        and not validation_details["nan_detected"]
    )

    return is_valid, validation_details


# =============================================================================
# REFINER (llms.py)
# =============================================================================


Refiner = Callable[..., str]


def make_refiner(**lm_kwargs: Any) -> Refiner:
    """GEPA's refiner: ``dspy.Predict(RefinerSignature)`` on GPT-5.1.

    ``lm_kwargs`` reach ``dspy.LM`` (``api_key``; ``api_base`` for a proxy).
    """
    import dspy

    class RefinerSignature(dspy.Signature):
        """Refine the code based on its evaluation results by fixing the errors and improving the performance."""

        refiner_prompt = dspy.InputField(desc="Instructions for how to refine the code")
        code_to_improve = dspy.InputField(desc="Code to improve")
        code_results = dspy.InputField(
            desc="Evaluation results of the code to improve by fixing the errors and improving the performance"
        )
        refined_code = dspy.OutputField(
            desc="Next iteration of improved code based on the evaluation results"
        )

    refiner_predictor = dspy.Predict(RefinerSignature)
    refiner_lm = dspy.LM(
        LLM_MODEL,
        temperature=1.0,
        max_tokens=32000,
        cache=True,
        **lm_kwargs,
    )

    def refine(**inputs: Any) -> str:
        with dspy.context(lm=refiner_lm):
            return refiner_predictor(**inputs).refined_code

    return refine


# =============================================================================
# FITNESS FUNCTION (main.py)
# =============================================================================


def compute_multiple_metrics(
    global_best_score: float, all_scores: list[float]
) -> dict[str, float]:
    candidate_best_score = max(all_scores)
    alpha_fixed = 0.1
    ema_fixed = all_scores[0]
    for s in all_scores[1:]:
        ema_fixed = alpha_fixed * s + (1 - alpha_fixed) * ema_fixed
    alpha_adaptive = 2.0 / (len(all_scores) + 1)
    ema_adaptive = all_scores[0]
    for s in all_scores[1:]:
        ema_adaptive = alpha_adaptive * s + (1 - alpha_adaptive) * ema_adaptive

    return {
        "max_score": max(all_scores),
        "mean_score": sum(all_scores) / len(all_scores),
        "ema_score_fixed": ema_fixed,
        "ema_score_adaptive": ema_adaptive,
        "score_improvement_from_previous_best": candidate_best_score
        - global_best_score,
    }


def refine_code(
    code: str,
    code_score: float,
    code_side_info: dict,
    refiner_prompt: str,
    refiner: Refiner,
    timeout: int,
    state_tracker: StateTracker,
    execute: Callable[..., dict] = execute_code,
):
    assert refiner is not None
    assert refiner_prompt is not None

    # Check cache first
    global_best_score, global_best_solution = state_tracker.get_best_solution()
    # cache_key = (code, refiner_prompt, global_best_solution)
    cache_key = (code, refiner_prompt)
    cache_results = state_tracker.get(cache_key)
    if cache_results is not None:
        return cache_results

    try:
        refined_code = (
            refiner(
                refiner_prompt=refiner_prompt,
                code_to_improve=code,
                code_results=code_side_info,
            ).strip()
            .replace("```python", "")
            .replace("```", "")
        )

        res = execute(refined_code, timeout, global_best_solution)
        refined_circles = None

        if res["success"]:
            refined_circles = res["result"]["circles"]
            multiple_metrics = compute_multiple_metrics(
                global_best_score, res["result"]["all_scores"]
            )
            refiner_score = res["result"]["validation_details"]["sum_radii"]
            refiner_improvement_rate = (
                (refiner_score - code_score) / code_score
                if code_score > 0
                else refiner_score
            )

            refined_side_info = {
                "scores": {
                    **multiple_metrics,
                    "refiner_improvement_rate": refiner_improvement_rate,
                },
                "Initial code": code,
                "Refined code": refined_code,
                "Global best circles at the time of evaluation": global_best_solution,
                "Circles": refined_circles,
                "Stdout": res.get("stdout", ""),
            }
        else:
            refiner_score = 0.0
            refined_side_info = {
                "scores": {
                    "sum_radii": 0.0,
                    "max_score": 0.0,
                    "refiner_improvement_rate": 0.0,
                },
                "Initial code": code,
                "Refined code": refined_code,
                "Error": res.get("error", "Unknown error"),
                "Traceback": res.get("traceback", ""),
                "Validation Details": res.get("result", {}).get(
                    "validation_details", {}
                ),
                "Stdout": res.get("stdout", ""),
            }

        # Set cache
        state_tracker.set(
            cache_key,
            (refiner_score, refined_code, refined_side_info),
            score=refiner_score,
            solution=refined_circles,
            artifact={
                "refined_code": refined_code,
                "refiner_prompt": refiner_prompt,
                "arg_current_best_solution": global_best_solution,
                "validation details": res.get("validation_details"),
            },
        )
        return refiner_score, refined_code, refined_side_info

    except Exception as e:
        raise Exception(f"Refinement failed: {e}")


def create_fitness_function(
    state_tracker: StateTracker,
    refiner: Refiner,
    timeout: int = TIMEOUT,
    execute: Callable[..., dict] = execute_code,
):
    """
    Create fitness function that evaluates code with optional refinement.
    """

    def fitness_fn(
        candidate: dict[str, str], **kwargs
    ) -> tuple[float, Any, dict]:
        """
        Evaluate code candidate on batch of problems with optional refinement.
        """
        code_candidate = candidate["code"]

        # Code candidate evaluation
        global_best_score, global_best_solution = state_tracker.get_best_solution()
        cache_key = code_candidate
        code_candidate_cache = state_tracker.get(cache_key)

        if code_candidate_cache is not None:
            code_score, code_side_info = code_candidate_cache
        else:
            code_result = execute(code_candidate, timeout, global_best_solution)
            circles = None

            if code_result["success"]:
                circles = code_result["result"]["circles"]
                all_scores = code_result["result"]["all_scores"]
                code_score = code_result["result"]["validation_details"]["sum_radii"]
                code_side_info = {
                    "scores": compute_multiple_metrics(global_best_score, all_scores),
                    "Code": code_candidate,
                    "Circles": circles,
                    "Global best circles at the time of evaluation": global_best_solution,
                    "Stdout": code_result["stdout"],
                }
            else:
                code_score = 0.0
                code_side_info = {
                    "scores": {"sum_radii": 0.0},
                    "Code": code_candidate,
                    "Error": code_result["error"],
                    "Traceback": code_result.get("traceback", ""),
                    "Stdout": code_result["stdout"],
                    "Validation Details": code_result.get("validation_details"),
                }

            # Cache after computing values
            state_tracker.set(
                cache_key,
                (code_score, code_side_info),
                score=code_score,
                solution=circles,
                artifact={
                    "code": code_candidate,
                    "arg_current_best_solution": global_best_solution,
                    "validation details": code_result.get("validation_details"),
                },
            )

        # Refiner prompt evaluation
        refiner_prompt_candidate = candidate["refiner_prompt"]
        global_best_score, global_best_solution = state_tracker.get_best_solution()

        # Refine code for this problem
        (
            refiner_score,
            refiner_code,
            refiner_side_info,
        ) = refine_code(
            code=code_candidate,
            code_score=code_score,
            code_side_info=code_side_info,
            refiner_prompt=refiner_prompt_candidate,
            refiner=refiner,
            timeout=timeout,
            state_tracker=state_tracker,
            execute=execute,
        )

        if refiner_score > code_score:
            best_score = refiner_score
            best_code = refiner_code
            best_circles = refiner_side_info.get("Circles", None)
        else:
            best_score = code_score
            best_code = code_candidate
            best_circles = code_side_info.get("Circles", None)

        if best_circles is not None:
            best_circles = best_circles.tolist()

        output = {
            "best_score": best_score,
            "best_code": best_code,
            "best_circles": best_circles,
            "code_candidate": code_candidate,
            "code_score": code_score,
            "refiner_prompt": refiner_prompt_candidate,
            "refiner_code": refiner_code,
            "refiner_score": refiner_score,
        }

        side_info = {
            "scores": {
                "best_score_from_code_and_refiner": max(code_score, refiner_score),
                "initial_code": code_score,
                "refiner_prompt": refiner_score,
            },
            "Input": {
                "Timeout (s)": timeout,
            },
            "code_specific_info": code_side_info,
            "refiner_prompt_specific_info": refiner_side_info,
        }

        return (best_score, output, side_info)

    return fitness_fn


# =============================================================================
# HELIX output
# =============================================================================


def json_safe(value: Any) -> Any:
    """Side info as strict JSON: arrays become lists, non-finite floats strings."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value
