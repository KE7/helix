"""Protected HELIX evaluator for the pinned FormulaCode smoke subset."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any, Iterator

from official_score import helix_result_pairs, score_measurements
from workloads import WORKLOADS, measure_workload


ROOT = Path.cwd().resolve()
PRIVATE_DIR = ROOT / ".formulacode"
_CREDENTIAL_SUFFIXES = (
    "_API_KEY",
    "_AUTH_TOKEN",
    "_ACCESS_TOKEN",
    "_SECRET_ACCESS_KEY",
    "_SECRET_KEY",
    "_CLIENT_SECRET",
)
_CREDENTIAL_NAMES = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
}


class EvaluationTimeout(TimeoutError):
    """Raised by the local wall-clock guard."""


def purge_mutation_credentials() -> None:
    """Remove provider credentials before candidate code can run in-process.

    This changes only the evaluator subprocess environment.  HELIX's parent
    process and isolated mutation containers retain their configured auth.
    """

    for name in tuple(os.environ):
        if name in _CREDENTIAL_NAMES or name.endswith(_CREDENTIAL_SUFFIXES):
            os.environ.pop(name, None)


@contextmanager
def candidate_environment() -> Iterator[None]:
    """Expose no mutation auth or host HOME to in-process candidate code."""

    purge_mutation_credentials()
    previous_home = os.environ.get("HOME")
    with tempfile.TemporaryDirectory(prefix="candidate-home-", dir=PRIVATE_DIR) as home:
        os.environ["HOME"] = home
        try:
            yield
        finally:
            if previous_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = previous_home


@contextmanager
def wall_clock_timeout(seconds: int) -> Iterator[None]:
    """Bound correctness and timing work without leaking exception text."""

    def handler(_signum: int, _frame: object) -> None:
        raise EvaluationTimeout("evaluation phase exceeded its pinned timeout")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def load_requested_ids(root: Path, split: str) -> list[str]:
    """Read HELIX's positional handoff and reject malformed/unknown ids."""

    if split not in WORKLOADS:
        raise ValueError("HELIX_SPLIT must be train or val")
    batch_path = root / "helix_batch.json"
    data = json.loads(batch_path.read_text())
    if (
        not isinstance(data, list)
        or not data
        or not all(isinstance(item, str) for item in data)
    ):
        raise ValueError("helix_batch.json must be a non-empty list[str]")
    unknown = [item for item in data if item not in WORKLOADS[split]]
    if unknown:
        raise ValueError("helix_batch.json contains an unknown instance id")
    return data


def run_correctness(config: dict[str, Any]) -> tuple[bool, str | None]:
    """Run the pinned upstream NetworkX correctness subset."""

    command = [sys.executable, "-m", "pytest", "-q", *config["correctness_tests"]]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=int(config["correctness_timeout_seconds"]),
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)},
        )
    except subprocess.TimeoutExpired:
        return False, "correctness_timeout"
    if result.returncode != 0:
        return False, "correctness_failure"
    return True, None


def _measure_agent(
    requested_ids: list[str], split: str, config: dict[str, Any]
) -> dict[str, list[float]]:
    with candidate_environment():
        sys.path.insert(0, str(ROOT))
        try:
            nx = importlib.import_module("networkx")
        finally:
            sys.path.pop(0)

        module_file = getattr(nx, "__file__", None)
        if module_file is None:
            raise RuntimeError("candidate NetworkX module has no source path")
        module_path = Path(module_file).resolve()
        if ROOT not in module_path.parents:
            raise RuntimeError("candidate NetworkX was not imported from its worktree")

        measured: dict[str, list[float]] = {}
        for instance_id in dict.fromkeys(requested_ids):
            workload = WORKLOADS[split][instance_id]
            measured[workload.benchmark_name] = measure_workload(
                nx,
                workload,
                warmups=int(config["warmups"]),
                repeats=int(config["repeats"]),
                loops=int(config["loops"]),
            )
        return measured


def evaluate() -> list[list[Any]]:
    purge_mutation_credentials()
    config = json.loads((PRIVATE_DIR / "measurement.json").read_text())
    baselines = json.loads((PRIVATE_DIR / "baselines.json").read_text())
    split = os.environ.get("HELIX_SPLIT", "val")
    requested_ids = load_requested_ids(ROOT, split)
    workloads = WORKLOADS[split]
    ordered_names = [workloads[item].benchmark_name for item in requested_ids]

    correctness_passed, failure_kind = run_correctness(config)
    agent_samples: dict[str, list[float]] = {}
    if correctness_passed:
        lock_path = Path(baselines["measurement_lock"])
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with wall_clock_timeout(int(config["measurement_timeout_seconds"])):
                with lock_path.open("a+") as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    agent_samples = _measure_agent(requested_ids, split, config)
        except (EvaluationTimeout, OSError, RuntimeError, ValueError):
            correctness_passed = False
            failure_kind = "measurement_failure"
            agent_samples = {}

    metrics = score_measurements(
        ordered_workloads=ordered_names,
        nop_samples=baselines["nop"][split],
        oracle_samples=baselines["oracle"][split],
        agent_samples=agent_samples,
        correctness_passed=correctness_passed,
    )
    return helix_result_pairs(
        metrics,
        requested_ids,
        {key: item.benchmark_name for key, item in workloads.items()},
        failure_kind=failure_kind,
    )


def main() -> int:
    try:
        pairs = evaluate()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        # Contract failures should be visible by type but never echo paths,
        # environment values, command output, or exception messages.
        print(
            f"FormulaCode evaluator contract error: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 2
    print("HELIX_RESULT=" + json.dumps(pairs, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
