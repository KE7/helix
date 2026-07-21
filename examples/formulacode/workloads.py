"""Pinned FormulaCode NetworkX #7971 workload subset and timing runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import subprocess
import sys
import timeit
from typing import Any


TASK_ID = "networkx_networkx_7971"
REPOSITORY = "https://github.com/networkx/networkx.git"
BASE_COMMIT = "a986762f2a1919126df2174644232c92c58be2be"
ORACLE_COMMIT = "3d0bb212f9fa4bac168c3b8c3f512a5f69b7920c"


@dataclass(frozen=True)
class Workload:
    instance_id: str
    benchmark_name: str
    component_kind: str
    nodes: int


# Both functions changed in the official optimization PR.  Training uses a
# smaller public shape; validation keeps a larger held-out size.  Each shape is
# the PR's adversarial "singleton first, complete component last" workload.
WORKLOADS: dict[str, dict[str, Workload]] = {
    "train": {
        "0": Workload(
            "0", "components.connected.single_plus_complete_n240", "connected", 240
        ),
        "1": Workload(
            "1",
            "components.weakly_connected.single_plus_complete_n240",
            "weakly_connected",
            240,
        ),
    },
    "val": {
        "0": Workload(
            "0", "components.connected.single_plus_complete_n420", "connected", 420
        ),
        "1": Workload(
            "1",
            "components.weakly_connected.single_plus_complete_n420",
            "weakly_connected",
            420,
        ),
    },
}


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _load_networkx(repo: Path) -> Any:
    resolved = repo.resolve()
    if not (resolved / "networkx" / "__init__.py").is_file():
        raise ValueError(f"not a NetworkX checkout: {resolved}")
    sys.path.insert(0, str(resolved))
    try:
        module = importlib.import_module("networkx")
    finally:
        sys.path.pop(0)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError("candidate NetworkX module has no source path")
    module_path = Path(module_file).resolve()
    if resolved not in module_path.parents:
        raise RuntimeError(f"NetworkX imported from unexpected path: {module_path}")
    return module


def _build_graph(nx: Any, workload: Workload) -> Any:
    if workload.component_kind == "connected":
        # Node insertion order is the essence of the official PR's adversarial
        # case: visit the singleton first, then encounter the dense component.
        graph = nx.Graph()
        graph.add_node(0)
        graph.update(nx.complete_graph(range(1, workload.nodes)))
        return graph
    if workload.component_kind == "weakly_connected":
        graph = nx.DiGraph()
        graph.add_node(0)
        graph.update(
            nx.complete_graph(range(1, workload.nodes), create_using=nx.DiGraph)
        )
        return graph
    raise ValueError(f"unknown workload kind: {workload.component_kind}")


def measure_workload(
    nx: Any,
    workload: Workload,
    *,
    warmups: int,
    repeats: int,
    loops: int,
) -> list[float]:
    """Measure one official-PR workload and return seconds per invocation."""

    graph = _build_graph(nx, workload)
    if workload.component_kind == "connected":

        def run() -> None:
            list(nx.connected_components(graph))
    else:

        def run() -> None:
            list(nx.weakly_connected_components(graph))

    for _ in range(warmups):
        run()
    timer = timeit.Timer(run)
    return [sample / loops for sample in timer.repeat(repeat=repeats, number=loops)]


def measure_repository(repo: Path, measurement: dict[str, int]) -> dict[str, Any]:
    """Measure every train/validation workload in a fresh Python process."""

    nx = _load_networkx(repo)
    samples: dict[str, dict[str, list[float]]] = {}
    for split, workloads in WORKLOADS.items():
        samples[split] = {}
        for workload in workloads.values():
            samples[split][workload.benchmark_name] = measure_workload(
                nx,
                workload,
                warmups=measurement["warmups"],
                repeats=measurement["repeats"],
                loops=measurement["loops"],
            )
    return {
        "commit": _git_head(repo),
        "python": sys.version.split()[0],
        "workloads": {
            split: {key: asdict(value) for key, value in values.items()}
            for split, values in WORKLOADS.items()
        },
        "samples": samples,
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    args = parser.parse_args()
    measurement = json.loads(args.measurement.read_text())
    print(json.dumps(measure_repository(args.repo, measurement), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
