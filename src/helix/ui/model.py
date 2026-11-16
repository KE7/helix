"""Static view of a HELIX project for ``helix ui``: config, loop, isolation, repro.

Everything here is derived from the loaded :class:`HelixConfig` and a bounded,
read-only look at files the config names.  Nothing about a run is read here;
``helix.ui.run_state`` owns that.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import helix
from helix.config import HelixConfig
from helix.evolution import _config_hash, _load_dataset_ids
from helix.sandbox import sandbox_auth_volume_name

_OBJECTIVE_CHARS = 300
_MAX_MODULE_FILES = 200
_MAX_MODULE_BYTES = 2 * 1024 * 1024
# A dataset file larger than this is not "cheaply knowable": its size is null.
_MAX_DATASET_BYTES = 64 * 1024 * 1024
# ``timed("name")`` and ``timed.record("name", ...)`` with a literal name.
_TIMED_RE = re.compile(r"""\btimed(?:\.record)?\(\s*(["'])(.+?)\1""")
# Docker networks that give a container a route out.  ``none`` has none, and a
# named network is taken to be the internal one the user set up for this.
_OPEN_NETWORKS = frozenset({"bridge", "host"})


def build_static(config: HelixConfig, project_root: Path, config_path: Path) -> dict[str, Any]:
    """Return the ``GET /api/static`` payload for one project."""
    agent_zone = "sandbox" if config.sandbox.enabled else "host"
    evaluator_zone = _evaluator_zone(config)
    sizes = dataset_sizes(config, project_root)
    return {
        "project": {
            "name": project_root.name,
            "root": str(project_root),
            "config_path": str(config_path),
            "helix_version": helix.__version__,
        },
        "config": _config_facts(config),
        "loop": _loop(config, agent_zone, evaluator_zone, sizes),
        "isolation": {
            "agent": agent_zone,
            "evaluator": evaluator_zone,
            "warnings": isolation_warnings(config, sizes["train"]),
        },
        "repro": _repro(config, project_root, config_path, evaluator_zone),
        "evaluator_modules": evaluator_modules(config, project_root),
        "sizes": sizes,
    }


def _evaluator_zone(config: HelixConfig) -> str:
    # executor.py:251-255: evaluator commands leave the host only when
    # sandbox.enabled and sandbox.evaluator are both set, and then they run in
    # runner containers that call [evaluator.sidecar] on a private network
    # agent containers cannot reach (docs/guides/sandboxing.md:8-18).
    if config.sandbox.enabled and config.sandbox.evaluator:
        return "sidecar"
    return "host"


def _auth_volume(config: HelixConfig) -> str | None:
    if not config.sandbox.enabled:
        return None
    try:
        return sandbox_auth_volume_name(config.agent.backend, config.sandbox)
    except ValueError:
        return None


def _config_facts(config: HelixConfig) -> dict[str, Any]:
    evo = config.evolution
    agent = config.agent
    sandbox = config.sandbox
    evaluator = config.evaluator
    return {
        "objective": config.objective[:_OBJECTIVE_CHARS],
        "seedless": config.seedless.enabled,
        "frontier_type": evo.frontier_type,
        "max_generations": evo.max_generations,
        # budget.py:100-101: any cap <= 0 disables the evaluation budget.
        "max_evaluations": evo.max_evaluations if evo.max_evaluations > 0 else None,
        "proposals_per_generation": evo.num_parallel_proposals,
        "mutations_per_parent": evo.mutations_per_parent,
        "minibatch_size": evo.minibatch_size,
        "batch_sampler": evo.batch_sampler,
        "acceptance_criterion": evo.acceptance_criterion,
        "val_stage_size": evo.val_stage_size or None,
        "perfect_score_threshold": evo.perfect_score_threshold,
        # Family of an instance id, as the Frontier strip groups them (run_state._frontier_payload).
        "group_key_separator": evo.group_key_separator,
        "stop_at_score": evo.stop_at_score,
        "merge_enabled": evo.merge_enabled,
        "cache_evaluation": evo.cache_evaluation,
        "sibling_history": evo.sibling_history,
        "max_workers": evo.max_workers,
        "rng_seed": config.rng_seed,
        "agent": {
            "backend": agent.backend,
            "model": agent.model,
            "effort": agent.effort,
            "max_turns": agent.max_turns,
            "web_access": agent.web_access,
        },
        "sandbox": {
            "enabled": sandbox.enabled,
            "evaluator_in_sandbox": sandbox.evaluator,
            "network": sandbox.network,
            "add_host_gateway": sandbox.add_host_gateway,
            "auth_volume": _auth_volume(config),
            "auth_volume_shared": sandbox.enabled and sandbox.auth_volume is None,
            "timeout_seconds": sandbox.timeout_seconds,
            "omit_from_agent": len(sandbox.omit_from_agent),
            "image": sandbox.image,
            "cpus": sandbox.cpus,
            "memory": sandbox.memory,
        },
        "evaluator": {
            "command": evaluator.command,
            "sidecar": evaluator.sidecar is not None,
            "protected_files": len(evaluator.protected_files),
            "extra_commands": len(evaluator.extra_commands),
            "include_stdout": evaluator.include_stdout,
            "include_stderr": evaluator.include_stderr,
        },
        # Names only: values may be secrets.
        "env_names": list(dict.fromkeys([*config.passthrough_env, *config.env])),
    }


def _gate_ids(config: HelixConfig) -> int:
    evo = config.evolution
    # batch_sampler.py:184-189: a stratified minibatch holds
    # num_sampled_groups (default minibatch_size) x num_examples_per_group
    # (default 1) ids; epoch_shuffled holds minibatch_size.
    if evo.batch_sampler == "stratified":
        groups = evo.num_sampled_groups or evo.minibatch_size
        return groups * (evo.num_examples_per_group or 1)
    return evo.minibatch_size


def _loop(
    config: HelixConfig, agent_zone: str, evaluator_zone: str, sizes: dict[str, int | None]
) -> dict[str, Any]:
    nodes = [
        {"id": "frontier", "zone": "host"},
        {"id": "mutation", "zone": agent_zone},
        {"id": "gate", "zone": evaluator_zone},
        {"id": "validation", "zone": evaluator_zone},
    ]
    if config.evolution.merge_enabled:
        # The merge agent runs where mutation agents do (merger.py:346).
        nodes.append({"id": "merge", "zone": agent_zone})
    agent = config.agent
    proposals = config.evolution.num_parallel_proposals * config.evolution.mutations_per_parent
    mutation = f"{proposals} proposal{'s' if proposals != 1 else ''} · " + " ".join(
        part for part in (agent.backend, agent.effort) if part
    )
    # evolution.py:1999: without train ids there is no minibatch gate.
    has_train = sizes["train"] is not None or config.seedless.train_path is not None
    return {
        "nodes": nodes,
        "planned": {
            "mutation": mutation,
            "gate": f"{_gate_ids(config)} ids" if has_train else None,
            "validation": f"{sizes['val']} ids" if sizes["val"] is not None else None,
        },
    }


def isolation_warnings(config: HelixConfig, train_size: int | None) -> list[dict[str, Any]]:
    """Return amber warnings for likely isolation or sampling mistakes in *config*."""
    warnings: list[dict[str, Any]] = []
    sandbox = config.sandbox
    # config.py:794-796: web_access = false disables only the backend's
    # built-in web tools; "the shell can still reach the network; a strict
    # no-web guarantee is the sandbox's job".
    if not config.agent.web_access and (
        not sandbox.enabled or sandbox.network in _OPEN_NETWORKS
    ):
        warnings.append(
            {
                "id": "web-access-open-network",
                "text": "web access off, network open",
                "detail": (
                    "agent.web_access = false turns off only the backend's built-in web "
                    "tools; the agent's shell can still reach the network."
                ),
                "keys": ["agent.web_access", "sandbox.network" if sandbox.enabled else "sandbox.enabled"],
            }
        )
    # sandbox.py:1385-1389: omit_from_agent only filters the sandbox workspace
    # copy, which unsandboxed runs never make.
    if sandbox.omit_from_agent and not sandbox.enabled:
        warnings.append(
            {
                "id": "omit-without-sandbox",
                "text": "omit_from_agent has no effect",
                "detail": (
                    "Without the sandbox the agent works in the worktree itself, so the "
                    "omitted paths stay readable."
                ),
                "keys": ["sandbox.omit_from_agent", "sandbox.enabled"],
            }
        )
    # evolution.py:2045-2061: P*N slots past len(train) // minibatch_size
    # disjoint minibatches make minibatches repeat within an iteration.
    if train_size is not None:
        slots = config.evolution.num_parallel_proposals * config.evolution.mutations_per_parent
        distinct = train_size // max(1, config.evolution.minibatch_size)
        if slots > distinct:
            warnings.append(
                {
                    "id": "minibatch-repeats",
                    "text": "minibatches repeat",
                    "detail": (
                        f"{slots} proposals per generation exceed the {distinct} disjoint "
                        f"minibatches of {config.evolution.minibatch_size} from {train_size} "
                        "train ids."
                    ),
                    "keys": [
                        "evolution.num_parallel_proposals",
                        "evolution.mutations_per_parent",
                        "evolution.minibatch_size",
                    ],
                }
            )
    return warnings


def _repro(
    config: HelixConfig, project_root: Path, config_path: Path, evaluator_zone: str
) -> list[list[str]]:
    try:
        shown_path = config_path.relative_to(project_root).as_posix()
    except ValueError:
        shown_path = str(config_path)
    agent = config.agent
    agent_parts = [agent.backend, agent.model, agent.effort]
    if agent.max_turns is not None:
        agent_parts.append(f"{agent.max_turns} turns")
    if config.sandbox.enabled:
        sandbox_parts = ["on", f"evaluator {'in sidecar' if evaluator_zone == 'sidecar' else 'on host'}"]
        volume = _auth_volume(config)
        if volume is not None:
            sandbox_parts.append(f"auth volume {volume}")
        sandbox_text = " · ".join(sandbox_parts)
    else:
        sandbox_text = "off"
    return [
        ["config", f"{shown_path} · {_config_hash(config)}"],
        ["frontier", config.evolution.frontier_type],
        ["rng_seed", str(config.rng_seed)],
        ["agent", " · ".join(part for part in agent_parts if part)],
        ["evaluator", config.evaluator.command],
        ["sandbox", sandbox_text],
        ["seed", "generated (seedless)" if config.seedless.enabled else "repository"],
        # A15: no .helix file records the version that ran; this is the viewer's.
        ["helix", f"{helix.__version__} (this viewer)"],
    ]


def _count_ids(path: Path | None, project_root: Path) -> int | None:
    """Count dataset ids the way ``helix.evolution.HelixDataLoader`` does, or None."""
    if path is None:
        return None
    # HELIX opens the path relative to its working directory, the project root
    # in the usual `helix evolve` from the project.
    resolved = path if path.is_absolute() else project_root / path
    try:
        if resolved.is_file() and resolved.stat().st_size > _MAX_DATASET_BYTES:
            return None
        count = len(_load_dataset_ids(resolved))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    # evolution.py:823-834: an empty loader counts as no loader.
    return count or None


def dataset_sizes(config: HelixConfig, project_root: Path) -> dict[str, int | None]:
    """Return the train and val id counts HELIX would use, None where unknown."""
    dataset = config.dataset
    seedless = config.seedless
    # evolution.py:1973-1996: train_path wins, then dataset.train_size.
    train = _count_ids(seedless.train_path, project_root)
    if train is None and dataset.train_size:
        train = dataset.train_size
    # evolution.py:1286-1291: dataset.val_size wins, then the val_path loader,
    # which falls back to the train_path loader only when val_path is unset.
    if dataset.val_size:
        val: int | None = dataset.val_size
    elif seedless.val_path is not None:
        val = _count_ids(seedless.val_path, project_root)
    else:
        val = _count_ids(seedless.train_path, project_root)
    return {"train": train, "val": val}


def _split(command: str) -> list[str]:
    try:
        return shlex.split(command, comments=True)
    except ValueError:
        return command.split()


def _inside(token: str, root: Path) -> Path | None:
    """Resolve *token* against *root*; None unless it is a file inside *root*."""
    path = Path(token)
    try:
        resolved = (path if path.is_absolute() else root / path).resolve()
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(root) or not resolved.is_file():
        return None
    return resolved


def _read_bounded(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_MODULE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _shell_python_tokens(text: str) -> list[str]:
    """Return ``.py`` tokens on ``python ... X.py`` and ``exec ... X.py`` lines."""
    tokens: list[str] = []
    for line in text.splitlines():
        words = _split(line)
        if not words:
            continue
        if words[0] == "exec" or any(Path(w).name.startswith("python") for w in words):
            tokens.extend(w for w in words if w.endswith(".py"))
    return tokens


def evaluator_modules(config: HelixConfig, project_root: Path) -> list[dict[str, str]]:
    """Return literal ``helix.timed`` names found in the evaluator's Python files."""
    root = project_root.resolve()
    files: list[Path] = []

    def add(token: str) -> None:
        path = _inside(token, root)
        if path is not None and path not in files and len(files) < _MAX_MODULE_FILES:
            files.append(path)

    evaluator = config.evaluator
    for command in [evaluator.command, *evaluator.extra_commands]:
        for token in _split(command):
            if token.endswith(".py"):
                add(token)
            elif token.endswith(".sh"):
                script = _inside(token, root)
                text = _read_bounded(script) if script is not None else None
                for py in _shell_python_tokens(text or ""):
                    add(py)
    for protected in evaluator.protected_files:
        if protected.endswith(".py"):
            add(protected)

    modules: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in files:
        text = _read_bounded(path)
        if text is None:
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _TIMED_RE.finditer(line):
                name = match.group(2)
                if name not in seen:
                    seen.add(name)
                    modules.append({"name": name, "source": f"{rel}:{lineno}"})
    return modules
