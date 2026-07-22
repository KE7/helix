"""Daemon-backed proof of parallel proposal sandbox isolation."""

from __future__ import annotations

import concurrent.futures
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import helix.sandbox as sandbox_module
from helix.config import SandboxConfig
from helix.sandbox import run_sandboxed_command


pytestmark = [
    pytest.mark.docker_integration,
    # PRE-EXISTING HAZARD, surfaced by the session-wide safety guard rather
    # than introduced by it.
    #
    # ``run_sandboxed_command(scope="agent", agent_backend="opencode")``
    # mounts the SHARED credential volume ``helix-auth-opencode`` at ``:rw``
    # — and because ``docker run -v`` silently CREATES a missing named
    # volume, running this test also provisions that volume on any host where
    # it does not yet exist. Neither is acceptable for a test: a container
    # holding a shared credential volume ``:rw`` can trigger an OAuth refresh,
    # and a successful refresh ROTATES the stored token for every lane.
    #
    # These tests assert workspace isolation and container cleanup, none of
    # which needs real credentials. They must be rewritten to run against a
    # DISPOSABLE volume with SYNTHETIC credentials before they can be enabled
    # again. Skipping rather than deleting keeps the gap visible.
    pytest.mark.skip(
        reason=(
            "mounts the SHARED helix-auth-opencode volume :rw (and would "
            "create it if absent); rewrite against a disposable volume with "
            "synthetic credentials first"
        )
    ),
]

_FIXTURE_IMAGE = os.environ.get("HELIX_DOCKER_TEST_IMAGE", "helix-runner-base:latest")


def _docker(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _require_local_docker_fixture() -> None:
    try:
        daemon = _docker("info")
        image = _docker("image", "inspect", _FIXTURE_IMAGE)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")
    if daemon.returncode != 0:
        pytest.skip(f"Docker daemon unavailable: {daemon.stderr.strip()}")
    if image.returncode != 0:
        pytest.skip(f"local fixture image {_FIXTURE_IMAGE!r} is not installed")


def _wait_until_running(container_names: list[str]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = _docker(
            "inspect",
            "--format",
            "{{.State.Running}}",
            *container_names,
        )
        if status.returncode == 0 and status.stdout.splitlines() == ["true", "true"]:
            return
        time.sleep(0.05)
    raise AssertionError(
        "parallel sandbox containers did not become simultaneously active: "
        + ", ".join(container_names)
    )


def test_parallel_sandboxes_overlap_isolate_state_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_local_docker_fixture()
    candidates = [tmp_path / "candidate-a", tmp_path / "candidate-b"]
    for candidate in candidates:
        candidate.mkdir()

    starts = threading.Barrier(3)
    observed: queue.Queue[tuple[str, Path, list[str]]] = queue.Queue()
    original_docker_args = sandbox_module._docker_args

    def record_docker_args(*args: Any, **kwargs: Any) -> list[str]:
        docker_args = original_docker_args(*args, **kwargs)
        name = docker_args[docker_args.index("--name") + 1]
        mount = docker_args[docker_args.index("-v") + 1]
        workspace = Path(mount.removesuffix(":/workspace:rw"))
        observed.put((name, workspace, docker_args))
        return docker_args

    monkeypatch.setattr(sandbox_module, "_docker_args", record_docker_args)
    command = [
        "sh",
        "-c",
        (
            'mkdir -p "$XDG_DATA_HOME/opencode"; '
            'printf "%s\\n" "$PROPOSAL_ID" > "$XDG_DATA_HOME/opencode/owner"; '
            'printf "%s\\n" "$PROPOSAL_ID" > /workspace/result.txt; '
            "while [ ! -e /workspace/release ]; do sleep 0.05; done"
        ),
    ]

    def run_proposal(index: int) -> subprocess.CompletedProcess[str]:
        starts.wait(timeout=10)
        return run_sandboxed_command(
            command,
            cwd=candidates[index],
            env={
                "PROPOSAL_ID": f"candidate-{index}",
                "XDG_DATA_HOME": "/workspace/.helix_opencode_state",
            },
            sandbox=SandboxConfig(
                enabled=True,
                image=_FIXTURE_IMAGE,
                network="none",
                timeout_seconds=30,
            ),
            scope="agent",
            sync_back=True,
            image=_FIXTURE_IMAGE,
            agent_backend="opencode",
        )

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    futures = [pool.submit(run_proposal, index) for index in range(2)]
    container_names: list[str] = []
    workspaces: list[Path] = []
    docker_runs: list[list[str]] = []
    try:
        starts.wait(timeout=10)
        for _ in range(2):
            name, workspace, docker_args = observed.get(timeout=15)
            container_names.append(name)
            workspaces.append(workspace)
            docker_runs.append(docker_args)

        assert len(set(container_names)) == 2
        assert len(set(workspaces)) == 2
        assert all("--rm" in args for args in docker_runs)
        assert all(
            "XDG_DATA_HOME=/workspace/.helix_opencode_state" in args
            for args in docker_runs
        )
        assert all(
            any(item.endswith(":/workspace:rw") for item in args)
            for args in docker_runs
        )

        _wait_until_running(container_names)
        owners = [
            _docker(
                "exec",
                name,
                "sh",
                "-c",
                'cat "$XDG_DATA_HOME/opencode/owner"',
                check=True,
            ).stdout.strip()
            for name in container_names
        ]
        assert set(owners) == {"candidate-0", "candidate-1"}
    finally:
        for name in container_names:
            _docker("exec", name, "touch", "/workspace/release")
        for future in futures:
            future.result(timeout=20)
        pool.shutdown(wait=True)
        for name in container_names:
            _docker("rm", "-f", name)

    assert (candidates[0] / "result.txt").read_text() == "candidate-0\n"
    assert (candidates[1] / "result.txt").read_text() == "candidate-1\n"
    assert all(not workspace.parent.exists() for workspace in workspaces)
    assert all(_docker("inspect", name).returncode != 0 for name in container_names)


@pytest.mark.parametrize(
    ("command", "timeout_seconds", "expected_returncode"),
    [
        (["sh", "-c", "exit 7"], 10, 7),
        (["sh", "-c", "sleep 5"], 1, None),
    ],
    ids=["nonzero-exit", "timeout"],
)
def test_sandbox_failure_paths_remove_container_and_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
    timeout_seconds: int,
    expected_returncode: int | None,
) -> None:
    """Daemon-backed failures leave neither containers nor temp workspaces."""
    _require_local_docker_fixture()
    candidate = tmp_path / "candidate"
    candidate.mkdir()

    observed: list[tuple[str, Path]] = []
    original_docker_args = sandbox_module._docker_args

    def record_docker_args(*args: Any, **kwargs: Any) -> list[str]:
        docker_args = original_docker_args(*args, **kwargs)
        name = docker_args[docker_args.index("--name") + 1]
        mount = docker_args[docker_args.index("-v") + 1]
        workspace = Path(mount.removesuffix(":/workspace:rw"))
        observed.append((name, workspace))
        return docker_args

    monkeypatch.setattr(sandbox_module, "_docker_args", record_docker_args)
    sandbox = SandboxConfig(
        enabled=True,
        image=_FIXTURE_IMAGE,
        network="none",
        timeout_seconds=timeout_seconds,
    )

    if expected_returncode is None:
        with pytest.raises(subprocess.TimeoutExpired):
            run_sandboxed_command(
                command,
                cwd=candidate,
                env={},
                sandbox=sandbox,
                scope="evaluator",
                sync_back=True,
                image=_FIXTURE_IMAGE,
            )
    else:
        result = run_sandboxed_command(
            command,
            cwd=candidate,
            env={},
            sandbox=sandbox,
            scope="evaluator",
            sync_back=True,
            image=_FIXTURE_IMAGE,
        )
        assert result.returncode == expected_returncode

    assert len(observed) == 1
    container_name, workspace = observed[0]
    assert not workspace.parent.exists()
    assert _docker("inspect", container_name).returncode != 0
