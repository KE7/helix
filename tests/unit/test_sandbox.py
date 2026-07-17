from __future__ import annotations

import concurrent.futures
import os
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import helix.sandbox as sandbox_module
from helix.config import EvaluatorSidecarConfig, SandboxConfig
from helix.sandbox import (
    EvaluatorSidecarRuntime,
    _healthcheck_docker_args,
    _run_docker,
    current_evaluator_sidecar_runtime,
    evaluator_sidecar_runtime,
    resolve_sandbox_image,
    run_sandboxed_command,
    run_sandboxed_commands,
    sandbox_auth_docker_args,
    sandbox_auth_volume_name,
    start_evaluator_sidecar,
)


_SYNTHETIC_SECRET = "synthetic-sentinel-secret-for-redaction-tests"
_SYNTHETIC_ENDPOINT_FRAGMENT = "synthetic-endpoint-fragment-for-redaction-tests"
_SYNTHETIC_PRIOR_SECRET = "synthetic-prior-invocation-redaction-value"
_NON_HEURISTIC_ENV_KEY = "SYNTHETIC_SETTING"


def _endpoint_env_args(endpoint: str, form: str) -> list[str]:
    assignment = f"HELIX_EVALUATOR_ENDPOINT={endpoint}"
    if form == "short-separated":
        return ["-e", assignment]
    if form == "long-separated":
        return ["--env", assignment]
    if form == "short-joined":
        return [f"-e{assignment}"]
    if form == "long-joined":
        return [f"--env={assignment}"]
    raise AssertionError(f"unexpected synthetic env form: {form}")


_SYNTHETIC_ENDPOINTS = [
    f"https://{_SYNTHETIC_ENDPOINT_FRAGMENT}@synthetic.invalid/evaluate",
    (
        "https://synthetic-user:"
        f"{_SYNTHETIC_ENDPOINT_FRAGMENT}@synthetic.invalid/evaluate"
    ),
    (f"https://synthetic.invalid/evaluate?opaque={_SYNTHETIC_ENDPOINT_FRAGMENT}"),
    f"https://synthetic.invalid/evaluate#{_SYNTHETIC_ENDPOINT_FRAGMENT}",
]


_SYNTHETIC_STRUCTURED_ENDPOINTS = [
    pytest.param(
        "https://synthetic-user%2Bdecoded@synthetic.invalid/evaluate",
        "synthetic-user%2Bdecoded",
        "synthetic-user+decoded",
        "short-separated",
        id="username",
    ),
    pytest.param(
        "https://synthetic-user:synthetic-password%2Fdecoded@"
        "synthetic.invalid/evaluate",
        "synthetic-password%2Fdecoded",
        "synthetic-password/decoded",
        "long-separated",
        id="password",
    ),
    pytest.param(
        "https://synthetic.invalid/evaluate?opaque=synthetic-query%3Fdecoded",
        "synthetic-query%3Fdecoded",
        "synthetic-query?decoded",
        "short-joined",
        id="query-value",
    ),
    pytest.param(
        "https://synthetic.invalid/evaluate#synthetic-fragment%23decoded",
        "synthetic-fragment%23decoded",
        "synthetic-fragment#decoded",
        "long-joined",
        id="fragment",
    ),
]


_SYNTHETIC_ENDPOINT_COMPONENT_CASES = [
    pytest.param(
        "https://example.invalid/execute?synthetic-query-bare-sentinel",
        ("synthetic-query-bare-sentinel",),
        id="query-bare",
    ),
    pytest.param(
        "https://example.invalid/execute?synthetic-query-blank-sentinel=",
        (
            "synthetic-query-blank-sentinel=",
            "synthetic-query-blank-sentinel",
        ),
        id="query-blank-value",
    ),
    pytest.param(
        "https://example.invalid/execute?"
        "synthetic-query-key-sentinel=synthetic-query-value-sentinel",
        (
            "synthetic-query-key-sentinel=synthetic-query-value-sentinel",
            "synthetic-query-key-sentinel",
            "synthetic-query-value-sentinel",
        ),
        id="query-key-value",
    ),
    pytest.param(
        "https://example.invalid/execute?"
        "synthetic-query%2Fkey-sentinel=synthetic-query%2Fvalue-sentinel",
        (
            "synthetic-query%2Fkey-sentinel=synthetic-query%2Fvalue-sentinel",
            "synthetic-query%2Fkey-sentinel",
            "synthetic-query/key-sentinel",
            "synthetic-query%2Fvalue-sentinel",
            "synthetic-query/value-sentinel",
        ),
        id="query-percent-decoded-key-value",
    ),
    pytest.param(
        "https://example.invalid/execute?"
        "synthetic+query+key+sentinel=synthetic+query+value+sentinel",
        (
            "synthetic+query+key+sentinel=synthetic+query+value+sentinel",
            "synthetic+query+key+sentinel",
            "synthetic query key sentinel",
            "synthetic+query+value+sentinel",
            "synthetic query value sentinel",
        ),
        id="query-plus-decoded-key-value",
    ),
    pytest.param(
        "https://example.invalid/execute?"
        "synthetic-query-first-key=synthetic-query-first-value&"
        "synthetic-query-second-bare",
        (
            "synthetic-query-first-key=synthetic-query-first-value&"
            "synthetic-query-second-bare",
            "synthetic-query-first-key=synthetic-query-first-value",
            "synthetic-query-first-key",
            "synthetic-query-first-value",
            "synthetic-query-second-bare",
        ),
        id="query-segments",
    ),
    pytest.param(
        "https://example.invalid/execute#synthetic-fragment-bare-sentinel",
        ("synthetic-fragment-bare-sentinel",),
        id="fragment-bare",
    ),
    pytest.param(
        "https://example.invalid/execute#synthetic-fragment-blank-sentinel=",
        (
            "synthetic-fragment-blank-sentinel=",
            "synthetic-fragment-blank-sentinel",
        ),
        id="fragment-blank-value",
    ),
    pytest.param(
        "https://example.invalid/execute#"
        "synthetic-fragment-key-sentinel=synthetic-fragment-value-sentinel",
        (
            "synthetic-fragment-key-sentinel=synthetic-fragment-value-sentinel",
            "synthetic-fragment-key-sentinel",
            "synthetic-fragment-value-sentinel",
        ),
        id="fragment-key-value",
    ),
    pytest.param(
        "https://example.invalid/execute#"
        "synthetic-fragment%2Fkey-sentinel=synthetic-fragment%2Fvalue-sentinel",
        (
            "synthetic-fragment%2Fkey-sentinel=synthetic-fragment%2Fvalue-sentinel",
            "synthetic-fragment%2Fkey-sentinel",
            "synthetic-fragment/key-sentinel",
            "synthetic-fragment%2Fvalue-sentinel",
            "synthetic-fragment/value-sentinel",
        ),
        id="fragment-percent-decoded-key-value",
    ),
    pytest.param(
        "https://example.invalid/execute#"
        "synthetic+fragment+key+sentinel=synthetic+fragment+value+sentinel",
        (
            "synthetic+fragment+key+sentinel=synthetic+fragment+value+sentinel",
            "synthetic+fragment+key+sentinel",
            "synthetic fragment key sentinel",
            "synthetic+fragment+value+sentinel",
            "synthetic fragment value sentinel",
        ),
        id="fragment-plus-decoded-key-value",
    ),
    pytest.param(
        "https://example.invalid/execute#"
        "synthetic-fragment-first-key=synthetic-fragment-first-value&"
        "synthetic-fragment-second-bare",
        (
            "synthetic-fragment-first-key=synthetic-fragment-first-value&"
            "synthetic-fragment-second-bare",
            "synthetic-fragment-first-key=synthetic-fragment-first-value",
            "synthetic-fragment-first-key",
            "synthetic-fragment-first-value",
            "synthetic-fragment-second-bare",
        ),
        id="fragment-segments",
    ),
]


def _endpoint_duplicated_argv(
    endpoint: str,
    raw_component: str,
    decoded_component: str,
    form: str,
) -> list[str]:
    return [
        "docker",
        "run",
        *_endpoint_env_args(endpoint, form),
        f"--endpoint={endpoint}",
        f"--raw-component={raw_component}",
        f"--decoded-component={decoded_component}",
        "synthetic-runner:latest",
        f"--healthcheck=probe {raw_component} {decoded_component}",
    ]


def _is_workspace_chown(args: list[str]) -> bool:
    """True for either workspace-recovery helper (chown or permission-relax).

    Existing call sites use this predicate to filter out housekeeping
    containers from the primary agent ``docker run``; both helpers share the
    same ``find /workspace`` boilerplate. Use :func:`_is_workspace_chown_only`
    or :func:`_is_workspace_permission_relax` when you need to distinguish
    them.
    """
    return args[:2] == ["docker", "run"] and any(
        "find /workspace -path /workspace/.git -prune" in item for item in args
    )


def _is_workspace_chown_only(args: list[str]) -> bool:
    return args[:2] == ["docker", "run"] and any(
        "chown" in item and "find /workspace" in item for item in args
    )


def _is_workspace_permission_relax(args: list[str]) -> bool:
    return args[:2] == ["docker", "run"] and any("chmod a+rwX" in item for item in args)


def test_resolve_sandbox_image_defaults_from_backend():
    cfg = SandboxConfig(enabled=True)
    assert (
        resolve_sandbox_image(cfg, "claude")
        == "ghcr.io/ke7/helix-evo-runner-claude:latest"
    )
    assert (
        resolve_sandbox_image(cfg, "codex")
        == "ghcr.io/ke7/helix-evo-runner-codex:latest"
    )
    assert (
        resolve_sandbox_image(cfg, "cursor")
        == "ghcr.io/ke7/helix-evo-runner-cursor:latest"
    )
    assert (
        resolve_sandbox_image(cfg, "gemini")
        == "ghcr.io/ke7/helix-evo-runner-gemini:latest"
    )
    assert (
        resolve_sandbox_image(cfg, "opencode")
        == "ghcr.io/ke7/helix-evo-runner-opencode:latest"
    )


def test_resolve_sandbox_image_honors_override():
    cfg = SandboxConfig(enabled=True, image="custom:latest")
    assert resolve_sandbox_image(cfg, "claude") == "custom:latest"


def test_docker_command_mounts_only_workspace_and_auth_volume(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value="1000:1000")

    cfg = SandboxConfig(
        enabled=True,
        image="helix-test:latest",
        network="none",
        cpus=1.5,
        memory="2g",
        timeout_seconds=10,
        add_host_gateway=True,
        extra_hosts={"env-endpoint": "host-gateway", "local-service": "127.0.0.1"},
    )

    run_sandboxed_command(
        ["codex", "exec", "prompt"],
        cwd=source,
        env={"HELIX_DEBUG": "1"},
        sandbox=cfg,
        scope="agent",
        sync_back=True,
        agent_backend="codex",
    )

    docker_call = next(
        call
        for call in calls
        if call[:2] == ["docker", "run"]
        and "--user" in call
        and call[call.index("--user") + 1] == "node"
    )
    joined = " ".join(docker_call)
    assert "--network none" in joined
    assert "--user node" in joined
    assert "--cpus 1.5" in joined
    assert "--memory 2g" in joined
    assert "--add-host" in joined
    assert "--add-host host.docker.internal:host-gateway" in joined
    assert "--add-host env-endpoint:host-gateway" in joined
    assert "--add-host local-service:127.0.0.1" in joined
    assert "helix-test:latest" in docker_call
    assert "-e" in docker_call
    assert "HOME=/home/node" in docker_call
    assert "helix-auth-codex:/home/node:rw" in docker_call
    assert f"{tmp_path}:" not in joined
    assert "/workspace:rw" in joined
    chown_calls = [call for call in calls if _is_workspace_chown(call)]
    assert len(chown_calls) == 2
    assert "node:node" in chown_calls[0]


def test_evaluator_scope_does_not_mount_agent_auth(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")

    mock_run = mocker.patch(
        "helix.sandbox.subprocess.run",
        return_value=MagicMock(stdout="", stderr="", returncode=0),
    )

    cfg = SandboxConfig(enabled=True)
    run_sandboxed_command(
        ["python", "evaluate.py"],
        cwd=source,
        env={},
        sandbox=cfg,
        scope="evaluator",
        sync_back=False,
        image="helix-test:latest",
    )

    docker_call = next(
        call.args[0]
        for call in mock_run.call_args_list
        if call.args[0][:2] == ["docker", "run"]
    )
    assert "helix-auth-codex:/home/node:rw" not in docker_call


def test_sidecar_runtime_switches_evaluator_to_private_network(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")

    seen_calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        seen_calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    runtime = EvaluatorSidecarRuntime(
        network="helix-eval-private",
        container_name="helix-evaluator-test",
        endpoint="http://helix-evaluator:8080/evaluate",
    )
    with evaluator_sidecar_runtime(runtime):
        run_sandboxed_command(
            ["python", "/runner/evaluate.py"],
            cwd=source,
            env={},
            sandbox=SandboxConfig(enabled=True),
            scope="evaluator",
            sync_back=False,
            image="helix-test:latest",
        )

    docker_call = next(
        call
        for call in seen_calls
        if call[:2] == ["docker", "run"] and not _is_workspace_chown(call)
    )
    assert docker_call[docker_call.index("--network") + 1] == "helix-eval-private"
    assert (
        "HELIX_EVALUATOR_ENDPOINT=http://helix-evaluator:8080/evaluate" in docker_call
    )
    assert "helix-auth-codex:/home/node:rw" not in docker_call


def test_sidecar_runtime_is_visible_to_worker_threads():
    runtime = EvaluatorSidecarRuntime(
        network="helix-eval-private",
        container_name="helix-evaluator-test",
        endpoint="http://helix-evaluator:8080/evaluate",
    )
    seen: list[EvaluatorSidecarRuntime | None] = []

    with evaluator_sidecar_runtime(runtime):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: seen.append(current_evaluator_sidecar_runtime())
            )
            future.result()

    assert seen == [runtime]
    # Stack must be empty after the context manager exits.
    assert current_evaluator_sidecar_runtime() is None


def test_parallel_sandboxes_use_distinct_mounts_and_sidecar_runtime(
    tmp_path: Path, mocker
):
    """Parallel evaluators inherit the sidecar and get disposable containers."""
    sources = [tmp_path / "candidate-a", tmp_path / "candidate-b"]
    for index, source in enumerate(sources):
        source.mkdir()
        (source / "candidate.txt").write_text(f"candidate-{index}\n")

    primary_runs: list[list[str]] = []
    cleanup_runs: list[list[str]] = []
    lock = threading.Lock()
    overlap = threading.Barrier(2)

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and "--name" in args:
            with lock:
                primary_runs.append(args)
            overlap.wait(timeout=10)
        elif args[:3] == ["docker", "rm", "-f"]:
            with lock:
                cleanup_runs.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)
    runtime = EvaluatorSidecarRuntime(
        network="helix-eval-private",
        container_name="helix-evaluator-test",
        endpoint="http://helix-evaluator:8080/evaluate",
    )

    with evaluator_sidecar_runtime(runtime):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    run_sandboxed_command,
                    ["sh", "-c", "true"],
                    cwd=source,
                    env={},
                    sandbox=SandboxConfig(enabled=True),
                    scope="evaluator",
                    sync_back=False,
                    image="helix-test:latest",
                )
                for source in sources
            ]
            for future in futures:
                assert future.result(timeout=15).returncode == 0

    assert len(primary_runs) == 2
    mounts = {call[call.index("-v") + 1] for call in primary_runs}
    assert len(mounts) == 2
    assert all(mount.endswith(":/workspace:rw") for mount in mounts)
    container_names = {call[call.index("--name") + 1] for call in primary_runs}
    assert len(container_names) == 2
    assert all("--rm" in call for call in primary_runs)
    assert all(
        call[call.index("--network") + 1] == runtime.network for call in primary_runs
    )
    assert all(
        f"HELIX_EVALUATOR_ENDPOINT={runtime.endpoint}" in call for call in primary_runs
    )
    assert {call[-1] for call in cleanup_runs} == container_names
    assert current_evaluator_sidecar_runtime() is None


def test_sandbox_exception_forces_container_and_workspace_cleanup(
    tmp_path: Path, mocker
):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")
    container_name: str | None = None
    workspace: Path | None = None
    cleanup_names: list[str] = []

    def fake_run(args, **kwargs):
        nonlocal container_name, workspace
        if args[:2] == ["docker", "run"] and "--name" in args:
            container_name = args[args.index("--name") + 1]
            workspace = Path(args[args.index("-v") + 1].split(":/workspace:rw", 1)[0])
            raise subprocess.TimeoutExpired(args, timeout=1)
        if args[:3] == ["docker", "rm", "-f"]:
            cleanup_names.append(args[-1])
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)

    with pytest.raises(subprocess.TimeoutExpired):
        run_sandboxed_command(
            ["sh", "-c", "true"],
            cwd=source,
            env={},
            sandbox=SandboxConfig(enabled=True, timeout_seconds=1),
            scope="evaluator",
            sync_back=False,
            image="helix-test:latest",
        )

    assert container_name is not None
    assert cleanup_names == [container_name]
    assert workspace is not None
    assert not workspace.parent.exists()


@pytest.mark.parametrize(
    "env_args",
    [
        ["-e", f"SYNTHETIC_API_KEY={_SYNTHETIC_SECRET}"],
        ["--env", f"SYNTHETIC_API_KEY={_SYNTHETIC_SECRET}"],
        [f"-eSYNTHETIC_API_KEY={_SYNTHETIC_SECRET}"],
        [f"--env=SYNTHETIC_API_KEY={_SYNTHETIC_SECRET}"],
    ],
)
def test_docker_argv_redaction_preserves_key_for_all_env_forms(env_args):
    rendered = repr(
        sandbox_module._redact_docker_argv(
            ["docker", "run", *env_args, "fixture:latest"]
        )
    )

    assert _SYNTHETIC_SECRET not in rendered
    assert "SYNTHETIC_API_KEY=<redacted>" in rendered


@pytest.mark.parametrize(
    ("endpoint", "components"), _SYNTHETIC_ENDPOINT_COMPONENT_CASES
)
def test_endpoint_component_redaction_values_cover_structured_fields(
    endpoint: str, components: tuple[str, ...]
):
    values = sandbox_module._endpoint_component_redaction_values(endpoint)

    assert set(components) <= values
    assert "" not in values
    assert not {"https", "example.invalid", "/execute"} & values


def test_short_query_key_redaction_does_not_corrupt_harmless_url_context():
    endpoint = "https://example.invalid/execute?x=synthetic-short-key-value-sentinel"
    values = sandbox_module._endpoint_component_redaction_values(endpoint)
    diagnostic = "https://example.invalid/execute failed for query key x"

    rendered = sandbox_module._redact_diagnostic_output(
        diagnostic, tuple(sorted(values, key=len, reverse=True))
    )

    assert "x" in values
    assert rendered == (
        "https://example.invalid/execute failed for query key <redacted>"
    )


@pytest.mark.parametrize(
    "form",
    ["short-separated", "long-separated", "short-joined", "long-joined"],
)
@pytest.mark.parametrize(
    ("endpoint", "components"), _SYNTHETIC_ENDPOINT_COMPONENT_CASES
)
def test_docker_argv_redacts_all_endpoint_component_variants(
    endpoint: str, components: tuple[str, ...], form: str
):
    raw_args = [
        "docker",
        "run",
        *_endpoint_env_args(endpoint, form),
        f"--endpoint={endpoint}",
        *(f"--duplicate-{index}={value}" for index, value in enumerate(components)),
        "--url-context=https://example.invalid/execute",
        "synthetic-runner:latest",
    ]
    original_args = list(raw_args)

    redacted = sandbox_module._redact_docker_argv(raw_args)
    rendered = repr(redacted)

    assert endpoint not in rendered
    assert all(component not in rendered for component in components)
    assert "HELIX_EVALUATOR_ENDPOINT=<redacted>" in rendered
    assert "--endpoint=<redacted>" in redacted
    assert all(
        f"--duplicate-{index}=<redacted>" in redacted
        for index in range(len(components))
    )
    assert "--url-context=https://example.invalid/execute" in redacted
    assert raw_args == original_args


@pytest.mark.parametrize(
    "form",
    ["short-separated", "long-separated", "short-joined", "long-joined"],
)
@pytest.mark.parametrize(
    ("endpoint", "components"), _SYNTHETIC_ENDPOINT_COMPONENT_CASES
)
@pytest.mark.parametrize("outcome", ["success", "nonzero", "timeout", "called-process"])
def test_docker_process_component_redaction_covers_every_diagnostic_surface(
    mocker,
    endpoint: str,
    components: tuple[str, ...],
    form: str,
    outcome: str,
):
    raw_args = [
        "docker",
        "run",
        *_endpoint_env_args(endpoint, form),
        f"--endpoint={endpoint}",
        *(f"--duplicate-{index}={value}" for index, value in enumerate(components)),
        f"--explicit={_SYNTHETIC_PRIOR_SECRET}",
        "--url-context=https://example.invalid/execute",
        "synthetic-runner:latest",
    ]
    original_args = list(raw_args)
    captured_subprocess_args: list[list[str]] = []
    secret_payload = " | ".join((endpoint, *components, _SYNTHETIC_PRIOR_SECRET))
    stdout = (
        "synthetic successful functional output"
        if outcome == "success"
        else f"synthetic diagnostic stdout: {secret_payload}"
    )
    stderr = (
        "synthetic successful functional stderr"
        if outcome == "success"
        else f"synthetic diagnostic stderr: {secret_payload}"
    )

    def fake_run(args, **kwargs):
        captured_subprocess_args.append(list(args))
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(
                args, timeout=1, output=stdout, stderr=stderr
            )
        if outcome == "called-process":
            raise subprocess.CalledProcessError(125, args, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(
            args,
            0 if outcome == "success" else 9,
            stdout=stdout,
            stderr=stderr,
        )

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    sentinels = (endpoint, *components, _SYNTHETIC_PRIOR_SECRET)

    if outcome in {"timeout", "called-process"}:
        expected_exception = (
            subprocess.TimeoutExpired
            if outcome == "timeout"
            else subprocess.CalledProcessError
        )
        with pytest.raises(expected_exception) as captured:
            _run_docker(
                raw_args,
                check=outcome == "called-process",
                redaction_values=[_SYNTHETIC_PRIOR_SECRET],
            )
        exc = captured.value
        renderings = [
            str(exc),
            repr(exc),
            repr(exc.cmd),
            repr(exc.args),
            str(exc.output),
            str(exc.stderr),
        ]
        safe_args = exc.cmd
    else:
        result = _run_docker(
            raw_args,
            check=False,
            redaction_values=[_SYNTHETIC_PRIOR_SECRET],
        )
        safe_args = result.args
        if outcome == "success":
            assert result.stdout == stdout
            assert result.stderr == stderr
            renderings = [repr(result.args)]
        else:
            renderings = [
                str(result),
                repr(result),
                repr(result.args),
                str(result.stdout),
                str(result.stderr),
            ]

    assert all(
        sentinel not in rendering for sentinel in sentinels for rendering in renderings
    )
    assert "HELIX_EVALUATOR_ENDPOINT=<redacted>" in repr(safe_args)
    assert "--endpoint=<redacted>" in safe_args
    assert "--explicit=<redacted>" in safe_args
    assert all(
        f"--duplicate-{index}=<redacted>" in safe_args
        for index in range(len(components))
    )
    assert "--url-context=https://example.invalid/execute" in safe_args
    assert captured_subprocess_args == [original_args]
    assert raw_args == original_args


@pytest.mark.parametrize(
    ("endpoint", "raw_component", "decoded_component", "form"),
    _SYNTHETIC_STRUCTURED_ENDPOINTS,
)
def test_docker_argv_redacts_endpoint_and_components_duplicated_in_other_tokens(
    endpoint: str,
    raw_component: str,
    decoded_component: str,
    form: str,
):
    raw_args = _endpoint_duplicated_argv(
        endpoint, raw_component, decoded_component, form
    )

    rendered = repr(sandbox_module._redact_docker_argv(raw_args))

    for sentinel in (endpoint, raw_component, decoded_component):
        assert sentinel not in rendered
    assert "HELIX_EVALUATOR_ENDPOINT=<redacted>" in rendered
    assert "--endpoint=<redacted>" in rendered
    assert "--raw-component=<redacted>" in rendered
    assert "--decoded-component=<redacted>" in rendered
    assert "--healthcheck=probe <redacted> <redacted>" in rendered


@pytest.mark.parametrize(
    ("endpoint", "raw_component", "decoded_component", "form"),
    _SYNTHETIC_STRUCTURED_ENDPOINTS,
)
@pytest.mark.parametrize("outcome", ["success", "nonzero", "timeout", "called-process"])
def test_docker_process_redacts_duplicated_endpoint_values_from_diagnostics(
    mocker,
    endpoint: str,
    raw_component: str,
    decoded_component: str,
    form: str,
    outcome: str,
):
    raw_args = _endpoint_duplicated_argv(
        endpoint, raw_component, decoded_component, form
    )
    raw_args.append(f"--prior-token={_SYNTHETIC_PRIOR_SECRET}")
    original_args = list(raw_args)
    captured_subprocess_args: list[list[str]] = []
    stdout = (
        f"functional stdout: {endpoint} {decoded_component} {_SYNTHETIC_PRIOR_SECRET}"
    )
    stderr = f"functional stderr: {raw_component}"

    def fake_run(args, **kwargs):
        captured_subprocess_args.append(list(args))
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(
                args, timeout=1, output=stdout, stderr=stderr
            )
        if outcome == "called-process":
            raise subprocess.CalledProcessError(125, args, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(
            args,
            0 if outcome == "success" else 9,
            stdout=stdout,
            stderr=stderr,
        )

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    sentinels = (
        endpoint,
        raw_component,
        decoded_component,
        _SYNTHETIC_PRIOR_SECRET,
    )

    if outcome in {"timeout", "called-process"}:
        expected_exception = (
            subprocess.TimeoutExpired
            if outcome == "timeout"
            else subprocess.CalledProcessError
        )
        with pytest.raises(expected_exception) as captured:
            _run_docker(
                raw_args,
                check=outcome == "called-process",
                redaction_values=[_SYNTHETIC_PRIOR_SECRET],
            )
        exc = captured.value
        renderings = [
            str(exc),
            repr(exc),
            repr(exc.cmd),
            repr(exc.args),
            str(exc.output),
            str(exc.stderr),
            f"subprocess diagnostic: {type(exc).__name__}: {exc}",
        ]
        assert all(
            sentinel not in rendering
            for sentinel in sentinels
            for rendering in renderings
        )
        safe_args = repr(exc.cmd)
    else:
        result = _run_docker(
            raw_args,
            check=False,
            redaction_values=[_SYNTHETIC_PRIOR_SECRET],
        )
        safe_args = repr(result.args)
        assert all(sentinel not in safe_args for sentinel in sentinels)
        if outcome == "success":
            assert result.stdout == stdout
            assert result.stderr == stderr
        else:
            renderings = [repr(result), str(result.stdout), str(result.stderr)]
            assert all(
                sentinel not in rendering
                for sentinel in sentinels
                for rendering in renderings
            )

    assert "HELIX_EVALUATOR_ENDPOINT=<redacted>" in safe_args
    assert "--endpoint=<redacted>" in safe_args
    assert "--raw-component=<redacted>" in safe_args
    assert "--decoded-component=<redacted>" in safe_args
    assert "--prior-token=<redacted>" in safe_args
    assert captured_subprocess_args == [original_args]
    assert raw_args == original_args


def test_explicit_prior_invocation_redaction_values_are_scrubbed_from_argv():
    raw_args = [
        "docker",
        "logs",
        "synthetic-container",
        f"--token={_SYNTHETIC_PRIOR_SECRET}",
        f"--healthcheck=probe {_SYNTHETIC_PRIOR_SECRET}",
    ]
    original_args = list(raw_args)

    rendered = repr(
        sandbox_module._redact_docker_argv(
            raw_args, redaction_values=[_SYNTHETIC_PRIOR_SECRET]
        )
    )

    assert _SYNTHETIC_PRIOR_SECRET not in rendered
    assert "--token=<redacted>" in rendered
    assert "--healthcheck=probe <redacted>" in rendered
    assert raw_args == original_args


@pytest.mark.parametrize(
    "form",
    ["short-separated", "long-separated", "short-joined", "long-joined"],
)
@pytest.mark.parametrize(
    "endpoint",
    _SYNTHETIC_ENDPOINTS,
    ids=["username", "password", "query-value", "fragment"],
)
@pytest.mark.parametrize("failure", ["timeout", "called-process"])
def test_endpoint_fragment_redacted_from_subprocess_exception(
    mocker, form: str, endpoint: str, failure: str
):
    raw_args = [
        "docker",
        "run",
        *_endpoint_env_args(endpoint, form),
        "synthetic-runner:latest",
    ]

    def fake_run(args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                args,
                timeout=1,
                output=f"stdout: {_SYNTHETIC_ENDPOINT_FRAGMENT}",
                stderr=f"stderr: {_SYNTHETIC_ENDPOINT_FRAGMENT}",
            )
        raise subprocess.CalledProcessError(
            125,
            args,
            output=f"stdout: {_SYNTHETIC_ENDPOINT_FRAGMENT}",
            stderr=f"stderr: {_SYNTHETIC_ENDPOINT_FRAGMENT}",
        )

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    expected_exception = (
        subprocess.TimeoutExpired
        if failure == "timeout"
        else subprocess.CalledProcessError
    )

    with pytest.raises(expected_exception) as captured:
        _run_docker(raw_args)

    exc = captured.value
    renderings = [
        str(exc),
        repr(exc),
        repr(exc.cmd),
        repr(exc.args),
        str(exc.output),
        str(exc.stderr),
        f"healthcheck exception: {type(exc).__name__}: {exc}",
    ]
    assert all(_SYNTHETIC_ENDPOINT_FRAGMENT not in item for item in renderings)
    assert "HELIX_EVALUATOR_ENDPOINT=<redacted>" in repr(exc.cmd)


@pytest.mark.parametrize(
    "form",
    ["short-separated", "long-separated", "short-joined", "long-joined"],
)
@pytest.mark.parametrize(
    "endpoint",
    _SYNTHETIC_ENDPOINTS,
    ids=["username", "password", "query-value", "fragment"],
)
def test_endpoint_fragment_redacted_from_nonzero_diagnostic(
    mocker, form: str, endpoint: str
):
    raw_args = [
        "docker",
        "run",
        *_endpoint_env_args(endpoint, form),
        "synthetic-runner:latest",
    ]
    mocker.patch(
        "helix.sandbox.subprocess.run",
        return_value=subprocess.CompletedProcess(
            raw_args,
            9,
            stdout=f"stdout: {_SYNTHETIC_ENDPOINT_FRAGMENT}",
            stderr=f"stderr: {_SYNTHETIC_ENDPOINT_FRAGMENT}",
        ),
    )

    result = _run_docker(raw_args, check=False)

    renderings = [repr(result), str(result.stdout), str(result.stderr)]
    assert all(_SYNTHETIC_ENDPOINT_FRAGMENT not in item for item in renderings)
    assert "HELIX_EVALUATOR_ENDPOINT=<redacted>" in repr(result.args)


def test_endpoint_redaction_preserves_harmless_url_context(mocker):
    endpoint = (
        f"https://synthetic.invalid/evaluate?opaque={_SYNTHETIC_ENDPOINT_FRAGMENT}"
    )
    raw_args = [
        "docker",
        "run",
        "-e",
        f"HELIX_EVALUATOR_ENDPOINT={endpoint}",
        "synthetic-runner:latest",
    ]
    mocker.patch(
        "helix.sandbox.subprocess.run",
        return_value=subprocess.CompletedProcess(
            raw_args,
            9,
            stdout=(
                "synthetic.invalid/evaluate failed: opaque="
                f"{_SYNTHETIC_ENDPOINT_FRAGMENT}"
            ),
            stderr="",
        ),
    )

    result = _run_docker(raw_args, check=False)

    assert result.stdout == ("synthetic.invalid/evaluate failed: <redacted>")


@pytest.mark.parametrize("as_bytes", [False, True], ids=["text", "bytes"])
def test_timeout_exception_redacts_docker_env_in_all_renderings(
    tmp_path: Path, mocker, as_bytes: bool
):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")
    stdout: str | bytes = f"stdout echoed {_SYNTHETIC_SECRET}"
    stderr: str | bytes = f"stderr echoed {_SYNTHETIC_SECRET}"
    if as_bytes:
        stdout = stdout.encode()
        stderr = stderr.encode()

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and "--name" in args:
            raise subprocess.TimeoutExpired(
                args,
                timeout=1,
                output=stdout,
                stderr=stderr,
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)

    with pytest.raises(subprocess.TimeoutExpired) as captured:
        run_sandboxed_command(
            ["sh", "-c", "true"],
            cwd=source,
            env={_NON_HEURISTIC_ENV_KEY: _SYNTHETIC_SECRET},
            sandbox=SandboxConfig(enabled=True, timeout_seconds=1),
            scope="evaluator",
            sync_back=False,
            image="helix-test:latest",
        )

    exc = captured.value
    renderings = [
        str(exc),
        repr(exc),
        repr(exc.cmd),
        repr(exc.args),
        str(exc.output),
        str(exc.stderr),
        f"worker exception: {type(exc).__name__}: {exc}",
    ]
    assert all(_SYNTHETIC_SECRET not in item for item in renderings)
    assert f"{_NON_HEURISTIC_ENV_KEY}=<redacted>" in repr(exc.cmd)


def test_nonzero_result_redacts_docker_argv_and_output_diagnostics(
    tmp_path: Path, mocker
):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and "--name" in args:
            return subprocess.CompletedProcess(
                args,
                7,
                stdout=f"stdout echoed {_SYNTHETIC_SECRET}",
                stderr=f"stderr echoed {_SYNTHETIC_SECRET}",
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)

    result = run_sandboxed_command(
        ["sh", "-c", "exit 7"],
        cwd=source,
        env={_NON_HEURISTIC_ENV_KEY: _SYNTHETIC_SECRET},
        sandbox=SandboxConfig(enabled=True),
        scope="evaluator",
        sync_back=False,
        image="helix-test:latest",
    )

    renderings = [repr(result), repr(result.args), result.stdout, result.stderr]
    assert all(_SYNTHETIC_SECRET not in item for item in renderings)
    assert f"{_NON_HEURISTIC_ENV_KEY}=<redacted>" in repr(result.args)
    assert result.returncode == 7


def test_called_process_exception_redacts_docker_env_and_indirect_context(mocker):
    raw_args = [
        "docker",
        "run",
        "--env",
        f"{_NON_HEURISTIC_ENV_KEY}={_SYNTHETIC_SECRET}",
        "fixture:latest",
    ]

    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(
            125,
            args,
            output=f"stdout echoed {_SYNTHETIC_SECRET}".encode(),
            stderr=f"stderr echoed {_SYNTHETIC_SECRET}".encode(),
        )

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    with pytest.raises(subprocess.CalledProcessError) as captured:
        _run_docker(raw_args)

    exc = captured.value
    renderings = [
        str(exc),
        repr(exc),
        repr(exc.cmd),
        repr(exc.args),
        str(exc.output),
        str(exc.stderr),
        f"sidecar setup failed: {exc}",
    ]
    assert all(_SYNTHETIC_SECRET not in item for item in renderings)
    assert f"{_NON_HEURISTIC_ENV_KEY}=<redacted>" in repr(exc.cmd)


def test_successful_result_preserves_functional_output_while_redacting_argv(mocker):
    env_key = "SYNTHETIC_API_KEY"
    raw_args = [
        "docker",
        "run",
        "-e",
        f"{env_key}={_SYNTHETIC_SECRET}",
        "fixture:latest",
    ]
    stdout = f"functional output {_SYNTHETIC_SECRET}"
    mocker.patch(
        "helix.sandbox.subprocess.run",
        return_value=subprocess.CompletedProcess(raw_args, 0, stdout=stdout, stderr=""),
    )

    result = _run_docker(raw_args)

    assert result.stdout == stdout
    assert _SYNTHETIC_SECRET not in repr(result.args)
    assert f"{env_key}=<redacted>" in repr(result.args)


@pytest.mark.parametrize(
    "env_args",
    [
        ["-e", f"{_NON_HEURISTIC_ENV_KEY}={_SYNTHETIC_SECRET}"],
        ["--env", f"{_NON_HEURISTIC_ENV_KEY}={_SYNTHETIC_SECRET}"],
        [f"-e{_NON_HEURISTIC_ENV_KEY}={_SYNTHETIC_SECRET}"],
        [f"--env={_NON_HEURISTIC_ENV_KEY}={_SYNTHETIC_SECRET}"],
    ],
)
@pytest.mark.parametrize("as_bytes", [False, True], ids=["text", "bytes"])
def test_nonzero_diagnostic_output_redacts_all_env_forms(
    mocker, env_args: list[str], as_bytes: bool
):
    raw_args = ["docker", "run", *env_args, "fixture:latest"]
    output: str | bytes = f"{_NON_HEURISTIC_ENV_KEY}={_SYNTHETIC_SECRET}"
    if as_bytes:
        output = output.encode()
    mocker.patch(
        "helix.sandbox.subprocess.run",
        return_value=subprocess.CompletedProcess(
            raw_args, 9, stdout=output, stderr=output
        ),
    )

    result = _run_docker(raw_args, check=False)

    renderings = [repr(result), str(result.stdout), str(result.stderr)]
    assert all(_SYNTHETIC_SECRET not in item for item in renderings)
    assert f"{_NON_HEURISTIC_ENV_KEY}=<redacted>" in str(result.stdout)


def test_container_running_exit_logs_redact_sidecar_startup_env(mocker):
    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="false exited\n", stderr=""
            )
        if args[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=f"{_NON_HEURISTIC_ENV_KEY}={_SYNTHETIC_SECRET}\n",
                stderr="",
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    sidecar = EvaluatorSidecarConfig(
        image="synthetic-sidecar:latest",
        command="python -m synthetic_server",
        endpoint="http://helix-evaluator:8080/evaluate",
    )

    with pytest.raises(RuntimeError) as captured:
        with start_evaluator_sidecar(
            sidecar,
            fixed_env={_NON_HEURISTIC_ENV_KEY: _SYNTHETIC_SECRET},
        ):
            pass

    rendered = str(captured.value)
    assert _SYNTHETIC_SECRET not in rendered
    assert f"{_NON_HEURISTIC_ENV_KEY}=<redacted>" in rendered


def test_sidecar_service_exit_logs_redact_sidecar_startup_env(mocker):
    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="false exited\n", stderr=""
            )
        if args[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="",
                stderr=f"{_NON_HEURISTIC_ENV_KEY}={_SYNTHETIC_SECRET}\n",
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._wait_for_container_running")
    sidecar = EvaluatorSidecarConfig(
        image="synthetic-sidecar:latest",
        command="python -m synthetic_server",
        endpoint="http://helix-evaluator:8080/evaluate",
    )

    with pytest.raises(RuntimeError) as captured:
        with start_evaluator_sidecar(
            sidecar,
            fixed_env={_NON_HEURISTIC_ENV_KEY: _SYNTHETIC_SECRET},
        ):
            pass

    rendered = str(captured.value)
    assert _SYNTHETIC_SECRET not in rendered
    assert f"{_NON_HEURISTIC_ENV_KEY}=<redacted>" in rendered


@pytest.mark.parametrize("failure_phase", ["container-start", "service-wait"])
def test_sidecar_logs_redact_endpoint_fragment(mocker, failure_phase: str):
    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="false exited\n", stderr=""
            )
        if args[:2] == ["docker", "logs"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=f"sidecar detail: {_SYNTHETIC_ENDPOINT_FRAGMENT}\n",
                stderr="",
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    if failure_phase == "service-wait":
        mocker.patch("helix.sandbox._wait_for_container_running")
    sidecar = EvaluatorSidecarConfig(
        image="synthetic-sidecar:latest",
        command="python -m synthetic_server",
        endpoint=(
            f"https://synthetic.invalid/evaluate?opaque={_SYNTHETIC_ENDPOINT_FRAGMENT}"
        ),
    )

    with pytest.raises(RuntimeError) as captured:
        with start_evaluator_sidecar(sidecar):
            pass

    rendered = str(captured.value)
    assert _SYNTHETIC_ENDPOINT_FRAGMENT not in rendered
    assert "stdout:\nsidecar detail: <redacted>" in rendered


def test_sidecar_healthcheck_timeout_redacts_endpoint_and_startup_env(mocker):
    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="true running\n", stderr=""
            )
        if any(item.startswith("HELIX_EVALUATOR_ENDPOINT=") for item in args):
            return subprocess.CompletedProcess(
                args,
                8,
                stdout=(
                    f"healthcheck detail: {_SYNTHETIC_ENDPOINT_FRAGMENT}\n"
                    f"{_NON_HEURISTIC_ENV_KEY}={_SYNTHETIC_SECRET}\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._wait_for_container_running")
    mocker.patch("helix.sandbox.time.monotonic", side_effect=[0.0, 0.0, 2.0])
    mocker.patch("helix.sandbox.time.sleep")
    endpoint = (
        f"https://synthetic.invalid/evaluate?opaque={_SYNTHETIC_ENDPOINT_FRAGMENT}"
    )
    sidecar = EvaluatorSidecarConfig(
        image="synthetic-sidecar:latest",
        command="python -m synthetic_server",
        endpoint=endpoint,
        startup_timeout_seconds=1,
    )

    with pytest.raises(TimeoutError) as captured:
        with start_evaluator_sidecar(
            sidecar,
            fixed_env={_NON_HEURISTIC_ENV_KEY: _SYNTHETIC_SECRET},
        ):
            pass

    rendered = str(captured.value)
    assert endpoint not in rendered
    assert _SYNTHETIC_ENDPOINT_FRAGMENT not in rendered
    assert _SYNTHETIC_SECRET not in rendered
    assert rendered.splitlines()[0] == (
        "Evaluator sidecar endpoint did not become reachable within 1s: "
        "HELIX_EVALUATOR_ENDPOINT=<redacted>"
    )
    assert f"{_NON_HEURISTIC_ENV_KEY}=<redacted>" in rendered


def test_sidecar_runtime_nested_same_runtime_restores_outer():
    """Nested ``with`` blocks must restore the outer runtime, not blank it."""
    outer = EvaluatorSidecarRuntime(
        network="net-outer", container_name="c-outer", endpoint="http://outer/"
    )
    inner = EvaluatorSidecarRuntime(
        network="net-inner", container_name="c-inner", endpoint="http://inner/"
    )
    with evaluator_sidecar_runtime(outer):
        assert current_evaluator_sidecar_runtime() is outer
        with evaluator_sidecar_runtime(inner):
            assert current_evaluator_sidecar_runtime() is inner
        assert current_evaluator_sidecar_runtime() is outer
    assert current_evaluator_sidecar_runtime() is None


def test_sidecar_runtime_accepts_unhashable_helixconfig_like_object():
    """Regression: prior design used a WeakKeyDictionary keyed on HelixConfig,
    which is a Pydantic ``BaseModel`` (unhashable). The new lock+stack design
    must not key on the config at all, so even an unhashable sentinel works.
    """

    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

    runtime = EvaluatorSidecarRuntime(
        network="net-x", container_name="c-x", endpoint="http://x/"
    )
    # Just exercising the API; unhashable config objects must not crash.
    Unhashable()  # constructed but never used as a key
    with evaluator_sidecar_runtime(runtime):
        assert current_evaluator_sidecar_runtime() is runtime
    assert current_evaluator_sidecar_runtime() is None


def test_sidecar_healthcheck_uses_runner_image_and_endpoint():
    sidecar = EvaluatorSidecarConfig(
        image="eval-service:latest",
        runner_image="eval-runner:latest",
        command="python -m server",
        endpoint="http://helix-evaluator:8080/evaluate",
        healthcheck_command="python /runner/healthcheck.py",
    )

    args = _healthcheck_docker_args(sidecar, network="helix-eval-private")

    assert args[args.index("--network") + 1] == "helix-eval-private"
    assert "HELIX_EVALUATOR_ENDPOINT=http://helix-evaluator:8080/evaluate" in args
    assert "eval-runner:latest" in args
    assert args[-2:] == ["python", "/runner/healthcheck.py"]


def test_start_evaluator_sidecar_injects_fixed_env(mocker):
    calls: list[list[str]] = []

    def fake_run_docker(args, **kwargs):
        calls.append(args)
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="true running\n", stderr=""
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox._run_docker", side_effect=fake_run_docker)

    sidecar = EvaluatorSidecarConfig(
        image="eval-service:latest",
        command="python -m server",
        endpoint="http://helix-evaluator:8080/evaluate",
    )

    with start_evaluator_sidecar(
        sidecar,
        fixed_env={"EVALUATOR_BASE_URL": "https://model-service.example.invalid/v1"},
    ):
        pass

    docker_run = next(call for call in calls if call[:3] == ["docker", "run", "-d"])
    assert "EVALUATOR_BASE_URL=https://model-service.example.invalid/v1" in docker_run


def test_agent_syncs_changes_back_but_excludes_git_and_artifacts(
    tmp_path: Path, mocker
):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "keep.py").write_text("old\n")
    (source / "delete.py").write_text("bye\n")
    (source / ".env").write_text("SECRET=value\n")
    (source / ".helix").mkdir()
    (source / ".helix" / "state.json").write_text("{}\n")
    (source / ".helix_artifacts").mkdir()
    (source / ".helix_artifacts" / "old.txt").write_text("old artifact\n")
    (source / "helix.toml").write_text("[evaluator.sidecar]\nendpoint = 'private'\n")

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            assert not (workspace / ".env").exists()
            assert not (workspace / ".helix").exists()
            assert not (workspace / ".helix_artifacts").exists()
            assert not (workspace / "helix.toml").exists()
            (workspace / "keep.py").write_text("new\n")
            (workspace / "delete.py").unlink()
            (workspace / "added.py").write_text("added\n")
            (workspace / "helix.toml").write_text("tampered\n")
            (workspace / ".env.local").write_text("NEW_SECRET=value\n")
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text("tampered\n")
            (workspace / ".helix_artifacts").mkdir()
            (workspace / ".helix_artifacts" / "new.txt").write_text("new artifact\n")
            (workspace / ".helix_backend_stdout.txt").write_text("artifact\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    cfg = SandboxConfig(enabled=True)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=cfg,
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "keep.py").read_text() == "new\n"
    assert not (source / "delete.py").exists()
    assert (source / "added.py").read_text() == "added\n"
    assert (source / ".env").read_text() == "SECRET=value\n"
    assert (
        source / "helix.toml"
    ).read_text() == "[evaluator.sidecar]\nendpoint = 'private'\n"
    assert (source / ".helix" / "state.json").read_text() == "{}\n"
    assert (source / ".helix_artifacts" / "old.txt").read_text() == "old artifact\n"
    assert not (source / ".helix_artifacts" / "new.txt").exists()
    assert not (source / ".helix_artifacts" / "backend_transcripts").exists()
    assert not (source / ".env.local").exists()
    assert not (source / ".git").exists()
    assert not (source / ".helix_backend_stdout.txt").exists()


def test_agent_sync_tolerates_inaccessible_workspace_paths(
    tmp_path: Path, mocker, monkeypatch
):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "keep.py").write_text("old\n")
    (source / ".gitignore").write_text("*.tmp\n")

    workspace_ready = False
    real_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if workspace_ready and path.name == ".gitignore" and "/workspace/" in str(path):
            raise PermissionError("permission denied")
        return real_exists(path)

    def fake_run(args, **kwargs):
        nonlocal workspace_ready
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace / "keep.py").write_text("new\n")
            workspace_ready = True
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    monkeypatch.setattr(Path, "exists", fake_exists)
    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "keep.py").read_text() == "new\n"
    assert (source / ".gitignore").read_text() == "*.tmp\n"


def test_agent_copies_claude_transcript_from_auth_volume(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            if args[-3:] and args[-3] == "sh" and "sess_123.jsonl" in args[-1]:
                transcript = (
                    workspace
                    / ".helix_artifacts"
                    / "backend_transcripts"
                    / "claude"
                    / "sess_123.jsonl"
                )
                transcript.parent.mkdir(parents=True)
                transcript.write_text('{"message":"saved"}\n')
            else:
                (workspace / "main.py").write_text("new\n")
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"type":"result","session_id":"sess_123"}\n',
            stderr="",
        )

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "main.py").read_text() == "new\n"
    transcript = (
        source
        / ".helix_artifacts"
        / "backend_transcripts"
        / "claude"
        / "sess_123.jsonl"
    )
    assert transcript.read_text() == '{"message":"saved"}\n'
    copy_call = next(
        call
        for call in calls
        if call[:2] == ["docker", "run"] and "sess_123.jsonl" in " ".join(call)
    )
    assert "helix-auth-claude:/home/node:ro" in copy_call


def test_agent_sync_tolerates_inaccessible_backend_transcripts(
    tmp_path: Path, mocker, monkeypatch
):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")

    workspace_root: Path | None = None
    real_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if (
            workspace_root is not None
            and path == workspace_root / ".helix_artifacts" / "backend_transcripts"
        ):
            raise PermissionError("permission denied")
        return real_exists(path)

    def fake_run(args, **kwargs):
        nonlocal workspace_root
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace_root = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace_root / "main.py").write_text("new\n")
            blocked = workspace_root / ".helix_artifacts" / "backend_transcripts"
            blocked.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"type":"result","session_id":"sess_123"}\n',
            stderr="",
        )

    monkeypatch.setattr(Path, "exists", fake_exists)
    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "main.py").read_text() == "new\n"
    assert not (source / ".helix_artifacts" / "backend_transcripts").exists()


def test_agent_sync_recovers_rootless_workspace_permissions(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")

    calls: list[list[str]] = []
    workspace_root: Path | None = None

    def fake_run(args, **kwargs):
        nonlocal workspace_root
        calls.append(args)
        if (
            args[:2] == ["docker", "run"]
            and not _is_workspace_chown(args)
            and not _is_workspace_permission_relax(args)
        ):
            workspace_root = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace_root / "main.py").write_text("new\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert workspace_root is not None
    assert (source / "main.py").read_text() == "new\n"
    relax_calls = [call for call in calls if _is_workspace_permission_relax(call)]
    assert relax_calls
    # The relax helper must run after the agent container exits and before
    # host-side sync-back reads the workspace; verify ordering relative to the
    # primary agent docker run.
    agent_run_idx = next(
        i
        for i, call in enumerate(calls)
        if call[:2] == ["docker", "run"]
        and not _is_workspace_chown(call)
        and not _is_workspace_permission_relax(call)
    )
    assert calls.index(relax_calls[0]) > agent_run_idx


def test_agent_sync_skips_relax_when_host_owner_available(tmp_path: Path, mocker):
    """When ``_host_owner`` returns a value, the chown path runs and the relax
    helper must NOT be invoked (the two recovery branches are mutually
    exclusive)."""
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if (
            args[:2] == ["docker", "run"]
            and not _is_workspace_chown(args)
            and not _is_workspace_permission_relax(args)
        ):
            workspace_root = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace_root / "main.py").write_text("new\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value="1000:1000")

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "main.py").read_text() == "new\n"
    assert not [call for call in calls if _is_workspace_permission_relax(call)]
    assert [call for call in calls if _is_workspace_chown_only(call)]


def test_safe_rmtree_uses_relax_helper_when_host_owner_missing(tmp_path: Path, mocker):
    """``_safe_rmtree`` must invoke the relax helper (not chown) when the host
    owner is unavailable, then retry the rmtree."""
    from helix.sandbox import _safe_rmtree

    target = tmp_path / "doomed"
    target.mkdir()
    (target / "f").write_text("x")

    rmtree_calls: list[Path] = []
    original_rmtree = __import__("shutil").rmtree

    def fake_rmtree(path, *args, **kwargs):
        rmtree_calls.append(Path(path))
        if len(rmtree_calls) == 1:
            raise OSError("permission denied")
        return original_rmtree(path, *args, **kwargs)

    mocker.patch("helix.sandbox.shutil.rmtree", side_effect=fake_rmtree)
    mocker.patch("helix.sandbox._host_owner", return_value=None)

    docker_calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        docker_calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    _safe_rmtree(target, docker_image="helix-test:latest")

    assert any(_is_workspace_permission_relax(call) for call in docker_calls)
    assert not any(_is_workspace_chown_only(call) for call in docker_calls)
    assert len(rmtree_calls) == 2  # initial failure + retry


def test_agent_sync_back_honors_omitted_paths(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")
    (source / "private").mkdir()
    (source / "private" / "token.txt").write_text("host secret\n")

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            assert not (workspace / "private" / "token.txt").exists()
            (workspace / "main.py").write_text("new\n")
            (workspace / "private").mkdir(exist_ok=True)
            (workspace / "private" / "token.txt").write_text("agent secret\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True, omit_from_agent=["private/token.txt"]),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "main.py").read_text() == "new\n"
    assert (source / "private" / "token.txt").read_text() == "host secret\n"


def test_agent_sync_back_does_not_create_omitted_paths(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace / "private").mkdir()
            (workspace / "private" / "token.txt").write_text("agent secret\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True, omit_from_agent=["private"]),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert not (source / "private").exists()


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="mkfifo is unavailable on this platform"
)
def test_agent_sync_skips_special_files_by_default(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            os.mkfifo(workspace / "agent.pipe")
            (workspace / "regular.txt").write_text("ok\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "regular.txt").read_text() == "ok\n"
    assert not (source / "agent.pipe").exists()


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="mkfifo is unavailable on this platform"
)
def test_sync_preserves_existing_host_special_files_when_skipped(
    tmp_path: Path, mocker
):
    source = tmp_path / "candidate"
    source.mkdir()
    os.mkfifo(source / "existing.pipe")

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            assert not (workspace / "existing.pipe").exists()
            (workspace / "regular.txt").write_text("ok\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "existing.pipe").exists()
    assert not (source / "existing.pipe").is_file()
    assert (source / "regular.txt").read_text() == "ok\n"


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="mkfifo is unavailable on this platform"
)
def test_special_file_skip_can_be_disabled(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            os.mkfifo(workspace / "agent.pipe")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    with pytest.raises(OSError):
        run_sandboxed_command(
            ["claude", "-p", "prompt"],
            cwd=source,
            env={},
            sandbox=SandboxConfig(enabled=True, skip_special_files=False),
            scope="agent",
            sync_back=True,
            image="helix-test:latest",
            agent_backend="claude",
        )


def test_evaluator_does_not_sync_changes_back(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")
    (source / "helix_batch.json").write_text('["0"]\n')

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            assert (workspace / "helix_batch.json").read_text() == '["0"]\n'
            (workspace / "main.py").write_text("mutated\n")
            (workspace / "helix_batch.json").write_text('["changed"]\n')
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    cfg = SandboxConfig(enabled=True)

    run_sandboxed_command(
        ["python", "evaluate.py"],
        cwd=source,
        env={},
        sandbox=cfg,
        scope="evaluator",
        sync_back=False,
        image="helix-test:latest",
    )

    assert (source / "main.py").read_text() == "old\n"
    assert (source / "helix_batch.json").read_text() == '["0"]\n'


def test_sandboxed_command_sequence_reuses_workspace(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    seen_workspaces: list[Path] = []

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            seen_workspaces.append(workspace)
            if args[-1] == "write":
                (workspace / "result.txt").write_text("ok\n")
            elif args[-1] == "read":
                assert (workspace / "result.txt").read_text() == "ok\n"
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_commands(
        [["sh", "-c", "write"], ["sh", "-c", "read"]],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="evaluator",
        sync_back=False,
        image="helix-test:latest",
    )

    assert len(seen_workspaces) == 2
    assert seen_workspaces[0] == seen_workspaces[1]
    assert not (source / "result.txt").exists()


def test_sandbox_auth_volume_name_is_backend_specific():
    assert sandbox_auth_volume_name("claude") == "helix-auth-claude"
    assert sandbox_auth_volume_name("cursor") == "helix-auth-cursor"


def test_sandbox_auth_login_command_uses_persistent_volume():
    args = sandbox_auth_docker_args(
        "cursor",
        image="helix-cursor:latest",
        action="login",
        extra_hosts={"local-service": "127.0.0.1"},
        interactive=True,
    )

    assert args[:3] == ["docker", "run", "-it"]
    assert "helix-auth-cursor:/home/node:rw" in args
    assert "helix-cursor:latest" in args
    assert args[-2:] == ["cursor-agent", "login"]
    assert "--add-host local-service:127.0.0.1" in " ".join(args)


def test_sandbox_auth_status_command_uses_backend_command():
    args = sandbox_auth_docker_args(
        "claude",
        image="helix-claude:latest",
        action="status",
    )

    assert "helix-auth-claude:/home/node:rw" in args
    assert args[-3:-1] == ["sh", "-lc"]
    script = args[-1]
    assert script.startswith("set -eu; ")
    assert "claude auth status --text" in script
    # Robust check: requires the on-disk credential file to exist and be
    # non-empty so the exit code is meaningful even when the CLI itself
    # exits 0 for an unauthenticated user.
    assert 'test -s "${HOME:-/home/node}/.claude/.credentials.json"' in script


def test_sandbox_auth_claude_login_uses_claudeai_flow():
    args = sandbox_auth_docker_args(
        "claude",
        image="helix-claude:latest",
        action="login",
        interactive=True,
    )

    assert "helix-auth-claude:/home/node:rw" in args
    assert args[-4:] == ["claude", "auth", "login", "--claudeai"]


def test_sandbox_auth_codex_login_uses_device_auth_flow():
    args = sandbox_auth_docker_args(
        "codex",
        image="helix-codex:latest",
        action="login",
        interactive=True,
    )

    assert "helix-auth-codex:/home/node:rw" in args
    assert args[-3:] == ["codex", "login", "--device-auth"]


def test_sandbox_auth_gemini_login_skips_workspace_trust_prompt():
    args = sandbox_auth_docker_args(
        "gemini",
        image="helix-gemini:latest",
        action="login",
        interactive=True,
    )

    assert "helix-auth-gemini:/home/node:rw" in args
    assert args[-2:] == ["gemini", "--skip-trust"]


def test_sandbox_auth_opencode_login_uses_full_setup_tui():
    args = sandbox_auth_docker_args(
        "opencode",
        image="helix-opencode:latest",
        action="login",
        interactive=True,
    )

    assert "helix-auth-opencode:/home/node:rw" in args
    assert args[-1:] == ["opencode"]
