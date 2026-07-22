"""Non-vacuity and completeness proofs for the Docker / shared-credential guard.

A guard nobody has watched fire is a guard nobody knows works, so these tests
make it trigger rather than merely asserting the suite is green with it
installed.

Context, stated once and not softened: wiring the auth preflight into
``run_evolution`` made a real authenticated probe against the shared
``helix-auth-claude`` volume reachable from ``pytest tests/unit/``.
Classification — prohibition BREACHED; credential record UNCHANGED;
non-credential shared-volume state MUTATED; OAuth POST CANNOT BE RULED OUT.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "helix"


# ---------------------------------------------------------------------------
# Gate 3 — the guard fires (deny direction)
# ---------------------------------------------------------------------------


def test_guard_blocks_real_docker_invocation():
    """Default-deny fires on a real docker command.

    Non-vacuity: without the guard this test fails, because ``docker version``
    would simply run and raise nothing.
    """
    with pytest.raises(Exception) as exc:
        subprocess.run(["docker", "version"], capture_output=True)
    assert "attempted to run Docker for real" in str(exc.value)


def test_guard_hard_denies_shared_auth_volume_even_when_opted_in(monkeypatch):
    """The ``helix-auth-*`` denial has NO override.

    Even with the integration opt-in environment variable set, and even for a
    command that does not look like docker, naming a shared credential volume
    is refused. There is no legitimate test reason to touch shared credential
    state: a refresh against it rotates the token for every lane.
    """
    monkeypatch.setenv("HELIX_ALLOW_DOCKER_TESTS", "1")
    with pytest.raises(Exception) as exc:
        subprocess.run(
            ["some-wrapper", "run", "-v", "helix-auth-claude:/home/node:rw", "img"],
            capture_output=True,
        )
    message = str(exc.value)
    assert "SHARED auth volume" in message
    assert "NO override" in message
    assert "DISPOSABLE" in message


@pytest.mark.parametrize(
    "volume",
    ["helix-auth-claude", "helix-auth-gemini", "helix-auth-opencode"],
)
def test_guard_denies_every_shared_auth_volume(volume):
    """The denial is by prefix, not by a hard-coded single volume name."""
    with pytest.raises(Exception) as exc:
        subprocess.run(["docker", "run", "-v", f"{volume}:/home/node:rw", "i"])
    assert volume in str(exc.value)


# ---------------------------------------------------------------------------
# Gate 3 — the guard permits safe work (allow direction)
# ---------------------------------------------------------------------------


def test_guard_allows_non_docker_subprocesses():
    """Ordinary subprocess use still works.

    Without this, the guard could be 'proved' by a version that blocks
    everything, which would mask real failures rather than prevent harm.
    """
    result = subprocess.run(["echo", "ok"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


@pytest.mark.docker_integration
def test_disposable_volume_control_is_permitted_when_opted_in(monkeypatch):
    """A DISPOSABLE volume name passes the guard's checks when opted in.

    This is the positive control for the allow direction: it proves the guard
    discriminates between shared credential volumes and disposable ones,
    rather than refusing everything. It deliberately does NOT start a
    container — it asserts the guard's own predicates, so the control is safe
    to run in ordinary CI.
    """
    from tests.conftest import _check_command, _named_shared_auth_volume

    disposable = ["docker", "run", "-v", "helix-testvol-synthetic:/home/node:rw", "i"]
    assert _named_shared_auth_volume(disposable) is None
    # Permitted when the caller has opted in...
    _check_command(disposable, docker_allowed=True)
    # ...and still denied by default.
    with pytest.raises(Exception):
        _check_command(disposable, docker_allowed=False)

    # A shared volume is refused in BOTH cases — no override.
    shared = ["docker", "run", "-v", "helix-auth-claude:/home/node:rw", "i"]
    for allowed in (True, False):
        with pytest.raises(Exception):
            _check_command(shared, docker_allowed=allowed)


# ---------------------------------------------------------------------------
# Gate 4 — the production preflight is unreachable from non-production paths
# ---------------------------------------------------------------------------


def _sandboxed_config():
    from helix.config import AgentConfig, EvaluatorConfig, HelixConfig, SandboxConfig

    return HelixConfig(
        objective="x",
        evaluator=EvaluatorConfig(command="true", score_parser="helix_result"),
        agent=AgentConfig(backend="claude"),
        sandbox=SandboxConfig(enabled=True, image="pinned@sha256:deadbeef"),
    )


def test_preflight_requires_an_injected_runner():
    """The Docker runner is a REQUIRED dependency, not a defaulted one.

    'Mocked in a fixture' and 'cannot reach Docker by construction' are
    different guarantees. This pins the second: a caller that omits the
    dependency gets a TypeError, so a non-production path cannot acquire the
    ability to run a real authenticated probe by omission.
    """
    from helix.authpreflight import preflight_auth, reset_preflight_cache

    reset_preflight_cache()
    with pytest.raises(TypeError):
        preflight_auth(_sandboxed_config())  # type: ignore[call-arg]


def test_preflight_with_the_production_runner_is_stopped_by_the_guard():
    """Even with the production runner, a test path cannot reach a container."""
    from helix.authpreflight import preflight_auth, reset_preflight_cache
    from helix.sandbox import production_docker_runner

    reset_preflight_cache()
    with pytest.raises(Exception) as exc:
        preflight_auth(_sandboxed_config(), runner=production_docker_runner())
    assert "helix-auth-claude" in str(exc.value)
    reset_preflight_cache()


def test_run_evolution_unit_path_cannot_reach_docker(tmp_path, mocker):
    """Gate 4, as a test rather than as prose.

    ``run_evolution`` with a sandboxed config must not reach Docker on the
    unit path. This is the exact shape of the original breach.
    """
    from helix.config import (
        AgentConfig,
        EvaluatorConfig,
        EvolutionConfig,
        HelixConfig,
        SandboxConfig,
    )
    from helix.evolution import run_evolution

    for target in (
        "helix.evolution.create_seed_worktree",
        "helix.evolution.run_evaluator",
        "helix.evolution.init_base_dir",
        "helix.evolution.save_state",
        "helix.evolution.record_entry",
        "helix.evolution.HelixLiveDisplay",
    ):
        mocker.patch(target)
    mocker.patch("helix.evolution.load_state", return_value=None)
    mocker.patch("helix.evolution.load_lineage", return_value={})

    config = HelixConfig(
        objective="x",
        evaluator=EvaluatorConfig(command="true", score_parser="helix_result"),
        agent=AgentConfig(backend="claude"),
        sandbox=SandboxConfig(enabled=True, image="pinned@sha256:deadbeef"),
        evolution=EvolutionConfig(max_generations=0, max_evaluations=1),
    )

    # NOT mocking preflight_auth here: the point is that the guard stops it
    # before a container starts, so the safety property does not depend on
    # every future test remembering to mock the right symbol.
    with pytest.raises(Exception) as exc:
        run_evolution(config, tmp_path, tmp_path / ".helix")
    assert "helix-auth-claude" in str(exc.value)


# ---------------------------------------------------------------------------
# Addition 1 — alias completeness is ENFORCED, not incidental
# ---------------------------------------------------------------------------


# The guard is complete because of what src/helix uses today. If someone later
# introduces os.system, os.popen, or the docker SDK, the guard would silently
# stop covering the codebase and nothing would fail. This test converts a
# true-today property into one that cannot silently stop being true — the same
# move applied to the no-copy probe rule.
_ALLOWED_EXEC_CALLS = {"subprocess.run"}

_FORBIDDEN_EXEC_CALLS = {
    "os.system",
    "os.popen",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.posix_spawn",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
}


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    if isinstance(node, ast.Name):
        return node.id
    return None


def _exec_call_sites() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted(node.func)
            if name in _FORBIDDEN_EXEC_CALLS or name in _ALLOWED_EXEC_CALLS:
                found.setdefault(name, []).append(
                    f"{path.relative_to(SRC)}:{node.lineno}"
                )
    return found


def test_src_uses_only_guarded_exec_entrypoints():
    """src/helix must not introduce an exec entrypoint the guard misses.

    Non-vacuity: the assertion below also requires that the ALLOWED
    entrypoint is actually present. If the scan silently found nothing (a
    broken parser, a wrong path), the test fails rather than passing
    trivially.
    """
    sites = _exec_call_sites()
    assert "subprocess.run" in sites, (
        "scan found no subprocess.run call sites in src/helix — the scanner "
        "is broken, so this test would otherwise pass vacuously"
    )
    offenders = {
        name: locations
        for name, locations in sites.items()
        if name in _FORBIDDEN_EXEC_CALLS
    }
    assert not offenders, (
        "src/helix introduced exec entrypoint(s) the Docker safety guard does "
        f"not necessarily cover: {offenders}. Either use subprocess.run, or "
        "extend the guard in tests/conftest.py AND add the new alias to "
        "_FORBIDDEN_EXEC_CALLS here."
    )


def test_docker_sdk_is_not_a_dependency():
    """No docker SDK usage in src/helix.

    The guard patches the SDK defensively if it is installed, but an import
    here would mean container creation could bypass the subprocess aliases
    entirely.
    """
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "docker":
                        offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "docker":
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"docker SDK imported in src/helix: {offenders}"


def test_synthetic_refresh_harness_refuses_shared_auth_volumes():
    """The synthetic-harness refusal rule, checked on EVERY PR.

    The T22-T25 refresh tests are integration-tier and skip by default, so
    their own safety rule would otherwise go unexercised in normal CI. This
    imports that rule and asserts it directly, at unit tier.

    The rule matters because a typo in a volume name does not fail loudly:
    ``docker run -v`` silently CREATES the volume it names, so a mistyped
    disposable name would provision a new volume, and a mistyped shared name
    would reach real credential state.
    """
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "integration"
        / "test_oauth_refresh_suppression.py"
    )
    spec = importlib.util.spec_from_file_location("_refresh_harness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for shared in ("helix-auth-claude", "helix-auth-opencode", "helix-auth-gemini"):
        with pytest.raises(AssertionError):
            module._assert_disposable(shared)
    # Positive counterpart: a genuinely disposable name is accepted, so the
    # rule discriminates rather than refusing everything.
    module._assert_disposable("helix-refreshtest-abc123")
