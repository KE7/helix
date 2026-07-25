"""Unit tests for container-image digest provenance.

Two failure modes are worth more than the feature itself, and both are tested
here rather than assumed:

1. **Recording a wrong-but-plausible identifier.**  A local-only image has no
   registry digest.  Reporting its local image ID under a field a reader will
   take for a pullable ``repo@sha256:...`` would be worse than reporting
   nothing -- it would make an unreproducible run *look* reproducible.  The
   local case must be labelled as local.

2. **Perturbing the run.**  Provenance is metadata.  No missing binary, dead
   daemon, absent image, or malformed output may raise out of this module.
   Every one of those paths is exercised.

These are unit-tier: a fake ``DockerRunner`` is injected everywhere and no test
touches a real Docker daemon.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from helix.config import EvaluatorSidecarConfig, HelixConfig, SandboxConfig
from helix.image_provenance import (
    RUNNER_ROLE,
    SIDECAR_RUNNER_ROLE,
    SIDECAR_SERVICE_ROLE,
    ImageProvenance,
    planned_run_images,
    record_image_provenance,
    resolve_image_provenance,
)

_IMAGE = "ghcr.io/ke7/helix-evo-runner-codex:latest"
_DIGEST = "ghcr.io/ke7/helix-evo-runner-codex@sha256:" + "a" * 64
_IMAGE_ID = "sha256:" + "b" * 64


def _runner(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    raises: BaseException | None = None,
    calls: list[list[str]] | None = None,
):
    """Return a fake ``DockerRunner``.

    Uses the real ``CompletedProcess`` type so an attribute typo in the
    production path (``result.exit_code`` vs ``returncode``) fails the test
    instead of being absorbed by a ``MagicMock``.
    """

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(args))
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(
            args=args, returncode=returncode, stdout=stdout, stderr=stderr
        )

    return run


# ---------------------------------------------------------------------------
# The happy path: a pulled image resolves to its registry digest
# ---------------------------------------------------------------------------


def test_pulled_image_records_the_repo_digest() -> None:
    """The whole point: a tag resolves to the immutable bytes it named."""
    prov = resolve_image_provenance(
        _IMAGE,
        role=RUNNER_ROLE,
        runner=_runner(stdout=f'["{_DIGEST}"]\t{_IMAGE_ID}\n'),
    )
    assert prov.source == "repo_digest"
    assert prov.digest == _DIGEST
    assert prov.image_id == _IMAGE_ID
    assert prov.image == _IMAGE
    assert prov.resolved
    assert prov.reason_code is None


def test_inspect_uses_json_repodigests_not_index_zero() -> None:
    """``{{index .RepoDigests 0}}`` raises on an empty list; ``json`` does not.

    Catches: a "simplification" back to the index form, which would turn the
    ordinary local-build case into a Docker template error and lose the
    fallback entirely.
    """
    calls: list[list[str]] = []
    resolve_image_provenance(
        _IMAGE,
        role=RUNNER_ROLE,
        runner=_runner(stdout=f'["{_DIGEST}"]\t{_IMAGE_ID}', calls=calls),
    )
    assert len(calls) == 1
    assert calls[0][:4] == ["docker", "image", "inspect", "--format"]
    assert "{{json .RepoDigests}}" in calls[0][4]
    assert "index .RepoDigests" not in calls[0][4]


def test_multiple_repo_digests_prefers_the_requested_repository() -> None:
    """A retagged/mirrored image carries several digests; pick ours."""
    other = "mirror.example.com/ke7/helix-evo-runner-codex@sha256:" + "c" * 64
    prov = resolve_image_provenance(
        _IMAGE,
        role=RUNNER_ROLE,
        runner=_runner(stdout=f'["{other}", "{_DIGEST}"]\t{_IMAGE_ID}'),
    )
    assert prov.digest == _DIGEST


def test_registry_port_is_not_mistaken_for_a_tag() -> None:
    """``localhost:5000/foo:v1`` -- the colon before the last ``/`` is a port."""
    image = "localhost:5000/helix-runner:v1"
    digest = "localhost:5000/helix-runner@sha256:" + "d" * 64
    prov = resolve_image_provenance(
        image, role=RUNNER_ROLE, runner=_runner(stdout=f'["{digest}"]\t{_IMAGE_ID}')
    )
    assert prov.digest == digest


# ---------------------------------------------------------------------------
# Local-only images: no RepoDigests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("repo_digests", ["[]", "null"])
def test_local_only_image_falls_back_to_image_id_and_says_so(
    repo_digests: str,
) -> None:
    """``docker build -t helix-runner-codex:latest .`` was never pushed.

    THE DISTINCTION THIS FEATURE TURNS ON: the fallback identifier must not be
    presented as a registry digest.  ``digest`` stays None so no reader can
    mistake a host-local ID for something pullable, and the reason states why.

    Docker renders a nil ``RepoDigests`` as JSON ``null`` and an empty one as
    ``[]``; both are the same case.
    """
    prov = resolve_image_provenance(
        "helix-runner-codex:latest",
        role=RUNNER_ROLE,
        runner=_runner(stdout=f"{repo_digests}\t{_IMAGE_ID}"),
    )
    assert prov.source == "local_image_id"
    assert prov.digest is None
    assert prov.image_id == _IMAGE_ID
    assert prov.resolved
    assert "never pushed" in (prov.reason or "")


# ---------------------------------------------------------------------------
# Failure paths: none of these may raise
# ---------------------------------------------------------------------------


def test_missing_docker_binary_is_recorded_not_raised() -> None:
    """No Docker on PATH must not perturb the run at all."""
    prov = resolve_image_provenance(
        _IMAGE,
        role=RUNNER_ROLE,
        runner=_runner(raises=FileNotFoundError("docker")),
    )
    assert prov.source == "unavailable"
    assert prov.reason_code == "docker_unavailable"
    assert "PATH" in (prov.reason or "")
    assert not prov.retryable


def test_image_absent_locally_is_recorded_as_retryable() -> None:
    """Docker pulls on first ``docker run``, so absent-at-start is normal.

    Catches: marking the pull-on-first-use case permanent, which would leave
    every ``auth = "env"`` run with no digest even though the image is present
    by the time the run ends.
    """
    prov = resolve_image_provenance(
        _IMAGE,
        role=RUNNER_ROLE,
        runner=_runner(returncode=1, stderr=f"Error: No such image: {_IMAGE}"),
    )
    assert prov.source == "unavailable"
    assert prov.reason_code == "image_not_found"
    assert prov.retryable


def test_daemon_error_is_recorded_and_not_retried() -> None:
    prov = resolve_image_provenance(
        _IMAGE,
        role=RUNNER_ROLE,
        runner=_runner(returncode=1, stderr="Cannot connect to the Docker daemon"),
    )
    assert prov.source == "unavailable"
    assert prov.reason_code == "docker_error"
    assert "Cannot connect" in (prov.reason or "")
    assert not prov.retryable


def test_timeout_is_recorded_not_raised() -> None:
    """A wedged daemon costs seconds of startup, never the run."""
    prov = resolve_image_provenance(
        _IMAGE,
        role=RUNNER_ROLE,
        runner=_runner(raises=subprocess.TimeoutExpired(cmd="docker", timeout=15.0)),
    )
    assert prov.source == "unavailable"
    assert prov.reason_code == "inspect_timeout"


def test_unexpected_exception_is_recorded_not_raised() -> None:
    """The broad catch is deliberate; this pins it.

    Catches: someone narrowing the ``except Exception`` to a tidy tuple and
    reintroducing a path where metadata collection can abort an evolution run.
    """
    prov = resolve_image_provenance(
        _IMAGE,
        role=RUNNER_ROLE,
        runner=_runner(raises=RuntimeError("daemon exploded")),
    )
    assert prov.source == "unavailable"
    assert prov.reason_code == "docker_error"
    assert "RuntimeError" in (prov.reason or "")


@pytest.mark.parametrize("stdout", ["", "not json\tsha256:x", "{{json}}"])
def test_unparseable_output_is_recorded_not_raised(stdout: str) -> None:
    prov = resolve_image_provenance(
        _IMAGE, role=RUNNER_ROLE, runner=_runner(stdout=stdout)
    )
    assert prov.source == "unavailable"
    assert prov.reason_code == "unparseable_output"


def test_reason_text_is_bounded() -> None:
    """Failure detail reaches a persisted artifact; it does not get to be huge."""
    prov = resolve_image_provenance(
        _IMAGE, role=RUNNER_ROLE, runner=_runner(returncode=1, stderr="x" * 5000)
    )
    assert prov.reason is not None
    assert len(prov.reason) <= 240


# ---------------------------------------------------------------------------
# Which images a run actually uses
# ---------------------------------------------------------------------------


def _sidecar_config() -> HelixConfig:
    return HelixConfig(
        objective="Test",
        evaluator={
            "command": "python /runner/evaluate.py",
            "score_parser": "helix_result",
            "sidecar": EvaluatorSidecarConfig(
                image="eval:latest",
                runner_image="eval-runner:latest",
                command="python -m server",
                endpoint="http://helix-evaluator:8080/evaluate",
            ),
        },
        sandbox=SandboxConfig(enabled=True, evaluator=True),
    )


def test_unsandboxed_run_has_no_images_to_attribute() -> None:
    """No container, no image, no claim about one."""
    config = HelixConfig(objective="Test", evaluator={"command": "pytest"})
    assert planned_run_images(config) == []


def test_sandboxed_run_covers_runner_and_both_sidecar_images() -> None:
    """Requirement 2: the mutation runner AND the evaluator sidecar.

    Catches: attributing only the mutation runner and leaving evaluation --
    the half of the run that produces the scores -- unattributed.
    """
    roles = dict(planned_run_images(_sidecar_config()))
    assert roles[RUNNER_ROLE].startswith("ghcr.io/ke7/helix-evo-runner-")
    assert roles[SIDECAR_SERVICE_ROLE] == "eval:latest"
    assert roles[SIDECAR_RUNNER_ROLE] == "eval-runner:latest"


def test_runner_only_when_sidecar_evaluation_is_off() -> None:
    config = HelixConfig(
        objective="Test",
        evaluator={"command": "pytest"},
        sandbox=SandboxConfig(enabled=True, evaluator=False),
    )
    assert [role for role, _ in planned_run_images(config)] == [RUNNER_ROLE]


# ---------------------------------------------------------------------------
# Run-level recording into state
# ---------------------------------------------------------------------------


def test_record_populates_every_role_and_reports_change() -> None:
    provenance: dict[str, Any] = {}
    changed = record_image_provenance(
        provenance,
        _sidecar_config(),
        runner=_runner(stdout=f'["{_DIGEST}"]\t{_IMAGE_ID}'),
    )
    assert changed
    assert set(provenance) == {RUNNER_ROLE, SIDECAR_SERVICE_ROLE, SIDECAR_RUNNER_ROLE}
    assert provenance[RUNNER_ROLE]["digest"] == _DIGEST
    assert provenance[SIDECAR_RUNNER_ROLE]["source"] == "repo_digest"


def test_record_is_json_serializable() -> None:
    """It lands in state.json, which is JSON -- not a pickle."""
    import json

    provenance: dict[str, Any] = {}
    record_image_provenance(
        provenance,
        _sidecar_config(),
        runner=_runner(stdout=f'["{_DIGEST}"]\t{_IMAGE_ID}'),
    )
    assert json.loads(json.dumps(provenance)) == provenance


def test_second_pass_resolves_a_role_that_was_absent_at_run_start() -> None:
    """``auth = "env"`` pulls nothing before the first proposal.

    The run-start lookup legitimately finds no image; the end-of-run pass must
    fill it in once Docker has pulled it.
    """
    config = HelixConfig(
        objective="Test",
        evaluator={"command": "pytest"},
        sandbox=SandboxConfig(enabled=True, evaluator=False),
    )
    provenance: dict[str, Any] = {}
    record_image_provenance(
        provenance, config, runner=_runner(returncode=1, stderr="No such image: x")
    )
    assert provenance[RUNNER_ROLE]["source"] == "unavailable"

    changed = record_image_provenance(
        provenance, config, runner=_runner(stdout=f'["{_DIGEST}"]\t{_IMAGE_ID}')
    )
    assert changed
    assert provenance[RUNNER_ROLE]["digest"] == _DIGEST


def test_permanent_failure_is_not_re_probed() -> None:
    """One failed lookup per run is enough for a non-retryable reason.

    Catches: burning a subprocess per pass on a host that has no Docker
    binary at all.
    """
    config = HelixConfig(
        objective="Test",
        evaluator={"command": "pytest"},
        sandbox=SandboxConfig(enabled=True, evaluator=False),
    )
    provenance: dict[str, Any] = {}
    record_image_provenance(
        provenance, config, runner=_runner(raises=FileNotFoundError("docker"))
    )

    calls: list[list[str]] = []
    changed = record_image_provenance(
        provenance,
        config,
        runner=_runner(stdout=f'["{_DIGEST}"]\t{_IMAGE_ID}', calls=calls),
    )
    assert calls == []
    assert not changed
    assert provenance[RUNNER_ROLE]["reason_code"] == "docker_unavailable"


def test_a_moved_tag_is_recorded_as_drift_not_an_overwrite() -> None:
    """The ``:latest`` failure mode, made visible.

    A run resumed after the tag moved genuinely executed two different images.
    Overwriting the first digest would erase the evidence and leave the
    artifact confidently wrong about which bytes produced generations 0..n.
    """
    config = HelixConfig(
        objective="Test",
        evaluator={"command": "pytest"},
        sandbox=SandboxConfig(enabled=True, evaluator=False),
    )
    provenance: dict[str, Any] = {}
    record_image_provenance(
        provenance, config, runner=_runner(stdout=f'["{_DIGEST}"]\t{_IMAGE_ID}')
    )
    moved = "ghcr.io/ke7/helix-evo-runner-codex@sha256:" + "e" * 64

    changed = record_image_provenance(
        provenance, config, runner=_runner(stdout=f'["{moved}"]\t{_IMAGE_ID}')
    )
    assert changed
    assert provenance[RUNNER_ROLE]["digest"] == _DIGEST, "original must survive"
    assert provenance[RUNNER_ROLE]["drift"][0]["digest"] == moved


def test_record_never_raises_even_when_the_runner_is_hostile() -> None:
    """Requirement 3 of the brief, enforced at the run-facing entry point."""
    config = _sidecar_config()
    provenance: dict[str, Any] = {}

    def exploding(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("no fds")

    assert record_image_provenance(provenance, config, runner=exploding) is True
    assert all(v["source"] == "unavailable" for v in provenance.values())


def test_from_dict_tolerates_garbage() -> None:
    """A hand-edited or newer-HELIX state.json must not crash a resume."""
    prov = ImageProvenance.from_dict({"source": "wishful", "digest": 17})
    assert prov.source == "unavailable"
    assert prov.digest is None
