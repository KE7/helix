"""Container-image provenance for a HELIX run.

HELIX resolves runner images by **tag** (``resolve_sandbox_image`` returns
``ghcr.io/ke7/helix-evo-runner-<backend>:latest`` by default, from
``backends.DEFAULT_BACKEND_IMAGES``).  A tag is a mutable pointer: two runs a
week apart can both say ``:latest`` in every artifact they persist and still
have executed different bytes.  Nothing in ``state.py`` / ``lineage.py`` /
``executor.py`` recorded what the tag actually resolved to, so a finished run
could not answer "which image ran this?" -- only "whatever ``:latest`` pointed
at, at some unrecorded moment".

The immutable ``sha256:`` OCI digest is the only thing that names the bytes.
This module resolves it once per run and hands it to ``EvolutionState`` for
persistence.

Three properties are load-bearing:

**It never raises.**  Provenance is metadata.  A missing ``docker`` binary, a
wedged daemon, an image that was never pulled -- none of these may perturb,
slow, or abort a run.  Every failure path returns an ``ImageProvenance`` whose
``source`` is ``"unavailable"`` and whose ``reason`` says why, so the artifact
records the *absence* explicitly instead of leaving a hole a reader has to
guess about.

**It runs off the hot path.**  ``mutator.py`` resolves the sandbox image once
per proposal; shelling out to Docker there would put an ``exec`` on every
mutation.  Resolution happens once per run in ``evolution._run_evolution_impl``
and is memoized into state.

**Local-only images degrade honestly.**  ``RepoDigests`` is empty for an image
built locally and never pushed or pulled -- the ordinary ``docker build -t
helix-runner-codex:latest .`` dev loop.  Such an image has no registry digest
because no registry has ever seen it.  We fall back to the local image ID
(``.Id``, a content hash of the image config) and label it ``local_image_id``
so the record is never mistaken for a pullable digest.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from helix.config import HelixConfig
from helix.sandbox import DockerRunner, production_docker_runner, resolve_sandbox_image

logger = logging.getLogger(__name__)


ProvenanceSource = Literal["repo_digest", "local_image_id", "unavailable"]

# Role keys under ``EvolutionState.image_provenance``.  A run uses at most
# three distinct images, and every image-resolution call site in the run
# resolves to one of them:
#   * RUNNER          -- sandbox.resolve_sandbox_image (mutator.py:1669,
#                        authpreflight.py:290)
#   * SIDECAR_SERVICE -- evaluator.sidecar.image (sandbox.py, sidecar container)
#   * SIDECAR_RUNNER  -- evaluator.sidecar.resolved_runner_image
#                        (executor.py:247, sandbox.py:996 healthcheck)
RUNNER_ROLE = "runner"
SIDECAR_SERVICE_ROLE = "sidecar_service"
SIDECAR_RUNNER_ROLE = "sidecar_runner"

# ``{{index .RepoDigests 0}}`` is the obvious formulation and is wrong: Go
# templates raise "index out of range" when the list is empty, which is exactly
# the local-build case this module has to handle.  ``{{json .RepoDigests}}``
# renders ``[]`` instead, so the empty case parses like any other.
_INSPECT_FORMAT = "{{json .RepoDigests}}\t{{.Id}}"
# A local metadata read against the daemon.  Bounded so a wedged daemon costs
# a few seconds of run startup rather than the run.
_INSPECT_TIMEOUT_SECONDS = 15.0
# Failure detail is bounded before it reaches a persisted artifact.
_MAX_REASON_CHARS = 240
# Cap on retained drift entries per role, so a run resumed many times across a
# moving tag cannot grow state.json without bound.
_MAX_DRIFT_ENTRIES = 8

# ``reason_code`` values.  ``image_not_found`` is the only retryable one: it
# means the image was not present locally *yet*, which is expected when Docker
# pulls on first ``docker run``.
_REASON_IMAGE_NOT_FOUND = "image_not_found"
_REASON_DOCKER_UNAVAILABLE = "docker_unavailable"
_REASON_DOCKER_ERROR = "docker_error"
_REASON_TIMEOUT = "inspect_timeout"
_REASON_UNPARSEABLE = "unparseable_output"


@dataclass(frozen=True)
class ImageProvenance:
    """What a single image reference resolved to at a point in time.

    ``digest`` is a fully-qualified ``repo@sha256:...`` reference and is set
    only when ``source == "repo_digest"`` -- i.e. only when it is a real
    registry digest that could be pulled again.  ``image_id`` is the local
    ``sha256:...`` image config hash, recorded whenever Docker reports it, and
    is the *only* identifier available for a local-only image.
    """

    role: str
    image: str
    digest: str | None = None
    image_id: str | None = None
    source: ProvenanceSource = "unavailable"
    reason: str | None = None
    reason_code: str | None = None

    @property
    def resolved(self) -> bool:
        """True when Docker answered with some identifier for the image."""
        return self.source != "unavailable"

    @property
    def retryable(self) -> bool:
        """True when a later re-resolution could plausibly succeed.

        Only the "image not present locally" case: Docker pulls on first
        ``docker run``, so an image that is absent at run start is routinely
        present by the end of the run.  A missing ``docker`` binary or a
        daemon error is not retried -- one failed lookup per run is enough.
        """
        return self.reason_code == _REASON_IMAGE_NOT_FOUND

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "image": self.image,
            "digest": self.digest,
            "image_id": self.image_id,
            "source": self.source,
            "reason": self.reason,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ImageProvenance:
        """Rebuild a record from persisted state, tolerating unknown shapes.

        Reads defensively: a state.json written by a newer HELIX, or one an
        operator hand-edited, must not crash a resume over metadata.
        """
        raw_source = data.get("source")
        source: ProvenanceSource = (
            raw_source
            if raw_source in ("repo_digest", "local_image_id", "unavailable")
            else "unavailable"
        )
        return cls(
            role=str(data.get("role", "")),
            image=str(data.get("image", "")),
            digest=_optional_str(data.get("digest")),
            image_id=_optional_str(data.get("image_id")),
            source=source,
            reason=_optional_str(data.get("reason")),
            reason_code=_optional_str(data.get("reason_code")),
        )


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _truncate(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _MAX_REASON_CHARS:
        return collapsed
    return collapsed[: _MAX_REASON_CHARS - 1] + "…"


def _unavailable(
    role: str, image: str, reason_code: str, reason: str
) -> ImageProvenance:
    return ImageProvenance(
        role=role,
        image=image,
        source="unavailable",
        reason=_truncate(reason),
        reason_code=reason_code,
    )


def _image_repository(image: str) -> str:
    """Return the repository part of an image reference.

    ``ghcr.io/ke7/helix-evo-runner-codex:latest`` -> ``ghcr.io/ke7/helix-evo-runner-codex``
    ``localhost:5000/foo:v1``                     -> ``localhost:5000/foo``
    ``repo@sha256:abc``                           -> ``repo``

    The tag separator is only a tag separator when it appears after the last
    ``/`` -- otherwise it is a registry port.
    """
    ref = image.split("@", 1)[0]
    last_slash = ref.rfind("/")
    colon = ref.rfind(":")
    if colon > last_slash:
        return ref[:colon]
    return ref


def _select_repo_digest(image: str, repo_digests: list[str]) -> str:
    """Pick the digest matching *image*'s repository.

    An image can carry several ``RepoDigests`` when it has been pushed to or
    pulled from more than one repository (mirrors, retags).  Prefer the one
    naming the repository HELIX actually asked for; otherwise fall back to a
    deterministic choice so two readers never disagree.
    """
    repository = _image_repository(image)
    matching = [d for d in repo_digests if _image_repository(d) == repository]
    if matching:
        return sorted(matching)[0]
    return sorted(repo_digests)[0]


def resolve_image_provenance(
    image: str,
    *,
    role: str,
    runner: DockerRunner | None = None,
) -> ImageProvenance:
    """Resolve *image* to a digest via ``docker image inspect``.

    Never raises.  Every failure -- no ``docker`` on PATH, daemon error,
    timeout, image absent, output HELIX cannot parse -- becomes an
    ``unavailable`` record carrying the reason.
    """
    run = runner if runner is not None else production_docker_runner()
    args = ["docker", "image", "inspect", "--format", _INSPECT_FORMAT, image]
    try:
        result = run(args, check=False, timeout=_INSPECT_TIMEOUT_SECONDS)
    except FileNotFoundError:
        return _unavailable(
            role,
            image,
            _REASON_DOCKER_UNAVAILABLE,
            "docker executable not found on PATH",
        )
    except subprocess.TimeoutExpired:
        return _unavailable(
            role,
            image,
            _REASON_TIMEOUT,
            f"docker image inspect timed out after {_INSPECT_TIMEOUT_SECONDS:g}s",
        )
    except Exception as exc:  # noqa: BLE001 - provenance must never abort a run
        # Deliberately broad.  This function's contract to its callers is that
        # it cannot be the reason a run dies, and an unforeseen Docker/OS error
        # is exactly the case that contract exists for.
        logger.debug("image provenance lookup raised for %s", image, exc_info=True)
        return _unavailable(
            role, image, _REASON_DOCKER_ERROR, f"{type(exc).__name__}: {exc}"
        )

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if "no such image" in stderr.lower() or "no such object" in stderr.lower():
            return _unavailable(
                role,
                image,
                _REASON_IMAGE_NOT_FOUND,
                f"image not present locally: {image}",
            )
        return _unavailable(
            role,
            image,
            _REASON_DOCKER_ERROR,
            f"docker image inspect exited {result.returncode}: {stderr}",
        )

    return _parse_inspect_output(role, image, result.stdout or "")


def _parse_inspect_output(role: str, image: str, stdout: str) -> ImageProvenance:
    raw = stdout.strip()
    repo_digests_field, _, image_id_field = raw.partition("\t")
    image_id = _optional_str(image_id_field.strip())
    try:
        decoded = json.loads(repo_digests_field.strip() or "[]")
    except json.JSONDecodeError:
        return _unavailable(
            role,
            image,
            _REASON_UNPARSEABLE,
            f"could not parse docker image inspect output: {raw!r}",
        )
    # ``{{json .RepoDigests}}`` renders JSON ``null`` when the field is nil,
    # which decodes to None rather than an empty list.
    repo_digests = [d for d in (decoded or []) if isinstance(d, str) and d]

    if repo_digests:
        return ImageProvenance(
            role=role,
            image=image,
            digest=_select_repo_digest(image, repo_digests),
            image_id=image_id,
            source="repo_digest",
        )
    if image_id:
        return ImageProvenance(
            role=role,
            image=image,
            digest=None,
            image_id=image_id,
            source="local_image_id",
            reason=_truncate(
                "image has no RepoDigests (built locally and never pushed or "
                "pulled); recorded the local image ID instead, which names the "
                "bytes on this host only"
            ),
        )
    return _unavailable(
        role,
        image,
        _REASON_UNPARSEABLE,
        f"docker image inspect returned neither RepoDigests nor Id: {raw!r}",
    )


def planned_run_images(config: HelixConfig) -> list[tuple[str, str]]:
    """Return ``(role, image)`` for every container image this run will use.

    Empty when the run is unsandboxed -- no container image is involved, so
    there is nothing to attribute.  Never raises: an image reference HELIX
    cannot even resolve to a tag is simply omitted (the run itself will fail
    on it later, with a better error than this module could produce).
    """
    images: list[tuple[str, str]] = []
    if not config.sandbox.enabled:
        return images

    try:
        images.append(
            (RUNNER_ROLE, resolve_sandbox_image(config.sandbox, config.agent.backend))
        )
    except ValueError:
        logger.debug("could not resolve runner image for provenance", exc_info=True)

    sidecar = config.evaluator.sidecar
    if config.sandbox.evaluator and sidecar is not None:
        images.append((SIDECAR_SERVICE_ROLE, sidecar.image))
        images.append((SIDECAR_RUNNER_ROLE, sidecar.resolved_runner_image))
    return images


def _record_drift(
    stored: dict[str, Any], previous: ImageProvenance, current: ImageProvenance
) -> None:
    """Note that a role's tag now resolves to different bytes than it did.

    This is the ``:latest`` failure mode made visible: a run resumed after the
    tag moved genuinely executed two different images, and silently
    overwriting the first digest would erase that.
    """
    drift = stored.get("drift")
    entries: list[Any] = list(drift) if isinstance(drift, list) else []
    if len(entries) >= _MAX_DRIFT_ENTRIES:
        return
    entries.append(current.to_dict())
    stored["drift"] = entries
    logger.warning(
        "image %s (role=%s) now resolves to %s, but this run recorded %s",
        current.image,
        current.role,
        current.digest or current.image_id,
        previous.digest or previous.image_id,
    )


def record_image_provenance(
    provenance: dict[str, Any],
    config: HelixConfig,
    *,
    runner: DockerRunner | None = None,
) -> bool:
    """Fill *provenance* in place with this run's image digests.

    Idempotent, and safe to call more than once per run.  Called at run start
    and once more before the run returns:

    * A role with no record yet is resolved.
    * A role whose record is ``unavailable`` **and** retryable (the image had
      not been pulled yet) is re-resolved.  This is what makes the second call
      worthwhile: with ``auth = "env"`` nothing pulls the runner image before
      the first proposal, so the run-start lookup legitimately finds nothing.
    * A role that already resolved keeps its original record -- that is the
      image the run started on -- and any later, *different* digest is
      appended under ``drift`` rather than overwriting it.

    Returns True when *provenance* changed, so callers can skip a needless
    checkpoint write.  Never raises.
    """
    changed = False
    try:
        for role, image in planned_run_images(config):
            stored_raw = provenance.get(role)
            stored = stored_raw if isinstance(stored_raw, dict) else None
            existing = ImageProvenance.from_dict(stored) if stored else None

            if existing is not None and existing.resolved and existing.image == image:
                # Already attributed.  Re-resolve only to detect drift, and
                # only when we would otherwise have done nothing at all.
                current = resolve_image_provenance(image, role=role, runner=runner)
                if (
                    current.resolved
                    and (current.digest, current.image_id)
                    != (existing.digest, existing.image_id)
                    and stored is not None
                ):
                    _record_drift(stored, existing, current)
                    changed = True
                continue

            if (
                existing is not None
                and not existing.resolved
                and not existing.retryable
            ):
                # Already looked up once this run and failed for a reason that
                # will not change (no docker binary, daemon error).  Do not
                # spend another subprocess on it.
                continue

            resolved = resolve_image_provenance(image, role=role, runner=runner)
            if existing is not None and existing.to_dict() == resolved.to_dict():
                continue
            provenance[role] = resolved.to_dict()
            changed = True
    except Exception:  # noqa: BLE001 - provenance must never abort a run
        # The loop body is already total; this is the outer guarantee that a
        # defect *here* cannot take down a run that is otherwise fine.
        logger.debug("recording image provenance failed", exc_info=True)
    return changed
