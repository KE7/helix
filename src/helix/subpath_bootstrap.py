"""Docker capability preflight and auth-subpath bootstrap for volume mode.

Volume mode mounts the persistent auth store with ``volume-subpath``, which
narrows the shared surface to the backend's auth directory instead of the whole
container HOME.  That mechanism carries two operational hazards, and both
present as *opaque daemon errors* rather than as the fixable problems they are.

1. **Engine floor.** ``volume-subpath`` requires Docker Engine 25.0+ / API 1.45+
   (developed against 29.6.1 / API 1.55).  On an older host the design cannot
   run **at all**.  A hard requirement bump is acceptable; surfacing it as a
   mystery failure is not.

2. **Bootstrap ordering.** ``volume-subpath`` requires the subpath to ALREADY
   EXIST.  Against a fresh or never-authenticated volume the daemon refuses with
   ``cannot access path …: no such file or directory`` and the container never
   starts -- so a plain "you are not logged in" problem presents as an internal
   Docker error at agent-dispatch time.

   ``helix sandbox login`` must therefore CREATE the subpath, and the login path
   itself cannot use a subpath mount (it is the thing establishing it).

   The distinction against the standing rule is deliberate and narrow:
   **creating the SUBPATH on login is legitimate; creating the VOLUME on status
   is not.** ``status`` must remain observation-only, because a status check
   that provisions storage makes "is this volume present" true by construction.
"""

from __future__ import annotations

import re

from helix.exceptions import HelixError


MIN_ENGINE_MAJOR = 25
MIN_API_VERSION = (1, 45)


class DockerCapabilityError(HelixError):
    """The Docker daemon cannot support volume-subpath isolation."""


def _parse_api_version(raw: str) -> tuple[int, int] | None:
    match = re.match(r"^\s*(\d+)\.(\d+)", raw or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_engine_major(raw: str) -> int | None:
    match = re.match(r"^\s*v?(\d+)\.", raw or "")
    return int(match.group(1)) if match else None


def supports_volume_subpath(*, server_version: str, api_version: str) -> bool:
    """Whether this daemon can mount a volume subpath.

    Both signals are checked and BOTH must be satisfiable.  The API version is
    authoritative (``volume-subpath`` landed in API 1.45), but it is not always
    reported in a parseable form, so the engine major serves as a fallback --
    and an unparseable pair is treated as UNSUPPORTED rather than assumed fine.
    Guessing "probably new enough" here produces the opaque daemon failure this
    module exists to prevent.
    """
    api = _parse_api_version(api_version)
    if api is not None:
        return api >= MIN_API_VERSION
    major = _parse_engine_major(server_version)
    if major is not None:
        return major >= MIN_ENGINE_MAJOR
    return False


def assert_volume_subpath_supported(*, server_version: str, api_version: str) -> None:
    """Fail with an actionable message rather than an opaque daemon error."""
    if supports_volume_subpath(server_version=server_version, api_version=api_version):
        return
    raise DockerCapabilityError(
        "this Docker daemon cannot provide per-run HOME isolation.\n"
        f"  server version: {server_version or '<unknown>'}\n"
        f"  API version:    {api_version or '<unknown>'}\n"
        f"  required:       Engine {MIN_ENGINE_MAJOR}.0+ "
        f"(API {MIN_API_VERSION[0]}.{MIN_API_VERSION[1]}+)\n"
        "\n"
        '  sandbox.auth = "volume" mounts the credential store with '
        "`volume-subpath`, which older daemons do not support. Without it the "
        "only way to mount the store is over the ENTIRE container HOME, which "
        "is the cross-candidate defect this release removes -- so HELIX will "
        "not fall back to it.\n"
        "\n"
        '  Remedy: upgrade Docker, or use sandbox.auth = "env", which mounts '
        "no persistent store at all and has no daemon requirement."
    )


def auth_subpath_bootstrap_command(subpath: str) -> str:
    """Shell that creates the auth subpath inside the mounted volume.

    Run during ``login`` ONLY, in a container that mounts the volume at its
    ROOT -- a subpath mount cannot be used to create the subpath it needs.

    ``mkdir -p`` is deliberate: it is idempotent, and it must never disturb an
    existing directory or its contents. This creates an empty directory and
    nothing else; it does not write, move, or remove any credential.
    """
    safe = subpath.strip("/")
    if not safe or ".." in safe.split("/"):
        raise ValueError(f"unsafe auth subpath: {subpath!r}")
    return f'set -eu; mkdir -p "/helix-auth-root/{safe}"'


def missing_subpath_error(*, backend: str, volume: str, subpath: str) -> str:
    """The message for a volume that exists but has no auth subpath yet."""
    return (
        f"auth volume {volume!r} has no {subpath!r} directory yet, so the "
        f"{backend} agent cannot start.\n"
        "\n"
        "  Volume mode mounts only the backend's auth directory, and Docker "
        "requires that directory to exist before the container starts. "
        "Without this check the run fails inside the daemon with "
        "`cannot access path ...: no such file or directory`, which looks like "
        "an internal error rather than a login problem.\n"
        "\n"
        f"  Remedy: helix sandbox login {backend}"
    )
