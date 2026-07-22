"""Remove only resources labeled for this pinned SWE-bench-Live demo."""

from __future__ import annotations

import argparse
import json
import subprocess

from pins import (
    CONTAINER_PREFIX,
    OFFICIAL_IMAGE,
    OFFICIAL_IMAGE_REPOSITORY,
    PRIVATE_VOLUME,
    RESOURCE_LABEL,
)


def _output(args: list[str]) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=120
    ).stdout.strip()


def image_id() -> str | None:
    result = subprocess.run(
        ["docker", "image", "inspect", OFFICIAL_IMAGE, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def inspect_image(reference: str) -> str | None:
    result = subprocess.run(
        ["docker", "image", "inspect", reference, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def cleanup(remove_image: bool) -> dict[str, object]:
    before_id = image_id()
    ids = _output(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label={RESOURCE_LABEL}",
            "--filter",
            f"name={CONTAINER_PREFIX}",
        ]
    ).splitlines()
    removed_containers: list[str] = []
    for container_id in filter(None, ids):
        _output(["docker", "rm", "--force", container_id])
        removed_containers.append(container_id)
    volume = subprocess.run(
        ["docker", "volume", "inspect", PRIVATE_VOLUME],
        capture_output=True,
        text=True,
        timeout=60,
    )
    volume_name_before: str | None = None
    removed_volume = False
    if volume.returncode == 0:
        volume_data = json.loads(volume.stdout)[0]
        labels = volume_data.get("Labels") or {}
        if labels.get("com.helix.demo") != "swebench-live-capstone-2743":
            raise SystemExit(f"refusing to remove unlabeled volume {PRIVATE_VOLUME}")
        volume_name_before = str(volume_data.get("Name") or PRIVATE_VOLUME)
        _output(["docker", "volume", "rm", PRIVATE_VOLUME])
        removed_volume = True
    removed_image_references: list[str] = []
    if remove_image and before_id is not None:
        _output(["docker", "image", "rm", OFFICIAL_IMAGE])
        removed_image_references.append(OFFICIAL_IMAGE)
        latest = f"{OFFICIAL_IMAGE_REPOSITORY}:latest"
        latest_id = inspect_image(latest)
        if latest_id is not None:
            if latest_id != before_id:
                raise SystemExit("refusing to remove latest tag with a different image ID")
            _output(["docker", "image", "rm", latest])
            removed_image_references.append(latest)
        if inspect_image(before_id) is not None:
            _output(["docker", "image", "rm", before_id])
            removed_image_references.append(before_id)
    remaining = _output(
        ["docker", "ps", "-aq", "--filter", f"label={RESOURCE_LABEL}"]
    ).splitlines()
    return {
        "removed_containers": removed_containers,
        "removed_private_volume": removed_volume,
        "private_volume_name_before": volume_name_before,
        "image_id_before": before_id,
        "image_id_after": image_id(),
        "image_id_after_by_id": inspect_image(before_id) if before_id else None,
        "latest_tag_id_after": inspect_image(f"{OFFICIAL_IMAGE_REPOSITORY}:latest"),
        "removed_image_references": removed_image_references,
        "remaining_labeled_containers": list(filter(None, remaining)),
        "private_volume_exists": subprocess.run(
            ["docker", "volume", "inspect", PRIVATE_VOLUME],
            capture_output=True,
            timeout=30,
        ).returncode
        == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-image", action="store_true")
    args = parser.parse_args()
    print(json.dumps(cleanup(args.remove_image), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
