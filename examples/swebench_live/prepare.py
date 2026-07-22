# mypy: disable-error-code=import-untyped
"""Prepare the pinned private task volume without retaining benchmark data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from pins import (
    DATASET_PARQUET_PATH,
    DATASET_PARQUET_SHA256,
    DATASET_PARQUET_SIZE,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    HARNESS_COMMIT,
    OFFICIAL_IMAGE,
    OFFICIAL_IMAGE_PLATFORM,
    PRIVATE_VOLUME,
    RESOURCE_LABEL,
    TASK_ID,
    validate_task_row,
)


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        cwd=cwd,
        env=env,
        capture_output=capture_output,
        input=input_text,
        text=True,
    )


def _check_prerequisites() -> None:
    for args in (["git", "--version"], ["git", "lfs", "version"], ["docker", "version"]):
        _run(list(args), capture_output=True)
    _run(["docker", "image", "inspect", OFFICIAL_IMAGE], capture_output=True)
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required; run: uv run --with pyarrow python prepare.py"
        ) from exc


def _load_pinned_row() -> dict[str, object]:
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory(prefix="helix-swebench-live-dataset-") as tmp:
        checkout = Path(tmp) / "dataset"
        env = dict(os.environ)
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        _run(
            ["git", "clone", "--quiet", "--no-checkout", DATASET_REPOSITORY, str(checkout)],
            env=env,
        )
        _run(["git", "checkout", "--quiet", DATASET_REVISION], cwd=checkout, env=env)
        _run(
            [
                "git",
                "lfs",
                "pull",
                f"--include={DATASET_PARQUET_PATH}",
                "--exclude=",
            ],
            cwd=checkout,
        )
        parquet = checkout / DATASET_PARQUET_PATH
        payload = parquet.read_bytes()
        if len(payload) != DATASET_PARQUET_SIZE:
            raise SystemExit("pinned parquet size mismatch")
        if hashlib.sha256(payload).hexdigest() != DATASET_PARQUET_SHA256:
            raise SystemExit("pinned parquet digest mismatch")
        for row in pq.read_table(parquet).to_pylist():
            if row.get("instance_id") == TASK_ID:
                validate_task_row(row)
                return dict(row)
    raise SystemExit(f"task {TASK_ID!r} not found at pinned dataset revision")


def _write_private_volume(row: dict[str, object]) -> None:
    inspect = subprocess.run(
        ["docker", "volume", "inspect", PRIVATE_VOLUME],
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        labels = (json.loads(inspect.stdout)[0].get("Labels") or {})
        if labels.get("com.helix.demo") != "swebench-live-capstone-2743":
            raise SystemExit(f"refusing to reuse unlabeled volume {PRIVATE_VOLUME}")
    else:
        _run(
            [
                "docker",
                "volume",
                "create",
                "--label",
                RESOURCE_LABEL,
                PRIVATE_VOLUME,
            ],
            capture_output=True,
        )
    envelope = {
        "source": {
            "harness_commit": HARNESS_COMMIT,
            "dataset_revision": DATASET_REVISION,
            "task_id": TASK_ID,
        },
        "task": row,
    }
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            OFFICIAL_IMAGE_PLATFORM,
            "--network",
            "none",
            "--label",
            RESOURCE_LABEL,
            "--mount",
            f"type=volume,src={PRIVATE_VOLUME},dst=/private",
            "-i",
            OFFICIAL_IMAGE,
            "sh",
            "-c",
            "umask 077; chmod 700 /private; dd of=/private/task.json status=none; chmod 600 /private/task.json",
        ],
        input_text=json.dumps(envelope, separators=(",", ":")),
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    _check_prerequisites()
    row = _load_pinned_row()
    _write_private_volume(row)
    print(
        json.dumps(
            {
                "prepared": True,
                "task_id": TASK_ID,
                "dataset_revision": DATASET_REVISION,
                "image": OFFICIAL_IMAGE,
                "platform": OFFICIAL_IMAGE_PLATFORM,
                "private_volume": PRIVATE_VOLUME,
                "benchmark_data_retained_on_host": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
