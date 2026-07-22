"""Immutable upstream pins for the SWE-bench-Live HELIX demo."""

from __future__ import annotations

from typing import Any


HARNESS_REPOSITORY = "https://github.com/microsoft/SWE-bench-Live.git"
HARNESS_COMMIT = "70ec57e852e3f2d195790fe71f553e272c691833"
REPOLAUNCH_COMMIT = "7735b1e7363dd3bbc69bd0ef80db646a2ae391fd"

DATASET_REPOSITORY = "https://huggingface.co/datasets/SWE-bench-Live/MultiLang"
DATASET_REVISION = "608f7ae9ab8ea1f9f0d030fe04562cf6bd1a0c8b"
DATASET_PARQUET_PATH = "data/c-00000-of-00001.parquet"
DATASET_PARQUET_SHA256 = (
    "0d3b31cc38c807160e3fef132ed0f86b1e33890a842372894c2340ad08794674"
)
DATASET_PARQUET_SIZE = 8_872_150

TASK_ID = "capstone-engine__capstone-2743"
TASK_REPOSITORY = "capstone-engine/capstone"
TASK_BASE_COMMIT = "56db8c2b690eb6372c91f8d76621f43a33c4dbe4"
# Public upstream commit that resolves the pinned issue.  This pin is never
# copied into a mutation sandbox; the root-only task runner uses it solely to
# prove that the candidate clone cannot reach (or even read) the gold object.
TASK_GOLD_FIX_COMMIT = "717d8b051997bacf48481eace9df357caedc0bca"

OFFICIAL_IMAGE_REPOSITORY = (
    "docker.io/starryzhang/sweb.eval.x86_64.capstone-engine_1776_capstone-2743"
)
OFFICIAL_IMAGE_DIGEST = (
    "sha256:c3d6222106db9afce1eaf6036f67d540011e46ea8e59419097c32d0555032ed9"
)
OFFICIAL_IMAGE = f"{OFFICIAL_IMAGE_REPOSITORY}@{OFFICIAL_IMAGE_DIGEST}"
OFFICIAL_IMAGE_PLATFORM = "linux/amd64"

PRIVATE_VOLUME = "helix-swebench-live-capstone-2743-private-v1"
RESOURCE_LABEL = "com.helix.demo=swebench-live-capstone-2743"
CONTAINER_PREFIX = "helix-swebench-live-capstone-2743-"

EXPECTED_FAIL_TO_PASS = ("IssueTests",)
EXPECTED_PASS_TO_PASS = (
    "legacy_test_customized_mnem",
    "legacy_test_skipdata",
    "unit_sstream",
    "DetailTests",
    "legacy_test_iter",
    "integration_compat_headers",
    "integration_test_litbase",
    "unit_utils",
    "FeaturesTests",
    "integration_cstest",
    "MCTests",
    "unit_cstest",
)
EXPECTED_REBUILD_COMMANDS = (
    "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "
    "-DCAPSTONE_BUILD_CSTEST=ON ; cmake --build build --parallel",
)
EXPECTED_TEST_COMMANDS = (
    "ctest --test-dir build --output-on-failure | tee test-output.log",
)
EXPECTED_PRINT_COMMANDS = ("cat test-output.log",)


class PinMismatch(ValueError):
    """The downloaded row no longer matches this reviewed demo contract."""


def validate_task_row(row: dict[str, Any]) -> None:
    """Reject silent dataset drift before private task data reaches Docker."""

    expected: dict[str, Any] = {
        "instance_id": TASK_ID,
        "repo": TASK_REPOSITORY,
        "base_commit": TASK_BASE_COMMIT,
        "docker_image": OFFICIAL_IMAGE_REPOSITORY.removeprefix("docker.io/"),
        "FAIL_TO_PASS": list(EXPECTED_FAIL_TO_PASS),
        "PASS_TO_PASS": list(EXPECTED_PASS_TO_PASS),
        "rebuild_cmds": list(EXPECTED_REBUILD_COMMANDS),
        "test_cmds": list(EXPECTED_TEST_COMMANDS),
        "print_cmds": list(EXPECTED_PRINT_COMMANDS),
    }
    mismatches = [
        key for key, value in expected.items() if row.get(key) != value
    ]
    if mismatches:
        raise PinMismatch("task row differs at: " + ", ".join(sorted(mismatches)))
    for required in ("patch", "test_patch", "problem_statement", "log_parser"):
        value = row.get(required)
        if not isinstance(value, str) or not value.strip():
            raise PinMismatch(f"task row has empty or invalid {required!r}")
