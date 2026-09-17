"""HELIX evaluator command for the sandboxed circle-packing run.

HELIX runs this in a short-lived evaluator-runner container whose working
directory is a copy of the candidate tree and whose only network is the
private evaluator network.  It never executes candidate code.  It packs the
candidate tree, sends it to the evaluator service at
``$HELIX_EVALUATOR_ENDPOINT`` and prints the service's answer as the single
``HELIX_RESULT=`` line HELIX parses.  Every result it prints comes from the
service, which records each scoring execution in its private ledger.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOL = 1
MAX_TREE_BYTES = 32 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 1800
SKIP_NAMES = {
    ".git",
    "helix_batch.json",
    ".agent_task_prompt.md",
    ".agent_internal",
    "__pycache__",
    ".DS_Store",
}


def _skipped(rel: Path) -> bool:
    return any(
        part in SKIP_NAMES or part.startswith(".helix") for part in rel.parts
    )


def collect_files(root: Path) -> list[Path]:
    """Regular files under ``root`` in sorted order; links are never followed."""
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(directory).relative_to(root)
        dirnames[:] = sorted(d for d in dirnames if not _skipped(rel_dir / d))
        for name in sorted(filenames):
            rel = rel_dir / name
            if _skipped(rel):
                continue
            full = root / rel
            if full.is_symlink() or not full.is_file():
                continue
            files.append(rel)
    return files


def build_archive(root: Path) -> tuple[bytes, str]:
    """Deterministic tar.gz of the candidate tree and a digest of its files."""
    buffer = io.BytesIO()
    files_digest = hashlib.sha256()
    total = 0
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for rel in collect_files(root):
                data = (root / rel).read_bytes()
                total += len(data)
                if total > MAX_TREE_BYTES:
                    raise ValueError(
                        f"candidate tree exceeds {MAX_TREE_BYTES} bytes of files"
                    )
                info = tarfile.TarInfo(rel.as_posix())
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
                files_digest.update(rel.as_posix().encode() + b"\0")
                files_digest.update(hashlib.sha256(data).hexdigest().encode() + b"\n")
    return buffer.getvalue(), files_digest.hexdigest()


def main() -> int:
    endpoint = os.environ.get("HELIX_EVALUATOR_ENDPOINT")
    if not endpoint:
        print("HELIX_EVALUATOR_ENDPOINT is not set", file=sys.stderr)
        return 2
    root = Path.cwd()
    ids = json.loads((root / "helix_batch.json").read_text())
    request: dict[str, object] = {
        "protocol": PROTOCOL,
        "split": os.environ.get("HELIX_SPLIT", ""),
        "ids": ids,
    }
    try:
        archive, files_sha256 = build_archive(root)
        request["archive_b64"] = base64.b64encode(archive).decode("ascii")
        request["archive_sha256"] = hashlib.sha256(archive).hexdigest()
        request["files_sha256"] = files_sha256
    except (OSError, ValueError) as exc:
        # The service still records this as an execution that scored 0, so the
        # ledger and HELIX's evaluation count stay in step.
        request["client_error"] = f"{type(exc).__name__}: {exc}"
    data = json.dumps(request).encode("utf-8")
    http_request = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(http_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        print(f"evaluator service refused the request: HTTP {exc.code}", file=sys.stderr)
        return 3
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"evaluator service unreachable: {type(exc).__name__}", file=sys.stderr)
        return 3
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(ids):
        print("evaluator service returned a malformed result", file=sys.stderr)
        return 3
    print(f"evaluation_id={payload.get('evaluation_id')}", file=sys.stderr)
    print("HELIX_RESULT=" + json.dumps(results, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
