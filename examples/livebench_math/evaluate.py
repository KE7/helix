"""Candidate-side adapter: send only the prompt and opaque batch positions."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def build_request(base_dir: Path, split: str) -> dict[str, Any]:
    prompt = (base_dir / "prompt.txt").read_text()
    ids = json.loads((base_dir / "helix_batch.json").read_text())
    if split not in {"train", "val"}:
        raise ValueError("HELIX_SPLIT must be train or val")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("helix_batch.json must contain a list of string IDs")
    return {"prompt": prompt, "split": split, "ids": ids}


def run_client(
    endpoint: str,
    request_payload: dict[str, Any],
    *,
    timeout: float = 240,
) -> list[list[Any]]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(request_payload["ids"]):
        raise ValueError("protected evaluator returned the wrong result cardinality")
    return results


def main() -> None:
    endpoint = os.environ["HELIX_EVALUATOR_ENDPOINT"]
    payload = build_request(Path.cwd(), os.environ.get("HELIX_SPLIT", "val"))
    results = run_client(endpoint, payload)
    print("HELIX_RESULT=" + json.dumps(results, separators=(",", ":")))


if __name__ == "__main__":
    main()
