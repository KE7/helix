"""Bake the pinned dataset revision into the disposable evaluator image."""

from __future__ import annotations

import json
from pathlib import Path

from constants import LIVEBENCH_DATA_REVISION
from dataset import select_smoke_rows, split_rows, validate_full_splits


def main() -> None:
    from datasets import load_dataset

    source = load_dataset(
        "livebench/math", revision=LIVEBENCH_DATA_REVISION, split="test"
    )
    # Keep only fields consumed by splitting, solving, and official scoring.
    # LiveBench also carries datetime metadata that is neither needed here nor
    # JSON serializable without lossy coercion.
    fields = ("question_id", "subtask", "turns", "ground_truth")
    rows = [{field: row[field] for field in fields} for row in source]
    splits = split_rows(rows)
    validate_full_splits(splits)
    payload = {
        "dataset_revision": LIVEBENCH_DATA_REVISION,
        "full_splits": splits,
        "smoke_splits": select_smoke_rows(splits),
    }
    target = Path("/opt/livebench-math/data.json")
    target.write_text(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
