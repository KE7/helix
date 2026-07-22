"""Terrarium-faithful deterministic split construction and validation."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from constants import FULL_SPLIT_SHA256, FULL_SPLIT_SIZES, SMOKE_IDS


def largest_remainder(target: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Match Terrarium's proportional, largest-remainder allocation."""
    grand_total = sum(sizes.values())
    if target < 0 or grand_total <= 0 or target > grand_total:
        raise ValueError("invalid largest-remainder allocation")
    raw = {name: target * size / grand_total for name, size in sizes.items()}
    result = {name: int(raw[name]) for name in sizes}
    deficit = target - sum(result.values())
    for name in sorted(sizes, key=lambda item: raw[item] - result[item], reverse=True)[
        :deficit
    ]:
        result[name] += 1
    return result


def split_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build the exact seed-0 100/100/168 Terrarium split."""
    normalized = [dict(row) for row in rows]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[str(row["subtask"])].append(row)
    sizes = {subtask: len(group) for subtask, group in grouped.items()}
    val_alloc = largest_remainder(FULL_SPLIT_SIZES["val"], sizes)
    train_alloc = largest_remainder(FULL_SPLIT_SIZES["train"], sizes)
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for subtask in sorted(grouped):
        group = sorted(grouped[subtask], key=lambda row: str(row["question_id"]))
        random.Random(0).shuffle(group)
        val_count = val_alloc[subtask]
        train_count = train_alloc[subtask]
        if val_count + train_count > len(group):
            raise ValueError(f"split allocation exceeds subtask {subtask!r}")
        splits["val"].extend(group[:val_count])
        splits["train"].extend(group[val_count : val_count + train_count])
        splits["test"].extend(group[val_count + train_count :])
    return splits


def ids_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "".join(f"{row['question_id']}\n" for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_full_splits(splits: Mapping[str, list[dict[str, Any]]]) -> None:
    for name, expected_size in FULL_SPLIT_SIZES.items():
        rows = splits.get(name)
        if rows is None or len(rows) != expected_size:
            raise ValueError(f"{name} split size does not match the pinned manifest")
        if ids_digest(rows) != FULL_SPLIT_SHA256[name]:
            raise ValueError(f"{name} split IDs do not match the pinned manifest")


def select_smoke_rows(
    splits: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    by_id = {str(row["question_id"]): row for rows in splits.values() for row in rows}
    selected: dict[str, list[dict[str, Any]]] = {}
    for name, ids in SMOKE_IDS.items():
        try:
            selected[name] = [dict(by_id[question_id]) for question_id in ids]
        except KeyError as exc:
            raise ValueError(f"pinned smoke ID is absent: {exc.args[0]}") from exc
    return selected
