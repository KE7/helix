"""Write train/, val/ and test/ from the Eterna100 benchmark's own files.

Usage: python make_splits.py <eterna100_puzzles.tsv> <results.tsv>

Both files are in https://github.com/eternagame/eterna100-benchmarking under
data/ (MIT license, see LICENSE-eterna100). The puzzles are the Eterna100-V2
target structures. Each puzzle's difficulty is how many of the 15 published
algorithm variants in results.tsv solve it under Vienna 2. Sorted by
difficulty, then length, each block of ten puzzles gives five to test, three
to val and two to train, so the three splits span the same range.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(1 << 30)


def main(puzzles_tsv: str, results_tsv: str) -> None:
    puzzles = [
        {"id": f"p{int(r['Puzzle #']):03d}", "name": r["Puzzle Name"].strip(), "structure": r["Secondary Structure V2"].strip()}
        for r in csv.DictReader(open(puzzles_tsv, newline=""), delimiter="\t")
    ]
    solvers: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for r in csv.DictReader(open(results_tsv, newline=""), delimiter="\t"):
        if r["Folder"] == "vienna2" and r["Success"].strip() in ("1", "1.0", "True"):
            solvers[r["Target Structure"].strip()].add((r["Algorithm"], r["Variant"]))
    for p in puzzles:
        p["solved_by"] = len(solvers[p["structure"]])
    ordered = sorted(puzzles, key=lambda p: (-p["solved_by"], len(p["structure"]), p["id"]))
    pattern = ["test", "val", "train", "test", "val", "test", "train", "test", "val", "test"]
    splits: dict[str, list[dict[str, object]]] = defaultdict(list)
    for i, p in enumerate(ordered):
        splits[pattern[i % 10]].append(p)
    here = Path(__file__).parent
    for split, rows in splits.items():
        (here / split).mkdir(exist_ok=True)
        with open(here / split / "puzzles.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["id", "name", "solved_by", "structure"])
            for p in sorted(rows, key=lambda p: p["id"]):
                w.writerow([p["id"], p["name"], p["solved_by"], p["structure"]])
        print(split, len(rows), "puzzles; mean solved_by", round(sum(p["solved_by"] for p in rows) / len(rows), 1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
