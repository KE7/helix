"""
Evaluator for the eterna100 example.

Scores ``designer.design`` on Eterna100-V2 puzzles. HELIX passes
``HELIX_SPLIT=train|val``; ``HELIX_SPLIT=test`` scores the held-out puzzles
after a run. For each requested puzzle:

  1. ``design(target, BUDGET_S, SEED)`` runs in its own Python process, killed
     after ``HARD_LIMIT_S`` seconds.
  2. The returned sequence is folded with ViennaRNA 2 (minimum free energy,
     37 C, default Turner 2004 parameters, dangles=2), the Eterna100-V2
     protocol.
  3. The puzzle scores 1.0 when the folded structure equals the target
     exactly, else 0.0.

A designer whose files store an RNA sequence of 40 or more letters scores 0 on
every puzzle: designs must be computed, not looked up.

Requested ids come from ``helix_batch.json``: a puzzle id (``p017``) or a
positional index (``"0"``, ``"1"``, ...) into the split's puzzles sorted by id,
which is what HELIX writes when ``[dataset]`` sizes are set. Without
``helix_batch.json`` every puzzle in the split is scored. Emits one
``HELIX_RESULT=`` line with a ``[score, side_info]`` pair per requested id.
"""

import concurrent.futures
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# Shows up in `helix timing`; a no-op outside a HELIX run.
from helix import timed

BUDGET_S = 8.0
HARD_LIMIT_S = BUDGET_S + 4.0
SEED = 0
WORKERS = 4

HERE = Path(__file__).resolve().parent
# Files of the example itself, not of the designer; train/ holds structures.
_NOT_DESIGNER = {"evaluate.py", "make_splits.py", "README.md", "LICENSE-eterna100", "helix.toml"}
# Only at the top: a nested train/ is the designer's. A workspace venv holds packages, not designs.
_SKIP_TOP = {".git", "train", "val", "test", ".venv", "node_modules"}
_SKIP_DIRS = {"__pycache__"}
# HELIX's own files in a candidate worktree, such as .agent_task_prompt.md,
# which quotes the training feedback and so the designs it scored.
_HELIX_PREFIXES = (".helix", ".agent")
_STORED_SEQUENCE = re.compile(rb"[ACGU]{40,}")

_RUNNER = r"""
import json, sys, time
sys.path.insert(0, ".")
import designer
start = time.perf_counter()
seq = designer.design(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))
print("HELIX_DESIGN=" + json.dumps({"sequence": seq, "seconds": time.perf_counter() - start}))
"""


def _puzzles(split: str) -> list[dict[str, str]]:
    path = HERE / split / "puzzles.tsv"
    if not path.exists():
        return []
    return sorted(csv.DictReader(open(path, newline=""), delimiter="\t"), key=lambda r: r["id"])


def _requested_ids(ordered: list[str]) -> list[str]:
    batch = Path.cwd() / "helix_batch.json"
    if not batch.exists():
        return ordered
    ids = json.loads(batch.read_text())
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError("helix_batch.json must contain a JSON list[str]")
    return ids


def _stored_sequence() -> str | None:
    """The first designer file that stores a 40+ letter RNA sequence, as ``path:line``."""
    root = Path.cwd()
    for dirpath, dirnames, filenames in os.walk(root):
        top = Path(dirpath) == root
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not (top and d in _SKIP_TOP) and not d.startswith(_HELIX_PREFIXES)
        ]
        for name in filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root)
            if str(rel) in _NOT_DESIGNER or name.startswith(_HELIX_PREFIXES) or path.stat().st_size > 20_000_000:
                continue
            data = path.read_bytes()
            match = _STORED_SEQUENCE.search(data)
            if match:
                line = data.count(b"\n", 0, match.start()) + 1
                return f"{rel}:{line}"
    return None


def _run_designer(target: str) -> tuple[str, float, str | None]:
    """(sequence, seconds, error) from one design() call in a fresh process."""
    start = time.perf_counter()
    # A session of its own, so a timeout also stops any worker processes design() started.
    proc = subprocess.Popen(
        [sys.executable, "-c", _RUNNER, target, str(BUDGET_S), str(SEED)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=Path.cwd(), start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=HARD_LIMIT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return "", time.perf_counter() - start, f"design() did not return within {HARD_LIMIT_S:.0f} s"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # workers a finished design() left behind
        except (ProcessLookupError, PermissionError):
            pass
    out = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("HELIX_DESIGN="):
            result = json.loads(line.removeprefix("HELIX_DESIGN="))
            return str(result["sequence"]), float(result["seconds"]), None
    tail = (out.stderr.strip().splitlines() or ["no output"])[-1]
    return "", time.perf_counter() - start, f"design() failed: {tail[:300]}"


def _check(target: str, sequence: str) -> tuple[str, str | None]:
    """(folded structure, problem with the sequence) for a design."""
    import RNA

    if len(sequence) != len(target):
        return "", f"sequence has {len(sequence)} letters, target has {len(target)}"
    if set(sequence) - set("ACGU"):
        return "", f"sequence has letters other than A, C, G, U: {''.join(sorted(set(sequence) - set('ACGU')))}"
    with timed("fold"):
        folded, _ = RNA.fold(sequence)
    return folded, None


def _score(puzzle: dict[str, str]) -> list[object]:
    target = puzzle["structure"]
    with timed("design", example=puzzle["id"]):
        sequence, seconds, error = _run_designer(target)
    folded, problem = _check(target, sequence) if not error else ("", None)
    error = error or problem
    solved = not error and folded == target
    info: dict[str, object] = {
        "puzzle": puzzle["id"],
        "name": puzzle["name"],
        "length": len(target),
        "target": target,
        "design": sequence,
        "folded": folded,
        "solved": solved,
        "seconds": round(seconds, 2),
        "scores": {"solved": 1.0 if solved else 0.0},
    }
    if folded and not solved:
        import RNA

        info["bp_distance"] = RNA.bp_distance(folded, target)
        info["mismatch"] = "".join("." if a == b else "^" for a, b in zip(folded, target))
    if error:
        info["error"] = error
    return [1.0 if solved else 0.0, info]


def evaluate() -> None:
    split = (os.environ.get("HELIX_SPLIT") or "val").strip().lower()
    puzzles = _puzzles(split if split in ("train", "val", "test") else "val")
    by_id = {p["id"]: p for p in puzzles}
    ordered = list(by_id)
    requested = _requested_ids(ordered)

    def lookup(pid: str) -> dict[str, str] | None:
        if pid in by_id:
            return by_id[pid]
        if pid.isdigit() and int(pid) < len(ordered):
            return by_id[ordered[int(pid)]]
        return None

    stored = _stored_sequence()
    if stored:
        error = f"{stored} stores an RNA sequence of 40+ letters; designs must be computed, not looked up"
        print("HELIX_RESULT=" + json.dumps([[0.0, {"error": error}] for _ in requested]))
        return

    found = [lookup(pid) for pid in requested]
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        scored = list(pool.map(_score, [p for p in found if p is not None]))
    results = iter(scored)
    payload = [
        next(results) if p is not None else [0.0, {"error": f"Unknown puzzle id: {pid}"}]
        for pid, p in zip(requested, found)
    ]
    print("HELIX_RESULT=" + json.dumps(payload))


if __name__ == "__main__":
    evaluate()
