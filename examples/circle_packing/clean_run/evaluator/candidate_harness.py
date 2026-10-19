"""Run a candidate's ``solve.pack_circles(26)`` and write its packing as JSON.

The evaluator service starts this script as an unprivileged user, in a
scratch copy of the candidate tree, with a wall-clock limit.  Everything the
script writes is under the candidate's control (candidate code runs in this
process), so the service treats the output file as untrusted input and
re-validates it with ``packing_rules``.

Usage: ``python3 candidate_harness.py <output.json>``
"""

from __future__ import annotations

import json
import numbers
import os
import sys
import traceback

N_CIRCLES = 26
MAX_ROWS = 10_000
TRACEBACK_TAIL = 4000


def _convert(value: object) -> object:
    if isinstance(value, bool):
        return {"__non_numeric__": "bool"}
    if isinstance(value, numbers.Real):
        try:
            return float(value)
        except (OverflowError, ValueError, TypeError):
            return {"__non_numeric__": type(value).__name__}
    return {"__non_numeric__": type(value).__name__}


def _rows(result: object) -> list[object]:
    try:
        items = list(result)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"pack_circles must return a sequence of (x, y, r) rows, got "
            f"{type(result).__name__}"
        ) from None
    if len(items) > MAX_ROWS:
        raise ValueError(f"pack_circles returned {len(items)} rows")
    rows: list[object] = []
    for item in items:
        try:
            values = list(item)  # type: ignore[call-overload]
        except TypeError:
            rows.append({"__non_row__": type(item).__name__})
            continue
        if len(values) > 16:
            rows.append({"__non_row__": f"sequence of length {len(values)}"})
            continue
        rows.append([_convert(v) for v in values])
    return rows


def _limit_resources() -> None:
    """Cap memory, process count and file size before candidate code runs."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Linux only in production
        return
    limits = {
        "RLIMIT_AS": int(os.environ.get("CP_CANDIDATE_MEMORY_BYTES", str(12 * 1024**3))),
        "RLIMIT_NPROC": int(os.environ.get("CP_CANDIDATE_MAX_PROCESSES", "512")),
        "RLIMIT_FSIZE": int(os.environ.get("CP_CANDIDATE_MAX_FILE_BYTES", str(256 * 1024**2))),
    }
    for name, value in limits.items():
        which = getattr(resource, name, None)
        if which is None or value <= 0:
            continue
        try:
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError):
            pass


def main() -> int:
    out_path = sys.argv[1]
    payload: dict[str, object]
    _limit_resources()
    sys.path.insert(0, os.getcwd())
    try:
        import solve  # noqa: PLC0415 - candidate module, imported at run time

        result = solve.pack_circles(N_CIRCLES)
        payload = {"status": "ok", "circles": _rows(result)}
    except BaseException as exc:  # noqa: BLE001 - report any candidate failure
        payload = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "traceback_tail": traceback.format_exc()[-TRACEBACK_TAIL:],
        }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
