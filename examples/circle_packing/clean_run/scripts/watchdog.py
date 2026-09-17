"""Stop HELIX when the model API keeps refusing the credential.

    python3 watchdog.py <gateway_requests.jsonl> <helix.pid> <stop_record.json>

If the mutation agent's requests start failing for a reason that retrying
cannot fix (authentication, permission or billing), every further generation
would still re-score its parent and so spend scoring executions without any
chance of a new candidate.  The watchdog polls the gateway log and, after
``CONSECUTIVE`` such failures in a row among forwarded requests, sends SIGINT
to HELIX's process group (HELIX saves its state; launch.sh --resume continues
the run once the credential is fixed) and records why.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONSECUTIVE = 3
POLL_SECONDS = 10
FATAL_TYPES = {"authentication_error", "permission_error", "billing_error"}
FATAL_MESSAGE_MARKERS = ("credit balance", "billing", "invalid x-api-key", "api key")


def is_fatal(record: dict[str, Any]) -> bool:
    status = record.get("status")
    if record.get("decision") != "allow" or not isinstance(status, int) or status < 400:
        return False
    if status in (401, 403):
        return True
    message = str(record.get("error_message") or "").lower()
    return record.get("error_type") in FATAL_TYPES or (
        status == 400 and any(marker in message for marker in FATAL_MESSAGE_MARKERS)
    )


def should_stop(records: list[dict[str, Any]], consecutive: int = CONSECUTIVE) -> bool:
    forwarded = [r for r in records if r.get("decision") == "allow"]
    tail = forwarded[-consecutive:]
    return len(tail) == consecutive and all(is_fatal(r) for r in tail)


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def main() -> int:
    log, pid_file, stop_record = (Path(a) for a in sys.argv[1:4])
    while True:
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
        except (OSError, ValueError):
            return 0
        records = _records(log)
        if should_stop(records):
            stop_record.write_text(json.dumps({
                "utc": datetime.now(timezone.utc).isoformat(),
                "reason": "model API refused the credential repeatedly",
                "last_failures": [
                    {k: r.get(k) for k in ("finished_utc", "status", "error_type", "error_message")}
                    for r in [r for r in records if r.get("decision") == "allow"][-CONSECUTIVE:]
                ],
            }, indent=2))
            try:
                os.killpg(pid, signal.SIGINT)
            except OSError:
                os.kill(pid, signal.SIGINT)
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
