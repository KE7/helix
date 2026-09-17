"""Stop HELIX when the run is over or its records stop adding up.

    python3 watchdog.py --pid-file <helix.pid> --stop-record <watchdog_stop.json>
        --gateway-log <gateway_requests.jsonl> --ledger-dir <ledger>
        --helix-state <.helix/state.json> --helix-log <helix_stdout.log>

The evaluator service, not HELIX, holds this protocol's budget: HELIX counts a
request the service answers from its cache, the service counts only
executions.  Once the budget is spent the service refuses every request that
starts a new proposal round (HTTP 429), and HELIX, which treats a failed
parent evaluation as a skipped slot, would keep looping.  The watchdog polls
the run's records and sends SIGINT to HELIX's process group (HELIX saves its
state; ``launch.sh --resume`` continues a run stopped for a fixable reason)
when any of these happens, and writes the reason to the stop record:

* ``budget_spent``: the service refused a request because the budget is spent;
* ``evaluator_refusal``: the service refused a request for any other reason
  than a request from inside the evaluator (403, which is candidate code
  misbehaving and is already harmless), e.g. 409 (an evaluation still
  running) or 400;
* ``evaluator_aborted``: the service aborted an evaluation that overran its
  deadline (it records the execution and exits);
* ``evaluator_failures``: HELIX reported ``EVALUATOR_FAILURES`` failed
  evaluator calls since the watchdog started (the service is unreachable or
  answered nothing);
* ``counter_divergence``: HELIX's persisted evaluation counter and the number
  of requests the service answered drift apart by more than a proposal's worth
  (``DIVERGENCE_TOLERANCE``), relative to their difference when the watchdog
  started, on two consecutive polls;
* ``credential``: ``CONSECUTIVE`` forwarded model API requests in a row failed
  for a reason retrying cannot fix (authentication, permission or billing).

If HELIX has not exited ``ESCALATE_SECONDS`` after SIGINT it gets SIGTERM, and
SIGKILL 30 s later.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONSECUTIVE = 3
POLL_SECONDS = 5
ESCALATE_SECONDS = 120
EVALUATOR_FAILURES = 3
# With one proposal per generation, HELIX's persisted counter can trail the
# service by the parent evaluation of the proposal in progress and a
# validation not yet saved; more than this is a real divergence.
DIVERGENCE_TOLERANCE = 3
FATAL_TYPES = {"authentication_error", "permission_error", "billing_error"}
FATAL_MESSAGE_MARKERS = ("credit balance", "billing", "invalid x-api-key", "api key")
EVALUATOR_FAILURE_RE = re.compile(r"(?:Parent|Child) eval for proposal \S+.*?failed", re.S)


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
    """Credential rule: the last ``consecutive`` forwarded requests all failed fatally."""
    if any(r.get("status") == 503 and str(r.get("reason", "")).startswith("subscription login")
           for r in records):
        return True
    forwarded = [r for r in records if r.get("decision") == "allow"]
    tail = forwarded[-consecutive:]
    return len(tail) == consecutive and all(is_fatal(r) for r in tail)


def evaluator_stop_reason(
    rejections: list[dict[str, Any]], service_events: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]] | None:
    """Reason to stop derived from the evaluator service's own records."""
    for event in service_events:
        if event.get("event") == "evaluation_aborted":
            return "evaluator_aborted", event
    for rejection in rejections:
        status = rejection.get("status")
        if status == 429:
            return "budget_spent", rejection
        if status != 403:
            return "evaluator_refusal", rejection
    return None


def counter_gap(helix_counter: int | None, answered_requests: int) -> int | None:
    """Requests the service answered that HELIX has not (yet) counted."""
    if helix_counter is None:
        return None
    return answered_requests - helix_counter


def diverged(gap: int | None, baseline: int) -> bool:
    """Whether HELIX's counter and the service's answered requests disagree.

    ``baseline`` is the gap when the watchdog started (0 for a fresh run; for
    a resumed run, charges an interrupted segment never saved).  HELIX can
    trail the service by up to ``DIVERGENCE_TOLERANCE`` in-flight requests,
    but it can never count a request the service did not answer.
    """
    if gap is None:
        return False
    return gap < baseline or gap - baseline > DIVERGENCE_TOLERANCE


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _helix_counter(path: Path) -> int | None:
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    value = (state.get("budget") or {}).get("evaluations") if isinstance(state, dict) else None
    return value if isinstance(value, int) else None


def _evaluator_failures(path: Path) -> int:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return 0
    return len(EVALUATOR_FAILURE_RE.findall(text))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _signal(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def stop(pid: int, record_path: Path, reason: str, evidence: Any) -> None:
    record_path.write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "evidence": evidence,
    }, indent=2, default=str))
    _signal(pid, signal.SIGINT)
    deadline = time.monotonic() + ESCALATE_SECONDS
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(1)
    if _alive(pid):
        _signal(pid, signal.SIGTERM)
        time.sleep(30)
        if _alive(pid):
            _signal(pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--stop-record", type=Path, required=True)
    parser.add_argument("--gateway-log", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--helix-state", type=Path, required=True)
    parser.add_argument("--helix-log", type=Path, required=True)
    args = parser.parse_args()

    ledger = args.ledger_dir
    # Only records written after this start count: a resumed run must not stop
    # on the refusal that ended its previous segment.
    start_rejections = len(_jsonl(ledger / "rejected_requests.jsonl"))
    start_events = len(_jsonl(ledger / "service_events.jsonl"))
    start_failures = _evaluator_failures(args.helix_log)
    start_counter = _helix_counter(args.helix_state)
    baseline = (
        0 if start_counter is None
        else max(0, _count_lines(ledger / "requests.jsonl") - start_counter)
    )
    previous: tuple[int | None, int] | None = None
    while True:
        try:
            pid = int(args.pid_file.read_text().strip())
        except (OSError, ValueError):
            return 0
        if not _alive(pid):
            return 0

        requests = _jsonl(args.gateway_log)
        if should_stop(requests):
            stop(pid, args.stop_record, "credential", [
                {k: r.get(k) for k in ("finished_utc", "status", "error_type", "error_message")}
                for r in [r for r in requests if r.get("decision") == "allow"][-CONSECUTIVE:]
            ])
            return 0

        found = evaluator_stop_reason(
            _jsonl(ledger / "rejected_requests.jsonl")[start_rejections:],
            _jsonl(ledger / "service_events.jsonl")[start_events:],
        )
        if found is not None:
            stop(pid, args.stop_record, found[0], found[1])
            return 0

        failures = _evaluator_failures(args.helix_log) - start_failures
        if failures >= EVALUATOR_FAILURES:
            stop(pid, args.stop_record, "evaluator_failures", {"failed_evaluator_calls": failures})
            return 0

        observed = (_helix_counter(args.helix_state), _count_lines(ledger / "requests.jsonl"))
        gap = counter_gap(*observed)
        if observed == previous and diverged(gap, baseline):
            stop(pid, args.stop_record, "counter_divergence", {
                "helix_counter": observed[0], "answered_requests": observed[1],
                "baseline_gap": baseline, "tolerance": DIVERGENCE_TOLERANCE,
            })
            return 0
        previous = observed
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
