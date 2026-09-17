"""Summarize the leak-free circle-packing run.

    .helix-run/bin/summarize.sh [--no-rerun]

Reads, never modifies, the run's records:

* ``.helix-run/ledger/``: the evaluator service's records.
  ``evaluations.jsonl`` holds one record per execution of the scoring
  function (the budget unit) with the packing it scored;
  ``execution_starts.jsonl`` one line per execution started;
  ``requests.jsonl`` every answered request (an execution or an answer from
  the service's cache of already-executed trees) with the HELIX candidate id
  the runner sent; ``rejected_requests.jsonl`` refused requests;
  ``service_events.jsonl`` service starts and aborts;
* ``.helix/state.json``, ``lineage.json``, ``evaluations/``: HELIX's state;
* ``.helix-run/logs/``: model API requests at the gateway, wall-clock segments,
  the watchdog's stop record and the agent containers' measured compute;
* ``.helix-run/transcripts/``: Claude Code session transcripts.

It reports, and writes to ``.helix-run/summary/summary.json`` and
``summary.md``:

* the best valid score among the first ``MAX_EVALUATIONS`` executions, with the
  HELIX candidate, generation and role (seed validation, gate evaluation of a
  new child, validation, parent re-evaluation) of the execution that scored it;
* an independent exact-arithmetic re-check of that packing
  (``independent_check.py``, which shares no code with the evaluator); if it
  fails, the headline says INVALID and the script exits 2;
* a re-execution of that candidate in a fresh network-less evaluator container
  (skipped with ``--no-rerun``), outside the budget; if it does not reproduce
  the scored packing, the headline says NOT REPRODUCED and the script exits 3;
* the best score by execution count (including optimize_anything's reference
  points) and by number of distinct candidate trees executed;
* whether the budget was spent or the run stopped early, why the watchdog
  stopped HELIX, gaps between started and recorded executions, and whether
  HELIX's evaluation counter matches the requests the service answered;
* the compute the mutation agent spent outside the budget: tool calls from the
  transcripts (every Bash command, how many ran Python, their durations),
  CPU-seconds measured in the agent containers, and whether the best
  candidate's source carries the scored coordinates as literals;
* subscription token usage, provider limit observations, and wall-clock time.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import uuid
import bisect
import json
import re
import shlex
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

BIN = Path(__file__).resolve().parent
PRIVATE = BIN.parent
sys.path.insert(0, str(BIN))

import independent_check  # noqa: E402

# optimize_anything's reported circle-packing run (gepa commit 133d185a, the
# state bundled with the CAIS'26 artifact), counted in executions of the
# scoring function, which is this protocol's unit: best >= 2.63598 first at
# execution 63, its final best at 114, 133 executions in total.  The run's
# configured cap was max_metric_calls = 200 (GEPA metric calls, never
# reached); 100 is the paper's controlled-comparison budget and 150 the
# artifact's default for a fresh run.
OA_REFERENCE_EXECUTIONS = (63, 100, 114, 133, 150, 200)
CHECKPOINTS = (25, 50, *OA_REFERENCE_EXECUTIONS)
GENERATION_RE = re.compile(r"^g(\d+)-")
FLOAT_LITERAL_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")
PYTHON_RE = re.compile(r"\bpython3?\b")


def _read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            parts = shlex.split(value)
            env[key.strip()] = parts[0] if parts else ""
    return env


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                record = json.loads(line)
            except ValueError:
                record = {"unparseable_line": line[:200]}
            if isinstance(record, dict):
                out.append(record)
    return out


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except ValueError:
        return None


def _utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def generation_of(candidate_id: Any) -> int | None:
    match = GENERATION_RE.match(candidate_id) if isinstance(candidate_id, str) else None
    return int(match.group(1)) if match else None


def best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [r for r in records if r.get("valid") and isinstance(r.get("score"), (int, float))]
    if not valid:
        return None
    return max(valid, key=lambda r: (float(r["score"]), -int(r["seq"])))


# ---------------------------------------------------------------------------
# Executions, requests and roles
# ---------------------------------------------------------------------------


def execution_roles(requests: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Map each execution seq to the candidate and role of the request that ran it.

    ``val`` executions are validations (the first one of the seed is the seed
    validation).  A ``train`` execution is the gate evaluation of a new child
    when it is the first request naming that candidate, otherwise a parent
    re-evaluation that the service could not answer from its cache.
    """
    roles: dict[int, dict[str, Any]] = {}
    seen_candidates: set[str] = set()
    for request in requests:
        candidate = request.get("candidate_id")
        first = isinstance(candidate, str) and candidate not in seen_candidates
        if isinstance(candidate, str):
            seen_candidates.add(candidate)
        if request.get("kind") != "execution" or not isinstance(request.get("seq"), int):
            continue
        split = request.get("split")
        if split == "val":
            role = "seed validation" if generation_of(candidate) == 0 else "validation"
        elif first and generation_of(candidate) not in (None, 0):
            role = "gate evaluation of a new child"
        else:
            role = "parent re-evaluation"
        roles[int(request["seq"])] = {
            "candidate_id": candidate,
            "generation": generation_of(candidate),
            "role": role,
            "phase": request.get("phase"),
        }
    return roles


def budget_accounting(
    records: list[dict[str, Any]],
    starts: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    budget: int,
    helix_counter: Any,
    marker_present: bool,
) -> dict[str, Any]:
    recorded = sorted(int(r["seq"]) for r in records if isinstance(r.get("seq"), int))
    started = sorted(int(s["seq"]) for s in starts if isinstance(s.get("seq"), int))
    highest = max([0, *recorded, *started])
    unfinished = sorted(set(started) - set(recorded))
    gaps = sorted(set(range(1, highest + 1)) - set(recorded) - set(unfinished))
    duplicates = sorted(seq for seq, count in Counter(recorded).items() if count > 1)
    answered = len(requests)
    cache_hits = sum(1 for r in requests if r.get("kind") == "cache_hit")
    budget_spent = highest >= budget
    return {
        "unit": "executions of the scoring function",
        "max_executions": budget,
        "executions_started": highest,
        "executions_recorded": len(recorded),
        "budget_spent": budget_spent,
        "budget_exhausted": budget_spent,
        "sidecar_budget_marker_present": marker_present,
        "run_incomplete": not budget_spent,
        "executions_short_of_budget": max(0, budget - highest),
        "unfinished_executions": unfinished,
        "missing_execution_numbers": gaps,
        "duplicate_execution_records": duplicates,
        "ledger_consistent": not unfinished and not gaps and not duplicates,
        "executions_beyond_budget": sum(1 for seq in recorded if seq > budget),
        "answered_requests": answered,
        "answered_from_cache": cache_hits,
        "helix_counter": helix_counter,
        "helix_counter_matches_answered_requests": helix_counter == answered,
        "helix_counter_minus_answered_requests": (
            helix_counter - answered if isinstance(helix_counter, int) else None
        ),
        "valid_executions": sum(1 for r in records if r.get("valid")),
        "timed_out_executions": sum(1 for r in records if r.get("timed_out")),
        "evaluator_internal_errors": sum(1 for r in records if r.get("internal_error")),
        "executions_with_candidate_state_left_behind": sum(
            1 for r in records if r.get("candidate_state_remaining")
        ),
        "refused_requests_by_status": dict(Counter(str(r.get("status")) for r in rejections)),
    }


def checkpoints(records: list[dict[str, Any]], points: tuple[int, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for point in points:
        subset = [r for r in records if int(r.get("seq", 0)) <= point]
        best = best_record(subset)
        entry: dict[str, Any] = {
            "executions": len(subset),
            "best_score": None if best is None or len(records) < point else best["score"],
            "reached": len(records) >= point,
            "execution_seq": None if best is None else best["seq"],
        }
        if len(records) < point:
            entry["note"] = f"run has only {len(records)} executions"
        out[str(point)] = entry
    return out


def execution_curve(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = None
    curve = []
    for record in sorted(records, key=lambda r: int(r.get("seq", 0))):
        score = record.get("score") if record.get("valid") else None
        if isinstance(score, (int, float)) and (best is None or score > best):
            best = score
        curve.append({"evaluation": record.get("seq"), "best_score": best})
    return curve


def distinct_tree_curve(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Best score over first executions of distinct candidate trees.

    optimize_anything's count never included re-running unchanged code, so its
    executions are first executions of distinct programs.  The service caches
    identical trees, so here the two counts coincide unless a tree was
    executed twice (an internal error, or the cache turned off).
    """
    seen: set[str] = set()
    best = None
    curve = []
    for record in sorted(records, key=lambda r: int(r.get("seq", 0))):
        key = record.get("files_sha256") or record.get("archive_sha256")
        if not isinstance(key, str) or key in seen:
            continue
        seen.add(key)
        score = record.get("score") if record.get("valid") else None
        if isinstance(score, (int, float)) and (best is None or score > best):
            best = score
            curve.append({"distinct_trees": len(seen), "execution_seq": record.get("seq"),
                          "best_score": best})
    by_count = {}
    for point in CHECKPOINTS:
        reached = [c for c in curve if c["distinct_trees"] <= point]
        by_count[str(point)] = reached[-1]["best_score"] if reached and len(seen) >= point else None
    return {"distinct_trees_executed": len(seen), "improvements": curve,
            "best_score_by_distinct_trees": by_count}


# ---------------------------------------------------------------------------
# Agent-side compute
# ---------------------------------------------------------------------------


def transcript_tool_use(paths: list[Path]) -> dict[str, Any]:
    """Tool calls in Claude Code transcripts, with Bash durations."""
    tools: Counter[str] = Counter()
    bash = {"invocations": 0, "running_python": 0, "background": 0,
            "errors": 0, "seconds_total": 0.0, "seconds_max": 0.0}
    python_seconds = 0.0
    for path in paths:
        uses: dict[str, tuple[str, dict[str, Any], datetime | None]] = {}
        for entry in _jsonl(path):
            message = entry.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            stamp = _utc(entry.get("timestamp"))
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    name = str(block.get("name"))
                    tools[name] += 1
                    tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
                    uses[str(block.get("id"))] = (name, tool_input, stamp)
                    if name == "Bash":
                        bash["invocations"] += 1
                        command = str(tool_input.get("command", ""))
                        if PYTHON_RE.search(command):
                            bash["running_python"] += 1
                        if tool_input.get("run_in_background"):
                            bash["background"] += 1
                elif block.get("type") == "tool_result":
                    use = uses.get(str(block.get("tool_use_id")))
                    if use is None or use[0] != "Bash":
                        continue
                    if block.get("is_error"):
                        bash["errors"] += 1
                    if use[2] is not None and stamp is not None:
                        seconds = max(0.0, (stamp - use[2]).total_seconds())
                        bash["seconds_total"] += seconds
                        bash["seconds_max"] = max(bash["seconds_max"], seconds)
                        if PYTHON_RE.search(str(use[1].get("command", ""))):
                            python_seconds += seconds
    bash["python_seconds_total"] = python_seconds
    return {"transcript_files": len(paths), "tool_calls_by_name": dict(tools), "bash": bash}


def container_compute(records: list[dict[str, Any]]) -> dict[str, Any]:
    cpu = [float(r["cpu_seconds"]) for r in records if isinstance(r.get("cpu_seconds"), (int, float))]
    wall = []
    for r in records:
        first, last = _utc(r.get("first_seen_utc")), _utc(r.get("last_seen_utc"))
        if first and last:
            wall.append((last - first).total_seconds())
    return {
        "agent_containers": len(records),
        "cpu_seconds_total": sum(cpu),
        "cpu_seconds_max_per_container": max(cpu) if cpu else None,
        "wall_seconds_total": sum(wall),
        "cpu_limits": sorted({r.get("cpu_limit") for r in records}, key=str),
        "peak_memory_bytes_max": max((r.get("peak_memory_bytes") or 0 for r in records), default=None),
        "note": "CPU time is sampled every second; the last second of a container can be missed.",
    }


def _significant_digits(literal: str) -> int:
    mantissa = re.split(r"[eE]", literal.lstrip("+-"))[0].replace(".", "").lstrip("0")
    return len(mantissa)


def hardcoded_evidence(sources: dict[str, str], circles: Any, atol: float = 1e-6) -> dict[str, Any]:
    """How many of the scored coordinates appear as numeric literals in the source."""
    values = [float(v) for row in circles for v in row] if isinstance(circles, list) else []
    literals = []
    for text in sources.values():
        for literal in FLOAT_LITERAL_RE.findall(text):
            if _significant_digits(literal) >= 6:
                try:
                    literals.append(float(literal))
                except ValueError:
                    continue
    ordered = sorted(literals)
    matched = 0
    for value in values:
        i = bisect.bisect_left(ordered, value - atol)
        if i < len(ordered) and abs(ordered[i] - value) <= atol:
            matched += 1
    return {
        "coordinate_values": len(values),
        "values_matched_by_literals": matched,
        "long_float_literals_in_source": len(literals),
        "appears_hard_coded": bool(values) and matched >= len(values) // 2,
        "rule": f"a value matches when a literal with >= 6 significant digits is within {atol}; "
                "hard-coded when at least half of the values match",
    }


def archive_sources(archive: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".py") and member.size < 4 * 1024 * 1024:
                handle = tar.extractfile(member)
                if handle is not None:
                    sources[member.name] = handle.read().decode("utf-8", errors="replace")
    return sources


# ---------------------------------------------------------------------------
# Model API usage
# ---------------------------------------------------------------------------


def usage_cost(requests: list[dict[str, Any]]) -> dict[str, Any]:
    per_model: dict[str, Counter[str]] = defaultdict(Counter)
    decisions: Counter[str] = Counter()
    denials: Counter[str] = Counter()
    efforts: Counter[str] = Counter()
    for req in requests:
        if req.get("event") == "start":
            continue
        decisions[str(req.get("decision"))] += 1
        if req.get("decision") == "deny":
            denials[str(req.get("reason"))] += 1
            continue
        model = str(req.get("response_model") or req.get("model"))
        efforts[f"{req.get('model')}:{req.get('effort')}"] += 1
        per_model[model]["requests"] += 1
        per_model[model][f"status_{req.get('status')}"] += 1
        for key, value in (req.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                per_model[model][key] += value
    return {
        "auth_mode": "claude_subscription",
        "requests_by_decision": dict(decisions),
        "denied_requests_by_reason": dict(denials),
        "requests_by_model_and_effort": dict(efforts),
        "tokens_by_model": {m: dict(c) for m, c in per_model.items()},
        "subscription_usage": {
            "unit": "provider-reported tokens and subscription rate-limit headers",
            "rate_limit_observations": [r["subscription_limits"] for r in requests
                                        if r.get("subscription_limits")],
            "dollars_charged": None,
            "note": "Token counts do not determine the fraction of a subscription allowance. "
                    "Claude Code cost estimates are not subscription charges.",
        },
    }


# ---------------------------------------------------------------------------
# Re-execution of the best candidate
# ---------------------------------------------------------------------------


RERUN_PROGRAM = """
import json, sys, pathlib, pwd
sys.path.insert(0, '/opt/cpeval/private')
import evaluator_server as e
candidate = pwd.getpwnam(e.CANDIDATE_USER)
e.WORK_ROOT.mkdir(exist_ok=True)
e.SCRUB_DIRS[:] = e.candidate_writable_dirs(candidate.pw_uid, candidate.pw_gid,
    exclude=[e.WORK_ROOT, e.LEDGER_DIR.parent, e.STREAM_ROOT])
ledger = e.Ledger(e.LEDGER_DIR)
e.evaluate_request(json.load(sys.stdin), ledger, 'rerun')
record = json.loads(ledger.evaluations.read_text().splitlines()[-1])
print(json.dumps({k:record.get(k) for k in ['exit_code','timed_out','wall_seconds','score','circles']}))
"""


def rerun(archive: Path, image: str, time_limit: float) -> dict[str, Any]:
    """Re-execute in a fresh container with the identical service and no host mount."""
    data = archive.read_bytes()
    request = {"protocol": 1, "ids": ["0"], "split": "val",
               "archive_b64": base64.b64encode(data).decode(),
               "archive_sha256": hashlib.sha256(data).hexdigest()}
    name = "helix-circle-clean-rerun-" + uuid.uuid4().hex[:12]
    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", "-i", "--name", name, "--label",
             "helix-circle-clean=rerun", "--network", "none",
             "-e", f"CP_CANDIDATE_TIMEOUT={time_limit}", image,
             "python3", "-c", RERUN_PROGRAM],
            input=json.dumps(request), capture_output=True, text=True,
            timeout=time_limit + 200,
        )
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        result["container_exit_code"] = proc.returncode
        return result
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {"circles": None, "score": None, "error": type(exc).__name__}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-rerun", action="store_true",
                        help="skip re-executing the best candidate (e.g. while the run is going)")
    args = parser.parse_args()

    env = _read_env(PRIVATE / "protocol.env")
    run_dir = Path(env["RUN_DIR"])
    budget = int(env["MAX_EVALUATIONS"])
    helix_dir = run_dir / ".helix"
    ledger_dir = PRIVATE / "ledger"
    logs = PRIVATE / "logs"

    records = sorted(_jsonl(ledger_dir / "evaluations.jsonl"), key=lambda r: int(r.get("seq", 0)))
    starts = _jsonl(ledger_dir / "execution_starts.jsonl")
    requests = _jsonl(ledger_dir / "requests.jsonl")
    rejections = _jsonl(ledger_dir / "rejected_requests.jsonl")
    service = _jsonl(ledger_dir / "service_events.jsonl")
    state = _json(helix_dir / "state.json") or {}
    roles = execution_roles(requests)
    helix_counter = (state.get("budget") or {}).get("evaluations")

    within = [r for r in records if int(r.get("seq", 0)) <= budget]

    def describe(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        role = roles.get(int(record["seq"]), {})
        check = independent_check.check(record["circles"], claimed_score=float(record["score"]))
        archive = ledger_dir / record["archive"] if record.get("archive") else None
        return {
            "score": record["score"],
            "execution_seq": record["seq"],
            "split": record.get("split"),
            "candidate_id": role.get("candidate_id") or record.get("candidate_id"),
            "generation": role.get("generation", generation_of(record.get("candidate_id"))),
            "role": role.get("role"),
            "received_utc": record.get("received_utc"),
            "candidate_wall_seconds": record.get("wall_seconds"),
            "time_limit_seconds": (record.get("rules") or {}).get("time_limit_seconds"),
            "candidate_archive": str(archive) if archive else None,
            "certification_shrink": (record.get("details") or {}).get("certification_shrink"),
            "radius_sum_before_shrink": (record.get("details") or {}).get("radius_sum_before_shrink"),
            "independent_check": check,
            "independent_check_passed": bool(
                check.get("valid_at_tolerance") and check.get("claimed_score_within_certified_sum")
            ),
            "hard_coding": hardcoded_evidence(archive_sources(archive), record["circles"])
            if archive and archive.is_file() else None,
            "circles": record["circles"],
        }

    best_within = describe(best_record(within))
    best_overall = describe(best_record(records))

    helix_best = None
    evaluations_dir = helix_dir / "evaluations"
    for path in sorted(evaluations_dir.glob("*.json")) if evaluations_dir.is_dir() else []:
        data = _json(path)
        score = (data.get("instance_scores") or {}).get("0") if isinstance(data, dict) else None
        if isinstance(score, (int, float)) and (helix_best is None or score > helix_best["val_score"]):
            cid = data.get("candidate_id")
            helix_best = {
                "candidate_id": cid,
                "val_score": score,
                "generation": generation_of(cid),
                "first_execution_seq_of_candidate": min(
                    (seq for seq, role in roles.items() if role.get("candidate_id") == cid), default=None
                ),
            }

    accounting = budget_accounting(
        records, starts, requests, rejections, budget, helix_counter,
        (ledger_dir / "budget_exhausted.json").exists(),
    )
    accounting["executions_by_role"] = dict(Counter(r.get("role") for r in roles.values()))
    accounting["executions_by_split"] = dict(Counter(str(r.get("split")) for r in records))
    accounting["sidecar_starts"] = sum(1 for e in service if e.get("event") == "start")
    accounting["evaluations_aborted_by_deadline"] = sum(
        1 for e in service if e.get("event") == "evaluation_aborted"
    )
    watchdog_stop = _json(logs / "watchdog_stop.json")

    runs = _jsonl(logs / "run_times.jsonl")
    segments, open_start = [], None
    windows: list[list[datetime | None]] = []
    for event in runs:
        if event.get("event") == "start":
            open_start = event
            windows.append([_utc(event.get("utc")), None])
        elif event.get("event") == "end" and open_start is not None:
            segments.append({"mode": event.get("mode"), "start_utc": open_start.get("utc"),
                             "end_utc": event.get("utc"), "wall_seconds": event.get("wall_seconds"),
                             "exit_code": event.get("exit_code")})
            open_start = None
            if windows and windows[-1][1] is None:
                windows[-1][1] = _utc(event.get("utc"))
    first = _utc(records[0].get("received_utc")) if records else None
    last = _utc(records[-1].get("finished_utc")) if records else None
    candidate_seconds = [float(r["wall_seconds"]) for r in records if isinstance(r.get("wall_seconds"), (int, float))]

    # Only requests made while HELIX ran: the log also holds the isolation
    # checks' deliberate policy probes, which launch.sh runs before a run.
    all_requests = _jsonl(logs / "gateway" / "gateway_requests.jsonl")
    in_run = []
    for request in all_requests:
        stamp = _utc(request.get("started_utc"))
        if stamp is not None and any(
            start is not None and start <= stamp and (end is None or stamp <= end)
            for start, end in windows
        ):
            in_run.append(request)
    gateway = usage_cost(in_run)
    gateway["requests_outside_run_segments"] = len(all_requests) - len(in_run)
    transcripts = sorted((PRIVATE / "transcripts").rglob("*.jsonl"))

    agent_compute = {
        "note": "The evaluation budget counts executions of the scoring function only. It does "
                "not bound what the mutation agent computes in its own container.",
        "cpu_limit_per_agent_container": env.get("AGENT_CPUS"),
        "transcripts": transcript_tool_use(transcripts),
        "containers": container_compute(_jsonl(logs / "agent_containers.jsonl")),
        "best_candidate_hard_coding": best_within and best_within.get("hard_coding"),
    }

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "settings": {k: env.get(k) for k in ("MAX_EVALUATIONS", "MAX_GENERATIONS", "MODEL", "EFFORT",
                                            "MAX_TURNS", "EVAL_CPUS", "AGENT_CPUS")},
        "best_valid_within_budget": best_within,
        "best_valid_overall": best_overall,
        "best_by_execution_checkpoint": checkpoints(records, tuple(sorted({*CHECKPOINTS, budget}))),
        "best_so_far_by_evaluation": execution_curve(within),
        "optimize_anything_comparable_curve": distinct_tree_curve(within),
        "helix_best_accepted_candidate": helix_best,
        "budget": accounting,
        "watchdog_stop": watchdog_stop,
        "helix": {
            "generation": state.get("generation"),
            "mutations": state.get("mutation_counter"),
            "max_generations_reached": isinstance(state.get("generation"), int)
            and str(state.get("generation")) == env.get("MAX_GENERATIONS"),
            "candidates_with_full_validation": len(list(evaluations_dir.glob("*.json")))
            if evaluations_dir.is_dir() else 0,
            "tokens_and_cost_reported_by_claude_code": {
                k: v for k, v in (state.get("budget") or {}).items() if k != "evaluations"
            },
        },
        "agent_side_compute": agent_compute,
        "model_api": gateway,
        "time": {
            "segments": segments,
            "wall_seconds_total": sum(s.get("wall_seconds") or 0 for s in segments),
            "first_to_last_scoring_seconds": (last - first).total_seconds() if first and last else None,
            "candidate_run_seconds": {
                "mean": sum(candidate_seconds) / len(candidate_seconds) if candidate_seconds else None,
                "max": max(candidate_seconds) if candidate_seconds else None,
                "total": sum(candidate_seconds),
            },
        },
        "records": {
            "transcripts": len(transcripts),
            "transcript_bytes": sum(p.stat().st_size for p in transcripts),
            "candidate_archives": len(list((ledger_dir / "candidate_archives").glob("*.tar.gz"))),
        },
    }

    exit_code = 0
    labels = []
    if best_within is None:
        labels.append("NO VALID PACKING")
        exit_code = 2
    else:
        if best_within["independent_check_passed"]:
            labels.append("VALID (independent exact re-check passed)")
        else:
            labels.append("INVALID (independent exact re-check failed)")
            exit_code = 2
        if args.no_rerun:
            labels.append("reproduction not checked (--no-rerun)")
        elif best_within.get("candidate_archive"):
            result = rerun(Path(best_within["candidate_archive"]), env["EVALUATOR_IMAGE"],
                           float(best_within.get("time_limit_seconds") or 600))
            circles = result.pop("circles")
            result["same_packing_as_scored"] = circles == best_within["circles"]
            result["same_score_as_scored"] = result.get("score") == best_within["score"]
            result["independent_check"] = (
                independent_check.check(circles) if isinstance(circles, list) and len(circles) == 26 else None
            )
            summary["rerun_best"] = result
            if (result["same_packing_as_scored"] and result["same_score_as_scored"]
                    and result.get("container_exit_code") == 0 and not result.get("timed_out")):
                labels.append("REPRODUCED (fresh container re-run returned the same packing)")
            else:
                labels.append(f"NOT REPRODUCED (fresh container re-run scored {result.get('score')})")
                exit_code = exit_code or 3
        else:
            labels.append("NOT REPRODUCED (candidate archive missing)")
            exit_code = exit_code or 3
    labels.append(
        "BUDGET SPENT" if accounting["budget_spent"]
        else f"RUN INCOMPLETE ({accounting['executions_started']} of {budget} executions)"
    )
    if not accounting["ledger_consistent"]:
        labels.append("LEDGER INCONSISTENT")
        exit_code = exit_code or 4
    if not accounting["helix_counter_matches_answered_requests"]:
        labels.append("HELIX COUNTER != ANSWERED REQUESTS")
    summary["headline"] = labels
    summary["exit_code"] = exit_code

    out_dir = PRIVATE / "summary"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if best_within is not None:
        (out_dir / "best_within_budget_packing.json").write_text(json.dumps(best_within["circles"]))

    def fmt(value: Any) -> str:
        return "n/a" if value is None else str(value)

    b = accounting
    compute = agent_compute
    bash = compute["transcripts"]["bash"]
    lines = [
        "# Circle packing (n=26) clean run: summary",
        "",
        "**" + " | ".join(labels) + "**",
        "",
        f"- Settings: model {env.get('MODEL')}, effort {env.get('EFFORT')}, max_turns {env.get('MAX_TURNS')}, "
        f"budget {budget} scoring executions, agent containers limited to {fmt(env.get('AGENT_CPUS'))} CPUs",
        f"- Best valid score within budget: {fmt(best_within and best_within['score'])} "
        f"(execution {fmt(best_within and best_within['execution_seq'])}, "
        f"candidate {fmt(best_within and best_within['candidate_id'])}, "
        f"generation {fmt(best_within and best_within['generation'])}, "
        f"{fmt(best_within and best_within['role'])})",
        f"- Independent exact re-check: valid at 1e-6 = "
        f"{fmt(best_within and best_within['independent_check']['valid_at_tolerance'])}, "
        f"exactly valid = {fmt(best_within and best_within['independent_check']['exactly_valid'])}, "
        f"certified radius sum = {fmt(best_within and best_within['independent_check']['certified_exactly_valid_radius_sum'])}, "
        f"score within certified sum = {fmt(best_within and best_within['independent_check'].get('claimed_score_within_certified_sum'))}",
        f"- Re-run of the best candidate: {fmt({k: v for k, v in summary.get('rerun_best', {}).items() if k not in ('independent_check', 'stderr_tail')} or None)}",
        f"- Budget: {b['executions_started']} of {budget} executions started, {b['executions_recorded']} recorded, "
        f"spent = {b['budget_spent']}; by role {b['executions_by_role']}; "
        f"unfinished {b['unfinished_executions']}, missing numbers {b['missing_execution_numbers']}; "
        f"answered requests {b['answered_requests']} ({b['answered_from_cache']} from cache), "
        f"HELIX counter {fmt(b['helix_counter'])} (match = {b['helix_counter_matches_answered_requests']}); "
        f"refused {b['refused_requests_by_status']}; watchdog stop: {fmt(watchdog_stop and watchdog_stop.get('reason'))}",
        f"- Best by execution count: {json.dumps(summary['best_by_execution_checkpoint'])}",
        f"- Best by distinct candidate trees executed: "
        f"{json.dumps(summary['optimize_anything_comparable_curve']['best_score_by_distinct_trees'])}",
        f"- HELIX best accepted candidate: {fmt(helix_best)}",
        f"- Agent-side compute (not bounded by the evaluation budget): {bash['invocations']} Bash calls "
        f"({bash['running_python']} running Python, {round(bash['seconds_total'], 1)} s total, "
        f"max {round(bash['seconds_max'], 1)} s), {round(compute['containers']['cpu_seconds_total'], 1)} CPU-seconds "
        f"in {compute['containers']['agent_containers']} agent containers; best candidate hard-coded: "
        f"{fmt(compute['best_candidate_hard_coding'] and compute['best_candidate_hard_coding']['appears_hard_coded'])} "
        f"({fmt(compute['best_candidate_hard_coding'] and compute['best_candidate_hard_coding']['values_matched_by_literals'])} "
        f"of 78 values as literals)",
        f"- Subscription usage: {json.dumps(gateway['tokens_by_model'])} "
        f"(requests {gateway['requests_by_decision']}, denied {gateway['denied_requests_by_reason']}); "
        f"Claude Code API-equivalent estimate (not a subscription charge): ${fmt(summary['helix']['tokens_and_cost_reported_by_claude_code'].get('cost_usd'))}",
        f"- Wall clock: {summary['time']['wall_seconds_total']} s over {len(segments)} segment(s)",
        f"- Records: {summary['records']['transcripts']} transcripts, "
        f"{summary['records']['candidate_archives']} candidate archives",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_dir / 'summary.json'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
