"""Summarize the leak-free circle-packing run.

    .helix-run/bin/summarize.sh [--rerun-best]

Reads, never modifies, the run's records:

* ``.helix-run/ledger/evaluations.jsonl``: one record per execution of the
  scoring function (the budget unit), with the packing it scored;
* ``.helix-run/ledger/rejected_requests.jsonl``: refused evaluator requests;
* ``.helix/state.json``, ``lineage.json``, ``evaluations/``, ``attempts/``:
  HELIX's own state and budget counter;
* ``.helix-run/logs/gateway/gateway_requests.jsonl``: every model API request
  (decision, model, effort, tools, token usage);
* ``.helix-run/logs/run_times.jsonl``: wall-clock start and end of each run
  segment;
* ``.helix-run/transcripts/``: Claude Code session transcripts.

It reports the best valid score within the budget (the first
``MAX_EVALUATIONS`` scoring executions) and overall, the generation and
candidate it came from, the budget consumed by both counters, token usage and
cost, wall-clock time, and an independent exact-arithmetic re-check of the
best packing (``independent_check.py``, which shares no code with the
evaluator).  ``--rerun-best`` also re-executes the best candidate once in a
fresh, network-less evaluator container; that execution is a post-hoc
determinism check and is not part of the search or its budget.

Writes ``.helix-run/summary/summary.json`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

BIN = Path(__file__).resolve().parent
PRIVATE = BIN.parent
sys.path.insert(0, str(BIN))
# In the HELIX checkout the client lives in clean_run/evaluator; in a run
# directory setup_run_dir.sh copies it next to this script.
sys.path.insert(1, str(BIN.parent / "evaluator"))

import evaluator_client  # noqa: E402  (same file-set digest the runner sends)
import independent_check  # noqa: E402

# Anthropic first-party API list prices, USD per million tokens (Claude API
# skill model table, cached 2026-06-24).  Cache writes cost 1.25x input (5-minute
# TTL) or 2x (1-hour TTL); cache reads 0.1x.  Confirm against the current
# pricing page before quoting a cost.
PRICES_PER_MTOK = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
}
# optimize_anything reference points, in scoring executions: 63 (paper, best
# reached), 100 (paper's controlled-comparison budget), 133 (executions in
# the bundled paper run), 150 (configured max_metric_calls).
CHECKPOINTS = (25, 50, 63, 100, 133, 150)


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
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"unparseable_line": line[:200]})
    return out


def _json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _model_family(model: str | None) -> str | None:
    if not model:
        return None
    for family in PRICES_PER_MTOK:
        if model == family or model.startswith(family + "-"):
            return family
    return None


def candidate_digests(run_dir: Path) -> dict[str, str]:
    """files_sha256 of every retained candidate worktree, keyed by candidate id."""
    digests: dict[str, str] = {}
    worktrees = run_dir / ".helix" / "worktrees"
    if not worktrees.is_dir():
        return digests
    for tree in sorted(p for p in worktrees.iterdir() if p.is_dir()):
        try:
            digests[tree.name] = evaluator_client.build_archive(tree)[1]
        except (OSError, ValueError):
            continue
    return digests


def best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [r for r in records if r.get("valid") and isinstance(r.get("score"), (int, float))]
    if not valid:
        return None
    return max(valid, key=lambda r: (float(r["score"]), -int(r["seq"])))


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
    total = 0.0
    priced: dict[str, Any] = {}
    unpriced: list[str] = []
    for model, counts in per_model.items():
        family = _model_family(model)
        if family is None:
            unpriced.append(model)
            continue
        price_in, price_out = PRICES_PER_MTOK[family]
        write_1h = counts.get("cache_creation.ephemeral_1h_input_tokens", 0)
        write_total = counts.get("cache_creation_input_tokens", 0)
        write_5m = max(0, write_total - write_1h)
        cost = (
            counts.get("input_tokens", 0) * price_in
            + write_5m * price_in * 1.25
            + write_1h * price_in * 2.0
            + counts.get("cache_read_input_tokens", 0) * price_in * 0.1
            + counts.get("output_tokens", 0) * price_out
        ) / 1e6
        priced[model] = round(cost, 4)
        total += cost
    return {
        "requests_by_decision": dict(decisions),
        "denied_requests_by_reason": dict(denials),
        "requests_by_model_and_effort": dict(efforts),
        "tokens_by_model": {m: dict(c) for m, c in per_model.items()},
        "cost_usd_by_model_list_price": priced,
        "cost_usd_list_price_total": round(total, 4),
        "unpriced_models": unpriced,
        "price_table_usd_per_mtok": PRICES_PER_MTOK,
    }


def rerun(archive: Path, image: str) -> dict[str, Any]:
    """Re-execute a stored candidate once, without network, and re-check it."""
    with tempfile.TemporaryDirectory(prefix="cp-rerun-") as tmp:
        work = Path(tmp) / "tree"
        work.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and not member.name.startswith(("/", "..")) and ".." not in Path(member.name).parts:
                    target = work / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is not None:
                        target.write_bytes(source.read())
        work.chmod(0o777)
        for path in work.rglob("*"):
            path.chmod(0o777 if path.is_dir() else 0o666)
        cmd = [
            "docker", "run", "--rm", "--label", "helix-circle-clean=rerun",
            "--network", "none", "--user", "candidate", "--workdir", "/tree",
            "-e", "PYTHONHASHSEED=0", "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
            "-e", "MKL_NUM_THREADS=1", "-e", "HOME=/tree",
            "-v", f"{work}:/tree", image,
            "timeout", "900", "python3", "/opt/cpeval/harness/candidate_harness.py", "/tree/.rerun.json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = work / ".rerun.json"
        payload = json.loads(out.read_text()) if out.exists() else None
        return {
            "exit_code": proc.returncode,
            "stderr_tail": proc.stderr[-1000:],
            "status": (payload or {}).get("status"),
            "circles": (payload or {}).get("circles"),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun-best", action="store_true")
    args = parser.parse_args()

    env = _read_env(PRIVATE / "protocol.env")
    run_dir = Path(env["RUN_DIR"])
    budget = int(env["MAX_EVALUATIONS"])
    helix_dir = run_dir / ".helix"
    ledger_dir = PRIVATE / "ledger"

    records = _jsonl(ledger_dir / "evaluations.jsonl")
    rejections = _jsonl(ledger_dir / "rejected_requests.jsonl")
    service = _jsonl(ledger_dir / "service_events.jsonl")
    state = _json(helix_dir / "state.json") or {}
    lineage = _json(helix_dir / "lineage.json") or []
    generation_of = {e["id"]: e.get("generation") for e in lineage if isinstance(e, dict)}
    digests = candidate_digests(run_dir)
    candidates_by_digest: dict[str, list[str]] = defaultdict(list)
    for cid, digest in digests.items():
        candidates_by_digest[digest].append(cid)

    seqs = [int(r.get("seq", 0)) for r in records]
    within = [r for r in records if int(r.get("seq", 0)) <= budget]

    def describe(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        ids = sorted(candidates_by_digest.get(str(record.get("files_sha256")), []))
        check = independent_check.check(record["circles"], claimed_score=float(record["score"]))
        return {
            "score": record["score"],
            "execution_seq": record["seq"],
            "split": record.get("split"),
            "received_utc": record.get("received_utc"),
            "candidate_ids": ids,
            "generations": sorted({generation_of.get(i) for i in ids if generation_of.get(i) is not None}),
            "candidate_archive": str(ledger_dir / record["archive"]) if record.get("archive") else None,
            "certification_shrink": (record.get("details") or {}).get("certification_shrink"),
            "radius_sum_before_shrink": (record.get("details") or {}).get("radius_sum_before_shrink"),
            "independent_check": check,
            "circles": record["circles"],
        }

    best_within = describe(best_record(within))
    best_overall = describe(best_record(records))

    checkpoints: dict[str, Any] = {}
    for point in CHECKPOINTS:
        subset = [r for r in records if int(r.get("seq", 0)) <= point]
        best = best_record(subset)
        entry: dict[str, Any] = {
            "executions": len(subset),
            "best_score": None if best is None else best["score"],
            "execution_seq": None if best is None else best["seq"],
        }
        if len(records) < point:
            entry["note"] = f"run has only {len(records)} executions"
        checkpoints[str(point)] = entry

    # HELIX's own view: the best accepted candidate on validation.
    helix_best = None
    for path in sorted((helix_dir / "evaluations").glob("*.json")) if (helix_dir / "evaluations").is_dir() else []:
        data = _json(path)
        score = (data.get("instance_scores") or {}).get("0") if isinstance(data, dict) else None
        if isinstance(score, (int, float)) and (helix_best is None or score > helix_best["val_score"]):
            helix_best = {
                "candidate_id": data.get("candidate_id"),
                "val_score": score,
                "generation": generation_of.get(data.get("candidate_id")),
                "discovered_at_helix_evaluation": (state.get("num_metric_calls_by_discovery") or {}).get(
                    data.get("candidate_id")
                ),
            }

    attempts_dir = helix_dir / "attempts"
    attempts = [_json(p) for p in sorted(attempts_dir.glob("*.json"))] if attempts_dir.is_dir() else []
    splits = Counter(str(r.get("split")) for r in records)
    ledger_consistent = seqs == list(range(1, len(records) + 1))
    helix_count = (state.get("budget") or {}).get("evaluations")

    runs = _jsonl(PRIVATE / "logs" / "run_times.jsonl")
    segments, open_start = [], None
    for event in runs:
        if event.get("event") == "start":
            open_start = event
        elif event.get("event") == "end" and open_start is not None:
            segments.append({"mode": event.get("mode"), "start_utc": open_start.get("utc"),
                             "end_utc": event.get("utc"), "wall_seconds": event.get("wall_seconds"),
                             "exit_code": event.get("exit_code")})
            open_start = None
    first = _utc(records[0].get("received_utc")) if records else None
    last = _utc(records[-1].get("finished_utc")) if records else None
    candidate_seconds = [float(r["wall_seconds"]) for r in records if isinstance(r.get("wall_seconds"), (int, float))]

    # Only requests made while HELIX ran: the log also holds the isolation
    # checks' deliberate policy probes, which launch.sh runs before a run.
    windows = []
    for event in runs:
        if event.get("event") == "start":
            windows.append([_utc(event.get("utc")), None])
        elif event.get("event") == "end" and windows and windows[-1][1] is None:
            windows[-1][1] = _utc(event.get("utc"))
    all_requests = _jsonl(PRIVATE / "logs" / "gateway" / "gateway_requests.jsonl")
    in_run = [
        r for r in all_requests
        if (t := _utc(r.get("started_utc"))) is not None
        and any(start is not None and start <= t and (end is None or t <= end) for start, end in windows)
    ]
    gateway = usage_cost(in_run)
    gateway["requests_outside_run_segments"] = len(all_requests) - len(in_run)
    transcripts = sorted((PRIVATE / "transcripts").rglob("*.jsonl"))

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "settings": {k: env.get(k) for k in ("MAX_EVALUATIONS", "MAX_EXECUTIONS", "MODEL", "EFFORT", "MAX_TURNS", "EVAL_CPUS")},
        "best_valid_within_budget": best_within,
        "best_valid_overall": best_overall,
        "best_by_execution_checkpoint": checkpoints,
        "helix_best_accepted_candidate": helix_best,
        "budget": {
            "unit": "executions of the scoring function",
            "max_evaluations": budget,
            "ledger_executions": len(records),
            "ledger_executions_by_split": dict(splits),
            "ledger_sequence_contiguous": ledger_consistent,
            "helix_counter": helix_count,
            "helix_counter_matches_ledger": helix_count == len(records),
            "executions_beyond_budget": max(0, len(records) - budget),
            "valid_executions": sum(1 for r in records if r.get("valid")),
            "timed_out_executions": sum(1 for r in records if r.get("timed_out")),
            "evaluator_internal_errors": sum(1 for r in records if r.get("internal_error")),
            "refused_requests_by_status": dict(Counter(str(r.get("status")) for r in rejections)),
            "sidecar_starts": sum(1 for e in service if e.get("event") == "start"),
        },
        "helix": {
            "generation": state.get("generation"),
            "mutations": state.get("mutation_counter"),
            "candidates_with_full_validation": len(list((helix_dir / "evaluations").glob("*.json")))
            if (helix_dir / "evaluations").is_dir() else 0,
            "rejected_attempts": len(attempts),
            "tokens_and_cost_reported_by_claude_code": {
                k: v for k, v in (state.get("budget") or {}).items() if k != "evaluations"
            },
        },
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

    if args.rerun_best and best_within and best_within.get("candidate_archive"):
        result = rerun(Path(best_within["candidate_archive"]), env["EVALUATOR_IMAGE"])
        circles = result.pop("circles")
        result["same_packing_as_scored"] = circles == best_within["circles"]
        result["independent_check"] = independent_check.check(circles) if isinstance(circles, list) and len(circles) == 26 else None
        summary["rerun_best"] = result

    out_dir = PRIVATE / "summary"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if best_within is not None:
        (out_dir / "best_within_budget_packing.json").write_text(json.dumps(best_within["circles"]))

    def fmt(value: Any) -> str:
        return "n/a" if value is None else str(value)

    b = summary["budget"]
    lines = [
        "# Circle packing (n=26) clean run: summary",
        "",
        f"- Settings: model {env.get('MODEL')}, effort {env.get('EFFORT')}, max_turns {env.get('MAX_TURNS')}, "
        f"budget {budget} scoring executions",
        f"- Best valid score within budget: {fmt(best_within and best_within['score'])} "
        f"(execution {fmt(best_within and best_within['execution_seq'])}, "
        f"candidates {fmt(best_within and best_within['candidate_ids'])}, "
        f"generations {fmt(best_within and best_within['generations'])})",
        f"- Independent exact re-check of that packing: valid at 1e-6 = "
        f"{fmt(best_within and best_within['independent_check']['valid_at_tolerance'])}, "
        f"exactly valid = {fmt(best_within and best_within['independent_check']['exactly_valid'])}, "
        f"certified radius sum = {fmt(best_within and best_within['independent_check']['certified_exactly_valid_radius_sum'])}, "
        f"score within certified sum = {fmt(best_within and best_within['independent_check'].get('claimed_score_within_certified_sum'))}",
        f"- Best valid score overall: {fmt(best_overall and best_overall['score'])}",
        f"- HELIX best accepted candidate: {fmt(helix_best)}",
        f"- Budget: {b['ledger_executions']} ledger executions ({b['ledger_executions_by_split']}), "
        f"HELIX counter {fmt(b['helix_counter'])}, match = {b['helix_counter_matches_ledger']}, "
        f"refused requests {b['refused_requests_by_status']}",
        f"- Best by checkpoint: {json.dumps(checkpoints)}",
        f"- Model API cost at list price: ${gateway['cost_usd_list_price_total']} "
        f"(requests {gateway['requests_by_decision']}, denied {gateway['denied_requests_by_reason']}); "
        f"Claude Code's own cost report: ${fmt(summary['helix']['tokens_and_cost_reported_by_claude_code'].get('cost_usd'))}",
        f"- Wall clock: {summary['time']['wall_seconds_total']} s over {len(segments)} segment(s)",
        f"- Records: {summary['records']['transcripts']} transcripts, "
        f"{summary['records']['candidate_archives']} candidate archives",
    ]
    if "rerun_best" in summary:
        lines.append(f"- Re-run of best candidate: {fmt({k: v for k, v in summary['rerun_best'].items() if k != 'independent_check'})}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
