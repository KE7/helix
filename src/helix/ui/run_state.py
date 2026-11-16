"""Read one HELIX run's ``.helix`` directory for ``helix ui``, read-only and incrementally.

:class:`RunReader` re-reads a file only when its (inode, mtime, size) changes,
reads ``timing.jsonl`` from the byte offset it reached last time, and derives
the ``GET /api/run`` and ``GET /api/frontier`` payloads from what HELIX
recorded.  It never writes, locks or deletes anything, and never calls
``setup_file_logging``.  Unknown values are ``None``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from helix.cli import _first_sentence
from helix.config import HelixConfig
from helix.lineage import LineageEntry, load_lineage
from helix.population import Candidate, EvalResult, ParetoFrontier
from helix.state import EvolutionState, load_state
from helix.timing import TIMING_FILE_NAME, WORK_KINDS, summarize
from helix.ui.redact import redact

Record = dict[str, Any]
_Sig = tuple[int, int, int]

_ID_GENERATION = re.compile(r"^g(\d+)-")

# Decision reasons that say whether a gated proposal cleared its gate
# (datamap A7).  Gate failures come from ``_safe_remove_worktree`` labels at
# evolution.py:3295 and :3346; the rest only happen after the gate passed
# (evolution.py:3428, :3483, :3515-3553, :3610-3651).
_GATE_FAILED = {"minibatch-gate-rejected candidate", "train-gate-rejected candidate"}
_GATE_PASSED = {
    "mutation",
    "val-stage-rejected candidate",
    "duplicate proposal candidate",
    "unselected proposal candidate",
}
# An agent timeout raises MutationError after ``sandbox.timeout_seconds``
# (mutator.py:2329-2341); no message is recorded, so a duration this close to
# the limit is the only signal.
_TIMEOUT_FRACTION = 0.98
_RECENT_GENERATIONS = 30
_AGENT_PHASE_SPANS = {"mutate", "clone", "snapshot", "agent", "summarize", "remove"}
# Decisions for proposals that left neither an evaluation nor an attempt file
# (datamap "Not recorded in attempts/"): evolution.py:2274, 2630, 2672, 3167,
# 3195, 3205.
_FAILED_REASONS = {
    "failed mutation",
    "tamper-rejected mutation candidate",
    "child whose minibatch eval raised",
    "failed merge",
    "tamper-rejected merge candidate",
    "failed seed generation",
}
_OUTCOMES = ("frontier", "dominated", "rejected", "skipped", "failed", "running")
_LINEAGE_NODES = 1500
_MERGE_ID = re.compile(r"^g\d+-m\d+$")
_TRAILING_INT = re.compile(r"(\d+)$")
_SEED = "g0-s0"
# The fields of a ``usage`` record (timing.py ``usage``) the UI sums.
_USAGE_FIELDS = ("cost_usd", "input_tokens", "output_tokens", "cached_input_tokens")
# A proposal row's outcome -> the Spend band's bucket.
_SPEND_OUTCOME = {"frontier": "accepted", "dominated": "accepted", "rejected": "rejected",
                  "skipped": "rejected", "failed": "failed", "running": "running"}


def _sig(path: Path) -> _Sig | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


class _CachedFile:
    """One file parsed by *load*, re-read only when its signature changes."""

    def __init__(self, path: Path, load: Callable[[Path], Any]) -> None:
        self.path = path
        self._load = load
        self.sig: _Sig | None = None
        self._value: Any = None

    def get(self) -> Any:
        """The parsed file, or ``None`` when it is missing or unreadable."""
        sig = _sig(self.path)
        if sig != self.sig:
            self.sig, self._value = sig, None
            if sig is not None:
                try:
                    self._value = self._load(self.path)
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    self._value = None
        return self._value


class _CachedDir:
    """Every ``*.json`` file of a directory (``*.tmp`` writes ignored), each cached."""

    def __init__(self, path: Path, load: Callable[[Path], Any]) -> None:
        self.path = path
        self._load = load
        self._files: dict[str, _CachedFile] = {}

    def get(self) -> dict[str, Any]:
        """File name -> parsed value, for every readable file, in name order."""
        try:
            names = sorted(n for n in os.listdir(self.path) if n.endswith(".json"))
        except OSError:
            names = []
        self._files = {
            n: self._files.get(n) or _CachedFile(self.path / n, self._load) for n in names
        }
        out: dict[str, Any] = {}
        for name, cached in self._files.items():
            value = cached.get()
            if value is not None:
                out[name] = value
        return out

    def sigs(self) -> tuple[tuple[str, _Sig | None], ...]:
        """The signatures seen by the last :meth:`get`."""
        return tuple((n, c.sig) for n, c in self._files.items())


def _load_evaluation(path: Path) -> EvalResult:
    """``EvalResult.from_dict`` without the bulky side info the run view never uses."""
    result = EvalResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
    result.asi, result.side_info, result.per_example_side_info = {}, None, None
    return result


def _load_attempt(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "candidate_id": data["candidate_id"],
        "instance_scores": dict(data.get("instance_scores") or {}),
        "attempt": dict(data.get("attempt") or {}),
    }


def _load_skips(path: Path) -> list[dict[str, Any]]:
    # Same tolerance as cli._load_skips: a list per generation, or one legacy dict.
    data = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in (data if isinstance(data, list) else [data]) if isinstance(r, dict)]


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_record(r: Any) -> bool:
    """True when *r* has every key ``timing.summarize`` and this module index."""
    if not isinstance(r, dict) or not isinstance(r.get("run_id"), str) or not _num(r.get("wall_time")):
        return False
    event = r.get("event")
    if event == "decision":
        return isinstance(r.get("candidate"), str)
    if event == "detail":
        return all(k in r for k in ("within", "source", "name", "count", "seconds", "max_seconds"))
    if event == "usage":
        return _usage_record(r)
    if event not in ("start", "end"):
        return False
    if not isinstance(r.get("span"), str) or not isinstance(r.get("span_id"), str):
        return False
    if not isinstance(r.get("attrs"), dict):
        return False
    if event == "end" and not _num(r.get("duration_seconds")):
        return False
    return not (r["span"] == "generation" and not isinstance(r["attrs"].get("generation"), int))


def _usage_record(r: Record) -> bool:
    """True for a well-formed ``usage`` record (``timing.usage``, one per ``budget.charge_llm_usage``)."""
    generation = r.get("generation")
    return (
        r.get("event") == "usage"
        and (r.get("candidate") is None or isinstance(r.get("candidate"), str))
        and (generation is None or (isinstance(generation, int) and not isinstance(generation, bool)))
        and all(_num(r.get(k, 0)) for k in _USAGE_FIELDS)
    )


def empty_usage() -> dict[str, Any]:
    """A zero usage total: ``cost_usd`` and the three token counts the UI shows."""
    return {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}


def add_usage(total: dict[str, Any], part: dict[str, Any]) -> None:
    """Add a ``usage`` record's (or another total's) cost and tokens to *total*."""
    total["cost_usd"] += float(part.get("cost_usd") or 0.0)
    for key in _USAGE_FIELDS[1:]:
        total[key] += int(part.get(key) or 0)


def candidate_usage(records: Iterable[Record], cid: str) -> dict[str, Any] | None:
    """The sum of *cid*'s ``usage`` records, or ``None`` when it has none (e.g. a run older than them)."""
    total: dict[str, Any] | None = None
    for r in records:
        if r.get("candidate") == cid and _usage_record(r):
            total = total or empty_usage()
            add_usage(total, r)
    return total


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _mean(scores: dict[str, float]) -> float | None:
    return sum(scores.values()) / len(scores) if scores else None


def _objective_means(result: EvalResult) -> dict[str, float] | None:
    """Per-objective mean over the result's slots, as ``ParetoFrontier._update_objective``."""
    values: dict[str, list[float]] = {}
    for slot in result.objective_scores or []:
        for name, score in slot.items():
            values.setdefault(name, []).append(float(score))
    return {name: sum(v) / len(v) for name, v in values.items()} or None


def _cid(r: Record) -> str | None:
    """The candidate a span evaluates or validates: ``attrs.candidate_id``, else ``candidate``."""
    value = r["attrs"].get("candidate_id", r.get("candidate"))
    return value if isinstance(value, str) else None


@contextmanager
def _quiet_population() -> Iterator[None]:
    """Drop the "no objective axes" WARNING (population.py:496) while rebuilding."""

    def keep(record: logging.LogRecord) -> bool:
        return not (record.levelno == logging.WARNING and str(record.msg).startswith("frontier_type="))

    log = logging.getLogger("helix.population")
    log.addFilter(keep)
    try:
        yield
    finally:
        log.removeFilter(keep)


@dataclass
class _Trace:
    """``timing.jsonl`` indexed once per new batch of records."""

    starts: dict[str, Record] = field(default_factory=dict)
    ends: dict[str, Record] = field(default_factory=dict)
    children: dict[str, list[Record]] = field(default_factory=lambda: defaultdict(list))
    last_decision: dict[str, Record] = field(default_factory=dict)
    latest_run: str | None = None
    usage_by_candidate: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage_by_generation: dict[int, dict[str, Any]] = field(default_factory=dict)
    usage_records: int = 0

    def is_open(self, r: Record) -> bool:
        return r["span_id"] not in self.ends

    def parent_span(self, r: Record) -> str | None:
        parent = self.starts.get(r.get("parent_span_id") or "")
        return None if parent is None else str(parent["span"])

    def open_in_latest(self) -> list[Record]:
        return [
            r for r in self.starts.values() if r["run_id"] == self.latest_run and self.is_open(r)
        ]


def _details(records: list[Record]) -> list[dict[str, Any]]:
    """Seconds and counts of ``detail`` records, totalled per (within, source, name).

    These are the agent transcript breakdown and the evaluator's own ``helix.timed`` names
    (timing.py ``detail``; the same grouping as ``timing.summarize`` ``details``).
    """
    totals: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in records:
        if r.get("event") != "detail":
            continue
        key = (str(r.get("within")), str(r.get("source")), str(r.get("name")))
        total = totals.setdefault(
            key, {"within": key[0], "source": key[1], "name": key[2], "seconds": 0.0, "count": 0}
        )
        total["seconds"] += float(r.get("seconds") or 0.0)
        total["count"] += int(r.get("count") or 0)
    return list(totals.values())


@dataclass
class _Front:
    """The accepted candidates and the frontier rebuilt from them."""

    frontier_type: str
    accepted: list[str]
    results: dict[str, EvalResult]
    non_dominated: set[str]
    active: dict[str, list[str]]


class RunReader:
    """Incremental, read-only view of one project's ``.helix`` run state."""

    def __init__(self, project_root: Path, config: HelixConfig) -> None:
        """Watch ``<project_root>/.helix``; *config* is the loaded ``helix.toml``."""
        self._root = project_root
        self._helix = project_root / ".helix"
        self._config = config
        # evolution._config_hash (evolution.py:152-154), recomputed from the
        # config the server already loaded.
        self._config_hash = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:16]
        self._lock = threading.RLock()
        self._state = _CachedFile(self._helix / "state.json", lambda p: load_state(p.parent.parent))
        self._lineage = _CachedFile(self._helix / "lineage.json", load_lineage)
        self._evaluations = _CachedDir(self._helix / "evaluations", _load_evaluation)
        self._attempts = _CachedDir(self._helix / "attempts", _load_attempt)
        self._skips = _CachedDir(self._helix / "skips", _load_skips)
        self._timing_path = self._helix / TIMING_FILE_NAME
        self._timing_ino: int | None = None
        self._timing_offset = 0
        self._timing_mtime: float | None = None
        self._records: list[Record] = []
        self._trace = _Trace()
        self._trace_size = -1
        self._front: _Front | None = None
        self._front_key: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self, now: float | None = None) -> dict[str, Any] | None:
        """The ``GET /api/run`` payload, or ``None`` before any run state exists."""
        with self._lock:
            now = time.time() if now is None else now
            state = self._state.get()
            self._read_timing()
            if state is None and self._timing_mtime is None:
                return None
            return self._snapshot(state, now)

    def frontier(self) -> dict[str, Any] | None:
        """The ``GET /api/frontier`` payload, or ``None`` when there is no state.json."""
        with self._lock:
            state = self._state.get()
            if state is None:
                return None
            return self._frontier_payload(self._front_of(state))

    def candidate_ids(self) -> set[str]:
        """Every candidate id the run has recorded anywhere."""
        with self._lock:
            state = self._state.get()
            self._read_timing()
            ids = set(self._eval_results()) | set(self._attempt_records())
            ids |= set(self._lineage_entries())
            if state is not None:
                ids |= set(state.frontier)
            # A merge has a `merge` span in place of a `proposal` (merger.py:200).
            for r in self._index().starts.values():
                if r["span"] in ("proposal", "merge") and _cid(r) is not None:
                    ids.add(str(_cid(r)))
            return ids

    def timing_records(self) -> list[dict[str, Any]]:
        """Every valid ``timing.jsonl`` record read so far, refreshed first."""
        with self._lock:
            self._read_timing()
            return list(self._records)

    def lineage(self) -> dict[str, Any]:
        """The ``GET /api/lineage`` payload: accepted and gate-rejected candidates as a DAG."""
        with self._lock:
            state = self._state.get()
            self._read_timing()
            return self._lineage_payload(state)

    def proposals(self) -> dict[str, Any]:
        """The ``GET /api/proposals`` payload: one row per proposal, in ``helix log`` order."""
        with self._lock:
            state = self._state.get()
            self._read_timing()
            rows = self._proposal_rows(state)
            counts = dict.fromkeys(_OUTCOMES, 0)
            for row in rows:
                counts[row["outcome"]] += 1
            return {"rows": rows, "counts": counts}

    def instance_history(self, iid: str) -> dict[str, Any] | None:
        """One validation instance's score across accepted candidates, or ``None`` if unknown."""
        with self._lock:
            state = self._state.get()
            if state is None:
                return None
            front = self._front_of(state)
            if iid not in set(_instances(front)):
                return None
            lineage = self._lineage_entries()
            scores = {
                c: front.results[c].instance_scores[iid]
                for c in sorted(front.accepted, key=lambda c: _run_order(c, lineage))
                if iid in front.results[c].instance_scores
            }
            best = max(scores.values(), default=None)
            return {
                "instance": iid,
                "points": [{"id": c, "generation": _generation_of(c, lineage), "score": s}
                           for c, s in scores.items()],
                "best": best,
                "holders": [c for c, s in scores.items() if s == best],
            }

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _read_timing(self) -> None:
        """Append the records written since the last call.

        A truncated or replaced file restarts from 0.  A last line without its
        newline is left for the next call; a line that never parses (torn by a
        killed run) is skipped, as ``timing.read_records`` does.
        """
        try:
            st = self._timing_path.stat()
        except OSError:
            self._timing_ino, self._timing_offset, self._timing_mtime = None, 0, None
            self._records = []
            return
        if st.st_ino != self._timing_ino or st.st_size < self._timing_offset:
            self._timing_ino, self._timing_offset, self._records = st.st_ino, 0, []
        self._timing_mtime = st.st_mtime
        if st.st_size == self._timing_offset:
            return
        try:
            with self._timing_path.open("rb") as fh:
                fh.seek(self._timing_offset)
                chunk = fh.read(st.st_size - self._timing_offset)
        except OSError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        for line in chunk[:end].split(b"\n"):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if _valid_record(record):
                self._records.append(record)
        self._timing_offset += end + 1

    def _index(self) -> _Trace:
        if self._trace_size == len(self._records) and self._trace_size >= 0:
            return self._trace
        trace = _Trace()
        for r in self._records:
            event = r["event"]
            if event == "start":
                trace.starts[r["span_id"]] = r
                if r.get("parent_span_id"):
                    trace.children[r["parent_span_id"]].append(r)
                if r["span"] == "run":
                    trace.latest_run = r["run_id"]
            elif event == "end":
                trace.ends[r["span_id"]] = r
            elif event == "decision":
                trace.last_decision[r["candidate"]] = r
            elif event == "usage":
                # Every run's charges count: state.json's budget is the total over resumes too.
                trace.usage_records += 1
                cid = r.get("candidate")
                generation = r.get("generation")
                if generation is None:  # seed generation runs before any generation span
                    generation = _generation_of(cid, None) if isinstance(cid, str) else 0
                add_usage(trace.usage_by_generation.setdefault(generation, empty_usage()), r)
                if isinstance(cid, str):
                    add_usage(trace.usage_by_candidate.setdefault(cid, empty_usage()), r)
        self._trace, self._trace_size = trace, len(self._records)
        return trace

    def _eval_results(self) -> dict[str, EvalResult]:
        return {r.candidate_id: r for r in self._evaluations.get().values()}

    def _attempt_records(self) -> dict[str, dict[str, Any]]:
        return {a["candidate_id"]: a for a in self._attempts.get().values()}

    def _lineage_entries(self) -> dict[str, LineageEntry]:
        lineage = self._lineage.get()
        return lineage if isinstance(lineage, dict) else {}

    def _front_of(self, state: EvolutionState) -> _Front:
        """Rebuild the frontier like ``cli._reconstruct_frontier``, without its worktree check."""
        results = self._eval_results()
        key = (self._state.sig, self._evaluations.sigs())
        if self._front is not None and key == self._front_key:
            return self._front
        frontier = ParetoFrontier(frontier_type=state.frontier_type)
        accepted: list[str] = []
        with _quiet_population():
            for cid in state.frontier:
                result = results.get(cid)
                if result is None or cid in accepted:
                    continue
                candidate = Candidate(
                    id=cid,
                    worktree_path=str(self._helix / "worktrees" / cid),
                    branch_name=f"helix/{cid}",
                    generation=_generation_of(cid, None),
                    parent_id=None,
                    parent_ids=[],
                    operation="restored",
                )
                try:
                    frontier.add(candidate, result)
                except ValueError:  # MissingObjectiveScoresError: malformed objective slots
                    continue
                accepted.append(cid)
        self._front = _Front(
            frontier_type=state.frontier_type,
            accepted=accepted,
            results={cid: results[cid] for cid in accepted},
            non_dominated=frontier.get_non_dominated(),
            active=frontier.active_frontier_snapshot(),
        )
        self._front_key = key
        return self._front

    # ------------------------------------------------------------------
    # GET /api/frontier
    # ------------------------------------------------------------------

    def _frontier_payload(self, front: _Front) -> dict[str, Any]:
        ev = self._config.evolution
        members = sorted(
            front.non_dominated, key=lambda c: (-front.results[c].aggregate_score(), c)
        )
        instances = _instances(front)
        best: list[float | None] = []
        for inst in instances:
            scores = [front.results[m].instance_scores[inst] for m in members
                      if inst in front.results[m].instance_scores]
            best.append(max(scores) if scores else None)
        held = {
            m: [i for i, inst in enumerate(instances)
                if best[i] is not None and front.results[m].instance_scores.get(inst) == best[i]]
            for m in members
        }
        groups = list(dict.fromkeys(family_of(i, ev.group_key_separator) for i in instances))
        objectives = _objective_names(front)
        all_means: dict[str, list[float | None]] | None = None
        if objectives:
            all_means = {}
            for cid in front.accepted:
                means = _objective_means(front.results[cid]) or {}
                all_means[cid] = [means.get(o) for o in objectives]
        threshold = ev.perfect_score_threshold
        if threshold is None and self._binary(front.results.values(), []):
            threshold = 1.0
        return {
            "type": front.frontier_type,
            "ids": members,
            "instances": instances,
            "groups": groups or [""],
            "best": best,
            "held": held,
            "objectives": objectives,
            "objective_means": None if all_means is None else {m: all_means[m] for m in members},
            "all_objective_means": all_means,
            "below_threshold": None if threshold is None
            else sum(1 for b in best if b is not None and b < threshold),
            "threshold": threshold,
        }

    # ------------------------------------------------------------------
    # GET /api/run
    # ------------------------------------------------------------------

    def _binary(self, results: Any, attempts: list[dict[str, Any]]) -> bool:
        seen = False
        for scores in [r.instance_scores for r in results] + [a["instance_scores"] for a in attempts]:
            for value in scores.values():
                if value not in (0, 1):
                    return False
                seen = True
        return seen

    def _snapshot(self, state: EvolutionState | None, now: float) -> dict[str, Any]:
        config = self._config
        ev = config.evolution
        trace = self._index()
        records = self._records
        lineage = self._lineage_entries()
        attempts = self._attempt_records()
        all_results = self._eval_results()
        front = self._front_of(state) if state is not None else None
        binary = self._binary(all_results.values(), list(attempts.values()))

        # --- run status (datamap A1): only the `run` span's end says it stopped.
        run_start = next(
            (r for r in reversed(records) if r["event"] == "start" and r["span"] == "run"), None
        )
        run_end = trace.ends.get(run_start["span_id"]) if run_start else None
        if run_start is None:
            state_name, detail = "stopped", None
        elif run_end is None:
            state_name, detail = "running", None
        elif run_end.get("outcome") == "ok":
            state_name, detail = "finished", None
        else:
            detail = run_end.get("error_type")
            state_name = "stopped" if detail == "KeyboardInterrupt" else "crashed"
        running = state_name == "running"
        stale = None if self._timing_mtime is None else max(0.0, now - self._timing_mtime)

        # --- generation (datamap A2): the open `generation` span leads state.json.
        latest_gens = [
            r for r in trace.starts.values()
            if r["run_id"] == trace.latest_run and r["span"] == "generation"
        ]
        open_gen = next((g for g in reversed(latest_gens) if trace.is_open(g)), None)
        gen_span = open_gen or (latest_gens[-1] if latest_gens else None)
        state_gen = state.generation if state is not None else 0
        current = int(open_gen["attrs"]["generation"]) if open_gen else state_gen
        done_gen = current - 1 if open_gen else state_gen

        summary = summarize(records)
        finished_gens = {g: s for g, s in summary.generation_seconds.items()
                         if open_gen is None or g != current}
        median_gen = _median(list(finished_gens.values()))

        elapsed: float | None = None
        if run_start is not None:
            elapsed = summary.run_seconds
            if run_end is None:
                latest = max(r["wall_time"] for r in records if r["run_id"] == run_start["run_id"])
                elapsed += max(0.0, now - latest)

        # --- budget and projections (datamap A3-A5).
        budget = state.budget if state is not None else None
        used = budget.evaluations if budget else 0
        max_evals = ev.max_evaluations if ev.max_evaluations > 0 else None
        gens_by_evals: float | None = None  # generations the evaluation budget still allows
        if max_evals is not None and state is not None and state.generation >= 1:
            seed_evals = state.num_metric_calls_by_discovery.get("g0-s0", 0)
            per_gen = (used - seed_evals) / state.generation
            if per_gen > 0:
                gens_by_evals = max(0, max_evals - used) / per_gen
        time_left, basis = None, None
        if running and median_gen is not None:
            if open_gen is not None:
                in_gen = now - float(open_gen["wall_time"])
                left_gens = max(0.0, median_gen - in_gen) + max(0, ev.max_generations - current) * median_gen
            else:
                left_gens = max(0, ev.max_generations - done_gen) * median_gen
            time_left, basis = left_gens, "generations"
            if gens_by_evals is not None and gens_by_evals * median_gen < left_gens:
                time_left, basis = gens_by_evals * median_gen, "evaluations"

        accepted = front.accepted if front else []
        results = front.results if front else {}
        cost = budget.cost_usd if budget else 0.0
        has_cost = cost > 0
        gens_recorded = state.generation if state is not None else 0
        per_gen_cost = cost / gens_recorded if has_cost and gens_recorded >= 1 else None
        # Usage records (timing.py ``usage``) give each generation's own cost; a
        # run older than them keeps the run-total estimate above.
        per_record = trace.usage_records > 0
        recorded = [trace.usage_by_generation[g]["cost_usd"] for g in finished_gens
                    if g in trace.usage_by_generation]
        if has_cost and per_record and recorded:
            per_gen_cost = _median(recorded)
        gens_left = float(max(0, ev.max_generations - gens_recorded))
        if gens_by_evals is not None:
            gens_left = min(gens_left, gens_by_evals)
        kept = [c for c in accepted if _generation_of(c, lineage) > 0]
        spend = {
            "cost_usd": cost,
            "input_tokens": budget.input_tokens if budget else 0,
            "output_tokens": budget.output_tokens if budget else 0,
            "cached_input_tokens": budget.cached_input_tokens if budget else 0,
            "projected_cost_usd": None if per_gen_cost is None or not running
            else cost + per_gen_cost * gens_left,
            "cost_per_generation": per_gen_cost,
            "cost_per_accepted": cost / len(kept) if has_cost and kept else None,
            "has_cost": has_cost,
            "per_record": per_record,
            "by_generation": [{"generation": g, **dict(u)}
                              for g, u in sorted(trace.usage_by_generation.items())],
            "by_outcome": self._spend_by_outcome(state, trace) if per_record else None,
        }

        # --- best, progress, frontier.
        best_id = max(accepted, key=lambda c: results[c].aggregate_score(), default=None)
        best = None
        if best_id is not None:
            best = self._score(results[best_id], binary)
            best = {"id": best_id, **best}
        points: list[dict[str, Any]] = sorted(
            ({"id": c, "generation": _generation_of(c, lineage), "mean": results[c].aggregate_score()}
             for c in accepted),
            key=lambda p: p["generation"],
        )
        steps = _running_best(points)
        objective_bests = None
        objective_progress = None
        if front is not None and front.frontier_type != "instance":
            means = {c: _objective_means(results[c]) or {} for c in accepted}
            names = _objective_names(front) or []
            objective_bests = [
                {"name": n, "id": b, "mean": means[b][n]}
                for n in names
                for b in [max((c for c in accepted if n in means[c]), key=lambda c: means[c][n])]
            ] or None
            # Higher is better per objective (population.py `_update_objective`).
            objective_progress = [
                {"name": n, "points": obj_points, "steps": _running_best(obj_points)}
                for n in names
                for obj_points in [[{"id": p["id"], "generation": p["generation"], "mean": means[p["id"]][n]}
                                    for p in points if n in means[p["id"]]]]
            ] or None
        seed = None
        if _SEED in results:
            seed = {"id": _SEED, **self._score(results[_SEED], binary),
                    "objective_means": _objective_means(results[_SEED])}
        non_dominated = front.non_dominated if front else set()

        # --- per-candidate timing.
        agent_seconds: dict[str, float] = defaultdict(float)
        validation_seconds: dict[str, float] = defaultdict(float)
        agent_durations: list[float] = []
        examples: list[float] = []
        for r in trace.ends.values():
            if r.get("error_type") == "KeyboardInterrupt":
                continue  # Ctrl-C: `helix resume` reruns the same candidate id
            name = r["span"]
            if name == "agent":
                if r.get("candidate"):
                    agent_seconds[r["candidate"]] += r["duration_seconds"]
                if trace.parent_span(r) != "summarize":
                    agent_durations.append(r["duration_seconds"])
            elif name == "validate" or (
                name == "cached_evaluate" and r["attrs"].get("split") == "val"
                and trace.parent_span(r) != "validate"
            ):
                cid = _cid(r)
                if cid:
                    validation_seconds[cid] += r["duration_seconds"]
        for r in trace.starts.values():
            attrs = r["attrs"]
            if (r["span"] == "cached_evaluate" and attrs.get("split") == "train"
                    and _num(attrs.get("examples")) and attrs.get("candidate_id") == r.get("candidate")):
                examples.append(float(attrs["examples"]))

        candidates = self._candidate_rows(state, front, lineage, binary, agent_seconds, validation_seconds)

        # --- loop numbers (datamap A7).
        open_latest = trace.open_in_latest() if running else []
        agents_running = sum(
            1 for r in open_latest if r["span"] == "agent" and trace.parent_span(r) != "summarize"
        )
        validations_running = sum(
            1 for r in open_latest
            if r["span"] == "validate" or (
                r["span"] == "cached_evaluate" and r["attrs"].get("split") == "val"
                and trace.parent_span(r) != "validate"
            )
        )
        gate: dict[int, list[int]] = defaultdict(lambda: [0, 0])  # generation -> [passed, tried]
        for d in trace.last_decision.values():
            g = d.get("generation")
            if not isinstance(g, int):
                continue
            if d.get("reason") in _GATE_PASSED:
                gate[g][0] += 1
                gate[g][1] += 1
            elif d.get("reason") in _GATE_FAILED:
                gate[g][1] += 1
        median_examples = _median(examples)
        val_values = list(validation_seconds.values())
        instances = _instances(front) if front else []
        loop = {
            "frontier": {"size": len(non_dominated), "accepted": len(accepted)},
            "mutation": {"running": agents_running, "median_seconds": _median(agent_durations),
                         "finished": len(agent_durations)},
            "gate": {"passed": sum(p for p, _ in gate.values()), "tried": sum(t for _, t in gate.values()),
                     "ids_each": int(median_examples) if median_examples is not None else ev.minibatch_size},
            "validation": {"median_seconds": _median(val_values), "count": len(val_values),
                           "ids": len(instances) or None, "running": validations_running},
        }

        # --- health.
        finished_rates = [
            (g, p / t) for g, (p, t) in sorted(gate.items())
            if t > 0 and not (open_gen is not None and g == current)
        ]
        finished_counts = [gate[g] for g, _ in finished_rates]
        recent = finished_counts[-_RECENT_GENERATIONS:]
        health = {
            "since_best": max(0, done_gen - steps[-1]["generation"]) if steps else None,
            "pass_rates": [rate for _, rate in finished_rates],
            "pass_recent": sum(p for p, _ in recent) / sum(t for _, t in recent) if recent else None,
            "pass_all": sum(p for p, _ in finished_counts) / sum(t for _, t in finished_counts)
            if finished_counts else None,
            "time_split": self._time_split(trace, now if running else None),
            "failures": self._failures(trace),
        }

        return {
            "status": {"state": state_name, "detail": detail, "stale_seconds": stale},
            "generation": {"current": current, "max": ev.max_generations},
            "elapsed_seconds": elapsed,
            "evaluations": {"used": used, "max": max_evals},
            "time_left_seconds": time_left,
            "time_left_basis": basis,
            "spend": spend,
            "config_changed": state is not None and state.config_hash != self._config_hash,
            "binary_scores": binary,
            "best": best,
            "seed": seed,
            "objective_bests": objective_bests,
            "progress": {"points": points, "steps": steps, "objectives": objective_progress},
            "frontier": {
                "ids": sorted(non_dominated, key=lambda c: (-results[c].aggregate_score(), c)),
                "accepted": len(accepted),
                "type": front.frontier_type if front else ev.frontier_type,
            },
            "candidates": candidates,
            "loop": loop,
            "now": {
                "generation": int(gen_span["attrs"]["generation"]) if gen_span else current,
                "started_at": float(gen_span["wall_time"]) if gen_span else None,
                "typical_seconds": median_gen,
                "lanes": self._lanes(trace, gen_span, all_results, attempts, binary),
            },
            "health": health,
            "details": _details(self._records),
        }

    def _spend_by_outcome(self, state: EvolutionState | None, trace: _Trace) -> dict[str, Any]:
        """Usage per proposal outcome, as ``/api/proposals`` decides the outcome."""
        totals = {bucket: empty_usage() for bucket in ("accepted", "rejected", "failed", "running")}
        for row in self._proposal_rows(state):
            usage = row["usage"]
            if usage is not None:
                add_usage(totals[_SPEND_OUTCOME[row["outcome"]]], usage)
        return totals

    @staticmethod
    def _score(result: EvalResult, binary: bool) -> dict[str, Any]:
        n = len(result.instance_scores)
        mean = result.aggregate_score()
        return {"mean": mean, "n": n, "successes": round(mean * n) if binary else None}

    def _candidate_rows(
        self,
        state: EvolutionState | None,
        front: _Front | None,
        lineage: dict[str, LineageEntry],
        binary: bool,
        agent_seconds: dict[str, float],
        validation_seconds: dict[str, float],
    ) -> list[dict[str, Any]]:
        if state is None or front is None:
            return []
        keys: dict[str, int] = defaultdict(int)
        sole: dict[str, int] = defaultdict(int)
        for members in front.active.values():
            for m in members:
                keys[m] += 1
            live = [m for m in members if m in front.non_dominated]
            if len(live) == 1:
                sole[live[0]] += 1
        notes: dict[str, str] = {}
        for entries in state.sibling_history.values():
            for sibling in entries:
                if sibling.get("summary"):
                    notes[str(sibling.get("child_id"))] = redact(str(sibling["summary"]))
        rows = []
        for cid in front.accepted:
            result = front.results[cid]
            entry = lineage.get(cid)
            rows.append({
                "id": cid,
                "generation": _generation_of(cid, lineage),
                "parent": entry.parent if entry else None,
                "parents": list(entry.parents) if entry else [],
                "operation": entry.operation if entry else ("seed" if cid == "g0-s0" else None),
                "mean": result.aggregate_score(),
                "n": len(result.instance_scores),
                "front": cid in front.non_dominated,
                "keys": keys.get(cid, 0),
                "sole": sole.get(cid, 0),
                "zero": sum(1 for v in result.instance_scores.values() if v == 0),
                "objective_means": _objective_means(result),
                "note": notes.get(cid),
                "agent_seconds": agent_seconds.get(cid),
                "validation_seconds": validation_seconds.get(cid),
            })
        return rows

    def _time_split(self, trace: _Trace, now: float | None) -> list[dict[str, Any]]:
        """Seconds per kind from the non-nesting ``timing.WORK_KINDS`` spans.

        ``evaluate`` time is split by its own ``split`` attr (train: gate, val:
        validation).  When *now* is given (a live run) open spans count up to now.
        """
        totals = {"mutation": 0.0, "gate": 0.0, "validation": 0.0, "other": 0.0}
        spans = [(r, r["duration_seconds"]) for r in trace.ends.values()]
        if now is not None:
            spans += [(r, max(0.0, now - r["wall_time"])) for r in trace.open_in_latest()]
        for r, seconds in spans:
            kind = WORK_KINDS.get(r["span"])
            if kind is None:
                continue
            if kind == "agent":
                totals["mutation"] += seconds
            elif kind == "evaluator":
                totals["validation" if r["attrs"].get("split") == "val" else "gate"] += seconds
            else:
                totals["other"] += seconds
        return [{"kind": k, "seconds": v} for k, v in totals.items()]

    def _failures(self, trace: _Trace) -> dict[str, Any]:
        sandbox = self._config.sandbox
        limit = sandbox.timeout_seconds if sandbox.enabled else None
        timeouts = errors = evaluator_errors = 0
        types: dict[str, int] = defaultdict(int)
        for r in trace.ends.values():
            error = r.get("error_type")
            if r.get("outcome") != "error" or error == "KeyboardInterrupt":
                continue
            if r["span"] == "agent":
                if (error == "MutationError" and limit
                        and r["duration_seconds"] >= _TIMEOUT_FRACTION * limit):
                    timeouts += 1
                else:
                    errors += 1
            elif r["span"] == "evaluate":
                evaluator_errors += 1
                types[str(error)] += 1
        return {"agent_timeouts": timeouts, "agent_errors": errors,
                "evaluator_errors": evaluator_errors, "evaluator_error_types": dict(types)}

    # ------------------------------------------------------------------
    # Lanes of the current generation
    # ------------------------------------------------------------------

    def _lanes(
        self,
        trace: _Trace,
        gen_span: Record | None,
        results: dict[str, EvalResult],
        attempts: dict[str, dict[str, Any]],
        binary: bool,
    ) -> list[dict[str, Any]]:
        if gen_span is None:
            return []
        generation = gen_span["attrs"].get("generation")
        under_gen = trace.children.get(gen_span["span_id"], [])
        work: dict[str, list[Record]] = defaultdict(list)
        for r in trace.starts.values():
            if r["run_id"] == gen_span["run_id"] and r["span"] in ("agent", "evaluate"):
                work[str(r.get("candidate"))].append(r)
        lanes = []
        # A merge generation has a `merge` span in place of proposals (merger.py:200).
        for proposal in (s for s in under_gen if s["span"] in ("proposal", "merge")):
            child = _cid(proposal)
            if child is None:
                continue
            if proposal["span"] == "merge":
                parents = proposal["attrs"].get("parents") or [None]
                parent = parents[0]
            else:
                parent = proposal["attrs"].get("parent")
            phase, started = self._phase(trace, proposal, parent, under_gen, child)
            result = self._result(trace, child, parent, generation, results, attempts, binary)
            if result is not None:
                decision = trace.last_decision.get(child)
                phase = "done"
                started = float(decision["wall_time"]) if decision else started
            segments: list[dict[str, Any]] = [
                {"kind": "agent" if r["span"] == "agent" else "eval",
                 "start": float(r["wall_time"]),
                 "end": None if trace.is_open(r) else float(trace.ends[r["span_id"]]["wall_time"])}
                for r in work.get(child, [])
            ]
            lanes.append({"id": child, "parent": parent, "phase": phase, "phase_started_at": started,
                          "segments": sorted(segments, key=lambda s: s["start"]), "result": result})
        return lanes

    def _phase(
        self, trace: _Trace, proposal: Record, parent: Any, under_gen: list[Record], child: str
    ) -> tuple[str, float]:
        """The lane's phase from its deepest open span (datamap "Phase per proposal")."""
        if not trace.is_open(proposal):
            # Apply phase on the main thread, directly under `generation`.
            after = [s for s in under_gen if s["span"] in ("validate", "cached_evaluate") and _cid(s) == child]
            live = [s for s in after if trace.is_open(s)]
            if live:
                s = live[-1]
                val = s["span"] == "validate" or s["attrs"].get("split") == "val"
                return ("validation" if val else "child_eval"), float(s["wall_time"])
            ended = [float(trace.ends[s["span_id"]]["wall_time"]) for s in after]
            return "waiting", max(ended + [float(trace.ends[proposal["span_id"]]["wall_time"])])
        path = [proposal]
        deepest = [proposal]
        stack = [[proposal]]
        while stack:
            path = stack.pop()
            if len(path) > len(deepest) or (
                len(path) == len(deepest) and path[-1]["wall_time"] > deepest[-1]["wall_time"]
            ):
                deepest = path
            for c in trace.children.get(path[-1]["span_id"], []):
                if trace.is_open(c):
                    stack.append(path + [c])
        mutate = next((s for s in deepest if s["span"] == "mutate"), None)
        if mutate is not None or deepest[-1]["span"] in _AGENT_PHASE_SPANS:
            return "agent", float((mutate or deepest[-1])["wall_time"])
        evaluation = next((s for s in deepest if s["span"] == "cached_evaluate"), None)
        if evaluation is not None:
            started = float(evaluation["wall_time"])
            if not any(c["span"] == "evaluate" for c in trace.children.get(evaluation["span_id"], [])):
                return "waiting", started  # worktree lock wait (evolution.py:1058-1062)
            return ("parent_eval" if evaluation["attrs"].get("candidate_id") == parent
                    else "child_eval"), started
        ended = [float(trace.ends[c["span_id"]]["wall_time"])
                 for c in trace.children.get(proposal["span_id"], []) if not trace.is_open(c)]
        return "waiting", max(ended + [float(proposal["wall_time"])])

    def _result(
        self,
        trace: _Trace,
        child: str,
        parent: Any,
        generation: Any,
        results: dict[str, EvalResult],
        attempts: dict[str, dict[str, Any]],
        binary: bool,
    ) -> dict[str, Any] | None:
        """What became of a proposal: its attempt file, else its decision."""
        attempt = attempts.get(child)
        if attempt is not None:
            scores = attempt["instance_scores"]
            mean = _mean(scores)
            return {"status": "rejected", "reason": attempt["attempt"].get("reason"),
                    "score": mean, "n": len(scores),
                    "successes": round(sum(scores.values())) if binary and scores else None,
                    "stage": attempt["attempt"].get("stage")}
        decision = trace.last_decision.get(child)
        if decision is None:
            return None
        if decision.get("accepted"):
            result = results.get(child)
            score = self._score(result, binary) if result else {"mean": None, "n": None, "successes": None}
            return {"status": "accepted", "reason": decision.get("reason"), "score": score["mean"],
                    "n": score["n"], "successes": score["successes"], "stage": None}
        if decision.get("reason") == "perfect parent subsample":
            # The parent's perfect minibatch is in skips/g<gen>.json (evolution.py:3171-3176).
            skip = next(
                (s for skips in self._skips.get().values() for s in skips
                 if s.get("generation") == generation and s.get("parent_id") == parent),
                None,
            )
            scores = dict((skip or {}).get("parent_eval", {}).get("instance_scores") or {})
            return {"status": "rejected", "reason": "perfect_subsample", "score": _mean(scores),
                    "n": len(scores) or None,
                    "successes": round(sum(scores.values())) if binary and scores else None,
                    "stage": "train_minibatch"}
        return {"status": "failed", "reason": decision.get("reason"), "score": None, "n": None,
                "successes": None, "stage": None}

    # ------------------------------------------------------------------
    # GET /api/proposals and GET /api/lineage
    # ------------------------------------------------------------------

    def _proposal_rows(self, state: EvolutionState | None) -> list[dict[str, Any]]:
        """Every proposal, joined from lineage, evaluations, attempts, skips and decisions.

        Follows ``cli._log_rows``: rows sorted by (generation, trailing id
        number, id); an attempt file outranks an evaluation; the summary is the
        first sentence of the ``sibling_history`` entry; the score delta is
        against ``evaluations/<parent>``.  Departs from it in one place:
        "frontier" means non-dominated (``ParetoFrontier.get_non_dominated``,
        as in the rest of the UI), where ``helix log`` counts any uncleaned
        ``active_frontier`` win.  Unlike ``helix log`` it also lists
        proposals with no lineage entry: perfect-subsample skips, failed and
        tampered proposals (decision only) and the ones still running.
        """
        trace = self._index()
        lineage = self._lineage_entries()
        attempts = self._attempt_records()
        results = self._eval_results()
        front = self._front_of(state) if state is not None else None
        accepted = set(front.accepted) if front else set()
        non_dominated = front.non_dominated if front else set()
        binary = self._binary(results.values(), list(attempts.values()))
        summaries: dict[str, str] = {}
        for entries in (state.sibling_history.values() if state is not None else []):
            for sibling in entries:
                child, text = sibling.get("child_id"), sibling.get("summary")
                if isinstance(child, str) and isinstance(text, str) and text.strip():
                    summaries[child] = redact(_first_sentence(text))
        spans: dict[str, Record] = {}  # the latest proposal or merge span per candidate
        for r in trace.starts.values():
            if r["span"] in ("proposal", "merge") and _cid(r) is not None:
                spans[str(_cid(r))] = r
        run_open = _run_open(trace)
        agent_seconds = _agent_seconds(trace)

        rows: list[dict[str, Any]] = []
        for cid in dict.fromkeys([*lineage, *attempts, *(front.accepted if front else []), *spans,
                                  *trace.last_decision]):
            entry, attempt = lineage.get(cid), attempts.get(cid)
            span, decision = spans.get(cid), trace.last_decision.get(cid)
            info = attempt["attempt"] if attempt else {}
            reason: Any = None
            stage: Any = None
            score: float | None = None
            n: int | None = None
            successes: int | None = None
            basis: str | None = None
            delta: float | None = None
            if attempt is not None:
                outcome, reason, stage = "rejected", info.get("reason"), info.get("stage")
                scores = attempt["instance_scores"]
                score, n = _mean(scores), len(scores) or None
                successes = round(sum(scores.values())) if binary and scores else None
                basis = stage if score is not None else None
            elif cid in accepted:
                assert front is not None
                outcome = "frontier" if cid in non_dominated else "dominated"
                scored = self._score(front.results[cid], binary)
                score, n, successes, basis = scored["mean"], scored["n"], scored["successes"], "val"
            elif (span is not None and run_open and span["run_id"] == trace.latest_run
                  and (decision is None or decision["wall_time"] < span["wall_time"])):
                outcome = "running"
            elif decision is not None and decision.get("reason") in _FAILED_REASONS:
                outcome, reason = "failed", decision.get("reason")
            elif entry is not None:
                # A lineage entry the run never finished (`helix log` shows "no result").
                outcome = "failed"
                reason = decision.get("reason") if decision else "no result"
            else:
                continue  # an interrupted proposal, or a decision about no proposal
            span_parents: list[Any] = []
            if span is not None:
                span_parents = (list(span["attrs"].get("parents") or []) if span["span"] == "merge"
                                else [span["attrs"].get("parent")])
            parents = (list(entry.parents) if entry
                       else [p for p in span_parents if isinstance(p, str)]
                       or [p for p in [info.get("parent_id")] if isinstance(p, str)])
            if basis == "val" and parents and parents[0] in results and score is not None:
                delta = score - results[parents[0]].aggregate_score()
            generation = next(
                (g for g in (entry.generation if entry else None, info.get("generation"),
                             (decision or {}).get("generation"), (span or {}).get("generation"))
                 if isinstance(g, int) and not isinstance(g, bool)),
                _generation_of(cid, None),
            )
            if entry is not None:
                operation = entry.operation
            elif (span is not None and span["span"] == "merge") or _MERGE_ID.match(cid):
                operation = "merge"
            else:
                operation = "seed" if cid == "g0-s0" else "mutate"
            rows.append({
                "id": cid, "generation": generation, "parents": parents, "operation": operation,
                "outcome": outcome, "reason": reason, "stage": stage,
                "score": score, "n": n, "successes": successes, "basis": basis, "delta": delta,
                "files": len(entry.files_changed) if entry else None,
                "summary": summaries.get(cid), "agent_seconds": agent_seconds.get(cid),
                "usage": dict(trace.usage_by_candidate[cid]) if cid in trace.usage_by_candidate else None,
            })

        # Perfect-parent skips: no child exists, only skips/g<gen>.json (evolution.py:3171-3176).
        seen: dict[str, int] = defaultdict(int)
        for skips in self._skips.get().values():
            for skip in skips:
                gen, parent = skip.get("generation"), skip.get("parent_id")
                if not isinstance(gen, int) or not isinstance(parent, str):
                    continue
                sid = f"skip:g{gen}:{parent}"
                seen[sid] += 1
                if seen[sid] > 1:  # several slots of one parent skipped in one generation
                    sid += f":{seen[sid]}"
                scores = dict((skip.get("parent_eval") or {}).get("instance_scores") or {})
                score = _mean(scores)
                rows.append({
                    "id": sid, "generation": gen, "parents": [parent], "operation": "mutate",
                    "outcome": "skipped", "reason": skip.get("reason") or "perfect_subsample",
                    "stage": "train_minibatch", "score": score, "n": len(scores) or None,
                    "successes": round(sum(scores.values())) if binary and scores else None,
                    "basis": "train_minibatch" if score is not None else None, "delta": None,
                    "files": None, "summary": None, "agent_seconds": None, "usage": None,
                })
        rows.sort(key=lambda row: _row_order(row["id"], row["generation"]))
        return rows

    def _lineage_payload(self, state: EvolutionState | None) -> dict[str, Any]:
        """Accepted and rejected proposals as DAG nodes; gate scores never in ``mean``."""
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for row in self._proposal_rows(state):
            node = {"id": row["id"], "generation": row["generation"], "parents": row["parents"],
                    "operation": row["operation"], "outcome": row["outcome"], "best_path": False}
            if row["outcome"] in ("frontier", "dominated"):
                accepted.append({**node, "mean": row["score"], "front": row["outcome"] == "frontier",
                                 "accepted": True})
            elif row["outcome"] == "rejected":
                rejected.append({**node, "mean": None, "gate_score": row["score"],
                                 "gate_basis": row["stage"], "front": False, "accepted": False,
                                 "reason": row["reason"]})
        objectives: list[str] | None = None
        if state is not None:
            front = self._front_of(state)
            objectives = _objective_names(front)
            for node in accepted:
                result = front.results.get(node["id"])
                node["objective_means"] = _objective_means(result) if result else None
        truncated = 0
        if len(accepted) + len(rejected) > _LINEAGE_NODES:
            keep = max(0, _LINEAGE_NODES - len(accepted))
            truncated = len(rejected) - keep
            rejected = rejected[len(rejected) - keep:]
        # Best as `helix best`: max mean, ties to the earliest accepted (population.py:890-897).
        best: str | None = None
        if state is not None:
            front = self._front_of(state)
            results = front.results
            best = max(front.accepted, key=lambda c: results[c].aggregate_score(), default=None)
        by_id = {node["id"]: node for node in accepted}
        todo = [best] if best in by_id else []
        while todo:  # the best candidate and every ancestor, both parents of a merge
            on_path = by_id.get(str(todo.pop()))
            if on_path is not None and not on_path["best_path"]:
                on_path["best_path"] = True
                todo.extend(on_path["parents"])
        nodes = sorted(accepted + rejected, key=lambda n: _row_order(n["id"], n["generation"]))
        generations = [n["generation"] for n in nodes] + [state.generation if state else 0]
        return {"nodes": nodes, "objectives": objectives, "best": best,
                "max_generation": max(generations), "truncated": truncated}


def family_of(instance: str, separator: str) -> str:
    """An instance's family: its id before ``evolution.group_key_separator``, else ``""``.

    The stratified sampler's group key (config.py ``group_key_separator``); the Frontier strip groups by it too.
    """
    return instance.split(separator, 1)[0] if separator and separator in instance else ""


def _running_best(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The points that strictly raise the best ``mean`` so far, in the given (generation) order."""
    steps: list[dict[str, Any]] = []
    for p in points:
        if not steps or p["mean"] > steps[-1]["mean"]:
            steps.append(p)
    return steps


def _run_open(trace: _Trace) -> bool:
    """True while the latest run's ``run`` span has no ``end`` (it may also have been killed)."""
    run = next((r for r in reversed(list(trace.starts.values()))
                if r["span"] == "run" and r["run_id"] == trace.latest_run), None)
    return run is not None and trace.is_open(run)


def _agent_seconds(trace: _Trace) -> dict[str, float]:
    """Agent seconds per candidate (change summary included), Ctrl-C'd calls left out."""
    totals: dict[str, float] = defaultdict(float)
    for r in trace.ends.values():
        if r["span"] == "agent" and r.get("candidate") and r.get("error_type") != "KeyboardInterrupt":
            totals[str(r["candidate"])] += r["duration_seconds"]
    return dict(totals)


def _row_order(cid: str, generation: int) -> tuple[int, int, str]:
    """``helix log`` order (cli._log_rows ``_run_order``): generation, trailing id number, id.

    Skip rows (``skip:g<N>:<parent>``) have no counter of their own; they go last in their generation.
    """
    if cid.startswith("skip:"):
        return (generation, sys.maxsize, cid)
    counter = _TRAILING_INT.search(cid)
    return (generation, int(counter.group(1)) if counter else 0, cid)


def _run_order(cid: str, lineage: dict[str, LineageEntry]) -> tuple[int, int, str]:
    return _row_order(cid, _generation_of(cid, lineage))


def _generation_of(cid: str, lineage: dict[str, LineageEntry] | None) -> int:
    """Lineage generation, else the number in the id (``g12-s40`` -> 12), else 0."""
    entry = (lineage or {}).get(cid)
    if entry is not None:
        return entry.generation
    match = _ID_GENERATION.match(cid)
    return int(match.group(1)) if match else 0


def _instances(front: _Front) -> list[str]:
    """Validation ids in evaluation order across the accepted candidates."""
    return list(dict.fromkeys(i for cid in front.accepted for i in front.results[cid].instance_scores))


def _objective_names(front: _Front) -> list[str] | None:
    names = dict.fromkeys(
        n for cid in front.accepted for slot in front.results[cid].objective_scores or [] for n in slot
    )
    return list(names) or None
