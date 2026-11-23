"""`helix ui` run view: RunReader over synthetic .helix dirs written with HELIX's own writers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from helix.config import DatasetConfig, HelixConfig
from helix.lineage import LineageEntry, record_entry
from helix.population import EvalResult
from helix.state import BudgetState, EvolutionState, save_state
from helix.ui.run_state import RunReader

VAL = ["a__1", "a__2", "b__1", "b__2"]


def _config(**evolution: Any) -> HelixConfig:
    return HelixConfig.model_validate(
        {
            "objective": "o",
            "evaluator": {"command": "python eval.py"},
            "evolution": {"max_generations": 3, "minibatch_size": 8, **evolution},
            "sandbox": {"enabled": True, "timeout_seconds": 100},
        }
    )


def _hash(config: HelixConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:16]


class Trace:
    """Writes records shaped like ``helix.timing``'s, at chosen wall times.

    ``generation`` and ``candidate`` follow the enclosing span, falling back to
    the span's own attrs, as ``timing.span`` does.
    """

    def __init__(self, run_id: str = "r1") -> None:
        self.run_id = run_id
        self.records: list[dict[str, Any]] = []

    def start(self, span: str, t: float, within: dict[str, Any] | None = None, **attrs: Any) -> dict[str, Any]:
        generation = within["generation"] if within else None
        candidate = within["candidate"] if within else None
        record = {
            "event": "start", "span": span, "span_id": uuid4().hex,
            "parent_span_id": within["span_id"] if within else None, "run_id": self.run_id,
            "generation": attrs.get("generation") if generation is None else generation,
            "candidate": attrs.get("candidate_id") if candidate is None else candidate,
            "wall_time": t, "thread_id": 1, "attrs": attrs,
        }
        self.records.append(dict(record))
        return record

    def end(self, record: dict[str, Any], t: float, error: str | None = None) -> None:
        out = {**record, "event": "end", "wall_time": t, "duration_seconds": t - record["wall_time"],
               "outcome": "error" if error else "ok"}
        if error:
            out["error_type"] = error
        self.records.append(out)

    def span(self, span: str, t0: float, t1: float, within: dict[str, Any] | None = None,
             error: str | None = None, **attrs: Any) -> dict[str, Any]:
        record = self.start(span, t0, within, **attrs)
        self.end(record, t1, error)
        return record

    def decision(self, candidate: str, t: float, reason: str, accepted: bool,
                 within: dict[str, Any] | None = None) -> None:
        self.records.append({
            "event": "decision", "candidate": candidate, "accepted": accepted, "reason": reason,
            "parent_span_id": within["span_id"] if within else None, "run_id": self.run_id,
            "generation": within["generation"] if within else None, "wall_time": t, "thread_id": 1,
        })

    def usage(self, candidate: str | None, generation: int | None, cost: float, tokens: tuple[int, int, int],
              source: str = "mutation", t: float = 0.0) -> None:
        """A ``usage`` record as ``timing.usage`` writes it (one per ``budget.charge_llm_usage``)."""
        self.records.append({
            "event": "usage", "candidate": candidate, "source": source,
            "input_tokens": tokens[0], "output_tokens": tokens[1], "cached_input_tokens": tokens[2],
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "reasoning_tokens": 0,
            "cost_usd": cost, "parent_span_id": None, "run_id": self.run_id, "generation": generation,
            "wall_time": t, "thread_id": 1,
        })

    def write(self, root: Path) -> None:
        path = root / ".helix" / "timing.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record) + "\n")
        self.records = []


def _evaluation(root: Path, cid: str, scores: list[float],
                objectives: list[dict[str, float]] | None = None) -> None:
    result = EvalResult(candidate_id=cid, scores={}, asi={},
                        instance_scores=dict(zip(VAL, scores)), objective_scores=objectives,
                        per_example_side_info=[{"feedback": "x"} for _ in scores])
    path = root / ".helix" / "evaluations" / f"{cid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict()))


def _attempt(root: Path, cid: str, parent: str, generation: int, reason: str, stage: str,
             scores: dict[str, float]) -> None:
    payload = EvalResult(candidate_id=cid, scores={}, asi={}, instance_scores=scores).to_dict()
    payload["attempt"] = {"status": "rejected", "reason": reason, "parent_id": parent,
                          "generation": generation, "stage": stage, "example_ids": list(scores),
                          "retained": False}
    path = root / ".helix" / "attempts" / f"{cid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _lineage(root: Path, cid: str, parent: str | None, generation: int) -> None:
    record_entry(root / ".helix" / "lineage.json", LineageEntry(
        id=cid, parent=parent, parents=[parent] if parent else [],
        operation="mutate" if parent else "seed", generation=generation, files_changed=[]))


def _state(root: Path, config: HelixConfig, generation: int, frontier: list[str], **kw: Any) -> None:
    kw.setdefault("budget", BudgetState())
    save_state(EvolutionState(generation=generation, frontier=frontier, instance_scores={},
                              config_hash=_hash(config), **kw), root)


def _seed(trace: Trace, run: dict[str, Any], t: float) -> None:
    """Seed validation over 4 val ids: validate t+0.2 .. t+20.2."""
    trace.span("snapshot", t, t + 0.2, run, candidate_id="g0-s0")
    validate = trace.start("validate", t + 0.2, run, candidate_id="g0-s0")
    cached = trace.start("cached_evaluate", t + 0.2, validate, candidate_id="g0-s0", split="val", examples=4)
    trace.span("evaluate", t + 0.5, t + 20.0, cached, candidate_id="g0-s0", split="val", examples=4)
    trace.end(cached, t + 20.2)
    trace.end(validate, t + 20.2)
    trace.decision("g0-s0", t + 20.3, "seed", True, run)


# ---------------------------------------------------------------------------


def test_no_run_returns_none(tmp_path: Path) -> None:
    reader = RunReader(tmp_path, _config())
    assert reader.snapshot() is None
    assert reader.frontier() is None
    assert reader.candidate_ids() == set()
    assert reader.timing_records() == []


def _seed_only_with_two_agents(root: Path, config: HelixConfig, *, upto: float) -> None:
    """Mirror of a real run caught mid-generation 1: two proposals, records up to *upto*."""
    trace = Trace()
    run = trace.start("run", 1000.0)
    _seed(trace, run, 1000.0)
    trace.span("save_state", 1020.4, 1020.5, run)
    gen = trace.start("generation", 1021.0, run, generation=1)
    trace.span("sandbox_reset", 1021.0, 1022.0, gen, backend="codex")
    p1 = trace.start("proposal", 1022.0, gen, candidate_id="g1-s1", parent="g0-s0")
    p2 = trace.start("proposal", 1022.0, gen, candidate_id="g1-s2", parent="g0-s0")
    c1 = trace.start("cached_evaluate", 1022.0, p1, candidate_id="g0-s0", split="train", examples=8)
    c2 = trace.start("cached_evaluate", 1022.0, p2, candidate_id="g0-s0", split="train", examples=8)
    e1 = trace.start("evaluate", 1022.5, c1, candidate_id="g0-s0", split="train", examples=8)
    if upto < 1200:
        trace.write(root)
        return
    trace.end(e1, 1200.0)
    trace.end(c1, 1200.0)
    m1 = trace.start("mutate", 1200.0, p1)
    trace.span("clone", 1200.0, 1200.5, m1)
    e2 = trace.start("evaluate", 1200.2, c2, candidate_id="g0-s0", split="train", examples=8)
    trace.span("snapshot", 1200.5, 1201.0, m1, candidate_id="g1-s1")
    trace.start("agent", 1201.0, m1, backend="codex", model="m")
    trace.end(e2, 1400.0)
    trace.end(c2, 1400.0)
    m2 = trace.start("mutate", 1400.0, p2)
    trace.span("clone", 1400.0, 1400.5, m2)
    trace.start("agent", 1401.0, m2, backend="codex", model="m")
    trace.write(root)


def test_a_dataset_runs_gate_size_before_any_gate_is_the_minibatch(tmp_path: Path) -> None:
    config = _config(max_generations=8).model_copy(update={"dataset": DatasetConfig(train_size=20)})
    _evaluation(tmp_path, "g0-s0", [1.0, 0.5, 0.0, 1.0])
    _lineage(tmp_path, "g0-s0", None, 0)
    _state(tmp_path, config, 0, ["g0-s0"], budget=BudgetState(evaluations=4, input_tokens=10),
           active_frontier={v: ["g0-s0"] for v in VAL})
    _seed_only_with_two_agents(tmp_path, config, upto=1401.0)
    snap = RunReader(tmp_path, config).snapshot(now=1500.0)
    assert snap is not None and snap["loop"]["gate"]["ids_each"] == 8


def test_seed_only_with_two_agents_running(tmp_path: Path) -> None:
    config = _config(max_generations=8)
    _evaluation(tmp_path, "g0-s0", [1.0, 0.5, 0.0, 1.0])
    _lineage(tmp_path, "g0-s0", None, 0)
    _state(tmp_path, config, 0, ["g0-s0"], budget=BudgetState(evaluations=4, input_tokens=10),
           active_frontier={v: ["g0-s0"] for v in VAL})
    _seed_only_with_two_agents(tmp_path, config, upto=1401.0)
    snap = RunReader(tmp_path, config).snapshot(now=1500.0)
    assert snap is not None

    assert snap["status"]["state"] == "running" and snap["status"]["detail"] is None
    assert snap["generation"] == {"current": 1, "max": 8}
    assert snap["elapsed_seconds"] == pytest.approx(500.0)
    assert snap["evaluations"] == {"used": 4, "max": None}
    assert snap["time_left_seconds"] is None and snap["time_left_basis"] is None
    assert snap["spend"]["has_cost"] is False and snap["spend"]["input_tokens"] == 10
    assert snap["config_changed"] is False
    assert snap["binary_scores"] is False
    assert snap["best"] == {"id": "g0-s0", "mean": 0.625, "n": 4, "successes": None}
    assert snap["objective_bests"] is None
    assert snap["progress"]["points"] == [{"id": "g0-s0", "generation": 0, "mean": 0.625}]
    assert snap["progress"]["steps"] == snap["progress"]["points"]
    assert snap["frontier"] == {"ids": ["g0-s0"], "accepted": 1, "type": "instance"}
    [row] = snap["candidates"]
    assert row == {
        "id": "g0-s0", "generation": 0, "parent": None, "parents": [], "operation": "seed",
        "mean": 0.625, "n": 4, "front": True, "keys": 4, "sole": 4, "zero": 1,
        "objective_means": None, "note": None, "agent_seconds": None,
        "validation_seconds": pytest.approx(20.0),
    }
    loop = snap["loop"]
    assert loop["frontier"] == {"size": 1, "accepted": 1}
    assert loop["mutation"] == {"running": 2, "median_seconds": None, "finished": 0}
    # No train set: each gate scores the single task once, whatever minibatch_size says.
    assert loop["gate"] == {"passed": 0, "tried": 0, "ids_each": 1}
    assert loop["validation"] == {"median_seconds": pytest.approx(20.0), "count": 1, "ids": 4, "running": 0}

    now = snap["now"]
    assert now["generation"] == 1 and now["started_at"] == 1021.0 and now["typical_seconds"] is None
    l1, l2 = now["lanes"]
    assert (l1["id"], l1["parent"], l1["phase"], l1["phase_started_at"]) == ("g1-s1", "g0-s0", "agent", 1200.0)
    assert l1["segments"] == [{"kind": "eval", "start": 1022.5, "end": 1200.0},
                              {"kind": "agent", "start": 1201.0, "end": None}]
    assert (l2["id"], l2["phase"], l2["phase_started_at"]) == ("g1-s2", "agent", 1400.0)
    assert l2["segments"] == [{"kind": "eval", "start": 1200.2, "end": 1400.0},
                              {"kind": "agent", "start": 1401.0, "end": None}]
    assert l1["result"] is None and l2["result"] is None

    health = snap["health"]
    assert health["since_best"] == 0 and health["pass_rates"] == []
    assert health["pass_recent"] is None and health["pass_all"] is None
    split = {s["kind"]: s["seconds"] for s in health["time_split"]}
    assert split["mutation"] == pytest.approx((1500 - 1201) + (1500 - 1401))
    assert split["gate"] == pytest.approx((1200 - 1022.5) + (1400 - 1200.2))
    assert split["validation"] == pytest.approx(19.5)
    assert split["other"] == pytest.approx(0.2 + 0.1 + 1.0 + 0.5 + 0.5 + 0.5)
    assert health["failures"] == {"agent_timeouts": 0, "agent_errors": 0, "evaluator_errors": 0,
                                  "evaluator_error_types": {}}


def test_phases_before_the_agents_then_incremental_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(max_generations=8)
    _evaluation(tmp_path, "g0-s0", [1.0, 0.5, 0.0, 1.0])
    _state(tmp_path, config, 0, ["g0-s0"])
    _seed_only_with_two_agents(tmp_path, config, upto=1100.0)
    reader = RunReader(tmp_path, config)
    snap = reader.snapshot(now=1100.0)
    assert snap is not None
    phases = {lane["id"]: (lane["phase"], lane["phase_started_at"]) for lane in snap["now"]["lanes"]}
    # g1-s1's parent minibatch is evaluating; g1-s2 waits on the worktree lock.
    assert phases == {"g1-s1": ("parent_eval", 1022.0), "g1-s2": ("waiting", 1022.0)}
    first = len(reader.timing_records())

    calls: list[Any] = []
    original = EvalResult.from_dict

    def counted(cls: type[EvalResult], data: dict[str, Any]) -> EvalResult:
        calls.append(data)
        return original(data)

    monkeypatch.setattr(EvalResult, "from_dict", classmethod(counted))
    # The rest of the generation, appended to the same file.
    path = tmp_path / ".helix" / "timing.jsonl"
    # Hold the old file open by a second name, so the new one gets a new inode
    # (Linux reuses a freed inode number at once; macOS does not).
    os.link(path, tmp_path / "old-timing.jsonl")
    path.unlink()
    _seed_only_with_two_agents(tmp_path, config, upto=1401.0)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:first]) + "\n")  # a new file: read from 0 ...
    reader.snapshot(now=1100.0)
    before = reader.timing_records()
    with path.open("a") as fh:  # ... then the new records
        fh.write("\n".join(lines[first:]) + "\n")
    snap = reader.snapshot(now=1500.0)
    assert snap is not None
    assert [lane["phase"] for lane in snap["now"]["lanes"]] == ["agent", "agent"]
    after = reader.timing_records()
    assert len(after) == len(lines)
    assert after[first - 1] is before[first - 1]  # read on from the offset, not from 0
    assert calls == []  # the unchanged evaluation file was not re-read


def test_torn_last_line_and_truncation(tmp_path: Path) -> None:
    config = _config()
    trace = Trace()
    run = trace.start("run", 1.0)
    trace.span("save_state", 1.0, 2.0, run)
    trace.write(tmp_path)
    path = tmp_path / ".helix" / "timing.jsonl"
    whole = json.dumps({**run, "event": "end", "wall_time": 5.0, "duration_seconds": 4.0, "outcome": "ok"})
    with path.open("a") as fh:
        fh.write("{not json\n" + whole[:20])  # a garbage line, then a torn last line
    reader = RunReader(tmp_path, config)
    snap = reader.snapshot(now=10.0)
    assert snap is not None and snap["status"]["state"] == "running"
    assert len(reader.timing_records()) == 3

    with path.open("a") as fh:  # the writer finishes the line
        fh.write(whole[20:] + "\n")
    snap = reader.snapshot(now=10.0)
    assert snap is not None and snap["status"]["state"] == "finished"
    assert snap["elapsed_seconds"] == pytest.approx(4.0)

    trace = Trace("r2")  # truncated and rewritten: read again from the start
    trace.start("run", 50.0)
    path.write_text("")
    trace.write(tmp_path)
    snap = reader.snapshot(now=60.0)
    assert snap is not None and snap["status"]["state"] == "running"
    assert [r["run_id"] for r in reader.timing_records()] == ["r2"]
    assert snap["elapsed_seconds"] == pytest.approx(10.0)


def _proposal(trace: Trace, gen: dict[str, Any], t: float, child: str, parent: str, *,
              agent: float, agent_error: str | None = None, child_eval: bool = True,
              eval_error: str | None = None, summarize: bool = False) -> float:
    """One proposal: parent minibatch 10 s, agent, child minibatch 10 s; returns its end time."""
    p = trace.start("proposal", t, gen, candidate_id=child, parent=parent)
    c = trace.start("cached_evaluate", t, p, candidate_id=parent, split="train", examples=3)
    trace.span("evaluate", t, t + 10, c, candidate_id=parent, split="train", examples=3)
    trace.end(c, t + 10)
    m = trace.start("mutate", t + 10, p)
    trace.span("agent", t + 10, t + 10 + agent, m, error=agent_error, backend="claude", model="m")
    t += 10 + agent
    if summarize:
        s = trace.start("summarize", t, m)
        trace.span("agent", t, t + 5, s, backend="claude", model="m")
        trace.end(s, t + 5)
        t += 5
    trace.end(m, t)
    if child_eval:
        c = trace.start("cached_evaluate", t, p, candidate_id=child, split="train", examples=3)
        trace.span("evaluate", t, t + 10, c, error=eval_error, candidate_id=child, split="train", examples=3)
        trace.end(c, t + 10, eval_error)
        t += 10
    trace.end(p, t)
    return t


def _validate(trace: Trace, gen: dict[str, Any], t: float, cid: str, seconds: float) -> None:
    v = trace.start("validate", t, gen, candidate_id=cid)
    c = trace.start("cached_evaluate", t, v, candidate_id=cid, split="val", examples=4)
    trace.span("evaluate", t, t + seconds, c, candidate_id=cid, split="val", examples=4)
    trace.end(c, t + seconds)
    trace.end(v, t + seconds)


def _finished_run(root: Path, config: HelixConfig) -> None:
    """Three generations: accepted, gate-rejected, timed out, agent error,
    evaluator error and val-stage-rejected proposals; binary scores."""
    _evaluation(root, "g0-s0", [1, 0, 0, 0])
    _evaluation(root, "g1-s1", [1, 1, 0, 0])
    _evaluation(root, "g3-s6", [0, 1, 1, 1])
    for cid, parent, g in [("g0-s0", None, 0), ("g1-s1", "g0-s0", 1), ("g1-s2", "g0-s0", 1),
                           ("g3-s6", "g1-s1", 3), ("g3-s7", "g1-s1", 3)]:
        _lineage(root, cid, parent, g)
    _attempt(root, "g1-s2", "g0-s0", 1, "minibatch_gate", "train_minibatch", {"a__1": 1, "a__2": 0, "b__1": 0})
    _attempt(root, "g3-s7", "g1-s1", 3, "val_stage", "val_stage", {"a__1": 0, "a__2": 0})
    history = {"g0-s0": [{"child_id": "g1-s1", "score": 1.0, "accepted": True,
                          "summary": "Retry the grasp. key sk-abcdefghijklmnopqrstuvwx"}]}
    _state(root, config, 3, ["g0-s0", "g1-s1", "g3-s6"], sibling_history=history,
           budget=BudgetState(evaluations=40, cost_usd=3.0, input_tokens=100, output_tokens=7))

    trace = Trace()
    run = trace.start("run", 0.0)
    _seed(trace, run, 0.0)
    # generation 1: 100 s
    gen = trace.start("generation", 100.0, run, generation=1)
    _proposal(trace, gen, 100.0, "g1-s1", "g0-s0", agent=50, summarize=True)
    _proposal(trace, gen, 100.0, "g1-s2", "g0-s0", agent=40)
    trace.decision("g1-s2", 175.0, "minibatch-gate-rejected candidate", False, gen)
    _validate(trace, gen, 175.0, "g1-s1", 20.0)
    trace.decision("g1-s1", 195.0, "mutation", True, gen)
    trace.end(gen, 200.0)
    # generation 2: 200 s, nothing kept
    gen = trace.start("generation", 200.0, run, generation=2)
    _proposal(trace, gen, 200.0, "g2-s3", "g1-s1", agent=99, agent_error="MutationError", child_eval=False)
    _proposal(trace, gen, 200.0, "g2-s4", "g1-s1", agent=30, eval_error="EvaluatorError")
    _proposal(trace, gen, 200.0, "g2-s5", "g1-s1", agent=10, agent_error="MutationError", child_eval=False)
    trace.decision("g2-s3", 310.0, "failed mutation", False, gen)
    trace.decision("g2-s4", 310.0, "child whose minibatch eval raised", False, gen)
    trace.decision("g2-s5", 310.0, "failed mutation", False, gen)
    trace.end(gen, 400.0)
    # generation 3: 300 s
    gen = trace.start("generation", 400.0, run, generation=3)
    _proposal(trace, gen, 400.0, "g3-s6", "g1-s1", agent=60)
    _proposal(trace, gen, 400.0, "g3-s7", "g1-s1", agent=70)
    trace.span("cached_evaluate", 500.0, 505.0, gen, candidate_id="g3-s7", split="val",
               examples=2, evaluation_phase="val_stage")
    trace.decision("g3-s7", 506.0, "val-stage-rejected candidate", False, gen)
    _validate(trace, gen, 510.0, "g3-s6", 30.0)
    trace.decision("g3-s6", 540.0, "mutation", True, gen)
    trace.end(gen, 700.0)
    trace.end(run, 701.0)
    trace.write(root)


def test_finished_multi_generation_run(tmp_path: Path) -> None:
    config = _config()
    _finished_run(tmp_path, config)
    reader = RunReader(tmp_path, config)
    snap = reader.snapshot(now=5000.0)
    assert snap is not None

    assert snap["status"] == {"state": "finished", "detail": None, "stale_seconds": snap["status"]["stale_seconds"]}
    assert snap["generation"] == {"current": 3, "max": 3}
    assert snap["elapsed_seconds"] == pytest.approx(701.0)
    assert snap["time_left_seconds"] is None
    assert snap["binary_scores"] is True
    assert snap["best"] == {"id": "g3-s6", "mean": 0.75, "n": 4, "successes": 3}
    assert snap["spend"] == {
        "cost_usd": 3.0, "input_tokens": 100, "output_tokens": 7, "cached_input_tokens": 0,
        "projected_cost_usd": None, "cost_per_generation": 1.0, "cost_per_accepted": 1.5, "has_cost": True,
        # A run without usage records keeps the run-total estimate.
        "per_record": False, "by_generation": [], "by_outcome": None,
    }
    assert [p["id"] for p in snap["progress"]["points"]] == ["g0-s0", "g1-s1", "g3-s6"]
    assert [(s["id"], s["generation"]) for s in snap["progress"]["steps"]] == [
        ("g0-s0", 0), ("g1-s1", 1), ("g3-s6", 3)]
    assert snap["frontier"] == {"ids": ["g3-s6", "g1-s1"], "accepted": 3, "type": "instance"}
    seed = snap["seed"]
    assert seed["id"] == "g0-s0" and seed["n"] == 4 and seed["objective_means"] is None
    assert seed["mean"] == snap["progress"]["points"][0]["mean"] and seed["successes"] == round(seed["mean"] * 4)
    assert snap["progress"]["objectives"] is None

    rows = {r["id"]: r for r in snap["candidates"]}
    # Fronts: a__1 {g0-s0, g1-s1}, a__2 {g1-s1, g3-s6}, b__1 and b__2 {g3-s6};
    # GEPA's elimination drops g0-s0.
    assert {c: (r["front"], r["keys"], r["sole"], r["zero"]) for c, r in rows.items()} == {
        "g0-s0": (False, 1, 0, 3), "g1-s1": (True, 2, 1, 2), "g3-s6": (True, 3, 2, 1)}
    assert rows["g1-s1"]["parent"] == "g0-s0" and rows["g1-s1"]["operation"] == "mutate"
    assert rows["g1-s1"]["note"] == "Retry the grasp. key ••••"
    assert rows["g1-s1"]["agent_seconds"] == pytest.approx(55.0)  # mutation + change summary
    assert rows["g1-s1"]["validation_seconds"] == pytest.approx(20.0)
    assert rows["g3-s6"]["validation_seconds"] == pytest.approx(30.0)

    loop = snap["loop"]
    assert loop["mutation"] == {"running": 0, "median_seconds": 50.0, "finished": 7}
    assert loop["gate"] == {"passed": 3, "tried": 4, "ids_each": 3}
    assert loop["validation"]["count"] == 4  # seed, g1-s1, g3-s6, and g3-s7's val stage
    assert loop["validation"]["median_seconds"] == pytest.approx(20.0)

    now = snap["now"]
    assert now["generation"] == 3 and now["started_at"] == 400.0 and now["typical_seconds"] == 200.0
    lanes = {lane["id"]: lane for lane in now["lanes"]}
    assert lanes["g3-s6"]["phase"] == "done" and lanes["g3-s6"]["phase_started_at"] == 540.0
    assert lanes["g3-s6"]["result"] == {"status": "accepted", "reason": "mutation", "score": 0.75,
                                        "n": 4, "successes": 3, "stage": None}
    assert lanes["g3-s7"]["result"] == {"status": "rejected", "reason": "val_stage", "score": 0.0,
                                        "n": 2, "successes": 0, "stage": "val_stage"}
    assert [s["kind"] for s in lanes["g3-s6"]["segments"]] == ["eval", "agent", "eval", "eval"]

    health = snap["health"]
    assert health["since_best"] == 0
    assert health["pass_rates"] == [0.5, 1.0]
    assert health["pass_recent"] == 0.75 and health["pass_all"] == 0.75
    assert health["failures"] == {"agent_timeouts": 1, "agent_errors": 1, "evaluator_errors": 1,
                                  "evaluator_error_types": {"EvaluatorError": 1}}
    split = {s["kind"]: s["seconds"] for s in health["time_split"]}
    assert split["mutation"] == pytest.approx(50 + 5 + 40 + 99 + 30 + 10 + 60 + 70)
    assert split["gate"] == pytest.approx(10 * 7 + 10 * 5)
    assert split["validation"] == pytest.approx(19.5 + 20 + 30)

    front = reader.frontier()
    assert front is not None
    assert front["ids"] == ["g3-s6", "g1-s1"]
    assert front["instances"] == VAL and front["groups"] == ["a", "b"]
    assert front["best"] == [1, 1, 1, 1]
    assert front["held"] == {"g3-s6": [1, 2, 3], "g1-s1": [0, 1]}
    assert front["objectives"] is None and front["all_objective_means"] is None
    assert (front["threshold"], front["below_threshold"]) == (1.0, 0)
    assert reader.candidate_ids() == {"g0-s0", "g1-s1", "g1-s2", "g2-s3", "g2-s4", "g2-s5", "g3-s6", "g3-s7"}


def test_failed_lane_and_gate_rejected_lane(tmp_path: Path) -> None:
    config = _config()
    _finished_run(tmp_path, config)
    # Re-read generation 2 as the current one: drop gen 3 from the trace.
    path = tmp_path / ".helix" / "timing.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(
        json.dumps(r) + "\n" for r in records
        if not (r.get("generation") == 3 or (r.get("span") == "run" and r["event"] == "end"))
    ))
    snap = RunReader(tmp_path, config).snapshot(now=450.0)
    assert snap is not None and snap["now"]["generation"] == 2
    results = {lane["id"]: lane["result"] for lane in snap["now"]["lanes"]}
    assert results["g2-s3"] == {"status": "failed", "reason": "failed mutation", "score": None,
                                "n": None, "successes": None, "stage": None}
    assert results["g2-s4"]["reason"] == "child whose minibatch eval raised"


def test_running_projection_by_generations_and_evaluations(tmp_path: Path) -> None:
    config = _config(max_generations=5)
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _state(tmp_path, config, 2, ["g0-s0"], num_metric_calls_by_discovery={"g0-s0": 20},
           budget=BudgetState(evaluations=100, cost_usd=4.0))
    trace = Trace()
    run = trace.start("run", 0.0)
    trace.span("generation", 0.0, 100.0, run, generation=1)
    trace.span("generation", 100.0, 400.0, run, generation=2)
    trace.start("generation", 400.0, run, generation=3)
    trace.write(tmp_path)

    snap = RunReader(tmp_path, config).snapshot(now=450.0)
    assert snap is not None
    assert snap["now"]["typical_seconds"] == 200.0
    # (200 - 50 into gen 3) + 2 generations after it
    assert (snap["time_left_seconds"], snap["time_left_basis"]) == (550.0, "generations")
    assert snap["spend"]["cost_per_generation"] == 2.0
    assert snap["spend"]["projected_cost_usd"] == 10.0
    assert snap["health"]["since_best"] == 2

    capped = _config(max_generations=5, max_evaluations=180)
    _state(tmp_path, capped, 2, ["g0-s0"], num_metric_calls_by_discovery={"g0-s0": 20},
           budget=BudgetState(evaluations=100))
    snap = RunReader(tmp_path, capped).snapshot(now=450.0)
    assert snap is not None
    # 80 evaluations left at 40 per generation: 2 generations of 200 s.
    assert (snap["time_left_seconds"], snap["time_left_basis"]) == (400.0, "evaluations")
    assert snap["evaluations"] == {"used": 100, "max": 180}
    assert snap["config_changed"] is False
    assert RunReader(tmp_path, config).snapshot(now=450.0)["config_changed"] is True  # type: ignore[index]


def test_run_past_the_toml_generation_limit_has_no_denominator(tmp_path: Path) -> None:
    """`helix evolve --generations 6` on a TOML saying 2 is not recorded anywhere."""
    config = _config(max_generations=2)
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _state(tmp_path, config, 2, ["g0-s0"], budget=BudgetState(evaluations=100, cost_usd=4.0))
    trace = Trace()
    run = trace.start("run", 0.0)
    trace.span("generation", 0.0, 100.0, run, generation=1)
    trace.span("generation", 100.0, 400.0, run, generation=2)
    trace.start("generation", 400.0, run, generation=3)
    trace.write(tmp_path)

    snap = RunReader(tmp_path, config).snapshot(now=450.0)
    assert snap is not None
    assert snap["generation"] == {"current": 3, "max": None}
    assert (snap["time_left_seconds"], snap["time_left_basis"]) == (None, None)
    assert snap["spend"]["projected_cost_usd"] is None
    assert snap["spend"]["cost_per_generation"] == 2.0


def test_hybrid_frontier_with_objective_scores(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = _config(frontier_type="hybrid")
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0], [{"x": 1.0, "y": 0.0}] * 4)
    _evaluation(tmp_path, "g1-s1", [0, 0.5, 1, 1], [{"x": 0.5, "y": 1.0}] * 4)
    _evaluation(tmp_path, "g2-s2", [0, 0, 0, 0])  # no objective axes: HELIX warns, the UI must not
    _state(tmp_path, config, 2, ["g0-s0", "g1-s1", "g2-s2"], frontier_type="hybrid")
    reader = RunReader(tmp_path, config)
    with caplog.at_level(logging.WARNING):
        snap = reader.snapshot(now=1.0)
        front = reader.frontier()
    assert not [r for r in caplog.records if r.name == "helix.population"]
    assert logging.getLogger("helix.population").filters == []
    assert snap is not None and front is not None

    assert snap["status"]["state"] == "stopped"  # state.json without a timing trace
    assert snap["frontier"]["type"] == "hybrid"
    assert snap["objective_bests"] == [{"name": "x", "id": "g0-s0", "mean": 1.0},
                                       {"name": "y", "id": "g1-s1", "mean": 1.0}]
    assert sorted(front["ids"]) == ["g0-s0", "g1-s1"]
    assert front["objectives"] == ["x", "y"]
    assert front["objective_means"] == {"g1-s1": [0.5, 1.0], "g0-s0": [1.0, 0.0]}
    assert front["all_objective_means"]["g2-s2"] == [None, None]
    rows = {r["id"]: r for r in snap["candidates"]}
    # hybrid keys: 4 instances + 2 objectives
    assert rows["g0-s0"]["keys"] == 2 and rows["g1-s1"]["keys"] == 4
    assert rows["g1-s1"]["objective_means"] == {"x": 0.5, "y": 1.0}
    assert front["threshold"] is None and front["below_threshold"] is None
    assert snap["seed"] == {"id": "g0-s0", "mean": 0.25, "n": 4, "successes": None,
                            "objective_means": {"x": 1.0, "y": 0.0}}
    # Per-objective progress: every accepted candidate with the objective, and its running best.
    objectives = {o["name"]: o for o in snap["progress"]["objectives"]}
    assert list(objectives) == ["x", "y"]
    assert [(p["id"], p["mean"]) for p in objectives["x"]["points"]] == [("g0-s0", 1.0), ("g1-s1", 0.5)]
    assert [s["id"] for s in objectives["x"]["steps"]] == ["g0-s0"]
    assert [(s["id"], s["generation"], s["mean"]) for s in objectives["y"]["steps"]] == [
        ("g0-s0", 0, 0.0), ("g1-s1", 1, 1.0)]
    lineage = reader.lineage()
    assert lineage["objectives"] == ["x", "y"]
    nodes = {n["id"]: n for n in lineage["nodes"]}
    assert nodes["g1-s1"]["objective_means"] == {"x": 0.5, "y": 1.0}
    assert nodes["g2-s2"]["objective_means"] is None


def test_resume_reads_the_latest_run(tmp_path: Path) -> None:
    config = _config()
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _state(tmp_path, config, 0, ["g0-s0"])
    first = Trace("r1")
    run = first.start("run", 0.0)
    _seed(first, run, 0.0)
    gen = first.start("generation", 30.0, run, generation=1)
    p = first.start("proposal", 30.0, gen, candidate_id="g1-s1", parent="g0-s0")
    m = first.start("mutate", 30.0, p)
    agent = first.start("agent", 30.0, m, backend="claude", model="m")
    for record, t in [(agent, 90.0), (m, 90.0), (p, 90.0), (gen, 90.0), (run, 90.0)]:
        first.end(record, t, "KeyboardInterrupt")
    first.write(tmp_path)
    reader = RunReader(tmp_path, config)
    snap = reader.snapshot(now=95.0)
    assert snap is not None
    assert snap["status"]["state"] == "stopped" and snap["status"]["detail"] == "KeyboardInterrupt"
    assert snap["health"]["failures"]["agent_errors"] == 0  # Ctrl-C is not an agent failure

    second = Trace("r2")
    run = second.start("run", 200.0)
    second.decision("g1-s1", 200.5, "orphan worktree from prior run", False, run)
    gen = second.start("generation", 201.0, run, generation=1)
    p = second.start("proposal", 201.0, gen, candidate_id="g1-s2", parent="g0-s0")
    second.start("mutate", 211.0, p)
    second.write(tmp_path)
    snap = reader.snapshot(now=300.0)
    assert snap is not None
    assert snap["status"]["state"] == "running"
    assert snap["generation"]["current"] == 1
    assert snap["elapsed_seconds"] == pytest.approx(90.0 + 100.0)
    assert [lane["id"] for lane in snap["now"]["lanes"]] == ["g1-s2"]
    assert snap["now"]["lanes"][0]["phase"] == "agent"
    assert snap["loop"]["mutation"]["running"] == 0  # the old run's agent is closed

    crashed = Trace("r2")
    crashed.end(run, 310.0, "RuntimeError")
    crashed.write(tmp_path)
    snap = reader.snapshot(now=320.0)
    assert snap is not None and snap["status"]["state"] == "crashed"
    assert snap["status"]["detail"] == "RuntimeError"


def test_detail_records_are_totalled_per_name(tmp_path: Path) -> None:
    """Agent transcript and evaluator ``helix.timed`` details reach the snapshot, summed per name."""
    from helix.ui.run_state import _details

    records = [
        {"event": "detail", "within": "evaluate", "source": "timed", "name": "MuJoCo", "count": 3, "seconds": 1.5},
        {"event": "detail", "within": "evaluate", "source": "timed", "name": "MuJoCo", "count": 1, "seconds": 0.5},
        {"event": "detail", "within": "agent", "source": "transcript", "name": "tool:exec", "count": 2, "seconds": 4.0},
        {"event": "start", "span": "run"},
    ]
    assert _details(records) == [
        {"within": "evaluate", "source": "timed", "name": "MuJoCo", "seconds": 2.0, "count": 4},
        {"within": "agent", "source": "transcript", "name": "tool:exec", "seconds": 4.0, "count": 2},
    ]


def test_projected_spend_stops_at_the_evaluation_budget(tmp_path: Path) -> None:
    """circle_packing: ``max_generations=1000`` is a guard; ``max_evaluations=200`` ends the run."""
    config = _config(max_generations=1000, max_evaluations=200)
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _state(tmp_path, config, 10, ["g0-s0"], num_metric_calls_by_discovery={"g0-s0": 4},
           budget=BudgetState(evaluations=54, cost_usd=1.0))
    trace = Trace()
    run = trace.start("run", 0.0)
    for g in range(1, 11):
        trace.span("generation", 10.0 * g, 10.0 * g + 10, run, generation=g)
    trace.start("generation", 110.0, run, generation=11)
    trace.write(tmp_path)

    snap = RunReader(tmp_path, config).snapshot(now=110.0)
    assert snap is not None
    assert snap["time_left_basis"] == "evaluations"
    # 5 evaluations a generation: 146 left is 29.2 generations at $0.10 each.
    assert snap["spend"]["projected_cost_usd"] == pytest.approx(1.0 + 0.1 * 29.2)


def test_interrupted_agent_is_not_counted_again_after_resume(tmp_path: Path) -> None:
    """Ctrl-C ends the agent span with KeyboardInterrupt; ``helix resume`` reuses the candidate id."""
    config = _config()
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _evaluation(tmp_path, "g1-s1", [1, 1, 0, 0])
    _lineage(tmp_path, "g0-s0", None, 0)
    _lineage(tmp_path, "g1-s1", "g0-s0", 1)
    _state(tmp_path, config, 1, ["g0-s0", "g1-s1"])
    first = Trace("r1")
    run = first.start("run", 0.0)
    _seed(first, run, 0.0)
    gen = first.start("generation", 30.0, run, generation=1)
    p = first.start("proposal", 30.0, gen, candidate_id="g1-s1", parent="g0-s0")
    m = first.start("mutate", 30.0, p)
    agent = first.start("agent", 30.0, m, backend="claude", model="m")
    for record in (agent, m, p, gen, run):
        first.end(record, 90.0, "KeyboardInterrupt")
    first.write(tmp_path)
    second = Trace("r2")
    run = second.start("run", 200.0)
    gen = second.start("generation", 200.0, run, generation=1)
    _proposal(second, gen, 200.0, "g1-s1", "g0-s0", agent=50)
    _validate(second, gen, 270.0, "g1-s1", 20.0)
    second.decision("g1-s1", 290.0, "mutation", True, gen)
    second.end(gen, 290.0)
    second.end(run, 291.0)
    second.write(tmp_path)

    snap = RunReader(tmp_path, config).snapshot(now=300.0)
    assert snap is not None
    rows = {r["id"]: r for r in snap["candidates"]}
    assert rows["g1-s1"]["agent_seconds"] == pytest.approx(50.0)
    assert snap["loop"]["mutation"]["median_seconds"] == pytest.approx(50.0)
    assert snap["loop"]["mutation"]["finished"] == 1


def test_merge_generation_has_a_lane(tmp_path: Path) -> None:
    """A merge generation has a ``merge`` span (merger.py:200) and no ``proposal``."""
    config = _config(merge_enabled=True)
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _evaluation(tmp_path, "g1-s1", [1, 1, 0, 0])
    _evaluation(tmp_path, "g1-s2", [0, 0, 1, 1])
    _state(tmp_path, config, 1, ["g0-s0", "g1-s1", "g1-s2"])
    trace = Trace()
    run = trace.start("run", 0.0)
    _seed(trace, run, 0.0)
    gen = trace.start("generation", 100.0, run, generation=2)
    merge = trace.start("merge", 100.0, gen, candidate_id="g2-m1", parents=["g1-s1", "g1-s2"])
    trace.span("clone", 100.0, 101.0, merge)
    agent = trace.start("agent", 101.0, merge, backend="claude", model="m")
    trace.write(tmp_path)
    reader = RunReader(tmp_path, config)

    snap = reader.snapshot(now=150.0)
    assert snap is not None
    assert snap["loop"]["mutation"]["running"] == 1
    [lane] = snap["now"]["lanes"]
    assert (lane["id"], lane["parent"], lane["phase"], lane["phase_started_at"]) == (
        "g2-m1", "g1-s1", "agent", 101.0)
    assert [s["kind"] for s in lane["segments"]] == ["agent"] and lane["result"] is None

    trace.end(agent, 160.0)
    trace.end(merge, 160.0)
    trace.span("cached_evaluate", 161.0, 170.0, gen, candidate_id="g2-m1", split="val", examples=2)
    trace.decision("g2-m1", 171.0, "score-rejected merge candidate", False, gen)
    trace.end(gen, 172.0)
    trace.write(tmp_path)
    _attempt(tmp_path, "g2-m1", "g1-s1", 2, "merge_below_parents", "merge_subsample", {"a__1": 1, "b__1": 0})
    snap = reader.snapshot(now=175.0)
    assert snap is not None
    [lane] = snap["now"]["lanes"]
    assert lane["phase"] == "done" and lane["phase_started_at"] == 171.0
    assert lane["result"] == {"status": "rejected", "reason": "merge_below_parents", "score": 0.5,
                              "n": 2, "successes": 1, "stage": "merge_subsample"}


# ---------------------------------------------------------------------------
# lineage, proposals, instance history
# ---------------------------------------------------------------------------


def test_proposals_of_a_finished_run(tmp_path: Path) -> None:
    config = _config()
    _finished_run(tmp_path, config)
    out = RunReader(tmp_path, config).proposals()
    rows = {r["id"]: r for r in out["rows"]}
    # helix log order: (generation, trailing id number, id)
    assert [r["id"] for r in out["rows"]] == [
        "g0-s0", "g1-s1", "g1-s2", "g2-s3", "g2-s4", "g2-s5", "g3-s6", "g3-s7"]
    assert out["counts"] == {"frontier": 2, "dominated": 1, "rejected": 2, "skipped": 0,
                             "failed": 3, "running": 0}

    seed = rows["g0-s0"]
    assert (seed["operation"], seed["outcome"], seed["basis"], seed["delta"]) == ("seed", "dominated", "val", None)
    assert rows["g1-s1"] == {
        "id": "g1-s1", "generation": 1, "parents": ["g0-s0"], "operation": "mutate",
        "outcome": "frontier", "reason": None, "stage": None, "score": 0.5, "n": 4, "successes": 2,
        "basis": "val", "delta": 0.25, "files": 0, "summary": "Retry the grasp.",
        "agent_seconds": pytest.approx(55.0), "usage": None,
    }
    assert rows["g3-s6"]["delta"] == pytest.approx(0.25)
    # Gate scores carry their basis and never a delta against the parent's validation score.
    gate = rows["g1-s2"]
    assert (gate["outcome"], gate["reason"], gate["stage"], gate["basis"]) == (
        "rejected", "minibatch_gate", "train_minibatch", "train_minibatch")
    assert gate["score"] == pytest.approx(1 / 3) and gate["n"] == 3 and gate["successes"] == 1
    assert gate["delta"] is None
    val_stage = rows["g3-s7"]
    assert (val_stage["basis"], val_stage["score"], val_stage["delta"]) == ("val_stage", 0.0, None)
    # Failed proposals: a decision only, parent from the proposal span.
    assert {c: (rows[c]["outcome"], rows[c]["reason"], rows[c]["parents"], rows[c]["generation"])
            for c in ("g2-s3", "g2-s4")} == {
        "g2-s3": ("failed", "failed mutation", ["g1-s1"], 2),
        "g2-s4": ("failed", "child whose minibatch eval raised", ["g1-s1"], 2)}
    assert rows["g2-s3"]["score"] is None and rows["g2-s3"]["basis"] is None
    assert rows["g2-s3"]["files"] is None and rows["g2-s3"]["agent_seconds"] == pytest.approx(99.0)


def test_lineage_of_a_finished_run(tmp_path: Path) -> None:
    config = _config()
    _finished_run(tmp_path, config)
    out = RunReader(tmp_path, config).lineage()
    nodes = {n["id"]: n for n in out["nodes"]}
    assert list(nodes) == ["g0-s0", "g1-s1", "g1-s2", "g3-s6", "g3-s7"]  # failed ones are not nodes
    assert (out["best"], out["max_generation"], out["truncated"]) == ("g3-s6", 3, 0)
    assert {c for c, n in nodes.items() if n["best_path"]} == {"g0-s0", "g1-s1", "g3-s6"}
    assert nodes["g3-s6"] == {"id": "g3-s6", "generation": 3, "parents": ["g1-s1"], "operation": "mutate",
                              "outcome": "frontier", "best_path": True, "mean": 0.75, "front": True,
                              "accepted": True, "objective_means": None}
    assert out["objectives"] is None
    assert nodes["g0-s0"]["front"] is False and nodes["g0-s0"]["accepted"] is True
    rejected = nodes["g1-s2"]
    assert rejected["mean"] is None and rejected["accepted"] is False and rejected["front"] is False
    assert rejected["gate_score"] == pytest.approx(1 / 3) and rejected["gate_basis"] == "train_minibatch"
    assert (rejected["outcome"], rejected["reason"], rejected["parents"]) == (
        "rejected", "minibatch_gate", ["g0-s0"])


def _merge_run(root: Path, config: HelixConfig) -> Trace:
    """g2-m3 merges g1-s1 and g1-s2 and is the best; g3 holds a no-change merge,
    two perfect-parent skips and a gate rejection; g4 is running."""
    from helix.evolution import _save_attempt_result, _save_skip_record

    helix = root / ".helix"
    _evaluation(root, "g0-s0", [1, 0, 0, 0])
    _evaluation(root, "g1-s1", [1, 1, 0, 0])
    _evaluation(root, "g1-s2", [1, 0, 1, 0])
    _evaluation(root, "g2-m3", [1, 1, 1, 0])
    for cid, parent, g in [("g0-s0", None, 0), ("g1-s1", "g0-s0", 1), ("g1-s2", "g0-s0", 1)]:
        _lineage(root, cid, parent, g)
    record_entry(helix / "lineage.json", LineageEntry(
        id="g2-m3", parent="g1-s1", parents=["g1-s1", "g1-s2"], operation="merge", generation=2,
        files_changed=["a.py", "b.py"]))
    _lineage(root, "g3-s5", "g2-m3", 3)
    _save_attempt_result(helix, EvalResult(candidate_id="g3-m4", scores={}, asi={}, instance_scores={}),
                         status="rejected", reason="merge_no_change", parent_id="g1-s1", generation=3,
                         stage="merge")
    _save_attempt_result(helix, EvalResult(candidate_id="g3-s5", scores={}, asi={},
                                           instance_scores={"a__1": 0.0, "b__2": 1.0}),
                         status="rejected", reason="minibatch_gate", parent_id="g2-m3", generation=3,
                         stage="train_minibatch", example_ids=["a__1", "b__2"])
    perfect = EvalResult(candidate_id="g2-m3", scores={}, asi={}, instance_scores={"a__1": 1.0, "a__2": 1.0})
    _save_skip_record(helix, generation=3, records=[
        {"generation": 3, "parent_id": "g2-m3", "reason": "perfect_subsample", "parent_eval": perfect.to_dict()}
        for _ in range(2)])
    _state(root, config, 3, ["g0-s0", "g1-s1", "g1-s2", "g2-m3"])

    trace = Trace()
    run = trace.start("run", 0.0)
    _seed(trace, run, 0.0)
    gen = trace.start("generation", 300.0, run, generation=3)
    trace.span("merge", 300.0, 310.0, gen, candidate_id="g3-m4", parents=["g1-s1", "g1-s2"])
    trace.decision("g3-m4", 311.0, "no-change merge candidate", False, gen)
    trace.end(gen, 400.0)
    gen = trace.start("generation", 400.0, run, generation=4)
    p = trace.start("proposal", 400.0, gen, candidate_id="g4-s6", parent="g2-m3")
    trace.start("mutate", 410.0, p)
    trace.start("merge", 400.0, gen, candidate_id="g4-m7", parents=["g2-m3", "g1-s1"])
    trace.write(root)
    return trace


def test_merges_skips_and_running_proposals(tmp_path: Path) -> None:
    config = _config(merge_enabled=True)
    _merge_run(tmp_path, config)
    reader = RunReader(tmp_path, config)
    out = reader.proposals()
    rows = {r["id"]: r for r in out["rows"]}
    assert [r["id"] for r in out["rows"]] == [
        "g0-s0", "g1-s1", "g1-s2", "g2-m3", "g3-m4", "g3-s5", "skip:g3:g2-m3", "skip:g3:g2-m3:2",
        "g4-s6", "g4-m7"]
    # g2-m3 [1,1,1,0] covers every other accepted candidate.
    assert out["counts"] == {"frontier": 1, "dominated": 3, "rejected": 2, "skipped": 2,
                             "failed": 0, "running": 2}
    merge = rows["g2-m3"]
    assert (merge["operation"], merge["parents"], merge["files"]) == ("merge", ["g1-s1", "g1-s2"], 2)
    assert merge["delta"] == pytest.approx(0.25)  # against the first parent
    # No lineage entry: both parents come from the merge span, not only attempt.parent_id.
    no_change = rows["g3-m4"]
    assert (no_change["outcome"], no_change["reason"], no_change["stage"]) == (
        "rejected", "merge_no_change", "merge")
    assert no_change["parents"] == ["g1-s1", "g1-s2"] and no_change["operation"] == "merge"
    assert no_change["score"] is None and no_change["basis"] is None
    skip = rows["skip:g3:g2-m3"]
    assert (skip["outcome"], skip["reason"], skip["parents"], skip["generation"]) == (
        "skipped", "perfect_subsample", ["g2-m3"], 3)
    assert (skip["score"], skip["n"], skip["basis"]) == (1.0, 2, "train_minibatch")
    assert (rows["g4-s6"]["outcome"], rows["g4-s6"]["parents"]) == ("running", ["g2-m3"])
    assert (rows["g4-m7"]["outcome"], rows["g4-m7"]["operation"], rows["g4-m7"]["parents"]) == (
        "running", "merge", ["g2-m3", "g1-s1"])
    # A running merge's candidate page must not 404.
    assert {"g4-m7", "g4-s6", "g3-m4"} <= reader.candidate_ids()

    lineage = reader.lineage()
    nodes = {n["id"]: n for n in lineage["nodes"]}
    assert set(nodes) == {"g0-s0", "g1-s1", "g1-s2", "g2-m3", "g3-m4", "g3-s5"}
    assert lineage["best"] == "g2-m3" and lineage["max_generation"] == 3
    # Both parents' chains of the best merge.
    assert {c for c, n in nodes.items() if n["best_path"]} == {"g0-s0", "g1-s1", "g1-s2", "g2-m3"}
    assert nodes["g2-m3"]["parents"] == ["g1-s1", "g1-s2"]
    assert nodes["g3-s5"]["gate_score"] == 0.5 and nodes["g3-s5"]["mean"] is None


def test_a_crashed_run_has_no_running_proposals(tmp_path: Path) -> None:
    config = _config(merge_enabled=True)
    _merge_run(tmp_path, config)
    records = [json.loads(line) for line in (tmp_path / ".helix" / "timing.jsonl").read_text().splitlines()]
    run = next(r for r in records if r["span"] == "run" if r.get("event") == "start")
    crashed = Trace()
    crashed.end(run, 500.0, "RuntimeError")
    crashed.write(tmp_path)
    rows = {r["id"]: r for r in RunReader(tmp_path, config).proposals()["rows"]}
    # Open spans of a run that ended are not running; with no lineage entry they are not rows.
    assert "g4-s6" not in rows and "g4-m7" not in rows
    # A lineage entry the run never finished is shown, as `helix log` shows "no result".
    (tmp_path / ".helix" / "attempts" / "g3-s5.json").unlink()
    rows = {r["id"]: r for r in RunReader(tmp_path, config).proposals()["rows"]}
    assert (rows["g3-s5"]["outcome"], rows["g3-s5"]["reason"]) == ("failed", "no result")


def test_a_decision_from_an_earlier_run_does_not_end_a_rerun(tmp_path: Path) -> None:
    """``helix resume`` reruns an interrupted candidate id; the old decision predates the new span."""
    config = _config()
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _state(tmp_path, config, 0, ["g0-s0"])
    first = Trace("r1")
    run = first.start("run", 0.0)
    gen = first.start("generation", 10.0, run, generation=1)
    p = first.start("proposal", 10.0, gen, candidate_id="g1-s1", parent="g0-s0")
    first.decision("g1-s1", 20.0, "failed mutation", False, gen)
    for record in (p, gen, run):
        first.end(record, 30.0, "KeyboardInterrupt")
    first.write(tmp_path)
    second = Trace("r2")
    run = second.start("run", 100.0)
    gen = second.start("generation", 100.0, run, generation=1)
    second.start("proposal", 101.0, gen, candidate_id="g1-s1", parent="g0-s0")
    second.write(tmp_path)
    rows = {r["id"]: r for r in RunReader(tmp_path, config).proposals()["rows"]}
    assert rows["g1-s1"]["outcome"] == "running"


def test_lineage_is_bounded_to_1500_nodes(tmp_path: Path) -> None:
    from helix.evolution import _save_attempt_result

    config = _config()
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _evaluation(tmp_path, "g1-s1", [1, 1, 0, 0])
    _state(tmp_path, config, 160, ["g0-s0", "g1-s1"])
    for i in range(2, 1602):
        gen = i // 10 + 1
        _save_attempt_result(tmp_path / ".helix",
                             EvalResult(candidate_id=f"g{gen}-s{i}", scores={}, asi={},
                                        instance_scores={"a__1": 0.0}),
                             status="rejected", reason="minibatch_gate", parent_id="g1-s1",
                             generation=gen, stage="train_minibatch")
    out = RunReader(tmp_path, config).lineage()
    assert len(out["nodes"]) == 1500 and out["truncated"] == 1602 - 1500
    ids = [n["id"] for n in out["nodes"]]
    assert ids[:2] == ["g0-s0", "g1-s1"]  # every accepted node is kept
    assert ids[2] == "g11-s104" and ids[-1] == "g161-s1601"  # the latest rejected ones
    assert RunReader(tmp_path, config).proposals()["counts"]["rejected"] == 1600  # rows are not bounded


def test_instance_history(tmp_path: Path) -> None:
    config = _config()
    _finished_run(tmp_path, config)
    reader = RunReader(tmp_path, config)
    assert reader.instance_history("a__2") == {
        "instance": "a__2",
        "points": [{"id": "g0-s0", "generation": 0, "score": 0}, {"id": "g1-s1", "generation": 1, "score": 1},
                   {"id": "g3-s6", "generation": 3, "score": 1}],
        "best": 1, "holders": ["g1-s1", "g3-s6"],
    }
    # Only the run's validation ids; a train id, a path or junk is unknown.
    for iid in ("zz", "../state.json", "", "a__2 ", "evaluations/g0-s0.json"):
        assert reader.instance_history(iid) is None
    assert RunReader(tmp_path / "none", config).instance_history("a__2") is None


def test_no_run_gives_empty_lineage_and_proposals(tmp_path: Path) -> None:
    reader = RunReader(tmp_path, _config())
    assert reader.lineage() == {"nodes": [], "objectives": None, "best": None, "max_generation": 0,
                                "truncated": 0}
    assert reader.proposals() == {"rows": [], "counts": {
        "frontier": 0, "dominated": 0, "rejected": 0, "skipped": 0, "failed": 0, "running": 0}}


# ---------------------------------------------------------------------------
# Usage records (timing.py ``usage``): spend per generation, outcome and candidate


def _usage(cost: float, tokens: tuple[int, int, int]) -> dict[str, Any]:
    return {"cost_usd": pytest.approx(cost), "input_tokens": tokens[0], "output_tokens": tokens[1],
            "cached_input_tokens": tokens[2]}


def test_usage_records_split_spend_by_generation_and_outcome(tmp_path: Path) -> None:
    from helix import timing

    config = _config()
    _finished_run(tmp_path, config)
    trace = Trace()
    trace.usage("g0-s0", 0, 0.1, (10, 1, 0), source="seed_generation", t=1.0)
    trace.usage("g1-s1", 1, 1.0, (100, 10, 50), t=150.0)     # accepted
    trace.usage("g1-s2", 1, 0.5, (50, 5, 0), t=150.0)        # rejected at the gate
    trace.usage("g2-s3", 2, 0.25, (20, 2, 0), source="mutation_failed", t=300.0)  # failed
    trace.usage("g2-s4", 2, 0.25, (20, 2, 0), t=300.0)       # failed: its minibatch eval raised
    trace.usage("g3-s6", 3, 0.6, (60, 6, 0), t=470.0)        # accepted
    trace.usage("g3-s7", 3, 0.6, (60, 6, 0), t=480.0)        # rejected at the val stage
    trace.records.append({**trace.records[-1], "cost_usd": "lots"})  # malformed: ignored
    trace.write(tmp_path)

    reader = RunReader(tmp_path, config)
    snap = reader.snapshot(now=5000.0)
    assert snap is not None
    spend = snap["spend"]
    assert spend["cost_usd"] == 3.0 and spend["has_cost"] is True  # the run total stays state.json's
    assert spend["per_record"] is True
    assert spend["by_generation"] == [
        {"generation": 0, **_usage(0.1, (10, 1, 0))},
        {"generation": 1, **_usage(1.5, (150, 15, 50))},
        {"generation": 2, **_usage(0.5, (40, 4, 0))},
        {"generation": 3, **_usage(1.2, (120, 12, 0))},
    ]
    # Median recorded cost of the finished generations 1-3 (not the seed), not 3.0 / 3.
    assert spend["cost_per_generation"] == pytest.approx(1.2)
    assert spend["projected_cost_usd"] is None  # finished run
    assert spend["by_outcome"] == {
        "accepted": _usage(1.7, (170, 17, 50)),
        "rejected": _usage(1.1, (110, 11, 0)),
        "failed": _usage(0.5, (40, 4, 0)),
        "running": _usage(0.0, (0, 0, 0)),
    }

    rows = {r["id"]: r for r in reader.proposals()["rows"]}
    assert rows["g1-s1"]["usage"] == _usage(1.0, (100, 10, 50))
    assert rows["g2-s3"]["usage"] == _usage(0.25, (20, 2, 0))
    assert rows["g2-s5"]["usage"] is None  # nothing charged
    # Usage lines leave the existing timing readers working.
    records = timing.read_records(tmp_path / ".helix" / "timing.jsonl")
    assert timing.summarize(records).run_seconds == pytest.approx(701.0)
    assert timing.to_perfetto(records)["traceEvents"]


def test_running_spend_projects_from_recorded_generations(tmp_path: Path) -> None:
    config = _config(max_generations=5)
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _state(tmp_path, config, 2, ["g0-s0"], budget=BudgetState(evaluations=100, cost_usd=4.0))
    trace = Trace()
    run = trace.start("run", 0.0)
    trace.span("generation", 0.0, 100.0, run, generation=1)
    trace.span("generation", 100.0, 400.0, run, generation=2)
    gen = trace.start("generation", 400.0, run, generation=3)
    trace.start("proposal", 400.0, gen, candidate_id="g3-s7", parent="g0-s0")
    trace.usage("g1-s1", 1, 0.5, (5, 1, 0))
    trace.usage("g2-s3", 2, 1.5, (15, 1, 0))
    trace.usage("g3-s7", 3, 0.7, (7, 1, 0))  # charged; its child minibatch is still running
    trace.write(tmp_path)

    spend = RunReader(tmp_path, config).snapshot(now=450.0)["spend"]  # type: ignore[index]
    # median(0.5, 1.5) per generation: the open generation 3 is not finished.
    assert spend["cost_per_generation"] == pytest.approx(1.0)
    assert spend["projected_cost_usd"] == pytest.approx(4.0 + 1.0 * 3)
    assert spend["by_outcome"]["running"] == _usage(0.7, (7, 1, 0))
    assert [g["generation"] for g in spend["by_generation"]] == [1, 2, 3]


def test_codex_usage_has_tokens_and_no_cost(tmp_path: Path) -> None:
    """Codex reports no cost: the records carry tokens with ``cost_usd`` 0."""
    config = _config()
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _evaluation(tmp_path, "g1-s1", [1, 1, 0, 0])
    _lineage(tmp_path, "g0-s0", None, 0)
    _lineage(tmp_path, "g1-s1", "g0-s0", 1)
    _state(tmp_path, config, 1, ["g0-s0", "g1-s1"],
           budget=BudgetState(input_tokens=9000, output_tokens=300, cached_input_tokens=8000))
    trace = Trace()
    run = trace.start("run", 0.0)
    _seed(trace, run, 0.0)
    gen = trace.start("generation", 30.0, run, generation=1)
    _proposal(trace, gen, 30.0, "g1-s1", "g0-s0", agent=50)
    trace.usage("g1-s1", 1, 0.0, (9000, 300, 8000), t=90.0)
    _validate(trace, gen, 100.0, "g1-s1", 10.0)
    trace.decision("g1-s1", 110.0, "mutation", True, gen)
    trace.end(gen, 110.0)
    trace.end(run, 111.0)
    trace.write(tmp_path)

    reader = RunReader(tmp_path, config)
    spend = reader.snapshot(now=200.0)["spend"]  # type: ignore[index]
    assert (spend["has_cost"], spend["per_record"]) == (False, True)
    assert (spend["cost_per_generation"], spend["projected_cost_usd"]) == (None, None)
    assert spend["by_generation"] == [{"generation": 1, **_usage(0.0, (9000, 300, 8000))}]
    assert spend["by_outcome"]["accepted"] == _usage(0.0, (9000, 300, 8000))
    rows = {r["id"]: r for r in reader.proposals()["rows"]}
    assert rows["g1-s1"]["usage"] == _usage(0.0, (9000, 300, 8000)) and rows["g0-s0"]["usage"] is None


def test_a_usage_record_without_a_generation_counts_by_its_id(tmp_path: Path) -> None:
    """The seed is generated before any generation span opens."""
    config = _config()
    _evaluation(tmp_path, "g0-s0", [1, 0, 0, 0])
    _state(tmp_path, config, 0, ["g0-s0"], budget=BudgetState(cost_usd=0.3))
    trace = Trace()
    trace.start("run", 0.0)
    trace.usage("g0-s0", None, 0.2, (2, 1, 0), source="seed_generation")
    trace.usage(None, None, 0.1, (1, 0, 0), source="other")
    trace.write(tmp_path)
    spend = RunReader(tmp_path, config).snapshot(now=10.0)["spend"]  # type: ignore[index]
    assert spend["by_generation"] == [{"generation": 0, **_usage(0.3, (3, 1, 0))}]
    # A charge for no candidate has no proposal row, so no outcome.
    assert spend["by_outcome"]["accepted"] == _usage(0.2, (2, 1, 0))
