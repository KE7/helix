"""``GET /api/candidate/{id}``: one candidate's scores, lineage, diff and weakest instances.

Read-only: files under ``.helix/`` are only read, and git runs only
``rev-parse`` and ``diff`` between two commits with ``GIT_OPTIONAL_LOCKS=0``,
so it never refreshes an index or takes a lock in a live run's worktrees.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from helix.config import HelixConfig
from helix.lineage import load_lineage
from helix.ui.redact import add_env_names, redact
from helix.ui.run_state import RunReader, add_usage, candidate_usage, empty_usage, family_of

CANDIDATE_ID_RE = re.compile(r"^g\d+-[sm]\d+$")

_DIFF_LINES = 400
_SIDE_INFO_CHARS = 400
# One instance's side_info, opened on demand from the Instances section.
_INSTANCE_SIDE_INFO_CHARS = 4000
_NOTE_CHARS = 2000
_LOWEST = 12
_GIT_TIMEOUT = 10.0
_PROMPT_CHARS = 20000
_SEED = "g0-s0"
# Written by the change-summary call (``helix.sibling_history``), gitignored.
_SUMMARY_FILE = ".agent_change_summary.md"
# The rendered mutation prompt kept in the worktree, then its fallback path
# (mutator.MUTATION_PROMPT_ARTIFACT_NAME / _FALLBACK_NAME, mutator.py:641-642).
_PROMPT_FILES = (".agent_task_prompt.md", ".agent_internal/task_prompt.md")
# What each backend call reported, written by HELIX into the worktree
# (mutator.py ``_backend_artifact_names``): the mutation, then its change-summary call.
_BACKEND_RESULTS = (".helix_backend_result.json", ".helix_backend_summary_result.json")
_BACKEND_RESULT_BYTES = 8 * 1024 * 1024
#: Diff bases ``?vs=`` accepts.
DIFF_BASES = ("parent", "seed")


def candidate_detail(
    project_root: Path, config: HelixConfig, reader: RunReader, cid: str, vs: str = "parent"
) -> dict[str, Any] | None:
    """Everything the Candidate view shows for *cid*, or ``None`` when HELIX has no record of it.

    *vs* picks the diff base: ``"parent"`` (the first parent) or ``"seed"`` (``g0-s0``).
    """
    if not CANDIDATE_ID_RE.match(cid) or vs not in DIFF_BASES:
        return None
    add_env_names([*config.passthrough_env, *config.env])
    helix_dir = project_root / ".helix"
    evaluation = _read_json(helix_dir / "evaluations" / f"{cid}.json")
    attempt_file = _read_json(helix_dir / "attempts" / f"{cid}.json")
    lineage = _lineage(helix_dir)
    snapshot = reader.snapshot() or {}
    rows = {row["id"]: row for row in snapshot.get("candidates") or []}
    row = rows.get(cid)
    entry = lineage.get(cid)
    if row is None and attempt_file is None and entry is None and cid not in reader.candidate_ids():
        return None

    accepted = row is not None
    attempt = (attempt_file or {}).get("attempt") or {}
    parent = row["parent"] if row else entry.parent if entry else attempt.get("parent_id")
    parents = row["parents"] if row else entry.parents if entry else [p for p in [parent] if p]
    generation = (
        row["generation"] if row else entry.generation if entry else attempt.get("generation")
    )
    operation = row["operation"] if row else entry.operation if entry else None
    binary = bool(snapshot.get("binary_scores"))
    mean = row.get("mean") if row else None
    n = row.get("n") if row else None

    worktree = helix_dir / "worktrees" / cid
    exists = worktree.is_dir()
    commit = files = diff = None
    truncated = False
    base = _SEED if vs == "seed" else parent
    base_wt = (helix_dir / "worktrees" / base
               if base and base != cid and CANDIDATE_ID_RE.match(base) else None)
    if exists:
        commit = _git(worktree, "rev-parse", "HEAD")
        base_sha = _git(base_wt, "rev-parse", "HEAD") if base_wt and base_wt.is_dir() else None
        if commit and base_sha:
            files = _numstat(worktree, base_sha, commit)
            diff, truncated = _diff(worktree, base_sha, commit)
    prompt, prompt_truncated = _prompt(worktree) if exists else (None, False)
    usage = candidate_usage(reader.timing_records(), cid)
    usage_source = "timing" if usage is not None else None
    if usage is None and exists:
        usage = _worktree_usage(worktree)
        usage_source = "worktree" if usage is not None else None

    scored = evaluation if accepted else attempt_file
    basis = _basis(accepted, scored)
    compare_to = _compare_to(parents, rows) if accepted else None
    parent_scores = _scores(_read_json(helix_dir / "evaluations" / f"{compare_to}.json")) if compare_to else {}
    instances = [
        {"id": iid, "score": score, "parent_score": parent_scores.get(iid)}
        for iid, score in _scores(scored).items()
    ]
    return {
        "id": cid,
        "generation": generation,
        "parent": parent,
        "parents": parents,
        "operation": operation,
        "accepted": accepted,
        "front": bool(row and row.get("front")),
        "rank": _rank(rows, cid) if accepted else None,
        "keys": row.get("keys") if row else None,
        "sole": row.get("sole") if row else None,
        "mean": mean,
        "n": n,
        "successes": round(mean * n) if binary and mean is not None and n else None,
        "objective_means": row.get("objective_means") if row else None,
        "note": _note(helix_dir, worktree, parent, cid),
        "seed": ("generated" if config.seedless.enabled else "repository") if operation == "seed" else None,
        "worktree": str(worktree.relative_to(project_root)),
        "worktree_exists": exists,
        "commit": commit[:7] if commit else None,
        "files": files,
        "diff": diff,
        "diff_truncated": truncated,
        "diff_base": vs,
        "prompt": prompt,
        "prompt_truncated": prompt_truncated,
        "vs_parent": _vs_parent(helix_dir, evaluation, parents[0] if parents else None, rows)
        if accepted else None,
        "children": [
            row for row in reader.proposals()["rows"]
            if cid in row["parents"] and row["outcome"] != "skipped"
        ],
        "lowest": _lowest(scored) if scored else [],
        "compare_to": compare_to,
        "instances_basis": basis,
        "instances": instances,
        "families": _families(instances, config.evolution.group_key_separator, binary,
                              compare_to is not None),
        "attempt": _attempt(attempt_file) if attempt_file else None,
        "timing": _timing(reader.timing_records(), cid),
        "usage": usage,
        "usage_source": usage_source,
    }


def instance_detail(
    project_root: Path, reader: RunReader, cid: str, iid: str
) -> dict[str, Any] | None:
    """One scored instance of *cid* with its compacted side_info (and its parent's), or ``None``.

    *iid* is only looked up among the candidate's own scored ids, never used as a path.
    ``per_example_side_info`` is positional to ``instance_scores`` (population.py ``EvalResult``).
    """
    if not CANDIDATE_ID_RE.match(cid):
        return None
    helix_dir = project_root / ".helix"
    snapshot = reader.snapshot() or {}
    rows = {row["id"]: row for row in snapshot.get("candidates") or []}
    accepted = cid in rows
    scored = _read_json(helix_dir / ("evaluations" if accepted else "attempts") / f"{cid}.json")
    found = _instance(scored, iid)
    if found is None:
        return None
    score, side_info = found
    parent = None
    compare_to = _compare_to(rows[cid]["parents"], rows) if accepted else None
    if compare_to is not None:
        theirs = _instance(_read_json(helix_dir / "evaluations" / f"{compare_to}.json"), iid)
        if theirs is not None:
            parent = {"id": compare_to, "score": theirs[0], "side_info": theirs[1]}
    return {"instance": iid, "basis": _basis(accepted, scored), "score": score,
            "side_info": side_info, "parent": parent}


def _instance(result: Any, iid: str) -> tuple[float, str | None] | None:
    """(score, compacted side_info) of *iid* in an evaluation or attempt file, or ``None``."""
    scores = _scores(result)
    if iid not in scores:
        return None
    side = result.get("per_example_side_info") or []
    index = list(scores).index(iid)
    info = side[index] if isinstance(side, list) and index < len(side) else None
    return scores[iid], compact_side_info(info, _INSTANCE_SIDE_INFO_CHARS)


def _scores(result: Any) -> dict[str, float]:
    scores = result.get("instance_scores") if isinstance(result, dict) else None
    return scores if isinstance(scores, dict) else {}


def _basis(accepted: bool, scored: Any) -> str | None:
    """``"val"`` for an accepted candidate; a rejected one's attempt stage (a gate score)."""
    if not _scores(scored):
        return None
    if accepted:
        return "val"
    stage = (scored.get("attempt") or {}).get("stage")
    return str(stage) if stage else None


def _compare_to(parents: list[str], rows: dict[str, dict[str, Any]]) -> str | None:
    """The first parent when it is accepted, so it has validation scores in ``evaluations/``."""
    parent = parents[0] if parents else None
    return parent if parent in rows and CANDIDATE_ID_RE.match(parent or "") else None


def _families(
    instances: list[dict[str, Any]], separator: str, binary: bool, has_parent: bool
) -> list[dict[str, Any]] | None:
    """Per-family candidate and parent scores; ``None`` when there are fewer than two families."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in instances:
        groups.setdefault(family_of(row["id"], separator), []).append(row)
    if len(groups) < 2:
        return None
    out = []
    for name, members in groups.items():
        mine = [r["score"] for r in members]
        theirs = [r["parent_score"] for r in members if r["parent_score"] is not None]
        mean = sum(mine) / len(mine)
        parent_mean = sum(theirs) / len(theirs) if has_parent and theirs else None
        out.append({
            "name": name,
            "n": len(mine),
            "mean": mean,
            "successes": round(sum(mine)) if binary else None,
            "parent_mean": parent_mean,
            "parent_successes": round(sum(theirs)) if binary and parent_mean is not None else None,
            "delta": None if parent_mean is None else mean - parent_mean,
        })
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _lineage(helix_dir: Path) -> dict[str, Any]:
    try:
        return load_lineage(helix_dir / "lineage.json")
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _rank(rows: dict[str, dict[str, Any]], cid: str) -> int | None:
    scored = sorted((r for r in rows.values() if r.get("mean") is not None), key=lambda r: -r["mean"])
    return next((i + 1 for i, r in enumerate(scored) if r["id"] == cid), None)


def _git(worktree: Path | None, *args: str) -> str | None:
    """Run a read-only git command in *worktree*; its stdout, or ``None`` on any failure."""
    if worktree is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), *args],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_GIT_TIMEOUT,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _numstat(worktree: Path, base: str, head: str) -> list[dict[str, Any]] | None:
    out = _git(worktree, "diff", "--no-ext-diff", "--no-color", "--numstat", base, head, "--", ".")
    if out is None:
        return None
    files = []
    for line in out.splitlines():
        added, removed, path = (line.split("\t", 2) + ["", "", ""])[:3]
        files.append(
            {
                "path": path,
                "added": int(added) if added.isdigit() else None,
                "removed": int(removed) if removed.isdigit() else None,
            }
        )
    return files


def _diff(worktree: Path, base: str, head: str) -> tuple[str | None, bool]:
    out = _git(
        worktree, "diff", "--no-ext-diff", "--no-textconv", "--no-color", base, head, "--", "."
    )
    if out is None:
        return None, False
    lines = out.splitlines()
    return redact("\n".join(lines[:_DIFF_LINES])), len(lines) > _DIFF_LINES


def _prompt(worktree: Path) -> tuple[str | None, bool]:
    """The candidate's own mutation prompt, redacted and capped; it is shown to the operator only."""
    root = worktree.resolve()
    for name in _PROMPT_FILES:
        path = worktree / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                text = fh.read(_PROMPT_CHARS + 1)
        except OSError:
            return None, False
        return redact(text[:_PROMPT_CHARS]), len(text) > _PROMPT_CHARS
    return None, False


def _worktree_usage(worktree: Path) -> dict[str, Any] | None:
    """Cost and tokens the worktree's backend result files report, for runs without ``usage`` records.

    The mutation's and its change-summary call's ``usage`` add up to what the
    budget charged for the candidate (evolution.py folds the summary into the
    child's usage).  ``None`` when neither file holds a usage object.
    """
    root = worktree.resolve()
    total: dict[str, Any] | None = None
    for name in _BACKEND_RESULTS:
        path = worktree / name
        try:
            if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                    or path.stat().st_size > _BACKEND_RESULT_BYTES):
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict) or not all(
            isinstance(usage.get(k, 0), (int, float)) and not isinstance(usage.get(k, 0), bool)
            for k in ("cost_usd", "input_tokens", "output_tokens", "cached_input_tokens")
        ):
            continue
        total = total or empty_usage()
        add_usage(total, usage)
    return total


def _vs_parent(
    helix_dir: Path, evaluation: Any, parent: str | None, rows: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Validation instances the candidate solved better or worse than its (first) parent.

    Both must be accepted, so both have full-validation scores in ``evaluations/``.
    """
    if not isinstance(evaluation, dict) or parent not in rows or not CANDIDATE_ID_RE.match(parent or ""):
        return None
    parent_eval = _read_json(helix_dir / "evaluations" / f"{parent}.json")
    if not isinstance(parent_eval, dict):
        return None
    mine = evaluation.get("instance_scores") or {}
    theirs = parent_eval.get("instance_scores") or {}
    shared = [i for i in mine if i in theirs]
    improved = [i for i in shared if mine[i] > theirs[i]]
    regressed = [i for i in shared if mine[i] < theirs[i]]
    return {"improved": len(improved), "regressed": len(regressed),
            "improved_ids": improved, "regressed_ids": regressed}


def _note(helix_dir: Path, worktree: Path, parent: str | None, cid: str) -> str | None:
    """The sibling-history summary HELIX kept, else the worktree's change summary."""
    state = _read_json(helix_dir / "state.json") or {}
    for entry in (state.get("sibling_history") or {}).get(parent or "", []):
        if isinstance(entry, dict) and entry.get("child_id") == cid and entry.get("summary"):
            return redact(str(entry["summary"])[:_NOTE_CHARS])
    path = worktree / _SUMMARY_FILE
    if path.is_file() and not path.is_symlink():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:_NOTE_CHARS]
        except OSError:
            return None
        return redact(text) if text.strip() else None
    return None


def _lowest(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The 12 lowest-scoring instances, with their side_info compacted to one line."""
    scores = result.get("instance_scores") or {}
    side = result.get("per_example_side_info") or []
    ranked = sorted(enumerate(scores.items()), key=lambda item: item[1][1])[:_LOWEST]
    return [
        {
            "instance": instance,
            "score": score,
            "side_info": compact_side_info(side[i]) if i < len(side) else None,
        }
        for i, (instance, score) in ranked
    ]


def compact_side_info(info: Any, limit: int = _SIDE_INFO_CHARS) -> str | None:
    """One redacted line of ``k: v`` pairs, JSON-object string values flattened one level, at most *limit* chars."""
    if not isinstance(info, dict):
        return None
    parts: list[str] = []
    for key, value in info.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                value = parsed
        if isinstance(value, dict):
            parts += [f"{key}.{k}: {_scalar(v)}" for k, v in value.items() if not str(k).startswith("_")]
        else:
            parts.append(f"{key}: {_scalar(value)}")
    text = redact(" ".join(" · ".join(parts).split()))
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text or None


def _scalar(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), default=str)


def _attempt(data: dict[str, Any]) -> dict[str, Any]:
    attempt = data.get("attempt") or {}
    scores = data.get("instance_scores") or {}
    return {
        "reason": attempt.get("reason"),
        "stage": attempt.get("stage"),
        "score": sum(scores.values()) / len(scores) if scores else None,
        "n": len(scores),
    }


def _timing(records: list[dict[str, Any]], cid: str) -> dict[str, float | None]:
    """Seconds this candidate spent in its agent calls, its own gate evaluations and validation.

    ``evaluate`` spans are the evaluator's own work (``timing.WORK_KINDS``).
    A parent's minibatch evaluation runs inside its child's proposal, so its
    record names the child as ``candidate`` and the parent as
    ``attrs.candidate_id`` (timing.py:20-23); requiring both to be *cid*
    counts neither against the other.
    """
    totals: dict[str, float | None] = {"agent_seconds": None, "gate_seconds": None, "validation_seconds": None}
    for rec in records:
        if rec.get("event") != "end" or not isinstance(rec.get("duration_seconds"), (int, float)):
            continue
        if rec.get("error_type") == "KeyboardInterrupt":
            continue  # Ctrl-C: `helix resume` reruns the same candidate id
        attrs = rec.get("attrs") or {}
        key = None
        if rec.get("candidate") != cid:
            continue
        if rec.get("span") == "agent":
            key = "agent_seconds"
        elif rec.get("span") == "evaluate" and attrs.get("candidate_id") == cid:
            key = {"train": "gate_seconds", "val": "validation_seconds"}.get(str(attrs.get("split")))
        if key:
            totals[key] = (totals[key] or 0.0) + float(rec["duration_seconds"])
    return totals
