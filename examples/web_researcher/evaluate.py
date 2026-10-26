"""
Evaluator for the web_researcher agent.

Loads question JSON files from the selected split directory (train/ or val/).
HELIX passes `HELIX_SPLIT=train|val`; this evaluator also accepts the legacy
`SPLIT` env var for standalone local runs. For each question:
  1. Resolves the expected answer. A ``ground_truth`` of ``FETCH:pypi:<package>``
     is the package's current version on PyPI, and ``FETCH:cpython`` is the
     latest CPython release listed by endoflife.date; both are fetched at
     evaluation time, with ``ground_truth_static_fallback`` used when the fetch
     fails. Any other ``ground_truth`` is used as is.
  2. Runs agent.solve(question) -> str
  3. Compares answer to ground truth using the question's match_mode:
       - 'exact':          normalized exact match
       - 'version_prefix': the answer equals the expected version, or its
                           dot-separated components, at least major.minor,
                           equal the leading components of the expected
                           version ("3.13" and "3.13.4" match "3.13.4";
                           "3" and "3.1" do not)

Requested ids come from ``helix_batch.json``. An id is either a question id
(the ``id`` field, e.g. ``q02_requests_version``) or a positional index
(``"0"``, ``"1"``, ...) into the split's questions sorted by file name, which is
what HELIX writes when ``[dataset]`` sizes are set. Without ``helix_batch.json``
the evaluator scores every question in the split.

Emits one ``HELIX_RESULT=`` line containing a ``[score, side_info]`` pair for
each requested id. The pair order is positional to ``helix_batch.json``.
"""

import json
import os
import pathlib
import urllib.request

# Shows up in `helix timing`; a no-op outside a HELIX run.
from helix import timed


def _read_batch_ids() -> list[str] | None:
    """Return HELIX's requested ids, or None for standalone full-split runs."""
    batch_file = pathlib.Path.cwd() / "helix_batch.json"
    if not batch_file.exists():
        return None
    ids = json.loads(batch_file.read_text())
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("helix_batch.json must contain a JSON list[str]")
    return ids


# ---------------------------------------------------------------------------
# Ground truth resolution
# ---------------------------------------------------------------------------

_PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
_CPYTHON_RELEASES_URL = "https://endoflife.date/api/python.json"


def _fetch_json(url: str):
    """Fetch a URL and parse its JSON body, or return None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _fetch_live_value(key: str) -> str | None:
    """Resolve a ``FETCH:`` key to its current value, or None on failure."""
    try:
        if key.startswith("pypi:"):
            data = _fetch_json(_PYPI_JSON_URL.format(package=key[len("pypi:"):]))
            value = data["info"]["version"] if data is not None else None
        elif key == "cpython":
            data = _fetch_json(_CPYTHON_RELEASES_URL)
            value = data[0]["latest"] if data else None
        else:
            value = None
    except (KeyError, IndexError, TypeError):
        value = None
    return str(value) if value else None


def resolve_ground_truth(question_data: dict) -> tuple[str, str]:
    """
    Resolve the ground truth for a question.

    Returns (ground_truth_value, source) where source is 'live', 'fallback', or 'static'.
    """
    gt = question_data.get("ground_truth", "")
    fallback = question_data.get("ground_truth_static_fallback", gt)

    # Static ground truth (no fetch needed)
    if not gt.startswith("FETCH:"):
        return gt, "static"

    live_value = _fetch_live_value(gt[len("FETCH:"):])
    if live_value:
        return live_value, "live"

    # Fallback to static value
    return fallback, "fallback"


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Normalize a string for comparison: lowercase, strip whitespace and quotes."""
    return s.strip().lower().strip('"\'`').strip()


def answers_match(got: str, expected: str, match_mode: str) -> bool:
    """
    Compare the agent's answer to the expected ground truth.

    match_mode:
        'exact'          - normalized strings must be equal
        'version_prefix' - normalized strings are equal, or the answer has at
                           least two dot-separated components (major.minor)
                           and they equal the leading components of the
                           expected version
    """
    got_n = _normalize(got)
    exp_n = _normalize(expected)

    if not got_n:
        return False

    if match_mode == "version_prefix":
        if got_n == exp_n:
            return True
        got_parts = got_n.split(".")
        exp_parts = exp_n.split(".")
        return (
            len(got_parts) >= 2
            and len(got_parts) < len(exp_parts)
            and exp_parts[: len(got_parts)] == got_parts
        )

    # 'exact' and any unknown mode
    return got_n == exp_n


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate():
    requested_split = os.environ.get("HELIX_SPLIT") or os.environ.get("SPLIT", "val")
    split = requested_split.strip().lower()
    if split not in ("train", "val"):
        split = "val"

    base_dir = pathlib.Path(__file__).parent
    question_dir = base_dir / split

    if not question_dir.exists():
        print("HELIX_RESULT=[]")
        return

    question_files = sorted(question_dir.glob("*.json"))
    if not question_files:
        print("HELIX_RESULT=[]")
        return

    questions_by_id = {}
    for qfile in question_files:
        try:
            qdata = json.loads(qfile.read_text())
        except Exception:
            continue
        questions_by_id[str(qdata.get("id", qfile.stem))] = (qfile, qdata)
    ordered_ids = list(questions_by_id)
    requested_ids = _read_batch_ids()
    if requested_ids is None:
        requested_ids = ordered_ids

    # Import agent
    try:
        import agent
    except ImportError as e:
        print("HELIX_RESULT=" + json.dumps([
            [0.0, {"error": f"Could not import agent.py: {e}"}]
            for _ in requested_ids
        ]))
        return

    payload = []
    for q_id in requested_ids:
        question_entry = questions_by_id.get(q_id)
        if question_entry is None and q_id.isdigit() and int(q_id) < len(ordered_ids):
            # Positional id from HELIX's [dataset] sampler.
            question_entry = questions_by_id[ordered_ids[int(q_id)]]
        if question_entry is None:
            payload.append([0.0, {"error": f"Unknown question id: {q_id}"}])
            continue
        qfile, qdata = question_entry
        question_text = qdata.get("question", "")
        match_mode = qdata.get("match_mode", "exact")

        # Resolve ground truth (may do a live network fetch)
        with timed("ground_truth", example=q_id):
            expected, gt_source = resolve_ground_truth(qdata)

        # Run the agent
        try:
            with timed("solve", example=q_id):
                got = agent.solve(question_text)
            if not isinstance(got, str):
                got = str(got) if got is not None else ""
        except Exception:
            got = ""

        # Score
        is_correct = answers_match(got, expected, match_mode)
        score = 1.0 if is_correct else 0.0
        payload.append([score, {
            "question": question_text,
            "expected": expected,
            "expected_source": gt_source,
            "got": got,
            "correct": is_correct,
            "match_mode": match_mode,
            "category": qdata.get("category", "unknown"),
            "scores": {"accuracy": score},
        }])

    print("HELIX_RESULT=" + json.dumps(payload))


if __name__ == "__main__":
    evaluate()
