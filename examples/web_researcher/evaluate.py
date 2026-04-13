"""
Evaluator for the web_researcher agent.

Loads question JSON files from the selected split directory (train/ or val/).
HELIX passes `HELIX_SPLIT=dev|val`; this evaluator maps `dev -> train` and also
accepts the legacy `SPLIT` env var for standalone local runs. For each question:
  1. Resolves ground truth — tries to fetch live answer from the source URL if
     ground_truth starts with 'FETCH:', else uses the static value.
  2. Runs agent.solve(question) -> str
  3. Compares answer to ground truth using the question's match_mode:
       - 'exact':          normalized exact match
       - 'version_prefix': ground truth starts with agent's answer (agent can give
                           major.minor when ground truth is major.minor.patch)

Outputs JSON:
  {
    "accuracy": float,
    "instance_scores": {q_id: 0.0 | 1.0, ...},
    "details": [{"question": str, "expected": str, "got": str, "correct": bool}, ...],
    "ground_truth_source": "live" | "fallback" | "static"
  }

Exits with code 0 always; HELIX reads the JSON stdout via score_parser=json_accuracy.

Note: This evaluator itself fetches live ground truth for version questions.
An evolved agent that *also* fetches live data will stay perfectly aligned with the
evaluator as library versions advance — a naive offline agent will drift.
"""

import json
import os
import pathlib
import sys
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Ground truth resolution
# ---------------------------------------------------------------------------

def _fetch_pypi_version(package: str) -> str | None:
    """Fetch the latest stable version of a PyPI package via the JSON API."""
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data["info"]["version"]
    except Exception:
        return None


def _fetch_url_json(url: str, jq_path: str) -> str | None:
    """
    Fetch a URL, parse JSON, and extract a value by a dotted path like '.info.version'
    or '.[0].latest'.
    """
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None

    # Walk the jq-like path manually (supports . prefix, array index .[N], .key)
    parts = jq_path.lstrip(".").split(".")
    node = data
    for part in parts:
        if not part:
            continue
        # Handle array index like '[0]'
        if part.startswith("[") and part.endswith("]"):
            try:
                idx = int(part[1:-1])
                node = node[idx]
            except (IndexError, ValueError, TypeError):
                return None
        else:
            # Strip trailing array index if any, e.g. 'foo[0]'
            if "[" in part:
                key, bracket = part.split("[", 1)
                try:
                    idx = int(bracket.rstrip("]"))
                    node = node[key][idx]
                except (KeyError, IndexError, ValueError, TypeError):
                    return None
            else:
                try:
                    node = node[part]
                except (KeyError, TypeError):
                    return None
    return str(node) if node is not None else None


def resolve_ground_truth(question_data: dict) -> tuple[str, str]:
    """
    Resolve the ground truth for a question.

    Returns (ground_truth_value, source) where source is 'live', 'fallback', or 'static'.
    """
    gt = question_data.get("ground_truth", "")
    fallback = question_data.get("ground_truth_static_fallback", gt)
    source_url = question_data.get("ground_truth_source")
    source_jq = question_data.get("ground_truth_source_jq", ".info.version")

    # Static ground truth (no fetch needed)
    if not gt.startswith("FETCH:"):
        return gt, "static"

    # Dynamic: try to fetch live value
    if source_url:
        live_value = _fetch_url_json(source_url, source_jq)
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
        'version_prefix' - expected starts with got (agent can give partial version)
                           OR got starts with expected (agent gives more specific version
                           that is still consistent)
    """
    got_n = _normalize(got)
    exp_n = _normalize(expected)

    if not got_n:
        return False

    if match_mode == "exact":
        return got_n == exp_n

    elif match_mode == "version_prefix":
        # Accept if:
        #   - exact match
        #   - agent gave major.minor and ground truth is major.minor.patch
        #     e.g. got="3.13", expected="3.13.2"  -> expected starts with got+"."
        #   - agent gave exact same major.minor.patch
        if got_n == exp_n:
            return True
        # got is a prefix of expected (e.g. got="3.13", exp="3.13.2")
        if exp_n.startswith(got_n + ".") or exp_n.startswith(got_n):
            return True
        # expected is a prefix of got (agent gave same series, slightly different patch)
        if got_n.startswith(exp_n + ".") or got_n.startswith(exp_n):
            return True
        return False

    # Default: exact
    return got_n == exp_n


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate():
    requested_split = os.environ.get("HELIX_SPLIT") or os.environ.get("SPLIT", "val")
    split = requested_split.strip().lower()
    if split == "dev":
        split = "train"
    if split not in ("train", "val"):
        split = "val"

    base_dir = pathlib.Path(__file__).parent
    question_dir = base_dir / split

    if not question_dir.exists():
        result = {
            "accuracy": 0.0,
            "instance_scores": {},
            "details": [],
            "error": f"Question directory not found: {question_dir}",
        }
        print(json.dumps(result, indent=2))
        sys.exit(0)

    question_files = sorted(question_dir.glob("*.json"))
    if not question_files:
        result = {
            "accuracy": 0.0,
            "instance_scores": {},
            "details": [],
            "error": f"No question files found in {question_dir}",
        }
        print(json.dumps(result, indent=2))
        sys.exit(0)

    # Import agent
    try:
        import agent
    except ImportError as e:
        result = {
            "accuracy": 0.0,
            "instance_scores": {},
            "details": [],
            "error": f"Could not import agent.py: {e}",
        }
        print(json.dumps(result, indent=2))
        sys.exit(0)

    instance_scores = {}
    details = []
    correct = 0
    total = 0

    for qfile in question_files:
        try:
            with open(qfile) as f:
                qdata = json.load(f)
        except Exception as e:
            q_id = qfile.stem
            instance_scores[q_id] = 0.0
            details.append({
                "question": str(qfile),
                "expected": "?",
                "got": "",
                "correct": False,
                "error": str(e),
            })
            total += 1
            continue

        q_id = qdata.get("id", qfile.stem)
        question_text = qdata.get("question", "")
        match_mode = qdata.get("match_mode", "exact")

        # Resolve ground truth (may do a live network fetch)
        expected, gt_source = resolve_ground_truth(qdata)

        # Run the agent
        try:
            got = agent.solve(question_text)
            if not isinstance(got, str):
                got = str(got) if got is not None else ""
        except Exception as ex:
            got = ""

        # Score
        is_correct = answers_match(got, expected, match_mode)
        score = 1.0 if is_correct else 0.0
        if is_correct:
            correct += 1

        instance_scores[q_id] = score
        details.append({
            "question": question_text,
            "expected": expected,
            "expected_source": gt_source,
            "got": got,
            "correct": is_correct,
            "match_mode": match_mode,
            "category": qdata.get("category", "unknown"),
        })
        total += 1

    accuracy = correct / total if total > 0 else 0.0

    result = {
        "accuracy": round(accuracy, 4),
        "instance_scores": instance_scores,
        "details": details,
        "total": total,
        "correct": correct,
    }

    print(json.dumps(result, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    evaluate()
