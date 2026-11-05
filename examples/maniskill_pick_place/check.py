"""Score the current solver/ on a few training trials.

Usage: python3 check.py pushcube__0 pickcube__0 stackcube__0

Only training ids (the file stems under splits/train) are accepted. The
trials run on the host through the same evaluator as in HELIX's own
evaluation, via the check service that run.sh starts; this script prints
each trial's score and the feedback HELIX would show.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = os.environ.get("MANISKILL_CHECK_URL", "http://host.docker.internal:8765")


def main(ids: list[str]) -> int:
    train = {p.stem for p in (ROOT / "splits" / "train").glob("*.json")}
    if not ids:
        print(__doc__)
        return 2
    if refused := sorted(set(ids) - train):
        print(f"Only training ids can be checked; refused: {refused}", file=sys.stderr)
        return 2
    files = {
        str(path.relative_to(ROOT)): path.read_text()
        for path in sorted((ROOT / "solver").rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    request = urllib.request.Request(
        URL + "/check",
        data=json.dumps({"ids": ids, "files": files}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        print(json.load(exc).get("error", exc), file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(
            f"The check service at {URL} is not reachable ({exc.reason}).",
            file=sys.stderr,
        )
        return 1
    for result in body["results"]:
        info = result["side_info"]
        print(f"=== {result['id']}: score {result['score']:.3f}")
        print(info.pop("feedback", ""))
        print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
