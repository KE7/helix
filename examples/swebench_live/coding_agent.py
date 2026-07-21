"""Seed autonomous repair agent evolved by HELIX.

The seed deliberately performs only safe repository reconnaissance. HELIX must
evolve it into an implementation that edits the task repository; the evaluator
scores only the resulting diff with the official SWE-bench-Live harness rule.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(repo: Path, issue_file: Path) -> None:
    """Inspect the task without guessing a source edit in the seed candidate."""

    issue_file.read_text(encoding="utf-8")
    subprocess.run(
        ["git", "status", "--short"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--issue-file", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.issue_file.resolve())


if __name__ == "__main__":
    main()
