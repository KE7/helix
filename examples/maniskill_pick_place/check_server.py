"""Training-only check service for the sandboxed mutation agent.

The agent runs in HELIX's Docker sandbox, which cannot run ManiSkill, so its
``check.py`` posts the candidate's ``solver/`` files and some training ids
here. This service writes them into a private copy of the example and runs
``evaluate.py`` on them, so the solver goes through the same container as in
a real evaluation, then returns each trial's score and side_info. Ids outside
``splits/train`` are refused. ``run.sh`` starts and stops it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent
TRAIN_IDS = frozenset(p.stem for p in (ROOT / "splits" / "train").glob("*.json"))
MAX_REQUEST_BYTES = 1 << 20
MAX_FILES = 100
CHECK_SECONDS = 600
# What the evaluator needs; never the validation split.
EXAMPLE_FILES = ("evaluate.py", "sim", "solver.Dockerfile", "requirements.txt")


def _solver_files(files: Any) -> dict[PurePosixPath, str]:
    """Validate uploaded files: Python sources under solver/, nothing else."""
    if not isinstance(files, dict) or not 0 < len(files) <= MAX_FILES:
        raise ValueError(f"files must map up to {MAX_FILES} paths to their text")
    checked = {}
    for name, text in files.items():
        path = PurePosixPath(name)
        if (
            not isinstance(text, str)
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[0] != "solver"
            or path.suffix != ".py"
        ):
            raise ValueError(f"not a solver source file: {name!r}")
        checked[path] = text
    if PurePosixPath("solver/__init__.py") not in checked:
        raise ValueError("solver/__init__.py is missing")
    return checked


def _check(ids: Any, files: Any, unsafe_local: bool) -> dict[str, Any]:
    if (
        not isinstance(ids, list)
        or not 0 < len(ids) <= len(TRAIN_IDS)
        or not all(isinstance(i, str) for i in ids)
    ):
        raise ValueError(f"ids must be a list of 1 to {len(TRAIN_IDS)} training ids")
    if refused := sorted(set(ids) - TRAIN_IDS):
        raise ValueError(f"only training ids can be checked; refused: {refused}")
    solver = _solver_files(files)
    with tempfile.TemporaryDirectory(prefix="helix-check-") as tmp:
        work = Path(tmp)
        for name in EXAMPLE_FILES:
            source = ROOT / name
            if source.is_dir():
                shutil.copytree(
                    source, work / name, ignore=shutil.ignore_patterns("__pycache__")
                )
            else:
                shutil.copy(source, work / name)
        for path, text in solver.items():
            (work / path).parent.mkdir(parents=True, exist_ok=True)
            (work / path).write_text(text)
        (work / "helix_batch.json").write_text(json.dumps(ids))
        command = [sys.executable, "-I", "evaluate.py"]
        if unsafe_local:
            command.append("--unsafe-local")
        proc = subprocess.run(
            command, cwd=work, capture_output=True, text=True, timeout=CHECK_SECONDS
        )
    lines = [
        line for line in proc.stdout.splitlines() if line.startswith("HELIX_RESULT=")
    ]
    if proc.returncode != 0 or len(lines) != 1:
        raise RuntimeError("the evaluator failed:\n" + proc.stderr[-4000:])
    results = json.loads(lines[0].removeprefix("HELIX_RESULT="))
    return {
        "results": [
            {"id": i, "score": score, "side_info": info}
            for i, (score, info) in zip(ids, results, strict=True)
        ]
    }


class _Handler(BaseHTTPRequestHandler):
    unsafe_local = False

    def _reply(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"ok": True})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/check":
            self._reply(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_REQUEST_BYTES:
            self._reply(
                413, {"error": f"requests are limited to {MAX_REQUEST_BYTES} bytes"}
            )
            return
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("the request must be a JSON object")
            self._reply(
                200, _check(request.get("ids"), request.get("files"), self.unsafe_local)
            )
        except ValueError as exc:
            self._reply(400, {"error": str(exc)})
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            self._reply(500, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("check_server: " + format % args + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--unsafe-local",
        action="store_true",
        help="pass --unsafe-local to evaluate.py (solvers run with no isolation)",
    )
    args = parser.parse_args()
    if not args.unsafe_local:
        # Build the solver image now rather than during the agent's first check.
        spec = importlib.util.spec_from_file_location("evaluate", ROOT / "evaluate.py")
        assert spec is not None and spec.loader is not None
        evaluate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluate)
        evaluate._solver_image()
    _Handler.unsafe_local = args.unsafe_local
    server = HTTPServer((args.host, args.port), _Handler)
    print(f"check_server: listening on {args.host}:{args.port}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
