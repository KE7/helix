"""Protected model-and-score sidecar for the LiveBench-Math smoke subset."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from constants import (
    LIVEBENCH_DATA_REVISION,
    SOLVER_MAX_TOKENS,
    SOLVER_MODEL,
    SOLVER_TEMPERATURE,
    SOLVER_TIMEOUT_SECONDS,
)
from scoring import score_livebench_math

_DATA = json.loads(Path("/opt/livebench-math/data.json").read_text())
_API_KEY = os.environ.pop("OPENAI_API_KEY", "")
_SECRET_PATTERN = re.compile(r"\b(?:sk|sess)-[A-Za-z0-9_-]{8,}\b")


def safe_error(exc: BaseException, secret: str = "") -> str:
    text = f"{type(exc).__name__}: {exc}"
    if secret:
        text = text.replace(secret, "<redacted>")
    return _SECRET_PATTERN.sub("<redacted>", text)[:500]


def solve_problem(prompt: str, row: dict[str, Any]) -> tuple[str, dict[str, int]]:
    if not _API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required by the protected solver")
    turns = row["turns"]
    question = str(turns[0] if isinstance(turns, (list, tuple)) else turns)
    body = json.dumps(
        {
            "model": SOLVER_MODEL,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            "temperature": SOLVER_TEMPERATURE,
            "max_tokens": SOLVER_MAX_TOKENS,
        }
    ).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=SOLVER_TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    answer = str(payload["choices"][0]["message"]["content"] or "")
    usage = payload.get("usage") or {}
    return answer, {
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }


def evaluate_one(
    prompt: str,
    row: dict[str, Any],
    solver: Callable[[str, dict[str, Any]], tuple[str, dict[str, int]]],
) -> list[Any]:
    try:
        answer, usage = solver(prompt, row)
        score = score_livebench_math(row, answer)
        expected = str(row["ground_truth"])
        feedback = (
            "Correct; preserve the exact-answer discipline."
            if score >= 1.0
            else f"Official score {score:.3f}; expected {expected!r}. Recheck the mathematics and final format."
        )
        return [
            score,
            {
                "question_id": str(row["question_id"]),
                "output": answer,
                "feedback": feedback,
                "solver_model": SOLVER_MODEL,
                "usage": usage,
                "scores": {"official_livebench": score},
            },
        ]
    except Exception as exc:
        return [
            0.0,
            {
                "question_id": str(row.get("question_id", "unknown")),
                "output": "",
                "feedback": "Solver or official scorer failed; treat this metric call as incorrect.",
                "error": safe_error(exc, _API_KEY),
                "solver_model": SOLVER_MODEL,
                "scores": {"official_livebench": 0.0},
            },
        ]


def evaluate_request(
    payload: dict[str, Any],
    *,
    data: dict[str, Any] | None = None,
    solver: Callable[[str, dict[str, Any]], tuple[str, dict[str, int]]] = solve_problem,
) -> list[list[Any]]:
    data = data or _DATA
    if data.get("dataset_revision") != LIVEBENCH_DATA_REVISION:
        raise ValueError("sidecar dataset revision mismatch")
    prompt = payload.get("prompt")
    split = payload.get("split")
    ids = payload.get("ids")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 100_000:
        raise ValueError("prompt must be a non-empty string of at most 100000 characters")
    if split not in {"train", "val"}:
        raise ValueError("smoke evaluator only accepts train or val")
    if not isinstance(ids, list) or not ids or len(ids) > 100:
        raise ValueError("ids must be a non-empty list of at most 100 positions")
    rows = data["smoke_splits"][split]
    selected = []
    for item in ids:
        if not isinstance(item, str) or not item.isdigit():
            raise ValueError("each HELIX ID must be a decimal string")
        index = int(item)
        if index < 0 or index >= len(rows):
            raise ValueError(f"HELIX ID is outside the pinned {split} smoke subset")
        selected.append(rows[index])
    with ThreadPoolExecutor(max_workers=min(4, len(selected))) as pool:
        # executor.map preserves request order and intentional duplicate/padded IDs.
        return list(pool.map(lambda row: evaluate_one(prompt, row, solver), selected))


class Handler(BaseHTTPRequestHandler):
    server_version = "helix-livebench-math/1"

    def do_GET(self) -> None:  # noqa: N802
        self._send(200, {"status": "ok", "dataset_revision": LIVEBENCH_DATA_REVISION})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 250_000:
                raise ValueError("invalid request length")
            payload = json.loads(self.rfile.read(length))
            self._send(200, {"results": evaluate_request(payload)})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"error": safe_error(exc)})
        except Exception as exc:
            self._send(500, {"error": safe_error(exc, _API_KEY)})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()

