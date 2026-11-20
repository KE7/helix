"""HTTP server for ``helix ui``: token-gated, GET-only, loopback by default.

Why the gate: with ``sandbox.add_host_gateway`` (and on Docker Desktop) a
sandboxed mutation agent can reach host ports, and this API serves validation
scores and evaluator feedback.  Data requests need the per-launch token in
``Authorization``, the ``Host`` header must name the bound host (a DNS-rebinding
guard), and anything but GET gets 405.  The browser loads the public page and
assets before making authenticated data requests.
"""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import json
import os
import re
import secrets
import socket
import threading
import time
import webbrowser
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import FrameType
from typing import Any, TypeVar

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from rich.markup import escape
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from helix.config import load_config
from helix.display import print_info, print_warning

STATIC_DIR = Path(__file__).parent / "static"

#: Candidate ids HELIX writes (``g<gen>-s<n>`` mutations, ``g<gen>-m<n>`` merges).
_CANDIDATE_ID = re.compile(r"^g\d+-[sm]\d+$")
#: ``?vs=`` values ``/api/candidate`` accepts (``candidate.DIFF_BASES``).
_DIFF_BASES = ("parent", "seed")
#: Ports tried after ``--port`` when it is taken.
_PORT_ATTEMPTS = 10
_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})
#: ``.helix`` entries whose (mtime_ns, size) signal a new run snapshot.
_SIGNATURE_FILES = ("state.json", "lineage.json", "timing.jsonl", "evaluations", "attempts")
_POLL_SECONDS = 1.5
_KEEPALIVE_SECONDS = 15.0
_NO_TOKEN = (
    "This dashboard needs the access token. "
    "Open the address that `helix ui` printed in the terminal.\n"
)

_T = TypeVar("_T")


def is_loopback(host: str) -> bool:
    """Return whether *host* names only this machine."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _host_name(header: str) -> str:
    """Return the host part of a ``Host`` header, without port or IPv6 brackets."""
    value = header.strip().lower()
    if value.startswith("["):
        return value[1 : value.find("]")] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _allowed_hosts(host: str) -> frozenset[str] | None:
    """Host names the guard accepts for a server bound to *host*; None accepts any."""
    if host in _WILDCARD_HOSTS:
        # Bound to every interface: no single name to check against; the
        # bearer token still gates access.
        return None
    names = {host.lower().strip("[]")}
    if is_loopback(host):
        names |= _LOOPBACK_NAMES
    return frozenset(names)


class _Guard:
    """ASGI middleware: Host check, GET only, and authentication for run data."""

    def __init__(
        self, app: ASGIApp, *, token: str, hosts: frozenset[str] | None
    ) -> None:
        self.app = app
        self.token = token
        self.hosts = hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            return  # no websockets
        request = Request(scope)
        response = self._reject(request)
        if response is not None:
            await response(scope, receive, send)
            return
        await self.app(scope, receive, _with_headers(send))

    def _reject(self, request: Request) -> Response | None:
        host = request.headers.get("host")
        if host is None or (
            self.hosts is not None and _host_name(host) not in self.hosts
        ):
            return PlainTextResponse("Invalid Host header.\n", status_code=400)
        if request.method != "GET":
            return PlainTextResponse(
                "Method not allowed.\n", status_code=405, headers={"Allow": "GET"}
            )
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            return None
        scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and secrets.compare_digest(supplied.encode(), self.token.encode()):
            return None
        return PlainTextResponse(
            _NO_TOKEN, status_code=401, headers={"WWW-Authenticate": "Bearer"}
        )


def _with_headers(send: Send) -> Send:
    """Add no-store caching and basic hardening headers to every response."""

    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = [
                (k, v)
                for k, v in message.get("headers", [])
                if k.lower() != b"cache-control"
            ]
            headers += [
                (b"cache-control", b"no-store"),
                (b"x-content-type-options", b"nosniff"),
                (b"referrer-policy", b"no-referrer"),
            ]
            message = {**message, "headers": headers}
        await send(message)

    return wrapped


def run_signature(helix_dir: Path) -> tuple[tuple[int, int] | None, ...]:
    """Return a cheap change signature of the run files under *helix_dir*."""
    out: list[tuple[int, int] | None] = []
    for name in _SIGNATURE_FILES:
        try:
            st = os.stat(helix_dir / name)
        except OSError:
            out.append(None)
        else:
            out.append((st.st_mtime_ns, st.st_size))
    return tuple(out)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


async def event_stream(
    static: Callable[[], dict[str, Any]],
    snapshot: Callable[[], dict[str, Any] | None],
    signature: Callable[[], object],
    stopping: threading.Event,
    *,
    poll_seconds: float,
    keepalive_seconds: float,
) -> AsyncIterator[str]:
    """Yield SSE frames: ``static`` once, ``run`` on each change, keepalives between."""
    yield _sse("static", await run_in_threadpool(static))
    last_signature: object = None
    last_sent = time.monotonic()
    while not stopping.is_set():
        current = signature()
        if current != last_signature:
            last_signature = current
            yield _sse("run", await run_in_threadpool(snapshot))
            last_sent = time.monotonic()
        elif time.monotonic() - last_sent >= keepalive_seconds:
            yield ": keepalive\n\n"
            last_sent = time.monotonic()
        await asyncio.sleep(poll_seconds)


def create_app(
    project_root: Path,
    config_path: Path,
    token: str,
    *,
    host: str = "127.0.0.1",
) -> Starlette:
    """Build the read-only dashboard app for one project."""
    # Imported here so the CLI starts fast and tests can swap the readers.
    from helix.ui.candidate import candidate_detail, instance_detail
    from helix.ui.model import build_static
    from helix.ui.run_state import RunReader
    from helix.ui.transcripts import TranscriptReader

    config = load_config(config_path)
    reader = RunReader(project_root, config)
    transcripts = TranscriptReader(project_root, config, reader)
    helix_dir = project_root / ".helix"
    # Set on Ctrl+C so open event streams end and the server can exit.
    stopping = threading.Event()
    # Readers keep caches; uvicorn serves sync endpoints from a thread pool.
    lock = threading.Lock()

    def locked(fn: Callable[[], _T]) -> _T:
        with lock:
            return fn()

    @functools.cache
    def static_payload() -> dict[str, Any]:
        return locked(lambda: build_static(config, project_root, config_path))

    def snapshot() -> dict[str, Any] | None:
        return locked(reader.snapshot)

    def index(request: Request) -> Response:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    def api_static(request: Request) -> Response:
        return JSONResponse(static_payload())

    def api_run(request: Request) -> Response:
        return JSONResponse(snapshot())

    def api_frontier(request: Request) -> Response:
        return JSONResponse(locked(reader.frontier))

    def api_candidate(request: Request) -> Response:
        cid = request.path_params["cid"]
        if not _CANDIDATE_ID.match(cid):
            return JSONResponse({"error": "unknown candidate"}, status_code=404)
        vs = request.query_params.get("vs", "parent")
        if vs not in _DIFF_BASES:
            return JSONResponse({"error": "vs must be parent or seed"}, status_code=400)
        detail = locked(lambda: candidate_detail(project_root, config, reader, cid, vs))
        if detail is None:
            return JSONResponse({"error": "unknown candidate"}, status_code=404)
        return JSONResponse(detail)

    def api_candidate_instance(request: Request) -> Response:
        # *iid* is looked up among the candidate's scored ids inside instance_detail; never a path.
        cid, iid = request.path_params["cid"], request.path_params["iid"]
        if not _CANDIDATE_ID.match(cid):
            return JSONResponse({"error": "unknown instance"}, status_code=404)
        detail = locked(lambda: instance_detail(project_root, reader, cid, iid))
        if detail is None:
            return JSONResponse({"error": "unknown instance"}, status_code=404)
        return JSONResponse(detail)

    def api_lineage(request: Request) -> Response:
        return JSONResponse(locked(reader.lineage))

    def api_proposals(request: Request) -> Response:
        return JSONResponse(locked(reader.proposals))

    def api_instance(request: Request) -> Response:
        # Checked against the run's validation ids inside the reader; never used as a path.
        iid = request.path_params["iid"]
        history = locked(lambda: reader.instance_history(iid))
        if history is None:
            return JSONResponse({"error": "unknown instance"}, status_code=404)
        return JSONResponse(history)

    def api_transcript(request: Request) -> Response:
        cid = request.path_params["cid"]
        if not _CANDIDATE_ID.match(cid):
            return JSONResponse({"error": "unknown candidate"}, status_code=404)
        try:
            cursor = int(request.query_params.get("cursor", "0"))
        except ValueError:
            cursor = -1
        if cursor < 0:
            return JSONResponse({"error": "bad cursor"}, status_code=400)
        # Not under the shared lock: a sandboxed live read runs ``docker run`` (up to 10 s) and
        # must not stall /api/run or the event stream. TranscriptReader and RunReader lock themselves.
        return JSONResponse(transcripts.read(cid, cursor))

    def api_events(request: Request) -> Response:
        stream = event_stream(
            static_payload,
            snapshot,
            lambda: run_signature(helix_dir),
            stopping,
            poll_seconds=_POLL_SECONDS,
            keepalive_seconds=_KEEPALIVE_SECONDS,
        )
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    routes = [
        Route("/", index),
        Route("/api/static", api_static),
        Route("/api/run", api_run),
        Route("/api/frontier", api_frontier),
        Route("/api/candidate/{cid}", api_candidate),
        # `path`: an instance id may contain "/" (the page URL-encodes it).
        Route("/api/candidate/{cid}/instance/{iid:path}", api_candidate_instance),
        Route("/api/lineage", api_lineage),
        Route("/api/proposals", api_proposals),
        # `path`: an instance id may contain "/" (the page URL-encodes it).
        Route("/api/instance/{iid:path}", api_instance),
        Route("/api/transcript/{cid}", api_transcript),
        Route("/api/events", api_events),
        Mount("/static", app=StaticFiles(directory=STATIC_DIR, check_dir=False)),
    ]
    app = Starlette(routes=routes)
    app.add_middleware(_Guard, token=token, hosts=_allowed_hosts(host))
    app.state.stopping = stopping
    return app


class _Server(uvicorn.Server):
    """uvicorn server that ends the app's event streams when asked to exit."""

    def __init__(self, config: uvicorn.Config, stopping: threading.Event) -> None:
        super().__init__(config)
        self.stopping = stopping

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        """Stop event streams, then exit as uvicorn does."""
        self.stopping.set()
        super().handle_exit(sig, frame)


def bind_socket(host: str, port: int) -> socket.socket:
    """Bind and listen on *port*, or on the first free one of the next ten."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    last_error: OSError | None = None
    for candidate in range(port, port + _PORT_ATTEMPTS + 1):
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.bind((host.strip("[]"), candidate))
            sock.listen()
        except OSError as exc:
            sock.close()
            last_error = exc
            continue
        return sock
    raise OSError(
        f"Ports {port}-{port + _PORT_ATTEMPTS} on {host} are all taken ({last_error})."
    )


def launch_url(host: str, port: int, token: str) -> str:
    """Return the tokenised address to open for a server bound to *host*:*port*."""
    shown = "127.0.0.1" if host in _WILDCARD_HOSTS else host.strip("[]")
    if ":" in shown:
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/#token={token}"


def serve(
    project_root: Path, config_path: Path, host: str, port: int, open_browser: bool
) -> None:
    """Run the dashboard until interrupted, printing (and opening) its address."""
    token = secrets.token_urlsafe(24)
    app = create_app(project_root, config_path, token, host=host)
    sock = bind_socket(host, port)
    url = launch_url(host, sock.getsockname()[1], token)
    print_info(f"HELIX UI for {escape(project_root.name)}: {escape(url)}")
    print_info("Read-only. Press Ctrl+C to stop.")
    if not is_loopback(host):
        print_warning(
            f"Listening on {escape(host)}, not loopback: anyone who can reach this port "
            "and learns the token can read scores and evaluator feedback, "
            "including a sandboxed agent with host access."
        )
    if open_browser:
        webbrowser.open(url)
    # Event streams end within one poll of Ctrl+C; the grace limit only
    # backstops a request stuck in a slow read.
    config = uvicorn.Config(
        app, log_level="warning", ws="none", lifespan="off", timeout_graceful_shutdown=5
    )
    server = _Server(config, app.state.stopping)
    server.run(sockets=[sock])
