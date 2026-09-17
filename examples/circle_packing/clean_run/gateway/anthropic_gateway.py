"""Egress gateway for sandboxed mutation agents: Anthropic Messages API only.

Agent containers sit on an ``--internal`` Docker network with no route out.
This gateway is the only other container on that network, and it is also
attached to a normal bridge network.  Claude Code in the agent container is
pointed at it with ``ANTHROPIC_BASE_URL=http://anthropic-gateway:8080`` and a
placeholder key; the gateway forwards allowed requests to
``https://api.anthropic.com`` with the real key, which never enters an agent
container.

Policy (``check_request``):

* only ``POST /v1/messages`` and ``POST /v1/messages/count_tokens``;
* the body must be a JSON object naming an allowed model;
* no ``mcp_servers`` (MCP connector) and no ``container`` (skills / hosted
  code-execution containers);
* every tool must be a client tool (no ``type``, or ``type == "custom"``):
  server tools such as ``web_search_*``, ``web_fetch_*`` and
  ``code_execution_*`` would let Anthropic's servers reach the web or run
  code on the agent's behalf;
* no tool named ``WebSearch`` / ``WebFetch`` / ``web_search`` / ``web_fetch``.

There is no CONNECT support and no other upstream host, so a request for any
other site from the agent container has nowhere to go.

Every request is logged as one JSON line (method, path, decision, model,
effort, thinking mode, tool names, top-level body keys, ``anthropic-beta``
values, HTTP status, token usage parsed from the response, timings).
Request and response bodies and credentials are never logged.

With ``GATEWAY_CAPTURE_ONLY=1`` the gateway records every request that passes
the policy and then refuses it (403) without contacting the upstream.  The
isolation checks use this to see exactly what the mutation agent would send,
at no API cost.
"""

from __future__ import annotations

import http.client
import json
import os
import ssl
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

UPSTREAM_HOST = os.environ.get("GATEWAY_UPSTREAM_HOST", "api.anthropic.com")
LISTEN_PORT = int(os.environ.get("GATEWAY_PORT", "8080"))
LOG_PATH = Path(os.environ.get("GATEWAY_LOG", "/logs/gateway_requests.jsonl"))
ALLOWED_PATHS = frozenset({"/v1/messages", "/v1/messages/count_tokens"})
BLOCKED_TOOL_NAMES = frozenset({"websearch", "webfetch", "web_search", "web_fetch"})
MAX_REQUEST_BYTES = 32 * 1024 * 1024
UPSTREAM_TIMEOUT_SECONDS = 1200
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
DROPPED_REQUEST_HEADERS = HOP_BY_HOP | {
    "host",
    "content-length",
    "authorization",
    "x-api-key",
    "accept-encoding",
    "cookie",
}

_log_lock = threading.Lock()


def allowed_models() -> frozenset[str]:
    raw = os.environ.get("GATEWAY_ALLOWED_MODELS", "")
    return frozenset(m.strip() for m in raw.split(",") if m.strip())


def check_request(
    method: str, path: str, body: bytes, models: frozenset[str]
) -> tuple[bool, str, dict[str, Any]]:
    """Return ``(allowed, reason, metadata)`` for one agent request."""
    meta: dict[str, Any] = {}
    route = path.split("?", 1)[0]
    if method != "POST" or route not in ALLOWED_PATHS:
        return False, f"route not allowed: {method} {route}", meta
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return False, "body is not JSON", meta
    if not isinstance(payload, dict):
        return False, "body is not a JSON object", meta
    model = payload.get("model")
    meta["body_keys"] = sorted(str(k) for k in payload)
    meta["model"] = model if isinstance(model, str) else None
    output_config = payload.get("output_config")
    if isinstance(output_config, dict):
        meta["effort"] = output_config.get("effort")
    thinking = payload.get("thinking")
    if isinstance(thinking, dict):
        meta["thinking"] = thinking.get("type")
    meta["stream"] = bool(payload.get("stream"))
    meta["max_tokens"] = payload.get("max_tokens")
    if not isinstance(model, str) or model not in models:
        return False, f"model not allowed: {model!r}", meta
    if "mcp_servers" in payload:
        return False, "mcp_servers is not allowed", meta
    if "container" in payload:
        return False, "container is not allowed", meta
    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        return False, "tools must be a list", meta
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            return False, "tool entry is not an object", meta
        tool_type = tool.get("type")
        name = tool.get("name")
        names.append(str(name))
        if tool_type not in (None, "custom"):
            meta["tool_names"] = sorted(names)
            return False, f"server tool not allowed: type={tool_type!r}", meta
        if isinstance(name, str) and name.lower() in BLOCKED_TOOL_NAMES:
            meta["tool_names"] = sorted(names)
            return False, f"tool not allowed: {name}", meta
    meta["tool_names"] = sorted(names)
    return True, "allowed", meta


class UsageParser:
    """Collect token usage from an SSE stream or a JSON body as it passes."""

    def __init__(self, event_stream: bool) -> None:
        self.event_stream = event_stream
        self.buffer = b""
        self.body = bytearray()
        self.usage: dict[str, Any] = {}
        self.response_model: str | None = None
        self.stop_reason: str | None = None
        self.error_type: str | None = None
        self.error_message: str | None = None

    def _merge(self, usage: Any, prefix: str = "") -> None:
        """Keep numeric usage fields, flattening one level of nesting.

        ``cache_creation`` splits cache writes by TTL
        (``cache_creation.ephemeral_5m_input_tokens`` /
        ``cache_creation.ephemeral_1h_input_tokens``), which prices them.
        """
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.usage[prefix + key] = value
                elif isinstance(value, dict) and not prefix:
                    self._merge(value, prefix=f"{key}.")

    def feed(self, chunk: bytes) -> None:
        if not self.event_stream:
            if len(self.body) < 8 * 1024 * 1024:
                self.body.extend(chunk)
            return
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "error":
                self._error(event.get("error"))
            elif event.get("type") == "message_start":
                message = event.get("message") or {}
                self.response_model = message.get("model")
                self._merge(message.get("usage"))
            elif event.get("type") == "message_delta":
                self._merge(event.get("usage"))
                delta = event.get("delta") or {}
                self.stop_reason = delta.get("stop_reason") or self.stop_reason

    def _error(self, error: Any) -> None:
        if isinstance(error, dict):
            self.error_type = str(error.get("type"))[:80]
            self.error_message = str(error.get("message"))[:200]

    def finish(self) -> None:
        if self.event_stream or not self.body:
            return
        try:
            payload = json.loads(bytes(self.body))
        except ValueError:
            return
        if isinstance(payload, dict):
            self._error(payload.get("error"))
            self.response_model = payload.get("model")
            self.stop_reason = payload.get("stop_reason")
            self._merge(payload.get("usage"))
            if "input_tokens" in payload and "usage" not in payload:
                self._merge({"count_tokens_input_tokens": payload["input_tokens"]})


def write_log(record: dict[str, Any]) -> None:
    line = json.dumps(record, sort_keys=True)
    with _log_lock:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "anthropic-gateway/1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _deny(self, record: dict[str, Any], status: int, reason: str) -> None:
        data = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "permission_error",
                    "message": f"blocked by the experiment egress gateway: {reason}",
                },
            }
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True
        record.update({"decision": "deny", "reason": reason, "status": status})
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_log(record)

    def _handle(self) -> None:
        started = time.monotonic()
        record: dict[str, Any] = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "client": self.client_address[0],
            "method": self.command,
            "path": self.path.split("?", 1)[0],
            "query": self.path.split("?", 1)[1] if "?" in self.path else "",
            "anthropic_beta": self.headers.get("anthropic-beta"),
            "user_agent": self.headers.get("user-agent"),
        }
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self._deny(record, 411, "chunked request bodies are not supported")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._deny(record, 413, "missing or oversized body")
            return
        body = self.rfile.read(length) if length else b""
        record["request_bytes"] = len(body)
        allowed, reason, meta = check_request(
            self.command, self.path, body, allowed_models()
        )
        record.update(meta)
        if not allowed:
            self._deny(record, 403, reason)
            return
        if os.environ.get("GATEWAY_CAPTURE_ONLY") == "1":
            self._deny(record, 403, "capture only: request recorded, not forwarded")
            return
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self._deny(record, 503, "gateway has no upstream credential")
            return

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in DROPPED_REQUEST_HEADERS
        }
        headers["x-api-key"] = api_key
        headers["accept-encoding"] = "identity"
        headers["content-length"] = str(len(body))
        connection = http.client.HTTPSConnection(
            UPSTREAM_HOST,
            443,
            timeout=UPSTREAM_TIMEOUT_SECONDS,
            context=ssl.create_default_context(),
        )
        response_bytes = 0
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            upstream = connection.getresponse()
            content_type = upstream.getheader("Content-Type", "")
            parser = UsageParser("text/event-stream" in content_type)
            self.send_response(upstream.status)
            for key, value in upstream.getheaders():
                if key.lower() in HOP_BY_HOP or key.lower() == "content-length":
                    continue
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while True:
                chunk = upstream.read1(65536)
                if not chunk:
                    break
                parser.feed(chunk)
                response_bytes += len(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
            parser.finish()
            record.update(
                {
                    "decision": "allow",
                    "status": upstream.status,
                    "request_id": upstream.getheader("request-id"),
                    "response_model": parser.response_model,
                    "stop_reason": parser.stop_reason,
                    "usage": parser.usage,
                    "error_type": parser.error_type,
                    "error_message": parser.error_message,
                }
            )
        except (OSError, http.client.HTTPException) as exc:
            record.update({"decision": "allow", "upstream_error": type(exc).__name__})
            self.close_connection = True
        finally:
            connection.close()
            record["response_bytes"] = response_bytes
            record["seconds"] = round(time.monotonic() - started, 3)
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_log(record)

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle()

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle()

    def do_CONNECT(self) -> None:  # noqa: N802
        self._handle()


def main() -> int:
    if not allowed_models():
        print("GATEWAY_ALLOWED_MODELS is empty", file=sys.stderr)
        return 2
    server = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    server.daemon_threads = True
    write_log(
        {
            "event": "start",
            "utc": datetime.now(timezone.utc).isoformat(),
            "upstream": UPSTREAM_HOST,
            "allowed_models": sorted(allowed_models()),
            "credential_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "capture_only": os.environ.get("GATEWAY_CAPTURE_ONLY") == "1",
        }
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
