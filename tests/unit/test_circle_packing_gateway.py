"""Tests for examples/circle_packing_sandbox/anthropic_gateway.py.

The gateway is the only route out of the circle-packing mutation agent's
--internal network: it forwards plain Anthropic Messages API requests with the
subscription token it holds and refuses everything that could reach the web.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

GATEWAY = Path(__file__).resolve().parents[2] / "examples" / "circle_packing_sandbox" / "anthropic_gateway.py"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gateway = _load("cp_anthropic_gateway", GATEWAY)


# ---------------------------------------------------------------------------
# Request policy
# ---------------------------------------------------------------------------


MODELS = frozenset({"claude-sonnet-5"})


def _msg(**extra: Any) -> bytes:
    body = {"model": "claude-sonnet-5", "max_tokens": 10, "messages": []}
    body.update(extra)
    return json.dumps(body).encode()


class TestGatewayPolicy:
    def test_allows_messages_with_client_tools(self) -> None:
        body = _msg(
            tools=[{"name": "Bash", "input_schema": {}}, {"type": "custom", "name": "Read"}],
            output_config={"effort": "xhigh"},
            thinking={"type": "adaptive"},
        )
        ok, reason, meta = gateway.check_request("POST", "/v1/messages?beta=true", body, MODELS)
        assert ok, reason
        assert meta["effort"] == "xhigh" and meta["thinking"] == "adaptive"
        assert meta["tool_names"] == ["Bash", "Read"]

    def test_allows_count_tokens(self) -> None:
        ok, _, _ = gateway.check_request("POST", "/v1/messages/count_tokens", _msg(), MODELS)
        assert ok

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/v1/models"),
            ("POST", "/v1/complete"),
            ("POST", "/v1/files"),
            ("CONNECT", "example.com:443"),
            ("GET", "http://example.com/"),
            ("POST", "http://example.com/v1/messages"),
            ("GET", "/v1/messages"),
        ],
    )
    def test_denies_other_routes(self, method: str, path: str) -> None:
        ok, _, _ = gateway.check_request(method, path, _msg(), MODELS)
        assert not ok

    @pytest.mark.parametrize(
        "body",
        [
            _msg(model="claude-opus-5"),
            _msg(tools=[{"type": "web_search_20260209", "name": "web_search"}]),
            _msg(tools=[{"type": "web_fetch_20260209", "name": "web_fetch"}]),
            _msg(tools=[{"type": "code_execution_20260521", "name": "code_execution"}]),
            _msg(tools=[{"type": "mcp_toolset", "mcp_server_name": "x"}]),
            _msg(tools=[{"name": "WebSearch", "input_schema": {}}]),
            _msg(tools=[{"name": "WebFetch", "input_schema": {}}]),
            _msg(mcp_servers=[{"type": "url", "url": "https://example.com", "name": "x"}]),
            _msg(container={"skills": []}),
            _msg(tools="web_search"),
            b"not json",
            b"[]",
        ],
    )
    def test_denies_web_capable_or_malformed_requests(self, body: bytes) -> None:
        ok, _, _ = gateway.check_request("POST", "/v1/messages", body, MODELS)
        assert not ok

    def test_usage_parser_reads_sse_usage(self) -> None:
        parser = gateway.UsageParser(event_stream=True)
        events = [
            {"type": "message_start", "message": {"model": "claude-sonnet-5",
             "usage": {"input_tokens": 12, "cache_read_input_tokens": 100}}},
            {"type": "content_block_delta", "delta": {"text": "hi"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 7}},
        ]
        stream = "".join(f"event: x\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        for i in range(0, len(stream), 13):  # arbitrary chunk boundaries
            parser.feed(stream[i : i + 13])
        parser.finish()
        assert parser.usage == {
            "input_tokens": 12, "cache_read_input_tokens": 100, "output_tokens": 7
        }
        assert parser.response_model == "claude-sonnet-5"
        assert parser.stop_reason == "end_turn"

    def test_usage_parser_flattens_cache_creation(self) -> None:
        parser = gateway.UsageParser(event_stream=False)
        parser.feed(json.dumps({"model": "claude-sonnet-5", "usage": {
            "input_tokens": 3, "cache_creation_input_tokens": 50,
            "cache_creation": {"ephemeral_5m_input_tokens": 20, "ephemeral_1h_input_tokens": 30},
        }}).encode())
        parser.finish()
        assert parser.usage["cache_creation.ephemeral_1h_input_tokens"] == 30
        assert parser.usage["cache_creation_input_tokens"] == 50

    def test_records_top_level_body_keys(self) -> None:
        ok, _, meta = gateway.check_request("POST", "/v1/messages", _msg(system="s"), MODELS)
        assert ok and meta["body_keys"] == ["max_tokens", "messages", "model", "system"]


def _user(*blocks: Any) -> dict[str, Any]:
    return {"role": "user", "content": list(blocks)}


PDF_URL = "https://arxiv.org/pdf/2605.19633"
IMAGE_URL = "http://hydra.nat.uni-magdeburg.de/packing/csq/csq26.png"
B64_PNG = {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}


def _tool_result(*content: Any) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": "toolu_1", "content": list(content)}


class TestGatewayContentPolicy:
    """Anthropic's servers must never fetch anything on the agent's behalf."""

    @pytest.mark.parametrize(
        "messages,where",
        [
            ([_user({"type": "document", "source": {"type": "url", "url": PDF_URL}})],
             "messages[0].content[0].source.type='url'"),
            ([_user({"type": "image", "source": {"type": "url", "url": IMAGE_URL}})],
             "messages[0].content[0].source.type='url'"),
            ([_user({"type": "text", "text": "hi"},
                    _tool_result({"type": "document", "source": {"type": "url", "url": PDF_URL}}))],
             "messages[0].content[1].content[0].source.type='url'"),
            ([_user(_tool_result({"type": "image", "source": {"type": "url", "url": IMAGE_URL}}))],
             "messages[0].content[0].content[0].source.type='url'"),
            ([_user({"type": "document", "source": {"type": "file", "file_id": "file_1"}})],
             "source.type='file'"),
            ([_user({"type": "image", "source": {"type": "file", "file_id": "file_1"}})],
             "source.type='file'"),
            ([_user({"type": "document", "source": {"type": "content", "content": [
                {"type": "image", "source": {"type": "url", "url": IMAGE_URL}}]}})],
             "source.content[0].source.type='url'"),
            ([_user({"type": "container_upload", "file_id": "file_1"})], "type='container_upload'"),
            ([{"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_fetch",
                 "input": {"url": PDF_URL}}]}], "type='server_tool_use'"),
            ([_user({"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_1", "content": {}})],
             "type='web_fetch_tool_result'"),
            ([_user({"type": "mcp_tool_use", "id": "x", "name": "fetch", "server_name": "s", "input": {}})],
             "type='mcp_tool_use'"),
            ([{"role": "system", "content": [{"type": "document", "source": {"type": "url", "url": PDF_URL}}]}],
             "type='document'"),
            ([{"role": "tool", "content": "x"}], "role='tool'"),
            ([_user({"source": {"type": "url", "url": PDF_URL}})], "type=None"),
            ([_user("not a block")], "is not an object"),
        ],
    )
    @pytest.mark.parametrize("path", ["/v1/messages?beta=true", "/v1/messages/count_tokens?beta=true"])
    def test_refuses_remote_content(self, messages: list[Any], where: str, path: str) -> None:
        ok, reason, meta = gateway.check_request("POST", path, _msg(messages=messages), MODELS)
        assert not ok
        assert reason.startswith("content not allowed") and where in reason, reason
        if "source.type" in where:
            # The refused block's source type is logged for auditing.
            assert any(k.endswith(":url") or k.endswith(":file") for k in meta["source_types"])

    def test_refuses_remote_content_in_system(self) -> None:
        body = _msg(system=[{"type": "document", "source": {"type": "url", "url": PDF_URL}}])
        ok, reason, _ = gateway.check_request("POST", "/v1/messages", body, MODELS)
        assert not ok and "system[0].type='document'" in reason

    def test_allows_inline_content_claude_code_sends(self) -> None:
        messages = [
            {"role": "user", "content": "Read .agent_task_prompt.md"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "...", "signature": "sig"},
                {"type": "redacted_thinking", "data": "abc"},
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "plot.png"}},
            ]},
            _user(
                _tool_result({"type": "text", "text": "image follows"},
                             {"type": "image", "source": B64_PNG},
                             {"type": "tool_reference", "tool_name": "Bash"}),
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "notes"}},
                {"type": "document", "source": {"type": "content", "content": [
                    {"type": "text", "text": "a"}, {"type": "image", "source": B64_PNG}]}},
                {"type": "search_result", "source": "local notes", "title": "t",
                 "content": [{"type": "text", "text": "x"}]},
                {"type": "text", "text": "continue", "cache_control": {"type": "ephemeral"}},
            ),
            {"role": "system", "content": "mid-conversation reminder"},
            {"role": "assistant", "content": [{"type": "compaction", "content": "summary"}]},
        ]
        body = _msg(messages=messages, system=[{"type": "text", "text": "You are Claude Code."}],
                    context_management={"edits": [{"type": "clear_thinking_20251015"}]})
        ok, reason, meta = gateway.check_request("POST", "/v1/messages?beta=true", body, MODELS)
        assert ok, reason
        assert meta["block_types"]["image"] == 2
        assert meta["source_types"] == {"image:base64": 2, "document:base64": 1,
                                        "document:text": 1, "document:content": 1}
        assert meta["context_edits"] == ["clear_thinking_20251015"]

    @pytest.mark.parametrize(
        "header",
        ["files-api-2025-04-14", "mcp-client-2025-11-20", "skills-2025-10-02",
         "code-execution-2025-08-25", "web-fetch-2025-09-10",
         "claude-code-20250219,web-search-2025-03-05"],
    )
    def test_refuses_remote_content_betas(self, header: str) -> None:
        ok, reason, meta = gateway.check_request("POST", "/v1/messages", _msg(), MODELS, beta_header=header)
        assert not ok and reason.startswith("anthropic-beta not allowed")
        assert meta["model"] == "claude-sonnet-5"

    def test_allows_the_betas_claude_code_sends(self) -> None:
        header = ("claude-code-20250219,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,"
                  "context-management-2025-06-27,prompt-caching-scope-2026-01-05,"
                  "mid-conversation-system-2026-04-07,effort-2025-11-24")
        ok, reason, _ = gateway.check_request("POST", "/v1/messages", _msg(), MODELS, beta_header=header)
        assert ok, reason

    def test_joins_repeated_beta_headers(self) -> None:
        assert gateway.joined_header(["claude-code-20250219", "files-api-2025-04-14"]) == (
            "claude-code-20250219,files-api-2025-04-14"
        )
        assert gateway.joined_header(None) is None

    @pytest.mark.parametrize("key", ["fallbacks", "fallback_credit_token", "user_profile_id", "betas"])
    def test_refuses_unreviewed_top_level_parameters(self, key: str) -> None:
        ok, reason, _ = gateway.check_request("POST", "/v1/messages", _msg(**{key: []}), MODELS)
        assert not ok and key in reason

    def test_deeply_nested_content_is_refused_not_crashing(self) -> None:
        # Built as text: json.dumps itself overflows the recursion limit on 3.11.
        block = '{"type": "text", "text": "x"}'
        for _ in range(3000):
            block = '{"type": "tool_result", "tool_use_id": "t", "content": [' + block + ']}'
        body = ('{"model": "claude-sonnet-5", "max_tokens": 1, "messages": '
                '[{"role": "user", "content": [' + block + ']}]}').encode()
        ok, _, _ = gateway.check_request("POST", "/v1/messages", body, MODELS)
        assert not ok

    def test_repeated_beta_header_reaches_the_policy(self, tmp_path: Path, monkeypatch) -> None:
        import http.client
        import threading
        from http.server import ThreadingHTTPServer

        monkeypatch.setenv("GATEWAY_CAPTURE_ONLY", "1")
        monkeypatch.setenv("GATEWAY_ALLOWED_MODELS", "claude-sonnet-5")
        monkeypatch.setattr(gateway, "LOG_PATH", tmp_path / "requests.jsonl")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
            body = _msg()
            conn.putrequest("POST", "/v1/messages")
            conn.putheader("content-type", "application/json")
            conn.putheader("content-length", str(len(body)))
            conn.putheader("anthropic-beta", "claude-code-20250219")
            conn.putheader("anthropic-beta", "files-api-2025-04-14")
            conn.endheaders(body)
            status = conn.getresponse().status
        finally:
            httpd.shutdown()
        assert status == 403
        [record] = [json.loads(t) for t in (tmp_path / "requests.jsonl").read_text().splitlines()]
        assert record["reason"] == "anthropic-beta not allowed: files-api-2025-04-14"


# ---------------------------------------------------------------------------
# Capture mode, upstream errors and subscription auth
# ---------------------------------------------------------------------------


def test_gateway_capture_mode_records_and_refuses_without_upstream(tmp_path: Path, monkeypatch) -> None:
    import http.client
    import threading
    from http.server import ThreadingHTTPServer

    monkeypatch.setenv("GATEWAY_CAPTURE_ONLY", "1")
    monkeypatch.setenv("GATEWAY_ALLOWED_MODELS", "claude-sonnet-5")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(gateway, "LOG_PATH", tmp_path / "requests.jsonl")

    def refuse_upstream(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("capture mode must not contact the upstream")

    monkeypatch.setattr(gateway.http.client, "HTTPSConnection", refuse_upstream)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), gateway.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    body = _msg(output_config={"effort": "xhigh"}, tools=[{"name": "Bash", "input_schema": {}}])
    try:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        conn.request("POST", "/v1/messages?beta=true", body=body, headers={
            "content-type": "application/json", "anthropic-beta": "x-2026-01-01",
            "x-api-key": "placeholder"})
        status = conn.getresponse().status
    finally:
        httpd.shutdown()
    assert status == 403
    [record] = [json.loads(text) for text in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert record["decision"] == "deny" and record["reason"].startswith("capture only")
    assert record["model"] == "claude-sonnet-5" and record["effort"] == "xhigh"
    assert record["tool_names"] == ["Bash"] and record["anthropic_beta"] == "x-2026-01-01"
    assert "placeholder" not in json.dumps(record)


def test_gateway_records_upstream_error_type() -> None:
    parser = gateway.UsageParser(event_stream=False)
    parser.feed(json.dumps({"type": "error", "error": {
        "type": "invalid_request_error", "message": "Your credit balance is too low"}}).encode())
    parser.finish()
    assert parser.error_type == "invalid_request_error"
    assert parser.error_message == "Your credit balance is too low"


def test_subscription_refresh_stays_in_auth_volume_and_never_logs_tokens(tmp_path, monkeypatch):
    from io import BytesIO
    path = tmp_path / '.credentials.json'
    path.write_text(json.dumps({'claudeAiOauth': {
        'accessToken': 'test-expired', 'refreshToken': 'test-refresh',
        'expiresAt': 0, 'scopes': ['user:inference'], 'subscriptionType': 'max'}}))
    monkeypatch.setattr(gateway, 'CREDENTIAL_PATH', path)
    monkeypatch.setattr(gateway, 'LOG_PATH', tmp_path / 'events.jsonl')
    calls = []
    def renew(request, **kwargs):
        calls.append(request)
        assert request.full_url == 'https://platform.claude.com/v1/oauth/token'
        assert json.loads(request.data)['refresh_token'] == 'test-refresh'
        return BytesIO(json.dumps({'access_token': 'test-new-access',
            'refresh_token': 'test-new-refresh', 'expires_in': 3600}).encode())
    monkeypatch.setattr(gateway.urllib.request, 'urlopen', renew)
    assert gateway.subscription_token() == 'test-new-access'
    assert gateway.subscription_token() == 'test-new-access'
    assert len(calls) == 1
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())['claudeAiOauth']['subscriptionType'] == 'max'
    assert 'test-' not in (tmp_path / 'events.jsonl').read_text()


def test_subscription_refresh_error_never_exposes_response_or_token(tmp_path, monkeypatch):
    path = tmp_path / '.credentials.json'
    path.write_text(json.dumps({'claudeAiOauth': {
        'accessToken': 'test-expired', 'refreshToken': 'test-refresh',
        'expiresAt': 0, 'scopes': ['user:inference']}}))
    monkeypatch.setattr(gateway, 'CREDENTIAL_PATH', path)
    def fail(*args, **kwargs):
        raise ValueError('test-refresh secret echoed by upstream')
    monkeypatch.setattr(gateway.urllib.request, 'urlopen', fail)
    with pytest.raises(gateway.CredentialError) as caught:
        gateway.subscription_token()
    assert 'test-refresh' not in str(caught.value)
    assert caught.value.__suppress_context__


def test_gateway_forwards_subscription_auth_without_logging_it(tmp_path, monkeypatch):
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    monkeypatch.delenv('GATEWAY_CAPTURE_ONLY', raising=False)
    monkeypatch.setenv('GATEWAY_ALLOWED_MODELS', 'claude-sonnet-5')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'unused-test-api-key')
    monkeypatch.setattr(gateway, 'LOG_PATH', tmp_path / 'requests.jsonl')
    monkeypatch.setattr(gateway, 'subscription_token', lambda: 'test-private-subscription-token')
    sent = {}
    class Response:
        status = 200
        def getheader(self, name, default=None):
            return 'application/json' if name == 'Content-Type' else default
        def getheaders(self):
            return [('Content-Type', 'application/json'), ('anthropic-ratelimit-unified-status', 'allowed')]
        def read1(self, size):
            return b''
    class Upstream:
        def __init__(self, host, *args, **kwargs):
            assert host == 'api.anthropic.com'
        def request(self, method, path, body, headers):
            sent.update(headers)
        def getresponse(self):
            return Response()
        def close(self):
            pass
    monkeypatch.setattr(gateway.http.client, 'HTTPSConnection', Upstream)
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), gateway.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection('127.0.0.1', httpd.server_address[1], timeout=10)
        conn.request('POST', '/v1/messages', body=_msg(), headers={
            'Authorization': 'Bearer placeholder', 'X-Api-Key': 'client-must-not-forward'})
        response = conn.getresponse()
        assert response.status == 200
        response.read()
    finally:
        httpd.shutdown()
    assert sent['authorization'] == 'Bearer test-private-subscription-token'
    assert not any(k.lower() == 'x-api-key' for k in sent)
    assert 'oauth-2025-04-20' in sent['anthropic-beta']
    text = (tmp_path / 'requests.jsonl').read_text()
    assert 'test-private-subscription-token' not in text
    assert 'unused-test-api-key' not in text
    assert json.loads(text)['auth_mode'] == 'claude_subscription'
