"""Tests for the ``helix ui`` server: token gate, Host guard, routes, SSE, CLI."""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from starlette.testclient import TestClient

from helix.cli import cli
from helix.ui import server

TOKEN = "test-token-abc"
SNAPSHOT = {"status": {"state": "running"}, "generation": {"current": 1, "max": 3}}
STATIC = {"project": {"name": "proj"}}
DETAIL = {"id": "g1-s1", "mean": 0.5}
LINEAGE = {"nodes": [{"id": "g0-s0"}], "best": "g0-s0", "max_generation": 0, "truncated": 0}
PROPOSALS = {"rows": [{"id": "g1-s1"}], "counts": {"frontier": 1}}
INSTANCES = {"lift__seed101", "tasks/a b"}


class FakeRunReader:
    """Stand-in for helix.ui.run_state.RunReader."""

    def __init__(self, project_root: Path, config: Any) -> None:
        self.project_root = project_root

    def snapshot(self, now: float | None = None) -> dict[str, Any] | None:
        return SNAPSHOT

    def frontier(self) -> dict[str, Any] | None:
        return {"type": "instance", "ids": ["g1-s1"]}

    def candidate_ids(self) -> set[str]:
        return {"g1-s1"}

    def timing_records(self) -> list[dict[str, Any]]:
        return []

    def lineage(self) -> dict[str, Any]:
        return LINEAGE

    def proposals(self) -> dict[str, Any]:
        return PROPOSALS

    def instance_history(self, iid: str) -> dict[str, Any] | None:
        return {"instance": iid, "points": [], "best": None, "holders": []} if iid in INSTANCES else None


class FakeTranscriptReader:
    """Stand-in for helix.ui.transcripts.TranscriptReader."""

    def __init__(self, project_root: Path, config: Any, reader: Any) -> None:
        pass

    def read(self, cid: str, cursor: int = 0) -> dict[str, Any]:
        return {"source": "none", "events": [], "cursor": cursor, "note": cid}


def _candidate_detail(
    project_root: Path, config: Any, reader: Any, cid: str, vs: str = "parent"
) -> dict[str, Any] | None:
    return {**DETAIL, "diff_base": vs} if cid == "g1-s1" else None


def _instance_detail(project_root: Path, reader: Any, cid: str, iid: str) -> dict[str, Any] | None:
    return {"instance": iid, "score": 1.0} if cid == "g1-s1" and iid in INSTANCES else None


@pytest.fixture
def fake_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the data modules for fakes; server.create_app imports them lazily."""
    fakes = {
        "helix.ui.model": {"build_static": lambda config, root, path: STATIC},
        "helix.ui.run_state": {"RunReader": FakeRunReader},
        "helix.ui.candidate": {"candidate_detail": _candidate_detail, "instance_detail": _instance_detail},
        "helix.ui.transcripts": {"TranscriptReader": FakeTranscriptReader},
    }
    for name, attrs in fakes.items():
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "helix.toml").write_text(
        'objective = "test"\n\n[evaluator]\ncommand = "true"\n'
    )
    return tmp_path


@pytest.fixture
def static_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "static"
    path.mkdir()
    (path / "index.html").write_text("<!doctype html><title>HELIX</title>")
    (path / "app.js").write_text("console.log('ok')")
    monkeypatch.setattr(server, "STATIC_DIR", path)
    return path


@pytest.fixture
def client(fake_modules: None, project: Path, static_dir: Path) -> TestClient:
    app = server.create_app(project, project / "helix.toml", TOKEN)
    return TestClient(app, base_url="http://127.0.0.1:8790", follow_redirects=False)


def _cookie_name() -> str:
    import hashlib

    return "helix_ui_" + hashlib.sha256(TOKEN.encode()).hexdigest()[:12]


API_ROUTES = [
    "/api/static",
    "/api/run",
    "/api/frontier",
    "/api/candidate/g1-s1",
    "/api/candidate/g1-s1?vs=seed",
    "/api/candidate/g1-s1/instance/lift__seed101",
    "/api/lineage",
    "/api/proposals",
    "/api/instance/lift__seed101",
    "/api/transcript/g1-s1",
    "/api/events",
    "/static/app.js",
]


@pytest.mark.parametrize("path", API_ROUTES)
def test_every_route_needs_the_token(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 401
    assert client.get(path, params={"token": "wrong"}).status_code == 401


def test_index_sets_cookie_and_redirects(client: TestClient) -> None:
    assert client.get("/").status_code == 401
    assert client.get("/", params={"token": "wrong"}).status_code == 401

    response = client.get("/", params={"token": TOKEN})
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    set_cookie = response.headers["set-cookie"].lower()
    assert _cookie_name() in set_cookie
    assert "httponly" in set_cookie and "samesite=strict" in set_cookie

    page = client.get("/")
    assert page.status_code == 200
    assert "HELIX" in page.text
    assert client.get("/api/run").status_code == 200
    assert client.get("/static/app.js").text == "console.log('ok')"


def test_api_routes_with_token(client: TestClient) -> None:
    params = {"token": TOKEN}
    run = client.get("/api/run", params=params)
    assert run.json() == SNAPSHOT
    assert run.headers["cache-control"] == "no-store"
    assert run.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/static", params=params).json() == STATIC
    assert client.get("/api/frontier", params=params).json()["ids"] == ["g1-s1"]
    assert client.get("/api/candidate/g1-s1", params=params).json() == {**DETAIL, "diff_base": "parent"}
    transcript = client.get(
        "/api/transcript/g1-s1", params={**params, "cursor": "42"}
    ).json()
    assert transcript["cursor"] == 42


def test_run_is_null_without_a_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeRunReader, "snapshot", lambda self, now=None: None)
    response = client.get("/api/run", params={"token": TOKEN})
    assert response.status_code == 200
    assert response.json() is None


@pytest.mark.parametrize("cid", ["g9-s9", "../state", "g1-s1x", "seed"])
def test_unknown_or_bad_candidate_is_404(client: TestClient, cid: str) -> None:
    response = client.get(f"/api/candidate/{cid}", params={"token": TOKEN})
    assert response.status_code == 404


def test_transcript_bad_id_and_cursor(client: TestClient) -> None:
    params = {"token": TOKEN}
    assert client.get("/api/transcript/nope", params=params).status_code == 404
    for cursor in ("-1", "abc"):
        response = client.get(
            "/api/transcript/g1-s1", params={**params, "cursor": cursor}
        )
        assert response.status_code == 400


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_non_get_is_405(client: TestClient, method: str) -> None:
    response = client.request(method, "/api/run", params={"token": TOKEN})
    assert response.status_code == 405
    assert client.request(method, "/").status_code == 405


@pytest.mark.parametrize(
    "host", ["evil.example:8790", "attacker.test", "127.0.0.2.nip.io:8790"]
)
def test_host_guard_rejects_rebinding(client: TestClient, host: str) -> None:
    response = client.get("/api/run", params={"token": TOKEN}, headers={"host": host})
    assert response.status_code == 400


@pytest.mark.parametrize("host", ["localhost:8790", "127.0.0.1", "[::1]:8790"])
def test_host_guard_accepts_loopback_names(client: TestClient, host: str) -> None:
    response = client.get("/api/run", params={"token": TOKEN}, headers={"host": host})
    assert response.status_code == 200


def test_host_guard_uses_bound_host(
    fake_modules: None, project: Path, static_dir: Path
) -> None:
    app = server.create_app(project, project / "helix.toml", TOKEN, host="10.0.0.5")
    client = TestClient(app, base_url="http://10.0.0.5:8790")
    assert client.get("/api/run", params={"token": TOKEN}).status_code == 200
    response = client.get(
        "/api/run", params={"token": TOKEN}, headers={"host": "localhost:8790"}
    )
    assert response.status_code == 400


def test_event_stream_sends_static_then_run_then_keepalive() -> None:
    signatures = iter([1, 1, 2])

    async def take(n: int) -> list[str]:
        stream = server.event_stream(
            lambda: STATIC,
            lambda: SNAPSHOT,
            lambda: next(signatures),
            threading.Event(),
            poll_seconds=0.001,
            keepalive_seconds=0.0,
        )
        frames = [await anext(stream) for _ in range(n)]
        await stream.aclose()
        return frames

    frames = asyncio.run(take(4))
    assert len(frames) == 4
    assert frames[0].startswith("event: static\ndata: ")
    assert '"name":"proj"' in frames[0]
    assert frames[1].startswith("event: run\ndata: ")
    assert '"current":1' in frames[1]
    assert frames[2] == ": keepalive\n\n"
    assert frames[3].startswith("event: run\n")


def test_run_signature_tracks_run_files(tmp_path: Path) -> None:
    helix_dir = tmp_path / ".helix"
    empty = server.run_signature(helix_dir)
    assert empty == (None,) * 5
    helix_dir.mkdir()
    (helix_dir / "timing.jsonl").write_text("{}\n")
    first = server.run_signature(helix_dir)
    with (helix_dir / "timing.jsonl").open("a") as fh:
        fh.write("{}\n")
    assert server.run_signature(helix_dir) != first


def test_events_route_rejects_without_token(client: TestClient) -> None:
    assert client.get("/api/events").status_code == 401


def test_bind_socket_skips_a_taken_port() -> None:
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    port = taken.getsockname()[1]
    try:
        sock = server.bind_socket("127.0.0.1", port)
        try:
            assert port < sock.getsockname()[1] <= port + 10
        finally:
            sock.close()
    finally:
        taken.close()


def test_launch_url() -> None:
    assert server.launch_url("127.0.0.1", 8790, "t") == "http://127.0.0.1:8790/?token=t"
    assert server.launch_url("0.0.0.0", 8791, "t") == "http://127.0.0.1:8791/?token=t"
    assert server.launch_url("::1", 8790, "t") == "http://[::1]:8790/?token=t"


def test_serve_prints_url_and_warns_off_loopback(
    fake_modules: None,
    project: Path,
    static_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}

    class FakeSocket:
        def getsockname(self) -> tuple[str, int]:
            return ("0.0.0.0", 8791)

    class FakeServer:
        def __init__(self, config: Any, stopping: threading.Event) -> None:
            calls["log_level"] = config.log_level

        def run(self, sockets: list[Any]) -> None:
            calls["sockets"] = sockets

    monkeypatch.setattr(server, "bind_socket", lambda host, port: FakeSocket())
    monkeypatch.setattr(server, "_Server", FakeServer)
    monkeypatch.setattr(server.webbrowser, "open", lambda url: calls.setdefault("opened", url))
    server.serve(project, project / "helix.toml", "0.0.0.0", 8790, open_browser=True)

    out = capsys.readouterr().out
    assert "http://127.0.0.1:8791/?token=" in out
    assert "not loopback" in out
    assert calls["opened"].startswith("http://127.0.0.1:8791/?token=")
    assert calls["log_level"] == "warning"
    assert len(calls["sockets"]) == 1


def test_cli_help() -> None:
    result = CliRunner().invoke(cli, ["ui", "--help"])
    assert result.exit_code == 0
    for flag in ("--config", "--dir", "--host", "--port", "--no-browser"):
        assert flag in result.output


def test_cli_starts_without_binding_or_browser(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_serve(
        root: Path, config_path: Path, host: str, port: int, open_browser: bool
    ) -> None:
        seen.update(
            root=root, config=config_path, host=host, port=port, open=open_browser
        )

    monkeypatch.setattr(server, "serve", fake_serve)
    result = CliRunner().invoke(
        cli, ["ui", "--dir", str(project), "--port", "9001", "--no-browser"]
    )
    assert result.exit_code == 0, result.output
    assert seen == {
        "root": project.resolve(),
        "config": project.resolve() / "helix.toml",
        "host": "127.0.0.1",
        "port": 9001,
        "open": False,
    }


def test_cli_missing_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "serve", lambda *a, **k: pytest.fail("served"))
    result = CliRunner().invoke(cli, ["ui", "--dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Config file not found" in result.output


def test_cli_reports_taken_ports(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(*args: Any, **kwargs: Any) -> None:
        raise OSError("Ports 8790-8800 on 127.0.0.1 are all taken")

    monkeypatch.setattr(server, "serve", busy)
    result = CliRunner().invoke(cli, ["ui", "--dir", str(project), "--no-browser"])
    assert result.exit_code == 1
    assert "all taken" in result.output


def test_event_stream_ends_when_stopping() -> None:
    stopping = threading.Event()
    stopping.set()

    async def collect() -> list[str]:
        stream = server.event_stream(
            lambda: STATIC,
            lambda: SNAPSHOT,
            lambda: 1,
            stopping,
            poll_seconds=0.001,
            keepalive_seconds=0.0,
        )
        return [frame async for frame in stream]

    frames = asyncio.run(collect())
    assert len(frames) == 1 and frames[0].startswith("event: static")


def test_lineage_proposals_and_instance_routes(client: TestClient) -> None:
    params = {"token": TOKEN}
    assert client.get("/api/lineage", params=params).json() == LINEAGE
    assert client.get("/api/proposals", params=params).json() == PROPOSALS
    history = client.get("/api/instance/lift__seed101", params=params)
    assert history.status_code == 200 and history.json()["instance"] == "lift__seed101"
    # An id with "/" and a space, URL-encoded by the page.
    assert client.get("/api/instance/tasks%2Fa%20b", params=params).json()["instance"] == "tasks/a b"
    for iid in ("nope", "..%2Fstate.json", "lift__seed101%00"):
        assert client.get(f"/api/instance/{iid}", params=params).status_code == 404


def test_candidate_instance_route(client: TestClient) -> None:
    params = {"token": TOKEN}
    got = client.get("/api/candidate/g1-s1/instance/tasks%2Fa%20b", params=params)
    assert got.status_code == 200 and got.json()["instance"] == "tasks/a b"
    for path in ("g1-s1/instance/nope", "g1-s1/instance/..%2Fstate.json", "g9-s9/instance/lift__seed101",
                 "bad/instance/lift__seed101"):
        assert client.get(f"/api/candidate/{path}", params=params).status_code == 404


def test_candidate_vs_query(client: TestClient) -> None:
    params = {"token": TOKEN}
    assert client.get("/api/candidate/g1-s1", params={**params, "vs": "seed"}).json()["diff_base"] == "seed"
    assert client.get("/api/candidate/g1-s1", params={**params, "vs": "parent"}).json()["diff_base"] == "parent"
    for vs in ("", "SEED", "g0-s0", "../x"):
        assert client.get("/api/candidate/g1-s1", params={**params, "vs": vs}).status_code == 400


def test_new_routes_with_the_real_readers(project: Path) -> None:
    """/api/lineage, /api/proposals and /api/instance over a synthetic seed-only run."""
    import json

    from helix.population import EvalResult
    from helix.state import BudgetState, EvolutionState, save_state

    evaluations = project / ".helix" / "evaluations"
    evaluations.mkdir(parents=True)
    seed = EvalResult(candidate_id="g0-s0", scores={}, asi={}, instance_scores={"lift__1": 1.0, "lift__2": 0.0})
    (evaluations / "g0-s0.json").write_text(json.dumps(seed.to_dict()))
    (project / ".helix" / "lineage.json").write_text(json.dumps([
        {"id": "g0-s0", "parent": None, "parents": [], "operation": "seed", "generation": 0, "files_changed": []}]))
    save_state(EvolutionState(generation=0, frontier=["g0-s0"], instance_scores={}, budget=BudgetState(),
                              config_hash="x"), project)
    app = server.create_app(project, project / "helix.toml", TOKEN)
    client = TestClient(app, base_url="http://127.0.0.1:8790")
    params = {"token": TOKEN}

    for path in ("/api/lineage", "/api/proposals", "/api/instance/lift__1"):
        assert client.get(path).status_code == 401
    lineage = client.get("/api/lineage", params=params).json()
    assert [(n["id"], n["best_path"]) for n in lineage["nodes"]] == [("g0-s0", True)]
    proposals = client.get("/api/proposals", params=params).json()
    assert [(r["id"], r["outcome"]) for r in proposals["rows"]] == [("g0-s0", "frontier")]
    history = client.get("/api/instance/lift__1", params=params).json()
    assert history == {"instance": "lift__1", "points": [{"id": "g0-s0", "generation": 0, "score": 1.0}],
                       "best": 1.0, "holders": ["g0-s0"]}
    assert client.get("/api/instance/train__9", params=params).status_code == 404
    detail = client.get("/api/candidate/g0-s0", params={**params, "vs": "seed"}).json()
    assert detail["diff_base"] == "seed" and detail["children"] == [] and detail["prompt"] is None
    assert client.get("/api/candidate/g0-s0", params={**params, "vs": "x"}).status_code == 400
    assert client.get("/api/candidate/g0-s0/instance/lift__1").status_code == 401
    one = client.get("/api/candidate/g0-s0/instance/lift__1", params=params).json()
    assert one == {"instance": "lift__1", "basis": "val", "score": 1.0, "side_info": None, "parent": None}
    assert client.get("/api/candidate/g0-s0/instance/train__9", params=params).status_code == 404


def test_usage_reaches_run_proposals_and_candidate(project: Path) -> None:
    """A ``usage`` line in timing.jsonl shows up in all three payloads through the real readers."""
    import json

    from helix.population import EvalResult
    from helix.state import BudgetState, EvolutionState, save_state

    helix = project / ".helix"
    (helix / "evaluations").mkdir(parents=True)
    seed = EvalResult(candidate_id="g0-s0", scores={}, asi={}, instance_scores={"lift__1": 1.0})
    (helix / "evaluations" / "g0-s0.json").write_text(json.dumps(seed.to_dict()))
    save_state(EvolutionState(generation=0, frontier=["g0-s0"], instance_scores={},
                              budget=BudgetState(cost_usd=0.2, input_tokens=30), config_hash="x"), project)
    usage = {"event": "usage", "candidate": "g0-s0", "source": "seed_generation", "input_tokens": 30,
             "output_tokens": 3, "cached_input_tokens": 0, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.2, "parent_span_id": None,
             "run_id": "r1", "generation": 0, "wall_time": 1.0, "thread_id": 1}
    (helix / "timing.jsonl").write_text(json.dumps(usage) + "\n")
    client = TestClient(server.create_app(project, project / "helix.toml", TOKEN), base_url="http://127.0.0.1:8790")
    params = {"token": TOKEN}
    expected = {"cost_usd": 0.2, "input_tokens": 30, "output_tokens": 3, "cached_input_tokens": 0}

    spend = client.get("/api/run", params=params).json()["spend"]
    assert spend["per_record"] is True and spend["by_generation"] == [{"generation": 0, **expected}]
    assert spend["by_outcome"]["accepted"] == expected
    row = client.get("/api/proposals", params=params).json()["rows"][0]
    assert row["id"] == "g0-s0" and row["usage"] == expected
    detail = client.get("/api/candidate/g0-s0", params=params).json()
    assert (detail["usage"], detail["usage_source"]) == (expected, "timing")
