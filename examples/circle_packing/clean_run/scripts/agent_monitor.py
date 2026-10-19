"""Record the compute every mutation-agent container uses.

    python3 agent_monitor.py --docker-host unix:///path/docker.sock \\
        --network helix-circle-clean-agent --out <agent_containers.jsonl> \\
        --pid-file <helix.pid>

The evaluation budget counts executions of the scoring function; it does not
bound what the mutation agent computes in its own container (for example its
own packing searches).  This monitor measures that.  Every ``POLL_SECONDS`` it
asks the Docker Engine API (read-only calls on the host's Docker socket; it
never enters a container) for the containers attached to the agent network,
and for each HELIX command container (``helix-cmd-*``) records the cumulative
CPU time its cgroup reports, its peak memory and process count, its CPU
limit, and when it was first and last seen.  When a container disappears, one
JSON line with its final numbers is appended to ``--out``.

The CPU time is the last sample before the container exited, so it can miss
at most ``POLL_SECONDS`` times the container's CPU limit.  The monitor exits
when HELIX's process has exited (after writing every tracked container).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

POLL_SECONDS = 1.0
NAME_PREFIX = "/helix-cmd-"


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float = 10.0) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock


def docker_get(socket_path: str, path: str) -> Any:
    conn = UnixHTTPConnection(socket_path)
    try:
        conn.request("GET", path, headers={"Host": "docker"})
        response = conn.getresponse()
        data = response.read()
        if response.status != 200:
            return None
        return json.loads(data)
    finally:
        conn.close()


def socket_path_from_host(host: str) -> str:
    if not host.startswith("unix://"):
        raise ValueError(f"only unix:// Docker hosts are supported, got {host!r}")
    return host[len("unix://"):]


def sample_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Cumulative CPU seconds, memory and pids from one Engine API stats answer."""
    cpu = ((stats.get("cpu_stats") or {}).get("cpu_usage") or {}).get("total_usage")
    memory = (stats.get("memory_stats") or {}).get("usage")
    pids = (stats.get("pids_stats") or {}).get("current")
    return {
        "cpu_seconds": cpu / 1e9 if isinstance(cpu, (int, float)) else None,
        "memory_bytes": memory if isinstance(memory, int) else None,
        "pids": pids if isinstance(pids, int) else None,
    }


def merge_sample(entry: dict[str, Any], sample: dict[str, Any], now: str) -> None:
    entry["last_seen_utc"] = now
    entry["samples"] = entry.get("samples", 0) + 1
    if sample["cpu_seconds"] is not None:
        entry["cpu_seconds"] = max(entry.get("cpu_seconds") or 0.0, sample["cpu_seconds"])
    for key, peak in (("memory_bytes", "peak_memory_bytes"), ("pids", "peak_pids")):
        if sample[key] is not None:
            entry[peak] = max(entry.get(peak) or 0, sample[key])


def poll(socket_path: str, network: str, tracked: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Update ``tracked`` in place; return entries of containers that are gone."""
    now = datetime.now(timezone.utc).isoformat()
    filters = urllib.parse.quote(json.dumps({"network": [network]}))
    listing = docker_get(socket_path, f"/containers/json?filters={filters}") or []
    seen = set()
    for container in listing:
        names = container.get("Names") or []
        if not any(str(n).startswith(NAME_PREFIX) for n in names):
            continue
        cid = str(container.get("Id"))
        seen.add(cid)
        entry = tracked.get(cid)
        if entry is None:
            inspect = docker_get(socket_path, f"/containers/{cid}/json") or {}
            host_config = inspect.get("HostConfig") or {}
            config = inspect.get("Config") or {}
            entry = {
                "container": names[0].lstrip("/"),
                "id": cid[:12],
                "command": (config.get("Cmd") or [None])[0],
                "image": config.get("Image"),
                "cpu_limit": (host_config.get("NanoCpus") or 0) / 1e9 or None,
                "memory_limit_bytes": host_config.get("Memory") or None,
                "created": inspect.get("Created"),
                "first_seen_utc": now,
            }
            tracked[cid] = entry
        stats = docker_get(socket_path, f"/containers/{cid}/stats?stream=false&one-shot=true")
        if isinstance(stats, dict):
            merge_sample(entry, sample_from_stats(stats), now)
    gone = [tracked.pop(cid) for cid in list(tracked) if cid not in seen]
    return gone


def _append(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docker-host", required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    args = parser.parse_args()
    socket_path = socket_path_from_host(args.docker_host)
    tracked: dict[str, dict[str, Any]] = {}
    while True:
        try:
            gone = poll(socket_path, args.network, tracked)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            print(f"agent_monitor: poll failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            gone = []
        _append(args.out, gone)
        try:
            pid = int(args.pid_file.read_text().strip())
            os.kill(pid, 0)
        except (OSError, ValueError):
            _append(args.out, list(tracked.values()))
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
