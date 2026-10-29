#!/usr/bin/env bash
# Run HELIX on this example. Starts the check service that the sandboxed
# agent's check.py calls, runs `helix evolve` with any arguments given to
# this script, and stops the service when HELIX exits.
set -euo pipefail
cd "$(dirname "$0")"

if ! docker info >/dev/null 2>&1; then
    echo "run.sh: Docker is not running; the agent and the solvers need it." >&2
    exit 1
fi

# Docker Desktop routes host.docker.internal to this machine's loopback. On
# Linux it is the bridge gateway, so the service listens there instead.
host=127.0.0.1
if [[ "$(uname -s)" == Linux ]]; then
    host="$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')"
fi
port=8765

log="$(mktemp -t helix-check-server.XXXXXX)"
echo "run.sh: check service log: $log" >&2
uv run --no-project --python 3.12 --with-requirements requirements.txt \
    python3 -I check_server.py --host "$host" --port "$port" >"$log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT

# The first start installs the pinned packages and builds the solver image.
for _ in $(seq 600); do
    curl --silent --fail "http://$host:$port/health" >/dev/null && break
    kill -0 "$server" 2>/dev/null || { echo "run.sh: the check service exited; see $log" >&2; exit 1; }
    sleep 1
done
curl --silent --fail "http://$host:$port/health" >/dev/null || {
    echo "run.sh: the check service did not start." >&2
    exit 1
}

helix evolve "$@"
