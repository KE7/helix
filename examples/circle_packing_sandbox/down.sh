#!/usr/bin/env bash
# Stop what up.sh started.  --purge also deletes the gateway's request logs.
set -uo pipefail
docker rm -f helix-circle-gateway >/dev/null 2>&1
docker network rm helix-circle-agent helix-circle-egress >/dev/null 2>&1
docker volume rm helix-circle-agent-home >/dev/null 2>&1
[ "${1:-}" = "--purge" ] && docker volume rm helix-circle-gateway-logs >/dev/null 2>&1
exit 0
