#!/usr/bin/env bash
# Start the network that examples/circle_packing's mutation agent runs on.
# Agent containers join an --internal Docker network whose only other member
# is anthropic_gateway.py, which forwards Anthropic Messages API requests and
# nothing else, so the agent can reach neither the web nor any other API.
#
#   up.sh [--capture]
#
# Builds the agent and gateway images, gives the agent a fresh, empty home
# volume, starts the gateway and checks the network (no API spend).  The
# gateway forwards with the Claude subscription in helix-auth-claude
# (`helix sandbox login claude`); the agent itself never holds a credential.
# With --capture it records and refuses every request instead, for trying the
# setup without a login.  MODEL (default claude-haiku-4-5) is the one model the
# gateway allows; keep it equal to [agent].model in helix.toml.
# Requests are logged in the helix-circle-gateway-logs volume.  down.sh stops it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-claude-haiku-4-5}"
AGENT_IMAGE="helix-circle/agent:1"
GATEWAY_IMAGE="helix-circle/gateway:1"
AGENT_NETWORK="helix-circle-agent"
EGRESS_NETWORK="helix-circle-egress"
GATEWAY="helix-circle-gateway"
HOME_VOLUME="helix-circle-agent-home"

capture=0
[ "${1:-}" = "--capture" ] && capture=1

docker build -q -t "${AGENT_IMAGE}" -f "${HERE}/agent.Dockerfile" "${HERE}/.." >/dev/null
docker build -q -t "${GATEWAY_IMAGE}" -f "${HERE}/gateway.Dockerfile" "${HERE}/.." >/dev/null

docker rm -f "${GATEWAY}" >/dev/null 2>&1 || true
docker network inspect "${AGENT_NETWORK}" >/dev/null 2>&1 \
  || docker network create --internal "${AGENT_NETWORK}" >/dev/null
[ "$(docker network inspect --format '{{.Internal}}' "${AGENT_NETWORK}")" = "true" ] \
  || { echo "${AGENT_NETWORK} exists and is not --internal" >&2; exit 1; }
docker network inspect "${EGRESS_NETWORK}" >/dev/null 2>&1 \
  || docker network create "${EGRESS_NETWORK}" >/dev/null

docker volume rm -f "${HOME_VOLUME}" >/dev/null
docker volume create "${HOME_VOLUME}" >/dev/null
docker run --rm --network none -v "${HOME_VOLUME}:/home/node" "${AGENT_IMAGE}" \
  chown node:node /home/node

gateway_args=(-e "GATEWAY_ALLOWED_MODELS=${MODEL}")
if [ "${capture}" = "1" ]; then
  gateway_args+=(-e GATEWAY_CAPTURE_ONLY=1)
else
  gateway_args+=(-v helix-auth-claude:/auth:rw)
fi
docker run -d --name "${GATEWAY}" --network "${EGRESS_NETWORK}" \
  --security-opt no-new-privileges "${gateway_args[@]}" \
  -v helix-circle-gateway-logs:/logs "${GATEWAY_IMAGE}" >/dev/null
docker network connect --alias anthropic-gateway "${AGENT_NETWORK}" "${GATEWAY}"
sleep 1
docker ps -q --filter "name=^${GATEWAY}$" --filter status=running | grep -q . \
  || { docker logs "${GATEWAY}" >&2; exit 1; }

# From an agent container: no direct route out, and the gateway refuses
# anything but plain Messages requests.
docker run --rm --network "${AGENT_NETWORK}" -e "MODEL=${MODEL}" \
  -e LITELLM_LOCAL_MODEL_COST_MAP=True "${AGENT_IMAGE}" sh -c '
  fail() { echo "FAIL: $*" >&2; exit 1; }
  status() { curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$@"; }
  for url in https://example.com https://api.anthropic.com https://erich-friedman.github.io; do
    curl -s -o /dev/null --max-time 5 "$url" && fail "$url is reachable"
  done
  [ "$(status http://anthropic-gateway:8080/v1/models)" = 403 ] \
    || fail "the gateway serves GET /v1/models"
  [ "$(status -X POST -H "content-type: application/json" \
        --data "{\"model\":\"$MODEL\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"web_search_20250305\",\"name\":\"web_search\"}]}" \
        http://anthropic-gateway:8080/v1/messages)" = 403 ] \
    || fail "the gateway forwards a web search tool"
  python3 -c "import numpy, scipy, dspy" || fail "the experiment environment is missing"
  echo "network checks passed: no route out except the Messages API"
'
