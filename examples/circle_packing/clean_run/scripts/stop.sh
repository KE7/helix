#!/usr/bin/env bash
# Stop the circle-packing run and remove the Docker objects that belong to it.
#
#   .helix-run/bin/stop.sh [--purge]
#
# Sends HELIX SIGINT (it saves state and can be resumed with
# launch.sh --resume), escalating to SIGTERM/SIGKILL after 60 s.  Removes the
# gateway and archiver, and any HELIX sidecar, sidecar network, agent or
# runner container left behind by this run (identified by this run's ledger
# mount or networks, never by name alone).  --purge also removes the run's
# labeled networks and agent home volume.  Nothing without the
# helix-circle-clean label or a tie to this run is touched.
set -uo pipefail

PRIVATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PRIVATE}/protocol.env"
LOGS="${PRIVATE}/logs"
PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1

signal_group_and_wait() {  # $1 pid file of a process-group leader
  [ -f "$1" ] || return 0
  local pid; pid="$(cat "$1")"
  kill -0 "${pid}" 2>/dev/null || return 0
  kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null
  for _ in $(seq 60); do kill -0 "${pid}" 2>/dev/null || return 0; sleep 1; done
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null
  sleep 5
  kill -0 "${pid}" 2>/dev/null && { kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null; }
  return 0
}
# run_wrapper.sh starts HELIX as its own process group (set -m); the wrapper
# then records the end time and stops the gateway and archiver itself.
signal_group_and_wait "${LOGS}/helix.pid"
if [ -f "${LOGS}/wrapper.pid" ]; then
  wrapper="$(cat "${LOGS}/wrapper.pid")"
  for _ in $(seq 30); do kill -0 "${wrapper}" 2>/dev/null || break; sleep 1; done
  kill -0 "${wrapper}" 2>/dev/null && kill -TERM "${wrapper}" 2>/dev/null
fi

# HELIX's sidecar for this run mounts this run's ledger directory.
for c in $(docker ps -aq --filter "name=helix-evaluator-"); do
  if docker inspect --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' "${c}" \
       | grep -q "${PRIVATE}/ledger\$"; then
    nets="$(docker inspect --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' "${c}")"
    for net in ${nets}; do
      for r in $(docker ps -aq --filter "network=${net}" --filter "name=helix-cmd-"); do
        docker rm -f "${r}" >/dev/null 2>&1
      done
    done
    docker rm -f "${c}" >/dev/null 2>&1
    for net in ${nets}; do
      case "${net}" in helix-eval-*) docker network rm "${net}" >/dev/null 2>&1 ;; esac
    done
  fi
done
# Agent containers of this run sit on the run's agent network.
for c in $(docker ps -aq --filter "network=${AGENT_NETWORK}" --filter "name=helix-cmd-"); do
  docker rm -f "${c}" >/dev/null 2>&1
done

for name in "${GATEWAY_CONTAINER}" "${ARCHIVER_CONTAINER}"; do
  if [ "$(docker inspect --format "{{ index .Config.Labels \"${LABEL}\" }}" "${name}" 2>/dev/null)" = "1" ]; then
    [ "${name}" = "${GATEWAY_CONTAINER}" ] && docker logs "${name}" > "${LOGS}/gateway/container.log" 2>&1
    docker rm -f "${name}" >/dev/null 2>&1
  fi
done
rm -f "${LOGS}/helix.pid" "${LOGS}/wrapper.pid"

if [ "${PURGE}" = "1" ]; then
  for net in "${AGENT_NETWORK}" "${EGRESS_NETWORK}"; do
    [ "$(docker network inspect --format "{{ index .Labels \"${LABEL}\" }}" "${net}" 2>/dev/null)" = "1" ] \
      && docker network rm "${net}" >/dev/null 2>&1
  done
  [ "$(docker volume inspect --format "{{ index .Labels \"${LABEL}\" }}" "${AUTH_VOLUME}" 2>/dev/null)" = "1" ] \
    && docker volume rm "${AUTH_VOLUME}" >/dev/null 2>&1
fi
echo "stopped"
