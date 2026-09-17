#!/usr/bin/env bash
# Background body of launch.sh: run HELIX with a scrubbed environment, record
# wall-clock times, then stop the gateway and archiver.
#
#   run_wrapper.sh evolve|resume
set -uo pipefail

PRIVATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PRIVATE}/protocol.env"
LOGS="${PRIVATE}/logs"
MODE="${1:?evolve or resume}"

started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started_epoch="$(date +%s)"
printf '{"event":"start","mode":"%s","utc":"%s","epoch":%s,"helix_commit":"%s"}\n' \
  "${MODE}" "${started_utc}" "${started_epoch}" "$(git -C "${HELIX_SRC}" rev-parse HEAD)" \
  >> "${LOGS}/run_times.jsonl"

cd "${RUN_DIR}" || exit 1
# Job control gives the HELIX pipeline its own process group, which stop.sh
# signals as a whole (caffeinate, uv, HELIX and its docker clients).
set -m
# env -i: HELIX sees no Anthropic credential and no HELIX_* variables from this
# shell (it forwards both into agent containers).  caffeinate keeps the Mac
# awake for the run.
caffeinate -i env -i \
  PATH="${PATH}" HOME="${HOME}" LANG="en_US.UTF-8" TERM="dumb" COLUMNS="200" \
  uv run --project "${HELIX_SRC}" helix "${MODE}" \
    --config "${PRIVATE}/helix.toml" --dir "${RUN_DIR}" \
  >> "${LOGS}/helix_stdout.log" 2>&1 &
helix_pid=$!
echo "${helix_pid}" > "${LOGS}/helix.pid"
set +m
/usr/bin/python3 "${PRIVATE}/bin/watchdog.py" \
  --gateway-log "${LOGS}/gateway/gateway_requests.jsonl" \
  --pid-file "${LOGS}/helix.pid" --stop-record "${LOGS}/watchdog_stop.json" \
  --ledger-dir "${PRIVATE}/ledger" --helix-state "${RUN_DIR}/.helix/state.json" \
  --helix-log "${LOGS}/helix_stdout.log" >> "${LOGS}/watchdog.log" 2>&1 &
watchdog_pid=$!
docker_host="$(docker context inspect --format '{{.Endpoints.docker.Host}}')"
/usr/bin/python3 "${PRIVATE}/bin/agent_monitor.py" --docker-host "${docker_host}" \
  --network "${AGENT_NETWORK}" --out "${LOGS}/agent_containers.jsonl" \
  --pid-file "${LOGS}/helix.pid" >> "${LOGS}/agent_monitor.log" 2>&1 &
monitor_pid=$!
wait "${helix_pid}"
rc=$?
wait "${monitor_pid}" || true
kill "${watchdog_pid}" 2>/dev/null || true

ended_epoch="$(date +%s)"
printf '{"event":"end","mode":"%s","utc":"%s","epoch":%s,"exit_code":%s,"wall_seconds":%s}\n' \
  "${MODE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ended_epoch}" "${rc}" \
  "$((ended_epoch - started_epoch))" >> "${LOGS}/run_times.jsonl"

# Let the archiver take a last pass before it and the gateway are stopped.
sleep 5
docker exec "${ARCHIVER_CONTAINER}" sh -c \
  'cd /src/.claude/projects 2>/dev/null && find . -name "*.jsonl" -type f -exec cp -u --parents {} /dst/ \;' \
  >/dev/null 2>&1 || true
docker logs "${GATEWAY_CONTAINER}" > "${LOGS}/gateway/container.log" 2>&1 || true
for name in "${GATEWAY_CONTAINER}" "${ARCHIVER_CONTAINER}"; do
  docker rm -f "${name}" >/dev/null 2>&1 || true
done
rm -f "${LOGS}/helix.pid" "${LOGS}/wrapper.pid"
exit "${rc}"
