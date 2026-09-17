#!/usr/bin/env bash
# Launch the leak-free circle-packing run in the background.
#
#   cd <run dir> && .helix-run/bin/launch.sh [--resume] [--skip-preflight]
#   cd <run dir> && .helix-run/bin/launch.sh --preflight-only [--api-auth-check]
#   cd <run dir> && .helix-run/bin/launch.sh --dry-run
#
# Requires ANTHROPIC_API_KEY in this shell.  The key is handed only to the
# egress-gateway container (docker run -e NAME, so it never appears in an argv
# or a file); HELIX itself runs with a scrubbed environment that contains no
# Anthropic credential, so HELIX cannot forward one into agent containers.
#
# Steps:
#   1. check preconditions (fresh run dir or --resume, clean HELIX checkout,
#      images present) and record the run's provenance in logs/infra.json;
#   2. create the labeled Docker objects: an --internal agent network, an
#      egress bridge network, a dedicated agent home volume, the egress
#      gateway (on both networks) and the transcript archiver;
#   3. run the isolation checks (isolation_probe.py; no API spend) and abort
#      if any fails, unless --skip-preflight;
#   4. start run_wrapper.sh under nohup and return.
#
# --preflight-only stops after step 3 and leaves the infrastructure up
# (stop.sh removes it); --api-auth-check adds one Messages request with
# max_tokens 16 through the gateway from an agent container.
#
# --dry-run runs the whole pipeline with the gateway in capture-only mode and
# no credential: every model request is recorded and refused, so mutations
# fail and no API budget is spent.  Use it on a throwaway run directory with a
# small MAX_EVALUATIONS to exercise HELIX, the sidecar, the ledger, the
# archiver, the watchdog, stop.sh and summarize.sh end to end.
#
# Stop with .helix-run/bin/stop.sh; summarize with .helix-run/bin/summarize.sh.
set -euo pipefail

PRIVATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PRIVATE}/protocol.env"
LOGS="${PRIVATE}/logs"
MODE="evolve"
PREFLIGHT=1
PREFLIGHT_ONLY=0
DRY_RUN=0
PROBE_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --resume) MODE="resume" ;;
    --skip-preflight) PREFLIGHT=0 ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --api-auth-check) PROBE_ARGS+=(--api-auth-check) ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

say() { printf '[launch %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "${LOGS}/launch.log"; }
die() { say "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------- 1. checks
if [ -f "${LOGS}/wrapper.pid" ] && kill -0 "$(cat "${LOGS}/wrapper.pid")" 2>/dev/null; then
  die "a run is already active (pid $(cat "${LOGS}/wrapper.pid"))"
fi
if [ "${MODE}" = "evolve" ] && [ -e "${RUN_DIR}/.helix" ]; then
  die "${RUN_DIR}/.helix exists; use --resume or start from a fresh run directory"
fi
if [ "${MODE}" = "resume" ] && [ ! -f "${RUN_DIR}/.helix/state.json" ]; then
  die "nothing to resume: ${RUN_DIR}/.helix/state.json is missing"
fi
if [ "${DRY_RUN}" = "1" ]; then
  unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
else
  [ -n "${ANTHROPIC_API_KEY:-}" ] || die "ANTHROPIC_API_KEY must be set in this shell"
fi
docker info >/dev/null 2>&1 || die "Docker is not running"
command -v uv >/dev/null || die "uv is required"

if [ -n "$(git -C "${RUN_DIR}" status --porcelain --untracked-files=all)" ]; then
  die "the run directory's candidate tree has uncommitted changes"
fi
HELIX_COMMIT="$(git -C "${HELIX_SRC}" rev-parse HEAD)"
if [ -n "$(git -C "${HELIX_SRC}" status --porcelain --untracked-files=no)" ] \
   && [ "${HELIX_ALLOW_DIRTY:-0}" != "1" ]; then
  die "HELIX checkout ${HELIX_SRC} has uncommitted changes (set HELIX_ALLOW_DIRTY=1 to override)"
fi

image_id() { docker image inspect --format '{{.Id}}' "$1" 2>/dev/null || true; }
AGENT_ID="$(image_id "${AGENT_IMAGE}")"
EVAL_ID="$(image_id "${EVALUATOR_IMAGE}")"
GATEWAY_ID="$(image_id "${GATEWAY_IMAGE}")"
[ -n "${AGENT_ID}" ] && [ -n "${EVAL_ID}" ] && [ -n "${GATEWAY_ID}" ] \
  || die "images missing; run examples/circle_packing/clean_run/scripts/build_images.sh"
CLAUDE_VERSION="$(docker run --rm --label "${LABEL}=launch" --network none "${AGENT_IMAGE}" claude --version)"
PY_VERSIONS="$(docker run --rm --label "${LABEL}=launch" --network none "${EVALUATOR_IMAGE}" \
  python3 -c 'import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)')"
cat > "${LOGS}/infra-$(date -u +%Y%m%dT%H%M%SZ).json" <<JSON
{
  "mode": "${MODE}",
  "dry_run": ${DRY_RUN},
  "utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "helix_src": "${HELIX_SRC}",
  "helix_commit": "${HELIX_COMMIT}",
  "helix_branch": "$(git -C "${HELIX_SRC}" rev-parse --abbrev-ref HEAD)",
  "run_dir_commit": "$(git -C "${RUN_DIR}" rev-parse HEAD)",
  "images": {"agent": "${AGENT_ID}", "evaluator": "${EVAL_ID}", "gateway": "${GATEWAY_ID}"},
  "claude_code": "${CLAUDE_VERSION}",
  "evaluator_python_numpy_scipy": "${PY_VERSIONS}",
  "docker": "$(docker version --format '{{.Server.Version}}')",
  "docker_cpus": "$(docker info --format '{{.NCPU}}')",
  "docker_memory_bytes": "$(docker info --format '{{.MemTotal}}')",
  "host": "$(uname -srm)",
  "settings": {"max_evaluations": ${MAX_EVALUATIONS}, "max_executions": ${MAX_EXECUTIONS},
               "model": "${MODEL}", "effort": "${EFFORT}", "max_turns": ${MAX_TURNS}}
}
JSON

# --------------------------------------------------------- 2. Docker objects
is_ours() {  # $1 = object kind (network|volume|container), $2 = name
  [ "$(docker "$1" inspect --format "{{ index .${3:-Labels} \"${LABEL}\" }}" "$2" 2>/dev/null)" = "1" ]
}
ensure_network() {  # $1 name, $2 extra flags
  if docker network inspect "$1" >/dev/null 2>&1; then
    is_ours network "$1" || die "network $1 exists and is not labeled ${LABEL}=1"
  else
    # shellcheck disable=SC2086
    docker network create $2 --label "${LABEL}=1" "$1" >/dev/null
  fi
}
ensure_network "${AGENT_NETWORK}" "--internal"
ensure_network "${EGRESS_NETWORK}" ""
[ "$(docker network inspect --format '{{.Internal}}' "${AGENT_NETWORK}")" = "true" ] \
  || die "${AGENT_NETWORK} is not an --internal network"

if docker volume inspect "${AUTH_VOLUME}" >/dev/null 2>&1; then
  is_ours volume "${AUTH_VOLUME}" || die "volume ${AUTH_VOLUME} exists and is not labeled ${LABEL}=1"
  if [ "${MODE}" = "evolve" ]; then
    docker volume rm "${AUTH_VOLUME}" >/dev/null
    docker volume create --label "${LABEL}=1" "${AUTH_VOLUME}" >/dev/null
  fi
else
  docker volume create --label "${LABEL}=1" "${AUTH_VOLUME}" >/dev/null
fi
# The agent runs as the image's node user; give it the fresh home volume.
docker run --rm --label "${LABEL}=launch" --network none -v "${AUTH_VOLUME}:/home/node" \
  "${AGENT_IMAGE}" chown node:node /home/node

for name in "${GATEWAY_CONTAINER}" "${ARCHIVER_CONTAINER}"; do
  if docker container inspect "${name}" >/dev/null 2>&1; then
    is_ours container "${name}" Config.Labels || die "container ${name} exists and is not labeled ${LABEL}=1"
    docker rm -f "${name}" >/dev/null
  fi
done
GATEWAY_ENV=(-e "GATEWAY_ALLOWED_MODELS=${GATEWAY_ALLOWED_MODELS}")
if [ "${DRY_RUN}" = "1" ]; then
  GATEWAY_ENV+=(-e GATEWAY_CAPTURE_ONLY=1)
else
  GATEWAY_ENV+=(-e ANTHROPIC_API_KEY)
fi
docker run -d --name "${GATEWAY_CONTAINER}" --label "${LABEL}=1" \
  --network "${EGRESS_NETWORK}" --security-opt no-new-privileges \
  "${GATEWAY_ENV[@]}" -v "${LOGS}/gateway:/logs" "${GATEWAY_IMAGE}" >/dev/null
docker network connect --alias "${GATEWAY_ALIAS}" "${AGENT_NETWORK}" "${GATEWAY_CONTAINER}"
docker run -d --name "${ARCHIVER_CONTAINER}" --label "${LABEL}=1" --network none \
  --security-opt no-new-privileges \
  -v "${AUTH_VOLUME}:/src:ro" -v "${PRIVATE}/transcripts:/dst" "${AGENT_IMAGE}" \
  sh -c 'while :; do
           if [ -d /src/.claude/projects ]; then
             cd /src/.claude/projects && find . -name "*.jsonl" -type f \
               -exec cp -u --parents {} /dst/ \; 2>/dev/null
           fi
           sleep 0.5
         done' >/dev/null
sleep 2
docker ps --filter "name=^${GATEWAY_CONTAINER}$" --filter status=running -q | grep -q . \
  || die "gateway did not start: $(docker logs "${GATEWAY_CONTAINER}" 2>&1 | tail -5)"
say "infrastructure ready (gateway on ${EGRESS_NETWORK} + ${AGENT_NETWORK} as ${GATEWAY_ALIAS})"

# ------------------------------------------------------- 3. isolation checks
if [ "${PREFLIGHT}" = "1" ]; then
  if [ "${PREFLIGHT_ONLY}" = "0" ] && [ "${DRY_RUN}" = "0" ]; then
    # A real launch first proves the credential works (one max_tokens=16
    # request), so a dead key cannot burn the evaluation budget.
    PROBE_ARGS+=(--api-auth-check)
  fi
  say "running isolation checks (${PROBE_ARGS[*]:-no API spend})"
  if ! (cd "${PRIVATE}" && env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
        uv run --project "${HELIX_SRC}" python "${PRIVATE}/bin/isolation_probe.py" \
        --run-dir "${RUN_DIR}" ${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"} >> "${LOGS}/launch.log" 2>&1); then
    say "isolation checks FAILED; see ${LOGS}/launch.log and ${PRIVATE}/isolation/"
    "${PRIVATE}/bin/stop.sh" >/dev/null 2>&1 || true
    exit 1
  fi
  say "isolation checks passed"
fi
if [ "${PREFLIGHT_ONLY}" = "1" ]; then
  say "preflight only: infrastructure left running; remove it with ${PRIVATE}/bin/stop.sh"
  exit 0
fi

# ------------------------------------------------------------- 4. background
nohup "${PRIVATE}/bin/run_wrapper.sh" "${MODE}" >> "${LOGS}/wrapper.log" 2>&1 < /dev/null &
echo $! > "${LOGS}/wrapper.pid"
disown || true
say "started ${MODE} (wrapper pid $(cat "${LOGS}/wrapper.pid"))"
say "follow: tail -f ${LOGS}/helix_stdout.log   stop: ${PRIVATE}/bin/stop.sh   summary: ${PRIVATE}/bin/summarize.sh"
