#!/usr/bin/env bash
# Create the run directory for the leak-free circle-packing run.
#
#   setup_run_dir.sh [RUN_DIR]
#
# Settings (environment, defaults in brackets):
#   MAX_EVALUATIONS [200]  scoring-function executions (optimize_anything's
#                          max_metric_calls for circle packing)
#   MODEL [claude-haiku-4-5]  EFFORT [unset]  MAX_TURNS [40]
# Haiku 4.5 does not send output_config.effort; omit the CLI effort override.
#
# The run directory is a fresh git repository outside any HELIX clone whose
# only commit holds the seed candidate (solve.py, check_packing.py and a
# .gitignore).  Everything else lives under RUN_DIR/.helix-run/, which
# .git/info/exclude keeps out of the candidate tree and which HELIX never
# copies into an agent workspace (it skips every path component starting with
# ".helix"):
#
#   .helix-run/helix.toml      rendered from clean_run/helix.toml.template
#   .helix-run/protocol.env    settings, names, image tags, HELIX checkout
#   .helix-run/bin/            launch.sh, stop.sh, summarize.sh and helpers
#   .helix-run/ledger/         the evaluator's private ledger (sidecar mount)
#   .helix-run/logs/           HELIX output, gateway log, timings
#   .helix-run/transcripts/    every Claude Code session transcript
#   .helix-run/isolation/      isolation-check reports
#   .helix-run/summary/        summarize.sh output
set -euo pipefail

CLEAN_RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="$(cd "${CLEAN_RUN}/.." && pwd)"
HELIX_SRC="$(git -C "${CLEAN_RUN}" rev-parse --show-toplevel)"
RUN_DIR="${1:-${HOME}/helix-circle-packing-clean-run}"
MAX_EVALUATIONS="${MAX_EVALUATIONS:-200}"
MODEL="${MODEL:-claude-haiku-4-5}"
EFFORT="${EFFORT:-}"
MAX_TURNS="${MAX_TURNS:-40}"
TAG="${HELIX_CIRCLE_CLEAN_TAG:-1}"
MAX_EXECUTIONS="${MAX_EVALUATIONS}"
MAX_GENERATIONS="${MAX_GENERATIONS:-1000}"
AGENT_CPUS="${AGENT_CPUS:-2}"
SUBSCRIPTION_AUTH_VOLUME="${SUBSCRIPTION_AUTH_VOLUME:-helix-auth-claude}"

case "${RUN_DIR}" in
  "${HELIX_SRC}"|"${HELIX_SRC}"/*)
    echo "refusing: ${RUN_DIR} is inside the HELIX checkout ${HELIX_SRC}" >&2; exit 1 ;;
esac
if [ -e "${RUN_DIR}" ]; then
  echo "refusing: ${RUN_DIR} already exists (the run directory must be fresh)" >&2
  exit 1
fi
if git -C "$(dirname "${RUN_DIR}")" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "refusing: $(dirname "${RUN_DIR}") is inside a git repository" >&2
  exit 1
fi
for image in agent-claude evaluator gateway; do
  docker image inspect "helix-circle-clean/${image}:${TAG}" >/dev/null \
    || { echo "missing image helix-circle-clean/${image}:${TAG}; run build_images.sh" >&2; exit 1; }
done

mkdir -p "${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
cp "${EXAMPLE}/solve.py" "${EXAMPLE}/check_packing.py" "${RUN_DIR}/"
printf '.helix/\n__pycache__/\n' > "${RUN_DIR}/.gitignore"

git -C "${RUN_DIR}" init -q -b main
git -C "${RUN_DIR}" config user.name "circle-packing clean run"
git -C "${RUN_DIR}" config user.email "circle-packing-clean-run@localhost"
git -C "${RUN_DIR}" config commit.gpgsign false
printf '/.helix-run/\n' >> "${RUN_DIR}/.git/info/exclude"
git -C "${RUN_DIR}" add .gitignore solve.py check_packing.py
git -C "${RUN_DIR}" commit -q -m "Seed candidate"

PRIVATE="${RUN_DIR}/.helix-run"
mkdir -p "${PRIVATE}/bin" "${PRIVATE}/ledger" "${PRIVATE}/logs/gateway" \
  "${PRIVATE}/transcripts" "${PRIVATE}/isolation" "${PRIVATE}/summary"
chmod 700 "${PRIVATE}" "${PRIVATE}/ledger"
for f in launch.sh run_wrapper.sh stop.sh summarize.sh summarize.py \
         independent_check.py isolation_probe.py watchdog.py agent_monitor.py; do
  cp "${CLEAN_RUN}/scripts/${f}" "${PRIVATE}/bin/${f}"
done
cp "${CLEAN_RUN}/evaluator/evaluator_client.py" "${PRIVATE}/bin/evaluator_client.py"
chmod +x "${PRIVATE}/bin/"*.sh

EVAL_CPUS="$(docker run --rm --label helix-circle-clean=setup --network none \
  "helix-circle-clean/evaluator:${TAG}" nproc)"
sed -e "s#@LEDGER_DIR@#${PRIVATE}/ledger#g" \
    -e "s#@EVAL_CPUS@#${EVAL_CPUS}#g" \
    -e "s#@MAX_EVALUATIONS@#${MAX_EVALUATIONS}#g" \
    -e "s#@MAX_EXECUTIONS@#${MAX_EXECUTIONS}#g" \
    -e "s#@MAX_GENERATIONS@#${MAX_GENERATIONS}#g" \
    -e "s#@AGENT_CPUS@#${AGENT_CPUS}#g" \
    -e "s#@MODEL@#${MODEL}#g" \
    -e "s#@EFFORT@#${EFFORT}#g" \
    -e "s#@MAX_TURNS@#${MAX_TURNS}#g" \
    -e '/^effort = ""$/d' \
    "${CLEAN_RUN}/helix.toml.template" > "${PRIVATE}/helix.toml"
if grep -q '@[A-Z_]*@' "${PRIVATE}/helix.toml"; then
  echo "unrendered placeholder in ${PRIVATE}/helix.toml" >&2; exit 1
fi

cat > "${PRIVATE}/protocol.env" <<ENV
RUN_DIR='${RUN_DIR}'
HELIX_SRC='${HELIX_SRC}'
HELIX_COMMIT_AT_SETUP='$(git -C "${HELIX_SRC}" rev-parse HEAD)'
MAX_EVALUATIONS='${MAX_EVALUATIONS}'
MAX_EXECUTIONS='${MAX_EXECUTIONS}'
MAX_GENERATIONS='${MAX_GENERATIONS}'
AGENT_CPUS='${AGENT_CPUS}'
SUBSCRIPTION_AUTH_VOLUME='${SUBSCRIPTION_AUTH_VOLUME}'
MODEL='${MODEL}'
EFFORT='${EFFORT}'
MAX_TURNS='${MAX_TURNS}'
EVAL_CPUS='${EVAL_CPUS}'
IMAGE_TAG='${TAG}'
AGENT_IMAGE='helix-circle-clean/agent-claude:${TAG}'
EVALUATOR_IMAGE='helix-circle-clean/evaluator:${TAG}'
GATEWAY_IMAGE='helix-circle-clean/gateway:${TAG}'
AGENT_NETWORK='helix-circle-clean-agent'
EGRESS_NETWORK='helix-circle-clean-egress'
AUTH_VOLUME='helix-circle-clean-claude-home'
GATEWAY_CONTAINER='helix-circle-clean-gateway'
GATEWAY_ALIAS='anthropic-gateway'
ARCHIVER_CONTAINER='helix-circle-clean-transcript-archiver'
GATEWAY_ALLOWED_MODELS='${MODEL},claude-haiku-4-5'
LABEL='helix-circle-clean'
ENV

echo "run directory ready: ${RUN_DIR}"
echo "candidate tree:"
git -C "${RUN_DIR}" ls-files | sed 's/^/  /'
