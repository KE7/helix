#!/usr/bin/env bash
# Build the four images of the leak-free circle-packing run.  Every image
# carries the label helix-circle-clean=1.  Re-running rebuilds in place.
set -euo pipefail
CLEAN_RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${HELIX_CIRCLE_CLEAN_TAG:-1}"

docker build --label helix-circle-clean=1 -t "helix-circle-clean/base:${TAG}" \
  -f "${CLEAN_RUN}/docker/base.Dockerfile" "${CLEAN_RUN}"
docker build --label helix-circle-clean=1 --build-arg "BASE_IMAGE=helix-circle-clean/base:${TAG}" \
  -t "helix-circle-clean/agent-claude:${TAG}" \
  -f "${CLEAN_RUN}/docker/agent-claude.Dockerfile" "${CLEAN_RUN}"
docker build --label helix-circle-clean=1 --build-arg "BASE_IMAGE=helix-circle-clean/base:${TAG}" \
  -t "helix-circle-clean/evaluator:${TAG}" \
  -f "${CLEAN_RUN}/docker/evaluator.Dockerfile" "${CLEAN_RUN}"
docker build --label helix-circle-clean=1 -t "helix-circle-clean/gateway:${TAG}" \
  -f "${CLEAN_RUN}/docker/gateway.Dockerfile" "${CLEAN_RUN}"

for image in base agent-claude evaluator gateway; do
  docker image inspect --format "helix-circle-clean/${image}:${TAG} {{.Id}}" \
    "helix-circle-clean/${image}:${TAG}"
done
docker run --rm --network none "helix-circle-clean/agent-claude:${TAG}" claude --version
docker run --rm --network none "helix-circle-clean/base:${TAG}" \
  python3 -c "import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"
