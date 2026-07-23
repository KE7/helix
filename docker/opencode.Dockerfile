ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

ARG CLI_VERSION=1.18.4
ARG CLI_TARBALL=https://registry.npmjs.org/opencode-ai/-/opencode-ai-1.18.4.tgz
ARG CLI_SHA512=07ca4502cd60d79f1953e0ba792889cffe59846ed5af6987ed9eebb981072c4940957f6d7a1ad14e7cd34e08d2c47636bf067fe55a65c11de874553e0da8bc8e

RUN curl -fsSL "${CLI_TARBALL}" -o /tmp/opencode.tgz \
    && echo "${CLI_SHA512}  /tmp/opencode.tgz" | sha512sum --check --strict \
    && npm install --global --include=optional /tmp/opencode.tgz \
    && npm cache clean --force \
    && rm -f /tmp/opencode.tgz

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      io.helix.runner.backend="opencode" \
      io.helix.runner.cli-version="${CLI_VERSION}" \
      io.helix.runner.cli-tarball="${CLI_TARBALL}" \
      io.helix.runner.cli-sha512="${CLI_SHA512}"

WORKDIR /workspace
