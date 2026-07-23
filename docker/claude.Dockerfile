ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

ARG CLI_VERSION=2.1.218
ARG CLI_TARBALL=https://registry.npmjs.org/@anthropic-ai/claude-code/-/claude-code-2.1.218.tgz
ARG CLI_SHA512=04757de75aee21ae905da6450c5d70461c3100e90089a7c1d803961bac064542786a9689f66973a750452c08467c6d12927c00caa05e9ea793ea78ade9ab78b9

RUN curl -fsSL "${CLI_TARBALL}" -o /tmp/claude-code.tgz \
    && echo "${CLI_SHA512}  /tmp/claude-code.tgz" | sha512sum --check --strict \
    && npm install --global --include=optional /tmp/claude-code.tgz \
    && npm cache clean --force \
    && rm -f /tmp/claude-code.tgz

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      io.helix.runner.backend="claude" \
      io.helix.runner.cli-version="${CLI_VERSION}" \
      io.helix.runner.cli-tarball="${CLI_TARBALL}" \
      io.helix.runner.cli-sha512="${CLI_SHA512}"

WORKDIR /workspace
