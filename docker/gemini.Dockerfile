ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

ARG CLI_VERSION=0.52.0
ARG CLI_TARBALL=https://registry.npmjs.org/@google/gemini-cli/-/gemini-cli-0.52.0.tgz
ARG CLI_SHA512=ffa16f7ef95cb0e26d9f7340813a79fdc6bdc52ea799ab78a2d43e3445684fa968be62363798c7c6d06a1ad8cb528f9cbc17c682d4765a69023f1c691e3afad5

RUN curl -fsSL "${CLI_TARBALL}" -o /tmp/gemini-cli.tgz \
    && echo "${CLI_SHA512}  /tmp/gemini-cli.tgz" | sha512sum --check --strict \
    && npm install --global --include=optional /tmp/gemini-cli.tgz \
    && npm cache clean --force \
    && rm -f /tmp/gemini-cli.tgz

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      io.helix.runner.backend="gemini" \
      io.helix.runner.cli-version="${CLI_VERSION}" \
      io.helix.runner.cli-tarball="${CLI_TARBALL}" \
      io.helix.runner.cli-sha512="${CLI_SHA512}"

WORKDIR /workspace
