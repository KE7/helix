ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG CLI_VERSION=2026.07.20-8cc9c0b
ARG CLI_AMD64_TARBALL=https://downloads.cursor.com/lab/2026.07.20-8cc9c0b/linux/x64/agent-cli-package.tar.gz
ARG CLI_AMD64_SHA256=6e9f17247ffeb5f8f7e2246b4bcd6bb26cb2d5a9f9a4b0012c9a80d868ed25b4
ARG CLI_ARM64_TARBALL=https://downloads.cursor.com/lab/2026.07.20-8cc9c0b/linux/arm64/agent-cli-package.tar.gz
ARG CLI_ARM64_SHA256=2986152b283c70a666b015035b2e99a96d13afd2660a587b8639417cfdd147fb

# The official installer is audited for version discovery, but the image does
# not pipe mutable network content to a shell.  It downloads the installer's
# exact platform artifact directly and verifies the discovered checksum.
RUN case "${TARGETARCH}" in \
        amd64) platform_url="${CLI_AMD64_TARBALL}"; platform_sha="${CLI_AMD64_SHA256}" ;; \
        arm64) platform_url="${CLI_ARM64_TARBALL}"; platform_sha="${CLI_ARM64_SHA256}" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 2 ;; \
    esac \
    && curl -fsSL "${platform_url}" -o /tmp/cursor-agent.tgz \
    && echo "${platform_sha}  /tmp/cursor-agent.tgz" | sha256sum --check --strict \
    && mkdir -p "/opt/cursor-agent/versions/${CLI_VERSION}" \
    && tar -xzf /tmp/cursor-agent.tgz --strip-components=1 \
        -C "/opt/cursor-agent/versions/${CLI_VERSION}" \
    && ln -s "/opt/cursor-agent/versions/${CLI_VERSION}/cursor-agent" /usr/local/bin/cursor-agent \
    && printf '%s\n' \
        '#!/bin/sh' \
        'if [ "$1" = "agent" ]; then shift; fi' \
        'exec cursor-agent "$@"' \
        > /usr/local/bin/cursor \
    && chmod +x /usr/local/bin/cursor \
    && rm -f /tmp/cursor-agent.tgz

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      io.helix.runner.backend="cursor" \
      io.helix.runner.cli-version="${CLI_VERSION}"

WORKDIR /workspace
