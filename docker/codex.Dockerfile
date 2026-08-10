ARG BASE_IMAGE=helix-runner-base:latest
FROM ${BASE_IMAGE}

ARG TARGETARCH
ARG CLI_VERSION=0.145.0
ARG CLI_TARBALL=https://registry.npmjs.org/@openai/codex/-/codex-0.145.0.tgz
ARG CLI_SHA512=fcf48f485ba38e39a2c9516f1b6caefe0ace161b167689131fcb762865a15d2a3f3399ff7480ec9dbb273bcdbfedb2ed2280eecd07fb013054316a8f59020d95
ARG CLI_AMD64_TARBALL=https://registry.npmjs.org/@openai/codex/-/codex-0.145.0-linux-x64.tgz
ARG CLI_AMD64_SHA512=bbcc3c2cbbf70efb1fac30a8b302c87a66744a83445f28b9d755ac7c5b1289851ace4f6a32c07f36d53c37dbe101f37b9990312c5a0c7b1e2feba2631ee656c0
ARG CLI_ARM64_TARBALL=https://registry.npmjs.org/@openai/codex/-/codex-0.145.0-linux-arm64.tgz
ARG CLI_ARM64_SHA512=f0e2dc3d7680a25fc53ab4680f15a12221c815aef7291b0ce3510aa1c8d164ec224f84dc7a5cc95a7ddd1f28ae49beed796176e6baec96f48fcaf8542d845109

# Install the exact launcher and exact architecture package by verified
# tarball.  This avoids a second, floating npm resolution for Codex's native
# optional dependency.
RUN case "${TARGETARCH}" in \
        amd64) platform_name=codex-linux-x64; platform_url="${CLI_AMD64_TARBALL}"; platform_sha="${CLI_AMD64_SHA512}" ;; \
        arm64) platform_name=codex-linux-arm64; platform_url="${CLI_ARM64_TARBALL}"; platform_sha="${CLI_ARM64_SHA512}" ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 2 ;; \
    esac \
    && curl -fsSL "${CLI_TARBALL}" -o /tmp/codex.tgz \
    && curl -fsSL "${platform_url}" -o /tmp/codex-platform.tgz \
    && echo "${CLI_SHA512}  /tmp/codex.tgz" | sha512sum --check --strict \
    && echo "${platform_sha}  /tmp/codex-platform.tgz" | sha512sum --check --strict \
    && mkdir -p "/opt/codex/node_modules/@openai/${platform_name}" \
    && tar -xzf /tmp/codex.tgz --strip-components=1 -C /opt/codex \
    && tar -xzf /tmp/codex-platform.tgz --strip-components=1 \
        -C "/opt/codex/node_modules/@openai/${platform_name}" \
    && ln -s /opt/codex/bin/codex.js /usr/local/bin/codex \
    && rm -f /tmp/codex.tgz /tmp/codex-platform.tgz

LABEL org.opencontainers.image.source="https://github.com/KE7/helix" \
      io.helix.runner.backend="codex" \
      io.helix.runner.cli-version="${CLI_VERSION}" \
      io.helix.runner.required-model="gpt-5.6-luna" \
      io.helix.runner.required-reasoning-effort="xhigh"

WORKDIR /workspace
